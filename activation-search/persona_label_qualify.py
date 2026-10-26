"""Finite discovery/held-out label qualification using six already leased B1 workers.

No model loading, lease renewal, original queue mutation or uncertain retries.
Run without --execute to validate inputs only. All statistics concern magnitude
of prefill activation including template tokens, never additive steering sign.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import re
from pathlib import Path
import signal
import sys
import socket
import sqlite3
import subprocess
import threading
import time
import numpy as np
from persona_overnight_map import descriptor, digest, fetch, file_hash, unit_state, validate_result, validate_artifact
# Keep flat deployment helpers first; support the repository's sibling package.
_sibling_persona=Path(__file__).resolve().parents[1]/'persona'
if _sibling_persona.is_dir() and str(_sibling_persona) not in sys.path:
    sys.path.append(str(_sibling_persona))
from persona_label_probes import load_probes

SHAPE = (64, 17408)
LABELS = {'joy':'happy','sadness':'sad','sexual_desire':'sexual desire','confusion':'confused','green':'green','dog':'dog'}
CLAIM = 'Empirical precision conditional on authored held-out text probes; not probability that a neuron means a word. Magnitude association does not determine steering sign.'


def wilson(tp, fp):
    n=tp+fp
    if not n:return 0.
    p=tp/n;z=1.959963984540054
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def select_candidate(values, labels, shape=SHAPE):
    """Discovery only: standardized contrast shortlist then midpoint balanced accuracy."""
    x=np.asarray(values,dtype=np.float32);y=np.asarray(labels,dtype=bool)
    if x.ndim!=2 or x.shape!=(len(y),math.prod(shape)) or y.sum()!=12 or (~y).sum()!=12 or not np.isfinite(x).all():
        raise ValueError('Require 12+12 finite discovery rows and all neuron coordinates')
    positive=x[y];negative=x[~y]
    delta=positive.mean(axis=0,dtype=np.float64)-negative.mean(axis=0,dtype=np.float64)
    sd=np.sqrt((positive.var(axis=0,ddof=1,dtype=np.float64)+negative.var(axis=0,ddof=1,dtype=np.float64))/2)
    score=np.divide(delta,np.maximum(sd,1e-12));score[(np.max(x,axis=0)-np.min(x,axis=0))<=1e-12]=0
    indices=np.lexsort((np.arange(x.shape[1]),-np.abs(score)))[:32]
    best=None
    for index in indices:
        if score[index]==0:continue
        unique=np.unique(x[:,index].astype(np.float64));thresholds=(unique[:-1]+unique[1:])/2
        direction='positive' if delta[index]>0 else 'inverse'
        for threshold in thresholds:
            prediction=x[:,index]>threshold if direction=='positive' else x[:,index]<threshold
            accuracy=float((prediction[y].mean()+(~prediction[~y]).mean())/2)
            key=(-accuracy,-abs(float(score[index])),int(index),float(threshold))
            if best is None or key<best[0]:best=(key,{'layer':int(index)//shape[1],'neuron':int(index)%shape[1],'direction':direction,'threshold':float(threshold),'discovery_balanced_accuracy':accuracy,'standardized_contrast':float(score[index]),'mean_difference':float(delta[index]),'measurement':'mean_absolute_all_template_prefill_activation','selection':'top32 absolute standardized contrasts; best discovery midpoint balanced accuracy; deterministic ties'})
    if best is None:raise ValueError('No varying discovery feature')
    return best[1]


NEGATION = re.compile(r"\b(?:no|not|never|neither|nor|without|cannot)\b|n['’]t\b", re.IGNORECASE)


def nuisance_values(rows, tokens):
    return {'encoded_token_length':[float(n) for n in tokens], 'explicit_negation_count':[float(len(NEGATION.findall(r['prompt']))) for r in rows]}


def fit_nuisance(values, labels):
    x=np.asarray(values,dtype=float);y=np.asarray(labels,dtype=bool)
    if len(x)!=24 or y.sum()!=12 or not np.isfinite(x).all():raise ValueError('Nuisance fitting uses discovery only')
    unique=np.unique(x);thresholds=(unique[:-1]+unique[1:])/2 if len(unique)>1 else unique
    best=None
    for direction in ('positive','inverse'):
        for threshold in thresholds:
            predicted=x>threshold if direction=='positive' else x<threshold
            balanced=float((predicted[y].mean()+(~predicted[~y]).mean())/2)
            key=(-balanced,direction,float(threshold))
            if best is None or key<best[0]:best=(key,{'direction':direction,'threshold':float(threshold),'discovery_balanced_accuracy':balanced})
    return best[1]


def nuisance_balanced_accuracy(frozen, values, rows):
    y=np.asarray([r['label'] for r in rows],dtype=bool);x=np.asarray(values)
    predicted=x>frozen['threshold'] if frozen['direction']=='positive' else x<frozen['threshold']
    return float((predicted[y].mean()+(~predicted[~y]).mean())/2)


def heldout_metrics(candidate, values, rows):
    if len(rows)!=48 or sum(r['label'] for r in rows)!=24 or len({r['family_id'] for r in rows})!=48:
        raise ValueError('Require 24+24 independent held-out families')
    if any(r['split']!='heldout' for r in rows):raise ValueError('Held-out split required')
    if len(values)!=len(rows) or not np.isfinite(values).all():raise ValueError('Invalid held-out activation values')
    predicted=np.asarray(values)>candidate['threshold'] if candidate['direction']=='positive' else np.asarray(values)<candidate['threshold']
    actual=np.asarray([r['label'] for r in rows],dtype=bool)
    tp=int((predicted&actual).sum());fp=int((predicted&~actual).sum());tn=int((~predicted&~actual).sum());fn=int((~predicted&actual).sum())
    recall=tp/(tp+fn);balanced=(recall+tn/(tn+fp))/2;lower=wilson(tp,fp)
    outcomes=[{'unit_id':r['family_id'],'correct':bool(r['label']),'probe_id':r['id']} for r,p in zip(rows,predicted) if p]
    return {'passed':balanced>=.8 and recall>=.75 and len(outcomes)>=20 and lower>.70,'tp':tp,'fp':fp,'tn':tn,'fn':fn,'samples':len(rows),'predicted_positive_families':len(outcomes),'precision':tp/(tp+fp) if tp+fp else None,'precision_wilson95_lower':lower,'recall':recall,'balanced_accuracy':balanced,'outcomes':outcomes,'all_outcomes':[{'probe_id':r['id'],'family_id':r['family_id'],'label':r['label'],'predicted':bool(p),'value':float(v)} for r,p,v in zip(rows,predicted,values)]}


def connect(path):
    db=sqlite3.connect(path,timeout=60);db.row_factory=sqlite3.Row;return db


def initialize(database, rows, binding):
    with connect(database) as db:
        db.executescript('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT);CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,concept TEXT,split TEXT,state TEXT,row TEXT,endpoint TEXT,request TEXT,response TEXT,validation TEXT,error TEXT);CREATE TABLE IF NOT EXISTS frozen(concept TEXT PRIMARY KEY,record TEXT,sha256 TEXT);CREATE TABLE IF NOT EXISTS events(time REAL,kind TEXT,body TEXT);')
        old=db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
        if old and json.loads(old[0])!=binding:raise ValueError('Resume source/config/dataset binding changed')
        if not old:
            db.execute('INSERT INTO metadata VALUES(?,?)',('binding',json.dumps(binding)))
            db.executemany('INSERT INTO tasks(id,concept,split,state,row) VALUES(?,?,?,?,?)',[(r['id'],r['concept'],r['split'],'pending',json.dumps(r)) for r in rows])
        db.execute("UPDATE tasks SET state='uncertain',error='Coordinator interrupted; no automatic inference retry' WHERE state='running'")


def pause_receipt(root, previous=None, now=None):
    now=time.time() if now is None else now
    pause=root/'overnight.pause';stat=pause.stat();receipt=(stat.st_dev,stat.st_ino,stat.st_mtime_ns,stat.st_size)
    status=json.loads((root/'overnight_status.json').read_text())
    if (previous is not None and tuple(previous)!=receipt) or status.get('paused') is not True or status.get('active')!=[] or not 0<=now-status['updated']<30 or status['updated']<stat.st_mtime:
        raise ValueError('Require unchanged overnight.pause plus fresh acknowledged drained status')
    return receipt


def lease_guard(endpoint, board, now):
    leases=[r for r in board['jobs'] if r.get('project')=='character-creation' and r.get('status')=='running' and r.get('assigned_uuid')==endpoint['gpu_uuid'] and r.get('unit_name')==endpoint['unit'] and r.get('unit_invocation_id')==endpoint['invocation_id']]
    if len(leases)!=1 or not math.isfinite(endpoint['lease_expires']) or endpoint['lease_expires']-now<120:
        raise ValueError('Missing matching healthy finite lease; pending work retained without renewal')
    return leases[0]['id']


def atomic_json(path, value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def immutable_json(path, value):
    content=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text()!=content:raise ValueError('Immutable publication differs: '+path.name)
    else:
        with path.open('x') as stream:stream.write(content)
    return {'artifact':path.name,'sha256':file_hash(path)}


def read_capture(task, root, endpoint):
    response=json.loads(task['response']);request=json.loads(task['request'])
    validation=validate_result(response,request,endpoint,root)
    path=validate_artifact(response['activation_artifact'],root)
    with np.load(path,allow_pickle=False) as arrays:values=arrays['search_sum_abs'].astype(np.float32).reshape(-1)/validation['tokens']
    return values,{'probe_id':task['id'],'endpoint':task['endpoint'],'request_sha256':digest(request),'activation_artifact':response['activation_artifact'],'provenance_artifact':response['provenance_artifact'],'profile_artifact':response['activation_search']['profile_artifact'],'profile_provenance_artifact':response['activation_search']['profile_provenance_artifact'],'result_artifact':response['activation_search']['result_artifact'],'validation':validation}


def freeze_concept(database, concept, root, endpoints, binding):
    with connect(database) as db:
        prior=db.execute('SELECT record,sha256 FROM frozen WHERE concept=?',(concept,)).fetchone()
        if prior:
            record=json.loads(prior[0]);assert digest(record)==prior[1];return record
        if db.execute("SELECT 1 FROM tasks WHERE split='heldout' AND state!='pending'").fetchone():raise ValueError('Held-out captures exist before freeze; refuse selection')
        tasks=db.execute("SELECT * FROM tasks WHERE concept=? AND split='discovery' ORDER BY id",(concept,)).fetchall()
    if len(tasks)!=24 or any(r['state']!='complete' for r in tasks):raise ValueError('Discovery incomplete; no held-out selection')
    values=np.empty((24,math.prod(SHAPE)),np.float32);rows=[];evidence=[]
    for i,task in enumerate(tasks):
        values[i],item=read_capture(task,root,endpoints[task['endpoint']]);rows.append(json.loads(task['row']));evidence.append(item)
    candidate=select_candidate(values,[r['label'] for r in rows]);del values
    nuisance={name:fit_nuisance(feature,[r['label'] for r in rows]) for name,feature in nuisance_values(rows,[item['validation']['tokens'] for item in evidence]).items()}
    record={'concept':concept,'candidate':candidate,'nuisance':nuisance,'binding':binding,'created':time.time(),'discovery_dataset_sha256':digest(rows),'discovery_evidence':evidence,'discovery_ids':[r['id'] for r in rows]}
    with connect(database) as db:db.execute('INSERT INTO frozen VALUES(?,?,?)',(concept,json.dumps(record),digest(record)))
    return record


def publish(database, dataset, root, endpoints, binding):
    labels=[]
    for concept in dataset['definitions']:
        with connect(database) as db:
            frozen=db.execute('SELECT record,sha256 FROM frozen WHERE concept=?',(concept,)).fetchone();record=json.loads(frozen[0])
            if digest(record)!=frozen[1] or record['binding']!=binding:raise ValueError('Frozen discovery binding differs')
            tasks=db.execute("SELECT * FROM tasks WHERE concept=? AND split='heldout' ORDER BY id",(concept,)).fetchall()
        if len(tasks)!=48 or any(r['state']!='complete' for r in tasks):raise ValueError('Held-out incomplete; no qualification publication')
        candidate=record['candidate'];rows=[];values=[];evidence=[];index=candidate['layer']*SHAPE[1]+candidate['neuron']
        for task in tasks:
            vector,item=read_capture(task,root,endpoints[task['endpoint']]);values.append(float(vector[index]));rows.append(json.loads(task['row']));evidence.append(item)
        metrics=heldout_metrics(candidate,values,rows)
        nuisance_scores={name:nuisance_balanced_accuracy(record['nuisance'][name],feature,rows) for name,feature in nuisance_values(rows,[item['validation']['tokens'] for item in evidence]).items()}
        metrics['nuisance_heldout_balanced_accuracy']=nuisance_scores
        metrics['nuisance_margin']=metrics['balanced_accuracy']-max(nuisance_scores.values())
        metrics['unconfounded']=metrics['nuisance_margin']>.05
        metrics['passed']=metrics['passed'] and metrics['unconfounded']
        validation={**metrics,'split':'held_out','dataset_sha256':digest(rows)}
        confidence={'value':metrics['precision'],'method':'heldout_precision','calibrated':metrics['passed']}
        row={k:candidate[k] for k in ('layer','neuron','direction')};row.update(label=LABELS[concept],confidence=confidence,association={'measurement':candidate['measurement'],'standardized_contrast':candidate['standardized_contrast'],'direction_semantics':'greater magnitude' if candidate['direction']=='positive' else 'lower magnitude','steering_sign_validated':False})
        proof={**{k:row[k] for k in ('layer','neuron','direction','label','confidence')},'schema':'character-creation.neuron-label-evidence.v1','model_identity':next(iter(endpoints.values()))['model_identity'],'strong_related':metrics['passed'],'claim':'probe_conditional_precision','probe':dataset['definitions'][concept],'discovery_dataset_sha256':record['discovery_dataset_sha256'],'calibration':{'method':'heldout_precision','split':'held_out','samples':len(rows),'dataset_sha256':digest(rows)},'validation':validation,'frozen_candidate':record,'frozen_sha256':frozen[1],'heldout_evidence':evidence,'binding':binding,'phase':'prefill','measurement':candidate['measurement'],'scope':CLAIM,'limitations':[*dataset['limitations'],'Distinct authored family IDs are a design constraint, not proof of statistical independence.','Length and explicit-negation baselines are discovery-fitted; held-out balanced accuracy must exceed both by more than 0.05.']}
        row['evidence']=immutable_json(root/f'persona_label_evidence_{concept}.json',proof);labels.append(row)
    registry={'schema':'character-creation.neuron-labels.v1','model_identity':next(iter(endpoints.values()))['model_identity'],'labels':labels,'binding':binding,'claim':CLAIM}
    immutable_json(root/'persona_neuron_labels.json',registry)
    return {'qualified':sum(r['confidence']['calibrated'] for r in labels),'candidates':len(labels),'registry_sha256':file_hash(root/'persona_neuron_labels.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--input-sha256',required=True);parser.add_argument('--database',type=Path,required=True);parser.add_argument('--workboard',default='/home/orwel/.local/bin/gpu-workboard');parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    dataset=load_probes(args.inputs)
    if file_hash(args.inputs)!=args.input_sha256:raise ValueError('Input file hash differs')
    if not args.execute:print(json.dumps({'state':'validated_only','rows':len(dataset['rows']),'dataset_sha256':dataset['sha256']}));return
    config=json.loads(args.config.read_text());root=Path(config['artifact_root']).resolve()
    if socket.gethostname()!='v100' or len(config['endpoints'])!=6 or args.database.resolve().parent!=root or args.database.name!='persona_label_qualification.sqlite':raise ValueError('Require V100 six-worker config and flat artifact database')
    endpoints={e['url']:e for e in config['endpoints']}
    if len(endpoints)!=6 or len({e['gpu_uuid'] for e in endpoints.values()})!=6 or any(len({e[k] for e in endpoints.values()})!=1 for k in ('model_identity','runtime_identity','prepared_identity')):raise ValueError('Require six distinct GPUs with common immutable identities')
    for name,sha in config['runtime_hashes'].items():
        if Path(name).name!=name or file_hash(root/name)!=sha:raise ValueError('Runtime source binding differs: '+name)
    binding={'config_sha256':file_hash(args.config),'dataset_sha256':dataset['sha256'],'input_file_sha256':args.input_sha256,'runner_sha256':file_hash(__file__),'capture_validator_sha256':file_hash(Path(__file__).with_name('persona_overnight_map.py')),'probe_validator_sha256':file_hash(Path(__file__).with_name('persona_label_probes.py')),'model_identity':next(iter(endpoints.values()))['model_identity'],'runtime_identity':next(iter(endpoints.values()))['runtime_identity'],'prepared_identity':next(iter(endpoints.values()))['prepared_identity'],'runtime_hashes':config['runtime_hashes'],'endpoint_receipts':[{k:e[k] for k in ('url','gpu_uuid','unit','invocation_id','lease_expires')} for e in endpoints.values()]}
    lock=args.database.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=pause_receipt(root);initialize(args.database,dataset['rows'],binding)
    stop=threading.Event();state_lock=threading.RLock();active=set();stage='admission';terminal=None
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.set())
    board_cache={'time':0,'value':None}
    def event(kind,**body):
        with connect(args.database) as db:db.execute('INSERT INTO events VALUES(?,?,?)',(time.time(),kind,json.dumps(body)))
        print(json.dumps({'time':time.time(),'event':kind,**body}),flush=True)
    def status():
        with state_lock:
            with connect(args.database) as db:counts=dict(db.execute('SELECT state,count(*) FROM tasks GROUP BY state'))
            atomic_json(root/'persona_label_qualification_status.json',{'updated':time.time(),'state':terminal or stage,'counts':counts,'active':sorted(active),'binding':binding})
    def guard(endpoint):
        pause_receipt(root,receipt)
        if time.time()>config['deadline']-120:raise ValueError('Finite deadline reached')
        current=unit_state(endpoint['unit'])
        if current.get('ActiveState')!='active' or current.get('InvocationID')!=endpoint['invocation_id']:raise ValueError('Owned worker invocation changed')
        with state_lock:
            if time.time()-board_cache['time']>5:
                board_cache['value']=json.loads(subprocess.check_output([args.workboard,'status','--json'],text=True,timeout=30));board_cache['time']=time.time()
            lease=lease_guard(endpoint,board_cache['value'],time.time())
        model=descriptor(endpoint)
        if model['flare'].get('batch_capacity')!=1:raise ValueError('Expected reviewed B1 capture worker')
        return model,lease
    def worker(endpoint,split):
        while not stop.is_set():
            with connect(args.database) as db:
                if not db.execute("SELECT 1 FROM tasks WHERE split=? AND state='pending'",(split,)).fetchone():return
            try:model,lease=guard(endpoint)
            except Exception as exc:event('admission_refused',endpoint=endpoint['url'],error=str(exc));stop.set();return
            with state_lock:
                if stop.is_set():return
                with connect(args.database) as db:
                    db.execute('BEGIN IMMEDIATE');found=db.execute("SELECT id,row FROM tasks WHERE split=? AND state='pending' ORDER BY id LIMIT 1",(split,)).fetchone()
                    if not found:return
                    identity,row=found[0],json.loads(found[1]);request={'model':model['id'],'messages':[{'role':'user','content':row['prompt']}],'seed':7035,'max_tokens':1,'temperature':0,'stream':False,'observation_only':True,'chat_template_kwargs':{'enable_thinking':False},'capture_search':{'id':digest({'binding':binding,'probe':identity})[:32],'top_k':32}}
                    db.execute("UPDATE tasks SET state='running',endpoint=?,request=? WHERE id=?",(endpoint['url'],json.dumps(request),identity))
                active.add(endpoint['url']);event('dispatch',id=identity,split=split,lease_id=lease,endpoint=endpoint['url']);status()
            response=None
            try:
                response=fetch(endpoint['url']+'/v1/chat/completions',request,timeout=min(1200,endpoint['lease_expires']-time.time(),config['deadline']-time.time()))
                with connect(args.database) as db:db.execute('UPDATE tasks SET response=? WHERE id=?',(json.dumps(response,allow_nan=False),identity))
                validation=validate_result(response,request,endpoint,root)
                with connect(args.database) as db:db.execute("UPDATE tasks SET state='complete',validation=? WHERE id=?",(json.dumps(validation),identity))
                event('complete',id=identity,validation=validation)
            except Exception as exc:
                state='failed' if response is not None else 'uncertain'
                with connect(args.database) as db:db.execute('UPDATE tasks SET state=?,error=? WHERE id=?',(state,repr(exc),identity))
                event(state,id=identity,error=repr(exc));stop.set()
            finally:
                with state_lock:active.discard(endpoint['url']);status()
    try:
        event('start',binding=binding,pause_receipt=receipt);status()
        for split in ('discovery','heldout'):
            stage=split;status()
            if split=='heldout':
                for concept in dataset['definitions']:
                    frozen=freeze_concept(args.database,concept,root,endpoints,binding);event('frozen',concept=concept,sha256=digest(frozen),candidate=frozen['candidate'])
                with connect(args.database) as db:
                    if db.execute('SELECT COUNT(*) FROM frozen').fetchone()[0]!=6:raise ValueError('All discovery candidates must be frozen before held-out')
                stage='heldout';status()
            with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(lambda endpoint:worker(endpoint,split),endpoints.values()))
            with connect(args.database) as db:
                counts=dict(db.execute('SELECT state,count(*) FROM tasks WHERE split=? GROUP BY state',(split,)))
            if stop.is_set() or counts!={'complete':144 if split=='discovery' else 288}:raise ValueError('Capture phase incomplete; no automatic retry: '+str(counts))
        stage='qualification';status();result=publish(args.database,dataset,root,endpoints,binding);event('published',**result);terminal='complete'
    except Exception as exc:
        terminal='incomplete';event('incomplete',error=repr(exc));raise
    finally:status();lock.close()

if __name__=='__main__':main()
