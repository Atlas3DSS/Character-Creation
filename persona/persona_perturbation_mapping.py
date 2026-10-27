"""Finite low-priority baseline map of original synthetic discovery prompts.

No leases or GPU services are created. Foreground studies and interactive viewer
work block new claims. An already claimed native batch drains without replay.
"""
import argparse
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import threading
import time
from urllib.parse import quote
from urllib.request import urlopen
from persona_manager import Queue,Manager
from persona_vectors import canonical_digest
from persona_reasoning_sweep import atomic_json,publication_due

CORPUS_SHA='c2eb38ebcba2f39f3ff786f9054183ee166cf629f81fe55ad8e6e4171ea47bff'
MANIFEST_SHA='b18cfa620ebb650da349df16b5c8ea5fb132c2a387f5a009103ca98e40f0eeea'
DEADLINE=1793113200
SEED=80617
REPO='Atlas3D/character-creation-big-five'
PUBLIC='persona_perturbation_mapping_public.json'
STUDIES=('persona_preset_overnight.sqlite','persona_preset_discovery_continuation.sqlite','persona_preset_confirmation.sqlite',
         'persona_preset_followon_discovery.sqlite','persona_preset_followon_confirmation.sqlite')
BUSY=('queued','running','submitting','stopping')
TERMINAL=('complete','incomplete','failed')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)


def foreground_busy(root):
    """Bounded read-only checks; lock/schema/read failures yield rather than risk priority."""
    checks=[(Path(root)/'results'/name,('jobs',)) for name in STUDIES]
    checks.append((Path(root)/'results/persona_explorer.sqlite',('batches','sculptor_campaigns')))
    for path,tables in checks:
        if not path.exists():continue
        try:
            with closing(sqlite3.connect('file:'+quote(str(path.resolve()),safe='/')+'?mode=ro',uri=True,timeout=.25)) as db:
                db.execute('PRAGMA query_only=ON')
                for table in tables:
                    # Tables are fixed module constants, never user input.
                    if db.execute(f'SELECT 1 FROM {table} WHERE state IN (?,?,?,?) LIMIT 1',BUSY).fetchone():return True
        except (sqlite3.Error,OSError):return True
    return False


class PriorityQueue(Queue):
    def __init__(self,path,root):
        super().__init__(path);self.root=Path(root);self.priority_lock=threading.RLock()
    def claim_many(self,endpoint,limit):
        with self.priority_lock:
            if foreground_busy(self.root):return []
            return super().claim_many(endpoint,limit)
    def audit(self,event,data):
        if event in ('native_batch_submitting','native_b1_submitting') and foreground_busy(self.root):
            # dispatch calls this BEFORE its only HTTP POST. Release only attempts
            # lacking a durable submission intent; Manager then sees no active
            # claim to terminalize. An already submitted request is never retried.
            claims=[]
            with self.connect() as db:
                for attempt in data['attempts']:
                    row=db.execute("SELECT body FROM jobs WHERE id=? AND state='running' AND attempts=?",
                        (attempt['job_id'],attempt['number'])).fetchone()
                    require(row is not None,'Priority release lost claim ownership')
                    claims.append((json.loads(row[0]),attempt['number']))
            self.release_unsubmitted(claims)
            super().audit('background_yield_before_submission',{'batch_id':data['batch_id'],'jobs':[j['id'] for j,n in claims]})
            raise RuntimeError('Foreground appeared before background submission; unsubmitted claims safely released')
        return super().audit(event,data)

    def claim(self,endpoint):
        with self.priority_lock:
            if foreground_busy(self.root):return None
            return super().claim(endpoint)


def prepare(root):
    root=Path(root);source=root/'data/persona_perturbation_prompts.jsonl';mp=root/'data/persona_perturbation_prompts_manifest.json'
    require(sha(source)==CORPUS_SHA and sha(mp)==MANIFEST_SHA,'Frozen original corpus/manifest SHA differs')
    manifest=json.loads(mp.read_text());require(manifest['status']=='complete' and manifest['records']==3072 and manifest['split_counts']=={'discovery':2443,'heldout':629},'Frozen corpus counts differ')
    original=[json.loads(line) for line in source.read_text().splitlines() if line.strip()];rows=[r for r in original if r['split']=='discovery']
    require(len(original)==3072 and len(rows)==2443 and len({r['id'] for r in rows})==2443,'Discovery coverage differs')
    config=json.loads((root/'results/persona_preset_optimized_config.json').read_text());gate=json.loads((root/'results/persona_preset_runtime_gate.json').read_text())
    require(gate['state']=='qualified' and gate['config_sha256']==sha(root/'results/persona_preset_optimized_config.json'),'Qualified runtime config differs')
    binding={k:config['endpoints'][0][k] for k in ('model_identity','runtime_identity','prepared_identity')}
    require(len(config['endpoints'])==6 and all({k:e[k] for k in binding}==binding for e in config['endpoints']),'Six homogeneous qualified endpoints required')
    calibration_sha=sha(root/'results/layer_32.npz');program={'phase':'all','sites':[]}
    plan={'schema':'character-creation.perturbation-mapping-plan.v1','source_sha256':CORPUS_SHA,'source_manifest_sha256':MANIFEST_SHA,
        'binding':binding,'seed':SEED,'max_tokens':2048,'jobs':2443,'split':'discovery','heldout_excluded':629,
        'program_sha256':canonical_digest(program),'calibration_sha256':calibration_sha,'deadline':DEADLINE,'dispatch_deadline':DEADLINE-120,
        'priority':'Claim only when fixed foreground queues/viewer tables have no queued/running/submitting/stopping work; admitted batches may drain.',
        'purpose':'Baseline contextual association observations, not persona validation or causal neuron labels.'}
    plan['input_sha256']=canonical_digest(plan);jobs=[];corpus=[]
    for row in rows:
        require(row.get('system','')=='' and row['source']['kind']=='synthetic_original' and row.get('thinking',False) is False,'Only original neutral nonthinking discovery prompts allowed')
        require(hashlib.sha256(row['prompt'].encode()).hexdigest()==row['source']['prompt_sha256'],'Authored prompt hash differs')
        prompt=deepcopy(row);prompt.update(max_tokens=2048,thinking=False,system='');corpus.append(prompt)
        job={'schema':'character-creation.manager-job.v1','proposal_id':plan['input_sha256'],'model_identity':binding['model_identity'],
            'experiment_index':0,'experiment':{'layer':32,'label':'original_synthetic_baseline_map','measurement':'rollout'},'arm':'baseline','pushes':[],
            'phase':'all','seed':SEED,'repetition':0,'prompt':prompt,'pair_id':prompt['pair_id'],
            'source_group':{'corpus':prompt['corpus'],'pair_id':prompt['pair_id'],'split':'discovery'},'corpus_sha256':CORPUS_SHA,
            'calibration_sha256':calibration_sha,'program_sha256':canonical_digest(program),'measurement':'rollout','kind':'rollout','promotion_eligible':False}
        job['id']=canonical_digest(job);jobs.append(job)
    data=''.join(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n' for row in corpus)
    corpus_digest=hashlib.sha256(data.encode()).hexdigest()
    for job in jobs:
        job['corpus_sha256']=corpus_digest;job.pop('id');job['id']=canonical_digest(job)
    plan['job_ids_sha256']=canonical_digest([j['id'] for j in jobs])
    # Input identity describes the immutable scientific specification, while this
    # additional digest binds its concrete deterministic expansion.
    config=deepcopy(config);config.update(max_total_jobs=2443,corpus=str(root/'results/persona_perturbation_mapping_corpus.jsonl'),
        corpus_manifest=str(root/'results/persona_perturbation_mapping_corpus_manifest.json'))
    data=''.join(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n' for row in corpus)
    for path,value in [(root/'results/persona_perturbation_mapping_plan.json',plan),(root/'results/persona_perturbation_mapping_config.json',config)]:
        if path.exists():require(json.loads(path.read_text())==value,'Existing frozen mapping input differs')
        else:atomic_json(path,value)
    cp=Path(config['corpus'])
    if cp.exists():require(cp.read_text()==data,'Existing mapping corpus differs')
    else:cp.write_text(data)
    cm={'status':'complete','output_sha256':sha(cp),'source_sha256':CORPUS_SHA,'rows':2443,'heldout_excluded':629}
    cmp=Path(config['corpus_manifest'])
    if cmp.exists():require(json.loads(cmp.read_text())==cm,'Existing mapping corpus manifest differs')
    else:atomic_json(cmp,cm)
    return plan,jobs,config


def summary(database,plan,jobs,final=False):
    expected={j['id']:j for j in jobs}
    with closing(sqlite3.connect('file:'+quote(str(Path(database).resolve()),safe='/')+'?mode=ro',uri=True,timeout=10)) as db:
        db.execute('BEGIN');saved=db.execute('SELECT id,state,body FROM jobs ORDER BY id').fetchall()
        attempts=db.execute('SELECT job_id,number,result FROM attempts ORDER BY number').fetchall()
    latest={i:(n,raw) for i,n,raw in attempts};rows=[];counts={};tokens=0
    require(len(saved)==2443,'Full frozen queue coverage required')
    for identity,state,body in saved:
        require(identity in expected and json.loads(body)==expected[identity],'Queue scientific input changed')
        counts[state]=counts.get(state,0)+1
        if state not in TERMINAL:continue
        job=expected[identity];number,raw=latest.get(identity,(None,None));result=json.loads(raw) if raw else None
        row={'job_id':identity,'source_id':job['prompt']['id'],'source_group':job['pair_id'],'source_prompt_sha256':job['prompt']['source']['prompt_sha256'],
            'prompt':job['prompt']['prompt'],'state':state,'attempt':number,'answer':None,'capture_validated':False}
        valid=result and not result.get('error') and (result.get('complete') is True or result.get('capture_validated') is True)
        if valid:
            response=result['response'];request=result['request'];rh=canonical_digest(request)
            require(result['binding']==plan['binding'] and all(response[k]==v for k,v in plan['binding'].items()),'Completed response identity differs')
            require(request['messages']==[{'role':'user','content':job['prompt']['prompt']}] and request['seed']==SEED and request['max_tokens']==2048
                and not request.get('steering') and not request.get('kv_control'),'Baseline request changed')
            pointers={}
            for name in ('activation_artifact','provenance_artifact'):
                a=response[name];require(a['request_sha256']==rh and all(a[k]==v for k,v in plan['binding'].items()),'Capture pointer binding differs')
                pointers[name]={k:a[k] for k in ('sha256','request_sha256','model_identity','runtime_identity','prepared_identity')}
            rc=response['response_capture'];require(rc['request_sha256']==rh,'Response capture request differs')
            row.update(answer=response['choices'][0]['message']['content'],finish_reason=response['choices'][0]['finish_reason'],
                complete_answer=result['complete'],capture_validated=True,request_sha256=rh,response_sha256=canonical_digest(response),
                usage=response['usage'],capture=pointers,response_capture={k:rc[k] for k in ('consumed_generated_tokens','sampled_generated_tokens','mean_defined',
                    'generated_token_ids_sha256','consumed_token_ids_sha256','consumed_positions_sha256')})
            tokens+=response['usage']['completion_tokens']
        elif result:row['failure_class']=result.get('error_class','execution_or_validation')
        rows.append(row)
    terminal=sum(counts.get(k,0) for k in TERMINAL)
    return {'schema':'character-creation.perturbation-mapping-public.v1','input_sha256':plan['input_sha256'],'source_sha256':CORPUS_SHA,
        'source_manifest_sha256':MANIFEST_SHA,'binding':plan['binding'],'counts':counts,'terminal_jobs':terminal,'total_jobs':2443,'heldout_excluded':629,
        'generated_tokens_validated':tokens,'final':final or terminal==2443,'state':'complete' if terminal==2443 else 'deadline' if final else 'running',
        'rows':rows,'scope':'Original synthetic discovery prompts and their baseline generated answers; source-bound capture hashes only. Raw arrays are not published by this watcher. No behavioral or causal validation claim.'}


def upload(path,timeout):
    result=subprocess.run(['/home/orwel/.local/bin/hf','upload',REPO,str(path),PUBLIC,'--type','dataset','--format','json',
        '--commit-message','Update original synthetic discovery mapping'],capture_output=True,text=True,timeout=timeout,check=True)
    def strings(v):
        if isinstance(v,str):yield v
        elif isinstance(v,dict):
            for x in v.values():yield from strings(x)
        elif isinstance(v,list):
            for x in v:yield from strings(x)
    pattern=r'https://huggingface.co/datasets/'+re.escape(REPO)+r'/commit/([0-9a-f]{40})'
    revisions={m.group(1) for s in strings(json.loads(result.stdout)) if (m:=re.fullmatch(pattern,s))}
    require(len(revisions)==1,'Upload did not return one dataset commit');revision=revisions.pop()
    with urlopen(f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{PUBLIC}',timeout=60) as response:raw=response.read(Path(path).stat().st_size+1)
    require(hashlib.sha256(raw).hexdigest()==sha(path),'Pinned uploaded bytes differ')
    return {'revision':revision,'commit_url':f'https://huggingface.co/datasets/{REPO}/commit/{revision}','sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def publish(path,receipt_path,receipt,terminal,final,uploader=upload):
    if not publication_due(receipt,terminal,final):return
    attempt={'state':'intent','terminal_jobs':terminal,'final':final,'sha256':sha(path),'bytes':Path(path).stat().st_size,'source':receipt['source'],'started':time.time()}
    receipt['attempts'].append(attempt);receipt['state']='publishing';atomic_json(receipt_path,receipt)
    try:
        result=uploader(path,120);require(result['sha256']==attempt['sha256'] and result['bytes']==attempt['bytes'],'Upload snapshot bytes differ')
        attempt.update(result,state='verified');receipt['state']='verified'
    except Exception as exc:attempt.update(state='uncertain',error_type=type(exc).__name__);receipt['state']='uncertain'
    attempt['finished']=time.time();atomic_json(receipt_path,receipt)


def watch(root,database,plan,jobs,publishing):
    root=Path(root);receipt_path=root/'results/persona_perturbation_mapping_publication.json';path=root/'results'/PUBLIC
    source={'input_sha256':plan['input_sha256'],'job_ids_sha256':plan['job_ids_sha256'],'module_sha256':sha(__file__),'source_sha256':CORPUS_SHA}
    receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {'schema':'character-creation.perturbation-mapping-publication.v1','source':source,'state':'idle','attempts':[]}
    require(receipt['source']==source,'Publication source differs');previous=None
    while True:
        with closing(sqlite3.connect('file:'+quote(str(Path(database).resolve()),safe='/')+'?mode=ro',uri=True)) as db:counts=dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
        final=time.time()>=DEADLINE or sum(counts.get(k,0) for k in TERMINAL)==2443
        if counts!=previous or final:
            value=summary(database,plan,jobs,final);atomic_json(path,value);previous=counts
            if publishing:publish(path,receipt_path,receipt,value['terminal_jobs'],value['final'])
        if final:return
        time.sleep(min(15,max(0,DEADLINE-time.time())))


def run(root,publishing):
    root=Path(root);require(time.time()<DEADLINE-120,'Dispatch deadline already passed')
    plan,jobs,config=prepare(root);q=PriorityQueue(root/'results/persona_perturbation_mapping.sqlite',root)
    manager=Manager(config,q);q.enqueue({'schema':'character-creation.fixed-baseline-mapping.v1','input_sha256':plan['input_sha256']},jobs,limit=2443)
    launch=root/'results/persona_perturbation_mapping_launch.json';require(not launch.exists(),'Existing launch requires explicit recovery; no automatic uncertain replay')
    record={'schema':'character-creation.perturbation-mapping-launch.v1','state':'running','started':time.time(),'input_sha256':plan['input_sha256'],
        'config_sha256':sha(root/'results/persona_perturbation_mapping_config.json'),'module_sha256':sha(__file__),'manager_sha256':sha(root/'persona/persona_manager.py'),
        'deadline':DEADLINE,'dispatch_deadline':DEADLINE-120,'priority_databases':list(STUDIES),'interactive_tables':['batches','sculptor_campaigns']};atomic_json(launch,record)
    errors=[]
    def reporter():
        try:watch(root,q.path,plan,jobs,publishing)
        except Exception as exc:errors.append(repr(exc))
    thread=threading.Thread(target=reporter,name='mapping-publication',daemon=True);thread.start()
    try:manager.run(DEADLINE-120)
    finally:
        record.update(state='dispatch_stopped',counts=q.counts(),stopped=time.time());atomic_json(launch,record)
    while thread.is_alive() and time.time()<DEADLINE+185:thread.join(timeout=15)
    record.update(state='stopped',publication_errors=errors,finished=time.time());atomic_json(launch,record)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--publish',action='store_true');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    if a.prepare_only:prepare(a.root)
    else:run(a.root,a.publish)


if __name__=='__main__':main()
