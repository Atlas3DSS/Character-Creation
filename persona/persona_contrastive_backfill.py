"""Fixed, low-priority discovery backfill; never promotes recipes or calls a judge."""
import argparse
from collections import defaultdict
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from persona_perturbation_mapping import PriorityQueue,CORPUS_SHA,MANIFEST_SHA,DEADLINE,TERMINAL,sha,require
from persona_manager import Manager
from persona_vectors import canonical_digest,VectorBank,ModelSpec
from persona_preset_overnight import attach_column
from persona_reasoning_sweep import atomic_json

PREFIX='persona_contrastive_backfill'
SEED=80411
ARMS=('candidate','reversal','random')

def choose(rows,plans,limit=512):
    excluded_pairs={c['pair_id'] for p in plans for stage in ('discovery','confirmation') for c in p[stage]}
    excluded_text={c['prompt'] for p in plans for stage in ('discovery','confirmation') for c in p[stage]}
    buckets=defaultdict(list);seen_pairs=set();seen_text=set()
    for row in sorted(rows,key=lambda r:r['id']):
        if row['split']!='discovery' or row['pair_id'] in excluded_pairs or row['prompt'] in excluded_text:continue
        require(row.get('system','')=='' and row.get('thinking',False) is False and row['source']['kind']=='synthetic_original','Non-neutral/private prompt')
        require(hashlib.sha256(row['prompt'].encode()).hexdigest()==row['source']['prompt_sha256'],'Prompt source hash differs')
        if row['pair_id'] in seen_pairs or row['prompt'] in seen_text:continue
        seen_pairs.add(row['pair_id']);seen_text.add(row['prompt']);buckets[row['category']].append(row)
    selected=[]
    while buckets and len(selected)<limit:
        for domain in sorted(list(buckets)):
            if len(selected)>=limit:break
            selected.append(buckets[domain].pop(0))
            if not buckets[domain]:del buckets[domain]
    return selected

def prototypes(rows,bank,binding):
    chosen={};program_cache={}
    for job in rows:
        require(canonical_digest({k:v for k,v in job.items() if k!='id'})==job['id'],'Source job ID differs')
        require(job['model_identity']==binding['model_identity'],'Source job model differs')
        if not job['recipe_id'].endswith('-distributed') or job['seed']!=SEED or job['arm'] not in ('baseline',*ARMS):continue
        key=(job['recipe_id'],job['arm']);spec=canonical_digest({'pushes':job['pushes'],'phase':job['phase']})
        if spec not in program_cache:program_cache[spec]=canonical_digest(bank.program(job['pushes'],job['phase'])[0])
        require(job['program_sha256']==program_cache[spec],'Prototype program differs')
        require(not job['promotion_eligible'] and job['prompt']['split']=='discovery','Prototype not discovery-only')
        if key in chosen:
            require(all(chosen[key][k]==job[k] for k in ('pushes','phase','program_sha256','calibration_sha256','seed')),'Conflicting prototype')
        else:chosen[key]=job
    recipes=sorted({k[0] for k in chosen});require(len(recipes)==8 and len(chosen)==32,'Expected eight complete distributed recipes')
    return chosen

def expand(prompts,proto,input_sha,corpus_sha):
    recipes=sorted({k[0] for k in proto});baseline=next(v for (r,a),v in sorted(proto.items()) if a=='baseline' and v['phase']=='all')
    jobs=[]
    for prompt in prompts:
        variants=[('shared_baseline','baseline',baseline)]+[(r,a,proto[r,a]) for r in recipes for a in ARMS]
        baseline_id=None
        for recipe,arm,source in variants:
            job=deepcopy(source);job.pop('id');job.update(proposal_id=input_sha,prompt=deepcopy(prompt),pair_id=prompt['pair_id'],
                source_group={'corpus':prompt['corpus'],'pair_id':prompt['pair_id'],'split':'discovery'},corpus_sha256=corpus_sha,
                recipe_id=recipe,arm=arm,study_stage='research_backfill',promotion_eligible=False,prototype_job_id=source['id'],
                shared_baseline_job_id=baseline_id,seed=SEED)
            job['id']=canonical_digest(job)
            if arm=='baseline':baseline_id=job['id']
            jobs.append(job)
    return jobs

def immutable(path,value):
    if path.exists():require(json.loads(path.read_text())==value,'Frozen backfill file changed: '+path.name)
    else:atomic_json(path,value)

def prepare(root):
    root=Path(root);results=root/'results';corpus=root/'data/persona_perturbation_prompts.jsonl';manifest=root/'data/persona_perturbation_prompts_manifest.json'
    require(sha(corpus)==CORPUS_SHA and sha(manifest)==MANIFEST_SHA,'Frozen Terra corpus differs')
    mp=json.loads(manifest.read_text());require(mp['status']=='complete' and mp['split_counts']=={'discovery':2443,'heldout':629},'Corpus not complete')
    paths=[results/'persona_preset_overnight_plan.json',results/'persona_preset_followon_plan.json'];plans=[json.loads(p.read_text()) for p in paths]
    config=json.loads((results/'persona_preset_optimized_config.json').read_text());binding={k:config['endpoints'][0][k] for k in ('model_identity','runtime_identity','prepared_identity')}
    require(len(config['endpoints'])==6 and all(all(e[k]==v for k,v in binding.items()) for e in config['endpoints']),'Six homogeneous endpoints required')
    gate=json.loads((results/'persona_preset_runtime_gate.json').read_text());require(gate['state']=='qualified' and gate['config_sha256']==sha(results/'persona_preset_optimized_config.json'),'Runtime gate differs')
    database=results/'persona_preset_followon_discovery.sqlite'
    with closing(sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True)) as db:
        db.execute('BEGIN');identity=json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0]);rows=[json.loads(r[0]) for r in db.execute('SELECT body FROM jobs ORDER BY id')]
    require(identity==binding,'Prototype database runtime binding differs')
    bank=attach_column(VectorBank(results,ModelSpec.load(root/'data/persona_model.json')),results/'persona_preset_columns.json')
    proto=prototypes(rows,bank,binding)
    prompts=choose([json.loads(s) for s in corpus.read_text().splitlines()],plans)
    for p in prompts:
        p['source_row_id']=p['id'];p['max_tokens']=2048;p['id']=canonical_digest({k:v for k,v in p.items() if k!='id'})
    data=''.join(json.dumps(p,sort_keys=True,separators=(',',':'))+'\n' for p in prompts);corpus_sha=hashlib.sha256(data.encode()).hexdigest()
    recipe_by_id={r['id']:r for r in plans[1]['recipes']}
    plan={'schema':'character-creation.contrastive-backfill-plan.v1','binding':binding,'source_sha256':CORPUS_SHA,'source_manifest_sha256':MANIFEST_SHA,
        'excluded_plan_sha256':{p.name:sha(p) for p in paths},'prototype_jobs_sha256':canonical_digest(rows),
        'prototype_ids':sorted(j['id'] for j in proto.values()),'recipes':{r:recipe_by_id[r] for r in sorted({k[0] for k in proto})},
        'corpus_sha256':corpus_sha,'prompts':len(prompts),'jobs':len(prompts)*25,'seed':SEED,'max_tokens':2048,
        'deadline':DEADLINE,'dispatch_deadline':DEADLINE-max(660,int(config.get('request_timeout_seconds',600))+60),'promotion_eligible':False,'heldout_excluded':629,
        'source_code_sha256':{n:sha(root/'persona'/n) for n in ('persona_contrastive_backfill.py','persona_perturbation_mapping.py','persona_manager.py')},
        'control_semantics':'One shared same-prompt/seed baseline; each of eight fixed recipes has candidate, sign reversal, and same-layer/site calibrated-norm seeded random control. Experimental research, not persona qualification.'}
    plan['input_sha256']=canonical_digest(plan);jobs=expand(prompts,proto,plan['input_sha256'],corpus_sha);plan['job_ids_sha256']=canonical_digest([j['id'] for j in jobs])
    config.update(max_total_jobs=len(jobs),corpus=str(results/(PREFIX+'_corpus.jsonl')),corpus_manifest=str(results/(PREFIX+'_corpus_manifest.json')))
    cp=results/(PREFIX+'_corpus.jsonl')
    if cp.exists():require(cp.read_text()==data,'Selected corpus changed')
    else:cp.write_text(data)
    for name,value in (('plan',plan),('jobs',jobs),('config',config),('corpus_manifest',{'status':'complete','output_sha256':corpus_sha,'input_sha256':plan['input_sha256'],'records':len(prompts)})):immutable(results/(PREFIX+'_'+name+'.json'),value)
    return plan,jobs,config

def public_snapshot(database,plan,jobs,final=False):
    expected={j['id']:j for j in jobs};out=[];counts={}
    with closing(sqlite3.connect('file:'+str(Path(database).resolve())+'?mode=ro',uri=True)) as db:
        db.execute('BEGIN');saved=db.execute('SELECT id,state,body FROM jobs ORDER BY id').fetchall();attempts=db.execute('SELECT job_id,number,result FROM attempts ORDER BY number').fetchall()
    require(len(saved)==len(jobs),'Queue coverage differs');latest={i:(n,r) for i,n,r in attempts}
    for identity,state,body in saved:
        require(identity in expected and json.loads(body)==expected[identity],'Frozen job changed');j=expected[identity];counts[state]=counts.get(state,0)+1
        if state not in TERMINAL:continue
        n,raw=latest.get(identity,(None,None));result=json.loads(raw) if raw else None
        row={'job_id':identity,'source_id':j['prompt']['source_row_id'],'source_prompt_sha256':j['prompt']['source']['prompt_sha256'],'prompt':j['prompt']['prompt'],
            'recipe_id':j['recipe_id'],'arm':j['arm'],'shared_baseline_job_id':j['shared_baseline_job_id'] or identity,'seed':j['seed'],
            'program_sha256':j['program_sha256'],'pushes':j['pushes'],'phase':j['phase'],'state':state,'attempt':n,'answer':None,'complete_answer':False,'capture_validated':False}
        if result and not result.get('error') and (result.get('complete') is True or result.get('capture_validated') is True):
            response=result['response'];request=result['request'];rh=canonical_digest(request)
            require(result['binding']==plan['binding'] and all(response[k]==v for k,v in plan['binding'].items()),'Response binding differs')
            require(request['messages']==[{'role':'user','content':j['prompt']['prompt']}] and request['seed']==SEED and request['max_tokens']==2048 and not request.get('kv_control'),'Response request changed')
            actual=request.get('steering',{'phase':j['phase'],'sites':[]})
            require(canonical_digest(actual)==j['program_sha256'],'Actual intervention differs')
            pointers={}
            for key in ('activation_artifact','provenance_artifact'):
                a=response[key];require(a['request_sha256']==rh and all(a[k]==v for k,v in plan['binding'].items()),'Artifact binding differs')
                pointers[key]={k:a[k] for k in ('sha256','request_sha256','model_identity','runtime_identity','prepared_identity')}
            rc=response['response_capture'];require(rc['request_sha256']==rh,'Response capture request differs')
            row.update(answer=response['choices'][0]['message']['content'],finish_reason=response['choices'][0]['finish_reason'],complete_answer=result['complete'],
                capture_validated=True,usage=response['usage'],request_sha256=rh,response_sha256=canonical_digest(response),capture=pointers,
                response_capture={k:rc[k] for k in ('consumed_generated_tokens','sampled_generated_tokens','mean_defined','generated_token_ids_sha256','consumed_token_ids_sha256','consumed_positions_sha256')})
        out.append(row)
    terminal=sum(counts.get(s,0) for s in TERMINAL);done=terminal==len(jobs)
    return {'schema':'character-creation.contrastive-backfill-public.v1','input_sha256':plan['input_sha256'],'source_sha256':CORPUS_SHA,'source_manifest_sha256':MANIFEST_SHA,
        'binding':plan['binding'],'counts':counts,'terminal_jobs':terminal,'total_jobs':len(jobs),'final':final or done,'state':'complete' if done else 'deadline' if final else 'running',
        'rows':out,'scope':plan['control_semantics'],'raw_arrays_included':False,'promotion_eligible':False,'dispatch_deadline':plan['dispatch_deadline'],'deadline':DEADLINE}

def run(root):
    root=Path(root);plan,jobs,config=prepare(root);require(time.time()<plan['dispatch_deadline'],'No new dispatch window')
    launch=root/'results'/f'{PREFIX}_launch.json';require(not launch.exists(),'Existing launch requires explicit recovery; no uncertain replay')
    queue=PriorityQueue(root/'results'/f'{PREFIX}.sqlite',root);manager=Manager(config,queue)
    queue.enqueue({'schema':'character-creation.fixed-contrastive-backfill.v1','input_sha256':plan['input_sha256']},jobs,limit=len(jobs))
    record={'state':'running','source_sha256':sha(__file__),'input_sha256':plan['input_sha256'],'started':time.time(),'dispatch_deadline':plan['dispatch_deadline'],'deadline':DEADLINE}
    atomic_json(launch,record);done=threading.Event();errors=[]
    def reporter():
        previous=None
        while True:
            counts=queue.counts();final=done.is_set() or time.time()>=DEADLINE
            try:
                if counts!=previous or final:
                    atomic_json(root/'results'/f'{PREFIX}_public.json',public_snapshot(queue.path,plan,jobs,final));previous=counts
            except Exception as exc:
                errors.append(type(exc).__name__);atomic_json(root/'results'/f'{PREFIX}_reporter_status.json',{'state':'blocked','error':str(exc),'time':time.time()})
            if final:return
            done.wait(15)
    thread=threading.Thread(target=reporter,daemon=True);thread.start()
    try:manager.run(plan['dispatch_deadline'])
    finally:
        done.set();thread.join(timeout=60);record.update(state='stopped',counts=queue.counts(),finished=time.time(),reporter_errors=errors);atomic_json(launch,record)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    if a.prepare_only:prepare(a.root)
    else:run(a.root)
