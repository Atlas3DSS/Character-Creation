"""Pure fixed reasoning queue construction and read-only saved-result summaries.

No inference, adaptive heldout use or queue writes. Explicit publication contains only redacted summaries. Algebra source
text/answers remain local because its redistribution license is unspecified.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.parse import quote
from persona_vectors import canonical_digest
from persona_reasoning_battery import validate as validate_battery,request as battery_request,assess
from persona_batch_qualify import prepare_manager_pilot,ids_binding,require

ARMS=('baseline','A','A_reverse','B','B_reverse','random')
SEED=80317
MAX_TOKENS=2048


def build(battery,qualification_plan,bank,target,*,qualification_spec,compatibility_spec):
    """Prepare48questions×6qualified arms, preserving every source split.

    Reuses the existing qualified pilot gate without modifying its frozen code.
    All288jobs must be atomically enqueued before Manager.run with max_total_jobs288
    to avoid its adaptive planner path. The caller owns queue/config operations.
    """
    validate_battery(battery)
    pilot=prepare_manager_pilot(qualification_plan,bank,target,qualification_spec=qualification_spec,compatibility_spec=compatibility_spec)
    target=ids_binding(target);recipes={}
    for job in pilot['jobs']:
        candidate={k:job[k] for k in ('pushes','phase','program_sha256','compatibility_sha256','calibration_sha256')}
        if job['arm'] in recipes:require(recipes[job['arm']]==candidate,'Qualified recipes vary across scenes')
        recipes[job['arm']]=deepcopy(candidate)
    require(set(recipes)==set(ARMS),'Six qualified arms required')
    proposal={'schema':'character-creation.reasoning-sweep.v1','battery_sha256':battery['sha256'],'source_snapshot_sha256':battery['source_snapshot_sha256'],
        'qualification_plan_sha256':qualification_plan['input_sha256'],'qualification_sha256':qualification_spec['sha256'],'compatibility_sha256':compatibility_spec['sha256'],
        'binding':target,'seed':SEED,'max_tokens':MAX_TOKENS,'arms':list(ARMS),'cases':48,'jobs':288,
        'heldout_policy':'Frozen confirmation only; no adaptation, selection or promotion from heldout responses.'}
    proposal_id=canonical_digest(proposal);corpus=[];jobs=[]
    for case_index,case in enumerate(battery['cases']):
        native=battery_request(case,seed=SEED,max_tokens=MAX_TOKENS)
        require(len(native['messages'])==1 and native['messages'][0]['role']=='user','Battery request must be neutral single-user text')
        prompt={'schema':'character-creation.manager-corpus.v1','corpus':'reasoning_'+case['kind'],'category':case['kind'],'split':case['split'],
            'system':'','prompt':native['messages'][0]['content'],'max_tokens':MAX_TOKENS,'thinking':False,'pair_id':case['question_id'],
            'battery_case_id':case['id'],'battery_sha256':battery['sha256'],'source_row_sha256':case['source_row_sha256'],'source_rows':case['source_rows'],
            'source':deepcopy(case['source']),'scoring':case['scoring'],'labels':{'scenario_domain':case['kind']}}
        prompt['id']=canonical_digest(prompt);corpus.append(prompt)
        for arm in ARMS:
            recipe=recipes[arm]
            job={'schema':'character-creation.manager-job.v1','proposal_id':proposal_id,'model_identity':bank.model.identity,'experiment_index':case_index,
                'experiment':{'label':'reasoning_fixed_neural_arms','kind':'direction','layer':32,'dose':.1,'phase':recipe['phase'],'corpora':[prompt['corpus']],
                    'prompts_per_corpus':48,'repetitions':1,'measurement':'rollout'},
                'arm':arm,'pushes':deepcopy(recipe['pushes']),'phase':recipe['phase'],'program_sha256':recipe['program_sha256'],
                'compatibility_sha256':recipe['compatibility_sha256'],'calibration_sha256':recipe['calibration_sha256'],
                'seed':SEED,'repetition':0,'prompt':deepcopy(prompt),'pair_id':case['question_id'],
                'source_group':{'corpus':prompt['corpus'],'pair_id':case['question_id'],'split':case['split']},
                'battery_case_id':case['id'],'battery_sha256':battery['sha256'],'source_row_sha256':case['source_row_sha256'],
                'measurement':'rollout','kind':'rollout','target_binding':target,'promotion_eligible':False,
                'random_control_semantics':'Same qualified seeded residual random vector; fixed relative norm, no outcome selection'}
            jobs.append(job)
    corpus_bytes=''.join(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n' for row in corpus).encode()
    corpus_sha=hashlib.sha256(corpus_bytes).hexdigest()
    for job in jobs:job['corpus_sha256']=corpus_sha;job['id']=canonical_digest(job)
    require(len(jobs)==len({j['id'] for j in jobs})==288 and len(corpus)==48,'Exact question/arm coverage required')
    return {'schema':'character-creation.reasoning-queue-input.v1','proposal':proposal,'jobs':jobs,'corpus':corpus,'corpus_sha256':corpus_sha,
        'input_sha256':canonical_digest({'proposal':proposal,'jobs':jobs,'corpus':corpus}),
        'config_patch':{'max_total_jobs':288,'per_gpu_concurrency':1,'native_batch':{'enabled':True,'compatibility_receipt':deepcopy(compatibility_spec)}},
        'startup_requirement':'Separate target-bound queue, all288jobs atomically enqueued before Manager.run; merge artifact transport settings. No planner, no system/persona/KV prompt insertion.'}


def validate_input(prepared,battery):
    validate_battery(battery)
    require(prepared['input_sha256']==canonical_digest({k:prepared[k] for k in ('proposal','jobs','corpus')}),'Queue input hash differs')
    require(prepared['proposal']['battery_sha256']==battery['sha256'],'Battery changed')
    cases={c['id']:c for c in battery['cases']};jobs=prepared['jobs']
    require(len(jobs)==288 and len({j['id'] for j in jobs})==288,'Job coverage differs')
    require(Counter((j['battery_case_id'],j['arm']) for j in jobs)==Counter((i,a) for i in cases for a in ARMS),'Question/arm coverage differs')
    for job in jobs:
        case=cases[job['battery_case_id']];expected=battery_request(case,seed=SEED,max_tokens=MAX_TOKENS)
        require(job['id']==canonical_digest({k:v for k,v in job.items() if k!='id'}),'Job content hash differs')
        require(job['prompt']['split']==case['split'] and job['source_row_sha256']==case['source_row_sha256'] and job['prompt']['source']==case['source'],'Source split/binding changed')
        require(job['seed']==SEED and job['prompt']['max_tokens']==MAX_TOKENS and job['prompt']['system']=='' and job['prompt']['prompt']==expected['messages'][0]['content'],'Neutral fixed request differs')
        require(job['promotion_eligible'] is False,'Promotion must remain disabled')
    return cases


def summarize(database,prepared,battery):
    """Read actual latest Manager attempts in one snapshot; return local/public data.

    Only completed and source-bound saved answers are scored. No missing answer
    is fabricated; unrun jobs remain not_evaluated. Source/reference checks are
    distinct from model answer correctness, and general tasks remain unscored.
    """
    cases=validate_input(prepared,battery);jobs={j['id']:j for j in prepared['jobs']};target=prepared['proposal']['binding']
    uri='file:'+quote(str(Path(database).resolve()))+'?mode=ro'
    with sqlite3.connect(uri,uri=True) as db:
        db.execute('BEGIN')
        placeholders=','.join('?' for _ in jobs)
        stored=db.execute(f'SELECT id,state,body FROM jobs WHERE id IN ({placeholders})',list(jobs)).fetchall()
        attempts=db.execute(f'SELECT job_id,number,state,result FROM attempts WHERE job_id IN ({placeholders}) ORDER BY job_id,number',list(jobs)).fetchall()
    latest={r[0]:r[1:] for r in attempts};saved={r[0]:r[1:] for r in stored};rows=[]
    usage={'logical_prompt_tokens':0,'generated_tokens':0,'consumed_generated_tokens':0,'responses_with_usage':0,'complete_captures':0,'incomplete_captures':0}
    for identity,job in jobs.items():
        case=cases[job['battery_case_id']];state,body=saved.get(identity,('not_enqueued',None))
        if body is not None:require(json.loads(body)==job,'Persisted Manager job differs from frozen input')
        number,attempt_state,raw=latest.get(identity,(None,None,None));result=json.loads(raw) if raw else None
        assessment={'state':'not_evaluated','reason':'No completed validated Manager response'};answer=None;response=None
        validated_partial=bool(result and state=='incomplete' and result.get('capture_validated') is True and result.get('generation_status')=='length_limited')
        if result and (state=='complete' or validated_partial):
            require(result.get('complete') is (state=='complete') and result.get('error') is None and result.get('binding')==target,'Validated result binding or completion differs')
            response=result['response'];request=result['request'];expected_messages=[{'role':'user','content':job['prompt']['prompt']}]
            require(ids_binding(response)==target and request['messages']==expected_messages and request['seed']==SEED and request['max_tokens']==MAX_TOKENS
                and not request.get('kv_control') and not request.get('observation_only'),'Saved inference request differs')
            require(canonical_digest(request.get('steering',{'phase':job['phase'],'sites':[]}))==job['program_sha256']==result['program_sha256'],'Saved exact neural program differs')
            request_hash=canonical_digest(request)
            for name in ('activation_artifact','provenance_artifact'):
                pointer=response[name];require(pointer['request_sha256']==request_hash and ids_binding(pointer)==target,'Saved artifact pointer binding differs')
            if validated_partial:
                require(response['diagnostics']['complete_final_answer'] is False and response['choices'][0]['finish_reason']=='length'
                    and response['usage']['completion_tokens']==request['max_tokens']
                    and response['response_capture']['generation_complete'] is False,'Length-limited capture termination differs')
                assessment={'state':'unscorable' if case['kind']=='algebra' else 'unscored',
                    'reason':'Validated length-limited capture; partial answer retained locally, never correctness-scored'}
            else:
                require(response['diagnostics']['complete_final_answer'] is True and response['choices'][0]['finish_reason']=='stop','Incomplete prose cannot be scored as a completed run')
                assessment=assess(case,response['choices'][0]['message']['content'])
            answer=response['choices'][0]['message']['content']
            usage['incomplete_captures' if validated_partial else 'complete_captures']+=1
            if 'usage' in response and 'response_capture' in response:
                values=(response['usage']['prompt_tokens'],response['usage']['completion_tokens'],response['response_capture']['consumed_generated_tokens'])
                require(all(type(v) is int and v>=0 for v in values),'Invalid saved token usage')
                for key,value in zip(('logical_prompt_tokens','generated_tokens','consumed_generated_tokens'),values):usage[key]+=value
                usage['responses_with_usage']+=1
        elif result and state in ('failed','incomplete'):
            assessment={'state':'unscorable' if case['kind']=='algebra' else 'unscored','reason':'Execution or validation did not complete; no correctness claim'}
        row={'job_id':identity,'case_id':case['id'],'question_id':case['question_id'],'kind':case['kind'],'split':case['split'],'arm':job['arm'],
            'source_row_sha256':case['source_row_sha256'],'prompt_sha256':hashlib.sha256(job['prompt']['prompt'].encode()).hexdigest(),
            'program_sha256':job['program_sha256'],'state':state,'attempt':number,'attempt_state':attempt_state,'assessment':assessment,
            'response_sha256':canonical_digest(response) if response else None,'answer':answer,'expected_answer':case.get('expected_answer')}
        rows.append(row)
    counts={}
    for kind in ('algebra','general'):
        counts[kind]={}
        for split in ('discovery','heldout'):
            counts[kind][split]={}
            for arm in ARMS:
                subset=[r for r in rows if (r['kind'],r['split'],r['arm'])==(kind,split,arm)]
                counts[kind][split][arm]={'total':len(subset),'execution':dict(Counter(r['state'] for r in subset)),
                    'assessment':{s:sum(r['assessment']['state']==s for r in subset) for s in ('correct','incorrect','unscorable','unscored','not_evaluated')}}
    local={'schema':'character-creation.reasoning-results.v1','input_sha256':prepared['input_sha256'],'battery_sha256':battery['sha256'],'binding':target,
        'counts':counts,'rows':rows,'heldout_policy':prepared['proposal']['heldout_policy'],'limitations':['Exact numeric equality to supplied source answers only; not independent proof that reference answers are correct.',
            'General prompts have no correctness supervision and remain unscored.','Execution failures and unscorable numeric outputs are reported separately from incorrect numeric answers.',
            'Heldout results are confirmation only; this helper performs no adaptation or selection.']}
    states=Counter(r['state'] for r in rows)
    local['state']='complete' if states['complete']==len(rows) else 'partial' if all(r['state'] in TERMINAL_STATES for r in rows) else 'running'
    local['usage']=usage
    local['usage_scope']='Validated complete and length-limited captures only; correctness is never scored on partial answers. Capture counts are separate; missing usage is excluded and counted by responses_with_usage. No GPU throughput claim.'
    return local,public_summary(local)


def public_summary(local):
    """Explicit allowlist: no questions, answers, source templates or local paths."""
    fields=('job_id','case_id','question_id','kind','split','arm','source_row_sha256','prompt_sha256','program_sha256','state','attempt','response_sha256')
    return {'schema':'character-creation.reasoning-public-summary.v1','input_sha256':local['input_sha256'],'battery_sha256':local['battery_sha256'],
        'binding':local['binding'],'counts':deepcopy(local['counts']),'heldout_policy':local['heldout_policy'],'limitations':list(local['limitations']),
        'state':local['state'],'usage':deepcopy(local['usage']),'usage_scope':local['usage_scope'],
        'rows':[{**{k:r[k] for k in fields},'assessment_state':r['assessment']['state']} for r in local['rows']],
        'publication_scope':'IDs, hashes and aggregate outcomes only. Algebra source question/answer/template text and generated answers are deliberately omitted.'}


# Publication is explicit and limited to the redacted summary, never saved answers.
REPOSITORY='Atlas3D/character-creation-big-five'
PUBLIC_FILE='persona_reasoning_public.json'
TERMINAL_STATES=frozenset(('complete','failed','incomplete'))


def atomic_json(path,value):
    import os
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    directory=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)


def publication_due(receipt,terminal,final):
    if terminal<=0 or any(a['state']!='verified' for a in receipt.get('attempts',[])):return False
    attempts=receipt.get('attempts',[])
    previous=max((a['terminal_jobs'] for a in attempts),default=0)
    return (terminal>previous and terminal//48>previous//48) or (final and terminal>=previous and not any(a.get('final') and a['terminal_jobs']==terminal for a in attempts))


def upload_summary(path,timeout):
    """One upload attempt, then byte verification at its immutable commit."""
    import subprocess
    import re
    from urllib.request import urlopen
    result=subprocess.run(['/home/orwel/.local/bin/hf','upload',REPOSITORY,str(path),PUBLIC_FILE,
        '--type','dataset','--format','json','--commit-message','Update fixed reasoning sweep summary'],
        capture_output=True,text=True,timeout=timeout,check=True)
    output=json.loads(result.stdout)
    # CLI versions expose either commit_url or a nested CommitInfo object.
    def strings(value):
        if isinstance(value,str):yield value
        elif isinstance(value,dict):
            for child in value.values():yield from strings(child)
        elif isinstance(value,list):
            for child in value:yield from strings(child)
    pattern=r'https://huggingface.co/datasets/'+re.escape(REPOSITORY)+r'/commit/([0-9a-f]{40})'
    matches=[re.fullmatch(pattern,s) for s in strings(output)]
    revisions={m.group(1) for m in matches if m}
    require(len(revisions)==1,'Upload did not return one exact dataset commit URL')
    revision=revisions.pop()
    with urlopen(f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{PUBLIC_FILE}',timeout=min(timeout,60)) as response:
        data=response.read(Path(path).stat().st_size+1)
    return {'revision':revision,'commit_url':f'https://huggingface.co/datasets/{REPOSITORY}/commit/{revision}',
        'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}


def publish_snapshot(path,receipt_path,receipt,terminal,final,timeout=120,upload=None):
    """Intent survives crashes; any ambiguous attempt permanently disables retries."""
    import time
    if not publication_due(receipt,terminal,final):return False
    data=Path(path).read_bytes()
    attempt={'state':'intent','terminal_jobs':terminal,'final':bool(final),'started':time.time(),
        'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'source':deepcopy(receipt['source'])}
    receipt['attempts'].append(attempt);receipt['state']='publishing';atomic_json(receipt_path,receipt)
    try:
        evidence=(upload or upload_summary)(path,timeout)
        require(evidence['sha256']==attempt['sha256'] and evidence['bytes']==attempt['bytes'],'Pinned uploaded bytes differ')
        attempt.update(evidence);attempt['state']='verified';receipt['state']='verified'
    except Exception as exc:
        # Even a CLI error can follow a successful remote commit. Never replay it.
        attempt['state']='uncertain';attempt['error_type']=type(exc).__name__;receipt['state']='uncertain'
    attempt['finished']=time.time();atomic_json(receipt_path,receipt)
    return attempt['state']=='verified'


def watch(root,until,*,continuous=False,publish=False):
    import math
    import time
    import fcntl
    from persona_reasoning_battery import load
    require(math.isfinite(until) and until>0,'A finite positive deadline is required')
    root=Path(root).resolve();results=root/'results'
    # One writer prevents concurrent uploads and ambiguous receipt ownership.
    with (results/'persona_reasoning_sweep.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        paths={name:root/path for name,path in {
            'input':'results/persona_reasoning_input.json','config':'results/persona_reasoning_config.json',
            'launch':'results/persona_reasoning_launch.json','battery':'data/persona_reasoning_battery.json',
            'sources':'data/persona_reasoning_sources.json'}.items()}
        hashes={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in paths.items()}
        prepared=json.loads(paths['input'].read_text());battery=load(paths['battery']);validate_input(prepared,battery)
        launch=json.loads(paths['launch'].read_text())
        require(launch['input_sha256']==hashes['input'] and launch['config_sha256']==hashes['config'],'Launch input/config file binding differs')
        source={**hashes,'input_identity':prepared['input_sha256'],'battery_identity':battery['sha256'],
            'watcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        receipt_path=results/'persona_reasoning_sweep_publication.json'
        receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {
            'schema':'character-creation.reasoning-sweep-publication.v1','repository':REPOSITORY,'path':PUBLIC_FILE,
            'source':source,'state':'idle','attempts':[]}
        require(receipt['source']==source and receipt['repository']==REPOSITORY and receipt['path']==PUBLIC_FILE,'Publication receipt source differs')
        if any(a['state']!='verified' for a in receipt['attempts']):
            receipt['state']='uncertain';atomic_json(receipt_path,receipt)
        while True:
            require(all(hashlib.sha256(path.read_bytes()).hexdigest()==hashes[key] for key,path in paths.items()),'Frozen watcher inputs changed')
            local,public=summarize(results/'persona_reasoning.sqlite',prepared,battery)
            terminal=sum(r['state'] in TERMINAL_STATES for r in local['rows'])
            now=time.time();final=terminal==len(prepared['jobs']) or now>=until
            status={'terminal_jobs':terminal,'total_jobs':len(prepared['jobs']),'final':final,
                'stop_reason':'all_terminal' if terminal==len(prepared['jobs']) else 'deadline' if final else None}
            local['watch_status']=status;public['watch_status']=status
            atomic_json(results/'persona_reasoning_results.json',local);atomic_json(results/PUBLIC_FILE,public)
            if publish:publish_snapshot(results/PUBLIC_FILE,receipt_path,receipt,terminal,final,timeout=120)
            print(json.dumps({**status,'publication_state':receipt['state']}),flush=True)
            if final or not continuous:return status
            time.sleep(min(15,max(0,until-time.time())))


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Finite read-only reasoning summaries; optional sanitized Hub publication.')
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--until',type=float,required=True)
    parser.add_argument('--watch',action='store_true');parser.add_argument('--publish',action='store_true')
    args=parser.parse_args();watch(args.root,args.until,continuous=args.watch,publish=args.publish)


if __name__=='__main__':main()
