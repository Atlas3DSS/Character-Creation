"""Finite same-engine host-capture comparison; no serving, queues or leases created.

Use `plan` offline, then `run` with the standard probe arguments and --performance-plan.
Only the operator launches GPU work. Timers are host elapsed, not kernel-only CUDA time.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import time
import uuid
import numpy as np
from persona_vectors import canonical_digest,ModelSpec,VectorBank
from persona_batch_qualify import load_plan,request_for,checked_parity,EXACT
from persona_batch_probe import sha,atomic

INDICES=tuple(range(24))
GROUPS=((0,),(1,),(2,3),(6,7,18,19),(4,5,8,9),(10,11,12,13),(14,15,16,17),(20,21,22,23))
SOURCE_FILES=('persona_batch_server.py','persona_batch_capture.py','persona_batch_probe.py','persona_batch_qualify.py','persona_probe.py','persona_vectors.py','persona_batch_performance.py')
POLICY={'widths':[1,2,4],'repeats':2,'orders':[['baseline','optimized'],['optimized','baseline']],
    'max_wall_seconds':1200,'exact_arrays':list(EXACT),'trace_ignored_columns':[0,1],
    'request_ignored_fields':['capture_response.id','metadata.benchmark_id'],'automatic_rollout':False}


def scientific_request(request):
    value=deepcopy(request)
    value.get('capture_response',{}).pop('id',None)
    value.get('metadata',{}).pop('benchmark_id',None)
    return value


def compare_observations(left,right):
    if scientific_request(left['request'])!=scientific_request(right['request']):raise ValueError('Scientific request differs')
    if left['evidence']['binding']!=right['evidence']['binding']:raise ValueError('Same-engine identity differs')
    for name in EXACT:
        a,b=left['arrays'][name],right['arrays'][name]
        if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():raise ValueError('Exact array differs: '+name)
    a,b=left['arrays']['completion_trace'],right['arrays']['completion_trace']
    if a.dtype!=b.dtype or a.shape!=b.shape or not np.array_equal(a[:,2:],b[:,2:]):raise ValueError('Native completion/token/RNG trace differs')
    if not np.array_equal(left['arrays']['completion_graph_kind'],right['arrays']['completion_graph_kind']):raise ValueError('Native graph-kind trace differs')
    if left['response']['choices']!=right['response']['choices']:raise ValueError('Answer/termination differs')
    return {'exact':True,'left_request_sha256':left['evidence']['request_sha256'],'right_request_sha256':right['evidence']['request_sha256'],
        'left_artifact_sha256':left['evidence']['artifact_sha256'],'right_artifact_sha256':right['evidence']['artifact_sha256'],
        'prose_complete':left['evidence']['prose_complete'] and right['evidence']['prose_complete']}


def schedule(plan):
    runs=[]
    for index,group in enumerate(GROUPS):
        for mode in ('baseline','optimized'):
            runs.append({'id':f'g{index}-r0-{mode}','width':len(group),'repeat':0,'mode':mode,'job_ids':[plan['jobs'][i]['id'] for i in group]})
    for mode in ('optimized','baseline'):
        runs.append({'id':f'g7-r1-{mode}','width':4,'repeat':1,'mode':mode,'job_ids':[plan['jobs'][i]['id'] for i in GROUPS[-1]]})
    return runs


def freeze(root,source_path,compatibility_path):
    source=load_plan(source_path);compat=json.loads(Path(compatibility_path).read_text())
    if compat.get('state')!='complete' or compat.get('passed') is not True or compat.get('schema')!='character-creation.batch-runtime-compatibility.v1':raise ValueError('Completed core compatibility receipt required')
    model=ModelSpec.load(root/'data/persona_model.json');bank=VectorBank(root/'results',model)
    jobs=[deepcopy(source['jobs'][i]) for i in INDICES]
    if len({j['arm'] for j in jobs})!=6 or any(sum(x['arm']==j['arm'] for x in jobs)!=4 for j in jobs):raise ValueError('Four cases per six arms required')
    requests={}
    for job in jobs:
        request,proof=request_for(job,bank)
        if any(p.get('kind')=='direction' for p in job['recipe']['pushes']) and proof['program_sha256'] not in compat['allowed_program_sha256']:raise ValueError('Unqualified original direction program')
        requests[job['id']]=request
    plan={'schema':'character-creation.batch-performance-plan.v1','policy':POLICY,'jobs':jobs,'requests':requests,
        'source_plan':source,'source_plan_sha256':sha(source_path),'compatibility_sha256':sha(compatibility_path),
        'expected_core_source_hashes':compat['new_core_source_hashes'],'model_identity':model.identity,'prepared_identity':compat['new_binding']['prepared_identity'],
        'direction_artifact':source['direction_artifact'],'source_sha256':{name:sha(Path(__file__).with_name(name)) for name in SOURCE_FILES},
        'planned_responses':56,'maximum_sampled_tokens':sum(j['max_tokens'] for j in jobs)*2+sum(jobs[i]['max_tokens'] for i in GROUPS[-1])*2,
        'timing_claim':'Host elapsed graph+existing CUDA synchronization, capture validation/readback/accumulation, and artifact serialization. Not kernel-only timing.',
        'deployment': 'Finite callback runtime; no production default promotion; same engine/weights/core for both modes.'}
    plan['input_sha256']=canonical_digest(plan);return plan


def validate_plan(plan):
    if plan.get('schema')!='character-creation.batch-performance-plan.v1' or plan.get('policy')!=POLICY or plan.get('input_sha256')!=canonical_digest({k:v for k,v in plan.items() if k!='input_sha256'}):raise ValueError('Frozen performance plan differs')
    if [j['id'] for j in plan['jobs']]!=[plan['source_plan']['jobs'][i]['id'] for i in INDICES] or plan['jobs']!=[plan['source_plan']['jobs'][i] for i in INDICES]:raise ValueError('Fixed source cases changed')
    if plan['planned_responses']!=56 or plan['maximum_sampled_tokens']!=sum(j['max_tokens'] for j in plan['jobs'])*2+sum(plan['jobs'][i]['max_tokens'] for i in GROUPS[-1])*2:raise ValueError('Attempt/token budget differs')
    for name,digest in plan['source_sha256'].items():
        if name not in SOURCE_FILES or sha(Path(__file__).with_name(name))!=digest:raise ValueError('Frozen implementation changed: '+name)
    if set(plan['source_sha256'])!=set(SOURCE_FILES):raise ValueError('Source coverage incomplete')
    return plan


class LocalStore:
    def __init__(self,root):self.root=Path(root).resolve()
    def get(self,pointer):
        path=Path(pointer['path'])
        if path.parent.resolve()!=self.root or not path.is_file() or sha(path)!=pointer['sha256']:raise ValueError('Local attempt artifact binding differs')
        return path


def experiment(engine,args):
    from persona_batch_server import BatchEngine,binding
    plan=validate_plan(json.loads(args.performance_plan.read_text()));output=args.output/'persona_batch_performance.json'
    journal=args.output/'persona_batch_performance.attempts.jsonl'
    if output.exists() or journal.exists():raise FileExistsError('Existing attempt evidence retained; no automatic retry')
    actual_core={**engine.manifest['source_sha256'],'heliopause':engine.manifest['heliopause_sha256']}
    if actual_core!=plan['expected_core_source_hashes'] or engine.model.identity!=plan['model_identity'] or engine.prepared.server_config['identity']!=plan['prepared_identity']:raise ValueError('Native model/core/prepared identity differs')
    if sha(args.output/plan['direction_artifact']['filename'])!=plan['direction_artifact']['sha256']:raise ValueError('Direction bytes changed')
    bank=VectorBank(args.output,engine.model)
    for job in plan['jobs']:
        if request_for(job,bank)[0]!=plan['requests'][job['id']]:raise ValueError('Frozen native payload/program differs from source case')
    report={'schema':'character-creation.batch-performance.v1','state':'running','passed':False,'plan':plan,'plan_sha256':sha(args.performance_plan),'binding':binding(engine),
        'started':time.time(),'runs':[],'comparisons':[],'holes_observed':False,'automatic_rollout':False}
    atomic(output,report);journal.touch(exist_ok=False)
    store=LocalStore(args.output);backends={mode:BatchEngine(engine,capture_readback=mode,phase_timing=True) for mode in ('baseline','optimized')}
    results={};repeat_reference={};start=time.monotonic();deadline=start+POLICY['max_wall_seconds'];jobs={j['id']:j for j in plan['jobs']}
    def event(value):
        import os
        with journal.open('a') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    import threading
    timer=threading.Timer(POLICY['max_wall_seconds'],engine.stop_event.set);timer.daemon=True;timer.start()
    try:
        for run in schedule(plan):
            if time.monotonic()>=deadline or engine.stop_event.is_set():raise TimeoutError('Finite profiling wall budget/lease stop reached')
            requests=[]
            for identity in run['job_ids']:
                request=deepcopy(plan['requests'][identity]);request['capture_response']['id']=canonical_digest([plan['input_sha256'],run['id'],identity])[:32]
                request.setdefault('metadata',{})['benchmark_id']=run['id'];requests.append(request)
            envelope={'batch_id':uuid.uuid4().hex,'requests':requests}
            event({'state':'submitting','run':run,'envelope':envelope,'time':time.time()});began=time.perf_counter()
            response=backends[run['mode']].complete(envelope);elapsed=time.perf_counter()-began
            event({'state':'received','run_id':run['id'],'response':response,'wall_seconds':elapsed})
            if response['batch_id']!=envelope['batch_id'] or len(response['results'])!=len(requests):raise ValueError('Native response coverage differs')
            validated=[]
            for identity,request,row in zip(run['job_ids'],requests,response['results']):
                if row['request_sha256']!=canonical_digest(request):raise ValueError('Native response order/request differs')
                record=checked_parity(row['response'],request,binding(engine),store)
                key=(run['width'],run['repeat'],identity);prior=results.get(key)
                if prior:report['comparisons'].append({'kind':'paired','case_id':identity,'width':run['width'],'repeat':run['repeat'],**compare_observations(prior,record)})
                else:results[key]=record
                if identity in [plan['jobs'][i]['id'] for i in GROUPS[-1]]:
                    if identity in repeat_reference:
                        report['comparisons'].append({'kind':'repeat','case_id':identity,'width':run['width'],'repeat':run['repeat'],**compare_observations(repeat_reference[identity],record)})
                    else:repeat_reference[identity]=record
                validated.append({'case_id':identity,**record['evidence'],'native_trace':record['arrays']['completion_trace'].tolist(),'native_graph_kinds':record['arrays']['completion_graph_kind'].tolist(),'response':row['response']})
            counts=[r['sampled_tokens'] for r in validated]
            if run['width']>1 and len(set(counts))>1:report['holes_observed']=True
            profile=response.get('profiling')
            if not profile or profile.get('capture_readback_mode')!=run['mode'] or not profile.get('phases'):raise ValueError('Source-bound phase timing absent')
            report['runs'].append({**run,'wall_seconds':elapsed,'profiling':response.get('profiling'),'records':validated})
            atomic(output,report)
            # Equality checked while both arrays resident; retain source artifacts, not all decoded matrices.
            if prior:
                for identity in run['job_ids']:results.pop((run['width'],run['repeat'],identity),None)
        if time.monotonic()>=deadline or engine.stop_event.is_set():raise TimeoutError('Finite profiling deadline reached before final qualification')
        expected=plan['planned_responses']//2
        report.update(state='complete',passed=sum(c['kind']=='paired' for c in report['comparisons'])==expected and report['holes_observed'],elapsed_seconds=time.monotonic()-start)
    except BaseException as error:
        report.update(state='failed',passed=False,error=repr(error),elapsed_seconds=time.monotonic()-start);raise
    finally:timer.cancel();report['finished']=time.time();atomic(output,report)
    return {'passed':report['passed'],'report':output.name,'sha256':sha(output)}


def main():
    if len(__import__('sys').argv)>1 and __import__('sys').argv[1]=='plan':
        parser=argparse.ArgumentParser();parser.add_argument('mode');parser.add_argument('--root',type=Path,required=True);parser.add_argument('--source-plan',type=Path,required=True);parser.add_argument('--compatibility',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
        value=freeze(args.root,args.source_plan,args.compatibility)
        with args.output.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    else:
        import persona_probe as probe
        from persona_batch_server import prepare_adapter
        parser=probe.argument_parser();parser.add_argument('--performance-plan',type=Path,required=True);args=parser.parse_args()
        probe.run(args,experiment=experiment,preparation_adapter=prepare_adapter)

if __name__=='__main__':main()
