"""Finite native capture qualification and fixed-input residual propagation.

Uses already-leased endpoints only. Never allocates GPUs, retries a POST, or
changes a model. NPZ/provenance files are selectively fetched and hash checked;
arrays stay in files. Reports describe intervention responses, not coherence.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
import urllib.request
import urllib.error
import uuid
import numpy as np
from persona_vectors import canonical_digest
from persona_trace import ResponseCapture

IDENTITIES=('model_identity','runtime_identity','prepared_identity')
LAYERS=[31,32,48,63]
LAGS=[0,1,2,4,8]


def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def atomic(path,value):
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def check_launch(value):
    if not isinstance(value,dict) or not math.isfinite(value.get('deadline',float('nan'))) or value['deadline']<=time.time():
        raise ValueError('A live finite launch deadline is required')
    binding=value['binding']
    if any(not re.fullmatch('[0-9a-f]{64}',str(binding.get(k,''))) for k in IDENTITIES):raise ValueError('Launch identities missing')
    endpoints=value['endpoints']
    if not isinstance(endpoints,list) or len(endpoints)!=6:raise ValueError('Exactly six existing endpoints required')
    urls=[e['url'] if isinstance(e,dict) else e for e in endpoints]
    if len(set(urls))!=6 or any(not re.fullmatch(r'http://127\.0\.0\.1:893[0-5]',u) for u in urls):
        raise ValueError('Expected the six existing loopback ports8930..8935')
    return sorted(urls),binding,float(value['deadline'])


class ArtifactStore:
    def __init__(self,directory,remote_host,remote_root):
        self.directory=Path(directory);self.host=remote_host;self.root=Path(remote_root)
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',remote_host) or remote_host.startswith('-'):raise ValueError('Invalid SSH host')
    def get(self,pointer):
        source=Path(pointer['path']);expected=pointer['sha256']
        if source.parent!=self.root or not re.fullmatch(r'probe-[0-9a-f]{32}\.(?:npz|jsonl)',source.name):
            raise ValueError('Artifact outside exact reviewed result directory/naming')
        if not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError('Invalid artifact hash')
        target=self.directory/source.name
        if target.exists():
            if file_sha(target)!=expected:raise ValueError('Existing artifact bytes differ')
            return target
        # Bounded remote read, no shell interpolation of an untrusted path.
        code='import pathlib,sys;p=pathlib.Path(sys.argv[1]);n=p.stat().st_size;assert n<=268435456;sys.stdout.buffer.write(p.read_bytes())'
        command=shlex.join(['python3','-c',code,str(source)])
        temporary=target.with_name(target.name+'.part')
        with temporary.open('xb') as stream:
            result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none',self.host,command],stdout=stream,stderr=subprocess.PIPE,timeout=120)
            stream.flush();os.fsync(stream.fileno())
        if result.returncode or file_sha(temporary)!=expected:raise ValueError('Remote artifact transfer/hash failed; partial retained')
        temporary.replace(target)
        return target


def checked(response,request,binding,store):
    digest=canonical_digest(request)
    for key in IDENTITIES:
        if response.get(key)!=binding[key]:raise ValueError('Response identity differs')
    paths={}
    for kind in ('activation_artifact','provenance_artifact'):
        pointer=response[kind]
        if pointer.get('request_sha256')!=digest or any(pointer.get(k)!=binding[k] for k in IDENTITIES):raise ValueError('Artifact binding differs')
        paths[kind]=store.get(pointer)
    report=json.loads(paths['provenance_artifact'].read_text())
    if (report.get('status')!='complete' or report.get('state_restored') is not True or report.get('request')!=request
            or report.get('request_sha256')!=digest or any(report.get(k)!=binding[k] for k in IDENTITIES)
            or report.get('response')!={k:v for k,v in response.items() if k!='provenance_artifact'}):
        raise ValueError('Provenance restoration/request/response differs')
    with np.load(paths['activation_artifact'],allow_pickle=False) as data:
        if str(data['request_sha256'])!=digest or any(str(data[k])!=binding[k] for k in IDENTITIES):raise ValueError('NPZ identity differs')
        arrays={k:data[k].copy() for k in data.files if data[k].dtype.kind not in 'US' or k=='completion_graph_kind'}
        if 'prefill_execution_json' in data.files and json.loads(str(data['prefill_execution_json']))!=response.get('prefill_execution'):
            raise ValueError('Saved prefill execution differs')
    for key,value in arrays.items():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():raise ValueError('Nonfinite NPZ field: '+key)
    ids=arrays['prompt_token_ids'];generated=arrays['generated_token_ids'];trace=arrays['completion_trace']
    if (ids.ndim!=1 or generated.ndim!=1 or not len(generated) or trace.ndim!=2 or trace.shape[1]!=8
            or not np.array_equal(arrays['prefill_positions'],np.arange(len(ids))) or np.any(np.diff(trace[:,1])<=0)
            or not len(trace) or np.any(np.diff(trace[:,7])<0) or trace[-1,2]!=len(ids)+len(generated)-1 or trace[-1,3]!=len(generated)):
        raise ValueError('Actual token/completion/RNG alignment differs')
    for field,shape in [('residual_last',(64,5120)),('pre_mlp_last',(64,5120)),('mlp_input_last',(64,17408))]:
        if arrays[field].shape!=shape:raise ValueError('Activation dimensions differ')
    if request.get('observation_only'):
        if not response['diagnostics']['prefill_observation_complete']:raise ValueError('Prefill observation incomplete')
    elif not response['diagnostics']['complete_final_answer']:
        raise ValueError('Qualification answer did not complete')
    if request.get('capture_response'):
        validate_response_capture(response,request,arrays)
    if request.get('capture_prefill'):
        validate_prefill_capture(response,request,arrays)
    return arrays,paths


def validate_response_capture(response,request,a):
    r=response['response_capture'];generated=a['generated_token_ids'];consumed=generated[:-1];n=len(consumed);start=len(a['prompt_token_ids'])
    if (r['id']!=request['capture_response']['id'] or r['artifact']!=response['activation_artifact']
            or r['request_sha256']!=canonical_digest(request) or any(r.get(k)!=response.get(k) for k in IDENTITIES)
            or r['mean_defined']!=(n>0) or r['generation_complete'] is not True or r['consumed_generated_tokens']!=n
            or r['sampled_generated_tokens']!=len(generated) or r['all_consumed_generated_tokens_captured'] is not True
            or r['all_sampled_generated_tokens_captured'] is not False or r['final_sampled_unconsumed_token_id']!=int(generated[-1])
            or not np.array_equal(a['response_consumed_token_ids'],consumed)
            or not np.array_equal(a['response_consumed_positions'],np.arange(start,start+n))
            or int(a['response_residual_count'])!=n or bool(a['response_residual_mean_defined'])!=(n>0)):
        raise ValueError('Response capture coverage semantics differ')
    for key,values in [('consumed_token_ids_sha256',consumed),('consumed_positions_sha256',np.arange(start,start+n)),('generated_token_ids_sha256',generated)]:
        if r[key]!=ResponseCapture.ids_hash(values):raise ValueError('Response sequence hash differs')
    if a['response_residual_sum'].shape!=(64,5120) or a['response_residual_mean'].shape!=(64,5120):raise ValueError('Response means shape differs')
    expected=a['response_residual_sum']/n if n else np.zeros((64,5120))
    if not np.array_equal(expected,a['response_residual_mean']):raise ValueError('Response mean/count arithmetic differs')
    layers=request['capture_response']['layers'];kept=min(n,request['capture_response']['max_positions']) if layers else 0
    if (not np.array_equal(a['response_selected_layers'],layers) or not np.array_equal(a['response_selected_positions'],np.arange(start,start+kept))
            or a['response_selected_residuals'].shape!=(kept,len(layers),5120)):
        raise ValueError('Response selected profile differs')
    if kept==n and n and not np.array_equal(a['response_selected_residuals'].astype(np.float64).sum(axis=0),a['response_residual_sum'][layers]):
        raise ValueError('Response means disagree with full retained profiles')


def validate_prefill_capture(response,request,a):
    spec=request['capture_prefill'];r=response['prefill_capture'];positions=spec['positions'];layers=spec['layers']
    if (r['id']!=spec['id'] or r['complete'] is not True or r['artifact']!=response['activation_artifact']
            or r.get('request_sha256')!=canonical_digest(request) or any(r.get(k)!=response.get(k) for k in IDENTITIES)
            or r.get('positions')!=positions or r.get('layers')!=layers
            or r.get('positions_sha256')!=ResponseCapture.ids_hash(positions)
            or r.get('token_ids_sha256')!=ResponseCapture.ids_hash(a['prompt_token_ids'][positions])
            or not np.array_equal(a['prefill_snapshot_positions'],positions) or not np.array_equal(a['prefill_snapshot_layers'],layers)
            or not np.array_equal(a['prefill_snapshot_token_ids'],a['prompt_token_ids'][positions])
            or a['prefill_snapshot_residuals'].shape!=(len(positions),len(layers),5120)):
        raise ValueError('Prefill snapshot alignment differs')
    execution=response['prefill_execution'];pulse=request.get('steering_pulse',{}).get('position')
    if execution['pulse_position']!=pulse or execution['consumed_token_ids_sha256']!=ResponseCapture.ids_hash(a['prompt_token_ids']):raise ValueError('Pulse execution metadata differs')
    spans=execution['spans']
    if [p for start,stop in spans for p in range(start,stop)]!=list(range(len(a['prompt_token_ids']))):raise ValueError('Prefill spans differ')
    if pulse is not None and [pulse,pulse+1] not in spans:raise ValueError('Source position not isolated')


def exact(left,right,*,partition_same=True):
    fields=('prompt_token_ids','generated_token_ids','mlp_input_last','residual_last','pre_mlp_last','residual_l2')
    mismatch=[k for k in fields if not np.array_equal(left[k],right[k])]
    l,r=left['completion_trace'],right['completion_trace']
    if not np.array_equal(l[:,2:] if partition_same else l[-1,2:],r[:,2:] if partition_same else r[-1,2:]):mismatch.append('completion_RNG_state')
    if partition_same and not np.array_equal(left['completion_graph_kind'],right['completion_graph_kind']):mismatch.append('completion_graph_kind')
    if mismatch:raise ValueError('Native equality failed: '+','.join(mismatch))


class Runner:
    def __init__(self,launch,output,store):
        self.endpoints,self.binding,self.deadline=check_launch(launch);self.output=Path(output);self.store=store;self.lock=threading.Lock();self.records=[]
        self.attempts=self.output.with_name(self.output.stem+'.attempts.jsonl')
        if self.attempts.exists():raise ValueError('Prior attempts exist; no uncertain POST retry')
    def journal(self,value):
        with self.lock:
            with self.attempts.open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    def fetch(self,url,value=None):
        remaining=self.deadline-time.time()
        if remaining<=15:raise RuntimeError('Finite launch deadline reached; no further requests')
        data=None if value is None else json.dumps(value,allow_nan=False).encode()
        with urllib.request.urlopen(urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'}),timeout=min(300,remaining)) as r:return json.load(r)
    def model(self,endpoint):
        ready_deadline=min(time.time()+1200,self.deadline-300)
        while True:
            try:
                model=self.fetch(endpoint+'/v1/models')['data'][0]
                break
            except urllib.error.HTTPError:
                raise
            except (urllib.error.URLError,ConnectionResetError,ConnectionRefusedError,TimeoutError):
                if time.time()+10>ready_deadline:raise
                time.sleep(10)
        flare=model['flare']
        if any(flare.get(k)!=self.binding[k] for k in IDENTITIES):raise ValueError('Endpoint differs from frozen launch binding')
        if not flare.get('response_capture',{}).get('supported') or not flare.get('prefill_instrumentation',{}).get('supported'):raise ValueError('Required native instrumentation unavailable')
        return model
    def infer(self,endpoint,request,label):
        identity=uuid.uuid4().hex;self.journal(dict(attempt=identity,state='started',endpoint=endpoint,label=label,request=request,time=time.time()))
        try:
            response=self.fetch(endpoint+'/v1/chat/completions',request)
            self.journal(dict(attempt=identity,state='response_received',response=response,time=time.time()))
            arrays,paths=checked(response,request,self.binding,self.store)
            record=dict(attempt=identity,endpoint=endpoint,label=label,request_sha256=canonical_digest(request),activation_artifact=response['activation_artifact'],provenance_artifact=response['provenance_artifact'],local_artifacts={k:str(v) for k,v in paths.items()})
            with self.lock:self.records.append(record)
            self.journal(dict(attempt=identity,state='validated',time=time.time()))
            return response,arrays,record
        except Exception as error:
            self.journal(dict(attempt=identity,state='failed_or_uncertain',error=repr(error),time=time.time()));raise


def base_request(model,prompt,*,seed=7035):
    return dict(model=model['id'],messages=[dict(role='user',content=prompt)],seed=seed,temperature=0,max_tokens=64,stream=False,chat_template_kwargs={'enable_thinking':False})


def steering(model,layer,vector):
    site=next(s for s in model['flare']['steering']['sites'] if s['name']==f'layers.{layer}.residual')
    if len(vector)!=site['width']:raise ValueError('Residual direction dimensions differ')
    return dict(phase='prefill',sites=[dict(site=site['name'],vector=np.asarray(vector,dtype=float).tolist())])


def source_invariant(baseline, intervention):
    if (not np.array_equal(intervention['prompt_token_ids'],baseline['prompt_token_ids'])
            or not np.array_equal(intervention['prefill_snapshot_residuals'][0,0],baseline['prefill_snapshot_residuals'][0,0])):
        raise ValueError('Fixed tokens/upstream source-position invariant failed')


def qualification_worker(runner,endpoint):
    model=runner.model(endpoint);results=[]
    prompt='A brass key lies beside an unopened letter in the quiet observatory. Answer with exactly these three words: the letter waits.'
    for mode in ('greedy','sampled','steered'):
        base=base_request(model,prompt)
        if mode!='greedy':base.update(temperature=.7,top_p=.95,top_k=20)
        if mode=='steered':base['steering']=steering(model,32,np.eye(1,5120,0)[0]*.1)
        _,off,off_record=runner.infer(endpoint,base,mode+'_off_before')
        enabled=dict(base,capture_response=dict(id=uuid.uuid4().hex,layers=LAYERS,max_positions=128))
        response,on,on_record=runner.infer(endpoint,enabled,mode+'_capture_on');exact(off,on)
        if int(on['response_residual_count'])<1:raise ValueError('Qualification failed to exercise consumed response tokens')
        _,after,after_record=runner.infer(endpoint,base,mode+'_off_after');exact(off,after)
        results.append(dict(mode=mode,passed=True,attempts=[off_record['attempt'],on_record['attempt'],after_record['attempt']],response_tokens=int(on['response_residual_count'])))
    pulse={'passed':False,'arms':[]}
    try:
        qualify_pulse(runner,endpoint,model,prompt,pulse)
        pulse['passed']=True
    except Exception as error:
        pulse['error']=repr(error)
    return dict(endpoint=endpoint,passed=True,response_capture_passed=True,response_capture=results,pulse=pulse)


def partition_difference(left,right):
    result={'same_actual_prompt_tokens':bool(np.array_equal(left['prompt_token_ids'],right['prompt_token_ids'])),
            'same_generated_tokens':bool(np.array_equal(left['generated_token_ids'],right['generated_token_ids'])),'activation_differences':{}}
    for key in ('mlp_input_last','residual_last','pre_mlp_last','residual_l2'):
        a,b=left[key].astype(float),right[key].astype(float)
        if a.shape!=b.shape:raise ValueError('Partition comparison shape differs')
        delta=b-a
        result['activation_differences'][key]={'exact_equal':bool(np.array_equal(a,b)),
            'max_abs':float(np.max(np.abs(delta))),'l2':float(np.linalg.norm(delta)),
            'relative_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(a),1e-30))}
    try:exact(left,right,partition_same=False);result['exact_equal']=True
    except ValueError as error:result.update(exact_equal=False,equality_error=str(error))
    result['interpretation']='Partition sensitivity only; pulse effects use identically partitioned zero baseline.'
    return result


def pulse_scale(snapshot):
    coordinate=float(snapshot[0]);norm=float(np.linalg.norm(np.asarray(snapshot,dtype=float)))
    spacing=2.**(math.floor(math.log2(max(abs(coordinate),2.**-126)))-7)
    scale=max(1.,16*spacing)
    # Qualification must exercise rounding without a disproportionately large pulse.
    if .1*scale>max(1.,.1*norm):raise ValueError('BF16-visible qualification pulse exceeds bounded residual norm scale')
    return {'basis_scale':scale,'baseline_coordinate':coordinate,'baseline_bf16_spacing':spacing,
            'baseline_residual_l2':norm,'largest_absolute_dose':.1*scale,
            'largest_dose_to_residual_norm':.1*scale/norm if norm else None}


def qualify_pulse(runner,endpoint,model,prompt,result):
    base=base_request(model,prompt);base.update(observation_only=True,max_tokens=1)
    _,plain,plain_record=runner.infer(endpoint,base,'unsplit_baseline')
    result['unsplit_attempt']=plain_record['attempt']
    position=8;positions=[position+lag for lag in LAGS]
    if positions[-1]>=len(plain['prompt_token_ids']):raise ValueError('Qualification prompt too short')
    split=dict(base,steering_pulse={'position':position},capture_prefill={'id':uuid.uuid4().hex,'layers':LAYERS,'positions':positions})
    _,zero,zrecord=runner.infer(endpoint,split,'partitioned_zero')
    result.update(source_position=position,layers=LAYERS,lags=LAGS,zero_attempt=zrecord['attempt'],partition_comparison=partition_difference(plain,zero))
    snapshot=zero['prefill_snapshot_residuals'];scaling=pulse_scale(snapshot[0,1]);result['scaling']=scaling
    for dose in (-.1,-.05,.05,.1):
        absolute=dose*scaling['basis_scale']
        request=dict(split,capture_prefill={**split['capture_prefill'],'id':uuid.uuid4().hex},steering=steering(model,32,np.eye(1,5120,0)[0]*absolute))
        _,a,record=runner.infer(endpoint,request,'pulse_'+str(dose))
        source_invariant(zero,a)
        delta=a['prefill_snapshot_residuals'][0,1].astype(float)-snapshot[0,1]
        if delta[0]*absolute<=0 or np.count_nonzero(delta[1:]):raise ValueError('Residual source pulse is not confined to requested coordinate/sign')
        result['arms'].append(dict(dose=dose,absolute_dose=absolute,attempt=record['attempt'],actual_source_delta=float(delta[0])))


def cosine(a,b):
    a=np.asarray(a,dtype=float).ravel();b=np.asarray(b,dtype=float).ravel();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den else None


def validate_plan(plan):
    if len(plan.get('contexts',[]))!=6 or len(plan.get('directions',[]))!=2:raise ValueError('Exactly six development contexts and two frozen directions required')
    if any(c.get('split')!='discovery' or not c.get('prompt') or type(c.get('source_position')) is not int or c['source_position']<0 for c in plan['contexts']):raise ValueError('Propagation fits only fixed discovery contexts and explicit source positions')
    if len({c['id'] for c in plan['contexts']})!=6 or len({d['id'] for d in plan['directions']})!=2:raise ValueError('Duplicate propagation IDs')
    for d in plan['directions']:
        v=np.asarray(d['vector'],dtype=float)
        if (d.get('layer')!=32 or v.shape!=(5120,) or not np.isfinite(v).all() or np.linalg.norm(v)==0
                or type(d.get('residual_median')) not in (int,float) or not math.isfinite(d['residual_median']) or d['residual_median']<=0
                or not re.fullmatch('[0-9a-f]{64}',str(d.get('source_sha256','')))):
            raise ValueError('Invalid model-bound layer32 direction/calibration')


def propagation_context(runner,endpoint,context,directions):
    model=runner.model(endpoint);position=context['source_position'];positions=[position+lag for lag in LAGS]
    base=base_request(model,context['prompt']);base.update(max_tokens=1,observation_only=True,steering_pulse={'position':position},capture_prefill=dict(id=uuid.uuid4().hex,layers=LAYERS,positions=positions))
    _,zero,zrecord=runner.infer(endpoint,base,context['id']+'_zero')
    _,repeat,rrecord=runner.infer(endpoint,dict(base,capture_prefill={**base['capture_prefill'],'id':uuid.uuid4().hex}),context['id']+'_zero_repeat')
    exact(zero,repeat);z=zero['prefill_snapshot_residuals'].astype(float);noise=np.abs(repeat['prefill_snapshot_residuals'].astype(float)-z)
    if noise.any():raise ValueError('Repeated zero snapshots differ')
    output=[]
    for direction in directions:
        unit=np.asarray(direction['vector'],dtype=float);unit=unit/np.linalg.norm(unit)*direction['residual_median'];arms={};records=[]
        for dose in (-.1,-.05,.05,.1):
            request=dict(base,capture_prefill={**base['capture_prefill'],'id':uuid.uuid4().hex},steering=steering(model,32,unit*dose))
            _,a,record=runner.infer(endpoint,request,context['id']+'_'+direction['id']+'_'+str(dose))
            current=a['prefill_snapshot_residuals'].astype(float)
            source_invariant(zero,a)
            arms[dose]=current-z;records.append(dict(dose=dose,attempt=record['attempt'],actual_source_delta_l2=float(np.linalg.norm(arms[dose][0,1])),source_delta_direction_cosine=cosine(arms[dose][0,1],unit*dose)))
        small=(arms[.05]-arms[-.05])/.1;large=(arms[.1]-arms[-.1])/.2
        metrics=[]
        for pi,lag in enumerate(LAGS):
            for li,layer in enumerate(LAYERS):
                # Baseline BF16 spacing is a descriptive rounding scale, not a confidence interval.
                spacing=np.exp2(np.floor(np.log2(np.maximum(np.abs(z[pi,li]),2.**-126)))-7)
                metrics.append(dict(lag=lag,position=position+lag,layer=layer,
                    small_symmetric_derivative_l2=float(np.linalg.norm(small[pi,li])),large_symmetric_derivative_l2=float(np.linalg.norm(large[pi,li])),
                    dose_derivative_cosine=cosine(small[pi,li],large[pi,li]),
                    even_response_l2_at_point1=float(np.linalg.norm((arms[.1][pi,li]+arms[-.1][pi,li])/2)),
                    baseline_bf16_spacing_rms=float(np.sqrt(np.mean(spacing**2))),
                    point05_delta_above_spacing_fraction=float(np.mean(np.abs(arms[.05][pi,li])>spacing))))
        output.append(dict(direction_id=direction['id'],direction_vector_sha256=hashlib.sha256(np.asarray(direction['vector'],dtype='<f8').tobytes()).hexdigest(),arms=records,metrics=metrics,
                           _small=small,_large=large))
    return dict(context_id=context['id'],endpoint=endpoint,zero_attempt=zrecord['attempt'],zero_repeat_attempt=rrecord['attempt'],zero_noise_max=float(noise.max()),directions=output)


def public_report(value):
    """Omit internal temporary vector arrays from durable JSON progress."""
    if isinstance(value,dict):return {k:public_report(v) for k,v in value.items() if k not in ('_small','_large')}
    if isinstance(value,list):return [public_report(v) for v in value]
    return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('qualify','propagate'));p.add_argument('--launch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path);p.add_argument('--qualification',type=Path);p.add_argument('--remote-host',default='v100');p.add_argument('--remote-root',default='/mnt/storage16tb/Character Creation/results');args=p.parse_args(argv)
    launch=json.loads(args.launch.read_text());endpoints,binding,deadline=check_launch(launch)
    if args.output.exists():raise ValueError('Report already exists; explicit review required, no blind retry')
    plan=None
    if args.mode=='propagate':
        if not args.plan or not args.qualification:raise ValueError('Propagation requires frozen plan and passed qualification')
        plan=json.loads(args.plan.read_text());validate_plan(plan)
        qualification=json.loads(args.qualification.read_text())
        if qualification.get('passed') is not True or qualification.get('pulse_qualification_passed') is not True or qualification.get('state')!='complete' or len(qualification.get('results',[]))!=6 or qualification.get('binding')!=binding or plan.get('binding')!=binding:raise ValueError('Qualification/plan binding differs')
    store=ArtifactStore(args.output.parent,args.remote_host,args.remote_root);runner=Runner(launch,args.output,store)
    report=dict(schema='character-creation.'+('capture-qualification' if args.mode=='qualify' else 'propagation')+'.v1',state='running',passed=False,started=time.time(),binding=binding,deadline=deadline,launch_snapshot=launch,launch_sha256=file_sha(args.launch),driver_sha256=file_sha(__file__),inputs=plan,plan_sha256=file_sha(args.plan) if args.plan else None,qualification_sha256=file_sha(args.qualification) if args.qualification else None,results=[],failures=[],attempts_file=str(runner.attempts))
    atomic(args.output,report)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(qualification_worker,runner,e):e for e in endpoints} if args.mode=='qualify' else {pool.submit(propagation_context,runner,e,c,plan['directions']):c['id'] for e,c in zip(endpoints,plan['contexts'])}
        for future in as_completed(futures):
            try:report['results'].append(future.result())
            except Exception as error:report['failures'].append(dict(worker=futures[future],error=repr(error)))
            report['records']=list(runner.records)
            atomic(args.output,public_report(report))
    if args.mode=='propagate':
        context_consistency=[]
        for direction in plan['directions']:
            rows=[(r['context_id'],next(d for d in r['directions'] if d['direction_id']==direction['id'])) for r in report['results']]
            for i,(cid,a) in enumerate(rows):
                for other,b in rows[i+1:]:
                    context_consistency.append(dict(direction_id=direction['id'],contexts=[cid,other],cosine_by_lag_and_layer=[[cosine(a['_small'][pi,li],b['_small'][pi,li]) for li in range(4)] for pi in range(5)]))
        report['context_consistency']=context_consistency
        for r in report['results']:
            for d in r['directions']:d.pop('_small');d.pop('_large')
        report['interpretation']='Fixed-token, one-position additive residual intervention total downstream effects. Multiple epsilons expose nonlinear/quantized response; zero-repeat noise and BF16 spacing are descriptive scales, not confidence. No coherence or direct causal wiring claim. Future-token upstream layers may change through recurrence.'
    if args.mode=='qualify':
        report['response_capture_passed']=not report['failures'] and len(report['results'])==6 and all(r.get('response_capture_passed') is True for r in report['results'])
        report['pulse_qualification_passed']=report['response_capture_passed'] and all(r['pulse'].get('passed') is True for r in report['results'])
    report.update(state='complete' if not report['failures'] else 'failed',passed=not report['failures'],finished=time.time(),records=runner.records)
    atomic(args.output,report);print(json.dumps({'report':str(args.output),'passed':report['passed'],'validated_requests':len(runner.records)}),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
