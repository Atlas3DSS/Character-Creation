"""Finite B1/B2/B4 capture qualification; no production HTTP integration."""
from copy import deepcopy
from contextlib import contextmanager
from types import MappingProxyType
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from persona_batch_probe import BatchTrial,sha,atomic
from persona_vectors import canonical_digest

PLAN_SCHEMA='character-creation.batch-capture-plan.v1'
PLANES=('residual_last','pre_mlp_last','mlp_input_last')
FIELDS=('row_slot_ids','row_generations','request_generations','physical_slots','token_rows','token_slots','token_positions','token_ids','token_generations','token_live','invalid_tokens','slot_match_counts','slot_last_index','slot_live','slot_positions','slot_token_ids','slot_generations','mapping_error',*PLANES)
POLICY={'budget':32,'unequal_budgets':[1,3,8,32],'exact_capture_equality':True,'maximum_artifact_bytes':768*1024**2,
        'scope':'Consumed generated input tokens only; prefill excluded; final sampled token is not consumed. No production HTTP rollout.'}


def prepare_adapter(original_prepare):
    def wrapped(*args,**kwargs):
        from ignition.recipes.qwen38_dense import bootstrap
        from activation_capture import RequestActivationCapture
        original=bootstrap.PreparedDecoder;shared=[]
        def factory(weights,scratch,metadata,pool,**options):
            if options.get('mode')=='decode' and metadata.positions.numel() in (2,4):
                if options.get('activation_capture') is not None:raise ValueError('Decode capture already instrumented')
                steering=options.get('steering')
                if steering is None:raise ValueError('Native steering ownership required')
                if not shared:
                    shared.append(RequestActivationCapture.allocate(len(weights.config.layers),steering.inputs.row_slot_ids.numel(),weights.config.hidden,
                        scratch.activations.device,intermediate=weights.config.text['intermediate_size'],row_slot_ids=steering.inputs.row_slot_ids,
                        row_generations=steering.inputs.row_generations,request_generations=steering.requests.generation))
                elif shared[0].row_slot_ids is not steering.inputs.row_slot_ids or shared[0].request_generations is not steering.requests.generation:
                    raise ValueError('Decode captures do not share native ownership')
                options['activation_capture']=shared[0].rows(metadata.positions.numel())
            return original(weights,scratch,metadata,pool,**options)
        bootstrap.PreparedDecoder=factory
        try:return original_prepare(*args,**kwargs)
        finally:bootstrap.PreparedDecoder=original
    return wrapped


class Accumulator:
    def __init__(self,n,layers,hidden,intermediate):
        self.widths=dict(residual_last=hidden,pre_mlp_last=hidden,mlp_input_last=intermediate)
        self.sums=np.zeros((n,layers,hidden),np.float64);self.first={p:np.zeros((n,layers,w),np.float32) for p,w in self.widths.items()}
        self.ids=[[] for _ in range(n)];self.positions=[[] for _ in range(n)];self.generations=[None]*n;self.hashes=[[] for _ in range(n)]
    def add(self,index,position,token,generation,planes):
        if self.positions[index] and position!=self.positions[index][-1]+1:raise ValueError('Capture position discontinuity')
        if self.generations[index] is not None and generation!=self.generations[index]:raise ValueError('Request generation changed')
        if type(position) is not int or position<0 or type(token) is not int or token<0 or generation<=0:raise ValueError('Invalid token evidence')
        hashes={}
        for name,width in self.widths.items():
            value=np.asarray(planes[name])
            if value.shape!=(self.sums.shape[1],width) or value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('Invalid capture plane')
            hashes[name]=hashlib.sha256(np.ascontiguousarray(value,dtype='<f4').tobytes()).hexdigest()
            if not self.ids[index]:self.first[name][index]=value
        self.sums[index]+=planes['residual_last'].astype(np.float64)
        self.ids[index].append(token);self.positions[index].append(position);self.generations[index]=generation;self.hashes[index].append(hashes)
    def finish(self,result):
        for i,generated in enumerate(result['generated']):
            if self.ids[i]!=generated[:-1] or self.positions[i]!=list(range(result['prompt_tokens'][i],result['prompt_tokens'][i]+len(generated)-1)):
                raise ValueError('Consumed capture coverage differs from generated[:-1]')
        count=np.asarray([len(v) for v in self.ids],np.int64)
        means=np.divide(self.sums,count[:,None,None],out=np.zeros_like(self.sums),where=count[:,None,None]>0)
        return {'counts':count.tolist(),'mean_defined':(count>0).tolist(),'consumed_token_ids':self.ids,'positions':self.positions,'generations':self.generations,
                'step_plane_sha256':self.hashes,'_count':count,'_sum':self.sums,'_mean':means,**{'_first_'+k:v for k,v in self.first.items()}}


def mapping_expected(before,slots,evidence):
    active={s for s in slots if int(before[s][4])==1}
    live=set(np.flatnonzero(evidence['slot_live']).tolist())
    if bool(np.any(evidence['mapping_error'])) or live!=active:raise ValueError('Capture mapping/live slots differ')
    for s in active:
        generation,position,_,token=map(int,before[s][:4])
        if (int(evidence['slot_positions'][s])!=position or int(evidence['slot_token_ids'][s])!=token
                or int(evidence['slot_generations'][s])!=generation or int(evidence['slot_match_counts'][s])!=1):raise ValueError('Capture input position/token/generation differs')
    if 'token_live' in evidence:
        live_rows=np.flatnonzero(evidence['token_live'])
        if np.any(evidence['invalid_tokens']) or len(live_rows)!=len(active):raise ValueError('Invalid packed token evidence')
        if {int(evidence['token_slots'][r]) for r in live_rows}!=active:raise ValueError('Packed token slots differ')
        for r in live_rows:
            slot=int(evidence['token_slots'][r]);generation,position,_,token=map(int,before[slot][:4])
            if (int(evidence['token_positions'][r]),int(evidence['token_ids'][r]),int(evidence['token_generations'][r]))!=(position,token,generation):raise ValueError('Packed token ownership differs')
    return active


class PreparedFacade:
    """Attribute-compatible prepared view; native graph registry stays untouched."""
    __slots__=('_original','graphs')
    def __init__(self,original,replacements):
        unknown=set(replacements)-set(original.graphs)
        if unknown:raise ValueError('Cannot wrap unknown prepared graphs')
        self._original=original
        self.graphs=MappingProxyType({**original.graphs,**replacements})
    def __getattr__(self,name):return getattr(self._original,name)


@contextmanager
def wrapped_prepared(engine,replacements):
    original=engine.prepared
    engine.prepared=PreparedFacade(original,replacements)
    try:yield
    finally:engine.prepared=original


KNOWN_UNSTARTED_ERROR=repr(TypeError("'mappingproxy' object does not support item assignment"))


def recovery_history(output,artifact,plan,enabled):
    if artifact.exists():raise FileExistsError('Existing capture arrays must not be overwritten')
    if not output.exists():
        if enabled:raise ValueError('Explicit recovery requires preserved failed report')
        return []
    old=json.loads(output.read_text());worker=old.get('failed_worker_manifest',{});prior=old.get('failed_plan',{})
    if (not enabled or old.get('schema')!='character-creation.batch-capture-proof.v1'
        or old.get('state')!='failed_pre_execution' or old.get('passed') is not False
        or old.get('runs')!={} or old.get('artifact') or old.get('error')!=KNOWN_UNSTARTED_ERROR
        or worker.get('status')!='failed' or worker.get('error')!="TypeError: 'mappingproxy' object does not support item assignment"
        or worker.get('calibration_equivalence',{}).get('passed') is not True):
        raise FileExistsError('Recovery limited to verified mappingproxy pre-execution failure')
    digest=hashlib.sha256(json.dumps(prior,indent=2,allow_nan=False).encode()).hexdigest()
    if digest!=old.get('failed_plan_sha256') or digest!=old.get('policy_sha256'):raise ValueError('Preserved failed plan hash differs')
    mutable={'adapter_sha256','probe_source_sha256','created'}
    if {k:v for k,v in prior.items() if k not in mutable}!={k:v for k,v in plan.items() if k not in mutable}:raise ValueError('Recovery changes scientific policy/cases/native core')
    if (old.get('driver_sha256')!=prior.get('adapter_sha256') or worker.get('experiment_source_sha256')!=prior.get('adapter_sha256')
        or worker.get('script_sha256')!=prior.get('probe_source_sha256')):raise ValueError('Failed worker/source lineage differs')
    return [old]


class CaptureTrial:
    def __init__(self,engine):
        self.engine=engine;self.base=BatchTrial(engine)
        self.capture=self.base.catalog[self.base.names[4]].decoder.activation_capture
        if self.capture is None or not hasattr(self.capture,'slot_live'):raise ValueError('B4 request capture absent')
        two=self.base.catalog[self.base.names[2]].decoder.activation_capture
        if two.residual_last.data_ptr()!=self.capture.residual_last.data_ptr():raise ValueError('B2/B4 backing differs')
    def execute(self,cases,**options):
        import torch
        from persona_probe import REQUEST_STATE_FIELDS
        e=self.engine;slots=options.get('slots',list(range(len(cases))));logical={s:i for i,s in enumerate(slots)}
        accumulator=Accumulator(len(cases),e.model.layers,e.model.hidden,e.model.intermediate);mapping_steps=[]
        # All new tensors explicitly restored; borrowed native vectors are small, not pools.
        snapshots={name:getattr(self.capture,name).clone() for name in FIELDS}
        legacy={name:getattr(e.capture,name).clone() for name in PLANES}
        original_prepared=e.prepared
        originals={name:e.prepared.graphs[name] for name in self.base.names.values()}
        class Wrapper:
            def __init__(self,original,width):self.original=original;self.width=width
            def __getattr__(self,name):return getattr(self.original,name)
            def replay(wrapper):
                before=np.stack([getattr(e.requests,k).cpu().numpy().copy() for k in REQUEST_STATE_FIELDS],axis=1)
                wrapper.original.replay();torch.cuda.synchronize(e.prepared.device)
                if wrapper.width==1:
                    active={s for s in slots if int(before[s][4])==1}
                    if len(active)!=1:raise ValueError('Legacy B1 expected one active request')
                    owner=self.base.catalog[self.base.names[1]];m=owner.batch.metadata
                    row=int(m.requests[0].item());slot=int(e.inputs.row_slot_ids[row].item());s=next(iter(active))
                    if slot!=s or int(m.positions[0].item())!=int(before[s][1]) or int(m.token_ids[0].item())!=int(before[s][3]) or int(e.inputs.row_generations[row].item())!=int(before[s][0]):raise ValueError('B1 native metadata differs')
                    mapping_steps.append({'width':1,'before':before.tolist(),'slot':s,'row':row,'position':int(m.positions[0].item()),'token_id':int(m.token_ids[0].item()),'generation':int(e.inputs.row_generations[row].item())})
                    planes={name:getattr(e.capture,name).cpu().numpy().copy() for name in PLANES}
                    accumulator.add(logical[s],int(before[s][1]),int(before[s][3]),int(before[s][0]),planes)
                else:
                    view=self.base.catalog[self.base.names[wrapper.width]].decoder.activation_capture
                    evidence={k:getattr(view,k).cpu().numpy().copy() for k in ('slot_live','slot_positions','slot_token_ids','slot_generations','slot_match_counts','slot_last_index','mapping_error','token_rows','token_slots','token_positions','token_ids','token_generations','token_live','invalid_tokens')}
                    active=mapping_expected(before,slots,evidence)
                    planes={name:getattr(self.capture,name).cpu().numpy().copy() for name in PLANES}
                    mapping_steps.append({'width':wrapper.width,'before':before.tolist(),'evidence':{k:v.tolist() for k,v in evidence.items()},'inactive_plane_sha256':{str(s):{k:hashlib.sha256(np.ascontiguousarray(v[:,s,:],dtype='<f4').tobytes()).hexdigest() for k,v in planes.items()} for s in set(range(4))-active}})
                    for s in active:accumulator.add(logical[s],int(before[s][1]),int(before[s][3]),int(before[s][0]),{k:v[:,s,:] for k,v in planes.items()})
                    for s in set(range(4))-active:
                        if any(np.any(v[:,s,:]) for v in planes.values()):raise ValueError('Inactive slot retains capture data')
        with torch.inference_mode():
            try:
                replacements={name:Wrapper(originals[name],width) for width,name in self.base.names.items()}
                with wrapped_prepared(e,replacements):
                    result=self.base.execute(cases,**options);result['capture']=accumulator.finish(result);result['capture']['mapping_steps']=mapping_steps
                return result
            finally:
                if e.prepared is not original_prepared:raise ValueError('Prepared facade not restored')
                for name,value in snapshots.items():getattr(self.capture,name).copy_(value)
                for name,value in legacy.items():getattr(e.capture,name).copy_(value)
                torch.cuda.synchronize(e.prepared.device)
                if any(not torch.equal(getattr(self.capture,k),v) for k,v in snapshots.items()) or any(not torch.equal(getattr(e.capture,k),v) for k,v in legacy.items()):raise ValueError('Capture tensor restoration differs')
                if 'result' in locals():result['capture_restored']=True;result['graph_wrappers_restored']=all(e.prepared.graphs[k] is v for k,v in originals.items())


def exact_capture(a,b,i,j):
    x,y=a['capture'],b['capture']
    return (a['generated'][i]==b['generated'][j] and x['consumed_token_ids'][i]==y['consumed_token_ids'][j]
        and x['positions'][i]==y['positions'][j] and x['step_plane_sha256'][i]==y['step_plane_sha256'][j]
        and all(np.array_equal(a[k][i],b[k][j]) for k in ('_logits','_hidden'))
        and all(np.array_equal(x[k][i],y[k][j]) for k in ('_count','_sum','_mean',*['_first_'+p for p in PLANES])))


def make_plan(original):
    if original['schema']!='character-creation.batch-proof-plan.v1' or len(original['cases'])!=4:raise ValueError('Original four-case plan required')
    result=deepcopy(original);result['schema']=PLAN_SCHEMA;result['policy']=deepcopy(POLICY)
    for case in result['cases']:
        if canonical_digest(case['program'])!=case['program_sha256']:raise ValueError('Original program changed')
        case['budget']=32
    result['batch_trial_source_sha256']=sha(Path(__file__).with_name('persona_batch_probe.py'))
    result['probe_source_sha256']=sha(Path(__file__).with_name('persona_probe.py'))
    capture=Path(__file__).resolve().parent.parent/'activation_capture.py'
    result['capture_source_sha256']=sha(capture);result['adapter_sha256']=sha(__file__)
    content=capture.read_bytes();boundary=content.index(b'@dataclass(frozen=True)\nclass RequestActivationCapture')
    result['legacy_capture_prefix_sha256']=hashlib.sha256(content[:boundary-2]).hexdigest()
    old_core=original.get('production_core_source_hashes')
    if old_core and result['legacy_capture_prefix_sha256']!=old_core['ignition/components/activation_capture.py']:raise ValueError('Legacy capture prefix changed')
    result['original_plan_canonical_sha256']=canonical_digest(original)
    result['created']=time.time();return result


def experiment(engine,args):
    output=args.output/'persona_batch_capture_proof.json';artifact=args.output/'persona_batch_capture_proof.npz'
    plan=json.loads(args.capture_plan.read_text())
    history=recovery_history(output,artifact,plan,getattr(args,'recover_unstarted',False))
    if plan['schema']!=PLAN_SCHEMA or plan['policy']!=POLICY or plan['batch_trial_source_sha256']!=sha(Path(__file__).with_name('persona_batch_probe.py')):raise ValueError('Frozen capture policy/BatchTrial source differs')
    if engine.manifest['script_sha256']!=plan['probe_source_sha256']:raise ValueError('Probe source differs from frozen plan')
    override=engine.manifest.get('capture_override') or {}
    if override.get('sha256')!=plan['capture_source_sha256'] or sha(__file__)!=plan['adapter_sha256']:raise ValueError('Capture/adapter source differs from frozen plan')
    core={**engine.manifest['source_sha256'],'heliopause':engine.manifest['heliopause_sha256']}
    expected_core=dict(plan['production_core_source_hashes']);expected_core['ignition/components/activation_capture.py']=plan['capture_source_sha256']
    if core!=expected_core or engine.prepared.server_config['identity']!=plan['production_binding']['prepared_identity']:raise ValueError('Native core/prepared binding changed')
    if any(c['budget']!=32 or canonical_digest(c['program'])!=c['program_sha256'] for c in plan['cases']):raise ValueError('Frozen program/budget differs')
    if plan['model_identity']!=engine.model.identity or sha(args.output/'layer_32.npz')!=plan['calibration_sha256']:raise ValueError('Model/calibration differs')
    report=dict(schema='character-creation.batch-capture-proof.v1',state='running',passed=False,policy=POLICY,policy_sha256=sha(args.capture_plan),driver_sha256=sha(__file__),batch_trial_source_sha256=plan['batch_trial_source_sha256'],
                binding={k:engine.manifest[k] if k!='prepared_identity' else engine.prepared.server_config['identity'] for k in ('model_identity','runtime_identity','prepared_identity')},runs={},checks={},automatic_rollout=False)
    report['prior_attempts']=history
    report['limitations']=['Timing includes capture synchronization and host readback, not kernel-only throughput.','Non-first per-token vectors are represented by hashes; sums rely on the source-bound accumulator.','All new B2/B4 graphs include capture; this is not a same-graph capture-off/on speed comparison.']
    report.update(eos_token_ids=list(engine.eos),source_sha256=engine.manifest['source_sha256'],capture_override=engine.manifest.get('capture_override'),heliopause_sha256=engine.manifest['heliopause_sha256'],adapter_sha256=sha(__file__),probe_source_sha256=plan['probe_source_sha256'],legacy_capture_prefix_sha256=plan['legacy_capture_prefix_sha256'])
    atomic(output,report);arrays={};values={};cases=plan['cases']
    def run(name,items,**options):
        print('batch-capture '+name,flush=True)
        value=trial.execute(items,**options);values[name]=value
        for k in ('_logits','_hidden'):arrays[name+k]=value[k]
        for k,v in value['capture'].items():
            if k.startswith('_'):arrays[name+k]=v
        report['runs'][name]={k:v for k,v in value.items() if not k.startswith('_') and k!='capture'}
        report['runs'][name]['capture']={k:v for k,v in value['capture'].items() if not k.startswith('_')};atomic(output,report)
        if sum(v.nbytes for v in arrays.values())>POLICY['maximum_artifact_bytes']:raise ValueError('Capture array budget exceeded')
        return value
    try:
        trial=CaptureTrial(engine)
        report['decode_graphs']={str(rows):{'name':name,'batch_capacity':rows,'token_capacity':engine.prepared.signatures[name].token_capacity,'speculation_enabled':False} for rows,name in trial.base.names.items()}
        for i,c in enumerate(cases):run('serial_'+str(i),[c],width=1)
        pair0=run('pair_0',cases[:2],width=2);pair1=run('pair_1',cases[2:],width=2)
        base=run('four',cases,width=4);repeat=run('four_repeat',cases,width=4)
        slots=run('slot_permutation',cases,width=4,slots=[2,0,3,1]);rows=run('row_permutation',cases,width=4,order=[3,1,0,2])
        zero=deepcopy(cases)
        for c in zero:c['program']={'phase':'all','sites':[]}
        zeros=run('zero_controls',zero,width=4)
        changed=deepcopy(cases);changed[1]['program']={'phase':'all','sites':[]};mutant=run('one_changed',changed,width=4)
        uneven=deepcopy(cases)
        for i,c in enumerate(uneven):c['budget']=POLICY['unequal_budgets'][i]
        holes=run('unequal_lengths',uneven,width=4);holes2=run('unequal_lengths_repeat',uneven,width=4)
        sampled=run('sampled_four',cases,width=4,sampled=True);sampled2=run('sampled_permutation',cases,width=4,slots=[3,1,0,2],order=[2,0,3,1],sampled=True)
        report['checks']={'serial_B4_exact':all(exact_capture(values['serial_'+str(i)],base,0,i) for i in range(4)),
            'serial_B2_exact':all(exact_capture(values['serial_'+str(i)],pair0 if i<2 else pair1,0,i%2) for i in range(4)),
            'four_distinct_native_effects':all(not np.array_equal(base['_logits'][i],zeros['_logits'][i]) for i in range(4)),
            'changed_row_effect':not np.array_equal(base['_logits'][1],mutant['_logits'][1]),
            'repeat_exact':all(exact_capture(base,repeat,i,i) for i in range(4)),
            'slot_permutation_exact':all(exact_capture(base,slots,i,i) for i in range(4)),
            'row_permutation_exact':all(exact_capture(base,rows,i,i) for i in range(4)),
            'unaffected_rows_exact':all(exact_capture(base,mutant,i,i) for i in (0,2,3)),
            'unequal_lengths_exact':all(exact_capture(holes,holes2,i,i) for i in range(4)),
            'sampled_permutation_exact':all(exact_capture(sampled,sampled2,i,i) for i in range(4))}
        report.update(state='complete',passed=all(report['checks'].values()))
    except BaseException as error:report.update(state='failed',passed=False,error=repr(error));raise
    finally:
        if arrays:
            with artifact.open('xb') as stream:np.savez_compressed(stream,**arrays)
            report['artifact']={'filename':artifact.name,'sha256':sha(artifact),'bytes':artifact.stat().st_size,'uncompressed_array_bytes':sum(v.nbytes for v in arrays.values()),'compression':'np.savez_compressed lossless'}
            if artifact.stat().st_size>POLICY['maximum_artifact_bytes']:report.update(state='failed',passed=False,error='Compressed artifact exceeds frozen bound')
        report['finished']=time.time();atomic(output,report)
    return {'passed':report['passed'],'report':output.name,'sha256':sha(output)}


def main():
    if len(sys.argv)>1 and sys.argv[1]=='plan':
        p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--original-plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
        with a.output.open('x') as stream:json.dump(make_plan(json.loads(a.original_plan.read_text())),stream,indent=2,allow_nan=False)
    else:
        import persona_probe as probe
        p=probe.argument_parser();p.add_argument('--capture-plan',type=Path,required=True);p.add_argument('--recover-unstarted',action='store_true');a=p.parse_args();probe.run(a,experiment=experiment,preparation_adapter=prepare_adapter)

if __name__=='__main__':main()
