"""Lossless public authored-response/prefill bundle; no inference or transport.

`plan --root ROOT --output MANIFEST` pins the existing approved source files.
After review and explicit transfer of those reports, `build --root ROOT
--manifest MANIFEST --output ROOT/results/persona_response_capture_bundle.npz`
reads source NPZs in place. No original checkpoint/capture files are changed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from persona_contrastive import scenes, input_digest, build_batches, POLICY
from persona_propagation import checked, IDENTITIES
from persona_vectors import canonical_digest

FILES={'scenes':'data/persona_contrastive_scenes.json',
       'development':'results/persona_contrastive_development.json',
       'pilot':'results/persona_contrastive_pilot.json',
       'propagation':'results/persona_propagation.json',
       'directions':'results/persona_response_directions.npz'}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def ids_sha(value):return hashlib.sha256(np.asarray(value,dtype='<i8').tobytes()).hexdigest()


def plan(root):
    root=Path(root);data,scene_sha=scenes(root)
    files={key:{'path':path,'sha256':sha(root/path)} for key,path in FILES.items()}
    calibration=[{'path':f'results/layer_{i:02d}.npz','sha256':sha(root/f'results/layer_{i:02d}.npz')} for i in range(64)]
    return {'schema':'character-creation.response-bundle-inputs.v1','files':files,'calibration':calibration,
            'counts':{'development':24,'pilot':256,'propagation':60},'scene_sha256':scene_sha,
            'privacy_scope':'Only frozen authored scenes; no private books or arbitrary user inputs.',
            'array_bytes_uncompressed':280*64*5120*8+60*5*4*5120*4}


def load_inputs(root,manifest):
    if manifest.get('schema')!='character-creation.response-bundle-inputs.v1' or set(manifest['files'])!=set(FILES):raise ValueError('Invalid bundle manifest')
    for key,path in FILES.items():
        spec=manifest['files'][key]
        if spec['path']!=path or sha(root/path)!=spec['sha256']:raise ValueError('Frozen input hash differs: '+key)
    if len(manifest['calibration'])!=64:raise ValueError('Need all 64 calibration layers')
    medians=[]
    development=json.loads((root/FILES['development']).read_text());binding=development['binding']
    for i,spec in enumerate(manifest['calibration']):
        if spec['path']!=f'results/layer_{i:02d}.npz' or sha(root/spec['path'])!=spec['sha256']:raise ValueError('Calibration source differs')
        with np.load(root/spec['path'],allow_pickle=False) as a:
            if str(a['model_identity'])!=binding['model_identity']:raise ValueError('Calibration checkpoint differs')
            value=float(a['residual_median'])
            if not np.isfinite(value) or value<=0:raise ValueError('Invalid calibration norm')
            medians.append(value)
    data,scene_sha=scenes(root)
    if scene_sha!=manifest['scene_sha256']:raise ValueError('Authored corpus differs')
    reports={k:json.loads((root/FILES[k]).read_text()) for k in ('development','pilot','propagation')}
    for stage in ('development','pilot'):
        report=reports[stage]
        if (report.get('state')!='complete' or report['stage']!=stage or report['binding']!=binding or report['policy']!=POLICY
                or report['scene_sha256']!=scene_sha or report['input_sha256']!=input_digest(report)):
            raise ValueError('Incomplete or changed controller report')
        expected=build_batches(data,binding,stage,report['direction_artifact'])
        if len(expected)!=len(report['batches']):raise ValueError('Batch coverage differs')
        for want,row in zip(expected,report['batches']):
            # Capture UUIDs differ on regeneration, so compare every other frozen field.
            actual=json.loads(json.dumps(row['input']))
            for a,b in zip(actual['cells'],want['cells']):
                a['capture_response']['id']=b['capture_response']['id']
            if actual!=want or row['state']!='complete' or len(row['result']['cells'])!=len(want['cells']):raise ValueError('Authored batch differs')
    p=reports['propagation'];inputs=p['inputs']
    if p['state']!='complete' or p['passed'] is not True or p['failures'] or p['binding']!=binding or inputs['binding']!=binding or inputs['scene_sha256']!=scene_sha:raise ValueError('Propagation not complete/bound')
    approved={r['id']:r for r in data['evaluation'] if r['split']=='discovery'}
    if len(inputs['contexts'])!=6 or len({c['id'] for c in inputs['contexts']})!=6:raise ValueError('Propagation context coverage differs')
    for c in inputs['contexts']:
        if c['id'] not in approved or c['prompt']!=approved[c['id']]['prompt'] or c['split']!='discovery':raise ValueError('Propagation outside authored discovery corpus')
    direction_sha=manifest['files']['directions']['sha256']
    if reports['pilot']['direction_artifact']!={'filename':Path(FILES['directions']).name,'sha256':direction_sha}:raise ValueError('Pilot direction binding differs')
    if len(inputs['directions'])!=2 or any(d['source_sha256']!=direction_sha for d in inputs['directions']):raise ValueError('Propagation direction binding differs')
    with np.load(root/FILES['directions'],allow_pickle=False) as a:
        for d in inputs['directions']:
            if d['layer']!=32 or d['residual_median']!=medians[32] or not np.array_equal(a['direction_'+d['id']][32],d['vector']):raise ValueError('Propagation direction values differ')
    return data,reports,binding,np.asarray(medians,dtype=np.float64)


class LocalStore:
    def __init__(self,root):self.root=(Path(root)/'results').resolve()
    def get(self,pointer):
        path=Path(pointer['path'])
        if path.parent!=self.root or path.is_symlink() or not re.fullmatch(r'probe-[0-9a-f]{32}\.(npz|jsonl)',path.name):raise ValueError('Unexpected source artifact path')
        if path.stat().st_size>256*1024**2 or sha(path)!=pointer['sha256']:raise ValueError('Source artifact hash/size differs')
        return path


def native(store,record,binding,prompt,model_id=None):
    provenance=store.get(record['provenance_artifact']);saved=json.loads(provenance.read_text())
    request=saved['request'];response=dict(saved['response'],provenance_artifact=record['provenance_artifact'])
    if (request.get('messages')!=[{'role':'user','content':prompt}] or (model_id is not None and request.get('model')!=model_id)
            or canonical_digest(request)!=record['request_sha256'] or response['activation_artifact']!=record['activation_artifact']):
        raise ValueError('Native request outside approved source/binding')
    arrays,_=checked(response,request,binding,store)
    return arrays,request,response


def compact_response(arrays):
    mean=arrays['response_residual_mean'];count=arrays['response_residual_count']
    if mean.dtype!=np.dtype('float64') or mean.shape!=(64,5120) or count.dtype!=np.dtype('int64') or int(count)<=0:raise ValueError('Response dtype/count differs')
    return mean,int(count)


def public_source(record):
    return {'artifact_sha256':record['activation_artifact']['sha256'],'provenance_sha256':record['provenance_artifact']['sha256'],'request_sha256':record['request_sha256']}


def worker_count(value):
    if isinstance(value,bool):raise ValueError('workers must be an integer from 1 to 86')
    try:n=int(value)
    except (ValueError,TypeError):raise ValueError('workers must be an integer from 1 to 86') from None
    if str(n)!=str(value) or not 1<=n<=86:raise ValueError('workers must be an integer from 1 to 86')
    return n


def default_workers():
    available=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    return max(1,min(86,available-2))


def ordered_extract(function,jobs,workers):
    # Futures retain only compact extraction results, never full raw NPZ dictionaries.
    with ThreadPoolExecutor(max_workers=worker_count(workers)) as pool:
        yield from pool.map(function,jobs)


def validate_native_pulse(program,expected_vector):
    if (not isinstance(program,dict) or set(program)!={'phase','sites'} or program['phase']!='prefill'
            or not isinstance(program['sites'],list) or len(program['sites'])!=1
            or set(program['sites'][0])!={'site','vector'} or program['sites'][0]['site']!='layers.32.residual'):
        raise ValueError('Actual pulse site/phase differs')
    actual=np.asarray(program['sites'][0]['vector'],dtype=np.float64);expected=np.asarray(expected_vector,dtype=np.float64)
    if actual.shape!=(5120,) or expected.shape!=actual.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():raise ValueError('Invalid pulse vector')
    # Probe validate_program casts request vectors to FP32 before native steering.
    # Cross-platform FP64 norm reconstruction can differ below one FP32 ULP;
    # require exact equality of the actual runtime representation, never tolerance.
    actual32=actual.astype('<f4');expected32=expected.astype('<f4')
    if not np.isfinite(actual32).all() or not np.array_equal(actual32,expected32):raise ValueError('Actual native FP32 pulse differs from frozen direction/dose')
    return hashlib.sha256(actual32.tobytes()).hexdigest()


def build(root,manifest,output,workers=None):
    root=Path(root);output=Path(output)
    if output.exists() or output.parent.resolve()!=(root/'results').resolve() or output.name!='persona_response_capture_bundle.npz':raise ValueError('Use the new exact flat bundle filename')
    data,reports,binding,medians=load_inputs(root,manifest);store=LocalStore(root)
    means=np.empty((280,64,5120),np.float64);counts=np.empty(280,np.int64);response_index=[]
    workers=default_workers() if workers is None else worker_count(workers)
    jobs=[]
    for stage in ('development','pilot'):
        report=reports[stage]
        for batch in report['batches']:
            for cell_index,(spec,cell) in enumerate(zip(batch['input']['cells'],batch['result']['cells'])):
                jobs.append((len(jobs),stage,report,batch,cell_index,spec,cell))
    if len(jobs)!=280:raise ValueError('Expected exactly 280 response means')
    def extract_response(job):
        index,stage,report,batch,cell_index,spec,cell=job
        out=cell['output'];capture=out['response_capture'];source=spec['_record']
        record={k:out[k] for k in ('activation_artifact','provenance_artifact')};record['request_sha256']=capture['request_sha256']
        a,request,response=native(store,record,binding,spec['prompt'],report['model_id'])
        if out.get('complete') is not True or response['response_capture']!=capture or request['seed']!=batch['input']['seed']:raise ValueError('Controller/native capture differs')
        mean,count=compact_response(a)
        metadata=dict(index=index,stage=stage,scene_id=source['scene_id'],split=source['split'],arm=source['arm'],seed=request['seed'],
            source_sha256=source['source_sha256'],replay='replay_of' in source,replay_of=source.get('replay_of'),batch_id=batch['batch_id'],cell_index=cell_index,
            **public_source(record),prompt_token_ids_sha256=ids_sha(a['prompt_token_ids']),consumed_token_ids_sha256=ids_sha(a['response_consumed_token_ids']),
            consumed_positions_sha256=ids_sha(a['response_consumed_positions']),generated_token_ids_sha256=ids_sha(a['generated_token_ids']),
            sampled_generated_tokens=len(a['generated_token_ids']),consumed_generated_tokens=count)
        return mean,count,metadata
    for index,(mean,count,metadata) in enumerate(ordered_extract(extract_response,jobs,workers)):
        means[index]=mean;counts[index]=count;response_index.append(metadata)
    p=reports['propagation'];records={r['attempt']:r for r in p['records']};contexts={c['id']:c for c in p['inputs']['contexts']}
    snapshots=[];positions=[];token_ids=[];prop_index=[];used=set();jobs=[]
    for result in p['results']:
        context=contexts[result['context_id']]
        arms=[('zero',0.,result['zero_attempt']),('zero_repeat',0.,result['zero_repeat_attempt'])]
        arms += [(d['direction_id'],arm['dose'],arm['attempt']) for d in result['directions'] for arm in d['arms']]
        if len(arms)!=10 or {(d,v) for d,v,_ in arms}!={('zero',0.),('zero_repeat',0.),*((d,v) for d in ('A','B') for v in (-.1,-.05,.05,.1))}:raise ValueError('Propagation arm coverage differs')
        for direction,dose,attempt in arms:
            if attempt in used:raise ValueError('Duplicate propagation attempt')
            used.add(attempt);jobs.append((len(jobs),context,direction,dose,attempt,records[attempt]))
    def extract_propagation(job):
        index,context,direction,dose,attempt,record=job
        a,request,response=native(store,record,binding,context['prompt'])
        expected_positions=[context['source_position']+lag for lag in (0,1,2,4,8)]
        if (request['steering_pulse']!={'position':context['source_position']} or request['capture_prefill']['positions']!=expected_positions
                or request['capture_prefill']['layers']!=[31,32,48,63] or request.get('observation_only') is not True):raise ValueError('Propagation geometry differs')
        pulse_vector_sha256=None
        if direction in ('zero','zero_repeat'):
            if request.get('steering') is not None:raise ValueError('Zero propagation arm contains steering')
        else:
            d=next(d for d in p['inputs']['directions'] if d['id']==direction)
            vector=np.asarray(d['vector'],dtype=float);vector=vector/np.linalg.norm(vector)*d['residual_median']*dose
            pulse_vector_sha256=validate_native_pulse(request.get('steering'),vector)
        values=a['prefill_snapshot_residuals']
        if values.dtype!=np.dtype('float32') or values.shape!=(5,4,5120):raise ValueError('Propagation snapshot dtype/shape differs')
        metadata=dict(index=index,scene_id=context['id'],split='discovery',source_position=context['source_position'],direction=direction,dose=dose,
            attempt=attempt,seed=request['seed'],native_fp32_pulse_vector_sha256=pulse_vector_sha256,source_sha256=hashlib.sha256(context['prompt'].encode()).hexdigest(),**public_source(record),prompt_token_ids_sha256=ids_sha(a['prompt_token_ids']),snapshot_token_ids_sha256=ids_sha(a['prefill_snapshot_token_ids']))
        return values,a['prefill_snapshot_positions'],a['prefill_snapshot_token_ids'],metadata
    for values,pos,tokens,metadata in ordered_extract(extract_propagation,jobs,workers):
        snapshots.append(values);positions.append(pos);token_ids.append(tokens);prop_index.append(metadata)
    if len(snapshots)!=60 or len({r['context_id'] for r in p['results']})!=6 or used!=set(records):raise ValueError('Expected all 60 propagation records exactly once')
    metadata={'schema':'character-creation.response-capture-bundle.v1','binding':binding,'inputs_sha256':canonical_digest(manifest),
        'input_hashes':{k:v['sha256'] for k,v in manifest['files'].items()},'calibration_source_sha256':[v['sha256'] for v in manifest['calibration']],
        'producer_sha256':sha(__file__),'pulse_vector_validation':'Exact FP32 array equality matching native validate_program cast; SHA256 uses contiguous little-endian FP32 bytes. Original request hashes bind pre-cast JSON. No numerical tolerance.',
        'responses':response_index,'propagation':prop_index,'semantics':'Response FP64 means over consumed generated tokens only; final sampled token excluded. Propagation FP32 fixed-input block-output residual snapshots. Replays are not independent evidence; no causal wiring or coherence claim.',
        'privacy_scope':'Frozen authored scenes only. No private corpus, arbitrary user prompts, answers, or local paths.'}
    with output.open('xb') as stream:
        np.savez(stream,metadata=np.asarray(json.dumps(metadata,sort_keys=True,allow_nan=False)),response_mean=means,response_count=counts,
            propagation_residual=np.stack(snapshots),propagation_positions=np.stack(positions),propagation_token_ids=np.stack(token_ids),
            propagation_layers=np.array([31,32,48,63],dtype=np.int64),calibration_residual_median=medians)
    if output.stat().st_size>=1024**3:raise ValueError('Bundle exceeds 1 GiB; artifact retained for inspection')
    return {'path':str(output),'sha256':sha(output),'bytes':output.stat().st_size,'responses':280,'propagation':60}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('plan','build'));p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=worker_count,default=default_workers());a=p.parse_args(argv)
    if a.mode=='plan':
        with a.output.open('x') as f:json.dump(plan(a.root),f,indent=2)
        print(a.output,sha(a.output))
    else:
        if not a.manifest:p.error('--manifest required')
        print(json.dumps(build(a.root,json.loads(a.manifest.read_text()),a.output,a.workers)))

if __name__=='__main__':main()
