"""Frozen response-contrast development and a 256-answer six-arm persona pilot.

Uses existing leased workers through the explorer queue. No training, lease
creation, automatic retries or preset promotion. Held-out outcomes never fit
vectors or select doses. All scenes are newly authored and redistributable.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import time
import urllib.request
import uuid
import numpy as np
from persona_vectors import canonical_digest, ModelSpec, VectorBank
from persona_coherence_eval import evaluate_surface
from persona_association_campaign import atomic, matching_binding

VIEWER='http://127.0.0.1:8792'
ARMS=('baseline','A','A_reverse','B','B_reverse','random')
POLICY={'answers':256,'balanced_answers':252,'replay_answers':4,'arms':list(ARMS),
        'seeds':[17031,29147],'layer':32,'dose':.1,'phase':'all','random_seed':481516,
        'max_tokens':2048,'development_seed':90731,'discovery_families':12,
        'heldout_families':9,'promotion_eligible':False}


def fetch(url, value=None):
    request=urllib.request.Request(url, data=None if value is None else json.dumps(value,allow_nan=False).encode(),
                                   headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=90) as response:return json.load(response)


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scenes(root):
    path=root/'data/persona_contrastive_scenes.json'; value=json.loads(path.read_text())
    if (value['schema']!='character-creation.contrastive-scenes.v1' or len(value['development'])!=8
            or len(value['evaluation'])!=21 or value['policy']['answers']!=256
            or value['policy']!=POLICY
            or Counter(x['split'] for x in value['evaluation'])!=Counter(discovery=12,heldout=9)):
        raise ValueError('Frozen scene counts/policy differ')
    rows=value['development']+value['evaluation']
    if (len({x['family_id'] for x in rows})!=29 or len({x['id'] for x in rows})!=29
            or any(x['split']!='development' for x in value['development'])):raise ValueError('Family/split leakage')
    for row in rows:
        if hashlib.sha256(row['prompt'].encode()).hexdigest()!=row['source_sha256']:raise ValueError('Scene hash differs')
    return value,file_sha(path)


def recipe(arm, policy, artifact=None):
    push=[]
    if arm=='random':push=[{'kind':'random','site':'residual','layer':policy['layer'],'dose':policy['dose'],'seed':policy['random_seed']}]
    elif arm!='baseline':
        if arm not in ARMS or artifact is None:raise ValueError('Unknown arm or missing fitted direction')
        push=[{'kind':'direction','layer':policy['layer'],'dose':-policy['dose'] if arm.endswith('_reverse') else policy['dose'],
               'artifact':artifact['filename'],'sha256':artifact['sha256'],'direction':arm[0]}]
    names={'baseline':'Baseline','A':'Candidate: dry skeptical wit','A_reverse':'Reversed: dry skeptical wit',
           'B':'Candidate: warm attentive candor','B_reverse':'Reversed: warm attentive candor','random':'Matched random residual control'}
    return {'name':names[arm],'pushes':push,'phase':policy['phase']}


def build_batches(data, binding, stage, artifact=None):
    """Arm order rotates before outcomes; two seeds share the same scenes."""
    policy=data['policy']; batches=[]
    if stage=='development':
        cells=[]
        for row in data['development']:
            for style in ('baseline','A','B'):
                cells.append({'label':row['id']+' / '+style,'prompt':data['styles'][style]+'\n\n'+row['prompt'],
                    'recipe':recipe('baseline',policy),'capture_response':{'id':uuid.uuid4().hex,'layers':[],'max_positions':0},
                    '_record':{'scene_id':row['id'],'split':'development','arm':style,'source_sha256':row['source_sha256']}})
        for i in range(0,24,6):batches.append({'seed':policy['development_seed'],'cells':cells[i:i+6]})
    elif stage=='pilot':
        for i,row in enumerate(data['evaluation']):
            for j,seed in enumerate(policy['seeds']):
                order=list(ARMS);offset=(i*2+j)%6;order=order[offset:]+order[:offset]
                cells=[{'label':row['id']+' / '+arm,'prompt':row['prompt'],'recipe':recipe(arm,policy,artifact),
                        'capture_response':{'id':uuid.uuid4().hex,'layers':[],'max_positions':0},
                        '_record':{'scene_id':row['id'],'split':row['split'],'arm':arm,'source_sha256':row['source_sha256']}}
                       for arm in order]
                batches.append({'seed':seed,'cells':cells})
        replay=[]
        # Four preregistered request replays; never counted as new scene evidence.
        for index,arm in enumerate(('baseline','A','B','random')):
            original=next(c for c in batches[index*2]['cells'] if c['_record']['arm']==arm)
            cell=deepcopy(original);cell['_record']['replay_of']={'batch_index':index*2,'arm':arm}
            cell['capture_response']['id']=uuid.uuid4().hex;cell['label']+=' / replay';replay.append(cell)
        batches.append({'seed':policy['seeds'][0],'cells':replay})
    else:raise ValueError('Unknown stage')
    for i,batch in enumerate(batches):
        batch.update(model_identity=binding['model_identity'],max_tokens=policy['max_tokens'],thinking=False,smoke=False)
    expected=24 if stage=='development' else 256
    if sum(len(b['cells']) for b in batches)!=expected:raise ValueError('Answer count differs')
    return batches


def input_digest(plan):
    frozen={k:plan[k] for k in ('schema','stage','binding','model_id','scene_sha256','policy','direction_artifact','qualification_sha256')}
    frozen['batches']=[{'index':row['index'],'input':row['input'],'state':'pending'} for row in plan['batches']]
    return canonical_digest(frozen)


def validate_result(result, row, binding, bank, model_id):
    request=deepcopy(row['input'])
    for cell in request['cells']:cell.pop('_record')
    if (result.get('id')!=row['batch_id'] or len(result['cells'])!=len(request['cells'])
            or any(result['input'].get(k)!=v for k,v in request.items())):
        raise ValueError('Returned batch input differs from frozen request')
    for index,(expected,cell) in enumerate(zip(request['cells'],result['cells'])):
        if (cell.get('index')!=index or cell['label']!=expected['label'] or cell['recipe']!=expected['recipe']
                or cell['prompt']['text']!=expected['prompt']):raise ValueError('Returned cell differs from frozen input')
        output=cell.get('output') or {};capture=output.get('response_capture') or {}
        if cell['state']!='complete' or output.get('complete') is not True:raise ValueError('Incomplete pilot cell; no automatic replay')
        program,details=bank.program(expected['recipe']['pushes'],expected['recipe']['phase'])
        native={'model':model_id,'messages':[{'role':'user','content':expected['prompt']}],
                'seed':request['seed'],'temperature':.7,'top_p':.95,'top_k':20,'max_tokens':request['max_tokens'],
                'stream':False,'chat_template_kwargs':{'enable_thinking':False},
                'capture_response':expected['capture_response'],'metadata':{'job_id':row['batch_id']+'-'+str(index)}}
        if program['sites']:native['steering']=program
        request_sha=canonical_digest(native)
        if (capture.get('id')!=expected['capture_response']['id'] or capture.get('mean_defined') is not True
                or capture.get('consumed_generated_tokens',-1)!=capture.get('sampled_generated_tokens',-1)-1
                or capture.get('all_consumed_generated_tokens_captured') is not True
                or capture.get('generation_complete') is not True
                or output.get('provenance',{}).get('program_sha256')!=details['program_sha256']):
            raise ValueError('Missing or inconsistent response capture/program')
        for evidence in (capture,capture.get('artifact',{}),output.get('provenance',{})):
            if (evidence.get('request_sha256')!=request_sha
                    or any(evidence.get(k)!=v for k,v in binding.items() if k!='layout_sha256')):
                raise ValueError('Capture request/runtime binding differs')
        cell['surface_evaluation']=evaluate_surface(output)


def artifact_local(root, artifact):
    remote=artifact['path']; name=Path(remote).name
    if not re.fullmatch(r'probe-[0-9a-f]{32}\.npz',name) or remote!='/mnt/storage16tb/Character Creation/results/'+name:
        raise ValueError('Unexpected native artifact location')
    target=root/'results'/name
    if not target.exists():
        subprocess.run(['scp','-q','v100:'+remote,str(target)],check=True,timeout=120)
    if file_sha(target)!=artifact['sha256']:raise ValueError('Downloaded native artifact hash differs')
    return target


def run(root, stage):
    data,scene_sha=scenes(root); launch=json.loads((root/'results/persona_contrastive_launch.json').read_text())
    binding=launch['binding']; deadline=launch['deadline']-180
    if not time.time()<deadline:raise ValueError('Serving lease expired')
    qualification=json.loads((root/'results/persona_capture_qualification.json').read_text())
    if qualification.get('passed') is not True or qualification.get('binding')!=binding:
        raise ValueError('Native capture is not qualified for this exact runtime')
    urls=[e['url'] for e in launch['endpoints']]
    models=[fetch(u+'/v1/models') for u in urls]
    live=matching_binding(models,binding['model_identity'])
    model_id=models[0]['data'][0]['id']
    if any(live.get(k)!=v for k,v in binding.items()):raise ValueError('Serving runtime differs')
    artifact=None
    if stage=='pilot':
        path=root/'results/persona_response_directions.npz'
        artifact={'filename':path.name,'sha256':file_sha(path)}
    path=root/('results/persona_contrastive_'+stage+'.json')
    if path.exists():raise ValueError('Existing stage is never silently retried; inspect its durable attempts')
    batches=build_batches(data,binding,stage,artifact)
    plan={'schema':'character-creation.contrastive-campaign.v1','stage':stage,'binding':binding,'model_id':model_id,'scene_sha256':scene_sha,
          'policy':data['policy'],'direction_artifact':artifact,'qualification_sha256':file_sha(root/'results/persona_capture_qualification.json'),
          'batches':[{'index':i,'input':b,'state':'pending'} for i,b in enumerate(batches)],'state':'frozen',
          'automatic_default_promotion':False,'created':time.time()}
    plan['input_sha256']=input_digest(plan)
    bank=VectorBank(root/'results',ModelSpec.load(root/'data/persona_model.json'))
    atomic(path,plan); active=[]; next_index=0
    while active or next_index<len(batches):
        if time.time()>=deadline:
            plan['state']='deadline';atomic(path,plan);raise RuntimeError('Lease deadline reached; inspect active batches')
        while len(active)<4 and next_index<len(batches):
            row=plan['batches'][next_index];request=deepcopy(row['input'])
            for cell in request['cells']:cell.pop('_record')
            row.update(state='submitting',submitted=time.time(),request_sha256=canonical_digest(request));atomic(path,plan)
            # Never retry ambiguous POSTs: durable state remains 'submitting'.
            response=fetch(VIEWER+'/api/batches',request)
            row.update(state='queued',batch_id=response['id']);active.append(row);next_index+=1;atomic(path,plan)
        for row in list(active):
            result=fetch(VIEWER+'/api/batch?id='+row['batch_id'])
            row['result']=result;row['state']=result['state']
            if result['state'] not in ('queued','running'):
                validate_result(result,row,binding,bank,model_id)
                active.remove(row)
                print(stage, sum(len(r.get('result',{}).get('cells',[])) for r in plan['batches'] if r['state']=='complete'), 'answers complete',flush=True)
            atomic(path,plan)
        time.sleep(2)
    plan['state']='complete';plan['completed']=time.time();atomic(path,plan)


def fit(root):
    data,scene_sha=scenes(root);path=root/'results/persona_contrastive_development.json';plan=json.loads(path.read_text())
    if plan['state']!='complete' or plan['scene_sha256']!=scene_sha:raise ValueError('Development incomplete or changed')
    if plan['input_sha256']!=input_digest(plan) or plan['policy']!=POLICY:raise ValueError('Frozen development input changed')
    cells={}
    bank=VectorBank(root/'results',ModelSpec.load(root/'data/persona_model.json'))
    for batch in plan['batches']:
        validate_result(batch['result'],batch,plan['binding'],bank,plan['model_id'])
        for spec,result in zip(batch['input']['cells'],batch['result']['cells']):
            key=(spec['_record']['scene_id'],spec['_record']['arm'])
            if key in cells:raise ValueError('Duplicate development cell')
            source=next((r for r in data['development'] if r['id']==key[0]),None)
            if (source is None or key[1] not in ('baseline','A','B') or spec['_record']['split']!='development'
                    or spec['prompt']!=data['styles'][key[1]]+'\n\n'+source['prompt']
                    or batch['input']['seed']!=POLICY['development_seed']):raise ValueError('Development source/split differs')
            cells[key]=result
    if set(cells)!={(r['id'],s) for r in data['development'] for s in ('baseline','A','B')}:raise ValueError('Development coverage differs')
    artifacts=[cell['output']['response_capture']['artifact'] for cell in cells.values()]
    with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(lambda a:artifact_local(root,a),artifacts))
    means={};sources=[]
    for key,cell in cells.items():
        artifact=cell['output']['response_capture']['artifact'];local=artifact_local(root,artifact)
        with np.load(local,allow_pickle=False) as arrays:
            mean=arrays['response_residual_mean'].astype(np.float64);count=int(arrays['response_residual_count'])
            if mean.shape!=(64,5120) or count<=0 or not np.isfinite(mean).all():raise ValueError('Invalid response mean')
            if not np.array_equal(arrays['response_consumed_token_ids'],arrays['generated_token_ids'][:-1]):raise ValueError('Response token coverage differs')
            capture=cell['output']['response_capture'];consumed=arrays['response_consumed_token_ids'];positions=arrays['response_consumed_positions']
            if (count!=len(consumed) or count!=capture['consumed_generated_tokens'] or not bool(arrays['response_residual_mean_defined'])
                    or not np.array_equal(positions,np.arange(len(arrays['prompt_token_ids']),len(arrays['prompt_token_ids'])+count))
                    or not np.allclose(mean*count,arrays['response_residual_sum'],rtol=1e-13,atol=1e-10)):
                raise ValueError('Response averaging count/positions differ')
            for name,hash_key in (('response_consumed_token_ids','consumed_token_ids_sha256'),('generated_token_ids','generated_token_ids_sha256'),('response_consumed_positions','consumed_positions_sha256')):
                if hashlib.sha256(np.asarray(arrays[name],dtype='<i8').tobytes()).hexdigest()!=capture[hash_key]:raise ValueError('Response token/position hash differs')
            if str(arrays['request_sha256'])!=cell['output']['provenance']['request_sha256'] or str(arrays['request_sha256'])!=artifact['request_sha256']:
                raise ValueError('NPZ request binding differs')
            if any(str(arrays[k])!=v for k,v in plan['binding'].items() if k!='layout_sha256'):raise ValueError('NPZ binding differs')
        means[key]=mean;sources.append({'scene_id':key[0],'style':key[1],'artifact_sha256':artifact['sha256'],'consumed_tokens':count,'request_sha256':artifact['request_sha256']})
    directions={};ranking={}
    for style in ('A','B'):
        paired=np.stack([means[(row['id'],style)]-means[(row['id'],'baseline')] for row in data['development']])
        direction=paired.mean(axis=0);directions['direction_'+style]=direction
        # Vector dimensions are residual coordinates, not MLP neuron IDs.
        norms=np.linalg.norm(paired,axis=2);mean_norm=np.linalg.norm(direction,axis=1)
        consistency=np.divide(mean_norm,norms.mean(axis=0),out=np.zeros(64),where=norms.mean(axis=0)>0)
        ranking[style]={'layer_contrast_l2':mean_norm.tolist(),'context_direction_consistency':consistency.tolist(),
                        'selected_layer':data['policy']['layer'],'selected_before_outputs':True,
                        'meaning':'Paired response residual contrast; context consistency is not causal or prose coherence.'}
        candidates=[]
        for layer in range(64):
            calibrated=bank.layer(layer);columns=calibrated['down_columns'].astype(np.float64)
            column_norms=np.linalg.norm(columns,axis=0)
            projections=paired[:,layer]@np.divide(columns,column_norms,out=np.zeros_like(columns),where=column_norms>0)
            denom=np.linalg.norm(paired[:,layer],axis=1)[:,None]
            cosines=np.divide(projections,denom,out=np.zeros_like(projections),where=denom>0)
            for index,neuron in enumerate(calibrated['neuron_ids']):
                score=float(cosines[:,index].mean())
                candidates.append({'layer':layer,'neuron':int(neuron),'mean_signed_cosine':score,
                    'paired_context_sign_agreement':float(np.mean(np.sign(cosines[:,index])==np.sign(score)))})
        ranking[style]['mapped_neuron_alignment']=sorted(candidates,key=lambda x:-abs(x['mean_signed_cosine']))[:32]
        ranking[style]['alignment_coverage']=len(candidates)
        ranking[style]['alignment_semantics']='Decoded output-column alignment to paired response residual differences over the existing calibrated candidate bank only. Not measured neuronal activity, causal influence, or label confidence.'
    metadata={'schema':'character-creation.response-directions.v1',**{k:plan['binding'][k] for k in ('model_identity','runtime_identity','prepared_identity')},
              'split':'development','sources':sources,'scene_sha256':scene_sha,'development_sha256':file_sha(path),
              'method':'Equal-scene mean of response-only per-token means: instructed style minus matched natural voice; excludes sampled-but-unconsumed final token.',
              'limitation':'Response contrasts can reflect lexical, length and instruction effects; only intervention evaluation tests behavioral usefulness.'}
    target=root/'results/persona_response_directions.npz'
    with target.open('xb') as stream:np.savez_compressed(stream,metadata=json.dumps(metadata,sort_keys=True),**directions)
    atomic(root/'results/persona_response_contrasts.json',{'metadata':metadata,'artifact':{'filename':target.name,'sha256':file_sha(target)},'ranking':ranking,'automatic_default_promotion':False})
    print('fitted',target.name,file_sha(target),flush=True)


def blind(root):
    data,scene_sha=scenes(root);plan=json.loads((root/'results/persona_contrastive_pilot.json').read_text())
    if plan['state']!='complete' or plan['scene_sha256']!=scene_sha:raise ValueError('Pilot incomplete')
    if plan['input_sha256']!=input_digest(plan):raise ValueError('Frozen pilot input changed')
    by_scene={r['id']:r for r in data['evaluation']};groups=[];key=[];rng=random.Random(112358)
    for batch in plan['batches']:
        if batch['index']==42:continue
        scene=by_scene[batch['input']['cells'][0]['_record']['scene_id']];order=list(range(6));rng.shuffle(order)
        answers=[]
        for i in order:
            cell=batch['result']['cells'][i];spec=batch['input']['cells'][i];code=hashlib.sha256((str(batch['index'])+':'+str(i)+':blinded112358').encode()).hexdigest()[:12]
            answers.append({'code':code,'answer':cell['output']['answer']})
            key.append({'code':code,'arm':spec['_record']['arm'],'scene_id':scene['id'],'split':scene['split'],'seed':batch['input']['seed'],'batch_index':batch['index'],'cell_index':i})
        groups.append({'group':batch['index'],'prompt':scene['prompt'],'fact_contract':scene['fact_contract'],'answers':answers})
    packet={'schema':'character-creation.blind-persona-eval.v1','rubric':{
        'dry_wit':'0 absent,1 slight,2 clear,3 strong,4 dominant; understated irony/skeptical specificity, not merely bluntness or cruelty',
        'attentive_candor':'0 absent,1 slight,2 clear,3 strong,4 dominant; grounded emotional attention plus honesty, not merely politeness/flattery',
        'fidelity':'ok/issue/uncertain; quote exact unsupported assertions, changed details or consequential omissions against fact_contract',
        'coherence':'ok/issue/uncertain; intelligibility and consistency, independent of tone',
        'instructions':'Judge each answer independently; allow both styles or neither. Do not infer intervention or reward positivity. JSON rows: code,dry_wit,attentive_candor,fidelity,coherence,evidence. Evidence concise and exact.'},'groups':groups}
    atomic(root/'results/persona_pilot_blind.json',packet);atomic(root/'results/persona_pilot_blind_key.json',key)
    # Replay comparators use token hashes, never string-only equality.
    replays=[]
    for spec,cell in zip(plan['batches'][42]['input']['cells'],plan['batches'][42]['result']['cells']):
        ref=spec['_record']['replay_of'];batch=plan['batches'][ref['batch_index']]
        i=next(i for i,c in enumerate(batch['input']['cells']) if c['_record']['arm']==ref['arm'])
        first=batch['result']['cells'][i]
        replays.append({'scene_id':spec['_record']['scene_id'],'arm':ref['arm'],
            'token_ids_equal':first['output']['response_capture']['generated_token_ids_sha256']==cell['output']['response_capture']['generated_token_ids_sha256'],
            'answer_equal':first['output']['answer']==cell['output']['answer'],
            'original_endpoint':first.get('endpoint_label'),'replay_endpoint':cell.get('endpoint_label')})
    atomic(root/'results/persona_pilot_replays.json',{'replays':replays,'independent_scene_evidence':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['development','fit','pilot','blind','campaign']);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    args=parser.parse_args()
    if args.stage=='campaign':
        deadline=json.loads((args.root/'results/persona_contrastive_launch.json').read_text())['deadline']-180
        while True:
            path=args.root/'results/persona_capture_qualification.json'
            if path.exists():
                qualification=json.loads(path.read_text())
                if qualification.get('state')!='running':
                    if qualification.get('passed') is not True:raise ValueError('Native capture qualification failed')
                    break
            if time.time()>=deadline:raise ValueError('Lease expired while waiting for qualification')
            time.sleep(5)
        run(args.root,'development');fit(args.root);run(args.root,'pilot');blind(args.root)
    elif args.stage in ('development','pilot'):run(args.root,args.stage)
    elif args.stage=='fit':fit(args.root)
    else:blind(args.root)

if __name__=='__main__':main()
