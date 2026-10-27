"""Offline runtime parity qualification. Length-limited prose is NOT complete prose.

Never submits requests, leases GPUs, retries uncertain work, or modifies inputs.
Production persona_propagation.checked remains unchanged.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from persona_vectors import canonical_digest
from persona_trace import ResponseCapture
from persona_probe import normalize_request,validate_completion,file_hash
from persona_propagation import ArtifactStore

IDENTITIES=('model_identity','runtime_identity','prepared_identity')
EXACT=('final_chunk_positions','virtual_prefix_slots','prompt_token_ids','generated_token_ids','prefill_positions','mlp_input_last','residual_last','pre_mlp_last',
       'residual_l2','residual_l2_count','residual_l2_min','residual_l2_max','residual_l2_mean','residual_l2_median',
       'response_residual_sum','response_residual_mean','response_residual_count','response_residual_mean_defined',
       'response_consumed_token_ids','response_consumed_positions','response_selected_layers','response_selected_positions','response_selected_residuals')


def require(value,message):
    if not value:raise ValueError(message)


def digest(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def ids_binding(value):return {k:value[k] for k in IDENTITIES}


def load_plan(path):
    plan=json.loads(Path(path).read_text())
    require(plan['schema']=='character-creation.batch-scheduler-qualification-plan.v1' and len(plan['jobs'])==24,'Frozen24-case plan required')
    require(canonical_digest({k:v for k,v in plan.items() if k!='input_sha256'})==plan['input_sha256'],'Plan input hash differs')
    require(len({j['id'] for j in plan['jobs']})==24 and all(hashlib.sha256(j['prompt'].encode()).hexdigest()==j['source_sha256'] for j in plan['jobs']),'Plan case/source identity differs')
    require(all(plan['policy'].get(k) is True for k in ('response_token_ids_exact','final_prefill_arrays_exact','response_sums_means_exact','request_sha256_exact','all_24_required','no_automatic_retries')),'Frozen parity policy differs')
    return plan


def request_for(job,bank,model_id='checkpoint'):
    """Create frozen original request; caller may POST only never-submitted jobs."""
    program,provenance=bank.program(job['recipe']['pushes'],job['recipe']['phase'])
    request={'model':model_id,'messages':[{'role':'user','content':job['prompt']}],'seed':job['seed'],'temperature':.7,'top_p':.95,'top_k':20,
        'max_tokens':job['max_tokens'],'stream':False,'chat_template_kwargs':{'enable_thinking':False},'capture_response':job['capture_response'],'metadata':{'job_id':job['id']}}
    if program['sites']:request['steering']=program
    return request,provenance


def check_job_request(job,request):
    require(request['messages']==[{'role':'user','content':job['prompt']}] and request['seed']==job['seed'] and request['max_tokens']==job['max_tokens']
        and request['capture_response']==job['capture_response'] and request.get('metadata')=={'job_id':job['id']},'Request differs from frozen job')
    spec=normalize_request(request,request['model'])
    require(not spec['observation_only'] and all(spec[k] is None for k in ('kv_control','capture_search','capture_trace','capture_prefill','steering_pulse','prefill_stop')),'Unsupported parity request')
    require(spec['requested_sampling']['temperature']==.7 and spec['requested_sampling']['top_p']==.95 and spec['requested_sampling']['top_k']==20,'Frozen sampling differs')
    if job['arm']=='baseline':require(not request.get('steering',{}).get('sites'),'Baseline contains steering')


def checked_parity(response,request,expected_binding,store):
    """Strict native evidence check accepting stop OR correct full-budget length.

    Unlike production checked(), truncated/empty prose does not invalidate neural
    parity. It remains explicitly classified and cannot be called complete prose.
    """
    bound=ids_binding(expected_binding);request_hash=canonical_digest(request);spec=normalize_request(request,request['model'])
    require(ids_binding(response)==bound and response['operation']=='rollout','Response binding/operation differs')
    paths={}
    for kind in ('activation_artifact','provenance_artifact'):
        pointer=response[kind];require(ids_binding(pointer)==bound and pointer['request_sha256']==request_hash,'Pointer binding differs')
        paths[kind]=store.get(pointer);require(file_hash(paths[kind])==pointer['sha256'],'Pointer bytes differ')
    report=json.loads(paths['provenance_artifact'].read_text())
    require(report['status']=='complete' and report['state_restored'] is True and report['request']==request and report['request_sha256']==request_hash
        and ids_binding(report)==bound and report['response']=={k:v for k,v in response.items() if k!='provenance_artifact'},'Original provenance differs')
    manifest=report['worker_manifest'];require(ids_binding(manifest)==bound and manifest['calibration_equivalence']['passed'] is True,'Worker source/calibration binding differs')
    source={**manifest['source_sha256'],'heliopause':manifest['heliopause_sha256']}
    require(source and all(re.fullmatch('[0-9a-f]{64}',v) for v in source.values()),'Invalid native source hashes')
    # Reconstruct the actual probe runtime identity from its exact source inputs.
    runtime={k:manifest[k] for k in ('script_sha256','helper_sha256','source_sha256','torch','heliopause_sha256')}
    for key in ('experiment_source_sha256','preparation_adapter_source_sha256'):
        if key in manifest:runtime[key]=manifest[key]
    runtime['preparation']={k:v for k,v in manifest['preparation'].items() if k not in ('compile_workers','kernel_cache','cutlass_include','headroom_bytes','snapshot_budget_bytes')}
    require(canonical_digest(runtime)==bound['runtime_identity'],'Worker runtime source digest differs')
    with np.load(paths['activation_artifact'],allow_pickle=False) as z:
        require(str(z['request_sha256'])==request_hash and all(str(z[k])==v for k,v in bound.items()),'NPZ identity differs')
        arrays={k:z[k].copy() for k in z.files if z[k].dtype.kind not in 'US' or k=='completion_graph_kind'}
    require(all(np.isfinite(a).all() for a in arrays.values() if a.dtype.kind in 'fc'),'Nonfinite arrays')
    ids=arrays['prompt_token_ids'];generated=arrays['generated_token_ids'];trace=arrays['completion_trace'];kinds=arrays['completion_graph_kind'].tolist()
    require(ids.dtype==generated.dtype==np.int64 and ids.ndim==generated.ndim==1 and len(ids)>0 and 1<=len(generated)<=spec['max_tokens'],'Token shape/budget')
    require(int(arrays['virtual_prefix_slots'])==0 and np.array_equal(arrays['prefill_positions'],np.arange(len(ids))),'Prompt token/position coverage')
    require(trace.ndim==2 and trace.shape[1]==8 and trace.dtype==np.int64 and len(trace)==len(kinds) and kinds[0]=='init' and set(kinds)<={'init','prefill','decode'} and kinds.count('init')==1,'Completion trace schema')
    require(np.all(np.diff(trace[:,1])>0),'Completion sequence must increase')
    admission=report['effective_admission'];eos=set(admission['eos_token_ids']);epoch=admission['generation']
    require(admission['seed']==spec['seed'] and admission['max_new_tokens']==spec['max_tokens'] and all(admission[k]==v for k,v in spec['sampling'].items()),'Native admission differs')
    require(not any(int(t) in eos for t in generated[:-1]),'Continued generation after EOS')
    consumed=count=0
    for row,kind in zip(trace,kinds):
        sampled=False
        if kind=='init':require(consumed==count==0,'Late init')
        elif kind=='prefill':
            require(count==0 and consumed<int(row[2])<=len(ids),'Prefill progress differs');consumed=int(row[2]);sampled=consumed==len(ids);count=int(sampled)
        else:
            require(consumed>=len(ids) and count>=1,'Decode before prefill');consumed+=1;count+=1;sampled=True
        observed=[epoch,consumed,count,int(generated[count-1]) if count else -1,int(row[5]),int(row[6]),int(row[7])]
        validate_completion(row[:7],observed,epoch=epoch,sequence=int(row[1]),consumed=consumed,generated=count,sampled=sampled,budget=spec['max_tokens'],eos=eos,vocabulary=2**31)
        require(int(row[0])==epoch and int(row[2])==consumed and int(row[3])==count and (not count or int(row[4])==int(generated[count-1])),'Raw token/epoch progression differs')
    require(count==len(generated) and consumed==len(ids)+len(generated)-1 and int(trace[-1,5])==2,'Terminal native coverage differs')
    finish='stop' if int(generated[-1]) in eos else 'length'
    require(finish=='stop' or len(generated)==spec['max_tokens'],'Unexplained non-EOS early termination')
    require(response['choices'][0]['finish_reason']==finish,'Finish reason differs')
    answer=response['choices'][0]['message']['content'];thinking='<think>' in answer or '</think>' in answer
    complete=finish=='stop' and bool(answer.strip()) and not thinking
    require(response['diagnostics']['complete_final_answer'] is complete and response['diagnostics']['unexpected_thinking'] is thinking,'Prose completion misclassified')
    require(response['usage']=={'prompt_tokens':len(ids),'completion_tokens':len(generated),'total_tokens':len(ids)+len(generated)},'Usage differs')
    for name,shape in [('residual_last',(64,5120)),('pre_mlp_last',(64,5120)),('mlp_input_last',(64,17408)),('residual_l2',(64,len(ids)-1))]:
        require(arrays[name].shape==shape and arrays[name].dtype==np.float32,'Native activation shape/dtype differs: '+name)
    require(int(arrays['residual_l2_count'])==len(ids)-1,'Norm count differs')
    norms=arrays['residual_l2']
    for name,fn in [('min',np.min),('max',np.max),('mean',lambda a,axis:np.mean(a,axis=axis,dtype=np.float64)),('median',np.median)]:
        expected=fn(norms,axis=1) if norms.shape[1] else np.zeros(64,np.float64)
        require(np.array_equal(arrays['residual_l2_'+name],expected),'Norm summary differs')
    validate_parity_response(response,request,arrays,complete)
    require(arrays['response_residual_sum'].dtype==arrays['response_residual_mean'].dtype==np.float64,'Response precision differs')
    evidence={'request_sha256':request_hash,'binding':bound,'source_hashes':source,'script_sha256':manifest['script_sha256'],'helper_sha256':manifest['helper_sha256'],
        'artifact_sha256':response['activation_artifact']['sha256'],'provenance_sha256':response['provenance_artifact']['sha256'],
        'finish_reason':finish,'prose_complete':complete,'truncated_prose':finish=='length','sampled_tokens':len(generated),'consumed_tokens':len(generated)-1,
        'program_sha256':canonical_digest(request.get('steering',{'phase':'all','sites':[]})),
        'array_sha256':{k:digest(arrays[k]) for k in EXACT},'batch':report.get('batch'),'checkpoint_hashes':{k:manifest[k] for k in ('checkpoint_config_sha256','checkpoint_index_sha256','tokenizer_sha256')},'graphs':report['graphs'],'experiment_source_sha256':manifest.get('experiment_source_sha256'),'physical_pages':report['physical_pages']}
    return {'arrays':arrays,'paths':paths,'evidence':evidence,'request':request,'response':response}



def validate_parity_response(response,request,a,complete):
    """Same capture invariants as production, with truthful prose completeness."""
    r=response['response_capture'];g=a['generated_token_ids'];consumed=g[:-1];n=len(consumed);start=len(a['prompt_token_ids'])
    require(r['id']==request['capture_response']['id'] and r['artifact']==response['activation_artifact']
        and r['request_sha256']==canonical_digest(request) and ids_binding(r)==ids_binding(response)
        and r['generation_complete'] is complete and r['mean_defined']==(n>0)
        and r['consumed_generated_tokens']==n and r['sampled_generated_tokens']==len(g)
        and r['all_consumed_generated_tokens_captured'] is True and r['all_sampled_generated_tokens_captured'] is False
        and r['final_sampled_unconsumed_token_id']==int(g[-1]),'Response capture completeness/identity differs')
    require(np.array_equal(a['response_consumed_token_ids'],consumed) and np.array_equal(a['response_consumed_positions'],np.arange(start,start+n))
        and int(a['response_residual_count'])==n and bool(a['response_residual_mean_defined'])==(n>0),'Response capture token coverage differs')
    for key,values in [('consumed_token_ids_sha256',consumed),('consumed_positions_sha256',np.arange(start,start+n)),('generated_token_ids_sha256',g)]:
        require(r[key]==ResponseCapture.ids_hash(values),'Response sequence hash differs')
    require(a['response_residual_sum'].shape==a['response_residual_mean'].shape==(64,5120),'Response means shape differs')
    expected=a['response_residual_sum']/n if n else np.zeros((64,5120))
    require(np.array_equal(expected,a['response_residual_mean']),'Response mean/count differs')
    layers=request['capture_response']['layers'];kept=min(n,request['capture_response']['max_positions']) if layers else 0
    require(np.array_equal(a['response_selected_layers'],layers) and np.array_equal(a['response_selected_positions'],np.arange(start,start+kept))
        and a['response_selected_residuals'].shape==(kept,len(layers),5120),'Selected response profiles differ')
    if kept==n and n:require(np.array_equal(a['response_selected_residuals'].astype(np.float64).sum(axis=0),a['response_residual_sum'][layers]),'Selected profiles disagree with sums')


def compare(left,right):
    require(left['request']==right['request'] and left['evidence']['request_sha256']==right['evidence']['request_sha256'],'Parity original request differs')
    require(left['evidence']['binding']['model_identity']==right['evidence']['binding']['model_identity'],'Parity checkpoint differs')
    for key in EXACT:
        a,b=left['arrays'][key],right['arrays'][key]
        require(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),'Exact parity differs: '+key)
    require(left['response']['choices']==right['response']['choices'],'Generated text/termination differs')
    return {'request_sha256':left['evidence']['request_sha256'],'tokens_exact':True,'final_prefill_exact':True,'response_capture_exact':True,
        'left_artifact_sha256':left['evidence']['artifact_sha256'],'right_artifact_sha256':right['evidence']['artifact_sha256'],
        'prose_complete':left['evidence']['prose_complete'] and right['evidence']['prose_complete']}


def recover_received(plan,journal,store,expected_binding=None):
    """Read-only recovery; every submitted-but-unreceived job remains uncertain."""
    jobs={j['id']:j for j in plan['jobs']};submitted={};received={}
    for line in Path(journal).read_text().splitlines():
        event=json.loads(line);identity=event.get('job_id')
        require(identity in jobs,'Journal job outside frozen plan')
        if event['state']=='submitting':
            require(identity not in submitted,'Duplicate submission: manual investigation required');check_job_request(jobs[identity],event['request'])
            require(canonical_digest(event['request'])==event['request_sha256'],'Journal request hash differs');submitted[identity]=event
        elif event['state']=='received':
            require(identity in submitted and identity not in received,'Unpaired/duplicate received event');received[identity]=event
    checked={}
    for identity,event in received.items():
        checked[identity]=checked_parity(event['response'],submitted[identity]['request'],expected_binding or plan['old_binding'],store)
    return {'records':checked,'never_submitted':[j for j in jobs if j not in submitted],
        'uncertain_no_retry':[j for j in submitted if j not in received],'reused_received':list(checked),'journal_sha256':file_hash(journal)}


def write_receipts(plan,old,new_b1,new_batch,*,qualification_path,compatibility_path,server_sha256,physical_pages,page_tokens):
    """Only all24 independently checked records in each mode can publish gates."""
    identities={j['id'] for j in plan['jobs']}
    require(all(set(records)==identities for records in (old,new_b1,new_batch)),'All24 records required in all three modes')
    require(not Path(qualification_path).exists() and not Path(compatibility_path).exists(),'Receipts cannot overwrite existing evidence')
    require(re.fullmatch('[0-9a-f]{64}',server_sha256) is not None and type(physical_pages) is int and physical_pages>0 and type(page_tokens) is int and page_tokens>0,'Invalid service configuration')
    pairs={};groups={};old_bound=ids_binding(plan['old_binding']);new_bound=None;old_core=None;new_core=None
    for identity in sorted(identities):
        a,b,c=old[identity],new_b1[identity],new_batch[identity]
        check_job_request(next(j for j in plan['jobs'] if j['id']==identity),a['request'])
        require(a['evidence']['binding']==old_bound,'Old binding differs')
        new_bound=new_bound or b['evidence']['binding'];require(b['evidence']['binding']==c['evidence']['binding']==new_bound,'New binding differs')
        old_core=old_core or a['evidence']['source_hashes'];new_core=new_core or b['evidence']['source_hashes']
        require(a['evidence']['source_hashes']==old_core and b['evidence']['source_hashes']==c['evidence']['source_hashes']==new_core,'Per-runtime core drift')
        require(set(old_core)==set(new_core) and all(old_core[k]==new_core[k] for k in old_core if k!='ignition/components/activation_capture.py'),'Native core changed outside capture override')
        require(a['evidence']['checkpoint_hashes']==b['evidence']['checkpoint_hashes']==c['evidence']['checkpoint_hashes'],'Checkpoint/tokenizer source drift')
        require(b['evidence']['batch'] is None,'New B1 evidence unexpectedly batched')
        require(b['evidence']['experiment_source_sha256']==c['evidence']['experiment_source_sha256']==server_sha256
            and b['evidence']['physical_pages']==c['evidence']['physical_pages']==physical_pages,'Service source/physical capacity differs')
        require(page_tokens==128,'Native checkpoint page size differs')
        batch=c['evidence']['batch'];require(batch and batch['requests']<=4 and batch['physical_slot']==batch['index'],'Batch slot evidence absent')
        groups.setdefault(batch['id'],[]).append(c)
        pairs[identity]={'old_to_new_b1':compare(a,b),'new_b1_to_batch':compare(b,c)}
    concurrency=[]
    for group,records in groups.items():
        metadata=[r['evidence']['batch'] for r in records];n=metadata[0]['requests']
        require(len(records)==n and {m['index'] for m in metadata}==set(range(n)) and all(m['requests']==n for m in metadata),'Batch group incomplete')
        sequences={}
        for r in records:
            for row,kind in zip(r['arrays']['completion_trace'],r['arrays']['completion_graph_kind']):
                if kind=='decode':sequences.setdefault(int(row[1]),set()).add(r['evidence']['batch']['physical_slot'])
        maximum=max(map(len,sequences.values()),default=0)
        if maximum==4:require(all(r['evidence']['graphs']['decode'].startswith('decode4') for r in records),'Shared4-token sequence lacks decode4 graph')
        concurrency.append({'batch_id':group,'requests':n,'maximum_shared_decode_slots':maximum})
    require(any(g['maximum_shared_decode_slots']==4 for g in concurrency),'No actual shared4-slot decode evidence')
    receipt={'schema':'character-creation.batch-service-qualification.v1','verified':True,'passed':True,'binding':new_bound,'server_sha256':server_sha256,
        'physical_pages':physical_pages,'page_tokens':page_tokens,'plan_sha256':plan['input_sha256'],'comparisons':pairs,'concurrency':concurrency,
        'prose_complete_pairs':sum(p['old_to_new_b1']['prose_complete'] for p in pairs.values()),'scope':'Exact native runtime parity, including deliberately length-limited traces. Not prose quality or personality efficacy.',
        'validator_sha256':file_hash(__file__)}
    programs=sorted({r['evidence']['program_sha256'] for identity,r in old.items() if any(p.get('kind')=='direction' for p in next(j for j in plan['jobs'] if j['id']==identity)['recipe']['pushes'])})
    require(programs,'No exact direction programs qualified')
    with Path(qualification_path).open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    compatibility={'schema':'character-creation.batch-runtime-compatibility.v1','state':'complete','passed':True,'old_binding':old_bound,'new_binding':new_bound,
        'artifact_sha256':plan['direction_artifact']['sha256'],'allowed_program_sha256':programs,'old_core_source_hashes':old_core,'new_core_source_hashes':new_core,
        'qualification_sha256':file_hash(qualification_path),'parity':{'programs_exact':True,'response_tokens_exact':True,'response_capture_exact':True},'scope':receipt['scope']}
    with Path(compatibility_path).open('x') as f:json.dump(compatibility,f,indent=2,allow_nan=False)
    return receipt,compatibility



def prepare_manager_pilot(plan,bank,target,*,qualification_spec,compatibility_spec):
    """Return24 new Manager jobs/corpus rows; no queue writes or inference.

    Enqueue all24 atomically before starting Manager with max_total_jobs=24.
    That existing job-budget branch prevents its planner from being called.
    """
    from persona_batch_dispatch import authorize_direction
    require(len(plan['jobs'])==24 and canonical_digest({k:v for k,v in plan.items() if k!='input_sha256'})==plan['input_sha256'],'Frozen24-case plan required')
    target=ids_binding(target)
    require(target['model_identity']==plan['old_binding']['model_identity']==bank.model.identity,'Pilot checkpoint differs')
    qpath=Path(qualification_spec['path']);cpath=Path(compatibility_spec['path'])
    require(file_hash(qpath)==qualification_spec['sha256'] and file_hash(cpath)==compatibility_spec['sha256'],'Pilot receipt bytes changed')
    qualification=json.loads(qpath.read_text());compatibility=json.loads(cpath.read_text())
    require(qualification.get('schema')=='character-creation.batch-service-qualification.v1' and qualification.get('verified') is True
        and qualification.get('passed') is True and qualification.get('binding')==target
        and qualification.get('plan_sha256')==plan['input_sha256'] and set(qualification.get('comparisons',{}))=={j['id'] for j in plan['jobs']},'Pilot qualification incomplete or wrong binding')
    require(compatibility.get('qualification_sha256')==qualification_spec['sha256'] and compatibility.get('new_binding')==target
        and compatibility.get('artifact_sha256')==plan['direction_artifact']['sha256'],'Pilot compatibility lineage differs')
    expected_arms={'baseline','A','A_reverse','B','B_reverse','random'};groups={}
    for row in plan['jobs']:groups.setdefault(row['case_id'],[]).append(row)
    require(len(groups)==4 and all(len(rows)==6 and {r['arm'] for r in rows}==expected_arms for rows in groups.values()),'Pilot must contain4scenes by6arms')
    calibration=file_hash(bank.root/'layer_32.npz')
    proposal={'schema':'character-creation.batch-manager-pilot.v1','model_identity':bank.model.identity,'binding':target,
        'source_plan_sha256':plan['input_sha256'],'qualification_sha256':qualification_spec['sha256'],'compatibility_sha256':compatibility_spec['sha256'],
        'max_tokens':2048,'jobs':24,'scope':'Authored personality probes with native neural interventions; correctness and style assessed separately; no automatic promotion.'}
    proposal_id=canonical_digest(proposal);corpus=[];jobs=[]
    for scene_index,(case_id,rows) in enumerate(groups.items()):
        first=rows[0]
        require(all(r['prompt']==first['prompt'] and r['source_sha256']==first['source_sha256'] and r['seed']==first['seed'] for r in rows),'Within-scene prompt/seed mismatch')
        require(hashlib.sha256(first['prompt'].encode()).hexdigest()==first['source_sha256'],'Authored source changed')
        prompt={'schema':'character-creation.manager-corpus.v1','corpus':'personality_diagnostics','split':'discovery','category':'native_batch_pilot',
            'system':'','prompt':first['prompt'],'max_tokens':2048,'thinking':False,'pair_id':case_id,'source_sha256':first['source_sha256'],
            'suite_sha256':plan['suite_sha256'],'source_plan_sha256':plan['input_sha256'],'labels':{'scenario_domain':'personality_diagnostics'}}
        prompt['id']=canonical_digest(prompt);corpus.append(prompt)
        for row in rows:
            pushes=deepcopy(row['recipe']['pushes']);phase=row['recipe']['phase']
            require(phase=='all' and all(push.get('layer')==32 for push in pushes),'Unqualified pilot layer/phase')
            program,provenance=bank.program(pushes,phase)
            authorized=authorize_direction(provenance,pushes,target,compatibility_spec)
            if any(push.get('kind')=='direction' for push in pushes):require(authorized==compatibility_spec['sha256'],'Direction compatibility missing')
            experiment={'label':'native_batch_personality_pilot','kind':'direction','layer':32,'dose':.1,'phase':phase,'corpora':['personality_diagnostics'],'prompts_per_corpus':4,'repetitions':1,'measurement':'rollout'}
            job={'schema':'character-creation.manager-job.v1','proposal_id':proposal_id,'model_identity':bank.model.identity,
                'experiment_index':scene_index,'experiment':experiment,'arm':row['arm'],'pushes':pushes,'phase':phase,'seed':row['seed'],'repetition':0,
                'prompt':deepcopy(prompt),'pair_id':case_id,'source_group':{'corpus':'personality_diagnostics','pair_id':case_id,'split':'discovery'},
                'calibration_sha256':calibration,'random_control_semantics':'Seeded residual random vector, matched configured relative norm',
                'measurement':'rollout','kind':'rollout','qualification_job_id':row['id'],'source_plan_sha256':plan['input_sha256'],
                'target_binding':target,'program_sha256':canonical_digest(program),'compatibility_sha256':authorized,'promotion_eligible':False}
            jobs.append(job)
    corpus_sha256=hashlib.sha256(''.join(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n' for row in corpus).encode()).hexdigest()
    for job in jobs:job['corpus_sha256']=corpus_sha256;job['id']=canonical_digest(job)
    require(len({j['id'] for j in jobs})==24 and not {j['id'] for j in jobs}&{j['id'] for j in plan['jobs']},'Pilot IDs not distinct')
    return {'proposal':proposal,'jobs':jobs,'corpus':corpus,'corpus_sha256':corpus_sha256,
        'config_patch':{'max_total_jobs':24,'per_gpu_concurrency':1,'native_batch':{'enabled':True,'compatibility_receipt':deepcopy(compatibility_spec)}},
        'startup_requirement':'Use a separate empty target-bound queue; register these4corpus rows and atomically enqueue all24jobs before Manager.run. Merge native_batch patch with artifact transport settings. No planner request occurs once total_jobs=max_total_jobs.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--journal',type=Path,required=True);p.add_argument('--artifact-directory',type=Path,required=True);p.add_argument('--remote-host',required=True);p.add_argument('--remote-root',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'Recovery receipt exists; no overwrite');plan=load_plan(a.plan)
    recovery=recover_received(plan,a.journal,ArtifactStore(a.artifact_directory,a.remote_host,a.remote_root))
    summary={k:v for k,v in recovery.items() if k!='records'};summary.update(schema='character-creation.batch-reference-recovery.v1',plan_sha256=plan['input_sha256'],validator_sha256=file_hash(__file__),records={k:v['evidence'] for k,v in recovery['records'].items()},semantics='Revalidated existing received evidence, not independent replication; no POST performed.')
    with a.output.open('x') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print(json.dumps({'recovered':len(recovery['records']),'never_submitted':len(recovery['never_submitted']),'uncertain':len(recovery['uncertain_no_retry'])}))
if __name__=='__main__':main()
