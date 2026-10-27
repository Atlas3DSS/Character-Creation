"""Independent offline validation of the finite native B4 proof, never inference.

Checks persisted physical completion ABI records and request-state/RNG traces.
A verified receipt can still have passed=false when a scientific gate fails.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from persona_vectors import canonical_digest

POLICY={'maximum_first_decode_kl':.01,'maximum_first_hidden_relative_l2':.02,
        'minimum_first_decode_top1_agreement':1.,'exact_same_batch_repeat':True,
        'exact_unaffected_rows':True,'exact_packed_and_slot_permutations':True,
        'generation_budget':64,'seeds':[801,802,803,804],
        'scope':'Four resident requests; serial per-request prefill then one true B4 ordinary decode graph per step. No intermediate B4 capture or production rollout.'}
BINDINGS=('model_identity','runtime_identity','prepared_identity')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def configurations():
    rows={f'serial_{i}':([i],1,[0],[0],False) for i in range(4)}
    rows.update(pair_0=([0,1],2,[0,1],[0,1],False),pair_1=([2,3],2,[0,1],[0,1],False))
    for name in ('four','four_repeat','zero_controls','one_changed','unequal_lengths','unequal_lengths_repeat'):
        rows[name]=([0,1,2,3],4,[0,1,2,3],[0,1,2,3],False)
    rows['slot_permutation']=([0,1,2,3],4,[2,0,3,1],[0,1,2,3],False)
    rows['row_permutation']=([0,1,2,3],4,[0,1,2,3],[3,1,0,2],False)
    rows['sampled_four']=([0,1,2,3],4,[0,1,2,3],[0,1,2,3],True)
    rows['sampled_permutation']=([0,1,2,3],4,[3,1,0,2],[2,0,3,1],True)
    return rows


def run_trace(name,run,cases,arrays,eos,hidden,graphs=None):
    indices,width,slots,order,sampled=configurations()[name];n=len(indices)
    require(run['cases']==[cases[i]['id'] for i in indices] and run['width']==width and run['slots']==slots and run['packed_order']==order and run['sampled'] is sampled,'Run configuration differs: '+name)
    budgets=[cases[i]['budget'] for i in indices]
    if name.startswith('unequal_lengths'):budgets=[1,8,24,64]
    lengths=run['prompt_tokens'];generated=run['generated']
    require(len(lengths)==n and len(generated)==n and all(type(x) is int and x>0 for x in lengths),'Prompt lengths differ')
    logits=arrays[name+'_logits'];h=arrays[name+'_hidden'];vocab=logits.shape[1]
    require(logits.ndim==2 and logits.shape[0]==n and h.shape==(n,hidden) and logits.dtype==h.dtype==np.float32,'Raw first-decode array dimensions/dtypes differ')
    require(np.isfinite(logits).all() and np.isfinite(h).all(),'Nonfinite raw observations')
    for tokens,budget in zip(generated,budgets):
        require(1<=len(tokens)<=budget and all(type(t) is int and 0<=t<vocab for t in tokens),'Invalid generated IDs/budget')
        require(len(tokens)==budget or tokens[-1] in eos,'Unexplained early generation termination')
        require(not any(t in eos for t in tokens[:-1]),'Generation continued after EOS')
    previous=[[0,0,0,-1,0,0,0] for _ in range(4)];last_sequence=-1;admitted=set();decode=[];seen=[False]*n
    counts=Counter();slot_to_local={slot:i for i,slot in enumerate(slots)}
    for trace in run['traces']:
        match=re.fullmatch(r'(init|prefill|decode)([0-9]+)(?:-no-mtp)?',trace['graph']);require(match is not None,'Unexpected graph name')
        kind,capacity=match[1],int(match[2]);counts[kind]+=1
        require(type(trace['sequence']) is int and trace['sequence']>last_sequence,'Nonmonotonic graph sequence');last_sequence=trace['sequence']
        active=trace['active_slots'];state=trace['state']
        require(len(state)==4 and all(len(row)==7 and all(type(x) is int for x in row) for row in state),'Malformed physical request state')
        require(active==sorted(set(active)) and set(active)<=set(slots) and active,'Invalid active physical slots')
        if kind=='decode':
            require(capacity==width and set(active)=={s for s in slots if previous[s][4]==1},'Decode must consume every live physical request')
            require(trace.get('actual_decode_tokens')==len(active),'Actual packed decode token count differs')
            if graphs is not None:require(trace['graph']==graphs[str(width)]['name'],'Decode graph name differs from signature')
            decode.append(active)
        else:require(capacity==1 if kind=='init' else capacity>=1,'Wrong maintenance capacity')
        require(kind=='decode' or len(active)==1,'Prefill/admission must be B1')
        for slot in range(4):
            before,after=previous[slot],state[slot]
            if slot not in active:require(before==after,'Inactive/unrelated physical request changed');continue
            i=slot_to_local[slot];epoch,consumed,count,pending,status,error,rng=after
            require(error==0 and rng==count,'Request error or RNG progression differs')
            if kind=='init':
                require(slot not in admitted and epoch>0 and after[1:]==[0,0,-1,1,0,0],'Bad physical-slot admission')
                admitted.add(slot)
            else:
                require(slot in admitted and epoch==before[0] and before[4]==1,'Invalid request epoch/status')
                if kind=='prefill':
                    require(before[2]==0 and before[1]<consumed<=lengths[i] and consumed-before[1]<=capacity,'Invalid prefill progress')
                    expected_count=int(consumed==lengths[i])
                else:
                    require(consumed==before[1]+1 and before[1]>=lengths[i],'Invalid decode consumption')
                    expected_count=before[2]+1;seen[i]=True
                require(count==expected_count and count<=len(generated[i]),'Generated count differs')
                require(pending==(generated[i][count-1] if count else -1),'Pending token differs from saved sequence')
                terminal=count>0 and (pending in eos or count==budgets[i])
                require(status==(2 if terminal else 1),'Terminal status differs from EOS/budget')
        records=trace.get('completion_records')
        require(isinstance(records,list) and len(records)==4 and len({len(r) for r in records})==1 and len(records[0]) in (8,24),'Raw physical completion records missing')
        resident=set(slots) if kind=='decode' else set(active)
        for slot,record in enumerate(records):
            require(all(type(x) is int for x in record),'Malformed raw completion values')
            epoch,consumed,count,pending,status,error,rng=state[slot]
            live=slot in resident;sampled=slot in active and kind!='init' and count>0
            token=pending if sampled else -1
            expected=[epoch,trace['sequence'],consumed,count,token,status,error,0] if live else [0,0,0,0,-1,0,0,0]
            if len(record)==24:
                tail=[0]*16;tail[0]=int(live and token>=0);tail[4]=-1;tail[5]=max(consumed-1,0) if live else 0;tail[7]=1
                tail[8:]=[-1]*8;tail[8]=token if live else -1;expected+=tail
            require(record==expected,'Raw completion physical-slot/token/state alignment differs')
        previous=state
    require(admitted==set(slots) and counts['init']==n and run['final_states']==previous,'Admission/final state coverage differs')
    for slot,i in slot_to_local.items():require(previous[slot][2]==len(generated[i]) and previous[slot][4]==2,'Request did not finish')
    require(run['state_restored'] is True and run['first_decode_seen']==seen,'Missing restoration or first-decode coverage')
    require(run['decode_rounds']==len(decode) and run['initial_active_decode_tokens']==(len(decode[0]) if decode else None),'Decode concurrency summary differs')
    steps=run['logit_step_hashes'];require(len(steps)==len(decode),'All-step hashes missing')
    first=set()
    for step,active in zip(steps,decode):
        step={int(k):v for k,v in step.items()};require(set(step)=={slot_to_local[s] for s in active},'Step hashes physical/logical mapping differs')
        for i,digest in step.items():
            require(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest),'Malformed per-step hash')
            if i not in first:
                require(hashlib.sha256(logits[i].tobytes()).hexdigest()==digest,'First raw logits/hash mismatch');first.add(i)
    for i,live in enumerate(seen):
        if not live:require(not logits[i].any() and not h[i].any(),'Unobserved first decode fabricated')
    require(all(type(run[k]) in (float,int) and math.isfinite(run[k]) and run[k]>=0 for k in ('wall_seconds','gpu_decode_ms')) and run['wall_seconds']>0,'Invalid timings')
    return {'validated_graph_replays':len(run['traces']),'decode_replays':len(decode),'maximum_simultaneous_tokens':max(map(len,decode),default=0)}


def histories(run,index):return [step.get(str(index),step.get(index)) for step in run['logit_step_hashes'] if str(index) in step or index in step]


def same(a,b,indices,arrays):
    return all(a['generated'][i]==b['generated'][i] and histories(a,i)==histories(b,i)
               and np.array_equal(arrays[a['_name']+'_logits'][i],arrays[b['_name']+'_logits'][i])
               and np.array_equal(arrays[a['_name']+'_hidden'][i],arrays[b['_name']+'_hidden'][i]) for i in indices)


def compare_summary(actual,expected):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),'Summary fields differ')
        for key in expected:compare_summary(actual[key],expected[key])
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),'Summary length differs')
        for a,b in zip(actual,expected):compare_summary(a,b)
    elif isinstance(expected,float):require(type(actual) in (int,float) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-12),'Numerical summary differs')
    else:require(type(actual) is type(expected) and actual==expected,'Summary value differs')


def recompute(report,plan,arrays,*,eos,hidden=5120):
    require(report['schema']=='character-creation.batch-proof.v1' and plan['schema']=='character-creation.batch-proof-plan.v1','Wrong schemas')
    require(report['state']=='complete' and report['policy']==plan['policy']==POLICY and report['automatic_rollout'] is False,'Incomplete or changed protocol')
    require(report['production_binding']==plan['production_binding'] and report['production_core_source_hashes']==plan['production_core_source_hashes'] and report['production_core_equal'] is True,'Production binding differs')
    require(plan['model_identity']==report['binding']['model_identity']==plan['production_binding']['model_identity'],'Checkpoint binding differs')
    require(report['binding']['prepared_identity']==plan['production_binding']['prepared_identity'],'Prepared identity differs')
    require(all(re.fullmatch('[0-9a-f]{64}',report['binding'][k]) for k in BINDINGS),'Invalid bound identities')
    cases=plan['cases'];require(len(cases)==4 and len({c['id'] for c in cases})==4 and len({c['program_sha256'] for c in cases})==4,'Four unique cases/programs required')
    for i,c in enumerate(cases):
        require(canonical_digest(c['program'])==c['program_sha256'] and c['program']['phase']=='all' and c['budget']==64 and c['seed']==POLICY['seeds'][i],'Frozen program/policy differs')
        require(any(np.any(np.asarray(s['vector'])!=0) for s in c['program']['sites']),'All four injections must be nonzero')
    require(len({c['prompt'] for c in cases})==1,'Four injections need the same prompt')
    require(isinstance(report.get('eos_token_ids'),list) and report['eos_token_ids'] and set(report['eos_token_ids'])==set(eos),'Authoritative EOS IDs differ')
    graphs=report.get('decode_graphs',{});require(set(graphs)=={'1','2','4'},'Decode signatures missing')
    for width in (1,2,4):
        signature=graphs[str(width)]
        require(signature['batch_capacity']==width and signature['token_capacity']==1 and signature['speculation_enabled'] is False,'Invalid ordinary decode signature')
    runs=report['runs'];require(set(runs)==set(configurations()),'Required run coverage differs')
    require(set(arrays)=={n+s for n in runs for s in ('_logits','_hidden')},'NPZ run coverage differs')
    trace_checks={n:run_trace(n,r,cases,arrays,set(eos),hidden,graphs) for n,r in runs.items()}
    for n,r in runs.items():r['_name']=n
    base=runs['four'];parity=[]
    for i in range(4):
        p=arrays[f'serial_{i}_logits'][0].astype(float);q=arrays['four_logits'][i].astype(float)
        lp=p-p.max();lq=q-q.max();lp-=np.log(np.exp(lp).sum());lq-=np.log(np.exp(lq).sum())
        h=arrays[f'serial_{i}_hidden'][0].astype(float);other=arrays['four_hidden'][i].astype(float)
        parity.append({'kl':float(np.sum(np.exp(lp)*(lp-lq))),'same_top1':bool(p.argmax()==q.argmax()),'max_abs':float(np.max(np.abs(p-q))),
                       'hidden_relative_l2':float(np.linalg.norm(h-other)/max(np.linalg.norm(h),1e-30)),
                       'entire_token_sequence_equal':runs[f'serial_{i}']['generated'][0]==base['generated'][i]})
    checks={'four_active_tokens_in_one_decode_graph':base['initial_active_decode_tokens']==4,
        'four_distinct_native_effects':all(not np.array_equal(arrays['four_logits'][i],arrays['zero_controls_logits'][i]) for i in range(4)),
        'same_batch_repeat_exact':same(base,runs['four_repeat'],range(4),arrays),
        'physical_slot_permutation_exact':same(base,runs['slot_permutation'],range(4),arrays),
        'packed_row_permutation_exact':same(base,runs['row_permutation'],range(4),arrays),
        'other_three_unchanged_exact':same(base,runs['one_changed'],(0,2,3),arrays),
        'early_finished_holes_repeat_exact':same(runs['unequal_lengths'],runs['unequal_lengths_repeat'],range(4),arrays),
        'sampled_seed_and_slot_isolation_exact':same(runs['sampled_four'],runs['sampled_permutation'],range(4),arrays),
        'serial_numerical_gate':all(m['kl']<=.01 and m['same_top1'] and m['hidden_relative_l2']<=.02 for m in parity)}
    serial=sum(runs[f'serial_{i}']['wall_seconds'] for i in range(4))
    throughput={'serial_four_wall_seconds':serial,'batched_four_wall_seconds':base['wall_seconds'],'wall_speedup':serial/base['wall_seconds'],
        'serial_generated_tokens':sum(len(runs[f'serial_{i}']['generated'][0]) for i in range(4)),
        'batched_generated_tokens':sum(map(len,base['generated'])),'serial_gpu_decode_ms':sum(runs[f'serial_{i}']['gpu_decode_ms'] for i in range(4)),
        'batched_gpu_decode_ms':base['gpu_decode_ms'],'scope':'Warm native trials including host validation/state restore before execution; excludes model preparation. Generated lengths reported; no fourfold assumption.'}
    compare_summary(report['checks'],checks);compare_summary(report['serial_comparison'],parity);compare_summary(report['throughput'],throughput)
    require(report['passed'] is all(checks.values()),'Passed flag differs')
    return {'checks':checks,'passed':all(checks.values()),'trace_validation':trace_checks,'serial_comparison':parity,'throughput':throughput}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--driver',type=Path,required=True);p.add_argument('--worker-report',type=Path);p.add_argument('--eos-token',type=int,action='append');p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    if a.output.exists():raise FileExistsError('Validation receipt exists; do not overwrite prior evidence')
    receipt={'schema':'character-creation.batch-proof-validation.v1','verified':False,'passed':False,'validator_sha256':sha(__file__)}
    try:
        require(sha(a.plan)==a.plan_sha256,'Frozen plan SHA differs');plan=json.loads(a.plan.read_text());report=json.loads(a.report.read_text())
        require(report['policy_sha256']==a.plan_sha256 and report['driver_sha256']==sha(a.driver),'Driver/plan byte binding differs')
        pointer=report['artifact'];require(pointer['filename']=='persona_batch_proof.npz','Unexpected artifact filename')
        artifact=a.report.parent/pointer['filename'];require(not artifact.is_symlink() and sha(artifact)==pointer['sha256'] and artifact.stat().st_size==pointer['bytes'],'Artifact bytes differ')
        if a.worker_report:
            worker=json.loads(a.worker_report.read_text())
            require(all(worker[k]==report['binding'][k] for k in BINDINGS) and worker['gpu_uuid']==report['gpu_uuid'] and worker['lease_id']==report['lease_id'],'Worker identity/lease differs')
            require(worker['experiment_source_sha256']==sha(a.driver) and worker['calibration_equivalence']['passed'] is True,'Worker experiment/calibration gate differs')
            if 'experiment' in worker:
                require(worker['experiment']['sha256']==sha(a.report) and worker['experiment']['passed'] is report['passed'],'Worker experiment receipt differs')
            require({**worker['source_sha256'],'heliopause':worker['heliopause_sha256']}==report['production_core_source_hashes'],'Worker core differs')
            receipt['worker_report_sha256']=sha(a.worker_report)
        with np.load(artifact,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        eos=a.eos_token if a.eos_token is not None else report['eos_token_ids']
        result=recompute(report,plan,arrays,eos=eos)
        receipt.update(result,verified=True,plan_sha256=a.plan_sha256,report_sha256=sha(a.report),artifact_sha256=sha(artifact),driver_sha256=sha(a.driver),binding=report['binding'],eos_token_ids=eos,
            evidence_limits=['Raw physical completion records and request-state/RNG traces are checked; this remains source-bound recorded evidence, not a second execution.', 'Only first-decode logits/hidden arrays are retained; later equality uses recorded per-step hashes, not reconstructed logits.', 'Restoration is source-bound runtime evidence; this offline reader cannot inspect released GPU state.'])
    except Exception as error:receipt['error']=repr(error)
    with a.output.open('x') as stream:json.dump(receipt,stream,indent=2,allow_nan=False)
    print(json.dumps({'verified':receipt['verified'],'passed':receipt['passed'],'receipt':str(a.output)}))
    return 0 if receipt['verified'] and receipt['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
