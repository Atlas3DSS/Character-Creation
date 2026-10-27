import persona
from copy import deepcopy
import hashlib
import numpy as np
import pytest
from persona_batch_validate import POLICY,configurations,recompute,run_trace,canonical_digest


def fixture():
    binding={k:'a'*64 for k in ('model_identity','runtime_identity','prepared_identity')}
    cases=[]
    for i in range(4):
        program={'phase':'all','sites':[{'site':'layers.32.residual','vector':[float(i+1)]}]}
        cases.append({'id':str(i),'prompt':'Same authored arithmetic problem','seed':801+i,'budget':64,'program':program,'program_sha256':canonical_digest(program)})
    plan={'schema':'character-creation.batch-proof-plan.v1','policy':deepcopy(POLICY),'model_identity':binding['model_identity'],'production_binding':binding,'production_core_source_hashes':{'core':'b'*64},'cases':cases}
    report={'schema':'character-creation.batch-proof.v1','state':'complete','policy':deepcopy(POLICY),'automatic_rollout':False,'production_binding':binding,'binding':binding,'production_core_source_hashes':plan['production_core_source_hashes'],'production_core_equal':True,'eos_token_ids':[2],'decode_graphs':{str(w):{'name':'decode'+str(w),'batch_capacity':w,'token_capacity':1,'speculation_enabled':False} for w in (1,2,4)},'runs':{}}
    arrays={}
    for name,(indices,width,slots,order,sampled) in configurations().items():
        n=len(indices);unequal=name.startswith('unequal_lengths');lengths=[3+i if unequal else 3 for i in range(n)];gens=[[2] if unequal and i==0 else [1,1,2] for i in range(n)]
        logits=np.array([[0.,5.+i*.01,2.] for i in indices],np.float32);hidden=np.array([[i+1.,2.,3.,4.] for i in indices],np.float32)
        if unequal:logits[0]=0;hidden[0]=0
        if name=='zero_controls':logits[:,0]+=.1
        if name=='one_changed':logits[1,0]+=.1
        arrays[name+'_logits']=logits;arrays[name+'_hidden']=hidden
        state=[[0,0,0,-1,0,0,0] for _ in range(4)];traces=[];seq=0
        def append(graph,active):
            nonlocal seq
            seq+=1;records=[]
            for slot in range(4):
                live=slot in slots if graph.startswith('decode') else slot in active
                epoch,consumed,count,pending,status,error,rng=state[slot]
                token=pending if slot in active and not graph.startswith('init') and count else -1
                records.append([epoch,seq,consumed,count,token,status,error,0] if live else [0,0,0,0,-1,0,0,0])
            traces.append({'graph':graph,'sequence':seq,'active_slots':sorted(active),'state':deepcopy(state),'completion_records':records,
                           **({'actual_decode_tokens':len(active)} if graph.startswith('decode') else {})})
        for i,slot in enumerate(slots):
            state[slot]=[i+1,0,0,-1,1,0,0];append('init1-no-mtp',[slot])
            state[slot]=[i+1,lengths[i],1,gens[i][0],2 if len(gens[i])==1 else 1,0,1];append('prefill128-no-mtp',[slot])
        steps=[];first=None
        while any(state[s][4]==1 for s in slots):
            active=[s for s in slots if state[s][4]==1]
            if first is None:first=len(active)
            hashes={}
            for i,slot in enumerate(slots):
                if slot not in active:continue
                state[slot][1]+=1;state[slot][2]+=1;count=state[slot][2];state[slot][3]=gens[i][count-1];state[slot][4]=2 if count==len(gens[i]) else 1;state[slot][6]=count
                hashes[str(i)]=hashlib.sha256(logits[i].tobytes()).hexdigest() if count==2 else 'b'*64
            append('decode'+str(width),active);steps.append(hashes)
        report['runs'][name]={'cases':[cases[i]['id'] for i in indices],'width':width,'slots':slots,'packed_order':order,'sampled':sampled,'prompt_tokens':lengths,'generated':gens,'traces':traces,'final_states':deepcopy(state),'state_restored':True,'first_decode_seen':[len(g)>1 for g in gens],'decode_rounds':len(steps),'initial_active_decode_tokens':first,'logit_step_hashes':steps,'wall_seconds':1.,'gpu_decode_ms':2.}
    report['checks']={k:True for k in ('four_active_tokens_in_one_decode_graph','four_distinct_native_effects','same_batch_repeat_exact','physical_slot_permutation_exact','packed_row_permutation_exact','other_three_unchanged_exact','early_finished_holes_repeat_exact','sampled_seed_and_slot_isolation_exact','serial_numerical_gate')}
    report['serial_comparison']=[{'kl':0.,'same_top1':True,'max_abs':0.,'hidden_relative_l2':0.,'entire_token_sequence_equal':True} for _ in range(4)]
    report['throughput']={'serial_four_wall_seconds':4.,'batched_four_wall_seconds':1.,'wall_speedup':4.,'serial_generated_tokens':12,'batched_generated_tokens':12,'serial_gpu_decode_ms':8.,'batched_gpu_decode_ms':2.,'scope':'Warm native trials including host validation/state restore before execution; excludes model preparation. Generated lengths reported; no fourfold assumption.'};report['passed']=True
    return report,plan,arrays


def test_complete_b4_proof_recomputes_from_arrays_and_physical_states():
    report,plan,arrays=fixture();r=recompute(report,plan,arrays,eos=[2],hidden=4)
    assert r['passed'] and r['throughput']['wall_speedup']==4.
    assert r['trace_validation']['four']['maximum_simultaneous_tokens']==4
    assert r['trace_validation']['unequal_lengths']['maximum_simultaneous_tokens']==3


@pytest.mark.parametrize('change',[
    lambda r:r['runs']['four']['traces'][-1]['state'][2].__setitem__(6,0),
    lambda r:r['runs']['four']['traces'][-1].__setitem__('active_slots',[0,1,2]),
    lambda r:r['runs']['four']['traces'][0]['state'][3].__setitem__(1,99),
    lambda r:r['runs']['four']['logit_step_hashes'][0].__setitem__('0','c'*64),
    lambda r:r['runs']['four_repeat']['logit_step_hashes'][1].__setitem__('0','c'*64),
    lambda r:r['throughput'].__setitem__('wall_speedup',40.),
    lambda r:r['serial_comparison'][0].__setitem__('kl',.001),
])
def test_tampered_rng_concurrency_inactive_state_later_hash_or_summary_rejected(change):
    report,plan,arrays=fixture();change(report)
    with pytest.raises(ValueError):recompute(report,plan,arrays,eos=[2],hidden=4)


def test_raw_array_change_cannot_hide_behind_pass_summary():
    report,plan,arrays=fixture();arrays['serial_0_hidden'][0,0]+=1
    with pytest.raises(ValueError,match='Summary|summary'):recompute(report,plan,arrays,eos=[2],hidden=4)


@pytest.mark.parametrize('change',[
    lambda r:r['runs']['slot_permutation']['traces'][0]['completion_records'].__setitem__(0,r['runs']['slot_permutation']['traces'][0]['completion_records'][2]),
    lambda r:r['runs']['four']['traces'][-1].__setitem__('actual_decode_tokens',3),
    lambda r:r['decode_graphs']['4'].__setitem__('batch_capacity',1),
    lambda r:r['eos_token_ids'].append(1),
])
def test_completion_physical_slot_and_actual_packing_are_required(change):
    report,plan,arrays=fixture();change(report)
    with pytest.raises(ValueError):recompute(report,plan,arrays,eos=[2],hidden=4)


def test_abi3_full_completion_records_validate():
    report,plan,arrays=fixture()
    for run in report['runs'].values():
        for trace in run['traces']:
            for record in trace['completion_records']:
                live=record[0]>0;token=record[4];tail=[0]*16;tail[0]=int(live and token>=0);tail[4]=-1;tail[5]=max(record[2]-1,0) if live else 0;tail[7]=1;tail[8:]=[-1]*8;tail[8]=token if live else -1
                record.extend(tail)
    assert recompute(report,plan,arrays,eos=[2],hidden=4)['passed']
