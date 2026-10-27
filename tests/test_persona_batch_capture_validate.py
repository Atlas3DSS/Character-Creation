import persona
from copy import deepcopy
import numpy as np
import pytest
from test_persona_batch_validate import fixture as native_fixture
from persona_batch_capture_validate import validate_capture,recompute,digest,PLANES


def fixture():
    r,p,a=native_fixture();p.update(schema='character-creation.batch-capture-plan.v1',policy={'budget':32,'unequal_budgets':[1,3,8,32],'exact_capture_equality':True})
    for c in p['cases']:c['budget']=32
    p['production_core_source_hashes']={'ignition/components/activation_capture.py':'b'*64,'heliopause':'c'*64}
    for k in ('adapter_sha256','probe_source_sha256','batch_trial_source_sha256'):p[k]='d'*64
    p['legacy_capture_prefix_sha256']='b'*64;p['capture_source_sha256']='e'*64
    r.update(schema='character-creation.batch-capture-proof.v1',policy=p['policy'],source_sha256={'ignition/components/activation_capture.py':'e'*64},heliopause_sha256='c'*64,capture_override={'sha256':'e'*64})
    for k in ('adapter_sha256','probe_source_sha256','batch_trial_source_sha256','legacy_capture_prefix_sha256'):r[k]=p[k]
    for name,run in r['runs'].items():
        n=len(run['cases']);count=np.array([len(g)-1 for g in run['generated']],np.int64)
        first={q:np.stack([np.full((2,4 if q!='mlp_input_last' else 6),int(case)+1,np.float32) if count[i] else np.zeros((2,4 if q!='mlp_input_last' else 6),np.float32) for i,case in enumerate(run['cases'])]) for q in PLANES}
        sums=first['residual_last'].astype(np.float64)*count[:,None,None]
        for k,v in {'_count':count,'_sum':sums,'_mean':first['residual_last'].astype(np.float64),**{'_first_'+q:v for q,v in first.items()}}.items():a[name+k]=v
        c={'counts':count.tolist(),'mean_defined':(count>0).tolist(),'consumed_token_ids':[g[:-1] for g in run['generated']], 'positions':[list(range(t,t+int(k))) for t,k in zip(run['prompt_tokens'],count)],'generations':[i+1 if k else None for i,k in enumerate(count)],'step_plane_sha256':[[{q:digest(v[i]) for q,v in first.items()} for _ in range(int(k))] for i,k in enumerate(count)],'mapping_steps':[]}
        previous=[[0,0,0,-1,0,0,0] for _ in range(4)]
        for t in run['traces']:
            if t['graph'].startswith('decode'):
                active=t['active_slots'];w=run['width'];m={'before':deepcopy(previous),'width':w}
                if w==1:
                    s=active[0];m.update(slot=s,row=0,generation=previous[s][0],position=previous[s][1],token_id=previous[s][3])
                else:
                    e={'slot_live':[s in active for s in range(4)],'slot_positions':[-1]*4,'slot_token_ids':[-1]*4,'slot_generations':[0]*4,'slot_match_counts':[0]*4,'slot_last_index':[0]*4,'mapping_error':[False],'token_rows':[-1]*w,'token_slots':[-1]*w,'token_positions':[-1]*w,'token_ids':[-1]*w,'token_generations':[0]*w,'token_live':[False]*w,'invalid_tokens':[False]*w}
                    for j,s in enumerate(active):
                        row=run['packed_order'].index(run['slots'].index(s));e['slot_positions'][s]=previous[s][1];e['slot_token_ids'][s]=previous[s][3];e['slot_generations'][s]=previous[s][0];e['slot_match_counts'][s]=1;e['slot_last_index'][s]=j
                        for key,value in [('rows',row),('slots',s),('positions',previous[s][1]),('ids',previous[s][3]),('generations',previous[s][0]),('live',True)]:e['token_'+key][j]=value
                    m.update(evidence=e,inactive_plane_sha256={str(s):{q:digest(np.zeros(v.shape[1:],np.float32)) for q,v in first.items()} for s in set(range(4))-set(active)})
                c['mapping_steps'].append(m)
            previous=t['state']
        run.update(capture=c,capture_restored=True,graph_wrappers_restored=True)
    r['checks']={k:True for k in ('serial_B4_exact','serial_B2_exact','four_distinct_native_effects','changed_row_effect','repeat_exact','slot_permutation_exact','row_permutation_exact','unaffected_rows_exact','unequal_lengths_exact','sampled_permutation_exact')}
    return r,p,a


def test_all_native_runs_and_capture_evidence():
    r,p,a=fixture();result=recompute(r,p,a,layers=2,hidden=4,intermediate=6)
    assert result['verified'] and result['passed']
    assert result['trace_validation']['four']['maximum_simultaneous_tokens']==4


@pytest.mark.parametrize('mutation',[
    lambda r,p,a:a['four_mean'].__setitem__((0,0,0),99),
    lambda r,p,a:a['four_first_residual_last'].__setitem__((0,0,0),99),
    lambda r,p,a:r['runs']['four']['capture']['mapping_steps'][0]['evidence']['token_slots'].__setitem__(0,2),
    lambda r,p,a:r['runs']['pair_0']['capture']['mapping_steps'][0]['inactive_plane_sha256']['2'].__setitem__('residual_last','a'*64),
    lambda r,p,a:r['runs']['four']['capture']['consumed_token_ids'][0].__setitem__(0,99),
    lambda r,p,a:r['runs']['four']['capture']['generations'].__setitem__(0,99),
    lambda r,p,a:r['runs']['four']['capture']['step_plane_sha256'][0][1].__setitem__('residual_last','a'*64),
    lambda r,p,a:r['source_sha256'].__setitem__('unexpected','a'*64),
    lambda r,p,a:r['runs']['four'].__setitem__('capture_restored',False),
])
def test_mutations_rejected(mutation):
    r,p,a=fixture();mutation(r,p,a)
    with pytest.raises(ValueError):recompute(r,p,a,layers=2,hidden=4,intermediate=6)
