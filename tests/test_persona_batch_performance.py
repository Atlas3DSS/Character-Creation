"""CPU-only schedule/equality checks; no inference, network or native engine."""
import persona
from copy import deepcopy
import numpy as np
import pytest
from persona_batch_performance import schedule,compare_observations,scientific_request,GROUPS,POLICY
from persona_batch_qualify import EXACT


def test_full24_cases_56_responses_all_widths_and_paired_long_repeat():
    plan={'jobs':[{'id':str(i)} for i in range(24)]};runs=schedule(plan)
    assert sum(len(r['job_ids']) for r in runs)==56
    for mode in ('baseline','optimized'):
        ids=[i for r in runs if r['repeat']==0 and r['mode']==mode for i in r['job_ids']]
        assert len(ids)==len(set(ids))==24
    assert {r['width'] for r in runs}=={1,2,4}
    assert [r['mode'] for r in runs[-2:]]==['optimized','baseline']
    assert runs[-1]['job_ids']==[str(i) for i in GROUPS[-1]]
    assert (6,7,18,19) in GROUPS and POLICY['max_wall_seconds']==1200


def observations():
    x={'request':{'seed':2,'capture_response':{'id':'a','layers':[]},'metadata':{'job_id':'stable','benchmark_id':'runA'}},
        'evidence':{'binding':{'model_identity':'same','runtime_identity':'same'},'request_sha256':'a','artifact_sha256':'b','prose_complete':False},
        'arrays':{k:np.ones((2,),np.float64) for k in EXACT},'response':{'choices':[{'finish_reason':'length','message':{'content':'partial'}}]}}
    x['arrays']['completion_trace']=np.array([[1,2,3,4,5,2,0,4]],np.int64)
    x['arrays']['completion_graph_kind']=np.array(['decode'])
    return x


def test_only_capture_and_benchmark_ids_and_trace_epochs_sequences_normalized():
    a=observations();b=deepcopy(a)
    b['request']['capture_response']['id']='b';b['request']['metadata']['benchmark_id']='runB'
    b['arrays']['completion_trace'][0,:2]=[100,101]
    assert compare_observations(a,b)['exact'] is True
    assert compare_observations(a,b)['prose_complete'] is False
    b['request']['metadata']['job_id']='different'
    with pytest.raises(ValueError,match='Scientific request'):compare_observations(a,b)


@pytest.mark.parametrize('change',['tokens','mean','rng','kind','termination','runtime','seed'])
def test_any_scientific_or_native_difference_rejects(change):
    a=observations();b=deepcopy(a)
    if change=='tokens':b['arrays']['generated_token_ids'][0]+=1
    elif change=='mean':b['arrays']['response_residual_mean'][0]=np.nextafter(1.,2.)
    elif change=='rng':b['arrays']['completion_trace'][0,7]+=1
    elif change=='kind':b['arrays']['completion_graph_kind']=np.array(['prefill'])
    elif change=='termination':b['response']['choices'][0]['finish_reason']='stop'
    elif change=='runtime':b['evidence']['binding']['runtime_identity']='other'
    else:b['request']['seed']=3
    with pytest.raises(ValueError):compare_observations(a,b)
