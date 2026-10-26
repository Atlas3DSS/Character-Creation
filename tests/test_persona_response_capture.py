import persona
import threading
from types import SimpleNamespace as NS
import numpy as np
import pytest
from persona_trace import normalize_response_capture, ResponseCapture
from persona_probe import normalize_request

class Tensor:
    def __init__(self,value):self.value=np.asarray(value)
    def cpu(self):return self
    def numpy(self):return self.value

def engine():
    capture=NS(positions=Tensor([10]),residual_last=Tensor(np.zeros((2,3),np.float32)))
    return NS(model=NS(layers=2,hidden=3,identity='model'),manifest={'runtime_identity':'runtime'},
        prepared=NS(server_config={'identity':'prepared'}),decode_generation=NS(decoder=NS(activation_capture=capture)),
        trace_lock=threading.RLock(),trace_records={})

def test_bounds_and_incompatible_requests():
    assert normalize_response_capture({'id':'a'*32})=={'id':'a'*32,'layers':[],'max_positions':0}
    for spec in ({'id':'a'*32,'layers':[[]]}, {'id':'a'*32,'layers':[True]}, {'id':'a'*32,'layers':[1,1]},
                 {'id':'a'*32,'layers':[0,1,2,3,4]}, {'id':'a'*32,'layers':[0],'max_positions':129},
                 {'id':'a'*32,'max_positions':1}):
        with pytest.raises(ValueError):normalize_response_capture(spec)
    base=dict(model='m',messages=[dict(role='user',content='scene')],max_tokens=10,capture_response={'id':'a'*32})
    assert normalize_request(base,'m')['capture_response']['id']=='a'*32
    for change in (dict(observation_only=True,max_tokens=1),dict(max_tokens=4097),dict(capture_search={}),dict(kv_control={})): 
        with pytest.raises(ValueError):normalize_request(dict(base,**change),'m')
    with pytest.raises(ValueError,match='distinct ids'):
        normalize_request(dict(base,capture_trace={'id':'a'*32,'neurons':[{'layer':0,'neuron':0}]}),'m')

def test_streaming_means_alignment_and_selected_prefix():
    e=engine();r=ResponseCapture(e,normalize_response_capture({'id':'a'*32,'layers':[1],'max_positions':2}),10,'request')
    originals=[]
    for index,value in enumerate((1e8,1.,-1e8)):
        a=np.full((2,3),value,np.float32);originals.append(a.copy())
        e.decode_generation.decoder.activation_capture.positions=Tensor([10+index])
        e.decode_generation.decoder.activation_capture.residual_last=Tensor(a)
        r.observe(10+index,7+index)
        assert np.array_equal(a,originals[-1])
    arrays=r.arrays([7,8,9,99])
    assert np.array_equal(arrays['response_residual_sum'],np.ones((2,3)))
    assert np.allclose(arrays['response_residual_mean'],1/3)
    assert arrays['response_consumed_positions'].tolist()==[10,11,12]
    assert arrays['response_selected_positions'].tolist()==[10,11]
    assert arrays['response_selected_residuals'].shape==(2,1,3)
    result=r.result([7,8,9,99],{'sha256':'artifact'},True)
    assert result['final_sampled_unconsumed_token_id']==99 and result['consumed_generated_tokens']==3
    assert result['all_consumed_generated_tokens_captured'] and not result['all_sampled_generated_tokens_captured']
    assert result['selected_profile_complete'] is False
    assert result['consumed_token_ids_sha256']==ResponseCapture.ids_hash([7,8,9])
    assert result['model_identity']=='model' and e.trace_records['a'*32]['state']=='complete'


def test_single_sample_has_no_fabricated_response_mean():
    e=engine();r=ResponseCapture(e,normalize_response_capture({'id':'a'*32}),10,'request')
    arrays=r.arrays([99]);assert not arrays['response_residual_mean_defined']
    assert arrays['response_selected_residuals'].shape==(0,0,3)
    result=r.result([99],{},False)
    assert result['consumed_generated_tokens']==0 and not result['mean_defined'] and not result['generation_complete']


def test_alignment_nonfinite_and_coverage_fail_closed():
    e=engine();r=ResponseCapture(e,normalize_response_capture({'id':'a'*32}),10,'request')
    with pytest.raises(ValueError,match='alignment'):r.observe(11,7)
    e.decode_generation.decoder.activation_capture.residual_last=Tensor(np.full((2,3),np.nan,np.float32))
    with pytest.raises(ValueError,match='Invalid full-layer'):r.observe(10,7)
    with pytest.raises(ValueError,match='coverage'):r.arrays([7,8])
    r.finish_failed('test');assert e.trace_records['a'*32]['state']=='failed'
