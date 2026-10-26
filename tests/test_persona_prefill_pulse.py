import persona
from types import SimpleNamespace as NS
import threading
import numpy as np
import pytest
from persona_trace import normalize_prefill_capture, plan_prefill, PrefillCapture, ResponseCapture
from persona_probe import normalize_request, configure_pulse_span


def test_exact_partition_and_prefix_no_text_roundtrip():
    ids=list(range(20));cap=normalize_prefill_capture({'id':'a'*32,'layers':[3,63],'positions':[0,6,8,12]})
    used,spans,meta=plan_prefill(ids,8,stop=13,pulse=6,capture=cap)
    assert used==ids[:13] and ids==list(range(20))
    assert spans==[(0,1),(1,6),(6,7),(7,8),(8,9),(9,13)]
    assert [i for start,stop in spans for i in range(start,stop)]==used
    assert max(b-a for a,b in spans)<=8
    assert meta['original_token_ids_sha256']==ResponseCapture.ids_hash(ids)
    assert meta['consumed_token_ids_sha256']==ResponseCapture.ids_hash(ids[:13])
    assert plan_prefill(ids,8,stop=13,pulse=6,capture=cap)[1]==spans  # independent of intervention magnitude/sign
    for kw in ({'stop':21},{'stop':0},{'stop':True},{'pulse':20},{'stop':5,'pulse':5},{'stop':5,'capture':cap}):
        with pytest.raises(ValueError):plan_prefill(ids,8,**kw)


def test_pulse_buffers_only_active_for_one_token_and_zero_arm():
    torch=pytest.importorskip('torch')
    ids,spans,_=plan_prefill(list(range(20)),8,pulse=8)
    for dose in (-.1,0,.1):
        addends=torch.full((2,3),999.);phases=torch.full((2,),7);seen=[]
        def install():addends[0,1]=dose;phases[0]=1
        for start,stop in spans:
            configure_pulse_span(addends,phases,8,start,stop,install)
            for position in range(start,stop):seen.append((position,float(addends[0,1]),int(phases[0])))
        assert [p for p,v,phase in seen if phase]==[8]
        assert all(v==0 for p,v,phase in seen if p!=8)
        assert seen[8][1]==pytest.approx(dose)
        configure_pulse_span(addends,phases,8,-1,-1,install)
        assert not torch.any(addends) and not torch.any(phases)


def test_protocol_rejects_mixed_offsets_and_requires_prefill():
    base=dict(model='m',messages=[dict(role='user',content='prompt')],max_tokens=1,observation_only=True,
              steering_pulse={'position':3},capture_prefill={'id':'a'*32,'layers':[0,63],'positions':[2,3,4]},prefill_stop=5)
    result=normalize_request(base,'m');assert result['prefill_stop']==5
    for change in (dict(steering={'phase':'all','sites':[]}),dict(steering_pulse={'position':True}),dict(kv_control={}),dict(capture_search={}),dict(observation_only=False),dict(prefill_stop=False)):
        with pytest.raises(ValueError):normalize_request(dict(base,**change),'m')
    no_stop=dict(base);no_stop.pop('prefill_stop');no_stop.update(observation_only=False,max_tokens=10,capture_response={'id':'b'*32})
    assert normalize_request(no_stop,'m')['capture_response']
    no_stop['capture_response']['id']='a'*32
    with pytest.raises(ValueError,match='distinct'):normalize_request(no_stop,'m')


class Tensor:
    def __init__(self,value):self.value=np.asarray(value)
    def cpu(self):return self
    def numpy(self):return self.value


def test_full_residual_snapshots_are_exact_and_position_bound():
    plane=np.arange(64*5,dtype=np.float32).reshape(64,5)
    e=NS(model=NS(layers=64,hidden=5,identity='model'),manifest={'runtime_identity':'runtime'},
         prepared=NS(server_config={'identity':'prepared'}),trace_lock=threading.RLock(),trace_records={},
         capture=NS(positions=Tensor([0,1,-1]),residual_last=Tensor(plane)))
    spec=normalize_prefill_capture({'id':'a'*32,'layers':[0,63],'positions':[1,3]})
    c=PrefillCapture(e,spec,[10,11,12,13],'request');c.observe(1)
    with pytest.raises(ValueError,match='coverage'):c.arrays()
    e.capture.positions=Tensor([2,3,-1]);e.capture.residual_last=Tensor(plane+1);c.observe(3)
    arrays=c.arrays();assert arrays['prefill_snapshot_token_ids'].tolist()==[11,13]
    assert np.array_equal(arrays['prefill_snapshot_residuals'],np.stack([plane[[0,63]],(plane+1)[[0,63]]]))
    result=c.result({'sha256':'bound'});assert result['complete'] and result['artifact']['sha256']=='bound'
    assert result['positions_sha256']==ResponseCapture.ids_hash([1,3])
    assert e.trace_records['a'*32]['state']=='complete'
