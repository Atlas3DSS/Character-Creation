"""Canonical search admission, two-pass evidence, and override cache binding."""
import persona
from copy import deepcopy
import json
import sys
import threading
from types import SimpleNamespace, ModuleType
import numpy as np
import pytest
import persona_probe as probe


def request():
    return dict(model='m',messages=[dict(role='user',content='A partial scene:')],max_tokens=1,
                observation_only=True,seed=7035,temperature=.7,capture_search=dict(id='a'*32,top_k=1))


def test_canonical_search_rejects_mixed_modes():
    value=request();assert probe.normalize_request(value,'m')['capture_search']['top_k']==1
    for change in (dict(observation_only=False),dict(max_tokens=2),dict(capture_trace={}),dict(kv_control={})):
        with pytest.raises(ValueError):probe.normalize_request(dict(value,**change),'m')


def engine(tmp_path, fail=False):
    calls=[];identity='a'*32
    result=dict(id=identity,coverage=dict(tokens=2),candidates=[dict(layer=3,neuron=7,peak=-4.,peak_position=1,mean_abs=3.)])
    first=dict(activation_search=result,activation_artifact={'sha256':'first'},provenance_artifact={'sha256':'first_provenance'})
    def complete(payload):
        calls.append(deepcopy(payload))
        if len(calls)==1:return first
        if fail:raise ValueError('profile failed')
        return dict(token_trace=dict(neurons=[dict(layer=3,neuron=7)],events=[dict(positions=[0,1],values=[[2.],[-4.]])],next_token_logits={'bound':'logits'}),
                    activation_artifact={'sha256':'second'},provenance_artifact={'sha256':'second_provenance'})
    owner=SimpleNamespace(_complete=complete,output=tmp_path,trace_lock=threading.RLock(),trace_records={identity:{'state':'running'}})
    return owner,calls,first


def test_two_pass_search_keeps_seed_tokens_and_evidence(tmp_path):
    owner,calls,first=engine(tmp_path)
    result=probe.ProbeEngine.complete(owner,request())
    assert result is first and len(calls)==2
    assert {k:v for k,v in calls[0].items() if k!='capture_search'}=={k:v for k,v in calls[1].items() if k!='capture_trace'}
    assert calls[1]['capture_trace']['neurons']==[dict(layer=3,neuron=7)]
    search=result['activation_search']
    assert search['candidates'][0]['values']==[2.,-4.]
    assert search['profile_verification']['peak_bit_exact']
    assert search['profile_artifact']=={'sha256':'second'}
    saved=json.loads((tmp_path/('search-'+'a'*32+'.json')).read_text())
    assert saved['activation_artifact']=={'sha256':'first'}
    assert saved['activation_search']['profile_provenance_artifact']=={'sha256':'second_provenance'}
    assert owner.trace_records['a'*32]['state']=='complete'


def test_failed_profile_cannot_publish_complete_search(tmp_path):
    owner,calls,_=engine(tmp_path,fail=True)
    with pytest.raises(ValueError,match='profile failed'):probe.ProbeEngine.complete(owner,request())
    assert owner.trace_records['a'*32]['state']=='failed'
    assert not list(tmp_path.iterdir())


def test_plain_request_still_uses_one_normal_pass(tmp_path):
    owner,calls,first=engine(tmp_path)
    payload=request();payload.pop('capture_search')
    assert probe.ProbeEngine.complete(owner,payload) is first
    assert len(calls)==1


def test_canonical_override_binds_actual_source_and_aot_namespace(tmp_path,monkeypatch):
    module=ModuleType('ignition.preparation.aot');seen=[]
    class Cache:
        def __init__(self,*,namespace,**kwargs):seen.append(namespace)
    module.ForwardCache=Cache
    monkeypatch.setitem(sys.modules,'ignition.preparation.aot',module)
    override=tmp_path/'activation_capture.py';override.write_text('source-one')
    other=tmp_path/'other.py';other.write_text('unchanged')
    hashes=probe.capture_source_hashes(tmp_path,['ignition/components/activation_capture.py','other.py'],override)
    assert hashes['ignition/components/activation_capture.py']==probe.file_hash(override)
    digest=probe.bind_capture_cache_namespace(override)
    Cache(namespace={'checkpoint':'bound'})
    assert seen==[{'checkpoint':'bound','activation_capture_override_sha256':digest}]
    assert probe.bind_capture_cache_namespace(override)==digest
    override.write_text('source-two')
    with pytest.raises(RuntimeError,match='another capture'):probe.bind_capture_cache_namespace(override)


def test_experimental_preparation_has_separate_cache_identity(tmp_path,monkeypatch):
    module=ModuleType('ignition.preparation.aot');seen=[]
    class Cache:
        def __init__(self,*,namespace,**kwargs):seen.append(namespace)
    module.ForwardCache=Cache
    monkeypatch.setitem(sys.modules,'ignition.preparation.aot',module)
    override=tmp_path/'activation_capture.py';override.write_text('same-capture')
    digest=probe.bind_capture_cache_namespace(override,'a'*64)
    Cache(namespace={'checkpoint':'bound'})
    assert seen==[{'checkpoint':'bound','activation_capture_override_sha256':probe.file_hash(override),'preparation_adapter_sha256':'a'*64}]
    assert digest!=probe.file_hash(override)
    assert probe.bind_capture_cache_namespace(override,'a'*64)==digest
    with pytest.raises(RuntimeError):probe.bind_capture_cache_namespace(override,'b'*64)
    with pytest.raises(RuntimeError):probe.bind_capture_cache_namespace(override)
