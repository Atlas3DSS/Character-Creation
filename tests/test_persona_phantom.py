import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from persona_phantom import SCHEMA, LAYERS, LearnedKVCache, file_sha, validate_control
from persona_probe import completion_outcome, normalize_request
from persona_vectors import canonical_digest


def fixture(tmp_path, slots=4):
    binding={k:v*64 for k,v in zip(('model_identity','runtime_identity','prepared_identity','layout_sha256'),'abcd')}
    backing=torch.full((3,64),123,dtype=torch.uint8)
    pages={f'layer{i}.{field}':torch.full(((slots+7)//8+1,8*4*256*2),77,dtype=torch.uint8) for i in LAYERS for field in ('key','value')}
    pool=NS(reserved_backing=backing,pages=pages,reserved_region=lambda name:backing[:,-8:].view(torch.int64))
    e=NS(output=tmp_path,pool=pool,page_tokens=8,manifest={'checkpoint_config_sha256':'e'*64,'checkpoint_index_sha256':'f'*64},
         requests=NS(consumed_tokens=torch.zeros(1,dtype=torch.int64),rng_counter=torch.tensor([37])))
    keys=np.arange(16*4*slots*256,dtype=np.float32).reshape(16,4,slots,256)/1000
    meta={'schema':SCHEMA,'model_identity':binding['model_identity'],'key_space':'post_norm_post_rope','prompt_position_start':slots,
          'recurrent_initial_state':'fresh_zero','optimizer_steps':1,'gradient_validation':{'passed':True},
          'checkpoint':{'config_sha256':'e'*64,'index_sha256':'f'*64}}
    path=tmp_path/'learned.npz'
    np.savez(path,keys=keys,values=-keys,attention_layers=np.asarray(LAYERS,dtype=np.int64),metadata=np.asarray(json.dumps(meta)))
    artifact={'schema':SCHEMA,'kind':'learned_attention_kv','file':path.name,'weights_sha256':file_sha(path),'slots':slots,**binding}
    artifact['sha256']=canonical_digest(artifact)
    return e,dict(artifact=artifact,dose=1,cache_policy='reuse'),keys


@pytest.mark.parametrize('slots',[1,4,9,128])
def test_learned_pages_fresh_recurrence_and_rng(tmp_path,slots):
    e,control,keys=fixture(tmp_path,slots);cache=LearnedKVCache(e,{k:control['artifact'][k] for k in ('model_identity','runtime_identity','prepared_identity','layout_sha256')})
    count,entry,hit=cache.prepare(control);assert count==slots and hit is False
    cache.install(entry)
    assert torch.count_nonzero(e.pool.reserved_backing[1,:-8])==0
    assert e.pool.reserved_region('consumed_tokens')[1].item()==slots
    assert e.requests.consumed_tokens.item()==slots and e.requests.rng_counter.item()==37
    for i,layer in enumerate(LAYERS):
        pages=(slots+7)//8
        view=e.pool.pages[f'layer{layer}.key'][:pages].view(torch.bfloat16).view(-1,4,256)
        assert torch.equal(view[:slots],torch.from_numpy(keys[i]).transpose(0,1).to(torch.bfloat16))
        assert torch.count_nonzero(view[slots:])==0
    assert cache.prepare(control)[2] is True
    assert cache.prepare(dict(control,cache_policy='rebuild'))[2] is False
    (tmp_path/'learned.npz').unlink()
    assert cache.prepare(dict(control,dose=0))==(0,None,False)
    with pytest.raises(OSError):cache.prepare(control)


def test_learned_protocol_and_hash_rejection(tmp_path):
    e,control,_=fixture(tmp_path)
    request=dict(model='m',messages=[dict(role='user',content='A fictional scene.')],kv_control=control)
    assert normalize_request(request,'m')['kv_control']==control
    for change in ({'file':'../learned.npz'},{'slots':0},{'weights_sha256':'bad'}):
        a=dict(control['artifact'],**change);a['sha256']=canonical_digest({k:v for k,v in a.items() if k!='sha256'})
        with pytest.raises(ValueError):validate_control(dict(control,artifact=a))
    with pytest.raises(ValueError):validate_control(dict(control,dose=.5))


def test_virtual_slots_are_not_claimed_as_observed_tokens():
    spec={'max_tokens':1,'observation_only':True}
    args=dict(prompt_tokens=3,generated=[5],graph_kinds=['prefill'],result={'finished':True,'finish_reason':'length'},answer='')
    assert completion_outcome(spec,positions=[4,5,6],position_offset=4,**args)['diagnostics']['prefill_observation_complete']
    with pytest.raises(RuntimeError):completion_outcome(spec,positions=list(range(7)),position_offset=4,**args)


def test_finite_fp32_overflowing_bfloat16_is_rejected(tmp_path):
    e,control,_=fixture(tmp_path);path=tmp_path/'learned.npz'
    with np.load(path,allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
    arrays['keys'][0,0,0,0]=np.finfo(np.float32).max
    np.savez(path,**arrays)
    a=control['artifact'];a['weights_sha256']=file_sha(path);a['sha256']=canonical_digest({k:v for k,v in a.items() if k!='sha256'})
    cache=LearnedKVCache(e,{k:a[k] for k in ('model_identity','runtime_identity','prepared_identity','layout_sha256')})
    with pytest.raises(ValueError,match='finite'):cache.prepare(control)
