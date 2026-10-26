import json
from pathlib import Path
import numpy as np
import pytest
from persona_vectors import canonical_digest
from persona_kv import file_sha
import persona_phantom_qualify as q
from persona_kv_qualify import checked_arrays
from persona_phantom import SCHEMA, LAYERS


def fixture(tmp_path):
    binding = dict(zip(('model_identity','runtime_identity','prepared_identity','layout_sha256'), (c*64 for c in 'abcd')))
    candidate = tmp_path/'candidate.npz'
    metadata = dict(schema=SCHEMA, model_identity=binding['model_identity'], prompt_position_start=4,
                    key_space='post_norm_post_rope', recurrent_initial_state='fresh_zero')
    np.savez(candidate, keys=np.zeros((16,4,4,256), np.float32), values=np.ones((16,4,4,256), np.float32),
             attention_layers=np.asarray(LAYERS), metadata=np.asarray(json.dumps(metadata)))
    artifact = q.make_artifact(candidate,binding)
    request = dict(model='m',kv_control=dict(artifact=artifact,dose=1,cache_policy='rebuild'))
    path = tmp_path/'activation.npz'
    values = dict(prompt_token_ids=np.asarray([1,2,3]),generated_token_ids=np.asarray([4]),
                  prefill_positions=np.asarray([4,5,6]),virtual_prefix_slots=np.int64(4),
                  completion_trace=np.asarray([[1,1,0,0,0,0,0,0],[1,2,7,1,4,1,0,1]]),
                  mlp_input_last=np.zeros((64,17408),np.float32),residual_last=np.zeros((64,5120),np.float32),
                  pre_mlp_last=np.zeros((64,5120),np.float32),residual_l2=np.zeros((64,2),np.float32),
                  request_sha256=np.asarray(canonical_digest(request)),**{k:np.asarray(v) for k,v in binding.items() if k!='layout_sha256'})
    np.savez(path, **values)
    response = dict(**{k:v for k,v in binding.items() if k!='layout_sha256'},
        activation_artifact=dict(path=str(path),sha256=file_sha(path),request_sha256=canonical_digest(request),**{k:v for k,v in binding.items() if k!='layout_sha256'}),
        diagnostics=dict(complete_final_answer=True,virtual_prefix_slots=4),
        kv_control=dict(**binding,artifact_sha256=artifact['sha256'],dose=1,cache_policy='rebuild',kind='learned_attention_kv',
                        virtual_prefix_slots=4,prefix_tokens=4,attention_only=True,recurrent_initial_state='fresh_zero',
                        state_artifact=dict(path=str(candidate),sha256=file_sha(candidate))))
    return binding,request,response,values,path


def test_virtual_real_token_alignment(tmp_path):
    binding,request,response,values,path=fixture(tmp_path)
    arrays=checked_arrays(response,request,binding)
    assert arrays['prompt_token_ids'].tolist()==[1,2,3]
    values['prefill_positions']=np.arange(3)
    np.savez(path,**values);response['activation_artifact']['sha256']=file_sha(path)
    with pytest.raises(ValueError,match='coverage'):checked_arrays(response,request,binding)


def test_reject_wrong_virtual_and_state_bindings(tmp_path):
    binding,request,response,values,path=fixture(tmp_path)
    response['kv_control']['recurrent_initial_state']='cached'
    with pytest.raises(ValueError,match='recurrence'):checked_arrays(response,request,binding)
    response['kv_control']['recurrent_initial_state']='fresh_zero'
    response['kv_control']['state_artifact']['sha256']='f'*64
    with pytest.raises(ValueError,match='artifact mismatch'):checked_arrays(response,request,binding)


def test_nonnoop_and_actual_ids():
    baseline=dict(prompt_token_ids=np.asarray([1,2]),mlp_input_last=np.zeros(2))
    with pytest.raises(ValueError,match='no-op'):q.prove_effect(baseline,baseline)
    grafted=dict(prompt_token_ids=np.asarray([1,2]),mlp_input_last=np.ones(2))
    q.prove_effect(baseline,grafted)
    grafted['prompt_token_ids']=np.asarray([9,1,2])
    with pytest.raises(ValueError,match='actual prompt'):q.prove_effect(baseline,grafted)


def test_exact_twelve_requests_and_no_post_retry(tmp_path,monkeypatch):
    binding,request,response,values,path=fixture(tmp_path)
    artifact=request['kv_control']['artifact'];calls=[]
    descriptor=dict(id='m',flare=dict(**{k:v for k,v in binding.items() if k!='layout_sha256'},
        kv_control=dict(supported=True,kinds=['learned_attention_kv'],layout_sha256=binding['layout_sha256']),
        steering=dict(sites=[dict(name='layer3.mlp_input',width=2)])))
    def fetch(url, request=None):
        if request is None:return dict(data=[descriptor])
        calls.append(request)
        c=request.get('kv_control');on=bool(c and c['dose'])
        return dict(kv_control=dict(cache_hit=c['cache_policy']=='reuse',state_artifact={'sha256':'x'})) if on else {}
    monkeypatch.setattr(q,'fetch',fetch)
    monkeypatch.setattr(q,'checked_native',lambda r,p,b:dict(prompt_token_ids=np.asarray([1,2]),mlp_input_last=np.asarray([bool(p.get('kv_control',{}).get('dose'))])))
    result=q.qualify_endpoint('http://worker:8930',tmp_path/'candidate.npz',tmp_path,'test')
    assert result['passed'] and len(calls)==12
    assert all('kv_control' not in calls[i] for i in (0,4,8))
    def fail(url, request=None):
        if request is None:return dict(data=[descriptor])
        calls.append(request);raise TimeoutError('uncertain')
    monkeypatch.setattr(q,'fetch',fail);calls.clear()
    with pytest.raises(TimeoutError):q.qualify_endpoint('http://worker:8930',tmp_path/'candidate.npz',tmp_path,'failed')
    assert len(calls)==1
    log=next(tmp_path.glob('phantom-attempt-failed-*.jsonl')).read_text()
    assert 'started' in log and 'failed_or_uncertain' in log

def test_publication_requires_all_six_and_keeps_partial_results(tmp_path,monkeypatch):
    binding,request,response,values,path=fixture(tmp_path)
    endpoints=['http://worker:'+str(8930+i) for i in range(6)]
    def qualify(endpoint,*args):
        if endpoint==endpoints[-1]:raise ValueError('one worker failed')
        return dict(endpoint=endpoint,passed=True,binding=binding,artifacts=[request['kv_control']['artifact']],records=[])
    monkeypatch.setattr(q,'qualify_endpoint',qualify)
    args=['--candidate',str(tmp_path/'candidate.npz'),'--output',str(tmp_path)]
    for endpoint in endpoints:args+=['--endpoint',endpoint]
    assert q.main(args)==1
    assert not (tmp_path/'phantom-qualification.json').exists()
    report=json.loads(next(tmp_path.glob('phantom-qualification-*.json')).read_text())
    assert len(report['results'])==5 and len(report['failures'])==1
