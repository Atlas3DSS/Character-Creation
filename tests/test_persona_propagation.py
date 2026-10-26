import persona
import copy
import json
import threading
import time
from types import SimpleNamespace as NS
import numpy as np
import pytest
from persona_propagation import (ArtifactStore, Runner, check_launch, exact, validate_plan,
    validate_response_capture, public_report, checked, canonical_digest, file_sha)
from persona_trace import ResponseCapture


def launch():
    return {'deadline':time.time()+600,'binding':{k:'a'*64 for k in ('model_identity','runtime_identity','prepared_identity')},
            'endpoints':[{'url':f'http://127.0.0.1:{p}'} for p in range(8930,8936)]}


def test_launch_requires_six_bound_live_endpoints():
    value=launch();assert len(check_launch(value)[0])==6
    for mutate in (lambda v:v.update(deadline=0),lambda v:v['endpoints'].pop(),
                   lambda v:v['endpoints'][0].update(url='http://example.org:8930'),
                   lambda v:v['binding'].update(runtime_identity='bad')):
        bad=copy.deepcopy(value);mutate(bad)
        with pytest.raises(ValueError):check_launch(bad)


def native():
    return dict(prompt_token_ids=np.arange(4),generated_token_ids=np.array([8,9]),
        mlp_input_last=np.zeros((64,17408),np.float32),residual_last=np.zeros((64,5120),np.float32),
        pre_mlp_last=np.zeros((64,5120),np.float32),residual_l2=np.zeros((64,4),np.float32),
        completion_trace=np.array([[1,1,4,1,8,0,0,1],[1,2,5,2,9,0,0,2]]),
        completion_graph_kind=np.array(['prefill','decode']),prefill_positions=np.arange(4))


def test_exact_checks_tokens_rng_activations_and_graphs():
    a=native();b=copy.deepcopy(a);b['completion_trace'][:,:2]+=9;exact(a,b)
    for key,index in [('generated_token_ids',0),('residual_last',(32,0)),('completion_trace',(1,7))]:
        b=copy.deepcopy(a);b[key][index]+=1
        with pytest.raises(ValueError):exact(a,b)
    b=copy.deepcopy(a);b['completion_graph_kind'][0]='decode'
    with pytest.raises(ValueError,match='graph'):exact(a,b)
    b=copy.deepcopy(a);b['completion_trace']=np.vstack([b['completion_trace'][0],b['completion_trace']]);b['completion_graph_kind']=np.array(['prefill','prefill','decode'])
    exact(a,b,partition_same=False)


class Tensor:
    def __init__(self,x):self.x=x
    def cpu(self):return self
    def numpy(self):return self.x


def response_fixture(layers=[31],count=2):
    binding=launch()['binding'];request={'capture_response':{'id':'b'*32,'layers':layers,'max_positions':128 if layers else 0}}
    capture=NS(positions=Tensor(np.array([4])),residual_last=Tensor(np.ones((64,5120),np.float32)))
    e=NS(model=NS(layers=64,hidden=5120,identity=binding['model_identity']),manifest=binding,
        prepared=NS(server_config={'identity':binding['prepared_identity']}),decode_generation=NS(decoder=NS(activation_capture=capture)),trace_lock=threading.RLock(),trace_records={})
    c=ResponseCapture(e,request['capture_response'],4,canonical_digest(request));generated=list(range(8,8+count+1))
    for i in range(count):
        capture.positions=Tensor(np.array([4+i]));c.observe(4+i,generated[i])
    artifact={};response={**binding,'activation_artifact':artifact,'response_capture':c.result(generated,artifact,True)}
    arrays={**c.arrays(generated),'prompt_token_ids':np.arange(4),'generated_token_ids':np.array(generated)}
    return response,request,arrays


def test_response_means_use_real_consumed_tokens_and_undefined_zero():
    for layers,count in (([31],2),([],2),([],0)):
        r,q,a=response_fixture(layers,count);validate_response_capture(r,q,a)
    r,q,a=response_fixture();a['response_residual_mean'][0,0]+=1
    with pytest.raises(ValueError,match='arithmetic'):validate_response_capture(r,q,a)
    r,q,a=response_fixture();a['response_selected_residuals'][0,0,0]+=1
    with pytest.raises(ValueError,match='profiles'):validate_response_capture(r,q,a)
    r,q,a=response_fixture();r['response_capture']['runtime_identity']='bad'
    with pytest.raises(ValueError):validate_response_capture(r,q,a)


def test_store_rejects_paths_and_checks_existing_bytes(tmp_path):
    store=ArtifactStore(tmp_path,'v100','/remote/results');name='probe-'+'a'*32+'.npz';p=tmp_path/name;p.write_bytes(b'good')
    pointer={'path':'/remote/results/'+name,'sha256':file_sha(p)};assert store.get(pointer)==p
    for bad in ({**pointer,'path':'/other/'+name},{**pointer,'path':'/remote/results/../'+name},{**pointer,'sha256':'0'*64}):
        with pytest.raises(ValueError):store.get(bad)


def test_post_failure_is_durable_and_never_retried(tmp_path):
    runner=Runner(launch(),tmp_path/'report.json',None);calls=[]
    def fail(*args):calls.append(args);raise TimeoutError('uncertain')
    runner.fetch=fail
    with pytest.raises(TimeoutError):runner.infer(runner.endpoints[0],{'prompt':'test'},'one')
    assert len(calls)==1
    events=[json.loads(line) for line in runner.attempts.read_text().splitlines()]
    assert [x['state'] for x in events]==['started','failed_or_uncertain']
    with pytest.raises(ValueError,match='Prior attempts'):Runner(launch(),tmp_path/'report.json',None)


def test_plan_rejects_heldout_wrong_layer_and_missing_calibration():
    p={'contexts':[{'id':str(i),'prompt':'fixed','split':'discovery','source_position':8} for i in range(6)],
       'directions':[{'id':str(i),'layer':32,'vector':[1.]+[0.]*5119,'residual_median':2.,'source_sha256':'a'*64} for i in range(2)]}
    validate_plan(p)
    for mutate in (lambda v:v['contexts'][0].update(split='heldout'),lambda v:v['directions'][0].update(layer=31),lambda v:v['directions'][0].update(residual_median=0)):
        bad=copy.deepcopy(p);mutate(bad)
        with pytest.raises(ValueError):validate_plan(bad)
    assert public_report({'results':[{'_small':np.ones(2),'metrics':[1]}]})=={'results':[{'metrics':[1]}]}


def test_checked_verifies_saved_response_request_and_npz_bindings(tmp_path):
    q={'observation_only':True};binding=launch()['binding'];a=native();digest=canonical_digest(q)
    path=tmp_path/('probe-'+'a'*32+'.npz');np.savez(path,**a,**binding,request_sha256=digest)
    pointer={**binding,'request_sha256':digest,'path':str(path),'sha256':file_sha(path)}
    r={**binding,'activation_artifact':pointer,'diagnostics':{'prefill_observation_complete':True}}
    provenance=tmp_path/('probe-'+'b'*32+'.jsonl')
    saved={'status':'complete','state_restored':True,'request':q,'request_sha256':digest,**binding,'response':copy.deepcopy(r)}
    provenance.write_text(json.dumps(saved));r['provenance_artifact']={**binding,'request_sha256':digest,'path':str(provenance),'sha256':file_sha(provenance)}
    store=ArtifactStore(tmp_path,'v100',str(tmp_path));checked(r,q,binding,store)
    saved['state_restored']=False;provenance.write_text(json.dumps(saved));r['provenance_artifact']['sha256']=file_sha(provenance)
    with pytest.raises(ValueError,match='Provenance'):checked(r,q,binding,store)


def test_source_invariance_allows_recurrent_future_upstream_effects():
    from persona_propagation import source_invariant
    a={'prompt_token_ids':np.arange(20),'prefill_snapshot_residuals':np.zeros((5,4,5120))}
    b=copy.deepcopy(a);b['prefill_snapshot_residuals'][1:,0]=1;source_invariant(a,b)
    b['prefill_snapshot_residuals'][0,1:]=2;source_invariant(a,b)
    b['prefill_snapshot_residuals'][0,0,0]=1
    with pytest.raises(ValueError,match='upstream'):source_invariant(a,b)
    b=copy.deepcopy(a);b['prompt_token_ids'][0]=99
    with pytest.raises(ValueError,match='Fixed tokens'):source_invariant(a,b)


def test_prefill_metadata_requires_isolated_source_and_hashed_actual_tokens():
    from persona_propagation import validate_prefill_capture
    binding=launch()['binding'];positions=[8,9,10,12,16];layers=[31,32,48,63];ids=np.arange(20)
    q={'steering_pulse':{'position':8},'capture_prefill':{'id':'a'*32,'positions':positions,'layers':layers}}
    a=dict(prompt_token_ids=ids,prefill_snapshot_positions=np.array(positions),prefill_snapshot_layers=np.array(layers),
           prefill_snapshot_token_ids=ids[positions],prefill_snapshot_residuals=np.zeros((5,4,5120)))
    r={**binding,'activation_artifact':{},'prefill_capture':{**binding,'id':'a'*32,'complete':True,'artifact':{},
       'request_sha256':canonical_digest(q),'positions':positions,'layers':layers,
       'positions_sha256':ResponseCapture.ids_hash(positions),'token_ids_sha256':ResponseCapture.ids_hash(ids[positions])},
       'prefill_execution':{'pulse_position':8,'consumed_token_ids_sha256':ResponseCapture.ids_hash(ids),'spans':[[0,8],[8,9],[9,20]]}}
    validate_prefill_capture(r,q,a)
    bad=copy.deepcopy(r);bad['prefill_execution']['spans']=[[0,9],[9,20]]
    with pytest.raises(ValueError,match='isolated'):validate_prefill_capture(bad,q,a)
    bad=copy.deepcopy(r);bad['prefill_capture']['token_ids_sha256']='0'*64
    with pytest.raises(ValueError,match='alignment'):validate_prefill_capture(bad,q,a)


def test_partition_sensitivity_is_reported_without_rejecting_pulse():
    from persona_propagation import partition_difference
    a=native();b=copy.deepcopy(a);b['residual_last'][1,0]=.125
    result=partition_difference(a,b)
    assert result['exact_equal'] is False and result['same_actual_prompt_tokens'] is True
    assert result['activation_differences']['residual_last']['max_abs']==.125


def test_basis_scale_exceeds_half_bf16_spacing_for_smallest_dose():
    from persona_propagation import pulse_scale
    for coordinate in (0.,1.,32.,4096.,-4096.):
        vector=np.zeros(5120);vector[0]=coordinate
        result=pulse_scale(vector)
        assert .05*result['basis_scale']>.5*result['baseline_bf16_spacing']
        assert result['largest_absolute_dose']<=max(1.,.1*np.linalg.norm(vector))


def test_pulse_failure_does_not_invalidate_response_capture(monkeypatch):
    import persona_propagation as module
    model={'id':'m','flare':{'steering':{'sites':[{'name':'layers.32.residual','width':5120}]}}}
    calls=[]
    class Stub:
        def model(self,endpoint):return model
        def infer(self,endpoint,request,label):
            calls.append(label);a=native();a['response_residual_count']=np.array(1)
            return {},a,{'attempt':label}
    def fail(*args):raise ValueError('pulse invariant failed')
    monkeypatch.setattr(module,'qualify_pulse',fail)
    result=module.qualification_worker(Stub(),'endpoint')
    assert len(calls)==9 and result['passed'] and result['response_capture_passed']
    assert not result['pulse']['passed'] and 'pulse invariant' in result['pulse']['error']


def test_pulse_only_source_coordinate_and_signed_actual_dose():
    from persona_propagation import qualify_pulse
    model={'id':'m','flare':{'steering':{'sites':[{'name':'layers.32.residual','width':5120}]}}}
    class Stub:
        contaminate=False
        def infer(self,endpoint,request,label):
            a=native();a['prompt_token_ids']=np.arange(30)
            a['prefill_snapshot_residuals']=np.zeros((5,4,5120),np.float32)
            a['prefill_snapshot_residuals'][0,1,0]=4096
            if request.get('steering'):
                value=request['steering']['sites'][0]['vector'][0]
                a['prefill_snapshot_residuals'][0,1,0]+=value
                if self.contaminate:a['prefill_snapshot_residuals'][0,1,1]=1
            if label=='partitioned_zero':a['residual_last'][0,0]=.125
            return {},a,{'attempt':label}
    result={'arms':[]};qualify_pulse(Stub(),'endpoint',model,'prompt',result)
    assert len(result['arms'])==4 and not result['partition_comparison']['exact_equal']
    assert all(x['actual_source_delta']*x['absolute_dose']>0 for x in result['arms'])
    stub=Stub();stub.contaminate=True
    with pytest.raises(ValueError,match='confined'):qualify_pulse(stub,'endpoint',model,'prompt',{'arms':[]})


def test_model_readiness_retries_only_get_transport_errors(tmp_path,monkeypatch):
    import persona_propagation as module
    import urllib.error
    runner=Runner(launch(),tmp_path/'report.json',None);calls=[];sleeps=[]
    model={'flare':{**runner.binding,'response_capture':{'supported':True},'prefill_instrumentation':{'supported':True}}}
    def fetch(url):
        calls.append(url)
        if len(calls)<3:raise ConnectionResetError('warming')
        return {'data':[model]}
    runner.fetch=fetch;monkeypatch.setattr(module.time,'sleep',sleeps.append)
    assert runner.model(runner.endpoints[0])==model
    assert sleeps==[10,10] and all(x.endswith('/v1/models') for x in calls)
    runner.fetch=lambda url:(_ for _ in ()).throw(urllib.error.HTTPError(url,500,'failure',{},None))
    with pytest.raises(urllib.error.HTTPError):runner.model(runner.endpoints[0])
    runner.deadline=time.time()+299
    runner.fetch=lambda url:(_ for _ in ()).throw(ConnectionResetError('warming'))
    with pytest.raises(ConnectionResetError):runner.model(runner.endpoints[0])
    assert sleeps==[10,10]
