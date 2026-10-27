import persona
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import numpy as np
import pytest
from persona_batch_server import BatchEngine,BatchResponseCapture,envelope,page_plan,binding,verify_sources
from persona_vectors import canonical_digest


def engine(tmp_path,layers=2,hidden=4):
    return SimpleNamespace(model=SimpleNamespace(identity='a'*64,layers=layers,hidden=hidden),manifest={'runtime_identity':'b'*64},
        prepared=SimpleNamespace(server_config={'identity':'c'*64,'max_sequence_tokens':1024}),capacity=(5,8),page_tokens=128,
        inputs=SimpleNamespace(page_tables=np.zeros((4,8))),trace_lock=threading.Lock(),trace_records={},model_id='checkpoint',
        catalog=[{'name':'layers.0.residual','width':hidden}],steering={'max_sites':64},tokenizer=SimpleNamespace(decode=lambda *args,**kw:'A clear answer.'),vocabulary=100,output=tmp_path,eos=(99,))


def payload(i=0):return {'model':'checkpoint','messages':[{'role':'user','content':'authored'}],'seed':i,'max_tokens':32,'temperature':0,'capture_response':{'id':f'{i:032x}','layers':[],'max_positions':0}}


def test_source_pins_match_reviewed_native_components():verify_sources()


def test_aggregate_capacity_and_disjoint_full_budget_pages():
    counts,pages,fits=page_plan([128,129,3,400],[1,128,32,100],page_tokens=128,physical_pages=8,logical_context=1024,table_pages=8)
    assert counts==[2,3,1,4] and fits is False
    assert len(set(sum(pages,[])))==sum(counts)
    assert page_plan([128]*4,[1]*4,page_tokens=128,physical_pages=8,logical_context=1024,table_pages=8)[2]


def test_envelope_rejects_duplicates_bad_ids_and_oversize():
    for value in ({'batch_id':'bad/path','requests':[payload()]},{'batch_id':'ok','requests':[payload()]*2},{'batch_id':'ok','requests':[payload(i) for i in range(5)]}):
        with pytest.raises(ValueError):envelope(value)


def test_preflight_normalizes_actual_sampler_and_rejects_unsupported(tmp_path,monkeypatch):
    import persona_capture
    monkeypatch.setattr(persona_capture,'tokenize_messages',lambda tokenizer,messages:[3]*129)
    service=BatchEngine.__new__(BatchEngine);service.e=engine(tmp_path)
    value={'batch_id':'four','requests':[payload(i) for i in range(4)]}
    before=deepcopy(value);summary,specs,tokens,programs,pages=service.prepare(value)
    assert summary['fits'] and summary['required_pages']==[2]*4 and value==before
    assert summary['request_sha256']==[canonical_digest(p) for p in value['requests']]
    assert all(s['sampling']['top_k']==1 and s['sampling']['temperature']==1 for s in specs)
    for field in ('kv_control','capture_search','capture_trace','capture_prefill','steering_pulse','prefill_stop'):
        changed=deepcopy(value);changed['requests'][0][field]={}
        with pytest.raises(ValueError,match='Unsupported batch feature'):service.prepare(changed)
    service.e.trace_records[value['requests'][0]['capture_response']['id']]={}
    with pytest.raises(ValueError,match='already exists'):service.prepare(value)


def test_response_capture_unequal_counts_and_fp64_exact(tmp_path):
    e=engine(tmp_path);captures=[BatchResponseCapture(e,payload(i)['capture_response'],3,canonical_digest(payload(i))) for i in range(4)]
    for i,c in enumerate(captures):
        for pos in range(i):c.observe_value(3+pos,10+pos,np.full((2,4),i+.25,np.float32))
        arrays=c.arrays(list(range(10,10+i))+[99])
        assert arrays['response_residual_count']==i and arrays['response_residual_sum'].dtype==np.float64
        assert np.array_equal(arrays['response_residual_sum'],np.full((2,4),(i+.25)*i,np.float64))
    with pytest.raises(Exception,match='position'):captures[0].observe_value(4,10,np.zeros((2,4),np.float32))


def test_qualification_exact_binding_gate(tmp_path):
    import persona_batch_server as module
    e=engine(tmp_path);s=BatchEngine.__new__(BatchEngine);s.e=e;s.receipt=tmp_path/'qualification.json'
    assert s.capability()['qualified'] is False
    receipt={'schema':'character-creation.batch-service-qualification.v1','verified':True,'passed':True,'binding':binding(e),'server_sha256':module.probe.file_hash(module.__file__),'physical_pages':8,'page_tokens':128}
    s.receipt.write_text(json.dumps(receipt));assert s.capability()['qualified'] is True
    receipt['binding']['runtime_identity']='f'*64;s.receipt.write_text(json.dumps(receipt));assert s.capability()['qualified'] is False


def test_saved_per_job_response_passes_existing_checked(tmp_path):
    from persona_propagation import checked
    e=engine(tmp_path,64,5120);s=BatchEngine.__new__(BatchEngine);s.e=e
    p=payload();spec=__import__('persona_probe').normalize_request(p,e.model_id);h=canonical_digest(p)
    rc=BatchResponseCapture(e,spec['capture_response'],3,h);rc.observe_value(3,10,np.ones((64,5120),np.float32))
    job={'id':'d'*32,'spec':spec,'ids':[1,2,3],'generated':[10,99],'response_capture':rc,'positions':[0,1,2],
        'trace':[[1,1,0,0,-1,1,0,0],[1,2,3,1,10,1,0,1],[1,8,4,2,99,2,0,2]],'graph_kinds':['init','prefill','decode'],
        'result':{'finished':True,'finish_reason':'stop'},'norms':[np.ones((64,2),np.float32)],
        'prefill':{'residual_last':np.zeros((64,5120),np.float32),'pre_mlp_last':np.zeros((64,5120),np.float32),'mlp_input_last':np.zeros((64,17408),np.float32),'final_chunk_positions':np.array([0,1,2])},
        'artifact':tmp_path/'probe.npz','report':{'status':'running','request':p,'request_sha256':h,'started_at':0.,**binding(e),'batch':{'index':0,'requests':4}}}
    s.save_job(job);job['report']['state_restored']=True
    provenance=tmp_path/'probe.jsonl';provenance.write_text(json.dumps(job['report']))
    response=job['response'];response['provenance_artifact']={'path':str(provenance),'sha256':__import__('persona_probe').file_hash(provenance),'request_sha256':h,**binding(e)}
    class Store:
        def get(self,pointer):
            path=Path(pointer['path']);assert __import__('persona_probe').file_hash(path)==pointer['sha256'];return path
    arrays,paths=checked(response,p,binding(e),Store())
    assert arrays['response_residual_count']==1 and arrays['completion_trace'][-1,1]==8
