"""CPU-only native scheduler safety tests; no inference or external services."""
import persona
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from persona_manager import Queue
from persona_batch_dispatch import authorize_direction,batch_eligible
from persona_propagation import file_sha


def queue_fixture(tmp_path,n=12):
    queue=Queue(tmp_path/'queue.sqlite')
    with queue.connect() as db:
        for i in range(n):
            job={'id':str(i)};db.execute("INSERT INTO jobs(id,created,state,body) VALUES (?,0,'queued',?)",(job['id'],json.dumps(job)))
    return queue


def test_atomic_four_per_worker_claims_no_duplicates(tmp_path):
    queue=queue_fixture(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:groups=list(pool.map(lambda i:queue.claim_many(str(i),4),range(4)))
    rows=[job['id'] for group in groups for job,attempt in group]
    assert len(rows)==len(set(rows))==12 and all(len(group)<=4 for group in groups)


def test_only_unsubmitted_preflight_leftovers_can_return(tmp_path):
    queue=queue_fixture(tmp_path,4);claims=queue.claim_many('endpoint',4)
    queue.audit('native_batch_submitting',{'batch_id':'a'*32,'attempts':[{'job_id':claims[0][0]['id'],'number':claims[0][1]}]})
    with pytest.raises(ValueError):queue.release_unsubmitted(claims)
    assert queue.counts()=={'running':4}  # entire release transaction rolled back
    queue.release_unsubmitted(claims[1:]);assert queue.counts()=={'queued':3,'running':1}
    assert queue.recover_native_batch()==1
    assert queue.counts()=={'queued':3,'failed':1}
    assert len(queue.claim_many('endpoint',4))==3


@pytest.mark.parametrize('field',['kv_control','capture_search','trace','capture_prefill','observation_only'])
def test_unsupported_controls_not_silently_batched(field):
    request={'stream':False,field:{'sentinel':'preserved'}}
    assert not batch_eligible(request) and field in request


def receipt_fixture(tmp_path):
    old={k:ch*64 for k,ch in [('model_identity','a'),('runtime_identity','b'),('prepared_identity','c')]}
    new=dict(old,runtime_identity='d'*64,prepared_identity='e'*64)
    receipt=dict(schema='character-creation.batch-runtime-compatibility.v1',state='complete',passed=True,old_binding=old,new_binding=new,artifact_sha256='f'*64,
        allowed_program_sha256=['1'*64],old_core_source_hashes={'core':'2'*64},new_core_source_hashes={'core':'3'*64},qualification_sha256='4'*64,
        parity=dict(programs_exact=True,response_tokens_exact=True,response_capture_exact=True))
    path=tmp_path/'compat.json';path.write_text(json.dumps(receipt));spec={'path':str(path),'sha256':file_sha(path)}
    return receipt,path,spec,{'direction_binding':old,'program_sha256':'1'*64},[{'kind':'direction','sha256':'f'*64}],new


def test_exact_compatibility_required_without_artifact_rebinding(tmp_path):
    receipt,path,spec,proof,pushes,target=receipt_fixture(tmp_path)
    original=path.read_bytes()
    assert authorize_direction(proof,pushes,target,spec)==spec['sha256'];assert path.read_bytes()==original
    with pytest.raises(ValueError):authorize_direction(proof,pushes,target)
    with pytest.raises(ValueError):authorize_direction(dict(proof,program_sha256='0'*64),pushes,target,spec)
    with pytest.raises(ValueError):authorize_direction(proof,[{'kind':'direction','sha256':'0'*64}],target,spec)
    with pytest.raises(ValueError):authorize_direction(proof,pushes,dict(target,model_identity='0'*64),spec)
    receipt['parity']['response_tokens_exact']=False;path.write_text(json.dumps(receipt));spec['sha256']=file_sha(path)
    with pytest.raises(ValueError):authorize_direction(proof,pushes,target,spec)


def dispatch_fixture(tmp_path,n=4):
    from persona_vectors import canonical_digest
    queue=queue_fixture(tmp_path,n);claims=queue.claim_many('endpoint',4)
    entries=[{'job':dict(j,pushes=[]),'number':number,'request':{'model':'checkpoint','messages':[{'role':'user','content':j['id']}],'max_tokens':32,'stream':False},'provenance':{'program_sha256':'a'*64}} for j,number in claims]
    binding={k:ch*64 for k,ch in [('model_identity','a'),('runtime_identity','b'),('prepared_identity','c')]}
    cap={'supported':True,'qualified':True,'max_batch_size':4,'physical_pages':128,'page_tokens':16,'preflight_path':'/v1/batch/preflight','completions_path':'/v1/batch/completions'}
    return queue,entries,dict(binding,url='http://127.0.0.1:8935'),{'data':[{'id':'checkpoint','flare':dict(binding,batch=cap)}]}


def test_one_native_batch_post_then_every_original_response_validated(tmp_path):
    from persona_batch_dispatch import dispatch
    from persona_vectors import canonical_digest
    import time
    queue,entries,endpoint,descriptor=dispatch_fixture(tmp_path);calls=[];validated=[]
    def fetch(url,payload=None,timeout=0):
        calls.append((url,payload))
        if url.endswith('/v1/models'):return descriptor
        if url.endswith('/preflight'):return {'batch_id':payload['batch_id'],'binding':{k:endpoint[k] for k in ('model_identity','runtime_identity','prepared_identity')},'fits':True,'reason':None,'prompt_tokens':[8]*4,'required_pages':[3]*4,'available_pages':128,'request_sha256':[canonical_digest(r) for r in payload['requests']]}
        assert url.endswith('/completions') and len(payload['requests'])==4
        return {'batch_id':payload['batch_id'],'results':[{'request_sha256':canonical_digest(r),'response':{'activation_artifact':{'sha256':str(i)},'provenance_artifact':{},'index':i}} for i,r in enumerate(payload['requests'])]}
    def check(response,request,binding,store):validated.append((response['index'],request));return {},{}
    outcomes=dispatch(entries,endpoint,queue,None,fetch=fetch,deadline=time.time()+10,admit=lambda:True,validator=check)
    assert len(outcomes)==4 and all(r['complete'] for e,r in outcomes)
    assert [r for i,r in validated]==[e['request'] for e in entries]
    assert len([url for url,p in calls if url.endswith('/completions')])==1
    for entry,result in outcomes:queue.finish(entry['job'],entry['number'],result)
    assert queue.counts()=={'complete':4}


def test_uncertain_batch_is_not_retried_or_released(tmp_path):
    from persona_batch_dispatch import dispatch
    from persona_vectors import canonical_digest
    import time
    queue,entries,endpoint,descriptor=dispatch_fixture(tmp_path);posts=[]
    def fetch(url,payload=None,timeout=0):
        if url.endswith('/models'):return descriptor
        if url.endswith('/preflight'):return {'batch_id':payload['batch_id'],'binding':{k:endpoint[k] for k in ('model_identity','runtime_identity','prepared_identity')},'fits':True,'reason':None,'prompt_tokens':[8]*4,'required_pages':[3]*4,'available_pages':128,'request_sha256':[canonical_digest(r) for r in payload['requests']]}
        posts.append(payload);raise TimeoutError('remote completion unknown')
    outcomes=dispatch(entries,endpoint,queue,None,fetch=fetch,deadline=time.time()+10,admit=lambda:True)
    assert len(posts)==1 and len(outcomes)==4
    assert all(r['error_class']=='uncertain_no_retry' for e,r in outcomes)
    with pytest.raises(ValueError):queue.release_unsubmitted([(e['job'],e['number']) for e in entries])
    for entry,result in outcomes:queue.finish(entry['job'],entry['number'],result)
    assert queue.counts()=={'failed':4};assert queue.claim_many('endpoint',4)==[]


def test_preflight_full_budget_and_false_fit_rejected(tmp_path):
    from persona_batch_dispatch import checked_preflight
    from persona_vectors import canonical_digest
    queue,entries,endpoint,descriptor=dispatch_fixture(tmp_path)
    requests=[e['request'] for e in entries];envelope={'batch_id':'x','requests':requests};binding={k:endpoint[k] for k in ('model_identity','runtime_identity','prepared_identity')}
    result={'batch_id':'x','binding':binding,'fits':True,'reason':None,'prompt_tokens':[8]*4,'required_pages':[1]*4,'available_pages':128,'request_sha256':[canonical_digest(r) for r in requests]}
    cap=descriptor['data'][0]['flare']['batch']
    with pytest.raises(ValueError):checked_preflight(result,envelope,binding,cap)
    result['required_pages']=[3]*4;result['available_pages']=11
    with pytest.raises(ValueError):checked_preflight(result,envelope,binding,cap)


def test_capacity_shrink_releases_only_unsubmitted_jobs(tmp_path):
    from persona_batch_dispatch import dispatch
    from persona_vectors import canonical_digest
    import time
    queue,entries,endpoint,descriptor=dispatch_fixture(tmp_path);submitted=[]
    def fetch(url,payload=None,timeout=0):
        if url.endswith('/models'):return descriptor
        requests=payload['requests'];n=len(requests)
        if url.endswith('/preflight'):return {'batch_id':payload['batch_id'],'binding':{k:endpoint[k] for k in ('model_identity','runtime_identity','prepared_identity')},'fits':n<=2,'reason':None if n<=2 else 'capacity','prompt_tokens':[8]*n,'required_pages':[3]*n,'available_pages':6,'request_sha256':[canonical_digest(r) for r in requests]}
        submitted.append(requests)
        return {'batch_id':payload['batch_id'],'results':[{'request_sha256':canonical_digest(r),'response':{'activation_artifact':{},'provenance_artifact':{}}} for r in requests]}
    outcomes=dispatch(entries,endpoint,queue,None,fetch=fetch,deadline=time.time()+10,admit=lambda:True,validator=lambda *a:({},{}))
    assert len(submitted)==1 and len(submitted[0])==2 and len(outcomes)==2
    assert queue.counts()=={'running':2,'queued':2}
    assert len(queue.claim_many('endpoint',4))==2


def test_trace_fallback_keeps_exact_payload_on_same_endpoint(tmp_path):
    from persona_batch_dispatch import dispatch
    import time
    queue,entries,endpoint,descriptor=dispatch_fixture(tmp_path)
    entries[0]['request']['capture_trace']={'id':'trace','neurons':[]}
    descriptor['data'][0]['flare']['token_trace']={'supported':True};posts=[]
    def fetch(url,payload=None,timeout=0):
        if url.endswith('/models'):return descriptor
        assert url==endpoint['url']+'/v1/chat/completions';posts.append(payload)
        return {'activation_artifact':{},'provenance_artifact':{}}
    outcomes=dispatch(entries,endpoint,queue,None,fetch=fetch,deadline=time.time()+10,admit=lambda:True,validator=lambda *a:({},{}))
    assert posts==[entries[0]['request']] and len(outcomes)==1
    assert queue.counts()=={'running':1,'queued':3}


def test_unqualified_capability_never_uses_batch_route(tmp_path):
    from persona_batch_dispatch import dispatch
    import time
    queue,entries,endpoint,descriptor=dispatch_fixture(tmp_path)
    descriptor['data'][0]['flare']['batch']['qualified']=False;posts=[]
    def fetch(url,payload=None,timeout=0):
        if url.endswith('/models'):return descriptor
        assert url.endswith('/v1/chat/completions');posts.append(payload)
        return {'activation_artifact':{},'provenance_artifact':{}}
    outcomes=dispatch(entries,endpoint,queue,None,fetch=fetch,deadline=time.time()+10,admit=lambda:True,validator=lambda *a:({},{}))
    assert len(posts)==len(outcomes)==1


def test_primary_manager_worker_uses_native_dispatch_and_completes_each(tmp_path,monkeypatch):
    import threading,time
    from types import SimpleNamespace
    from persona_manager import Manager
    queue=queue_fixture(tmp_path,4)
    manager=Manager.__new__(Manager);manager.native_batch=True;manager.queue=queue
    manager.stop=threading.Event();manager.model=object();manager.bank=SimpleNamespace(root=tmp_path);manager.allow_unmapped=False
    manager.config={'native_batch':{'enabled':True,'artifact_directory':str(tmp_path),'artifact_remote_host':'v100','artifact_remote_root':'/approved'},'request_timeout_seconds':30}
    endpoint={'url':'http://127.0.0.1:8935','lease_id':'lease',**{k:'a'*64 for k in ('model_identity','runtime_identity','prepared_identity')}}
    monkeypatch.setattr('persona_manager.fetch',lambda *a,**k:{'data':[{'flare':{**{k:endpoint[k] for k in ('model_identity','runtime_identity','prepared_identity')},'batch':{'qualified':True,'supported':True,'max_batch_size':4}}}]})
    manager.admission=SimpleNamespace(allowed=lambda e:(True,''),board={'jobs':[{'id':'lease','lease_expires_epoch':time.time()+1000}]})
    monkeypatch.setattr('persona_manager.VectorBank',lambda *a,**k:object())
    monkeypatch.setattr('persona_manager.prepare_job',lambda job,*a:({'model':'checkpoint','stream':False},{'sites':[]},{'program_sha256':'a'*64},'rollout'))
    calls=[]
    def dispatch(entries,endpoint,queue,store,**kwargs):
        assert kwargs['admit']() and len(entries)==4
        assert all(e['request']['capture_response']['layers']==[] for e in entries)
        calls.append(entries);manager.stop.set()
        return [(e,{'complete':True,'error':None,'batch_id':'example'}) for e in entries]
    monkeypatch.setattr('persona_batch_dispatch.dispatch',dispatch)
    manager.worker(endpoint,time.time()+10)
    assert len(calls)==1 and queue.counts()=={'complete':4}


@pytest.mark.parametrize('unready',['transport','empty','unqualified','503'])
def test_warming_endpoint_does_not_claim_any_jobs(tmp_path,monkeypatch,unready):
    import time,urllib.error
    from types import SimpleNamespace
    from persona_manager import Manager
    queue=queue_fixture(tmp_path,4);manager=Manager.__new__(Manager)
    class Stop:
        stopped=False
        waits=[]
        def is_set(self):return self.stopped
        def wait(self,seconds):self.waits.append(seconds);self.stopped=True
    manager.stop=Stop();manager.native_batch=True;manager.queue=queue;manager.model=object();manager.bank=SimpleNamespace(root=tmp_path);manager.allow_unmapped=False
    manager.config={'native_batch':{'enabled':True,'artifact_directory':str(tmp_path),'artifact_remote_host':'v100','artifact_remote_root':'/approved'}}
    endpoint={'url':'http://127.0.0.1:8935','lease_id':'lease',**{k:'a'*64 for k in ('model_identity','runtime_identity','prepared_identity')}}
    manager.admission=SimpleNamespace(allowed=lambda e:(True,''),board={'jobs':[{'id':'lease','lease_expires_epoch':time.time()+1000}]})
    monkeypatch.setattr('persona_manager.VectorBank',lambda *a,**k:object())
    def fetch(*a,**kw):
        if unready=='transport':raise ConnectionResetError('warming')
        if unready=='503':raise urllib.error.HTTPError(endpoint['url'],503,'warming',{},None)
        if unready=='empty':return {'data':[]}
        return {'data':[{'flare':{**{k:endpoint[k] for k in ('model_identity','runtime_identity','prepared_identity')},'batch':{'qualified':False}}}]}
    monkeypatch.setattr('persona_manager.fetch',fetch)
    def must_not_claim(*args):raise AssertionError('Warming endpoint claimed work')
    monkeypatch.setattr(queue,'claim_many',must_not_claim)
    manager.batch_worker(endpoint,time.time()+10)
    assert queue.counts()=={'queued':4} and manager.stop.waits==[2]
    with queue.connect() as db:assert db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]==0


@pytest.mark.parametrize('field',['model_identity','runtime_identity','prepared_identity'])
def test_readiness_wrong_identity_fails_closed(monkeypatch,field):
    from persona_manager import native_batch_ready
    endpoint={'url':'http://127.0.0.1:8935',**{k:'a'*64 for k in ('model_identity','runtime_identity','prepared_identity')}}
    flare={**{k:endpoint[k] for k in ('model_identity','runtime_identity','prepared_identity')},'batch':{'qualified':True,'supported':True,'max_batch_size':4}};flare[field]='b'*64
    monkeypatch.setattr('persona_manager.fetch',lambda *a,**k:{'data':[{'flare':flare}]})
    with pytest.raises(RuntimeError,match='identity mismatch: '+field):native_batch_ready(endpoint)


def test_fixed_job_budget_waits_without_calling_planner(tmp_path,monkeypatch):
    import time
    from types import SimpleNamespace
    from persona_manager import Manager
    queue=queue_fixture(tmp_path,4);manager=Manager.__new__(Manager)
    class Stop:
        stopped=False
        waits=[]
        def is_set(self):return self.stopped
        def set(self):self.stopped=True
        def wait(self,seconds):self.waits.append(seconds);self.stopped=True
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,*args):return SimpleNamespace(done=lambda:False)
    manager.stop=Stop();manager.stop_signal=None;manager.queue=queue;manager.native_batch=True
    manager.model=SimpleNamespace(identity='a'*64);manager.runtime_identity='b'*64
    manager.endpoints=[{'prepared_identity':'c'*64}];manager.concurrency=1
    manager.config={'max_total_jobs':4}
    monkeypatch.setattr('persona_manager.ThreadPoolExecutor',Pool)
    monkeypatch.setattr('persona_manager.signal.signal',lambda *a:None)
    def forbidden(*args):raise AssertionError('Fixed full queue must not call planner')
    manager.propose=forbidden
    manager.run(time.time()+10)
    assert manager.stop.waits==[.2] and queue.counts()=={'queued':4}


@pytest.mark.parametrize('corrupt',[False,True])
def test_real_length_limited_native_artifacts_are_incomplete_not_failed(tmp_path,monkeypatch,corrupt):
    import time
    import test_persona_batch_qualify as qualification
    from persona_batch_dispatch import dispatch
    from persona_vectors import canonical_digest
    original_payload=qualification.payload
    monkeypatch.setattr(qualification,'payload',lambda:dict(original_payload(),stream=False))
    response,request,bound,store,job=qualification.fixture(tmp_path)
    if corrupt:job['artifact'].write_bytes(job['artifact'].read_bytes()+b'corrupt capture bytes')
    queue=queue_fixture(tmp_path,1);claimed,number=queue.claim_many('endpoint',1)[0]
    entries=[{'job':dict(claimed,pushes=[]),'number':number,'request':request,'provenance':{'program_sha256':canonical_digest({'phase':'all','sites':[]})}}]
    endpoint=dict(bound,url='http://127.0.0.1:8935')
    def fetch(url,payload=None,timeout=0):
        if url.endswith('/models'):return {'data':[{'id':'checkpoint','flare':dict(bound,batch={'supported':True,'qualified':True,'max_batch_size':4,'physical_pages':8,'page_tokens':128,'preflight_path':'/v1/batch/preflight','completions_path':'/v1/batch/completions'})}]}
        if url.endswith('/preflight'):return {'batch_id':payload['batch_id'],'binding':bound,'request_sha256':[canonical_digest(request)],'fits':True,'reason':None,'prompt_tokens':[3],'required_pages':[1],'available_pages':8}
        return {'batch_id':payload['batch_id'],'results':[{'request_sha256':canonical_digest(request),'response':response}]}
    outcomes=dispatch(entries,endpoint,queue,store,fetch=fetch,deadline=time.time()+30,admit=lambda:True)
    entry,result=outcomes[0];queue.finish(entry['job'],entry['number'],result,retry=False)
    assert result['complete'] is False
    if corrupt:
        assert result['error'] and result['error_class']=='artifact_validation_no_retry'
        assert queue.counts()=={'failed':1}
    else:
        assert result['error'] is None and result['generation_status']=='length_limited' and result['capture_validated'] is True
        assert result['request']==request and result['response']==response
        assert queue.counts()=={'incomplete':1}
        with queue.connect() as db:events=[json.loads(row[0]) for row in db.execute("SELECT data FROM audit WHERE event='native_job_validated'")]
        assert len(events)==1 and events[0]['complete_answer'] is False
        from persona_propagation import checked
        with pytest.raises(ValueError,match='answer did not complete'):checked(response,request,bound,store)


def test_native_infrastructure_budget_excludes_validated_scientific_incomplete():
    from persona_manager import request_error_budget_exceeded
    assert not request_error_budget_exceeded({'complete':39,'incomplete':9},{},native_batch=True)
    assert request_error_budget_exceeded({'complete':39,'failed':9},{},native_batch=True)
    assert request_error_budget_exceeded({'complete':39,'incomplete':9},{},native_batch=False)
    assert request_error_budget_exceeded({'failed':2,'complete':100},{'error_budget':2},native_batch=True)
    assert request_error_budget_exceeded({'complete':15,'incomplete':2,'failed':4},{'max_error_fraction':.15},native_batch=True)


@pytest.mark.parametrize('mutation',['thinking','under_budget','wrong_diag','not_native','nonordinary'])
def test_length_exception_gate_cannot_bypass_normal_validator(monkeypatch,mutation):
    from persona_batch_dispatch import validate_generation
    request={'stream':False,'max_tokens':2048,'capture_response':{'id':'a'*32}}
    response={'choices':[{'finish_reason':'length'}],'diagnostics':{'complete_final_answer':False,'unexpected_thinking':False},'usage':{'completion_tokens':2048}}
    native=True
    if mutation=='thinking':response['diagnostics']['unexpected_thinking']=True
    elif mutation=='under_budget':response['usage']['completion_tokens']=2047
    elif mutation=='wrong_diag':response['diagnostics']['complete_final_answer']=True
    elif mutation=='not_native':native=False
    else:request['capture_trace']={}
    monkeypatch.setattr('persona_batch_qualify.checked_parity',lambda *a:pytest.fail('Length-only exception must not apply'))
    def strict(*args):raise ValueError('ordinary strict validator rejected')
    with pytest.raises(ValueError,match='strict validator'):validate_generation(response,request,{},None,native_batch=native,validator=strict)
