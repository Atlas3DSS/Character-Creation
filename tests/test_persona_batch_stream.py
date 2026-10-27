"""Pure CPU ownership tests: never load native decoder or submit work."""
import persona
import pytest
from persona_batch_stream import Slots,POLICY


def test_full_budget_allocation_and_capacity_backpressure():
    s=Slots(8)
    a=s.admit('a',5);b=s.admit('b',3)
    assert not set(a[1]['pages'])&set(b[1]['pages'])
    assert s.admit('c',1) is None and len(s.live)==2
    s.retire(a[0],'a',a[1]['generation'])
    c=s.admit('c',5)
    assert c[0]==a[0] and c[1]['generation']==2 and c[1]['pages']==a[1]['pages']
    assert b[1]==s.live[b[0]]


def test_stale_retirement_never_releases_new_owner_pages():
    s=Slots(4);slot,a=s.admit('a',2);s.retire(slot,'a',a['generation']);_,b=s.admit('b',2)
    before=set(s.free)
    with pytest.raises(ValueError,match='Stale'):s.retire(slot,'a',a['generation'])
    with pytest.raises(ValueError,match='Stale'):s.retire(slot,'b',a['generation'])
    assert s.free==before and s.live[slot]==b


def test_four_slot_limit_and_repeat_generation():
    s=Slots(20)
    for i in range(4):assert s.admit(str(i),1)[0]==i
    assert s.admit('fifth',1) is None
    for i in range(3):
        old=s.live[0];s.retire(0,old['id'],old['generation']);assert s.admit('next'+str(i),2)[0]==0
    assert s.epochs[0]==4 and len(s.history)==7


@pytest.mark.parametrize('count',[0,-1,True,1.5])
def test_invalid_pages_fail_before_mutation(count):
    s=Slots(4)
    with pytest.raises(ValueError):s.admit('x',count)
    assert s.free==set(range(4)) and s.live=={}


def test_duplicate_live_identity_fails_closed():
    s=Slots(4);s.admit('x',1)
    with pytest.raises(ValueError,match='Duplicate'):s.admit('x',1)
    assert len(s.live)==1


def test_finite_no_mtp_no_prefix_policy():
    assert POLICY['requests']==8 and POLICY['isolated_references']==8
    assert POLICY['deadline_seconds']==1200
    assert POLICY['prefix_cache'] is POLICY['mtp'] is POLICY['automatic_rollout'] is False


def test_actual_capture_mapping_rejects_stale_reused_generation():
    import numpy as np
    from persona_batch_capture import mapping_expected
    before=[[8,10,1,99,1,0,1],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
    ev={'slot_live':np.array([True,False,False,False]),'mapping_error':np.array([False]),'slot_positions':np.array([10,-1,-1,-1]),'slot_token_ids':np.array([99,-1,-1,-1]),'slot_generations':np.array([8,0,0,0]),'slot_match_counts':np.array([1,0,0,0])}
    assert mapping_expected(before,[0],ev)=={0}
    ev['slot_generations'][0]=7
    with pytest.raises(ValueError,match='generation'):mapping_expected(before,[0],ev)


def test_loaded_server_binding_survives_disk_restore_but_startup_rejects(monkeypatch):
    from pathlib import Path
    import persona_batch_stream as m
    from persona_vectors import canonical_digest
    digests={n:m.sha(Path(m.__file__).with_name(n)) for n in m.SOURCES}
    p={'schema':'character-creation.batch-stream-plan.v1','policy':m.POLICY,'jobs':[{'source_job':{'id':str(i)},'request':{'max_tokens':v}} for i,v in enumerate([1,8,64,128]*2)],'source_sha256':digests}
    p['input_sha256']=canonical_digest(p)
    original=m.sha
    monkeypatch.setattr(m,'sha',lambda path:'0'*64 if Path(path).name=='persona_batch_server.py' else original(path))
    with pytest.raises(ValueError,match='Source changed'):m.validate_plan(p)
    assert m.validate_plan(p,loaded_server_sha256=digests['persona_batch_server.py']) is p
    with pytest.raises(ValueError,match='Source changed'):m.validate_plan(p,loaded_server_sha256='f'*64)


def test_empty_pool_readmits_pending_after_all_current_owners_finish():
    from persona_batch_stream import check_empty_readmission
    s=Slots(4);slot,row=s.admit('first',4)
    assert s.admit('pending',4) is None
    s.retire(slot,'first',row['generation'])
    assert check_empty_readmission(s,4)
    assert s.admit('pending',4)[0]==slot
    with pytest.raises(ValueError):check_empty_readmission(Slots(4),5)


def payload(i):
    return {'metadata':{'job_id':str(i)},'capture_response':{'id':f'{i:032x}','layers':[],'max_positions':0}}


def test_stream_envelope_rejects_duplicates_missing_capture_and_overcapacity():
    from persona_batch_stream import stream_envelope
    assert len(stream_envelope({'batch_id':'x','requests':[payload(i) for i in range(16)]})['requests'])==16
    for rows in [[payload(1),payload(1)],[{}],[payload(i) for i in range(17)]]:
        with pytest.raises(ValueError):stream_envelope({'batch_id':'x','requests':rows})
    with pytest.raises(ValueError):stream_envelope({'batch_id':'../x','requests':[payload(1)]})


def fake_trial(tmp_path):
    from types import SimpleNamespace as NS
    e=NS(output=tmp_path,model=NS(identity='model',layers=64,hidden=5120,intermediate=17408),manifest={'runtime_identity':'runtime'},prepared=NS(server_config={'identity':'prepared'}),capacity=(4,8),page_tokens=256)
    def prepare(value):
        r=value['requests'][0]
        return ({'fits':True,'prompt_tokens':[100],'required_pages':[8]},[{'capture_response':r['capture_response'],'max_tokens':128}],[[0]*100],[],[])
    return NS(e=e,prepare=prepare)


def test_preflight_allows_queue_larger_than_simultaneous_pages_but_bounds_memory(tmp_path):
    from persona_batch_stream import stream_preflight
    t=fake_trial(tmp_path);v={'batch_id':'x','requests':[payload(i) for i in range(16)]}
    r=stream_preflight(t,v)
    assert r['fits'] and sum(r['required_pages'])>r['available_pages'] and r['max_active']==4
    for p in v['requests']:p['capture_response'].update(layers=list(range(64)),max_positions=128)
    with pytest.raises(ValueError,match='memory'):stream_preflight(t,v)


def test_capability_receipt_exact_loaded_sources_and_limits(tmp_path,monkeypatch):
    import json
    import persona_batch_stream as m
    from persona_batch_server import SERVER_SOURCE_SHA256
    t=fake_trial(tmp_path);path=tmp_path/'receipt.json'
    receipt={'schema':'character-creation.stream-service-qualification.v1','verified':True,'passed':True,'binding':m.binding(t.e),'stream_sha256':m.STREAM_SOURCE_SHA256,'server_sha256':SERVER_SOURCE_SHA256,'physical_pages':8,'page_tokens':256,'max_queued':16,'limits':m.STREAM_LIMITS}
    path.write_text(json.dumps(receipt));assert m.stream_capability(t.e,path)['qualified']
    monkeypatch.setattr(m,'sha',lambda path:'changed-on-disk')
    assert m.stream_capability(t.e,path)['qualified']
    for key in ('stream_sha256','server_sha256','binding','limits'):
        bad=dict(receipt);bad[key]='wrong';path.write_text(json.dumps(bad));assert not m.stream_capability(t.e,path)['qualified']


def test_service_durable_received_and_uncertain_no_retry(tmp_path,monkeypatch):
    import json
    import persona_batch_stream as m
    t=fake_trial(tmp_path);calls=[]
    def execute(rows,deadline):calls.append(rows);raise RuntimeError('uncertain GPU failure')
    t.execute=execute;s=m.StreamService(t.e,None,trial=t);monkeypatch.setattr(s,'capability',lambda:{'qualified':True})
    v={'batch_id':'attempt','requests':[payload(1)]}
    with pytest.raises(RuntimeError):s.complete(v)
    with pytest.raises(ValueError,match='no retry'):s.complete(v)
    s2=m.StreamService(t.e,None,trial=t);monkeypatch.setattr(s2,'capability',lambda:{'qualified':True})
    with pytest.raises(ValueError,match='no retry'):s2.complete(v)
    assert len(calls)==1
    events=[json.loads(x) for x in next(tmp_path.glob('stream-*.jsonl')).read_text().splitlines()]
    assert [e['state'] for e in events]==['submitting','uncertain_no_retry']


def test_service_missing_response_fails_no_silent_drop(tmp_path,monkeypatch):
    import persona_batch_stream as m
    t=fake_trial(tmp_path);t.execute=lambda rows,deadline:({},{});s=m.StreamService(t.e,None,trial=t);monkeypatch.setattr(s,'capability',lambda:{'qualified':True})
    with pytest.raises(ValueError,match='Missing/extra'):s.complete({'batch_id':'x','requests':[payload(1)]})
    assert '"state": "received"' in next(tmp_path.glob('stream-*.jsonl')).read_text()


def test_busy_rejects_before_creating_intent(tmp_path,monkeypatch):
    import persona_batch_stream as m
    t=fake_trial(tmp_path);s=m.StreamService(t.e,None,trial=t);s.lock.acquire()
    try:
        with pytest.raises(ValueError,match='busy'):s.complete({'batch_id':'x','requests':[payload(1)]})
    finally:s.lock.release()
    assert list(tmp_path.iterdir())==[]


def test_service_validates_every_response_preserves_order_and_truncation(tmp_path,monkeypatch):
    import json
    import persona_batch_stream as m
    t=fake_trial(tmp_path);seen=[]
    def execute(rows,deadline):return ({m.canonical_digest(r):{'original':r['metadata']['job_id'],'complete':False} for r in reversed(rows)},{'active':4})
    def checked(response,request,bound,store):seen.append(request['metadata']['job_id']);return {'evidence':{'prose_complete':False}}
    t.execute=execute;s=m.StreamService(t.e,None,trial=t);monkeypatch.setattr(s,'capability',lambda:{'qualified':True});monkeypatch.setattr(m,'checked_parity',checked)
    rows=[payload(i) for i in range(8)];out=s.complete({'batch_id':'x','requests':rows})
    assert seen==[str(i) for i in range(8)]
    assert [r['response']['original'] for r in out['results']]==seen
    assert all(r['response']['complete'] is False for r in out['results'])
    events=[json.loads(x) for x in next(tmp_path.glob('stream-*.jsonl')).read_text().splitlines()]
    assert [e['state'] for e in events]==['submitting','received','validated']
    assert events[0]['envelope']['requests']==rows


def test_received_bad_capture_is_terminal_uncertain_not_success(tmp_path,monkeypatch):
    import persona_batch_stream as m
    t=fake_trial(tmp_path);t.execute=lambda rows,deadline:({m.canonical_digest(r):{} for r in rows},{})
    s=m.StreamService(t.e,None,trial=t);monkeypatch.setattr(s,'capability',lambda:{'qualified':True})
    def bad(*args):raise ValueError('capture hash mismatch')
    monkeypatch.setattr(m,'checked_parity',bad)
    with pytest.raises(ValueError,match='capture hash'):s.complete({'batch_id':'x','requests':[payload(1)]})
    log=next(tmp_path.glob('stream-*.jsonl')).read_text()
    assert 'uncertain_no_retry' in log and '"state": "validated"' not in log
