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
