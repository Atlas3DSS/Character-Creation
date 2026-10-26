from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'persona'))
import persona_literature_embedding_lease as lease


def test_exact_command_and_core_reserve(monkeypatch):
    assert lease.command()[-5:] == ['--build','--device','cuda','--batch-size','256']
    monkeypatch.setattr(lease.os, 'sched_getaffinity', lambda _: set(range(32)))
    original = lease.Path.read_text
    monkeypatch.setattr(lease.Path, 'read_text', lambda self,*a,**kw: '0,16' if str(self).endswith('thread_siblings_list') else original(self,*a,**kw))
    affinity = lease.cpu_affinity()
    assert len(affinity) == 30 and 0 not in affinity and 16 not in affinity


def test_release_binds_exact_receipt_and_never_other_gpu(monkeypatch):
    receipt = {'id':'owned','project':'character-creation','label':'literature-bge-embeddings','target_host':'wtatum84','gpu_selector':lease.GPU}
    calls = []
    current = {**receipt,'status':'running','assigned_uuid':lease.GPU}
    def wb(*args,**kwargs):
        calls.append(args)
        return {'jobs':[current]} if args[0]=='status' else {}
    monkeypatch.setattr(lease,'wb',wb)
    lease.release('owned',receipt,True,'checkpointed')
    assert calls[-1][:4] == ('finish','owned','--outcome','released')
    current['assigned_uuid']='other'
    with pytest.raises(RuntimeError,match='Unexpected'):
        lease.release('owned',receipt,True,'completed')
    assert sum(c[0]=='finish' for c in calls)==1
    current.update(assigned_uuid=lease.GPU,status='queued')
    lease.release('owned',receipt,False,'failed')
    assert calls[-1]==('cancel','owned')
