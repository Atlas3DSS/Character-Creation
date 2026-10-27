import json
from types import SimpleNamespace
import pytest
from persona_publication_recovery import Recovery,REPORTS,GUIDES,due

class FakeHub:
    def __init__(self):self.current='p0';self.data={'p0':{}};self.calls=[];self.fail=None;self.read_fail=False
    def head(self):return self.current
    def history(self,head,parent):return list(reversed(list(self.data)))[0:list(reversed(list(self.data))).index(parent)]
    def read(self,n,oid,missing=False):
        if self.read_fail:raise TimeoutError()
        return self.data[oid].get(n)
    def commit(self,files,parent,message):
        self.calls.append((dict(files),parent))
        if parent!=self.current:
            e=RuntimeError();e.response=SimpleNamespace(status_code=412);raise e
        if self.fail=='before':self.fail=None;raise TimeoutError()
        self.current='p'+str(len(self.data));self.data[self.current]=dict(files)
        if self.fail=='after':self.fail=None;raise TimeoutError()
        return self.current

def setup(tmp_path):
    hub=FakeHub();r=Recovery(tmp_path/'receipt.sqlite',hub,{'old_receipt_sha':'x'})
    files={n:n.encode() for n in (*REPORTS,*GUIDES)};r.freeze(files,{'terminal':48});return r,hub,files

def test_success_unknown_reconciled_without_second_post(tmp_path):
    r,h,f=setup(tmp_path);h.fail='after';assert not r.step();assert not r.step();assert r.step();assert len(h.calls)==1

def test_unknown_retries_identical_parent_and_bytes(tmp_path):
    r,h,f=setup(tmp_path);h.fail='before';r.step();r.step();assert h.calls==[(f,'p0'),(f,'p0')];assert r.step()

def test_412_only_refreshes_parent(tmp_path):
    r,h,f=setup(tmp_path);h.current='other';h.data['other']={};r.step();assert h.calls[0][1]=='p0';r.step();assert h.calls[1][1]=='other';assert r.step()

def test_source_mutation_after_freeze_cannot_change_payload(tmp_path):
    r,h,f=setup(tmp_path);f['README_human.md']=b'changed';r.step();assert h.calls[0][0]['README_human.md']==b'README_human.md'

def test_known_oid_verification_failure_never_reposts(tmp_path):
    r,h,f=setup(tmp_path);r.step();h.read_fail=True;r.step();r.step();assert len(h.calls)==1;h.read_fail=False;assert r.step()

def test_restart_retains_unknown_and_immutable_payload(tmp_path):
    r,h,f=setup(tmp_path);h.fail='after';r.step();r.db.close();r=Recovery(tmp_path/'receipt.sqlite',h,{'old_receipt_sha':'x'});r.step();assert r.step();assert len(h.calls)==1

def test_cadence_zero_and_review_evaluation_changes():
    def m(n=0,r=0,e='x'):return {k:{'terminal':n,'reviews':r,'evaluations':e,'final':False} for k in REPORTS}
    assert not due(None,m());assert due(None,m(48));assert not due(m(48),m(60));assert due(m(48),m(96));assert due(m(48),m(48,8));assert due(m(48),m(48,0,'changed'))

def test_allowlist_and_lineage_fail_closed(tmp_path):
    r,h,f=setup(tmp_path)
    with pytest.raises(ValueError):Recovery(tmp_path/'receipt.sqlite',h,{'different':True})
    r.step();r.step()
    with pytest.raises(ValueError):r.freeze({'private.json':b'x'}, {})


def test_unknown_success_then_other_writer_changes_guide(tmp_path):
    r,h,f=setup(tmp_path);h.fail='after';r.step();committed=h.current
    h.current='later';h.data['later']={**f,'README_human.md':b'another writer'}
    r.step();assert r.pending()[2]==committed;assert r.step();assert len(h.calls)==1
