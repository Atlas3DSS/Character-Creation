from copy import deepcopy
import json
import pytest
import persona_adaptive_publication as p

H='a'*64

def fixture():
    c={'target':'target','source_recipe_id':'source','promotion_eligible':False,'pushes':[],'program_sha256':H};c['id']=p.canonical(c)
    menu={'binding':{'model_identity':H,'runtime_identity':H,'prepared_identity':H},'candidates':[c],'excluded_gpu':'private'};menu['menu_sha256']=p.canonical(menu)
    source={'controller':H,'trial_store':H,'identity_fixture':p.IDENTITY_SHA,'corpus':p.CORPUS_SHA}
    projection={k:v for k,v in menu.items() if k!='excluded_gpu'}
    report={'schema':'character-creation.adaptive-public.v1','binding':menu['binding'],'menu':projection,'menu_sha256':menu['menu_sha256'],'menu_projection_sha256':p.canonical(projection),
        'counts':{},'total_jobs':0,'terminal_jobs':0,'final':False,'state':'running','rows':[],'proposals':[],'reviews':[],
        'source_hashes':source,'promotion_eligible':False,'scope':'Discovery only'}
    return report,menu,source

def raw(v):
    v=deepcopy(v);v.pop('revision_sha256',None);v['revision_sha256']=p.canonical(v);return json.dumps(v).encode()

def test_empty_reports_and_zero_answer_proposal_allowed():
    v,m,s=fixture();p.validate(raw(v),{},m,{},s)
    v['proposals']=[{'id':H,'candidate_ids':[m['candidates'][0]['id']],'rationale':'Test fixed choice','state':'planned','evidence_sha256':H,'menu_sha256':m['menu_sha256']}]
    p.validate(raw(v),{},m,{},s)
    assert p.due(None,{'terminal':0,'events':0,'proposals':1,'final':False,'sha256':H})

@pytest.mark.parametrize('change',[lambda v:v.update(promotion_eligible=True),lambda v:v.update(private_path='/home/private'),lambda v:v['menu'].update(arbitrary='unbound')])
def test_false_promotion_private_fields_or_menu_edits_block(change):
    v,m,s=fixture();change(v)
    with pytest.raises(ValueError):p.validate(raw(v),{},m,{},s)

def test_unknown_or_historical_adaptive_proposal_blocked():
    v,m,s=fixture();v['proposals']=[{'id':H,'candidate_ids':['b'*64],'rationale':'Unknown','state':'planned','evidence_sha256':H,'menu_sha256':m['menu_sha256']}]
    with pytest.raises(ValueError,match='candidate'):p.validate(raw(v),{},m,{},s)

def test_cadence_sixteen_answers_four_events_or_final():
    previous={'terminal':16,'events':4,'proposals':1,'final':False,'sha256':'old'}
    current={**previous,'sha256':'new','terminal':31,'events':7}
    assert not p.due(previous,current)
    assert p.due(previous,{**current,'terminal':32})
    assert p.due(previous,{**current,'events':8})
    assert p.due(previous,{**current,'final':True})
    with pytest.raises(ValueError):p.due(previous,{**current,'terminal':0})

class Hub:
    def __init__(self):self.oid='old';self.files={};self.calls=0;self.fail=True
    def head(self):return self.oid
    def history(self,head,parent):return [head]
    def read(self,name,oid,missing=False):return self.files.get(name)
    def commit(self,files,parent,message):
        self.calls+=1;assert parent=='old';self.oid='new';self.files=dict(files)
        if self.fail:self.fail=False;raise TimeoutError()
        return self.oid

def test_two_file_snapshot_immutable_and_unknown_commit_no_duplicate(tmp_path,monkeypatch):
    fixture=b'unchanged identity';monkeypatch.setattr(p,'IDENTITY_SHA',p.digest(fixture));h=Hub();o=p.Observer(tmp_path/'receipt.sqlite',h,{'source':'test'})
    files={p.REPORT:b'report',p.IDENTITY:fixture};o.freeze(files,{'terminal':16});files[p.REPORT]=b'mutated'
    assert not o.step();assert not o.step();assert o.step();assert h.calls==1 and h.files[p.REPORT]==b'report'
    with pytest.raises(ValueError):o.freeze({p.REPORT:b'x','private.json':fixture},{})

def test_durable_allowlist_checked_before_recovery(tmp_path,monkeypatch):
    fixture=b'fixture';monkeypatch.setattr(p,'IDENTITY_SHA',p.digest(fixture));h=Hub();o=p.Observer(tmp_path/'receipt.sqlite',h,{})
    o.freeze({p.REPORT:b'x',p.IDENTITY:fixture},{})
    with o.db:o.db.execute("UPDATE files SET name='private.json' WHERE name=?",(p.REPORT,))
    with pytest.raises(ValueError):o.step()
    assert h.calls==0


def terminal_fixture():
    v,m,s=fixture();c=m['candidates'][0]
    job={'candidate_id':c['id'],'menu_sha256':m['menu_sha256'],'seed':80711,'arm':'baseline','phase':'all','pushes':[],'program_sha256':H,'pair_id':'pair','identity_probe':False,'prompt':{'prompt':'Fresh source','source_row_id':'source'}}
    job['id']=p.canonical(job)
    row={'job_id':job['id'],'candidate_id':c['id'],'target':c['target'],'source_recipe_id':c['source_recipe_id'],
        **{k:job[k] for k in ('seed','arm','phase','pushes','program_sha256','pair_id','identity_probe')},'state':'failed','split':'discovery','source_id':'source','prompt':'Fresh source',
        'source_sha256':p.CORPUS_SHA,'prompt_sha256':p.digest(b'Fresh source'),'complete':False,'capture_validated':False,'answer':None}
    v.update(rows=[row],counts={'failed':1},total_jobs=1,terminal_jobs=1)
    return v,m,s,{job['id']:job}

def test_source_text_and_authoritative_control_identity_fail_closed():
    v,m,s,jobs=terminal_fixture();p.validate(raw(v),{'source':'Fresh source'},m,jobs,s)
    with pytest.raises(ValueError,match='source text'):p.validate(raw(v),{'heldout':'Fresh source'},m,jobs,s)
    changed=deepcopy(v);changed['rows'][0]['program_sha256']='b'*64
    with pytest.raises(ValueError,match='Scientific job'):p.validate(raw(changed),{'source':'Fresh source'},m,jobs,s)
    changed=deepcopy(v);changed['rows'][0].update(complete=True,capture_validated=True,answer='fabricated')
    with pytest.raises(ValueError,match='complete-answer'):p.validate(raw(changed),{'source':'Fresh source'},m,jobs,s)

@pytest.mark.parametrize('value',['http://127.0.0.1:8930','gpu-20261027T051428Z-id','GPU-6f2541c5-80ed-71d7-5892-8fe9af1398eb'])
def test_private_transport_strings_blocked(value):
    report,menu,sources=fixture();report['scope']=value
    with pytest.raises(ValueError,match='Private transport'):p.validate(raw(report),{},menu,{},sources)
