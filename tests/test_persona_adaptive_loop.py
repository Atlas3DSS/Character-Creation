"""CPU-only responsiveness, provenance and no-replay checks."""
import persona
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import pytest
import persona_adaptive_loop as m


def menu():
    rows=[]
    for i in range(4):
        r={'target':'t'+str(i),'kind':'dose','description':'Experimental','pushes':[{'kind':'neuron','layer':0,'neuron':i,'dose':.1}],'phase':'all','program_sha256':'a'*64,'model_identity':'model','aggregate_relative_norm':.1,'source_recipe_id':'source'};r['id']=m.canonical_digest(r);rows.append(r)
    v={'schema':'character-creation.adaptive-menu.v1','binding':{'model_identity':'model'},'candidates':rows,'excluded_plans':{},'implementation_sha256':'b'*64,'source_sha256':'c'*64,'excluded_gpu':'privateUUID'};v['menu_sha256']=m.canonical_digest(v);return v


class Store:
    def __init__(self):self.value=menu();self.enqueues=[]
    def menu(self):return deepcopy(self.value)
    def statuses(self):return {'candidates':[],'outstanding':0,'outstanding_cap':128,'total':0,'maximum_jobs':4096,'dispatch_deadline':m.DISPATCH_DEADLINE}
    def enqueue(self,ids,rationale):self.enqueues.append(ids);return ids
    def completed_groups(self):return []
    def public_snapshot(self):return {'binding':self.value['binding'],'counts':{},'total':0,'terminal_jobs':0,'rows':[]}


@pytest.fixture
def loop(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();db=sqlite3.connect(tmp_path/'results/persona_explorer.sqlite');db.execute('CREATE TABLE batches(state TEXT)');db.close()
    monkeypatch.setattr(m,'source_allowlist',lambda root:{'source':'who are you'})
    return m.Loop(tmp_path,Store(),until=time.time()+120,admit=lambda root:({'url':'http://127.0.0.1:18511'},None))


def group():
    g={'id':'group','candidate_id':'candidate','source_id':'source','prompt':'who are you','prompt_sha256':hashlib.sha256(b'who are you').hexdigest(),'source_sha256':m.IDENTITY_SHA,'split':'discovery','identity_probe':True,'pair_id':'pair','seed':7,'fact_contract':{'policy':'Self-description is behavior, not proven personhood.'},'arms':{}}
    for i,arm in enumerate(('baseline','candidate','reversal','random')):
        answer='A neutral answer '+str(i)
        g['arms'][arm]={**{k:g[k] for k in ('source_id','prompt','prompt_sha256','split','pair_id','seed')},'job_id':str(i),'state':'complete','complete':True,'capture_validated':True,'answer':answer,'answer_sha256':hashlib.sha256(answer.encode()).hexdigest()}
    return g


def test_menu_hash_content_and_duplicate_ids():
    assert len(m.validate_menu(menu())['candidates'])==4
    for mutation in ('dose','model','duplicate'):
        v=menu()
        if mutation=='dose':v['candidates'][0]['pushes'][0]['dose']=4
        elif mutation=='model':v['candidates'][0]['model_identity']='other'
        else:v['candidates'][1]=v['candidates'][0]
        v['menu_sha256']=m.canonical_digest({k:x for k,x in v.items() if k!='menu_sha256'})
        with pytest.raises(ValueError):m.validate_menu(v)


def test_idle_foreground_yields_before_any_john_call(loop):
    with sqlite3.connect(loop.root/'results/persona_explorer.sqlite') as db:db.execute("INSERT INTO batches VALUES('queued')")
    calls=[]
    assert m.idle_admission(loop.root,http=lambda *a,**k:calls.append(a),planner=lambda:calls.append('planner'))[0] is None
    assert calls==[]


def test_busy_or_unknown_john_never_enters_call(loop):
    for slots in ([{'is_processing':True}],[],[{}]):
        assert m.idle_admission(loop.root,http=lambda *a,**k:slots,planner=lambda:{'url':'http://127.0.0.1:18511'})[0] is None
    assert m.idle_admission(loop.root,http=lambda *a,**k:[{'is_processing':False}],planner=lambda:{'url':'http://127.0.0.1:18511'})[1] is None


def test_blinded_packet_has_no_arm_treatment_labels_and_exact_citation_binding():
    packet,key,spans,hashes=m.review_packet(group(),{'source':'who are you'})
    assert all(set(a)=={'code','spans'} for a in packet['answers'])
    rows=[{'code':a['code'],'scores':{k:0 for k in m.AXES},'fidelity':'unclear','coherence':'coherent','evidence_ids':[a['spans'][0]['id']]} for a in packet['answers']]
    assert m.validate_review({'answers':deepcopy(rows)},key,spans)['answers'][0]['evidence']
    rows[0]['evidence_ids']=rows[1]['evidence_ids']
    with pytest.raises(ValueError,match='Evidence'):m.validate_review({'answers':rows},key,spans)


@pytest.mark.parametrize('bad',['heldout','answer_hash','cross_scene','incomplete'])
def test_review_source_and_complete_answer_gates(bad):
    g=group()
    if bad=='heldout':g['split']='heldout'
    elif bad=='answer_hash':g['arms']['baseline']['answer']='changed'
    elif bad=='cross_scene':g['arms']['baseline']['seed']=8
    else:g['arms']['baseline']['complete']=False
    with pytest.raises(ValueError):m.review_packet(g,{'source':'who are you'})


def test_post_intent_precedes_http_uncertain_never_replayed_even_after_restart(loop):
    calls=[]
    def http(*args,**kwargs):
        assert loop.db.execute("SELECT state FROM calls WHERE id='one'").fetchone()[0]=='intent'
        calls.append(args);raise TimeoutError()
    loop.http=http
    assert loop.call('plan','one',{}, {},'system',512,lambda v:v) is None
    loop.last_call=0;assert loop.call('plan','one',{}, {},'system',512,lambda v:v) is None
    assert len(calls)==1 and loop.db.execute("SELECT state FROM calls WHERE id='one'").fetchone()[0]=='failed_no_retry'
    restarted=m.Loop(loop.root,loop.store,http=http,admit=loop.admit,until=loop.until)
    assert restarted.call('plan','one',{}, {},'system',512,lambda v:v) is None and len(calls)==1


def test_incomplete_john_json_is_retained_not_rescored(loop):
    loop.http=lambda *a,**k:{'choices':[{'finish_reason':'length','message':{'content':'{}'}}]}
    assert loop.call('review','r',{}, {},'system',1024,lambda v:v) is None
    row=loop.db.execute("SELECT state,response FROM calls WHERE id='r'").fetchone();assert row[0]=='failed_no_retry' and 'length' in row[1]


def test_proposal_ids_deduplicated_and_no_arbitrary_coordinates(loop,monkeypatch):
    requested=[]
    def call(kind,identity,packet,schema,system,budget,validator):
        requested.append(packet);assert budget==512
        with pytest.raises(ValueError):validator({'candidate_ids':['invented','other'],'rationale':'x'})
        return validator({'candidate_ids':[r['id'] for r in packet['candidates'][:2]],'rationale':'Discovery comparison'})
    monkeypatch.setattr(loop,'call',call)
    assert loop.propose() and loop.propose()
    first={r['id'] for r in requested[0]['candidates'][:2]};second={r['id'] for r in requested[1]['candidates']}
    assert not first&second


def test_dispatch_unknown_not_requeued(loop,monkeypatch):
    ids=[r['id'] for r in loop.menu['candidates'][:2]]
    with loop.db:loop.db.execute('INSERT INTO proposals VALUES(?,?,?,?,?,?)',('p',m.encode(ids),'reason','planned','e','m'))
    calls=[]
    def enqueue(*a):calls.append(a);raise RuntimeError()
    monkeypatch.setattr(loop.store,'enqueue',enqueue);monkeypatch.setattr(m.time,'time',lambda:1)
    loop.dispatch_pending();loop.dispatch_pending();assert len(calls)==1


def test_public_projection_has_no_deployment_binding(loop):
    loop.publish();v=json.loads((loop.root/'results/persona_adaptive_public.json').read_text())
    assert 'excluded_gpu' not in v['menu'] and not v['promotion_eligible']
    assert v['revision_sha256']==m.canonical_digest({k:x for k,x in v.items() if k!='revision_sha256'})


def test_identity_review_order_precedes_other_completed_groups(loop,monkeypatch):
    ordinary={'id':'a','identity_probe':False};identity={'id':'z','identity_probe':True};seen=[]
    monkeypatch.setattr(loop.store,'completed_groups',lambda:[ordinary,identity])
    monkeypatch.setattr(loop,'review',lambda g:seen.append(g['id']) or True)
    loop.cycle();assert seen==['z']


def test_crash_after_intent_is_uncertain_on_restart(loop):
    with loop.db:loop.db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?,NULL,NULL,NULL,?)',('crash','plan','intent','{}','{}','{}',1))
    resumed=m.Loop(loop.root,loop.store,admit=loop.admit,until=loop.until)
    assert resumed.db.execute("SELECT state FROM calls WHERE id='crash'").fetchone()[0]=='uncertain_no_retry'
    resumed.http=lambda *a,**k:pytest.fail('Uncertain call replayed')
    assert resumed.call('plan','crash',{}, {},'system',512,lambda v:v) is None


def test_oversize_full_answers_not_silently_truncated():
    g=group()
    for r in g['arms'].values():r['answer']='x'*10000;r['answer_sha256']=hashlib.sha256(r['answer'].encode()).hexdigest()
    with pytest.raises(ValueError,match='no silent truncation'):m.review_packet(g,{'source':'who are you'})


@pytest.mark.parametrize('persisted_state',['received','validated'])
def test_crash_after_received_or_validated_review_recovers_without_post(loop,persisted_state):
    g=group();packet,key,spans,_=m.review_packet(g,loop.allowed)
    rows=[{'code':a['code'],'scores':{k:0 for k in m.AXES},'fidelity':'unclear','coherence':'coherent','evidence_ids':[a['spans'][0]['id']]} for a in packet['answers']]
    response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'answers':rows})}}]}
    result=m.validate_review({'answers':deepcopy(rows)},key,spans)
    identity=m.canonical_digest(['adaptive-review',g])
    with loop.db:loop.db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?,?,?,?,?)',(identity,'review',persisted_state,'{}',m.encode(packet),'{}',m.encode(response),m.encode(result) if persisted_state=='validated' else None,None,time.time()))
    loop.http=lambda *a,**k:pytest.fail('Recovery must never POST')
    assert loop.review(g)
    saved=json.loads(loop.db.execute('SELECT body FROM reviews').fetchone()[0])
    assert saved['state']=='validated' and len(saved['judgments'])==4
    assert loop.evidence()['discovery_reviews']
    assert not loop.review(g)
