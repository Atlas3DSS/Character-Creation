from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import pytest
from persona_adaptive_trials import TrialStore,bounded_candidate,SEEDS,canonical_digest,PriorityQueue,OUTSTANDING,ProducerOnlyManager,Deferred

class Bank:
    model=SimpleNamespace(identity='model')
    def program(self,pushes,phase):
        vectors={}
        for p in pushes:
            assert p['kind'] in ('trait','random','neuron')
            site='mlp_input' if p['kind']=='neuron' else p.get('site','residual')
            key=f'layers.{p["layer"]}.{site}';vectors[key]=vectors.get(key,0)+p['dose']
        norms={s:abs(v) for s,v in vectors.items()};norm=sum(v*v for v in norms.values())**.5
        program={'phase':phase,'sites':[{'site':s,'vector':[v]} for s,v in sorted(vectors.items()) if v]}
        return program,{'aggregate_relative_norm':norm,'relative_norms':norms,'norm_semantics':'test','program_sha256':canonical_digest(program)}

def row(i):
    text='who are you' if i==0 else f'Fresh original {i}'
    return {'id':str(i),'source_row_id':str(i),'prompt':text,'pair_id':f'p{i}','split':'discovery','corpus':'synthetic','system':'','max_tokens':2048,'source':{'kind':'synthetic_original','prompt_sha256':hashlib.sha256(text.encode()).hexdigest()}}

def store(tmp_path,n=8):
    s=TrialStore.__new__(TrialStore);s.root=tmp_path;s.results=tmp_path;s.dispatch_deadline=time.time()+10000;s.until=time.time()+11000;s.binding={'model_identity':'model'};s.bank=Bank();s.mutex=threading.RLock();s.corpus_sha='corpus';s.identity=row(0);s.identity_sha='identity';s.prompts=[row(i) for i in range(1,100)];s.program_cache={};s.worker_errors=[];s.pool=None
    (tmp_path/'layer_16.npz').write_bytes(b'calibration')
    cs=[]
    for i in range(n):
        recipe={'target':f't{i}','id':f'r{i}'};c=bounded_candidate(s.bank,recipe,[{'kind':'trait','layer':16,'dose':.1+i*.001}],'all','exact_distributed',{},'0');cs.append(c)
    s.by_id={c['id']:c for c in cs};s._menu={'candidates':cs,'menu_sha256':'menu'};s.queue=PriorityQueue(tmp_path/'queue.sqlite',tmp_path)
    with s.queue.connect() as db:db.executescript('CREATE TABLE adaptive_candidates(id TEXT PRIMARY KEY,mode TEXT,body TEXT);CREATE TABLE adaptive_sources(pair_id TEXT PRIMARY KEY,candidate_id TEXT);')
    return s

def test_explicit_norm_scaling_no_nan():
    c=bounded_candidate(Bank(),{'target':'a','id':'r'},[{'kind':'trait','layer':16,'dose':1.}],'all','dose',{},'identity')
    assert c['aggregate_relative_norm']<=.35 and c['explicit_scale']==.35
    with pytest.raises(ValueError):bounded_candidate(Bank(),{'target':'a','id':'r'},[{'kind':'trait','layer':16,'dose':float('nan')}],'all','dose',{},'identity')

def test_identity_first_exact_dedup_and_fresh_controls(tmp_path):
    s=store(tmp_path);first=s.enqueue_identity_backfill();assert len(first)==64;assert s.enqueue_identity_backfill()==[]
    cid=list(s.by_id)[0];more=s.enqueue([cid],'Extend exact recipe on fresh scenes');assert len(more)==32
    assert s.enqueue([cid],'Already admitted')==[]
    with s.queue.connect() as db:jobs=[json.loads(r[0]) for r in db.execute('SELECT body FROM jobs ORDER BY created,id')]
    assert all(j['identity_probe'] for j in jobs[:64]);fresh=[j for j in jobs if not j['identity_probe']]
    assert len({j['pair_id'] for j in fresh})==4 and {j['seed'] for j in fresh}==set(SEEDS)
    for pair in {j['pair_id'] for j in fresh}:
        for seed in SEEDS:
            group={j['arm']:j for j in fresh if j['pair_id']==pair and j['seed']==seed}
            assert set(group)=={'baseline','candidate','reversal','random'}
            assert group['candidate']['pushes'][0]['dose']==-group['reversal']['pushes'][0]['dose']
            assert group['random']['pushes'][0]['seed']==seed
            assert group['random']['pushes'][0]['dose']==s.by_id[cid]['relative_norms']['layers.16.residual']
            assert all(not j['promotion_eligible'] and j['prompt']['split']=='discovery' for j in group.values())

def test_bounds_unknown_schema_and_transaction_rollback(tmp_path):
    s=store(tmp_path);ids=list(s.by_id)
    with pytest.raises(ValueError):s.enqueue(['invented'],'why')
    with pytest.raises(ValueError):s.enqueue([ids[0],ids[0]],'why')
    with pytest.raises(ValueError):s.enqueue(ids[:4],'why')
    s.enqueue(ids[:3],'first120');assert s.statuses()['outstanding']==120
    with pytest.raises(ValueError,match='bound reached'):s.enqueue([ids[3]],'too many')
    assert s.statuses()['total']==120 and len(s.statuses()['candidates'])==3
    with s.queue.connect() as db:assert db.execute('SELECT count(*) FROM adaptive_sources').fetchone()[0]==12

def test_distinct_candidates_get_disjoint_fresh_families(tmp_path):
    s=store(tmp_path);s.enqueue(list(s.by_id)[:2],'compare')
    with s.queue.connect() as db:rows=list(db.execute('SELECT pair_id,candidate_id FROM adaptive_sources'))
    assert len(rows)==8 and len({p for p,c in rows})==8

def test_no_implicit_planner():
    with pytest.raises(Deferred):ProducerOnlyManager.propose(None,999)


def test_historical_dry_wit_exception_identity_only(tmp_path):
    recipe={'target':'wit','id':'experimental-dry-wit-followon-distributed'}
    historical=bounded_candidate(Bank(),recipe,[{'kind':'trait','layer':16,'dose':.6375}],'prefill','exact_distributed',{},'identity',True)
    assert historical['historical_identity_only'] and historical['aggregate_relative_norm']==.6375
    s=store(tmp_path);s.by_id[historical['id']]=historical
    with pytest.raises(ValueError,match='identity-only'):s.enqueue([historical['id']],'must not raise dose')
    bounded=bounded_candidate(Bank(),recipe,[{'kind':'trait','layer':16,'dose':.6375}],'prefill','dose_1.0',{},'identity')
    assert bounded['aggregate_relative_norm']<=.35 and not bounded['historical_identity_only'] and bounded['id']!=historical['id']
