import persona
import json
from pathlib import Path
import sqlite3
import pytest
import persona_perturbation_mapping as mapping


def jobs(n=8):return [{'id':str(i),'prompt':{'split':'discovery'}} for i in range(n)]


def test_priority_changes_between_claims_and_inflight_may_finish(tmp_path):
    (tmp_path/'results').mkdir();q=mapping.PriorityQueue(tmp_path/'results/background.sqlite',tmp_path);q.enqueue({},jobs(),limit=8)
    claims=q.claim_many('worker',4);assert len(claims)==4
    foreground=tmp_path/'results'/mapping.STUDIES[2]
    with sqlite3.connect(foreground) as db:db.execute('CREATE TABLE jobs(state TEXT)');db.execute("INSERT INTO jobs VALUES('queued')")
    assert q.claim_many('worker',4)==[] and q.claim('worker') is None
    for job,number in claims:q.finish(job,number,{'complete':True,'error':None})
    assert q.counts()=={'complete':4,'queued':4}
    with sqlite3.connect(foreground) as db:db.execute("UPDATE jobs SET state='complete'")
    assert len(q.claim_many('worker',4))==4


@pytest.mark.parametrize('state',['queued','running','submitting','stopping'])
def test_interactive_work_blocks_and_terminal_states_do_not(tmp_path,state):
    (tmp_path/'results').mkdir();p=tmp_path/'results/persona_explorer.sqlite'
    with sqlite3.connect(p) as db:
        db.execute('CREATE TABLE batches(state TEXT)');db.execute('CREATE TABLE sculptor_campaigns(state TEXT)');db.execute('INSERT INTO sculptor_campaigns VALUES(?)',(state,))
    assert mapping.foreground_busy(tmp_path)
    for terminal in ('complete','partial','interrupted','failed','incomplete'):
        with sqlite3.connect(p) as db:db.execute('UPDATE sculptor_campaigns SET state=?',(terminal,))
        assert not mapping.foreground_busy(tmp_path)


def test_missing_future_queue_rechecked_and_malformed_queue_fails_closed(tmp_path):
    (tmp_path/'results').mkdir();assert not mapping.foreground_busy(tmp_path)
    (tmp_path/'results'/mapping.STUDIES[-1]).touch();assert mapping.foreground_busy(tmp_path)


def prepared(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();(tmp_path/'data').mkdir();rows=[]
    for i in range(3072):
        prompt=f'Original synthetic case {i}.'
        rows.append({'id':str(i),'pair_id':str(i),'category':'fixture','corpus':'synthetic','split':'discovery' if i<2443 else 'heldout',
            'system':'','thinking':False,'prompt':prompt,'source':{'kind':'synthetic_original','prompt_sha256':mapping.hashlib.sha256(prompt.encode()).hexdigest()}})
    cp=tmp_path/'data/persona_perturbation_prompts.jsonl';cp.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    mp=tmp_path/'data/persona_perturbation_prompts_manifest.json';mp.write_text(json.dumps({'status':'complete','records':3072,'split_counts':{'discovery':2443,'heldout':629}}))
    monkeypatch.setattr(mapping,'CORPUS_SHA',mapping.sha(cp));monkeypatch.setattr(mapping,'MANIFEST_SHA',mapping.sha(mp))
    config={'endpoints':[{'model_identity':'m','runtime_identity':'r','prepared_identity':'p','url':str(i)} for i in range(6)]}
    cf=tmp_path/'results/persona_preset_optimized_config.json';cf.write_text(json.dumps(config))
    (tmp_path/'results/persona_preset_runtime_gate.json').write_text(json.dumps({'state':'qualified','config_sha256':mapping.sha(cf)}))
    (tmp_path/'results/layer_32.npz').write_bytes(b'fixture')
    return mapping.prepare(tmp_path)


def test_exact_discovery_only_baseline_and_public_source_allowlist(tmp_path,monkeypatch):
    plan,jobset,config=prepared(tmp_path,monkeypatch)
    assert len(jobset)==2443 and len({j['id'] for j in jobset})==2443
    assert all(j['prompt']['split']=='discovery' and j['pushes']==[] and j['prompt']['system']=='' and j['prompt']['max_tokens']==2048 for j in jobset)
    assert not any(int(j['prompt']['id'])>=2443 for j in jobset)
    q=mapping.PriorityQueue(tmp_path/'results/background.sqlite',tmp_path);q.enqueue({},jobset,limit=2443)
    public=mapping.summary(q.path,plan,jobset)
    assert public['rows']==[] and public['heldout_excluded']==629
    assert 'Raw arrays are not published' in public['scope']
    assert str(tmp_path) not in json.dumps(public)


def test_publication_intent_before_call_and_no_uncertain_retry(tmp_path):
    path=tmp_path/'public.json';path.write_text('{}');rp=tmp_path/'receipt.json';receipt={'source':{'hash':'a'},'state':'idle','attempts':[]};calls=[]
    def fail(path,timeout):
        assert json.loads(rp.read_text())['attempts'][-1]['state']=='intent';calls.append(1);raise TimeoutError('Unknown outcome')
    mapping.publish(path,rp,receipt,0,True,uploader=fail);assert not calls
    mapping.publish(path,rp,receipt,48,False,uploader=fail);assert receipt['state']=='uncertain'
    mapping.publish(path,rp,receipt,2443,True,uploader=fail);assert len(calls)==1


def test_foreground_arriving_after_claim_yields_before_durable_post_intent(tmp_path):
    (tmp_path/'results').mkdir();q=mapping.PriorityQueue(tmp_path/'results/background.sqlite',tmp_path);q.enqueue({},jobs(4),limit=4)
    claims=q.claim_many('worker',4)
    with sqlite3.connect(tmp_path/'results'/mapping.STUDIES[0]) as db:db.execute('CREATE TABLE jobs(state TEXT)');db.execute("INSERT INTO jobs VALUES('running')")
    event={'batch_id':'b','attempts':[{'job_id':j['id'],'number':n} for j,n in claims]}
    with pytest.raises(RuntimeError,match='safely released'):q.audit('native_batch_submitting',event)
    assert q.counts()=={'queued':4}
    with q.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM audit WHERE event='native_batch_submitting'").fetchone()[0]==0
        assert db.execute("SELECT COUNT(*) FROM attempts WHERE state='unsubmitted_preflight'").fetchone()[0]==4


def test_submitted_uncertain_attempt_never_requeued_for_priority(tmp_path):
    (tmp_path/'results').mkdir();q=mapping.PriorityQueue(tmp_path/'results/background.sqlite',tmp_path);q.enqueue({},jobs(1),limit=1)
    claims=q.claim_many('worker',1);job,n=claims[0]
    q.audit('native_batch_submitting',{'batch_id':'b','attempts':[{'job_id':job['id'],'number':n}]})
    with sqlite3.connect(tmp_path/'results'/mapping.STUDIES[0]) as db:db.execute('CREATE TABLE jobs(state TEXT)');db.execute("INSERT INTO jobs VALUES('queued')")
    q.finish(job,n,{'error':'transport uncertainty','complete':False},retry=False)
    assert q.counts()=={'failed':1}
    with pytest.raises(ValueError,match='Submitted attempt'):q.release_unsubmitted(claims)
