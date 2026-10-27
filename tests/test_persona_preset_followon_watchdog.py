import json
import sqlite3
import pytest
from persona_preset_followon_watchdog import terminal_stage,stage_name,eligible,REASON

def db(tmp_path,states=('complete','failed')):
    p=tmp_path/'stage.sqlite';c=sqlite3.connect(p);c.executescript('CREATE TABLE jobs(id TEXT,state TEXT,body TEXT);CREATE TABLE attempts(state TEXT);')
    jobs=[{'id':str(i),'prompt':'fixed'+str(i)} for i in range(len(states))]
    c.executemany('INSERT INTO jobs VALUES(?,?,?)',[(j['id'],s,json.dumps(j)) for j,s in zip(jobs,states)]);c.commit();c.close();return p,jobs

def test_terminal_failed_is_terminal_but_not_replayed(tmp_path):
    p,jobs=db(tmp_path);assert terminal_stage(p,jobs)['counts']=={'complete':1,'failed':1,'incomplete':0}

@pytest.mark.parametrize('state',['queued','running','submitting','unknown'])
def test_nonterminal_blocks(tmp_path,state):
    p,jobs=db(tmp_path,('complete',state))
    with pytest.raises(ValueError,match='not fully terminal'):terminal_stage(p,jobs)

def test_missing_extra_or_changed_frozen_job_blocks(tmp_path):
    p,jobs=db(tmp_path)
    with pytest.raises(ValueError,match='coverage'):terminal_stage(p,jobs[:1])
    jobs[0]['prompt']='changed'
    with pytest.raises(ValueError,match='body'):terminal_stage(p,jobs)

def test_active_attempt_blocks_even_terminal_jobs(tmp_path):
    p,jobs=db(tmp_path);c=sqlite3.connect(p);c.execute("INSERT INTO attempts VALUES('running')");c.commit();c.close()
    with pytest.raises(ValueError,match='Attempt still active'):terminal_stage(p,jobs)

def test_stage_from_actual_database_not_missing_status(tmp_path):
    (tmp_path/'results').mkdir();assert stage_name(tmp_path)=='discovery'
    (tmp_path/'results/persona_preset_followon_selection.json').write_text('{}');assert stage_name(tmp_path)=='discovery'
    (tmp_path/'results/persona_preset_followon_confirmation.sqlite').touch();assert stage_name(tmp_path)=='confirmation'

def test_only_exact_cause_gone_process_and_time_window():
    status={'state':'stopped','reason':REASON};unit={'ActiveState':'inactive','MainPID':'0'}
    assert eligible(status,unit,0,3600)
    assert not eligible(status,{**unit,'MainPID':'12'},0,3600)
    assert not eligible(status,{**unit,'ActiveState':'active'},0,3600)
    assert not eligible({'state':'stopped','reason':'timeout'},unit,0,3600)
    assert not eligible({'state':'complete'},unit,0,3600)
    assert not eligible(status,unit,1900,3600)


@pytest.mark.parametrize('state',['intent','failed_no_retry','unknown'])
def test_ambiguous_review_never_retried(tmp_path,state):
    from persona_preset_followon_watchdog import review_evidence
    (tmp_path/'results').mkdir();(tmp_path/'results/persona_preset_followon_judge_one.json').write_text(json.dumps({'state':state,'input_sha256':'fixed','protocol':'answer_span_ids_v2'}))
    with pytest.raises(ValueError,match='no restart'):review_evidence(tmp_path,'fixed')


def test_dynamic_execstart_fields_normalized_for_runner_owner(monkeypatch):
    import persona_preset_followon_watchdog as w
    def state(time,pid):return w.normalize_service('InvocationID=owned\nMainPID=0\nActiveState=inactive\nExecStart={ path=/python ; argv[]=/python fixed.py ; start_time='+time+' ; pid='+pid+' ; code=exited ; status=0 }')
    a=state('old','12');b=state('new','0');assert a==b
    monkeypatch.setattr(w,'process_gone',lambda p:True)
    assert w.owner({'attachment':{'unit':a,'process':{}},'attempts':[]},b)==('owned',False)


def test_actual_runner_recovers_collected_unit_once(monkeypatch,tmp_path):
    import persona_preset_followon_watchdog as w
    results=tmp_path/'results';results.mkdir()
    status=results/'persona_preset_followon_status.json';status.write_text(json.dumps({'state':'stopped','reason':REASON}))
    live={'InvocationID':'owned','MainPID':'12','ActiveState':'active','ExecStart':'/python fixed.py','LoadState':'loaded'}
    gone={'InvocationID':'','MainPID':'0','ActiveState':'inactive','ExecStart':'','LoadState':'not-found'}
    new={**live,'InvocationID':'new','MainPID':'13'}
    sequence=iter([live,gone,gone,new,new])
    monkeypatch.setattr(w,'service',lambda:next(sequence))
    monkeypatch.setattr(w,'process_identity',lambda pid:{'pid':int(pid),'argv':['/python','fixed.py'],'cwd':str(tmp_path),'start_ticks':'1','boot_id':'b'})
    monkeypatch.setattr(w,'process_gone',lambda p:True)
    monkeypatch.setattr(w,'verify',lambda root:{'stage':'discovery','jobs':960})
    monkeypatch.setattr(w.time,'time',lambda:1000)
    def sleep(_):status.write_text(json.dumps({'state':'complete'}));new.update(ActiveState='inactive',MainPID='0')
    monkeypatch.setattr(w.time,'sleep',sleep)
    calls=[]
    monkeypatch.setattr(w.subprocess,'run',lambda cmd,**kw:calls.append(cmd))
    w.run(tmp_path,5000,'owned')
    assert len(calls)==1 and calls[0][:2]==['systemd-run','--user']
    assert calls[0][-2:]==['/python','fixed.py']
    receipt=json.loads((results/'persona_preset_followon_watchdog.json').read_text())
    assert receipt['state']=='finished' and receipt['attempts'][0]['prior_invocation']=='owned'


def test_collected_unit_not_recovered_if_owned_pid_alive(monkeypatch):
    import persona_preset_followon_watchdog as w
    monkeypatch.setattr(w,'process_gone',lambda p:False)
    with pytest.raises(ValueError,match='still has owned process'):
        w.owner({'attachment':{'unit':{'InvocationID':'old'},'process':{}},'attempts':[]},{'LoadState':'not-found'})
