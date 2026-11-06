"""v4 successor policy tests: gate, ledger import/audit, continuation, tamper detection, admission baseline, dispatch scope.
Run with: PYTHONPATH=<v4 dir>:<root>:<root>/monkey python -m pytest -q test_persona_threshold_v4.py
"""
import hashlib, json, os, sqlite3, tempfile, time
from pathlib import Path
import pytest
import persona_threshold as policy
import persona_threshold_audit as audit
import persona_threshold_metrics as metrics
import threshold_admissions as admissions
import persona_wide_identity_sweep as runner

PRED='4e3e'+'0'*60
B4={'model_identity':'m'*64,'prepared_identity':'p4'*32,'runtime_identity':'r4'*32}
B32={'model_identity':'m'*64,'prepared_identity':'p32'*21+'x','runtime_identity':'r32'*21+'x'}
B16={'model_identity':'m'*64,'prepared_identity':'p16'*21+'x','runtime_identity':'r16'*21+'x'}
BOILER=metrics.BASELINE
DRIFT='Hello there, I am just a helpful assistant here to talk about the weather, cooking, music, travel and books whenever you like.'
LOOP='fit fit fit fit fit fit fit fit fit fit fit fit fit fit fit fit fit fit fit fit'

def desc(binding,width,qual,ctx=1024):
    return {'id':'checkpoint','flare':{'model_identity':binding['model_identity'],'logical_context_tokens':ctx,
            'batch':{'qualified':True,'supported':True,'max_batch_size':width,'binding':binding,'qualification_sha256':qual,'physical_pages':8,'page_tokens':64}}}

def measured(answer,tokens=40):
    m=metrics.measure(answer);m['coherent']=int(runner.coherent(answer,tokens));return m

class Fixture:
    def __init__(self):
        self.root=Path(tempfile.mkdtemp());(self.root/'results').mkdir()
        self.cal=self.root/'results'/'layer_01.npz';self.cal.write_bytes(b'calibration-bytes')
        self.builder=self.root/'builder.py';self.builder.write_text('# builder\n')
        self.db=runner.connect(self.root);policy.schema(self.db);admissions.schema(self.db)
        self.art=self.root/'artifacts';self.art.mkdir()
        self.cal_sha=hashlib.sha256(self.cal.read_bytes()).hexdigest()
        self.targets=[{'layer':1,'neuron':5,'priority':0,'target_bits':5},{'layer':1,'neuron':6,'priority':1,'target_bits':5},{'layer':1,'neuron':7,'priority':2,'target_bits':5}]
        self.spec={'schema':'identity-threshold.targets.v1','targets':self.targets,'generator_sha256':'g'*64}
        self.config={'schema':policy.SCHEMA_V3,'spec':self.spec,'bindings':{str(p):{'binding':B4,'qualification_sha256':'q4'*32} for p in range(8931,8936)},
                     'metric_sha256':policy.METRIC_SHA,'policy_sha256':'old'*21+'x','auditor_sha256':'aud'*21+'x','prompt':'who are you','max_tokens':160,'seed':0,
                     'temperature':0,'phase':'all','thinking':False,'grid':list(policy.GRID),'budget_per_sign':policy.BUDGET,
                     'calibration_files':{str(self.cal):self.cal_sha},'builder_file':str(self.builder),'builder_sha256':hashlib.sha256(self.builder.read_bytes()).hexdigest(),
                     'reuse_source_snapshot_sha256':'s'*64,'reuse_source_count':1}
        db=self.db
        db.execute("INSERT INTO threshold_campaigns VALUES(?,?,'blocked',?,?)",(PRED,policy.encode(self.config),time.time()-1000,'Uncertain sample deadbeef'))
        for p in range(8931,8936):
            self.job(policy.digest(['base',p]),-1,-1,0.0,BOILER,role='baseline',port=p,target_port=p,binding=B4)
        # one admitted endpoint (8930, B32) with an immutable admission document and seals for its probes
        self.doc={'schema':'identity-threshold.admission.v1','campaign_id':PRED,'port':8930,'binding':B32,'advertised_capacity':32,'dispatch_width':32,'startup_receipt':{'sha256':'q32'*21+'x'}}
        self.aid=policy.digest(self.doc)
        db.execute('INSERT INTO threshold_endpoint_admissions VALUES(?,?,?,?,?)',(self.aid,PRED,8930,policy.encode(self.doc),time.time()))
        # legacy source rows (non-campaign, |dose| 0.5) reused by the +1 searches
        self.legacy_by={n:self.job('legacy'+str(n)+'0'*25,1,n,0.5,DRIFT,port=8932,binding=B4,campaign=None) for n in (5,6,7)}
        self.legacy=self.legacy_by[6]
        self.searches={}
        db.commit()

    def job(self,identity,layer,neuron,dose,answer,role='coarse',port=8931,target_port=None,binding=B4,campaign=PRED,sid=None,state='done',admission=None,tokens=40,finished=None):
        m=measured(answer,tokens) if state=='done' else {'coherent':None}
        finished=finished if finished is not None else (time.time()-700 if state in ('done','uncertain') else None)
        artifact=artifact_sha=program_sha=request_sha=None
        if state=='done':  # real on-disk evidence: activation artifact + provenance sidecar, as the servers write them
            artifact=str(self.art/f'{identity}.npz');raw=os.urandom(64);Path(artifact).write_bytes(raw);artifact_sha=hashlib.sha256(raw).hexdigest()
            vector=[0.0]*17408
            if layer>=0:vector[neuron]=float(dose)
            steering={'phase':'all','sites':[{'site':f'layers.{layer}.mlp_input','vector':vector}]} if layer>=0 else None
            request={'model':'checkpoint','messages':[{'role':'user','content':'who are you'}],'seed':0,'max_tokens':160,'temperature':0,'stream':False,
                     'chat_template_kwargs':{'enable_thinking':False},'metadata':{'job_id':identity},**({'steering':steering} if steering else {})}
            request_sha=policy.digest(request);program_sha=policy.digest(steering)
            report={'status':'complete','request':request,'request_sha256':request_sha,**binding,'activation_artifact':{'sha256':artifact_sha,'path':artifact,'request_sha256':request_sha},
                    'response':{**binding,'choices':[{'message':{'content':answer},'finish_reason':'stop'}],'usage':{'completion_tokens':tokens,'prompt_tokens':9}}}
            Path(artifact).with_suffix('.jsonl').write_text(json.dumps(report))
        self.db.execute("""INSERT INTO jobs(id,sweep,layer,neuron,dose,dose_hex,kind,campaign_id,search_id,probe_role,target_port,state,created,started,finished,port,answer,finish,tokens,
            artifact,artifact_sha256,program_sha256,metrics_json,coherent,request_sha256,binding_json,admission_id)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (identity,'threshold-phase-a' if campaign else 'manual',layer,neuron,dose,float(dose).hex(),'threshold:x',campaign,sid,role if campaign else None,target_port,state,
             time.time()-900,time.time()-800,finished,port,answer if state=='done' else None,'stop',tokens if state=='done' else None,
             artifact,artifact_sha,program_sha,policy.encode(m) if state=='done' else None,m['coherent'] if state=='done' else None,
             request_sha,policy.encode(binding) if state=='done' else None,admission))
        if admission and state=='done':
            seal=policy.digest([campaign,sid,'done',answer,tokens,policy.encode(binding),artifact,artifact_sha,port,admission,role])
            self.db.execute('INSERT INTO threshold_completion_seals VALUES(?,?)',(identity,seal))
        return identity

    def search(self,layer,neuron,sign,magnitudes,answers,state,reuse=False,queued_next=None,uncertain_next=None,priority=0,admitted_every=None,post_block=None):
        sid=policy.digest([PRED,layer,neuron,sign]);db=self.db
        db.execute("INSERT INTO threshold_searches(id,campaign_id,layer,neuron,sign,priority,target_bits,state,audit_done,source_snapshot_json,stop_reason) VALUES(?,?,?,?,?,?,5,?,1,'[]',?)",
                   (sid,PRED,layer,neuron,sign,priority,state,'budget_exhausted' if state=='complete' else 'uncertain_no_retry' if state=='uncertain' else None))
        points=[{'job_id':'b','magnitude':0.0,**measured(BOILER)}]
        for i,(mag,ans) in enumerate(zip(magnitudes,answers)):
            adm=self.aid if admitted_every and i%admitted_every==0 else None
            jid=self.job(policy.digest([sid,mag]),layer,neuron,sign*mag,ans,role='coarse' if mag in policy.GRID else 'explore',port=8930 if adm else 8931,binding=B32 if adm else B4,sid=sid,admission=adm)
            points.append({'job_id':jid,'magnitude':mag,**measured(ans)})
        reused=0
        if reuse:
            legacy=self.legacy_by[neuron]
            src=dict(zip(audit.FIELDS,db.execute('SELECT '+','.join(audit.FIELDS)+' FROM jobs WHERE id=?',(legacy,)).fetchone()))
            inner={'source_job_id':legacy,'source_row_sha256':audit.row_fingerprint(src),'artifact_sha256':src['artifact_sha256'],'sidecar_sha256':'s'*64,'request_sha256':'r'*64,
                   'program_sha256':src['program_sha256'],'binding':B4,'qualification_sha256':'q4'*32,'current_calibration_sha256':self.cal_sha,'dose_hex':float(0.5).hex(),'auditor_sha256':self.config['auditor_sha256']}
            ev={**inner,'audit_sha256':policy.digest(inner)}
            db.execute('INSERT INTO threshold_reuse VALUES(?,?,?,?,1,NULL,?,?,?)',(sid,legacy,0.5,float(0.5).hex(),policy.digest(ev),policy.encode(ev),policy.encode(measured(DRIFT))))
            points.append({'job_id':legacy,'magnitude':0.5,'reused':True,**measured(DRIFT)});reused=1
        result=policy.observed(points,5)
        db.execute("UPDATE threshold_searches SET probes_used=?,reused_observations=?,first_change_json=?,first_coherent_departure_json=?,collapse_json=?,reversions_json=? WHERE id=?",
                   (len(magnitudes),reused,*(None if result[k] is None else policy.encode(result[k]) for k in ('first_change','first_coherent_departure','collapse')),policy.encode(result['reversions']),sid))
        if post_block is not None:  # done after the block: counted in the ledger, not in the stale counters/summary
            self.job(policy.digest([sid,'post',post_block]),layer,neuron,sign*post_block,BOILER,role='explore',sid=sid,finished=time.time()-100)
        if queued_next is not None:
            self.job(policy.digest([sid,'queued',queued_next]),layer,neuron,sign*queued_next,None,role='explore',sid=sid,state='queued')
        if uncertain_next is not None:
            self.job(policy.digest([sid,'uncertain',uncertain_next]),layer,neuron,sign*uncertain_next,None,role='explore',sid=sid,state='uncertain',port=8933)
        self.searches[(layer,neuron,sign)]=sid;db.commit();return sid

def build(reuse_on_plus=True):
    f=Fixture()
    grid=list(policy.GRID);fill=list(policy.FILL)
    # (1,5,+1): complete; 0.5 reused, so 4 coarse + 4 explore = 8 fresh, boilerplate everywhere
    f.search(1,5,1,[0.125,0.25,0.65,0.85]+fill[:4],[BOILER]*8,'complete',reuse=reuse_on_plus,admitted_every=2)
    # (1,5,-1): complete without reuse: 5 coarse + 3 explore, a drift at 0.85 with refine steps already resolved to 1/32
    f.search(1,5,-1,grid+[0.75,0.7,0.675],[BOILER]*4+[DRIFT,DRIFT,DRIFT,DRIFT],'complete')
    # (1,6,+1): running, 7 probes used (4 coarse with the reused drift at 0.5 + 3 explore); the predecessor had queued the 8th,
    # a refine of the first_change bracket (0.375, 0.5) at 0.4375
    f.search(1,6,1,[0.125,0.25,0.65,0.85]+fill[:3],[BOILER]*7,'running',reuse=reuse_on_plus,queued_next=0.4375,priority=1,admitted_every=3)
    # (1,6,-1): uncertain, 7 probes used; the 8th probe died on 8933
    f.search(1,6,-1,grid+fill[:2],[BOILER]*7,'uncertain',uncertain_next=fill[2],priority=1)
    # (1,7,+1): running with 7 counted probes and an 8th that finished after the block (never advanced): complete on import
    f.search(1,7,1,grid+fill[:2],[BOILER]*7,'running',priority=2,post_block=fill[2])
    # (1,7,-1): running, 6 probes used, 7th queued
    f.search(1,7,-1,grid+fill[:1],[BOILER]*6,'running',priority=2,queued_next=fill[1])
    return f

DESCS={8930:desc(B32,32,'q32'*21+'x'),8931:desc(B16,16,'q16'*21+'x',4096)}

def test_gate_requires_declared_blocked_predecessor():
    f=build()
    with pytest.raises(ValueError,match='must be resolved first'):
        policy.enqueue(f.db,f.spec,{p:desc(B4,4,'q4'*32) for p in range(8931,8936)})
    with pytest.raises(ValueError,match='unresolved blocked'):
        policy.enqueue(f.db,None,DESCS,predecessor='f'*64,coherence=runner.coherent)
    with pytest.raises(ValueError,match='must equal the predecessor spec'):
        policy.enqueue(f.db,{'targets':[]},DESCS,predecessor=PRED,coherence=runner.coherent)
    with pytest.raises(ValueError,match='coherence required'):
        policy.enqueue(f.db,None,DESCS,predecessor=PRED)
    assert f.db.execute("SELECT count(*) FROM threshold_campaigns").fetchone()[0]==1  # nothing created, no stale gate

def test_successor_imports_and_continues():
    f=build();db=f.db
    out=policy.enqueue(db,None,DESCS,predecessor=PRED,coherence=runner.coherent,deep_modulus=1)
    cid=out['campaign_id']
    assert out=={'campaign_id':cid,'added':3,'searches':6,'imported_complete':3,'continued':3,'superseded':2}
    assert db.execute('SELECT state FROM threshold_campaigns WHERE id=?',(cid,)).fetchone()[0]=='searching'
    assert db.execute('SELECT state,error FROM threshold_campaigns WHERE id=?',(PRED,)).fetchone()==('blocked','Uncertain sample deadbeef')
    assert db.execute("SELECT count(*) FROM threshold_campaigns WHERE state='auditing'").fetchone()[0]==0
    cfg=policy.campaign_config(db,cid)
    assert cfg['schema']==policy.SCHEMA_V4 and cfg['spec']==f.spec and cfg['grid']==list(policy.GRID) and cfg['budget_per_sign']==8
    assert set(cfg['bindings'])=={'8930','8931'} and cfg['bindings']['8931']['max_batch_size']==16 and cfg['bindings']['8930']['binding']==B32
    assert cfg['excluded_endpoints']['8932'].startswith('frozen original B4 shard') and '8930' not in cfg['excluded_endpoints']
    pred=cfg['predecessor']
    assert pred['campaign_id']==PRED and pred['state']=='blocked' and pred['error']=='Uncertain sample deadbeef'
    assert pred['searches_imported_complete']==3 and pred['searches_continued']==3 and pred['probes_carried']==8+8+7+7+8+6 and pred['reused_observations_carried']==2
    assert pred['queued_probes_superseded']==2 and pred['uncertain_probes_left_as_record']==1 and pred['baseline']['port']==8931 and pred['probes_completed_after_block']==1
    assert pred['uncertain_probes_rerun_by_successor']==1 and pred['lineage']==[PRED] and pred['root_campaign_id']==PRED and pred['block_at'] is not None
    assert pred['anchors']=={'rows':44,'deep_audited':44,'deep_modulus':1}
    assert cfg['policy_sha256']==hashlib.sha256(Path(policy.__file__).read_bytes()).hexdigest()!=f.config['policy_sha256']
    assert cfg['calibration_files']==f.config['calibration_files']
    states=dict(db.execute("SELECT layer*100+neuron*10+(sign>0),state FROM threshold_searches WHERE campaign_id=?",(cid,)).fetchall())
    assert states=={151:'complete',150:'complete',161:'running',160:'running',171:'complete',170:'running'}
    rows=db.execute("SELECT layer,neuron,sign,probes_used,reused_observations,stop_reason FROM threshold_searches WHERE campaign_id=? ORDER BY neuron,sign",(cid,)).fetchall()
    assert rows==[(1,5,-1,8,0,'budget_exhausted'),(1,5,1,8,1,'budget_exhausted'),(1,6,-1,7,0,None),(1,6,1,7,1,None),(1,7,-1,6,0,None),(1,7,1,8,0,'budget_exhausted')]
    # continuation: exactly one fresh successor probe per unfinished search, at the magnitude the predecessor would have run next
    new=db.execute("SELECT s.neuron,s.sign,j.dose,j.probe_role,j.state,j.target_port,j.admission_id FROM jobs j JOIN threshold_searches s ON s.id=j.search_id WHERE j.campaign_id=? ORDER BY s.sign,s.neuron",(cid,)).fetchall()
    assert new==[(6,-1,-0.75,'explore','queued',None,None),(7,-1,-0.575,'explore','queued',None,None),(6,1,0.4375,'refine','queued',None,None)]
    ev7=json.loads(db.execute('SELECT evidence_json FROM threshold_successor_searches WHERE search_id=?',(policy.digest([cid,1,7,1]),)).fetchone()[0])
    assert ev7['mode']=='complete' and len(ev7['post_block_probe_ids'])==1 and len(ev7['jobs'])==8
    # the predecessor's queued probe is superseded with the successor named; its uncertain probe stays as the record
    assert db.execute("SELECT state,error FROM jobs WHERE campaign_id=? AND dose=0.4375",(PRED,)).fetchone()==('superseded',policy.SUPERSEDED+' '+cid)
    assert db.execute("SELECT count(*) FROM jobs WHERE campaign_id=? AND state='uncertain'",(PRED,)).fetchone()[0]==1
    assert db.execute("SELECT count(*) FROM jobs WHERE campaign_id=? AND state='done'",(PRED,)).fetchone()[0]==5+44  # predecessor rows untouched
    # provenance rows and copied reuse
    succ=db.execute('SELECT mode,imported_probes,imported_reused,predecessor_campaign_id FROM threshold_successor_searches WHERE campaign_id=? ORDER BY imported_probes DESC,imported_reused',(cid,)).fetchall()
    assert succ==[('complete',8,0,PRED),('complete',8,0,PRED),('complete',8,1,PRED),('continue',7,0,PRED),('continue',7,1,PRED),('continue',6,0,PRED)]
    ev=json.loads(db.execute('SELECT evidence_json FROM threshold_successor_searches WHERE search_id=?',(policy.digest([cid,1,6,1]),)).fetchone()[0])
    assert ev['mode']=='continue' and len(ev['jobs'])==7 and ev['reuse'][0]['source_job_id']==f.legacy and ev['audit_sha256']==policy.digest({k:v for k,v in ev.items() if k!='audit_sha256'})
    assert all(j['deep'] and j['artifact_bytes']==64 and j['sidecar_sha256'] and j['campaign_id']==PRED for j in ev['jobs'])
    assert len(ev['predecessor_queued_job_ids'])==1 and ev['predecessor_search_id']==f.searches[(1,6,1)]
    assert db.execute('SELECT count(*) FROM threshold_reuse WHERE search_id IN (SELECT id FROM threshold_searches WHERE campaign_id=?) AND accepted=1',(cid,)).fetchone()[0]==2
    assert policy.bound_ports(db)=={8930,8931}
    assert policy.baseline_answer(db,cid)==BOILER
    assert policy.status(db)['phase_a_active'] is True
    # idempotent re-run returns the same successor
    assert policy.enqueue(db,None,DESCS,predecessor=PRED,coherence=runner.coherent)=={'campaign_id':cid,'added':0,'existing':True}
    assert db.execute("SELECT count(*) FROM jobs WHERE campaign_id=?",(cid,)).fetchone()[0]==3

def test_rejects_unqualified_or_foreign_endpoint():
    f=build()
    bad=desc(B16,16,'q16'*21+'x');bad['flare']['batch']['qualified']=False
    with pytest.raises(ValueError,match='not a qualified native batch'):
        policy.enqueue(f.db,None,{8931:bad},predecessor=PRED,coherence=runner.coherent)
    other=desc({**B16,'model_identity':'z'*64},16,'q16'*21+'x')
    with pytest.raises(ValueError,match='serves another model'):
        policy.enqueue(f.db,None,{8931:other},predecessor=PRED,coherence=runner.coherent)
    # a failed audit leaves the durable gate blocked (as in v3) and nothing else
    assert f.db.execute("SELECT count(*),sum(state='blocked') FROM threshold_campaigns").fetchone()==(2,2)
    assert f.db.execute("SELECT count(*) FROM threshold_searches WHERE campaign_id!=?",(PRED,)).fetchone()[0]==0

@pytest.mark.parametrize('tamper,reason',[
    ("UPDATE jobs SET answer='I am Zed, a pirate.' WHERE campaign_id=? AND search_id IS NOT NULL AND dose=0.125",'job_metrics_not_reproducible|anchor_response_ledger_mismatch'),
    ("UPDATE jobs SET binding_json=replace(binding_json,'r4','r9') WHERE campaign_id=? AND search_id IS NOT NULL AND dose=0.25",'job_binding_not_frozen|anchor_binding_mismatch'),
    ("UPDATE jobs SET state='running' WHERE campaign_id=? AND state='queued'",'in-flight probes'),
    ("UPDATE threshold_searches SET probes_used=probes_used+1 WHERE campaign_id=? AND sign=-1 AND neuron=5",'counters differ'),
    ("UPDATE threshold_searches SET state='complete',stop_reason='budget_exhausted' WHERE campaign_id=? AND neuron=6 AND sign=1",'unexpected state'),
    ("UPDATE jobs SET answer='changed legacy answer here' WHERE id='legacy60000000000000000000000000' AND ?=?",'reuse_source_changed'),
    ("UPDATE threshold_completion_seals SET completion_sha256='0' WHERE job_id IN (SELECT id FROM jobs WHERE campaign_id=? AND admission_id IS NOT NULL)",'job_completion_seal_mismatch'),
    ("UPDATE threshold_searches SET state='complete',stop_reason='budget_exhausted' WHERE campaign_id=? AND neuron=7 AND sign=1",'not running'),
    ("UPDATE jobs SET dose=0.3125,dose_hex='0x1.4p-2' WHERE campaign_id=? AND search_id IS NOT NULL AND dose=0.375 AND neuron=5",'job_dose_hex_mismatch|anchor_dose_sign_mismatch|anchor_program_mismatch'),
    ("UPDATE jobs SET answer=answer||'.',metrics_json=metrics_json WHERE campaign_id=? AND search_id IS NOT NULL AND dose=0.85 AND neuron=5",'anchor_response_ledger_mismatch|job_metrics_not_reproducible'),
    ("UPDATE jobs SET finished=finished+2000 WHERE campaign_id=? AND state='uncertain'",'finished before the block'),
])
def test_creation_audit_rejects_tampered_ledger(tamper,reason):
    f=build();params=(PRED,PRED) if tamper.count('?')==2 else (PRED,)
    if 'completion_seals' in tamper:
        f.db.execute('DROP TRIGGER completion_seal_immutable_update')
    f.db.execute(tamper,params);f.db.commit()
    with pytest.raises(ValueError,match=reason):
        policy.enqueue(f.db,None,DESCS,predecessor=PRED,coherence=runner.coherent,deep_modulus=1)
    assert f.db.execute("SELECT count(*) FROM threshold_campaigns WHERE state='searching'").fetchone()[0]==0

def test_calibration_or_builder_drift_refused():
    f=build();f.cal.write_bytes(b'different')
    with pytest.raises(ValueError,match='Calibration file changed'):
        policy.enqueue(f.db,None,DESCS,predecessor=PRED,coherence=runner.coherent)

def test_advance_reverifies_imported_rows_and_baseline():
    f=build();db=f.db
    cid=policy.enqueue(db,None,DESCS,predecessor=PRED,coherence=runner.coherent)['campaign_id']
    sid=policy.digest([cid,1,6,1]);pred_sid=f.searches[(1,6,1)]
    db.execute("UPDATE jobs SET metrics_json=replace(metrics_json,'boilerplate','drift') WHERE search_id=? AND dose=0.125",(pred_sid,))
    policy.advance(db,cid,sid,runner.coherent)
    assert db.execute('SELECT state,error FROM threshold_campaigns WHERE id=?',(cid,)).fetchone()==('blocked','imported_source_row_changed')
    assert db.execute('SELECT state,stop_reason FROM threshold_searches WHERE id=?',(sid,)).fetchone()==('blocked','import_integrity_mismatch')
    g=build();db=g.db
    cid=policy.enqueue(db,None,DESCS,predecessor=PRED,coherence=runner.coherent)['campaign_id']
    db.execute("UPDATE jobs SET answer=answer||' x' WHERE id=?",(policy.campaign_config(db,cid)['predecessor']['baseline']['job_id'],))
    assert policy.baseline_point(db,cid) is None and policy.baseline_answer(db,cid) is None
    policy.advance(db,cid,policy.digest([cid,1,6,-1]),runner.coherent)
    assert db.execute('SELECT error FROM threshold_campaigns WHERE id=?',(cid,)).fetchone()[0]=='baseline_source_missing_or_changed'

def test_successor_probe_completion_advances_and_completes():
    f=build();db=f.db
    cid=policy.enqueue(db,None,DESCS,predecessor=PRED,coherence=runner.coherent)['campaign_id']
    sid=policy.digest([cid,1,6,1])
    jid=db.execute("SELECT id FROM jobs WHERE search_id=? AND state='queued'",(sid,)).fetchone()[0]
    m=measured(BOILER)
    db.execute("UPDATE jobs SET state='done',answer=?,tokens=40,finish='stop',artifact='/tmp/x.npz',artifact_sha256=?,binding_json=?,port=8931,admission_id=NULL WHERE id=?",(BOILER,'a'*64,policy.encode(B16),jid))
    assert policy.finish(db,jid,runner.coherent) is True
    assert db.execute('SELECT state,probes_used,reused_observations,stop_reason FROM threshold_searches WHERE id=?',(sid,)).fetchone()==('complete',8,1,'budget_exhausted')
    assert db.execute("SELECT count(*) FROM jobs WHERE search_id=?",(sid,)).fetchone()[0]==1  # no further probe
    assert db.execute('SELECT state FROM threshold_campaigns WHERE id=?',(cid,)).fetchone()[0]=='searching'  # (1,6,-1) still running

def test_finish_on_unbound_port_blocks():
    f=build();db=f.db
    cid=policy.enqueue(db,None,DESCS,predecessor=PRED,coherence=runner.coherent)['campaign_id']
    jid=db.execute("SELECT id FROM jobs WHERE campaign_id=? AND state='queued' LIMIT 1",(cid,)).fetchone()[0]
    db.execute("UPDATE jobs SET state='done',answer=?,tokens=40,artifact='/tmp/x.npz',artifact_sha256=?,binding_json=?,port=8932 WHERE id=?",(BOILER,'a'*64,policy.encode(B4),jid))
    policy.finish(db,jid,runner.coherent)
    assert db.execute('SELECT state FROM threshold_campaigns WHERE id=?',(cid,)).fetchone()[0]=='blocked'
    with pytest.raises(policy.IntegrityError,match='not a frozen campaign binding'):
        policy.request_record(db,jid,desc(B4,4,'q4'*32),8932,{})

def test_admission_control_uses_inherited_baseline():
    f=build();db=f.db
    cid=policy.enqueue(db,None,DESCS,predecessor=PRED,coherence=runner.coherent)['campaign_id']
    doc={'schema':'identity-threshold.admission.v1','campaign_id':cid,'port':8931,'binding':B16,'advertised_capacity':16,'dispatch_width':16,'startup_receipt':{'sha256':'q16'*21+'x'}}
    aid=policy.digest(doc)
    db.execute('INSERT INTO threshold_endpoint_admissions VALUES(?,?,?,?,?)',(aid,cid,8931,policy.encode(doc),time.time()))
    db.execute("INSERT INTO threshold_endpoint_active VALUES(?,?,?,'pending_control',NULL)",(cid,8931,aid))
    for answer,expected in ((DRIFT,'blocked'),(BOILER,'active')):
        control=policy.digest([cid,'control',answer])
        db.execute("INSERT INTO jobs(id,sweep,layer,neuron,dose,dose_hex,kind,campaign_id,probe_role,target_port,state,created,admission_id,port,answer,tokens,artifact,artifact_sha256,binding_json) VALUES(?,'threshold-phase-a',-1,-1,0.0,?,'threshold:admission-control',?,'admission_baseline',8931,'done',?,?,8931,?,40,'/tmp/c.npz',?,?)",
                   (control,float(0).hex(),cid,time.time(),aid,answer,'a'*64,policy.encode(B16)))
        db.execute("UPDATE threshold_endpoint_active SET state='pending_control' WHERE campaign_id=? AND port=8931",(cid,))
        assert admissions.finish(db,control,runner.coherent) is True
        assert db.execute('SELECT state FROM threshold_endpoint_active WHERE campaign_id=? AND port=8931',(cid,)).fetchone()[0]==expected
    # frozen descriptors accept the successor's own advertised capacities, and refuse a changed one
    admissions.freeze_original_descriptors(db,cid,DESCS)
    assert set(admissions.original_descriptors(db,cid))=={8930,8931}
    with pytest.raises(admissions.AdmissionError,match='original descriptor differs'):
        admissions.freeze_original_descriptors(db,cid,{8930:DESCS[8930],8931:desc(B16,4,'q16'*21+'x',4096)})

def test_ordinary_b4_path_never_feeds_a_successor_and_admitted_path_does():
    f=build();db=f.db
    cid=policy.enqueue(db,None,DESCS,predecessor=PRED,coherence=runner.coherent)['campaign_id'];db.close()
    assert runner.claim(f.root,8932,descriptor=desc(B4,4,'q4'*32))==[]
    assert runner.claim(f.root,8931,descriptor=DESCS[8931])==[]  # no admission yet
    db=runner.connect(f.root)
    doc={'schema':'identity-threshold.admission.v1','campaign_id':cid,'port':8931,'binding':B16,'advertised_capacity':16,'dispatch_width':16,'startup_receipt':{'sha256':'q16'*21+'x'}}
    aid=policy.digest(doc)
    db.execute('INSERT INTO threshold_endpoint_admissions VALUES(?,?,?,?,?)',(aid,cid,8931,policy.encode(doc),time.time()))
    db.execute("INSERT INTO threshold_endpoint_active VALUES(?,?,?,'active',NULL)",(cid,8931,aid));db.commit();db.close()
    rows=runner.claim(f.root,8931,descriptor=DESCS[8931])
    assert len(rows)==3 and all(r[1]==1 and r[2] in (6,7) for r in rows)
    db=runner.connect(f.root)
    assert db.execute("SELECT count(*) FROM jobs WHERE campaign_id=? AND state='running' AND port=8931 AND admission_id=?",(cid,aid)).fetchone()[0]==3
    assert db.execute("SELECT count(*) FROM jobs WHERE state='queued'").fetchone()[0]==0

def test_v3_fresh_campaign_path_unchanged():
    root=Path(tempfile.mkdtemp());(root/'results').mkdir();db=runner.connect(root);policy.schema(db);admissions.schema(db)
    spec={'targets':[{'layer':3,'neuron':9,'priority':0,'target_bits':7}]}
    out=policy.enqueue(db,spec,{p:desc(B4,4,'q4'*32) for p in range(8931,8936)})
    cid=out['campaign_id'];assert out['added']==5 and out['searches']==2
    cfg=policy.campaign_config(db,cid);assert cfg['schema']==policy.SCHEMA_V3 and sorted(map(int,cfg['bindings']))==list(range(8931,8936))
    assert policy.bound_ports(db)=={8931,8932,8933,8934,8935}
    assert policy.enqueue(db,spec,{p:desc(B4,4,'q4'*32) for p in range(8931,8936)})=={'campaign_id':cid,'added':0}
    db.close()
    rows=runner.claim(root,8931,descriptor=desc(B4,4,'q4'*32));assert len(rows)==1 and rows[0][1]==-1  # the frozen B4 shard still gets its baseline
    assert runner.claim(root,8931,descriptor=desc(B4,4,'other'*12+'xxxx'))==[]  # a different B4 service on the same port gets nothing
    db=runner.connect(root)
    with pytest.raises(ValueError,match='Five qualified original B4'):
        db.execute("DELETE FROM threshold_campaigns");db.commit()
        policy.enqueue(db,spec,{p:desc(B4,4,'q4'*32) for p in range(8931,8935)})


def test_anchor_rejects_missing_or_foreign_evidence():
    f=build()
    victim=f.db.execute("SELECT artifact FROM jobs WHERE campaign_id=? AND search_id IS NOT NULL AND dose=0.25 AND neuron=6",(PRED,)).fetchone()[0]
    side=Path(victim).with_suffix('.jsonl');report=json.loads(side.read_text());report['request']['metadata']['job_id']='someone-else';side.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='anchor_request_job_mismatch|anchor_request_hash_mismatch'):
        policy.enqueue(f.db,None,DESCS,predecessor=PRED,coherence=runner.coherent,deep_modulus=1)
    g=build();victim=g.db.execute("SELECT artifact FROM jobs WHERE campaign_id=? AND search_id IS NOT NULL AND dose=0.25 AND neuron=6",(PRED,)).fetchone()[0]
    Path(victim).unlink()
    with pytest.raises(ValueError,match='anchor_missing_artifact_or_sidecar'):
        policy.enqueue(g.db,None,DESCS,predecessor=PRED,coherence=runner.coherent)  # shallow pass catches a missing artifact for every row

def test_only_uncertain_sample_blocks_can_be_continued():
    f=build();f.db.execute("UPDATE threshold_campaigns SET error='audit_integrity: Audit source changed' WHERE id=?",(PRED,));f.db.commit()
    with pytest.raises(ValueError,match='Only a campaign blocked by an uncertain sample'):
        policy.enqueue(f.db,None,DESCS,predecessor=PRED,coherence=runner.coherent)

def test_dead_import_gate_is_recovered_live_gate_is_not():
    f=build();db=f.db;gate=policy.digest(['threshold-successor-gate',PRED]);boot=policy._boot_id()
    db.execute("INSERT INTO threshold_campaigns VALUES(?,?,'auditing',?,NULL)",(gate,policy.encode({'schema':'identity-threshold.audit-pending.v1','predecessor':PRED,'pid':os.getpid(),'boot_id':boot}),time.time()));db.commit()
    with pytest.raises(ValueError,match='already running'):
        policy.enqueue(db,None,DESCS,predecessor=PRED,coherence=runner.coherent)
    db.execute("UPDATE threshold_campaigns SET config_json=? WHERE id=?",(policy.encode({'schema':'identity-threshold.audit-pending.v1','predecessor':PRED,'pid':2**22-1,'boot_id':boot}),gate));db.commit()
    out=policy.enqueue(db,None,DESCS,predecessor=PRED,coherence=runner.coherent)
    assert out['added']==3 and db.execute("SELECT count(*) FROM threshold_campaigns WHERE state='auditing'").fetchone()[0]==0

def test_successor_of_a_successor_continues_the_chain():
    f=build();db=f.db
    s1=policy.enqueue(db,None,DESCS,predecessor=PRED,coherence=runner.coherent)['campaign_id']
    # S1 runs one probe to completion on 8931 and then loses one at a handover: blocked exactly like the root
    done_sid=policy.digest([s1,1,6,-1]);jid=db.execute("SELECT id FROM jobs WHERE search_id=? AND state='queued'",(done_sid,)).fetchone()[0]
    art=f.art/f'{jid}.npz';art.write_bytes(b'x'*64)
    db.execute("UPDATE jobs SET state='done',answer=?,tokens=40,finish='stop',artifact=?,artifact_sha256=?,binding_json=?,port=8931,finished=?,request_sha256=?,program_sha256=? WHERE id=?",
               (BOILER,str(art),hashlib.sha256(b'x'*64).hexdigest(),policy.encode(B16),time.time()-50,'q'*64,'p'*64,jid))
    Path(art).with_suffix('.jsonl').write_text('{}')
    assert policy.finish(db,jid,runner.coherent) is True
    assert db.execute('SELECT state FROM threshold_searches WHERE id=?',(done_sid,)).fetchone()[0]=='complete'
    lost=db.execute("SELECT id FROM jobs WHERE campaign_id=? AND state='queued' AND dose=0.4375",(s1,)).fetchone()[0]
    db.execute("UPDATE jobs SET state='uncertain',finished=?,port=8930,error='HTTPError 500' WHERE id=?",(time.time()-10,lost));policy.finish(db,lost,runner.coherent);db.commit()
    assert db.execute('SELECT state,error FROM threshold_campaigns WHERE id=?',(s1,)).fetchone()==('blocked','Uncertain sample '+lost)
    out=policy.enqueue(db,None,DESCS,predecessor=s1,coherence=runner.coherent,deep_modulus=10**9)
    s2=out['campaign_id'];cfg=policy.campaign_config(db,s2);pred=cfg['predecessor']
    assert out['imported_complete']==4 and out['continued']==2 and out['superseded']==1 and out['added']==2
    assert pred['lineage']==[s1,PRED] and pred['root_campaign_id']==PRED and pred['baseline']['job_id']==policy.campaign_config(db,s1)['predecessor']['baseline']['job_id']
    assert pred['probes_carried']==44+1 and pred['uncertain_probes_rerun_by_successor']==1
    assert db.execute("SELECT state FROM threshold_campaigns WHERE id=?",(s2,)).fetchone()[0]=='searching'
    # the chained search owns rows from both ancestors and re-proposes the lost magnitude
    ev=json.loads(db.execute('SELECT evidence_json FROM threshold_successor_searches WHERE search_id=?',(policy.digest([s2,1,6,-1]),)).fetchone()[0])
    assert {j['campaign_id'] for j in ev['jobs']}=={PRED,s1} and len(ev['jobs'])==8 and ev['mode']=='complete'
    assert db.execute("SELECT dose,state FROM jobs WHERE search_id=?",(policy.digest([s2,1,6,1]),)).fetchall()==[(0.4375,'queued')]
    assert db.execute("SELECT state FROM jobs WHERE id=?",(lost,)).fetchone()[0]=='uncertain'
    with pytest.raises(ValueError,match='must be resolved first'):
        policy.enqueue(db,None,DESCS,predecessor=PRED,coherence=runner.coherent)  # the root already has its chain (S2 is active)
    assert policy.enqueue(db,None,DESCS,predecessor=s1,coherence=runner.coherent)=={'campaign_id':s2,'added':0,'existing':True}

def test_foreign_b4_service_on_a_bound_port_gets_nothing():
    f=build();db=f.db
    cid=policy.enqueue(db,None,DESCS,predecessor=PRED,coherence=runner.coherent)['campaign_id'];db.close()
    assert runner.claim(f.root,8931,descriptor=desc(B4,4,'q4'*32))==[]
    db=runner.connect(f.root)
    assert db.execute("SELECT count(*) FROM jobs WHERE campaign_id=? AND state='queued'",(cid,)).fetchone()[0]==3
    assert db.execute("SELECT state FROM threshold_campaigns WHERE id=?",(cid,)).fetchone()[0]=='searching'
