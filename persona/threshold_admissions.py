"""Append-only threshold endpoint admission; original Phase A policy stays frozen.

v4 (2026-10-30): frozen descriptors carry their own advertised capacity (v3 configs default to B4) and the
admission control answer is compared with the campaign's baseline through the policy (a v4 successor inherits
its predecessor's validated baseline).

CPU preparation only. Deploy only after independent review and a drained runner.
"""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import persona_threshold as policy
import claim_guard

class AdmissionError(ValueError): pass

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def schema(db):
    if 'admission_id' not in {r[1] for r in db.execute('PRAGMA table_info(jobs)')}:
        db.execute('ALTER TABLE jobs ADD COLUMN admission_id TEXT')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS threshold_original_descriptors(
        campaign_id TEXT NOT NULL, port INTEGER NOT NULL, descriptor_json TEXT NOT NULL,
        descriptor_sha256 TEXT NOT NULL, PRIMARY KEY(campaign_id,port));
      CREATE TABLE IF NOT EXISTS threshold_endpoint_admissions(
        id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL, port INTEGER NOT NULL,
        document_json TEXT NOT NULL, created REAL NOT NULL);
      CREATE TABLE IF NOT EXISTS threshold_endpoint_active(
        campaign_id TEXT NOT NULL, port INTEGER NOT NULL, admission_id TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('pending_control','active','draining','blocked')),
        reason TEXT, PRIMARY KEY(campaign_id,port));
      CREATE TABLE IF NOT EXISTS threshold_completion_seals(job_id TEXT PRIMARY KEY, completion_sha256 TEXT NOT NULL);
      CREATE TRIGGER IF NOT EXISTS completion_seal_immutable_update BEFORE UPDATE ON threshold_completion_seals BEGIN SELECT RAISE(ABORT,'immutable completion seal'); END;
      CREATE TRIGGER IF NOT EXISTS completion_seal_immutable_delete BEFORE DELETE ON threshold_completion_seals BEGIN SELECT RAISE(ABORT,'immutable completion seal'); END;
      CREATE TRIGGER IF NOT EXISTS admission_immutable_update BEFORE UPDATE ON threshold_endpoint_admissions BEGIN SELECT RAISE(ABORT,'immutable admission'); END;
      CREATE TRIGGER IF NOT EXISTS admission_immutable_delete BEFORE DELETE ON threshold_endpoint_admissions BEGIN SELECT RAISE(ABORT,'immutable admission'); END;
      CREATE TRIGGER IF NOT EXISTS original_descriptor_immutable_update BEFORE UPDATE ON threshold_original_descriptors BEGIN SELECT RAISE(ABORT,'immutable descriptor'); END;
      CREATE TRIGGER IF NOT EXISTS original_descriptor_immutable_delete BEFORE DELETE ON threshold_original_descriptors BEGIN SELECT RAISE(ABORT,'immutable descriptor'); END;
    ''')

def freeze_original_descriptors(db,cid,descriptors):
    row=db.execute('SELECT config_json FROM threshold_campaigns WHERE id=?',(cid,)).fetchone()
    if not row:raise AdmissionError('campaign missing')
    config=json.loads(row[0]); expected=config['bindings']
    if set(map(str,descriptors))!=set(expected):raise AdmissionError('original descriptor coverage')
    values=[]
    for port,d in sorted(descriptors.items()):
        b=d['flare']['batch'];e=expected[str(port)]
        if not b.get('qualified') or b['binding']!=e['binding'] or b['qualification_sha256']!=e['qualification_sha256'] or b['max_batch_size']!=e.get('max_batch_size',4):
            raise AdmissionError('original descriptor differs from frozen campaign')
        values.append((cid,int(port),policy.encode(d),policy.digest(d)))
    with db:
        for v in values:
            old=db.execute('SELECT descriptor_json,descriptor_sha256 FROM threshold_original_descriptors WHERE campaign_id=? AND port=?',v[:2]).fetchone()
            if old and old!=(v[2],v[3]):raise AdmissionError('original snapshot changed')
            if not old:db.execute('INSERT INTO threshold_original_descriptors VALUES(?,?,?,?)',v)

def original_descriptors(db,cid):
    row=db.execute('SELECT config_json FROM threshold_campaigns WHERE id=?',(cid,)).fetchone()
    if not row:raise AdmissionError('campaign missing')
    expected=json.loads(row[0])['bindings'];rows=db.execute('SELECT port,descriptor_json,descriptor_sha256 FROM threshold_original_descriptors WHERE campaign_id=?',(cid,)).fetchall()
    out={}
    for port,raw,h in rows:
        d=json.loads(raw);b=d['flare']['batch'];e=expected.get(str(port))
        if not e or policy.digest(d)!=h or b['binding']!=e['binding'] or b['qualification_sha256']!=e['qualification_sha256'] or not b['qualified'] or b['max_batch_size']!=e.get('max_batch_size',4):raise AdmissionError('frozen descriptor integrity')
        out[port]=d
    if set(map(str,out))!=set(expected):raise AdmissionError('frozen descriptors incomplete')
    return out

def load_receipt(path,expected_hash):
    if sha(path)!=expected_hash:raise AdmissionError('receipt hash mismatch')
    r=json.loads(Path(path).read_text())
    if not (r.get('verified') is True and r.get('passed') is True):raise AdmissionError('receipt not verified and passed')
    return r

RESOURCE_RESOLVER_SHA='90ab2a92d17deb0a7c519523cea7c9c11d5859f7a627bac41081932427a0097a'
RESOURCE_RECEIPT_SHA='8da7d835a09924595d9b845a5880c365d31c9a393602fcbd41526ad8050b540a'

def resource_sources(adapter,server):
    resolver=Path(adapter).with_name('resource_quiescence_resolver.py')
    receipt=Path(server).parent/'resource-quiescence-receipt.json'
    if sha(resolver)!=RESOURCE_RESOLVER_SHA or sha(receipt)!=RESOURCE_RECEIPT_SHA:raise AdmissionError('resource proof source/receipt differs')
    return {'resolver_path':str(resolver),'resolver_sha256':RESOURCE_RESOLVER_SHA,'receipt_path':str(receipt),'receipt_sha256':RESOURCE_RECEIPT_SHA}

def joint_barrier_held(token,epoch,inspections,port):
    if not token or token.port!=port: return False
    try: return claim_guard.ready_held(token,epoch,inspections)
    except (ValueError,TypeError,OSError) as exc: raise AdmissionError('port barrier proof invalid') from exc

def verify_same_service_attach(lace_db,worker_manifest,descriptor,uuid,startup_hash):
    """Initial 8930 attach only: prove current live lace uses this exact service."""
    import sqlite3
    import urllib.request
    path=Path(lace_db).resolve();connection=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    try:
        value=connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        if value is None:raise AdmissionError('active lace manifest missing')
        manifest=json.loads(value[0]);url='http://127.0.0.1:8930';route=manifest['routing'];endpoint=manifest['endpoints'][url]
        batch=descriptor['flare']['batch']
        if (route['wide_endpoint']!=url or url not in route['persona_endpoints'] or endpoint['binding']!=batch['binding']
            or endpoint['qualification_sha256']!=startup_hash or endpoint['width']!=32 or endpoint['logical_context_tokens']!=1024):
            raise AdmissionError('active lace service/binding differs')
    finally:connection.close()
    wm=json.loads(Path(worker_manifest).read_text());pid=wm['pid']
    if wm['gpu_uuid']!=uuid or wm['runtime_identity']!=batch['binding']['runtime_identity'] or wm['prepared_identity']!=batch['binding']['prepared_identity'] or wm['model_identity']!=batch['binding']['model_identity']:
        raise AdmissionError('worker manifest differs')
    if not Path(f'/proc/{pid}').is_dir():raise AdmissionError('current worker absent')
    argv=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if not any(Path(x.decode(errors='replace')).name=='persona_wide_server.py' for x in argv if x):raise AdmissionError('worker source process differs')
    # Re-read model descriptor; do not assert no queued lace or interrupt it.
    live=json.load(urllib.request.urlopen('http://127.0.0.1:8930/v1/models',timeout=10))['data'][0]
    if live['flare']['batch']['binding']!=batch['binding'] or live['flare']['batch']['qualification_sha256']!=startup_hash or not live['flare']['batch']['qualified']:
        raise AdmissionError('live descriptor changed')
    return {'mode':'same_existing_lace_service','lace_campaign_id':manifest['campaign_id'],'lace_manifest_sha256':policy.digest(manifest),'worker_manifest_sha256':sha(worker_manifest),'worker_pid':pid,'lease_id':wm['lease_id'],'gpu_uuid':uuid}

def verify_threshold_scope(receipt,binding,width,context,max_tokens,startup_sha,server_sha,probe_sha):
    if receipt.get('wide_binding')!=binding or receipt.get('checks')!=44 or 'greedy 160 tokens' not in receipt.get('scope',''):
        raise AdmissionError('long threshold receipt differs')
    schema=receipt.get('schema')
    if schema=='identity-threshold.b4-b32-parity.v1':return 'b32'
    if schema=='identity-threshold.b4-b16-4k-parity.v1':
        if (width!=16 or context!=4096 or max_tokens!=160 or receipt.get('dispatch_width')!=16 or receipt.get('context')!=4096 or receipt.get('max_tokens')!=160 or receipt.get('startup_sha256')!=startup_sha or receipt.get('server_sha256')!=server_sha or receipt.get('probe_sha256')!=probe_sha):
            raise AdmissionError('B16/4K scope or source differs')
        return 'b16-4k'
    raise AdmissionError('unknown threshold qualification schema')


def admit(db,cid,port,uuid,descriptor,width,context,max_tokens,startup,long_parity,server,probe,runner,adapter,telemetry,guard_token,guard_epoch,inspections,predecessor=None,initial_attach=None):
    startup=(str(Path(startup[0]).resolve()),startup[1]);long_parity=(str(Path(long_parity[0]).resolve()),long_parity[1])
    server,probe,runner,adapter,telemetry=(str(Path(x).resolve()) for x in (server,probe,runner,adapter,telemetry))
    row=db.execute('SELECT config_json,state FROM threshold_campaigns WHERE id=?',(cid,)).fetchone()
    if not row or row[1] not in ('baseline','searching'):raise AdmissionError('campaign must be active')
    config=json.loads(row[0]);batch=descriptor['flare']['batch'];binding=batch['binding']
    if not batch.get('supported') or not batch.get('qualified') or binding['model_identity']!=next(iter(config['bindings'].values()))['binding']['model_identity']:
        raise AdmissionError('candidate descriptor is not qualified/model bound')
    if type(width) is not int or width not in (1,2,4,8,16,32) or width>batch['max_batch_size'] or context!=descriptor['flare']['logical_context_tokens'] or max_tokens!=config['max_tokens'] or max_tokens>context:
        raise AdmissionError('dispatch/context/output capability invalid')
    q=load_receipt(startup[0],startup[1]);t=load_receipt(long_parity[0],long_parity[1])
    if (q.get('schema')!='character-creation.batch-service-qualification.v1' or q.get('binding')!=binding or q.get('gpu_uuid')!=uuid or q.get('context')!=context or q.get('max_active')!=batch['max_batch_size'] or q.get('physical_pages')!=batch['physical_pages'] or q.get('page_tokens')!=batch['page_tokens'] or q.get('server_sha256')!=sha(server) or q.get('probe_sha256')!=sha(probe) or batch['qualification_sha256']!=startup[1]):
        raise AdmissionError('startup receipt differs')
    verify_threshold_scope(t,binding,width,context,max_tokens,startup[1],sha(server),sha(probe))
    if config['policy_sha256']!=sha(policy.__file__) or config['auditor_sha256']!=sha(policy.reuse_audit.__file__) or config['metric_sha256']!=policy.METRIC_SHA:
        raise AdmissionError('frozen policy changed')
    original_descriptors(db,cid)
    try:
        verify_receipt_evidence(startup[0],long_parity[0],telemetry,server,probe,uuid)
    except (OSError,KeyError,ValueError,ImportError) as exc:
        raise AdmissionError('independent receipt evidence validation failed') from exc
    if initial_attach is None:
        if not guard_token or getattr(guard_token,'port',None)!=port: raise AdmissionError('port guard token missing')
        mode={'mode':'joint_barrier','guard_authority':str(guard_token.path),'guard_epoch':guard_epoch,'resource_proof':resource_sources(adapter,server)}
    else:
        if port!=8930 or guard_token is not None or guard_epoch is not None or inspections is not None:raise AdmissionError('invalid same-service attach scope')
        mode=verify_same_service_attach(initial_attach['lace_db'],initial_attach['worker_manifest'],descriptor,uuid,startup[1])
        if mode['lease_id']!=q.get('workboard_job_id'):raise AdmissionError('startup lease differs from current worker')
    doc={'schema':'identity-threshold.admission.v1','campaign_id':cid,'port':port,'gpu_uuid':uuid,'binding':binding,'advertised_capacity':batch['max_batch_size'],'dispatch_width':width,'context':context,'max_tokens':max_tokens,'sampling':'greedy','startup_receipt':{'path':str(startup[0]),'sha256':startup[1]},'threshold_receipt':{'path':str(long_parity[0]),'sha256':long_parity[1]},'telemetry':{'path':str(telemetry),'sha256':sha(telemetry)},'server_path':str(server),'server_sha256':sha(server),'probe_path':str(probe),'probe_sha256':sha(probe),'runner_sha256':sha(runner),'adapter_sha256':sha(adapter),'policy_sha256':config['policy_sha256'],'guard_source_sha256':sha(claim_guard.__file__),'resource_resolver_sha256':'90ab2a92d17deb0a7c519523cea7c9c11d5859f7a627bac41081932427a0097a','resource_receipt_sha256':'8da7d835a09924595d9b845a5880c365d31c9a393602fcbd41526ad8050b540a','admission_mode':mode,'predecessor':predecessor}
    aid=policy.digest(doc);control=policy.digest([cid,'admission-control',aid]);db.execute('BEGIN IMMEDIATE')
    try:
        if initial_attach is None and not joint_barrier_held(guard_token,guard_epoch,inspections,port): raise AdmissionError('durable joint dispatcher barrier unavailable')
        current=db.execute('SELECT admission_id,state FROM threshold_endpoint_active WHERE campaign_id=? AND port=?',(cid,port)).fetchone()
        if initial_attach is not None and current is not None:
            previous=db.execute('SELECT document_json FROM threshold_endpoint_admissions WHERE id=?',(current[0],)).fetchone()
            prior=json.loads(previous[0]) if previous else {}
            pending=db.execute("SELECT id FROM jobs WHERE admission_id=? AND probe_role='admission_baseline' AND state='queued'",(current[0],)).fetchall()
            other=db.execute("SELECT 1 FROM jobs WHERE admission_id=? AND probe_role!='admission_baseline' LIMIT 1",(current[0],)).fetchone()
            if (current[0]!=predecessor or current[1]!='draining' or prior.get('admission_mode',{}).get('mode')!='same_existing_lace_service'
                or prior.get('binding')!=binding or len(pending)!=1 or other):raise AdmissionError('same-service revision requires one unsubmitted control and no probes')
            db.execute("UPDATE jobs SET state='superseded',error='same-service priority correction' WHERE id=?",(pending[0][0],))
        if (current[0] if current else None)!=predecessor or (current and current[1]!='draining'):
            raise AdmissionError('predecessor must be drained')
        if db.execute("SELECT 1 FROM jobs WHERE port=? AND state='running' LIMIT 1",(port,)).fetchone():raise AdmissionError('port still has in-flight identity jobs')
        if db.execute("SELECT 1 FROM jobs WHERE campaign_id=? AND probe_role='baseline' AND target_port=? AND state!='done' LIMIT 1",(cid,port)).fetchone():raise AdmissionError('original port baseline is unresolved')
        if current and db.execute("SELECT 1 FROM jobs WHERE admission_id=? AND state='running' LIMIT 1",(current[0],)).fetchone():raise AdmissionError('predecessor still has in-flight jobs')
        db.execute('INSERT INTO threshold_endpoint_admissions VALUES(?,?,?,?,?)',(aid,cid,port,policy.encode(doc),time.time()))
        db.execute('INSERT OR REPLACE INTO threshold_endpoint_active VALUES(?,?,?,\'pending_control\',NULL)',(cid,port,aid))
        db.execute('''INSERT INTO jobs(id,sweep,layer,neuron,dose,dose_hex,level,kind,campaign_id,search_id,probe_role,target_port,state,created,admission_id)
          VALUES(?, 'threshold-phase-a',-1,-1,0.0,?,NULL,'threshold:admission-control',?,NULL,'admission_baseline',?,'queued',?,?)''',
          (control,float(0).hex(),cid,port,time.time(),aid))
        db.commit();return aid
    except BaseException:
        db.rollback();raise

def active(db,cid,port,descriptor):
    row=db.execute('''SELECT a.id,a.document_json,s.state FROM threshold_endpoint_active s JOIN threshold_endpoint_admissions a ON a.id=s.admission_id
      WHERE s.campaign_id=? AND s.port=?''',(cid,port)).fetchone()
    if not row:return None
    aid,raw,state=row;doc=json.loads(raw);b=descriptor['flare']['batch']
    if policy.digest(doc)!=aid or doc['binding']!=b['binding'] or doc['advertised_capacity']!=b['max_batch_size'] or doc['startup_receipt']['sha256']!=b['qualification_sha256'] or not b.get('qualified'):
        raise AdmissionError('active endpoint differs from immutable admission')
    return doc | {'id':aid,'state':state}

def request_record(db,identity,descriptor,port,request,runner_path):
    row=db.execute('SELECT campaign_id,layer,admission_id FROM jobs WHERE id=?',(identity,)).fetchone()
    if not row or not row[0] or not row[2]:return policy.request_record(db,identity,descriptor,port,request)
    cid,layer,aid=row
    saved=db.execute('SELECT document_json FROM threshold_endpoint_admissions WHERE id=? AND campaign_id=? AND port=?',(aid,cid,port)).fetchone()
    if not saved:raise policy.IntegrityError('missing admission')
    doc=json.loads(saved[0]);b=descriptor['flare']['batch'];config=json.loads(db.execute('SELECT config_json FROM threshold_campaigns WHERE id=?',(cid,)).fetchone()[0])
    try:
        if (policy.digest(doc)!=aid or doc['binding']!=b['binding'] or not b['qualified'] or b['qualification_sha256']!=doc['startup_receipt']['sha256']
            or doc['runner_sha256']!=sha(runner_path) or doc['adapter_sha256']!=sha(__file__) or doc['policy_sha256']!=sha(policy.__file__)
            or doc['guard_source_sha256']!=sha(claim_guard.__file__) or doc['server_sha256']!=sha(doc['server_path']) or doc['probe_sha256']!=sha(doc['probe_path'])
            or config['auditor_sha256']!=sha(policy.reuse_audit.__file__) or config['metric_sha256']!=policy.METRIC_SHA):
            raise policy.IntegrityError('admission/source binding changed')
        for key in ('startup_receipt','threshold_receipt'):
            load_receipt(doc[key]['path'],doc[key]['sha256'])
        proof=doc['admission_mode'].get('resource_proof')
        if proof and (sha(proof['resolver_path'])!=proof['resolver_sha256'] or sha(proof['receipt_path'])!=proof['receipt_sha256']):raise policy.IntegrityError('resource proof changed')
        if config['builder_file'] and sha(config['builder_file'])!=config['builder_sha256']:raise policy.IntegrityError('builder changed')
        for path,h in config['calibration_files'].items():
            if layer<0 or Path(path).name==f'layer_{layer:02d}.npz':
                if sha(path)!=h:raise policy.IntegrityError('calibration changed')
    except (OSError,AdmissionError,KeyError) as exc:
        raise policy.IntegrityError('admission evidence unavailable') from exc
    if not (request['messages']==[{'role':'user','content':config['prompt']}] and request['seed']==config['seed'] and request['max_tokens']==config['max_tokens'] and request['temperature']==0 and request['stream'] is False and request['chat_template_kwargs']=={'enable_thinking':False}):
        raise policy.IntegrityError('request settings changed')
    db.execute('UPDATE jobs SET request_json=?,request_sha256=? WHERE id=?',(policy.encode({k:v for k,v in request.items() if k!='steering'}),policy.digest(request),identity))

def finish(db,identity,coherence):
    row=db.execute('SELECT campaign_id,search_id,state,answer,tokens,binding_json,artifact,artifact_sha256,port,admission_id,probe_role,metrics_json FROM jobs WHERE id=?',(identity,)).fetchone()
    if not row or not row[0] or not row[9]:return policy.finish(db,identity,coherence)
    cid,sid,state,answer,tokens,binding,artifact,artifact_sha,port,aid,role,metrics_json=row
    saved=db.execute('SELECT document_json FROM threshold_endpoint_admissions WHERE id=? AND campaign_id=? AND port=?',(aid,cid,port)).fetchone()
    if state!='done' or not saved:
        db.execute("UPDATE threshold_campaigns SET state='blocked',error=? WHERE id=?",('Uncertain admitted sample '+identity,cid))
        if sid:db.execute("UPDATE threshold_searches SET state='uncertain',stop_reason='uncertain_no_retry' WHERE id=?",(sid,))
        return True
    doc=json.loads(saved[0])
    if policy.digest(doc)!=aid or not binding or json.loads(binding)!=doc['binding'] or not artifact or not artifact_sha:
        db.execute("UPDATE threshold_campaigns SET state='blocked',error=? WHERE id=?",('Admission binding/artifact differs '+identity,cid));return True
    m=policy.metrics.measure(answer);m['coherent']=int(coherence(answer,tokens))
    seal=policy.digest([cid,sid,state,answer,tokens,binding,artifact,artifact_sha,port,aid,role])
    saved_seal=db.execute('SELECT completion_sha256 FROM threshold_completion_seals WHERE job_id=?',(identity,)).fetchone()
    if metrics_json is not None or saved_seal:
        if metrics_json!=policy.encode(m) or not saved_seal or saved_seal[0]!=seal:
            db.execute("UPDATE threshold_campaigns SET state='blocked',error=? WHERE id=?",('Contradictory duplicate '+identity,cid))
        return True
    db.execute('INSERT INTO threshold_completion_seals VALUES(?,?)',(identity,seal))
    db.execute('UPDATE jobs SET metrics_json=?,coherent=? WHERE id=?',(policy.encode(m),m['coherent'],identity))
    if role=='admission_baseline':
        base=policy.baseline_answer(db,cid)
        if base is None or answer!=base or m['mode']!='boilerplate' or not m['coherent']:
            db.execute("UPDATE threshold_endpoint_active SET state='blocked',reason='baseline mismatch' WHERE campaign_id=? AND port=? AND admission_id=?",(cid,port,aid));return True
        db.execute("UPDATE threshold_endpoint_active SET state='active' WHERE campaign_id=? AND port=? AND admission_id=? AND state='pending_control'",(cid,port,aid));return True
    if sid is None:raise policy.IntegrityError('admitted search has no search ID')
    if db.execute('SELECT state FROM threshold_campaigns WHERE id=?',(cid,)).fetchone()[0]!='blocked':policy.advance(db,cid,sid,coherence)
    return True

def drain(db,cid,port,aid):
    db.execute('BEGIN IMMEDIATE')
    try:
        changed=db.execute("UPDATE threshold_endpoint_active SET state='draining',reason='operator drain' WHERE campaign_id=? AND port=? AND admission_id=? AND state IN ('pending_control','active')",(cid,port,aid)).rowcount
        if changed!=1:raise AdmissionError('active admission not drainable')
        inflight=db.execute("SELECT count(*) FROM jobs WHERE admission_id=? AND state='running'",(aid,)).fetchone()[0]
        db.commit();return inflight
    except BaseException:
        db.rollback();raise

def verify_receipt_evidence(startup,long_parity,telemetry,server,probe,uuid):
    """Re-run independent readback for the card-0 qualification evidence.

    These specific validators cover the named card-0 fixture catalog. Other
    cards require their own validated catalog/validator and are not admitted by
    this implementation.
    """
    q=json.loads(Path(startup).read_text());t=json.loads(Path(long_parity).read_text())
    if t.get('schema')=='identity-threshold.b4-b16-4k-parity.v1':
        return verify_b16_receipt_evidence(startup,long_parity,telemetry,server,probe,uuid,q,t)
    trusted={'projection_receipt.py':'f565dc888c51abaea74c9645f920fbe7bfd8d779ec8a25d6434a8eb485d9a252',
             'projection_threshold_validate.py':'6419e64f2c77d873009b21a175fcfd4c8ffe71bd76fbf05a434fa108264f8b24'}
    evidence_dir=Path(startup).resolve().parent
    for name,digest in trusted.items():
        if sha(evidence_dir/name)!=digest:raise AdmissionError('trusted validator source changed')
    import sys
    if str(evidence_dir) not in sys.path:sys.path.insert(0,str(evidence_dir))
    import projection_receipt as qr
    import projection_threshold_validate as tv
    if Path(qr.__file__).resolve()!=(evidence_dir/'projection_receipt.py').resolve() or Path(tv.__file__).resolve()!=(evidence_dir/'projection_threshold_validate.py').resolve():raise AdmissionError('validator import differs')
    q=json.loads(Path(startup).read_text());t=json.loads(Path(long_parity).read_text())
    if q.get('validator_sha256')!=trusted['projection_receipt.py'] or t.get('validator_sha256')!=trusted['projection_threshold_validate.py']:raise AdmissionError('untrusted validator identity')
    if (q.get('b4_short_reference_sha256')!='6f4a3da088c6b0a33e9551860ef9ad3757dbbab86d56502adab760f7568a9c08'
        or q.get('b4_edge_reference_sha256')!='cea5b08ad08caa9737e31f85196050a5adbd7fdbf0469a8806b8a68821e13bb2'
        or t.get('manifest_sha256')!='9260405a9a8c788ed74b2c9aa44a68089228c4d4f52a6c5211d00cc487b9007e'
        or t.get('reference_sha256')!='dce6dd34e1953a75034f95d5ded68567fd2c3f0914393e214e14a3aef7148873'):
        raise AdmissionError('trusted B4 fixture identity differs')
    fresh=qr.build(telemetry,server,probe,uuid,q['workboard_job_id'])
    for k in fresh:
        if k!='created_at' and fresh[k]!=q.get(k):raise AdmissionError('startup evidence readback mismatch: '+k)
    out=Path(startup).parent;manifest=out/'threshold-parity-requests.json';reference=out/'threshold-parity-reference.json';wide=out/'threshold-parity-projection.json'
    if any(sha(p)!=t[k] for p,k in ((manifest,'manifest_sha256'),(reference,'reference_sha256'),(wide,'wide_sha256'))):raise AdmissionError('long evidence hash mismatch')
    plan=json.loads(manifest.read_text());ref=json.loads(reference.read_text());new=json.loads(wide.read_text());requests=plan['requests']
    if len(requests)!=44 or len(ref['evidence'])!=44 or len(new['evidence'])!=44 or t['checks']!=44:raise AdmissionError('long evidence coverage')
    for phase,data,width,gpu in [('reference',ref,4,'GPU-8747fa38-17e6-3fb3-f9da-4ebb5ac128a1'),('projection',new,32,uuid)]:
        d=data['descriptor'];batch=d['flare']['batch'];binding=batch['binding'];journal=out/f'threshold-parity-{phase}.jsonl'
        if data['manifest_sha256']!=sha(manifest) or data['journal_sha256']!=sha(journal) or not batch.get('qualified') or batch['max_batch_size']!=width:
            raise AdmissionError('long descriptor or journal mismatch')
        lines=[json.loads(s) for s in journal.read_text().splitlines()];submitted=[r for r in lines if r['state']=='submitted'];done=[r for r in lines if r['state']=='completed'];offsets=list(range(0,44,width))
        if [r['offset'] for r in submitted]!=offsets or [r['offset'] for r in done]!=offsets:raise AdmissionError('long journal coverage')
        collected=[]
        for start,sub,complete in zip(offsets,submitted,done):
            subset=requests[start:start+width];rows=complete['evidence'];pages=[]
            if sub['request_sha256']!=[policy.digest(req) for req in subset] or len(rows)!=len(subset):raise AdmissionError('long request coverage')
            for i,(row,req) in enumerate(zip(rows,subset)):
                b=row['batch'];n=len(subset)
                if b['id']!=sub['batch_id'] or b['index']!=i or b['physical_slot']!=i or b['requests']!=n or b['decode_capacity']!=1<<(n-1).bit_length():raise AdmissionError('long slot ownership')
                pages.extend(b['physical_pages']);tv.verify(row,req,binding,gpu);collected.append(row)
            if len(pages)!=len(set(pages)):raise AdmissionError('long page ownership')
        if collected!=data['evidence']:raise AdmissionError('long summary mismatch')
    if new['descriptor']['flare']['batch']['binding']!=q['binding'] or t['wide_binding']!=q['binding'] or t['reference_binding']!=ref['descriptor']['flare']['batch']['binding']:
        raise AdmissionError('long binding differs')
    if any(a['arrays']!=b['arrays'] or a['finish_reason']!=b['finish_reason'] for a,b in zip(ref['evidence'],new['evidence'])):raise AdmissionError('long parity differs')

def verify_b16_receipt_evidence(startup,long_parity,telemetry,server,probe,uuid,q,t):
    """Re-read the explicitly pinned B16/4K fixture, without weakening B32."""
    import importlib.util
    evidence_dir=Path(startup).resolve().parent
    trusted={'conductor_recovery_receipt.py':'2c15c08f4cd5be70447b5eb2b73d904ed273d284c2488b870b8de5ebe64458bb',
             'conductor_recovery_context_receipt.py':'b07627e41a87597e385e0b7fe6c333635ed9993e6474131a87a1f4477fe61ff2',
             'threshold_b16_4k_validate.py':'c1a50caf363e3a5939ce6b102cfa2e7c278b82a6dcdee32559ae503b0080c918',
             'wide_qualify.py':'dc051488c53fbdd3c88d16a23735aae2d46a5973ddf46ace125416255085ab13'}
    for name,digest in trusted.items():
        if sha(evidence_dir/name)!=digest:raise AdmissionError('B16 validator source changed')
    if q.get('validator_sha256')!=trusted['conductor_recovery_receipt.py'] or t.get('validator_sha256')!=trusted['threshold_b16_4k_validate.py'] or t.get('gpu_uuid')!=uuid or t.get('workboard_job_id')!=q.get('workboard_job_id'):
        raise AdmissionError('B16 validator or lease differs')
    import sys
    if str(evidence_dir) not in sys.path:sys.path.insert(0,str(evidence_dir))
    def module(name):
        path=evidence_dir/name;spec=importlib.util.spec_from_file_location('admission_'+path.stem,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
    capacity=evidence_dir/'conductor-recovery-capacity-receipt.json'
    if sha(capacity)!=q.get('capacity_receipt_sha256'):raise AdmissionError('B16 capacity receipt differs')
    startup_fresh=module('conductor_recovery_receipt.py').build(telemetry,server,probe,uuid,q['workboard_job_id'],capacity)
    for k,v in startup_fresh.items():
        if k!='created_at' and q.get(k)!=v:raise AdmissionError('B16 startup readback mismatch: '+k)
    parity_fresh=module('threshold_b16_4k_validate.py').build()
    if parity_fresh!=t:raise AdmissionError('B16 threshold readback mismatch')


def inspect_identity(ledger,port):
    """Count unresolved rows for this port, conservatively including unknown ports."""
    import sqlite3
    db=sqlite3.connect(Path(ledger).resolve().as_uri()+'?mode=ro',uri=True)
    try:
        counts={state:0 for state in claim_guard.BLOCKING}
        for state,n in db.execute("SELECT state,count(*) FROM jobs WHERE (port=? OR port IS NULL) AND state IN ('preparing','submitted','uncertain','running') GROUP BY state",(port,)):
            counts[state]=n
        return counts
    finally:db.close()
