"""Fixed public-corpus neutral answer + final-prefill capture; no planner or steering.
Run on the endpoint host so raw artifacts can be verified. Never acquires GPUs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import threading
import tempfile
import time
from zoneinfo import ZoneInfo
import numpy as np
from persona_manager import (Admission, encoded, file_hash, native_capability,
                             response_complete, verify_diagnostic_artifacts, shared_runtime)
from persona_vectors import ModelSpec, canonical_digest
from persona_fleet import fetch

JOB = '72aff97cff954831aa1623ee01efe95e'
COUNT = 40000

def source_rows(database):
    with sqlite3.connect('file:' + str(database.resolve()) + '?mode=ro', uri=True) as c:
        job = c.execute('SELECT state,spec,counts FROM jobs WHERE id=?', (JOB,)).fetchone()
        if not job or job[0] != 'complete':
            raise ValueError('Required generation job is not complete')
        rows = c.execute('SELECT id,request_id,normalized_hash,body FROM samples WHERE job_id=? ORDER BY id', (JOB,)).fetchall()
    if len(rows) != COUNT:
        raise ValueError('Expected exactly 40000 accepted source rows')
    result = []
    for sample_id, request_id, normalized, body in rows:
        p = json.loads(body)
        if (p.get('corpus') != 'big_five' or p.get('synthetic') is not True
                or p.get('generator', {}).get('job_id') != JOB or p.get('system')
                or p.get('thinking') is not False or not p.get('prompt')):
            raise ValueError('Non-public, non-neutral, or wrong-job source row')
        result.append({'sample_id': sample_id, 'request_id': request_id,
                       'normalized_hash': normalized, 'source_body_sha256': hashlib.sha256(body.encode()).hexdigest(),
                       'prompt': p})
    return result, {'job_id': JOB, 'job_spec': json.loads(job[1]), 'counts': json.loads(job[2])}

def export_source(args):
    rows, provenance = source_rows(args.source_database)
    content = ''.join(encoded(r) + '\n' for r in rows)
    with args.corpus.open('x') as f:
        f.write(content)
    manifest = {**provenance, 'count': COUNT, 'corpus_sha256': file_hash(args.corpus),
                'source': 'accepted samples only; excludes books and SCOTUS',
                'schema': 'character-creation.fixed-baseline-corpus.v1'}
    with args.corpus.with_suffix('.manifest.json').open('x') as f:
        f.write(encoded(manifest) + '\n')
    print(encoded(manifest))

def validate(response, endpoint, payload):
    if response.get('operation') != 'rollout' or response.get('choices', [{}])[0].get('finish_reason') not in ('stop','length'):
        raise ValueError('Expected terminated baseline rollout')
    complete = response_complete(response, diagnostic=True, measurement='rollout')
    verify_diagnostic_artifacts(response, endpoint, payload)
    a = response['activation_artifact']
    if a.get('stage') != 'final_prefill':
        raise ValueError('Activation stage must be final_prefill')
    if file_hash(Path(a['path'])) != a['sha256']:
        raise ValueError('Raw activation hash mismatch')
    with np.load(a['path'], allow_pickle=False) as z:
        for k in ('model_identity', 'prepared_identity', 'runtime_identity', 'request_sha256', 'operation', 'stage'):
            if str(z[k]) != a[k]:
                raise ValueError('Raw binding mismatch: ' + k)
        for k, shape in [('mlp_input_last', (64,17408)), ('residual_last', (64,5120)), ('pre_mlp_last', (64,5120))]:
            if z[k].shape != shape or not np.isfinite(z[k]).all():
                raise ValueError('Invalid full capture: ' + k)
        if (z['generated_token_ids'].ndim != 1 or not 1 <= len(z['generated_token_ids']) <= payload['max_tokens']
                or not np.array_equal(z['prefill_positions'], np.arange(len(z['prompt_token_ids'])))
                or response.get('usage', {}).get('completion_tokens') != len(z['generated_token_ids'])):
            raise ValueError('Answer budget/usage or exact prompt coverage mismatch')
    return {'complete':complete, 'activation': a, 'provenance': response['provenance_artifact'],
            'raw_bytes': Path(a['path']).stat().st_size}

def atomic_write(path, chunks):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.'+path.name+'.',delete=False) as out:
            temporary = out.name
            for chunk in chunks: out.write(chunk)
            out.flush(); os.fsync(out.fileno())
        os.replace(temporary,path)
        temporary = None
    finally:
        if temporary is not None: os.unlink(temporary)

def in_daily_window(now, timeout, end):
    local = datetime.fromtimestamp(now, ZoneInfo('America/Los_Angeles'))
    if not (local.hour >= 20 or local.hour < 8):
        return False
    boundary = local.replace(hour=8,minute=0,second=0,microsecond=0)
    if local.hour >= 20:
        boundary += timedelta(days=1)
    return now + timeout < min(end, boundary.timestamp())

def request_window(now, timeout, end, opportunistic=False):
    return now + timeout < end if opportunistic else in_daily_window(now, timeout, end)


def capture_binding(manifest, endpoints, max_tokens):
    return {'corpus': manifest, 'code_sha256': file_hash(Path(__file__)),
               'endpoints': [{k:e[k] for k in ('gpu_uuid','model_identity','prepared_identity','runtime_identity')} for e in endpoints],
               'operation':'rollout', 'steering':'absent', 'answer_rollout':True,
               'max_tokens':max_tokens, 'thinking':False, 'temperature':0,
               'activation_scope':'all layers/all neurons at final valid prompt token only; not answer-token neuron trajectories', 'arrays':{'mlp_input_last':[64,17408],'residual_last':[64,5120],'pre_mlp_last':[64,5120]}}

def recovery_evidence(connection, paths, original_binding):
    """Bind quarantines to exact retained rows and exact raw reports, never retry."""
    rows = {r[0]: r for r in connection.execute("SELECT * FROM captures WHERE state!='complete'")}
    by_digest = {canonical_digest(json.loads(r[3])): r for r in rows.values()}
    evidence = []
    for path in paths:
        path = Path(path).resolve()
        report = json.loads(path.read_text())
        row = by_digest.get(report.get('request_sha256'))
        if row is None or report.get('state_restored') is not True:
            raise ValueError('Recovery report does not bind a restored retained request')
        endpoint = original_binding['endpoints'][row[1] % 6]
        worker = report.get('worker_manifest') or {}
        if (report.get('request') != json.loads(row[3]) or report.get('operation') != 'rollout'
                or worker.get('gpu_uuid') != endpoint['gpu_uuid']
                or any(report.get(k) != endpoint[k] or worker.get(k) != endpoint[k]
                       for k in ('model_identity','prepared_identity','runtime_identity'))):
            raise ValueError('Recovery report model/prepared/runtime/request identity differs from original endpoint')
        if row[2] == 'failed':
            if (row[4] is not None or report.get('status') != 'failed'
                    or report.get('response') or report.get('error') != 'ProtocolError: Worker interrupted before graph replay'
                    or path.with_suffix('.npz').exists()):
                raise ValueError('Only verified interrupted failures without output may be quarantined')
        elif row[2] == 'incomplete':
            response = json.loads(row[4])
            artifact = response['activation_artifact']
            if (report.get('status') != 'complete' or not report.get('response')
                    or report['response'] != {k:v for k,v in response.items() if k!='provenance_artifact'}
                    or report.get('activation_artifact') != artifact
                    or response['choices'][0]['finish_reason'] != 'length'
                    or file_hash(Path(artifact['path'])) != artifact['sha256']):
                raise ValueError('Incomplete quarantine requires intact truncated output')
            pointer = response.get('provenance_artifact') or {}
            expected = {k:report[k] for k in ('operation','model_identity','prepared_identity','runtime_identity','request_sha256')}
            expected.update(path=str(path),sha256=file_hash(path))
            if any(pointer.get(k) != v for k,v in expected.items()):
                raise ValueError('Retained response provenance does not bind exact recovery report')
        else:
            raise ValueError('Unclassified capture cannot be quarantined')
        evidence.append({'sample_id': row[0], 'state': row[2], 'row_sha256': canonical_digest(list(row)),
                         'request_sha256': report['request_sha256'], 'report_path': str(path),
                         'report_sha256': file_hash(path),
                         'activation': json.loads(row[4])['activation_artifact'] if row[4] else None})
    if len(evidence) != len(rows) or {e['sample_id'] for e in evidence} != set(rows):
        raise ValueError('Review must account for every non-complete row exactly once')
    return sorted(evidence, key=lambda e: e['sample_id'])


def verify_recovery(connection, record, binding):
    original = record['original_binding']
    if (record.get('schema') != 'character-creation.baseline-reviewed-recovery.v1'
            or not record.get('reason') or record['original_binding_sha256'] != canonical_digest(original)
            or record['old_runner_sha256'] != original['code_sha256']
            or record['new_runner_sha256'] != binding['code_sha256']
            or {k:v for k,v in original.items() if k!='code_sha256'} != {k:v for k,v in binding.items() if k!='code_sha256'}):
        raise ValueError('Reviewed migration changed scientific binding or runner identity')
    # Validate exactly approved rows, including after more successful captures exist.
    for evidence in record['quarantines']:
        row = connection.execute('SELECT * FROM captures WHERE sample_id=?', (evidence['sample_id'],)).fetchone()
        if row is None or canonical_digest(list(row)) != evidence['row_sha256']:
            raise ValueError('Approved quarantine row changed')
        if file_hash(Path(evidence['report_path'])) != evidence['report_sha256']:
            raise ValueError('Approved quarantine report changed')
        artifact = evidence.get('activation')
        if artifact and file_hash(Path(artifact['path'])) != artifact['sha256']:
            raise ValueError('Approved quarantine activation changed')
    return {e['sample_id'] for e in record['quarantines']}


def reviewed_recovery(args):
    """Explicit offline prepare/approve; no GPU or existing capture mutation."""
    with args.ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with sqlite3.connect(args.ledger) as c:
            c.execute('PRAGMA synchronous=FULL')
            original = json.loads(c.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()[0])
            if c.execute("SELECT 1 FROM metadata WHERE key='reviewed_recovery'").fetchone():
                raise ValueError('Recovery already approved; no replacement permitted')
            manifest = json.loads(args.corpus.with_suffix('.manifest.json').read_text())
            if manifest['corpus_sha256'] != file_hash(args.corpus):
                raise ValueError('Corpus manifest mismatch')
            config = json.loads(args.config.read_text())
            binding = capture_binding(manifest, config['endpoints'], args.max_tokens)
            if args.command == 'prepare-recovery':
                record = {'schema':'character-creation.baseline-reviewed-recovery.v1',
                          'reason':args.recovery_reason, 'original_binding':original,
                          'original_binding_sha256':canonical_digest(original),
                          'old_runner_sha256':original['code_sha256'], 'new_runner_sha256':binding['code_sha256'],
                          'quarantines':recovery_evidence(c,args.quarantine_report,original),
                          'policy':'Only unattempted samples may dispatch; quarantines retain needs_review; no retry'}
                verify_recovery(c,record,binding)
                if getattr(args,'replace_draft',False):
                    prior = json.loads(args.recovery_record.read_text())
                    if prior.get('original_binding') != original:
                        raise ValueError('Cannot replace draft for a different original binding')
                    record['superseded_draft'] = {'sha256':file_hash(args.recovery_record),'record':prior}
                    atomic_write(args.recovery_record,[encoded(record)+'\n'])
                else:
                    with args.recovery_record.open('x') as out:
                        out.write(encoded(record)+'\n');out.flush();os.fsync(out.fileno())
                print(encoded(record));return
            record = json.loads(args.recovery_record.read_text())
            if record['original_binding'] != original:
                raise ValueError('Ledger binding changed since review')
            if recovery_evidence(c,[e['report_path'] for e in record['quarantines']],original) != record['quarantines']:
                raise ValueError('Quarantine evidence changed since review')
            verify_recovery(c,record,binding)
            c.execute("INSERT INTO metadata VALUES('reviewed_recovery',?)", (encoded(record),))
            c.execute("UPDATE metadata SET value=? WHERE key='binding'", (encoded(binding),))
        print('Reviewed recovery approved. Run an offline expired session to publish continuation eligibility before GPU admission.')


def run(args):
    with args.ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original_handlers = {s:signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            return run_locked(args)
        finally:
            for s,handler in original_handlers.items(): signal.signal(s,handler)

def run_locked(args):
    config = json.loads(args.config.read_text())
    endpoints = config['endpoints']
    if len(endpoints) != 6 or len({e['gpu_uuid'] for e in endpoints}) != 6:
        raise ValueError('Require six distinct fixed endpoints')
    shared_runtime(endpoints)
    model = ModelSpec.load(Path(config['model']))
    manifest = json.loads(args.corpus.with_suffix('.manifest.json').read_text())
    if manifest['job_id'] != JOB or manifest['corpus_sha256'] != file_hash(args.corpus):
        raise ValueError('Corpus manifest mismatch')
    rows = [json.loads(s) for s in args.corpus.read_text().splitlines()]
    if len(rows) != COUNT or len({r['sample_id'] for r in rows}) != COUNT:
        raise ValueError('Coverage must be exactly 40000 unique samples')
    for r in rows:
        p = r['prompt']
        if p.get('corpus') != 'big_five' or p.get('generator', {}).get('job_id') != JOB or not p.get('synthetic') or p.get('system'):
            raise ValueError('Corpus row out of scope')
    binding = capture_binding(manifest, endpoints, args.max_tokens)
    def connect():
        c = sqlite3.connect(args.ledger, timeout=60)
        c.execute('PRAGMA synchronous=FULL')
        return c
    with connect() as c:
        c.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        c.execute('CREATE TABLE IF NOT EXISTS captures(sample_id TEXT PRIMARY KEY,ordinal INTEGER UNIQUE,state TEXT NOT NULL,payload TEXT NOT NULL,response TEXT,error TEXT,started REAL,finished REAL)')
        old = c.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
        if old and old[0] != encoded(binding):
            raise ValueError('Resume binding changed')
        c.execute("INSERT OR IGNORE INTO metadata VALUES('binding',?)", (encoded(binding),))
        review = c.execute("SELECT value FROM metadata WHERE key='reviewed_recovery'").fetchone()
        recovery = json.loads(review[0]) if review else None
        quarantined = verify_recovery(c,recovery,binding) if recovery else set()
    # Recover replies lost after server completion, without resubmitting an uncertain request.
    with connect() as c:
        unresolved = {canonical_digest(json.loads(payload)):sid for sid,payload in
                      c.execute('SELECT sample_id,payload FROM captures WHERE response IS NULL') if sid not in quarantined}
    if unresolved and args.artifact_dir:
        for entry in os.scandir(args.artifact_dir):
            if not entry.name.startswith('probe-') or not entry.name.endswith('.jsonl') or not entry.is_file():
                continue
            path = Path(entry.path)
            try:
                report = json.loads(path.read_text())
            except (ValueError, OSError):
                continue
            digest = report.get('request_sha256')
            if digest not in unresolved or report.get('status') != 'complete' or report.get('state_restored') is not True:
                continue
            response = report['response']
            response['provenance_artifact'] = {k:report[k] for k in ('operation','model_identity','prepared_identity','request_sha256')}
            response['provenance_artifact'].update(path=str(path.resolve()),sha256=file_hash(path),runtime_identity=report['worker_manifest']['runtime_identity'])
            with connect() as c:
                c.execute('UPDATE captures SET response=? WHERE sample_id=?',(encoded(response),unresolved[digest]))
    stop = threading.Event()
    for s in (signal.SIGINT, signal.SIGTERM): signal.signal(s, lambda *_:stop.set())
    end = datetime.fromisoformat(args.until).timestamp()
    def permitted():
        return not stop.is_set() and request_window(time.time(), args.timeout, end, getattr(args, 'opportunistic', False))
    capacity = threading.Condition()
    new_successes = 0
    active = 0
    def reserve():
        nonlocal active
        with capacity:
            while permitted():
                if not args.max_new or new_successes + active < args.max_new:
                    active += 1
                    return True
                if new_successes >= args.max_new:
                    return False
                capacity.wait(timeout=1)
            return False
    def release(success):
        nonlocal active, new_successes
        with capacity:
            active -= 1
            new_successes += int(success)
            capacity.notify_all()
    validated_ids = set()
    validation_lock = threading.Lock()
    admission = Admission(config['workboard'])
    def worker(shard):
        if not permitted(): return
        endpoint = endpoints[shard]
        model_id = native_capability(endpoint, model, {'sites':[]}, measurement='rollout')
        for ordinal in range(shard, COUNT, 6):
            row = rows[ordinal]; sid = row['sample_id']; p = row['prompt']
            if sid in quarantined: continue
            payload = {'model':model_id, 'messages':[{'role':'user','content':p['prompt']}],
                'seed':7035, 'max_tokens':args.max_tokens, 'temperature':0, 'stream':False,
                'chat_template_kwargs':{'enable_thinking':False},
                'metadata':{'job_id':'baseline40k-'+sid, 'sample_id':sid, 'source_job_id':JOB, 'arm':'baseline', 'split':p['split']}}
            with connect() as c:
                previous = c.execute('SELECT state,payload,response FROM captures WHERE sample_id=?',(sid,)).fetchone()
            if previous:
                if not permitted(): return
                try:
                    if previous[1] != encoded(payload): raise ValueError('Request changed on resume')
                    if not previous[2]:
                        raise RuntimeError('Uncertain prior request; recover its raw report before retry: '+sid)
                    verified = validate(json.loads(previous[2]), endpoint, payload)
                    with validation_lock: validated_ids.add(sid)
                    state = 'complete' if verified['complete'] else 'incomplete'
                    with connect() as c: c.execute('UPDATE captures SET state=?,error=? WHERE sample_id=?',(state,None if verified['complete'] else 'Answer incomplete; no automatic retry',sid))
                except Exception as error:
                    with connect() as c: c.execute("UPDATE captures SET state='failed',error=? WHERE sample_id=?",(str(error),sid))
                    raise
                continue
            if not permitted(): return
            allowed, reason = admission.allowed(endpoint)
            if not allowed: raise RuntimeError(reason)
            if not reserve(): return
            allowed, reason = admission.allowed(endpoint)
            if not allowed:
                release(False)
                raise RuntimeError(reason)
            success = False
            started = time.time()
            try:
                with connect() as c:
                    c.execute('INSERT INTO captures VALUES(?,?,?,?,?,?,?,?)',(sid,ordinal,'inflight',encoded(payload),None,None,time.time(),None))
                response = fetch(endpoint['url']+'/v1/chat/completions',payload,timeout=args.timeout)
                with connect() as c: c.execute('UPDATE captures SET response=? WHERE sample_id=?',(encoded(response),sid))
                verified = validate(response,endpoint,payload)
                with validation_lock: validated_ids.add(sid)
                success = verified['complete']
                state = 'complete' if success else 'incomplete'
                with connect() as c: c.execute('UPDATE captures SET state=?,error=?,finished=? WHERE sample_id=?',(state,None if success else 'Answer incomplete; no automatic retry',time.time(),sid))
                if ordinal < 60 or ordinal % 600 < 6:
                    print(encoded({'event':'captured','complete_pair':success,'ordinal':ordinal,'shard':shard,'sample_id':sid,
                                   'elapsed_seconds':time.time()-started,'raw_bytes':Path(response['activation_artifact']['path']).stat().st_size}),flush=True)
            except Exception as e:
                with connect() as c: c.execute("UPDATE captures SET state='failed',error=? WHERE sample_id=?",(str(e),sid))
                stop.set(); raise
            finally:
                release(success)
    def guarded_worker(shard):
        try:
            return worker(shard)
        except Exception:
            stop.set()
            raise
    errors = []
    with connect() as c:
        prior_states = dict(c.execute('SELECT state,count(*) FROM captures GROUP BY state'))
        unresolved_ids = [sid for sid, in c.execute('SELECT sample_id FROM captures WHERE response IS NULL') if sid not in quarantined]
        for sid in unresolved_ids:
            c.execute("UPDATE captures SET state='failed',error='Uncertain request without recoverable response; needs operator review' WHERE sample_id=?",(sid,))
        unreviewed_failed = [sid for sid, in c.execute("SELECT sample_id FROM captures WHERE state='failed'") if sid not in quarantined]
    if unresolved_ids or unreviewed_failed:
        errors.append('Prior failed/uncertain capture needs operator review; no new requests dispatched')
    elif sum(prior_states.values()) == COUNT and prior_states.get('incomplete'):
        errors.append('All prompts attempted but incomplete answers remain; explicit scientific revision/review required')
    else:
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(guarded_worker,s) for s in range(6)]
            for future in futures:
                try: future.result()
                except Exception as error: errors.append(str(error))
    with connect() as c:
        states = dict(c.execute('SELECT state,count(*) FROM captures GROUP BY state'))
        count = states.get('complete',0)
        def index_lines():
            for sid,ordinal,state,payload,response,error in c.execute("SELECT sample_id,ordinal,state,payload,response,error FROM captures ORDER BY ordinal"):
                r = json.loads(response) if response else {}
                yield encoded({'source':rows[ordinal], 'state':state, 'complete_pair':state=='complete',
                               'validated_this_session':sid in validated_ids,
                               'error':error,'request':json.loads(payload), 'response':r,
                               'activation':r.get('activation_artifact'), 'provenance':r.get('provenance_artifact')})+'\n'
        # Invalidate any old complete publication before replacing its indexed bytes.
        atomic_write(args.index.with_suffix('.manifest.json'), [encoded({'complete':False,'publication_state':'updating','needs_review':False})+'\n'])
        atomic_write(args.index,index_lines())
    complete = count == COUNT and len(validated_ids) == COUNT and not errors
    needs_review = bool(errors or states.get('failed') or states.get('inflight') or states.get('incomplete'))
    with connect() as c:
        unreviewed = [sid for sid, in c.execute("SELECT sample_id FROM captures WHERE state!='complete'") if sid not in quarantined]
    can_continue = bool(recovery and not errors and not unreviewed and COUNT-sum(states.values()) > 0)
    report = {**binding, 'complete':complete,'needs_review':needs_review,'completed':count,'missing':COUNT-count,
              'can_continue_unattempted':can_continue,
              'reviewed_recovery':recovery,
              'states':states,'unattempted':COUNT-sum(states.values()),'new_successes':new_successes,
              'validated_this_session':len(validated_ids),'publication_state':'published',
              'daily_window':'20:00–08:00 America/Los_Angeles',
              'opportunistic':getattr(args, 'opportunistic', False),
              'stopped_by_signal':stop.is_set() and not errors,
              'index_sha256':file_hash(args.index),'ledger':str(args.ledger),'corpus':str(args.corpus),'errors':errors}
    atomic_write(args.index.with_suffix('.manifest.json'),[encoded(report)+'\n'])
    print(encoded({'completed':count,'missing':COUNT-count,'complete':complete,'needs_review':needs_review,'errors':errors}))
    if errors: raise RuntimeError('Capture stopped; incomplete/raw results retained: '+'; '.join(errors))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['export','run','prepare-recovery','approve-recovery'])
    p.add_argument('--source-database',type=Path)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--ledger',type=Path)
    p.add_argument('--index',type=Path)
    p.add_argument('--until',help='Required offset-aware stop time; daily baseline window is 20:00–08:00 LA')
    p.add_argument('--opportunistic',action='store_true',help='Permit idle backfill outside the daily window; --until remains mandatory')
    p.add_argument('--artifact-dir',type=Path,help='Existing flat probe output directory, for lost-response recovery')
    p.add_argument('--timeout',type=float,default=600)
    p.add_argument('--max-tokens',type=int,default=2048)
    p.add_argument('--max-new',type=int,default=0,help='Global new successful pair limit per session; 0 means unlimited; obeys session scheduling mode')
    p.add_argument('--recovery-record',type=Path)
    p.add_argument('--recovery-reason')
    p.add_argument('--replace-draft',action='store_true',help='Prepare only: replace an unapproved draft in place, preserving its full contents and hash inside the new draft')
    p.add_argument('--quarantine-report',type=Path,action='append',default=[])
    a=p.parse_args()
    if a.command=='export':
        if not a.source_database:p.error('--source-database required')
        export_source(a)
    elif a.command in ('prepare-recovery','approve-recovery'):
        if not all((a.config,a.ledger,a.recovery_record)):p.error('recovery requires config, ledger, recovery-record')
        if a.command=='prepare-recovery' and not (a.recovery_reason and a.quarantine_report):p.error('prepare requires recovery-reason and quarantine-report evidence')
        reviewed_recovery(a)
    else:
        if not all((a.config,a.ledger,a.index,a.until)):p.error('run requires config, ledger, index, until')
        if datetime.fromisoformat(a.until).tzinfo is None:p.error('--until must include UTC offset')
        if a.max_tokens < 2048:p.error('--max-tokens must be >=2048 for complete-answer baseline')
        if a.max_new < 0 or not 0 < a.timeout <= 3600:p.error('max-new must be nonnegative; timeout must be 0..3600 seconds')
        run(a)
if __name__=='__main__':main()
