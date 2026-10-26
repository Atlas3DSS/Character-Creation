"""Finite V100-only all-neuron prefill mapping. Never retry uncertain inference."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import threading
import time
import urllib.request

import numpy as np

IDENTITIES = ('model_identity', 'runtime_identity', 'prepared_identity')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(url, payload=None, timeout=10):
    request = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def descriptor(endpoint):
    data = fetch(endpoint['url'] + '/v1/models')['data']
    if len(data) != 1:
        raise ValueError('Expected exactly one diagnostic model')
    flare = data[0]['flare']
    if any(flare.get(k) != endpoint[k] for k in IDENTITIES):
        raise ValueError('Endpoint identity changed')
    if not flare.get('activation_search', {}).get('supported') or not flare.get('calibration_equivalence', {}).get('passed'):
        raise ValueError('Unqualified all-neuron runtime')
    return data[0]


def validate_artifact(pointer, root):
    path = Path(pointer['path']).resolve()
    if path.parent != root.resolve() or not path.is_file() or file_hash(path) != pointer['sha256']:
        raise ValueError('Artifact location or SHA256 differs')
    return path


def validate_result(response, request, endpoint, root):
    """Validate full raw reductions, not merely the top32 response ranking."""
    if any(response.get(k) != endpoint[k] for k in IDENTITIES):
        raise ValueError('Response identity changed')
    search = response['activation_search']
    if search['id'] != request['capture_search']['id']:
        raise ValueError('Search identity changed')
    coverage = search['coverage']
    if coverage['all_neurons'] is not True or coverage['all_prompt_positions'] is not True:
        raise ValueError('Incomplete coverage flags')
    shape = (64, 17408)
    count = len(search['tokens'])
    if not 1 <= count <= 4096 or (coverage['layers'], coverage['neurons_per_layer'], coverage['tokens']) != (*shape, count):
        raise ValueError('Coverage dimensions changed')
    if coverage['neuron_token_values'] != math.prod(shape) * count:
        raise ValueError('Coverage count changed')
    pointer = response['activation_artifact']
    if pointer['request_sha256'] != digest(request):
        raise ValueError('Request binding differs')
    path = validate_artifact(pointer, root)
    with np.load(path, allow_pickle=False) as arrays:
        for key in IDENTITIES:
            if str(arrays[key]) != endpoint[key]:
                raise ValueError('NPZ identity differs')
        if str(arrays['request_sha256']) != digest(request) or str(arrays['stage']) != 'all_prefill_search':
            raise ValueError('NPZ request/stage differs')
        peak, positions, total = (arrays[k] for k in ('search_peak', 'search_peak_position', 'search_sum_abs'))
        if any(a.shape != shape for a in (peak, positions, total)):
            raise ValueError('Incomplete raw all-neuron planes')
        if not np.isfinite(peak).all() or not np.isfinite(total).all():
            raise ValueError('Nonfinite raw reductions')
        if np.any(positions < 0) or np.any(positions >= count) or np.any(total < 0):
            raise ValueError('Invalid raw reduction coverage')
        if np.any(total + 1e-5 < np.abs(peak)):
            raise ValueError('Sum below maximum')
        if not np.array_equal(arrays['search_positions'], np.arange(count)):
            raise ValueError('Prompt positions not complete')
        if arrays['prompt_token_ids'].tolist() != [t['id'] for t in search['tokens']]:
            raise ValueError('Token alignment differs')
        for row in search['candidates']:
            key = (row['layer'], row['neuron'])
            values = np.asarray(row['values'], dtype=np.float64)
            if values.shape != (count,) or not np.isfinite(values).all():
                raise ValueError('Invalid selected profile')
            if not np.isclose(peak[key], row['peak'], rtol=1e-5, atol=1e-6) or positions[key] != row['peak_position']:
                raise ValueError('Raw peak differs from replay')
            if not np.isclose(total[key] / count, row['mean_abs'], rtol=1e-5, atol=1e-6):
                raise ValueError('Raw mean differs from profile')
            if not np.isclose(values[row['peak_position']], peak[key], rtol=1e-5, atol=1e-6):
                raise ValueError('Replay peak differs')
            if not np.isclose(np.abs(values).mean(), total[key] / count, rtol=1e-5, atol=1e-6):
                raise ValueError('Replay mean differs')
    pointers = [response['provenance_artifact'], search['profile_artifact'],
                search['profile_provenance_artifact'], search['result_artifact']]
    for item in pointers:
        validate_artifact(item, root)
    return {'tokens': count, 'neurons': math.prod(shape), 'neuron_token_values': count * math.prod(shape),
            'raw_artifact_sha256': pointer['sha256'], 'artifact_bytes': sum(Path(p['path']).stat().st_size for p in [pointer, *pointers])}


def initialize(database, inputs, expected_hash):
    if file_hash(inputs) != expected_hash:
        raise ValueError('Frozen input manifest changed')
    with sqlite3.connect(database) as db:
        db.executescript('''CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,ordinal INTEGER UNIQUE,state TEXT,row TEXT,
        endpoint TEXT,started REAL,finished REAL,request TEXT,response TEXT,error TEXT,validation TEXT);
        CREATE TABLE IF NOT EXISTS events(time REAL,kind TEXT,body TEXT);''')
        prior = db.execute("SELECT value FROM metadata WHERE key='input_sha256'").fetchone()
        if prior and prior[0] != expected_hash:
            raise ValueError('Resume manifest binding differs')
        if not prior:
            db.execute('INSERT INTO metadata VALUES(?,?)', ('input_sha256', expected_hash))
            with inputs.open() as stream:
                for ordinal, line in enumerate(stream):
                    row = json.loads(line)
                    if not isinstance(row['id'], str) or not row['id'] or not row['messages'] or not row.get('source_sha256'):
                        raise ValueError('Invalid frozen input record')
                    if any(m.get('role') not in ('system', 'user', 'assistant') or not isinstance(m.get('content'), str) for m in row['messages']):
                        raise ValueError('Invalid source messages')
                    db.execute('INSERT INTO tasks(id,ordinal,state,row) VALUES(?,?,?,?)',
                               (row['id'], ordinal, 'pending', json.dumps(row, allow_nan=False)))
        db.execute("UPDATE tasks SET state='uncertain',error='Coordinator interrupted; remote inference may have completed; no automatic retry' WHERE state='running'")
        db.execute("CREATE INDEX IF NOT EXISTS tasks_tier_state ON tasks(json_extract(row,'$.tier'),state)")


def unit_state(unit):
    text = subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'InvocationID'], text=True, timeout=15)
    return dict(line.split('=', 1) for line in text.splitlines() if '=' in line)


def renewal_command(command, deadline):
    command = list(command)
    remaining = int((deadline - time.time()) // 60)
    if not 45 <= remaining <= 600:
        raise ValueError('Insufficient finite renewal window')
    for index, value in enumerate(command):
        if value.startswith('--property=RuntimeMaxSec='):
            command[index] = '--property=RuntimeMaxSec=' + str(remaining * 60)
    command[command.index('run') + 4] = str(remaining)
    command.insert(command.index('/usr/bin/python3'), '--setenv=PERSONA_JOB_DEADLINE=' + str(deadline))
    return command


def bind_continuation(database, manifest, *, config_sha256, runner_sha256, input_sha256, endpoints, now=None):
    """Explicit new scheduling window; retain original capture/input bindings."""
    now = time.time() if now is None else now
    required = {'schema','id','parent_runner_sha256','runner_sha256','config_sha256','input_sha256',
                'deadline','initial_counts','prior_endpoints','feature_artifact_sha256','reason'}
    if not isinstance(manifest,dict) or set(manifest)!=required or manifest['schema']!='character-creation.overnight-continuation.v1':
        raise ValueError('Invalid explicit continuation manifest')
    if manifest['id']!=digest({k:v for k,v in manifest.items() if k!='id'}):
        raise ValueError('Continuation content binding differs')
    if (manifest['config_sha256']!=config_sha256 or manifest['runner_sha256']!=runner_sha256
            or manifest['input_sha256']!=input_sha256):
        raise ValueError('Continuation source/config/input binding differs')
    if type(manifest['deadline']) not in (int,float) or not now<manifest['deadline']<=now+10*3600+60:
        raise ValueError('Continuation requires a finite unexpired window')
    if not isinstance(manifest['reason'],str) or not manifest['reason'].strip():
        raise ValueError('Continuation requires an explicit reason')
    with sqlite3.connect(database) as db:
        db.execute('BEGIN IMMEDIATE')
        metadata=dict(db.execute('SELECT key,value FROM metadata'))
        if (metadata.get('config_sha256')!=config_sha256 or metadata.get('runner_sha256')!=manifest['parent_runner_sha256']
                or metadata.get('input_sha256')!=input_sha256):
            raise ValueError('Original immutable ledger binding differs')
        key='continuation:'+manifest['id']
        encoded=json.dumps(manifest,sort_keys=True,allow_nan=False)
        if key in metadata:
            if metadata[key]!=encoded or metadata.get('active_continuation')!=manifest['id']:
                raise ValueError('A different continuation owns this ledger')
            return manifest['id']
        if metadata.get('active_continuation'):
            raise ValueError('Prior continuation needs explicit separate reconciliation')
        counts=dict(db.execute('SELECT state,count(*) FROM tasks GROUP BY state'))
        if counts!=manifest['initial_counts'] or counts.get('running',0):
            raise ValueError('Continuation ledger snapshot changed or has active tasks')
        prior=json.loads(metadata.get('renewed_endpoints','{}'))
        if prior!=manifest['prior_endpoints'] or set(prior)!={e['url'] for e in endpoints}:
            raise ValueError('Prior endpoint ownership receipts differ')
        for endpoint in endpoints:
            state=unit_state(endpoint['unit'])
            if state.get('ActiveState') not in ('inactive','failed') or state.get('InvocationID') not in ('',prior[endpoint['url']]['invocation_id']):
                raise ValueError('Prior worker is active or invocation ownership changed')
        db.execute('INSERT INTO metadata VALUES(?,?)',(key,encoded))
        db.execute('INSERT INTO metadata VALUES(?,?)',('active_continuation',manifest['id']))
        db.execute('INSERT INTO events VALUES(?,?,?)',(now,'continuation_authorized',encoded))
    return manifest['id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--continuation', type=Path, help='Explicit source-bound continuation of a finished scheduling window')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config['artifact_root']).resolve()
    if socket.gethostname() != 'v100' or len(config['endpoints']) != 6:
        raise ValueError('Require configured six-endpoint V100 host')
    for filename, expected in config['runtime_hashes'].items():
        if file_hash(root / filename) != expected:
            raise ValueError('Runtime source changed: ' + filename)
    continuation=json.loads(args.continuation.read_text()) if args.continuation else None
    deadline = continuation['deadline'] if continuation else config['deadline']
    if not time.time() < deadline <= time.time() + 10 * 3600 + 60:
        raise ValueError('Require finite remaining deadline of at most10 hours')
    lock = args.database.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initialize(args.database, args.inputs, args.input_sha256)
    if continuation:
        bind_continuation(args.database,continuation,config_sha256=file_hash(args.config),runner_sha256=file_hash(__file__),
                          input_sha256=args.input_sha256,endpoints=config['endpoints'])
    else:
        with sqlite3.connect(args.database) as db:
            for key, value in (('config_sha256', file_hash(args.config)), ('runner_sha256', file_hash(__file__))):
                prior = db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
                if prior and prior[0] != value:
                    raise ValueError('Resume source/config binding changed: ' + key)
                db.execute('INSERT OR IGNORE INTO metadata VALUES(?,?)', (key, value))
    stop = threading.Event()
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: stop.set())
    state_lock = threading.Lock()
    active = set()
    renewed = set()
    with sqlite3.connect(args.database) as db:
        saved = db.execute("SELECT value FROM metadata WHERE key='renewed_endpoints'").fetchone()
        saved = json.loads(saved[0]) if saved else {}
    for endpoint in config['endpoints']:
        if continuation:
            endpoint.update(continuation['prior_endpoints'][endpoint['url']])
            endpoint['lease_expires']=0
        if endpoint['url'] in saved and (not continuation or saved[endpoint['url']]['lease_expires']==deadline):
            endpoint.update(saved[endpoint['url']])
            renewed.add(endpoint['url'])
        current = unit_state(endpoint['unit'])
        if current.get('InvocationID') not in ('', endpoint['invocation_id']):
            raise ValueError('Existing worker invocation differs from owned receipt')
    paused = root / 'overnight.pause'
    status_path = root / 'overnight_status.json'
    terminal_state = None
    feature_process = None
    feature_attempted = False
    feature_state = 'waiting_for_authored_captures'
    if continuation and continuation['feature_artifact_sha256']:
        if file_hash(root/'overnight_feature_map.json')!=continuation['feature_artifact_sha256']:
            raise ValueError('Prior completed feature analysis artifact changed')
        feature_state='complete'

    def connect():
        return sqlite3.connect(args.database, timeout=60)

    def event(kind, **body):
        with connect() as db:
            db.execute('INSERT INTO events VALUES(?,?,?)', (time.time(), kind, json.dumps(body)))
        print(json.dumps({'time': time.time(), 'event': kind, **body}), flush=True)

    def publish():
        with connect() as db:
            counts = dict(db.execute('SELECT state,count(*) FROM tasks GROUP BY state'))
        with state_lock:
            body = {'updated': time.time(), 'deadline': deadline, 'paused': paused.exists(),
                    'active': sorted(active), 'counts': counts, 'input_sha256': args.input_sha256,
                    'feature_analysis': feature_state, 'continuation_id':continuation['id'] if continuation else None,
                    'state': terminal_state or ('stopping' if stop.is_set() else 'running')}
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(body, indent=2) + '\n'); temporary.replace(status_path)

    def renew_expired():
        for endpoint in config['endpoints']:
            if endpoint['url'] in renewed or time.time() < endpoint['lease_expires']:
                continue
            state = unit_state(endpoint['unit'])
            if state['ActiveState'] not in ('inactive', 'failed'):
                # Never stop existing workers to accelerate the renewal.
                if time.time() > endpoint['lease_expires'] + 180:
                    raise RuntimeError('Existing worker exceeded finite expiry; no replacement started')
                continue
            if state.get('InvocationID') not in ('', endpoint['invocation_id']):
                raise ValueError('Worker invocation ownership changed')
            if deadline - time.time() < 45 * 60:
                continue
            for filename, expected in config['runtime_hashes'].items():
                if file_hash(root / filename) != expected:
                    raise ValueError('Runtime source changed before renewal: ' + filename)
            command = renewal_command(endpoint['renewal_command'], deadline)
            subprocess.run(command, check=True, timeout=60)
            state = unit_state(endpoint['unit'])
            if not state.get('InvocationID'):
                raise RuntimeError('Renewal has no named invocation')
            endpoint['invocation_id'] = state['InvocationID']
            endpoint['lease_expires'] = deadline
            renewed.add(endpoint['url'])
            with connect() as db:
                saved = {e['url']: {k: e[k] for k in ('invocation_id', 'lease_expires')}
                         for e in config['endpoints'] if e['url'] in renewed}
                db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', ('renewed_endpoints', json.dumps(saved)))
            event('renewed', unit=endpoint['unit'], invocation=state['InvocationID'], deadline=deadline)

    def maybe_features():
        nonlocal feature_process, feature_attempted, feature_state
        if feature_state=='complete' and feature_process is None:
            return
        if feature_attempted:
            code = feature_process.poll()
            if code is not None and feature_state == 'running':
                feature_state = 'complete' if code == 0 else 'failed'
                event('feature_analysis_finished', exit_code=code)
            return
        if stop.is_set() or paused.exists():
            return
        with connect() as db:
            states = dict(db.execute("SELECT state,count(*) FROM tasks WHERE json_extract(row,'$.tier')='authored_emotion_contrast' GROUP BY state"))
        if sum(states.values()) != 768 or states.get('pending', 0) or states.get('running', 0):
            return
        script = root / 'persona_overnight_features.py'
        if file_hash(script) != config['feature_script_sha256']:
            raise ValueError('Feature analysis source changed')
        feature_process = subprocess.Popen([config['python'], str(script), '--database', str(args.database), '--root', str(root)], start_new_session=True)
        feature_attempted = True
        feature_state = 'running'
        event('feature_analysis_started', pid=feature_process.pid, source_sha256=config['feature_script_sha256'], capture_counts=states)

    def worker(endpoint):
        unavailable_since = None
        consecutive_failures = 0
        while not stop.is_set() and time.time() < deadline - 60:
            if paused.exists():
                stop.wait(2); continue
            if endpoint['lease_expires'] - time.time() < 120:
                stop.wait(5); continue
            with connect() as db:
                if not db.execute("SELECT 1 FROM tasks WHERE state='pending' LIMIT 1").fetchone():
                    return
            try:
                model = descriptor(endpoint)
                unavailable_since = None
            except Exception as error:
                unavailable_since = unavailable_since or time.time()
                if time.time() - unavailable_since > 2100:
                    event('endpoint_unavailable', endpoint=endpoint['url'], error=str(error)); return
                stop.wait(5); continue
            with state_lock:
                if paused.exists() or stop.is_set():
                    continue
                with connect() as db:
                    db.execute('BEGIN IMMEDIATE')
                    found = db.execute("SELECT id,row FROM tasks WHERE state='pending' ORDER BY ordinal LIMIT 1").fetchone()
                    if not found:
                        return
                    identity, source = found[0], json.loads(found[1])
                    request = {'model': model['id'], 'messages': source['messages'], 'seed': 7035,
                               'max_tokens': 1, 'temperature': 0, 'stream': False, 'observation_only': True,
                               'chat_template_kwargs': {'enable_thinking': False},
                               'capture_search': {'id': digest({'manifest': args.input_sha256, 'id': identity})[:32], 'top_k': 32}}
                    db.execute("UPDATE tasks SET state='running',endpoint=?,started=?,request=? WHERE id=?",
                               (endpoint['url'], time.time(), json.dumps(request), identity))
                active.add(endpoint['url'])
            response = None
            try:
                response = fetch(endpoint['url'] + '/v1/chat/completions', request, timeout=min(1200, max(1, deadline-time.time())))
                with connect() as db:
                    db.execute('UPDATE tasks SET response=? WHERE id=?', (json.dumps(response, allow_nan=False), identity))
                validation = validate_result(response, request, endpoint, root)
                with connect() as db:
                    db.execute("UPDATE tasks SET state='complete',finished=?,validation=? WHERE id=?", (time.time(), json.dumps(validation), identity))
                event('complete', id=identity, endpoint=endpoint['url'], **validation)
                consecutive_failures = 0
            except Exception as error:
                # A transport failure has unknown remote outcome; no resubmission.
                state = 'failed' if response is not None else 'uncertain'
                with connect() as db:
                    db.execute('UPDATE tasks SET state=?,finished=?,error=? WHERE id=?', (state, time.time(), repr(error), identity))
                event(state, id=identity, endpoint=endpoint['url'], error=repr(error))
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    event('endpoint_halted', endpoint=endpoint['url'], reason='Three consecutive capture/validation failures; pending tasks retained')
                    return
            finally:
                with state_lock:
                    active.discard(endpoint['url'])

    event('start', source_sha256=file_hash(__file__), config_sha256=file_hash(args.config), deadline=deadline,
          continuation_id=continuation['id'] if continuation else None)
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            jobs = [pool.submit(worker, endpoint) for endpoint in config['endpoints']]
            while not all(job.done() for job in jobs):
                publish()
                try:
                    if not stop.is_set():
                        renew_expired()
                        maybe_features()
                except Exception:
                    stop.set(); raise
                if time.time() >= deadline:
                    stop.set()
                stop.wait(5) if not stop.is_set() else time.sleep(1)
            for job in jobs:
                job.result()
            maybe_features()
    except Exception as error:
        stop.set(); terminal_state = 'failed'; publish()
        event('supervisor_failed', error=repr(error))
        raise
    finally:
        if feature_process is not None:
            # Finish bounded CPU derivation if captures ended early; never run
            # past the immutable overnight deadline or leave orphan children.
            try:
                if stop.is_set() and feature_process.poll() is None:
                    os.killpg(feature_process.pid, signal.SIGTERM)
                code = feature_process.wait(timeout=max(1, min(1800, deadline-time.time())))
                if feature_state == 'running':
                    feature_state = 'complete' if code == 0 else 'failed'
                    event('feature_analysis_finished', exit_code=code)
            except subprocess.TimeoutExpired:
                os.killpg(feature_process.pid, signal.SIGTERM)
                try:
                    feature_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(feature_process.pid, signal.SIGKILL); feature_process.wait()
                event('feature_analysis_deadline')
        # Only stop renewed invocations created by this coordinator. Initial
        # serving leases retain their previously authorized finite lifetime.
        for endpoint in config['endpoints']:
            if endpoint['url'] in renewed and unit_state(endpoint['unit']).get('InvocationID') == endpoint['invocation_id']:
                subprocess.run(['systemctl', '--user', 'stop', endpoint['unit']], check=True, timeout=90)
    stop.set()
    with connect() as db:
        counts = dict(db.execute('SELECT state,count(*) FROM tasks GROUP BY state'))
    terminal_state = 'complete' if set(counts) == {'complete'} else 'deadline' if time.time() >= deadline else 'partial'
    publish(); event('stopped', outcome=terminal_state, counts=counts)


if __name__ == '__main__':
    main()
