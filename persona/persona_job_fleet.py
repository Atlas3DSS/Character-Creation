"""Fixed operator-owned resource adapter for scheduled persona jobs.

Only named, workboard-leased services are started. Cleanup is bound to the exact
systemd InvocationID created here; another owner's process is never stopped.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import shlex
import socket
import sqlite3
import subprocess
import threading
import time
import uuid

from persona_fleet import fetch
from persona_launch import command as probe_command
from persona_manager import check_admission, local_url, shared_runtime
from persona_planner import GPU as JOHN, UNIT as PLANNER_UNIT, readiness
from persona_vectors import ModelSpec
from persona_service_control import JOB_ENV, TOKEN_ENV, UNITS, validate_owner

TUNNEL_UNIT = 'character-persona-jobs-tunnel.service'


def source_path(root: Path, name: str) -> Path:
    """Find source in this checkout or in the existing flat worker deployment."""
    root = Path(root)
    organized = root / 'persona' / name
    return organized if '/' not in name and organized.is_file() else root / name


def run(argv, *, ssh=None, timeout=40):
    invocation = [*ssh, shlex.join(argv)] if ssh else argv
    reply = subprocess.run(invocation, text=True, capture_output=True, timeout=timeout)
    if reply.returncode:
        raise RuntimeError(f'{argv[0]} failed ({reply.returncode}): {reply.stderr[-1500:]}')
    return reply.stdout


def unit_state(unit, ssh=None, *, helper):
    if unit not in UNITS: raise ValueError('Only fixed persona service units are allowed')
    return json.loads(run(['/usr/bin/python3', helper, 'state', '--unit', unit], ssh=ssh))


def stop_owned(unit, invocation_id, ssh=None, *, helper, job_id, token):
    validate_owner(unit, job_id, token)
    argv = ['/usr/bin/python3', helper, 'stop', '--unit', unit, '--job-id', job_id, '--token', token]
    if invocation_id: argv += ['--invocation-id', invocation_id]
    result = json.loads(run(argv, ssh=ssh, timeout=125))
    if result.get('status') == 'identity_changed_before_signal':
        raise RuntimeError('Service changed during verified cleanup; no process was signalled')
    return result


def prior_launch_intents(job, root):
    """Read the full immutable job history, not the UI's latest100 events."""
    path = root / 'results/persona_jobs.sqlite'
    if path.exists():
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=10) as db:
            rows = db.execute("SELECT data FROM events WHERE job_id=? AND event='service_launch_intent' ORDER BY sequence LIMIT 10001", (job['id'],)).fetchall()
        if len(rows) > 10000: raise ValueError('Too many prior launch intents to reconcile safely')
        intents = [json.loads(row[0]) for row in rows]
    else:
        intents = [row['data'] for row in job.get('events', []) if row.get('event') == 'service_launch_intent']
    for intent in intents:
        validate_owner(intent['unit'], intent['job_id'], intent['token'])
        expected_host = 'v100' if intent['unit'].startswith('character-persona-probe-') else 'wtatum84'
        if intent['job_id'] != job['id'] or intent.get('host') != expected_host:
            raise ValueError('Prior launch intent does not bind this job and fixed host')
    return intents


def cleanup_intents(intents, event, *, ssh, local_helper, remote_helper):
    errors = []
    for intent in reversed(intents):
        unit = intent['unit']; remote = intent['host'] == 'v100'
        try:
            result = stop_owned(unit, intent.get('invocation_id'), ssh if remote else None,
                helper=remote_helper if remote else local_helper, job_id=intent['job_id'], token=intent['token'])
            name, details = 'service_cleanup', {'unit': unit, 'token': intent['token'], **result}
        except Exception as error:
            errors.append(f'{unit}: {error}')
            name, details = 'cleanup_error', {'unit': unit, 'token': intent['token'], 'error': repr(error)}
        # A database/logging failure must never skip cleanup of another intent.
        try: event(name, details)
        except Exception as error: errors.append(f'{unit} cleanup audit: {error}')
    if errors: raise RuntimeError('Owned cleanup could not be fully verified: ' + '; '.join(errors))


def launch_owned(unit, argv, job, event, intents, *, host, helper, ssh=None):
    """Save an immutable receipt BEFORE a potentially uncertain systemd call."""
    intent = {'unit': unit, 'job_id': job['id'], 'token': uuid.uuid4().hex, 'host': host}
    validate_owner(unit, job['id'], intent['token'])
    if argv[0] != 'systemd-run' or not any(value in ('--unit=' + unit, '--unit=' + unit.removesuffix('.service')) for value in argv):
        raise ValueError('Launch argv must name the fixed intended service')
    intents.append(intent)
    event('service_launch_intent', dict(intent))
    executable = next((i for i, value in enumerate(argv[1:], 1) if not value.startswith('-')), None)
    if executable is None or argv[executable] not in ('/usr/bin/python3', '/usr/bin/ssh'):
        raise ValueError('Unexpected service executable')
    remaining = int(float(job['deadline']) - time.time())
    if remaining < 1: raise InterruptedError('Scheduled window ended before launch')
    invocation = [('--property=RuntimeMaxSec=' + str(remaining)) if value.startswith('--property=RuntimeMaxSec=') else value for value in argv]
    invocation[executable:executable] = [f'--setenv={JOB_ENV}={job["id"]}', f'--setenv={TOKEN_ENV}={intent["token"]}',
        f'--setenv=PERSONA_JOB_DEADLINE={job["deadline"]}', '--property=Restart=no']
    run(invocation, ssh=ssh)
    state = unit_state(unit, ssh, helper=helper)
    if (not state.get('InvocationID') or state.get('launch_token') != intent['token'] or state.get('job_id') != job['id']):
        raise RuntimeError('New service did not expose its exact launch ownership: ' + unit)
    intent['invocation_id'] = state['InvocationID']
    event('service_started', dict(intent))
    return intent


def live_board(config):
    return json.loads(run([config['workboard'], 'status', '--json']))


def worker_command(fleet, config, shard, minutes, seconds):
    if not math.isfinite(seconds) or seconds < 1: raise InterruptedError('Scheduled worker window ended')
    argv = probe_command(fleet, shard, False, mode='probe', minutes=minutes, base_port=8930)
    wrapper_index = argv.index(fleet['lease_wrapper'])
    argv[wrapper_index] = fleet['remote_root'] + '/persona_lease.py'
    argv[wrapper_index-1:wrapper_index-1] = [
        '--setenv=PERSONA_WORKER_UNIT=' + f'character-persona-probe-{shard}.service',
        '--setenv=PERSONA_LEASE_WRAPPER=' + fleet['lease_wrapper'],
        '--setenv=PERSONA_LEASE_WRAPPER_SHA256=' + config['lease_wrapper_sha256']]
    return [('--setenv=WB_HOLD_WAIT=' + str(int(seconds))) if value.startswith('--setenv=WB_HOLD_WAIT=')
            else ('--property=RuntimeMaxSec=' + str(int(seconds))) if value.startswith('--property=RuntimeMaxSec=') else value for value in argv]


def control_context(config):
    """Verify the fixed hosts/helper without creating services or GPU leases."""
    if socket.gethostname() != 'wtatum84':
        raise ValueError('The scheduled resource owner must run on wtatum84')
    root = Path(config['root']).resolve()
    fleet = json.loads(Path(config['fleet_config']).read_text())
    if fleet['hostname'] != 'v100' or len(fleet['gpus']) != 6 or len(set(fleet['gpus'])) != 6:
        raise ValueError('Expected six explicitly configured Quincies on v100')
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
           '-o', 'ConnectTimeout=10', fleet['ssh_host']]
    if run(['hostname'], ssh=ssh).strip() != fleet['hostname']:
        raise ValueError('Remote hostname does not match the configured fleet')
    local_helper, remote_helper = str(source_path(root, 'persona_service_control.py')), fleet['remote_root'] + '/persona_service_control.py'
    if run(['sha256sum', remote_helper], ssh=ssh).split()[0] != hashlib.sha256(Path(local_helper).read_bytes()).hexdigest():
        raise ValueError('Deployed ownership-control helper differs')
    return root, fleet, ssh, local_helper, remote_helper


def reconcile_recovered(job, config, event):
    """Clean a recovered terminal job; never admit or launch fresh resources.

    The daemon selects cancelled/expired jobs with a recovered event newer than
    their last resource_reconciled marker. A failed/interrupted cleanup writes
    no completion marker, making the next daemon startup retry the same intents.
    """
    root, _, ssh, local_helper, remote_helper = control_context(config)
    intents = prior_launch_intents(job, root)
    cleanup_intents(intents, event, ssh=ssh, local_helper=local_helper, remote_helper=remote_helper)
    result = {'job_id': job['id'], 'intent_count': len(intents),
        'intents_sha256': hashlib.sha256(json.dumps(intents, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'resources_started': False, 'scope': 'matching immutable launch tokens only'}
    # This durable marker is deliberately last. If its write fails, retrying
    # cleanup is safe: inactive or differently owned units are left untouched.
    event('resource_reconciled', result)
    return result


def verified_session(job, config, *, owned=None):
    root = Path(config['root'])
    fleet = json.loads(Path(config['fleet_config']).read_text())
    model = ModelSpec.load(root / 'data/persona_model.json')
    board = live_board(fleet)
    endpoints = []
    for index, gpu in enumerate(fleet['gpus']):
        unit = f'character-persona-probe-{index}.service'
        leases = [row for row in board['jobs'] if row.get('status') == 'running'
                  and row.get('project') == 'character-creation' and row.get('assigned_uuid') == gpu
                  and (row.get('unit_name') == unit or (owned is None and not row.get('unit_name')
                       and row.get('label') == f'persona-probe-{index}-v100-{index}'))]
        if len(leases) != 1:
            raise RuntimeError(f'Quincy {index + 1} does not have one verified running probe lease')
        lease = leases[0]
        if owned is not None and lease.get('unit_invocation_id') != owned.get(unit):
            raise RuntimeError('Probe lease does not belong to this scheduled job invocation')
        url = f'http://127.0.0.1:{8930 + index}'
        flare = fetch(url + '/v1/models', timeout=3)['data'][0]['flare']
        for key, expected in [('model_identity', model.identity),
                              ('prepared_identity', config['expected_prepared_identity']),
                              ('runtime_identity', config['expected_runtime_identity'])]:
            if flare.get(key) != expected:
                raise ValueError('Scheduled worker identity mismatch: ' + key)
        endpoint = {'url': url, 'model_identity': model.identity,
            'prepared_identity': flare['prepared_identity'], 'runtime_identity': flare['runtime_identity'],
            'diagnostic': True, 'label': f'Quincy {index + 1}', 'gpu_uuid': gpu, 'lease_id': lease['id'],
            'project': 'character-creation', 'host': fleet['hostname']}
        allowed, reason = check_admission(board, endpoint)
        if not allowed:
            raise RuntimeError('Quincy admission unavailable: ' + reason)
        endpoints.append(endpoint)
    shared_runtime(endpoints)
    planners = [row for row in board['jobs'] if row.get('status') == 'running'
                and row.get('project') == 'character-creation' and row.get('assigned_uuid') == JOHN
                and row.get('unit_name') == PLANNER_UNIT]
    if len(planners) != 1 or readiness() is None:
        raise RuntimeError('John planner is not ready with its exact 128K context and lease')
    lease = planners[0]
    if owned is not None and lease.get('unit_invocation_id') != owned.get(PLANNER_UNIT):
        raise RuntimeError('John planner lease does not belong to this scheduled job invocation')
    planner = {'url': 'http://127.0.0.1:18511', 'model': 'persona-manager', 'context_tokens': 131072,
               'lease_id': lease['id'], 'gpu_uuid': JOHN, 'project': 'character-creation', 'host': 'wtatum84'}
    allowed, reason = check_admission(board, planner)
    if not allowed:
        raise RuntimeError('John admission unavailable: ' + reason)
    return {'endpoints': endpoints, 'planner': planner, 'model': model.__dict__, 'root': str(root),
            'remote_root': fleet['remote_root'], 'config': config, 'fleet': fleet}


@contextmanager
def fleet_session(job, config, event, *, stop_event=None):
    stop_event = stop_event or threading.Event()
    root, fleet, ssh, local_helper, remote_helper = control_context(config)
    previous = prior_launch_intents(job, root)
    if previous:
        cleanup_intents(previous, event, ssh=ssh, local_helper=local_helper, remote_helper=remote_helper)
    deadline = float(job['deadline'])
    def remaining():
        return deadline - time.time()
    def check_stop():
        if stop_event.is_set() or remaining() <= 0:
            raise InterruptedError('Job cancelled or its scheduled window ended')
    # Keep resource ownership independent of whether a port happens to answer.
    resources = [(PLANNER_UNIT, None), (TUNNEL_UNIT, None)] + [
        (f'character-persona-probe-{i}.service', ssh) for i in range(6)]
    last_notice = 0
    while True:
        check_stop()
        occupied = [unit for unit, host in resources
                    if unit_state(unit, host, helper=remote_helper if host else local_helper).get('ActiveState') in ('active', 'activating', 'deactivating', 'reloading')]
        ports = []
        for port in (18511, *range(8930, 8936)):
            with socket.socket() as connection:
                connection.settimeout(.2)
                if connection.connect_ex(('127.0.0.1', port)) == 0: ports.append(port)
        if not occupied and not ports:
            break
        if time.time() - last_notice >= 60:
            event('waiting_for_resources', {'units': occupied, 'ports': ports, 'deadline': deadline})
            last_notice = time.time()
        stop_event.wait(min(10, max(0, remaining())))
    # Verify deployed code used by the fixed worker before creating any lease.
    required_sources = ('persona_lyra_worker.py', 'persona_lease.py') if job.get('kind')=='lyra_gift' else ('persona_probe.py', 'persona_capture.py', 'persona_vectors.py', 'persona_fleet.py', 'persona_qualify.py', 'persona_lease.py', 'persona_trace.py', 'persona_map.py')
    for filename in required_sources:
        local = hashlib.sha256(source_path(root, filename).read_bytes()).hexdigest()
        remote = run(['sha256sum', fleet['remote_root'] + '/' + filename], ssh=ssh).split()[0]
        if local != remote:
            raise ValueError('Deployed worker source differs: ' + filename)
    if run(['sha256sum', fleet['lease_wrapper']], ssh=ssh).split()[0] != config['lease_wrapper_sha256']:
        raise ValueError('Installed V100 lease supervisor differs from its verified hash')
    owned, launched, intents = {}, [], []
    def start(unit, argv, host=None):
        check_stop()
        intent = launch_owned(unit, argv, job, event, intents, host=fleet['hostname'] if host else 'wtatum84',
            helper=remote_helper if host else local_helper, ssh=host)
        invocation = intent['invocation_id']
        owned[unit] = invocation; launched.append((unit, invocation, host))
    try:
        minutes = max(1, min(960, math.ceil(remaining() / 60)))
        start(PLANNER_UNIT, ['systemd-run', '--user', '--unit=' + PLANNER_UNIT, '--collect',
              f'--property=RuntimeMaxSec={max(1, int(remaining()))}', '--property=TimeoutStopSec=60',
              '--working-directory=' + str(root), '/usr/bin/python3', str(source_path(root, 'persona_planner.py')),
              '--minutes', str(minutes), '--deadline', str(deadline), '--admission-wait-seconds', str(max(1, int(remaining())))])
        for shard in range(6):
            if job.get('kind')=='lyra_gift':
                from persona_lyra_job import worker_command as gift_command
                argv = gift_command(fleet,config,job,shard,minutes,remaining())
            else:
                argv = worker_command(fleet, config, shard, minutes, remaining())
            start(f'character-persona-probe-{shard}.service', argv, ssh)
        tunnel = ['systemd-run', '--user', '--unit=' + TUNNEL_UNIT, '--collect',
                  f'--property=RuntimeMaxSec={max(1, int(remaining()))}', '/usr/bin/ssh', '-NT', '-o', 'BatchMode=yes',
                  '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'ExitOnForwardFailure=yes',
                  '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3']
        for port in range(8930, 8936):
            tunnel += ['-L', f'127.0.0.1:{port}:127.0.0.1:{port}']
        tunnel += ['-R', '127.0.0.1:18511:127.0.0.1:18511', fleet['ssh_host']]
        start(TUNNEL_UNIT, tunnel)
        last_notice = 0
        while True:
            check_stop()
            for unit, invocation, host in launched:
                state = unit_state(unit, host, helper=remote_helper if host else local_helper)
                if (state.get('InvocationID') != invocation
                        or state.get('ActiveState') not in ('active', 'activating', 'reloading')):
                    raise RuntimeError('Owned service exited before readiness: ' + unit)
            try:
                if job.get('kind')=='lyra_gift':
                    from persona_lyra_job import verified
                    session=verified(job,config,owned)
                else:
                    session = verified_session(job, config, owned=owned)
                break
            except ValueError:
                raise
            except Exception as error:
                if time.time() - last_notice >= 60:
                    event('warming_workers', {'detail': str(error), 'deadline': deadline})
                    last_notice = time.time()
                stop_event.wait(min(5, max(0, remaining())))
        event('fleet_ready', {'endpoints': session['endpoints'], 'planner': session['planner']})
        yield session
    finally:
        # Reverse creation order: close tunnels, then Quincies, then John.
        cleanup_intents(intents, event, ssh=ssh, local_helper=local_helper, remote_helper=remote_helper)


def run_neighborhood(job, session, event, stop_event):
    from persona_hillclimb import run_neighborhood as run
    return run(job, session, event, stop_event)
