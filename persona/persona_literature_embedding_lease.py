"""One source-pinned, preemptible60-minute John embedding lease.

The child commits SQLite batches and exits75 on graceful interruption. This
supervisor never resumes automatically or signals unrelated processes.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

from persona_planner import wb, memory_available, stop_child

ROOT = Path(__file__).resolve().parents[1]
GPU = 'GPU-d739b9c5-bfbb-e95a-bbf1-7122f38c2cf1'
UNIT = 'character-literature-embeddings.service'
PYTHON = '/home/orwel/dev_genius/venv/bin/python'
SCRIPT = ROOT / 'persona/persona_literature_embeddings.py'
MANIFEST = ROOT / 'results/persona_literature_embedding_lease.jsonl'
POLICY = Path('/home/orwel/dev_genius/gpu_workboard/john_thermal_policy.py')


def file_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def command() -> list[str]:
    return [PYTHON, str(SCRIPT), '--build', '--device', 'cuda', '--batch-size', '256']


def protected_services() -> dict:
    snapshot = {}
    for scope in ([], ['--user']):
        result = subprocess.run(['systemctl', *scope, 'list-units', '--type=service', '--state=running',
                                 '--no-legend', '--plain', '--no-pager'], capture_output=True, text=True, timeout=15)
        if result.returncode:
            raise RuntimeError('Cannot inspect protected co-hosted services: ' + result.stderr[:200])
        for line in result.stdout.splitlines():
            name = line.split()[0]
            if re.search(r'bob|palworld|tunnel|hdmi|synthwave', name, re.I):
                active = subprocess.check_output(['systemctl', *scope, 'show', name, '-p', 'ActiveState', '-p', 'InvocationID'], text=True, timeout=15)
                snapshot[('user:' if scope else 'system:') + name] = dict(v.split('=', 1) for v in active.splitlines() if '=' in v)
    return snapshot


def owned_memory(child_pid: int) -> int:
    lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                     '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()
    total = 0
    for line in lines:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 3 or parts[1] != GPU:
            continue
        pid = int(parts[0])
        for _ in range(64):
            if pid == child_pid:
                total += int(parts[2]); break
            try:
                stat = Path(f'/proc/{pid}/stat').read_text()
            except (FileNotFoundError, ProcessLookupError):
                break
            pid = int(stat[stat.rfind(')') + 2:].split()[1])
            if pid <= 1:
                break
    return total


def cpu_affinity() -> list[int]:
    available = set(os.sched_getaffinity(0))
    # Match the NLP executor's shared OS reserve: physical core0/sibling16.
    siblings = Path('/sys/devices/system/cpu/cpu0/topology/thread_siblings_list').read_text().strip()
    reserve = {0, 16}
    for part in siblings.split(','):
        bounds = part.split('-')
        reserve.update(range(int(bounds[0]), int(bounds[-1]) + 1))
    selected = sorted(available - reserve)
    if not selected:
        raise RuntimeError('No CPUs remain after reserving one physical core')
    return selected


def release(job: str, receipt: dict, acquired: bool, outcome: str) -> None:
    try:
        current = next((r for r in wb('status', json_result=True)['jobs'] if r['id'] == job), None)
    except Exception:
        if not acquired:
            raise RuntimeError('Acquire outcome uncertain; exact lease retained: ' + job)
        current = {**receipt, 'status': 'leased', 'assigned_uuid': GPU}
    if current is None:
        return
    for key in ('id', 'project', 'label', 'target_host', 'gpu_selector'):
        if key not in receipt or current.get(key) != receipt[key]:
            raise RuntimeError('Lease receipt identity changed; refusing release')
    if current['status'] == 'queued':
        wb('cancel', job); return
    if current['status'] not in ('leased', 'running', 'stale') or current.get('assigned_uuid') != GPU:
        raise RuntimeError('Unexpected lease state; exact lease retained')
    state = 'completed' if outcome == 'completed' else 'released' if outcome in ('checkpointed', 'interrupted') else 'failed'
    wb('finish', job, '--outcome', state, '--detail', outcome + ': embedding child stopped; committed SQLite batches retained for explicit resume')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script-sha256', required=True)
    parser.add_argument('--admission-wait-seconds', type=int, default=300)
    args = parser.parse_args()
    if (socket.gethostname() != 'wtatum84' or UNIT not in Path('/proc/self/cgroup').read_text()
            or not re.fullmatch('[0-9a-f]{64}', args.script_sha256)
            or not 1 <= args.admission_wait_seconds <= 1800):
        raise ValueError('Require exact Devbox named unit, source SHA256 and bounded admission wait')
    if not Path(PYTHON).is_file() or not SCRIPT.is_file() or file_hash(SCRIPT) != args.script_sha256:
        raise ValueError('Embedding runtime/script differs from reviewed source')
    if memory_available() < 32 * 1024**3:
        raise RuntimeError('Existing shared32GiB RAM guard blocks requested compute; no smaller job launched')
    before = protected_services()
    spec = importlib.util.spec_from_file_location('literature_john_thermal_policy', POLICY)
    policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)
    thermal = policy.ThermalStrikeController()
    signals = {'value': None}
    child = None
    def interrupted(number, frame):
        signals['value'] = number
        # Forward checkpoint immediately, even during a telemetry/board call.
        if child is not None and child.poll() is None:
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: pass
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(signum, interrupted)
    def record(event: str, **values) -> None:
        value = {'time': time.time(), 'event': event, **values}
        with MANIFEST.open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps(value, allow_nan=False), flush=True)
    job = receipt = child = None
    acquired, outcome, peak = False, 'failed', 0
    window_end = time.monotonic() + 3600
    record('admission_check', hostname=socket.gethostname(), gpu_uuid=GPU, script_sha256=args.script_sha256,
           thermal_policy=policy.policy_record(), protected_services=before, command=command())
    try:
        receipt = wb('submit', '--project', 'character-creation', '--label', 'literature-bge-embeddings',
                     '--vram-gib', '16', '--duration-min', '60', '--host', 'wtatum84', '--gpu', GPU,
                     '--lane', 'batch', '--preemptible', '--checkpoint-signal', 'SIGTERM', '--checkpoint-grace-s', '30', json_result=True)
        job = receipt['id']
        record('submitted', lease_id=job, receipt=receipt)
        admission_end = min(window_end, time.monotonic() + args.admission_wait_seconds)
        while not signals['value'] and time.monotonic() < admission_end:
            reply = wb('acquire', job, json_result=True)
            acquired = reply.get('acquired') is True
            if acquired:
                break
            time.sleep(2)
        if not acquired:
            raise RuntimeError('No admission within bounded wait; no automatic retry')
        owned = next(r for r in wb('status', json_result=True)['jobs'] if r['id'] == job)
        if owned.get('assigned_uuid') != GPU or owned.get('lane') != 'batch' or not owned.get('preemptible'):
            raise ValueError('Assigned GPU/lane/preemption contract differs')
        wb('start', job, '--pid', str(os.getpid()), '--unit', UNIT, '--observed-mib', '0')
        if signals['value']:
            outcome = 'interrupted'; return
        if file_hash(SCRIPT) != args.script_sha256:
            raise ValueError('Embedding source changed after admission')
        affinity = cpu_affinity()
        os.sched_setaffinity(0, affinity)
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': GPU, 'HF_HUB_OFFLINE': '1',
            'OMP_NUM_THREADS': str(len(affinity)), 'MKL_NUM_THREADS': str(len(affinity)),
            'RAYON_NUM_THREADS': str(len(affinity)), 'TOKENIZERS_PARALLELISM': 'true'}
        child = subprocess.Popen(command(), env=environment, start_new_session=True)
        record('started', lease_id=job, child_pid=child.pid, cpu_affinity=affinity, script_sha256=args.script_sha256,
               parent_lease_pid=os.getpid(), maximum_process_tree_mib=16384)
        last_heartbeat, checkpoint_deadline = 0., None
        while child.poll() is None:
            now = time.monotonic()
            temperature = int(subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], text=True, timeout=10).strip())
            event = thermal.observe(temperature, now)
            if event['started'] or event['cleared'] or event['request_checkpoint'] or event['stop_after_checkpoint']:
                record('thermal', detail=event, temperature_c=temperature)
            used = owned_memory(child.pid); peak = max(peak, used)
            reason = ('thermal_stop' if event['request_checkpoint'] else 'ram_guard' if memory_available() < 32 * 1024**3
                      else 'vram_limit' if used > 16384 else 'checkpointed' if signals['value']
                      else 'interrupted' if now >= window_end - 30 else None)
            if reason and checkpoint_deadline is None:
                outcome = reason
                checkpoint_deadline = min(now + 25, window_end)
                if event.get('active'):
                    checkpoint_deadline = min(checkpoint_deadline, event['active']['started_at'] + event['active']['grace_seconds'])
                try: os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                record('checkpoint_requested', lease_id=job, reason=reason, grace_seconds=max(0, checkpoint_deadline-now))
            if event['stop_after_checkpoint'] or temperature >= policy.HARD_STOP_C or (checkpoint_deadline is not None and now >= checkpoint_deadline):
                try: os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                if temperature >= policy.HARD_STOP_C: outcome = 'thermal_stop'
                break
            if now - last_heartbeat >= 30:
                wb('heartbeat', job, '--pid', str(os.getpid()), '--observed-mib', str(used))
                record('heartbeat', lease_id=job, observed_mib=used, peak_mib=peak, temperature_c=temperature)
                last_heartbeat = now
            time.sleep(1)
        if outcome == 'failed':
            outcome = 'completed' if child.poll() == 0 else 'checkpointed' if child.poll() == 75 else 'failed'
    finally:
        stop_child(child)
        try:
            after = protected_services()
            changed = [name for name, state in before.items() if after.get(name) != state]
        except Exception as error:
            after, changed = {}, ['inspection_failed']
            outcome = 'protected_service_check_failed'
            record('protected_service_inspection_failed', error=str(error))
        if changed:
            record('protected_service_change', units=changed, before=before, after=after)
        if job is not None:
            release(job, receipt, acquired, outcome)
        record('finished', lease_id=job, outcome=outcome, child_exit=None if child is None else child.returncode,
               observed_peak_mib=peak, protected_services_unchanged=not changed)
    if outcome not in ('completed', 'checkpointed', 'interrupted'):
        raise RuntimeError(outcome)


if __name__ == '__main__':
    main()
