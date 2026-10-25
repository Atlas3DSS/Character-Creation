"""Own one bounded John planner lease; never stop another owner's service.

Run under a named systemd user unit. This launcher is operator configuration,
not callable by the planning model. A preemption stops the stateless server;
the separate experiment manager retains the durable queue and audit trail.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

GPU = 'GPU-d739b9c5-bfbb-e95a-bbf1-7122f38c2cf1'
WB = '/home/orwel/.local/bin/gpu-workboard'
RUNTIME = Path('/home/orwel/dev_genius/engines/llama.cpp-b10299/build-cuda-pro6000/bin/llama-server')
MODEL = Path('/home/orwel/dev_genius/models/gguf/qwen38-coder390/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-Q4_K_M.gguf')
MODEL_SHA = '299b3c3d0e8713a99b0c926ebb85d79370fce27cee166decc3e3a7217c06fed6'
UNIT = 'persona-planner-john.service'


def command() -> list[str]:
    return [str(RUNTIME), '-m', str(MODEL), '--alias', 'persona-manager', '-ngl', '999',
        '-fa', 'on', '-c', '131072', '-ctk', 'q8_0', '-ctv', 'q8_0', '-np', '1',
        '-b', '2048', '-ub', '512', '--fit', 'off', '--jinja', '--reasoning-budget', '0',
        '--threads', '8', '--threads-batch', '22', '--host', '127.0.0.1', '--port', '18511', '--no-webui']


def wb(*args: str, json_result: bool = False) -> dict:
    result = subprocess.run([WB, *args, *(['--json'] if json_result else [])],
                            capture_output=True, text=True, timeout=30)
    if result.returncode == 3 and args[0] == 'acquire':
        value = json.loads(result.stdout)
        if value.get('acquired') is False:
            return value
    if result.returncode:
        raise RuntimeError(f'workboard {args[0]} failed: {result.stderr[:500]}')
    return json.loads(result.stdout) if json_result else {}


def memory_available() -> int:
    for row in Path('/proc/meminfo').read_text().splitlines():
        if row.startswith('MemAvailable:'):
            return int(row.split()[1]) * 1024
    raise RuntimeError('MemAvailable unavailable')


def readiness() -> dict | None:
    try:
        with urllib.request.urlopen('http://127.0.0.1:18511/health', timeout=1) as response:
            health = json.load(response)
        if health.get('status') != 'ok':
            return None
        with urllib.request.urlopen('http://127.0.0.1:18511/props', timeout=1) as response:
            properties = json.load(response)
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return None
    context = properties.get('default_generation_settings', {}).get('n_ctx')
    if context != 131072 or properties.get('model_alias') != 'persona-manager':
        raise ValueError('Planner did not expose the exact131072 context and configured alias')
    return {'context_tokens': context, 'model_alias': properties['model_alias'], 'health': health['status']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minutes', type=int, default=600)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parent / 'results/persona_planner.jsonl')
    parser.add_argument('--print-command', action='store_true')
    args = parser.parse_args()
    if args.print_command:
        print(json.dumps({'command': command(), 'vram_gib': 40, 'context_tokens': 131072,
                          'gpu_uuid': GPU, 'unit': UNIT, 'maximum_minutes': args.minutes})); return
    if socket.gethostname() != 'wtatum84' or not 1 <= args.minutes <= 960:
        raise ValueError('Planner must run on John for at most sixteen hours')
    if not args.manifest.parent.is_dir() or not RUNTIME.is_file() or not MODEL.is_file():
        raise ValueError('Existing model, runtime and manifest parent are required')
    cgroup = Path('/proc/self/cgroup').read_text()
    if UNIT not in cgroup:
        raise ValueError('Launch under the named persona-planner-john.service unit')
    if memory_available() < 32 * 1024**3:
        raise RuntimeError('The shared32GiB system-RAM guard blocks admission')
    with MODEL.open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != MODEL_SHA:
            raise ValueError('Cached planner GGUF differs from recorded checkpoint')
    spec = importlib.util.spec_from_file_location('john_thermal_policy', '/home/orwel/dev_genius/gpu_workboard/john_thermal_policy.py')
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    thermal = policy.ThermalStrikeController()
    stop = {'signal': None}
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, lambda signum, frame: stop.update(signal=signum))
    def record(event: str, **value) -> None:
        with args.manifest.open('a') as stream:
            stream.write(json.dumps({'time': time.time(), 'event': event, **value}, sort_keys=True) + '\n')
            stream.flush(); os.fsync(stream.fileno())
    job, acquired, child, outcome, peak_mib = None, False, None, 'failed', 0
    try:
        job = wb('submit', '--project', 'character-creation', '--label', 'persona-planner-john',
            '--vram-gib', '40', '--duration-min', str(args.minutes), '--host', 'wtatum84', '--gpu', GPU,
            '--lane', 'batch', '--checkpoint-signal', 'SIGUSR1', '--checkpoint-grace-s', '30', json_result=True)['id']
        record('submitted', lease_id=job, model_sha256=MODEL_SHA, command=command(), gpu_uuid=GPU)
        acquire_deadline = time.monotonic() + 300
        while not stop['signal'] and time.monotonic() < acquire_deadline:
            reply = wb('acquire', job, json_result=True)
            acquired = reply.get('acquired') is True
            if acquired:
                break
            time.sleep(2)
        if not acquired:
            raise RuntimeError('No planner admission within300s; existing leases remain untouched')
        board = wb('status', json_result=True)
        owned = next(row for row in board['jobs'] if row['id'] == job)
        if owned['assigned_uuid'] != GPU:
            raise ValueError('Workboard assigned an unexpected GPU')
        wb('start', job, '--pid', str(os.getpid()), '--unit', UNIT, '--observed-mib', '0')
        if stop['signal']:
            return
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': GPU,
                       'LD_LIBRARY_PATH': str(RUNTIME.parent) + ':' + os.environ.get('LD_LIBRARY_PATH', '')}
        child = subprocess.Popen(command(), env=environment, start_new_session=True)
        record('started', lease_id=job, child_pid=child.pid, runtime_sha256=hashlib.sha256(RUNTIME.read_bytes()).hexdigest(),
               requested_context_tokens=131072, maximum_process_mib=40 * 1024, unit=UNIT)
        started, last_heartbeat, last_probe, ready = time.monotonic(), 0.0, 0.0, False
        end = started + args.minutes * 60
        while child.poll() is None and not stop['signal'] and time.monotonic() < end:
            now = time.monotonic()
            temperatures = subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-gpu=temperature.gpu',
                                                   '--format=csv,noheader,nounits'], text=True, timeout=10)
            event = thermal.observe(int(temperatures.strip()), now)
            if event['started'] or event['cleared'] or event['request_checkpoint'] or event['stop_after_checkpoint']:
                record('thermal', detail=event)
            if event['request_checkpoint'] or event['stop_after_checkpoint']:
                outcome = 'thermal_stop'; break
            if memory_available() < 32 * 1024**3:
                outcome = 'ram_guard'; break
            lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                             '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()
            used = sum(int(parts[2]) for line in lines if len(parts := [p.strip() for p in line.split(',')]) == 3
                       and parts[0] == str(child.pid) and parts[1] == GPU)
            peak_mib = max(peak_mib, used)
            if used > 40 * 1024:
                record('vram_limit', observed_mib=used, limit_mib=40 * 1024)
                outcome = 'vram_limit'; break
            if not ready and now - last_probe >= 5:
                properties = readiness()
                last_probe = now
                if properties:
                    ready = True
                    record('ready', lease_id=job, **properties, observed_process_peak_mib=peak_mib,
                        allocator_peak_mib=None, allocator_peak_status='llama.cpp does not expose allocator peak here; process VRAM sampled every second')
                elif now - started >= 600:
                    outcome = 'readiness_timeout'; break
            if now - last_heartbeat >= 30:
                wb('heartbeat', job, '--pid', str(os.getpid()), '--observed-mib', str(used))
                record('heartbeat', lease_id=job, observed_mib=used, observed_process_peak_mib=peak_mib, ready=ready)
                last_heartbeat = now
            time.sleep(1)
        if stop['signal']:
            outcome = 'checkpointed' if stop['signal'] == signal.SIGUSR1 else 'interrupted'
        elif outcome == 'failed' and (child.poll() == 0 or time.monotonic() >= end):
            outcome = 'completed'
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
        if job is not None:
            if acquired:
                board_outcome = ('completed' if outcome == 'completed' else 'released'
                    if outcome in ('checkpointed', 'interrupted') else 'failed')
                wb('finish', job, '--outcome', board_outcome, '--detail', outcome + ': stateless planner stopped; durable experiment queue belongs to persona_manager.py')
            else:
                wb('cancel', job)
            record('finished', lease_id=job, outcome=outcome, signal=stop['signal'], observed_process_peak_mib=peak_mib)
    if outcome not in ('completed', 'checkpointed', 'interrupted'):
        raise RuntimeError(outcome)


if __name__ == '__main__':
    main()
