"""Finite, source-bound six-Quincy supervisor; never acquires GPUs on import.

Run only inside character-persona-phantom-train.service after serving leases
have ended. --bindings is a JSON object mapping absolute input/source paths to
SHA256. The operator supplies the exact trainer command after --. The installed
supervisor provides ordered admission, uncertain-receipt handling and descendant
cleanup; this adapter never changes its installed bytes or workboard database.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

UNIT = 'character-persona-phantom-train.service'
WRAPPER = '/opt/atlas/v100-qwen38/wb_hold.py'
WRAPPER_SHA = '1ea64d4a6f0c87781f16695476505f3c5b044b78f666a4097c8ba860bf907c35'
CARDS = dict(zip(('v100-' + str(i) for i in range(6)), (
    'GPU-725106e3-11df-e1fc-003e-99f071a4e353',
    'GPU-8747fa38-17e6-3fb3-f9da-4ebb5ac128a1',
    'GPU-30ab07ae-8f44-df0e-ce62-9b6b088f1d00',
    'GPU-5b95eabd-c3e3-a8bf-e8a0-f2717e731f07',
    'GPU-f2b6541e-8121-c7fd-8a73-78d4b1743d3e',
    'GPU-6f2541c5-80ed-71d7-5892-8fe9af1398eb')))

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def verify_bindings(bindings):
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError('Nonempty source/input bindings required')
    for path, expected in bindings.items():
        if not Path(path).is_absolute() or not isinstance(expected, str) or len(expected) != 64 or sha(path) != expected:
            raise ValueError('Source/input binding mismatch: ' + path)

def validate_command(command, trainer, bindings):
    if len(command) < 4 or Path(command[1]).resolve() != Path(trainer).resolve() or command[2] not in ('validate', 'train'):
        raise ValueError('Expected exact Python trainer validate/train command')
    if str(Path(trainer).resolve()) not in bindings:
        raise ValueError('Trainer must have a source binding')
    for option in ('--dataset', '--checkpoint', '--model-identity', '--output', '--devices'):
        if command.count(option) != 1:
            raise ValueError('Required unique trainer option: ' + option)
    dataset = str(Path(command[command.index('--dataset') + 1]).resolve())
    if dataset not in bindings:
        raise ValueError('Dataset must have a source binding')
    at = command.index('--devices') + 1
    devices = []
    while at < len(command) and not command[at].startswith('--'):
        devices.append(command[at]); at += 1
    if devices != ['cuda:' + str(i) for i in range(6)]:
        raise ValueError('Trainer must use exactly all six mapped devices in order')

def batch_args(args):
    values = [v for v in args if v not in ('--protected', '--preemptible')]
    return [*values, '--preemptible', '--checkpoint-signal', 'SIGTERM', '--checkpoint-grace-s', '60']

def memory_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable unavailable')

def card_usage(text, pids):
    result = dict.fromkeys(CARDS.values(), 0)
    for line in text.splitlines():
        if not line.strip(): continue
        pid, uuid, mib = (v.strip() for v in line.split(','))
        if int(pid) in pids:
            if uuid not in result:
                raise RuntimeError('Owned process used an unleased GPU')
            result[uuid] += int(mib)
    return result

def check_resources(temperatures, usage, ram):
    if ram < 32 * 1024**3:
        raise RuntimeError('32 GiB system RAM floor reached; stop, do not silently shrink')
    for uuid in CARDS.values():
        if uuid not in temperatures or not math.isfinite(temperatures[uuid]) or temperatures[uuid] >= 85:
            raise RuntimeError('Missing/unsafe GPU temperature: ' + uuid)
        if usage.get(uuid, 0) > 31 * 1024:
            raise RuntimeError('Owned GPU allocation exceeds lease: ' + uuid)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings', required=True)
    parser.add_argument('--trainer', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--deadline', required=True, type=float)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    remaining = args.deadline - time.time()
    if not math.isfinite(remaining) or not 0 < remaining <= 90 * 60:
        raise ValueError('Deadline must be future and within 90 minutes')
    if socket.gethostname().split('.')[0] != 'v100' or UNIT not in Path('/proc/self/cgroup').read_text():
        raise RuntimeError('Run only in the exact named unit on v100')
    bindings = json.loads(Path(args.bindings).read_text())
    verify_bindings(bindings)
    validate_command(command, args.trainer, bindings)
    if sha(WRAPPER) != WRAPPER_SHA:
        raise RuntimeError('Installed supervisor source differs')
    report = Path(args.report)
    # Exclusive report creation prevents accidental run-lineage overwrite.
    with report.open('x') as stream:
        json.dump({'state': 'admitting', 'bindings': bindings, 'command': command,
                   'deadline': args.deadline, 'gpus': CARDS, 'supervisor_sha256': sha(__file__)}, stream)
    rc, jobs = 1, {}
    done = threading.Event()
    try:
        spec = importlib.util.spec_from_file_location('phantom_installed_hold', WRAPPER)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        original_wb, original_stop = module.wb, module.stop_child
        jobs, events = {}, []
        done = threading.Event()
        def write_event(kind, **fields):
            event = {'time': time.time(), 'kind': kind, **fields}
            events.append(event)
            with Path(str(report) + '.jsonl').open('a') as stream:
                stream.write(json.dumps(event) + '\n'); stream.flush(); os.fsync(stream.fileno())
        def bound(*values):
            if values[0] == 'submit':
                values = tuple(batch_args(values))
            elif values[0] == 'start':
                status = original_wb('status')
                job = next((j for j in status['jobs'] if j['id'] == values[1]), None)
                if job is None or job.get('assigned_uuid') != jobs.get(values[1]) or job.get('lane') != 'batch' or not job.get('preemptible'):
                    raise RuntimeError('Acquired lease UUID/lane/preemption identity differs')
                values = (*values, '--unit', UNIT)
            result = original_wb(*values)
            if values[0] == 'submit':
                card = values[values.index('--gpu') + 1]
                jobs[result['id']] = CARDS[card]
            if values[0] in ('submit', 'acquire', 'start', 'finish', 'cancel'):
                write_event(values[0], arguments=list(values), receipt=result)
            return result
        def sample(_root=None):
            owned = module.tree(os.getpid())
            raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=10)
            usage = card_usage(raw, owned)
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,temperature.gpu', '--format=csv,noheader,nounits'], text=True, timeout=10)
            temperatures = {p[0].strip(): float(p[1]) for line in raw.splitlines() if len(p := line.split(',')) == 2}
            check_resources(temperatures, usage, memory_available())
            write_event('telemetry', usage_mib=usage, temperatures=temperatures)
            return usage
        def heartbeat(held, ignored):
            usage = sample()
            for job in held:
                bound('heartbeat', job, '--pid', str(os.getpid()), '--observed-mib', str(usage[jobs[job]]))
        def graceful_stop(proc):
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=50)
                except subprocess.TimeoutExpired: pass
            original_stop(proc)  # Verified pidfd/subreaper cleanup before releasing leases.
        class ChildProcesses:
            def __getattr__(self, name):
                return getattr(subprocess, name)
            def Popen(self, *values, **options):
                verify_bindings(bindings)
                sample()
                if time.time() >= args.deadline:
                    raise RuntimeError('Deadline expired before launch')
                return subprocess.Popen(*values, **options)
        module.subprocess = ChildProcesses()
        module.wb, module.heartbeat, module.vram_mib, module.stop_child = bound, heartbeat, sample, graceful_stop
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(CARDS.values())
        os.environ['WB_HOLD_WAIT'] = str(min(300, max(1, int(remaining))))
        affinity = set(os.sched_getaffinity(0)) - {43, 87}
        if not affinity: raise RuntimeError('No CPUs available outside reserved physical core')
        os.sched_setaffinity(0, affinity)
        sample()
        verify_bindings(bindings)
        signal.signal(signal.SIGTERM, module.on_stop)
        signal.signal(signal.SIGINT, module.on_stop)
        def deadline_watch():
            if not done.wait(max(0, args.deadline - time.time())):
                os.kill(os.getpid(), signal.SIGTERM)
        threading.Thread(target=deadline_watch, daemon=True).start()
        rc = module.main(['phantom', 'run', 'character-persona-phantom', 'frozen-hybrid-graft',
                          '31', str(max(1, math.ceil(remaining / 60))), ','.join(CARDS), '--', *command])
        return rc
    finally:
        done.set()
        terminal = {'state': 'completed' if rc == 0 else 'stopped_or_failed', 'exit_code': rc,
                    'bindings': bindings, 'command': command, 'deadline': args.deadline,
                    'gpus': CARDS, 'jobs': jobs, 'ended': time.time(),
                    'supervisor_sha256': sha(__file__), 'events': str(report) + '.jsonl'}
        temporary = report.with_name(report.name + '.tmp')
        with temporary.open('w') as stream:
            json.dump(terminal, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, report)

if __name__ == '__main__':
    sys.exit(main())
