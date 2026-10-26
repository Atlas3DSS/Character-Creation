"""Bind the installed V100 lease supervisor to this exact named user unit.

The installed supervisor retains its admission, heartbeat, thermal-child and
process-tree cleanup behavior. Only workboard start registration gains --unit.
"""
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time


def submission_args(args, lane):
    values = list(args)
    while '--lane' in values:
        at = values.index('--lane')
        if at + 1 >= len(values): raise ValueError('Malformed installed lane option')
        del values[at:at + 2]
    if lane == 'batch':
        values = [value for value in values if value not in ('--protected', '--preemptible')]
        for option in ('--checkpoint-signal', '--checkpoint-grace-s'):
            while option in values:
                at = values.index(option)
                if at + 1 >= len(values): raise ValueError('Malformed checkpoint option')
                del values[at:at + 2]
        values += ['--preemptible', '--checkpoint-signal', 'SIGTERM', '--checkpoint-grace-s', '60']
    return [*values, '--lane', lane]


def watch_deadline(done, deadline):
    """Ask this supervisor to run its installed cleanup at the absolute deadline."""
    while not done.is_set():
        remaining = deadline - time.time()
        if remaining <= 0:
            # Deliver to the main thread's installed handler. Calling on_stop
            # in this watcher would only exit the watcher, not the supervisor.
            os.kill(os.getpid(), signal.SIGTERM)
            return
        done.wait(min(remaining, 1.))


def main():
    lane = os.environ.get('PERSONA_WORKBOARD_LANE', 'interactive')
    if lane not in ('interactive', 'batch'):
        raise ValueError('Workboard lane must be interactive or batch')
    unit = os.environ['PERSONA_WORKER_UNIT']
    wrapper = Path(os.environ['PERSONA_LEASE_WRAPPER'])
    expected = os.environ['PERSONA_LEASE_WRAPPER_SHA256']
    if (not re.fullmatch(r'character-persona-probe-[0-5]\.service', unit)
            or unit not in Path('/proc/self/cgroup').read_text()
            or len(sys.argv) < 2 or sys.argv[1] != 'run'):
        raise ValueError('A fixed probe unit and run invocation are required')
    if hashlib.sha256(wrapper.read_bytes()).hexdigest() != expected:
        raise ValueError('Installed lease supervisor differs from the operator-verified hash')
    spec = importlib.util.spec_from_file_location('persona_installed_lease', wrapper)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original = module.wb
    def bound(*args):
        if args[0] == 'submit':
            return original(*submission_args(args, lane))
        if args[0] == 'start':
            return original(*args, '--unit', unit)
        return original(*args)
    module.wb = bound
    signal.signal(signal.SIGTERM, module.on_stop)
    signal.signal(signal.SIGINT, module.on_stop)
    done = threading.Event()
    try:
        raw = os.environ.get('PERSONA_JOB_DEADLINE')
        if raw is not None:
            deadline = float(raw)
            if not math.isfinite(deadline): raise ValueError('Scheduled lease deadline must be finite')
            if deadline <= time.time(): return 143  # No lease submission or child after expiry.
            threading.Thread(target=watch_deadline, args=(done, deadline), daemon=True).start()
        return module.main(sys.argv)
    finally:
        done.set()


if __name__ == '__main__':
    sys.exit(main())
