"""Fixed persona-unit inspection and token-bound pidfd cleanup.

Never stop a service by its reusable name. A pidfd is opened for its MainPID,
then invocation, owner token and cgroup are rechecked before SIGTERM. Only that
immutable process handle is signalled; a replacement invocation is untouched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import select
import shlex
import signal
import socket
import subprocess
import time

PLANNER_UNIT = 'persona-planner-john.service'
TUNNEL_UNIT = 'character-persona-jobs-tunnel.service'
PROBE_UNITS = tuple(f'character-persona-probe-{i}.service' for i in range(6))
UNITS = (PLANNER_UNIT, TUNNEL_UNIT, *PROBE_UNITS)
TOKEN_ENV, JOB_ENV = 'PERSONA_LAUNCH_TOKEN', 'PERSONA_JOB_ID'


def validate_owner(unit, job_id, token):
    if unit not in UNITS: raise ValueError('Only fixed persona service units are allowed')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', job_id): raise ValueError('Invalid scheduled job identity')
    if not re.fullmatch(r'[0-9a-f]{32}', token): raise ValueError('Invalid unique launch token')


def unit_state(unit):
    if unit not in UNITS: raise ValueError('Only fixed persona service units are allowed')
    result = subprocess.run(['systemctl', '--user', 'show', unit,
        '--property=LoadState,ActiveState,InvocationID,MainPID,ControlGroup,Environment'],
        text=True, capture_output=True, timeout=15)
    state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if result.returncode and state.get('LoadState') != 'not-found':
        raise RuntimeError('Cannot inspect fixed service: ' + result.stderr[-500:])
    if not state.get('LoadState') or not state.get('ActiveState'):
        raise RuntimeError('Service inspection did not return a complete state')
    environment = dict(item.split('=', 1) for item in shlex.split(state.pop('Environment', '')) if '=' in item)
    state['job_id'], state['launch_token'] = environment.get(JOB_ENV), environment.get(TOKEN_ENV)
    state['MainPID'] = int(state.get('MainPID') or 0)
    return state


def proc_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        groups = [line.split(':', 2)[2] for line in Path(f'/proc/{pid}/cgroup').read_text().splitlines()]
    except (FileNotFoundError, ProcessLookupError): return None
    return {'start_time': int(fields[19]), 'groups': groups}


def in_group(identity, group):
    return bool(identity and group and group.startswith('/') and any(
        path == group or path.startswith(group + '/') for path in identity['groups']))


def stop_owned(unit, job_id, token, *, invocation_id=None, timeout=90):
    validate_owner(unit, job_id, token)
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Linux pidfd support is required for ownership-safe cleanup')
    if not 0 < timeout <= 100: raise ValueError('Cleanup timeout must be bounded by100 seconds')
    until = time.monotonic() + timeout
    while True:
        first = unit_state(unit)
        if first.get('LoadState') == 'not-found' or first.get('ActiveState') in ('inactive', 'failed'):
            return {'stopped': False, 'status': 'already_inactive'}
        if first.get('job_id') != job_id or first.get('launch_token') != token:
            return {'stopped': False, 'status': 'different_owner'}
        if invocation_id and first.get('InvocationID') != invocation_id:
            return {'stopped': False, 'status': 'different_invocation'}
        pid, invocation, group = first['MainPID'], first.get('InvocationID'), first.get('ControlGroup')
        if not pid or not invocation:
            if time.monotonic() >= until: raise RuntimeError('Owned unit has no verifiable MainPID yet')
            time.sleep(.1); continue
        try: descriptor = os.pidfd_open(pid)
        except ProcessLookupError:
            if time.monotonic() >= until: raise RuntimeError('Owned MainPID changed repeatedly during cleanup')
            continue
        try:
            identity = proc_identity(pid)
            second = unit_state(unit)
            repeated = proc_identity(pid)
            keys = ('MainPID', 'InvocationID', 'ControlGroup', 'job_id', 'launch_token')
            if any(first.get(key) != second.get(key) for key in keys):
                return {'stopped': False, 'status': 'identity_changed_before_signal'}
            if not in_group(identity, group) or not in_group(repeated, group) or identity['start_time'] != repeated['start_time']:
                raise RuntimeError('MainPID does not have stable membership in its verified unit cgroup')
            try: signal.pidfd_send_signal(descriptor, signal.SIGTERM)
            except ProcessLookupError:
                return {'stopped': False, 'status': 'original_process_already_exited', 'invocation_id': invocation}
            poll = select.poll(); poll.register(descriptor, select.POLLIN)
            while time.monotonic() < until:
                if poll.poll(max(1, min(1000, int((until - time.monotonic()) * 1000)))):
                    return {'stopped': True, 'status': 'owned_process_exited', 'invocation_id': invocation, 'pid': pid}
            # Do not kill a supervisor before it confirms descendant cleanup and
            # releases its exact leases. The unit's RuntimeMaxSec still bounds it.
            raise RuntimeError('Owned supervisor did not finish cooperative cleanup within its bound')
        finally: os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['state', 'stop'])
    parser.add_argument('--unit', choices=UNITS, required=True)
    parser.add_argument('--job-id'); parser.add_argument('--token'); parser.add_argument('--invocation-id')
    parser.add_argument('--timeout', type=float, default=90)
    args = parser.parse_args()
    expected_host = 'v100' if args.unit in PROBE_UNITS else 'wtatum84'
    if socket.gethostname() != expected_host: raise ValueError('Fixed service is on the wrong configured host')
    if args.command == 'state': result = unit_state(args.unit)
    else: result = stop_owned(args.unit, args.job_id or '', args.token or '', invocation_id=args.invocation_id, timeout=args.timeout)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__': main()
