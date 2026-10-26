"""One finite six-interior-excerpt qualification after worker readiness.

No full-corpus LLM annotation loop. Workers remain available to the UI until
existing finite workboard leases and systemd RuntimeMaxSec expire naturally.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
import time

from persona_literature_annotate import fetch, verify_descriptor

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / 'results/persona_literature_batch.json'
CONFIG = ROOT / 'activation-search/persona_fleet.local.json'
UNITS = [f'character-activation-search-probe-{i}.service' for i in range(6)]
TUNNEL = 'character-activation-search-tunnel.service'
STARTED = time.time()


def publish(state: str, **details) -> None:
    value = {'state': state, 'started': STARTED, 'updated': time.time(), 'finite': True,
             'annotation_limit': 6, 'full_corpus_llm_loop': False, **details}
    STATUS.write_text(json.dumps(value, indent=2, allow_nan=False))
    print(json.dumps(value, allow_nan=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attach-existing', action='store_true')
    args = parser.parse_args()
    previous = json.loads(STATUS.read_text()) if args.attach_existing else None
    config = json.loads(CONFIG.read_text())
    endpoints = json.loads((ROOT / 'persona_explorer.local.json').read_text())['activation_endpoints']
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ControlMaster=no', '-o', 'ControlPath=none', config['ssh_host']]
    def remote(arguments: list[str]) -> str:
        return subprocess.check_output([*ssh, shlex.join(arguments)], text=True, timeout=30).strip()
    if remote(['hostname']) != config['hostname']:
        raise ValueError('Remote hostname differs')
    def properties() -> dict:
        raw = remote(['systemctl', '--user', 'show', *UNITS, '-p', 'Id', '-p', 'ActiveState', '-p', 'InvocationID', '-p', 'RuntimeMaxUSec'])
        result = {}
        for block in raw.split('\n\n'):
            values = dict(line.split('=', 1) for line in block.splitlines() if '=' in line)
            if values.get('Id'):
                result[values['Id']] = values
        return result
    if args.attach_existing:
        owned = previous.get('invocations', {})
        states = properties()
        if set(owned) != set(UNITS) or any(states.get(name, {}).get('InvocationID') != identity for name, identity in owned.items()):
            raise ValueError('Existing worker invocation ownership differs from saved coordinator state')
        launch_time = previous.get('lease_started', previous['started'])
    else:
        deadline = STARTED + 20 * 60
        while True:
            states = properties()
            active = [name for name in UNITS if states.get(name, {}).get('ActiveState') not in ('inactive', 'failed')]
            publish('waiting_for_natural_interactive_expiry', active_units=active, deadline=deadline)
            if not active:
                break
            if time.time() >= deadline:
                raise TimeoutError('Existing session did not finish; no new leases submitted')
            time.sleep(15)
        subprocess.run(['systemctl', '--user', 'stop', TUNNEL], check=True, timeout=30)
        launch_time = time.time()
        publish('launching_batch_workers', lane='batch', minutes=240)
        subprocess.run([sys.executable, str(ROOT / 'activation-search/launch.py'), '--lane', 'batch', '--minutes', '240'], check=True, timeout=180)
        owned = {name: values['InvocationID'] for name, values in properties().items() if values.get('InvocationID')}
        if set(owned) != set(UNITS):
            raise RuntimeError('Could not bind all six new worker invocation IDs')
    readiness_deadline = min(time.time() + 20 * 60, launch_time + 240 * 60 - 1260)
    while True:
        ready, failures = [], {}
        for endpoint in endpoints:
            try:
                verify_descriptor(endpoint, fetch(endpoint['url'] + '/v1/models', timeout=5))
                ready.append(endpoint['url'])
            except Exception as error:
                failures[endpoint['url']] = f'{type(error).__name__}: {error}'
        publish('waiting_for_qualification_readiness', ready=ready, failures=failures, deadline=readiness_deadline,
                invocations=owned, lease_started=launch_time)
        if len(ready) == 6:
            break
        if time.time() >= readiness_deadline:
            raise TimeoutError('Workers did not become ready; pending annotation units untouched')
        time.sleep(15)
    board = json.loads(subprocess.check_output(['/home/orwel/.local/bin/gpu-workboard', 'status', '--json'], text=True, timeout=30))
    leases = [j for j in board['jobs'] if j.get('unit_name') in owned and j.get('status') == 'running' and j.get('lane') == 'batch']
    if {j['assigned_uuid'] for j in leases} != set(config['gpus']):
        raise RuntimeError('Six identity-matched running batch leases were not verified')
    minutes = min(30, int((launch_time + 240 * 60 - time.time() - 1260) // 60))
    if minutes < 1:
        raise TimeoutError('No safe annotation qualification window remains')
    qualification_started = time.time()
    publish('qualifying_six_interior_excerpts', minutes=minutes, lease_ids=[j['id'] for j in leases],
            invocations=owned, lease_started=launch_time)
    subprocess.run([sys.executable, str(ROOT / 'persona/persona_literature_annotate.py'),
                    '--database', str(ROOT / 'results/persona_literature.sqlite'),
                    '--config', str(ROOT / 'persona_explorer.local.json'), '--minutes', str(minutes), '--limit', '6'],
                   check=True, timeout=minutes * 60 + 1200)
    with sqlite3.connect(ROOT / 'results/persona_literature.sqlite') as db:
        rows = db.execute('SELECT unit_id,state,annotation_json,error FROM literature_attempts WHERE started>=? ORDER BY started', (qualification_started,)).fetchall()
    summary = []
    for identity, state, annotation, error in rows:
        data = json.loads(annotation) if annotation else {}
        summary.append({'unit_id': identity, 'state': state, 'scene_present': data.get('scene') is not None,
                        'entities': len(data.get('entities', [])), 'themes': len(data.get('themes', [])), 'error': error})
    publish('qualification_complete' if len(summary) == 6 and all(r['state'] == 'complete' for r in summary) else 'qualification_incomplete',
            units=summary, invocations=owned, lease_started=launch_time,
            worker_policy='Kept available to UI until existing finite lease/systemd expiry; no bulk annotation follows.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        publish('failed', error=f'{type(error).__name__}: {error}')
        raise
