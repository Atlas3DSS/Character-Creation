"""Keeps the "who are you" sweep running unattended: workers, runner and the next batch.

Runs beside the explorer. Every minute it checks the GPU host. When the six batch workers have
all exited (their leased window ended) it relaunches them from the recorded
command with a fresh window; it never stops a running worker. It restarts the
runner when it is down and work is queued. When the queue runs low it queues the
next batch from a fixed menu (emotion words first, then wider feeling and
character groups). If the local planner model is already up, it picks the menu
item; otherwise the menu order is used. The planner never runs commands.
Host names, paths and URLs come from the git-ignored identity_sweep.local.json.
"""
from __future__ import annotations
import argparse
import json
import re
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / 'results/neuron_atlas/identity_sweep_workers_launch.json'
STATE = ROOT / 'results/neuron_atlas/identity_supervisor.json'
WINDOW_MIN, LOW_WATER, BATCH = 720, 1500, 2000
DISK_BUDGET = 5 * 1024**4  # raw-capture budget on the GPU host's disk
EMOTIONS = ('joy happy delight grief sad sorrow lonely anger rage furious fear afraid terror anxious dread disgust contempt '
            'surprise awe love affection tender lust desire horny arous seduc shame guilt embarrass regret pride proud envy '
            'jealous trust loyal hope grateful compassion empathy pity bored amus laugh calm serene cruel evil kind sarcas '
            'mock smug bitter tease flirt nervous').split()
MENU = ([{'id': 'word:' + w, 'name': f'Neurons labelled “{w}”', 'query': {'q': w}} for w in EMOTIONS] + [
    {'id': 'activation:feelings', 'name': 'Neurons that respond to feelings prompts', 'query': {'mode': 'activation', 'category': 'feelings'}},
    {'id': 'evidence:any', 'name': 'Every neuron with recorded evidence', 'query': {'evidence': 1}},
    {'id': 'lens:persona-up', 'name': 'Feeling & character lens, pushed up', 'query': {'mode': 'up', 'category': 'persona'}},
    {'id': 'lens:persona-down', 'name': 'Feeling & character lens, pushed down', 'query': {'mode': 'down', 'category': 'persona'}},
    {'id': 'activation:relationships', 'name': 'Neurons that respond to relationship prompts', 'query': {'mode': 'activation', 'category': 'relationships'}},
    {'id': 'activation:conflict', 'name': 'Neurons that respond to disagreement prompts', 'query': {'mode': 'activation', 'category': 'conflict'}},
    {'id': 'activation:values', 'name': 'Neurons that respond to values prompts', 'query': {'mode': 'activation', 'category': 'values'}},
    {'id': 'lens:word-up', 'name': 'Whole-word lens neurons, pushed up', 'query': {'mode': 'up', 'category': 'word'}}])


def deployment():
    """Host, paths and URLs of this installation, from the git-ignored identity_sweep.local.json."""
    path = Path(__file__).resolve().parents[1] / 'identity_sweep.local.json'
    if not path.exists(): raise SystemExit(f'Create {path.name} (see README_agent.md, identity sweep section)')
    return json.loads(path.read_text())


D = deployment()
REMOTE, PYTHON, EXPLORER, PLANNER = D['remote_root'], D['remote_python'], D['explorer_url'], D['planner_url']


def ssh(command, timeout=60):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', D['host'], command],
                          capture_output=True, text=True, timeout=timeout)


def post(url, value, timeout=120):
    request = urllib.request.Request(url, json.dumps(value).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as reply: return json.loads(reply.read())


def remote_state():
    units = [f'character-persona-probe-{i}' for i in range(6)] + ['character-identity-sweep']
    reply = ssh('systemctl --user is-active ' + ' '.join(units) + f'; cd {shlex.quote(REMOTE)} && {PYTHON} persona_identity_sweep.py status')
    lines = reply.stdout.strip().splitlines()
    return dict(zip(units, lines[:7])), json.loads(lines[-1]) if lines and lines[-1].startswith('{') else None


def relaunch_workers(log):
    record = json.loads(LAUNCH.read_text()); deadline = time.time() + WINDOW_MIN * 60
    for worker in record['workers']:
        command = [f'--property=RuntimeMaxSec={WINDOW_MIN * 60 + 32}' if x.startswith('--property=RuntimeMaxSec=')
                   else f'--setenv=PERSONA_JOB_DEADLINE={deadline:.1f}' if x.startswith('--setenv=PERSONA_JOB_DEADLINE=')
                   else x for x in worker['command']]
        reply = ssh(shlex.join(command)); worker.update(command=command, returncode=reply.returncode, receipt=(reply.stdout + reply.stderr)[:300])
    record.update(started=time.time(), deadline=deadline); LAUNCH.write_text(json.dumps(record, indent=1))
    log('workers_relaunched', deadline=deadline, codes=[w['returncode'] for w in record['workers']])
    return deadline


def start_runner(until, log):
    reply = ssh(f'systemd-run --user --unit=character-identity-sweep --collect --property=RuntimeMaxSec={WINDOW_MIN * 60} '
                f'--working-directory={shlex.quote(REMOTE)} --setenv=OMP_NUM_THREADS=4 taskset -c {D["runner_cpus"]} {PYTHON} '
                f'{shlex.quote(REMOTE + "/persona_identity_sweep.py")} run --until {int(until)} --checkpoint {shlex.quote(D["checkpoint"])}')
    log('runner_started', returncode=reply.returncode, detail=(reply.stdout + reply.stderr)[:200])


def planner_pick(options, status):
    """One short, bounded call; any malformed reply falls back to the menu order."""
    try:
        urllib.request.urlopen(PLANNER.replace('/chat/completions', '/models'), timeout=3)
    except Exception: return options[0], 'menu order (planner is not running)'
    menu = '\n'.join(f"{o['id']}: {o['name']}" for o in options[:12])
    ask = ('You choose the next batch for a neuron sweep that asks the 27B model "who are you" with one neuron pushed. '
           f"Progress so far: {json.dumps(status.get('counts'))}. Typical strongest coherent push by layer: "
           f"{json.dumps(status.get('median_strongest_coherent_dose'))}. Prefer core human emotions not yet covered.\n"
           f'Options:\n{menu}\nReply with JSON only: {{"choice": "<id>", "why": "<one sentence>"}}')
    try:
        reply = post(PLANNER, {'model': D['planner_model'], 'messages': [{'role': 'user', 'content': ask}], 'temperature': 0.2, 'max_tokens': 120}, timeout=90)
        text = reply['choices'][0]['message']['content']
        value = json.loads(re.search(r'\{.*\}', text, re.S).group(0))
        chosen = next(o for o in options if o['id'] == value['choice'])
        return chosen, 'planner: ' + str(value.get('why', ''))[:200]
    except Exception as exc: return options[0], f'menu order (planner reply unusable: {type(exc).__name__})'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--until', type=float, default=float('inf'), help='Epoch at which supervision ends (default: never)')
    args = parser.parse_args()
    state = json.loads(STATE.read_text()) if STATE.exists() else {'queued_menu': [], 'events': []}
    def log(kind, **fields):
        state['events'] = (state['events'] + [{'time': time.time(), 'event': kind, **fields}])[-40:]
    while time.time() < args.until:
        try:
            units, status = remote_state(); record = json.loads(LAUNCH.read_text())
            workers_up = sum(units.get(f'character-persona-probe-{i}') == 'active' for i in range(6))
            # None running: the window ended, a reboot or a crash. Relaunch, at most every 30 minutes.
            if workers_up == 0 and time.time() - state.get('last_relaunch', 0) > 1800:
                state['last_relaunch'] = time.time(); record['deadline'] = relaunch_workers(log)
            queued = (status or {}).get('counts', {}).get('queued', 0)
            free = int(ssh(f'df -B1 --output=avail {shlex.quote(REMOTE)} | tail -1').stdout.strip() or 0)
            state.setdefault('free_at_start', free); used = state['free_at_start'] - free
            state['disk_used_bytes'] = used
            if used > DISK_BUDGET: log('disk_budget_reached', used=used)
            elif status and queued < LOW_WATER:
                options = [o for o in MENU if o['id'] not in state['queued_menu']]
                if options:
                    chosen, why = planner_pick(options, status)
                    result = post(EXPLORER + '/api/neuron-atlas/sweep', {'name': chosen['id'], **chosen['query']})
                    state['queued_menu'].append(chosen['id']); log('batch_queued', menu=chosen['id'], why=why, result=result)
            if units.get('character-identity-sweep') != 'active' and workers_up and (queued or status is None):
                start_runner(record['deadline'] - 120, log)
            state.update(updated=time.time(), workers_up=workers_up, runner=units.get('character-identity-sweep'),
                         worker_deadline=record['deadline'], status=status, until=args.until, last_error=None)
        except Exception as exc:
            state.update(updated=time.time(), last_error=f'{type(exc).__name__}: {str(exc)[:300]}')
        STATE.write_text(json.dumps(state, indent=1, default=str))
        time.sleep(60)


if __name__ == '__main__': main()
