"""Keeps the "who are you" sweep running unattended: workers, runner and the next batch.

Runs beside the explorer. Every minute it checks the GPU host. Workers follow the work
(2026-10-30 fix): it launches batch workers only when there is work for them (sweep or
baseline40k rows queued/running, a threshold campaign searching, label qualification pending,
or a menu batch it has just queued) and only as many as that work needs (at most six). Leases
ask the workboard for any card; the granted card fixes the shard and port. Workers it launched
that sit idle for 15 minutes are stopped so their cards go back to the workboard. The rollover
hold file suppresses runner starts, worker launches and idle stops. It restarts the
runner when it is down and work is queued. When the queue runs low it queues the
next batch from a fixed menu (emotion words first, then wider feeling and
character groups). If the local planner model is already up, it picks the menu
item; otherwise the menu order is used. The planner never runs commands.
Host names, paths and URLs come from the git-ignored identity_sweep.local.json.
"""
from __future__ import annotations
import argparse
import json
import math
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / 'results/neuron_atlas/identity_sweep_workers_launch.json'
STATE = ROOT / 'results/neuron_atlas/identity_supervisor.json'
WINDOW_MIN, LOW_WATER, BATCH = 720, 1500, 2000
RUNNER_STOP_S, RUNNER_DRAIN_S = 1000, 1500  # SIGTERM drain budget (runner joins in-flight batches <= 960 s); runner `until` ends this long before RuntimeMaxSec
HOLD = 'monkey/identity_supervisor.hold'  # on the GPU host under REMOTE: JSON {reason, by, at, expires}; present + unexpired = no runner start, no worker launch, no idle stop
# Worker scheduling (2026-10-30): launch only for work, size to the work, give idle cards back.
MAX_WORKERS = 6                # unit slots character-persona-probe-0..5 (not cards: any-card leases pick the card)
WORKER_UNIT = 'character-persona-probe-{}'
PER_WORKER_ROWS = 500          # queued+running rows one worker is launched for: need = ceil(rows / 500), capped at MAX_WORKERS
IDLE_STOP_S = 900              # workers up with no work and no answers this long are stopped (cards back to the workboard)
RELAUNCH_GAP_S = 1800          # at most one launch round per 30 minutes (crash/lease-wait loops)
WORKER_CAMPAIGN_STATES = ('baseline', 'searching')  # threshold campaigns the runner serves from the workers
FREE_UNIT_STATES = ('inactive', 'failed', '')       # `systemctl is-active` states of an unused slot
TERMINAL_JOB_STATES = ('preempted', 'done', 'completed', 'finished', 'failed', 'cancelled', 'canceled', 'released', 'expired', 'reaped', 'dead')
BINDER = 'monkey/identity_anycard_worker.py'        # on the GPU host under REMOTE: binds shard/port from the granted lease
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
    units = [f'character-persona-probe-{i}' for i in range(6)] + ['character-identity-sweep', 'character-label-qualify']
    reply = ssh('systemctl --user is-active ' + ' '.join(units) + f'; cd {shlex.quote(REMOTE)} && {PYTHON} persona_wide_identity_sweep.py status')
    lines = reply.stdout.strip().splitlines()
    return dict(zip(units, lines[:8])), json.loads(lines[-1]) if lines and lines[-1].startswith('{') else None


def worker_need(status, labels, queued):
    """(workers the current work needs, reasons). None when the sweep status is unknown: never launch or stop on it."""
    if status is None: return None, ['sweep status unavailable']
    rows = queued + (status.get('counts') or {}).get('running', 0)
    base = sum(v for k, v in ((status.get('baseline40k') or {}).get('counts') or {}).items() if k in ('queued', 'running'))
    need, why = math.ceil((rows + base) / PER_WORKER_ROWS), []
    if rows: why.append(f'sweep queued+running {rows}')
    if base: why.append(f'baseline40k queued+running {base}')
    searching = [c.get('id', '')[:8] for c in (status.get('threshold') or {}).get('campaigns') or [] if c.get('state') in WORKER_CAMPAIGN_STATES]
    if searching: need = MAX_WORKERS; why.append('threshold campaign ' + ','.join(searching))
    if labels and not labels.get('complete'):
        pending = sum(v for k, v in (labels.get('tasks') or {}).items() if not k.endswith(':done'))
        need = max(need, math.ceil(pending / PER_WORKER_ROWS), 1); why.append(f'label qualification pending ({pending} tasks)')
    return min(need, MAX_WORKERS), why


def answering(status):
    """Any answers in the last 10 minutes (sweep or baseline): someone is using the workers."""
    status = status or {}
    return bool((status.get('last_10_min') or {}).get('answers') or ((status.get('baseline40k') or {}).get('last_10_min') or {}).get('answers'))


def split_worker(command):
    """A recorded worker command -> (lease part before '--', taskset CPU list, server python, server path, server args after 'run')."""
    sep = command.index('--'); tail = command[sep + 1:]
    if tail[:2] != ['taskset', '-c'] or tail[5] != 'run': raise ValueError('unexpected recorded worker command shape')
    return command[:sep], tail[2], tail[3], tail[4], tail[6:]


def drop_option(args, name):
    out, skip = [], False
    for x in args:
        if skip: skip = False; continue
        if x == name: skip = True; continue
        out.append(x)
    return out


def windowed(x, deadline):
    return (f'--property=RuntimeMaxSec={WINDOW_MIN * 60 + 32}' if x.startswith('--property=RuntimeMaxSec=')
            else f'--setenv=PERSONA_JOB_DEADLINE={deadline:.1f}' if x.startswith('--setenv=PERSONA_JOB_DEADLINE=') else x)


def anycard_command(record, slot, deadline):
    """Slot unit with an any-card lease: no CUDA_VISIBLE_DEVICES, PERSONA_LEASE_ANY_CARD=1, and the binder picks
    --shard/--port/CPU set from the card the workboard grants. Built from the recorded command (no card named here)."""
    template = record['workers'][0]; head, _, python, server, args = split_worker(template['command'])
    port = int(args[args.index('--port') + 1]); base = port - int(template['shard'])
    cpus = {str(w['shard']): split_worker(w['command'])[1] for w in record['workers']}
    unit = WORKER_UNIT.format(slot); out = []
    for x in head:
        if x.startswith('--setenv=CUDA_VISIBLE_DEVICES='): continue
        x = windowed(x, deadline)
        if x.startswith('--unit='): x = f'--unit={unit}'
        elif x.startswith('--setenv=PERSONA_WORKER_UNIT='): x = f'--setenv=PERSONA_WORKER_UNIT={unit}.service'
        out.append(x)
    lease = next(i for i, x in enumerate(out) if x.endswith('/persona_lease.py'))
    out[lease + 3] = re.sub(r'-\d+$', f'-{slot}', out[lease + 3])  # lease label persona-identity-sweep-<slot>; becomes ...-<slot>-any
    out.insert(lease - 1, '--setenv=PERSONA_LEASE_ANY_CARD=1')
    rest = drop_option(drop_option(args, '--shard'), '--port')
    return out + ['--', python, f'{REMOTE}/{BINDER}', '--server', server, '--base-port', str(base), '--cpus', json.dumps(cpus), '--python', python, '--'] + rest


def launch_mode():
    """'anycard' when the host's lease wrapper honours PERSONA_LEASE_ANY_CARD and the binder is installed, else None."""
    reply = ssh(f'cd {shlex.quote(REMOTE)} && grep -q PERSONA_LEASE_ANY_CARD persona_lease.py && test -f {shlex.quote(BINDER)} && echo anycard')
    return 'anycard' if reply.stdout.strip() == 'anycard' else None


BOARD = ("import json, subprocess; w = json.load(open('persona_fleet.local.json'))['workboard']; "
         "print(subprocess.check_output([w, 'status', '--json'], text=True, timeout=60))")


def busy_cards():
    """GPU UUIDs held or claimed by any live workboard job (pinned fallback only: never pre-empt a card in use)."""
    board = json.loads(ssh(f'cd {shlex.quote(REMOTE)} && {PYTHON} -c {shlex.quote(BOARD)}', timeout=90).stdout)
    busy = {j['assigned_uuid'] for j in board.get('jobs', []) if j.get('assigned_uuid') and j.get('status') not in TERMINAL_JOB_STATES}
    for card in (board.get('fleet') or {}).get('cards', []):
        if card.get('active_job_ids') or card.get('starting_job_ids') or card.get('exclusive_job_ids'): busy.add(card.get('uuid'))
    return busy


def recorded_card(command):
    return next((x.split('=', 2)[2] for x in command if x.startswith('--setenv=CUDA_VISIBLE_DEVICES=')), None)


def relaunch_workers(count, units, log, why=()):
    """Start up to `count` workers in free unit slots. Any-card leases when the host supports them; otherwise (blocker:
    installed persona_lease.py has no any-card block) the recorded pinned commands, only on cards no live job holds."""
    record = json.loads(LAUNCH.read_text()); deadline = time.time() + WINDOW_MIN * 60
    free = [i for i in range(MAX_WORKERS) if units.get(WORKER_UNIT.format(i), '') in FREE_UNIT_STATES]
    mode, launched = launch_mode(), []
    if mode == 'anycard':
        for slot in free[:count]:
            command = anycard_command(record, slot, deadline); reply = ssh(shlex.join(command))
            record.setdefault('anycard', {})[str(slot)] = {'command': command, 'returncode': reply.returncode, 'receipt': (reply.stdout + reply.stderr)[:300], 'at': time.time()}
            launched.append((WORKER_UNIT.format(slot), reply.returncode))
    else:
        mode, busy = 'pinned-free-cards', busy_cards()
        for worker in [w for w in record['workers'] if w['shard'] in free and recorded_card(w['command']) not in busy][:count]:
            command = [windowed(x, deadline) for x in worker['command']]
            reply = ssh(shlex.join(command)); worker.update(command=command, returncode=reply.returncode, receipt=(reply.stdout + reply.stderr)[:300])
            launched.append((WORKER_UNIT.format(worker['shard']), reply.returncode))
    if launched: record.update(started=time.time(), deadline=deadline)
    LAUNCH.write_text(json.dumps(record, indent=1))
    log('workers_relaunched', mode=mode, requested=count, why=list(why), deadline=deadline, units=[u for u, _ in launched], codes=[c for _, c in launched],
        **({} if mode == 'anycard' else {'blocker': 'installed persona_lease.py lacks PERSONA_LEASE_ANY_CARD (or binder missing); pinned recorded commands on free cards only'}))
    return (deadline if launched else record['deadline']), len(launched)


def owned_invocations(record):
    """unit -> systemd invocation ID of the workers this supervisor launched (from the launch receipts)."""
    found = {}
    for entry in list(record.get('workers', [])) + list((record.get('anycard') or {}).values()):
        m = re.search(r'Running as unit: ([\w.@-]+?)\.service; invocation ID: (\w+)', entry.get('receipt') or '')
        if m: found[m.group(1)] = m.group(2)
    return found


def stop_idle_workers(units, log, idle_s):
    """Stop the active workers this supervisor launched (same invocation), so their leases end and the cards return."""
    record = json.loads(LAUNCH.read_text()); owned = owned_invocations(record)
    active = [WORKER_UNIT.format(i) for i in range(MAX_WORKERS) if units.get(WORKER_UNIT.format(i)) == 'active']
    reply = ssh('for u in ' + ' '.join(active) + '; do echo "$u $(systemctl --user show -p InvocationID --value $u.service)"; done')
    current = dict(line.split(' ', 1) for line in reply.stdout.strip().splitlines() if ' ' in line)
    mine = [u for u in active if owned.get(u) and current.get(u, '').strip() == owned[u]]
    stopped = ssh('systemctl --user stop ' + ' '.join(f'{u}.service' for u in mine), timeout=240) if mine else None
    log('workers_idle_stopped', idle_s=int(idle_s), units=mine, not_owned=sorted(set(active) - set(mine)),
        returncode=None if stopped is None else stopped.returncode)
    return mine


def held(text, now):
    """A hold file written by a rollover (phasea_successor_launch.py hold/release). Unparseable = not held; expired = not held."""
    try: record = json.loads(text)
    except (TypeError, ValueError): return None
    if not isinstance(record, dict): return None
    expires = record.get('expires')
    if expires is not None and float(expires) <= now: return None
    return record


def hold_state(log, state):
    reply = ssh(f'cat {shlex.quote(REMOTE + "/" + HOLD)} 2>/dev/null')
    record = held(reply.stdout, time.time())
    if bool(record) != bool(state.get('held')):
        log('runner_hold' if record else 'runner_hold_released', record=record or state.get('held'))
    state['held'] = record
    return record


def start_runner(until, log):
    reply = ssh(f'systemd-run --user --unit=character-identity-sweep --collect --property=RuntimeMaxSec={WINDOW_MIN * 60} --property=TimeoutStopSec={RUNNER_STOP_S} '
                f'--working-directory={shlex.quote(REMOTE)} --setenv=OMP_NUM_THREADS=4 taskset -c {D["runner_cpus"]} {PYTHON} '
                f'{shlex.quote(REMOTE + "/persona_wide_identity_sweep.py")} run --until {int(until)} --checkpoint {shlex.quote(D["checkpoint"])} --ports 8930-8935')
    log('runner_started', returncode=reply.returncode, detail=(reply.stdout + reply.stderr)[:200])


def label_status():
    reply = ssh(f'cd {shlex.quote(REMOTE)} && test -f results/label_qualify/qualify.sqlite && {PYTHON} persona_label_qualify.py status')
    return json.loads(reply.stdout.strip().splitlines()[-1]) if reply.returncode == 0 and reply.stdout.strip() else None


def start_labels(until, log):
    reply = ssh(f'systemd-run --user --unit=character-label-qualify --collect --property=RuntimeMaxSec={WINDOW_MIN * 60} '
                f'--working-directory={shlex.quote(REMOTE)} --setenv=OMP_NUM_THREADS=2 {PYTHON} '
                f'{shlex.quote(REMOTE + "/persona_label_qualify.py")} run --until {int(until)} --workers 16')
    log('labels_started', returncode=reply.returncode, detail=(reply.stdout + reply.stderr)[:200])


def publish_labels(log):
    """Copy v2 evidence beside the panel's registry and add or replace one row per concept; v1 rows stay."""
    import hashlib
    listing = ssh(f'ls {shlex.quote(REMOTE)}/results/label_qualify/ | grep "^persona_label_evidence_v2_" || true').stdout.split()
    registry_path = ROOT / 'results/persona_neuron_labels.json'
    registry = json.loads(registry_path.read_text())
    rows = {r['evidence']['artifact']: r for r in registry['labels'] if r.get('evidence', {}).get('artifact', '').startswith('persona_label_evidence_v2_')}
    passed = []
    for name in listing:
        local = ROOT / 'results' / name
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', f"{D['host']}:{REMOTE}/results/label_qualify/{name}", str(local)], timeout=120, check=True)
        raw = local.read_bytes(); proof = json.loads(raw)
        rows[name] = {'layer': proof['layer'], 'neuron': proof['neuron'], 'label': proof['label'], 'direction': proof['direction'],
                      'confidence': proof['confidence'], 'evidence': {'artifact': name, 'sha256': hashlib.sha256(raw).hexdigest()},
                      'association': {'measurement': proof['measurement'], 'standardized_contrast': proof['frozen_candidate']['standardized_contrast'],
                                      'steering_sign_validated': False}}
        if proof['confidence']['calibrated']: passed.append(proof['label'])
    registry['labels'] = [r for r in registry['labels'] if not r.get('evidence', {}).get('artifact', '').startswith('persona_label_evidence_v2_')] + list(rows.values())
    temporary = registry_path.with_suffix('.tmp'); temporary.write_text(json.dumps(registry, indent=1)); temporary.replace(registry_path)
    log('labels_published', evidence=len(rows), passed=sorted(passed))
    return sorted(passed)


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


def schedule_workers(state, log, units, status, labels, queued, hold):
    """Launch for work, size to the work, stop idle workers after IDLE_STOP_S; the hold file freezes all of it."""
    now = time.time()
    workers_up = sum(units.get(WORKER_UNIT.format(i)) == 'active' for i in range(MAX_WORKERS))
    occupied = sum(units.get(WORKER_UNIT.format(i), '') not in FREE_UNIT_STATES for i in range(MAX_WORKERS))
    need, why = worker_need(status, labels, queued)
    state.update(worker_need=need, worker_need_why=why)
    if hold or need is None:
        state['idle_since'] = None; return None
    if need > occupied:
        if now - state.get('last_relaunch', 0) > RELAUNCH_GAP_S:
            deadline, launched = relaunch_workers(need - occupied, units, log, why)
            if launched: state['last_relaunch'] = now
            state['idle_since'] = None; return deadline
    elif need == 0 and workers_up and not answering(status) and units.get('character-label-qualify') != 'active':
        state['idle_since'] = state.get('idle_since') or now
        if now - state['idle_since'] >= IDLE_STOP_S:
            stop_idle_workers(units, log, now - state['idle_since']); state['idle_since'] = None
        return None
    state['idle_since'] = None
    return None


def tick(state, log, until):
    """One supervision pass (the body of main's loop)."""
    units, status = remote_state(); record = json.loads(LAUNCH.read_text())
    workers_up = sum(units.get(WORKER_UNIT.format(i)) == 'active' for i in range(MAX_WORKERS))
    hold = hold_state(log, state)
    queued = (status or {}).get('counts', {}).get('queued', 0)
    free = int(ssh(f'df -B1 --output=avail {shlex.quote(REMOTE)} | tail -1').stdout.strip() or 0)
    state.setdefault('free_at_start', free); used = state['free_at_start'] - free
    state['disk_used_bytes'] = used
    # Owner order: the core-emotion batch finishes, then label qualification, then further identity batches.
    running = (status or {}).get('counts', {}).get('running', 0)
    labels = label_status()
    if status and not queued and not running and labels and not labels.get('complete'):
        if units.get('character-label-qualify') != 'active' and workers_up:
            start_labels(record['deadline'] - 120, log)
    finished = (labels or {}).get('concepts', {}).get('done', 0)
    if finished and finished != state.get('labels_finished_published'):
        state['labels_passed'] = publish_labels(log); state['labels_finished_published'] = finished
    # Keep the atlas colours current: rebuild once when labels finish, then nightly (03:00 local).
    builder = state.get('atlas_build_pid')
    if builder and Path(f'/proc/{builder}').exists(): pass
    elif (labels and labels.get('complete') and not state.get('atlas_built_after_labels')) or (
            time.localtime().tm_hour == 3 and time.time() - state.get('atlas_built_at', 0) > 20 * 3600):
        process = subprocess.Popen(['nice', '-n', '10', sys.executable, str(ROOT / 'persona/persona_neuron_atlas.py'), 'build', '--workers', '24'],
                                   cwd=ROOT, stdout=open(ROOT / 'results/neuron_atlas/build.log', 'w'), stderr=subprocess.STDOUT)
        state.update(atlas_build_pid=process.pid, atlas_built_at=time.time(),
                     atlas_built_after_labels=bool(labels and labels.get('complete')) or state.get('atlas_built_after_labels', False))
        log('atlas_rebuild_started', pid=process.pid)
    if used > DISK_BUDGET: log('disk_budget_reached', used=used)
    elif status and queued < LOW_WATER and (labels is None or labels.get('complete')):
        # Several picks per cycle: a menu item whose neurons were already swept adds almost nothing, and
        # one pick a minute would leave the GPUs idle for an hour of such items.
        for _ in range(10):
            # The remote campaign state is authoritative; do not mark skipped menus consumed.
            pause_ladders = status.get('threshold', {}).get('pause_ladder_menus', False)
            options = [o for o in MENU if o['id'] not in state['queued_menu']
                       and not (pause_ladders and o['id'].startswith(('lens:', 'word:')))]
            if not options or queued >= LOW_WATER: break
            chosen, why = planner_pick(options, status)
            try:
                result = post(EXPLORER + '/api/neuron-atlas/sweep', {'name': re.sub(r'[^A-Za-z0-9 _.-]', '-', chosen['id']), **chosen['query']})  # ids hold ':'
            except urllib.error.HTTPError as exc:  # e.g. a word no neuron label contains: skip it, never retry forever
                result = {'skipped': f'HTTP {exc.code}: ' + exc.read().decode(errors='replace')[:200]}
            queued += int(result.get('added') or 0)
            state['queued_menu'].append(chosen['id']); log('batch_queued', menu=chosen['id'], why=why, result=result)
    # Workers after the menu step: a batch queued this pass counts as work; options that add nothing do not.
    deadline = schedule_workers(state, log, units, status, labels, queued, hold)
    if deadline: record['deadline'] = deadline
    baseline_left = sum(v for k, v in ((status or {}).get('baseline40k') or {}).get('counts', {}).items() if k in ('queued', 'running'))
    if units.get('character-identity-sweep') != 'active' and workers_up and (queued or baseline_left or status is None) and not hold:
        runner_until = record['deadline'] - 120
        # the runner stops claiming DRAIN_MARGIN before `until` and joins in-flight batches, so `until` ends before RuntimeMaxSec
        start_runner(runner_until if runner_until > time.time() else time.time() + WINDOW_MIN * 60 - RUNNER_DRAIN_S, log)
    state.update(updated=time.time(), workers_up=workers_up, runner=units.get('character-identity-sweep'),
                 worker_deadline=record['deadline'], status=status, until=until, last_error=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--until', type=float, default=float('inf'), help='Epoch at which supervision ends (default: never)')
    args = parser.parse_args()
    state = json.loads(STATE.read_text()) if STATE.exists() else {'queued_menu': [], 'events': []}
    def log(kind, **fields):
        state['events'] = (state['events'] + [{'time': time.time(), 'event': kind, **fields}])[-40:]
    while time.time() < args.until:
        try:
            tick(state, log, args.until)
        except Exception as exc:
            state.update(updated=time.time(), last_error=f'{type(exc).__name__}: {str(exc)[:300]}')
        STATE.write_text(json.dumps(state, indent=1, default=str))
        time.sleep(60)


if __name__ == '__main__': main()
