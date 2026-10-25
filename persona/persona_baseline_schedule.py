"""Fixed baseline session with optional idle-time backfill and cooperative yield."""
import argparse
from datetime import datetime, timedelta
import json
import os
import tempfile
from pathlib import Path
import shlex
import signal
import sqlite3
import subprocess
import threading
import time
import uuid
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from persona_job_fleet import JOHN, verified_session

ROOT = Path(__file__).resolve().parent.parent
ZONE = ZoneInfo('America/Los_Angeles')
UNIT = 'character-baseline40k-capture.service'
STOP = threading.Event()

def api(path, body=None):
    request = Request('http://127.0.0.1:8792'+path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type':'application/json'})
    with urlopen(request, timeout=30) as response: return json.load(response)

def command(argv, timeout=90):
    return subprocess.check_output(argv, text=True, timeout=timeout)

def board(fleet):
    return json.loads(command([fleet['workboard'], 'status', '--json']))

def priority_waiting(jobs):
    """Owner preference at the client; never changes board priority or other jobs."""
    waiting = [j for j in jobs if j.get('status') in ('queued', 'pending', 'waiting')]
    for job in waiting:
        if job.get('lane') == 'interactive' or 'atlas' in str(job.get('project', '')).lower():
            return 'interactive'
    for job in waiting:
        project = str(job.get('project', '')).lower()
        label = str(job.get('label', '')).lower()
        if (project in ('dsv41', 'chemistry', 'chem', 'orca')
                or 'chem' in project or label.startswith(('chem-', 'orca-'))):
            return 'chemistry'
    return None


def interactive_waiting(fleet):
    return priority_waiting(board(fleet)['jobs'])


def required_fleet_idle(snapshot, fleet):
    """Conservative preflight only; workboard still owns atomic admission."""
    required = set(fleet['gpus']) | {JOHN}
    if len(set(fleet['gpus'])) != 6 or len(required) != 7:
        raise ValueError('Baseline requires six distinct probe GPUs plus John')
    cards = snapshot.get('fleet', {}).get('cards', [])
    for gpu in required:
        matches = [c for c in cards if c.get('uuid') == gpu]
        if len(matches) != 1 or not matches[0].get('fresh'):
            return False
        card = matches[0]
        if any(card.get(field) != [] for field in
               ('active_job_ids', 'starting_job_ids', 'exclusive_job_ids')):
            return False
    # Cover owner reservations even if the card's summary is incomplete.
    for job in snapshot.get('jobs', []):
        if (job.get('assigned_uuid') in required
                and job.get('status') in ('leased', 'starting', 'running', 'stale', 'preempting')):
            return False
    return True


def session_end(now, opportunistic=False):
    """Dedicated window, or bounded daytime backfill until the next window."""
    now = now.astimezone(ZONE)
    if now.hour >= 20 or now.hour < 8:
        end = now.replace(hour=8, minute=0, second=0, microsecond=0)
        return end + timedelta(days=1) if now.hour >= 20 else end
    if opportunistic:
        return min(now + timedelta(hours=2), now.replace(hour=20, minute=0, second=0, microsecond=0))
    return None

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--opportunistic', action='store_true', help='Also backfill idle time outside 20:00–08:00; yield to chemistry and Atlas')
    args=parser.parse_args()
    config=json.loads((ROOT/'persona_jobs.local.json').read_text())
    fleet=json.loads(Path(config['fleet_config']).read_text())
    ssh=['ssh','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none',fleet['ssh_host']]
    remote=Path(fleet['remote_root'])
    def remote_run(argv, timeout=90): return command([*ssh,shlex.join(map(str,argv))],timeout)
    def copy(source,destination):
        command(['scp','-q','-o','ControlMaster=no','-o','ControlPath=none',str(source),
                 fleet['ssh_host']+':'+str(destination)],300)
    if args.check:
        assert remote_run(['hostname']).strip()==fleet['hostname']
        assert (ROOT/'dataset/baseline40k.jsonl').is_file()
        print('Fixed source, host and schedule configuration present; no GPU launch.')
        return
    now=datetime.now(ZONE)
    end=session_end(now,args.opportunistic)
    if end is None:
        print('Outside baseline window (20:00–08:00 America/Los_Angeles).',flush=True);return
    # At most one owner service instance; systemd provides the process ownership.
    job_id=None; capture_invocation=None; capture_token=None
    state_path=ROOT/'results/baseline40k-schedule.json'
    def save(**values):
        payload=json.dumps({'updated':time.time(),'job_id':job_id,
            'capture_invocation':capture_invocation,'capture_token':capture_token,
            'opportunistic':args.opportunistic,**values},indent=2)+'\n'
        with tempfile.NamedTemporaryFile(mode='w',dir=state_path.parent,delete=False) as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno());temporary=stream.name
        os.replace(temporary,state_path)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:STOP.set())
    try:
        if state_path.is_file():
            previous_owner=json.loads(state_path.read_text())
            prior_id=previous_owner.get('job_id')
            job_id=prior_id;capture_token=previous_owner.get('capture_token');capture_invocation=previous_owner.get('capture_invocation')
            token=previous_owner.get('capture_token')
            if token:
                raw=remote_run(['systemctl','--user','show',UNIT,'-p','Environment','--value'])
                env=dict(v.split('=',1) for v in shlex.split(raw) if '=' in v)
                if env.get('BASELINE_CAPTURE_TOKEN')==token and env.get('BASELINE_SESSION_JOB')==prior_id:
                    remote_run(['systemctl','--user','stop',UNIT],700)
            if prior_id:
                prior=api('/api/job?id='+prior_id)
                if prior['state'] in ('queued','running','stopping'):
                    api('/api/job/cancel',{'id':prior_id})
                    # Cancel only the recorded session; never adopt another owner's work.
                    for _ in range(45):
                        if api('/api/job?id='+prior_id)['state'] not in ('queued','running','stopping'):break
                        STOP.wait(2)
                    else:raise RuntimeError('Prior owned session cleanup still pending')
            if previous_owner.get('state') in ('failed','capture_failed','needs_review'):
                print('Prior failure requires review; no automatic new GPU session.',flush=True);return
        job_id=None;capture_token=None;capture_invocation=None
        manifest=remote_run(['/usr/bin/python3','-c',
            'import pathlib; p=pathlib.Path('+repr(str(remote/'results/baseline40k-index.manifest.json'))+'); print(p.read_text() if p.exists() else "{}")'])
        previous=json.loads(manifest)
        if previous.get('complete'):
            save(state='complete');return
        if previous.get('needs_review') or previous.get('errors') or (previous.get('unattempted')==0 and previous.get('states',{}).get('incomplete',0)):
            save(state='needs_review',reason='Retained failed/incomplete captures need explicit resolution.');return
        snapshot=board(fleet)
        if priority_waiting(snapshot['jobs']):
            save(state='waiting_for_priority_work');return
        if not required_fleet_idle(snapshot,fleet):
            save(state='waiting_for_idle_fleet');return
        if STOP.is_set():return
        minutes=int((end-datetime.now(ZONE)).total_seconds()//60)
        if minutes<20:return
        job=api('/api/jobs',{'kind':'studio_session','name':'Lyra gift: 40k baseline answers + activations',
            'start_at':datetime.now(ZONE).isoformat(),'timezone':str(ZONE),'duration_minutes':minutes})
        job_id=job['id'];save(state='warming')
        while not STOP.is_set() and datetime.now(ZONE)<end:
            job=api('/api/job?id='+job_id)
            if job['state'] in ('failed','partial','complete','cancelled'):raise RuntimeError('Session ended before capture: '+str(job.get('error')))
            priority=interactive_waiting(fleet)
            if priority:save(state='yielded_to_'+priority);return
            if job['counts']['accepted_unique']>=1:break
            STOP.wait(20)
        if STOP.is_set() or datetime.now(ZONE)>=end:return
        with sqlite3.connect(ROOT/'results/persona_jobs.sqlite') as db:
            receipts=[json.loads(row[0]) for row in db.execute(
                "SELECT data FROM events WHERE job_id=? AND event='service_started'",(job_id,))]
        owned={r['unit']:r['invocation_id'] for r in receipts}
        session=verified_session(job,config,owned=owned)
        capture_config={'model':str(remote/'data/persona_model.json'),
            'workboard':'/home/orwel/.local/bin/gpu-workboard','endpoints':session['endpoints']}
        local_config=ROOT/'results/baseline40k-config.json'
        local_config.write_text(json.dumps(capture_config,indent=2)+'\n')
        for src,dst in [(ROOT/'persona/persona_baseline_capture.py',remote/'persona_baseline_capture.py'),
                        (ROOT/'dataset/baseline40k.jsonl',remote/'data/baseline40k.jsonl'),
                        (ROOT/'dataset/baseline40k.manifest.json',remote/'data/baseline40k.manifest.json'),
                        (local_config,remote/'results/baseline40k-config.json')]:copy(src,dst)
        # Linux/Atlas get scheduling preference; existing guards retain RAM headroom.
        for unit in owned:
            argv=['systemctl','--user','set-property','--runtime',unit,'CPUWeight=10','IOWeight=10']
            (remote_run if unit.startswith('character-persona-probe-') else command)(argv)
        seconds=int((end-datetime.now(ZONE)).total_seconds())
        capture_token=uuid.uuid4().hex
        save(state='capture_launch_intent',until=end.isoformat())
        remote_run(['systemd-run','--user','--unit='+UNIT,
            '--setenv=BASELINE_CAPTURE_TOKEN='+capture_token,'--setenv=BASELINE_SESSION_JOB='+job_id,'--property=RuntimeMaxSec='+str(seconds),
            '--property=TimeoutStopSec=660','--property=CPUWeight=10','--property=IOWeight=10',
            '--working-directory='+str(remote),fleet['python'],'-u',str(remote/'persona_baseline_capture.py'),'run',
            '--config',str(remote/'results/baseline40k-config.json'),'--corpus',str(remote/'data/baseline40k.jsonl'),
            '--ledger',str(remote/'results/baseline40k.sqlite'),'--index',str(remote/'results/baseline40k-index.jsonl'),
            '--artifact-dir',str(remote/'results'),'--until',end.isoformat(),'--max-tokens','2048','--timeout','600',
            *(['--opportunistic'] if args.opportunistic else [])])
        capture_invocation=remote_run(['systemctl','--user','show',UNIT,'-p','InvocationID','--value']).strip()
        save(state='capturing',until=end.isoformat())
        while not STOP.wait(20) and datetime.now(ZONE)<end:
            priority=interactive_waiting(fleet)
            if priority:save(state='yielded_to_'+priority);return
            active=remote_run(['systemctl','--user','show',UNIT,'-p','ActiveState','--value']).strip()
            if active not in ('active','activating','deactivating'):
                code=remote_run(['systemctl','--user','show',UNIT,'-p','ExecMainStatus','--value']).strip()
                save(state='capture_finished' if code=='0' else 'capture_failed',exit_code=code)
                return
    except Exception as error:
        save(state='failed',error=repr(error));raise
    finally:
        try:
            # Atlas/interactive releases endpoints immediately. Chemistry drains the
            # current capture first; interrupted/ambiguous outputs remain review-only.
            if job_id and state_path.is_file() and json.loads(state_path.read_text()).get('state')=='yielded_to_interactive':
                api('/api/job/cancel',{'id':job_id})
            if capture_token:
                raw=remote_run(['systemctl','--user','show',UNIT,'-p','Environment','--value'])
                environment=dict(v.split('=',1) for v in shlex.split(raw) if '=' in v)
                actual=remote_run(['systemctl','--user','show',UNIT,'-p','InvocationID','--value']).strip()
                if (environment.get('BASELINE_CAPTURE_TOKEN')==capture_token
                        and environment.get('BASELINE_SESSION_JOB')==job_id
                        and (capture_invocation is None or actual==capture_invocation)):
                    remote_run(['systemctl','--user','stop','--no-block',UNIT])
                    drain_deadline=time.monotonic()+700
                    while True:
                        actual=remote_run(['systemctl','--user','show',UNIT,'-p','InvocationID','--value']).strip()
                        active=remote_run(['systemctl','--user','show',UNIT,'-p','ActiveState','--value']).strip()
                        if active not in ('active','activating','deactivating'):
                            break
                        if capture_invocation is not None and actual!=capture_invocation:
                            raise RuntimeError('Capture ownership changed during drain')
                        # Atlas can arrive while a chemistry yield is draining.
                        if job_id and interactive_waiting(fleet)=='interactive':
                            api('/api/job/cancel',{'id':job_id})
                        if time.monotonic()>=drain_deadline:
                            raise TimeoutError('Owned capture did not finish its bounded drain')
                        time.sleep(2)
        finally:
            if job_id:api('/api/job/cancel',{'id':job_id})

if __name__=='__main__':main()
