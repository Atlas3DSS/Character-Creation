"""Bounded external guard for the known terminal-snapshot follow-on exit race.

Never recovers job states, calls inference, or retries judge requests. Restarts
only the existing CPU controller unit after independently checking frozen jobs.
"""
import argparse
from contextlib import closing
import fcntl
import hashlib
import json
import re
from pathlib import Path
import sqlite3
import subprocess
import time

UNIT='character-persona-preset-followon.service'
REASON='Manager exited before all jobs terminal; no uncertain replay'
DEADLINE=1793113200
TERMINAL={'complete','incomplete','failed'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)

def terminal_stage(path,expected):
    with closing(sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True,timeout=2)) as db:
        db.execute('BEGIN');rows=db.execute('SELECT id,state,body FROM jobs').fetchall()
        active=db.execute("SELECT count(*) FROM attempts WHERE state IN ('running','submitting')").fetchone()[0]
    expected={j['id']:j for j in expected}
    require(len(rows)==len(expected) and {r[0] for r in rows}==set(expected),'Frozen job coverage differs')
    require(not active,'Attempt still active')
    for identity,state,body in rows:
        require(json.loads(body)==expected[identity],'Frozen job body differs')
        require(state in TERMINAL,'Stage not fully terminal')
    return {'jobs':len(rows),'counts':{s:sum(r[1]==s for r in rows) for s in sorted(TERMINAL)}}

def stage_name(root):
    # A stopped status from the race omits stage. Actual durable phase artifacts
    # establish it; never infer confirmation merely from a selected candidate.
    root=Path(root)
    if (root/'results/persona_preset_followon_confirmation.sqlite').exists():return 'confirmation'
    return 'discovery'

def review_evidence(root,input_sha256):
    reviews=[]
    for path in sorted((Path(root)/'results').glob('persona_preset_followon_judge_*.json')):
        review=json.loads(path.read_text())
        require(review.get('input_sha256')==input_sha256,'Review belongs to another plan')
        require(review.get('state')=='validated' and review.get('protocol')=='answer_span_ids_v2','Ambiguous or failed review request; no restart')
        reviews.append({'sha256':sha(path),'state':review['state']})
    return reviews


def verify(root):
    import persona_preset_followon as follow
    base=follow.base;root=Path(root)
    status=json.loads((root/'results/persona_preset_followon_status.json').read_text())
    require(status.get('state')=='stopped' and status.get('reason')==REASON,'Not the authorized terminal-snapshot race')
    plan=json.loads((root/'results/persona_preset_followon_plan.json').read_text())
    require(plan['implementation_sha256']=={'followon':sha(follow.__file__),'shared':sha(base.__file__)},'Frozen implementation changed')
    require(all(sha(root/k)==v for k,v in plan['sources'].items()),'Frozen source changed')
    bank=base.attach_column(follow.VectorBank(root/'results',follow.ModelSpec.load(root/'data/persona_model.json')),root/'results/persona_preset_columns.json')
    base.validate_plan(plan,bank)
    stage=stage_name(root);selected=None
    if stage=='confirmation':
        selection=json.loads((root/'results/persona_preset_followon_selection.json').read_text())
        require(selection['input_sha256']==plan['input_sha256'],'Selection binding changed');selected=selection['selected']
        require(bool(selected),'Confirmation without selected candidates')
    expected=follow.jobs(plan,bank,stage,selected)
    evidence=terminal_stage(root/f'results/persona_preset_followon_{stage}.sqlite',expected)
    launch=json.loads((root/f'results/persona_preset_followon_{stage}_launch.json').read_text())
    require(launch['input_sha256']==plan['input_sha256'] and launch['source_sha256']==sha(follow.__file__) and launch['shared_source_sha256']==sha(base.__file__) and launch['manager_sha256']==sha(root/'persona/persona_manager.py'),'Stage launch source binding changed')
    reviews=review_evidence(root,plan['input_sha256'])
    return {'stage':stage,**evidence,'input_sha256':plan['input_sha256'],'plan_sha256':sha(root/'results/persona_preset_followon_plan.json'),
        'status_sha256':sha(root/'results/persona_preset_followon_status.json'),'implementation_sha256':plan['implementation_sha256'],'reviews':reviews}

def normalize_service(text):
    result=dict(line.split('=',1) for line in text.splitlines() if '=' in line)
    raw=result.get('ExecStart','')
    if raw:
        command=re.search(r'argv\[\]=(.*?) ;',raw)
        require(command is not None,'Cannot verify owned service command')
        result['ExecStart']=command.group(1)
    return result

def service():
    value=subprocess.run(['systemctl','--user','show',UNIT,'-p','ActiveState','-p','SubState','-p','MainPID','-p','InvocationID','-p','ExecStart','-p','LoadState'],capture_output=True,text=True,timeout=15)
    result=normalize_service(value.stdout)
    require(value.returncode==0 or result.get('LoadState')=='not-found','Cannot read controller unit')
    return result

def process_identity(pid):
    path=Path('/proc')/str(int(pid))
    stat=(path/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
    return {'pid':int(pid),'start_ticks':tail[19],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'argv':(path/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'),'cwd':str((path/'cwd').resolve())}

def process_gone(identity):
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==identity['boot_id'],'Host reboot requires explicit recovery')
    try:current=process_identity(identity['pid'])
    except FileNotFoundError:return True
    # A reused PID proves the original process ended, but creates ambiguous
    # ownership for unattended recovery. Fail closed rather than inspect/kill it.
    require(current['start_ticks']==identity['start_ticks'],'Prior PID reused; ownership ambiguous')
    return False

def owner(record,unit):
    latest=next((a for a in reversed(record['attempts']) if a.get('state')=='started'),record['attachment'])
    previous=latest['unit'];identity=latest['process']
    collected=unit.get('LoadState')=='not-found' or (not unit.get('InvocationID') and not unit.get('ExecStart') and unit.get('MainPID')=='0' and unit.get('ActiveState')=='inactive')
    if collected:
        require(unit.get('MainPID','0')=='0' and unit.get('ActiveState','inactive')=='inactive','Collected unit state ambiguous')
        require(process_gone(identity),'Collected unit still has owned process')
    else:
        require(unit.get('InvocationID')==previous['InvocationID'],'Unowned invocation replaced attached controller')
        require(unit.get('ExecStart')==record['attachment']['unit'].get('ExecStart'),'Owned service command changed')
        if unit.get('MainPID')=='0':require(process_gone(identity),'Unit stopped but owned process remains')
    return previous['InvocationID'],collected

def start_controller(record,collected,until):
    if collected:
        process=record['attachment']['process']
        command=['systemd-run','--user','--unit='+UNIT,'--property=WorkingDirectory='+process['cwd'],
            '--property=RuntimeMaxSec='+str(max(1,int(until-time.time()+65))),'--property=TimeoutStopSec=65',
            '--property=Nice=10','--setenv=PYTHONUNBUFFERED=1',*process['argv']]
    else:command=['systemctl','--user','start',UNIT]
    subprocess.run(command,check=True,timeout=30)


def eligible(status,unit,now,until):
    return now+1800<until and unit.get('ActiveState') in ('inactive','failed') and unit.get('MainPID')=='0' and status.get('state')=='stopped' and status.get('reason')==REASON

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');tmp.replace(path)

def run(root,until,expected_invocation):
    root=Path(root);require(time.time()<until<=DEADLINE,'Finite 08:00 deadline required')
    lock=(root/'results/persona_preset_followon_watchdog.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=root/'results/persona_preset_followon_watchdog.json'
    record=json.loads(path.read_text()) if path.exists() else {'schema':'character-creation.preset-followon-watchdog.v1','source_sha256':sha(__file__),'attempts':[]}
    require(record['source_sha256']==sha(__file__),'Watchdog source changed')
    attached=service();require(attached.get('InvocationID')==expected_invocation,'Attachment invocation changed')
    if 'attachment' not in record:
        record['attachment']={'unit':attached,'process':process_identity(attached['MainPID']),'time':time.time(),'prior_recovery_cause':REASON,'controller_status':json.loads((root/'results/persona_preset_followon_status.json').read_text())};write(path,record)
    while time.time()<until:
        try:
            unit=service();status=json.loads((root/'results/persona_preset_followon_status.json').read_text())
            prior_invocation,collected=owner(record,unit)
            record.update(state='watching',updated=time.time(),unit=unit,controller_state=status.get('state'),deadline=until)
            if unit.get('ActiveState') in ('inactive','failed') and status.get('state') in ('complete','deadline'):
                record['state']='finished';write(path,record);return
            if eligible(status,unit,time.time(),until):
                require(len(record['attempts'])<2,'Bounded restart budget exhausted')
                require(not any(a['prior_invocation']==prior_invocation for a in record['attempts']),'Invocation already attempted; no uncertain restart retry')
                evidence=verify(root)
                # Check both the unit and singleton controller lock before restart.
                with (root/'results/persona_preset_followon.lock').open('a') as controller_lock:
                    fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    require(service()==unit,'Service changed during verification')
                    attempt={'state':'intent','prior_invocation':prior_invocation,'started':time.time(),'evidence':evidence}
                    record['attempts'].append(attempt);write(path,record)
                # Existing runner also owns this singleton lock, preventing overlap
                # if an operator starts it between the check and systemd start.
                start_controller(record,collected,until)
                restarted=service();require(restarted.get('MainPID','0')!='0','Restart did not expose owned live PID; no retry')
                attempt.update(state='started',unit=restarted,process=process_identity(restarted['MainPID']),finished=time.time());write(path,record)
                print(json.dumps({'state':'restarted','stage':evidence['stage'],'jobs':evidence['jobs']}),flush=True)
        except Exception as exc:
            record.update(state='blocked',error=str(exc),updated=time.time());print(json.dumps({'state':'blocked','error':str(exc)}),flush=True)
        write(path,record);time.sleep(min(30,max(0,until-time.time())))
    record.update(state='deadline',updated=time.time());write(path,record)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--until',type=float,default=DEADLINE);p.add_argument('--expected-invocation',required=True);a=p.parse_args();run(a.root,a.until,a.expected_invocation)
