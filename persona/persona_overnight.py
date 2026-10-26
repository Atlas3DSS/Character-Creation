"""Thin UI reservation and pause/drain bridge for the remote overnight mapper."""
from pathlib import Path, PurePosixPath
import json
import re
import shlex
import subprocess
import threading
import time

REMOTE = '/mnt/storage16tb/Character Creation/activation-search'

def remote_config(config=None):
 """Validate shared Explorer/handoff transport settings; no network operation."""
 if config is not None and not isinstance(config,dict):raise ValueError('Expected an Explorer configuration object')
 value=(config or {}).get('overnight_remote',{})
 if not isinstance(value,dict) or set(value)-{'ssh_host','hostname','remote_root'}:raise ValueError('Invalid overnight_remote fields')
 result={'ssh_host':'v100','hostname':'v100','remote_root':REMOTE,**value}
 if not isinstance(result['ssh_host'],str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@:\[\]-]*',result['ssh_host']):raise ValueError('Invalid overnight SSH host')
 if not isinstance(result['hostname'],str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',result['hostname']):raise ValueError('Invalid expected remote hostname')
 root=result['remote_root']
 if not isinstance(root,str) or not root.startswith('/') or any(ord(c)<32 for c in root) or '..' in PurePosixPath(root).parts:raise ValueError('Remote root must be an absolute POSIX path without traversal')
 result['remote_root']=str(PurePosixPath(root))
 return result

class OvernightMap:
 def __init__(self,app):
  self.app=app;self.path=app.root/'results/persona_overnight_session.json';self.lock=threading.RLock();self.cached=None;self.updated=0
 def remote(self,action='status'):
  settings=remote_config(getattr(self.app,'config',{}))
  script="""import json,socket,subprocess
from pathlib import Path, PurePosixPath
if socket.gethostname()!=%r:raise RuntimeError('Remote hostname differs from configuration')
p=Path(%r)
a=%r
if a=='pause':(p/'overnight.pause').touch()
if a=='resume':(p/'overnight.pause').unlink(missing_ok=True)
s=p/('overnight_feature_map.json' if a=='features' else 'overnight_status.json')
v=json.loads(s.read_text()) if s.exists() else {'state':'not_started','active':[]}
if a=='features':
 print(json.dumps(v));raise SystemExit(0)
q=p/'persona_label_qualification_status.json'
if q.exists():v['label_qualification']=json.loads(q.read_text())
v['pause_requested_at']=(p/'overnight.pause').stat().st_mtime if (p/'overnight.pause').exists() else None
print(json.dumps(v))
"""%(settings['hostname'],settings['remote_root'],action)
  result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=3',settings['ssh_host'],shlex.join(['python3','-c',script])],capture_output=True,text=True,timeout=12)
  if result.returncode:raise ValueError('Overnight status bridge unavailable: '+result.stderr[-300:])
  return json.loads(result.stdout)
 def status(self,force=False):
  with self.lock:
   session=json.loads(self.path.read_text()) if self.path.exists() else {'active':False}
   if not session.get('active'):return {'active':False,'interactive_allowed':True}
   if not force and self.cached is not None and getattr(self,'cached_session',None)==session and time.monotonic()-self.updated<5:return self.cached
   result={**session,'interactive_allowed':False}
   try:
    remote=self.remote();result['remote']=remote
    pause=remote.get('pause_requested_at')
    drained=bool(pause and remote.get('paused') is True and not remote.get('active') and remote.get('updated',0)>=pause)
    finished=remote.get('state') in ('complete','partial','deadline','failed') and not remote.get('active')
    result['interactive_allowed']=drained or finished
    result['state']='paused' if drained else remote.get('state','unknown')
    if session.get('auxiliary')=='label_qualification':
     qualification=remote.get('label_qualification',{})
     auxiliary_done=qualification.get('state') in ('complete','incomplete','failed') and not qualification.get('active')
     result['operation']='label_qualification';result['auxiliary_active']=not auxiliary_done
     if not auxiliary_done:result['interactive_allowed']=False
    if finished and not result.get('auxiliary_active'):
     result['active']=False;self.path.write_text(json.dumps({**session,'active':False,'finished_state':remote.get('state')}))
   except (OSError,ValueError,subprocess.TimeoutExpired) as error:result['error']=str(error)
   self.cached=result;self.cached_session=session;self.updated=time.monotonic();return result
 def features(self):
  # Remote artifacts are immutable evidence; a short cache keeps browsing cheap.
  with self.lock:
   if getattr(self,'feature_cache',None) and time.monotonic()-self.feature_updated<60:return self.feature_cache
   value=self.remote('features');self.feature_cache=value;self.feature_updated=time.monotonic();return value
 def require_interactive(self):
  if not self.status().get('interactive_allowed'):raise ValueError('The six Quincies are mapping overnight. Pause and drain the map using the overnight control before interactive inference.')
 def action(self,value):
  if not isinstance(value,dict) or set(value)!={'action'} or value['action'] not in ('pause','resume'):raise ValueError('Choose pause or resume')
  with self.lock:
   if not self.path.exists() or not json.loads(self.path.read_text()).get('active'):raise ValueError('No active overnight mapping session')
   if json.loads(self.path.read_text()).get('auxiliary')=='label_qualification' and self.status(force=True).get('auxiliary_active'):raise ValueError('Label qualification is using the paused interval. Wait for its captures to drain before changing the map state.')
   if value['action']=='resume':
    with self.app.connect() as db:
     db.execute('BEGIN IMMEDIATE')
     for table in ('cells','activation_searches','comparisons','token_traces','sculptor_campaigns','coherence_runs'):
      if db.execute("SELECT 1 FROM sqlite_master WHERE name=?",(table,)).fetchone() and db.execute("SELECT 1 FROM "+table+" WHERE state IN ('queued','running','stopping') LIMIT 1").fetchone():raise ValueError('Wait for current interactive work to finish before resuming the map')
     self.cached={**self.status(),'interactive_allowed':False};self.updated=time.monotonic();self.remote('resume')
   else:self.remote('pause')
   return self.status(force=True)
