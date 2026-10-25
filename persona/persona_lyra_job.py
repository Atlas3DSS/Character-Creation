"""Fixed-model gift campaign: pinned corpus, six leased workers, bounded John choices."""
from __future__ import annotations
import fcntl,hashlib,json,math,shlex,subprocess,time
from pathlib import Path
from persona_lyra_worker import IDENTITY,MODEL_REPO,REVISION,digest

def prepare(job,config,event):
 from persona_job_fleet import control_context,run,source_path
 root,fleet,ssh,_,_=control_context(config)
 with (root/'data/persona_manager_corpus.jsonl').open('rb') as lock:
  fcntl.flock(lock,fcntl.LOCK_SH)
  raw=(root/'data/persona_generated_big_five.jsonl').read_bytes()
  publication=json.loads((root/'data/persona_generated_big_five_manifest.json').read_text())
 if publication['status']!='complete' or hashlib.sha256(raw).hexdigest()!=publication['output_sha256']:raise ValueError('Synthetic snapshot changed')
 report={'source_sha256':publication['output_sha256']};rows=[]
 for line in raw.splitlines():
  r=json.loads(line)
  if r.get('synthetic') is not True or r.get('corpus')!='big_five' or r.get('category')!='synthetic_scenario' or r.get('source',{}).get('kind')!='synthetic_model_response':raise ValueError('Gift input must be generated Big Five scenarios')
  rows.append({'id':r['id'],'prompt':r['prompt'],'domain':r['labels']['scenario_domain'],'facet_design':r['labels']['prompt_design_facet']})
 if len(rows)<1000:raise ValueError('Gift requires at least1000 accepted public discovery prompts')
 rows=sorted(rows,key=lambda r:hashlib.sha256((str(job['seed'])+r['id']).encode()).hexdigest())[:job['max_entries']]
 selected=[]
 for row in rows:
  family=hashlib.sha256((' '.join(row['domain'].lower().split())+'|'+row['facet_design']).encode()).hexdigest()
  bucket=int(family[:8],16)%100
  selected.append({'id':row['id'],'prompt':row['prompt'],'synthetic':True,'task_family':family,'split':'discovery' if bucket<70 else 'replication' if bucket<85 else 'heldout',
   'source':'Atlas3D/character-creation-big-five','source_sha256':report['source_sha256']})
 corpus=root/'results'/f"lyra_{job['id']}_corpus.jsonl"
 body=''.join(json.dumps(r,sort_keys=True)+'\n' for r in selected)
 if corpus.exists() and corpus.read_text()!=body:raise ValueError('Gift input already pinned to a different snapshot')
 corpus.write_text(body)
 checkpoint='/mnt/nvme/models/Qwen3.5-4B'
 code="from pathlib import Path;import hashlib,json;p=Path("+repr(checkpoint)+");print(json.dumps({x.name:hashlib.file_digest(x.open('rb'),'sha256').hexdigest() for x in sorted(p.iterdir()) if x.is_file() and x.suffix in ('.json','.safetensors','.jinja')}))"
 files=json.loads(run([fleet['python'],'-c',code],ssh=ssh,timeout=180))
 pinned=root/'data/persona_lyra_checkpoint.json'
 reference=json.loads(pinned.read_text())
 if reference['revision']!=REVISION or reference['files']!=files:raise ValueError('Pinned 4B checkpoint changed')
 worker=source_path(root,'persona_lyra_worker.py');remote_worker=fleet['remote_root']+'/persona_lyra_worker.py'
 if run(['sha256sum',remote_worker],ssh=ssh).split()[0]!=digest(worker):raise ValueError('Gift worker deployment differs')
 remote_corpus=fleet['remote_root']+'/results/'+corpus.name
 manifest={'schema':'character-creation.lyra-gift.v1','job_id':job['id'],'model_identity':IDENTITY,'repo':MODEL_REPO,'revision':REVISION,
  'checkpoint':checkpoint,'checkpoint_files':files,'worker_sha256':digest(worker),'corpus':remote_corpus,'corpus_sha256':digest(corpus),
  'records':len(rows),'max_prompt_tokens':512,'seed':job['seed'],'deadline':job['deadline'],
  'observation':'all MLP inputs to down_proj at final observed prompt token; prompts beyond512tokens explicitly marked truncated',
  'limits':'Float16 independent HF backend, not BF16 equivalence to Lyra; correlations not wiring; adjacent-layer ablation checks not an exhaustive connectome'}
 path=root/'results'/f"lyra_{job['id']}_manifest.json";path.write_text(json.dumps(manifest,indent=2))
 for local in (corpus,path):
  # Copy selected, authored experiment artifacts only; retain original basenames.
  subprocess.run(['scp','-o','ControlMaster=no','-o','ControlPath=none',str(local),fleet['ssh_host']+':'+fleet['remote_root']+'/results/'+local.name],check=True,capture_output=True)
 event('gift_input_pinned',{'manifest':str(path),'manifest_sha256':digest(path),'records':len(rows),'revision':REVISION})
 return manifest

def worker_command(fleet,config,job,shard,minutes,seconds):
 unit=f'character-persona-probe-{shard}.service'
 return ['systemd-run','--user','--unit='+unit,'--collect',f'--property=RuntimeMaxSec={int(seconds)}','--property=TimeoutStopSec=45',
  '--working-directory='+fleet['remote_root'],'--setenv=CUDA_VISIBLE_DEVICES='+fleet['gpus'][shard],
  '--setenv=OMP_NUM_THREADS='+str(fleet['compile_workers'][shard]),'--setenv=HF_HUB_OFFLINE=1','--setenv=WB_HOLD_WAIT='+str(int(seconds)),
  '--setenv=PERSONA_WORKER_UNIT='+unit,'--setenv=PERSONA_LEASE_WRAPPER='+fleet['lease_wrapper'],
  '--setenv=PERSONA_LEASE_WRAPPER_SHA256='+config['lease_wrapper_sha256'],
  '/usr/bin/python3',fleet['remote_root']+'/persona_lease.py','run','character-creation','lyra-gift-'+str(shard),'28',str(minutes),fleet['gpu_labels'][shard],
  '--','taskset','-c',fleet['cpu_sets'][shard],fleet['python'],fleet['remote_root']+'/persona_lyra_worker.py',
  '--manifest',fleet['remote_root']+f"/results/lyra_{job['id']}_manifest.json",'--shard',str(shard),'--port',str(8930+shard)]

def verified(job,config,owned):
 from persona_job_fleet import live_board
 from persona_planner import readiness,GPU,UNIT
 from persona_manager import check_admission
 from persona_fleet import fetch
 fleet=json.loads(Path(config['fleet_config']).read_text());board=live_board(fleet);endpoints=[]
 for i,gpu in enumerate(fleet['gpus']):
  unit=f'character-persona-probe-{i}.service';leases=[r for r in board['jobs'] if r.get('status')=='running' and r.get('assigned_uuid')==gpu and r.get('unit_invocation_id')==owned[unit] and r.get('project')=='character-creation']
  if len(leases)!=1:raise RuntimeError('Gift lease not ready')
  status=fetch(f'http://127.0.0.1:{8930+i}/gift/status',timeout=3)
  if status['job_id']!=job['id'] or status['model_identity']!=IDENTITY:raise ValueError('Gift worker identity mismatch')
  if status.get('qualification',{}).get('gpu_uuid') not in (None,gpu):raise ValueError('Gift physical GPU mismatch')
  if status['phase']=='failed':raise ValueError(status.get('error','Gift qualification failed'))
  if not status['ready']:raise RuntimeError('4B worker loading and checking neutral equivalence')
  endpoint={'url':f'http://127.0.0.1:{8930+i}','gpu_uuid':gpu,'lease_id':leases[0]['id'],'host':'v100','project':'character-creation','model_identity':IDENTITY,'label':f'Quincy {i+1}'}
  ok,reason=check_admission(board,endpoint)
  if not ok:raise RuntimeError(reason)
  endpoints.append(endpoint)
 planners=[r for r in board['jobs'] if r.get('status')=='running' and r.get('assigned_uuid')==GPU and r.get('unit_invocation_id')==owned[UNIT] and r.get('project')=='character-creation']
 if len(planners)!=1 or not readiness():raise RuntimeError('Gift planner not ready')
 return {'endpoints':endpoints,'planner':{'url':'http://127.0.0.1:18511','lease_id':planners[0]['id'],'gpu_uuid':GPU,'project':'character-creation','host':'wtatum84'},'root':config['root'],'fleet':fleet,'config':config}

def run(job,session,event,stop_event):
 from persona_fleet import fetch
 from persona_job_fleet import live_board
 from persona_manager import check_admission
 statuses=[]
 while not stop_event.is_set() and time.time()<job['deadline']:
  board=live_board(session['fleet'])
  for endpoint in session['endpoints']+[session['planner']]:
   ok,reason=check_admission(board,endpoint)
   if not ok:raise RuntimeError(reason)
  statuses=[fetch(e['url']+'/gift/status',timeout=5) for e in session['endpoints']]
  if any(s.get('job_id')!=job['id'] or s.get('model_identity')!=IDENTITY for s in statuses):raise ValueError('Gift ownership changed')
  count=sum(s['observations'] for s in statuses)
  event('progress',{'accepted_unique':count,'requested':job['max_entries'],'requests':sum(s['causal_checks'] for s in statuses),'errors':sum(s['phase']=='failed' for s in statuses)})
  event('gift_phases',{'workers':[{'shard':s['shard'],'phase':s['phase'],'causal_checks':s['causal_checks']} for s in statuses]})
  if any(s['phase']=='failed' for s in statuses):raise RuntimeError('Gift worker failed: '+str([s.get('error') for s in statuses if s['phase']=='failed']))
  if all(s['phase']=='complete' for s in statuses):break
  stop_event.wait(30)
 result={'model_identity':IDENTITY,'workers':statuses,'all_phases_complete':bool(statuses) and all(s['phase']=='complete' for s in statuses),
   'artifacts':session['fleet']['remote_root']+'/results/lyra_'+job['id']+'_*','scope':'observed coactivity and bounded adjacent-layer ablation checks; no exhaustive causal map claim'}
 path=Path(session['root'])/'results'/f"lyra_{job['id']}_summary.json";path.write_text(json.dumps(result,indent=2))
 event('gift_summary',result)
 if not result['all_phases_complete']:raise RuntimeError('Gift deadline reached with partial artifacts; no complete dataset claim')
 return result
