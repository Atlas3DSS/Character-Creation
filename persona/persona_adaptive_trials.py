"""Bounded discovery-only menu and five-card native dispatcher for John-selected IDs.

No inferred neuron labels, heldout adaptation, automatic promotion, or GPU leases.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from persona_perturbation_mapping import PriorityQueue,CORPUS_SHA,MANIFEST_SHA,DEADLINE,sha,require,TERMINAL
from persona_contrastive_backfill import choose,immutable
from persona_manager import Manager,Deferred,request_error_budget_exceeded
from persona_vectors import canonical_digest,VectorBank,ModelSpec
from persona_preset_overnight import attach_column,validate_plan
from persona_reasoning_sweep import atomic_json

EXCLUDED_GPU='GPU-6f2541c5-80ed-71d7-5892-8fe9af1398eb'
SEEDS=(80711,80729)
CAP=4096
OUTSTANDING=128
ARMS=('baseline','candidate','reversal','random')

class ProducerOnlyManager(Manager):
    def propose(self,finish):raise Deferred('Only the explicit adaptive controller may enqueue known menu IDs')

def bounded_candidate(bank,recipe,pushes,phase,change,evidence,identity_id,historical_identity_only=False):
    pushes=deepcopy(pushes);program,details=bank.program(pushes,phase)
    norm=details['aggregate_relative_norm'];require(math.isfinite(norm) and norm>0,'Invalid candidate norm')
    require(not historical_identity_only or change=='exact_distributed' and recipe['id']=='experimental-dry-wit-followon-distributed' and abs(norm-.6375)<1e-9,'Unauthorized historical dose exception')
    scale=1. if historical_identity_only else min(1.,.35/norm)
    if scale<1:
        for push in pushes:push['dose']*=scale*(1-1e-12)
        program,details=bank.program(pushes,phase)
    require(historical_identity_only or details['aggregate_relative_norm']<=.35+1e-10,'Candidate exceeds aggregate bound')
    value={'target':recipe['target'],'kind':change,'description':change+'; discovery hypothesis only','pushes':pushes,'phase':phase,
        'program_sha256':canonical_digest(program),'aggregate_relative_norm':details['aggregate_relative_norm'],'relative_norms':details['relative_norms'],
        'explicit_scale':scale,'historical_identity_only':historical_identity_only,'source_evidence':evidence,'source_recipe_id':recipe['id'],'identity_probe_id':identity_id,
        'model_identity':bank.model.identity,'promotion_eligible':False,'norm_semantics':details['norm_semantics']}
    value['id']=canonical_digest(value);return value

def build_menu(bank,plan,plan_sha,identity_id):
    candidates=[];seen=set()
    for recipe in sorted((r for r in plan['recipes'] if r['id'].endswith('-distributed')),key=lambda r:r['id']):
        evidence={'plan_sha256':plan_sha,'recipe_id':recipe['id'],'recipe_program_sha256':recipe['program_sha256'],'selection_scope':'source discovery recipes only; no confirmation results read'}
        p=recipe['pushes'];layers=sorted({v['layer'] for v in p});layer=layers[len(layers)//2]
        neurons=sorted(map(int,bank.layer(layer)['neuron_ids']));require(neurons,'No actual mapped neurons')
        neuron=neurons[int(canonical_digest(recipe['id'])[:8],16)%len(neurons)]
        variants=[('exact_distributed',p,recipe['phase'])]
        variants.extend(('dose_'+str(f),[{**v,'dose':v['dose']*f} for v in p],recipe['phase']) for f in (.65,.85,1.2))
        variants.extend(('phase_'+phase,p,phase) for phase in ('all','prefill') if phase!=recipe['phase'])
        if len(layers)>1:variants.extend(('drop_layer_'+str(l),[v for v in p if v['layer']!=l],recipe['phase']) for l in (layers[0],layers[-1]))
        variants.append(('mapped_neuron_add',p+[{'kind':'neuron','layer':layer,'neuron':neuron,'dose':.025}],recipe['phase']))
        variants.append(('mapped_neuron_probe',[{'kind':'neuron','layer':layer,'neuron':neuron,'dose':.05}],recipe['phase']))
        for change,pushes,phase in variants:
            ev={**evidence,'calibration_sha256':{str(l):sha(bank.root/f'layer_{l:02d}.npz') for l in sorted({x['layer'] for x in pushes})}}
            if change.startswith('mapped'):ev['coordinate_evidence']={'layer':layer,'neuron':neuron,'source':'actual calibrated neuron_ids; no semantic neuron-label claim'}
            candidate=bounded_candidate(bank,recipe,pushes,phase,change,ev,identity_id,historical_identity_only=change=='exact_distributed' and recipe['id']=='experimental-dry-wit-followon-distributed')
            if change=='exact_distributed':require(candidate['explicit_scale']==1 and candidate['program_sha256']==recipe['program_sha256'],'Exact source recipe exceeds limits or changed')
            if candidate['program_sha256'] in seen:continue
            seen.add(candidate['program_sha256']);candidates.append(candidate)
    require(sum(c['kind']=='exact_distributed' for c in candidates)==8,'Eight distinct exact source recipes required')
    return candidates

class TrialStore:
    def __init__(self,root,until=DEADLINE):
        self.root=Path(root);self.results=self.root/'results';self.until=float(until)
        require(time.time()<self.until<=DEADLINE,'Finite 08:00 deadline required')
        self.mutex=threading.RLock();self.pool=None;self.manager=None;self.monitor=None;self.lock=None;self.worker_errors=[]
        config=json.loads((self.results/'persona_preset_optimized_config.json').read_text());config['endpoints']=[e for e in config['endpoints'] if e['gpu_uuid']!=EXCLUDED_GPU]
        require(len(config['endpoints'])==5 and len({e['gpu_uuid'] for e in config['endpoints']})==5,'Exactly five non-excluded cards required')
        self.binding={k:config['endpoints'][0][k] for k in ('model_identity','runtime_identity','prepared_identity')}
        require(all(all(e[k]==v for k,v in self.binding.items()) for e in config['endpoints']),'Mixed runtime bindings')
        gate=json.loads((self.results/'persona_preset_runtime_gate.json').read_text())
        require(gate['state']=='qualified' and gate['config_sha256']==sha(self.results/'persona_preset_optimized_config.json'),'Qualified source config changed')
        self.dispatch_deadline=self.until-max(660,int(config.get('request_timeout_seconds',600))+60)
        require(time.time()<self.dispatch_deadline,'Dispatch deadline passed')
        config.update(max_total_jobs=CAP,per_gpu_concurrency=1,dispatch_deadline=self.dispatch_deadline,until=self.until)
        self.bank=attach_column(VectorBank(self.results,ModelSpec.load(self.root/'data/persona_model.json')),self.results/'persona_preset_columns.json')
        require(self.bank.model.identity==self.binding['model_identity'],'Model binding differs')
        paths=[self.results/'persona_preset_overnight_plan.json',self.results/'persona_preset_followon_plan.json'];plans=[json.loads(p.read_text()) for p in paths]
        for p in plans:validate_plan(p,self.bank)
        source=self.root/'data/persona_perturbation_prompts.jsonl';manifest=self.root/'data/persona_perturbation_prompts_manifest.json'
        require(sha(source)==CORPUS_SHA and sha(manifest)==MANIFEST_SHA,'Terra source differs')
        self.prompts=choose([json.loads(s) for s in source.read_text().splitlines()],plans,4096)
        ip=self.root/'data/identity_probe_questions.json';require(sha(ip)=='0a7c4b5f76cfcd3e54ef8ba021c6cea8d45de4a096013c54231998e5598d8e79','Pinned identity fixture differs');identity=json.loads(ip.read_text())
        self.identity=identity['baseline_probe']
        require(self.identity['prompt']=='who are you' and not self.identity.get('system') and self.identity['split']=='discovery' and self.identity['source']['kind']=='synthetic_original','Identity anchor changed')
        self.identity=deepcopy(self.identity);self.identity['max_tokens']=2048;self.identity['thinking']=False
        for row in self.prompts+[self.identity]:
            row['source_row_id']=row['id'];row['max_tokens']=2048;row['id']=canonical_digest({k:v for k,v in row.items() if k!='id'})
        self.identity_sha=sha(ip)
        candidates=build_menu(self.bank,plans[1],sha(paths[1]),self.identity['id'])
        menu={'schema':'character-creation.adaptive-menu.v1','binding':self.binding,'candidates':candidates,'identity_probe_id':self.identity['id'],
            'identity_source_sha256':self.identity_sha,'source_sha256':CORPUS_SHA,'source_manifest_sha256':MANIFEST_SHA,
            'excluded_plans':{p.name:sha(p) for p in paths},'implementation_sha256':sha(__file__),'seeds':list(SEEDS),'jobs_per_candidate':40,
            'maximum_jobs':CAP,'outstanding_cap':OUTSTANDING,'excluded_gpu':EXCLUDED_GPU,'deadline':self.until,'dispatch_deadline':self.dispatch_deadline}
        menu['menu_sha256']=canonical_digest(menu);self._menu=menu;self.by_id={c['id']:c for c in candidates};self.program_cache={}
        data=''.join(json.dumps(p,sort_keys=True,separators=(',',':'))+'\n' for p in [self.identity]+self.prompts);self.corpus_sha=hashlib.sha256(data.encode()).hexdigest()
        cp=self.results/'persona_adaptive_corpus.jsonl'
        if cp.exists():require(cp.read_text()==data,'Adaptive corpus changed')
        else:cp.write_text(data)
        config.update(corpus=str(cp),corpus_manifest=str(self.results/'persona_adaptive_corpus_manifest.json'))
        immutable(self.results/'persona_adaptive_menu.json',menu);immutable(self.results/'persona_adaptive_config.json',config)
        immutable(self.results/'persona_adaptive_corpus_manifest.json',{'status':'complete','output_sha256':self.corpus_sha,'rows':len(self.prompts)+1})
        self.config=config;self.queue=PriorityQueue(self.results/'persona_adaptive.sqlite',self.root);self.queue.bind(self.binding)
        with self.queue.connect() as db:
            db.executescript('CREATE TABLE IF NOT EXISTS adaptive_candidates(id TEXT PRIMARY KEY,mode TEXT,body TEXT);CREATE TABLE IF NOT EXISTS adaptive_sources(pair_id TEXT PRIMARY KEY,candidate_id TEXT);')
    def menu(self):return deepcopy(self._menu)
    def _program(self,pushes,phase):
        key=canonical_digest([pushes,phase])
        if key not in self.program_cache:self.program_cache[key]=self.bank.program(pushes,phase)
        return self.program_cache[key]
    def _jobs(self,candidate,cases):
        jobs=[];phase=candidate['phase'];base=candidate['pushes'];layer=base[0]['layer']
        for case in cases:
            for repeat,seed in enumerate(SEEDS):
                random=[{'kind':'random','layer':int(site.split('.')[1]),'site':'mlp_out' if site.endswith('.mlp_input') else site.split('.')[2],'dose':norm,'seed':seed} for site,norm in candidate['relative_norms'].items() if norm]
                for arm in ARMS:
                    pushes=[] if arm=='baseline' else random if arm=='random' else [{**p,'dose':-p['dose']} for p in base] if arm=='reversal' else deepcopy(base)
                    program,prov=self._program(pushes,phase)
                    job={'schema':'character-creation.manager-job.v1','proposal_id':candidate['id'],'model_identity':self.binding['model_identity'],
                        'experiment_index':0,'experiment':{'layer':layer,'label':candidate['id'],'measurement':'rollout'},'arm':arm,'pushes':pushes,'phase':phase,
                        'seed':seed,'repetition':repeat,'prompt':deepcopy(case),'pair_id':case['pair_id'],'source_group':{'corpus':case['corpus'],'pair_id':case['pair_id'],'split':'discovery'},
                        'corpus_sha256':self.corpus_sha,'calibration_sha256':sha(self.results/f'layer_{layer:02d}.npz'),'program_sha256':canonical_digest(program),
                        'measurement':'rollout','kind':'rollout','promotion_eligible':False,'candidate_id':candidate['id'],'target':candidate['target'],
                        'identity_probe':case['id']==self.identity['id'],'identity_probe_id':self.identity['id'],'menu_sha256':self._menu['menu_sha256'],
                        'random_control_semantics':'Same-layer/site decoded-output relative-norm match; neuron controls use post-projection random hooks, not identical hook location.'}
                    job['id']=canonical_digest(job);jobs.append(job)
        return jobs
    def enqueue(self,candidate_ids,rationale):return self._enqueue(candidate_ids,rationale,False)
    def enqueue_identity_backfill(self):return self._enqueue([c['id'] for c in self._menu['candidates'] if c['kind']=='exact_distributed'],'User-authorized fixed identity anchor for eight exact source recipes',True)
    def _enqueue(self,ids,rationale,identity_only):
        require(isinstance(ids,list) and 1<=len(ids)<=(8 if identity_only else 3) and all(type(i)is str for i in ids) and len(set(ids))==len(ids),'Candidate IDs must be a bounded unique list')
        require(isinstance(rationale,str) and 0<len(rationale.strip())<=4000,'Bounded rationale required')
        require(all(i in self.by_id for i in ids),'Unknown menu candidate')
        require(identity_only or all(not self.by_id[i].get('historical_identity_only') for i in ids),'Historical exact dose is identity-only; select a bounded adaptive variant')
        require(time.time()<self.dispatch_deadline,'No dispatch window')
        with self.mutex,self.queue.connect() as db:
            db.execute('BEGIN IMMEDIATE');states=dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'));total=sum(states.values());outstanding=sum(states.get(s,0) for s in ('queued','running','submitting'))
            used={r[0] for r in db.execute('SELECT pair_id FROM adaptive_sources')};jobs=[];new=[]
            for cid in ids:
                old=db.execute('SELECT mode FROM adaptive_candidates WHERE id=?',(cid,)).fetchone()
                if old and (old[0]=='full' or identity_only):continue
                cases=[] if old else [self.identity]
                selected=[]
                if not identity_only:
                    selected=sorted((p for p in self.prompts if p['pair_id'] not in used),key=lambda p:canonical_digest([cid,p['pair_id']]))[:4]
                    require(len(selected)==4,'Insufficient fresh discovery families');used.update(p['pair_id'] for p in selected);cases+=selected
                jobs+=self._jobs(self.by_id[cid],cases);new.append((cid,'identity_only' if identity_only else 'full',selected))
            require(total+len(jobs)<=CAP and outstanding+len(jobs)<=OUTSTANDING,'Total or outstanding job bound reached')
            now=time.time();proposal={'candidate_ids':ids,'rationale':rationale,'menu_sha256':self._menu['menu_sha256'],'identity_only':identity_only,'job_ids':[j['id'] for j in jobs]};pid=canonical_digest(proposal)
            if not jobs:return []
            db.execute('INSERT INTO proposals VALUES(?,?,?)',(pid,now,json.dumps(proposal)))
            for cid,mode,selected in new:
                db.execute('INSERT OR REPLACE INTO adaptive_candidates VALUES(?,?,?)',(cid,mode,json.dumps(self.by_id[cid])))
                db.executemany('INSERT INTO adaptive_sources VALUES(?,?)',[(p['pair_id'],cid) for p in selected])
            # Preserve identity-first order even though Queue orders same-time rows by ID.
            db.executemany("INSERT INTO jobs(id,created,state,body) VALUES(?,?,'queued',?)",[(j['id'],now+n*1e-5,json.dumps(j)) for n,j in enumerate(jobs)])
            db.execute('INSERT INTO audit(time,event,data) VALUES(?,?,?)',(now,'adaptive_enqueue',json.dumps(proposal)))
            return [j['id'] for j in jobs]
    def statuses(self):
        counts=self.queue.counts()
        with self.queue.connect() as db:candidates=[{'id':r[0],'mode':r[1]} for r in db.execute('SELECT id,mode FROM adaptive_candidates ORDER BY id')]
        return {'counts':counts,'total':sum(counts.values()),'outstanding':sum(counts.get(s,0) for s in ('queued','running','submitting')),
            'candidates':candidates,'maximum_jobs':CAP,'outstanding_cap':OUTSTANDING,'dispatch_deadline':self.dispatch_deadline,'deadline':self.until,
            'worker_errors':list(self.worker_errors),'workers_started':self.pool is not None}
    def _rows(self):
        with self.queue.connect() as db:
            db.execute('BEGIN');saved=db.execute('SELECT id,state,body FROM jobs ORDER BY created,id').fetchall();attempts=db.execute('SELECT job_id,number,result FROM attempts ORDER BY number').fetchall()
        latest={i:r for i,n,r in attempts};rows=[]
        for identity,state,body in saved:
            job=json.loads(body);require(canonical_digest({k:v for k,v in job.items() if k!='id'})==identity,'Adaptive job altered')
            require(job['candidate_id'] in self.by_id and job['menu_sha256']==self._menu['menu_sha256'] and job['prompt']['split']=='discovery','Adaptive source differs')
            result=json.loads(latest[identity]) if latest.get(identity) else None;p=job['prompt']
            row={'job_id':identity,'candidate_id':job['candidate_id'],'target':job['target'],'source_recipe_id':self.by_id[job['candidate_id']]['source_recipe_id'],'pair_id':job['pair_id'],'seed':job['seed'],'arm':job['arm'],'state':state,
                'prompt':p['prompt'],'fact_contract':p.get('fact_contract',{}),'source_id':p['source_row_id'],'source_sha256':self.identity_sha if job['identity_probe'] else CORPUS_SHA,
                'prompt_sha256':hashlib.sha256(p['prompt'].encode()).hexdigest(),'split':'discovery','identity_probe':job['identity_probe'],
                'program_sha256':job['program_sha256'],'phase':job['phase'],'pushes':job['pushes'],'answer':None,'complete':False,'capture_validated':False}
            if result and not result.get('error') and (result.get('complete') is True or result.get('capture_validated') is True):
                r=result['response'];request=result['request'];rh=canonical_digest(request)
                require(result['binding']==self.binding and all(r[k]==v for k,v in self.binding.items()),'Adaptive response binding differs')
                require(request['messages']==[{'role':'user','content':p['prompt']}] and request['seed']==job['seed'] and request['max_tokens']==2048 and not request.get('kv_control'),'Adaptive request differs')
                require(canonical_digest(request.get('steering',{'phase':job['phase'],'sites':[]}))==job['program_sha256'],'Adaptive program differs')
                capture={}
                for key in ('activation_artifact','provenance_artifact'):
                    a=r[key];require(a['request_sha256']==rh and all(a[k]==v for k,v in self.binding.items()),'Adaptive artifact binding differs');capture[key]={k:a[k] for k in ('sha256','request_sha256','model_identity','runtime_identity','prepared_identity')}
                answer=r['choices'][0]['message']['content'];row.update(answer=answer,answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),complete=result['complete'],capture_validated=True,
                    request_sha256=rh,response_sha256=canonical_digest(r),capture=capture,usage=r['usage'],finish_reason=r['choices'][0]['finish_reason'])
            rows.append(row)
        return rows
    def completed_groups(self):
        grouped=defaultdict(list)
        for row in self._rows():grouped[(row['candidate_id'],row['pair_id'],row['seed'])].append(row)
        result=[]
        for (cid,pair,seed),rows in grouped.items():
            if len(rows)!=4 or {r['arm'] for r in rows}!=set(ARMS) or any(r['state'] not in TERMINAL for r in rows):continue
            r=rows[0];item={k:r[k] for k in ('prompt','fact_contract','source_id','source_sha256','prompt_sha256','split','identity_probe')}
            item.update(candidate_id=cid,pair_id=pair,seed=seed,arms={r['arm']:r for r in rows},behavior_axes=['agreeableness','assertiveness','sarcasm','hostility','empathy','imagination','anxiety','directness'],
                scope='Discovery comparison only; automatic judgments cannot establish personhood or true identity.')
            item['id']=canonical_digest([self._menu['menu_sha256'],cid,pair,seed]);result.append(item)
        return sorted(result,key=lambda r:(not r['identity_probe'],r['id']))
    def public_snapshot(self):
        status=self.statuses();rows=[r for r in self._rows() if r['state'] in TERMINAL]
        return {'schema':'character-creation.adaptive-public.v1','menu_sha256':self._menu['menu_sha256'],'binding':self.binding,'menu':self.menu(),**status,
            'rows':rows,'terminal_jobs':len(rows),'final':time.time()>=self.dispatch_deadline or sum(status['counts'].values())>=CAP and not status['outstanding'],
            'scope':'Original discovery scenes and fixed identity anchor; exact calibrated experimental programs. No automatic promotion, causal neuron-label claim, or general persona validation. Two processes may each queue one B4 request per card; foreground yields before submission, without hard preemption.'}
    def start_workers(self):
        require(self.pool is None and time.time()<self.dispatch_deadline,'Workers already started or window closed')
        self.lock=self.queue.path.with_suffix('.sqlite.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.manager=ProducerOnlyManager(self.config,self.queue);self.queue.recover_native_batch()
        self.pool=ThreadPoolExecutor(max_workers=5,thread_name_prefix='adaptive-native')
        self.futures=[self.pool.submit(self.manager.worker,e,self.dispatch_deadline) for e in self.manager.endpoints]
        self.queue.audit('adaptive_workers_start',{'menu_sha256':self._menu['menu_sha256'],'endpoints':[e['url'] for e in self.manager.endpoints],'deadline':self.dispatch_deadline})
        def monitor():
            while not self.manager.stop.wait(2):
                if time.time()>=self.dispatch_deadline:self.manager.stop.set();break
                if request_error_budget_exceeded(self.queue.counts(),self.config,native_batch=True):
                    self.worker_errors.append('request_error_budget');self.manager.stop.set();break
                ended=[f for f in self.futures if f.done()]
                if ended:
                    self.worker_errors.extend(repr(f.exception()) for f in ended);self.manager.stop.set();break
        self.monitor=threading.Thread(target=monitor,name='adaptive-monitor',daemon=True);self.monitor.start();return self.monitor
    def stop_workers(self):
        if self.manager:self.manager.stop.set()
        if self.pool:self.pool.shutdown(wait=True)
        if self.monitor:self.monitor.join(timeout=5)
        if self.lock:self.lock.close();self.lock=None
        if self.manager:self.queue.audit('adaptive_workers_stop',{'counts':self.queue.counts(),'worker_errors':self.worker_errors})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--until',type=float,default=DEADLINE);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    require(a.prepare_only,'Use the separate John controller for explicit live scheduling')
    store=TrialStore(a.root,a.until);print(json.dumps({'menu_sha256':store.menu()['menu_sha256'],'candidates':len(store.menu()['candidates']),'status':store.statuses()}))
