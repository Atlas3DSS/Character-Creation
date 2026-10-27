"""Finite two-file adaptive publisher; no inference, queue mutation or worker control."""
import argparse
from collections import Counter
from contextlib import closing
import fcntl
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from persona_publication_recovery import Recovery,BoundedHub,digest,encoded,DEADLINE
from persona_contrastive_publication import approved_prompts,CORPUS_SHA,MANIFEST_SHA,TERMINAL
from persona_hub_research import reject_private_keys

REPORT='persona_adaptive_public.json'
IDENTITY='data/identity_probe_questions.json'
IDENTITY_SHA='0a7c4b5f76cfcd3e54ef8ba021c6cea8d45de4a096013c54231998e5598d8e79'
HEX=re.compile('[a-f0-9]{64}')
AXES={'agreeableness','assertiveness','sarcasm','hostility','empathy','imagination','anxiety','directness'}
FIELDS={'schema','binding','menu','menu_sha256','menu_projection_sha256','counts','total_jobs','terminal_jobs','final','state','rows','proposals','reviews','source_hashes','promotion_eligible','scope','revision_sha256'}
ROW_FIELDS={'job_id','candidate_id','target','source_recipe_id','pair_id','seed','arm','state','prompt','fact_contract','source_id','source_sha256','prompt_sha256','split','identity_probe','program_sha256','phase','pushes','answer','complete','capture_validated','answer_sha256','request_sha256','response_sha256','capture','usage','finish_reason'}

def require(ok,message):
    if not ok:raise ValueError(message)
def canonical(v):return digest(encoded(v).encode())

def private_transport(value):
    if isinstance(value,dict):
        for item in value.values():private_transport(item)
    elif isinstance(value,list):
        for item in value:private_transport(item)
    elif isinstance(value,str):
        require(not re.search(r'(?:gpu-\d{8}T|GPU-[a-f0-9-]{20,}|https?://(?:localhost|127\.|10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.|v100(?:[:/]|$)|wtatum84(?:[:/]|$)))',value),'Private transport identifier')

def validate(raw,allowed,menu,jobs,source_hashes):
    require(len(raw)<=256*1024**2,'Report size bound exceeded');v=json.loads(raw);reject_private_keys(v);private_transport(v)
    require(set(v)==FIELDS and v['schema']=='character-creation.adaptive-public.v1','Adaptive report fields/schema differ')
    require(v['revision_sha256']==canonical({k:x for k,x in v.items() if k!='revision_sha256'}),'Revision hash differs')
    projection={k:x for k,x in menu.items() if k!='excluded_gpu'}
    require(v['menu']==projection and v['menu_sha256']==menu['menu_sha256'] and v['menu_projection_sha256']==canonical(projection),'Frozen menu projection differs')
    require(v['binding']==menu['binding'] and v['source_hashes']==source_hashes,'Model/runtime/source binding differs')
    require(v['promotion_eligible'] is False,'Unsupported promotion claim')
    candidates={c['id']:c for c in menu['candidates']};require(1<=len(candidates)<=66,'Candidate menu bound differs')
    require(all(c['id']==canonical({k:x for k,x in c.items() if k!='id'}) and c['promotion_eligible'] is False for c in candidates.values()),'Candidate content hash/promotion differs')
    counts=v['counts'];require(isinstance(counts,dict) and all(k in TERMINAL|{'queued','running','submitting'} and type(n)is int and n>=0 for k,n in counts.items()),'Invalid counts')
    require(type(v['total_jobs'])is int and 0<=v['total_jobs']<=4096 and sum(counts.values())==v['total_jobs'],'Job total differs')
    rows=v['rows'];require(isinstance(rows,list) and len(rows)==v['terminal_jobs']==sum(counts.get(k,0) for k in TERMINAL),'Terminal coverage differs')
    require(type(v['final'])is bool and v['state'] in ('running','deadline'),'Invalid final status')
    if v['final']:require(v['state']=='deadline','Final report state differs')
    observed=Counter(r.get('state') for r in rows);require(all(observed[k]==counts.get(k,0) for k in TERMINAL) and set(observed)<=TERMINAL,'Terminal states differ')
    by_id={}
    for row in rows:
        require(isinstance(row,dict) and set(row)<=ROW_FIELDS,'Unapproved row field')
        jid=row['job_id'];require(jid in jobs and jid not in by_id,'Missing/duplicate authoritative job');by_id[jid]=row;job=jobs[jid]
        require(jid==canonical({k:x for k,x in job.items() if k!='id'}) and job['menu_sha256']==menu['menu_sha256'],'Frozen job identity differs')
        candidate=candidates.get(row['candidate_id']);require(candidate is not None and row['candidate_id']==job['candidate_id'],'Unknown candidate')
        require(row['target']==candidate['target'] and row['source_recipe_id']==candidate['source_recipe_id'],'Candidate ancestry differs')
        for key in ('seed','arm','phase','pushes','program_sha256','pair_id','identity_probe'):
            require(row[key]==job[key],'Scientific job field differs: '+key)
        require(HEX.fullmatch(row['program_sha256']) and row['arm'] in ('baseline','candidate','reversal','random'),'Invalid program/arm')
        if row['arm']=='candidate':require(row['program_sha256']==candidate['program_sha256'] and row['pushes']==candidate['pushes'],'Candidate program differs from menu')
        if row['arm']=='baseline':require(not row['pushes'],'Baseline is steered')
        if row['arm']=='reversal':require(row['pushes']==[{**p,'dose':-p['dose']} for p in candidate['pushes']],'Reversal differs')
        require(row['split']=='discovery' and row['source_id'] in allowed and allowed[row['source_id']]==row['prompt']==job['prompt']['prompt'],'Unapproved source text')
        require(row['source_id']==job['prompt']['source_row_id'] and row['prompt_sha256']==digest(row['prompt'].encode()),'Prompt source hash differs')
        require(row['source_sha256']==(IDENTITY_SHA if row['identity_probe'] else CORPUS_SHA),'Prompt provenance differs')
        require(row['identity_probe'] is (row['prompt']=='who are you'),'Identity anchor flag differs')
        if candidate.get('historical_identity_only'):require(row['identity_probe'],'Historical high-dose exception escaped identity scope')
        require(type(row['complete'])is bool and type(row['capture_validated'])is bool,'Invalid completion flags')
        answer=row['answer'];require(answer is None or isinstance(answer,str) and len(answer)<=100000,'Answer bound/type differs')
        if row['complete']:require(row['state']=='complete' and row['capture_validated'] and bool(answer),'False complete-answer claim')
        if answer is not None:
            require(row['capture_validated'] and row['answer_sha256']==digest(answer.encode()),'Answer not validated/hash differs')
            require(all(HEX.fullmatch(row[k]) for k in ('request_sha256','response_sha256')),'Missing response bindings')
            require(set(row['capture'])=={'activation_artifact','provenance_artifact'},'Capture references missing')
            for pointer in row['capture'].values():
                require(set(pointer)=={'sha256','request_sha256','model_identity','runtime_identity','prepared_identity'} and HEX.fullmatch(pointer['sha256']) and pointer['request_sha256']==row['request_sha256'] and all(pointer[k]==x for k,x in v['binding'].items()),'Artifact binding differs')
    require(isinstance(v['proposals'],list) and len(v['proposals'])<=1024 and isinstance(v['reviews'],list) and len(v['reviews'])<=1024,'Review/proposal bound differs')
    for p in v['proposals']:
        require(set(p)=={'id','candidate_ids','rationale','state','evidence_sha256','menu_sha256'} and p['menu_sha256']==menu['menu_sha256'],'Proposal fields/menu differ')
        require(p['state'] in ('planned','dispatch_intent','enqueued','dispatch_uncertain_no_retry') and isinstance(p['candidate_ids'],list) and 1<=len(p['candidate_ids'])<=3 and len(set(p['candidate_ids']))==len(p['candidate_ids']),'Proposal shape/state differs')
        require(all(i in candidates and not candidates[i].get('historical_identity_only') for i in p['candidate_ids']),'Unapproved adaptive proposal candidate')
        require(isinstance(p['rationale'],str) and len(p['rationale'])<=4000,'Unbounded rationale')
    for review in v['reviews']:
        require(set(review)<={'id','candidate_id','source_id','pair_id','seed','identity_probe','source_sha256','source_controller_sha256','state','reason','packet_sha256','anonymous_key','answer_sha256','judgments'},'Unapproved review fields')
        require(review['candidate_id'] in candidates and review['source_id'] in allowed and review['state'] in ('ineligible','validated','failed_no_retry','uncertain_no_retry','received'),'Unknown review source/state')
        require(review['source_controller_sha256']==source_hashes['controller'],'Review controller differs')
        if review['state']=='validated':
            key=review['anonymous_key'];require(len(key)==4 and len(review['judgments'])==4,'Blinded coverage differs')
            require({r['code'] for r in review['judgments']}==set(key),'Judge code coverage differs')
            for judgment in review['judgments']:
                require(set(judgment)=={'code','scores','fidelity','coherence','evidence_ids','evidence'} and set(judgment['scores'])==AXES and all(type(n)is int and 0<=n<=2 for n in judgment['scores'].values()),'Judgment score/schema differs')
                require(judgment['fidelity'] in ('supported','violation','unclear') and judgment['coherence'] in ('coherent','collapse','unclear'),'Judgment category differs')
                code=judgment['code'];answer=by_id.get(key[code]['job_id']);require(answer and answer['capture_validated'] and answer['complete'],'Judge references missing/incomplete answer')
                require(answer['candidate_id']==review['candidate_id'] and answer['source_id']==review['source_id'] and answer['seed']==review['seed'] and answer['arm']==key[code]['arm'],'Judge cross-answer reference')
                require(review['answer_sha256'][code]==answer['answer_sha256'],'Judged answer changed')
                require(1<=len(judgment['evidence'])<=2 and all(isinstance(s,str) and s and s in answer['answer'] for s in judgment['evidence']),'Judge quote not supported')
    return v


def snapshot(root):
    root=Path(root);allowed=approved_prompts(root);identity=(root/IDENTITY).read_bytes();require(digest(identity)==IDENTITY_SHA,'Identity fixture changed')
    fixture=json.loads(identity);anchor=fixture['baseline_probe'];require(anchor['prompt']=='who are you' and anchor.get('system','')=='' and anchor['source']['kind']=='synthetic_original','Identity anchor differs');allowed[anchor['id']]=anchor['prompt']
    menu=json.loads((root/'results/persona_adaptive_menu.json').read_text());require(menu['menu_sha256']==canonical({k:v for k,v in menu.items() if k!='menu_sha256'}),'Private menu hash differs')
    raw=(root/'results'/REPORT).read_bytes()
    with closing(sqlite3.connect('file:'+str((root/'results/persona_adaptive.sqlite').resolve())+'?mode=ro',uri=True)) as db:
        jobs={r[0]:json.loads(r[1]) for r in db.execute('SELECT id,body FROM jobs')}
    sources={'controller':digest((root/'persona/persona_adaptive_loop.py').read_bytes()),'trial_store':menu['implementation_sha256'],'identity_fixture':IDENTITY_SHA,'corpus':CORPUS_SHA}
    require(sources['trial_store']==digest((root/'persona/persona_adaptive_trials.py').read_bytes()),'Trial source changed')
    v=validate(raw,allowed,menu,jobs,sources)
    terminal_events=sum(p['state'] in ('enqueued','dispatch_uncertain_no_retry') for p in v['proposals'])+sum(r['state'] in ('validated','ineligible','failed_no_retry','uncertain_no_retry') for r in v['reviews'])
    return {REPORT:raw,IDENTITY:identity},{'terminal':v['terminal_jobs'],'events':terminal_events,'proposals':len(v['proposals']),'final':v['final'],'sha256':digest(raw),'menu_sha256':v['menu_sha256']}

def due(previous,current,cutoff=False):
    if previous is None:return current['terminal']>=16 or current['proposals']>0 or current['events']>=4 or current['final'] or cutoff
    require(current['terminal']>=previous['terminal'] and current['events']>=previous['events'],'Publication progress regressed')
    if current['sha256']==previous['sha256']:return False
    return current['terminal']//16>previous['terminal']//16 or current['events']//4>previous['events']//4 or current['final'] or cutoff

class Observer(Recovery):
    def freeze(self,files,markers):
        require(not self.pending(),'Pending snapshot must resolve first');require(set(files)=={REPORT,IDENTITY} and all(type(v)is bytes for v in files.values()),'Two-file byte allowlist differs')
        require(digest(files[IDENTITY])==IDENTITY_SHA,'Identity upload bytes differ');parent=self.hub.head()
        with self.db:
            i=self.db.execute("INSERT INTO snapshots(parent,state,markers) VALUES(?,'intent',?)",(parent,encoded(markers))).lastrowid
            for name,data in files.items():self.db.execute('INSERT INTO files VALUES(?,?,?,?)',(i,name,data,digest(data)))
            self.event(i,'durable_intent',{'parent':parent,'hashes':{n:digest(b) for n,b in files.items()}})
        return i
    def step(self):
        row=self.pending()
        if row:
            files=dict(self.db.execute('SELECT name,data FROM files WHERE snapshot=?',(row[0],)))
            require(set(files)=={REPORT,IDENTITY} and digest(files[IDENTITY])==IDENTITY_SHA,'Durable upload allowlist/fixture changed')
        return super().step()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--until',type=float,default=DEADLINE);p.add_argument('--publish',action='store_true');a=p.parse_args()
    require(math.isfinite(a.until) and time.time()<a.until<=DEADLINE,'Finite08:05 deadline required')
    if not a.publish:
        files,markers=snapshot(a.root);print(encoded({'hashes':{n:digest(b) for n,b in files.items()},'markers':markers}));return
    with (a.root/'results/persona_adaptive_publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lineage={'files':[REPORT,IDENTITY],'identity_sha256':IDENTITY_SHA,'corpus_sha256':CORPUS_SHA,'source_sha256':digest(Path(__file__).read_bytes()),'recovery_sha256':digest(Path(__file__).with_name('persona_publication_recovery.py').read_bytes())}
        observer=Observer(a.root/'results/persona_adaptive_publication.sqlite',BoundedHub(),lineage)
        while time.time()<a.until:
            try:
                if observer.pending():observer.step()
                else:
                    files,markers=snapshot(a.root)
                    if due(observer.previous(),markers,time.time()>=a.until-90):observer.freeze(files,markers)
                    elif markers['final']:break
            except Exception as exc:print(encoded({'state':'blocked','error_type':type(exc).__name__,'reason':str(exc)}),flush=True)
            time.sleep(min(15,max(0,a.until-time.time())))

if __name__=='__main__':main()
