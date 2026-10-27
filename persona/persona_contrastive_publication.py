"""One-report finite public observer. No inference imports, queues or worker control."""
import argparse
from collections import Counter
import fcntl
import json
import math
from pathlib import Path
import re
import time
from persona_publication_recovery import Recovery, BoundedHub, digest, encoded, DEADLINE
from persona_hub_research import reject_private_keys

REPORT='persona_contrastive_backfill_public.json'
SCHEMA='character-creation.contrastive-backfill-public.v1'
CORPUS_SHA='c2eb38ebcba2f39f3ff786f9054183ee166cf629f81fe55ad8e6e4171ea47bff'
MANIFEST_SHA='b18cfa620ebb650da349df16b5c8ea5fb132c2a387f5a009103ca98e40f0eeea'
TERMINAL={'complete','incomplete','failed','interrupted','cancelled'}
HEX=re.compile('[a-f0-9]{64}')


def require(ok,message):
    if not ok:raise ValueError(message)


def approved_prompts(root):
    raw=(Path(root)/'data/persona_perturbation_prompts.jsonl').read_bytes()
    manifest=(Path(root)/'data/persona_perturbation_prompts_manifest.json').read_bytes()
    require(digest(raw)==CORPUS_SHA and digest(manifest)==MANIFEST_SHA,'Authored corpus/manifest bytes differ')
    rows=[json.loads(line) for line in raw.splitlines() if line.strip()];allowed={}
    for row in rows:
        if row.get('split')!='discovery':continue
        require(row.get('system','')=='' and row['source']['kind']=='synthetic_original' and row['source']['prompt_sha256']==digest(row['prompt'].encode()),'Source is not exact original authored discovery')
        require(row['id'] not in allowed,'Duplicate source ID');allowed[row['id']]=row['prompt']
    require(allowed,'No allowed discovery rows');return allowed


def validate_report(raw,allowed,input_sha256):
    require(len(raw)<=256*1024**2,'Report byte bound exceeded')
    value=json.loads(raw);reject_private_keys(value)
    require(isinstance(value,dict) and set(value)<={'schema','input_sha256','source_sha256','source_manifest_sha256','binding','counts','terminal_jobs','total_jobs','final','state','rows','scope','raw_arrays_included','promotion_eligible','dispatch_deadline','deadline'},'Unapproved report fields')
    require(value.get('raw_arrays_included') is False and value.get('promotion_eligible') is False,'Unapproved coverage/promotion claim')
    require(value.get('dispatch_deadline')==1793112540 and value.get('deadline')==1793113200,'Study deadline differs')
    require(value.get('schema')==SCHEMA and value.get('input_sha256')==input_sha256 and HEX.fullmatch(input_sha256),'Report plan/schema differs')
    require(value.get('source_sha256')==CORPUS_SHA and value.get('source_manifest_sha256')==MANIFEST_SHA,'Report authored source differs')
    rows=value.get('rows');counts=value.get('counts');total=value.get('total_jobs');terminal=value.get('terminal_jobs')
    require(type(total) is int and 1<=total<=12800 and type(terminal) is int and 0<=terminal<=total,'Invalid bounded job totals')
    require(isinstance(rows,list) and len(rows)==terminal and isinstance(counts,dict),'Terminal row coverage differs')
    require(all(k in TERMINAL|{'queued','running'} and type(n) is int and n>=0 for k,n in counts.items()),'Invalid state counts')
    observed=Counter(r.get('state') for r in rows)
    require(set(observed)<=TERMINAL and all(counts.get(k,0)==observed[k] for k in TERMINAL) and sum(counts.values())==total,'State count mismatch')
    require(type(value.get('final')) is bool and value.get('state') in ('running','complete','deadline'),'Invalid publication status')
    if value['state']=='complete':require(value['final'] and terminal==total,'Incomplete study called complete')
    if value['final']:require(value['state'] in ('complete','deadline'),'Final state differs')
    bound=value.get('binding');require(isinstance(bound,dict) and set(bound)=={'model_identity','runtime_identity','prepared_identity'} and all(isinstance(v,str) and HEX.fullmatch(v) for v in bound.values()),'Invalid runtime binding')
    seen=set();baseline={}
    for row in rows:
        require(isinstance(row,dict) and set(row)<={'job_id','source_id','source_prompt_sha256','prompt','recipe_id','arm','shared_baseline_job_id','seed','program_sha256','pushes','phase','state','attempt','answer','complete_answer','capture_validated','finish_reason','usage','request_sha256','response_sha256','capture','response_capture'},'Unapproved row fields')
        require(row.get('source_id') in allowed and row.get('prompt')==allowed[row['source_id']],'Unapproved or changed source prompt')
        require(row.get('source_prompt_sha256')==digest(row['prompt'].encode()),'Prompt byte hash differs')
        require(isinstance(row.get('recipe_id'),str) and 1<=len(row['recipe_id'])<=160,'Invalid recipe identifier')
        identity=row.get('job_id');require(isinstance(identity,str) and HEX.fullmatch(identity) and identity not in seen,'Duplicate/invalid job ID');seen.add(identity)
        require(row.get('arm') in {'candidate','reversal','random','baseline'} and row.get('phase') in {'all','prefill','decode'},'Unknown intervention arm/phase')
        require(type(row.get('seed')) is int and 0<=row['seed']<2**32 and isinstance(row.get('pushes'),list),'Invalid recipe settings')
        require(isinstance(row.get('program_sha256'),str) and HEX.fullmatch(row['program_sha256']),'Missing program binding')
        ref=row.get('shared_baseline_job_id');require(isinstance(ref,str) and HEX.fullmatch(ref),'Missing shared baseline binding')
        pair=(row['source_id'],row['seed']);require(pair not in baseline or baseline[pair]==ref,'Shared baseline changed within scene/seed');baseline[pair]=ref
        if row['arm']=='baseline':require(identity==ref and row['pushes']==[],'Shared baseline is not unsteered/self-bound')
        require(type(row.get('complete_answer')) is bool and type(row.get('capture_validated')) is bool,'Invalid scientific completion flags')
        answer=row.get('answer');require(answer is None or isinstance(answer,str) and len(answer)<=100000,'Invalid answer text bound')
        if row['complete_answer']:require(row['state']=='complete' and row['capture_validated'] and bool(answer),'False complete answer claim')
        if row['state']!='complete':require(not row['complete_answer'],'Noncomplete row called complete')
        for key in ('request_sha256','response_sha256'):
            h=row.get(key);require(h is None or isinstance(h,str) and HEX.fullmatch(h),'Invalid response/request hash')
        if answer is not None:
            require(row.get('request_sha256') and row.get('response_sha256'),'Answer lacks request/response binding')
            require(set(row.get('capture',{}))=={'activation_artifact','provenance_artifact'},'Capture references missing')
            for pointer in row['capture'].values():
                require(set(pointer)=={'sha256','request_sha256','model_identity','runtime_identity','prepared_identity'} and all(pointer[k]==v for k,v in bound.items()) and pointer['request_sha256']==row['request_sha256'] and HEX.fullmatch(pointer['sha256']),'Capture artifact binding differs')
            rc=row.get('response_capture',{});require(set(rc)=={'consumed_generated_tokens','sampled_generated_tokens','mean_defined','generated_token_ids_sha256','consumed_token_ids_sha256','consumed_positions_sha256'},'Response capture fields differ')
            require(type(rc['sampled_generated_tokens']) is int and 1<=rc['sampled_generated_tokens']<=2048 and rc['consumed_generated_tokens']==rc['sampled_generated_tokens']-1 and rc['mean_defined'] is (rc['consumed_generated_tokens']>0),'Response capture coverage differs')
            require(all(HEX.fullmatch(rc[k]) for k in ('generated_token_ids_sha256','consumed_token_ids_sha256','consumed_positions_sha256')),'Token/position hashes missing')
    return value


def snapshot(root,allowed,input_sha256):
    raw=(Path(root)/'results'/REPORT).read_bytes();v=validate_report(raw,allowed,input_sha256)
    return {REPORT:raw},{'terminal':v['terminal_jobs'],'total':v['total_jobs'],'final':v['final'],'sha256':digest(raw)}


def due(previous,current,cutoff=False):
    if previous is None:return current['terminal']>=48 or current['final'] or cutoff
    if current['terminal']<previous['terminal'] or current['total']!=previous['total']:raise ValueError('Report progress regressed')
    if current['sha256']==previous['sha256']:return False
    return current['terminal']//48>previous['terminal']//48 or current['final'] or cutoff


class Observer(Recovery):
    """Reuse immutable CAS/reconciliation unchanged; specialize only upload admission."""
    def freeze(self,files,markers):
        if self.pending():raise ValueError('Pending snapshot must resolve first')
        if set(files)!={REPORT} or any(type(b) is not bytes for b in files.values()):raise ValueError('One-report byte allowlist differs')
        parent=self.hub.head()
        with self.db:
            i=self.db.execute("INSERT INTO snapshots(parent,state,markers) VALUES(?,'intent',?)",(parent,encoded(markers))).lastrowid
            for name,data in files.items():self.db.execute('INSERT INTO files VALUES(?,?,?,?)',(i,name,data,digest(data)))
            self.event(i,'durable_intent',{'parent':parent,'hashes':{n:digest(b) for n,b in files.items()}})
        return i
    def step(self):
        row=self.pending()
        if row:
            names={x[0] for x in self.db.execute('SELECT name FROM files WHERE snapshot=?',(row[0],))}
            if names!={REPORT}:raise ValueError('Durable snapshot allowlist differs')
        return super().step()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--input-sha256',required=True);p.add_argument('--until',type=float,default=DEADLINE);p.add_argument('--publish',action='store_true');a=p.parse_args()
    require(HEX.fullmatch(a.input_sha256) and math.isfinite(a.until) and a.until<=DEADLINE,'Invalid frozen plan hash/deadline')
    allowed=approved_prompts(a.root)
    if not a.publish:
        files,markers=snapshot(a.root,allowed,a.input_sha256);print(encoded({'files':{n:digest(b) for n,b in files.items()},'markers':markers}));return
    require(time.time()<a.until,'Finite publication deadline expired')
    results=a.root/'results'
    with (results/'persona_contrastive_publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lineage={'report':REPORT,'input_sha256':a.input_sha256,'source_sha256':CORPUS_SHA,'source_manifest_sha256':MANIFEST_SHA,'publisher_sources':{n:digest(Path(__file__).with_name(n).read_bytes()) for n in ('persona_contrastive_publication.py','persona_publication_recovery.py','persona_hub_research.py')}}
        observer=Observer(results/'persona_contrastive_publication.sqlite',BoundedHub(),lineage)
        while time.time()<a.until:
            try:
                if observer.pending():observer.step()
                else:
                    files,markers=snapshot(a.root,allowed,a.input_sha256)
                    if due(observer.previous(),markers,cutoff=time.time()>=a.until-90):observer.freeze(files,markers)
                    elif markers['final']:break
            except Exception as exc:print(encoded({'state':'deferred','error_type':type(exc).__name__}),flush=True)
            time.sleep(min(15,max(0,a.until-time.time())))

if __name__=='__main__':main()
