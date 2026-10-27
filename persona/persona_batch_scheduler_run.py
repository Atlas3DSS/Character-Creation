"""Finite, journaled runtime parity requests; never retries submitted inference.

This operator qualification driver does not mark truncated answers as completed
prose. Production jobs continue through Manager and its stricter validator.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid
import urllib.error
from persona_batch_qualify import load_plan, request_for, checked_parity, recover_received
from persona_manager import fetch, check_admission
from persona_propagation import ArtifactStore, atomic, file_sha, IDENTITIES
from persona_vectors import ModelSpec, VectorBank, canonical_digest


def run(plan, endpoints, binding, *, root, journal, output, deadline, batch=False):
    binding={k:binding[k] for k in IDENTITIES}
    if output.exists():
        raise ValueError('Qualification phase receipt already exists')
    if not time.time() < deadline < time.time()+86400:
        raise ValueError('Finite qualification deadline required')
    if len({e['gpu_uuid'] for e in endpoints}) != len(endpoints):
        raise ValueError('One dispatcher per physical GPU required')
    store=ArtifactStore(root/'results','v100','/mnt/storage16tb/Character Creation/results')
    if journal.exists():
        recovery=recover_received(plan,journal,store,binding)
        if recovery['uncertain_no_retry']:
            raise ValueError('Uncertain prior submissions; no replay allowed')
    else:
        journal.touch(exist_ok=False)
        recovery={'records':{},'never_submitted':[j['id'] for j in plan['jobs']]}
    pending=[j for j in plan['jobs'] if j['id'] in recovery['never_submitted']]
    results={key:value['evidence'] for key,value in recovery['records'].items()}
    lock=threading.RLock();stop=threading.Event();errors=[]
    def event(value):
        with lock,journal.open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(),**value),allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
    def worker(endpoint):
        bank=VectorBank(root/'results',ModelSpec.load(root/'data/persona_model.json'))
        try:
            # Read-only readiness polling does not retry any inference.
            while not stop.is_set() and time.time()+15<deadline:
                with lock:
                    if not pending:return
                try:
                    fetch(endpoint['url']+'/v1/models',timeout=2)
                    break
                except (OSError,urllib.error.URLError):stop.wait(3)
            while not stop.is_set():
                with lock:
                    if not pending:return
                    jobs=pending[:4 if batch else 1];del pending[:len(jobs)]
                if time.time()+15>=deadline:raise TimeoutError('Qualification deadline reached')
                board=json.loads(subprocess.check_output(['/home/orwel/.local/bin/gpu-workboard','status','--json'],text=True,timeout=30))
                allowed,reason=check_admission(board,endpoint)
                if not allowed:raise ValueError('Admission: '+reason)
                descriptor=fetch(endpoint['url']+'/v1/models',timeout=10)['data'][0]
                if any(descriptor['flare'].get(k)!=v for k,v in binding.items()):raise ValueError('Runtime binding differs')
                requests=[request_for(job,bank,descriptor['id'])[0] for job in jobs]
                envelope={'batch_id':uuid.uuid4().hex,'requests':requests}
                if batch:
                    from persona_batch_dispatch import checked_preflight
                    preflight=fetch(endpoint['url']+'/v1/batch/preflight',envelope,timeout=30)
                    if not checked_preflight(preflight,envelope,binding,descriptor['flare']['batch']):raise ValueError('Frozen batch full budgets do not fit')
                board=json.loads(subprocess.check_output(['/home/orwel/.local/bin/gpu-workboard','status','--json'],text=True,timeout=30))
                allowed,reason=check_admission(board,endpoint)
                lease=next((j for j in board['jobs'] if j['id']==endpoint['lease_id']),{})
                request_deadline=min(deadline,lease.get('lease_expires_epoch',0))
                if not allowed or time.time()+15>=request_deadline:raise ValueError('Lease unavailable before submission: '+reason)
                for job,request in zip(jobs,requests):
                    event(dict(state='submitting',job_id=job['id'],endpoint=endpoint,request=request,request_sha256=canonical_digest(request),batch_id=envelope['batch_id'] if batch else None))
                timeout=min(600,request_deadline-time.time())
                if batch:
                    reply=fetch(endpoint['url']+'/v1/batch/completions',envelope,timeout=timeout)
                    if reply.get('batch_id')!=envelope['batch_id'] or [r.get('request_sha256') for r in reply.get('results',[])]!=[canonical_digest(r) for r in requests]:raise ValueError('Batch response order/coverage differs')
                    replies=[r['response'] for r in reply['results']]
                else:replies=[fetch(endpoint['url']+'/v1/chat/completions',requests[0],timeout=timeout)]
                # Preserve every received response before validating any one artifact.
                for job,response in zip(jobs,replies):event(dict(state='received',job_id=job['id'],response=response))
                for job,request,response in zip(jobs,requests,replies):
                    record=checked_parity(response,request,binding,store)
                    with lock:results[job['id']]=record['evidence']
                    print(json.dumps({'validated':job['id'],'count':len(results),'prose_complete':record['evidence']['prose_complete']}),flush=True)
        except Exception as error:
            stop.set()
            with lock:errors.append({'endpoint':endpoint['url'],'error':repr(error)})
    atomic(output,{'state':'running','plan_sha256':plan['input_sha256'],'binding':binding,'deadline':deadline,'recovered':len(results)})
    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:list(pool.map(worker,endpoints))
    report={'schema':'character-creation.batch-runtime-phase.v1','state':'complete' if len(results)==24 and not errors else 'incomplete_no_retry','plan_sha256':plan['input_sha256'],'binding':binding,'deadline':deadline,'finished':time.time(),'results':results,'errors':errors,'journal':str(journal),'journal_sha256':file_sha(journal),'driver_sha256':file_sha(__file__),'scope':'Runtime parity, not completed prose or personality efficacy'}
    atomic(output,report)
    if report['state']!='complete':raise RuntimeError(json.dumps(errors))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','endpoints','journal','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--batch',action='store_true');a=p.parse_args();launch=json.loads(a.endpoints.read_text())
    run(load_plan(a.plan),launch['endpoints'],launch['binding'],root=a.root,journal=a.journal,output=a.output,deadline=launch['deadline'],batch=a.batch)


if __name__=='__main__':main()
