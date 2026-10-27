"""Finite eight-request continuous-refill canary; no HTTP service or scheduler mutation."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import time
import uuid
import numpy as np
import persona_probe as probe
from persona_batch_server import BatchEngine,BatchResponseCapture,PLANES,binding,read_response_residual,prepare_adapter
from persona_batch_performance import LocalStore,compare_observations
from persona_batch_qualify import load_plan,request_for,checked_parity
from persona_batch_probe import atomic,sha
from persona_vectors import canonical_digest,ModelSpec,VectorBank

SOURCES=('persona_batch_stream.py','persona_batch_server.py','persona_batch_performance.py','persona_batch_capture.py','persona_batch_probe.py','persona_batch_qualify.py','persona_probe.py','persona_vectors.py')
POLICY={'requests':8,'isolated_references':8,'capacity':4,'deadline_seconds':1200,'initial_budgets':[1,8,64,128],'replacement_budgets':[1,8,64,128],'prefix_cache':False,'mtp':False,'automatic_rollout':False}


STREAM_SOURCE_SHA256=sha(__file__)
STREAM_LIMITS={'max_queued':16,'max_envelope_bytes':32*1024**2,'max_capture_bytes':1024*1024**2,'deadline_seconds':1200}


def stream_envelope(value,*,max_queued=16):
    import re
    if max_queued not in (8,16):raise ValueError('Queue capacity must be8or16')
    if not isinstance(value,dict) or set(value)!={'batch_id','requests'} or not isinstance(value['batch_id'],str) or not re.fullmatch('[A-Za-z0-9_-]{1,128}',value['batch_id']):raise ValueError('Invalid stream envelope')
    rows=value['requests']
    if not isinstance(rows,list) or not 1<=len(rows)<=max_queued or any(not isinstance(r,dict) for r in rows):raise ValueError('Bounded request list required')
    if len(json.dumps(value,allow_nan=False).encode())>STREAM_LIMITS['max_envelope_bytes']:raise ValueError('Envelope byte bound exceeded')
    hashes=[canonical_digest(r) for r in rows]
    if len(set(hashes))!=len(hashes):raise ValueError('Duplicate request hashes')
    ids=[r.get('capture_response',{}).get('id') if isinstance(r.get('capture_response'),dict) else None for r in rows]
    if None in ids or len(set(ids))!=len(ids):raise ValueError('Stream requires distinct response captures; use B1 fallback otherwise')
    return value


def check_empty_readmission(ledger,required):
    if ledger.live:raise ValueError('Expected empty ledger')
    if required>len(ledger.free):raise ValueError('Head request cannot fit empty page pool')
    return True


def stream_preflight(trial,value,*,max_queued=16):
    """Tokenizer/program validation only; never replay a graph or reserve pages."""
    value=stream_envelope(value,max_queued=max_queued);rows=[];estimated=0
    for request in value['requests']:
        summary,specs,tokens,_,_=trial.prepare({'batch_id':value['batch_id'],'requests':[request]})
        if not summary['fits']:raise ValueError('Request cannot fit full budget on this card')
        spec=specs[0];cap=spec['capture_response'];e=trial.e
        # Bound queued host capture arrays, including copies during serialization/validation.
        layers=e.model.layers;hidden=e.model.hidden;intermediate=e.model.intermediate
        base=4*layers*(2*hidden+intermediate)+16*layers*hidden+4*layers*len(tokens[0])
        selected=4*len(cap['layers'])*hidden*min(cap['max_positions'],spec['max_tokens']-1)
        estimated+=3*(base+selected)+32*(len(tokens[0])+spec['max_tokens'])
        rows.append(summary)
    if estimated>STREAM_LIMITS['max_capture_bytes']:raise ValueError('Queued capture memory estimate exceeds finite bound; split envelope')
    return {'batch_id':value['batch_id'],'fits':True,'binding':binding(trial.e),'max_active':4,'max_queued':max_queued,
        'prompt_tokens':[r['prompt_tokens'][0] for r in rows],'required_pages':[r['required_pages'][0] for r in rows],
        'available_pages':trial.e.capacity[1],'request_sha256':[canonical_digest(r) for r in value['requests']],
        'estimated_capture_bytes':estimated,'max_capture_bytes':STREAM_LIMITS['max_capture_bytes'],
        'page_policy':'Reserve full prompt+answer budget per admitted request; queued requests need not fit simultaneously.'}


def stream_capability(engine,receipt=None,*,max_queued=16):
    from persona_batch_server import SERVER_SOURCE_SHA256,UNSUPPORTED
    if max_queued not in (8,16):raise ValueError('Queue capacity must be8or16')
    qualified=False;receipt_sha=None
    if receipt is not None:
        try:
            raw=Path(receipt).read_bytes();r=json.loads(raw)
            qualified=(r.get('schema')=='character-creation.stream-service-qualification.v1' and r.get('verified') is True and r.get('passed') is True
                and r.get('binding')==binding(engine) and r.get('stream_sha256')==STREAM_SOURCE_SHA256 and r.get('server_sha256')==SERVER_SOURCE_SHA256
                and r.get('physical_pages')==engine.capacity[1] and r.get('page_tokens')==engine.page_tokens and r.get('max_queued')==max_queued
                and r.get('limits')==STREAM_LIMITS)
            if qualified:receipt_sha=__import__('hashlib').sha256(raw).hexdigest()
        except (OSError,ValueError,TypeError):pass
    return {'supported':True,'qualified':qualified,'qualification_sha256':receipt_sha,'binding':binding(engine),'stream_sha256':STREAM_SOURCE_SHA256,'server_sha256':SERVER_SOURCE_SHA256,
        'max_batch_size':max_queued,'max_active':4,'limits':deepcopy(STREAM_LIMITS),'unsupported_fields':list(UNSUPPORTED),
        'requires_response_capture':True,'prefix_cache':False,'mtp':False,'preflight_path':'/v1/stream/preflight','completions_path':'/v1/stream/completions'}


class StreamService:
    """Route-neutral bounded API. Single serialized owner; durable IDs survive restart."""
    def __init__(self,engine,receipt,*,max_queued=16,trial=None):
        import threading
        self.engine=engine;self.receipt=receipt;self.max_queued=max_queued;self.trial=trial or StreamTrial(engine);self.lock=threading.Lock()
    def capability(self):return stream_capability(self.engine,self.receipt,max_queued=self.max_queued)
    def preflight(self,value):return stream_preflight(self.trial,value,max_queued=self.max_queued)
    def complete(self,value):
        import os
        from contextlib import contextmanager
        @contextmanager
        def exclusive():
            if not self.lock.acquire(blocking=False):raise ValueError('Stream card busy; envelope not submitted')
            try:yield
            finally:self.lock.release()
        with exclusive():
            cap=self.capability()
            if not cap['qualified']:raise ValueError('Stream runtime is not qualified')
            value=deepcopy(stream_envelope(value,max_queued=self.max_queued))
            path=self.engine.output/('stream-'+canonical_digest(value['batch_id'])+'.jsonl')
            if path.exists():raise ValueError('Prior stream submission retained; no retry after uncertain result')
            summary=self.preflight(value)
            def append(f,event):f.write(json.dumps(event,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            with path.open('x') as f:
                append(f,{'state':'submitting','envelope':value,'preflight':summary,'capability':cap,'time':time.time()})
                try:
                    responses,metrics=self.trial.execute(value['requests'],time.monotonic()+STREAM_LIMITS['deadline_seconds'])
                    append(f,{'state':'received','responses':responses,'metrics':metrics})
                    expected=summary['request_sha256']
                    if set(responses)!=set(expected) or len(responses)!=len(expected):raise ValueError('Missing/extra stream responses; no partial success')
                    evidence=[]
                    for request,digest in zip(value['requests'],expected):
                        record=checked_parity(responses[digest],request,binding(self.engine),LocalStore(self.engine.output));evidence.append(record['evidence'])
                    result={'batch_id':value['batch_id'],'results':[{'request_sha256':h,'response':responses[h]} for h in expected],'profiling':metrics}
                    append(f,{'state':'validated','evidence':evidence,'response':result});return result
                except BaseException as error:
                    append(f,{'state':'uncertain_no_retry','error':repr(error),'time':time.time()});raise


class Slots:
    """CPU ownership ledger: reserve entire budget before native admission."""
    def __init__(self,pages):self.free=set(range(pages));self.live={};self.epochs=[0]*4;self.history=[]
    def admit(self,identity,count):
        if type(count) is not int or count<=0:raise ValueError('Invalid page requirement')
        if identity in [v['id'] for v in self.live.values()]:raise ValueError('Duplicate owner')
        slot=next((s for s in range(4) if s not in self.live),None)
        if slot is None or len(self.free)<count:return None
        pages=sorted(self.free)[:count];self.free.difference_update(pages);self.epochs[slot]+=1
        row={'id':identity,'pages':pages,'generation':self.epochs[slot]};self.live[slot]=row;self.history.append({'slot':slot,**deepcopy(row)});return slot,row
    def retire(self,slot,identity,generation):
        row=self.live.get(slot)
        if row is None or row['id']!=identity or row['generation']!=generation:raise ValueError('Stale retirement generation/owner')
        self.free.update(row['pages']);del self.live[slot]


def freeze(root,source,compatibility):
    src=load_plan(source);receipt=json.loads(compatibility.read_text())
    if receipt.get('passed') is not True or receipt.get('state')!='complete':raise ValueError('Completed compatibility required')
    bank=VectorBank(root/'results',ModelSpec.load(root/'data/persona_model.json'))
    jobs=[]
    for i,index in enumerate((0,1,2,3,4,5,12,13)):
        row=deepcopy(src['jobs'][index]);row['max_tokens']=(1,8,64,128)[i%4]
        if i>=4:row['seed']=80317+i
        row['id']=canonical_digest({'source':row,'stream_index':i});request,proof=request_for(row,bank)
        if any(p.get('kind')=='direction' for p in row['recipe']['pushes']) and proof['program_sha256'] not in receipt['allowed_program_sha256']:raise ValueError('Unproved direction program')
        jobs.append({'source_job':row,'request':request})
    p={'schema':'character-creation.batch-stream-plan.v1','policy':POLICY,'jobs':jobs,'source_plan_sha256':sha(source),'compatibility_sha256':sha(compatibility),'core':receipt['new_core_source_hashes'],'model_identity':bank.model.identity,'prepared_identity':receipt['new_binding']['prepared_identity'],'source_sha256':{n:sha(Path(__file__).with_name(n)) for n in SOURCES},'maximum_sampled_tokens':sum(j['request']['max_tokens'] for j in jobs)*2}
    p['input_sha256']=canonical_digest(p);return p


def validate_plan(p,*,loaded_server_sha256=None):
    if p.get('schema')!='character-creation.batch-stream-plan.v1' or p.get('policy')!=POLICY or p.get('input_sha256')!=canonical_digest({k:v for k,v in p.items() if k!='input_sha256'}):raise ValueError('Frozen stream plan differs')
    if len(p['jobs'])!=8 or len({j['source_job']['id'] for j in p['jobs']})!=8:raise ValueError('Eight unique requests required')
    if [j['request']['max_tokens'] for j in p['jobs']]!=[1,8,64,128]*2:raise ValueError('Fixed budgets differ')
    if set(p['source_sha256'])!=set(SOURCES):raise ValueError('Source coverage changed')
    for n,digest in p['source_sha256'].items():
        actual=loaded_server_sha256 if n=='persona_batch_server.py' and loaded_server_sha256 is not None else sha(Path(__file__).with_name(n))
        if actual!=digest:raise ValueError('Source changed: '+n)
    return p


class StreamTrial(BatchEngine):
    def execute(self,requests,deadline):
        import torch
        from ignition.components.sampling import RequestAdmission
        from validate_candidate import _restore
        from persona_qualify import site_layout
        from persona_batch_capture import FIELDS,mapping_expected
        e=self.e;prepared=[self.prepare({'batch_id':uuid.uuid4().hex,'requests':[r]}) for r in requests]
        ledger=Slots(e.capacity[1]);owners={};completed=[];all_jobs=[];next_job=0;start=time.monotonic();metrics={'decode_steps':0,'active_slot_steps':0,'prefill_seconds':0.,'prefill_stall_seconds':0.,'graph_and_sync_seconds':{},'queue_wait_seconds':[],'events':[]}
        def check():
            if e.stop_event.is_set() or time.monotonic()>=deadline:raise TimeoutError('Finite stream deadline/interruption')
        def state():return [list(x) for x in zip(*(getattr(e.requests,k).cpu().tolist() for k in probe.REQUEST_STATE_FIELDS))]
        def replay(name,kind,slots,expected):
            check();before=state();e.sequence+=1;frame=e.sequence%2;e.buffers['frame_index'].fill_(frame);e.buffers['submission_sequence'].fill_(e.sequence)
            stamp=time.monotonic();e.prepared.graphs[name].replay();torch.cuda.synchronize(e.prepared.device);metrics['graph_and_sync_seconds'][kind]=metrics['graph_and_sync_seconds'].get(kind,0.)+time.monotonic()-stamp;after=state();records=e.buffers['completion_records'][frame].cpu().tolist()
            if any(before[s]!=after[s] for s in range(4) if s not in slots):raise probe.ProtocolError('Unrelated request state changed')
            for s in slots:
                j=owners[s];consumed,count,sampled=expected[s]
                j['result']=probe.validate_completion(records[s],after[s],epoch=j['epoch'],sequence=e.sequence,consumed=consumed,generated=count,sampled=sampled,budget=j['spec']['max_tokens'],eos=e.eos,vocabulary=e.vocabulary)
                j['trace'].append(records[s][:7]+[after[s][-1]]);j['graph_kinds'].append(kind)
            if kind=='init' and (bool(e.commands.errors.any()) or bool(e.maintenance.state.admission_errors.any())):raise probe.ProtocolError('Native refill admission failed')
            if kind=='decode':
                c=self.capture;evidence={k:getattr(c,k).cpu().numpy().copy() for k in ('slot_live','slot_positions','slot_token_ids','slot_generations','slot_match_counts','mapping_error','token_live','invalid_tokens','token_slots','token_positions','token_ids','token_generations')}
                if mapping_expected(before,slots,evidence)!=set(slots):raise probe.ProtocolError('Refill capture ownership differs')
                values=read_response_residual(c,slots)
                for s in slots:owners[s]['response_capture'].observe_value(before[s][1],before[s][3],values[:,s,:])
                metrics['decode_steps']+=1;metrics['active_slot_steps']+=len(slots)
        def retire(s):
            j=owners[s];self.save_job(j);completed.append(j);ledger.retire(s,j['id'],j['ledger_generation']);del owners[s]
            metrics['events'].append({'kind':'retire','slot':s,'generation':j['epoch'],'request_sha256':j['report']['request_sha256']})
        with e.lock,torch.inference_mode(),torch.cuda.device(e.prepared.device):
            snapshots={k:getattr(self.capture,k).clone() for k in FIELDS};success=False
            try:
                _restore(e.prepared,e.pool,e.snapshots)
                inp,cmd=e.inputs,e.commands;inp.request_state_rows.zero_();inp.page_tables.fill_(-1)
                for key in ('steering_addends','steering_phases','steering_prompt_tokens'):e.buffers[key].zero_()
                while next_job<len(requests) or owners:
                    check()
                    while next_job<len(requests):
                        summary,specs,tokens,programs,_=prepared[next_job];identity=uuid.uuid4().hex;reserved=ledger.admit(identity,summary['required_pages'][0])
                        if reserved is None:break
                        s,allocation=reserved;spec=specs[0];ids=tokens[0];payload=requests[next_job];e.epoch+=1
                        admission=RequestAdmission(generation=e.epoch,seed=spec['seed'],max_new_tokens=spec['max_tokens'],eos_token_ids=e.eos,**spec['sampling']);admission.validate(e.vocabulary)
                        j={'id':identity,'epoch':e.epoch,'ledger_generation':allocation['generation'],'spec':spec,'ids':ids,'generated':[],'trace':[],'graph_kinds':[],'positions':[],'norms':[],'finished':False,'artifact':e.output/f'probe-{identity}.npz','provenance':e.output/f'probe-{identity}.jsonl'}
                        j['report']={'schema':'character-creation.probe-request.v1','status':'running','id':identity,'operation':'rollout','runtime':'finite continuous-refill canary',**binding(e),'worker_manifest':e.manifest,'request':payload,'request_sha256':canonical_digest(payload),'started_at':time.time(),'state_restored':False,'effective_admission':admission.__dict__,'graphs':{**e.names,'decode':self.native.names[4]},'physical_pages':e.capacity[1],'batch':{'physical_slot':s,'generation':e.epoch,'physical_pages':allocation['pages']}}
                        j['response_capture']=BatchResponseCapture(e,spec['capture_response'],len(ids),canonical_digest(payload));owners[s]=j;all_jobs.append(j);next_job+=1
                        metrics['queue_wait_seconds'].append(time.monotonic()-start);metrics['events'].append({'kind':'admit','slot':s,'generation':e.epoch,'pages':allocation['pages'],'request_sha256':canonical_digest(payload)})
                        inp.request_state_rows[s,0]=s+1;inp.page_tables[s].fill_(-1);inp.page_tables[s,:len(allocation['pages'])].copy_(torch.tensor(allocation['pages'],dtype=torch.int32,device=e.prepared.device))
                        inp.row_slot_ids.fill_(-1);inp.row_slot_ids[0]=s;inp.row_generations.zero_();inp.row_generations[0]=j['epoch'];inp.live_rows.fill_(1);inp.query_lengths.zero_();inp.query_lengths[0]=1;inp.token_ids.zero_();inp.sample_enabled.zero_()
                        e.admission.stage(0,s,admission,ids);e.admission.admission_count.fill_(1)
                        cmd.request_slots.fill_(-1);cmd.source_slots.fill_(-1);cmd.destination_slots.fill_(-1);cmd.operation.zero_();cmd.expected_consumed.fill_(-1);cmd.request_slots[0]=s;cmd.destination_slots[0]=s+1;cmd.operation[0]=1;cmd.expected_consumed[0]=0;cmd.num_ops.fill_(1)
                        replay(e.names['init'],'init',[s],{s:(0,0,False)});e.admission.admission_count.zero_();cmd.num_ops.zero_()
                        e.buffers['steering_addends'][s].zero_();e.buffers['steering_phases'][s].zero_();e.buffers['steering_prompt_tokens'][s]=len(ids)
                        for site in programs[0]['sites']:
                            ix,offset,length=site_layout(e.catalog,e.buffers['steering_addends'].shape,site['site']);e.buffers['steering_addends'][s,offset:offset+length].copy_(torch.from_numpy(site['vector']));e.buffers['steering_phases'][s,ix]=probe.PHASE_MASKS[programs[0]['phase']]
                        other_live=any(not x['finished'] for slot,x in owners.items() if slot!=s);began=time.monotonic()
                        for a in range(0,len(ids),e.batch.tokens):
                            b=min(len(ids),a+e.batch.tokens);last=b==len(ids);inp.query_lengths[0]=b-a;inp.token_ids.zero_();inp.token_ids[:b-a].copy_(torch.tensor(ids[a:b],dtype=torch.int64,device=e.prepared.device));inp.sample_enabled[0]=int(last)
                            replay(e.names['prefill'],'prefill',[s],{s:(b,int(last),last)});pos=e.capture.positions.cpu().numpy().copy()
                            if pos[pos>=0].tolist()!=list(range(a,b)):raise probe.ProtocolError('Refill prefill positions differ')
                            j['positions'].extend(pos[pos>=0].tolist());norm=e.capture.residual_l2.cpu().numpy().copy()[:,pos>0]
                            if not np.isfinite(norm).all():raise probe.ProtocolError('Nonfinite prefill')
                            j['norms'].append(norm)
                            if last:j['prefill']={p:getattr(e.capture,p).cpu().numpy().copy() for p in PLANES};j['prefill']['final_chunk_positions']=pos;j['generated'].append(j['result']['token']);j['finished']=j['result']['finished']
                        duration=time.monotonic()-began;metrics['prefill_seconds']+=duration
                        if other_live:metrics['prefill_stall_seconds']+=duration
                    for s in list(owners):
                        if owners[s]['finished']:retire(s)
                    if not owners:
                        if next_job<len(requests):check_empty_readmission(ledger,prepared[next_job][0]['required_pages'][0])
                        continue
                    slots=sorted(owners);inp.row_slot_ids.fill_(-1);inp.row_generations.zero_();inp.sample_enabled.zero_();inp.live_rows.fill_(len(slots));inp.query_lengths.fill_(1)
                    for row,s in enumerate(slots):inp.row_slot_ids[row]=s;inp.row_generations[row]=owners[s]['epoch'];inp.sample_enabled[row]=1
                    replay(self.native.names[4],'decode',slots,{s:(len(owners[s]['ids'])+len(owners[s]['generated']),len(owners[s]['generated'])+1,True) for s in slots})
                    for s in slots:
                        j=owners[s];j['generated'].append(j['result']['token']);j['finished']=j['result']['finished']
                        if j['finished']:retire(s)
                success=True
            finally:
                try:
                    _restore(e.prepared,e.pool,e.snapshots)
                    for k,v in snapshots.items():getattr(self.capture,k).copy_(v)
                    torch.cuda.synchronize(e.prepared.device)
                    if any(not torch.equal(getattr(self.capture,k),v) for k,v in snapshots.items()):raise probe.ProtocolError('Capture restoration differs')
                    for j in all_jobs:j['report']['state_restored']=True
                finally:
                    for j in all_jobs:
                        if not success:j['report']['status']='failed';j['report']['error']='Finite refill attempt failed; no retry'
                        j['report']['finished_at']=time.time()
                        with j['provenance'].open('x') as f:json.dump(j['report'],f,allow_nan=False)
        responses={}
        for j in completed:
            r=j['response'];r['provenance_artifact']={'path':str(j['provenance'].resolve()),'sha256':sha(j['provenance']),**binding(e),'request_sha256':j['report']['request_sha256']};responses[j['report']['request_sha256']]=r
        metrics.update(wall_seconds=time.monotonic()-start,slot_history=ledger.history,mean_decode_occupancy=metrics['active_slot_steps']/max(1,metrics['decode_steps']),timing_scope='Host graph plus synchronization; prefill stall is time with another live request waiting; not CUDA kernel time')
        return responses,metrics


def experiment(engine,args):
    import threading
    from persona_batch_server import SERVER_SOURCE_SHA256
    p=validate_plan(json.loads(args.stream_plan.read_text()),loaded_server_sha256=SERVER_SOURCE_SHA256)
    if engine.manifest.get('preparation_adapter_source_sha256')!=SERVER_SOURCE_SHA256:raise ValueError('Loaded adapter manifest binding differs')
    out=args.output/'persona_batch_stream.json';journal=args.output/'persona_batch_stream.attempts.jsonl'
    if out.exists() or journal.exists():raise FileExistsError('Prior attempt retained; no automatic retry')
    if {**engine.manifest['source_sha256'],'heliopause':engine.manifest['heliopause_sha256']}!=p['core'] or engine.model.identity!=p['model_identity'] or binding(engine)['prepared_identity']!=p['prepared_identity']:raise ValueError('Native binding differs')
    bank=VectorBank(args.output,engine.model)
    for j in p['jobs']:
        if request_for(j['source_job'],bank)[0]!=j['request']:raise ValueError('Source program/prompt changed')
    report={'schema':'character-creation.batch-stream.v1','state':'running','passed':False,'plan':p,'loaded_server_sha256':SERVER_SOURCE_SHA256,'binding':binding(engine),'references':[],'comparisons':[],'automatic_rollout':False};atomic(out,report);journal.touch(exist_ok=False)
    def event(value):
        import os
        with journal.open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    def payloads(mode):
        rows=[]
        for j in p['jobs']:
            r=deepcopy(j['request']);r['capture_response']['id']=canonical_digest([p['input_sha256'],mode,j['source_job']['id']])[:32];r['metadata']['benchmark_id']=mode;rows.append(r)
        return rows
    start=time.monotonic();deadline=start+POLICY['deadline_seconds'];timer=threading.Timer(POLICY['deadline_seconds'],engine.stop_event.set);timer.daemon=True;timer.start()
    store=LocalStore(args.output);baseline=BatchEngine(engine);refs=[]
    try:
        for r in payloads('isolated'):
            if engine.stop_event.is_set() or time.monotonic()>=deadline:raise TimeoutError('Finite deadline')
            body={'batch_id':uuid.uuid4().hex,'requests':[r]};event({'state':'submitting_reference','request':body});answer=baseline.complete(body)['results'][0]['response'];event({'state':'received_reference','response':answer})
            checked=checked_parity(answer,r,binding(engine),store);refs.append(checked);report['references'].append(checked['evidence']);atomic(out,report)
        requests=payloads('stream');event({'state':'submitting_stream','requests':requests})
        responses,metrics=StreamTrial(engine).execute(requests,deadline);event({'state':'received_stream','responses':responses,'metrics':metrics});report['metrics']=metrics
        for r,ref in zip(requests,refs):
            checked=checked_parity(responses[canonical_digest(r)],r,binding(engine),store)
            report['comparisons'].append({'case_id':r['metadata']['job_id'],**compare_observations(ref,checked),'stream_evidence':checked['evidence'],'response':checked['response']})
        history=metrics['slot_history'];reused=any(sum(x['slot']==s for x in history)>=3 for s in range(4))
        if not reused:raise ValueError('Repeated physical-slot reuse was not exercised')
        if engine.stop_event.is_set() or time.monotonic()>=deadline:raise TimeoutError('Deadline before final validation')
        report.update(state='complete',passed=len(report['comparisons'])==8,repeated_slot_reuse=True)
    except BaseException as error:report.update(state='failed',passed=False,error=repr(error));raise
    finally:timer.cancel();report['wall_seconds']=time.monotonic()-start;atomic(out,report)
    return {'passed':report['passed'],'report':out.name,'sha256':sha(out)}


def main():
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='plan':
        p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--root',type=Path,required=True);p.add_argument('--source-plan',type=Path,required=True);p.add_argument('--compatibility',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
        value=freeze(a.root,a.source_plan,a.compatibility)
        with a.output.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    else:
        p=probe.argument_parser();p.add_argument('--stream-plan',type=Path,required=True);a=p.parse_args();validate_plan(json.loads(a.stream_plan.read_text()));probe.run(a,experiment=experiment,preparation_adapter=prepare_adapter)

if __name__=='__main__':main()
