"""Leased native batch service: cascaded B1 prefill, true B1/B2/B4 decode.

No leasing here. The normal probe supplies calibration, monitoring and cleanup.
Admission is opt-in and fail-closed; uncertain batch POSTs must never be retried.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import signal
import time
import uuid
import numpy as np
import persona_probe as probe
from persona_trace import ResponseCapture
from persona_vectors import canonical_digest

REUSED={'persona_batch_probe.py':'297cb97da6ab68ed747860a5a8540262eb83665db7023e01dd63b3d230a3b1ee',
        'persona_batch_capture.py':'04541454c5fbb0b71223dcf9aa0b2f832f1303b3dd231bbce416ab8b71b04f58'}
UNSUPPORTED=('kv_control','capture_search','capture_trace','capture_prefill','steering_pulse','prefill_stop')
PLANES=('mlp_input_last','residual_last','pre_mlp_last')
BATCH_RUNTIME='Native independent request slots; cascaded B1 prefill; ordinary B1/B2/B4 decode; actual PreparedSampling'


def verify_sources():
    for name,digest in REUSED.items():
        if probe.file_hash(Path(__file__).with_name(name))!=digest:raise ValueError('Reused batch source changed: '+name)


def prepare_adapter(prepare):
    verify_sources()
    from persona_batch_capture import prepare_adapter as capture_adapter
    return capture_adapter(prepare)


def binding(e):return {'model_identity':e.model.identity,'runtime_identity':e.manifest['runtime_identity'],'prepared_identity':e.prepared.server_config['identity']}


def envelope(value):
    if not isinstance(value,dict) or set(value)!={'batch_id','requests'} or not isinstance(value['batch_id'],str) or not re.fullmatch('[A-Za-z0-9_-]{1,128}',value['batch_id']):raise ValueError('Batch requires bounded batch_id and requests')
    if not isinstance(value['requests'],list) or not 1<=len(value['requests'])<=4:raise ValueError('Batch requires 1..4 requests')
    if len({canonical_digest(p) for p in value['requests']})!=len(value['requests']):raise ValueError('Duplicate request payloads require distinct metadata IDs')
    return value


def page_plan(lengths,budgets,*,page_tokens,physical_pages,logical_context,table_pages):
    counts=[probe.validate_capacity(n,b,physical_pages=physical_pages,page_tokens=page_tokens,logical_context=logical_context,table_pages=table_pages) for n,b in zip(lengths,budgets)]
    pages=[];offset=0
    for count in counts:pages.append(list(range(offset,offset+count)));offset+=count
    return counts,pages,offset<=physical_pages


class BatchResponseCapture(ResponseCapture):
    def observe_value(self,position,token,value):
        if len(self.ids)>=4095 or position!=self.prompt_tokens+len(self.ids):raise probe.ProtocolError('Batch response position/count differs')
        value=np.asarray(value)
        if value.dtype!=np.float32 or value.shape!=self.sum.shape or not np.isfinite(value).all():raise probe.ProtocolError('Invalid batch residual')
        self.sum+=value;self.ids.append(int(token));self.positions.append(int(position))
        if len(self.vectors)<self.spec['max_positions']:self.vectors.append(value[self.spec['layers']].copy())
        with self.engine.trace_lock:self.record['consumed_generated_tokens']=len(self.ids)


class BatchEngine:
    def __init__(self,engine,qualification=None):
        from persona_batch_probe import BatchTrial
        self.e=engine;self.native=BatchTrial(engine);self.receipt=qualification;self.seen=set()
        for width in (2,4):
            c=self.native.catalog[self.native.names[width]].decoder.activation_capture
            if c is None or not hasattr(c,'slot_live'):raise ValueError('Request-owned batch capture required')
        self.capture=self.native.catalog[self.native.names[4]].decoder.activation_capture
        if self.native.catalog[self.native.names[2]].decoder.activation_capture.residual_last.data_ptr()!=self.capture.residual_last.data_ptr():raise ValueError('Batch capture backing differs')

    def capability(self):
        e=self.e;qualified=False;receipt_sha=None
        if self.receipt and Path(self.receipt).exists():
            r=json.loads(Path(self.receipt).read_text())
            qualified=(r.get('schema')=='character-creation.batch-service-qualification.v1' and r.get('verified') is True and r.get('passed') is True
                and r.get('binding')==binding(e) and r.get('server_sha256')==probe.file_hash(__file__)
                and r.get('physical_pages')==e.capacity[1] and r.get('page_tokens')==e.page_tokens)
            if qualified:receipt_sha=probe.file_hash(self.receipt)
        return {'supported':True,'qualified':qualified,'qualification_sha256':receipt_sha,'max_batch_size':4,
            'preflight_path':'/v1/batch/preflight','completions_path':'/v1/batch/completions',
            'physical_pages':e.capacity[1],'page_tokens':e.page_tokens,'binding':binding(e),
            'unsupported_fields':list(UNSUPPORTED),'prefill':'cascaded B1','decode':'native ordinary B1/B2/B4'}

    def prepare(self,value):
        from persona_capture import tokenize_messages
        value=envelope(value);e=self.e;specs=[];tokens=[];programs=[];capture_ids=[]
        for payload in value['requests']:
            if any(payload.get(k) is not None for k in UNSUPPORTED) or payload.get('observation_only',False):raise ValueError('Unsupported batch feature; use ordinary B1 before submission')
            spec=probe.normalize_request(payload,e.model_id)
            if spec['max_tokens']>4096:raise ValueError('Batch generation budget exceeds4096')
            program=probe.validate_program(spec['steering'],e.catalog,e.steering['max_sites'])
            ids=tokenize_messages(e.tokenizer,spec['messages'])
            if not ids or any(type(t) is not int or not 0<=t<e.vocabulary for t in ids):raise ValueError('Invalid encoded batch prompt')
            if spec['capture_response']:capture_ids.append(spec['capture_response']['id'])
            specs.append(spec);tokens.append(ids);programs.append(program)
        if len(set(capture_ids))!=len(capture_ids):raise ValueError('Response capture IDs must be unique within batch')
        with e.trace_lock:
            if any(i in e.trace_records for i in capture_ids):raise ValueError('Response capture ID already exists')
        counts,pages,fits=page_plan([len(t) for t in tokens],[s['max_tokens'] for s in specs],page_tokens=e.page_tokens,physical_pages=e.capacity[1],logical_context=int(e.prepared.server_config['max_sequence_tokens']),table_pages=e.inputs.page_tables.shape[1])
        summary={'batch_id':value['batch_id'],'fits':fits,'prompt_tokens':[len(t) for t in tokens],'required_pages':counts,'available_pages':e.capacity[1],
            'reason':None if fits else 'Aggregate full-budget physical page capacity exceeded','binding':binding(e),'request_sha256':[canonical_digest(p) for p in value['requests']]}
        return summary,specs,tokens,programs,pages

    def preflight(self,value):return self.prepare(value)[0]

    def complete(self,value):
        import torch
        from ignition.components.sampling import RequestAdmission
        from validate_candidate import _restore
        from persona_qualify import site_layout
        from persona_batch_capture import FIELDS,mapping_expected
        e=self.e
        with e.lock,torch.inference_mode(),torch.cuda.device(e.prepared.device):
            summary,specs,tokens,programs,pages=self.prepare(value)
            if not summary['fits']:raise ValueError(summary['reason'])
            if value['batch_id'] in self.seen:raise ValueError('Batch ID already submitted; no uncertain retry')
            if e.stop_event.is_set():raise probe.ProtocolError('Worker stopping')
            if len(self.seen)>=100000:raise ValueError('Finite batch receipt capacity exhausted')
            self.seen.add(value['batch_id']);n=len(specs);width=1 if n==1 else 2 if n==2 else 4
            snapshots={k:getattr(self.capture,k).clone() for k in FIELDS}
            jobs=[];started=time.time();success=False
            try:
                for i,(payload,spec,ids) in enumerate(zip(value['requests'],specs,tokens)):
                    identity=uuid.uuid4().hex;e.epoch+=1
                    admission=RequestAdmission(generation=e.epoch,seed=spec['seed'],max_new_tokens=spec['max_tokens'],eos_token_ids=e.eos,**spec['sampling']);admission.validate(e.vocabulary)
                    job={'id':identity,'epoch':e.epoch,'spec':spec,'ids':ids,'generated':[],'trace':[],'graph_kinds':[],'positions':[],'norms':[],'finished':False,'artifact':e.output/f'probe-{identity}.npz','provenance':e.output/f'probe-{identity}.jsonl'}
                    job['report']={'schema':'character-creation.probe-request.v1','status':'running','id':identity,'operation':'rollout','runtime':BATCH_RUNTIME,**binding(e),'worker_manifest':e.manifest,
                        'request':payload,'request_sha256':summary['request_sha256'][i],'started_at':started,'requested_sampling':spec['requested_sampling'],'effective_admission':admission.__dict__,
                        'graphs':{**e.names,'decode':self.native.names[width]},'physical_pages':e.capacity[1],'mapped_request_pages':len(pages[i]),'state_restored':False,'generated_tokens':0,
                        'batch':{'id':value['batch_id'],'index':i,'requests':n,'decode_capacity':width,'physical_slot':i,'physical_pages':pages[i]}}
                    job['admission']=admission;jobs.append(job)
                    job['response_capture']=BatchResponseCapture(e,spec['capture_response'],len(ids),summary['request_sha256'][i]) if spec['capture_response'] else None
                _restore(e.prepared,e.pool,e.snapshots)
                inp,cmd=e.inputs,e.commands;inp.request_state_rows.zero_();inp.page_tables.fill_(-1)
                for key in ('steering_addends','steering_phases','steering_prompt_tokens'):e.buffers[key].zero_()
                def state():return [list(x) for x in zip(*(getattr(e.requests,k).cpu().tolist() for k in probe.REQUEST_STATE_FIELDS))]
                def replay(name,kind,active,expected):
                    if e.stop_event.is_set():raise probe.ProtocolError('Batch interrupted')
                    before=state();e.sequence+=1;frame=e.sequence%2;e.buffers['frame_index'].fill_(frame);e.buffers['submission_sequence'].fill_(e.sequence)
                    e.prepared.graphs[name].replay();torch.cuda.synchronize(e.prepared.device)
                    after=state();records=e.buffers['completion_records'][frame].cpu().tolist()
                    for slot in range(4):
                        if slot not in active and before[slot]!=after[slot]:raise probe.ProtocolError('Inactive batch request state changed')
                    for i in active:
                        j=jobs[i];consumed,count,sampled=expected[i]
                        result=probe.validate_completion(records[i],after[i],epoch=j['epoch'],sequence=e.sequence,consumed=consumed,generated=count,sampled=sampled,budget=j['spec']['max_tokens'],eos=e.eos,vocabulary=e.vocabulary)
                        j['trace'].append(records[i][:7]+[after[i][-1]]);j['graph_kinds'].append(kind);j['result']=result
                    if kind=='init' and (bool(cmd.errors.any()) or bool(e.maintenance.state.admission_errors.any())):raise probe.ProtocolError('Batch native admission failed')
                    if kind=='decode':
                        owner=self.native.catalog[name]
                        if int(owner.batch.metadata.live_counts[0].item())!=len(active):raise probe.ProtocolError('Native concurrent decode token count differs')
                        if width==1:
                            c=owner.decoder.activation_capture
                            if c.positions.cpu().numpy()[c.positions.cpu().numpy()>=0].tolist()!=[before[0][1]]:raise probe.ProtocolError('B1 consumed position differs')
                            values=c.residual_last.cpu().numpy().copy()[:,None,:]
                        else:
                            c=owner.decoder.activation_capture
                            evidence={k:getattr(c,k).cpu().numpy().copy() for k in ('slot_live','slot_positions','slot_token_ids','slot_generations','slot_match_counts','mapping_error','token_live','invalid_tokens','token_slots','token_positions','token_ids','token_generations')}
                            if mapping_expected(before,list(range(n)),evidence)!=set(active):raise probe.ProtocolError('Batch capture ownership differs')
                            values=c.residual_last.cpu().numpy().copy()
                            for p in PLANES:
                                plane=getattr(c,p).cpu().numpy()
                                if any(np.any(plane[:,s,:]) for s in set(range(4))-set(active)):raise probe.ProtocolError('Inactive capture plane not zero')
                        for i in active:
                            rc=jobs[i]['response_capture']
                            if rc:rc.observe_value(before[i][1],before[i][3],values[:,i,:])
                for i,j in enumerate(jobs):
                    ids=j['ids'];inp.request_state_rows[i,0]=i+1
                    inp.page_tables[i,:len(pages[i])].copy_(torch.tensor(pages[i],dtype=torch.int32,device=e.prepared.device))
                    inp.row_slot_ids.fill_(-1);inp.row_slot_ids[0]=i;inp.row_generations.zero_();inp.row_generations[0]=j['epoch'];inp.live_rows.fill_(1)
                    inp.query_lengths.zero_();inp.query_lengths[0]=1;inp.token_ids.zero_();inp.sample_enabled.zero_()
                    e.admission.stage(0,i,j['admission'],ids);e.admission.admission_count.fill_(1)
                    cmd.request_slots.fill_(-1);cmd.source_slots.fill_(-1);cmd.destination_slots.fill_(-1);cmd.operation.zero_();cmd.expected_consumed.fill_(-1)
                    cmd.request_slots[0]=i;cmd.destination_slots[0]=i+1;cmd.operation[0]=1;cmd.expected_consumed[0]=0;cmd.num_ops.fill_(1)
                    replay(e.names['init'],'init',[i],{i:(0,0,False)});e.admission.admission_count.zero_();cmd.num_ops.zero_()
                    e.buffers['steering_prompt_tokens'][i]=len(ids)
                    for site in programs[i]['sites']:
                        index,offset,length=site_layout(e.catalog,e.buffers['steering_addends'].shape,site['site'])
                        e.buffers['steering_addends'][i,offset:offset+length].copy_(torch.from_numpy(site['vector']));e.buffers['steering_phases'][i,index]=probe.PHASE_MASKS[programs[i]['phase']]
                    for start in range(0,len(ids),e.batch.tokens):
                        stop=min(len(ids),start+e.batch.tokens);last=stop==len(ids)
                        inp.query_lengths[0]=stop-start;inp.token_ids.zero_();inp.token_ids[:stop-start].copy_(torch.tensor(ids[start:stop],dtype=torch.int64,device=e.prepared.device));inp.sample_enabled[0]=int(last)
                        replay(e.names['prefill'],'prefill',[i],{i:(stop,int(last),last)})
                        pos=e.capture.positions.cpu().numpy().copy()
                        if pos[pos>=0].tolist()!=list(range(start,stop)):raise probe.ProtocolError('Batch prefill positions differ')
                        j['positions'].extend(pos[pos>=0].tolist());norm=e.capture.residual_l2.cpu().numpy().copy()[:,pos>0]
                        if not np.isfinite(norm).all():raise probe.ProtocolError('Nonfinite prefill norms')
                        j['norms'].append(norm)
                        if last:
                            j['prefill']={p:getattr(e.capture,p).cpu().numpy().copy() for p in PLANES};j['prefill']['final_chunk_positions']=pos
                            if any(not np.isfinite(a).all() for a in j['prefill'].values()):raise probe.ProtocolError('Nonfinite prefill capture')
                            j['generated'].append(j['result']['token']);j['finished']=j['result']['finished']
                inp.row_slot_ids.fill_(-1);inp.row_generations.zero_();inp.sample_enabled.zero_();inp.live_rows.fill_(n);inp.query_lengths.fill_(1)
                for i,j in enumerate(jobs):inp.row_slot_ids[i]=i;inp.row_generations[i]=j['epoch'];inp.sample_enabled[i]=1
                while not all(j['finished'] for j in jobs):
                    active=[i for i,j in enumerate(jobs) if not j['finished']]
                    replay(self.native.names[width],'decode',active,{i:(len(jobs[i]['ids'])+len(jobs[i]['generated']),len(jobs[i]['generated'])+1,True) for i in active})
                    for i in active:j=jobs[i];j['generated'].append(j['result']['token']);j['finished']=j['result']['finished']
                for j in jobs:self.save_job(j)
                success=True
            except BaseException as error:
                for j in jobs:
                    j['report'].update(status='failed',error=f'{type(error).__name__}: {error}',generated_tokens=len(j['generated']))
                    if j.get('response_capture'):j['response_capture'].finish_failed(error)
                raise probe.ProtocolError('Batch replay/artifact failure: '+str(error)) from error
            finally:
                try:
                    _restore(e.prepared,e.pool,e.snapshots)
                    for k,v in snapshots.items():getattr(self.capture,k).copy_(v)
                    torch.cuda.synchronize(e.prepared.device)
                    if any(not torch.equal(getattr(self.capture,k),v) for k,v in snapshots.items()):raise probe.ProtocolError('Batch capture restoration differs')
                    for j in jobs:j['report']['state_restored']=True
                except BaseException as error:
                    success=False
                    for j in jobs:j['report'].update(status='failed',cleanup_error=str(error))
                    raise probe.ProtocolError('Batch cleanup failed') from error
                finally:
                    for j in jobs:
                        j['report']['finished_at']=time.time()
                        with j['provenance'].open('x') as stream:stream.write(json.dumps(j['report'],allow_nan=False)+'\n')
            if not success:raise probe.ProtocolError('Batch not successful')
            results=[]
            for j in jobs:
                response=j['response'];response['provenance_artifact']={'path':str(j['provenance'].resolve()),'sha256':probe.file_hash(j['provenance']),'operation':'rollout',**binding(e),'request_sha256':j['report']['request_sha256'],'self_reference':'Saved response excludes this post-write provenance pointer'}
                results.append({'request_sha256':j['report']['request_sha256'],'response':response})
            return {'batch_id':value['batch_id'],'results':results}

    def save_job(self,j):
        e=self.e;spec=j['spec'];ids=j['ids'];generated=j['generated'];rc=j['response_capture'];request_hash=j['report']['request_sha256']
        answer=e.tokenizer.decode(generated[:-1] if generated[-1] in e.eos else generated,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        outcome=probe.completion_outcome(spec,positions=j['positions'],prompt_tokens=len(ids),generated=generated,graph_kinds=j['graph_kinds'],result=j['result'],answer=answer)
        norms=np.concatenate(j['norms'],axis=1)
        summaries={'residual_l2_'+k:(fn(norms,axis=1) if norms.shape[1] else np.zeros(e.model.layers,np.float64)) for k,fn in [('min',np.min),('max',np.max),('mean',lambda a,axis:np.mean(a,axis=axis,dtype=np.float64)),('median',np.median)]}
        extra=rc.arrays(generated) if rc else {}
        with j['artifact'].open('xb') as stream:
            probe.save_activation_npz(stream,**j['prefill'],**summaries,**extra,residual_l2=norms,residual_l2_count=np.int64(norms.shape[1]),prefill_positions=np.asarray(j['positions'],np.int64),prompt_token_ids=np.asarray(ids,np.int64),generated_token_ids=np.asarray(generated,np.int64),virtual_prefix_slots=np.int64(0),completion_trace=np.asarray(j['trace'],np.int64),completion_graph_kind=np.asarray(j['graph_kinds']),completion_trace_columns=np.asarray(['epoch','sequence','consumed','generated','token','status','error','rng_counter']),**binding(e),request_sha256=request_hash,stage='final_prefill',runtime=BATCH_RUNTIME,operation='rollout')
        artifact={'path':str(j['artifact'].resolve()),'sha256':probe.file_hash(j['artifact']),**binding(e),'operation':'rollout','request_sha256':request_hash,'stage':'final_prefill','fields':list(PLANES)+list(extra),'semantics':'Native final-prefill SwiGLU product, residual output and normalized MLP input; request-owned response residuals.'}
        response={'id':'chatcmpl-'+j['id'],'object':'chat.completion','created':int(j['report']['started_at']),'model':e.model_id,'operation':'rollout','choices':[{'index':0,'message':{'role':'assistant','content':answer},'finish_reason':j['result']['finish_reason']}],
            'usage':{'prompt_tokens':len(ids),'completion_tokens':len(generated),'total_tokens':len(ids)+len(generated)},'runtime':BATCH_RUNTIME,**binding(e),'activation_artifact':artifact,
            'diagnostics':{**outcome['diagnostics'],'sampling':spec['sampling'],'requested_sampling':spec['requested_sampling'],'native_cache_qualified':False,'mtp_qualified':False,'batch':j['report']['batch']}}
        if rc:response['response_capture']=rc.result(generated,artifact,outcome['diagnostics']['complete_final_answer'])
        j['response']=response;j['report'].update(status='complete',response=response,generated_tokens=len(generated),sampled_text=answer,elapsed_seconds=time.time()-j['report']['started_at'],activation_artifact=artifact,final_completion=j['trace'][-1],completion_count=len(j['trace']))


def serve(engine,args):
    """Runs under the existing finite leased probe callback, with unchanged B1 handler."""
    batch=BatchEngine(engine,args.batch_qualification);original_descriptor=engine.descriptor
    def descriptor():
        value=original_descriptor();value['flare']['batch']=batch.capability();return value
    engine.descriptor=descriptor
    server=probe.server_for(engine,args.host,args.port,engine.stop_event);base=server.RequestHandlerClass
    class Handler(base):
        def do_POST(self):
            if self.path not in ('/v1/batch/preflight','/v1/batch/completions'):return super().do_POST()
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 1<=length<=32*1024**2:raise ValueError('Invalid batch request size')
                payload=json.loads(self.rfile.read(length))
                result=batch.preflight(payload) if self.path.endswith('/preflight') else batch.complete(payload)
            except (ValueError,KeyError,TypeError) as error:self.send_json(400,{'error':{'type':'invalid_request_error','message':str(error)[:2000]}});return
            except Exception as error:
                engine.stop_event.set()
                try:self.send_json(500,{'error':{'type':'diagnostic_worker_failure','message':str(error)[:2000]}})
                finally:os.kill(os.getpid(),signal.SIGTERM)
                return
            try:self.send_json(200,result)
            except (BrokenPipeError,ConnectionResetError):pass
    server.RequestHandlerClass=Handler
    engine.manifest.update(status='ready',ready_at=time.time(),url=f'http://{args.host}:{args.port}',physical_capacity=engine.capacity,descriptor=descriptor())
    # The callback source is included in probe runtime identity. Write its normal
    # worker report now; probe.run retains ownership of final terminal publication.
    report=args.output/f'{args.report_stem}-shard{args.shard}.manifest.json'
    from persona_batch_probe import atomic
    atomic(report,engine.manifest)
    print(json.dumps({'status':'ready','batch':batch.capability(),'binding':binding(engine)}),flush=True)
    try:server.serve_forever(poll_interval=.5)
    finally:engine.stop_event.set();server.server_close();engine.descriptor=original_descriptor
    return {'passed':True,'operation':'leased_batch_service_stopped','binding':binding(engine)}


def main():
    verify_sources();p=probe.argument_parser();p.add_argument('--batch-qualification',type=Path);args=p.parse_args()
    probe.run(args,experiment=serve,preparation_adapter=prepare_adapter)
if __name__=='__main__':main()
