"""Bounded selected-neuron traces aligned with consumed input positions."""
from __future__ import annotations
import base64
import json
import re
import zlib
import numpy as np


def normalize_trace(value):
    if value is None:return None
    if not isinstance(value,dict) or set(value)!={'id','neurons'} or not re.fullmatch('[0-9a-f]{32}',str(value['id'])):
        raise ValueError('Trace requires a unique 32-hex id and neuron coordinates')
    coordinates=value['neurons']
    if not isinstance(coordinates,list) or not 1<=len(coordinates)<=64:raise ValueError('Select 1–64 trace neurons')
    seen=set();counts={}
    for row in coordinates:
        if not isinstance(row,dict) or set(row)!={'layer','neuron'}:raise ValueError('Invalid trace coordinate')
        l,n=row['layer'],row['neuron']
        if type(l) is not int or type(n) is not int or not 0<=l<64 or not 0<=n<17408 or (l,n) in seen:
            raise ValueError('Trace coordinates must be unique in-model integers')
        seen.add((l,n));counts[l]=counts.get(l,0)+1
    if max(counts.values())>32:raise ValueError('At most 32 trace neurons per layer')
    return value


def pack_logits(logits):
    logits=np.asarray(logits,dtype='<f4')
    if logits.ndim!=1 or not np.isfinite(logits).all():raise ValueError('Invalid next-token logits')
    return {'encoding':'zlib-base64-f32le','count':len(logits),'data':base64.b64encode(zlib.compress(logits.tobytes(),1)).decode()}


def unpack_logits(value):
    if value.get('encoding')!='zlib-base64-f32le' or type(value.get('count')) is not int or not 1<=value['count']<=300000:raise ValueError('Invalid logit payload')
    decoder=zlib.decompressobj();raw=decoder.decompress(base64.b64decode(value['data'],validate=True),value['count']*4+1)
    if len(raw)!=value['count']*4 or not decoder.eof or decoder.unused_data:raise ValueError('Invalid compressed logit size')
    array=np.frombuffer(raw,dtype='<f4').astype(np.float64)
    if not np.isfinite(array).all():raise ValueError('Nonfinite logits')
    return array


def compare_logits(baseline,intervention,decode,k=24):
    a,b=unpack_logits(baseline),unpack_logits(intervention)
    if a.shape!=b.shape:raise ValueError('Vocabulary dimensions differ')
    def logsoftmax(x):
        shifted=x-x.max();return shifted-np.log(np.exp(shifted).sum())
    a,b=logsoftmax(a),logsoftmax(b);delta=b-a
    def rows(indices):return [{'id':int(i),'text':decode(int(i)),'delta_logp':float(delta[i]),'baseline_logp':float(a[i]),'intervention_logp':float(b[i])} for i in indices]
    return {'kl_baseline_to_intervention':max(0.,float(np.sum(np.exp(a)*(a-b)))),
        'kl_intervention_to_baseline':max(0.,float(np.sum(np.exp(b)*(b-a)))),
        'increases':rows(np.argsort(-delta,kind='stable')[:k]),'decreases':rows(np.argsort(delta,kind='stable')[:k]),
        'semantics':'Full-vocabulary next-token distributions at the shared final prompt position, before sampling filters/penalties. Later free-running tokens may diverge and are not paired KL measurements.'}


class TokenTrace:
    def __init__(self,engine,spec,ids,request_sha):
        self.engine=engine;self.spec=spec;self.ids=ids;self.events=[];self.logits=None
        self.layers=[];self.slots=[];counts={}
        for row in spec['neurons']:
            l,n=row['layer'],row['neuron'];slot=counts.get(l,0);counts[l]=slot+1
            if l>=engine.model.layers or n>=engine.model.intermediate:raise ValueError('Trace outside loaded model')
            self.layers.append(l);self.slots.append(slot)
        with engine.trace_lock:
            if spec['id'] in engine.trace_records:raise ValueError('Trace id already exists')
            while len(engine.trace_records)>=16:
                old=next((key for key,value in engine.trace_records.items() if value['state']!='running'),None)
                if old is None:raise ValueError('Trace storage busy')
                del engine.trace_records[old]
            self.record={'id':spec['id'],'state':'running','neurons':spec['neurons'],'events':self.events,
                'model_identity':engine.model.identity,'runtime_identity':engine.manifest['runtime_identity'],
                'request_sha256':request_sha,'prompt_tokens':len(ids),
                'semantics':'SwiGLU values at consumed input tokens after pre-down steering; selected neurons only. Residual norms observe all layers. Final sampled token has no activation unless subsequently consumed.'}
            engine.trace_records[spec['id']]=self.record

    def configure(self):
        capture=self.engine.capture
        capture.token_indices.zero_();capture.token_enabled.zero_()
        for row,slot in zip(self.spec['neurons'],self.slots):
            capture.token_indices[row['layer'],slot]=row['neuron'];capture.token_enabled[row['layer'],slot]=True

    def observe(self,kind,expected_positions,token_ids,last=False):
        owner=self.engine.generation if kind=='prefill' else self.engine.decode_generation
        capture=owner.decoder.activation_capture
        positions=capture.positions.cpu().numpy();valid=positions>=0
        if positions[valid].tolist()!=expected_positions or len(token_ids)!=int(valid.sum()):raise ValueError('Trace consumed-token alignment differs')
        values=capture.token_values[self.layers,:,self.slots].T.cpu().numpy()[valid].copy()
        norms=capture.residual_l2.T.cpu().numpy()[valid].copy()
        if not np.isfinite(values).all() or not np.isfinite(norms).all():raise ValueError('Nonfinite token trace')
        event={'kind':kind,'positions':expected_positions,'tokens':[{'id':int(i),'text':self.engine.tokenizer.decode([i],skip_special_tokens=False,clean_up_tokenization_spaces=False)} for i in token_ids],
            'values':values.tolist(),'residual_l2':norms.tolist()}
        with self.engine.trace_lock:self.events.append(event)
        if last:self.logits=owner.decoder.scratch.logits[0].cpu().numpy().astype(np.float32,copy=True)

    def arrays(self):
        return {'trace_positions':np.array([p for e in self.events for p in e['positions']],dtype=np.int64),
            'trace_token_ids':np.array([t['id'] for e in self.events for t in e['tokens']],dtype=np.int64),
            'trace_values':np.array([v for e in self.events for v in e['values']],dtype=np.float32),
            'trace_residual_l2':np.array([v for e in self.events for v in e['residual_l2']],dtype=np.float32),
            'trace_neurons':np.array([[n['layer'],n['neuron']] for n in self.spec['neurons']],dtype=np.int64),
            'next_token_logits':self.logits}

    def finish(self,state,error=None):
        with self.engine.trace_lock:self.record.update(state=state,error=error)

    def result(self):
        return {'id':self.spec['id'],'neurons':self.spec['neurons'],'events':self.events,
            'next_token_logits':pack_logits(self.logits),'semantics':self.record['semantics']}


def normalize_response_capture(value):
    """Bounded optional response-token residual evidence; no GPU allocation."""
    if value is None: return None
    if (not isinstance(value,dict) or set(value)-{'id','layers','max_positions'}
            or not isinstance(value.get('id'),str) or not re.fullmatch('[0-9a-f]{32}',value['id'])):
        raise ValueError('Response capture requires a unique 32-hex id, optional layers and max_positions')
    layers=value.get('layers',[])
    if (not isinstance(layers,list) or len(layers)>4 or any(type(layer) is not int or not 0<=layer<64 for layer in layers)
            or len(set(layers))!=len(layers)):
        raise ValueError('Response capture selects at most four unique layers in 0..63')
    maximum=value.get('max_positions',128 if layers else 0)
    if type(maximum) is not int or not 0<=maximum<=128 or bool(layers)!=(maximum>0):
        raise ValueError('Selected layers require 1..128 retained positions; means-only requires zero')
    return {'id':value['id'],'layers':list(layers),'max_positions':maximum}


class ResponseCapture:
    """Accumulate existing B1 decode residuals, never replay or sample extra tokens.

    Each ordinary decode consumes the previous generated token. The final
    sampled token (including EOS) is not consumed and has no residual evidence.
    FP64 accumulation is over the native FP32 capture values, not extra model
    precision. Optional per-token vectors retain only a bounded first prefix.
    """
    def __init__(self,engine,spec,prompt_tokens,request_sha):
        self.engine,self.spec,self.prompt_tokens,self.request_sha=engine,spec,prompt_tokens,request_sha
        self.sum=np.zeros((engine.model.layers,engine.model.hidden),dtype=np.float64)
        self.positions=[];self.ids=[];self.vectors=[]
        if any(layer>=engine.model.layers for layer in spec['layers']):raise ValueError('Response layer outside loaded model')
        with engine.trace_lock:
            if spec['id'] in engine.trace_records:raise ValueError('Response capture id already exists')
            while len(engine.trace_records)>=16:
                old=next((k for k,v in engine.trace_records.items() if v['state']!='running'),None)
                if old is None:raise ValueError('Trace storage busy')
                del engine.trace_records[old]
            self.record={'id':spec['id'],'kind':'response_residual_capture','state':'running','events':[],
                'consumed_generated_tokens':0,'request_sha256':request_sha}
            engine.trace_records[spec['id']]=self.record

    def observe(self,position,token_id):
        if len(self.ids)>=4095:raise ValueError('Response capture exceeds bounded generation budget')
        capture=self.engine.decode_generation.decoder.activation_capture
        positions=capture.positions.cpu().numpy()
        if positions[positions>=0].tolist()!=[position] or position!=self.prompt_tokens+len(self.ids):
            raise ValueError('Response residual consumed-token alignment differs')
        value=capture.residual_last.cpu().numpy().copy()
        if value.shape!=self.sum.shape or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError('Invalid full-layer response residual capture')
        self.sum+=value
        self.positions.append(int(position));self.ids.append(int(token_id))
        if len(self.vectors)<self.spec['max_positions']:
            self.vectors.append(value[self.spec['layers']].copy())
        with self.engine.trace_lock:self.record['consumed_generated_tokens']=len(self.ids)

    @staticmethod
    def ids_hash(values):
        import hashlib
        return hashlib.sha256(np.asarray(values,dtype='<i8').tobytes()).hexdigest()

    def arrays(self,generated):
        if self.ids!=generated[:-1] or self.positions!=list(range(self.prompt_tokens,self.prompt_tokens+max(0,len(generated)-1))):
            raise ValueError('Response residual coverage differs from actual generated tokens')
        count=len(self.ids)
        return {'response_residual_sum':self.sum,'response_residual_mean':self.sum/count if count else np.zeros_like(self.sum),
                'response_residual_count':np.int64(count),'response_residual_mean_defined':np.bool_(count>0),
                'response_consumed_token_ids':np.asarray(self.ids,dtype=np.int64),
                'response_consumed_positions':np.asarray(self.positions,dtype=np.int64),
                'response_selected_layers':np.asarray(self.spec['layers'],dtype=np.int64),
                'response_selected_positions':np.asarray(self.positions[:len(self.vectors)],dtype=np.int64),
                'response_selected_residuals':np.asarray(self.vectors,dtype=np.float32).reshape(len(self.vectors),len(self.spec['layers']),self.engine.model.hidden)}

    def result(self,generated,artifact,generation_complete):
        self.arrays(generated)
        result={'id':self.spec['id'],'artifact':artifact,'request_sha256':self.request_sha,
            'model_identity':self.engine.model.identity,'runtime_identity':self.engine.manifest['runtime_identity'],
            'prepared_identity':self.engine.prepared.server_config['identity'],
            'layers':self.sum.shape[0],'hidden':self.sum.shape[1],'consumed_generated_tokens':len(self.ids),
            'sampled_generated_tokens':len(generated),'mean_defined':bool(self.ids),
            'consumed_token_ids_sha256':self.ids_hash(self.ids),'consumed_positions_sha256':self.ids_hash(self.positions),
            'generated_token_ids_sha256':self.ids_hash(generated),'hash_encoding':'signed int64 little-endian contiguous bytes',
            'consumed_positions_start':self.prompt_tokens if self.ids else None,
            'consumed_positions_end_exclusive':self.prompt_tokens+len(self.ids) if self.ids else None,
            'all_consumed_generated_tokens_captured':True,'all_sampled_generated_tokens_captured':False,
            'final_sampled_unconsumed_token_id':int(generated[-1]) if generated else None,
            'generation_complete':bool(generation_complete),'selected_layers':self.spec['layers'],
            'selected_positions_retained':len(self.vectors),'selected_positions_limit':self.spec['max_positions'],
            'selected_profile_complete':len(self.vectors)==len(self.ids) if self.spec['layers'] else None,
            'semantics':'Native block-output residual vectors for consumed generated input tokens only, after residual steering. Full-layer FP64 sums/means of FP32 capture values. Prompt/template tokens excluded. Final sampled token is not consumed; no extra decode or teacher forcing. Mean is undefined at count zero; stored zeros are placeholders.'}
        with self.engine.trace_lock:self.record.update(state='complete',result=result)
        return result

    def finish_failed(self,error):
        with self.engine.trace_lock:self.record.update(state='failed',error=str(error))


def normalize_prefill_capture(value):
    if value is None:return None
    if (not isinstance(value,dict) or set(value)!={'id','layers','positions'}
            or not isinstance(value['id'],str) or not re.fullmatch('[0-9a-f]{32}',value['id'])):
        raise ValueError('Prefill capture requires id, layers and positions')
    layers,positions=value['layers'],value['positions']
    if (not isinstance(layers,list) or not 1<=len(layers)<=4
            or any(type(v) is not int or not 0<=v<64 for v in layers) or len(set(layers))!=len(layers)):
        raise ValueError('Prefill capture requires 1..4 unique layer IDs')
    if (not isinstance(positions,list) or not 1<=len(positions)<=128
            or any(type(v) is not int or v<0 for v in positions) or positions!=sorted(set(positions))):
        raise ValueError('Prefill capture requires 1..128 sorted unique nonnegative positions')
    return {'id':value['id'],'layers':list(layers),'positions':list(positions)}


def plan_prefill(ids,batch_tokens,*,stop=None,pulse=None,capture=None):
    """Partition original encoded IDs; never decode/re-tokenize or add a template.

    A zero/baseline control must supply the same pulse/capture/stop to use the
    identical partition. Equality with ordinary unsplit execution is a separate
    native qualification question, not assumed here.
    """
    if type(batch_tokens) is not int or batch_tokens<1:raise ValueError('Invalid prefill chunk capacity')
    if stop is not None and (type(stop) is not int or not 1<=stop<=len(ids)):
        raise ValueError('prefill_stop exceeds original encoded prompt or is empty')
    used=list(ids if stop is None else ids[:stop]);length=len(used)
    if not length:raise ValueError('Empty encoded prompt')
    if pulse is not None and (type(pulse) is not int or not 0<=pulse<length):
        raise ValueError('Pulse position outside consumed prompt prefix')
    positions=[] if capture is None else capture['positions']
    if any(p>=length for p in positions):raise ValueError('Snapshot position outside consumed prompt prefix')
    cuts={0,length,*range(batch_tokens,length,batch_tokens),*(p+1 for p in positions)}
    if pulse is not None:cuts.update((pulse,pulse+1))
    cuts=sorted(cuts);spans=list(zip(cuts,cuts[1:]))
    return used,spans,{'original_encoded_tokens':len(ids),'consumed_prompt_tokens':length,
        'original_token_ids_sha256':ResponseCapture.ids_hash(ids),'consumed_token_ids_sha256':ResponseCapture.ids_hash(used),
        'hash_encoding':'signed int64 little-endian contiguous bytes','prefill_stop':stop,
        'pulse_position':pulse,'spans':[list(s) for s in spans],
        'semantics':'Exact original encoded prefix; fixed tokens, no text round-trip. Steering active only in the single-token pulse span when requested. Compare against an identically partitioned zero/baseline arm; chunk partition equivalence requires native qualification.'}


class PrefillCapture:
    def __init__(self,engine,spec,ids,request_sha):
        self.engine,self.spec,self.ids,self.request_sha=engine,spec,ids,request_sha
        self.positions=[];self.vectors=[]
        if any(l>=engine.model.layers for l in spec['layers']):raise ValueError('Snapshot layer outside loaded model')
        with engine.trace_lock:
            if spec['id'] in engine.trace_records:raise ValueError('Prefill capture id already exists')
            while len(engine.trace_records)>=16:
                old=next((k for k,v in engine.trace_records.items() if v['state']!='running'),None)
                if old is None:raise ValueError('Trace storage busy')
                del engine.trace_records[old]
            self.record={'id':spec['id'],'kind':'prefill_residual_snapshots','state':'running','events':[],'positions_completed':0}
            engine.trace_records[spec['id']]=self.record

    def observe(self,position):
        if position not in self.spec['positions']:return
        c=self.engine.capture;actual=c.positions.cpu().numpy();actual=actual[actual>=0]
        if not len(actual) or int(actual[-1])!=position or position!=self.spec['positions'][len(self.positions)]:
            raise ValueError('Prefill snapshot position alignment differs')
        value=c.residual_last.cpu().numpy()[self.spec['layers']].copy()
        if value.shape!=(len(self.spec['layers']),self.engine.model.hidden) or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError('Invalid prefill residual snapshot')
        self.positions.append(position);self.vectors.append(value)
        with self.engine.trace_lock:self.record['positions_completed']=len(self.positions)

    def arrays(self):
        if self.positions!=self.spec['positions']:raise ValueError('Prefill snapshot coverage incomplete')
        return {'prefill_snapshot_positions':np.asarray(self.positions,dtype=np.int64),
                'prefill_snapshot_token_ids':np.asarray([self.ids[p] for p in self.positions],dtype=np.int64),
                'prefill_snapshot_layers':np.asarray(self.spec['layers'],dtype=np.int64),
                'prefill_snapshot_residuals':np.asarray(self.vectors,dtype=np.float32)}

    def result(self,artifact):
        self.arrays()
        result={'id':self.spec['id'],'artifact':artifact,'request_sha256':self.request_sha,
            'model_identity':self.engine.model.identity,'runtime_identity':self.engine.manifest['runtime_identity'],
            'prepared_identity':self.engine.prepared.server_config['identity'],
            'positions':self.positions,'layers':self.spec['layers'],'token_ids':[self.ids[p] for p in self.positions],
            'positions_sha256':ResponseCapture.ids_hash(self.positions),'token_ids_sha256':ResponseCapture.ids_hash([self.ids[p] for p in self.positions]),
            'complete':True,'semantics':'Native full block-output residual vectors at specified consumed original prompt positions after residual steering; layers selected explicitly. Fixed-input total downstream propagation, not direct wiring.'}
        with self.engine.trace_lock:self.record.update(state='complete',result=result)
        return result

    def finish_failed(self,error):
        with self.engine.trace_lock:self.record.update(state='failed',error=str(error))
