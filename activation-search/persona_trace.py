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
