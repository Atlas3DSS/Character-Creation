"""Experimental learned attention-KV training on a frozen, decoded-weight HF surrogate.

No model load or work starts on import. No native-Flare equivalence is implied.
Only reviewed benign literary pairs are accepted; held-out rows are never optimized.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
import signal
import threading
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

SCHEMA = 'character-creation.learned-attention-kv.v1'
DATA_SCHEMA = 'character-creation.phantom-literary-pairs.v1'
ATTENTION_LAYERS = tuple(range(3, 64, 4))


class TrainingInterrupted(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def load_pairs(path):
    raw = Path(path).read_bytes(); data = json.loads(raw)
    if data.get('schema') != DATA_SCHEMA or data.get('approval') != {'scope': 'benign_literary', 'approved': True}:
        raise ValueError('Dataset requires explicit reviewed benign-literary approval')
    rows = data.get('rows'); seen = set(); families = {}; sources = {}; prompts = {}
    if not isinstance(rows, list) or not 2 <= len(rows) <= 100000:
        raise ValueError('Expected bounded discovery and held-out pair rows')
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'id','family_id','split','source_sha256','prompt','target','domain','provenance'}:
            raise ValueError('Invalid literary pair fields')
        if row['id'] != digest({k:v for k,v in row.items() if k != 'id'}) or row['id'] in seen:
            raise ValueError('Pair identity changed or duplicated')
        seen.add(row['id'])
        if row['domain'] != 'benign_literary' or row['split'] not in ('discovery','heldout') or not valid_sha(row['source_sha256']):
            raise ValueError('Only bound benign literary discovery/heldout rows are accepted')
        if any(not isinstance(row[k], str) or not 1 <= len(row[k].strip()) <= 20000 for k in ('prompt','target','family_id')):
            raise ValueError('Invalid pair text/family')
        if row['family_id'] in families and families[row['family_id']] != row['split']:
            raise ValueError('A source family crosses discovery/heldout splits')
        families[row['family_id']] = row['split']
        for mapping,key in ((sources,row['source_sha256']),(prompts,digest(row['prompt']))):
            if key in mapping and mapping[key] != row['split']:
                raise ValueError('Duplicate source/prompt crosses discovery/heldout splits')
            mapping[key] = row['split']
        provenance = row['provenance']
        if not isinstance(provenance, dict) or provenance.get('source_kind') not in ('authored','literature') or provenance.get('source_sha256') != row['source_sha256']:
            raise ValueError('Missing source provenance')
        if provenance['source_kind'] == 'literature' and (not valid_sha(provenance.get('book_sha256')) or provenance.get('workgroup') != row['family_id'] or not provenance.get('scene_id')):
            raise ValueError('Literature pairs require book/workgroup/scene bindings')
    if {row['split'] for row in rows} != {'discovery','heldout'}:
        raise ValueError('Both discovery and held-out source families are required')
    return {'rows':rows, 'sha256':hashlib.sha256(raw).hexdigest(), 'approval':data['approval']}


def causal_prefix_mask(length, slots, *, device, dtype):
    """All graft slots visible; real-token suffix strictly causal. No padding support."""
    if length < 1 or slots < 0: raise ValueError('Invalid token or graft length')
    real = torch.arange(length, device=device)
    visible = torch.cat((torch.ones((length,slots),device=device,dtype=torch.bool), real[None,:] <= real[:,None]), dim=1)
    return torch.zeros((1,1,length,slots+length),device=device,dtype=dtype).masked_fill(~visible[None,None], float('-inf'))


def rotate_queries_keys(query, key, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    width = cos.shape[-1]
    if width % 2 or width > query.shape[-1]:raise ValueError('Unsupported rotary width')
    def rotate(x):
        part, tail = x[...,:width], x[...,width:]
        half = width//2
        return torch.cat((part*cos + torch.cat((-part[...,half:],part[...,:half]),dim=-1)*sin, tail), dim=-1)
    return rotate(query), rotate(key)


def attention_with_graft(attention, hidden, position_embeddings, keys=None, values=None):
    """Qwen3.5 gated attention with explicit post-norm/post-RoPE graft injection."""
    batch, length, _ = hidden.shape
    head_dim = attention.head_dim
    query, gate = attention.q_proj(hidden).view(batch,length,-1,head_dim*2).chunk(2,dim=-1)
    gate = gate.reshape(batch,length,-1)
    query = attention.q_norm(query).transpose(1,2)
    key = attention.k_norm(attention.k_proj(hidden).view(batch,length,-1,head_dim)).transpose(1,2)
    value = attention.v_proj(hidden).view(batch,length,-1,head_dim).transpose(1,2)
    query,key = rotate_queries_keys(query,key,*position_embeddings)
    slots = 0
    if keys is not None:
        if values is None or keys.shape != values.shape or tuple(keys.shape[::2]) != (key.shape[1],head_dim):
            raise ValueError('Graft shape must be KV-heads × slots × head-dim')
        slots = keys.shape[1]
        key = torch.cat((keys.to(key).unsqueeze(0).expand(batch,-1,-1,-1), key),dim=2)
        value = torch.cat((values.to(value).unsqueeze(0).expand(batch,-1,-1,-1), value),dim=2)
    elif values is not None:raise ValueError('Both keys and values are required')
    groups = attention.num_key_value_groups
    key,value = key.repeat_interleave(groups,dim=1),value.repeat_interleave(groups,dim=1)
    mask = causal_prefix_mask(length,slots,device=hidden.device,dtype=query.dtype)
    # Explicit math attention avoids unqualified fused kernels; autograd remains intact.
    weights = torch.softmax((query @ key.transpose(-2,-1)).float()*attention.scaling + mask.float(),dim=-1).to(query.dtype)
    mixed = (weights @ value).transpose(1,2).reshape(batch,length,-1)
    return attention.o_proj(mixed*torch.sigmoid(gate))


class GraftBank(nn.Module):
    def __init__(self, layers, kv_heads, slots, head_dim, *, devices=None, seed=7035, dtype=torch.float32):
        super().__init__()
        if not 1 <= slots <= 128 or not layers or len(set(layers)) != len(layers):raise ValueError('Invalid graft dimensions')
        self.layers=tuple(layers);self.slots=slots;self.keys=nn.ParameterDict();self.values=nn.ParameterDict()
        generator=torch.Generator(device='cpu').manual_seed(seed)
        for layer in self.layers:
            device='cpu' if devices is None else devices[layer]
            for bank in (self.keys,self.values):
                tensor=torch.randn((kv_heads,slots,head_dim),generator=generator,dtype=dtype)*.02
                bank[str(layer)]=nn.Parameter(tensor.to(device))


class FunctionalDecoder(nn.Module):
    """Fresh full-sequence recurrence per call; no HF cache or generation state is used."""
    def __init__(self, model, bank, *, checkpoint_blocks=True, guard=None):
        super().__init__();self.model=model;self.bank=bank;self.checkpoint_blocks=checkpoint_blocks;self.guard=guard
        for parameter in model.parameters():parameter.requires_grad_(False)
        model.eval()
        if hasattr(model,'gradient_checkpointing_disable'):model.gradient_checkpointing_disable()
        self.text=model.model
        observed=tuple(i for i,layer in enumerate(self.text.layers) if layer.block_type=='full_attention')
        if observed!=bank.layers:raise ValueError('Graft layers differ from actual full-attention layers')

    def hidden(self, input_ids, *, enabled=True, captures=None):
        if self.guard:self.guard()
        if input_ids.ndim!=2 or input_ids.shape[0]!=1:raise ValueError('Prototype requires one unpadded sequence')
        hidden=self.text.embed_tokens(input_ids.to(self.text.embed_tokens.weight.device))
        offset=self.bank.slots if enabled else 0
        positions=torch.arange(hidden.shape[1],device=hidden.device)[None,:]+offset
        # Text-only mRoPE uses equal temporal/height/width positions.
        rotary=self.text.rotary_emb(hidden,positions.unsqueeze(0).expand(3,-1,-1))
        for index,layer in enumerate(self.text.layers):
            if self.guard:self.guard()
            device=next(layer.parameters()).device;hidden=hidden.to(device)
            embeddings=tuple(x.to(device) for x in rotary)
            def block(x, layer=layer, index=index, embeddings=embeddings):
                if self.guard:self.guard()
                residual=x;normalized=layer.input_layernorm(x)
                if layer.block_type=='full_attention':
                    keys=self.bank.keys[str(index)] if enabled else None
                    values=self.bank.values[str(index)] if enabled else None
                    mixed=attention_with_graft(layer.self_attn,normalized,embeddings,keys,values)
                elif layer.block_type=='linear_attention':
                    # No cached prefix is synthesized for GDN; its initial state is zero.
                    mixed=layer.linear_attn(hidden_states=normalized,cache_params=None,attention_mask=None)
                else:raise ValueError('Unsupported hybrid layer')
                x=residual+mixed
                return x+layer.mlp(layer.post_attention_layernorm(x))
            hidden=checkpoint(block,hidden,use_reentrant=False) if enabled and self.checkpoint_blocks and torch.is_grad_enabled() else block(hidden)
            if captures is not None:captures[index]=hidden[0,-1].detach().float().cpu().numpy().copy()
        return self.text.norm(hidden.to(self.text.norm.weight.device))

    def loss(self, ids, target_start, *, enabled=True, kl_weight=0.1, logit_chunk=32):
        if not 1<=target_start<ids.shape[1] or not 0<=kl_weight<=1:raise ValueError('Invalid target boundary or KL weight')
        hidden=self.hidden(ids,enabled=enabled)
        # Baseline KL preservation is evaluated on prompt positions only.
        with torch.no_grad():baseline=self.hidden(ids,enabled=False) if enabled and kl_weight else None
        device=self.model.lm_head.weight.device
        hidden=hidden.to(device);baseline=baseline.to(device) if baseline is not None else None
        labels=ids[:,1:].to(device);tokens=ids.shape[1]-1
        losses=[];target_count=tokens-target_start+1;prompt_count=max(1,target_start-1)
        for start in range(0,tokens,logit_chunk):
            stop=min(tokens,start+logit_chunk)
            def chunk(h,start=start,stop=stop):
                if self.guard:self.guard()
                logits=self.model.lm_head(h).float();truth=labels[:,start:stop]
                target_mask=torch.arange(start,stop,device=device)>=target_start-1
                ce=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),truth.reshape(-1),reduction='none')
                loss=(ce*target_mask).sum()/target_count
                if baseline is not None and start<target_start-1:
                    with torch.no_grad():reference=self.model.lm_head(baseline[:,start:stop]).float().softmax(-1)
                    per_token=F.kl_div(logits.log_softmax(-1),reference,reduction='none').sum(-1)
                    loss=loss+kl_weight*(per_token*(~target_mask)).sum()/prompt_count
                return loss
            part=hidden[:,start:stop]
            losses.append(checkpoint(chunk,part,use_reentrant=False) if enabled and torch.is_grad_enabled() else chunk(part))
        return torch.stack(losses).sum()


def precision_settings(weight_dtype):
    if weight_dtype not in ('float16','float32'):raise ValueError('Unsupported frozen weight dtype')
    return {'weight_dtype':weight_dtype,'training_backend':'decoded-'+('fp32' if weight_dtype=='float32' else 'fp16')+'-hf-surrogate',
            'gradient_protocol':{'epsilons':[.1,.2] if weight_dtype=='float32' else [.01],
                                 'max_relative_error':.12,'require_positive_direction':True}}


def frozen_weight_dtype(name,weight_dtype):
    precision_settings(weight_dtype)
    return torch.float32 if weight_dtype=='float32' or any(s in name for s in ('norm.weight','A_log','dt_bias')) else torch.float16


def load_frozen_model(checkpoint_path, devices, guard=None, weight_dtype='float16'):
    """Expand one tensor at a time into a frozen model shard; never write weights."""
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from persona_phantom_weights import PackedCheckpoint
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextRotaryEmbedding
    reader=PackedCheckpoint(checkpoint_path)
    config=Qwen3_5TextConfig(**reader.text_config);config._attn_implementation='eager';config.use_cache=False
    if (config.num_hidden_layers!=64 or config.num_key_value_heads!=4 or config.head_dim!=256
            or tuple(i for i,t in enumerate(config.layer_types) if t=='full_attention')!=ATTENTION_LAYERS):
        raise ValueError('This prototype requires the inspected 48-GDN/16-attention 27B architecture')
    with torch.device('meta'):model=Qwen3_5ForCausalLM(config)
    def destination(name):
        if name.startswith('model.layers.'):
            index=int(name.split('.')[2]);return devices[min(len(devices)-1,index*len(devices)//64)]
        return devices[0] if name.startswith('model.embed_tokens.') else devices[-1]
    for name,parameter in list(model.named_parameters()):
        if guard:guard()
        dtype=frozen_weight_dtype(name,weight_dtype)
        value=reader.tensor(name,device=destination(name),dtype=dtype)
        if value.shape!=parameter.shape:raise ValueError('Decoded weight shape differs: '+name)
        parent=model.get_submodule(name.rsplit('.',1)[0]);setattr(parent,name.rsplit('.',1)[1],nn.Parameter(value,requires_grad=False))
    model.model.rotary_emb=Qwen3_5TextRotaryEmbedding(config).to(devices[0])
    if any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers()):raise ValueError('Unmaterialized model tensor remains')
    model.eval()
    return model,reader.manifest



def token_ids(value):
    """Unwrap HF return containers only; never re-tokenize or change chat framing."""
    if isinstance(value,Mapping):
        if 'input_ids' not in value:raise ValueError('Tokenizer mapping omitted input_ids')
        value=value['input_ids']
    if torch.is_tensor(value):
        if value.device.type!='cpu':raise ValueError('Tokenization must return CPU IDs')
        value=value.tolist()
    elif isinstance(value,np.ndarray):value=value.tolist()
    if isinstance(value,tuple):value=list(value)
    if isinstance(value,list) and len(value)==1 and isinstance(value[0],(list,tuple)):
        value=list(value[0])
    if not isinstance(value,list) or any(type(i) is not int or i<0 for i in value):
        raise ValueError('Tokenizer must return one flat sequence of nonnegative integer IDs')
    return list(value)

def encode_pair(tokenizer,row,max_tokens):
    prompt=token_ids(tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False))
    target=token_ids(tokenizer.encode(row['target'],add_special_tokens=False))
    if tokenizer.eos_token_id is not None:target.append(tokenizer.eos_token_id)
    if not prompt or not target or len(prompt)+len(target)>max_tokens:raise ValueError('Pair exceeds token limit; no truncation allowed: '+row['id'])
    return torch.tensor([prompt+target],dtype=torch.long),len(prompt)


def gradient_check(decoder,ids,target_start,*,epsilon=1e-2,tolerance=.12,raise_on_failure=True):
    """Directional central difference, evaluated without optimizer updates."""
    parameters=list(decoder.bank.parameters());decoder.bank.zero_grad(set_to_none=True)
    loss=decoder.loss(ids,target_start,kl_weight=0);loss.backward()
    if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):raise ValueError('Missing/nonfinite graft gradient')
    norm=math.sqrt(sum(p.grad.double().square().sum().item() for p in parameters))
    if not norm>0:raise ValueError('Zero graft gradient')
    directions=[p.grad.detach().clone()/norm for p in parameters];original=[p.detach().clone() for p in parameters]
    try:
        values=[]
        for sign in (1,-1):
            with torch.no_grad():
                for p,base,direction in zip(parameters,original,directions):p.copy_(base+sign*epsilon*direction)
                values.append(decoder.loss(ids,target_start,kl_weight=0).item())
        numerical=(values[0]-values[1])/(2*epsilon)
    finally:
        with torch.no_grad():
            for p,base in zip(parameters,original):p.copy_(base)
    relative=abs(numerical-norm)/max(abs(numerical),norm,1e-12)
    passed=math.isfinite(relative) and numerical>0 and relative<=tolerance
    result={'analytic_directional_derivative':norm,'numerical_directional_derivative':numerical,'relative_error':relative,
            'epsilon':epsilon,'positive_direction':numerical>0,'passed':passed,'loss_plus':values[0],'loss_minus':values[1]}
    if not passed and raise_on_failure:raise ValueError(f'Gradient check failed: {result}')
    return result


def precision_gradient_check(decoder,ids,target_start,weight_dtype):
    protocol=precision_settings(weight_dtype)['gradient_protocol'];measurements=[]
    for epsilon in protocol['epsilons']:
        try:result=gradient_check(decoder,ids,target_start,epsilon=epsilon,tolerance=protocol['max_relative_error'],raise_on_failure=False)
        except ValueError as error:result={'epsilon':epsilon,'passed':False,'error':str(error)}
        measurements.append(result)
    return {'passed':all(row['passed'] for row in measurements),'protocol':protocol,'measurements':measurements}



def write_candidate(path,bank,metadata):
    if metadata.get('optimizer_steps',0)<1 or metadata.get('native_flare_qualified') is not False:
        raise ValueError('Only an actually updated, explicitly unqualified learned candidate can be exported')
    settings=precision_settings(metadata.get('weight_dtype'))
    if any(metadata.get(k)!=v for k,v in settings.items()):raise ValueError('Candidate precision/backend binding differs')
    keys=torch.stack([bank.keys[str(i)].detach().float().cpu() for i in bank.layers]).numpy()
    values=torch.stack([bank.values[str(i)].detach().float().cpu() for i in bank.layers]).numpy()
    if not np.isfinite(keys).all() or not np.isfinite(values).all():raise ValueError('Nonfinite learned candidate')
    metadata={**metadata,'schema':SCHEMA,'key_space':'post_norm_post_rope','prompt_position_start':bank.slots,
              'recurrent_initial_state':'fresh_zero',
              'claim':'Learned candidate only; native inference compatibility, genre efficacy and coherence recovery unqualified.'}
    with Path(path).open('xb') as stream:
        np.savez(stream,keys=keys,values=values,attention_layers=np.asarray(bank.layers,dtype=np.int64),metadata=np.asarray(json.dumps(metadata,sort_keys=True,allow_nan=False)))




NATIVE_THRESHOLDS={'max_relative_l2':.10,'min_cosine':.99,'all_next_token_argmax_equal':True}


def load_native_reference(path,data,model_identity,tokenizer):
    value=json.loads(Path(path).read_text());records=value.get('records',[])
    if (value.get('schema')!='character-creation.phantom-native-reference.v1' or value.get('state')!='complete'
            or value.get('model_identity')!=model_identity or value.get('dataset_sha256')!=data['sha256']
            or value.get('thresholds')!=NATIVE_THRESHOLDS or len(records)!=6):
        raise ValueError('Native reference must contain six completed, identically bound records and fixed thresholds')
    by_id={row['id']:row for row in data['rows']};seen=set();loaded=[]
    for record in records:
        identity=record['row_id'];row=by_id.get(identity);request=record['request'];response=record['response'];endpoint=record['endpoint'];artifact=response['activation_artifact']
        if record.get('state')!='complete' or identity in seen or row is None or row['split']!='discovery':raise ValueError('Reference row missing, duplicated or not discovery')
        seen.add(identity)
        if (request.get('messages')!=[{'role':'user','content':row['prompt']}] or request.get('temperature')!=0
                or request.get('max_tokens')!=1 or request.get('observation_only') is not True
                or request.get('steering') is not None or request.get('kv_control') is not None
                or request.get('chat_template_kwargs')!={'enable_thinking':False}):raise ValueError('Reference is not an exact unsteered prompt observation')
        if endpoint.get('model_identity')!=model_identity or endpoint.get('runtime_identity')!=value['runtime_identity']:
            raise ValueError('Reference endpoint identity differs')
        for key in ('model_identity','runtime_identity','prepared_identity'):
            if not valid_sha(endpoint.get(key)) or response.get(key)!=endpoint[key] or artifact.get(key)!=endpoint[key]:raise ValueError('Reference response identity differs')
        if response.get('operation')!='prefill_observation' or artifact.get('operation')!='prefill_observation' or artifact.get('stage')!='final_prefill' or response.get('diagnostics',{}).get('prefill_observation_complete') is not True:
            raise ValueError('Native reference is not a completed final-prefill observation')
        if artifact.get('request_sha256')!=digest(request) or file_digest(artifact['path'])!=artifact['sha256']:
            raise ValueError('Reference artifact/request digest differs')
        with np.load(artifact['path'],allow_pickle=False) as arrays:
            for key in ('model_identity','runtime_identity','prepared_identity'):
                if str(arrays[key].item())!=endpoint[key]:raise ValueError('Reference NPZ identity differs')
            if str(arrays['stage'].item())!='final_prefill' or str(arrays['operation'].item())!='prefill_observation':raise ValueError('Reference NPZ phase differs')
            if str(arrays['request_sha256'].item())!=digest(request):raise ValueError('Reference NPZ request differs')
            ids=arrays['prompt_token_ids'].copy();generated=arrays['generated_token_ids'].copy();residual=arrays['residual_last'].copy()
        expected=token_ids(tokenizer.apply_chat_template(request['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False))
        if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or not np.array_equal(ids,expected) or generated.shape!=(1,)
                or generated.dtype.kind not in 'iu' or residual.shape!=(64,5120) or not np.isfinite(residual).all()):
            raise ValueError('Reference token/activation shape or exact tokenizer binding differs')
        loaded.append({'row_id':identity,'ids':ids,'generated':int(generated[0]),'residual':residual,'artifact_sha256':artifact['sha256']})
    return {'sha256':file_digest(path),'records':loaded,'runtime_identity':value['runtime_identity']}


def native_forward_check(decoder,reference):
    results=[]
    with torch.no_grad():
        for row in reference['records']:
            captured={};hidden=decoder.hidden(torch.from_numpy(row['ids'].astype(np.int64))[None,:],enabled=False,captures=captured)
            observed=np.stack([captured[i] for i in range(64)]).astype(np.float64);expected=row['residual'].astype(np.float64)
            if not np.isfinite(observed).all():raise ValueError('Surrogate native comparison produced nonfinite activations')
            left=np.linalg.norm(observed,axis=1);right=np.linalg.norm(expected,axis=1)
            relative=np.linalg.norm(observed-expected,axis=1)/np.maximum(right,1e-30)
            cosine=(observed*expected).sum(axis=1)/np.maximum(left*right,1e-30)
            cosine=np.where((left==0)&(right==0),1.,cosine)
            logits=decoder.model.lm_head(hidden[:,-1].to(decoder.model.lm_head.weight.device))
            predicted=int(logits.argmax(-1).item())
            passed=bool(relative.max()<=.10 and cosine.min()>=.99 and predicted==row['generated'])
            results.append({'row_id':row['row_id'],'artifact_sha256':row['artifact_sha256'],'relative_l2_per_layer':relative.tolist(),
                            'cosine_per_layer':cosine.tolist(),'max_relative_l2':float(relative.max()),'min_cosine':float(cosine.min()),
                            'native_next_token':row['generated'],'surrogate_next_token':predicted,'next_token_equal':predicted==row['generated'],'passed':passed})
    return {'passed':all(r['passed'] for r in results) and len(results)==6,'thresholds':NATIVE_THRESHOLDS,'records':results,
            'native_reference_sha256':reference['sha256'],'runtime_identity':reference['runtime_identity']}


def require_validation(path,metadata,manifest,native_reference):
    report=json.loads(Path(path).read_text())
    if (report.get('state')!='validation_passed' or report.get('validation',{}).get('passed') is not True
            or report.get('native_forward',{}).get('passed') is not True or report.get('checkpoint')!=manifest
            or report['native_forward'].get('native_reference_sha256')!=native_reference['sha256']
            or any(report.get(k)!=metadata[k] for k in ('model_identity','dataset_sha256','trainer_sha256','reader_sha256','seed','slots','max_tokens','weight_dtype','training_backend','gradient_protocol'))):
        raise ValueError('Training requires a successful, unchanged full-model native-forward and gradient validation report')
    settings=precision_settings(metadata['weight_dtype']);validation=report['validation']
    rows=validation.get('measurements',[]);protocol=settings['gradient_protocol']
    if (any(metadata.get(k)!=v for k,v in settings.items()) or validation.get('protocol')!=protocol
            or [row.get('epsilon') for row in rows]!=protocol['epsilons']
            or any(row.get('passed') is not True or row.get('positive_direction') is not True
                   or not isinstance(row.get('relative_error'),(int,float)) or not 0<=row['relative_error']<=.12
                   or not row.get('numerical_directional_derivative',0)>0 for row in rows)):
        raise ValueError('Gradient precision protocol or recorded measurements differ')
    return {'path':str(Path(path).resolve()),'sha256':file_digest(path)}

def write_interrupted(path,bank,optimizer,metadata):
    """Atomic recovery evidence, never an inference-ready or completed learned artifact."""
    arrays={}
    if bank is not None:
        arrays['keys']=torch.stack([bank.keys[str(i)].detach().float().cpu() for i in bank.layers]).numpy()
        arrays['values']=torch.stack([bank.values[str(i)].detach().float().cpu() for i in bank.layers]).numpy()
        arrays['attention_layers']=np.asarray(bank.layers,dtype=np.int64)
        if optimizer is not None:
            for index,parameter in enumerate(bank.parameters()):
                for name,value in optimizer.state.get(parameter,{}).items():
                    if torch.is_tensor(value):arrays[f'optimizer_{index}_{name}']=value.detach().cpu().numpy()
        arrays['rng_cpu']=torch.get_rng_state().numpy()
    arrays['metadata']=np.asarray(json.dumps({**metadata,'schema':'character-creation.phantom-training-checkpoint.v1',
        'training_complete':False,'native_flare_qualified':False,'state':'interrupted','bank_initialized':bank is not None,
        'claim':'Partial optimizer state only; never load as a qualified learned graft.'},sort_keys=True,allow_nan=False))
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream:
        np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)



PREFLIGHT_TOLERANCES={'rtol':.02,'atol':.002}


def device_preflight(device,weight_dtype='float16'):
    """Tiny synthetic selected-precision test; no checkpoint weights involved."""
    generator=torch.Generator(device='cpu').manual_seed(1907)
    def sample(shape):return (torch.randn(shape,generator=generator)*.125).half().float()
    cases=[('linear',sample((2,17,32)),sample((48,32))),
           ('depthwise_causal_conv',sample((1,37,64)).transpose(1,2),sample((64,1,4)))]
    metrics=[]
    for name,input_cpu,weight_cpu in cases:
        def operation(x,w):
            return F.linear(x,w) if name=='linear' else F.silu(F.conv1d(x,w,padding=3,groups=64)[:,:,:37])
        reference_input=input_cpu.detach().clone().requires_grad_(True)
        reference_weight=weight_cpu.detach().clone().requires_grad_(True)
        reference=operation(reference_input,reference_weight)
        probe=torch.linspace(-.5,.5,reference.numel(),dtype=torch.float32).reshape(reference.shape)
        reference_grads=torch.autograd.grad((reference*probe).sum(),(reference_input,reference_weight))
        x=input_cpu.to(device=device,dtype=getattr(torch,weight_dtype)).detach().requires_grad_(True)
        w=weight_cpu.to(device=device,dtype=getattr(torch,weight_dtype)).detach().requires_grad_(True)
        observed=operation(x,w)
        grads=torch.autograd.grad((observed.float()*probe.to(device)).sum(),(x,w))
        tensors=[('output',observed,reference),('input_gradient',grads[0],reference_grads[0]),('weight_gradient',grads[1],reference_grads[1])]
        for field,actual,expected in tensors:
            actual=actual.detach().float().cpu();expected=expected.detach()
            finite=bool(torch.isfinite(actual).all())
            passed=finite and torch.allclose(actual,expected,**PREFLIGHT_TOLERANCES)
            metrics.append({'operation':name,'field':field,'finite':finite,'passed':passed,
                            'max_absolute_error':float((actual-expected).abs().max()) if finite else None})
    return {'device':str(device),'dtype':weight_dtype,'reference':'CPU float32 with identically rounded inputs',
            'cudnn_enabled':torch.backends.cudnn.enabled,'tolerances':PREFLIGHT_TOLERANCES,
            'passed':all(row['passed'] for row in metrics),'metrics':metrics}


def hardware_preflight(devices,guard,weight_dtype='float16'):
    report={'passed':False,'backend':'ATen convolution; cuDNN explicitly disabled for SM70 compatibility',
            'torch_version':torch.__version__,'cudnn_enabled':torch.backends.cudnn.enabled,
            'tolerances':PREFLIGHT_TOLERANCES,'devices':[]}
    for device in devices:
        guard()
        try:result=device_preflight(device,weight_dtype)
        except TrainingInterrupted:raise
        except Exception as error:result={'device':str(device),'passed':False,'error':str(error)}
        report['devices'].append(result)
    report['passed']=not report['cudnn_enabled'] and len(report['devices'])==6 and all(row['passed'] for row in report['devices'])
    return report

def run(args):
    if args.output.exists() or args.partial_output.exists():raise ValueError('Output/partial path exists; explicit fresh paths required')
    settings=precision_settings(args.weight_dtype)
    if not math.isfinite(args.max_gpu_gib) or not 0<args.max_gpu_gib<=31:raise ValueError('GPU allocator cap must be at most 31 GiB; no offload')
    data=load_pairs(args.dataset);discovery=[r for r in data['rows'] if r['split']=='discovery'];heldout=[r for r in data['rows'] if r['split']=='heldout']
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
    encoded={r['id']:encode_pair(tokenizer,r,args.max_tokens) for r in data['rows']}
    native=load_native_reference(args.native_reference,data,args.model_identity,tokenizer) if args.native_reference else None
    if args.command=='train' and (native is None or args.validation_report is None):raise ValueError('Train requires --native-reference and --validation-report from successful validate-only run')
    stopped=threading.Event();old_handlers={}
    def interrupt(signum,frame):stopped.set()
    for signum in (signal.SIGTERM,signal.SIGINT):old_handlers[signum]=signal.signal(signum,interrupt)
    cap=args.max_gpu_gib*2**30
    def guard():
        if stopped.is_set():raise TrainingInterrupted('Stop requested at safe functional boundary')
        for device in args.devices:
            if torch.cuda.memory_reserved(device)>cap:raise TrainingInterrupted('Declared GPU allocator budget exceeded')
    bank=optimizer=None;manifest=None;history=[];validation=None;stage='hardware_preflight'
    previous_cudnn=torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled=False
    metadata={**settings,'model_identity':args.model_identity,'dataset_sha256':data['sha256'],'seed':args.seed,'optimizer_steps':0,
              'trainer_sha256':file_digest(__file__),'reader_sha256':file_digest(Path(__file__).with_name('persona_phantom_weights.py')),
              'learning_rate':args.learning_rate,'kl_weight':args.kl_weight,'max_gpu_gib':args.max_gpu_gib,'slots':args.slots,'max_tokens':args.max_tokens}
    try:
        if args.command=='train':
            from persona_phantom_weights import PackedCheckpoint
            metadata['admission_validation']=require_validation(args.validation_report,metadata,PackedCheckpoint(args.checkpoint).manifest,native)
        # Must be launched by the fleet lease supervisor; cap must match its declaration.
        for device in args.devices:
            total=torch.cuda.get_device_properties(device).total_memory
            if cap>=total:raise ValueError('Declared allocator cap must fit each physical GPU')
            torch.cuda.set_per_process_memory_fraction(cap/total,device)
        metadata['hardware_preflight']=hardware_preflight(args.devices,guard,args.weight_dtype)
        if not metadata['hardware_preflight']['passed']:
            result={**metadata,'state':'validation_failed','stage':'hardware_preflight','native_flare_qualified':False}
            with args.output.open('x') as stream:json.dump(result,stream,indent=2)
            print(json.dumps({'report':str(args.output),'state':result['state'],'stage':'hardware_preflight'}));return result
        stage='loading';torch.manual_seed(args.seed)
        model,manifest=load_frozen_model(args.checkpoint,args.devices,guard=guard,weight_dtype=args.weight_dtype)
        device_map={i:next(model.model.layers[i].parameters()).device for i in ATTENTION_LAYERS}
        bank=GraftBank(ATTENTION_LAYERS,4,args.slots,256,devices=device_map,seed=args.seed)
        decoder=FunctionalDecoder(model,bank,guard=guard)
        if args.command=='validate':
            stage='native_forward';native_result=native_forward_check(decoder,native) if native else {'passed':False,'reason':'No native reference supplied'}
            stage='gradient_validation'
            validation=precision_gradient_check(decoder,*encoded[discovery[0]['id']],args.weight_dtype)
            guard();passed=native_result['passed'] and validation['passed']
            result={**metadata,'state':'validation_passed' if passed else 'validation_failed','native_flare_qualified':False,
                    'validation':validation,'native_forward':native_result,'checkpoint':manifest}
            with args.output.open('x') as stream:json.dump(result,stream,indent=2)
            print(json.dumps({'report':str(args.output),'state':result['state']}));return result
        metadata['admission_validation']=require_validation(args.validation_report,metadata,manifest,native)
        optimizer=torch.optim.Adam(bank.parameters(),lr=args.learning_rate);stage='training'
        for step in range(args.steps):
            guard();row=discovery[step%len(discovery)];optimizer.zero_grad(set_to_none=True)
            loss=decoder.loss(*encoded[row['id']],kl_weight=args.kl_weight);loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in bank.parameters()):raise ValueError('Nonfinite or missing training gradient')
            norm=torch.nn.utils.clip_grad_norm_(bank.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            metadata['optimizer_steps']=step+1
            history.append({'step':step+1,'row_id':row['id'],'loss':loss.item(),'gradient_norm':float(norm)})
            print(json.dumps(history[-1]),flush=True);guard()
        # Frozen final candidate only: held-out observations never affect optimizer or selection.
        stage='heldout';evaluation=[]
        with torch.no_grad():
            for row in heldout:
                guard();inputs=encoded[row['id']]
                evaluation.append({'row_id':row['id'],'family_id':row['family_id'],'candidate_target_nll':decoder.loss(*inputs,kl_weight=0).item(),'baseline_target_nll':decoder.loss(*inputs,enabled=False,kl_weight=0).item()})
        guard()
        write_candidate(args.output,bank,{**metadata,'checkpoint':manifest,'native_flare_qualified':False,
           'torch_version':torch.__version__,'transformers_version':__import__('transformers').__version__,
           'checkpoint_binding_scope':'reader metadata/config/index/header bindings; native runtime verification required',
           'gradient_validation':validation,'training_history':history,'heldout':evaluation,'frozen_before_heldout':True,'completed':time.time()})
        result={'artifact':str(args.output),'sha256':file_digest(args.output),'native_flare_qualified':False}
        print(json.dumps(result),flush=True);return result
    except (TrainingInterrupted,torch.cuda.OutOfMemoryError) as error:
        write_interrupted(args.partial_output,bank,optimizer,{**metadata,'stage':stage,'reason':str(error),'checkpoint':manifest,
                         'gradient_validation':validation,'training_history':history,'completed':time.time()})
        result={'state':'interrupted','partial_checkpoint':str(args.partial_output),'training_complete':False}
        print(json.dumps(result),flush=True);return result
    finally:
        torch.backends.cudnn.enabled=previous_cudnn
        for signum,handler in old_handlers.items():signal.signal(signum,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('validate','train'))
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--model-identity',required=True);parser.add_argument('--devices',nargs=6,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--partial-output',type=Path,required=True)
    parser.add_argument('--weight-dtype',choices=('float16','float32'),default='float16')
    parser.add_argument('--native-reference',type=Path);parser.add_argument('--validation-report',type=Path)
    parser.add_argument('--max-gpu-gib',type=float,required=True,help='Per-device allocator cap from fleet admission; not an admission bypass')
    parser.add_argument('--steps',type=int,default=10);parser.add_argument('--slots',type=int,default=64);parser.add_argument('--max-tokens',type=int,default=512)
    parser.add_argument('--seed',type=int,default=7035);parser.add_argument('--learning-rate',type=float,default=1e-3);parser.add_argument('--kl-weight',type=float,default=.1)
    args=parser.parse_args()
    if (not valid_sha(args.model_identity) or len(set(args.devices))!=6 or any(not d.startswith('cuda:') or not d[5:].isdigit() for d in args.devices)
            or not 0<=args.seed<2**32 or not 1<=args.steps<=1000 or not 1<=args.slots<=128 or not 16<=args.max_tokens<=2048
            or not math.isfinite(args.learning_rate) or not 0<args.learning_rate<=.01 or not 0<=args.kl_weight<=1
            or not math.isfinite(args.max_gpu_gib) or not 0<args.max_gpu_gib<=31 or args.output.resolve()==args.partial_output.resolve()):parser.error('Invalid bounded training settings')
    result=run(args)
    if result.get('state')=='interrupted':raise SystemExit(75)
    if result.get('state')=='validation_failed':raise SystemExit(2)


if __name__=='__main__':main()
