"""Checkpoint-bound static neuron lenses and adjacent-layer read/write affinities.

Independent implementation inspired by Lyra's neuron-explorer. Static projection
and weight affinity are hypotheses, not observed token effects or causal paths.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict, Counter
import hashlib
import json
import os
import signal
import re
from pathlib import Path
import threading
import zipfile
import time
import numpy as np
from persona_map import Checkpoint, file_sha256, e4m3, decode_columns
from persona_vectors import ModelSpec, canonical_digest

SCHEMA = 'character-creation.weight-atlas.v1'
SEMANTICS = {
    'lens': 'FP32 lm_head @ RMSNorm(+/- W_down[:,n]), using checkpoint NVFP4 weights and zero-centered final norm. Static logits, not probabilities or observed generation effects.',
    'web': 'Signed cosine between a source down column and next-layer gate/up rows. Each pair uses the channel with larger absolute cosine. Attention, normalization, gating state and other paths are omitted; affinity is not causal transmission.',
    'layout': 'Node depth is model layer; within-layer placement is a deterministic projection of output directions. This navigable 3D embedding is not literal physical geometry. Curves route the measured affinities.',
    'coverage': 'Every MLP neuron; top/bottom 24 vocabulary projections and strongest 16 adjacent-layer affinities in each direction. Omitted edges are not zero.'}


def norm_direction(value, weight, eps):
    return value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + eps) * (1 + weight)


def cosine_channels(down, gate, up):
    """CPU reference: rows are source output vectors and destination read vectors."""
    def unit(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)
    a, b = unit(down) @ unit(gate).T, unit(down) @ unit(up).T
    choose = np.abs(b) > np.abs(a)
    return np.where(choose, b, a), choose.astype(np.uint8)


def decode_gpu(checkpoint, prefix, device):
    """Decode in bounded row blocks, avoiding multi-GB temporary CPU tensors."""
    import torch
    packed = checkpoint.raw(prefix + '.weight', 'U8')
    scales = checkpoint.raw(prefix + '.weight_scale', 'F8_E4M3')
    gain = checkpoint.raw(prefix + '.weight_scale_2', 'F32')
    if gain.size != 1 or not np.isfinite(gain).all() or float(gain.flat[0]) <= 0:
        raise ValueError('Invalid global NVFP4 scale')
    if scales.shape != (packed.shape[0], packed.shape[1] // 8): raise ValueError('Invalid NVFP4 group shape')
    output = torch.empty((packed.shape[0], packed.shape[1]*2), dtype=torch.float32, device=device)
    magnitudes = torch.tensor([0,.5,1,1.5,2,3,4,6], device=device)
    for start in range(0, len(packed), 1024):
        raw = torch.from_numpy(np.array(packed[start:start+1024])).to(device)
        code = torch.stack((raw & 15, raw >> 4), dim=-1).reshape(len(raw), -1).long()
        scale = torch.from_numpy(e4m3(scales[start:start+1024])).to(device).repeat_interleave(16, dim=1)
        if bool((scale < 0).any()): raise ValueError('Negative block scale')
        output[start:start+len(raw)] = magnitudes[code & 7] * torch.where((code & 8) != 0, -1., 1.) * scale * float(gain.flat[0])
    if not bool(torch.isfinite(output).all()): raise ValueError('Nonfinite decoded matrix')
    return output


def build(args):
    import torch
    from transformers import AutoTokenizer
    from persona_capture import require_lease
    config = json.loads(args.config.read_text()); model = ModelSpec.load(args.root / 'data/persona_model.json')
    lease = require_lease(config, args.shard)
    from persona_fleet import telemetry
    def monitor():
        while True:
            try:
                sample=telemetry(config['gpus'][args.shard])
                if sample['temperature']>=88 or sample['memory_mib']>20*1024 or sample['available_ram_bytes']<config['ram_floor_bytes']:
                    raise RuntimeError('Atlas thermal/VRAM/RAM guard')
            except Exception as error:
                print(str(error),flush=True);os.kill(os.getpid(),signal.SIGTERM);return
            time.sleep(30)
    threading.Thread(target=monitor,daemon=True).start()
    torch.set_num_threads(len(os.sched_getaffinity(0)))
    torch.backends.cuda.matmul.allow_tf32 = False
    cp = Checkpoint(args.checkpoint)
    raw_config = json.loads((args.checkpoint/'config.json').read_text()); text = raw_config['text_config']
    if (text['num_hidden_layers'],text['hidden_size'],text['intermediate_size']) != (model.layers,model.hidden,model.intermediate):
        raise ValueError('Checkpoint dimensions differ from pinned model')
    if raw_config['model_type'] != 'qwen3_5': raise ValueError('Atlas requires verified Qwen3.5 zero-centered norm architecture')
    binding = {name:file_sha256(args.checkpoint/name) for name in ('config.json','model.safetensors.index.json','tokenizer.json')}
    # Verify the exact immutable shard bytes, not merely their filenames/config.
    weights = {name:file_sha256(args.checkpoint/name) for name in sorted(set(cp.index.values()))}
    if len(weights)==1 and next(iter(weights.values()))!=config['checkpoint_weights_sha256']:
        raise ValueError('Checkpoint shard differs from verified model weights')
    reference=json.loads((args.root/'results'/f'capture-shard{args.shard}.json').read_text())
    for name,key in [('config.json','checkpoint_config_sha256'),('model.safetensors.index.json','checkpoint_index_sha256'),('tokenizer.json','tokenizer_sha256')]:
        if binding[name]!=reference[key]:raise ValueError('Checkpoint differs from calibration')
    source = {'schema':SCHEMA,'model_identity':model.identity,'model':model.__dict__, 'checkpoint':binding,
        'weights':weights,'script_sha256':file_sha256(Path(__file__)), 'semantics':SEMANTICS,
        'precision':'FP32 decoded-weight reference; not native quantized activation arithmetic', 'top_tokens':24,'top_edges':16}
    source_id = canonical_digest(source)
    out = args.root/'results'
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    if args.shard == 0:
        tokens = [tokenizer.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in range(text['vocab_size'])]
        (out/'persona_atlas_tokens.json').write_text(json.dumps({'source_id':source_id,'tokens':tokens}, ensure_ascii=False))
    head = decode_gpu(cp, 'lm_head', 'cuda')
    raw_norm = cp.raw('model.language_model.norm.weight', 'BF16')
    norm = torch.from_numpy((np.asarray(raw_norm,dtype=np.uint32)<<16).view(np.float32).copy()).cuda()
    if norm.shape != (model.hidden,): raise ValueError('Final norm dimension differs')
    eps = float(text['rms_norm_eps'])
    # Fixed projection makes layout comparable across layers without claiming
    # a faithful 2D embedding of a 5120-dimensional output vector.
    projection = torch.from_numpy(np.random.default_rng(7035).normal(size=(model.hidden,2)).astype(np.float32)).cuda()
    for layer in range(args.shard, model.layers, 6):
        memory = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(memory['MemAvailable'].split()[0])*1024 < config['ram_floor_bytes']: raise RuntimeError('System RAM guard')
        down = decode_gpu(cp, f'model.language_model.layers.{layer}.mlp.down_proj', 'cuda').T.contiguous()
        if down.shape != (model.intermediate,model.hidden): raise ValueError('Down projection dimensions differ')
        check_indices=np.array([0,model.intermediate//2,model.intermediate-1])
        packed,scales,gain=cp.down(layer)
        np.testing.assert_allclose(down[check_indices].cpu().numpy().T,decode_columns(packed,scales,gain,check_indices),rtol=2e-6,atol=1e-8)
        lens_ids, lens_scores, anti_ids, anti_scores = [],[],[],[]
        for start in range(0,len(down),128):
            values = down[start:start+128]
            normalized = values*torch.rsqrt(values.square().mean(-1,keepdim=True)+eps)*(1+norm)
            logits = normalized @ head.T
            pos = torch.topk(logits,24,dim=1); neg = torch.topk(-logits,24,dim=1)
            lens_ids.append(pos.indices.cpu().numpy().astype(np.int32));lens_scores.append(pos.values.cpu().numpy())
            anti_ids.append(neg.indices.cpu().numpy().astype(np.int32));anti_scores.append(neg.values.cpu().numpy())
        unit = torch.nn.functional.normalize(down,dim=1)
        arrays = {'positive_ids':np.concatenate(lens_ids),'positive_scores':np.concatenate(lens_scores),
            'negative_ids':np.concatenate(anti_ids),'negative_scores':np.concatenate(anti_scores),
            'positions':(unit @ projection).cpu().numpy()}
        if layer+1 < model.layers:
            gate = torch.nn.functional.normalize(decode_gpu(cp,f'model.language_model.layers.{layer+1}.mlp.gate_proj','cuda'),dim=1)
            up = torch.nn.functional.normalize(decode_gpu(cp,f'model.language_model.layers.{layer+1}.mlp.up_proj','cuda'),dim=1)
            # Keep the full adjacent affinity matrix only on this GPU (~1.2 GB).
            scores = torch.empty((model.intermediate,model.intermediate),device='cuda')
            channels = torch.empty_like(scores,dtype=torch.uint8)
            for start in range(0,len(unit),128):
                a,b = unit[start:start+128] @ gate.T, unit[start:start+128] @ up.T
                choose = b.abs()>a.abs();scores[start:start+128]=torch.where(choose,b,a);channels[start:start+128]=choose
            for direction, value, channel in [('out',scores,channels),('in',scores.T,channels.T)]:
                indices = value.abs().topk(16,dim=1).indices
                arrays[direction+'_ids']=indices.cpu().numpy().astype(np.int32)
                arrays[direction+'_scores']=torch.gather(value,1,indices).cpu().numpy()
                arrays[direction+'_channels']=torch.gather(channel,1,indices).cpu().numpy()
            del gate,up,scores,channels,a,b,choose,value,channel
        metadata = {**source,'source_id':source_id,'layer':layer,'lease_id':lease['id'],'gpu_uuid':config['gpus'][args.shard], 'created':time.time()}
        path = out/f'persona_atlas_L{layer}.npz'
        with path.open('xb') as stream: np.savez_compressed(stream, metadata=np.array(json.dumps(metadata)), **arrays)
        print(json.dumps({'layer':layer,'sha256':file_sha256(path),'peak_cuda_bytes':torch.cuda.max_memory_allocated()}),flush=True)
        del down,unit,arrays
    print('ATLAS_SHARD_COMPLETE',args.shard,flush=True)


class WeightAtlas:
    def __init__(self, root, model, *, require_manifest=True):
        self.root=Path(root);self.model=model;self.lock=threading.RLock();self.cache=OrderedDict();self.tokens=None;self.source_id=None;self.require_manifest=require_manifest;self.manifest=None

    def publication(self):
        if self.manifest is None:
            value=json.loads((self.root/'results/persona_atlas_manifest.json').read_text())
            if value.get('model_identity')!=self.model.identity or value.get('layers')!=self.model.layers or not isinstance(value.get('files'),dict):
                raise ValueError('Atlas publication model or coverage differs')
            self.manifest=value
        return self.manifest

    def layer(self, layer):
        if type(layer) is not int or not 0<=layer<self.model.layers: raise ValueError('Invalid layer')
        path=self.root/'results'/f'persona_atlas_L{layer}.npz'
        signature=(path.stat().st_mtime_ns,path.stat().st_size)
        with self.lock:
            cached=self.cache.get(layer)
            if cached and cached[0]==signature:
                self.cache.move_to_end(layer);return cached[1]
            if self.require_manifest:
                manifest=self.publication()
                if file_sha256(path)!=manifest['files'].get(path.name):raise ValueError('Atlas artifact hash differs from publication')
            with zipfile.ZipFile(path) as archive:
                if len(archive.infolist())>12 or sum(i.file_size for i in archive.infolist())>100<<20:raise ValueError('Atlas archive exceeds bounds')
            with np.load(path,allow_pickle=False) as z:
                metadata=json.loads(str(z['metadata']))
                if (metadata['schema']!=SCHEMA or metadata['model_identity']!=self.model.identity or metadata['layer']!=layer):
                    raise ValueError('Atlas model/layer binding differs')
                source={k:v for k,v in metadata.items() if k not in ('source_id','layer','lease_id','gpu_uuid','created')}
                if canonical_digest(source)!=metadata['source_id']:raise ValueError('Atlas source binding differs')
                if self.source_id and self.source_id!=metadata['source_id']:raise ValueError('Mixed atlas checkpoints/builds')
                if self.require_manifest and metadata['source_id']!=manifest['source_id']:raise ValueError('Atlas publication source differs')
                arrays={k:z[k] for k in z.files if k!='metadata'}
            for name,array in arrays.items():
                width=2 if name=='positions' else 24 if name.startswith(('positive','negative')) else 16
                if array.shape!=(self.model.intermediate,width) or array.dtype.kind not in 'ifu' or not np.isfinite(array).all():raise ValueError('Invalid atlas array')
                if name.endswith('_ids'):
                    bound=248320 if name.startswith(('positive','negative')) else self.model.intermediate
                    if array.dtype.kind not in 'iu' or np.any(array<0) or np.any(array>=bound):raise ValueError('Invalid atlas indices')
                if name.endswith('_channels') and (array.dtype.kind not in 'iu' or np.any(array>1)):raise ValueError('Invalid affinity channel')
                if name in ('in_scores','out_scores') and np.any(np.abs(array)>1.00001):raise ValueError('Affinity exceeds cosine range')
                array.setflags(write=False)
            required={'positions','positive_ids','positive_scores','negative_ids','negative_scores'}
            if layer+1<self.model.layers:required|={d+'_'+v for d in ('in','out') for v in ('ids','scores','channels')}
            if set(arrays)!=required:raise ValueError('Atlas fields differ')
            self.source_id=metadata['source_id'];value={'metadata':metadata,**arrays}
            self.cache[layer]=(signature,value)
            if len(self.cache)>8:self.cache.popitem(last=False)
            return value

    def vocabulary(self):
        with self.lock:
            if self.tokens is None:
                if self.source_id is None:self.layer(0)
                path=self.root/'results/persona_atlas_tokens.json'
                if self.require_manifest and file_sha256(path)!=self.publication()['files'].get(path.name):raise ValueError('Tokenizer artifact hash differs')
                value=json.loads(path.read_text())
                if value['source_id']!=self.source_id:raise ValueError('Tokenizer and atlas source differ')
                if not isinstance(value['tokens'],list) or not 1<=len(value['tokens'])<=300000 or any(not isinstance(t,str) for t in value['tokens']):raise ValueError('Invalid token table')
                self.tokens=value['tokens']
            return self.tokens

    def query(self,layer,neuron):
        if type(neuron) is not int or not 0<=neuron<self.model.intermediate:raise ValueError('Invalid neuron')
        try:
            data=self.layer(layer);tokens=self.vocabulary()
            result={'available':True,'target':{'layer':layer,'neuron':neuron},'source':data['metadata']}
            for direction in ('positive','negative'):
                result[direction]=[{'id':int(i),'text':tokens[i],'logit':float(s)} for i,s in zip(data[direction+'_ids'][neuron],data[direction+'_scores'][neuron])]
            return result
        except FileNotFoundError:
            return {'available':False,'status':'missing_source','message':'This layer has no published checkpoint atlas yet.'}

    def web(self,layer,neuron,*,hops=2,fanout=4,direction='both',threshold=0.):
        if type(neuron) is not int or not 0<=neuron<self.model.intermediate or type(hops) is not int or not 1<=hops<=3 or type(fanout) is not int or not 1<=fanout<=8 or direction not in ('in','out','both') or not 0<=threshold<=1:raise ValueError('Invalid web controls')
        nodes={};edges={};missing=set();frontier=[(layer,neuron)]
        def add(l,n):
            data=self.layer(l);p=data['positions'][n]
            nodes[(l,n)]={'id':f'L{l}N{n}','layer':l,'neuron':n,'position':[float(p[0]),float(p[1])], 'position3d':[float(p[0]),float(p[1]),float(l)], 'selected':(l,n)==(layer,neuron)}
        try:add(layer,neuron)
        except FileNotFoundError:return {'available':False,'status':'missing_source','message':'No atlas for this layer.'}
        for depth in range(hops):
            following=[]
            for l,n in frontier:
                for d in ('in','out') if direction=='both' else (direction,):
                    source_layer=l if d=='out' else l-1
                    if not 0<=source_layer<self.model.layers-1:continue
                    try:data=self.layer(source_layer)
                    except FileNotFoundError:missing.add(source_layer);continue
                    for other,score,ch in zip(data[d+'_ids'][n][:fanout],data[d+'_scores'][n][:fanout],data[d+'_channels'][n][:fanout]):
                        other=int(other);score=float(score)
                        if score==0 or abs(score)<threshold:continue
                        target=(l+1 if d=='out' else l-1,other)
                        if target not in nodes:
                            if len(nodes)>=64:continue
                            try:add(*target)
                            except FileNotFoundError:missing.add(target[0]);continue
                            following.append(target)
                        a,b=((l,n),target) if d=='out' else (target,(l,n))
                        edges[(a,b)]={'source':nodes[a]['id'],'target':nodes[b]['id'],'score':score,'channel':'up' if ch else 'gate'}
            frontier=following
        # A volumetric alternative to layer planes: spectral coordinates of
        # the retained unsigned affinity graph. Edge signs remain displayed.
        ordered=list(nodes.values());lookup={node['id']:i for i,node in enumerate(ordered)}
        adjacency=np.zeros((len(ordered),len(ordered)),dtype=np.float64)
        for edge in edges.values():
            a,b=lookup[edge['source']],lookup[edge['target']]
            adjacency[a,b]=adjacency[b,a]=abs(edge['score'])
        degree=adjacency.sum(axis=1);scale=np.divide(1,np.sqrt(degree),out=np.zeros_like(degree),where=degree>0)
        laplacian=np.eye(len(ordered))-scale[:,None]*adjacency*scale[None,:]
        eigenvalues,eigenvectors=np.linalg.eigh(laplacian)
        topology=eigenvectors[:,1:4]
        if topology.shape[1]<3:topology=np.pad(topology,((0,0),(0,3-topology.shape[1])))
        for axis in range(3):
            if topology[np.argmax(abs(topology[:,axis])),axis]<0:topology[:,axis]*=-1
        for node,position in zip(ordered,topology):node['topology3d']=position.tolist()
        pooled={}
        tokens=self.vocabulary()
        for sign in ('positive','negative'):
            counts=Counter(int(i) for l,n in nodes for i in self.layer(l)[sign+'_ids'][n])
            pooled[sign]=[{'id':i,'text':tokens[i],'neurons':count} for i,count in counts.most_common(12)]
        return {'available':True,'source_id':self.source_id,'semantics':SEMANTICS,'missing_layers':sorted(missing),'pooled_tokens':pooled,'topology_semantics':'Three nontrivial normalized-Laplacian coordinates of the retained absolute-weight graph. Signs remain on edges. This is a sparse graph embedding, not physical geometry or a full-weight causal map.',
            'graph':{'nodes':list(nodes.values()),'edges':list(edges.values()),'directed':True,'layout':SEMANTICS['layout']}}

    def search(self,text,limit=40,word_start=False):
        if not isinstance(text,str) or not 1<=len(text)<=100:raise ValueError('Search needs 1–100 characters')
        matches=[];missing=[];token_ids=None
        for l in range(self.model.layers):
            try:data=self.layer(l);tokens=self.vocabulary()
            except FileNotFoundError:missing.append(l);continue
            if token_ids is None:
                pattern = re.compile(r'(?<!\w)' + re.escape(text.casefold())) if word_start else None
                token_ids=np.array([i for i,t in enumerate(tokens) if (pattern.search(t.casefold()) if pattern else text.casefold() in t.casefold())],dtype=np.int32)
            for direction in ('positive','negative'):
                rows,cols=np.where(np.isin(data[direction+'_ids'],token_ids))
                strongest=np.argsort(-data[direction+'_scores'][rows,cols],kind='stable')[:limit]
                for n,c in zip(rows[strongest],cols[strongest]):
                    i=int(data[direction+'_ids'][n,c]);matches.append({'layer':l,'neuron':int(n),'direction':direction,'token':tokens[i],'logit':float(data[direction+'_scores'][n,c])})
            matches=sorted(matches,key=lambda x:(-x['logit'],x['layer'],x['neuron']))[:limit]
        return {'matches':matches,'missing_layers':missing,'scope':'Stored top 24 tokens per sign, not an exhaustive vocabulary ranking.'}


def publish_geometry(root):
    """PCA of already calibrated output columns; no new model execution."""
    model=ModelSpec.load(root/'data/persona_model.json')
    chart_path=root/'results/persona_map.json'
    chart=json.loads(chart_path.read_text())
    if chart.get('model_identity')!=model.identity:raise ValueError('Map model differs')
    positions={};explained={};sources={}
    for layer in range(model.layers):
        path=root/'results'/f'layer_{layer:02d}.npz'
        if not path.exists():continue
        with np.load(path,allow_pickle=False) as z:
            if str(z['model_identity'])!=model.identity:raise ValueError('Calibration model differs')
            ids=z['neuron_ids'];vectors=z['down_columns'].T.astype(np.float64)
        if vectors.shape!=(len(ids),model.hidden) or not np.isfinite(vectors).all():raise ValueError('Invalid calibrated output columns')
        norms=np.linalg.norm(vectors,axis=1);valid=norms>0
        ids=ids[valid];vectors=vectors[valid]
        if not len(ids):continue
        vectors/=norms[valid,None];vectors-=vectors.mean(axis=0,keepdims=True)
        eig,vec=np.linalg.eigh(vectors@vectors.T);order=np.argsort(eig)[::-1][:2]
        xy=vec[:,order]*np.sqrt(np.maximum(eig[order],0))
        if xy.shape[1]==1:xy=np.pad(xy,((0,0),(0,1)))
        for axis in range(2):
            if xy[np.argmax(abs(xy[:,axis])),axis]<0:xy[:,axis]*=-1
        xy/=max(float(abs(xy).max()),1e-12)
        positions.update({f'L{layer}N{int(n)}':v.tolist() for n,v in zip(ids,xy)})
        explained[str(layer)]=float(np.maximum(eig[order],0).sum()/max(np.maximum(eig,0).sum(),1e-12))
        sources[path.name]=file_sha256(path)
    geometry={'model_identity':model.identity,'map_sha256':file_sha256(chart_path),'positions':positions,
        'explained_fraction':explained,'source_sha256':sources,
        'semantics':'Per-layer PCA of normalized decoded output columns. Local axes differ between layers; proximity is a 2D weight-direction approximation, not wiring.'}
    (root/'results/persona_map_geometry.json').write_text(json.dumps(geometry))
    return {'mapped_positions':len(positions),'layers':len(explained)}


def publish(root):
    model=ModelSpec.load(root/'data/persona_model.json')
    atlas=WeightAtlas(root,model,require_manifest=False);files={}
    for layer in range(model.layers):
        atlas.layer(layer)
        name=f'persona_atlas_L{layer}.npz';files[name]=file_sha256(root/'results'/name)
    atlas.vocabulary()
    name='persona_atlas_tokens.json';files[name]=file_sha256(root/'results'/name)
    manifest={'schema':'character-creation.weight-atlas-publication.v1','model_identity':model.identity,'source_id':atlas.source_id,
        'layers':model.layers,'neurons':model.layers*model.intermediate,'files':files}
    (root/'results/persona_atlas_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--config',type=Path);parser.add_argument('--shard',type=int,choices=range(6))
    parser.add_argument('--publish',action='store_true',help='Validate all shards and publish a hash-bound reader manifest')
    parser.add_argument('--geometry',action='store_true',help='Build main-map PCA from existing calibrated output columns')
    args=parser.parse_args()
    if args.geometry:print(json.dumps(publish_geometry(args.root)))
    elif args.publish:print(json.dumps(publish(args.root)))
    elif args.config and args.checkpoint and args.shard is not None:build(args)
    else:parser.error('Build requires config, checkpoint and shard; publication requires --publish')

if __name__=='__main__':main()
