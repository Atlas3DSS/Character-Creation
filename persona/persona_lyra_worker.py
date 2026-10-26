"""Bounded Qwen3.5-4B corpus observations and causal checks for Lyra's gift.

Independent HF implementation; no physical wiring or exhaustive causal map claim.
Run only inside the existing workboard lease supervisor.
"""
from __future__ import annotations
import argparse,hashlib,json,os,signal,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

MODEL_REPO='Qwen/Qwen3.5-4B'
REVISION='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
REFERENCE_NEURONS={(14,8949),(13,7956),(15,3920),(12,2920),(16,8168),(10,6295),(11,6261),(17,7870),(18,5796)}
IDENTITY=hashlib.sha256((MODEL_REPO+'@'+REVISION).encode()).hexdigest()

def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def atomic(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,sort_keys=True));os.replace(tmp,path)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--shard',type=int,required=True);ap.add_argument('--port',type=int,required=True);a=ap.parse_args()
 spec=json.loads(a.manifest.read_text());root=a.manifest.parent;job=spec['job_id'];prefix=root/f'lyra_{job}';status_path=Path(str(prefix)+f'_status_{a.shard}.json')
 if not 0<=a.shard<6 or spec['model_identity']!=IDENTITY or spec['revision']!=REVISION:raise ValueError('Invalid fixed-model job')
 if digest(__file__)!=spec['worker_sha256']:raise ValueError('Worker source changed')
 checkpoint=Path(spec['checkpoint'])
 for name,sha in spec['checkpoint_files'].items():
  if '/' in name or digest(checkpoint/name)!=sha:raise ValueError('Checkpoint file mismatch: '+name)
 if digest(spec['corpus'])!=spec['corpus_sha256']:raise ValueError('Corpus snapshot changed')
 rows=[json.loads(line) for line in Path(spec['corpus']).read_text().splitlines() if line.strip()]
 if len(rows)!=spec['records'] or any(r['synthetic'] is not True for r in rows):raise ValueError('Invalid corpus')
 stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
 def check():
  if stop.is_set() or time.time()>=spec['deadline']:raise InterruptedError('Bounded gift window ended')
 status={'job_id':job,'model_identity':IDENTITY,'worker_sha256':spec['worker_sha256'],'shard':a.shard,'phase':'loading','observations':0,'causal_checks':0,'ready':False}
 lock=threading.Lock()
 def update(**values):
  with lock:status.update(values,time=time.time());atomic(status_path,status)
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*_):pass
  def do_GET(self):
   if self.path!='/gift/status':self.send_error(404);return
   with lock:raw=json.dumps(status).encode()
   self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
 server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);threading.Thread(target=server.serve_forever,daemon=True).start();update()
 try:
  import numpy as np
  import torch
  from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
  torch.set_num_threads(len(os.sched_getaffinity(0)));torch.manual_seed(spec['seed'])
  model=Qwen3_5ForConditionalGeneration.from_pretrained(checkpoint,local_files_only=True,dtype=torch.float16,attn_implementation='eager').eval().to('cuda')
  lm=model.model.language_model
  if (len(lm.layers),model.config.text_config.intermediate_size)!=(32,9216):raise ValueError('Unexpected 4B architecture')
  tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True);captured={};intervention=None
  def hook(layer):
   def apply(mod,args):
    value=args[0]
    if intervention and intervention[0]==layer:
     value=value.clone();value[...,intervention[1]]*=intervention[2]
    captured[layer]=value[0,-1].detach().float()
    return (value,*args[1:])
   return apply
  handles=[layer.mlp.down_proj.register_forward_pre_hook(hook(i)) for i,layer in enumerate(lm.layers)]
  def observe(text,action=None):
   nonlocal intervention
   check();intervention=action;captured.clear()
   encoded=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
   if hasattr(encoded,'keys'):encoded=encoded['input_ids']
   original=len(encoded);encoded=encoded[:spec['max_prompt_tokens']]
   ids=torch.tensor([encoded],device='cuda')
   with torch.inference_mode():
    result=lm(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True)
    logits=model.lm_head(result.last_hidden_state[:,-1]).float()[0]
   values=torch.stack([captured[i] for i in range(32)])
   if not torch.isfinite(values).all() or not torch.isfinite(logits).all():raise ValueError('Nonfinite observations')
   return values,logits,{'input_tokens':original,'observed_tokens':len(encoded),'truncated':original>len(encoded),'position':len(encoded)-1}
  # Neutral multiplicative gain is an exact no-op; checks actual installed GPU backend.
  text='Compare two explanations for a failed experiment and suggest a reversible test.'
  v0,l0,_=observe(text);v1,l1,_=observe(text,(14,8949,1.0))
  if not torch.equal(v0,v1) or not torch.equal(l0,l1):raise ValueError('Neutral intervention changed observations')
  proof={'zero_equal':True,'gpu':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),
   'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,'dtype':'float16','attention':'eager','worker_sha256':spec['worker_sha256'],'checkpoint_files':spec['checkpoint_files']}
  atomic(Path(str(prefix)+f'_qualification_{a.shard}.json'),proof)
  update(ready=True,phase='observing',qualification=proof)
  selected=list(range(a.shard,len(rows),6));array_path=Path(str(prefix)+f'_activations_{a.shard}.npy');ids_path=Path(str(prefix)+f'_rows_{a.shard}.jsonl')
  if array_path.exists() or ids_path.exists():raise ValueError('Refusing to overwrite an existing attempt')
  matrix=np.lib.format.open_memmap(array_path,mode='w+',dtype=np.float16,shape=(len(selected),32,9216))
  with ids_path.open('x') as audit:
   for j,index in enumerate(selected):
    values,_,meta=observe(rows[index]['prompt']);matrix[j]=values.cpu().numpy().astype(np.float16)
    audit.write(json.dumps({'row':j,'global_index':index,'id':rows[index]['id'],'split':rows[index]['split'],**meta})+'\n')
    if j%16==0:matrix.flush();audit.flush();update(observations=j+1)
   matrix.flush();audit.flush();os.fsync(audit.fileno())
  del matrix;update(observations=len(selected),phase='capture_complete')
  # Shared filesystem barrier, always bounded by the absolute deadline.
  while True:
   check();states=[]
   for shard in range(6):
    path=Path(str(prefix)+f'_status_{shard}.json')
    if path.exists():states.append(json.loads(path.read_text()))
   if any(s.get('phase')=='failed' for s in states):raise RuntimeError('Peer failed; no derived atlas published')
   if len(states)==6 and all(s.get('job_id')==job and s.get('phase') in ('capture_complete','correlations','causal_checks','complete') for s in states):break
   stop.wait(2)
  update(phase='correlations')
  arrays=[np.load(Path(str(prefix)+f'_activations_{shard}.npy'),mmap_mode='r') for shard in range(6)]
  order=[i for shard in range(6) for i in range(shard,len(rows),6)]
  discover=np.array([rows[i]['split']=='discovery' for i in order]);replicate=np.array([rows[i]['split']=='replication' for i in order]);held=np.flatnonzero([rows[i]['split']=='heldout' for i in order])
  if discover.sum()<100 or replicate.sum()<20 or len(held)<20:raise ValueError('Insufficient independent split observations')
  def layer_values(layer):return np.concatenate([array[:,layer,:] for array in arrays]).astype(np.float32)
  def standardized(value,mask):
   t=torch.tensor(value[mask],device='cuda');t-=t.mean(0);norm=torch.linalg.vector_norm(t,dim=0);valid=torch.isfinite(norm)&(norm>1e-7)
   return t/torch.where(valid,norm,torch.ones_like(norm)),valid
  candidates=[];link_paths=[]
  for layer in range(a.shard,32,6):
   check();raw=layer_values(layer);x,valid=standardized(raw,discover)
   for target_layer in (layer,layer+1):
    if target_layer>=32:continue
    other=raw if target_layer==layer else layer_values(target_layer);y,target_valid=standardized(other,discover)
    top_ids=[];top_values=[]
    for offset in range(0,9216,256):
     check();corr=x[:,offset:offset+256].T@y
     supported=valid[offset:offset+256,None]&target_valid[None,:]
     if target_layer==layer:
      rr=torch.arange(corr.shape[0],device='cuda');supported[rr,rr+offset]=False
     rank=corr.abs().masked_fill(~supported,-1);score,index=rank.topk(24,dim=1);values=corr.gather(1,index)
     ids=index.cpu().numpy().astype(np.int32);vals=values.cpu().numpy();ids[score.cpu().numpy()<0]=-1;vals[ids<0]=np.nan
     top_ids.append(ids);top_values.append(vals)
    ids=np.concatenate(top_ids);vals=np.concatenate(top_values)
    # Validate only selected discovery links on an untouched, fixed prompt split.
    hx,hvalid=standardized(raw,replicate);hy,hyvalid=standardized(other,replicate);held_values=np.full(vals.shape,np.nan,np.float32)
    for column in range(24):
     safe=np.maximum(ids[:,column],0);r=(hx*hy[:,torch.tensor(safe,device='cuda')]).sum(0).cpu().numpy()
     supported=(ids[:,column]>=0)&hvalid.cpu().numpy()&hyvalid.cpu().numpy()[safe];held_values[supported,column]=r[supported]
    link_path=Path(str(prefix)+f'_links_L{layer}_L{target_layer}.npz');link_paths.append(link_path)
    np.savez_compressed(link_path,ids=ids,r=vals,heldout_r=held_values,
      discovery_n=int(discover.sum()),heldout_n=int(replicate.sum()),source_layer=layer,target_layer=target_layer,
      source_valid=valid.cpu().numpy(),target_valid=target_valid.cpu().numpy())
    if target_layer==layer+1:
     top=np.argsort(-np.nan_to_num(abs(vals[:,0]),nan=-1),kind='stable')[:12].tolist()
     flat=list(dict.fromkeys([n for l,n in sorted(REFERENCE_NEURONS) if l==layer]+top))
     candidates.extend((layer,int(n),target_layer,int(ids[n,0]),float(vals[n,0]),float(held_values[n,0])) for n in flat if ids[n,0]>=0 and np.isfinite(held_values[n,0]))
    del y,hx,hy
   del x
  # John ranks only existing candidate IDs. Model text is never executable.
  mandatory=[i for i,c in enumerate(candidates) if (c[0],c[1]) in REFERENCE_NEURONS]
  chosen=list(dict.fromkeys(mandatory+list(range(len(candidates)))))[:24];planner_record={'method':'fixed_strongest_fallback','selected':chosen}
  try:
   import urllib.request
   payload={'model':'persona-manager','messages':[{'role':'system','content':'Rank candidate neuron-link experiments. Return JSON only with selected: up to24 distinct integer candidate indices. Use discovery evidence and varied source layers. Evaluation evidence is withheld. No commands.'},{'role':'user','content':json.dumps({'candidates':[list(c[:5]) for c in candidates]})}], 'temperature':0,'max_tokens':512,'response_format':{'type':'json_object'}}
   request=urllib.request.Request('http://127.0.0.1:18511/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
   response=json.load(urllib.request.urlopen(request,timeout=min(60,max(1,spec['deadline']-time.time()))));selection=json.loads(response['choices'][0]['message']['content'])['selected']
   if not isinstance(selection,list) or not 1<=len(selection)<=24 or any(type(i)!=int or not 0<=i<len(candidates) for i in selection) or len(set(selection))!=len(selection):raise ValueError('Invalid bounded choice')
   chosen=list(dict.fromkeys(mandatory+selection))[:24];planner_record={'method':'john_ranked_fixed_candidates','selected':chosen,'mandatory_reference_indices':mandatory,'request':payload,'response':response}
  except Exception as error:planner_record['error']=str(error)
  atomic(Path(str(prefix)+f'_planner_{a.shard}.json'),planner_record);update(phase='causal_checks')
  rng=np.random.default_rng(spec['seed']+a.shard);checks=0
  with Path(str(prefix)+f'_causal_{a.shard}.jsonl').open('x') as stream:
   for index in chosen:
    layer,neuron,target_layer,target,r,held_r=candidates[index]
    for prompt_index in [order[i] for i in held[:8]]:
     check();baseline,base_logits,meta=observe(rows[prompt_index]['prompt']);base_logp=base_logits.log_softmax(-1)
     random_neuron=int(rng.integers(9215));random_neuron+=random_neuron>=neuron
     for control,source in [('candidate',neuron),('random',random_neuron)]:
      changed,logits,_=observe(rows[prompt_index]['prompt'],(layer,source,0.0));logp=logits.log_softmax(-1)
      record={'source_layer':layer,'source_neuron':source,'target_layer':target_layer,'target_neuron':target,'control':control,'mode':'mul','gain':0,
       'prompt_id':rows[prompt_index]['id'],'split':'heldout','control_matching':'random same-layer neuron, not amplitude matched','discovery_r':r,'heldout_r':held_r,
       'baseline_activation':float(baseline[target_layer,target]),'ablated_activation':float(changed[target_layer,target]),
       'kl_baseline_to_ablated':float((base_logp.exp()*(base_logp-logp)).sum()),'position':meta}
      stream.write(json.dumps(record,allow_nan=False)+'\n');checks+=1
     stream.flush();update(causal_checks=checks)
  artifact_paths=[array_path,ids_path,Path(str(prefix)+f'_qualification_{a.shard}.json'),Path(str(prefix)+f'_planner_{a.shard}.json'),Path(str(prefix)+f'_causal_{a.shard}.jsonl'),*link_paths]
  artifact_manifest=Path(str(prefix)+f'_artifacts_{a.shard}.json')
  atomic(artifact_manifest,{'job_id':job,'model_identity':IDENTITY,'worker_sha256':spec['worker_sha256'],'corpus_sha256':spec['corpus_sha256'],
   'activation_shape':[len(selected),32,9216],'activation_dtype':'float16','files':[{'name':p.name,'sha256':digest(p),'bytes':p.stat().st_size} for p in artifact_paths]})
  update(phase='complete',artifact_manifest=artifact_manifest.name,artifact_manifest_sha256=digest(artifact_manifest))
 except Exception as error:update(phase='failed',error=repr(error))
 finally:
  # Keep this exact leased service inspectable until its owning controller cleans it up.
  while not stop.is_set() and time.time()<spec['deadline']:stop.wait(1)
  server.shutdown()
if __name__=='__main__':main()
