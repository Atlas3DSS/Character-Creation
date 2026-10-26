"""Small CPU hybrid fixtures; no checkpoint weights, GPU allocations or downloads."""
import persona
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from persona_phantom_train import (GraftBank,FunctionalDecoder,attention_with_graft,causal_prefix_mask,
                                  gradient_check,load_pairs,digest,write_candidate,DATA_SCHEMA)

class Rotary(nn.Module):
 def forward(self,hidden,positions):
  phase=positions[0].to(hidden.dtype).unsqueeze(-1)*.17
  return torch.cat((phase,phase),-1).cos(),torch.cat((phase,phase),-1).sin()

class Attention(nn.Module):
 def __init__(self):
  super().__init__();self.head_dim=4;self.num_key_value_groups=2;self.scaling=.5
  self.q_proj=nn.Linear(8,16,bias=False);self.k_proj=nn.Linear(8,4,bias=False);self.v_proj=nn.Linear(8,4,bias=False);self.o_proj=nn.Linear(8,8,bias=False)
  self.q_norm=nn.Identity();self.k_norm=nn.Identity()

class Recurrence(nn.Module):
 def __init__(self):super().__init__();self.input=nn.Linear(8,8,bias=False);self.transition=nn.Linear(8,8,bias=False)
 def forward(self,hidden_states,cache_params=None,attention_mask=None):
  assert cache_params is None and attention_mask is None
  state=torch.zeros_like(hidden_states[:,0]);values=[]
  for position in range(hidden_states.shape[1]):
   state=torch.tanh(self.input(hidden_states[:,position])+self.transition(state));values.append(state)
  return torch.stack(values,dim=1)

class Block(nn.Module):
 def __init__(self,kind):
  super().__init__();self.block_type=kind;self.input_layernorm=nn.LayerNorm(8);self.post_attention_layernorm=nn.LayerNorm(8);self.mlp=nn.Sequential(nn.Linear(8,12),nn.SiLU(),nn.Linear(12,8))
  if kind=='full_attention':self.self_attn=Attention()
  else:self.linear_attn=Recurrence()

class Tiny(nn.Module):
 def __init__(self):
  super().__init__();self.model=nn.Module();self.model.embed_tokens=nn.Embedding(19,8);self.model.layers=nn.ModuleList([Block(t) for t in ['linear_attention','full_attention']*2]);self.model.norm=nn.LayerNorm(8);self.model.rotary_emb=Rotary();self.lm_head=nn.Linear(8,19,bias=False)

class FunctionalTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(37);self.base=Tiny().double();self.bank=GraftBank((1,3),1,3,4,dtype=torch.float64);self.decoder=FunctionalDecoder(self.base,self.bank);self.ids=torch.tensor([[1,3,4,2,6,8]])
 def test_explicit_prefix_mask_is_causal(self):
  mask=causal_prefix_mask(3,2,device='cpu',dtype=torch.float32)[0,0]
  self.assertTrue(torch.equal(mask[:,:2],torch.zeros(3,2)));self.assertTrue(torch.isneginf(mask[0,3:] ).all());self.assertEqual(mask[2,4],0)
 def test_frozen_weights_and_hybrid_gradient_finite_difference(self):
  result=gradient_check(self.decoder,self.ids,3,epsilon=.005,tolerance=.025)
  self.assertTrue(result['passed']);self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.base.parameters()))
  self.assertTrue(all(p.grad is not None and p.grad.abs().sum()>0 for p in self.bank.parameters()))
 def test_checkpoint_preserves_values_and_gradients(self):
  direct=FunctionalDecoder(deepcopy(self.base),deepcopy(self.bank),checkpoint_blocks=False)
  a=self.decoder.loss(self.ids,3,kl_weight=.1,logit_chunk=2);a.backward()
  b=direct.loss(self.ids,3,kl_weight=.1,logit_chunk=2);b.backward()
  torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
  for left,right in zip(self.bank.parameters(),direct.bank.parameters()):torch.testing.assert_close(left.grad,right.grad,rtol=1e-5,atol=1e-7)
 def test_fresh_recurrence_true_off_and_no_future_leak(self):
  before=self.decoder.hidden(self.ids,enabled=False).detach();on=self.decoder.hidden(self.ids).detach()
  self.assertFalse(torch.equal(before,on));torch.testing.assert_close(self.decoder.hidden(self.ids,enabled=False),before)
  changed=self.ids.clone();changed[0,-1]=11
  torch.testing.assert_close(self.decoder.hidden(changed)[:,:-1],on[:,:-1],rtol=1e-5,atol=1e-7)
  with torch.no_grad():
   for p in self.bank.parameters():p.add_(3)
  torch.testing.assert_close(self.decoder.hidden(self.ids,enabled=False),before)
 def test_actual_update_and_export_is_explicit_unqualified_candidate(self):
  initial=[p.detach().clone() for p in self.bank.parameters()];optimizer=torch.optim.Adam(self.bank.parameters(),lr=.001)
  self.decoder.loss(self.ids,3).backward();optimizer.step()
  self.assertTrue(any(not torch.equal(a,b) for a,b in zip(initial,self.bank.parameters())))
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'graft.npz'
   with self.assertRaises(ValueError):write_candidate(path,self.bank,{'optimizer_steps':0,'native_flare_qualified':False})
   write_candidate(path,self.bank,{'optimizer_steps':1,'native_flare_qualified':False})
   with np.load(path,allow_pickle=False) as data:
    self.assertEqual(data['keys'].shape,(2,1,3,4));self.assertEqual(data['values'].shape,(2,1,3,4));meta=json.loads(str(data['metadata']));self.assertFalse(meta['native_flare_qualified']);self.assertEqual(meta['recurrent_initial_state'],'fresh_zero');self.assertEqual(meta['prompt_position_start'],3)

class DataTests(unittest.TestCase):
 def row(self,family,split):
  row={'family_id':family,'split':split,'source_sha256':digest(family),'prompt':family+': An adult meets a friend at a library.','target':'She greets her friend calmly.','domain':'benign_literary','provenance':{'source_kind':'authored','source_sha256':digest(family)}}
  row['id']=digest(row);return row
 def test_approval_and_family_holdout_are_required(self):
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'pairs.json';data={'schema':DATA_SCHEMA,'approval':{'scope':'benign_literary','approved':True},'rows':[self.row('a','discovery'),self.row('b','heldout')]};path.write_text(json.dumps(data));self.assertEqual(len(load_pairs(path)['rows']),2)
   data['rows'][1]=self.row('a','heldout');path.write_text(json.dumps(data))
   with self.assertRaisesRegex(ValueError,'crosses'):load_pairs(path)
   data['rows'][1]=self.row('b','heldout');data['approval']['approved']=False;path.write_text(json.dumps(data))
   with self.assertRaisesRegex(ValueError,'approval'):load_pairs(path)

class ActualHFHybridTests(unittest.TestCase):
 def test_tiny_qwen_hybrid_off_agreement_and_real_graft_gradient(self):
  try:
   from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
   from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
  except ImportError:self.skipTest('Qwen3.5 Transformers optional for portable CPU fixtures')
  torch.manual_seed(91)
  config=Qwen3_5TextConfig(vocab_size=31,hidden_size=16,intermediate_size=24,num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,head_dim=8,linear_key_head_dim=8,linear_value_head_dim=8,linear_num_key_heads=1,linear_num_value_heads=1,layer_types=['linear_attention','full_attention']*2,rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':.5,'mrope_section':[1,1,0]},use_cache=False)
  config._attn_implementation='eager';model=Qwen3_5ForCausalLM(config).float().eval()
  bank=GraftBank((1,3),1,2,8);decoder=FunctionalDecoder(model,bank);ids=torch.tensor([[1,2,3,4,5,6]])
  with torch.no_grad():
   expected=model(input_ids=ids,use_cache=False).logits
   observed=model.lm_head(decoder.hidden(ids,enabled=False))
   torch.testing.assert_close(observed,expected,rtol=1e-5,atol=1e-6)
  result=gradient_check(decoder,ids,3,epsilon=.03,tolerance=.05)
  self.assertTrue(result['passed']);self.assertTrue(all(p.grad is None for p in model.parameters()))

class AdmissionTests(unittest.TestCase):
 def test_native_metrics_and_next_token_are_all_required(self):
  from persona_phantom_train import native_forward_check
  class Decoder:
   def __init__(self):
    self.factor=1.;self.model=SimpleNamespace(lm_head=nn.Linear(5120,3))
    with torch.no_grad():self.model.lm_head.weight.zero_();self.model.lm_head.bias.copy_(torch.tensor([0.,1.,0.]))
   def hidden(self,ids,enabled,captures):
    assert enabled is False
    for layer in range(64):captures[layer]=np.ones(5120,np.float32)*(self.factor if layer==7 else 1.)
    return torch.zeros(1,2,5120)
  reference={'sha256':'a'*64,'runtime_identity':'b'*64,'records':[{'row_id':str(i),'ids':np.array([1,2]),'generated':1,'residual':np.ones((64,5120),np.float32),'artifact_sha256':'c'*64} for i in range(6)]}
  decoder=Decoder();self.assertTrue(native_forward_check(decoder,reference)['passed'])
  decoder.factor=1.2;result=native_forward_check(decoder,reference);self.assertFalse(result['passed']);self.assertGreater(result['records'][0]['max_relative_l2'],.1)
  decoder.factor=1.;reference['records'][0]['generated']=2;self.assertFalse(native_forward_check(decoder,reference)['passed'])
 def test_training_report_cannot_bypass_failed_or_changed_validation(self):
  from persona_phantom_train import require_validation
  metadata={k:'a'*64 for k in ('model_identity','dataset_sha256','trainer_sha256','reader_sha256')};metadata.update(seed=1,slots=2,max_tokens=32)
  native={'sha256':'b'*64};manifest={'config_sha256':'c'*64}
  report={**metadata,'state':'validation_passed','validation':{'passed':True},'native_forward':{'passed':True,'native_reference_sha256':native['sha256']},'checkpoint':manifest}
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'validation.json';path.write_text(json.dumps(report));self.assertEqual(require_validation(path,metadata,manifest,native)['sha256'],__import__('hashlib').sha256(path.read_bytes()).hexdigest())
   for field,value in [('state','validation_failed'),('reader_sha256','d'*64),('slots',3)]:
    path.write_text(json.dumps({**report,field:value}))
    with self.assertRaises(ValueError):require_validation(path,metadata,manifest,native)
 def test_backward_can_stop_and_partial_checkpoint_is_not_trained(self):
  from persona_phantom_train import TrainingInterrupted,write_interrupted
  stopped=[False]
  def guard():
   if stopped[0]:raise TrainingInterrupted('test stop')
  torch.manual_seed(2);bank=GraftBank((1,3),1,2,4);decoder=FunctionalDecoder(Tiny(),bank,guard=guard)
  loss=decoder.loss(torch.tensor([[1,2,3,4]]),2);stopped[0]=True
  with self.assertRaises(TrainingInterrupted):loss.backward()
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'partial.npz';write_interrupted(path,bank,None,{'optimizer_steps':0})
   with np.load(path,allow_pickle=False) as data:
    metadata=json.loads(str(data['metadata']));self.assertFalse(metadata['training_complete']);self.assertFalse(metadata['native_flare_qualified']);self.assertEqual(metadata['state'],'interrupted');self.assertEqual(metadata['schema'],'character-creation.phantom-training-checkpoint.v1')

class SignalTests(unittest.TestCase):
 def test_sigterm_during_loading_saves_terminal_partial_without_gpu(self):
  import signal
  from unittest.mock import patch
  from persona_phantom_train import run
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);dataset=root/'pairs.json'
   dataset.write_text(json.dumps({'schema':DATA_SCHEMA,'approval':{'scope':'benign_literary','approved':True},'rows':[DataTests().row('a','discovery'),DataTests().row('b','heldout')]}))
   args=SimpleNamespace(output=root/'report.json',partial_output=root/'partial.npz',dataset=dataset,checkpoint=root,model_identity='a'*64,command='validate',native_reference=None,validation_report=None,devices=['cuda:'+str(i) for i in range(6)],max_gpu_gib=1.,seed=7,learning_rate=.001,kl_weight=.1,slots=2,max_tokens=32)
   tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**k:[1,2],encode=lambda *a,**k:[3,4],eos_token_id=5)
   def stopped_loader(*a,**kw):signal.raise_signal(signal.SIGTERM);kw['guard']();raise AssertionError('Stop was not honored')
   original=signal.getsignal(signal.SIGTERM)
   with patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer),patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(total_memory=2*2**30)),patch('torch.cuda.set_per_process_memory_fraction'),patch('torch.cuda.memory_reserved',return_value=0),patch('persona_phantom_train.load_frozen_model',side_effect=stopped_loader):
    result=run(args)
   self.assertEqual(result['state'],'interrupted');self.assertFalse(args.output.exists());self.assertEqual(signal.getsignal(signal.SIGTERM),original)
   with np.load(args.partial_output,allow_pickle=False) as data:
    metadata=json.loads(str(data['metadata']));self.assertFalse(metadata['bank_initialized']);self.assertFalse(metadata['training_complete']);self.assertEqual(metadata['stage'],'loading')

class TokenizerContainerTests(unittest.TestCase):
 def test_batchencoding_flat_batched_and_tensor_preserve_exact_ids(self):
  from persona_phantom_train import token_ids,encode_pair
  from transformers.tokenization_utils_base import BatchEncoding
  variants=([7,8,9],{'input_ids':[7,8,9]},BatchEncoding({'input_ids':[7,8,9],'attention_mask':[1,1,1]}),BatchEncoding({'input_ids':[[7,8,9]]}),{'input_ids':torch.tensor([[7,8,9]])},np.array([7,8,9],dtype=np.int64))
  for value in variants:
   self.assertEqual(token_ids(value),[7,8,9])
   tokenizer=SimpleNamespace(apply_chat_template=lambda *a,value=value,**kw:value,encode=lambda *a,**kw:[10,11],eos_token_id=12)
   ids,start=encode_pair(tokenizer,{'id':'row','prompt':'Exact prompt','target':'Exact answer'},10)
   self.assertEqual(ids.tolist(),[[7,8,9,10,11,12]]);self.assertEqual(start,3)
 def test_multiple_sequences_and_non_ids_fail_closed(self):
  from persona_phantom_train import token_ids
  for value in ({'attention_mask':[1]},[[1],[2]],[True],[1.5],'1 2',[-1]):
   with self.assertRaises(ValueError):token_ids(value)
