import persona
import io,json,sqlite3,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import persona_hub_research as r

class HubTimeoutTests(unittest.TestCase):
 def test_hub_uses_bounded_sdk_http_client(self):
  from huggingface_hub.utils._http import hf_request_event_hook
  with patch('huggingface_hub.set_client_factory') as factory,patch('huggingface_hub.HfApi') as api:
   hub=r.Hub()
  factory.assert_called_once();api.assert_called_once_with();self.assertIs(hub.api,api.return_value)
  client=factory.call_args.args[0]()
  try:
   self.assertEqual(client.timeout.as_dict(),{'connect':15,'read':120,'write':120,'pool':60})
   self.assertTrue(client.follow_redirects);self.assertIn(hf_request_event_hook,client.event_hooks['request'])
  finally:client.close()

class FakeHub:
 def __init__(self,files):self.revisions={'head0':dict(files)};self.current='head0';self.calls=0;self.uncertain=False;self.corrupt=False
 def head(self):return self.current
 def read(self,path,revision,missing=False):
  value=self.revisions[revision].get(path)
  if value is None and not missing:raise FileNotFoundError(path)
  return value
 def file_sha(self,path,revision):return 'bad' if self.corrupt and path.startswith('research_batch_') else r.sha(self.read(path,revision))
 def commit(self,files,parent,message):
  if parent!=self.current:raise RuntimeError('CAS conflict')
  self.calls+=1;new='head'+str(self.calls);self.revisions[new]={**self.revisions[parent],**{k:v.read_bytes() if isinstance(v,Path) else v for k,v in files.items()}};self.current=new
  if self.uncertain:self.uncertain=False;raise TimeoutError('Reply lost after successful commit')
  return new

class Conflict412(Exception):
 class response:
  status_code=412

class ResearchTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.policy={'scenarios_sha256':r.sha(b'public'),'manifest_sha256':r.sha(b'original manifest'),'model_identity':'m'};self.hub=FakeHub({'scenarios.jsonl':b'public','manifest.json':b'original manifest','README.md':b'Original card\n'});self.check=self.root/'checkpoint.json'
 def publisher(self):return r.Publisher(self.hub,self.check,self.policy)
 def transaction(self,publisher):
  parent,manifest=publisher.snapshot();name,body,batch=r.metadata_batch([{'id':'x','kind':'prefill_activation','source':{'id':'public-id'}}]);manifest['batches'].append(batch);manifest['records']=1;return publisher.publish({name:body},manifest,parent,self.root)
 def test_commit_verified_checkpoint_and_subset_preserved(self):
  p=self.publisher();revision=self.transaction(p);self.assertIsNone(p.state['pending']);self.assertEqual(p.state['last_commit'],revision);self.assertEqual(self.hub.read('scenarios.jsonl',revision),b'public');self.assertEqual(self.hub.read('README.md',revision),b'Original card\n')
 def test_uncertain_commit_reconciles_without_duplicate(self):
  p=self.publisher();self.hub.uncertain=True
  with self.assertRaises(TimeoutError):self.transaction(p)
  self.assertIsNotNone(p.state['pending']);restarted=self.publisher();restarted.recover();self.assertIsNone(restarted.state['pending']);self.assertEqual(self.hub.calls,1)
 def test_failed_remote_verification_never_advances(self):
  p=self.publisher();self.hub.corrupt=True
  with self.assertRaises(ValueError):self.transaction(p)
  self.assertIsNone(p.state['last_commit']);self.assertIsNotNone(p.state['pending'])
 def test_remote_original_changed_fails_closed(self):
  self.hub.revisions['head0']['scenarios.jsonl']=b'changed'
  with self.assertRaises(ValueError):self.publisher().snapshot()
 def test_other_lane_advance_reconciles_same_intent(self):
  p=self.publisher();original=self.hub.commit
  def conflict(files,parent,message):
   self.hub.commit=original;original({'research_raw_manifest.json':b'raw'},parent,'otherlane');raise RuntimeError('CAS')
  self.hub.commit=conflict
  with self.assertRaises(RuntimeError):self.transaction(p)
  restarted=self.publisher();restarted.recover();self.assertEqual(self.hub.calls,2);self.assertIsNone(restarted.state['pending'])
 def test_412_other_lane_retries_exact_bytes_without_restart(self):
  p=self.publisher();original=self.hub.commit;seen=[]
  def conflict(files,parent,message):
   seen.append({name:path.read_bytes() for name,path in files.items()})
   if len(seen)==1:
    original({'research_raw_manifest.json':b'raw'},parent,'otherlane');raise Conflict412()
   return original(files,parent,message)
  self.hub.commit=conflict;revision=self.transaction(p)
  self.assertEqual(seen[0],seen[1]);self.assertEqual(self.hub.calls,2);self.assertEqual(p.state['last_commit'],revision);self.assertIsNone(p.state['pending'])
 def test_412_same_lane_change_fails_closed(self):
  p=self.publisher();original=self.hub.commit
  def conflict(files,parent,message):
   self.hub.commit=original;original({'research_manifest.json':b'changed'},parent,'same lane');raise Conflict412()
  self.hub.commit=conflict
  with self.assertRaises(ValueError):self.transaction(p)
  self.assertEqual(self.hub.calls,1);self.assertIsNotNone(p.state['pending']);self.assertIsNone(p.state['last_commit'])
 def test_412_retries_are_bounded_and_keep_immutable_intent(self):
  p=self.publisher();calls=[]
  def conflict(files,parent,message):calls.append(parent);raise Conflict412()
  self.hub.commit=conflict
  with self.assertRaises(Conflict412):self.transaction(p)
  self.assertEqual(len(calls),3);self.assertIsNotNone(p.state['pending']);self.assertIsNone(p.state['last_commit'])
 def test_public_allowlist_requires_exact_text_and_source_hash(self):
  item={'id':'approved','prompt':'Public prompt','response_sha256':'source','normalized_prompt_sha256':'normal'};row={'id':'approved','tier':'registered_big_five','messages':[{'role':'user','content':'Public prompt'}],'source_sha256':'source','provenance':{'registered_id':'approved','source':{'kind':'synthetic_model_response','normalized_prompt_sha256':'normal'}}}
  value=r.approved_source(row,{'approved':item},{});self.assertEqual(value['prompt'],'Public prompt');row['messages'][0]['content']='PRIVATE BOOK'
  with self.assertRaises(ValueError):r.approved_source(row,{'approved':item},{})
  self.assertIsNone(r.approved_source({'tier':'book_diverse','messages':[{'content':'PRIVATE BOOK'}]}, {}, {}))
 def test_authored_hash_guard(self):
  policy={'authored_emotion_source_sha256':'graph'};row={'id':'probe','tier':'authored_emotion_contrast','source_sha256':'graph','messages':[{'role':'user','content':'Original authored text'}],'provenance':{'authored_graph_sha256':'graph','raw_text_sha256':r.sha(b'Original authored text'),'feature':'joy','frame':'feeling','arm':'positive','pair_family':'pair'}}
  self.assertEqual(r.approved_source(row,{},policy)['tier'],'authored_emotion_contrast');row['provenance']['raw_text_sha256']='wrong'
  with self.assertRaises(ValueError):r.approved_source(row,{},policy)
 def test_raw_original_bytes_and_entry_shapes(self):
  path=self.root/'original.npz';np.savez_compressed(path,search_peak=np.ones((2,3),dtype=np.float32),model_identity=np.array('m'))
  row={'id':'a'*64,'kind':'prefill_activation','path':str(path),'sha256':r.file_hash(path),'source_id':'public','prompt_sha256':'p','identity':{'model_identity':'m'},'request_sha256':'request'};files,batch=r.raw_shard([row],self.root)
  with tarfile.open(files[batch['file']]) as archive:self.assertEqual(archive.extractfile(row['id']+'.npz').read(),path.read_bytes())
  entry=json.loads(files[batch['entry_manifest']])['entries'][0];self.assertEqual(entry['arrays']['search_peak']['shape'],[2,3]);self.assertNotIn('path',entry);self.assertEqual(r.raw_shard([row],self.root)[1],batch)
 def test_raw_private_metadata_rejected(self):
  path=self.root/'private.npz';np.savez(path,private_book=np.array('secret'))
  with self.assertRaises(ValueError):r.inspect_raw(path)
  np.savez(path,runtime=np.array('/home/person/private'))
  with self.assertRaises(ValueError):r.inspect_raw(path)
 def test_card_append_is_idempotent_preserves_original(self):
  card=b'---\nconfigs: original\n---\n# Original card\n';once=r.append_card(card,r.card_section(20));self.assertTrue(once.startswith(card));self.assertEqual(once,r.append_card(once,r.card_section(20)));self.assertIn(b'20 records',once.replace(b'**',b''))
 def test_guides_prefer_canonical_root_and_fall_back_without_staging(self):
  dataset=self.root/'dataset';dataset.mkdir();fallback={'README_human.md':b'published human','README_agent.md':b'published agent'}
  for name,body in fallback.items():(dataset/name).write_bytes(body)
  self.assertEqual(r.read_guides(self.root),fallback)
  (self.root/'README_human.md').write_bytes(b'canonical human');self.assertEqual(r.read_guides(self.root),{**fallback,'README_human.md':b'canonical human'})
  (self.root/'README_agent.md').write_bytes(b'canonical agent');self.assertEqual(r.read_guides(self.root),{'README_human.md':b'canonical human','README_agent.md':b'canonical agent'})
  self.assertEqual({name:(dataset/name).read_bytes() for name in fallback},fallback)
 def test_reviewed_bundle_private_fields_rejected(self):
  for value in ({'payload':{'book_id':'private'}},{'payload':{'path':'/mnt/secret'}},{'payload':'hf_abcdefghijklmno'}):
   with self.assertRaises(ValueError):r.reject_private_keys(value)
 def test_immutable_batch_stable_order_and_source_join(self):
  a={'id':'a','kind':'baseline_answer','source':{'id':'prompt-a'}};b={'id':'b','kind':'prefill_activation','source':{'id':'prompt-b'}};self.assertEqual(r.metadata_batch([b,a]),r.metadata_batch([a,b]));self.assertEqual(r.metadata_batch([a,b])[2]['source_keys'],['baseline_answer:prompt-a','prefill_activation:prompt-b'])
if __name__=='__main__':unittest.main()
