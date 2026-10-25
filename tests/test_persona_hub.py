"""Only synthetic Big Five rows may enter the public export."""
import persona
import hashlib,json,tempfile,unittest
from pathlib import Path
from persona_hub import export

class HubExportTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);(self.root/'data').mkdir()
  (self.root/'data/persona_manager_corpus.jsonl').write_text('private corpus stays private')
  self.row={'id':'a','corpus':'big_five','category':'synthetic_scenario','synthetic':True,'split':'discovery','prompt':'Propose a reversible experiment.',
   'labels':{'validated_inventory':False,'prompt_design_trait':'openness','prompt_design_facet':'imagination','scenario_domain':'research'},
   'generator':dict.fromkeys(['job_id','request_id','model_identity','runtime_identity','prepared_identity','request_sha256','response_sha256'],'test'),
   'source':{'kind':'synthetic_model_response','normalized_prompt_sha256':'test','path':'PRIVATE_PATH'}}
 def save(self):
  raw=(json.dumps(self.row)+'\n').encode();(self.root/'data/persona_generated_big_five.jsonl').write_bytes(raw)
  (self.root/'data/persona_generated_big_five_manifest.json').write_text(json.dumps({'status':'complete','output_sha256':hashlib.sha256(raw).hexdigest(),'model_identity':'test','records':1}))
  (self.root/'data/persona_public_release.local.json').write_text(json.dumps({'classification':'public','source_sha256':hashlib.sha256(raw).hexdigest()}))
 def test_whitelisted_export_drops_private_paths(self):
  self.save();x=export(self.root);self.assertEqual(x['records'],1);self.assertNotIn('PRIVATE_PATH',(self.root/'dataset/scenarios.jsonl').read_text());self.assertNotIn('private corpus',(self.root/'dataset/scenarios.jsonl').read_text())
 def test_rejects_book_or_validated_inventory(self):
  for field,value in [('corpus','skippy'),('synthetic',False)]:
   old=self.row[field];self.row[field]=value;self.save()
   with self.assertRaises(ValueError):export(self.root)
   self.row[field]=old
 def test_source_tampering_fails_before_export(self):
  self.save();(self.root/'data/persona_generated_big_five.jsonl').write_text('{}\n')
  with self.assertRaises(ValueError):export(self.root)
  self.assertFalse((self.root/'dataset').exists())

 def test_unapproved_snapshot_is_not_exported(self):
  self.save();(self.root/'data/persona_public_release.local.json').unlink()
  with self.assertRaisesRegex(ValueError,'approval required'):export(self.root)
  self.assertFalse((self.root/'dataset').exists())
