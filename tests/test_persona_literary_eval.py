import persona
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
from persona_vectors import canonical_digest
from persona_literary_eval import annotate_response, assess, validate_control, check_control, compiled_prompt, submit, offline_plan
from persona_literature import initialize
from persona_literature_harness import save_prompt

class LiteraryTests(unittest.TestCase):
 def output(self,text,**kwargs):return dict(answer=text,complete=True,finish_reason='stop',**kwargs)
 def control(self):
  a=dict(schema='character-creation.hybrid-prefix.v1',kind='hybrid_text_prefix',model_identity='a'*64,runtime_identity='b'*64,prepared_identity='c'*64,layout_sha256='d'*64,prefix_text='A quiet fictional library.')
  a['sha256']=canonical_digest(a);return dict(artifact=a,dose=1,cache_policy='reuse')
 def test_refusal_is_separate_conservative_and_exact_span_bound(self):
  text='I cannot write that story.';row=annotate_response(self.output(text));self.assertEqual(row['label'],'refusal');self.assertTrue(row['review_required'])
  for span in row['evidence']:self.assertEqual(text[span['start']:span['end']],span['text'])
  for text in ['“I cannot help you,” Mara said.', 'Mara said I cannot help you.', 'The lantern flickered.']:
   self.assertEqual(annotate_response(self.output(text))['label'],'uncertain')
  self.assertEqual(annotate_response(dict(answer='I cannot write',complete=False,finish_reason='length'))['label'],'uncertain')
  self.assertEqual(annotate_response(self.output(''))['label'],'nonanswer')
  result=assess(self.output('The lantern flickered.'));self.assertEqual(result['semantic_coherence'],'unknown');self.assertEqual(result['coherence']['semantic_state'],'unknown')
 def test_strict_artifact_qualification_and_binding(self):
  control=self.control();validate_control(control,'a'*64)
  artifact=control['artifact'];endpoint={k:artifact[k] for k in ('model_identity','runtime_identity','prepared_identity')}
  capability=dict(supported=True,kinds=['hybrid_text_prefix'],doses=[0,1],layout_sha256=artifact['layout_sha256'],qualified_artifacts=[artifact['sha256']]);descriptor={'flare':{'kv_control':capability}}
  check_control(control,descriptor,endpoint)
  for key,value in [('qualified_artifacts',[]),('layout_sha256','e'*64),('supported',False)]:
   bad=deepcopy(descriptor);bad['flare']['kv_control'][key]=value
   with self.assertRaises(ValueError):check_control(control,bad,endpoint)
  for value in [.5,True,-1]:
   with self.assertRaises(ValueError):validate_control({**control,'dose':value},'a'*64)
  bad=deepcopy(control);bad['artifact']['prefix_text']='tampered'
  with self.assertRaises(ValueError):validate_control(bad,'a'*64)
 def test_compiler_source_binding_and_four_arm_plan(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'results').mkdir();dbpath=root/'results/persona_literature.sqlite';initialize(dbpath)
   with sqlite3.connect(dbpath) as db:
    db.execute("INSERT INTO books(id,title,sha256,workgroup,split) VALUES('b','Book','hash','family','heldout')")
    db.execute("INSERT INTO documents(id,book_id,member,text) VALUES('d','b','chapter','A lamp glows.')")
    db.execute("INSERT INTO units(id,book_id,document_id,start,end,text,split,boundary,canonical_id) VALUES('u','b','d',0,13,'A lamp glows.','heldout','window','u')")
   prompt=save_prompt(dbpath,dict(unit_id='u',mode='response',variant='original',seed=7));self.assertEqual(compiled_prompt(root,prompt['id'])['provenance']['family'],'family')
   recipe=dict(id='recipe',name='Neuron',model_identity='a'*64,pushes=[dict(kind='neuron',layer=1,neuron=2,dose=.2)],phase='all')
   app=SimpleNamespace(root=root,model=SimpleNamespace(identity='a'*64),saved_recipes=lambda:{'recipes':[recipe]},overnight=Mock(),submit_batch=Mock(return_value='batch'))
   with patch('persona_literary_eval.catalog',return_value={'kv':{'artifacts':[{'artifact':self.control()['artifact'],'endpoints':[{'url':'qualified'}]}]}}):
    result=submit(app,dict(prompt_id=prompt['id'],recipe_id='recipe',artifact_sha256=self.control()['artifact']['sha256'],seed=9,max_tokens=2048))
   self.assertEqual(result['split'],'heldout');cells=app.submit_batch.call_args.args[0]['cells']
   self.assertEqual([c['label'] for c in cells],['baseline','neuron_only','kv_only','combined']);self.assertEqual([bool(c.get('kv_control')) for c in cells],[False,False,True,True]);self.assertEqual(len({c['prompt'] for c in cells}),1)
   self.assertTrue(all(c['literature_prompt_id']==prompt['id'] for c in cells));self.assertFalse(offline_plan()['automatic_execution'])
   with sqlite3.connect(dbpath) as db:db.execute("UPDATE units SET text='Changed source' WHERE id='u'")
   with self.assertRaisesRegex(ValueError,'binding changed'):compiled_prompt(root,prompt['id'])

class TransportTests(unittest.TestCase):
 def test_request_and_response_binding_are_independent_of_neuron_program(self):
  from persona_explorer import Explorer
  control=LiteraryTests().control();artifact=control['artifact'];endpoint={k:artifact[k] for k in ('model_identity','prepared_identity','runtime_identity')};endpoint['url']='http://qualified'
  descriptor={'id':'model','flare':{**endpoint,'kv_control':dict(supported=True,kinds=['hybrid_text_prefix'],doses=[0,1],layout_sha256=artifact['layout_sha256'],qualified_artifacts=[artifact['sha256']])}}
  app=SimpleNamespace(model=SimpleNamespace(identity=artifact['model_identity']),config={},check_program=Explorer.check_program)
  settings=dict(seed=7,max_tokens=2048,thinking=False);program={'sites':[]};response={**{k:endpoint[k] for k in ('model_identity','prepared_identity','runtime_identity')},'choices':[{'message':{'content':'A story.'},'finish_reason':'stop'}],'diagnostics':{'complete_final_answer':True},'kv_control':{**{k:artifact[k] for k in ('model_identity','prepared_identity','runtime_identity','layout_sha256')},'artifact_sha256':artifact['sha256'],'dose':1,'cache_policy':'reuse','kind':'hybrid_text_prefix','prefix_tokens':8,'cache_hit':True,'state_artifact':{'path':'private.npz','sha256':'f'*64}}}
  calls=[]
  def api(url,payload=None,**kwargs):
   if payload is None:return {'data':[descriptor]}
   calls.append(payload);return response
  with patch('persona_explorer.fetch',api),patch('persona_explorer.check_generation_settings'):
   out,raw=Explorer.infer(app,endpoint,'Source',settings,program,{},extra={'kv_control':control})
   self.assertEqual(calls[0]['kv_control'],control);self.assertEqual(out['kv_control']['artifact_sha256'],artifact['sha256']);self.assertNotIn('steering',calls[0])
   Explorer.infer(app,endpoint,'Source',settings,program,{})
   self.assertNotIn('kv_control',calls[1])
   for key,bad in [('dose',0),('state_artifact',{}),('prefix_tokens',1025)]:
    old=response['kv_control'][key];response['kv_control'][key]=bad
    with self.subTest(key=key),self.assertRaisesRegex(ValueError,'KV response'):Explorer.infer(app,endpoint,'Source',settings,program,{},extra={'kv_control':control})
    response['kv_control'][key]=old
   response['kv_control']['artifact_sha256']='e'*64
   with self.assertRaisesRegex(ValueError,'KV response'):Explorer.infer(app,endpoint,'Source',settings,program,{},extra={'kv_control':control})

class ReviewTests(unittest.TestCase):
 def test_manual_answer_review_is_persisted_and_hash_bound(self):
  from persona_literary_eval import review,evaluated_batch
  with tempfile.TemporaryDirectory() as directory:
   dbpath=Path(directory)/'state.sqlite';output=dict(answer='A lamp glows.',complete=True,finish_reason='stop')
   record={'id':'batch','input':{'cells':[{'literature_prompt_id':'a'*64}]},'cells':[{'index':0,'output':output}]}
   app=SimpleNamespace(batch=lambda _:deepcopy(record),connect=lambda:sqlite3.connect(dbpath))
   value=dict(batch_id='batch',cell_index=0,answer_sha256=canonical_digest(output['answer']),label='answer',start=0,end=13,note='Responds to the scene without narrator refusal.')
   result=review(app,value);self.assertEqual(result['evidence'],output['answer']);self.assertEqual(evaluated_batch(app,'batch')['cells'][0]['response_review']['label'],'answer')
   with self.assertRaisesRegex(ValueError,'binding'):review(app,{**value,'answer_sha256':'b'*64})
   with self.assertRaisesRegex(ValueError,'span'):review(app,{**value,'end':100})

class SnapshotTests(unittest.TestCase):
 def test_batch_retains_compiled_provenance_and_rejects_text_substitution(self):
  from tests.test_persona_explorer import ExplorerWorkspaceTests
  fixture=ExplorerWorkspaceTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
  binding=dict(book_id='book',book_sha256='a'*64,family='family',canonical_id='scene',split='heldout',source_sha256='b'*64,harness_sha256='c'*64)
  compiled=dict(id='d'*64,prompt='Exact source-bound scene',provenance=binding)
  value=fixture.batch_input();value['cells'][0].update(prompt=compiled['prompt'],literature_prompt_id=compiled['id'])
  with patch('persona_literary_eval.compiled_prompt',return_value=compiled),patch('persona_explorer.fetch',fixture.api),patch.object(fixture.app.cell_executor,'submit'):
   identity=fixture.app.submit_batch(value);batch=fixture.app.batch(identity)
   self.assertEqual(batch['cells'][0]['source_binding'],{'literature_prompt_id':compiled['id'],**binding})
   self.assertEqual(batch['cells'][0]['prompt']['split'],'heldout')
   value['cells'][0]['prompt']='Substituted scene'
   with self.assertRaisesRegex(ValueError,'text differs'):fixture.app.submit_batch(value)


class CatalogLabelTests(unittest.TestCase):
 def test_named_prefix_fallback_and_invalid_labels(self):
  from persona_literary_eval import catalog
  control=LiteraryTests().control();artifact=control['artifact']
  endpoint={k:artifact[k] for k in ('model_identity','runtime_identity','prepared_identity')};endpoint['url']='http://qualified'
  descriptor={'flare':{'kv_control':dict(supported=True,kinds=['hybrid_text_prefix'],doses=[0,1],layout_sha256=artifact['layout_sha256'],qualified_artifacts=[artifact['sha256']])}}
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'results').mkdir();path=root/'results/persona_kv_artifacts.json'
   app=SimpleNamespace(root=root,model=SimpleNamespace(identity=artifact['model_identity']),endpoints=[endpoint],activation_endpoints=[],check_program=Mock(),batches=lambda:{'batches':[]},saved_recipes=lambda:{'recipes':[]})
   with patch('persona_fleet.fetch',return_value={'data':[descriptor]}):
    path.write_text(json.dumps({'artifacts':[artifact],'labels':{artifact['sha256']:'Horror'}}))
    self.assertEqual(catalog(app)['kv']['artifacts'][0]['label'],'Horror')
    path.write_text(json.dumps({'artifacts':[artifact]}))
    self.assertEqual(catalog(app)['kv']['artifacts'][0]['label'],artifact['prefix_text'])
    for label in ['', 'x'*81, 'line\nbreak']:
     path.write_text(json.dumps({'artifacts':[artifact],'labels':{artifact['sha256']:label}}))
     result=catalog(app);self.assertFalse(result['kv']['available']);self.assertIn('labels',result['kv']['reason'])
