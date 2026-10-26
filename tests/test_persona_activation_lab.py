import persona
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch
from persona_activation_lab import ActivationLab, validate_result, starter_recipes


def result():
    return {'id':'a'*32,'tokens':[{'id':1,'text':'A','position':0},{'id':2,'text':'B','position':1}],
            'coverage':{'layers':2,'neurons_per_layer':3,'tokens':2,'all_neurons':True,'all_prompt_positions':True},
            'candidates':[{'layer':1,'neuron':2,'peak':-4.,'peak_position':1,'mean_abs':3.,'values':[2.,-4.]}]}

class ValidationTests(unittest.TestCase):
    def setUp(self):self.model=SimpleNamespace(layers=2,intermediate=3)
    def test_full_coverage_and_profiles(self):self.assertEqual(validate_result(result(),self.model,1),result())
    def test_incomplete_or_fabricated_profiles_rejected(self):
        for mutate in [lambda r:r['coverage'].update(all_neurons=False),lambda r:r['tokens'][1].update(position=0),
                       lambda r:r['candidates'][0].update(values=[2.]),lambda r:r['candidates'][0].update(peak=4.),
                       lambda r:r['candidates'][0].update(mean_abs=2.),lambda r:r['candidates'][0].update(neuron=3),
                       lambda r:r['candidates'][0].update(values=[float('nan'),-4.])]:
            r=result();mutate(r)
            with self.assertRaises(ValueError):validate_result(r,self.model,1)

    def test_starters_use_only_ranked_candidates_and_separate_layers(self):
        data={'candidates':[{'layer':l,'neuron':n} for l,n in [(1,2),(1,3),(4,5),(7,8),(9,10)]]}
        recipes=starter_recipes(data)
        self.assertEqual([len(r['neurons']) for r in recipes],[1,1,3])
        self.assertEqual(recipes[1]['neurons'],[{'layer':4,'neuron':5}])
        self.assertEqual([r['layer'] for r in recipes[2]['neurons']],[1,4,7])
        self.assertTrue(all(r['dose']==.1 and r['status']=='untested' and 'confidence' not in r for r in recipes))
        self.assertEqual(starter_recipes(None),[])
        self.assertEqual(len(starter_recipes(result())),1)

class LabTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'db'
        @contextmanager
        def connect():
            with sqlite3.connect(self.path) as db:yield db
        self.endpoint={'url':'http://127.0.0.1:1','model_identity':'model','runtime_identity':'runtime','prepared_identity':'prepared'}
        @contextmanager
        def slot(eligible):yield self.endpoint
        self.app=SimpleNamespace(connect=connect,model=SimpleNamespace(identity='model',layers=2,intermediate=3),
            endpoints=[self.endpoint],check_program=Mock(return_value={'activation_search':{'supported':True}}),
            endpoint_slot=slot,submit_batch=Mock(return_value='b'*32))
        self.lab=ActivationLab(self.app);self.addCleanup(self.lab.executor.shutdown)
    def insert(self,state='complete'):
        with self.app.connect() as db:db.execute('INSERT INTO activation_searches VALUES(?,?,?,?,?,?,?)',
            ('a'*32,1,state,json.dumps({'text':'source','top_k':1,'model_identity':'model'}),json.dumps(result()),None,None))
    def test_compare_binds_candidates_and_divides_collection_dose(self):
        self.insert();value={'search_id':'a'*32,'neurons':[{'layer':1,'neuron':2}],'prompt':'new scenario','seed':7035,'dose':.5,'max_tokens':2048}
        response=self.lab.compare(value);self.assertEqual(response['id'],'b'*32)
        cells=self.app.submit_batch.call_args.args[0]['cells']
        self.assertEqual(cells[0]['recipe']['pushes'],[])
        self.assertEqual(cells[1]['recipe']['pushes'][0]['dose'],.5)
        self.assertEqual(cells[2]['recipe']['pushes'][0]['dose'],-.5)
        self.assertEqual(len(self.lab.get('a'*32)['comparisons']),1)
        value['neurons']=[{'layer':0,'neuron':0}]
        with self.assertRaises(ValueError):self.lab.compare(value)
    def test_no_capable_worker_does_not_enqueue(self):
        self.app.check_program.return_value={}
        with patch('persona_activation_lab.fetch',return_value={'data':[{'id':'model'}]}):
            with self.assertRaisesRegex(ValueError,'No identity-verified'):self.lab.submit({'text':'test','top_k':1})
        self.assertEqual(self.lab.listing(),[])
    def test_request_validation(self):
        for value in [{'text':'','top_k':1},{'text':'x','top_k':33},{'text':'x','top_k':True},{'text':'x','top_k':1,'extra':0}]:
            with self.assertRaises(ValueError):self.lab.submit(value)
    def test_failed_identity_is_retained_not_retried(self):
        self.insert(state='queued')
        response={'model_identity':'wrong'}
        with patch('persona_activation_lab.fetch',side_effect=[{'data':[{'id':'model'}]},response]) as fetch:
            self.lab.run('a'*32,{'text':'source','top_k':1},[self.endpoint['url']])
        self.assertEqual(fetch.call_count,2)
        record=self.lab.get('a'*32);self.assertEqual(record['state'],'failed');self.assertIn('identity differs',record['error'])
        with self.app.connect() as db:raw=json.loads(db.execute('SELECT raw FROM activation_searches').fetchone()[0])
        self.assertEqual(raw['response'],response)
    def test_restart_marks_inflight_interrupted(self):
        self.insert('running');other=ActivationLab(self.app);other.executor.shutdown()
        self.assertEqual(other.get('a'*32)['state'],'interrupted')

    def test_canonical_workers_remain_eligible_with_isolated_endpoints(self):
        self.app.activation_endpoints=[{**self.endpoint,'url':'http://127.0.0.1:2'}]
        other=ActivationLab(self.app);self.addCleanup(other.executor.shutdown)
        self.assertEqual([e['url'] for e in other.endpoints],[self.endpoint['url'],'http://127.0.0.1:2'])
        self.insert()
        self.assertEqual(other.get('a'*32)['starter_recipes'][0]['neurons'],[{'layer':1,'neuron':2}])

    def test_search_queue_can_fill_six_cards_and_has_a_finite_bound(self):
        self.lab.executor.submit=Mock()
        with patch('persona_activation_lab.fetch',return_value={'data':[{'id':'model'}]}):
            for i in range(64):self.lab.submit({'text':f'scene {i}','top_k':1})
            self.assertEqual(self.lab.executor.submit.call_count,64)
            with self.assertRaisesRegex(ValueError,'queue is full'):self.lab.submit({'text':'overflow','top_k':1})
