import persona
from copy import deepcopy
from collections import Counter
from pathlib import Path
import time
import json
import tempfile
from unittest.mock import patch
import unittest
from persona_personality_probes import load_suite
from persona_personality_sweep import freeze,validate_plan,validate_launch,planner_data,SEEDS,ARMS,canonical_digest,blind_packet,request_for,validate_gate,request_planner

class SweepTests(unittest.TestCase):
    def setUp(self):
        self.suite=load_suite(Path(__file__).resolve().parents[1]/'data/persona_personality_probes.json')
        self.binding=dict(model_identity='a'*64,runtime_identity='b'*64,prepared_identity='c'*64)
        self.launch=dict(start_time=time.time()-1,deadline=time.time()+3600,binding=self.binding,endpoints=[dict(self.binding,url=f'http://127.0.0.1:{8930+i}',gpu_uuid=f'gpu{i}',lease_id=f'lease{i}',project='test',host='v100',diagnostic=True) for i in range(5)])
        self.artifact=dict(filename='direction.npz',sha256='d'*64)
    def test_exact288_neural_only_matched_arms(self):
        plan=freeze(self.suite,self.launch,self.artifact);validate_plan(plan,self.suite,self.launch)
        self.assertEqual(len(plan['jobs']),288)
        self.assertEqual(Counter(j['arm'] for j in plan['jobs']),Counter({a:48 for a in ARMS}))
        for case in self.suite['cases']:
            for seed in SEEDS:
                jobs=[j for j in plan['jobs'] if j['case_id']==case['id'] and j['seed']==seed]
                self.assertEqual(len(jobs),6);self.assertEqual({j['prompt'] for j in jobs},{case['prompt']})
                self.assertTrue(all(j['split']=='discovery' and j['recipe']['phase']=='all' for j in jobs))
                self.assertEqual(jobs[0]['recipe']['pushes'],[])
                self.assertEqual([j['recipe']['pushes'][0]['dose'] for j in jobs[1:]],[.1,-.1,.1,-.1,.1])
    def test_rehashed_scientific_tampering_rejected(self):
        for field,value in [('prompt','changed'),('seed',7035),('split','heldout')]:
            plan=freeze(self.suite,self.launch,self.artifact);plan['jobs'][0][field]=value
            plan['input_sha256']=canonical_digest({k:v for k,v in plan.items() if k!='input_sha256'})
            with self.assertRaises(ValueError):validate_plan(plan,self.suite,self.launch)
    def test_five_six_unique_bindings_deadlines(self):
        validate_launch(self.launch)
        six=deepcopy(self.launch);six['endpoints'].append(dict(six['endpoints'][0],url='http://127.0.0.1:8935',gpu_uuid='gpu5'));validate_launch(six)
        for change in ('duplicate','runtime','expired','four'):
            launch=deepcopy(self.launch)
            if change=='duplicate':launch['endpoints'][1]['gpu_uuid']='gpu0'
            elif change=='runtime':launch['endpoints'][0]['runtime_identity']='e'*64
            elif change=='expired':launch['deadline']=time.time()-1
            else:launch['endpoints'].pop()
            with self.assertRaises(ValueError):validate_launch(launch)
    def test_planner_is_order_only_no_prompt_or_execution(self):
        value=dict(family_order=list(reversed(list(dict.fromkeys(c['pair_id'] for c in self.suite['cases'])))),followups=[],rationale='Synthetic order')
        self.assertEqual(planner_data(value,self.suite),value)
        plan=freeze(self.suite,self.launch,self.artifact,value)
        self.assertEqual(plan['jobs'][0]['family_id'],value['family_order'][0])
        value['system']='persona injection'
        with self.assertRaises(ValueError):planner_data(value,self.suite)

    def test_request_never_injects_style_system_or_kv(self):
        class Bank:
            def program(self,pushes,phase):return {'phase':phase,'sites':[]},{'program_sha256':'a'*64}
        job=freeze(self.suite,self.launch,self.artifact)['jobs'][0]
        request,_=request_for(job,'model',Bank())
        self.assertEqual(request['messages'],[{'role':'user','content':job['prompt']}])
        self.assertFalse(request['chat_template_kwargs']['enable_thinking'])
        self.assertNotIn('kv_control',request)
        self.assertEqual(request['max_tokens'],2048)
    def test_complete_blind_export_coverage_and_no_arm_leak(self):
        plan=freeze(self.suite,self.launch,self.artifact)
        report=dict(state='complete',plan_sha256=plan['input_sha256'],binding=self.binding,results=[dict(job_id=j['id'],state='complete',request_sha256='e'*64,
            response=dict(self.binding,response_capture={'request_sha256':'e'*64},choices=[{'message':{'content':'Synthetic answer'}}])) for j in plan['jobs']])
        packet,key=blind_packet(plan,report,self.suite)
        self.assertEqual(len(packet['groups']),48);self.assertEqual(len(key),288)
        self.assertEqual(len({k['code'] for k in key}),288)
        self.assertTrue(all(len(g['answers'])==6 for g in packet['groups']))
        self.assertTrue(all(set(a)=={'code','answer'} for g in packet['groups'] for a in g['answers']))
        report['results'].pop()
        with self.assertRaises(ValueError):blind_packet(plan,report,self.suite)


    def test_experimental_proof_runtime_separate_but_core_bound(self):
        launch=deepcopy(self.launch);launch.update(gate_sha256='e'*64,production_core_source_hashes={'core.py':'f'*64,'heliopause':'1'*64})
        gate=dict(schema='character-creation.batch-proof.v1',state='complete',passed=True,production_binding=self.binding,
                  binding=dict(self.binding,runtime_identity='2'*64),production_core_equal=True,
                  production_core_source_hashes=launch['production_core_source_hashes'],driver_sha256='3'*64,policy_sha256='4'*64,
                  artifact={'sha256':'5'*64,'bytes':100,'filename':'proof.npz'})
        validate_gate(gate,launch,'e'*64)
        for field,value in [('passed',False),('state','running'),('production_core_equal',False),('driver_sha256','bad'),
                            ('production_core_source_hashes',{'core.py':'0'*64}),('binding',dict(gate['binding'],prepared_identity='0'*64))]:
            with self.subTest(field=field),self.assertRaises(ValueError):validate_gate(dict(gate,**{field:value}),launch,'e'*64)
        with self.assertRaises(ValueError):validate_gate(gate,launch,'0'*64)


    def test_john_one_shot_receipt_and_no_uncertain_retry(self):
        from persona_planner import GPU,MODEL_SHA
        planner=dict(url='http://127.0.0.1:18511',model='persona-manager',gpu_uuid=GPU,host='wtatum84',model_sha256=MODEL_SHA,lease_id='john',project='test')
        order=list(dict.fromkeys(c['pair_id'] for c in self.suite['cases']))
        value={'family_order':order,'followups':[],'rationale':'Fixture ordering only'}
        class Response:
            def __init__(self,value):self.value=value
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return json.dumps(self.value).encode()
        for finish in ('stop','length'):
            calls=[]
            def fetch(request,timeout):
                calls.append(request)
                if request.full_url.endswith('/health'):return Response({'status':'ok'})
                if request.full_url.endswith('/props'):return Response({'model_alias':'persona-manager','default_generation_settings':{'n_ctx':131072}})
                return Response({'model':'persona-manager','choices':[{'finish_reason':finish,'message':{'content':json.dumps(value)}}]})
            with tempfile.TemporaryDirectory() as folder,patch('persona_personality_sweep.check_admission',return_value=(True,'ok')),patch('persona_personality_sweep.subprocess.check_output',return_value=json.dumps({'jobs':[{'id':'john','lease_expires_epoch':time.time()+1000}]})),patch('persona_personality_sweep.urllib.request.urlopen',side_effect=fetch):
                path=Path(folder)/'attempt.json'
                if finish=='stop':self.assertEqual(request_planner(self.suite,planner,path,deadline=time.time()+500),value)
                else:
                    with self.assertRaises(ValueError):request_planner(self.suite,planner,path,deadline=time.time()+500)
                posts=[r for r in calls if r.data]
                self.assertEqual(len(posts),1)
                request=json.loads(posts[0].data)
                self.assertNotIn('tools',request)
                self.assertEqual(request['model'],'persona-manager')
                self.assertNotIn('maxLength',json.dumps(request['response_format']))
                with self.assertRaises(ValueError):request_planner(self.suite,planner,path,deadline=time.time()+500)
                with self.assertRaises(ValueError):request_planner(self.suite,planner,path,deadline=time.time()+500,recover_rejected=True)
                self.assertEqual(len([r for r in calls if r.data]),1)
                self.assertEqual(json.loads(path.read_text())['state'],'validated' if finish=='stop' else 'failed_or_uncertain')
                if finish=='stop':
                    rejected=json.loads(path.read_text())
                    rejected.update(state='rejected_before_generation',rejection={'http_status':400,'error':'Failed to initialize samplers: failed to parse grammar'})
                    path.write_text(json.dumps(rejected))
                    self.assertEqual(request_planner(self.suite,planner,path,deadline=time.time()+500,recover_rejected=True),value)
                    self.assertEqual(json.loads(path.read_text())['prior_attempts'],[rejected])
    def test_john_wrong_gpu_and_private_evidence_rejected_before_call(self):
        with tempfile.TemporaryDirectory() as folder,patch('persona_personality_sweep.urllib.request.urlopen') as fetch:
            with self.assertRaises(ValueError):request_planner(self.suite,{'url':'http://example.com'},Path(folder)/'attempt.json',deadline=time.time()+100)
            fetch.assert_not_called()


if __name__=='__main__':unittest.main()
