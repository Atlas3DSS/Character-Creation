import persona
from copy import deepcopy
import json
from pathlib import Path
import unittest
from persona_personality_review import validate_inputs,validate_judgments,request_for,summarize,SCORES,ISSUES
from persona_personality_probes import load_suite
from persona_personality_sweep import ARMS,SEEDS


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.suite=load_suite(Path(__file__).resolve().parents[1]/'data/persona_personality_probes.json')
        self.plan={'input_sha256':'a'*64,'suite_sha256':self.suite['suite_sha256'],'binding':{'runtime_identity':'b'*64},'jobs':[]}
        self.packet={'schema':'character-creation.personality-blind.v1','plan_sha256':'a'*64,'groups':[]}
        self.run={'plan_sha256':'a'*64,'binding':self.plan['binding'],'state':'complete','results':[]};self.key=[];self.judgments=[]
        for case in self.suite['cases']:
            for seed in SEEDS:
                g={'group':str(len(self.packet['groups'])),'prompt':case['prompt'],'fact_contract':case['fact_contract'],'observations':case['observations'],'answers':[]}
                for arm in ARMS:
                    code=f'{len(self.key):024x}';jobid=f'job-{code}'
                    g['answers'].append({'code':code,'answer':'Fixture answer '+code})
                    self.key.append({'code':code,'job_id':jobid,'case_id':case['id'],'family_id':case['pair_id'],'arm':arm,'seed':seed,'split':'discovery'})
                    self.plan['jobs'].append({'id':jobid,'case_id':case['id'],'arm':arm,'seed':seed})
                    self.run['results'].append({'job_id':jobid,'state':'complete','response':{'choices':[{'message':{'content':g['answers'][-1]['answer']}}]}})
                    self.judgments.append({'code':code,**{f:1 for f in SCORES},**{f:'uncertain' for f in ISSUES},'evidence':'Synthetic fixture, not a judgment.'})
                self.packet['groups'].append(g)
    def test_exact_authored_source_answer_and_runtime_binding(self):
        validate_inputs(self.packet,self.key,self.suite,self.plan,self.run)
        for field in ('prompt','fact_contract','answers'):
            altered=deepcopy(self.packet)
            if field=='answers':altered['groups'][0][field][0]['answer']='Substituted response'
            elif field=='prompt':altered['groups'][0][field]+=' Altered prompt'
            else:altered['groups'][0][field]={}
            with self.assertRaises(ValueError):validate_inputs(altered,self.key,self.suite,self.plan,self.run)
        altered=deepcopy(self.run);altered['binding']={'runtime_identity':'other'}
        with self.assertRaises(ValueError):validate_inputs(self.packet,self.key,self.suite,self.plan,altered)
    def test_judge_never_receives_arm_key_or_execution_tools(self):
        request=request_for(self.packet['groups'][0]);payload=json.loads(request['messages'][1]['content'])
        self.assertEqual(payload,self.packet['groups'][0]);self.assertNotIn('tools',request)
        self.assertNotIn('arm',payload);self.assertEqual(request['max_tokens'],4096)
        self.assertNotIn('maxLength',json.dumps(request['response_format']))
    def test_coverage_bounds_and_descriptive_aggregation(self):
        group=self.packet['groups'][0];rows=self.judgments[:6]
        validate_judgments(group,{'judgments':rows})
        for bad in (rows[:-1],rows+[rows[0]],[dict(rows[0],warmth=True)]+rows[1:],[dict(rows[0],coherence='safe')]+rows[1:]):
            with self.assertRaises(ValueError):validate_judgments(group,{'judgments':bad})
        result=summarize(self.packet,self.key,self.judgments)
        self.assertEqual(len(result),72);self.assertTrue(all(r['complete']==4 for r in result))
        self.assertTrue(all(r['mean_scores']['hostility']==1 for r in result))
        with self.assertRaises(ValueError):summarize(self.packet,self.key,self.judgments[:-1])

if __name__=='__main__':unittest.main()
