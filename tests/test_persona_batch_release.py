import persona
from copy import deepcopy
import unittest
from persona_batch_release import project,RUNS,APPROVED_PROMPT
from persona_vectors import canonical_digest

class BatchReleaseTests(unittest.TestCase):
    def setUp(self):
        self.binding={'model_identity':'a'*64,'runtime_identity':'b'*64,'prepared_identity':'c'*64}
        program={'phase':'all','sites':[{'site':'layers.32.residual','vector':[0.,.25,0.,-.5]}]}
        cases=[dict(id=f'injection_{i}',prompt=APPROVED_PROMPT,seed=801+i,budget=64,push={'kind':'random','dose':.1,'layer':32},program=program,program_sha256=canonical_digest(program)) for i in range(4)]
        self.plan=dict(schema='character-creation.batch-proof-plan.v1',policy={'generation_budget':64},cases=cases,production_binding=self.binding,production_core_source_hashes={'decoder.py':'d'*64})
        self.proof=dict(schema='character-creation.batch-proof.v1',state='complete',passed=True,production_core_equal=True,binding=self.binding,production_binding=self.binding,production_core_source_hashes=self.plan['production_core_source_hashes'],
            runs={name:{'width':4,'answers':['Synthetic'],'private_path':'/home/private','lease_id':'private'} for name in RUNS},checks={'exact':True},serial_comparison=[{'kl':0.}],throughput={'wall_speedup':2.},artifact={'filename':'persona_batch_proof.npz','sha256':'e'*64,'bytes':10},
            prior_attempts=[{'state':'failed_pre_execution','failed_worker_manifest':{'path':'/mnt/private'},'lease_id':'secret'}])
        self.validation=dict(schema='character-creation.batch-proof-validation.v1',verified=True,passed=True,binding=self.binding,checks=self.proof['checks'],serial_comparison=self.proof['serial_comparison'],throughput=self.proof['throughput'],trace_validation={n:{'maximum_simultaneous_tokens':4} for n in RUNS},**{k:'f'*64 for k in ('driver_sha256','validator_sha256','plan_sha256','report_sha256')})
    def export(self):return project(self.plan,self.proof,self.validation,{APPROVED_PROMPT})
    def test_projection_drops_private_receipts_and_preserves_sparse_program(self):
        import json
        result=self.export();text=json.dumps(result)
        for private in ('/home/private','/mnt/private','lease_id','failed_worker_manifest'):self.assertNotIn(private,text)
        self.assertEqual(result['cases'][0]['program']['sites'][0]['nonzero'],[[1,.25],[3,-.5]])
        self.assertEqual(len(result['runs']),16);self.assertEqual(len(result['prior_events']),1)
        self.assertFalse(result['automatic_production_rollout'])
    def test_unverified_changed_or_incomplete_evidence_rejected(self):
        for field,value in [('verified',False),('passed',False),('throughput',{'wall_speedup':99.})]:
            old=self.validation;self.validation=dict(old,**{field:value})
            with self.assertRaises(ValueError):self.export()
            self.validation=old
        self.proof['runs'].pop('four')
        with self.assertRaises(ValueError):self.export()
    def test_authored_prompt_program_and_output_paths_checked(self):
        self.plan['cases'][0]['prompt']='private text'
        with self.assertRaises(ValueError):self.export()
        self.plan['cases'][0]['prompt']=APPROVED_PROMPT
        self.plan['cases'][0]['program_sha256']='0'*64
        with self.assertRaises(ValueError):self.export()
        self.plan['cases'][0]['program_sha256']=canonical_digest(self.plan['cases'][0]['program'])
        self.proof['runs']['four']['answers']=['/home/private']
        with self.assertRaises(ValueError):self.export()

if __name__=='__main__':unittest.main()
