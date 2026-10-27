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


def capture_fixture():
    from test_persona_batch_capture_validate import fixture
    from persona_batch_capture_validate import recompute
    r,p,a=fixture()
    for c in p['cases']:
        c['prompt']=APPROVED_PROMPT;c['push']={'kind':'neuron','layer':32,'dose':.1}
    p['original_plan_canonical_sha256']='a'*64
    r['driver_sha256']=p['adapter_sha256'];r['artifact']={'filename':'persona_batch_capture_proof.npz','sha256':'e'*64,'bytes':10,'uncompressed_array_bytes':100,'compression':'np.savez_compressed lossless'}
    v=recompute(r,p,a,layers=2,hidden=4,intermediate=6)
    v.update(schema='character-creation.batch-capture-validation.v1',validator_sha256='f'*64,plan_sha256='f'*64,report_sha256='f'*64)
    return r,p,v


def test_capture_projection_preserves_all_counts_programs_and_drops_private_receipts():
    import json
    from persona_batch_release import project_capture
    r,p,v=capture_fixture()
    r['capture_override']={'path':'/mnt/private','sha256':'e'*64}
    r['lease_id']='private-lease';r['hostname']='private-host'
    r['runs']['four']['capture']['private_path']='/home/private'
    result=project_capture(p,r,v,{APPROVED_PROMPT});text=json.dumps(result)
    assert len(result['runs'])==16 and result['runs']['four']['capture']['counts']==[2]*4
    assert result['cases'][0]['program']['sites'][0]['nonzero']==[[0,1.0]]
    assert result['artifact']['filename']=='persona_batch_capture_proof.npz'
    assert result['automatic_production_rollout'] is False
    for secret in ('/mnt/private','/home/private','private-lease','private-host','mapping_steps'):assert secret not in text


def test_capture_projection_fails_closed_on_missing_validation_or_private_output():
    import pytest
    from persona_batch_release import project_capture
    r,p,v=capture_fixture()
    for key in ('verified','passed'):
        bad=deepcopy(v);bad[key]=False
        with pytest.raises(ValueError):project_capture(p,r,bad,{APPROVED_PROMPT})
    bad=deepcopy(v);bad['checks']['serial_B4_exact']=False
    with pytest.raises(ValueError):project_capture(p,r,bad,{APPROVED_PROMPT})
    r['runs']['four']['answers']=['/home/private']
    with pytest.raises(ValueError):project_capture(p,r,v,{APPROVED_PROMPT})


def test_capture_export_rejects_changed_frozen_plan_before_artifact_read(tmp_path):
    import json
    import pytest
    from persona_batch_release import export_capture
    results=tmp_path/'results';results.mkdir()
    r,p,v=capture_fixture()
    for name,value in [('persona_batch_capture_plan.json',p),('persona_batch_capture_proof.json',r),('persona_batch_capture_validation.json',v),('persona_batch_proof_plan.json',{})]:
        (results/name).write_text(json.dumps(value))
    with pytest.raises(ValueError,match='byte binding'):export_capture(tmp_path)
