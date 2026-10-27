"""CPU-only evidence tests; these do not substitute for native capture qualification."""
import persona
from copy import deepcopy
import unittest
import numpy as np
from persona_batch_capture import Accumulator, mapping_expected, exact_capture, make_plan, PLANES, POLICY
from persona_vectors import canonical_digest

class CaptureTests(unittest.TestCase):
    def planes(self,value):
        return {k:np.full((2,4 if k=='mlp_input_last' else 3),value,np.float32) for k in PLANES}
    def test_consumed_tokens_mean_and_first_planes(self):
        a=Accumulator(1,2,3,4)
        a.add(0,10,42,7,self.planes(1));a.add(0,11,43,7,self.planes(3))
        out=a.finish({'generated':[[42,43,44]],'prompt_tokens':[10]})
        np.testing.assert_array_equal(out['_mean'],2);self.assertEqual(out['_sum'].dtype,np.float64)
        self.assertEqual(out['counts'],[2]);np.testing.assert_array_equal(out['_first_mlp_input_last'],1)
        self.assertNotEqual(out['step_plane_sha256'][0][0],out['step_plane_sha256'][0][1])
    def test_budget_one_has_undefined_zero_mean(self):
        out=Accumulator(1,2,3,4).finish({'generated':[[9]],'prompt_tokens':[8]})
        self.assertEqual(out['mean_defined'],[False]);self.assertFalse(out['_mean'].any())
    def test_missing_final_decode_and_prefill_rejected(self):
        for position,tokens in [(9,[42,43]),(10,[42,43,44])]:
            a=Accumulator(1,2,3,4);a.add(0,position,42,7,self.planes(1))
            with self.assertRaises(ValueError):a.finish({'generated':[tokens],'prompt_tokens':[10]})
    def test_generation_position_dtype_finite_guards(self):
        for pos,gen in [(12,7),(11,8)]:
            a=Accumulator(1,2,3,4);a.add(0,10,42,7,self.planes(1))
            with self.assertRaises(ValueError):a.add(0,pos,43,gen,self.planes(2))
        for value in [np.nan,np.inf]:
            with self.assertRaises(ValueError):Accumulator(1,2,3,4).add(0,0,1,1,self.planes(value))
    def evidence(self):
        before=np.array([[3,10,1,42,1,0,0],[4,11,1,43,2,0,0]])
        ev={'slot_live':[True,False],'slot_positions':[10,-1],'slot_token_ids':[42,-1],'slot_generations':[3,0],'slot_match_counts':[1,0],'mapping_error':[False],
            'token_live':[True,False],'invalid_tokens':[False,False],'token_slots':[0,-1],'token_positions':[10,-1],'token_ids':[42,-1],'token_generations':[3,0]}
        return before,{k:np.array(v) for k,v in ev.items()}
    def test_exact_mapping_and_reject_stale_or_duplicate(self):
        before,ev=self.evidence();self.assertEqual(mapping_expected(before,[0,1],ev),{0})
        for field,index,value in [('slot_live',1,True),('slot_generations',0,2),('slot_match_counts',0,2),('mapping_error',0,True),('token_ids',0,99),('invalid_tokens',1,True)]:
            bad=deepcopy(ev);bad[field][index]=value
            with self.assertRaises(ValueError):mapping_expected(before,[0,1],bad)
    def test_plan_preserves_original_programs_and_freezes_32(self):
        program={'phase':'all','sites':[]};old={'schema':'character-creation.batch-proof-plan.v1','cases':[{'program':program,'program_sha256':canonical_digest(program),'budget':64} for _ in range(4)]}
        plan=make_plan(old);self.assertEqual([c['budget'] for c in plan['cases']],[32]*4);self.assertEqual(old['cases'][0]['budget'],64)
        old['cases'][0]['program_sha256']='bad'
        with self.assertRaises(ValueError):make_plan(old)
    def test_capture_equality_includes_native_outputs(self):
        a=Accumulator(1,2,3,4);a.add(0,10,42,7,self.planes(1))
        x={'generated':[[42,43]],'prompt_tokens':[10],'_logits':np.ones((1,2)),'_hidden':np.ones((1,3))};x['capture']=a.finish(x)
        y=deepcopy(x);self.assertTrue(exact_capture(x,y,0,0));y['_logits'][0,0]=2;self.assertFalse(exact_capture(x,y,0,0))

if __name__=='__main__':unittest.main()

def test_prepare_factory_restored_even_when_preparation_fails(monkeypatch):
    import sys
    from types import ModuleType,SimpleNamespace
    from persona_batch_capture import prepare_adapter
    bootstrap=ModuleType('ignition.recipes.qwen38_dense.bootstrap')
    original=lambda *a,**k:None
    bootstrap.PreparedDecoder=original
    package=ModuleType('ignition.recipes.qwen38_dense');package.bootstrap=bootstrap
    capture=ModuleType('activation_capture');capture.RequestActivationCapture=object
    monkeypatch.setitem(sys.modules,'ignition.recipes.qwen38_dense',package)
    monkeypatch.setitem(sys.modules,'activation_capture',capture)
    def fail():
        assert bootstrap.PreparedDecoder is not original
        raise RuntimeError('expected')
    import pytest
    with pytest.raises(RuntimeError,match='expected'):prepare_adapter(fail)()
    assert bootstrap.PreparedDecoder is original

def test_prepare_rejects_overwriting_existing_capture(monkeypatch):
    import sys
    from types import ModuleType,SimpleNamespace
    from persona_batch_capture import prepare_adapter
    bootstrap=ModuleType('ignition.recipes.qwen38_dense.bootstrap');original=lambda *a,**k:None;bootstrap.PreparedDecoder=original
    package=ModuleType('ignition.recipes.qwen38_dense');package.bootstrap=bootstrap
    capture=ModuleType('activation_capture');capture.RequestActivationCapture=object
    monkeypatch.setitem(sys.modules,'ignition.recipes.qwen38_dense',package);monkeypatch.setitem(sys.modules,'activation_capture',capture)
    def prepare():return bootstrap.PreparedDecoder(None,None,SimpleNamespace(positions=SimpleNamespace(numel=lambda:4)),None,mode='decode',activation_capture=object())
    import pytest
    with pytest.raises(ValueError,match='already instrumented'):prepare_adapter(prepare)()
    assert bootstrap.PreparedDecoder is original
