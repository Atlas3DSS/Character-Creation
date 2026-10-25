
import persona  # Make the flat, deployment-compatible source modules importable.
import unittest
import numpy as np
from persona_trace import normalize_trace,pack_logits,unpack_logits,compare_logits

class TraceTests(unittest.TestCase):
    def test_selected_coordinate_bounds(self):
        valid={'id':'a'*32,'neurons':[{'layer':14,'neuron':8949}]}
        self.assertEqual(normalize_trace(valid),valid)
        for value in [{**valid,'neurons':valid['neurons']*2},{**valid,'id':'../oops'},
            {**valid,'neurons':[{'layer':0,'neuron':i} for i in range(33)]},
            {**valid,'neurons':[{'layer':64,'neuron':1}]}]:
            with self.assertRaises(ValueError):normalize_trace(value)

    def test_full_vocabulary_kl_and_log_probability_changes(self):
        a=pack_logits(np.array([2.,1.,-3.]));b=pack_logits(np.array([1.,2.,-3.]))
        np.testing.assert_array_equal(unpack_logits(a),[2.,1.,-3.])
        same=compare_logits(a,a,str)
        self.assertEqual(same['kl_baseline_to_intervention'],0)
        changed=compare_logits(a,b,str)
        self.assertGreater(changed['kl_baseline_to_intervention'],0)
        self.assertEqual(changed['increases'][0]['id'],1)
        self.assertAlmostEqual(changed['increases'][0]['delta_logp'],1)
        with self.assertRaises(ValueError):unpack_logits({**a,'count':2})

if __name__=='__main__':unittest.main()
