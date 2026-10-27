"""CPU checks for the finite native concurrency proof; no CUDA substitute."""
import persona
from types import SimpleNamespace
import unittest
import numpy as np
from persona_batch_probe import allocation,select_graphs,metrics,POLICY
from persona_probe import argument_parser,reviewed_preparation_options


class BatchProofTests(unittest.TestCase):
    def test_disjoint_pages_cover_every_private_budget(self):
        pages=allocation([120,250,100,500],[64,64,1,64],32,4096)
        self.assertEqual([len(x) for x in pages],[2,3,1,5])
        flat=[p for row in pages for p in row]
        self.assertEqual(len(flat),len(set(flat)));self.assertEqual(flat,list(range(11)))
    def test_physical_capacity_is_aggregate_not_per_request(self):
        with self.assertRaises(ValueError):allocation([100]*4,[64]*4,7,4096)
        with self.assertRaises(ValueError):allocation([4090],[64],100,4096)
        with self.assertRaises(ValueError):allocation([100],[0],32,4096)
    def prepared(self):
        signatures={f'decode{n}':SimpleNamespace(kind='decode',batch_capacity=n,speculation_enabled=False) for n in (1,2,4)}
        return SimpleNamespace(signatures=signatures,resources=[{n:object() for n in signatures}])
    def test_selects_actual_four_request_decode(self):
        p=self.prepared();names,_=select_graphs(p);self.assertEqual(names[4],'decode4')
        p.signatures['decode4'].speculation_enabled=True
        with self.assertRaises(ValueError):select_graphs(p)
    def test_missing_or_ambiguous_catalog_rejected(self):
        p=self.prepared();del p.signatures['decode4']
        with self.assertRaises(ValueError):select_graphs(p)
        p=self.prepared();p.resources.append(p.resources[0])
        with self.assertRaises(ValueError):select_graphs(p)
    def test_distribution_metric_is_shift_invariant(self):
        a=np.array([1.,3.,-1.]);m=metrics(a,a+1000)
        self.assertLess(abs(m['kl']),1e-12);self.assertTrue(m['same_top1'])
        self.assertGreater(metrics(a,np.array([3.,1.,-1.]))['kl'],0)
    def test_standard_worker_remains_serial_and_proof_policy_frozen(self):
        self.assertEqual(reviewed_preparation_options()['max_active'],4)
        self.assertEqual(POLICY['minimum_first_decode_top1_agreement'],1.)
        args=argument_parser().parse_args(['run','--config','c','--flare','f','--checkpoint','m','--heliopause','h','--shard','5','--compile-workers','86','--port','8935'])
        self.assertFalse(hasattr(args,'proof_plan'));self.assertEqual(args.physical_pages,32)


if __name__=='__main__':unittest.main()
