"""Focused measured concept ranking, missing support, and binding checks."""
import persona
import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from persona_concepts import ConceptSearch


class ConceptTests(unittest.TestCase):
    def setUp(self):
        self.rows = {str(i): {'id': str(i), 'corpus': 'big_five', 'split': 'discovery',
            'category': 'inventory', 'prompt': 'Curious scientist' if i < 4 else 'Calm writer'} for i in range(8)}
        x = np.array([1., 2., 3., 4., -1., -2., -3., -4.])
        self.arrays = {'values': np.column_stack((x, -x, np.ones(8))),
                      'prompt_ids': np.array(list(self.rows)), 'corpora': np.array(['big_five'] * 8)}
        self.data = {'arrays': self.arrays, 'coordinates': [(0, 0), (0, 1), (1, 0)],
                     'metadata': {}, 'artifact_sha256': 'a'*64, 'map_sha256': 'b'*64}
        self.index = SimpleNamespace(lock=nullcontext(), model=SimpleNamespace(identity='test'), _load=lambda: self.data)
        self.search = ConceptSearch(self.index, lambda: (self.rows, {'corpus_sha256': 'c'*64}))

    def test_ranks_measured_signed_contrasts_and_collection(self):
        result = self.search.query('CURIOUS scientist')
        self.assertTrue(result['available'])
        self.assertEqual(len(result['candidates']), 2)  # constant candidate excluded
        expected = 5 / self.arrays['values'][:, 0].std(ddof=1)
        self.assertAlmostEqual(result['candidates'][0]['score'], expected)
        self.assertAlmostEqual(result['candidates'][1]['score'], -expected)
        self.assertAlmostEqual(result['graph']['edges'][0]['r'], -1)
        self.assertEqual(result['source']['matched_used_rows'], 4)
        self.assertEqual(result['source']['control_used_rows'], 4)

    def test_no_semantic_fallback_or_substring_match(self):
        for query in ('curiosity', 'curio', 'inventory'):
            self.assertEqual(self.search.query(query)['status'], 'insufficient_support')

    def test_unmeasured_matches_are_not_used_for_activation_support(self):
        self.rows['extra'] = dict(self.rows['0'], id='extra')
        result = self.search.query('curious')
        self.assertEqual(result['source']['matching_corpus_rows'], 5)
        self.assertEqual(result['source']['matching_measured_rows'], 4)
        self.assertEqual(result['source']['unmeasured_matches'], 1)

    def test_cross_corpus_confounds_cannot_supply_controls(self):
        self.arrays['corpora'] = np.array(['big_five'] * 4 + ['scotus'] * 4)
        for i in range(4, 8): self.rows[str(i)]['corpus'] = 'scotus'
        self.assertEqual(self.search.query('curious')['status'], 'insufficient_support')

    def test_invalid_observation_binding_is_rejected(self):
        self.rows['0']['split'] = 'heldout'
        self.assertEqual(self.search.query('curious')['status'], 'invalid_binding')

    def test_missing_activation_source_is_explicit(self):
        def missing(): raise FileNotFoundError('missing matrix')
        self.index._load = missing
        result = self.search.query('curious')
        self.assertFalse(result['available'])
        self.assertEqual(result['status'], 'missing_source')

    def test_unmeasured_adult_facet_search_keeps_evidence_visible(self):
        self.rows['adult'] = {'id': 'adult', 'corpus': 'adult_sexuality', 'split': 'discovery',
            'category': 'adult', 'prompt': 'An adult relationship question',
            'labels': {'prompt_design_facet': 'libido_intensity'}}
        result = self.search.query('libido', corpus='adult_sexuality')
        self.assertEqual(result['status'], 'insufficient_support')
        self.assertEqual(result['source']['matching_corpus_rows'], 1)
        self.assertEqual(result['source']['matching_measured_rows'], 0)
        self.assertFalse(result['examples'][0]['measured'])
        self.assertEqual(result['candidates'], [])

    def test_missing_matrix_still_returns_prompt_matches(self):
        def missing(): raise FileNotFoundError('missing matrix')
        self.index._load = missing
        self.assertEqual(len(self.search.query('curious')['examples']), 4)

    def test_invalid_query(self):
        for query in ('', '!', 'a' * 161):
            with self.assertRaises(ValueError): self.search.query(query)


if __name__ == '__main__': unittest.main()
