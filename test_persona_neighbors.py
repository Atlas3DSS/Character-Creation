"""CPU fixtures for measured Pearson signs, exclusions and immutable provenance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from persona_explorer import handler
from persona_neighbors import CoactivityIndex, MEASUREMENT, SCHEMA
from persona_vectors import ModelSpec


class CoactivityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); (self.root / 'results').mkdir()
        self.model = ModelSpec('test/qwen', 'a' * 40, 'test', 3, 4, 4)
        self.coordinates = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        self.chart = {'schema': 'character-creation.map.v1', 'model_identity': self.model.identity,
                      'neurons': [{'layer': layer, 'neuron': neuron} for layer, neuron in self.coordinates + [(2, 2)]]}
        self.map_path = self.root / 'results/persona_map.json'
        self.map_path.write_text(json.dumps(self.chart))
        self.path = self.root / 'results/persona_coactivity.npz'
        x = np.tile(np.arange(8, dtype=np.float64) - 3.5, 2)
        orthogonal = x * x; orthogonal -= orthogonal.mean()
        moderate = .5 * x / np.linalg.norm(x) + np.sqrt(.75) * orthogonal / np.linalg.norm(orthogonal)
        self.metadata = {'schema': SCHEMA, 'model_identity': self.model.identity,
            'map_sha256': hashlib.sha256(self.map_path.read_bytes()).hexdigest(), 'split': 'discovery',
            'measurement': MEASUREMENT, 'unsteered': True,
            'selection': 'Unique original discovery baseline prompts', 'runtime_identities': ['b' * 64],
            'limitations': ['Small corpus slices are exploratory.']}
        self.arrays = {'layers': np.array([item[0] for item in self.coordinates], dtype=np.int64),
            'neurons': np.array([item[1] for item in self.coordinates], dtype=np.int64),
            'values': np.column_stack([x, 2 * x + 3, -3 * x + 2, orthogonal, np.ones(16), moderate]),
            'corpora': np.array(['big_five'] * 8 + ['scotus'] * 8),
            'prompt_ids': np.array([f'prompt-{i}' for i in range(16)]),
            'source_hashes': np.array([hashlib.sha256(str(i).encode()).hexdigest() for i in range(16)])}
        self.publish()
        self.index = CoactivityIndex(self.root, self.model)

    def publish(self, arrays=None, metadata=None):
        np.savez_compressed(self.path, metadata=np.array(json.dumps(self.metadata if metadata is None else metadata)),
                            **(self.arrays if arrays is None else arrays))

    def test_signed_strong_moderate_and_weak_correlations_match_measured_covariance(self):
        result = self.index.query(0, 0)
        self.assertTrue(result['available']); self.assertEqual(result['status'], 'ready')
        positive, negative, weak = (result['groups'][key] for key in ('positive', 'negative', 'weak'))
        self.assertEqual([(row['layer'], row['neuron']) for row in positive], [(0, 1), (2, 1)])
        self.assertEqual(positive[0]['strength'], 'strong'); self.assertAlmostEqual(positive[0]['r'], 1.)
        self.assertEqual(positive[1]['strength'], 'moderate'); self.assertAlmostEqual(positive[1]['r'], .5)
        self.assertEqual([(row['layer'], row['neuron']) for row in negative], [(1, 0)])
        self.assertAlmostEqual(negative[0]['r'], -1.)
        self.assertEqual([(row['layer'], row['neuron']) for row in weak], [(1, 1)])
        self.assertAlmostEqual(weak[0]['r'], 0.); self.assertEqual(weak[0]['strength'], 'weak')
        for row in positive + negative + weak:
            coordinate = (row['layer'], row['neuron'])
            self.assertNotIn(coordinate, [(0, 0), (2, 0)])
            measured = np.corrcoef(self.arrays['values'][:, 0], self.arrays['values'][:, self.coordinates.index(coordinate)])[0, 1]
            self.assertAlmostEqual(row['r'], measured)
            self.assertEqual(row['n'], 16)
        self.assertEqual(result['source']['artifact_sha256'], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(result['source']['map_sha256'], self.metadata['map_sha256'])
        self.assertEqual(result['source']['runtime_identities'], ['b' * 64])
        self.assertIn('Small corpus slices are exploratory.', result['source']['limitations'])
        self.assertTrue(any('not wiring' in note for note in result['source']['limitations']))

    def test_corpus_and_layer_filters_use_the_selected_rows_and_exclude_target(self):
        self.arrays['values'][8:, 1] = -self.arrays['values'][8:, 0]
        self.publish()
        first = self.index.query(0, 0, corpus='big_five', scope='same_layer')
        second = self.index.query(0, 0, corpus='scotus', scope='same_layer')
        self.assertEqual(first['source']['n_samples'], 8); self.assertEqual(second['source']['n_samples'], 8)
        self.assertEqual(first['groups']['positive'][0]['neuron'], 1)
        self.assertAlmostEqual(first['groups']['positive'][0]['r'], 1.)
        self.assertEqual(second['groups']['negative'][0]['neuron'], 1)
        self.assertAlmostEqual(second['groups']['negative'][0]['r'], -1.)
        crossed = self.index.query(0, 0, scope='cross_layer')
        self.assertTrue(all(row['layer'] != 0 for group in crossed['groups'].values() for row in group))
        limited = self.index.query(0, 0, limit=1)
        self.assertTrue(all(len(group) <= 1 for group in limited['groups'].values()))

    def test_constant_unobserved_missing_and_insufficient_sources_never_fabricate_neighbors(self):
        for target, options, status in [((2, 0), {}, 'target_constant'), ((2, 2), {}, 'target_unobserved'),
                ((0, 0), {'corpus': 'skippy'}, 'insufficient_samples')]:
            result = self.index.query(*target, **options)
            self.assertFalse(result['available']); self.assertEqual(result['status'], status)
            self.assertTrue(all(not rows for rows in result['groups'].values()))
        truncated = {name: value[:7] if name in ('values', 'corpora', 'prompt_ids', 'source_hashes') else value
                     for name, value in self.arrays.items()}
        self.publish(truncated)
        result = self.index.query(0, 0)
        self.assertEqual(result['status'], 'insufficient_samples'); self.assertEqual(result['source']['n_samples'], 7)
        self.path.unlink()
        self.assertEqual(self.index.query(0, 0)['status'], 'missing_source')

    def test_duplicate_prompts_and_candidates_are_rejected_instead_of_reweighted(self):
        duplicate = deepcopy(self.arrays); duplicate['prompt_ids'][1] = duplicate['prompt_ids'][0]
        self.publish(duplicate)
        result = self.index.query(0, 0)
        self.assertEqual(result['status'], 'invalid_source'); self.assertIn('Duplicate prompt', result['message'])
        duplicate = deepcopy(self.arrays); duplicate['neurons'][1] = duplicate['neurons'][0]
        self.publish(duplicate)
        result = self.index.query(0, 0)
        self.assertEqual(result['status'], 'invalid_source'); self.assertIn('unique observed', result['message'])

    def test_model_map_measurement_split_and_source_hash_bindings_fail_closed(self):
        for field, value in [('schema', 'unknown'), ('model_identity', 'other'), ('map_sha256', '0' * 64),
                             ('measurement', 'after_decode'), ('split', 'heldout'), ('unsteered', False)]:
            self.publish(metadata={**self.metadata, field: value})
            with self.subTest(field=field):
                result = self.index.query(0, 0)
                self.assertEqual(result['status'], 'invalid_source'); self.assertFalse(result['available'])
        broken = deepcopy(self.arrays); broken['source_hashes'][0] = 'bad-source-hash'
        self.publish(broken)
        self.assertIn('artifact SHA256', self.index.query(0, 0)['message'])
        chart = {**self.chart, 'model_identity': 'different-model'}
        self.map_path.write_text(json.dumps(chart))
        self.publish(metadata={**self.metadata, 'map_sha256': hashlib.sha256(self.map_path.read_bytes()).hexdigest()})
        self.assertIn('another model', self.index.query(0, 0)['message'])

    def test_finite_shapes_supported_corpora_and_observed_coordinates_are_required(self):
        variants = []
        bad = deepcopy(self.arrays); bad['values'][0, 0] = np.nan; variants.append(bad)
        bad = deepcopy(self.arrays); bad['values'][0, 0] = np.inf; variants.append(bad)
        bad = deepcopy(self.arrays); bad['values'] = bad['values'][:, :-1]; variants.append(bad)
        bad = deepcopy(self.arrays); bad['corpora'][0] = 'unknown'; variants.append(bad)
        bad = deepcopy(self.arrays); bad['layers'] = bad['layers'].astype(float); variants.append(bad)
        bad = deepcopy(self.arrays); bad['neurons'][-1] = 3; variants.append(bad)
        bad = deepcopy(self.arrays); bad['prompt_ids'] = bad['prompt_ids'][:-1]; variants.append(bad)
        bad = deepcopy(self.arrays); bad['extra'] = np.array('unregistered'); variants.append(bad)
        for number, arrays in enumerate(variants):
            self.publish(arrays)
            with self.subTest(number=number): self.assertEqual(self.index.query(0, 0)['status'], 'invalid_source')

    def test_content_identity_caches_statistics_and_invalidates_on_source_or_map_change(self):
        before = self.index.query(0, 0)
        statistics = self.index.statistics['all']
        with patch('persona_neighbors.np.load', side_effect=AssertionError('must reuse compact cache')):
            self.assertTrue(self.index.query(0, 0, scope='cross_layer', limit=1)['available'])
            self.assertIs(self.index.statistics['all'], statistics)
        self.arrays['values'][:, 1] = -self.arrays['values'][:, 0]
        self.publish()
        after = self.index.query(0, 0)
        self.assertNotEqual(before['source']['artifact_sha256'], after['source']['artifact_sha256'])
        self.assertIsNot(self.index.statistics['all'], statistics)
        self.assertIn((0, 1), [(row['layer'], row['neuron']) for row in after['groups']['negative']])
        self.map_path.write_text(json.dumps(self.chart, indent=2))
        self.assertEqual(self.index.query(0, 0)['status'], 'invalid_source')
        with patch('persona_neighbors.np.load', side_effect=AssertionError('invalid identity is cached too')):
            self.assertEqual(self.index.query(0, 0)['status'], 'invalid_source')

    def test_query_bounds_and_http_route_preserve_explicit_unavailable_status(self):
        for arguments in [(-1, 0, {}), (0, 4, {}), (True, 0, {}), (0, 0, {'corpus': 'other'}),
                          (0, 0, {'scope': 'directed'}), (0, 0, {'limit': 0}), (0, 0, {'limit': 51})]:
            layer, neuron, kwargs = arguments
            with self.subTest(arguments=arguments), self.assertRaises(ValueError): self.index.query(layer, neuron, **kwargs)
        request = object.__new__(handler(SimpleNamespace(neighbors=self.index)))
        request.send = Mock(); request.path = '/api/neighbors?layer=0&neuron=0&corpus=skippy&scope=all&limit=12'
        request.do_GET()
        result = request.send.call_args.args[0]
        self.assertEqual(result['status'], 'insufficient_samples'); self.assertEqual(result['source']['n_samples'], 0)
        request.path = '/api/neighbors?layer=0&neuron=0&limit=-1'; request.do_GET()
        self.assertEqual(request.send.call_args.args[1], 400)


if __name__ == '__main__':
    unittest.main()
