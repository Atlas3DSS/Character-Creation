"""CPU fixtures for measured Pearson signs, exclusions and immutable provenance."""

import persona  # Make the flat, deployment-compatible source modules importable.
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

    def test_induced_graph_contains_measured_neighbor_edges_and_stable_positions(self):
        graph = self.index.query(0, 0)['graph']
        nodes = {node['id']: node for node in graph['nodes']}
        self.assertNotIn('L2N0', nodes)
        self.assertEqual(len(graph['edges']), len(nodes) * (len(nodes) - 1) // 2)
        self.assertFalse(graph['directed'])
        self.assertGreater(graph['explained_fraction'], 0)
        self.assertLessEqual(graph['explained_fraction'], 1 + 1e-12)
        for edge in graph['edges']:
            a, b = nodes[edge['source']], nodes[edge['target']]
            ia = self.coordinates.index((a['layer'], a['neuron']))
            ib = self.coordinates.index((b['layer'], b['neuron']))
            expected = np.corrcoef(self.arrays['values'][:, ia], self.arrays['values'][:, ib])[0, 1]
            self.assertAlmostEqual(edge['r'], expected)
            self.assertEqual(edge['n'], 16)
        self.assertEqual(graph, self.index.query(0, 0)['graph'])
        self.assertTrue(np.isfinite([n['position'] for n in nodes.values()]).all())
        self.assertTrue(all(len(n['position3d'])==3 for n in nodes.values()))
        self.assertGreaterEqual(graph['explained_fraction_3d'],graph['explained_fraction'])
        for coordinate, options in [((2, 0), {}), ((2, 2), {}), ((0, 0), {'corpus': 'skippy'})]:
            self.assertNotIn('graph', self.index.query(*coordinate, **options))

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

class CohortTests(unittest.TestCase):
    setUp=CoactivityTests.setUp
    publish=CoactivityTests.publish
    def overnight(self):
        from persona_neighbors import CoactivityCohorts
        self.chart['neurons']=self.chart['neurons'][:-1];self.map_path.write_text(json.dumps(self.chart));self.metadata['map_sha256']=hashlib.sha256(self.map_path.read_bytes()).hexdigest();self.publish()
        meta={**self.metadata,'runtime_identity':'c'*64,'prepared_identity':'d'*64,'runtime_identities':['c'*64],'prepared_identities':['d'*64]}
        arrays={k:v.copy() for k,v in self.arrays.items()};arrays['values']=arrays['values'].astype(np.float32);arrays['values'][:,1]*=-1;arrays['corpora']=np.array(['authored_emotion_contrast']*8+['book_diverse']*8)
        path=self.root/'results/persona_overnight_viewer.npz';np.savez_compressed(path,metadata=np.array(json.dumps(meta)),**arrays)
        records=[{'split':'discovery','id':str(identity),'tier':str(tier),'artifact_sha256':str(artifact),'messages':[{'role':'user','content':'Source '+str(i)}],'identities':{k:meta[k] for k in ('model_identity','runtime_identity','prepared_identity')},'duplicate_count':1} for i,(identity,tier,artifact) in enumerate(zip(arrays['prompt_ids'],arrays['corpora'],arrays['source_hashes']))]
        rec=path.with_name('persona_overnight_viewer_records.jsonl');rec.write_text(''.join(json.dumps(row)+'\n' for row in records))
        summary={k:meta[k] for k in ('model_identity','runtime_identity','prepared_identity','map_sha256')};summary.update(state='ready',unique_observations=16,discovery_observations=16,heldout_observations=0,imported_captures=16,pending_captures=4,total_tasks=20,completed_captures=16,npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),records_sha256=hashlib.sha256(rec.read_bytes()).hexdigest());path.with_suffix('.json').write_text(json.dumps(summary))
        return CoactivityCohorts(self.root,self.model),path,rec
    def test_default_and_historical_never_pool(self):
        index,_,_=self.overnight();new=index.query(0,0);old=index.query(0,0,cohort='historical')
        self.assertEqual(new['source']['cohort'],'overnight');self.assertEqual(old['source']['cohort'],'historical');self.assertEqual(new['source']['n_samples'],16);self.assertEqual(old['source']['n_samples'],16)
        self.assertIn((0,1),[(x['layer'],x['neuron']) for x in new['groups']['negative']]);self.assertIn((0,1),[(x['layer'],x['neuron']) for x in old['groups']['positive']])
    def test_tampered_records_fallback_and_explicit_failure(self):
        index,_,records=self.overnight();self.assertEqual(index.status()['active_cohort'],'overnight');records.write_text(records.read_text()+' ')
        self.assertEqual(index.status()['active_cohort'],'historical')
        with self.assertRaises(ValueError):index.query(0,0,cohort='overnight')
    def test_examples_bound_sorted_filtered_and_limited(self):
        index,_,_=self.overnight();result=index.examples(0,0,corpus='authored_emotion_contrast',limit=2)
        self.assertEqual(len(result['groups']['positive']),2);self.assertEqual(result['groups']['positive'][0]['messages'][0]['content'],'Source 7');self.assertTrue(all(r['tier']=='authored_emotion_contrast' for group in result['groups'].values() for r in group))
        with self.assertRaises(ValueError):index.examples(0,0,limit=9)
    def test_missing_overnight_uses_historical(self):
        from persona_neighbors import CoactivityCohorts
        index=CoactivityCohorts(self.root,self.model);self.assertEqual(index.query(0,0)['source']['cohort'],'historical');self.assertFalse(index.examples(0,0)['available'])

    def test_heldout_is_explicit_and_never_pooled(self):
        index,path,records=self.overnight()
        with np.load(path,allow_pickle=False) as archive:
            arrays={k:archive[k].copy() for k in archive.files if k!='metadata'}
            meta=json.loads(str(archive['metadata']))
        meta['split']='heldout';arrays['values'][:,1]*=-1
        arrays['prompt_ids']=np.array(['heldout-'+str(i) for i in range(16)])
        heldout=path.with_name('persona_overnight_viewer_heldout.npz')
        np.savez_compressed(heldout,metadata=np.array(json.dumps(meta)),**arrays)
        rows=[json.loads(line) for line in records.read_text().splitlines()]
        extra=[{**row,'split':'heldout','id':'heldout-'+str(i),'messages':[{'role':'user','content':'Heldout '+str(i)}]} for i,row in enumerate(rows)]
        records.write_text(''.join(json.dumps(row)+'\n' for row in rows+extra))
        summary=json.loads(path.with_suffix('.json').read_text());summary.update(unique_observations=32,heldout_observations=16,heldout_npz_sha256=hashlib.sha256(heldout.read_bytes()).hexdigest(),records_sha256=hashlib.sha256(records.read_bytes()).hexdigest());path.with_suffix('.json').write_text(json.dumps(summary))
        discovery=index.query(0,0);evaluation=index.query(0,0,cohort='overnight_heldout')
        self.assertEqual(discovery['source']['split'],'discovery');self.assertEqual(evaluation['source']['split'],'heldout')
        self.assertIn('filtered heldout prompts',evaluation['graph']['semantics']);self.assertNotIn('discovery',str(evaluation['source']['limitations']))
        self.assertEqual(discovery['source']['n_samples'],16);self.assertEqual(evaluation['source']['n_samples'],16)
        self.assertIn((0,1),[(r['layer'],r['neuron']) for r in evaluation['groups']['positive']])
        self.assertIn((0,1),[(r['layer'],r['neuron']) for r in discovery['groups']['negative']])
        for cohort,split in [('overnight','discovery'),('overnight_heldout','heldout')]:
            examples=index.examples(0,0,cohort=cohort)
            self.assertTrue(all(row['split']==split for group in examples['groups'].values() for row in group))
        # A held-out matrix mislabeled discovery must fail closed, never become the default.
        meta['split']='discovery';np.savez_compressed(heldout,metadata=np.array(json.dumps(meta)),**arrays)
        with self.assertRaises(ValueError):index.query(0,0,cohort='overnight_heldout')
        self.assertEqual(index.query(0,0)['source']['cohort'],'overnight')

    def test_empty_heldout_is_valid_but_has_no_correlation_evidence(self):
        index,path,records=self.overnight()
        with np.load(path,allow_pickle=False) as archive:
            arrays={k:archive[k].copy() for k in archive.files if k!='metadata'}
            meta=json.loads(str(archive['metadata']))
        meta['split']='heldout'
        for key in ('values','corpora','prompt_ids','source_hashes'):arrays[key]=arrays[key][:0]
        heldout=path.with_name('persona_overnight_viewer_heldout.npz');np.savez_compressed(heldout,metadata=np.array(json.dumps(meta)),**arrays)
        summary=json.loads(path.with_suffix('.json').read_text());summary['heldout_npz_sha256']=hashlib.sha256(heldout.read_bytes()).hexdigest();path.with_suffix('.json').write_text(json.dumps(summary))
        result=index.query(0,0,cohort='overnight_heldout')
        self.assertFalse(result['available']);self.assertEqual(result['source']['n_samples'],0)
        self.assertEqual(index.examples(0,0,cohort='overnight_heldout')['groups'],{'positive':[],'negative':[],'near_zero':[]})
        self.assertEqual(index.query(0,0)['source']['cohort'],'overnight')
