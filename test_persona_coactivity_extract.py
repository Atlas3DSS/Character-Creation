import copy
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from persona_coactivity_extract import (records, selected, read_candidate_values, file_hash,
    published_corpus, registry_records, unique_rows, prompt_key, preserve_previous, write_matrix)
from persona_coactivity_extract import duplicate_audit
from persona_coactivity_extract import matrix_snapshot, previous_records
from persona_vectors import ModelSpec, canonical_digest


class CoactivityExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = ModelSpec('fixture', 'a' * 40, 'test', 2, 3, 4)
        self.identity = dict(model_identity=self.model.identity, prepared_identity='b' * 64,
                             runtime_identity='c' * 64)

    def example(self, prompt_id='prompt-1', group='group-1'):
        prompt = dict(id=prompt_id, pair_id=group, corpus='skippy', split='discovery',
                      mutation={'kind': 'original'}, labels={}, prompt='Synthetic passage', system='',
                      category='synthetic_fixture', max_tokens=2048, source_sha256='e' * 64)
        job = dict(operator_coverage=True, arm='baseline', pushes=[], measurement='prefill',
                   model_identity=self.model.identity, prompt=prompt)
        job['id'] = canonical_digest(job)
        artifact = {**self.identity, 'path': str(self.root / 'capture.npz'), 'request_sha256': 'd' * 64,
                    'stage': 'final_prefill', 'operation': 'prefill_observation', 'sha256': 'f' * 64}
        request = dict(messages=[{'role': 'user', 'content': prompt['prompt']}],
                       observation_only=True, max_tokens=1,
                       metadata={'job_id': job['id'], 'split': 'discovery', 'arm': 'baseline'})
        response = {'activation_artifact': artifact, 'provenance_artifact': {'path': str(self.root / 'record.json')}}
        result = dict(complete=True, complete_observation=True, error=None, provenance={'pushes': []},
                      request=request, response=response, activation_artifact=artifact,
                      endpoint={**self.identity})
        return job, result

    def database(self, rows):
        path = self.root / 'manager.sqlite'
        with sqlite3.connect(path) as db:
            db.executescript('CREATE TABLE metadata(key,value); CREATE TABLE jobs(id,body,state); '
                             'CREATE TABLE attempts(job_id,number,result,state);')
            db.execute('INSERT INTO metadata VALUES (?,?)', ('identity', json.dumps(self.identity)))
            for job, result in rows:
                db.execute('INSERT INTO jobs VALUES (?,?,?)', (job['id'], json.dumps(job), 'complete'))
                db.execute('INSERT INTO attempts VALUES (?,?,?,?)', (job['id'], 1, json.dumps(result), 'complete'))
        return path

    def test_selection_excludes_heldout_mutations_profiles_and_interventions(self):
        job, _ = self.example()
        self.assertTrue(selected(job))
        for key, value in [('operator_coverage', False), ('arm', 'positive'), ('pushes', [{}]), ('measurement', 'rollout')]:
            changed = copy.deepcopy(job); changed[key] = value
            self.assertFalse(selected(changed), key)
        for key, value in [('split', 'heldout'), ('mutation', {'kind': 'instruction_negation'}),
                           ('labels', {'profile': 'openness_high'})]:
            changed = copy.deepcopy(job); changed['prompt'][key] = value
            self.assertFalse(selected(changed), key)

    @patch('persona_coactivity_extract.response_complete', return_value=True)
    def test_records_bind_original_prompt_and_reject_duplicate_source_groups(self, _):
        path = self.database([self.example()])
        self.assertEqual(len(records(path, self.model)[0]), 1)
        job, result = self.example('prompt-2')
        with sqlite3.connect(path) as db:
            db.execute('INSERT INTO jobs VALUES (?,?,?)', (job['id'], json.dumps(job), 'complete'))
            db.execute('INSERT INTO attempts VALUES (?,?,?,?)', (job['id'], 1, json.dumps(result), 'complete'))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            records(path, self.model)

    @patch('persona_coactivity_extract.response_complete', return_value=True)
    def test_records_reject_intervention_payload_even_with_baseline_label(self, _):
        job, result = self.example()
        result['request']['steering'] = {'sites': [{'site': 'bad'}]}
        path = self.database([(job, result)])
        with self.assertRaisesRegex(ValueError, 'Request'):
            records(path, self.model)

    @patch('persona_coactivity_extract.verify_diagnostic_artifacts')
    def test_arrays_are_selected_by_coordinate_and_tampering_is_rejected(self, _):
        job, result = self.example()
        artifact = result['activation_artifact']; path = Path(artifact['path'])
        data = np.arange(8, dtype=np.float32).reshape(2, 4)
        np.savez(path, **{k: np.array(v) for k, v in artifact.items() if k != 'path'}, mlp_input_last=data)
        artifact['sha256'] = file_hash(path)
        actual = read_candidate_values((job, result), self.model, np.array([1, 0]), np.array([2, 3]), self.root)
        np.testing.assert_array_equal(actual, [6., 3.])
        with path.open('ab') as stream: stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'digest'):
            read_candidate_values((job, result), self.model, np.array([1]), np.array([2]), self.root)

    def registry(self, mode='prefill', name='entry', prompt=None):
        job, _ = self.example()
        prompt = prompt or job['prompt']
        request = dict(messages=([{'role': 'system', 'content': prompt['system']}] if prompt.get('system') else []) +
            [{'role': 'user', 'content': prompt['prompt']}], max_tokens=1 if mode == 'prefill' else 2048,
            metadata={'job_id': job['id'], 'split': 'discovery', 'arm': 'baseline'})
        if mode == 'prefill': request['observation_only'] = True
        operation = 'prefill_observation' if mode == 'prefill' else 'rollout'
        artifact = {**self.identity, 'path': str(self.root / (name + '.npz')),
            'request_sha256': canonical_digest(request), 'stage': 'final_prefill', 'operation': operation}
        np.savez(artifact['path'], **{key: np.array(value) for key, value in artifact.items() if key != 'path'},
                 mlp_input_last=np.arange(8, dtype=np.float32).reshape(2, 4))
        artifact['sha256'] = file_hash(artifact['path'])
        response = {**self.identity, 'operation': operation, 'activation_artifact': artifact,
            'choices': [{'finish_reason': 'stop', 'message': {'content': 'Complete fixture answer'}}],
            'diagnostics': {'prefill_observation_complete': mode == 'prefill', 'complete_final_answer': mode == 'rollout', 'unexpected_thinking': False}}
        report = {**self.identity, 'operation': operation, 'request': request, 'request_sha256': canonical_digest(request),
            'response': response, 'activation_artifact': artifact, 'status': 'complete', 'state_restored': True,
            'worker_manifest': {'runtime_identity': self.identity['runtime_identity']}}
        path = self.root / (name + '.json')
        path.write_text(json.dumps(report))
        pointer = {**self.identity, 'operation': operation, 'path': str(path), 'sha256': file_hash(path), 'request_sha256': canonical_digest(request)}
        entry = {**self.identity, 'job_id': job['id'], 'prompt_id': prompt['id'], 'corpus': prompt['corpus'],
            'pair_id': prompt['pair_id'], 'source_sha256': prompt['source_sha256'], 'source_row_sha256': canonical_digest(prompt),
            'labels': prompt['labels'], 'mutation': prompt['mutation'], 'original': True, 'coactivity_original_eligible': True,
            'candidate_registry_only': True, 'heldout_used': False, 'arm': 'baseline', 'corpus_sha256': 'f' * 64,
            'source_group': {'corpus': prompt['corpus'], 'pair_id': prompt['pair_id'], 'split': 'discovery'},
            'exact_prompt_sha256': canonical_digest([prompt.get('system', ''), prompt['prompt']]),
            'prompt_sha256': canonical_digest([' '.join(prompt.get('system', '').split()), ' '.join(prompt['prompt'].split())]),
            'identity': prompt_key(prompt, self.identity), 'artifact': artifact, 'provenance_artifact': pointer,
            'measurement': mode, 'target_values': [999999.]}
        registry = self.root / (name + '.jsonl')
        registry.write_text(json.dumps(entry) + '\n')
        return registry, entry, prompt

    @patch('persona_coactivity_extract.response_complete', return_value=True)
    def test_published_append_preserves_old_record_binding_and_rejects_rewrite(self, _):
        (self.root / 'data').mkdir()
        job, result = self.example(); database = self.database([(job, result)])
        path = self.root / 'data/persona_manager_corpus.jsonl'
        def publish(rows):
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            path.with_name('persona_manager_corpus_manifest.json').write_text(json.dumps({'status': 'complete', 'output_sha256': file_hash(path)}))
        publish([job['prompt']]); before = file_hash(path)
        extra = {**job['prompt'], 'id': 'new-synthetic', 'pair_id': 'new-group', 'prompt': 'A new discovery scenario.'}
        publish([job['prompt'], extra]); corpus, current = published_corpus(self.root)
        self.assertNotEqual(before, current)
        self.assertEqual(len(records(database, self.model, corpus)[0]), 1)
        changed = {**job['prompt'], 'labels': {'profile': 'different'}}
        publish([changed, extra])
        with self.assertRaisesRegex(ValueError, 'immutable corpus row'):
            records(database, self.model, published_corpus(self.root)[0])

    def test_registry_verifies_real_reports_and_deduplicates_modes_without_using_claimed_values(self):
        first, _, prompt = self.registry()
        second, _, _ = self.registry(mode='rollout', name='rollout')
        rows, sources = registry_records([first, second], self.model, {prompt['id']: prompt}, self.identity, self.root)
        self.assertEqual(len(rows), 2); self.assertEqual(len(sources), 2)
        unique, discarded = unique_rows(rows, self.identity)
        self.assertEqual((len(unique), discarded), (1, 1))
        values = read_candidate_values(unique[0], self.model, np.array([1, 0]), np.array([2, 3]), self.root)
        np.testing.assert_array_equal(values, [6., 3.])
        # A genuine completed rollout can contribute its prefill plane if unique.
        np.testing.assert_array_equal(read_candidate_values(rows[1], self.model, np.array([0]), np.array([2]), self.root), [2.])

    def test_deterministic_dedup_prefers_published_representatives_then_smallest_id(self):
        first = self.example('old-1', 'one')
        second = self.example('old-2', 'two')
        new = self.example('new', 'three')
        rows, discarded = unique_rows([new, first, second, first], self.identity, {'old-1', 'old-2'})
        self.assertEqual([job['prompt']['id'] for job, _ in rows], ['old-1'])
        self.assertEqual(discarded, 3)
        changed = copy.deepcopy(new); changed[0]['prompt']['system'] = 'Different actual conditioning'
        self.assertEqual(len(unique_rows([first, changed], self.identity)[0]), 2)

    def test_registry_rejects_untrusted_binding_intervention_and_incomplete_cleanup(self):
        path, original, prompt = self.registry()
        for changes in ({'runtime_identity': '0' * 64}, {'source_row_sha256': '0' * 64},
                        {'original': False}, {'arm': 'positive'}, {'exact_prompt_sha256': '0' * 64}):
            with self.subTest(changes=changes):
                path.write_text(json.dumps({**original, **changes}) + '\n')
                with self.assertRaises(ValueError): registry_records([path], self.model, {prompt['id']: prompt}, self.identity, self.root)
        for change in ('steering', 'cleanup'):
            path, entry, prompt = self.registry(name=change)
            report_path = Path(entry['provenance_artifact']['path']); report = json.loads(report_path.read_text())
            if change == 'steering': report['request']['steering'] = {'sites': []}
            else: report['state_restored'] = False
            report_path.write_text(json.dumps(report)); entry['provenance_artifact']['sha256'] = file_hash(report_path)
            path.write_text(json.dumps(entry) + '\n')
            with self.subTest(change=change), self.assertRaises(ValueError):
                registry_records([path], self.model, {prompt['id']: prompt}, self.identity, self.root)

    def test_matrix_refresh_preserves_published_values_history_and_atomic_destination(self):
        metadata = {'schema': 'character-creation.coactivity.v1', 'model_identity': self.model.identity,
            'map_sha256': 'f' * 64, 'split': 'discovery', 'measurement': 'final_prefill_mlp_input_last',
            'unsteered': True, 'runtime_identities': [self.identity['runtime_identity']],
            'prepared_identities': [self.identity['prepared_identity']]}
        original = dict(metadata=np.array(json.dumps(metadata)), layers=np.array([0]), neurons=np.array([1]),
            values=np.array([[1.]], dtype=np.float32), corpora=np.array(['skippy']), prompt_ids=np.array(['one']), source_hashes=np.array(['a' * 64]))
        path = self.root / 'matrix.npz'; write_matrix(path, original); initial = path.read_bytes()
        expanded = {**original, 'values': np.array([[1.], [2.]], dtype=np.float32), 'corpora': np.array(['skippy', 'big_five']),
            'prompt_ids': np.array(['one', 'two']), 'source_hashes': np.array(['a' * 64, 'b' * 64])}
        history = preserve_previous(path, expanded)
        self.assertEqual(history[0]['rows'], 1); self.assertEqual(history[0]['artifact_sha256'], file_hash(path))
        with self.assertRaises(FileExistsError): write_matrix(path, expanded)
        self.assertEqual(path.read_bytes(), initial)
        wrong = {**expanded, 'values': np.array([[9.], [2.]], dtype=np.float32)}
        with self.assertRaisesRegex(ValueError, 'numeric observations'): preserve_previous(path, wrong)
        with self.assertRaisesRegex(ValueError, 'drop'): preserve_previous(path, {**original, 'prompt_ids': np.array(['other'])})
        write_matrix(path, expanded, replace=True)
        with np.load(path) as data: self.assertEqual(data['values'].shape, (2, 1))

    def test_dedup_transition_records_excluded_sources_and_checks_identical_prompt_activations(self):
        first, second = self.example('one', 'group-1'), self.example('two', 'group-2')
        second[1]['activation_artifact']['sha256'] = 'a' * 64
        values = {'one': np.array([1., 2.], dtype=np.float32), 'two': np.array([1., 2.], dtype=np.float32)}
        read = lambda row: values[row[0]['prompt']['id']]
        audit = duplicate_audit([first, second], [first], values['one'][None], self.identity, read, 2)
        self.assertEqual(len(audit), 1); self.assertTrue(audit[0]['prompt_bytes_identical'])
        self.assertTrue(audit[0]['candidate_vectors_bit_equal'])
        metadata = {'schema': 'character-creation.coactivity.v1', 'model_identity': self.model.identity,
            'map_sha256': 'e' * 64, 'split': 'discovery', 'measurement': 'final_prefill_mlp_input_last',
            'unsteered': True, 'runtime_identities': [self.identity['runtime_identity']],
            'prepared_identities': [self.identity['prepared_identity']]}
        prior = dict(metadata=np.array(json.dumps(metadata)), layers=np.array([0, 0]), neurons=np.array([1, 2]),
            values=np.stack([values['one'], values['two']]), corpora=np.array(['skippy', 'skippy']),
            prompt_ids=np.array(['one', 'two']), source_hashes=np.array(['f' * 64, 'a' * 64]))
        path = self.root / 'prior.npz'; write_matrix(path, prior)
        current = {**prior, **{key: prior[key][:1] for key in ('values', 'corpora', 'prompt_ids', 'source_hashes')}}
        history = preserve_previous(path, current, audit)
        self.assertEqual(history[-1]['deduplication_transition'][0]['excluded_prompt_id'], 'two')
        self.assertEqual(history[-1]['artifact_sha256'], file_hash(path))
        self.assertTrue(history[-1]['raw_captures_preserved'])
        values['two'] = np.array([1., 3.], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'Byte-identical'):
            duplicate_audit([first, second], [first], values['one'][None], self.identity, read, 2)
        second[0]['prompt']['prompt'] = '  Synthetic   passage  '
        audit = duplicate_audit([first, second], [first], values['one'][None], self.identity, read, 2)
        self.assertFalse(audit[0]['prompt_bytes_identical']); self.assertFalse(audit[0]['candidate_vectors_bit_equal'])
        self.assertEqual(audit[0]['maximum_abs_candidate_difference'], 1.)

    def test_bounded_stdin_snapshot_and_registry_only_reuse_bind_content_addressed_source(self):
        prompt = self.example()[0]['prompt']; prompt.pop('id')
        prompt['id'] = canonical_digest(prompt)
        chart = {'model_identity': self.model.identity, 'neurons': [{'layer': 0, 'neuron': 1}]}
        metadata = {'schema': 'character-creation.coactivity.v1', 'model_identity': self.model.identity,
            'map_sha256': 'e' * 64, 'split': 'discovery', 'measurement': 'final_prefill_mlp_input_last',
            'unsteered': True, 'runtime_identities': [self.identity['runtime_identity']],
            'prepared_identities': [self.identity['prepared_identity']], 'corpus_sha256': 'a' * 64}
        arrays = dict(metadata=np.array(json.dumps(metadata)), layers=np.array([0]), neurons=np.array([1]),
            values=np.array([[2.]], dtype=np.float32), corpora=np.array(['skippy']),
            prompt_ids=np.array([prompt['id']]), source_hashes=np.array(['b' * 64]))
        output = io.BytesIO(); np.savez_compressed(output, **arrays); raw = output.getvalue()
        snapshot = matrix_snapshot(io.BytesIO(raw))
        rows, identity = previous_records(snapshot, {prompt['id']: prompt}, self.model, self.root, chart,
                                          'e' * 64, np.array([0]), np.array([1]))
        self.assertEqual(identity, self.identity); self.assertEqual(rows[0][0]['previous_vector_index'], 0)
        with self.assertRaisesRegex(ValueError, 'bounded'):
            matrix_snapshot(io.BytesIO(raw), maximum=len(raw) - 1)
        changed = {**prompt, 'prompt': 'Changed bytes with the old content ID'}
        with self.assertRaisesRegex(ValueError, 'unchanged original'):
            previous_records(snapshot, {prompt['id']: changed}, self.model, self.root, chart, 'e' * 64, np.array([0]), np.array([1]))


if __name__ == '__main__':
    unittest.main()
