import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from persona_coactivity_extract import records, selected, read_candidate_values, file_hash
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
                      mutation={'kind': 'original'}, labels={}, prompt='Synthetic passage', system='')
        job = dict(operator_coverage=True, arm='baseline', pushes=[], measurement='prefill',
                   model_identity=self.model.identity, prompt=prompt)
        job['id'] = canonical_digest(job)
        artifact = {**self.identity, 'path': str(self.root / 'capture.npz'), 'request_sha256': 'd' * 64,
                    'stage': 'final_prefill', 'operation': 'prefill_observation'}
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


if __name__ == '__main__':
    unittest.main()
