"""CPU-only queue, controls, provenance and planner-boundary checks."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from persona_manager import (ARMS, SCHEMA, Manager, Queue, activation_summary, check_admission,
    deadline, execute_job, expand_proposal, file_hash, load_corpus, local_url, native_capability,
    response_complete, shared_runtime, validate_proposal, verify_diagnostic_artifacts)
from persona_vectors import ModelSpec, VectorBank, canonical_digest


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = ModelSpec('test/qwen', 'a' * 40, 'test-fp32', 2, 4, 5)
        np.savez(self.root / 'layer_00.npz', model_identity=self.model.identity,
            residual_median=10., neuron_ids=np.array([1, 3]),
            down_columns=np.array([[3., 0.], [4., 0.], [0., 0.], [0., 2.]]),
            trait_openness=np.array([0., 0., 2., 0.]))
        self.bank = VectorBank(self.root, self.model)
        self.corpus = {split: {'id': split, 'prompt': 'A registered prompt', 'split': split,
            'corpus': 'big_five', 'category': 'openness', 'max_tokens': 2048}
            for split in ('discovery', 'heldout')}
        self.proposal = {'schema': SCHEMA, 'model_identity': self.model.identity, 'batch_seed': 17,
            'rationale': 'Test both polarities without assuming transfer', 'experiments': [
                {'label': 'candidate', 'kind': 'neuron', 'layer': 0, 'neuron': 1, 'dose': .2,
                 'phase': 'prefill', 'corpora': ['big_five'], 'prompts_per_corpus': 1, 'repetitions': 1}]}
        self.queue = Queue(self.root / 'queue.sqlite')

    def jobs(self, split='discovery'):
        return expand_proposal(self.proposal, self.corpus, self.bank, split=split, corpus_sha256='c' * 64)

    def test_strict_proposals_cannot_select_shell_paths_or_unbound_coordinates(self):
        self.assertEqual(validate_proposal(self.proposal, self.model, self.corpus, self.bank), self.proposal)
        for key, value in [('layer', True), ('layer', 2), ('neuron', 5), ('dose', float('nan')),
                           ('dose', 0), ('phase', 'shell'), ('corpora', [{'url': 'x'}])]:
            proposal = deepcopy(self.proposal)
            proposal['experiments'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_proposal(proposal, self.model, self.corpus, self.bank)
        for key in ('shell', 'endpoint', 'system', 'remove_controls'):
            proposal = deepcopy(self.proposal); proposal[key] = 'do something'
            with self.assertRaises(ValueError):
                validate_proposal(proposal, self.model, self.corpus, self.bank)

    def test_all_paired_controls_are_deterministic_distinct_and_same_seed(self):
        jobs = self.jobs()
        self.assertEqual(jobs, self.jobs())
        self.assertEqual(set(job['arm'] for job in jobs), set(ARMS))
        self.assertEqual(len({job['id'] for job in jobs}), 5)
        self.assertEqual(len({job['seed'] for job in jobs}), 1)
        arms = {job['arm']: job for job in jobs}
        self.assertEqual(arms['baseline']['pushes'], [])
        self.assertNotEqual(arms['random']['pushes'][0]['neuron'], 1)
        self.assertEqual(arms['positive']['pushes'][0]['dose'], -arms['negative']['pushes'][0]['dose'])
        self.assertEqual(arms['positive']['calibration_sha256'], file_hash(self.root / 'layer_00.npz'))

    def test_source_groups_keep_every_mutation_profile_and_match_seeds(self):
        corpus = {}
        for pair_id, variants, split in [('source-a', ['original', 'negation', 'speaker_swap'], 'discovery'),
                                         ('source-b', ['original', 'trait_profile'], 'discovery'),
                                         ('heldout-source', ['original', 'negation'], 'heldout')]:
            for variant in variants:
                key = pair_id + ':' + variant
                corpus[key] = {**self.corpus[split], 'id': key, 'pair_id': pair_id,
                               'mutation': {'kind': variant}, 'prompt': f'{pair_id} {variant}'}
        proposal = deepcopy(self.proposal)
        proposal['experiments'][0].update(prompts_per_corpus=2, repetitions=2)
        jobs = expand_proposal(proposal, corpus, self.bank)
        reversed_jobs = expand_proposal(proposal, dict(reversed(list(corpus.items()))), self.bank)
        self.assertEqual(jobs, reversed_jobs)
        self.assertEqual(len(jobs), 5 * 2 * len(ARMS))
        self.assertEqual({job['prompt']['id'] for job in jobs},
                         {row['id'] for row in corpus.values() if row['split'] == 'discovery'})
        for pair_id, count in [('source-a', 3), ('source-b', 2)]:
            for repetition in range(2):
                group = [job for job in jobs if job['pair_id'] == pair_id and job['repetition'] == repetition]
                self.assertEqual(len(group), count * len(ARMS))
                self.assertEqual(len({job['seed'] for job in group}), 1)
                self.assertTrue(all(job['source_group'] == {'corpus': 'big_five', 'pair_id': pair_id, 'split': 'discovery'} for job in group))
                self.assertEqual(len({job['pushes'][0]['neuron'] for job in group if job['arm'] == 'random'}), 1)
        proposal['experiments'][0]['prompts_per_corpus'] = 1
        selected = expand_proposal(proposal, corpus, self.bank)
        self.assertEqual(len({job['pair_id'] for job in selected}), 1)
        chosen = selected[0]['pair_id']
        self.assertEqual({job['prompt']['id'] for job in selected}, {row['id'] for row in corpus.values() if row['pair_id'] == chosen})

    def test_group_expansion_rejects_budget_overflow_without_partial_enqueue(self):
        corpus = {str(i): {**self.corpus['discovery'], 'id': str(i), 'pair_id': 'same-source'} for i in range(3)}
        jobs = expand_proposal(self.proposal, corpus, self.bank)
        self.assertEqual(len(jobs), 15)
        with self.assertRaises(ValueError):
            self.queue.enqueue(self.proposal, jobs, limit=14)
        self.assertEqual(self.queue.counts(), {})
        with self.queue.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM proposals').fetchone()[0], 0)

    def test_source_group_cannot_cross_discovery_and_heldout(self):
        path = self.root / 'corpus.jsonl'
        path.write_text(''.join(json.dumps({**row, 'pair_id': 'shared-source'}) + '\n' for row in self.corpus.values()))
        with self.assertRaisesRegex(ValueError, 'same split'):
            load_corpus(path)

    def test_queue_claim_is_atomic_and_attempt_ownership_is_exact(self):
        jobs = self.jobs()
        self.assertEqual(self.queue.enqueue(self.proposal, jobs), 5)
        self.assertEqual(self.queue.enqueue(self.proposal, jobs), 0)
        with ThreadPoolExecutor(max_workers=6) as pool:
            claims = list(pool.map(self.queue.claim, [f'endpoint{i}' for i in range(12)]))
        claims = [claim for claim in claims if claim]
        self.assertEqual(len(claims), 5)
        self.assertEqual(len({claim[0]['id'] for claim in claims}), 5)
        job, attempt = claims[0]
        self.queue.finish(job, attempt, {'complete': True})
        with self.assertRaises(RuntimeError):
            self.queue.finish(job, attempt, {'complete': True})
        self.assertEqual(self.queue.counts(), {'complete': 1, 'running': 4})
        with self.queue.connect() as db:
            ids = {row[0] for row in db.execute('SELECT id FROM proposals')}
        self.assertEqual(ids, {job['proposal_id']})

    def test_resume_records_unknown_execution_and_bounds_retries(self):
        self.queue.enqueue(self.proposal, self.jobs())
        first, _ = self.queue.claim('a')
        self.assertEqual(self.queue.recover(), 1)
        second, attempt = self.queue.claim('a')
        self.assertEqual(second['id'], first['id'])
        self.assertEqual(attempt, 2)
        self.assertEqual(self.queue.recover(), 1)
        self.assertEqual(self.queue.counts(), {'failed': 1, 'queued': 4})
        with self.queue.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM attempts WHERE state='interrupted'").fetchone()[0], 2)
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute('DELETE FROM audit')

    def test_corpus_can_append_but_not_rewrite_existing_split_or_prompt(self):
        self.queue.register_corpus(self.corpus, 'old')
        changed = deepcopy(self.corpus); changed['discovery']['split'] = 'heldout'
        with self.assertRaises(ValueError):
            self.queue.register_corpus(changed, 'changed')
        appended = {**self.corpus, 'new': {**self.corpus['discovery'], 'id': 'new'}}
        self.queue.register_corpus(appended, 'new')
        with self.assertRaises(ValueError):
            self.queue.register_corpus(self.corpus, 'old')

    def test_heldout_has_separate_proposal_and_never_enters_adaptive_feedback(self):
        self.queue.enqueue(self.proposal, self.jobs())
        self.queue.enqueue(self.proposal, self.jobs('heldout'))
        while claim := self.queue.claim('a'):
            job, attempt = claim
            self.queue.finish(job, attempt, {'complete': True, 'response': {'choices': [{'message': {'content': job['prompt']['split']}}]}})
        self.assertEqual(self.queue.counts(), {'complete': 10})
        self.assertEqual(sum(row['count'] for row in self.queue.planning_summary()), 5)
        self.assertEqual(len(self.queue.observations()), 5)
        self.assertTrue(all(row['answer'] == 'discovery' for row in self.queue.observations()))

    def endpoint(self):
        return {'url': 'http://127.0.0.1:8910', 'gpu_uuid': 'GPU-test', 'lease_id': 'lease-test',
                'project': 'character-creation', 'model_identity': self.model.identity, 'prepared_identity': 'prepared'}

    def test_explicit_zero_control_reaches_wire_and_baseline_omits_steering(self):
        requests = []
        def api(url, payload=None, timeout=0):
            if url.endswith('/v1/models'):
                return {'data': [{'id': 'served', 'flare': {'prepared_identity': 'prepared', 'steering': {
                    'protocol': 'additive-bf16-sites-v1', 'max_sites': 6,
                    'sites': [{'name': 'layers.0.mlp_input', 'width': 5}]}}}]}
            requests.append(payload)
            return {'choices': [{'message': {'content': 'answer'}, 'finish_reason': 'stop'}]}
        with patch('persona_manager.fetch', api):
            for job in self.jobs():
                if job['arm'] in ('baseline', 'zero'):
                    self.assertTrue(execute_job(job, self.endpoint(), self.bank, 30)['complete'])
        self.assertNotIn('steering', requests[0])
        self.assertEqual(requests[1]['steering']['sites'][0]['vector'], [0.] * 5)

    def test_old_package_and_modified_calibration_rejected_before_generation(self):
        with patch('persona_manager.fetch', return_value={'data': [{'id': 'old'}]}), self.assertRaises(ValueError):
            execute_job(self.jobs()[0], self.endpoint(), self.bank, 30)
        job = self.jobs()[0]
        job['calibration_sha256'] = 'wrong'
        with patch('persona_manager.fetch') as api, self.assertRaises(ValueError):
            execute_job(job, self.endpoint(), self.bank, 30)
        api.assert_not_called()

    def test_diagnostic_runtime_binding_rejects_mixed_changed_and_missing_runtimes(self):
        endpoint = {**self.endpoint(), 'diagnostic': True, 'runtime_identity': 'a' * 64}
        config = {'model': 'model.json', 'bank': str(self.root), 'workboard': '/unused',
            'endpoints': [{**endpoint, 'gpu_uuid': f'GPU-{i}'} for i in range(6)],
            'planner': {'url': 'http://127.0.0.1:18511', 'gpu_uuid': 'John', 'lease_id': 'planner', 'project': 'character-creation'}}
        with patch('persona_manager.ModelSpec.load', return_value=self.model), patch.object(Manager, 'refresh_corpus'):
            manager = Manager(config, self.queue)
            self.assertEqual(manager.runtime_identity, 'a' * 64)
            changed = deepcopy(config)
            for server in changed['endpoints']:
                server['runtime_identity'] = 'b' * 64
            with self.assertRaisesRegex(ValueError, 'Queue.*runtime'):
                Manager(changed, self.queue)
        for servers in ([endpoint, {**endpoint, 'runtime_identity': 'b' * 64}],
                        [endpoint, {**endpoint, 'runtime_identity': None}],
                        [endpoint, {**endpoint, 'diagnostic': False}]):
            with self.assertRaises(ValueError):
                shared_runtime(servers)
        flare = {'prepared_identity': 'prepared', 'model_identity': self.model.identity, 'runtime_identity': 'b' * 64}
        with patch('persona_manager.fetch', return_value={'data': [{'id': 'served', 'flare': flare}]}), self.assertRaisesRegex(ValueError, 'runtime_identity changed'):
            native_capability(endpoint, self.model, {'sites': []})

    def test_diagnostic_provenance_binds_runtime_request_and_restored_state(self):
        endpoint = {**self.endpoint(), 'diagnostic': True, 'runtime_identity': 'a' * 64}
        payload = {'model': 'served', 'messages': [{'role': 'user', 'content': 'Original prompt'}]}
        digest = canonical_digest(payload)
        binding = {key: endpoint[key] for key in ('model_identity', 'prepared_identity', 'runtime_identity')}
        report = {**binding, 'status': 'complete', 'state_restored': True, 'request': payload,
                  'request_sha256': digest, 'worker_manifest': {'runtime_identity': 'a' * 64}}
        path = self.root / 'provenance.jsonl'
        path.write_text(json.dumps(report))
        response = {**binding, 'activation_artifact': {**binding, 'request_sha256': digest},
            'provenance_artifact': {'path': str(path), 'sha256': file_hash(path), 'runtime_identity': 'a' * 64, 'request_sha256': digest}}
        verify_diagnostic_artifacts(response, endpoint, payload)
        for target, key, bad in [('response', 'runtime_identity', 'b' * 64),
                                 ('activation_artifact', 'runtime_identity', 'b' * 64),
                                 ('provenance_artifact', 'request_sha256', 'b' * 64)]:
            changed = deepcopy(response)
            (changed if target == 'response' else changed[target])[key] = bad
            with self.assertRaises(ValueError):
                verify_diagnostic_artifacts(changed, endpoint, payload)
        for key, bad in [('worker_manifest', {'runtime_identity': 'b' * 64}), ('state_restored', False),
                         ('request', {'model': 'served', 'messages': []})]:
            path.write_text(json.dumps({**report, key: bad}))
            response['provenance_artifact']['sha256'] = file_hash(path)
            with self.assertRaises(ValueError):
                verify_diagnostic_artifacts(response, endpoint, payload)

    def test_diagnostic_completeness_rejects_thinking_missing_flags_and_truncation(self):
        response = {'choices': [{'message': {'content': 'answer'}, 'finish_reason': 'stop'}]}
        self.assertTrue(response_complete(response, diagnostic=False))
        self.assertFalse(response_complete(response, diagnostic=True))
        for complete, thinking, expected in [(True, False, True), (False, False, False), (True, True, False)]:
            response['diagnostics'] = {'complete_final_answer': complete, 'unexpected_thinking': thinking}
            self.assertEqual(response_complete(response, diagnostic=True), expected)
        response['diagnostics'] = {'complete_final_answer': True, 'unexpected_thinking': False}
        response['choices'][0]['finish_reason'] = 'length'
        self.assertFalse(response_complete(response, diagnostic=True))

    def test_activation_summary_is_measured_and_digest_bound(self):
        artifacts = []
        for suffix, amount in [('baseline', 0.), ('positive', 2.)]:
            neurons = np.zeros((2, 5), np.float32); neurons[0, 1] = amount
            path = self.root / (suffix + '.npz')
            np.savez(path, mlp_input_last=neurons, residual_last=np.ones((2, 4), np.float32),
                     model_identity=self.model.identity, request_sha256='r', stage='final_prefill',
                     prepared_identity='prepared', runtime_identity='a' * 64)
            artifacts.append({'path': str(path), 'sha256': file_hash(path), 'model_identity': self.model.identity,
                              'request_sha256': 'r', 'stage': 'final_prefill',
                              'prepared_identity': 'prepared', 'runtime_identity': 'a' * 64})
        summary = activation_summary(artifacts[1], self.jobs()[3], artifacts[0])
        self.assertEqual(summary['selected_neuron_value'], 2.)
        self.assertEqual(summary['baseline_delta']['largest_coordinates'][0], {'layer': 0, 'neuron': 1, 'delta': 2.})
        with self.assertRaisesRegex(ValueError, 'runtime_identity'):
            activation_summary({**artifacts[1], 'runtime_identity': 'b' * 64}, self.jobs()[3])
        artifacts[1]['sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            activation_summary(artifacts[1], self.jobs()[3])

    def test_admission_requires_owned_fresh_cool_gpu(self):
        endpoint = {**self.endpoint(), 'host': 'v100', 'diagnostic': True}
        board = {'jobs': [{'id': 'lease-test', 'status': 'running', 'assigned_uuid': 'GPU-test', 'project': 'character-creation'}],
            'fleet': {'cards': [{'uuid': 'GPU-test', 'fresh': True, 'temperature_c': 70, 'host': 'v100', 'name': 'Tesla V100'}]},
            'policy': {'gpu_temperature_policies': [{'gpu_uuid': 'GPU-test', 'temperature_ceiling_c': 85}]}}
        self.assertTrue(check_admission(board, endpoint)[0])
        for key, value in [('temperature_c', 85), ('fresh', False), ('host', 'other'), ('name', 'otherGPU')]:
            changed = deepcopy(board); changed['fleet']['cards'][0][key] = value
            self.assertFalse(check_admission(changed, endpoint)[0])
        board['jobs'][0]['project'] = 'somebody-else'
        self.assertFalse(check_admission(board, endpoint)[0])

    def test_bounded_deadline_and_loopback_only_endpoints(self):
        now = datetime.fromisoformat('2026-10-25T01:00:00-07:00').timestamp()
        self.assertEqual(deadline(None, now) - now, 7 * 3600)
        self.assertEqual(deadline('2026-11-25T01:00:00-07:00', now) - now, 16 * 3600)
        for value in ('https://127.0.0.1', 'http://external.test', 'http://user:pass@localhost', 'http://localhost/path'):
            with self.assertRaises(ValueError):
                local_url(value)


if __name__ == '__main__':
    unittest.main()
