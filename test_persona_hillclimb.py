"""CPU protocol/evidence tests. No endpoints, GPU model or subprocess is launched."""
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from persona_hillclimb import (DECISION, DOSES, Evidence, RequestRecorder, bounded_options, build_round, decision_schema, families,
    propose, remote_run, run_neighborhood, search_options, validate_decision, validate_job)
from persona_manager import ARMS, Queue, file_hash
from persona_vectors import ModelSpec, VectorBank, canonical_digest


class HillclimbTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = ModelSpec('test/qwen', 'a' * 40, 'test-fp32', 3, 4, 5)
        np.savez(self.root / 'layer_00.npz', model_identity=self.model.identity, residual_median=10.,
            neuron_ids=np.array([1, 3]), down_columns=np.array([[3., 0.], [4., 0.], [0., 0.], [0., 2.]]))
        self.bank = VectorBank(self.root, self.model)
        self.job = {'id': 'job-test', 'kind': 'neighborhood_hillclimb', 'seed': 7035,
            'target_entries': 20, 'max_entries': 100, 'deadline': time.time() + 60,
            'targets': [{'layer': 0, 'neuron': 1}], 'corpora': ['big_five']}
        self.corpus = {}
        for index in range(3):
            for variant in range(2):
                key = f'p{index}v{variant}'
                self.corpus[key] = {'id': key, 'pair_id': f'source{index}',
                    'prompt': f'Explicit registered prompt {index} variant{variant}', 'corpus': 'big_five',
                    'category': 'openness', 'split': 'discovery' if index < 2 else 'heldout',
                    'max_tokens': 2048, 'source_sha256': 'e' * 64,
                    'mutation': {'kind': 'original' if not variant else 'instruction_negation'}}
        self.catalog = families(self.corpus, self.job['corpora']); self.family = next(iter(self.catalog))
        self.decision = {'schema': DECISION, 'rationale': 'Find variation and replicate downstream signs.',
            'experiments': [{'target_index': 0, 'family': self.family, 'dose': .25,
                             'measurement': 'prefill', 'source_groups': 2}]}
        self.queue = Queue(self.root / 'queue.sqlite'); self.evidence = Evidence(self.queue, self.job)

    def plan(self, decision=None, feedback=None, remaining=100):
        return build_round(decision or self.decision, self.job, self.catalog, self.bank, 'c' * 64,
                           0, feedback or {}, remaining)

    def result(self, task, *, zero_bad=False, offset=0.):
        base = np.zeros((3, 5), dtype=np.float32)
        base[0, 1] = float(task['prompt']['id'][1]) + offset
        arm = task['arm']
        if arm == 'positive': base[0, 1] += 1.; base[1, 2] += 2.
        if arm == 'negative': base[0, 1] -= 1.; base[1, 2] -= 2.
        if arm == 'random': base[1, 2] += .125
        if arm == 'zero' and zero_bad: base[2, 4] = -0.
        operation = 'prefill_observation' if task['measurement'] == 'prefill' else 'rollout'
        request = canonical_digest([task['id'], task['seed']])
        path = self.root / (task['id'] + '.npz')
        np.savez(path, mlp_input_last=base, residual_last=np.zeros((3, 4), np.float32),
            model_identity=self.model.identity, prepared_identity='p' * 64,
            runtime_identity='b' * 64, request_sha256=request, stage='final_prefill', operation=operation)
        return {'complete': True, 'complete_answer': operation == 'rollout',
            'activation_artifact': {'path': str(path), 'sha256': file_hash(path), 'stage': 'final_prefill',
                'model_identity': self.model.identity, 'prepared_identity': 'p' * 64,
                'runtime_identity': 'b' * 64, 'request_sha256': request, 'operation': operation}}

    def finish_all(self, zero_bad=False):
        while True:
            claim = self.queue.claim('test-worker')
            if not claim: break
            task, attempt = claim
            self.queue.finish(task, attempt, self.result(task, zero_bad=zero_bad))

    def test_allowlists_model_bounds_and_no_shell_fields(self):
        validate_job(self.job, self.model)
        unmapped = copy.deepcopy(self.job); unmapped['targets'][0]['neuron'] = 4
        validate_job(unmapped, self.model)  # Mapping membership is not the model boundary.
        for field, value in [('layer', 3), ('neuron', 5), ('neuron', True)]:
            changed = copy.deepcopy(self.job); changed['targets'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError): validate_job(changed, self.model)
        for change in ({'shell': 'echo bad'}, {'family': 'heldout'}, {'dose': True}, {'measurement': 'decode'}):
            value = copy.deepcopy(self.decision); value['experiments'][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError): validate_decision(value, self.job, self.catalog, {})

    def test_all_five_controls_every_variant_same_seed_deterministic(self):
        proposal, tasks = self.plan(); self.assertEqual(len(tasks), 20)
        self.assertEqual(tasks, self.plan()[1])
        self.assertEqual({task['prompt']['id'] for task in tasks}, {'p0v0', 'p0v1', 'p1v0', 'p1v1'})
        for prompt in {task['prompt']['id'] for task in tasks}:
            group = [task for task in tasks if task['prompt']['id'] == prompt]
            self.assertEqual({task['arm'] for task in group}, set(ARMS))
            self.assertEqual(len({task['seed'] for task in group}), 1)
            self.assertTrue(all(task['prompt']['split'] == 'discovery' for task in group))
            self.assertEqual(next(task for task in group if task['arm'] == 'baseline')['pushes'], [])
        self.assertTrue(all(task['calibration_sha256'] == file_hash(self.root / 'layer_00.npz') for task in tasks))
        self.assertEqual(self.queue.enqueue(proposal, tasks, limit=100), 20)
        self.assertEqual(self.queue.enqueue(proposal, tasks, limit=100), 0)

    def test_cap_does_not_trim_controls_or_variants(self):
        with self.assertRaisesRegex(ValueError, 'Whole matched-control'): self.plan(remaining=19)
        self.assertEqual(self.queue.counts(), {})

    def test_followup_requires_two_complete_independent_source_groups(self):
        followup = copy.deepcopy(self.decision); followup['experiments'][0]['measurement'] = 'rollout'
        with self.assertRaisesRegex(ValueError, 'two source groups'): self.plan(followup)
        proposal, tasks = self.plan(); self.queue.enqueue(proposal, tasks, limit=100)
        self.finish_all(); self.evidence.collect(); feedback = self.evidence.feedback()
        self.assertEqual(feedback['original_source_groups'], 2)
        self.assertEqual(len(feedback['followup_eligible']), 1)
        self.assertEqual(feedback['followup_eligible'][0]['groups'], ['source0', 'source1'])
        _, rollouts = self.plan(followup, feedback)
        self.assertEqual(len(rollouts), 40)
        self.assertTrue(all(task['prompt']['max_tokens'] >= 2048 for task in rollouts))
        self.assertEqual({task['repetition'] for task in rollouts}, {0, 1})

    def test_signed_downstream_recurrence_excludes_directly_pushed_coordinate(self):
        proposal, tasks = self.plan(); self.queue.enqueue(proposal, tasks, limit=100)
        self.finish_all(); self.evidence.collect(); feedback = self.evidence.feedback()
        effects = feedback['replicated_effect_candidates']
        self.assertEqual(len(effects), 1)
        self.assertEqual((effects[0]['layer'], effects[0]['neuron']), (1, 2))
        self.assertEqual(effects[0]['independent_source_groups'], 2)
        self.assertEqual(feedback['baseline_variation'][0]['sample_sd'], np.std([0., 1.], ddof=1))
        self.assertNotIn('correlation', effects[0])

    def test_zero_bit_difference_stops_analysis(self):
        proposal, tasks = self.plan(); self.queue.enqueue(proposal, tasks, limit=100)
        self.finish_all(zero_bad=True)
        with self.assertRaisesRegex(ValueError, 'zero-control invariant'): self.evidence.collect()

    def test_prefill_seed_repeats_are_not_new_baseline_evidence(self):
        task = next(task for task in self.plan()[1] if task['arm'] == 'baseline')
        self.evidence.record(task, self.result(task))
        repeated = copy.deepcopy(task); repeated['seed'] += 1; repeated['id'] = 'repeat'
        self.evidence.record(repeated, self.result(repeated)); self.evidence.record(repeated, self.result(repeated))
        self.assertEqual(self.evidence.counts()['accepted_unique'], 2)
        self.assertEqual(self.evidence.feedback()['unique_unsteered_prompts'], 1)
        target = self.root / 'registry.jsonl'; pointer = self.evidence.export(target)
        self.assertEqual(pointer['rows'], 1); self.assertFalse(pointer['map_updated'])
        self.assertEqual(pointer['sha256'], file_hash(target))
        row = json.loads(target.read_text()); self.assertEqual(row['source_sha256'], 'e' * 64)
        self.assertEqual(row['artifact']['sha256'], task and self.result(task)['activation_artifact']['sha256'])
        self.assertNotIn('prompt', row); self.assertFalse(row['heldout_used'])

    def test_heldout_or_invalid_artifact_cannot_enter_evidence(self):
        task = next(task for task in self.plan()[1] if task['arm'] == 'baseline')
        heldout = copy.deepcopy(task); heldout['prompt']['split'] = 'heldout'
        with self.assertRaisesRegex(ValueError, 'Heldout'): self.evidence.record(heldout, self.result(heldout))
        result = self.result(task); result['activation_artifact']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'digest changed'): self.evidence.record(task, result)
        self.assertEqual(self.evidence.counts()['accepted_unique'], 0)

    def test_resume_preserves_completes_and_interrupted_attempt_history(self):
        proposal, tasks = self.plan(); self.queue.enqueue(proposal, tasks, limit=100)
        task, attempt = self.queue.claim('one'); self.queue.finish(task, attempt, self.result(task))
        interrupted, _ = self.queue.claim('two')
        self.assertEqual(self.queue.recover(), 1)
        self.assertEqual(self.queue.counts(), {'complete': 1, 'queued': 19})
        self.evidence.collect(); before = self.evidence.counts()
        self.evidence.collect(); self.assertEqual(before, self.evidence.counts())
        with self.queue.connect() as db:
            self.assertEqual(db.execute('SELECT state FROM attempts WHERE job_id=?', (interrupted['id'],)).fetchone()[0], 'interrupted')

    def test_already_attempted_groups_are_not_silently_repeated(self):
        feedback = {'attempted_groups': [(0, self.family, .25, 'prefill', 'source0'), (0, self.family, .25, 'prefill', 'source1')]}
        with self.assertRaisesRegex(ValueError, 'no unattempted'): self.plan(feedback=feedback)

    def test_search_exhaustion_and_whole_set_cost(self):
        options = search_options(self.job, self.catalog, {})
        self.assertEqual(len(options), len(DOSES))
        self.assertTrue(all(row['minimum_observations'] == 10 for row in options))
        used = [(0, self.family, dose, 'prefill', pair) for dose in DOSES for pair in ('source0', 'source1')]
        self.assertEqual(search_options(self.job, self.catalog, {'attempted_groups': used}), [])
        with self.assertRaisesRegex(ValueError, 'requires an eligible'):
            validate_decision(self.decision, self.job, self.catalog, {'require_rollout_followup': True})

    def test_exact_wire_request_survives_transport_failure(self):
        def failing(*args, **kwargs): raise TimeoutError('Simulated transport failure')
        recorder = RequestRecorder(self.queue, failing)
        recorder.bind({'id': 'attempted-job'}, 2)
        payload = {'seed': 17, 'steering': {'phase': 'prefill', 'sites': [{'site': 'L0.mlp_input', 'vector': [0., -1.]}]}}
        with self.assertRaises(TimeoutError): recorder('http://127.0.0.1:8930/v1/chat/completions', payload, timeout=10)
        with self.queue.connect() as db:
            event, raw = db.execute('SELECT event,data FROM audit').fetchone()
        self.assertEqual(event, 'hillclimb_wire_request'); record = json.loads(raw)
        self.assertEqual(record['request'], payload); self.assertEqual(record['attempt'], 2)
        self.assertEqual(record['request_sha256'], canonical_digest(payload))

    def test_planner_schema_and_rejected_attempt_correction_are_audited(self):
        planner = {'url': 'http://127.0.0.1:18511', 'model': 'persona-manager', 'context_tokens': 131072}
        payloads = []
        decision = copy.deepcopy(self.decision); decision['experiments'][0]['source_groups'] = 1
        def transport(url, payload=None, **kwargs):
            if url.endswith('/props'):
                return {'model_alias': planner['model'], 'default_generation_settings': {'n_ctx': 131072}}
            payloads.append(copy.deepcopy(payload))
            # The immutable request must exist before any HTTP POST is attempted.
            with self.queue.connect() as db:
                records = [json.loads(row[0]) for row in db.execute("SELECT data FROM audit WHERE event='hillclimb_planner_request' ORDER BY sequence")]
            self.assertEqual(len(records), len(payloads))
            self.assertEqual(records[-1]['request'], payload)
            self.assertEqual(records[-1]['request_sha256'], canonical_digest(payload))
            value = [] if len(payloads) == 1 else decision
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}
        feedback = self.evidence.feedback()
        with patch('persona_hillclimb.fetch', side_effect=transport):
            with self.assertRaises(ValueError) as failed:
                propose(planner, self.job, self.catalog, feedback, self.queue, 100, 10)
            reason = str(failed.exception)
            self.queue.audit('hillclimb_planner_rejected', {'error': reason, 'consecutive': 1})
            self.assertEqual(propose(planner, self.job, self.catalog, feedback, self.queue, 100, 10), decision)
        self.assertNotEqual(payloads[0]['seed'], payloads[1]['seed'])
        correction = json.loads(payloads[1]['messages'][1]['content'])['correction']
        self.assertEqual(correction['previous_rejection'], reason); self.assertEqual(correction['attempt'], 2)
        format = payloads[0]['response_format']; self.assertEqual(format['type'], 'json_schema')
        self.assertEqual(set(format), {'type', 'json_schema'})
        self.assertTrue(format['json_schema']['strict'])
        schema = format['json_schema']['schema']
        self.assertEqual(schema['properties']['experiments']['maxItems'], 6)
        self.assertEqual(schema['type'], 'object'); self.assertFalse(schema['additionalProperties'])
        self.assertEqual(set(schema['required']), {'schema', 'rationale', 'experiments'})
        experiment = schema['properties']['experiments']['items']; self.assertFalse(experiment['additionalProperties'])
        self.assertEqual(experiment['properties']['family']['enum'], [self.family])
        self.assertEqual(experiment['properties']['target_index']['enum'], [0])
        self.assertEqual(experiment['properties']['dose']['enum'], list(DOSES))
        self.assertEqual(experiment['properties']['measurement']['enum'], ['prefill'])
        self.assertEqual(experiment['properties']['source_groups']['enum'], [1])

    def test_planner_grammar_caps_whole_control_sets_before_model_arithmetic(self):
        family = copy.deepcopy(self.catalog[self.family])
        family['groups'] = {'source': [dict(next(iter(self.corpus.values()))) for _ in range(11)]}
        planner = {'url': 'http://127.0.0.1:18511', 'model': 'persona-manager', 'context_tokens': 131072}
        def transport(url, payload=None, **_):
            if payload is None: return {'model_alias': planner['model'], 'default_generation_settings': {'n_ctx': 131072}}
            schema = payload['response_format']['json_schema']['schema']['properties']['experiments']
            self.assertEqual(schema['maxItems'], 3)
            self.assertEqual(schema['items']['properties']['source_groups']['enum'], [1])
            brief = json.loads(payload['messages'][-1]['content'])
            self.assertEqual(brief['per_round_limits']['maximum_observations_per_source_group'], 55)
            value = copy.deepcopy(self.decision); value['experiments'] *= 6
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}
        with patch('persona_hillclimb.fetch', transport), self.assertRaisesRegex(ValueError, 'grammar-sized'):
            propose(planner, self.job, {self.family: family}, self.evidence.feedback(), self.queue, 180, 10)

    def test_large_corpus_brief_and_schema_are_bounded_with_all_targets_visible(self):
        catalog = {}
        for index in range(4000):
            family = copy.deepcopy(self.catalog[self.family])
            family['id'] = canonical_digest(index)
            catalog[family['id']] = family
        job = copy.deepcopy(self.job)
        job['targets'] = [{'layer': 0, 'neuron': 1}, {'layer': 1, 'neuron': 3}]
        planner = {'url': 'http://127.0.0.1:18511', 'model': 'persona-manager', 'context_tokens': 131072}
        payloads = []
        def transport(url, payload=None, **_):
            if payload is None: return {'model_alias': planner['model'], 'default_generation_settings': {'n_ctx': 131072}}
            payloads.append(payload)
            brief = json.loads(payload['messages'][-1]['content'])
            option = brief['available_options'][0]
            value = {'schema': DECISION, 'rationale': 'Select visible under-observed evidence.',
                'experiments': [{key: option[key] for key in ('target_index', 'family', 'dose', 'measurement')} | {'source_groups': 1}]}
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}
        with patch('persona_hillclimb.fetch', transport):
            propose(planner, job, catalog, self.evidence.feedback(), self.queue, 100, 10)
        payload = payloads[0]; brief = json.loads(payload['messages'][-1]['content'])
        self.assertEqual(brief['catalog_families'], 4000)
        self.assertLessEqual(len(brief['families']), 160)
        self.assertEqual({row['target_index'] for row in brief['available_options']}, {0, 1})
        visible = {row['id'] for row in brief['families']}
        self.assertEqual(set(payload['response_format']['json_schema']['schema']['properties']['experiments']['items']['properties']['family']['enum']), visible)
        self.assertLess(len(json.dumps(payload)), 131072 * 2)
        options = search_options(job, catalog, self.evidence.feedback())
        self.assertEqual(bounded_options(options, 128, 7), bounded_options(options, 128, 7))
        self.assertNotEqual(bounded_options(options, 128, 7), bounded_options(options, 128, 8))

    def test_planner_transport_failure_retains_request_and_advances_attempt(self):
        planner = {'url': 'http://127.0.0.1:18511', 'model': 'persona-manager', 'context_tokens': 131072}
        def transport(url, *args, **kwargs):
            if url.endswith('/props'):
                return {'model_alias': planner['model'], 'default_generation_settings': {'n_ctx': 131072}}
            raise TimeoutError('simulated planner connection failure')
        with patch('persona_hillclimb.fetch', side_effect=transport):
            for _ in range(2):
                with self.assertRaises(TimeoutError):
                    propose(planner, self.job, self.catalog, self.evidence.feedback(), self.queue, 100, 10)
        with self.queue.connect() as db:
            requests = [json.loads(row[0]) for row in db.execute("SELECT data FROM audit WHERE event='hillclimb_planner_request' ORDER BY sequence")]
            self.assertEqual(db.execute("SELECT COUNT(*) FROM audit WHERE event='hillclimb_planner_response'").fetchone()[0], 0)
        self.assertEqual([row['attempt'] for row in requests], [1, 2])
        self.assertNotEqual(requests[0]['request']['seed'], requests[1]['request']['seed'])

    def test_six_worker_remote_dispatch_and_resume_without_network_or_gpu(self):
        import persona_manager
        data = self.root / 'data'; data.mkdir(); results = self.root / 'results'; results.mkdir()
        (data / 'persona_model.json').write_text(json.dumps(self.model.__dict__))
        (results / 'layer_00.npz').write_bytes((self.root / 'layer_00.npz').read_bytes())
        corpus = data / 'persona_manager_corpus.jsonl'
        corpus.write_text(''.join(json.dumps(row) + '\n' for row in self.corpus.values()))
        (data / 'persona_manager_corpus_manifest.json').write_text(json.dumps({'status': 'complete', 'output_sha256': file_hash(corpus)}))
        endpoints = [{'url': f'http://127.0.0.1:{8930+i}', 'gpu_uuid': f'GPU-{i}', 'lease_id': f'lease{i}',
            'project': 'character-creation', 'model_identity': self.model.identity, 'prepared_identity': 'p' * 64,
            'runtime_identity': 'b' * 64, 'diagnostic': True} for i in range(6)]
        config = {'root': str(self.root), 'hostname': 'v100', 'endpoints': endpoints,
            'planner': {'url': 'http://127.0.0.1:18511'}, 'checkpoint': str(self.root), 'workboard': '/unused'}
        job = {**self.job, 'max_entries': 20}; lock = threading.Lock(); barrier = threading.Barrier(6); active = peak = calls = 0
        def execute(task, *args):
            nonlocal active, peak, calls
            with lock: active += 1; calls += 1; peak = max(peak, active); ordinal = calls
            try:
                if ordinal <= 6: barrier.wait(timeout=5)
                return self.result(task)
            finally:
                with lock: active -= 1
        before = persona_manager.fetch
        with patch('persona_hillclimb.socket.gethostname', return_value='v100'), \
             patch('persona_hillclimb.Admission.allowed', return_value=(True, 'test lease')), \
             patch('persona_hillclimb.propose', return_value=self.decision) as planner, \
             patch('persona_hillclimb.execute_job', side_effect=execute):
            summary = remote_run(job, config, lambda *_: None, threading.Event())
            self.assertEqual(summary['accepted_unique'], 20); self.assertEqual(summary['matched_control_sets'], 4)
            self.assertEqual(summary['reason'], 'target_reached'); self.assertEqual(peak, 6)
            self.assertEqual(summary['baseline_registry']['rows'], 4); self.assertFalse(summary['map_updated'])
            resumed = remote_run(job, config, lambda *_: None, threading.Event())
            self.assertEqual(resumed['accepted_unique'], 20); self.assertEqual(calls, 20); self.assertEqual(planner.call_count, 1)
        self.assertIs(persona_manager.fetch, before)

    def test_remote_wrapper_uses_fixed_argv_and_no_model_commands(self):
        captured = []
        class Process:
            def __init__(self, argv, **kwargs):
                captured.append(argv); self.stdin = io.StringIO(); self.stderr = io.StringIO('')
                self.stdout = io.StringIO(json.dumps({'type': 'event', 'event': 'progress', 'data': {'accepted_unique': 4}}) + '\n' + json.dumps({'type': 'result', 'result': {'accepted_unique': 4}}) + '\n')
            def wait(self, timeout=None): return 0
            def poll(self): return 0
        session = {'model': self.model.__dict__, 'remote_root': '/remote/Character Creation', 'root': str(self.root),
            'endpoints': [], 'planner': {}, 'fleet': {'hostname': 'v100', 'ssh_host': 'v100', 'python': '/venv/python',
                'checkpoint': '/checkpoint', 'workboard': '/workboard'}}
        events = []
        with patch('persona_hillclimb.subprocess.Popen', Process), patch('persona_hillclimb.file_hash', return_value='a' * 64):
            result = run_neighborhood(self.job, session, lambda name, data: events.append((name, data)), threading.Event())
        self.assertEqual(result, {'accepted_unique': 4}); self.assertEqual(events[0][0], 'progress')
        self.assertIn('ControlMaster=no', captured[0]); self.assertIn("'/remote/Character Creation/persona_hillclimb.py' remote", captured[0][-1])
        self.assertNotIn(self.decision['rationale'], captured[0][-1])


if __name__ == '__main__': unittest.main()
