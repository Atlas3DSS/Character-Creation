"""CPU-only evidence normalization and shared-pool campaign orchestration tests."""

import persona  # Make the flat, deployment-compatible source modules importable.
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from persona_corpus import record
from persona_explorer import Explorer, handler
from persona_sculptor import PLAN_SCHEMA, TRAITS, trusted_planner
from persona_vectors import ModelSpec, canonical_digest


class PersonaSculptorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('data', 'results', 'ui'): (self.root / name).mkdir()
        self.model = ModelSpec('fixture/qwen', 'a' * 40, 'test-fp32', 2, 5, 7)
        (self.root / 'data/persona_model.json').write_text(json.dumps(self.model.__dict__))
        for layer in range(2):
            np.savez(self.root / f'results/layer_{layer:02d}.npz', model_identity=self.model.identity,
                residual_median=10., neuron_ids=np.array([1, 2]), down_columns=np.eye(5)[:, :2],
                **{'trait_' + trait: np.eye(5)[index] if layer == 0 else np.eye(5)[0]
                   for index, trait in enumerate(TRAITS)})
        source = {'path': '/private/fixture.jsonl', 'sha256': 'a' * 64, 'bytes': 100}
        self.prompts = [record(source, text, corpus='big_five', category=category, split=split,
            locator={'row': index}, labels={'profile': 'baseline'}, system='Registered personality conditioning must be omitted')
            for index, (text, category, split) in enumerate([
                ('How would you handle a changed plan?', 'social', 'discovery'),
                ('Describe an unusual library.', 'creative', 'heldout')])]
        raw = ''.join(json.dumps(row) + '\n' for row in self.prompts).encode()
        import hashlib
        (self.root / 'data/persona_manager_corpus.jsonl').write_bytes(raw)
        (self.root / 'data/persona_manager_corpus_manifest.json').write_text(json.dumps({'status': 'complete', 'output_sha256': hashlib.sha256(raw).hexdigest()}))
        self.config = {'endpoints': [{'url': f'http://127.0.0.1:{9000 + index}', 'label': 'Quincy ' + str(index),
            'model_identity': self.model.identity, 'prepared_identity': 'b' * 64, 'runtime_identity': 'c' * 64}
            for index in range(6)]}
        self.app = Explorer(self.root, self.config); self.addCleanup(lambda: self.app.close() if not self.app.closed else None)
        self.controls = {trait: .5 for trait in TRAITS}
        self.requests, self.active, self.peak = [], set(), 0
        self.lock = threading.Lock()

    def descriptor(self, url):
        endpoint = next(endpoint for endpoint in self.config['endpoints'] if url.startswith(endpoint['url'] + '/'))
        return {'data': [{'id': 'served', 'flare': {**{key: endpoint[key] for key in ('model_identity', 'prepared_identity', 'runtime_identity')},
            'thinking_enabled_supported': False, 'physical_context_tokens': 4096,
            'steering': {'protocol': 'additive-bf16-sites-v1', 'max_sites': 64,
                        'sites': [{'name': f'layers.{layer}.residual', 'width': 5} for layer in range(2)]}}}]}

    def api(self, url, payload=None, timeout=None):
        if payload is None: return self.descriptor(url)
        with self.lock:
            self.assertNotIn(url, self.active, 'One endpoint received concurrent requests')
            self.active.add(url); self.peak = max(self.peak, len(self.active)); self.requests.append(deepcopy(payload))
        try:
            time.sleep(.03)
            return {'model_identity': self.model.identity, 'prepared_identity': 'b' * 64, 'runtime_identity': 'c' * 64,
                'choices': [{'finish_reason': 'stop', 'message': {'content': 'Complete answer for ' + payload['messages'][-1]['content']}}],
                'usage': {'completion_tokens': 10}, 'diagnostics': {'complete_final_answer': True, 'unexpected_thinking': False}}
        finally:
            with self.lock: self.active.remove(url)

    def build(self, **kwargs):
        return self.app.sculptor.build({'brief': 'A witty captain with perfect recall', 'dials': self.controls,
            'layers': [0, 1], 'count': 6, 'dose_budget': .5, **kwargs})

    def campaign_input(self, build):
        return {'build_id': build['id'], 'prompts': [{'prompt_id': row['id']} for row in self.prompts], 'seed': 17, 'max_tokens': 2048}

    def wait(self, identity):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            value = self.app.sculptor.get(identity)
            if value['state'] not in ('queued', 'running', 'stopping'): return value
            time.sleep(.01)
        self.fail('Campaign did not finish within bounded CPU test time')

    def test_correlated_trait_vectors_are_normalized_and_variants_deterministic_without_neuron_claims(self):
        maximum = {trait: 1. for trait in TRAITS}
        first, second = self.build(dials=maximum), self.build(dials=maximum)
        self.assertEqual(first['candidates'], second['candidates'])
        self.assertEqual(first['method'], 'manual_heuristic')
        self.assertEqual(first['candidates'][0]['dials'], maximum)
        self.assertEqual(len({canonical_digest(item['dials']) for item in first['candidates']}), 6)
        for candidate in first['candidates']:
            _, actual = self.app.bank.program(candidate['pushes'], candidate['phase'])
            self.assertLessEqual(actual['aggregate_relative_norm'], .5 + 1e-9)
            self.assertTrue(all(push['kind'] == 'trait' and push['dose'] > 0 for push in candidate['pushes']))
            self.assertIn('polarity require comparison', candidate['direction_status'])
        self.assertLess(first['candidates'][0]['normalization_scale'], 1.)
        with self.assertRaises(ValueError): self.build(dose_budget=.51)
        with self.assertRaises(ValueError): self.build(dials={**self.controls, 'openness': float('nan')})
        neutral = self.build(dials={key: 0 for key in TRAITS}, count=1)
        self.assertEqual(neutral['candidates'][0]['pushes'], [])

    def test_john_plan_is_strict_lease_bound_and_raw_attempts_are_persisted_without_silent_fallback(self):
        binding = {'url': 'http://127.0.0.1:18511', 'lease_id': 'owned', 'gpu_uuid': 'John', 'unit_invocation_id': 'invocation'}
        response = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'schema': PLAN_SCHEMA,
            'dials': self.controls, 'rationale': 'Tentative style mapping', 'unsupported_traits': ['Perfect recall is not a Big Five trait']})}}]}
        with patch('persona_sculptor.trusted_planner', return_value=binding) as admission, patch('persona_sculptor.fetch', return_value=response) as api:
            plan = self.app.sculptor.plan({'brief': 'A warm captain with perfect recall', 'mode': 'john'})
        self.assertEqual(admission.call_count, 2); self.assertEqual(plan['method'], 'john_tentative_interpretation')
        self.assertEqual(api.call_args.args[1]['response_format']['type'], 'json_schema')
        self.assertEqual(api.call_args.args[1]['response_format']['json_schema']['schema']['properties']['schema']['const'], PLAN_SCHEMA)
        self.assertNotIn('schema', api.call_args.args[1]['response_format'])
        with self.app.connect() as db:
            request, raw, status = db.execute('SELECT request,response,state FROM sculptor_plans WHERE id=?', (plan['id'],)).fetchone()
        self.assertEqual(json.loads(raw), response); self.assertEqual(status, 'complete')
        self.assertNotIn('manual_reviews', request)
        malformed = deepcopy(response); malformed['choices'][0]['message']['content'] = json.dumps([self.controls])
        with patch('persona_sculptor.trusted_planner', return_value=binding), patch('persona_sculptor.fetch', return_value=malformed):
            with self.assertRaisesRegex(ValueError, 'no silent fallback'): self.app.sculptor.plan({'brief': 'Anything'})
        with self.app.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM sculptor_plans WHERE state='failed' AND response IS NOT NULL").fetchone()[0], 1)
        with patch('persona_sculptor.trusted_planner', side_effect=ValueError('No owner lease')), patch('persona_sculptor.fetch') as api:
            with self.assertRaisesRegex(ValueError, 'No owner lease'): self.app.sculptor.plan({'brief': 'Offline'})
            api.assert_not_called()
        manual = self.app.sculptor.plan({'brief': 'Offline', 'mode': 'manual', 'dials': self.controls})
        self.assertEqual(manual['method'], 'manual_heuristic')

    def test_planner_requires_correct_running_owner_uuid_unit_and_verified_context(self):
        from persona_planner import GPU, UNIT
        lease = {'id': 'owned', 'project': 'character-creation', 'assigned_uuid': GPU, 'unit_name': UNIT,
                 'unit_invocation_id': 'invocation', 'status': 'running'}
        with patch('persona_planner.wb', return_value={'jobs': [lease]}), patch('persona_sculptor.check_admission', return_value=(True, 'ready')), \
                patch('persona_planner.readiness', return_value={'context_tokens': 131072, 'model_alias': 'persona-manager'}):
            self.assertEqual(trusted_planner()['lease_id'], 'owned')
        for changed in ({'assigned_uuid': 'other'}, {'unit_name': 'other.service'}, {'status': 'queued'}, {'unit_invocation_id': None}):
            with patch('persona_planner.wb', return_value={'jobs': [{**lease, **changed}]}), self.assertRaises(ValueError): trusted_planner()

    def test_campaign_uses_shared_six_endpoint_pool_and_one_neutral_baseline_per_prompt(self):
        build = self.build()
        barrier = threading.Barrier(6)
        def api(url, payload=None, timeout=None):
            if payload and payload['messages'][-1]['content'].startswith('Existing recipe lab work'):
                barrier.wait(5)
            return self.api(url, payload, timeout)
        legacy = {'model_identity': self.model.identity, 'seed': 99, 'max_tokens': 2048, 'thinking': False, 'smoke': False,
            'cells': [{'prompt': 'Existing recipe lab work ' + str(index), 'recipe': {'name': 'Baseline', 'pushes': [], 'phase': 'all'}} for index in range(6)]}
        with patch('persona_sculptor.fetch', api), patch('persona_explorer.fetch', api):
            legacy_id = self.app.submit_batch(legacy)
            campaign = self.app.sculptor.submit(self.campaign_input(build))
            result = self.wait(campaign['id'])
        self.assertEqual(result['state'], 'complete'); self.assertEqual(result['counts']['total'], 14)
        self.assertEqual(result['counts']['complete_answers'], 14)
        self.assertEqual(self.app.batch(legacy_id)['state'], 'complete')
        self.assertEqual(self.peak, 6)
        for prompt in self.prompts:
            requests = [request for request in self.requests if request['messages'][-1]['content'] == prompt['prompt']]
            self.assertEqual(len(requests), 7); self.assertEqual(sum('steering' not in request for request in requests), 1)
            self.assertTrue(all(request['messages'] == [{'role': 'user', 'content': prompt['prompt']}] for request in requests))
            self.assertTrue(all(request['seed'] == 17 and request['max_tokens'] == 2048 for request in requests))
            self.assertTrue(all('witty captain' not in json.dumps(request) for request in requests))
        with self.app.connect() as db:
            waves = [json.loads(row[0]) for row in db.execute('SELECT input FROM sculptor_waves WHERE campaign_id=? ORDER BY wave', (campaign['id'],))]
            raw = db.execute('SELECT COUNT(*) FROM cells WHERE raw IS NOT NULL').fetchone()[0]
        self.assertEqual(len(waves[0]['cells']), 2)
        self.assertTrue(all(not cell['recipe']['pushes'] for cell in waves[0]['cells']))
        self.assertTrue(all(len(wave['cells']) <= 6 for wave in waves)); self.assertEqual(raw, 20)
        self.assertTrue(result['prompts'][1]['read_only_evaluation']); self.assertFalse(result['prompts'][1]['adaptive_feedback_allowed'])
        self.assertTrue(all(cell['output']['provenance']['persona_campaign']['brief_injected'] is False for cell in result['cells']))

    def test_hhd_discovery_battery_preserves_provenance_without_injecting_rubric(self):
        authored = {'schema': 'character-creation.lab-battery.v1', 'status': 'authored-discovery-battery-not-validated',
            'prompts': [{'label': 'HHD fixture', 'prompt': 'Assess a fictional measurement anomaly.'}],
            'review_dimensions': ['Raw evidence'], 'review_scale': 'Discovery only'}
        path = self.root / 'data/persona_hhd_battery.json'
        path.write_text(json.dumps(authored))
        value = self.campaign_input(self.build())
        digest = self.app.sculptor.catalog()['hhd_battery']['canonical_sha256']
        value['prompts'] = [{'prompt': authored['prompts'][0]['prompt'], 'authored_battery_sha256': digest}]
        changed = dict(authored, review_scale='Changed rubric')
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'battery changed'):
            self.app.sculptor.submit(value)
        path.write_text(json.dumps(authored))
        with patch('persona_sculptor.fetch', self.api), patch('persona_explorer.fetch', self.api):
            campaign = self.app.sculptor.submit(value); result = self.wait(campaign['id'])
        path.write_text('{}')
        prompt = self.app.sculptor.get(campaign['id'])['prompts'][0]
        self.assertEqual(prompt['split'], 'discovery')
        self.assertEqual(prompt['source_binding']['snapshot'], authored)
        self.assertEqual(len(prompt['source_binding']['canonical_sha256']), 64)
        self.assertEqual(prompt['authored_battery_sha256'], digest)
        self.assertFalse(prompt['adaptive_feedback_allowed'])
        self.assertTrue(all(r['messages'] == [{'role': 'user', 'content': authored['prompts'][0]['prompt']}] for r in self.requests))

    def test_partial_generation_is_visible_not_retried_and_manual_reviews_are_durable(self):
        build = self.build(); failed = []
        def api(url, payload=None, timeout=None):
            result = self.api(url, payload, timeout)
            if payload and 'steering' in payload:
                with self.lock:
                    if not failed:
                        failed.append(True); result['choices'][0]['finish_reason'] = 'length'; result['diagnostics']['complete_final_answer'] = False
            return result
        with patch('persona_sculptor.fetch', api), patch('persona_explorer.fetch', api):
            campaign = self.app.sculptor.submit(self.campaign_input(build)); result = self.wait(campaign['id'])
        self.assertEqual(result['state'], 'partial'); self.assertEqual(len(self.requests), 14)
        self.assertEqual(result['counts']['complete_answers'], 13); self.assertEqual(result['counts']['failed_or_incomplete'], 1)
        review = self.app.sculptor.review({'campaign_id': campaign['id'], 'candidate_id': 'candidate-2', 'rating': 4, 'preferred': True})
        self.assertFalse(review['used_for_automated_planning'])
        self.assertEqual(self.app.sculptor.get(campaign['id'])['manual_reviews'], [review])

    def test_invalid_battery_identity_or_changed_calibration_reject_before_any_queue(self):
        build = self.build(); value = self.campaign_input(build)
        bad = {**value, 'prompts': [value['prompts'][0], {'prompt_id': 'missing'}]}
        with patch('persona_sculptor.fetch', self.api), patch('persona_explorer.fetch', self.api):
            with self.assertRaises(ValueError): self.app.sculptor.submit(bad)
            with self.assertRaisesRegex(ValueError, 'smoke'): self.app.sculptor.submit({**value, 'max_tokens': 256})
            for change in ({'candidate_ids': [{}]}, {'build_id': {}}, {'prompts': [{'prompt_id': {}}]}):
                with self.subTest(change=change), self.assertRaises(ValueError): self.app.sculptor.submit({**value, **change})
            def mismatch(url, *args, **kwargs):
                result = self.descriptor(url); result['data'][0]['flare']['runtime_identity'] = '0' * 64; return result
            with patch('persona_sculptor.fetch', mismatch), self.assertRaisesRegex(ValueError, 'runtime'):
                self.app.sculptor.submit(value)
        with self.app.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM sculptor_campaigns').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM batches').fetchone()[0], 0)
        with (self.root / 'results/layer_00.npz').open('ab') as stream: stream.write(b'changed calibration')
        with self.assertRaisesRegex(ValueError, 'Calibration changed'): self.app.sculptor.submit(value)

    def test_default_battery_uses_only_original_baseline_discovery_unique_source_groups(self):
        base = self.prompts[0]
        rows = [base, self.prompts[1], {**base, 'id': 'profile', 'labels': {'profile': 'extraversion_high'}},
                {**base, 'id': 'mutation', 'mutation': {'kind': 'instruction_negation'}},
                {**base, 'id': 'duplicate-group', 'category': 'other', 'prompt': 'Different wording from the same source group.'}]
        with patch.object(self.app, 'corpus_snapshot', return_value=({row['id']: row for row in rows}, {})), \
                patch.object(self.app, 'capacity', return_value={'configured': 6, 'available': 0, 'idle_status': 'unknown'}):
            catalog = self.app.sculptor.catalog()
        self.assertEqual(len(catalog['default_prompts']), 1)
        self.assertNotIn(catalog['default_prompts'][0]['id'], ('profile', 'mutation', self.prompts[1]['id']))
        self.assertEqual(catalog['defaults']['count'], 6); self.assertEqual(catalog['capacity']['available'], 0)

    def test_allowlisted_presets_preserve_approximation_sources_and_save_center_without_john(self):
        preset = {'id': 'four-o-approximation', 'name': '4o-style approximation', 'brief': 'A warm, conversational helper',
            'description': 'An approximate style target, not original model weights.', 'dials': self.controls,
            'dose_budget': .2, 'variation': .1, 'phase': 'all', 'evidence_basis': 'Human style description mapped heuristically to calibrated controls.',
            'unsupported_traits': ['Exact historical model behavior'], 'sources': [{'title': 'Fixture source', 'url': 'https://example.test/source'}],
            'notes': ['No claim of recovering original weights.']}
        path = self.root / 'data/persona_presets.json'
        path.write_text(json.dumps({'schema': 'character-creation.persona-presets.v1', 'presets': [preset]}))
        with patch('persona_sculptor.trusted_planner') as planner, patch('persona_sculptor.fetch') as api:
            first = self.app.sculptor.preset({'id': preset['id'], 'count': 6, 'seed': 19})
            second = self.app.sculptor.preset({'id': preset['id'], 'count': 1, 'seed': 19})
            planner.assert_not_called(); api.assert_not_called()
        self.assertEqual(first['plan']['method'], 'preset_heuristic'); self.assertTrue(first['preset']['approximation'])
        self.assertEqual(first['build']['preset']['sources'], preset['sources'])
        self.assertEqual(first['build']['preset']['notes'], preset['notes'])
        self.assertEqual(first['saved_recipe']['pushes'], first['build']['candidates'][0]['pushes'])
        self.assertEqual(first['saved_recipe']['id'], second['saved_recipe']['id'])
        self.assertEqual(len(self.app.saved_recipes()['recipes']), 1)
        rebuilt = self.app.sculptor.build({'plan_id': first['plan']['id'], 'count': 6})
        self.assertEqual(rebuilt['candidates'], first['build']['candidates'])
        for candidate in first['build']['candidates']: self.assertLessEqual(candidate['aggregate_relative_norm'], .2 + 1e-9)
        with self.assertRaisesRegex(ValueError, 'allowlisted'): self.app.sculptor.preset({'id': 'invented-by-model'})
        with self.assertRaisesRegex(ValueError, 'strict proposal schema'): self.app.sculptor.preset({'id': preset['id'], 'dials': self.controls})
        preset['sources'][0]['url'] = 'javascript:alert(1)'
        path.write_text(json.dumps({'schema': 'character-creation.persona-presets.v1', 'presets': [preset]}))
        with self.assertRaisesRegex(ValueError, 'HTTP'): self.app.sculptor.presets()

    def test_cancellation_drains_current_baseline_wave_and_never_submits_candidates(self):
        build = self.build(); entered = threading.Event(); release = threading.Event()
        def api(url, payload=None, timeout=None):
            if payload:
                entered.set(); self.assertTrue(release.wait(3))
            return self.api(url, payload, timeout)
        with patch('persona_sculptor.fetch', api), patch('persona_explorer.fetch', api):
            campaign = self.app.sculptor.submit(self.campaign_input(build)); self.assertTrue(entered.wait(2))
            self.app.sculptor.cancel({'id': campaign['id']}); release.set(); result = self.wait(campaign['id'])
        self.assertEqual(result['state'], 'cancelled'); self.assertEqual(len(self.requests), 2)
        self.assertEqual(sum(cell['state'] == 'complete' for cell in result['cells']), 2)
        self.assertEqual(sum(cell['state'] == 'cancelled' for cell in result['cells']), 12)

    def test_restart_preserves_snapshot_marks_unknown_work_interrupted_and_never_retries(self):
        build = self.build()
        with patch('persona_sculptor.fetch', self.api), patch.object(self.app.executor, 'submit'):
            campaign = self.app.sculptor.submit({**self.campaign_input(build), 'candidate_ids': ['candidate-3']})
        self.app.close()
        with patch('persona_sculptor.fetch') as api:
            self.app = Explorer(self.root, self.config); result = self.app.sculptor.get(campaign['id']); api.assert_not_called()
        self.assertEqual(result['state'], 'interrupted'); self.assertEqual(result['build'], build)
        self.assertEqual(result['input']['candidate_ids'], ['candidate-3'])
        self.assertTrue(all(cell['state'] == 'interrupted' for cell in result['cells']))

    def test_sculptor_routes_require_json_and_serve_only_explicit_assets(self):
        (self.root / 'ui/persona_sculptor.js').write_text('fixed studio asset')
        request = object.__new__(handler(self.app)); request.send = Mock()
        request.send_response = Mock(); request.send_header = Mock(); request.end_headers = Mock(); request.wfile = io.BytesIO()
        request.path = '/persona_sculptor.js'; request.do_GET(); self.assertEqual(request.wfile.getvalue(), b'fixed studio asset')
        body = json.dumps({'brief': 'Manual offline', 'mode': 'manual', 'dials': self.controls}).encode()
        request.path = '/api/sculptor/plan'; request.headers = {'Host': 'localhost:8792', 'Content-Length': str(len(body)), 'Content-Type': 'text/plain'}
        request.rfile = io.BytesIO(body); request.do_POST(); self.assertEqual(request.send.call_args.args[1], 400)
        request.headers['Content-Type'] = 'application/json'; request.rfile = io.BytesIO(body); request.do_POST()
        self.assertEqual(request.send.call_args.args[0]['method'], 'manual_heuristic')


if __name__ == '__main__': unittest.main()
