"""CPU-only persistence, source binding and shared inference-slot checks."""
from copy import deepcopy
import hashlib
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
from persona_explorer import Explorer, handler, verify_endpoint
from persona_vectors import ModelSpec


class ExplorerIdentityTests(unittest.TestCase):
    def test_diagnostic_binding_cannot_be_omitted_or_reused_for_changed_code(self):
        endpoint = {"prepared_identity": "p", "model_identity": "m", "runtime_identity": "r"}
        descriptor = {"flare": dict(endpoint)}
        self.assertEqual(verify_endpoint(descriptor, endpoint)["runtime_identity"], "r")
        for change in ({"runtime_identity": "other"}, {"model_identity": "other"}, {"prepared_identity": "other"}):
            with self.assertRaises(ValueError):
                verify_endpoint({"flare": {**descriptor["flare"], **change}}, endpoint)
        with self.assertRaises(ValueError):
            verify_endpoint(descriptor, {k: v for k, v in endpoint.items() if k != "runtime_identity"})

    def test_native_package_remains_usable_without_diagnostic_runtime_extension(self):
        endpoint = {"prepared_identity": "p", "model_identity": "m"}
        self.assertEqual(verify_endpoint({"flare": {"prepared_identity": "p"}}, endpoint), {"prepared_identity": "p"})


class ExplorerWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ('data', 'results', 'ui'): (self.root / name).mkdir()
        self.model = ModelSpec('test/qwen', 'a' * 40, 'test-fp32', 2, 4, 5)
        (self.root / 'data/persona_model.json').write_text(json.dumps(self.model.__dict__))
        np.savez(self.root / 'results/layer_00.npz', model_identity=self.model.identity,
            residual_median=10., neuron_ids=np.array([1, 3]),
            down_columns=np.array([[3., 0.], [4., 0.], [0., 0.], [0., 2.]]))
        self.config = {'endpoints': [{'url': f'http://127.0.0.1:{9000 + i}', 'label': f'Quincy {i + 1}',
            'model_identity': self.model.identity, 'prepared_identity': 'prepared', 'runtime_identity': 'r' * 64}
            for i in range(6)]}
        self.app = Explorer(self.root, self.config); self.addCleanup(self.app.close)
        source = {'path': '/private/corpus.jsonl', 'sha256': 'b' * 64, 'bytes': 100}
        self.rows = [record(source, text, corpus=corpus, category='reasoning', split=split,
            locator={'row': i}, system='Registered system instruction' if i == 0 else '',
            max_tokens=3072 if corpus == 'scotus' else 2048)
            for i, (text, corpus, split) in enumerate([
                ('Analyze this legal holding.', 'scotus', 'discovery'),
                ('A heldout legal question.', 'scotus', 'heldout'),
                ('Describe a rainy library.', 'big_five', 'discovery')])]
        self.publish(self.rows)

    def publish(self, rows):
        raw = ''.join(json.dumps(row) + '\n' for row in rows).encode()
        (self.root / 'data/persona_manager_corpus.jsonl').write_bytes(raw)
        (self.root / 'data/persona_manager_corpus_manifest.json').write_text(json.dumps({
            'status': 'complete', 'output_sha256': hashlib.sha256(raw).hexdigest()}))

    def descriptor(self, url):
        endpoint = next(item for item in self.config['endpoints'] if url.startswith(item['url'] + '/'))
        return {'data': [{'id': 'served', 'flare': {**{key: endpoint[key] for key in
            ('model_identity', 'prepared_identity', 'runtime_identity')}, 'steering': {
                'protocol': 'additive-bf16-sites-v1', 'max_sites': 64,
                'sites': [{'name': 'layers.0.mlp_input', 'width': 5}]}}}]}

    def response(self, payload):
        return {'model_identity': self.model.identity, 'prepared_identity': 'prepared', 'runtime_identity': 'r' * 64,
            'choices': [{'message': {'content': 'Answer: ' + payload['messages'][-1]['content']}, 'finish_reason': 'stop'}],
            'usage': {'completion_tokens': 12}, 'diagnostics': {'complete_final_answer': True, 'unexpected_thinking': False}}

    def api(self, url, payload=None, timeout=None):
        return self.descriptor(url) if payload is None else self.response(payload)

    def batch_input(self, count=1):
        return {'model_identity': self.model.identity, 'seed': 7, 'max_tokens': 2048, 'thinking': False, 'smoke': False,
                'cells': [{'label': f'Cell {i}', 'prompt': f'Prompt {i}',
                           'recipe': {'name': 'Baseline', 'pushes': [], 'phase': 'all'}} for i in range(count)]}

    def wait_batch(self, identity):
        end = time.monotonic() + 5
        while time.monotonic() < end:
            value = self.app.batch(identity)
            if all(cell['state'] not in ('queued', 'running') for cell in value['cells']): return value
            time.sleep(.01)
        self.fail('Batch did not finish in bounded CPU test')

    def test_corpus_filters_hash_gate_and_recipe_roundtrip(self):
        self.assertEqual(self.app.corpora()['corpora'], [
            {'name': 'big_five', 'rows': 1, 'discovery': 1, 'heldout': 0},
            {'name': 'scotus', 'rows': 2, 'discovery': 1, 'heldout': 1}])
        result = self.app.prompts(corpus='scotus', split='all', q='legal', offset=1, limit=1)
        self.assertEqual(result['total'], 2); self.assertEqual(result['rows'][0]['split'], 'heldout')
        recipe = {'name': 'Named push', 'model_identity': self.model.identity,
                  'pushes': [{'kind': 'neuron', 'layer': 0, 'neuron': 1, 'dose': .2}], 'phase': 'prefill'}
        saved = self.app.save_recipe(recipe)
        updated = self.app.save_recipe({**recipe, 'id': saved['id'], 'name': 'Updated', 'phase': 'decode'})
        self.assertEqual(updated['created'], saved['created'])
        self.assertEqual(self.app.saved_recipes()['recipes'], [updated])
        for bad in ({**recipe, 'model_identity': 'other'}, {**recipe, 'id': 'missing'},
                    {**recipe, 'pushes': [{'layer': 0, 'neuron': 999, 'dose': .2}]}):
            with self.assertRaises(ValueError): self.app.save_recipe(bad)
        manifest = self.root / 'data/persona_manager_corpus_manifest.json'
        value = json.loads(manifest.read_text()); value['status'] = 'ingesting'; manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'not complete'): self.app.corpora()
        value['status'] = 'complete'; value['output_sha256'] = '0' * 64; manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'does not match'): self.app.prompts()

    def test_prompt_labels_distinguish_sources_mutations_and_profiles_without_exposing_paths(self):
        source = {'path': '/private/books/Example Novel.epub', 'sha256': 'b' * 64, 'bytes': 100}
        book = record(source, 'A passage.', corpus='skippy', category='book_character_dialogue', split='discovery',
                      locator={'member': 'spine/chapter2.xhtml', 'char_start': 120}, mutation={'kind': 'speaker_counterfactual'})
        task = record(source, 'A task.', corpus='big_five', category='behavior', split='discovery',
                      locator={'prompt_id': 'task-7'}, labels={'profile': 'openness_high'})
        inventory = record(source, 'An inventory item.', corpus='big_five', category='ipip50_inventory', split='heldout',
                      locator={'inventory_item_id': 12}, labels={'inventory_item': {'id': 12, 'dimension': 'agreeableness'}})
        legal = record(source, 'A legal task.', corpus='scotus', category='legal', split='discovery',
                       locator={'jsonl_line': 5}, original_prompt_id='case-5', mutation={'kind': 'instruction_negation'}, max_tokens=3072)
        rows = self.rows + [book, task, inventory, legal]; before = deepcopy(rows)
        self.publish(rows)
        checks = [('Example Novel', book, ('chapter2.xhtml', '@120', 'speaker_counterfactual')),
                  ('openness_high', task, ('task-7', 'openness_high')),
                  ('IPIP 12', inventory, ('12', 'agreeableness')),
                  ('case-5', legal, ('case-5', 'instruction_negation'))]
        for search, row, required in checks:
            result = self.app.prompts(split='all', q=search)
            self.assertEqual(result['total'], 1)
            label = result['rows'][0]['label']
            self.assertLessEqual(len(label), 160); self.assertTrue(label.endswith(row['id'][:12]))
            self.assertNotIn('/private/', label)
            self.assertTrue(all(value in label for value in required))
        self.assertEqual(rows, before)

    def test_invalid_cells_and_capabilities_reject_the_whole_batch_before_inference(self):
        original = self.batch_input(2)
        variants = []
        bad = deepcopy(original); bad['cells'][1]['prompt_id'] = self.rows[0]['id']; variants.append(bad)
        bad = deepcopy(original); bad['cells'][1]['recipe']['phase'] = 'bad'; variants.append(bad)
        bad = deepcopy(original); bad['cells'][1]['recipe']['pushes'] = [{'layer': 0, 'neuron': 99, 'dose': .2}]; variants.append(bad)
        bad = deepcopy(original); bad['thinking'] = 'false'; variants.append(bad)
        bad = deepcopy(original); bad['model_identity'] = 'other'; variants.append(bad)
        with patch('persona_explorer.fetch', self.api) as api:
            for value in variants:
                with self.assertRaises(ValueError): self.app.submit_batch(value)
        self.assertEqual(self.app.batches(), {'batches': []})
        with self.app.connect() as db: self.assertEqual(db.execute('SELECT COUNT(*) FROM cells').fetchone()[0], 0)
        bad = self.batch_input(); bad['cells'][0]['recipe']['pushes'] = [{'layer': 0, 'neuron': 1, 'dose': .2}]
        def unsupported(url, payload=None, timeout=None):
            self.assertIsNone(payload)
            value = self.descriptor(url); value['data'][0]['flare']['steering']['max_sites'] = 0
            return value
        with patch('persona_explorer.fetch', unsupported), self.assertRaisesRegex(ValueError, 'site capacity'):
            self.app.submit_batch(bad)
        self.assertEqual(self.app.batches(), {'batches': []})

    def test_six_cells_use_distinct_endpoints_and_legacy_comparison_shares_pool(self):
        gate, all_six, legacy_started = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(gate.set)
        lock = threading.Lock(); active = {}; maximum = {}; calls = []; peak = [0]
        def api(url, payload=None, timeout=None):
            if payload is None: return self.descriptor(url)
            endpoint = url.rsplit('/', 2)[0]
            with lock:
                active[endpoint] = active.get(endpoint, 0) + 1
                maximum[endpoint] = max(maximum.get(endpoint, 0), active[endpoint])
                calls.append(endpoint); peak[0] = max(peak[0], sum(active.values()))
                if len(calls) == 6: all_six.set()
            try:
                if payload['messages'][-1]['content'].startswith('Prompt'):
                    if not gate.wait(5): raise RuntimeError('CPU test gate expired')
                return self.response(payload)
            finally:
                with lock: active[endpoint] -= 1
        original_run = self.app.run_comparison
        def legacy(*args): legacy_started.set(); return original_run(*args)
        with patch('persona_explorer.fetch', api), patch.object(self.app, 'run_comparison', legacy):
            identity = self.app.submit_batch(self.batch_input(6))
            self.assertTrue(all_six.wait(5)); self.assertEqual(len(set(calls)), 6)
            comparison = self.app.submit({'model_identity': self.model.identity, 'prompt': 'Legacy prompt', 'pushes': []})
            self.assertTrue(legacy_started.wait(2))
            self.assertEqual(len(calls), 6)
            gate.set(); result = self.wait_batch(identity)
            end = time.monotonic() + 5
            while self.app.comparison(comparison)['state'] in ('queued', 'running') and time.monotonic() < end: time.sleep(.01)
            self.assertEqual(self.app.comparison(comparison)['state'], 'complete')
        self.assertEqual(result['state'], 'complete'); self.assertEqual(peak[0], 6)
        self.assertTrue(all(value == 1 for value in maximum.values())); self.assertEqual(len(calls), 9)

    def test_prompt_recipe_and_program_snapshots_are_immutable_and_vectors_stay_private(self):
        saved = self.app.save_recipe({'name': 'Original', 'model_identity': self.model.identity, 'phase': 'prefill',
            'pushes': [{'layer': 0, 'neuron': 1, 'dose': .2}]})
        value = self.batch_input(); value['cells'] = [{'prompt_id': self.rows[0]['id'], 'recipe_id': saved['id']}]
        with patch('persona_explorer.fetch', self.api), patch.object(self.app.cell_executor, 'submit'):
            identity = self.app.submit_batch(value)
        self.app.save_recipe({'name': 'Changed', 'id': saved['id'], 'model_identity': self.model.identity, 'phase': 'decode', 'pushes': []})
        self.publish(self.rows[1:])
        with self.assertRaisesRegex(ValueError, 'preserve existing'): self.app.prompts()
        with self.app.connect() as db:
            snapshot = json.loads(db.execute('SELECT snapshot FROM cells WHERE batch_id=?', (identity,)).fetchone()[0])
        with patch('persona_explorer.fetch', self.api): self.app.run_cell(identity, 0, snapshot, value)
        result = self.app.batch(identity); cell = result['cells'][0]
        self.assertEqual(cell['recipe']['name'], 'Original'); self.assertEqual(cell['prompt']['text'], self.rows[0]['prompt'])
        self.assertEqual(cell['output']['provenance']['source']['id'], self.rows[0]['id'])
        self.assertNotIn('"vector"', json.dumps(result))
        with self.app.connect() as db:
            raw = json.loads(db.execute('SELECT raw FROM cells WHERE batch_id=?', (identity,)).fetchone()[0])
        self.assertEqual(raw['request']['messages'][0]['content'], 'Registered system instruction')
        self.assertEqual(len(raw['request']['steering']['sites'][0]['vector']), 5)
        self.assertIn('response', raw)

    def test_scotus_short_budget_requires_explicit_nonpromotable_smoke(self):
        value = self.batch_input(); value['max_tokens'] = 512
        value['cells'][0].pop('prompt'); value['cells'][0]['prompt_id'] = self.rows[0]['id']
        with patch('persona_explorer.fetch', self.api):
            with self.assertRaisesRegex(ValueError, 'smoke:true'): self.app.submit_batch(value)
            value['smoke'] = True
            result = self.wait_batch(self.app.submit_batch(value))
        budget = result['cells'][0]['output']['provenance']['budget']
        self.assertTrue(budget['short_answer_budget']); self.assertTrue(budget['smoke'])
        self.assertFalse(budget['promotion_eligible']); self.assertFalse(budget['promotion_eligible_budget'])
        self.assertIn('SMOKE', budget['budget_note'])

    def test_partial_results_are_incremental_and_failed_inference_is_not_retried(self):
        gate, slow_entered, fast_finished = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(gate.set)
        calls = []
        def api(url, payload=None, timeout=None):
            if payload is None: return self.descriptor(url)
            prompt = payload['messages'][-1]['content']; calls.append(prompt)
            if prompt == 'Prompt 0': raise RuntimeError('one worker failed')
            if prompt == 'Prompt 1': slow_entered.set(); gate.wait(5)
            else: fast_finished.set()
            return self.response(payload)
        with patch('persona_explorer.fetch', api):
            identity = self.app.submit_batch(self.batch_input(3))
            self.assertTrue(slow_entered.wait(3)); self.assertTrue(fast_finished.wait(3))
            end = time.monotonic() + 3
            while time.monotonic() < end:
                cells = self.app.batch(identity)['cells']
                if cells[0]['state'] == 'failed' and cells[2]['state'] == 'complete': break
                time.sleep(.01)
            self.assertEqual([row['state'] for row in cells], ['failed', 'running', 'complete'])
            gate.set(); result = self.wait_batch(identity)
        self.assertEqual(result['state'], 'partial'); self.assertEqual(calls.count('Prompt 0'), 1)
        self.assertEqual(self.app.batches()['batches'][0]['completed'], 3)

    def test_restart_marks_unfinished_cells_interrupted_without_replaying_successes(self):
        value = self.batch_input(2)
        with patch('persona_explorer.fetch', self.api), patch.object(self.app.cell_executor, 'submit'):
            identity = self.app.submit_batch(value)
        with self.app.connect() as db:
            snapshot = json.loads(db.execute('SELECT snapshot FROM cells WHERE batch_id=? AND cell_index=0', (identity,)).fetchone()[0])
        with patch('persona_explorer.fetch', self.api): self.app.run_cell(identity, 0, snapshot, value)
        before = self.app.batch(identity)['cells'][0]['output']
        self.app.close()
        with patch('persona_explorer.fetch') as api:
            restarted = Explorer(self.root, self.config); self.addCleanup(restarted.close)
            result = restarted.batch(identity); api.assert_not_called()
        self.assertEqual([row['state'] for row in result['cells']], ['complete', 'interrupted'])
        self.assertEqual(result['cells'][0]['output'], before)
        self.assertIn('no automatic retry', result['cells'][1]['error'])

    def test_response_identity_failure_retains_raw_evidence_and_capacity_is_cached(self):
        def changed(url, payload=None, timeout=None):
            if payload is None: return self.descriptor(url)
            value = self.response(payload); value['runtime_identity'] = 'other'; return value
        with patch('persona_explorer.fetch', changed): result = self.wait_batch(self.app.submit_batch(self.batch_input()))
        self.assertEqual(result['state'], 'failed'); self.assertIn('runtime_identity', result['cells'][0]['error'])
        with self.app.connect() as db:
            raw = json.loads(db.execute('SELECT raw FROM cells WHERE batch_id=?', (result['id'],)).fetchone()[0])
        self.assertEqual(raw['response']['runtime_identity'], 'other')
        parallel = threading.Barrier(6)
        def capacity(url, payload=None, timeout=None):
            self.assertEqual(timeout, 2)
            parallel.wait(2)
            if ':9000/' in url: raise OSError('worker stopped')
            value = self.descriptor(url)
            if ':9001/' in url: value['data'][0]['flare']['runtime_identity'] = 'other'
            return value
        with patch('persona_explorer.fetch', side_effect=capacity) as api:
            first = self.app.capacity(); second = self.app.capacity()
            self.assertEqual(api.call_count, 6); self.assertEqual(first, second)
        self.assertEqual(first['configured'], 6); self.assertEqual(first['available'], 4)
        self.assertTrue(all(card['idle'] is None for card in first['cards']))

    def test_one_offline_endpoint_admits_six_cells_in_waves_on_only_five_verified_workers(self):
        gate, all_five = threading.Event(), threading.Event()
        self.addCleanup(gate.set)
        lock = threading.Lock(); active = {}; calls = []; peak = [0]
        def api(url, payload=None, timeout=None):
            if payload is None:
                if ':9000/' in url: raise OSError('worker stopped')
                return self.descriptor(url)
            self.assertNotIn(':9000/', url)
            with lock:
                active[url] = active.get(url, 0) + 1
                self.assertEqual(active[url], 1)
                calls.append(url); number = len(calls)
                peak[0] = max(peak[0], sum(active.values()))
                if number == 5: all_five.set()
            try:
                if number <= 5 and not gate.wait(5): raise RuntimeError('CPU test gate expired')
                return self.response(payload)
            finally:
                with lock: active[url] -= 1
        with patch('persona_explorer.fetch', api):
            identity = self.app.submit_batch(self.batch_input(6))
            self.assertTrue(all_five.wait(3)); self.assertEqual(len(calls), 5)
            admitted = self.app.batch(identity)['input']['admitted_endpoints']
            self.assertEqual(len(admitted), 5)
            self.assertTrue(all(item['runtime_identity'] == 'r' * 64 for item in admitted))
            gate.set(); result = self.wait_batch(identity)
        self.assertEqual(result['state'], 'complete'); self.assertEqual(len(calls), 6); self.assertEqual(peak[0], 5)
        bad = self.batch_input()
        def mismatch(url, payload=None, timeout=None):
            if ':9000/' in url: raise OSError('worker stopped')
            value = self.descriptor(url)
            if ':9001/' in url: value['data'][0]['flare']['runtime_identity'] = 'wrong'
            return value
        with patch('persona_explorer.fetch', mismatch), self.assertRaisesRegex(ValueError, 'runtime identity'):
            self.app.submit_batch(bad)
        self.assertEqual(len(self.app.batches()['batches']), 1)

    def test_worker_thinking_and_context_limits_reject_before_queue_and_appear_in_capacity(self):
        posts = []
        def api(url, payload=None, timeout=None):
            if payload is not None: posts.append(payload); return self.response(payload)
            value = self.descriptor(url)
            value['data'][0]['flare'].update(thinking_enabled_supported=False,
                physical_context_tokens=4096, logical_context_tokens=4096)
            return value
        with patch('persona_explorer.fetch', api):
            for change, message in (({'thinking': True}, 'thinking-enabled'), ({'max_tokens': 4096}, 'prompt also needs space')):
                with self.assertRaisesRegex(ValueError, message): self.app.submit_batch({**self.batch_input(), **change})
            self.assertEqual(self.app.batches(), {'batches': []}); self.assertEqual(posts, [])
            capacity = self.app.capacity()
            self.assertIs(capacity['thinking_supported'], False); self.assertEqual(capacity['context_tokens'], 4096)
            self.assertTrue(all(card['context_tokens'] == 4096 for card in capacity['cards']))
            self.assertEqual(self.wait_batch(self.app.submit_batch(self.batch_input()))['state'], 'complete')

    def test_static_routes_are_explicit_and_batch_routes_return_contract(self):
        (self.root / 'ui/persona_lab.js').write_text('safe asset')
        request = object.__new__(handler(self.app))
        request.path = '/persona_lab.js'; request.wfile = io.BytesIO()
        request.send_response = Mock(); request.send_header = Mock(); request.end_headers = Mock(); request.send = Mock()
        request.do_GET(); self.assertEqual(request.wfile.getvalue(), b'safe asset')
        request.path = '/ui/../../persona_explorer.py'; request.do_GET()
        request.send.assert_called_with({'error': 'Not found'}, 404)
        request.path = '/api/prompts?corpus=scotus&split=heldout&limit=1'; request.do_GET()
        self.assertEqual(request.send.call_args.args[0]['total'], 1)


if __name__ == "__main__":
    unittest.main()
