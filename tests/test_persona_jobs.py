"""CPU-only scheduling, synthetic validation, resume and publication checks."""

import persona  # Make the flat, deployment-compatible source modules importable.
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from persona_jobs import FACETS, ITEM_SCHEMA, PLAN_SCHEMA, JobStore, publish_samples, run_big_five, run_scheduled_job, scheduled_time, validate_items, validate_plan
from persona_vectors import ModelSpec


class PersonaJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'data').mkdir(); (self.root / 'results').mkdir()
        self.model = ModelSpec('test/qwen', 'a' * 40, 'test', 2, 4, 5)
        (self.root / 'data/persona_model.json').write_text(json.dumps(self.model.__dict__))
        raw = (json.dumps({'id': 'existing', 'prompt': 'An existing source prompt stays unchanged.', 'split': 'heldout'}) + '\n').encode()
        (self.root / 'data/persona_manager_corpus.jsonl').write_bytes(raw)
        (self.root / 'data/persona_manager_corpus_manifest.json').write_text(json.dumps({'status': 'complete', 'output_sha256': hashlib.sha256(raw).hexdigest()}))
        self.store = JobStore(self.root)
        self.value = {'kind': 'big_five_expand', 'name': 'CPU smoke', 'start_at': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
                      'timezone': 'America/Los_Angeles', 'duration_minutes': 15, 'target_entries': 72, 'max_entries': 72, 'seed': 7}
        self.session = {'planner': {'url': 'http://127.0.0.1:18511', 'model': 'planner'}, 'endpoints': [
            {'url': f'http://127.0.0.1:{8930+i}', 'model_identity': self.model.identity, 'prepared_identity': 'prepared'} for i in range(6)]}
        self.assignments = [{'trait': trait, 'facet': FACETS[trait][0], 'domain': f'domain-{i}', 'scenario_focus': 'Original decisions'}
                            for i, trait in enumerate([*FACETS, 'openness'])]

    def api(self, url, payload=None, timeout=None):
        if payload is None:
            return {'data': [{'id': 'served', 'flare': {'prepared_identity': 'prepared', 'model_identity': self.model.identity,
                'steering': {'protocol': 'additive-bf16-sites-v1', 'max_sites': 64, 'sites': []}}}]}
        if payload['model'] == 'planner': value = {'schema': PLAN_SCHEMA, 'batches': self.assignments, 'rationale': 'Diverse coverage'}
        else:
            task = json.loads(payload['messages'][-1]['content']); assignment = task['assignment']
            value = {'schema': ITEM_SCHEMA, 'prompts': [f"Scenario {task['worker_index']}-{i}: Your group faces an unexpected choice. Explain what you would do and why." for i in range(12)]}
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}

    def test_real_runner_contract_uses_all_six_workers_and_counts_only_valid_synthetic_rows(self):
        job = self.store.create(self.value)
        barrier = threading.Barrier(6); workers = set(); lock = threading.Lock()
        def api(url, payload=None, timeout=None):
            if payload and payload['model'] == 'served':
                with lock: workers.add(url)
                barrier.wait(3)
            return self.api(url, payload, timeout)
        with patch('persona_jobs.fetch', api), patch('persona_manager.fetch', api):
            report = run_big_five(job, self.session, self.store, threading.Event(), max_rounds=1, publish=False)
        self.assertEqual(len(workers), 6); self.assertEqual(report['accepted_unique'], 72)
        self.assertEqual(report['requests'], 6); self.assertTrue(report['target_met'])
        with self.store.connect() as db:
            rows = [json.loads(row[0]) for row in db.execute('SELECT body FROM samples')]
            self.assertEqual(db.execute('SELECT COUNT(*) FROM requests WHERE response IS NOT NULL').fetchone()[0], 7)
        self.assertTrue(all(row['split'] == 'discovery' and row['synthetic'] and not row['labels']['validated_inventory'] for row in rows))
        self.assertTrue(all(row['labels']['measured_trait'] is None for row in rows))
        self.assertTrue(all(row['generator']['request_sha256'] for row in rows))

    def test_incomplete_generation_is_quarantined_without_retry_or_partial_acceptance(self):
        job = self.store.create(self.value); calls = []
        def api(url, payload=None, timeout=None):
            response = self.api(url, payload, timeout)
            if payload and payload['model'] == 'served':
                calls.append(url)
                if ':8930/' in url: response['choices'][0]['finish_reason'] = 'length'
            return response
        with patch('persona_jobs.fetch', api), patch('persona_manager.fetch', api):
            result = run_big_five(job, self.session, self.store, threading.Event(), max_rounds=1)
        self.assertEqual(result['accepted_unique'], 60); self.assertEqual(result['errors'], 1)
        self.assertEqual(len(calls), 6); self.assertEqual(calls.count('http://127.0.0.1:8930/v1/chat/completions'), 1)

    def test_normalized_duplicates_and_wrong_labels_do_not_count(self):
        job = self.store.create(self.value)
        def api(url, payload=None, timeout=None):
            response = self.api(url, payload, timeout)
            if payload and payload['model'] == 'served':
                data = json.loads(response['choices'][0]['message']['content'])
                data['prompts'][0] = 'An existing source prompt stays unchanged!'
                data['prompts'][1] = {'prompt': 'untrusted object', 'trait': 'invented-trait'}
                response['choices'][0]['message']['content'] = json.dumps(data)
            return response
        with patch('persona_jobs.fetch', api), patch('persona_manager.fetch', api):
            result = run_big_five(job, self.session, self.store, threading.Event(), max_rounds=1)
        self.assertEqual(result['accepted_unique'], 60); self.assertEqual(result['rejected'], 12)
        self.assertEqual(result['errors'], 0)

    def test_planner_text_bounds_remain_enforced_after_compact_sampler_grammar(self):
        plan = {'schema': PLAN_SCHEMA, 'batches': deepcopy(self.assignments), 'rationale': 'Bounded explanation'}
        for field, limit in (('domain', 80), ('scenario_focus', 300)):
            with self.subTest(field=field):
                invalid = deepcopy(plan); invalid['batches'][0][field] = 'x' * (limit + 1)
                with self.assertRaisesRegex(ValueError, 'excessive'): validate_plan(invalid)
        invalid = deepcopy(plan); invalid['rationale'] = 'x' * 2001
        with self.assertRaisesRegex(ValueError, 'rationale'): validate_plan(invalid)
        invalid = deepcopy(plan); invalid['batches'][0]['facet'] = 'diligence'
        with self.assertRaisesRegex(ValueError, 'trait/facet'): validate_plan(invalid)

    def test_invalid_bare_planner_array_is_audited_then_new_schema_constrained_attempt_succeeds(self):
        job = self.store.create(self.value); plans = []
        def api(url, payload=None, timeout=None):
            response = self.api(url, payload, timeout)
            if payload and payload['model'] == 'planner':
                plans.append(deepcopy(payload))
                if len(plans) == 1: response['choices'][0]['message']['content'] = json.dumps(self.assignments)
            return response
        with patch('persona_jobs.fetch', api), patch('persona_manager.fetch', api):
            result = run_big_five(job, self.session, self.store, threading.Event(), max_rounds=1)
        self.assertEqual(result['accepted_unique'], 72); self.assertEqual(len(plans), 2)
        self.assertEqual(plans[1]['response_format']['type'], 'json_schema')
        self.assertEqual(plans[1]['response_format']['json_schema']['schema']['properties']['schema']['const'], PLAN_SCHEMA)
        self.assertNotIn('schema', plans[1]['response_format'])
        self.assertNotEqual(plans[0]['seed'], plans[1]['seed'])
        self.assertIn('previous_validation_error', plans[1]['messages'][-1]['content'])
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM requests WHERE role='planner' AND state='error'").fetchone()[0], 1)

    def test_schedule_timezone_dst_and_untrusted_fields_fail_closed(self):
        with self.assertRaises(ValueError): scheduled_time({'local_start': '2026-03-08T02:30'}, 'America/Los_Angeles')
        with self.assertRaises(ValueError): scheduled_time({'local_start': '2026-11-01T01:30'}, 'America/Los_Angeles')
        a = scheduled_time({'start_at': '2026-11-01T01:30:00-07:00'}, 'America/Los_Angeles')
        b = scheduled_time({'start_at': '2026-11-01T01:30:00-08:00'}, 'America/Los_Angeles')
        self.assertEqual(b.timestamp() - a.timestamp(), 3600)
        for field, value in [('shell', 'anything'), ('kind', 'shell'), ('duration_minutes', 961), ('target_entries', 73), ('seed', True)]:
            with self.subTest(field=field), self.assertRaises(ValueError): self.store.create({**self.value, field: value})
        with self.assertRaisesRegex(ValueError, 'explicit target'):
            self.store.create({**self.value, 'kind': 'neighborhood_hillclimb'})
        unmapped = self.store.create({**self.value, 'kind': 'neighborhood_hillclimb', 'targets': [{'layer': 1, 'neuron': 4}]})
        self.assertEqual(unmapped['targets'], [{'layer': 1, 'neuron': 4}])

    def test_atomic_claim_single_active_cancel_and_expired_windows(self):
        first = self.store.create(self.value); second = self.store.create({**self.value, 'name': 'Second'})
        with ThreadPoolExecutor(max_workers=6) as pool:
            claims = list(pool.map(lambda _: self.store.claim_due(), range(6)))
        self.assertEqual(sum(job is not None for job in claims), 1)
        active = next(job for job in claims if job is not None)
        other = second if active['id'] == first['id'] else first
        cancelled = self.store.cancel(other['id'])
        self.assertEqual(cancelled['state'], 'cancelled')
        self.assertIsNotNone(cancelled['finished'])
        self.assertEqual(self.store.cancel(active['id'])['state'], 'stopping')
        self.assertIsNone(self.store.claim_due())
        self.store.mark(active['id'], 'cancelled')
        third = self.store.create({**self.value, 'name': 'Expired'})
        self.assertIsNone(self.store.claim_due(now=third['deadline'] + 1))
        self.assertEqual(self.store.get(third['id'])['state'], 'expired')

    def test_recovered_cancelled_and_expired_resource_obligations_survive_another_restart(self):
        for cancelled in (True, False):
            job = self.store.create({**self.value, 'name': f'Recovery {cancelled}'})
            self.store.claim_due()
            self.store.event(job['id'], 'service_launch_intent', {'unit': 'owned-test-token'})
            if cancelled: self.store.cancel(job['id'])
            else:
                with self.store.connect() as db:
                    spec = {**job, 'deadline': 0}
                    db.execute('UPDATE jobs SET spec=? WHERE id=?', (json.dumps(spec), job['id']))
            self.assertIn(job['id'], self.store.recover())
            self.assertIsNotNone(self.store.get(job['id'])['finished'])
            self.assertIn(job['id'], self.store.recover())
            self.store.event(job['id'], 'resource_reconciled', {'admitted_new_work': False})
            self.assertNotIn(job['id'], self.store.recover())

    def test_studio_session_reserves_workers_without_running_automatic_inference(self):
        value = {key: item for key, item in self.value.items() if key not in ('target_entries', 'max_entries')}
        value['kind'] = 'studio_session'
        job = self.store.create(value)
        self.assertEqual(job['counts_unit'], 'ready_sessions')
        self.assertEqual(job['corpora'], [])
        cleaned = []
        @contextmanager
        def fleet(job, config, event, *, stop_event):
            job['deadline'] = __import__('time').time() + .02
            try: yield self.session
            finally: cleaned.append(True)
        with patch('persona_jobs.fetch') as inference, patch('persona_coactivity_refresh.sync_corpus') as sync:
            result = run_scheduled_job(job, {}, self.store, threading.Event(), fleet_callback=fleet)
        self.assertEqual(result['state'], 'complete')
        self.assertEqual(result['counts']['accepted_unique'], 1)
        self.assertEqual(cleaned, [True]); inference.assert_not_called(); sync.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'Studio sessions'):
            self.store.create({**value, 'target_entries': 2, 'max_entries': 2})

    def test_restart_retains_samples_and_marks_inflight_unknown_without_replay(self):
        job = self.store.create(self.value); self.store.claim_due()
        request_id, _ = self.store.request(job['id'], 'generator', self.session['endpoints'][0], {'unreplayed': True})
        self.store.recover()
        self.assertEqual(self.store.get(job['id'])['state'], 'queued')
        with self.store.connect() as db:
            row = db.execute('SELECT state,error,body FROM requests WHERE id=?', (request_id,)).fetchone()
        self.assertEqual(row[0], 'interrupted'); self.assertIn('will not be replayed', row[1])
        self.assertEqual(json.loads(row[2]), {'unreplayed': True})

    def test_publication_preserves_existing_heldout_bytes_and_recovers_owned_append(self):
        job = self.store.create(self.value)
        with patch('persona_jobs.fetch', self.api), patch('persona_manager.fetch', self.api):
            run_big_five(job, self.session, self.store, threading.Event(), max_rounds=1)
        corpus = self.root / 'data/persona_manager_corpus.jsonl'
        original = corpus.read_bytes()
        from persona_jobs import _publish_manifest
        def fail_after_append(path, value):
            if path.name == 'persona_manager_corpus_manifest.json' and value['status'] == 'complete':
                raise RuntimeError('simulated crash after append')
            return _publish_manifest(path, value)
        with patch('persona_jobs._publish_manifest', fail_after_append), self.assertRaisesRegex(RuntimeError, 'simulated crash'):
            publish_samples(self.store)
        self.assertTrue(corpus.read_bytes().startswith(original))
        with self.store.connect() as db:
            self.assertIsNotNone(db.execute("SELECT value FROM metadata WHERE key='pending_publication'").fetchone())
        self.store.recover()  # Owned publication recovers before any fleet admission.
        result = publish_samples(self.store)
        self.assertEqual(result['appended'], 0)
        raw = corpus.read_bytes(); self.assertTrue(raw.startswith(original))
        rows = [json.loads(line) for line in raw.splitlines()]
        self.assertEqual(len(rows), 73); self.assertEqual(rows[0]['split'], 'heldout')
        self.assertTrue(all(row['split'] == 'discovery' and row['synthetic'] for row in rows[1:]))
        self.assertEqual(publish_samples(self.store)['appended'], 0)
        self.assertEqual(corpus.read_bytes(), raw)
        manifest = json.loads((self.root / 'data/persona_manager_corpus_manifest.json').read_text())
        self.assertEqual(manifest['output_sha256'], hashlib.sha256(raw).hexdigest())

    def test_scheduled_executor_uses_only_callback_and_cooperatively_cancels(self):
        job = self.store.create(self.value); self.store.claim_due()
        cleaned = []
        @contextmanager
        def fleet(job, config, event, *, stop_event):
            event('owned_resources', {'test': True})
            self.store.cancel(job['id'])
            stop_event.set()
            try: yield self.session
            finally: cleaned.append(True)
        with patch('persona_jobs.fetch') as api:
            result = run_scheduled_job(job, {}, self.store, threading.Event(), fleet_callback=fleet)
            api.assert_not_called()
        self.assertEqual(result['state'], 'cancelled'); self.assertEqual(cleaned, [True])

    def test_raw_completion_and_samples_roll_back_together_on_failed_acceptance(self):
        job = self.store.create(self.value)
        endpoint = self.session['endpoints'][0]; payload = {'test': True}; response = {'test': 'complete'}
        identifier, _ = self.store.request(job['id'], 'generator', endpoint, payload)
        items = [{'prompt': f'Your team faces a difficult scheduling choice number {i}. What would you do?',
                  'trait': 'openness', 'facet': 'imagination', 'domain': 'workplace'} for i in range(2)]
        from persona_corpus import record
        calls = []
        def interrupted(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2: raise RuntimeError('simulated storage interruption')
            return record(*args, **kwargs)
        with patch('persona_jobs.record', interrupted), self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.store.accept(job, identifier, items, [], payload, response, endpoint, self.assignments[0])
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM samples').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT state,response FROM requests WHERE id=?', (identifier,)).fetchone(), ('running', None))
        self.store.accept(job, identifier, items, [], payload, response, endpoint, self.assignments[0])
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT state,response FROM requests WHERE id=?', (identifier,)).fetchone(), ('complete', json.dumps(response, separators=(',', ':'), sort_keys=True)))
        self.assertEqual(self.store.feedback(job['id'])['most_used_domains'], [{'domain': 'workplace', 'count': 2}])
        with self.assertRaisesRegex(ValueError, 'unfinished request'):
            self.store.accept(job, identifier, items, [], payload, response, endpoint, self.assignments[0])

    def test_deadline_timeout_retains_partial_counts_and_audited_error(self):
        job = self.store.create(self.value); self.store.claim_due()
        clock = [time.time()]
        @contextmanager
        def fleet(*args, **kwargs):
            clock[0] = job['deadline'] + 1
            raise TimeoutError('window ended during admission')
            yield
        with patch('persona_jobs.time.time', side_effect=lambda: clock[0]):
            result = run_scheduled_job(job, {}, self.store, threading.Event(), fleet_callback=fleet)
        self.assertEqual(result['state'], 'partial')
        self.assertIn('window ended', result['error'])
        self.assertTrue(any(event['event'] == 'execution_error' for event in result['events']))

    def test_terminal_counts_come_from_committed_samples_after_missing_progress_checkpoint(self):
        job = self.store.create(self.value); self.store.claim_due()
        with patch('persona_jobs.fetch', self.api), patch('persona_manager.fetch', self.api):
            run_big_five(job, self.session, self.store, threading.Event(), max_rounds=1)
        with self.store.connect() as db:
            db.execute('UPDATE jobs SET counts=? WHERE id=?', (json.dumps(dict(accepted_unique=0, requested=0, rejected=0, requests=0, errors=0)), job['id']))
        @contextmanager
        def fleet(*args, **kwargs): yield self.session
        with patch('persona_jobs.fetch') as api:
            result = run_scheduled_job(job, {}, self.store, threading.Event(), fleet_callback=fleet)
            api.assert_not_called()
        self.assertEqual(result['state'], 'complete'); self.assertEqual(result['counts']['accepted_unique'], 72)

    def test_job_http_mutations_require_json_same_origin_and_operator_enablement(self):
        from persona_explorer import handler
        request = object.__new__(handler(SimpleNamespace(jobs=self.store)))
        request.path = '/api/jobs'; request.send = Mock()
        raw = json.dumps(self.value).encode()
        def post(**headers):
            request.headers = {'Content-Length': str(len(raw)), 'Host': 'localhost:8899', **headers}
            request.rfile = io.BytesIO(raw); request.send.reset_mock(); request.do_POST()
            return request.send.call_args.args
        self.assertEqual(post(**{'Origin': 'http://evil.example', 'Content-Type': 'application/json'})[1], 403)
        self.assertEqual(post(**{'Content-Type': 'text/plain'})[1], 400)
        self.assertEqual(post(**{'Content-Type': 'application/json'})[1], 400)
        self.assertEqual(self.store.listing()['jobs'], [])
        (self.root / 'persona_jobs.local.json').write_text(json.dumps({'enabled': True}))
        result = post(**{'Origin': 'http://localhost:8899', 'Content-Type': 'application/json; charset=utf-8'})
        self.assertEqual(result[1], 201); self.assertEqual(result[0]['state'], 'queued')
        self.assertEqual(len(self.store.listing()['jobs']), 1)

    def test_neighborhood_sync_precedes_lease_and_publication_follows_cleanup_after_gpu_deadline(self):
        job = self.store.create({**self.value, 'kind': 'neighborhood_hillclimb', 'targets': [{'layer': 1, 'neuron': 3}]})
        self.store.claim_due(); order = []; clock = [time.time()]
        def sync(config, event, stop):
            self.assertFalse(stop.is_set()); order.append('sync')
        @contextmanager
        def fleet(job, config, event, *, stop_event):
            order.append('lease')
            try: yield self.session
            finally:
                self.assertTrue(stop_event.is_set()); order.append('cleanup')
        def neighborhood(job, session, event, stop):
            order.append('measure')
            event('progress', {'accepted_unique': 72, 'requested': 72, 'requests': 72})
            clock[0] = job['deadline'] + 1; stop.set()
            return {'baseline_registry': {'opaque_test_receipt': True}}
        def refresh(current, config, summary, event, publication_stop):
            self.assertEqual(order[-1], 'cleanup'); self.assertGreater(clock[0], current['deadline'])
            self.assertFalse(publication_stop.is_set()); order.append('refresh')
            return {'status': 'complete', 'map_updated': True, 'rows': 8}
        with patch('persona_coactivity_refresh.sync_corpus', sync), patch('persona_coactivity_refresh.refresh_coactivity', refresh), \
                patch('persona_jobs.time.time', side_effect=lambda: clock[0]):
            result = run_scheduled_job(job, {}, self.store, threading.Event(), fleet_callback=fleet, neighborhood_callback=neighborhood)
        self.assertEqual(order, ['sync', 'lease', 'measure', 'cleanup', 'refresh'])
        self.assertEqual(result['state'], 'complete'); self.assertIsNone(result['error'])
        attempt = next(event['data'] for event in result['events'] if event['event'] == 'finished_attempt')
        self.assertTrue(attempt['deadline_reached']); self.assertTrue(attempt['summary']['map_updated'])

    def test_ui_cancellation_skips_or_interrupts_cpu_publication_after_owned_cleanup(self):
        for when in ('before_publication', 'during_publication'):
            with self.subTest(when=when):
                job = self.store.create({**self.value, 'kind': 'neighborhood_hillclimb', 'targets': [{'layer': 1, 'neuron': 3}]})
                self.store.claim_due(); order = []
                @contextmanager
                def fleet(job, config, event, *, stop_event):
                    try: yield self.session
                    finally:
                        order.append('cleanup')
                        if when == 'before_publication': self.store.cancel(job['id'])
                def neighborhood(*args): return {'baseline_registry': {'opaque_test_receipt': True}}
                def refresh(current, config, summary, event, publication_stop):
                    self.assertEqual(order, ['cleanup']); order.append('publication')
                    if when == 'before_publication':
                        self.assertTrue(publication_stop.is_set())
                        return {'status': 'skipped', 'map_updated': False}
                    self.assertFalse(publication_stop.is_set())
                    self.store.cancel(job['id'])
                    self.assertTrue(publication_stop.wait(2.5), 'Watcher did not propagate UI cancellation to the CPU transport')
                    raise InterruptedError('CPU publication cancelled cooperatively')
                with patch('persona_coactivity_refresh.sync_corpus'), patch('persona_coactivity_refresh.refresh_coactivity', refresh):
                    result = run_scheduled_job(job, {}, self.store, threading.Event(), fleet_callback=fleet, neighborhood_callback=neighborhood)
                self.assertEqual(result['state'], 'cancelled')
                self.assertEqual(order, ['cleanup', 'publication'])
                if when == 'during_publication': self.assertIn('cancelled cooperatively', result['error'])

    def test_already_stopped_job_never_enters_sync_or_fleet_admission(self):
        for reason in ('shutdown', 'cancel', 'deadline', 'cancel_during_sync'):
            with self.subTest(reason=reason):
                job = self.store.create({**self.value, 'kind': 'neighborhood_hillclimb', 'targets': [{'layer': 1, 'neuron': 3}]})
                self.store.claim_due(); shutdown = threading.Event()
                if reason == 'shutdown': shutdown.set()
                if reason == 'cancel': self.store.cancel(job['id'])
                now = job['deadline'] + 1 if reason == 'deadline' else time.time()
                def sync(*args):
                    if reason == 'cancel_during_sync': self.store.cancel(job['id'])
                with patch('persona_coactivity_refresh.sync_corpus', side_effect=sync) as synchronized, \
                        patch('persona_jobs.time.time', return_value=now):
                    fleet, neighborhood = Mock(), Mock()
                    result = run_scheduled_job(job, {}, self.store, shutdown, fleet_callback=fleet, neighborhood_callback=neighborhood)
                    fleet.assert_not_called(); neighborhood.assert_not_called()
                    if reason != 'cancel_during_sync': synchronized.assert_not_called()
                expected = 'queued' if reason == 'shutdown' else 'partial' if reason == 'deadline' else 'cancelled'
                self.assertEqual(result['state'], expected)
                # A shutdown checkpoint remains queued; finish this fixture so
                # the next subcase independently exercises admission.
                if result['state'] == 'queued': self.store.mark(job['id'], 'cancelled')


if __name__ == '__main__': unittest.main()
