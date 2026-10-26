"""Matched degradation tests and finite, evidence-preserving dose backoff.

This calibrates a checkable task contract, not personality desirability. A passing
fact ledger does not certify its accompanying prose or arbitrary future chats.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
import time
import uuid

from persona_coherence_eval import load_suite, evaluate
from persona_manager import keys, uint
from persona_vectors import canonical_digest

SCALES = (1., .9, .75, .5, .25)
ACTIVE = ('queued', 'running', 'stopping')
SCOPE = {
    'measures': 'Source-grounded fact/requirement contract and observable response collapse',
    'does_not_measure': ['Desired personality strength', 'General prose factuality',
                         'Causal wiring', 'Universal neuron meaning'],
    'automatic_application': False,
    'reason': 'Selection is specific to this model, recipe, suite, phases and seeds. Free-prose claims remain unverified.',
}


def scale_recipe(recipe, scale):
    if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError('Backoff scale must be finite and between zero and one')
    result = deepcopy(recipe)
    result['name'] = (recipe['name'][:90] + f' · scale {scale:.4g}')
    result['pushes'] = [{**push, 'dose': push['dose'] * scale} for push in recipe['pushes'] if push['dose'] * scale]
    return result


def summarize(results, baseline=None):
    baseline = baseline or {}
    counts = {state: sum(item['evaluation']['state'] == state for item in results) for state in ('pass', 'fail', 'unknown')}
    issues, regressions = [], []
    for item in results:
        key = (item['case_id'], item['seed'])
        for issue in item['evaluation'].get('issues', []):
            issues.append({'case_id': item['case_id'], 'seed': item['seed'], **issue})
        if key in baseline and baseline[key]['state'] == 'pass' and item['evaluation']['state'] != 'pass':
            regressions.append({'case_id': item['case_id'], 'seed': item['seed'], 'state': item['evaluation']['state']})
    # A canonical representation mismatch may contain a perfectly correct
    # sentence. Keep it visible, but never call it evidence for dose reduction.
    hard = [issue for issue in issues if issue.get('kind') not in
            ('contract_representation_unknown', 'ledger_format_failure', 'duplicate_fact', 'duplicate_unknown', 'operational')]
    guard = 'fail' if hard else 'unknown' if issues or counts['unknown'] or not results else 'pass'
    return {'state': 'fail' if counts['fail'] else 'unknown' if counts['unknown'] or not results else 'pass',
            'guard_state': guard, 'hard_issues': hard,
            **{name + '_count': count for name, count in counts.items()}, 'issues': issues,
            'baseline_regressions': len(regressions), 'regression_units': regressions, 'results': results,
            'scope': deepcopy(SCOPE)}


class CoherenceLab:
    def __init__(self, app):
        self.app = app
        self.path = app.root / 'data/persona_coherence_suite.json'
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='coherence')
        with app.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS coherence_runs(id TEXT PRIMARY KEY,created REAL,updated REAL,state TEXT,body TEXT);
                CREATE TABLE IF NOT EXISTS coherence_defaults(binding TEXT PRIMARY KEY,created REAL,body TEXT);
                CREATE TABLE IF NOT EXISTS coherence_interactions(batch_id TEXT PRIMARY KEY,created REAL,body TEXT);
            ''')
            rows = db.execute("SELECT id,body FROM coherence_runs WHERE state IN ('queued','running','stopping')").fetchall()
            for identity, encoded in rows:
                body = json.loads(encoded)
                body.update(state='interrupted', error='Explorer restarted; possible remote outcomes retained; no automatic retry')
                db.execute('UPDATE coherence_runs SET state=?,updated=?,body=? WHERE id=?',
                           ('interrupted', time.time(), json.dumps(body), identity))

    def suite(self):
        suite = load_suite(self.path)
        return suite, hashlib.sha256(self.path.read_bytes()).hexdigest()

    def catalog(self):
        suite, digest = self.suite()
        with self.app.connect() as db:
            rows = db.execute('SELECT body FROM coherence_runs ORDER BY created DESC LIMIT 30').fetchall()
        return {'suite': {**suite, 'sha256': digest},
                'runs': [{k: row.get(k) for k in ('id', 'created', 'state', 'input', 'recommendation', 'error')}
                         for row in map(lambda item: json.loads(item[0]), rows)],
                'defaults': {'seed': 7035, 'max_tokens': 2048, 'scales': list(SCALES), 'seeds_per_case': 2,
                             'refinement_steps': 2}, 'scope': deepcopy(SCOPE)}

    def get(self, identity):
        with self.app.connect() as db:
            row = db.execute('SELECT body FROM coherence_runs WHERE id=?', (identity,)).fetchone()
        if not row:
            raise ValueError('Unknown coherence run')
        return json.loads(row[0])

    def save(self, run):
        run['updated'] = time.time()
        with self.app.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT body FROM coherence_runs WHERE id=?', (run['id'],)).fetchone()
            if previous and json.loads(previous[0]).get('cancel_requested'):
                run['cancel_requested'] = True
                if run['state'] in ACTIVE:
                    run['state'] = 'stopping'
            db.execute('UPDATE coherence_runs SET state=?,updated=?,body=? WHERE id=?',
                       (run['state'], run['updated'], json.dumps(run, allow_nan=False), run['id']))

    def submit(self, value):
        keys(value, {'recipe_id', 'seed', 'max_tokens'}, {'reuse_run'})
        self.app.overnight.require_interactive()
        seed = uint(value['seed'], 0, 2**32 - 1, 'Seed')
        budget = uint(value['max_tokens'], 2048, 4096, 'Complete-answer budget')
        saved = next((r for r in self.app.saved_recipes()['recipes'] if r['id'] == value['recipe_id']), None)
        if not saved or saved['model_identity'] != self.app.model.identity:
            raise ValueError('Choose a saved recipe for the current model')
        recipe = {key: deepcopy(saved[key]) for key in ('name', 'pushes', 'phase')}
        if not recipe['pushes'] or not any(p['dose'] for p in recipe['pushes']):
            raise ValueError('Choose a nonzero intervention; baseline is included automatically')
        self.app.validate_recipe(recipe)
        suite, digest = self.suite()
        if value.get('reuse_run'):
            prior = self.get(value['reuse_run'])
            if (prior['state'] in ACTIVE or prior['model_identity'] != self.app.model.identity
                    or prior['suite_sha256'] != digest or prior['max_tokens'] != budget
                    or prior['seeds'] != [seed, (seed+104729) % 2**32]):
                raise ValueError('Reuse requires a terminal run with the exact model, suite, seeds and budget')
        endpoints = self.app.sculptor.comparison_endpoints()
        if self.app.closed or not endpoints:
            raise ValueError('No active comparison endpoints')
        run = {'id': uuid.uuid4().hex, 'created': time.time(), 'state': 'queued', 'input': deepcopy(value),
               'model_identity': self.app.model.identity, 'recipe': recipe, 'recipe_sha256': canonical_digest(recipe),
               'suite': suite, 'suite_sha256': digest,
               'evaluator_sha256': hashlib.sha256((self.app.root/'persona/persona_coherence_eval.py').read_bytes()).hexdigest(),
               'controller_sha256': hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
               'seeds': [seed, (seed + 104729) % 2**32], 'max_tokens': budget,
               'scales': list(SCALES), 'trials': [], 'recommendation': None, 'scope': deepcopy(SCOPE),
               'deadline': time.time() + 10800, 'cancel_requested': False,
               'concurrency': {'configured_endpoints': len(endpoints), 'max_parallel_cells': min(6, len(endpoints)),
                               'cells_per_batch': 6, 'within_card': 'runtime-advertised; diagnostic capacity currently one'}}
        with self.app.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM coherence_runs WHERE state IN ('queued','running','stopping') LIMIT 1").fetchone():
                raise ValueError('A coherence calibration is already active')
            db.execute('INSERT INTO coherence_runs VALUES(?,?,?,?,?)',
                       (run['id'], run['created'], run['created'], run['state'], json.dumps(run, allow_nan=False)))
        self.executor.submit(self.execute, run, endpoints)
        return {'id': run['id'], 'state': run['state']}

    def cancel(self, value):
        keys(value, {'id'})
        with self.app.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT body FROM coherence_runs WHERE id=?', (value['id'],)).fetchone()
            if not row:
                raise ValueError('Unknown coherence run')
            run = json.loads(row[0])
            if run['state'] in ACTIVE:
                run.update(cancel_requested=True, state='stopping')
                db.execute('UPDATE coherence_runs SET state=?,body=? WHERE id=?', ('stopping', json.dumps(run), run['id']))
        return {'id': run['id'], 'state': run['state'], 'note': 'In-flight cells drain; no further trials are dispatched'}

    def check_stop(self, run):
        if self.app.closed or time.time() >= run['deadline']:
            raise InterruptedError('Calibration stopped at shutdown or finite deadline')
        if self.get(run['id']).get('cancel_requested'):
            raise InterruptedError('Cancelled; completed evidence retained')

    def trial(self, run, recipe, scale, split, baseline=None, purpose='dose'):
        self.check_stop(run)
        cases = [case for case in run['suite']['cases'] if case['split'] == split]
        if not cases:
            raise ValueError('Suite requires discovery and heldout cases')
        entry = {'id': len(run['trials']), 'scale': scale, 'split': split, 'purpose': purpose,
                 'state': 'running', 'recipe': deepcopy(recipe), 'batch_ids': [], 'cells': []}
        run['trials'].append(entry); self.save(run)
        if run['input'].get('reuse_run'):
            prior = self.get(run['input']['reuse_run'])
            source = next((t for t in prior['trials'] if t.get('state') == 'complete'
                           and t['recipe'] == recipe and t['split'] == split), None)
            if source:
                _, _, provenance = self.app.validate_recipe(recipe)
                old_results = source['summary']['results']
                expected = {(c['id'], seed) for c in cases for seed in run['seeds']}
                reusable = (len(old_results) == len(expected)
                    and {(x['case_id'], x['seed']) for x in old_results} == expected
                    and all(x['cell_state'] in ('complete','incomplete') and x.get('output')
                            and x['output'].get('provenance', {}).get('program_sha256') == provenance['program_sha256']
                            for x in old_results))
                if reusable:
                    by_id = {c['id']: c for c in cases}
                    results = [{**deepcopy(x), 'evaluation': evaluate(by_id[x['case_id']], x['output'])} for x in old_results]
                    entry.update(state='complete', batch_ids=source['batch_ids'], cells=source['cells'],
                                 reused_from={'run_id': prior['id'], 'trial_id': source['id'], 'evaluator_sha256': prior['evaluator_sha256']},
                                 summary=summarize(results, baseline))
                    self.save(run); return entry
        for seed in run['seeds']:
            for start in range(0, len(cases), 6):
                self.check_stop(run)
                chunk = cases[start:start+6]
                batch_id = self.app.submit_batch({'model_identity': run['model_identity'], 'seed': seed,
                    'max_tokens': run['max_tokens'], 'thinking': False, 'smoke': False,
                    'cells': [{'label': f"{case['id']} · {scale:g}", 'prompt': case['prompt'], 'recipe': recipe} for case in chunk]},
                    endpoints=self.app.sculptor.comparison_endpoints())
                entry['batch_ids'].append(batch_id)
                entry['cells'].extend({'batch_id': batch_id, 'index': index, 'case_id': case['id'], 'seed': seed}
                                      for index, case in enumerate(chunk))
                self.save(run)
        # Wait for all submitted cells even on cancellation, preserving ownership.
        while True:
            batches = {identity: self.app.batch(identity) for identity in entry['batch_ids']}
            if all(all(cell['state'] not in ('queued', 'running') for cell in batch['cells']) for batch in batches.values()):
                break
            if time.time() > run['deadline'] + self.app.config.get('request_timeout_seconds', 600):
                raise RuntimeError('Remote outcomes remain active beyond deadline; no retry')
            time.sleep(.5)
        cases_by_id = {case['id']: case for case in cases}
        results = []
        for item in entry['cells']:
            cell = batches[item['batch_id']]['cells'][item['index']]
            if cell['state'] in ('failed', 'interrupted') or not cell.get('output'):
                assessment = {'state': 'unknown', 'issues': [{'kind': 'operational', 'detail': cell.get('error', cell['state'])}]}
            else:
                assessment = evaluate(cases_by_id[item['case_id']], cell['output'])
            results.append({**item, 'evaluation': assessment, 'output': cell.get('output'), 'cell_state': cell['state']})
        entry['summary'] = summarize(results, baseline)
        entry['state'] = 'complete'; self.save(run)
        if any(item['cell_state'] in ('failed', 'interrupted') for item in results):
            raise RuntimeError('Operational cell failure; not treated as cognitive degradation and not retried')
        self.check_stop(run)
        return entry

    def execute(self, run, endpoints):
        try:
            run['state'] = 'running'; self.save(run)
            baseline_recipe = {'name': 'Unsteered baseline', 'pushes': [], 'phase': 'all'}
            baseline = self.trial(run, baseline_recipe, 0., 'discovery', purpose='baseline')
            references = {(x['case_id'], x['seed']): x['evaluation'] for x in baseline['summary']['results']}
            chosen, upper = None, None
            unresolved = None
            for scale in run['scales']:
                item = self.trial(run, scale_recipe(run['recipe'], scale), scale, 'discovery', references)
                guard = item['summary'].get('guard_state', item['summary']['state'])
                if guard == 'unknown':
                    unresolved = item; break
                if guard == 'pass':
                    chosen = item; break
                upper = scale
            # Every proposed midpoint is actually tested. No monotonicity claim.
            if chosen and upper is not None:
                for _ in range(2):
                    midpoint = (upper + chosen['scale']) / 2
                    item = self.trial(run, scale_recipe(run['recipe'], midpoint), midpoint, 'discovery', references, 'refinement')
                    guard = item['summary'].get('guard_state', item['summary']['state'])
                    if guard == 'unknown':
                        unresolved = item; break
                    if guard == 'pass':
                        chosen = item
                    else:
                        upper = midpoint
            if unresolved is not None:
                run['recommendation'] = {'status': 'review_needed', 'scale': None, 'recipe': None,
                    'reason': 'Only ambiguous representation/format evidence remains at the tested dose. No automatic reduction is justified by that evidence.',
                    'review_trial': unresolved['id'], 'scope': deepcopy(SCOPE)}
            elif chosen is None:
                run['recommendation'] = {'status': 'no_tested_nonzero_pass', 'scale': None, 'recipe': None,
                    'reason': 'No tested nonzero dose passed all checkable discovery contracts. Baseline and failures remain visible.', 'scope': deepcopy(SCOPE)}
            else:
                run['frozen_selection'] = {'scale': chosen['scale'], 'recipe': chosen['recipe'], 'selected_before_heldout': True}
                self.save(run)
                base_holdout = self.trial(run, baseline_recipe, 0., 'heldout', purpose='baseline')
                held_refs = {(x['case_id'], x['seed']): x['evaluation'] for x in base_holdout['summary']['results']}
                heldout = self.trial(run, chosen['recipe'], chosen['scale'], 'heldout', held_refs, 'confirmation')
                passed = heldout['summary']['state'] == 'pass'
                run['recommendation'] = {'status': 'bounded_probe_pass' if passed else 'heldout_failed',
                    'scale': chosen['scale'] if passed else None, 'recipe': chosen['recipe'] if passed else None,
                    'reason': 'Largest actually tested passing scale within the finite search, confirmed on untouched scenarios.' if passed
                              else 'Frozen candidate failed confirmation; heldout results are not reused to tune a replacement.',
                    'scope': deepcopy(SCOPE), 'phase': run['recipe']['phase'], 'tested_seeds': run['seeds']}
                if passed:
                    binding = canonical_digest({'model_identity': run['model_identity'], 'recipe_sha256': run['recipe_sha256'],
                                                'suite_sha256': run['suite_sha256'], 'evaluator_sha256': run['evaluator_sha256']})
                    with self.app.connect() as db:
                        db.execute('INSERT OR REPLACE INTO coherence_defaults VALUES(?,?,?)',
                                   (binding, time.time(), json.dumps({'binding': binding, 'run_id': run['id'], **run['recommendation']})))
            run['state'] = 'complete'
        except InterruptedError as error:
            run.update(state='cancelled', cancel_requested=True, error=str(error))
        except Exception as error:
            run.update(state='failed', error=f'{type(error).__name__}: {error}')
        finally:
            self.save(run)

    def interaction_plan(self, value):
        """Propose a factorial intervention, never infer compensation from r alone."""
        keys(value, {'recipe_id', 'layer', 'neuron', 'dose'})
        recipe = next((r for r in self.app.saved_recipes()['recipes'] if r['id'] == value['recipe_id']), None)
        if not recipe or recipe['model_identity'] != self.app.model.identity:
            raise ValueError('Choose a current-model saved recipe')
        primary = {key: deepcopy(recipe[key]) for key in ('name', 'pushes', 'phase')}
        if len(primary['pushes']) != 1 or primary['pushes'][0]['kind'] != 'neuron':
            raise ValueError('Interaction controls require one primary neuron to isolate effects')
        target = primary['pushes'][0]
        neighbors = self.app.neighbors.query(target['layer'], target['neuron'], limit=50)
        matches = [row for group in neighbors['groups'].values() for row in group
                   if row['layer'] == value['layer'] and row['neuron'] == value['neuron']]
        if not neighbors.get('available') or not matches:
            raise ValueError('Compensator must be an actually observed co-activity neighbor, not an assumed wire')
        dose = value['dose']
        if type(dose) not in (int, float) or not math.isfinite(dose) or not 0 < abs(dose) <= .5:
            raise ValueError('Test a finite nonzero compensator dose of magnitude at most .5')
        b = {'kind': 'neuron', 'layer': value['layer'], 'neuron': value['neuron'], 'dose': dose}
        variants = [{'name': 'Baseline', 'pushes': [], 'phase': primary['phase']}, primary,
                    {'name': 'Compensator alone', 'pushes': [b], 'phase': primary['phase']},
                    {'name': 'Primary plus compensator', 'pushes': [target, b], 'phase': primary['phase']},
                    {'name': 'Primary minus compensator', 'pushes': [target, {**b, 'dose': -dose}], 'phase': primary['phase']},
                    {'name': 'Primary sign reversal', 'pushes': [{**target, 'dose': -target['dose']}], 'phase': primary['phase']}]
        for variant in variants:
            self.app.validate_recipe(variant)
        return {'model_identity': self.app.model.identity, 'variants': variants, 'neighbor': matches[0],
                'source': neighbors['source'], 'claim': 'Factorial test proposal, not a validated correction or physical wiring.',
                'analysis': 'Compare AB−A−B+baseline on identical prompt/seed metrics. Test both signs; do not assume layer-wide reversals.'}

    def interaction_test(self, value):
        keys(value, {'recipe_id', 'layer', 'neuron', 'dose', 'case_id', 'seed', 'max_tokens'})
        self.app.overnight.require_interactive()
        seed = uint(value['seed'], 0, 2**32-1, 'Seed')
        budget = uint(value['max_tokens'], 2048, 4096, 'Complete-answer budget')
        plan = self.interaction_plan({k: value[k] for k in ('recipe_id', 'layer', 'neuron', 'dose')})
        suite, digest = self.suite()
        case = next((c for c in suite['cases'] if c['id'] == value['case_id'] and c['split'] == 'discovery'), None)
        if not case:
            raise ValueError('Exploratory interaction tests use discovery cases only')
        identity = self.app.submit_batch({'model_identity': self.app.model.identity, 'seed': seed, 'max_tokens': budget,
            'thinking': False, 'smoke': False,
            'cells': [{'label': recipe['name'], 'prompt': case['prompt'], 'recipe': recipe} for recipe in plan['variants']]},
            endpoints=self.app.sculptor.comparison_endpoints())
        with self.app.connect() as db:
            db.execute('INSERT INTO coherence_interactions VALUES(?,?,?)',
                       (identity, time.time(), json.dumps({'input': value, 'plan': plan, 'case': case, 'suite_sha256': digest})))
        return {'id': identity, 'state': 'queued'}

    def interaction(self, identity):
        with self.app.connect() as db:
            record = db.execute('SELECT body FROM coherence_interactions WHERE batch_id=?', (identity,)).fetchone()
        if not record:
            raise ValueError('Unknown interaction test')
        record = json.loads(record); batch = self.app.batch(identity)
        rows = [{**cell, 'evaluation': evaluate(record['case'], cell['output']) if cell.get('output') else None}
                for cell in batch['cells']]
        effect = None
        if all(row['evaluation'] and row['evaluation']['state'] != 'unknown' for row in rows[:4]):
            errors = [len(row['evaluation']['issues']) for row in rows[:4]]
            effect = {'metric': 'checkable issue count; lower is better',
                      'interaction': errors[3]-errors[1]-errors[2]+errors[0],
                      'combined_minus_primary': errors[3]-errors[1],
                      'claim': 'One prompt/seed exploratory contrast; not sufficient for a corrective default.'}
        return {'id': identity, 'state': batch['state'], **record, 'cells': rows, 'interaction_effect': effect}
