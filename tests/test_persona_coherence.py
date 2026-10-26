import persona
from copy import deepcopy
import json
import sqlite3
from types import SimpleNamespace
import pytest

from persona_coherence import CoherenceLab, scale_recipe, summarize


def test_backoff_preserves_signed_recipe_and_phase():
    recipe = {'name': 'A', 'phase': 'prefill', 'pushes': [
        {'kind': 'neuron', 'layer': 40, 'neuron': 1661, 'dose': 1.},
        {'kind': 'neuron', 'layer': 13, 'neuron': 20, 'dose': -.4}]}
    old = deepcopy(recipe)
    scaled = scale_recipe(recipe, .75)
    assert [p['dose'] for p in scaled['pushes']] == [.75, -.30000000000000004]
    assert scaled['phase'] == 'prefill' and recipe == old
    assert scale_recipe(recipe, 0)['pushes'] == []
    for bad in (True, float('nan'), 1.01, -1):
        with pytest.raises(ValueError): scale_recipe(recipe, bad)


def test_baseline_failure_not_mislabeled_as_new_regression():
    results = [{'case_id': 'x', 'seed': 1, 'evaluation': {'state': 'fail', 'issues': [{'kind': 'missing'}]}},
               {'case_id': 'y', 'seed': 1, 'evaluation': {'state': 'unknown', 'issues': []}}]
    summary = summarize(results, {('x', 1): {'state': 'fail'}, ('y', 1): {'state': 'pass'}})
    assert summary['baseline_regressions'] == 1
    assert summary['regression_units'][0]['case_id'] == 'y'
    assert summary['fail_count'] == 1 and summary['unknown_count'] == 1
    assert summarize([])['state'] == 'unknown'


def make_lab(tmp_path):
    app = SimpleNamespace(root=tmp_path, connect=lambda: sqlite3.connect(tmp_path/'db.sqlite'))
    return CoherenceLab(app)


def insert(lab):
    run = {'id': 'test', 'created': 1., 'state': 'running', 'cancel_requested': False,
           'recipe': {'name': 'x', 'phase': 'prefill', 'pushes': [{'kind': 'neuron', 'layer': 1, 'neuron': 1, 'dose': 1.}]},
           'model_identity': 'm', 'recipe_sha256': 'r', 'suite_sha256': 's', 'evaluator_sha256': 'e',
           'scales': [1., .9, .75, .5, .25], 'trials': [], 'seeds': [1,2]}
    with lab.app.connect() as db:
        db.execute('INSERT INTO coherence_runs VALUES(?,?,?,?,?)', ('test', 1, 1, 'running', json.dumps(run)))
    return run


def test_cancellation_survives_stale_worker_save(tmp_path):
    lab = make_lab(tmp_path); run = insert(lab)
    lab.cancel({'id': 'test'})
    run['trials'].append({'state': 'running'})
    lab.save(run)
    assert lab.get('test')['cancel_requested'] is True
    assert lab.get('test')['state'] == 'stopping'
    lab.executor.shutdown()


def test_failed_holdout_does_not_feed_retuning(tmp_path):
    lab = make_lab(tmp_path); run = insert(lab); calls = []
    def trial(run, recipe, scale, split, baseline=None, purpose='dose'):
        calls.append((scale, split, purpose))
        # Nonmonotonic response: .9 passes, .95 fails, .925 passes.
        passed = purpose == 'baseline' or (split == 'discovery' and scale in (.9, .925))
        return {'scale': scale, 'recipe': recipe, 'summary': {'state': 'pass' if passed else 'fail', 'results': []}}
    lab.trial = trial
    lab.execute(run, [])
    assert run['frozen_selection']['scale'] == pytest.approx(.925)
    assert run['recommendation']['status'] == 'heldout_failed'
    assert run['recommendation']['recipe'] is None
    assert calls[-1] == (.925, 'heldout', 'confirmation')
    assert len([x for x in calls if x[1] == 'heldout']) == 2
    with lab.app.connect() as db: assert db.execute('SELECT COUNT(*) FROM coherence_defaults').fetchone()[0] == 0
    lab.executor.shutdown()


def test_pass_is_scoped_and_never_silently_applied(tmp_path):
    lab = make_lab(tmp_path); run = insert(lab)
    lab.trial = lambda run, recipe, scale, split, baseline=None, purpose='dose': {
        'scale': scale, 'recipe': recipe, 'summary': {'state': 'pass', 'results': []}}
    lab.execute(run, [])
    assert run['recommendation']['scale'] == 1
    assert run['recommendation']['scope']['automatic_application'] is False
    with lab.app.connect() as db: assert db.execute('SELECT COUNT(*) FROM coherence_defaults').fetchone()[0] == 1
    lab.executor.shutdown()


def test_representation_uncertainty_does_not_trigger_backoff(tmp_path):
    lab = make_lab(tmp_path); run = insert(lab); scales = []
    def trial(run, recipe, scale, split, baseline=None, purpose='dose'):
        scales.append(scale)
        return {'id': len(scales), 'scale': scale, 'recipe': recipe,
                'summary': {'state': 'fail', 'guard_state': 'unknown', 'results': []}}
    lab.trial = trial; lab.execute(run, [])
    assert scales == [0, 1]
    assert run['recommendation']['status'] == 'review_needed'
    lab.executor.shutdown()


def test_correct_sentence_in_boolean_slot_is_not_a_factual_contradiction():
    from persona_coherence_eval import load_suite, evaluate
    from pathlib import Path
    suite = load_suite(Path(__file__).resolve().parents[1]/'data/persona_coherence_suite.json')
    case = next(c for c in suite['cases'] if 'heater_tested' in c['groundtruth']['answers'])
    truth = case['groundtruth']
    answer = {'prose': 'The heater was not tested.', 'facts': [{k:f[k] for k in ('id','subject','predicate','value')} for f in truth['facts']],
              'answers': {k:v['value'] for k,v in truth['answers'].items()}, 'unknowns': truth['unknowns']}
    answer['answers']['heater_tested'] = 'The heater was not tested, so whether it works is not established.'
    evaluated = evaluate(case, {'answer':json.dumps(answer),'complete':True,'finish_reason':'stop'})
    assert [x['kind'] for x in evaluated['issues']] == ['contract_representation_unknown']
    summary = summarize([{'case_id':case['id'],'seed':1,'evaluation':evaluated}])
    assert summary['guard_state'] == 'unknown' and not summary['hard_issues']
    answer['answers']['heater_tested'] = True
    evaluated = evaluate(case, {'answer':json.dumps(answer),'complete':True,'finish_reason':'stop'})
    assert 'negation_contradiction' in [x['kind'] for x in evaluated['issues']]


def test_unspaced_cjk_loop_is_detected_even_with_normal_stop():
    from persona_coherence_eval import evaluate_surface
    result = evaluate_surface({'answer': '梦想快乐'*80, 'complete': True, 'finish_reason': 'stop'})
    assert result['state'] == 'fail'
    assert 'repetition_collapse' in [x['kind'] for x in result['issues']]
