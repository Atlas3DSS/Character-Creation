"""Authored scorer calibration, not model-generated behavioral validation."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from persona.persona_coherence_eval import evaluate, evaluate_surface, load_suite, canonical_digest

PATH = Path(__file__).resolve().parents[1] / 'data/persona_coherence_suite.json'
SUITE = load_suite(PATH)


def ledger(case, prose='Well, that went brilliantly. Here are the stated facts.'):
    truth = deepcopy(case['groundtruth'])
    return dict(prose=prose, facts=[{k: f[k] for k in ('id', 'subject', 'predicate', 'value')} for f in truth['facts']],
                answers={k: v['value'] for k, v in truth['answers'].items()}, unknowns=list(truth['unknowns']))


def output(value, **kwargs):
    return dict(answer=json.dumps(value) if isinstance(value, dict) else value, complete=True, finish_reason='stop', **kwargs)


def kinds(result):
    return {i['kind'] for i in result['issues']}


def test_suite_binding_and_split(tmp_path):
    assert len(SUITE['cases']) == 12
    assert [c['split'] for c in SUITE['cases']].count('heldout') == 6
    assert len({c['source_id'] for c in SUITE['cases']}) == 12
    for field in ('source', 'prompt'):
        changed = deepcopy(SUITE)
        changed['cases'][0][field] += ' Changed.'
        file = tmp_path / 'suite.json'
        file.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match='binding'):
            load_suite(file)
        changed['suite_sha256'] = canonical_digest({k: v for k, v in changed.items() if k != 'suite_sha256'})
        file.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match='identity'):
            load_suite(file)


@pytest.mark.parametrize('case', SUITE['cases'], ids=lambda c: c['title'])
def test_reference_and_tone_controls(case):
    for prose in ('A concise paraphrase of the source follows.', 'Wonderful. Another triumph of bureaucratic incompetence.',
                  'I am terrified; this uncertainty is hard to bear.', 'The queen imagined a malicious shadow stalking the corridor.'):
        value = ledger(case, prose)
        value['facts'].reverse()
        result = evaluate(case, output(value))
        assert result['state'] == 'pass'
        assert result['prose_state'] == 'unknown'
        assert result['metrics']['prose_claims_checked'] == 0
        assert 'free prose remains unverified' in result['pass_meaning']


def test_numeric_negation_role_and_timeline():
    expectations = [(0, 1, 21, 'number_error'), (1, 0, 'mira', 'role_or_attribution_error'),
                    (3, 0, True, 'negation_contradiction'), (4, 0, '10:00', 'timeline_error')]
    for index, fact, wrong, kind in expectations:
        case = SUITE['cases'][index]
        value = ledger(case); value['facts'][fact]['value'] = wrong
        result = evaluate(case, output(value))
        assert result['state'] == 'fail'
        assert kind in kinds(result)
        assert any(i['evidence'].get('source_spans') for i in result['issues'])
    case = SUITE['cases'][0]
    value = ledger(case); value['facts'][1]['value'] = 20.0
    assert evaluate(case, output(value))['state'] == 'pass'
    case = SUITE['cases'][3]
    value = ledger(case); value['facts'][1]['value'] = 1
    assert evaluate(case, output(value))['state'] == 'fail'
    case = SUITE['cases'][4]
    value = ledger(case); value['answers']['event_order'].reverse()
    assert 'timeline_error' in kinds(evaluate(case, output(value)))


@pytest.mark.parametrize('mutation,kind', [
    ('fact', 'omitted_fact'), ('answer', 'omitted_answer'), ('unknown', 'omitted_uncertainty'),
    ('extra_fact', 'unsupported_ledger_claim'), ('extra_answer', 'unsupported_answer'),
    ('extra_unknown', 'unsupported_uncertainty'), ('duplicate', 'duplicate_fact'),
    ('conflict', 'literal_self_contradiction'), ('empty_prose', 'missing_prose')])
def test_closed_ledger_contract(mutation, kind):
    case = SUITE['cases'][0]; value = ledger(case)
    if mutation == 'fact': value['facts'].pop()
    elif mutation == 'answer': value['answers'].pop(next(iter(value['answers'])))
    elif mutation == 'unknown': value['unknowns'].pop()
    elif mutation == 'extra_fact': value['facts'].append(dict(id='invented',subject='driver',predicate='name',value='sam'))
    elif mutation == 'extra_answer': value['answers']['weather'] = 'sunny'
    elif mutation == 'extra_unknown': value['unknowns'].append('arrival')
    elif mutation in ('duplicate', 'conflict'):
        value['facts'].append(deepcopy(value['facts'][0]))
        if mutation == 'conflict': value['facts'][-1]['value'] = '22:00'
    elif mutation == 'empty_prose': value['prose'] = ''
    result = evaluate(case, output(value))
    assert result['state'] == 'fail'
    assert kind in kinds(result)


@pytest.mark.parametrize('answer', ['plain prose', '[]', '{"x":1,"x":2}', '{"x":NaN}', None,
                                     'intro\n```json\n{}\n```', '{"prose":"hello"}'])
def test_contract_failure_not_hallucination(answer):
    result = evaluate(SUITE['cases'][0], output(answer))
    assert result['state'] == 'fail'
    assert 'ledger_format_failure' in kinds(result)
    assert 'unsupported_ledger_claim' not in kinds(result)


def test_whole_fence_and_canonical_whitespace():
    case = SUITE['cases'][1]; value = ledger(case)
    value['facts'][0]['value'] = '  JON  '
    assert evaluate(case, output('```json\n' + json.dumps(value) + '\n```'))['state'] == 'pass'


def test_completion_and_repetition():
    for answer, kind in [('', 'empty_response'), ('! ' * 24, 'repetition_collapse'),
                         ('the door is closed. ' * 20, 'repetition_collapse')]:
        result = evaluate_surface(output(answer))
        assert result['state'] == 'fail' and kind in kinds(result)
    for text in ('No, no, no. That is a terrible plan.', 'The door was open. The door was closed.', 'Very '*10):
        result = evaluate_surface(output(text))
        assert result['state'] == 'pass' and result['semantic_state'] == 'unknown'
    for fields, kind in [({'complete':False},'incomplete'), ({'finish_reason':'length'},'truncated')]:
        value = output('A readable answer.'); value.update(fields)
        assert kind in kinds(evaluate_surface(value))
    assert evaluate_surface({'answer':'A readable answer.'})['state'] == 'unknown'
    assert evaluate_surface(None)['state'] == 'unknown'


def test_counterfactual_not_paraphrase():
    case = deepcopy(SUITE['cases'][0]); original = ledger(case)
    case['source'] = case['source'].replace('20 minutes', '25 minutes')
    case['groundtruth']['facts'][1]['value'] = 25
    case['groundtruth']['facts'][1]['evidence'] = ['25 minutes late']
    case['groundtruth']['answers']['arrival']['value'] = '09:35'
    case['groundtruth']['answers']['spare_minutes']['value'] = 10
    assert evaluate(case, output(original))['state'] == 'fail'
    assert evaluate(case, output(ledger(case, 'An entirely different tone.')))['state'] == 'pass'


def test_fiction_policy_and_prose_limit():
    for case in SUITE['cases']:
        result = evaluate(case, output(ledger(case, 'A dragon appears from an unmentioned purple cupboard.')))
        assert result['prose_state'] == 'unknown'
        assert result['state'] == 'pass'  # Deliberate bounded check, never prose certification.
        assert result['task_policy']['kind'] in ('fiction_continuation','factual_response')
        assert result['limitations']
    case = deepcopy(SUITE['cases'][0]); case['groundtruth']['require_ledger'] = False
    assert evaluate(case, output('Unstructured prose.'))['state'] == 'unknown'


def test_authored_mutation_calibration():
    # This matrix quantifies deterministic fixture sensitivity only. It is not
    # empirical accuracy on natural prose or generated model responses.
    tp = fp = fn = tn = 0
    for case in SUITE['cases']:
        reference = ledger(case)
        variants = [(False, reference), (False, ledger(case, 'How delightfully inconvenient.'))]
        for change in ('omission', 'role', 'unsupported', 'answer', 'unknown', 'conflict'):
            bad = deepcopy(reference)
            if change == 'omission': bad['facts'].pop()
            elif change == 'role': bad['facts'][0]['subject'] = 'invented_actor'
            elif change == 'unsupported': bad['facts'].append(dict(id='invented',subject='x',predicate='y',value=42))
            elif change == 'answer': bad['answers'][next(iter(bad['answers']))] = '__wrong__'
            elif change == 'unknown': bad['unknowns'] = []
            elif change == 'conflict':
                bad['facts'].append(deepcopy(bad['facts'][0])); bad['facts'][-1]['value'] = '__wrong__'
            variants.append((True,bad))
        for expected_failure, value in variants:
            predicted_failure = evaluate(case, output(value))['state'] == 'fail'
            tp += expected_failure and predicted_failure
            fn += expected_failure and not predicted_failure
            fp += not expected_failure and predicted_failure
            tn += not expected_failure and not predicted_failure
    report = dict(scope='authored mutations only', true_positive=tp,false_positive=fp,false_negative=fn,true_negative=tn,
                  precision=tp/(tp+fp),recall=tp/(tp+fn))
    print(json.dumps(report,sort_keys=True))
    assert report == dict(scope='authored mutations only',true_positive=72,false_positive=0,false_negative=0,true_negative=24,precision=1.0,recall=1.0)
