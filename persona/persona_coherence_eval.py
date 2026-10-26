"""Deterministic fact-ledger checks, not a general prose truth or taste judge.

``pass`` always names a bounded check scope. A correct ledger does not establish
that accompanying prose is grounded. Novel fictional details, sarcasm, cruelty,
anxiety and other tones are not scored for desirability.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any

SCHEMA = 'character-creation.coherence-suite.v1'
RESULT_SCHEMA = 'character-creation.coherence-result.v1'
LIMITS = [
    'Free prose claims, implications, omissions and contradictions are not generally understood by this scorer.',
    'Ledger correctness does not certify accompanying prose as factual or consistent.',
    'Repetition thresholds are mechanical flags; intentional refrains can trigger them.',
    'No personality, morality, politeness, emotional positivity or aesthetic quality is graded.',
]


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _issue(kind: str, detail: str, evidence: Any) -> dict:
    return {'kind': kind, 'detail': detail, 'evidence': evidence}


def _normalized(value: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def _equal(expected: Any, observed: Any) -> bool:
    # Python's True == 1 must not turn a negation/number error into a match.
    if isinstance(expected, bool) or isinstance(observed, bool):
        return type(expected) is type(observed) and expected == observed
    if type(expected) in (int, float) and type(observed) in (int, float):
        return (not isinstance(expected, float) or math.isfinite(expected)) and (not isinstance(observed, float) or math.isfinite(observed)) and expected == observed
    if isinstance(expected, str) and isinstance(observed, str):
        return _normalized(expected) == _normalized(observed)
    if isinstance(expected, list) and isinstance(observed, list):
        return len(expected) == len(observed) and all(_equal(a, b) for a, b in zip(expected, observed))
    return expected is None and observed is None


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object key: ' + key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError('Nonfinite JSON constant: ' + value)


def _parse(answer: str) -> dict:
    if not isinstance(answer, str):
        raise ValueError('Answer must be text')
    text = answer.strip()
    fence = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.I | re.S)
    if fence:
        text = fence.group(1).strip()
    result = json.loads(text, object_pairs_hook=_pairs, parse_constant=_reject_constant)
    if not isinstance(result, dict):
        raise ValueError('The answer must be one JSON object')
    return result


def _repetition(text: str) -> tuple[list[dict], dict]:
    # Inspect prose, not repetitive required JSON field names and punctuation.
    # CJK text need not contain spaces. Treat ideographs/kana as units instead
    # of collapsing an entire unspaced loop into one enormous ``\w+`` token.
    tokens = re.findall(r'[\u3400-\u9fff\u3040-\u30ff]|[^\W\d_\u3400-\u9fff\u3040-\u30ff]+|\d+|[^\w\s]',
                        unicodedata.normalize('NFKC', text).casefold())
    longest, run = 0, 0
    previous = None
    for token in tokens:
        run = run + 1 if token == previous else 1
        previous = token
        longest = max(longest, run)
    repeated = None
    # A contiguous periodic suffix is a conservative detector of generation
    # loops. Noncontiguous word frequency is not a reliable collapse signal.
    for width in range(2, 13):
        if len(tokens) < width * 8:
            continue
        unit = tokens[-width:]
        repeats = 1
        while (repeats + 1) * width <= len(tokens) and tokens[-(repeats + 1) * width:-repeats * width] == unit:
            repeats += 1
        covered = repeats * width
        if repeats >= 8 and covered >= 48 and covered >= len(tokens) * .5:
            repeated = {'pattern': unit, 'repeats': repeats, 'tokens': covered}
            break
    issues = []
    if longest >= 24:
        issues.append(_issue('repetition_collapse', 'At least 24 identical consecutive tokens or symbols.',
                             {'longest_run': longest, 'excerpt': text[-240:]}))
    elif repeated:
        issues.append(_issue('repetition_collapse', 'A repeated suffix occupies at least half of the response and 48 tokens.', repeated))
    return issues, {'surface_tokens': len(tokens), 'longest_identical_token_run': longest,
                    'repeated_suffix': repeated, 'repetition_detected': bool(issues)}


def evaluate_surface(output: dict) -> dict:
    """Cheap checks for existing free-prose batches; semantic status stays unknown.

    Literal conflicts are reported only for an explicit structured ledger with
    duplicate IDs. Ordinary prose is not converted into purported facts by
    fragile keyword matching. Passing these checks does not mean coherent prose.
    """
    issues, uncertainties = [], []
    if not isinstance(output, dict):
        return {'state': 'unknown', 'issues': [_issue('invalid_output', 'Expected an inference output object.', type(output).__name__)],
                'metrics': {}, 'scope': 'surface_only', 'semantic_state': 'unknown', 'limitations': LIMITS}
    answer = output.get('answer')
    if not isinstance(answer, str):
        return {'state': 'unknown', 'issues': [_issue('missing_answer', 'Answer is absent or not text.', {'type': type(answer).__name__})],
                'metrics': {}, 'scope': 'surface_only', 'semantic_state': 'unknown', 'limitations': LIMITS}
    if not answer.strip():
        issues.append(_issue('empty_response', 'The answer contains no non-whitespace text.', {'characters': len(answer)}))
    reason = output.get('finish_reason')
    complete = output.get('complete')
    if reason in ('length', 'max_tokens', 'max_length', 'token_limit'):
        issues.append(_issue('truncated', 'The runtime reports a generation-length limit.', {'finish_reason': reason, 'complete': complete}))
    elif complete is False:
        issues.append(_issue('incomplete', 'The runtime does not mark this as a complete answer.', {'finish_reason': reason, 'complete': complete}))
    elif complete is not True or reason not in ('stop', 'eos', 'end_turn'):
        uncertainties.append('Completion metadata does not establish normal completion.')
    prose = answer
    try:
        parsed = _parse(answer)
        if isinstance(parsed.get('prose'), str):
            prose = parsed['prose']
        # These exact canonical-ID contradictions need no natural-language
        # inference; do not attempt to interpret arbitrary prose JSON objects.
        if set(parsed) == {'prose', 'facts', 'answers', 'unknowns'} and isinstance(parsed['facts'], list):
            seen = {}
            for fact in parsed['facts']:
                if not isinstance(fact, dict) or set(fact) != {'id', 'subject', 'predicate', 'value'} or not isinstance(fact['id'], str):
                    continue
                key = fact['id']
                if key in seen and any(not _equal(seen[key][field], fact[field]) for field in ('subject', 'predicate', 'value')):
                    issues.append(_issue('literal_self_contradiction', 'The same canonical fact ID has conflicting ledger assertions.',
                                         {'first': seen[key], 'second': fact}))
                seen[key] = fact
    except (ValueError, TypeError):
        pass  # Free prose is supported by surface checks; it need not be JSON.
    repetition, metrics = _repetition(prose)
    issues.extend(repetition)
    metrics.update(answer_characters=len(answer), prose_characters=len(prose), completion_known=not uncertainties)
    return {'state': 'fail' if issues else 'unknown' if uncertainties else 'pass', 'issues': issues,
            'uncertainties': uncertainties, 'metrics': metrics, 'scope': 'surface_only',
            'semantic_state': 'unknown', 'limitations': LIMITS}


def _mismatch_kind(spec: dict, expected: Any, observed: Any) -> str:
    # A sentence in a boolean/number slot is a representation violation, not
    # proof that the sentence contradicts the source. Do not infer its meaning.
    if type(expected) is not type(observed) and not (
            type(expected) in (int, float) and type(observed) in (int, float)):
        return 'contract_representation_unknown'
    if isinstance(expected, str) and isinstance(observed, str):
        # Canonical slots are not a natural-language entailment judge. A value
        # outside their explicit domain may be a correct paraphrase.
        domain = spec.get('canonical_domain', [])
        if domain and _normalized(observed) not in {_normalized(x) for x in domain} and len(observed.split()) > 3:
            return 'contract_representation_unknown'
    if isinstance(expected, bool) and isinstance(observed, bool):
        return 'negation_contradiction'
    kind = spec.get('kind')
    if kind in ('role', 'attribution'):
        return 'role_or_attribution_error'
    if kind in ('timeline', 'time'):
        return 'timeline_error'
    if type(expected) in (int, float):
        return 'number_error'
    return 'fact_contradiction'


def evaluate(case: dict, output: dict) -> dict:
    surface = evaluate_surface(output)
    result = {'schema': RESULT_SCHEMA, 'case_id': case.get('id') if isinstance(case, dict) else None,
              'state': surface['state'], 'issues': deepcopy(surface['issues']), 'metrics': dict(surface['metrics']),
              'surface_state': surface['state'], 'ledger_state': 'unknown', 'prose_state': 'unknown',
              'scope': 'surface_and_required_fact_ledger_only', 'limitations': list(LIMITS)}
    if not isinstance(case, dict) or not isinstance(case.get('groundtruth'), dict):
        result['state'] = 'unknown' if surface['state'] != 'fail' else 'fail'
        result['issues'].append(_issue('invalid_case', 'No structured ground truth is available.', None))
        return result
    truth = case['groundtruth']
    if not isinstance(truth.get('facts'), list) or not isinstance(truth.get('answers'), dict) or not isinstance(truth.get('unknowns'), list):
        result['state'] = 'fail' if surface['state'] == 'fail' else 'unknown'
        result['issues'].append(_issue('invalid_case', 'Ground truth requires facts, answers and unknowns.', None))
        return result
    result.update(task_policy=deepcopy(case.get('task_policy', {})), prose_scope=case.get('prose_scope', 'unverified'))
    if truth.get('require_ledger', True) is not True:
        result['state'] = 'fail' if surface['state'] == 'fail' else 'unknown'
        result['limitations'].append('This case does not require a ledger; factual prose cannot be automatically certified.')
        return result
    try:
        parsed = _parse(output.get('answer', '') if isinstance(output, dict) else '')
    except (ValueError, TypeError) as error:
        result['state'] = 'fail'
        result['issues'].append(_issue('ledger_format_failure', 'The required JSON ledger cannot be parsed; this is a contract failure, not proof of hallucination.', str(error)))
        return result
    expected_keys = {'prose', 'facts', 'answers', 'unknowns'}
    if set(parsed) != expected_keys or not isinstance(parsed.get('prose'), str) or not isinstance(parsed.get('facts'), list) or not isinstance(parsed.get('answers'), dict) or not isinstance(parsed.get('unknowns'), list):
        result['state'] = 'fail'
        result['issues'].append(_issue('ledger_format_failure', 'Required top-level fields/types differ.', {'expected_fields': sorted(expected_keys), 'observed_fields': sorted(parsed)}))
        return result
    issues = result['issues']
    if not parsed['prose'].strip():
        issues.append(_issue('missing_prose', 'The required prose field is empty.', {'field': 'prose'}))
    expected_facts = {f['id']: f for f in truth['facts']}
    canonical_domain = [str(f[key]) for f in truth['facts'] for key in ('subject', 'predicate', 'value')
                        if isinstance(f.get(key), str)]
    canonical_domain += [spec['value'] for spec in truth['answers'].values() if isinstance(spec.get('value'), str)]
    observed_ids, matched = set(), 0
    for fact in parsed['facts']:
        if not isinstance(fact, dict) or set(fact) != {'id', 'subject', 'predicate', 'value'} or not isinstance(fact.get('id'), str):
            issues.append(_issue('ledger_format_failure', 'Each fact requires id, subject, predicate and value.', fact)); continue
        identity = fact['id']
        if identity in observed_ids:
            issues.append(_issue('duplicate_fact', 'A required fact ID was supplied more than once.', fact)); continue
        observed_ids.add(identity)
        spec = expected_facts.get(identity)
        if spec is None:
            issues.append(_issue('unsupported_ledger_claim', 'This assertion has no allowed source fact ID.', fact)); continue
        wrong = [key for key in ('subject', 'predicate', 'value') if not _equal(spec[key], fact[key])]
        if wrong:
            kind = 'role_or_attribution_error' if any(k in wrong for k in ('subject', 'predicate')) else _mismatch_kind({**spec, 'canonical_domain': canonical_domain}, spec['value'], fact['value'])
            issues.append(_issue(kind, 'The ledger differs from an explicit source fact.',
                                 {'id': identity, 'fields': wrong, 'expected': {k: spec[k] for k in wrong},
                                  'observed': {k: fact[k] for k in wrong}, 'source_spans': spec.get('evidence', [])}))
        else:
            matched += 1
    for identity in sorted(set(expected_facts) - observed_ids):
        issues.append(_issue('omitted_fact', 'A required source fact is missing.', {'id': identity, 'source_spans': expected_facts[identity].get('evidence', [])}))
    matched_answers = 0
    for key, spec in truth['answers'].items():
        if key not in parsed['answers']:
            issues.append(_issue('omitted_answer', 'A required answer slot is missing.', {'slot': key}))
        elif not _equal(spec['value'], parsed['answers'][key]):
            kind = _mismatch_kind({**spec, 'canonical_domain': canonical_domain}, spec['value'], parsed['answers'][key])
            issues.append(_issue(kind, 'The answer does not match the required canonical representation; its prose meaning is unverified.' if kind == 'contract_representation_unknown' else 'The required answer differs from the supplied facts.',
                                 {'slot': key, 'expected': spec['value'], 'observed': parsed['answers'][key],
                                  'support': spec.get('support', []), 'derivation': spec.get('derivation', '')}))
        else:
            matched_answers += 1
    for key in sorted(set(parsed['answers']) - set(truth['answers'])):
        issues.append(_issue('unsupported_answer', 'An additional answer slot is not licensed by this fixture.', {'slot': key, 'value': parsed['answers'][key]}))
    unknowns = parsed['unknowns']
    if any(not isinstance(x, str) for x in unknowns):
        issues.append(_issue('ledger_format_failure', 'Unknown fields must be canonical string IDs.', unknowns))
        unknowns = [x for x in unknowns if isinstance(x, str)]
    if len(unknowns) != len(set(unknowns)):
        issues.append(_issue('duplicate_unknown', 'Unknown field IDs must be unique.', unknowns))
    expected_unknowns = set(truth['unknowns'])
    for key in sorted(expected_unknowns - set(unknowns)):
        issues.append(_issue('omitted_uncertainty', 'The source does not establish this requested detail; acknowledge it as unknown.', {'field': key}))
    for key in sorted(set(unknowns) - expected_unknowns):
        issues.append(_issue('unsupported_uncertainty', 'An unrequested or source-established field was labeled unknown.', {'field': key}))
    result['metrics'].update(required_facts=len(expected_facts), matched_facts=matched,
                             required_answers=len(truth['answers']), matched_answers=matched_answers,
                             required_unknowns=len(expected_unknowns), acknowledged_unknowns=len(expected_unknowns & set(unknowns)),
                             prose_claims_checked=0)
    ledger_issues = [x for x in issues if x['kind'] not in ('empty_response', 'truncated', 'incomplete', 'repetition_collapse')]
    result['ledger_state'] = 'fail' if ledger_issues else 'pass'
    result['state'] = 'fail' if issues else 'unknown' if surface['state'] == 'unknown' else 'pass'
    result['pass_meaning'] = 'Required canonical facts, derived answer slots, uncertainty acknowledgments and surface checks only; free prose remains unverified.'
    return result


def score(case: dict, output: dict) -> dict:
    return evaluate(case, output)


def load_suite(path: str | Path) -> dict:
    suite = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_pairs, parse_constant=_reject_constant)
    if suite.get('schema') != SCHEMA or not isinstance(suite.get('cases'), list):
        raise ValueError('Unknown coherence suite schema')
    binding = {k: v for k, v in suite.items() if k != 'suite_sha256'}
    if suite.get('suite_sha256') != canonical_digest(binding):
        raise ValueError('Suite content changed without a new immutable binding')
    ids, sources = set(), set()
    for case in suite['cases']:
        body = {k: v for k, v in case.items() if k not in ('id', 'case_sha256')}
        sha = canonical_digest(body)
        if case.get('case_sha256') != sha or case.get('id') != sha[:32] or case['id'] in ids:
            raise ValueError('Case identity is not unique and source-bound')
        if case.get('source_id') != hashlib.sha256(case['source'].encode()).hexdigest() or case['source_id'] in sources:
            raise ValueError('Source identity changed or overlaps another split')
        ids.add(case['id']); sources.add(case['source_id'])
        truth = case['groundtruth']
        if case['split'] not in ('discovery', 'heldout') or case['source'] not in case['prompt']:
            raise ValueError('Invalid split or missing source in prompt')
        if case['task_policy']['kind'] not in ('factual_response', 'fiction_continuation') or not case['prose_scope']:
            raise ValueError('Missing task/prose policy')
        fact_ids = set()
        for fact in truth['facts']:
            if fact['id'] in fact_ids or not fact.get('evidence') or any(not span or span not in case['source'] for span in fact['evidence']):
                raise ValueError('Fact evidence is duplicated or not an exact source span')
            fact_ids.add(fact['id'])
        for spec in truth['answers'].values():
            if not set(spec['support']) <= fact_ids or not spec.get('derivation'):
                raise ValueError('Derived answer lacks source support')
        if len(truth['unknowns']) != len(set(truth['unknowns'])):
            raise ValueError('Duplicate unknown fields')
    if Counter(c['split'] for c in suite['cases']) != {'discovery': 6, 'heldout': 6}:
        raise ValueError('Require six discovery and six heldout cases')
    return suite
