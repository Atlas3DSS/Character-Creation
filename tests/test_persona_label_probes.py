"""Structural and adversarial validation; not automated semantic truth grading."""
from collections import Counter
from copy import deepcopy
from difflib import SequenceMatcher
import json
from pathlib import Path
import pytest
from persona.persona_label_probes import load_probes, validate_probes, bind_row, digest, DEFINITIONS

PATH = Path(__file__).resolve().parents[1] / 'data/persona_label_probes.json'
DATA = load_probes(PATH)


def seal(data):
    data['sha256'] = digest({k:v for k,v in data.items() if k != 'sha256'})
    return data


def test_balanced_disjoint_source_bound_dataset():
    report = validate_probes(DATA)
    assert report['rows'] == report['families'] == 432
    assert report['concepts'] == 6
    for concept in DEFINITIONS:
        rows = [r for r in DATA['rows'] if r['concept'] == concept]
        assert Counter((r['split'],r['label']) for r in rows) == {
            ('discovery',True):12,('discovery',False):12,('heldout',True):24,('heldout',False):24}
    for row in DATA['rows']:
        assert row['evidence_span'] in row['prompt']
        assert row['id'] == bind_row(row)['id']
    assert len({r['id'] for r in DATA['rows']}) == 432


def test_no_highly_similar_cross_split_frames():
    # This is a text-overlap alarm, not proof of semantic independence.
    for concept in DEFINITIONS:
        discovery = [r for r in DATA['rows'] if r['concept']==concept and r['split']=='discovery']
        heldout = [r for r in DATA['rows'] if r['concept']==concept and r['split']=='heldout']
        for first in discovery:
            for second in heldout:
                ratio = SequenceMatcher(None,first['prompt'].casefold(),second['prompt'].casefold(),autojunk=False).ratio()
                assert ratio < .8, (first['family_id'],second['family_id'],ratio)


@pytest.mark.parametrize('field,value', [('prompt','changed source'),('label',False),('split','heldout'),('family_id','changed-family')])
def test_unbound_edits_rejected(field,value):
    data = deepcopy(DATA); data['rows'][0][field] = value
    with pytest.raises(ValueError,match='binding'):
        validate_probes(data)
    seal(data)
    with pytest.raises(ValueError,match='content-bound'):
        validate_probes(data)


@pytest.mark.parametrize('mutation,message', [
    ('family','family reused'),('prompt','Duplicate source'),('evidence','exact nonempty'),
    ('nonboolean','non-boolean'),('balance','12 discovery'),('adult','explicitly establish adults'),
    ('definition','inconsistent definition'),('extra','Unexpected row fields')])
def test_resealed_invalid_data_rejected(mutation,message):
    data = deepcopy(DATA); row = data['rows'][1]
    if mutation=='family': row['family_id'] = data['rows'][0]['family_id']
    elif mutation=='prompt': row['prompt'] = data['rows'][0]['prompt']; row['evidence_span']=row['prompt']
    elif mutation=='evidence': row['evidence_span'] = 'invented supporting words'
    elif mutation=='nonboolean': row['label'] = 1
    elif mutation=='balance': row['split'] = 'heldout'
    elif mutation=='adult':
        row = next(r for r in data['rows'] if r['concept']=='sexual_desire' and r['label'])
        row['prompt'] = 'A person experiences attraction.';row['evidence_span']=row['prompt']
    elif mutation=='definition': row['definition'] = 'arbitrary'
    elif mutation=='extra': row['extra'] = 'not in contract'
    row['id'] = bind_row(row)['id'];seal(data)
    with pytest.raises(ValueError,match=message): validate_probes(data)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path/'bad.json';path.write_text('{"schema":1,"schema":2}')
    with pytest.raises(ValueError,match='Duplicate JSON key'): load_probes(path)


def test_authored_lexical_and_relation_controls():
    indexed = {r['family_id']:r for r in DATA['rows']}
    pairs = [
        ('joy-choir-harmony','joy-joy-definition'),
        ('sadness-empty-chair','sadness-joyful-tears'),
        ('sexual_desire-first-date-boundary','sexual_desire-romantic-without-erotic'),
        ('confusion-unclear-pronoun','confusion-uncertain-weather'),
        ('green-emerald-scarf','green-printed-color-word'),
        ('dog-guide-at-crossing','dog-photograph-only'),
    ]
    for positive,negative in pairs:
        assert indexed[positive]['label'] is True
        assert indexed[negative]['label'] is False
    for concept in DEFINITIONS:
        negatives = [r for r in DATA['rows'] if r['concept']==concept and not r['label']]
        assert any(concept.replace('_',' ') in r['prompt'].casefold() for r in negatives)
    assert 'no accuracy threshold' in ' '.join(DATA['limitations']).lower()
