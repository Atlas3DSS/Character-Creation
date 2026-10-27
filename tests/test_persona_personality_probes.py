import persona
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest
from persona_personality_probes import load_suite,registered_rows,validate_suite,AXES
from persona_vectors import canonical_digest

PATH=Path(__file__).resolve().parents[1]/'data/persona_personality_probes.json'


def test_24_probes_are_twelve_matched_factual_families():
    suite=load_suite(PATH)
    assert len(suite['cases'])==24
    assert Counter(c['axis'] for c in suite['cases'])==Counter({axis:4 for axis in AXES})
    assert Counter(c['pressure'] for c in suite['cases'])=={'lower':12,'higher':12}
    assert len({c['pair_id'] for c in suite['cases']})==12
    assert suite['suite_sha256']==hashlib.sha256(PATH.read_bytes()).hexdigest()
    for c in suite['cases']:
        assert all(f['source_span'] in c['prompt'] for f in c['fact_contract']['facts'])
        assert c['fact_contract']['required_distinctions'] and c['observations']


def test_registration_has_exact_prompt_no_persona_or_kv_and_hashes_all_fields():
    suite=load_suite(PATH);rows=registered_rows(suite)
    assert rows==registered_rows(suite)
    assert len({r['id'] for r in rows})==24
    for case,row in zip(suite['cases'],rows):
        assert row['id']==canonical_digest({k:v for k,v in row.items() if k!='id'})
        assert row['prompt']==case['prompt'] and row['system']==''
        assert row['max_tokens']==2048 and row['thinking'] is False
        assert 'kv_control' not in row and 'recipe' not in row
        assert row['split']=='discovery' and row['corpus']=='personality_diagnostics'
        assert row['labels']['prompt_design_trait']==case['axis']
        assert row['labels']['prompt_design_facet']==case['pressure']
        assert row['fact_contract']==case['fact_contract']
    rows[0]['fact_contract']['facts'][0]['expected']='tampered'
    assert suite['cases'][0]['fact_contract']['facts'][0]['expected']!='tampered'


@pytest.mark.parametrize('change',[
    lambda s:s['cases'][0].update(prompt='Unbound replacement'),
    lambda s:s['cases'][0]['fact_contract']['facts'][0].update(source_span='not present in the prompt'),
    lambda s:s['cases'][0].update(system='You are sarcastic.'),
    lambda s:s['policy'].update(kv_control=True),
    lambda s:s['cases'][0].update(axis='reassurance'),
    lambda s:s['cases'][0]['fact_contract']['unknowns'].append('A changed contract in only one variant'),
])
def test_source_evidence_policy_and_pair_changes_fail_closed(change):
    suite=load_suite(PATH);change(suite)
    with pytest.raises(ValueError):validate_suite(suite)


def test_correctness_not_confused_with_pressure_or_style_ranking():
    suite=load_suite(PATH);cases={c['id']:c for c in suite['cases']}
    assert cases['receipt_lower']['fact_contract']==cases['receipt_higher']['fact_contract']
    assert '$34' in cases['receipt_higher']['fact_contract']['facts'][0]['expected']
    assert 'previous answer was correct' in cases['false_accusation_higher']['fact_contract']['facts'][0]['expected']
    assert cases['interview_pause_higher']['fact_contract']['unknowns']
    assert 'no preferred style' in suite['policy']['style']
    assert 'not untouched heldout' in suite['policy']['transfer']
    assert all('score' not in c and 'desired_style' not in c for c in suite['cases'])
