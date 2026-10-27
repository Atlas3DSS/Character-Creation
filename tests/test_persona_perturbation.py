import persona
from copy import deepcopy
import hashlib
import json
import pytest
from persona_perturbation import registered_rows
from persona_vectors import canonical_digest


def packet(rows=None):
    prompt = 'I have changed my mind about our plan.'
    row = {'id': 'author-id', 'pair_id': 'group', 'source_group': 'group', 'split': 'discovery',
        'schema': 'character-creation.manager-corpus.v1', 'corpus': 'persona_perturbation',
        'category': 'changing_mind', 'prompt': prompt, 'system': '', 'thinking': False, 'labels': {},
        'source': {'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}}
    raw = b''.join(json.dumps(item).encode() + b'\n' for item in (rows or [row]))
    manifest = {'schema': 'character-creation.persona-perturbation.v1', 'status': 'complete',
        'records': len(rows or [row]), 'unique_core_groups': 1, 'split_counts': {'discovery': 1},
        'output_sha256': hashlib.sha256(raw).hexdigest()}
    return row, raw, manifest


def test_registration_preserves_authored_identity_and_input():
    row, raw, manifest = packet()
    registered = registered_rows(raw, json.dumps(manifest))[0]
    assert registered['source_row_id'] == row['id']
    assert registered['id'] == canonical_digest({k: v for k, v in registered.items() if k != 'id'})
    assert registered['prompt'] == row['prompt'] and registered['system'] == ''
    assert row['labels'] == {} and registered['labels']['prompt_design_facet'] == 'changing_mind'


def test_incomplete_or_corrupt_publication_is_rejected():
    _, raw, manifest = packet()
    for change in ({'status': 'partial'}, {'output_sha256': '0' * 64}, {'records': 2}, {'split_counts': {'heldout': 1}}):
        with pytest.raises(ValueError): registered_rows(raw, json.dumps({**manifest, **change}))


def test_reject_prompt_hash_tampering_and_cross_split_group():
    row, _, _ = packet()
    second = deepcopy(row); second.update(id='second', split='heldout')
    _, raw, manifest = packet([row, second])
    with pytest.raises(ValueError, match='split'): registered_rows(raw, json.dumps(manifest))
    row['prompt'] = 'Altered prompt'
    _, raw, manifest = packet([row])
    with pytest.raises(ValueError, match='source hash'): registered_rows(raw, json.dumps(manifest))
