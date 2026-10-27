import persona
import json
import sqlite3
import pytest
from persona_preset_publication import snapshot, due
from persona_vectors import canonical_digest


def fixture(tmp_path):
    results = tmp_path / 'results'; results.mkdir()
    prompt = {'id': 'p', 'prompt': 'I changed my mind.', 'split': 'discovery'}
    plan = {'model_identity': 'm', 'discovery': [prompt], 'confirmation': []}
    plan['input_sha256'] = canonical_digest(plan)
    (results / 'persona_preset_overnight_plan.json').write_text(json.dumps(plan))
    job = {'id': 'j', 'model_identity': 'm', 'prompt': prompt}
    with sqlite3.connect(results / 'persona_preset_overnight.sqlite') as db:
        db.executescript('CREATE TABLE jobs(id,state,body); CREATE TABLE attempts(job_id,number,result);')
        db.execute('INSERT INTO jobs VALUES(?,?,?)', ('j', 'complete', json.dumps(job)))
        result = {'complete': True, 'capture_validated': True, 'request': {'api_key': 'private'},
            'response': {'choices': [{'message': {'content': 'Tell me more.'}, 'finish_reason': 'stop'}]},
            'activation_artifact': {'sha256': 'a' * 64, 'path': '/private/activation.npz'}, 'error': None}
        db.execute('INSERT INTO attempts VALUES(?,?,?)', ('j', 1, json.dumps(result)))
    return results


def test_public_allowlist_preserves_answer_and_hash_not_private_transport(tmp_path):
    fixture(tmp_path); value = snapshot(tmp_path)
    assert value['terminal_jobs'] == 1 and value['rows'][0]['answer'] == 'Tell me more.'
    assert value['rows'][0]['activation_artifact'] == {'sha256': 'a' * 64}
    assert 'private' not in json.dumps(value) and not value['raw_arrays']['included']


def test_reject_unknown_or_changed_prompt(tmp_path):
    results = fixture(tmp_path)
    with sqlite3.connect(results / 'persona_preset_overnight.sqlite') as db:
        job = json.loads(db.execute('SELECT body FROM jobs').fetchone()[0]); job['prompt']['prompt'] = 'Other'
        db.execute('UPDATE jobs SET body=?', (json.dumps(job),))
    with pytest.raises(ValueError, match='authored study'): snapshot(tmp_path)


def test_ambiguous_upload_is_never_retried_and_judgment_batches_publish():
    value = {'terminal_jobs': 96, 'evaluations': {}, 'reviews': []}
    assert not due({'attempts': [{'state': 'uncertain_no_retry'}]}, value, True)
    previous = {'state': 'verified', 'terminal_jobs': 96, 'evaluation_sha256': canonical_digest({}), 'review_groups': 0}
    assert not due({'attempts': [previous]}, value, False)
    value['reviews'] = [{}] * 8
    assert due({'attempts': [previous]}, value, False)
    assert due({'attempts': [previous]}, value, True)
