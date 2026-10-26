import argparse
import copy
import json
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'persona'))
import persona_literature_annotate as literature


def annotation():
    return {'entities': [{'name': 'Mara', 'kind': 'character', 'evidence': {'text': 'Mara said, “Stay.”'}, 'certainty': 'explicit'}],
            'themes': [{'theme': 'trust', 'evidence': {'text': 'I trust Mara.'}, 'certainty': 'uncertain'}],
            'scene': {'description': 'Mara asks someone to remain.', 'evidence': {'text': 'Mara said, “Stay.”'},
                'certainty': 'explicit', 'events': [], 'boundary': {'kind': 'unclear',
                    'evidence': {'text': 'Mara said, “Stay.”'}, 'certainty': 'uncertain'}},
            'aliases': [], 'speakers': [{'speaker': 'Mara', 'utterance': {'text': 'Stay.'},
                'evidence': {'text': 'Mara said, “Stay.”'}, 'certainty': 'explicit'}]}


TEXT = 'Mara said, “Stay.” I trust Mara.'


def test_exact_grounding_offsets_and_uncertainty():
    output = literature.annotate_text(TEXT, '```json\n' + json.dumps(annotation()) + '\n```', start=91)
    assert output['entities'][0]['mention']['document_start'] == 91
    assert output['speakers'][0]['utterance']['start'] == TEXT.index('Stay.')
    assert output['themes'][0]['certainty'] == 'uncertain'
    assert output['neural_finding'] is False
    assert output['scene']['boundary']['kind'] == 'unclear'
    raw = annotation()
    raw['themes'][0]['evidence']['text'] = 'I trust Mars.'
    with pytest.raises(ValueError, match='exact source'):
        literature.annotate_text(TEXT, raw)


def test_repeated_evidence_and_invalid_schema():
    empty = {'entities': [], 'themes': [], 'scene': None, 'aliases': [], 'speakers': []}
    raw = copy.deepcopy(empty)
    raw['themes'] = [{'theme': 'fear', 'evidence': {'text': 'Run.'}, 'certainty': 'uncertain'}]
    with pytest.raises(ValueError, match='occurrence'):
        literature.annotate_text('Run. Run.', raw)
    raw['themes'][0]['evidence']['occurrence'] = 1
    assert literature.annotate_text('Run. Run.', raw)['themes'][0]['evidence']['start'] == 5
    raw['themes'][0]['theme'] = 'firing neuron'
    with pytest.raises(ValueError, match='controlled'):
        literature.annotate_text('Run. Run.', raw)
    with pytest.raises(ValueError, match='Duplicate'):
        literature.parse_json('{"a":1,"a":2}')
    with pytest.raises(ValueError):
        literature.parse_json('{"a":NaN}')
    raw = annotation()
    raw['speakers'][0]['utterance'] = {'text': 'I trust Mara.'}
    with pytest.raises(ValueError, match='this exact utterance'):
        literature.annotate_text(TEXT, raw)
    raw = annotation()
    raw['aliases'] = [{'name': 'Mara', 'alias': 'Captain', 'evidence': {'text': 'Mara said, “Stay.”'}, 'certainty': 'uncertain'}]
    with pytest.raises(ValueError, match='both exact'):
        literature.annotate_text(TEXT, raw)


def database(tmp_path, n=6):
    path = tmp_path / 'literature.sqlite'
    with literature.connect(path) as db:
        db.execute('CREATE TABLE books(id TEXT PRIMARY KEY,sha256 TEXT)')
        db.execute('CREATE TABLE units(id TEXT PRIMARY KEY,book_id TEXT,document_id TEXT,ordinal INTEGER,start INTEGER,end INTEGER,text TEXT,split TEXT,boundary TEXT,state TEXT DEFAULT \'pending\',annotation TEXT,error TEXT)')
        db.executemany('INSERT INTO books VALUES(?,?)', [('a', 'hash-a'), ('b', 'hash-b')])
        for i in range(n):
            db.execute('INSERT INTO units(id,book_id,document_id,ordinal,start,end,text,split,boundary) VALUES(?,?,?,?,?,?,?,?,?)',
                (str(i), 'a' if i < n//2 else 'b', 'doc', i, 0, len(TEXT), TEXT, 'discovery', 'paragraph'))
    return path


def endpoint(port):
    return {'url': 'http://127.0.0.1:' + str(port), 'model_identity': 'model', 'runtime_identity': 'runtime', 'prepared_identity': 'prepared'}


def descriptor(ep):
    return {'data': [{'id': 'checkpoint', 'flare': {**{k: ep[k] for k in literature.IDENTITIES},
             'logical_context_tokens': 4096, 'physical_context_tokens': 4096, 'calibration_equivalence': {'passed': True}}}]}


def response(ep, payload):
    return {**{k: ep[k] for k in literature.IDENTITIES}, 'model': 'checkpoint',
            'activation_artifact': {'request_sha256': literature.canonical_digest(payload)},
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(annotation())}}],
            'diagnostics': {'complete_final_answer': True}, 'usage': {'total_tokens': 1500, 'completion_tokens': 800}}


def test_resume_round_robin_and_owned_recovery(tmp_path):
    path = database(tmp_path)
    assert literature.pending_order(path, 4) == ['0', '3', '1', '4']
    assert literature.prepare_attempts(path) == 0
    with literature.connect(path) as db:
        db.execute("UPDATE units SET state='running' WHERE id IN ('0','1')")
        db.execute("UPDATE units SET state='failed' WHERE id='2'")
        db.execute('INSERT INTO literature_attempts(id,run_id,schema,unit_id,state,started,endpoint_json,source_sha256,book_sha256) VALUES(?,?,?,?,?,?,?,?,?)',
                   ('owned', 'dead-run', literature.SCHEMA, '0', 'running', 0, '{}', 'source', 'hash-a'))
    assert literature.prepare_attempts(path) == 1
    with literature.connect(path) as db:
        states = dict(db.execute('SELECT id,state FROM units'))
        assert states['0'] == 'pending' and states['1'] == 'running' and states['2'] == 'failed'
        assert db.execute("SELECT state FROM literature_attempts WHERE id='owned'").fetchone()[0] == 'interrupted'
    assert '2' not in literature.pending_order(path, None)


def test_runner_exclusive_endpoints_raw_attempts_and_resume(tmp_path, monkeypatch):
    path = database(tmp_path)
    endpoints = [endpoint(8940), endpoint(8941)]
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'activation_endpoints': endpoints}))
    active = {ep['url']: 0 for ep in endpoints}
    mutex = threading.Lock()
    def fetch(url, payload=None, timeout=None):
        ep = next(ep for ep in endpoints if url.startswith(ep['url']))
        if payload is None:
            return descriptor(ep)
        assert payload['max_tokens'] == 2048 and payload['chat_template_kwargs']['enable_thinking'] is False
        with mutex:
            active[ep['url']] += 1
            assert active[ep['url']] == 1
        time.sleep(.005)
        with mutex:
            active[ep['url']] -= 1
        return response(ep, payload)
    monkeypatch.setattr(literature, 'fetch', fetch)
    args = argparse.Namespace(database=path, config=config, limit=None, minutes=None)
    result = literature.run(args)
    assert result['complete'] == 6 and result['failed'] == 0 and result['interrupted'] is False
    with literature.connect(path) as db:
        attempts = db.execute('SELECT * FROM literature_attempts').fetchall()
        assert len(attempts) == 6
        assert all(row['request_json'] and row['response_json'] and row['source_sha256'] and row['annotation_json'] and row['descriptor_json'] for row in attempts)
    assert literature.run(args)['attempted'] == 0


def test_identity_failure_stops_endpoint_without_failing_queue(tmp_path, monkeypatch):
    path = database(tmp_path)
    ep = endpoint(8940)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'activation_endpoints': [ep]}))
    bad = descriptor(ep)
    bad['data'][0]['flare']['runtime_identity'] = 'unexpected'
    monkeypatch.setattr(literature, 'fetch', lambda *a, **kw: bad)
    result = literature.run(argparse.Namespace(database=path, config=config, limit=None, minutes=None))
    assert result['failed'] == 1 and result['unit_states']['pending'] == 5
    with literature.connect(path) as db:
        assert 'runtime_identity' in db.execute('SELECT error FROM literature_attempts').fetchone()[0]


def test_rejects_truncation_and_request_mismatch():
    ep = endpoint(8940)
    payload = literature.request_payload({'id': 'unit', 'text': TEXT}, 'checkpoint')
    result = response(ep, payload)
    result['choices'][0]['finish_reason'] = 'length'
    with pytest.raises(ValueError, match='truncated'):
        literature.verify_response(ep, payload, result)
    result = response(ep, payload)
    result['activation_artifact']['request_sha256'] = 'different'
    with pytest.raises(ValueError, match='binding'):
        literature.verify_response(ep, payload, result)


def test_bulk_yields_without_claiming_units(tmp_path, monkeypatch):
    import fcntl
    path = database(tmp_path)
    ep = endpoint(8940)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'activation_endpoints': [ep]}))
    monkeypatch.setattr(literature, 'fetch', lambda *a, **kw: pytest.fail('No inference while interactive endpoint is occupied'))
    args = argparse.Namespace(database=path, config=config, limit=None, minutes=.0002)
    lock = literature.endpoint_lock_path(tmp_path, ep['url'])
    with lock.open('a+') as occupied:
        fcntl.flock(occupied, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert literature.run(args)['attempted'] == 0
    with literature.connect(tmp_path / 'persona_explorer.sqlite') as db:
        db.execute('CREATE TABLE cells(state TEXT)')
        db.execute("INSERT INTO cells VALUES('queued')")
    assert literature.interactive_pending((tmp_path / 'persona_explorer.sqlite').resolve())
    assert literature.run(args)['attempted'] == 0
    with literature.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM units WHERE state='pending'").fetchone()[0] == 6
        assert db.execute('SELECT COUNT(*) FROM literature_attempts').fetchone()[0] == 0


def test_first_round_prefers_long_interior_then_keeps_all(tmp_path):
    path = database(tmp_path, n=12)
    with literature.connect(path) as db:
        for identity in ('2', '8'):
            text = 'Interior narrative. ' * 70
            db.execute('UPDATE units SET text=?,end=? WHERE id=?', (text, len(text), identity))
    order = literature.pending_order(path, None)
    assert order[:2] == ['2', '8']
    assert len(order) == len(set(order)) == 12
    assert order[2:4] == ['0', '6']
