import json
from pathlib import Path

import numpy as np
import pytest
import persona
import persona_overnight_integrate as m


def fixture(tmp_path):
    identities = dict(zip(m.IDENTITIES, ('model', 'runtime', 'prepared')))
    source = {'id': 'one', 'split': 'discovery', 'tier': 'book_diverse',
              'source_sha256': 'source', 'messages': [{'role': 'user', 'content': 'Exact passage.'}]}
    request = {'messages': source['messages'], 'observation_only': True, 'max_tokens': 1}
    raw = tmp_path / 'capture.npz'
    plane = np.zeros((64, 17408), np.float32); plane[4, 7] = -3
    np.savez(raw, **identities, request_sha256=m.digest(request), stage='all_prefill_search',
             mlp_input_last=plane, prompt_token_ids=np.array([1, 2]), search_positions=np.arange(2),
             final_chunk_positions=np.array([0, 1]))
    artifact = {'path': str(raw), 'sha256': m.file_hash(raw), 'request_sha256': m.digest(request)}
    provenance = tmp_path / 'capture.json'
    provenance.write_text(json.dumps({'request': request, 'activation_artifact': artifact}))
    pointer = {'path': str(provenance), 'sha256': m.file_hash(provenance)}
    validation = {'tokens': 2, 'neurons': 64 * 17408, 'raw_artifact_sha256': artifact['sha256']}
    coverage = {'tokens': 2, 'layers': 64, 'neurons_per_layer': 17408, 'all_neurons': True, 'all_prompt_positions': True}
    row = ('one', 0, *map(json.dumps, (source, request, validation, artifact, pointer)),
           *[identities[k] for k in m.IDENTITIES], json.dumps(coverage))
    return row, {'one': source}, np.array([[4, 7]], np.int64)


def test_real_coordinate_read_and_binding(tmp_path):
    row, frozen, coordinates = fixture(tmp_path)
    out = m.extract(row, frozen, tmp_path, coordinates, 'model')
    assert np.frombuffer(out[-1], '<f4').tolist() == [-3]
    assert json.loads(out[-2])['messages'] == frozen['one']['messages']
    changed = dict(frozen['one'], messages=[{'role': 'user', 'content': 'Changed'}])
    with pytest.raises(ValueError, match='frozen input'):
        m.extract(row, {'one': changed}, tmp_path, coordinates, 'model')
    with (tmp_path / 'capture.npz').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='bytes changed'):
        m.extract(row, frozen, tmp_path, coordinates, 'model')


def test_intervention_and_unknown_split_rejected(tmp_path):
    row, frozen, coordinates = fixture(tmp_path)
    bad = list(row); request = json.loads(bad[3]); request['steering'] = {}; bad[3] = json.dumps(request)
    with pytest.raises(ValueError, match='unsteered'):
        m.extract(bad, frozen, tmp_path, coordinates, 'model')
    source = dict(frozen['one'], split='unknown'); bad = list(row); bad[2] = json.dumps(source)
    with pytest.raises(ValueError, match='discovery/heldout tiers'):
        m.extract(bad, {'one': source}, tmp_path, coordinates, 'model')


def test_publish_keeps_all_capture_counts_but_deduplicates_correlations(tmp_path):
    row, frozen, coordinates = fixture(tmp_path)
    result = m.extract(row, frozen, tmp_path, coordinates, 'model')
    binding = {'input_sha256': 'input', 'map_sha256': 'map', 'extractor_sha256': 'extractor'}
    db = m.cache_database(tmp_path / 'cache.sqlite', binding)
    db.execute('INSERT INTO captures VALUES(?,?,?,?,?,?,?,?)', result)
    duplicate = list(result); duplicate[0] = 'two'; duplicate[1] = 1
    record = json.loads(duplicate[-2]); record['id'] = 'two'; duplicate[-2] = json.dumps(record)
    db.execute('INSERT INTO captures VALUES(?,?,?,?,?,?,?,?)', duplicate); db.commit()
    summary = m.publish(db, tmp_path, {'neurons': [{'layer': 4, 'neuron': 7}]}, 'map', binding,
                        m.Counter(complete=2, pending=3))
    assert summary['imported_captures'] == 2 and summary['unique_observations'] == 1
    assert summary['duplicates'] == 1 and summary['total_tasks'] == 5
    records = [json.loads(line) for line in (tmp_path / m.FILES[1]).read_text().splitlines()]
    assert records[0]['duplicate_count'] == 2 and records[0]['duplicate_ids'] == ['two']
    with np.load(tmp_path / m.FILES[0], allow_pickle=False) as a:
        assert a['values'].shape == (1, 1)
        assert a['source_hashes'].tolist() == [result[3]]
    db.execute('UPDATE captures SET values_blob=? WHERE id=?', (np.array([5], '<f4').tobytes(), 'two')); db.commit()
    with pytest.raises(ValueError, match='unequal'):
        m.publish(db, tmp_path, {'neurons': [{'layer': 4, 'neuron': 7}]}, 'map', binding, m.Counter(complete=2))


def test_cache_cannot_silently_rebind_source(tmp_path):
    path = tmp_path / 'cache.sqlite'; m.cache_database(path, {'input_sha256': 'first'}).close()
    with pytest.raises(ValueError, match='binding changed'):
        m.cache_database(path, {'input_sha256': 'second'})


def test_heldout_never_enters_discovery_matrix(tmp_path):
    row, frozen, coordinates = fixture(tmp_path)
    discovery = m.extract(row, frozen, tmp_path, coordinates, 'model')
    held_source = dict(frozen['one'], id='held', split='heldout')
    held_row = list(row); held_row[0] = 'held'; held_row[1] = 1; held_row[2] = json.dumps(held_source)
    held = m.extract(held_row, {'held': held_source}, tmp_path, coordinates, 'model')
    assert discovery[4] != held[4]
    binding = {'input_sha256': 'input', 'map_sha256': 'map', 'extractor_sha256': 'extractor'}
    db = m.cache_database(tmp_path / 'cache.sqlite', binding)
    db.executemany('INSERT INTO captures VALUES(?,?,?,?,?,?,?,?)', [discovery, held]); db.commit()
    summary = m.publish(db, tmp_path, {'neurons': [{'layer': 4, 'neuron': 7}]}, 'map', binding, m.Counter(complete=2))
    assert summary['imported_captures'] == 2 and summary['unique_observations'] == 2
    assert summary['discovery_observations'] == 1 and summary['heldout_observations'] == 1
    with np.load(tmp_path / m.FILES[0], allow_pickle=False) as a:
        assert a['prompt_ids'].tolist() == ['one'] and json.loads(str(a['metadata']))['split'] == 'discovery'
    with np.load(tmp_path / m.HELDOUT, allow_pickle=False) as a:
        assert a['prompt_ids'].tolist() == ['held'] and json.loads(str(a['metadata']))['split'] == 'heldout'


def test_mapping_budget_finite_and_shared(tmp_path, monkeypatch):
    status = tmp_path / 'status.json'
    clock = [100.0]; sleeps = []
    monkeypatch.setattr(m.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(m.time, 'time', lambda: clock[0])
    def sleep(delay):
        sleeps.append(delay); clock[0] += delay
    monkeypatch.setattr(m.time, 'sleep', sleep)
    status.write_text(json.dumps({'state': 'running', 'deadline': 200, 'counts': {'pending': 2}}))
    budget = m.MappingReadBudget(status, rate=1024)
    budget.charge(1024); budget.charge(2048)
    assert sleeps == [1, 2]
    status.write_text(json.dumps({'state': 'complete', 'deadline': 200, 'counts': {}}))
    budget.charge(1024)
    assert sleeps == [1, 2]
    clock[0] = 201
    status.write_text(json.dumps({'state': 'running', 'deadline': 200, 'counts': {'pending': 2}}))
    budget.charge(1024)
    assert sleeps == [1, 2]


def test_capture_read_budget_preserves_bytes(tmp_path, monkeypatch):
    data = b'abcd' * 700000; path = tmp_path / 'raw'; path.write_bytes(data)
    charged = []
    class Budget:
        def charge(self, size): charged.append(size)
    monkeypatch.setattr(m, 'READ_BUDGET', Budget())
    assert m.read_capture(path) == data
    assert sum(charged) == len(data)
    assert max(charged) <= 1024 * 1024


def test_sql_budget_runs_after_connection_closed(tmp_path, monkeypatch):
    path = tmp_path / 'ledger.sqlite'
    with m.sqlite3.connect(path) as db:
        db.execute('create table tasks(id,ordinal,row,request,validation,response,state)')
        db.execute('insert into tasks values(?,?,?,?,?,?,?)', ('x', 0, '{}', '{}', '{}', '{}', 'complete'))
    class Budget:
        def charge(self, size):
            assert size == 9
            db = m.sqlite3.connect(path, timeout=0)
            db.execute('begin exclusive'); db.rollback(); db.close()
    monkeypatch.setattr(m, 'READ_BUDGET', Budget())
    assert len(m.capture_rows(path, [0])) == 1


def test_active_read_rate_validation():
    assert m.active_read_rate('4') == 4
    assert m.active_read_rate('0.5') == 0.5
    assert m.active_read_rate('1024') == 1024
    for value in ('0', '-1', 'nan', 'inf', '-inf', '1025', 'bad', None):
        with pytest.raises(m.argparse.ArgumentTypeError):
            m.active_read_rate(value)
