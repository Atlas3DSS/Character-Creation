import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'activation-search/persona_overnight_map.py'
spec = importlib.util.spec_from_file_location('overnight', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_resume_never_requeues_uncertain_inference(tmp_path):
    inputs = tmp_path / 'inputs.jsonl'
    row = {'id': 'source-1', 'messages': [{'role': 'user', 'content': 'Exact source.'}], 'source_sha256': 'source-hash'}
    inputs.write_text(json.dumps(row) + '\n')
    database = tmp_path / 'map.sqlite'
    m.initialize(database, inputs, m.file_hash(inputs))
    with sqlite3.connect(database) as db:
        db.execute("UPDATE tasks SET state='running'")
    m.initialize(database, inputs, m.file_hash(inputs))
    with sqlite3.connect(database) as db:
        assert db.execute('SELECT state FROM tasks').fetchone()[0] == 'uncertain'
    inputs.write_text(json.dumps({**row, 'id': 'changed'}) + '\n')
    with pytest.raises(ValueError, match='Resume manifest'):
        m.initialize(database, inputs, m.file_hash(inputs))


def test_renewal_is_bounded_and_preserves_admission_wrapper(monkeypatch):
    monkeypatch.setattr(m.time, 'time', lambda: 1000)
    command = ['systemd-run', '--property=RuntimeMaxSec=14400', '/usr/bin/python3', '/installed/lease.py',
               'run', 'character-creation', 'activation-search-probe-0', '31', '240', 'v100-0', '--', 'taskset', '-c', '0-3', 'python', 'probe.py']
    result = m.renewal_command(command, 1000 + 3600)
    assert result[result.index('run') + 4] == '60'
    assert '--property=RuntimeMaxSec=3600' in result
    assert '--setenv=PERSONA_JOB_DEADLINE=4600' in result
    assert result[result.index('run') + 3] == '31'
    assert '/installed/lease.py' in result
    assert command[1] == '--property=RuntimeMaxSec=14400'
    with pytest.raises(ValueError):
        m.renewal_command(command, 1500)


def test_artifact_rejects_outside_root_or_modified_bytes(tmp_path):
    path = tmp_path / 'capture.npz'
    path.write_bytes(b'immutable')
    pointer = {'path': str(path), 'sha256': m.file_hash(path)}
    assert m.validate_artifact(pointer, tmp_path) == path
    with pytest.raises(ValueError):
        m.validate_artifact(pointer, tmp_path / 'other')
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):
        m.validate_artifact(pointer, tmp_path)


def test_validates_unranked_neurons_not_only_profiles(tmp_path):
    endpoint = dict(zip(m.IDENTITIES, ('model', 'runtime', 'prepared')))
    request = {'capture_search': {'id': 'a' * 32}}
    path = tmp_path / 'raw.npz'
    ancillary = tmp_path / 'evidence.json'
    ancillary.write_text('{}')
    other = {'path': str(ancillary), 'sha256': m.file_hash(ancillary)}
    shape = (64, 17408)
    peak = np.full(shape, 2, dtype=np.float32)
    total = np.full(shape, 4, dtype=np.float64)
    def save():
        np.savez(path, **endpoint, request_sha256=m.digest(request), stage='all_prefill_search',
                 search_peak=peak, search_peak_position=np.zeros(shape, dtype=np.int64),
                 search_sum_abs=total, search_positions=np.arange(2), prompt_token_ids=[100, 101])
        return {'path': str(path), 'sha256': m.file_hash(path), 'request_sha256': m.digest(request)}
    response = {**endpoint, 'activation_artifact': save(), 'provenance_artifact': other,
                'activation_search': {'id': 'a' * 32, 'tokens': [{'id': 100}, {'id': 101}],
                    'coverage': {'layers': 64, 'neurons_per_layer': 17408, 'tokens': 2,
                                 'all_neurons': True, 'all_prompt_positions': True, 'neuron_token_values': 64*17408*2},
                    'candidates': [{'layer': 0, 'neuron': 0, 'peak': 2, 'peak_position': 0, 'mean_abs': 2, 'values': [2, 2]}],
                    'profile_artifact': other, 'profile_provenance_artifact': other, 'result_artifact': other}}
    assert m.validate_result(response, request, endpoint, tmp_path)['neurons'] == 64*17408
    total[63, 17407] = np.nan
    response['activation_artifact'] = save()
    with pytest.raises(ValueError, match='Nonfinite raw'):
        m.validate_result(response, request, endpoint, tmp_path)
