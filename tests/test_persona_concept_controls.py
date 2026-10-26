import copy
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'persona'))
from persona_concept_controls import Registry
from persona_vectors import canonical_digest


class App:
    def __init__(self, path):
        self.path = path
        self.model = SimpleNamespace(identity='model', layers=2, intermediate=8)
        self.lens = [{'layer': 0, 'neuron': 1, 'direction': 'positive', 'token': 'rage', 'logit': 3.},
                     {'layer': 1, 'neuron': 2, 'direction': 'negative', 'token': 'rage', 'logit': 2.}]
        self.atlas = SimpleNamespace(model=self.model, search=lambda *a, **kw: {'matches': self.lens, 'scope': 'stored tokens'}, query=self.query)
        self.measured = {'available': True, 'status': 'ready',
            'source': {'model_identity': 'model', 'artifact_sha256': 'a' * 64, 'map_sha256': 'b' * 64, 'split': 'discovery'},
            'candidates': [{'layer': 0, 'neuron': 1, 'score': -2.}, {'layer': 1, 'neuron': 2, 'score': 1.}]}
        self.concepts = SimpleNamespace(query=lambda **kw: copy.deepcopy(self.measured))
    def connect(self):
        return sqlite3.connect(self.path)
    def query(self, layer, neuron):
        row = next(r for r in self.lens if (r['layer'], r['neuron']) == (layer, neuron))
        return {'available': True, 'source': {'model_identity': self.model.identity, 'source_id': 'source'},
                row['direction']: [{'text': row['token'], 'logit': row['logit']}]}


def payload(source='lens'):
    return {'name': 'Rage association', 'source': source, 'query': 'rage',
            'neurons': [{'layer': 0, 'neuron': 1}, {'layer': 1, 'neuron': 2}]}


def test_source_signed_equal_weights_and_bounded_expansion(tmp_path):
    app = App(tmp_path / 'controls.sqlite')
    registry = Registry(app)
    lens = registry.create(payload())
    measured = registry.create(payload('measured'))
    assert [r['weight'] for r in lens['neurons']] == [.5, -.5]
    assert [r['weight'] for r in measured['neurons']] == [-.5, .5]
    pushes, snapshots = registry.expand([{'id': lens['id'], 'value': .5}], .4)
    assert [r['dose'] for r in pushes] == [.1, -.1]
    assert snapshots[0]['validation'] == 'association_only'
    pushes, snapshots = registry.expand([{'id': lens['id'], 'value': 1}, {'id': measured['id'], 'value': -1}], .4)
    assert sum(abs(r['dose']) for r in pushes) == pytest.approx(.4)
    assert snapshots[0]['dial_l1_scale'] == .5
    cancelled, _ = registry.expand([{'id': lens['id'], 'value': 1}, {'id': measured['id'], 'value': 1}], .4)
    assert cancelled == []
    assert len(registry.listing()['controls']) == 2
    assert Registry(app).listing()['controls'] == registry.listing()['controls']


def test_reject_fabricated_coordinates_weights_and_wrong_binding(tmp_path):
    app = App(tmp_path / 'controls.sqlite'); registry = Registry(app)
    value = payload(); value['neurons'][0]['sign'] = -1
    with pytest.raises(ValueError, match='weights or signs'):
        registry.create(value)
    value = payload(); value['neurons'][0]['neuron'] = 7
    with pytest.raises(ValueError, match='not returned'):
        registry.create(value)
    app.measured['source']['model_identity'] = 'other'
    with pytest.raises(ValueError, match='validated'):
        registry.create(payload('measured'))
    control = registry.create(payload())
    with pytest.raises(ValueError):
        registry.expand([{'id': control['id'], 'value': True}], .4)
    with pytest.raises(ValueError):
        registry.expand([{'id': control['id'], 'value': 1}] * 2, .4)
    with app.connect() as db:
        db.execute("UPDATE concept_controls SET record='{}' WHERE id=?", (control['id'],))
    with pytest.raises(ValueError, match='integrity'):
        registry.listing()


def test_activation_control_requires_complete_bound_raw_evidence(tmp_path):
    app = App(tmp_path / 'controls.sqlite'); registry = Registry(app)
    key = 'a' * 32
    request = {'model': 'checkpoint', 'messages': [{'role': 'user', 'content': 'rage'}]}
    request_sha = canonical_digest(request)
    artifact = {'sha256': 'c' * 64, 'request_sha256': request_sha}
    ids = {'model_identity': 'model', 'runtime_identity': 'runtime', 'prepared_identity': 'prepared'}
    result = {'id': key, 'coverage': {'all_neurons': True, 'all_prompt_positions': True, 'layers': 2, 'neurons_per_layer': 8, 'tokens': 2},
              'tokens': [{'position': 0, 'id': 1, 'text': 'a'}, {'position': 1, 'id': 2, 'text': 'b'}],
              'candidates': [{'layer': 0, 'neuron': 1, 'peak': -2., 'peak_position': 1, 'mean_abs': 1.5, 'values': [-1., -2.]}]}
    output = {**result, **ids, 'request_sha256': request_sha, 'activation_artifact': artifact}
    record = {'id': key, 'state': 'complete', 'input': {'model_identity': 'model', 'top_k': 1}, 'output': output}
    app.activation_lab = SimpleNamespace(get=lambda _: copy.deepcopy(record))
    raw = {'request': request, 'request_sha256': request_sha, 'endpoint': ids,
           'response': {**ids, 'activation_search': result, 'activation_artifact': artifact}}
    with app.connect() as db:
        db.execute('CREATE TABLE activation_searches(id TEXT,raw TEXT)')
        db.execute('INSERT INTO activation_searches VALUES(?,?)', (key, json.dumps(raw)))
    value = {'name': 'Observed rage text', 'source': 'activation', 'query': 'rage', 'search_id': key,
             'neurons': [{'layer': 0, 'neuron': 1}]}
    control = registry.create(value)
    assert control['neurons'][0]['weight'] == -1
    assert control['evidence']['request_sha256'] == request_sha
    raw['response']['runtime_identity'] = 'other'
    with app.connect() as db:
        db.execute('UPDATE activation_searches SET raw=?', (json.dumps(raw),))
    with pytest.raises(ValueError, match='identity'):
        registry.create(value)


def test_old_model_control_cannot_expand(tmp_path):
    app = App(tmp_path / 'controls.sqlite'); registry = Registry(app)
    control = registry.create(payload())
    app.model = SimpleNamespace(identity='changed', layers=2, intermediate=8)
    assert registry.listing()['controls'] == []
    with pytest.raises(ValueError, match='binding'):
        registry.expand([{'id': control['id'], 'value': 1}], .2)
