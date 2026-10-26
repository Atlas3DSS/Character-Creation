"""Immutable model-bound neuron collections from recorded associations.

Signs come from source evidence, never names or user-supplied weights. These
controls are additive hypotheses, not causal concept validation or ablation.
"""
from __future__ import annotations
from copy import deepcopy
import json
import math
import re
import time
import uuid

from persona_activation_lab import validate_result
from persona_vectors import canonical_digest

SCHEMA = 'character-creation.concept-control.v1'
SEMANTICS = ('Source association only; labels are user descriptions, not established neuron meanings. '
             'Negative dial values reverse an additive direction; they do not zero or ablate neurons. '
             'Behavior and polarity require paired comparisons; no causal or held-out validation is claimed.')


def finite(value: float, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + ' must be finite')
    return float(value)


def coordinates(rows: list, model) -> list[tuple[int, int]]:
    if not isinstance(rows, list) or not 1 <= len(rows) <= 6:
        raise ValueError('Select 1–6 distinct source neurons')
    result = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'layer', 'neuron'}:
            raise ValueError('Coordinates cannot supply weights or signs')
        l, n = row['layer'], row['neuron']
        if (type(l) is not int or type(n) is not int or not 0 <= l < model.layers
                or not 0 <= n < model.intermediate or (l, n) in result):
            raise ValueError('Coordinates must be unique in-model integers')
        result.append((l, n))
    return result


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{32}', value):
        raise ValueError('Invalid saved concept control ID')
    return value


class Registry:
    def __init__(self, app):
        self.app = app
        with app.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS concept_controls(
                id TEXT PRIMARY KEY,model_identity TEXT NOT NULL,created REAL NOT NULL,
                record TEXT NOT NULL,record_sha256 TEXT NOT NULL)''')

    def _decode(self, row) -> dict:
        value = json.loads(row[3])
        if (canonical_digest(value) != row[4] or value.get('id') != row[0]
                or value.get('model_identity') != row[1] or row[1] != self.app.model.identity
                or value.get('created') != row[2] or value.get('schema') != SCHEMA
                or value.get('validation') != 'association_only'):
            raise ValueError('Saved concept control integrity/model binding differs')
        coordinates([{k: r[k] for k in ('layer', 'neuron')} for r in value['neurons']], self.app.model)
        count = len(value['neurons'])
        for neuron in value['neurons']:
            weight = finite(neuron['weight'], 'Saved weight')
            if neuron.get('sign') not in (-1, 1) or weight != neuron['sign'] / count:
                raise ValueError('Saved source signs/equal L1 weights differ')
        return value

    def listing(self) -> dict:
        with self.app.connect() as db:
            rows = db.execute('SELECT id,model_identity,created,record,record_sha256 FROM concept_controls WHERE model_identity=? ORDER BY created DESC,id', (self.app.model.identity,)).fetchall()
        return {'controls': [self._decode(row) for row in rows], 'model_identity': self.app.model.identity,
                'validation': 'association_only', 'semantics': SEMANTICS}

    def _source(self, value: dict, chosen: list[tuple[int, int]]) -> tuple[list[dict], dict]:
        source, query = value['source'], value['query']
        if source == 'lens':
            if self.app.atlas.model.identity != self.app.model.identity:
                raise ValueError('Checkpoint atlas belongs to another model')
            result = self.app.atlas.search(query, word_start=True)
            # Match the UI's first returned row for repeated coordinates.
            allowed = {}
            for row in result.get('matches', []):
                allowed.setdefault((row['layer'], row['neuron']), row)
            selected, publications = [], []
            for key in chosen:
                if key not in allowed:
                    raise ValueError('Coordinate was not returned by the checkpoint query')
                row = deepcopy(allowed[key])
                if row.get('direction') not in ('positive', 'negative') or finite(row.get('logit'), 'Lens logit') == 0:
                    raise ValueError('Lens evidence has no usable signed direction')
                evidence = self.app.atlas.query(*key)
                binding = evidence.get('source', {})
                if not evidence.get('available') or binding.get('model_identity') != self.app.model.identity or not binding.get('source_id'):
                    raise ValueError('Missing model-bound checkpoint evidence')
                if not any(t.get('text') == row['token'] and t.get('logit') == row['logit'] for t in evidence[row['direction']]):
                    raise ValueError('Checkpoint token evidence changed during selection')
                row['sign'] = 1 if row['direction'] == 'positive' else -1
                selected.append(row); publications.append(deepcopy(binding))
            return selected, {'kind': 'checkpoint_token_projection', 'scope': result.get('scope'),
                              'missing_layers': result.get('missing_layers', []), 'sources': publications}
        if source == 'measured':
            result = self.app.concepts.query(q=query)
            binding = result.get('source', {})
            if (not result.get('available') or result.get('status') != 'ready'
                    or binding.get('model_identity') != self.app.model.identity
                    or not binding.get('artifact_sha256') or not binding.get('map_sha256')
                    or binding.get('split') != 'discovery'):
                raise ValueError('Missing validated discovery activation associations')
            allowed = {(r['layer'], r['neuron']): r for r in result.get('candidates', [])}
            selected = []
            for key in chosen:
                if key not in allowed:
                    raise ValueError('Coordinate was not returned by the measured query')
                row = deepcopy(allowed[key]); score = finite(row.get('score'), 'Measured score')
                if score == 0:
                    raise ValueError('A zero association does not define a direction')
                row['sign'] = 1 if score > 0 else -1
                selected.append(row)
            return selected, {'kind': 'discovery_measured_association', 'source': deepcopy(binding),
                              'strata': deepcopy(result.get('strata', [])), 'examples': deepcopy(result.get('examples', []))}
        record = self.app.activation_lab.get(identifier(value['search_id']))
        output = record.get('output') or {}
        if (record.get('state') != 'complete' or record.get('input', {}).get('model_identity') != self.app.model.identity
                or output.get('model_identity') != self.app.model.identity):
            raise ValueError('Select a complete activation search from this model')
        validate_result(output, self.app.model, record['input']['top_k'])
        with self.app.connect() as db:
            row = db.execute('SELECT raw FROM activation_searches WHERE id=?', (record['id'],)).fetchone()
        raw = json.loads(row[0]) if row and row[0] else {}
        request, response, endpoint = raw.get('request'), raw.get('response', {}), raw.get('endpoint', {})
        digest = canonical_digest(request) if isinstance(request, dict) else None
        artifact = response.get('activation_artifact', {})
        if (not digest or digest != raw.get('request_sha256') or artifact.get('request_sha256') != digest
                or output.get('request_sha256') != digest or not artifact.get('sha256')
                or output.get('activation_artifact') != artifact
                or response.get('activation_search', {}).get('candidates') != output['candidates']):
            raise ValueError('Activation search request/evidence binding differs')
        for field in ('model_identity', 'runtime_identity', 'prepared_identity'):
            if not endpoint.get(field) or response.get(field) != endpoint[field] or output.get(field) != endpoint[field]:
                raise ValueError('Activation search identity differs: ' + field)
        allowed = {(r['layer'], r['neuron']): r for r in output['candidates']}
        selected = []
        for key in chosen:
            if key not in allowed:
                raise ValueError('Coordinate was not returned by this activation search')
            row = deepcopy(allowed[key]); peak = finite(row.get('peak'), 'Activation peak')
            if peak == 0:
                raise ValueError('A zero activation peak does not define a direction')
            row['sign'] = 1 if peak > 0 else -1
            selected.append(row)
        return selected, {'kind': 'text_activation_observation', 'search_id': record['id'],
                          'request_sha256': digest, 'activation_artifact': deepcopy(artifact),
                          'coverage': deepcopy(output['coverage']),
                          'runtime_identity': output['runtime_identity'], 'prepared_identity': output['prepared_identity'],
                          'profile_verification': deepcopy(output.get('profile_verification'))}

    def create(self, value: dict) -> dict:
        if not isinstance(value, dict):
            raise ValueError('Concept control must be an object')
        source = value.get('source')
        required = {'name', 'source', 'query', 'neurons'} | ({'search_id'} if source == 'activation' else set())
        if set(value) != required or source not in ('lens', 'measured', 'activation'):
            raise ValueError('Invalid concept control fields/source')
        for field, maximum in (('name', 100), ('query', 100 if source == 'lens' else 160)):
            if not isinstance(value[field], str) or not 1 <= len(value[field].strip()) <= maximum:
                raise ValueError('Invalid concept control ' + field)
        value = {**value, 'query': value['query'].strip(), 'name': value['name'].strip()}
        chosen = coordinates(value['neurons'], self.app.model)
        selected, evidence = self._source(value, chosen)
        neurons = [{'layer': row['layer'], 'neuron': row['neuron'], 'sign': row['sign'],
                    'weight': row['sign'] / len(selected), 'source_evidence': row} for row in selected]
        record = {'schema': SCHEMA, 'id': uuid.uuid4().hex, 'created': time.time(), 'name': value['name'],
                  'source': source, 'query': value['query'], 'model_identity': self.app.model.identity,
                  'neurons': neurons, 'evidence': evidence, 'validation': 'association_only',
                  'normalization': 'Equal absolute neuron weights; sum(abs(weight))=1', 'semantics': SEMANTICS}
        with self.app.connect() as db:
            db.execute('INSERT INTO concept_controls VALUES(?,?,?,?,?)',
                (record['id'], record['model_identity'], record['created'], json.dumps(record, allow_nan=False), canonical_digest(record)))
        return deepcopy(record)

    def expand(self, selection: list, budget: float) -> tuple[list[dict], list[dict]]:
        budget = finite(budget, 'Concept dose budget')
        if not 0 <= budget <= .5 or not isinstance(selection, list) or len(selection) > 6:
            raise ValueError('Select at most six controls and a dose budget between0 and0.5')
        selected, seen = [], set()
        with self.app.connect() as db:
            for item in selection:
                if not isinstance(item, dict) or set(item) != {'id', 'value'}:
                    raise ValueError('Select controls using id and value only')
                key, value = identifier(item['id']), finite(item['value'], 'Concept dial')
                if key in seen or not -1 <= value <= 1:
                    raise ValueError('Concept dials must be unique and between-1 and1')
                seen.add(key)
                row = db.execute('SELECT id,model_identity,created,record,record_sha256 FROM concept_controls WHERE id=?', (key,)).fetchone()
                if row is None:
                    raise ValueError('Unknown saved concept control')
                selected.append((self._decode(row), value))
        normalization = max(1., sum(abs(value) for _, value in selected))
        combined, snapshots = {}, []
        for record, value in selected:
            scale = budget * value / normalization
            snapshots.append({**deepcopy(record), 'value': value, 'dial_l1_scale': 1 / normalization,
                              'allocated_l1_dose': abs(scale), 'snapshot_sha256': canonical_digest(record)})
            for neuron in record['neurons']:
                key = (neuron['layer'], neuron['neuron'])
                combined[key] = combined.get(key, 0.) + scale * neuron['weight']
        pushes = [{'kind': 'neuron', 'layer': l, 'neuron': n, 'dose': dose}
                  for (l, n), dose in sorted(combined.items()) if dose != 0]
        if sum(abs(p['dose']) for p in pushes) > budget + 1e-12:
            raise ValueError('Concept controls exceeded aggregate L1 budget')
        return pushes, snapshots
