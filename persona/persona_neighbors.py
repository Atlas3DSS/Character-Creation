"""Measured, nondirected Pearson co-activity among mapped MLP candidates.

The compact NPZ is produced offline from verified unsteered discovery captures.
This reader never opens raw captures, launches inference, or invents missing data.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import threading
import zipfile

import numpy as np

from persona_vectors import ModelSpec


SCHEMA = 'character-creation.coactivity.v1'
MEASUREMENT = 'final_prefill_mlp_input_last'
CORPORA = ('big_five', 'scotus', 'skippy')
MIN_SAMPLES = 8
WEAK_MAX = .2
STRONG_MIN = .6
FIELDS = {'metadata', 'layers', 'neurons', 'values', 'corpora', 'prompt_ids', 'source_hashes'}
LIMITATIONS = [
    'Signed co-activity is correlation across prompts, not wiring, direction, causation or temporal phase.',
    'Each value is one final-prefill SwiGLU activation; token-by-token trajectories are not observed.',
    'Only measured map candidates are included; unobserved neurons are not ranked.',
    'Strength thresholds are display conventions, not significance tests or multiple-comparison qualification.',
    'Corpus and task mixture can affect pooled correlations; these discovery results remain exploratory.',
]


def _signature(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


class CoactivityIndex:
    """Thread-safe content-bound cache of corpus statistics and target correlations."""

    def __init__(self, root: Path, model: ModelSpec):
        self.model = model
        self.path = Path(root) / 'results/persona_coactivity.npz'
        self.map_path = Path(root) / 'results/persona_map.json'
        self.lock = threading.RLock()
        self.signature = self.content_key = None
        self.data = None
        self.error = None
        self.statistics = {}
        self.correlations = OrderedDict()

    def _load(self):
        signature = (_signature(self.path), _signature(self.map_path))
        if signature == self.signature:
            if self.error is not None: raise ValueError(self.error)
            return self.data
        try:
            raw_map = self.map_path.read_bytes()
            map_sha = hashlib.sha256(raw_map).hexdigest()
            with self.path.open('rb') as stream:
                artifact_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
                key = (self.model.identity, map_sha, artifact_sha)
                if key == self.content_key and self.data is not None:
                    if signature != (_signature(self.path), _signature(self.map_path)):
                        raise ValueError('Co-activity publication changed while reading')
                    self.signature, self.error = signature, None
                    return self.data
                stream.seek(0)
                with zipfile.ZipFile(stream) as archive:
                    members = archive.infolist()
                    if (len(members) != len(FIELDS) or {item.filename for item in members} != {name + '.npy' for name in FIELDS}
                            or sum(item.file_size for item in members) > 512 << 20
                            or archive.getinfo('metadata.npy').file_size > 1 << 20):
                        raise ValueError('Co-activity archive fields or expanded size exceed the compact input contract')
                stream.seek(0)
                with np.load(stream, allow_pickle=False) as archive:
                    arrays = {name: archive[name] for name in FIELDS}
            data = self._validate(arrays, json.loads(raw_map), map_sha, artifact_sha)
            if signature != (_signature(self.path), _signature(self.map_path)):
                raise ValueError('Co-activity publication changed while reading')
        except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as error:
            self.signature, self.error = signature, str(error)
            self.data = self.content_key = None
            self.statistics.clear(); self.correlations.clear()
            raise ValueError(str(error)) from error
        self.signature, self.content_key, self.data, self.error = signature, key, data, None
        self.statistics.clear(); self.correlations.clear()
        return data

    def _validate(self, arrays, chart, map_sha, artifact_sha):
        raw_metadata = arrays['metadata']
        if raw_metadata.shape != () or raw_metadata.dtype.kind not in 'US':
            raise ValueError('Co-activity metadata must be a scalar JSON string')
        encoded = raw_metadata.item()
        metadata = json.loads(encoded.decode('utf-8') if isinstance(encoded, bytes) else encoded)
        if (not isinstance(metadata, dict) or metadata.get('schema') != SCHEMA
                or metadata.get('model_identity') != self.model.identity
                or metadata.get('map_sha256') != map_sha or metadata.get('split') != 'discovery'
                or metadata.get('measurement') != MEASUREMENT or metadata.get('unsteered') is not True):
            raise ValueError('Co-activity model/map/discovery/unsteered measurement binding differs')
        if not isinstance(chart, dict) or chart.get('model_identity') != self.model.identity or not isinstance(chart.get('neurons'), list):
            raise ValueError('Measured map belongs to another model or omits candidate coordinates')
        mapped = set()
        for node in chart['neurons']:
            if (not isinstance(node, dict) or type(node.get('layer')) is not int or type(node.get('neuron')) is not int
                    or not 0 <= node['layer'] < self.model.layers or not 0 <= node['neuron'] < self.model.intermediate
                    or (node['layer'], node['neuron']) in mapped):
                raise ValueError('Measured map contains invalid or duplicate coordinates')
            mapped.add((node['layer'], node['neuron']))
        layers, neurons, values = (arrays[name] for name in ('layers', 'neurons', 'values'))
        if (layers.ndim != 1 or neurons.shape != layers.shape or not len(layers)
                or layers.dtype.kind not in 'iu' or neurons.dtype.kind not in 'iu'
                or values.ndim != 2 or values.shape[1] != len(layers)
                or values.dtype.kind != 'f' or values.dtype.itemsize not in (4, 8) or not np.isfinite(values).all()):
            raise ValueError('Co-activity coordinates and finite row-by-candidate float arrays are required')
        coordinates = [(int(layer), int(neuron)) for layer, neuron in zip(layers, neurons)]
        if len(set(coordinates)) != len(coordinates) or any(coordinate not in mapped for coordinate in coordinates):
            raise ValueError('Co-activity candidates must be unique observed coordinates in the bound map')
        rows = values.shape[0]
        for name in ('corpora', 'prompt_ids', 'source_hashes'):
            array = arrays[name]
            if array.shape != (rows,) or array.dtype.kind != 'U' or any(not value for value in array.tolist()):
                raise ValueError('Co-activity row labels must be nonempty Unicode vectors matching the row count')
        if any(name not in CORPORA for name in arrays['corpora'].tolist()):
            raise ValueError('Co-activity rows contain an unsupported corpus')
        if len(set(arrays['prompt_ids'].tolist())) != rows:
            raise ValueError('Duplicate prompt IDs would overweight repeated baseline observations')
        if any(not _digest(value) for value in arrays['source_hashes'].tolist()):
            raise ValueError('Every co-activity row requires its originating artifact SHA256')
        for array in arrays.values(): array.setflags(write=False)
        return {'metadata': metadata, 'arrays': arrays, 'coordinates': coordinates,
                'indices': {coordinate: index for index, coordinate in enumerate(coordinates)},
                'map_sha256': map_sha, 'artifact_sha256': artifact_sha, 'total_samples': rows}

    def _statistics(self, corpus):
        if corpus in self.statistics: return self.statistics[corpus]
        arrays = self.data['arrays']
        selected = arrays['values'] if corpus == 'all' else arrays['values'][arrays['corpora'] == corpus]
        normalized = np.array(selected, dtype=np.float64, copy=True)
        if len(normalized):
            # Shift first for stable centering when values have a large common offset.
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                try:
                    normalized -= normalized[0].copy()
                    normalized -= normalized.mean(axis=0)
                    norms = np.linalg.norm(normalized, axis=0)
                    if not np.isfinite(norms).all(): raise ValueError('Nonfinite co-activity variance')
                    valid = norms > 0
                    normalized[:, valid] /= norms[valid]
                except FloatingPointError as error:
                    raise ValueError('Co-activity variance cannot be represented safely') from error
            if not np.isfinite(normalized).all(): raise ValueError('Nonfinite co-activity statistics')
        else:
            valid = np.zeros(normalized.shape[1], dtype=bool)
        normalized.setflags(write=False); valid.setflags(write=False)
        result = {'n': len(normalized), 'normalized': normalized, 'valid': valid}
        self.statistics[corpus] = result
        return result

    def _correlation(self, corpus, target, statistics):
        key = (self.content_key, corpus, target)
        if key not in self.correlations:
            centered = statistics['normalized']
            correlation = np.clip(centered.T @ centered[:, target], -1., 1.)
            correlation.setflags(write=False)
            self.correlations[key] = correlation
            if len(self.correlations) > 256: self.correlations.popitem(last=False)
        self.correlations.move_to_end(key)
        return self.correlations[key]

    def query(self, layer: int, neuron: int, *, corpus='all', scope='all', limit=12):
        if (type(layer) is not int or not 0 <= layer < self.model.layers or type(neuron) is not int
                or not 0 <= neuron < self.model.intermediate or corpus not in ('all', *CORPORA)
                or scope not in ('same_layer', 'cross_layer', 'all') or type(limit) is not int or not 1 <= limit <= 50):
            raise ValueError('Invalid neuron, corpus, scope or neighbor limit')
        result = {'available': False, 'status': 'missing_source', 'message': '', 'target': {'layer': layer, 'neuron': neuron},
                  'groups': {'positive': [], 'negative': [], 'weak': []}, 'source': {
                      'model_identity': self.model.identity, 'n_samples': 0, 'measurement': MEASUREMENT,
                      'split': 'discovery', 'candidates': 0, 'corpus': corpus, 'scope': scope,
                      'min_samples': MIN_SAMPLES, 'statistic': 'Pearson correlation across prompts',
                      'thresholds': {'strong_min_abs_r': STRONG_MIN, 'weak_max_abs_r': WEAK_MAX},
                      'limitations': list(LIMITATIONS)}}
        with self.lock:
            try:
                data = self._load()
            except FileNotFoundError:
                result['message'] = 'No validated compact co-activity source is available.'
                return result
            except ValueError as error:
                result.update(status='invalid_source', message=str(error))
                return result
            metadata = data['metadata']
            source = result['source']
            source.update(map_sha256=data['map_sha256'], artifact_sha256=data['artifact_sha256'],
                          total_samples=data['total_samples'], candidates=len(data['coordinates']), unsteered=True)
            for name in ('selection', 'runtime_identities', 'prepared_identities', 'corpus_sha256', 'extraction_sha256', 'created'):
                if name in metadata: source[name] = deepcopy(metadata[name])
            extra = metadata.get('limitations', [])
            if isinstance(extra, str): extra = [extra]
            if isinstance(extra, list): source['limitations'].extend(value for value in extra if isinstance(value, str))
            arrays = data['arrays']
            source['n_samples'] = data['total_samples'] if corpus == 'all' else int(np.count_nonzero(arrays['corpora'] == corpus))
            target = data['indices'].get((layer, neuron))
            if target is None:
                result.update(status='target_unobserved', message='This neuron is not observed in the compact candidate matrix.')
                return result
            if source['n_samples'] < MIN_SAMPLES:
                result.update(status='insufficient_samples', message=f'At least {MIN_SAMPLES} unique prompts are required for this corpus filter.')
                return result
            try:
                statistics = self._statistics(corpus)
                if not statistics['valid'][target]:
                    result.update(status='target_constant', message='The target has zero activation variance in this corpus filter.')
                    return result
                correlations = self._correlation(corpus, target, statistics)
            except ValueError as error:
                result.update(status='invalid_source', message=str(error))
                return result
            eligible = statistics['valid'].copy()
            eligible[target] = False
            if scope == 'same_layer': eligible &= arrays['layers'] == layer
            elif scope == 'cross_layer': eligible &= arrays['layers'] != layer
            source['eligible_candidates'] = int(np.count_nonzero(eligible))
            indices = np.flatnonzero(eligible).tolist()
            def entry(index):
                r = float(correlations[index]); coordinate = data['coordinates'][index]
                return {'layer': coordinate[0], 'neuron': coordinate[1], 'r': r, 'n': statistics['n'],
                        'strength': 'strong' if abs(r) >= STRONG_MIN else 'weak' if abs(r) <= WEAK_MAX else 'moderate'}
            def tie(index): return data['coordinates'][index]
            result['groups'] = {
                'positive': [entry(i) for i in sorted((i for i in indices if correlations[i] > WEAK_MAX), key=lambda i: (-correlations[i], tie(i)))[:limit]],
                'negative': [entry(i) for i in sorted((i for i in indices if correlations[i] < -WEAK_MAX), key=lambda i: (correlations[i], tie(i)))[:limit]],
                'weak': [entry(i) for i in sorted((i for i in indices if abs(correlations[i]) <= WEAK_MAX), key=lambda i: (abs(correlations[i]), tie(i)))[:limit]],
            }
            # The induced graph retains neighbor-to-neighbor measurements rather
            # than forcing every selection into the same center-and-spokes shape.
            selected = [target]
            for rank in range(limit):
                for group in ('positive', 'negative', 'weak'):
                    if rank < len(result['groups'][group]):
                        item = result['groups'][group][rank]
                        index = data['indices'][(item['layer'], item['neuron'])]
                        if index not in selected and len(selected) < 64: selected.append(index)
            profiles = statistics['normalized'][:, selected]
            matrix = np.clip(profiles.T @ profiles, -1., 1.)
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            order = np.argsort(eigenvalues)[::-1][:3]
            layout = eigenvectors[:, order] * np.sqrt(np.maximum(eigenvalues[order], 0))
            if layout.shape[1] < 3: layout = np.pad(layout, ((0, 0), (0, 3-layout.shape[1])))
            for axis in range(3):
                anchor = int(np.argmax(np.abs(layout[:, axis])))
                if layout[anchor, axis] < 0: layout[:, axis] *= -1
            nodes = []
            for offset, index in enumerate(selected):
                l, n = data['coordinates'][index]
                nodes.append({'id': f'L{l}N{n}', 'layer': l, 'neuron': n,
                              'r': float(correlations[index]), 'n': statistics['n'],
                              'position': layout[offset, :2].tolist(), 'position3d': layout[offset].tolist(), 'selected': index == target})
            edges = [{'source': nodes[a]['id'], 'target': nodes[b]['id'],
                      'r': float(matrix[a, b]), 'n': statistics['n']}
                     for a in range(len(nodes)) for b in range(a + 1, len(nodes))]
            result['graph'] = {'nodes': nodes, 'edges': edges, 'directed': False,
                'layout': 'Three principal coordinates of normalized activation profiles; a 3D projection of high-dimensional measurements.',
                'explained_fraction': float(np.maximum(eigenvalues[order[:2]], 0).sum() / len(nodes)),
                'explained_fraction_3d': float(np.maximum(eigenvalues[order], 0).sum() / len(nodes)),
                'selection': 'Rank-interleaved positive, negative and weak neighbors; at most 64 nodes.',
                'semantics': 'Every edge is measured on the same filtered discovery prompts; no causal or temporal direction.'}
            result.update(available=True, status='ready' if indices else 'no_eligible_neighbors',
                          message='Measured signed co-activity; strength labels are display conventions.' if indices else 'No varying candidate neighbors match this scope.')
            return result
