"""Exact all-neuron/all-prefill-token discovery with bounded chunk reductions.

One discovery pass reduces every SwiGLU activation on device. A second ordinary
prefill records only the ranked neurons' full token profiles. No tensor-sized
JSON or selected-neuron search sweeps are used.
"""
from __future__ import annotations

import re
import json
import numpy as np


def normalize_search(value: dict | None) -> dict | None:
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) - {'id', 'top_k', 'token_position'}
            or not re.fullmatch('[0-9a-f]{32}', str(value.get('id', '')))):
        raise ValueError('Search requires a unique 32-hex id, optional top_k and token_position')
    k, position = value.get('top_k', 32), value.get('token_position')
    if type(k) is not int or not 1 <= k <= 32:
        raise ValueError('Search top_k must be 1–32')
    if position is not None and (type(position) is not int or position < 0):
        raise ValueError('Search token_position must be a nonnegative integer or null')
    return {'id': value['id'], 'top_k': k, 'token_position': position}


class ActivationSearch:
    def __init__(self, engine, spec: dict, ids: list[int], request_sha: str):
        self.engine, self.spec, self.ids = engine, spec, ids
        if spec['token_position'] is not None and spec['token_position'] >= len(ids):
            raise ValueError('Search token_position exceeds encoded prompt; no truncation performed')
        shape = (engine.model.layers, engine.model.intermediate)
        self.peak = np.zeros(shape, dtype=np.float32)
        self.peak_position = np.full(shape, -1, dtype=np.int64)
        self.sum_abs = np.zeros(shape, dtype=np.float64)
        self.selected = np.zeros(shape, dtype=np.float32)
        self.positions = []
        self.token_peaks = []
        self.tokens = [{'position': p, 'id': int(t), 'text': engine.tokenizer.decode(
            [t], skip_special_tokens=False, clean_up_tokenization_spaces=False)} for p, t in enumerate(ids)]
        with engine.trace_lock:
            if spec['id'] in engine.trace_records:
                raise ValueError('Search id already exists')
            while len(engine.trace_records) >= 16:
                old = next((k for k, v in engine.trace_records.items() if v['state'] != 'running'), None)
                if old is None:
                    raise ValueError('Search storage busy')
                del engine.trace_records[old]
            self.record = {'id': spec['id'], 'state': 'running', 'kind': 'activation_search',
                           'request_sha256': request_sha, 'model_identity': engine.model.identity,
                           'runtime_identity': engine.manifest['runtime_identity'],
                           'tokens': self.tokens, 'positions_completed': 0, 'token_peaks': self.token_peaks, 'events': []}
            engine.trace_records[spec['id']] = self.record

    def configure(self) -> None:
        c = self.engine.capture
        if not hasattr(c, 'search_peak'):
            raise ValueError('Loaded graphs do not support all-neuron activation search')
        c.search_enabled.fill_(True)
        c.search_position.fill_(-1 if self.spec['token_position'] is None else self.spec['token_position'])

    def observe(self, expected: list[int]) -> None:
        c = self.engine.capture
        positions = c.positions.cpu().numpy()
        valid = positions >= 0
        if positions[valid].tolist() != expected or expected != list(range(len(self.positions), len(self.positions) + len(expected))):
            raise ValueError('Search prefill position coverage differs')
        peak = c.search_peak.cpu().numpy().copy()
        pos = c.search_peak_position.cpu().numpy().copy()
        total = c.search_sum_abs.cpu().numpy().copy()
        token_peak = c.search_token_peak.cpu().numpy()[:, valid].copy()
        token_neuron = c.search_token_neuron.cpu().numpy()[:, valid].copy()
        selected = c.search_selected.cpu().numpy().copy()
        if any(not np.isfinite(x).all() for x in (peak, total, token_peak, selected)):
            raise ValueError('Nonfinite all-neuron search capture')
        if peak.shape != self.peak.shape or np.any((pos < expected[0]) | (pos > expected[-1])):
            invalid = (pos < expected[0]) | (pos > expected[-1])
            diagnostic = {'expected_shape': list(self.peak.shape), 'actual_shape': list(peak.shape),
                'expected_positions': [expected[0], expected[-1]],
                'position_min': int(pos.min()), 'position_max': int(pos.max()),
                'invalid_per_layer': invalid.sum(axis=1).tolist(),
                'peak_nonzero_per_layer': np.count_nonzero(peak, axis=1).tolist(),
                'sum_abs_nonzero_per_layer': np.count_nonzero(total, axis=1).tolist(),
                'search_enabled': c.search_enabled.cpu().tolist(),
                'search_position': c.search_position.cpu().tolist()}
            raise ValueError('Search neuron coverage or peak positions differ: ' + json.dumps(diagnostic))
        replace = (self.peak_position < 0) | (np.abs(peak) > np.abs(self.peak))
        self.peak[replace], self.peak_position[replace] = peak[replace], pos[replace]
        self.sum_abs += total
        if self.spec['token_position'] in expected:
            self.selected[:] = selected
        layer = np.abs(token_peak).argmax(axis=0)
        rows = [{'position': p, 'layer': int(layer[i]), 'neuron': int(token_neuron[layer[i], i]),
                 'peak': float(token_peak[layer[i], i])} for i, p in enumerate(expected)]
        with self.engine.trace_lock:
            self.positions.extend(expected)
            self.token_peaks.extend(rows)
            self.record["events"].append({"kind": "activation_search", "positions": expected, "token_peaks": rows})
            self.record['positions_completed'] = len(self.positions)

    def arrays(self) -> dict:
        if self.positions != list(range(len(self.ids))):
            raise ValueError('Search did not cover every prompt position')
        return {'search_peak': self.peak, 'search_peak_position': self.peak_position,
                'search_sum_abs': self.sum_abs, 'search_selected_position_values': self.selected,
                'search_selected_position': np.int64(-1 if self.spec['token_position'] is None else self.spec['token_position']),
                'search_positions': np.asarray(self.positions, dtype=np.int64)}

    def result(self) -> dict:
        self.arrays()  # assert complete coverage before publishing any ranking
        def rank(values: np.ndarray) -> list[dict]:
            # Stable flattened order breaks equal absolute-value ties by layer/neuron.
            order = np.argsort(-np.abs(values).ravel(), kind='stable')[:self.spec['top_k']]
            rows = []
            for i in order:
                l, n = np.unravel_index(i, values.shape)
                rows.append({'layer': int(l), 'neuron': int(n), 'peak': float(self.peak[l, n]),
                    'peak_abs': float(abs(self.peak[l, n])), 'peak_position': int(self.peak_position[l, n]),
                    'mean_abs': float(self.sum_abs[l, n] / len(self.ids)),
                    **({'activation': float(values[l, n])} if self.spec['token_position'] is not None and values is self.selected else {})})
            return rows
        return {'id': self.spec['id'], 'tokens': self.tokens, 'candidates': rank(self.peak),
                'token_peaks': self.token_peaks, 'token_position': self.spec['token_position'],
                'token_candidates': rank(self.selected) if self.spec['token_position'] is not None else [],
                'coverage': {'layers': self.peak.shape[0], 'neurons_per_layer': self.peak.shape[1],
                    'tokens': len(self.ids), 'neuron_token_values': int(self.peak.size * len(self.ids)),
                    'all_neurons': True, 'all_prompt_positions': True, 'includes_template_tokens': True},
                'ranking': 'descending absolute peak across all prompt positions; ties by layer then neuron',
                'semantics': 'Continuous SwiGLU activation after pre-down steering, before down projection. Every encoded prompt position including template/special tokens; no firing threshold. Mean absolute values use FP32 chunk sums with FP64 accumulation. Profiles use a second state-restored prefill.'}

    def finish(self, state: str, error: str | None = None) -> None:
        with self.engine.trace_lock:
            self.record.update(state=state, error=error)


def attach_profiles(result: dict, trace: dict) -> None:
    """Verify the second pass at every selected neuron's discovery peak."""
    positions = [p for e in trace['events'] for p in e['positions']]
    if positions != list(range(result['coverage']['tokens'])):
        raise ValueError('Search profile pass did not cover every prompt token')
    values = np.asarray([v for e in trace['events'] for v in e['values']], dtype=np.float32)
    for i, row in enumerate(result['candidates']):
        if trace['neurons'][i] != {'layer': row['layer'], 'neuron': row['neuron']}:
            raise ValueError('Search profile neuron coordinates differ')
        profile = values[:, i]
        peak_position = int(np.abs(profile).argmax())
        if (peak_position != row['peak_position']
                or profile[peak_position].tobytes() != np.float32(row['peak']).tobytes()):
            raise ValueError('Search profile replay peak is not bit-exact with discovery')
        if not np.isclose(np.abs(profile.astype(np.float64)).mean(), row['mean_abs'], rtol=1e-5, atol=1e-6):
            raise ValueError('Search profile replay mean differs from discovery reduction')
        row['values'] = profile.tolist()
    result['profile_verification'] = {'peak_bit_exact': True, 'mean_abs_rtol': 1e-5, 'mean_abs_atol': 1e-6, 'all_positions': True, 'passes': 2}
