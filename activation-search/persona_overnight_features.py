"""Discovery-only, lexical-controlled associations from immutable prefill maps."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3

import numpy as np

ARMS = ('semantic_positive', 'semantic_neutral', 'lexical_mention_control')
FRAMES = ('feeling', 'expression', 'regulation', 'recognition')
IDENTITIES = ('model_identity', 'runtime_identity', 'prepared_identity')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative_difference(positive, reference):
    return (positive - reference) / (np.abs(positive) + np.abs(reference) + 1e-6)


def contrasts(means, peaks):
    """Arrays have frame, arm, layer, neuron axes; no heldout fitting."""
    semantic = relative_difference(means[:, 0], means[:, 1])
    lexical = relative_difference(means[:, 2], means[:, 1])
    signed_peak = relative_difference(peaks[:, 0], peaks[:, 1])
    average = semantic.mean(axis=0)
    direction = np.sign(average)
    agreement = (np.sign(semantic) == direction).sum(axis=0)
    lexical_magnitude = np.abs(lexical).mean(axis=0)
    activity = np.max(means, axis=(0, 1))
    # A descriptive ranking, not a p-value or neuron meaning. Small denominators
    # are gated explicitly; lexical controls and frame disagreement are retained.
    score = np.maximum(0, np.abs(average) - lexical_magnitude) * (agreement / len(FRAMES))
    score[(agreement < 3) | (activity < 1e-3)] = 0
    return {'semantic_relative_mean': average.astype(np.float32),
            'lexical_absolute_mean': lexical_magnitude.astype(np.float32),
            'signed_peak_relative_mean': signed_peak.mean(axis=0).astype(np.float32),
            'frame_agreement': agreement.astype(np.uint8), 'ranking_score': score.astype(np.float32)}, semantic, lexical, signed_peak


def analyze_feature(job):
    feature, rows, root, binding = job
    root = Path(root)
    slots = {(r['frame'], r['arm']): r for r in rows}
    expected = {(f, a) for f in FRAMES for a in ARMS}
    if len(rows) != 12 or set(slots) != expected:
        return {'feature': feature, 'state': 'missing_evidence', 'complete_rows': len(rows),
                'missing': [list(x) for x in sorted(expected - set(slots))], 'candidates': []}
    identities = {tuple(r[k] for k in IDENTITIES) for r in rows}
    if len(identities) != 1:
        raise ValueError('Feature source runtime identities differ')
    means, peaks, evidence = [], [], []
    for frame in FRAMES:
        frame_means, frame_peaks = [], []
        family = {slots[(frame, arm)]['pair_family'] for arm in ARMS}
        if len(family) != 1:
            raise ValueError('Unmatched contrast family')
        for arm in ARMS:
            record = slots[(frame, arm)]
            path = Path(record['artifact_path']).resolve()
            if path.parent != root.resolve() or file_hash(path) != record['artifact_sha256']:
                raise ValueError('Raw artifact binding changed')
            with np.load(path, allow_pickle=False) as data:
                if any(str(data[k]) != record[k] for k in IDENTITIES) or str(data['request_sha256']) != record['request_sha256']:
                    raise ValueError('NPZ model/request binding changed')
                if str(data['stage']) != 'all_prefill_search':
                    raise ValueError('Not an all-position capture')
                mean = data['search_sum_abs'].astype(np.float32) / record['tokens']
                peak = data['search_peak'].astype(np.float32)
                if mean.shape != (64, 17408) or peak.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(peak).all():
                    raise ValueError('Invalid all-neuron feature input')
                frame_means.append(mean); frame_peaks.append(peak)
            evidence.append(record)
        means.append(frame_means); peaks.append(frame_peaks)
    arrays, semantic, lexical, signed_peak = contrasts(np.asarray(means), np.asarray(peaks))
    candidates = []
    score = arrays['ranking_score']
    order = np.argsort(-score.ravel(), kind='stable')[:32]
    for flat in order:
        l, n = np.unravel_index(flat, score.shape)
        if score[l, n] <= 0:
            continue
        candidates.append({'layer': int(l), 'neuron': int(n), 'score': float(score[l, n]),
            'direction': 'greater_mean_absolute_activation' if arrays['semantic_relative_mean'][l, n] > 0 else 'lower_mean_absolute_activation',
            'semantic_relative_mean': float(arrays['semantic_relative_mean'][l, n]),
            'lexical_absolute_mean': float(arrays['lexical_absolute_mean'][l, n]),
            'signed_peak_relative_mean': float(arrays['signed_peak_relative_mean'][l, n]),
            'frame_agreement': int(arrays['frame_agreement'][l, n]),
            'frames': [{'frame': frame, 'semantic': float(semantic[i, l, n]), 'lexical': float(lexical[i, l, n]),
                        'signed_peak': float(signed_peak[i, l, n])} for i, frame in enumerate(FRAMES)]})
    # Report strong semantic-looking coordinates suppressed by lexical controls.
    confounded = np.abs(arrays['semantic_relative_mean']) - arrays['ranking_score']
    counterevidence = []
    for flat in np.argsort(-confounded.ravel(), kind='stable')[:8]:
        l, n = np.unravel_index(flat, score.shape)
        counterevidence.append({'layer': int(l), 'neuron': int(n),
            'semantic_relative_mean': float(arrays['semantic_relative_mean'][l, n]),
            'lexical_absolute_mean': float(arrays['lexical_absolute_mean'][l, n]),
            'frame_agreement': int(arrays['frame_agreement'][l, n]),
            'reason': 'Large apparent contrast discounted by lexical response and/or inconsistent frames'})
    identity = digest({'binding': binding, 'feature': feature, 'evidence': evidence})[:32]
    path = root / ('feature-' + identity + '.npz')
    if path.exists():
        with np.load(path, allow_pickle=False) as prior:
            if any(not np.array_equal(prior[k], value) for k, value in arrays.items()):
                raise ValueError('Immutable derived artifact differs')
    else:
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays, feature=feature, input_binding=digest(binding))
    return {'feature': feature, 'state': 'complete', 'evidence': evidence, 'candidates': candidates,
            'counterevidence': counterevidence, 'artifact': {'path': str(path), 'sha256': file_hash(path)},
            'eligible_neurons': int(np.count_nonzero(score)), 'all_neurons_ranked': 64*17408,
            'claim': 'Prompt-conditioned association across four authored frames; not causality, emotion recognition, or a validated neuron concept.'}


def collect(database):
    groups = {}
    with sqlite3.connect(database) as db:
        binding = dict(db.execute('SELECT key,value FROM metadata'))
        rows = db.execute("SELECT id,state,row,request,response FROM tasks WHERE json_extract(row,'$.tier')='authored_emotion_contrast'")
        for identity, state, encoded, request, response in rows:
            source = json.loads(encoded)
            if source['split'] != 'discovery':
                continue
            provenance = source['provenance']
            group = groups.setdefault(provenance['feature'], [])
            if state != 'complete':
                continue
            response, request = json.loads(response), json.loads(request)
            artifact = response['activation_artifact']
            if artifact['request_sha256'] != digest(request):
                raise ValueError('Completed request binding differs')
            group.append({'id': identity, 'row_sha256': digest(source), 'source_sha256': source['source_sha256'],
                          'frame': provenance['frame'], 'arm': provenance['arm'], 'pair_family': provenance['pair_family'],
                          'artifact_path': artifact['path'], 'artifact_sha256': artifact['sha256'],
                          'request_sha256': digest(request), 'tokens': response['activation_search']['coverage']['tokens'],
                          **{k: response[k] for k in IDENTITIES}})
    # Renewal state is operational, not statistical input identity.
    binding.pop('renewed_endpoints', None)
    binding['feature_script_sha256'] = file_hash(__file__)
    return groups, binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'v100':
        raise ValueError('Derived analysis is V100-CPU only')
    groups, binding = collect(args.database)
    jobs = [(feature, rows, str(args.root), binding) for feature, rows in sorted(groups.items())]
    workers = min(len(jobs), max(1, len(os.sched_getaffinity(0))))
    if not jobs:
        raise ValueError('No discovery feature groups')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        features = list(pool.map(analyze_feature, jobs))
    edges = []
    sets = [{(r['layer'], r['neuron']) for r in f['candidates']} for f in features]
    for i, first in enumerate(features):
        for j in range(i + 1, len(features)):
            shared = sets[i] & sets[j]
            if shared:
                edges.append({'source': first['feature'], 'target': features[j]['feature'], 'shared_coordinates': len(shared),
                              'jaccard': len(shared) / len(sets[i] | sets[j])})
    result = {'schema': 'character-creation.discovery-feature-map.v1', 'binding': binding,
              'state': 'complete' if all(f['state'] == 'complete' for f in features) else 'partial',
              'method': {'statistic': '(positive-neutral)/(abs(positive)+abs(neutral)+1e-6), per frame',
                         'rank': 'max(0, abs(mean semantic contrast)-mean absolute lexical contrast)*frame sign agreement/4',
                         'minimum_frame_agreement': 3, 'minimum_mean_absolute_activity': 1e-3,
                         'frame_count': 4, 'heldout_used': False, 'template_tokens_included': True,
                         'limitation': 'Unequal-length authored texts and template tokens can confound reductions; lexical controls are partial. No significance tests, causal claims or behavioral labels.'},
              'features': features, 'edges': sorted(edges, key=lambda e: -e['jaccard'])[:256],
              'edge_semantics': 'Overlap of top ranked coordinates, not measured coactivation or neural connectivity.'}
    target = args.root / 'overnight_feature_map.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n'); temporary.replace(target)
    print(json.dumps({'state': result['state'], 'features': len(features), 'workers': workers, 'path': str(target), 'sha256': file_hash(target)}), flush=True)


if __name__ == '__main__':
    main()
