"""Concept traces: where each qualified concept lives, layer by layer, over all 1,114,112 MLP neurons.

Label qualification v2 captures the final-prompt-token SwiGLU product of every verified scene (concept holds vs hard
negative). Its analysis keeps one neuron per concept; this keeps the whole picture. For each concept, the standardized
contrast (positive mean - negative mean) / pooled SD is computed for every neuron on the discovery scenes, the top
neurons of each layer are chosen on discovery only, and the same neurons are re-measured on the held-out scenes, so
every dot in a trace says whether it replicated. Streaming sums: no plane matrix is ever held in memory.

Runs where the captures live (the GPU host); writes results/neuron_atlas/concept_trace.json for the atlas page.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import time

LAYERS, WIDTH = 64, 17408
TOP = 24            # neurons kept per layer per concept (discovery selection)
STRONG = 1.0        # |discovery contrast| counted as a strong neuron
REPLICATES = 0.5    # held-out contrast of the same sign and at least this size = replicated
MIN_SCENES = 12     # per class in each split
DB = 'results/label_qualify/qualify.sqlite'
OUT = 'results/neuron_atlas/concept_trace.json'


def scenes(root):
    db = sqlite3.connect(f'file:{Path(root) / DB}?mode=ro', uri=True)
    try:
        concepts = {name: {'state': state, 'result': json.loads(result) if result else {}}
                    for name, state, result in db.execute('SELECT name,state,result FROM concepts')}
        rows = {}
        for concept, label, split, artifact in db.execute(
                "SELECT concept,label,split,artifact FROM scenes WHERE verified=1 AND split IN ('discovery','heldout') AND artifact IS NOT NULL"):
            rows.setdefault(concept, []).append((int(label), split, artifact))
    finally:
        db.close()
    return concepts, rows


def contrast(acc):
    import numpy as np
    (n1, s1, q1), (n0, s0, q0) = acc[1], acc[0]
    m1, m0 = s1 / n1, s0 / n0
    pooled = np.sqrt((np.maximum(q1 / n1 - m1 ** 2, 0) + np.maximum(q0 / n0 - m0 ** 2, 0)) / 2) + 1e-6
    return ((m1 - m0) / pooled).astype(np.float32)


def trace(args):
    import numpy as np
    name, rows = args
    acc = {split: {label: [0, np.zeros(LAYERS * WIDTH), np.zeros(LAYERS * WIDTH)] for label in (0, 1)}
           for split in ('discovery', 'heldout')}
    for label, split, path in rows:
        try:
            with np.load(path, allow_pickle=False) as data:
                plane = data['mlp_input_last'].reshape(-1).astype(np.float64)
        except (OSError, KeyError, ValueError):
            continue
        if plane.size != LAYERS * WIDTH or not np.isfinite(plane).all():
            continue
        slot = acc[split][label]; slot[0] += 1; slot[1] += plane; slot[2] += plane * plane
    counts = {s: {l: acc[s][l][0] for l in (0, 1)} for s in acc}
    if min(counts['discovery'].values()) < MIN_SCENES or min(counts['heldout'].values()) < MIN_SCENES:
        return name, {'skipped': 'too few captured scenes', 'scenes': counts}
    disc = contrast(acc['discovery']).reshape(LAYERS, WIDTH)
    held = contrast(acc['heldout']).reshape(LAYERS, WIDTH)
    layers = []
    for L in range(LAYERS):
        top = np.argsort(-np.abs(disc[L]))[:TOP]
        d, h = disc[L, top], held[L, top]
        replicated = (np.sign(d) == np.sign(h)) & (np.abs(h) >= REPLICATES)
        layers.append({'max': round(float(np.abs(disc[L]).max()), 3), 'strong': int((np.abs(disc[L]) >= STRONG).sum()),
                       'replicated': round(float(replicated.mean()), 3),
                       'neurons': [[int(n), round(float(a), 3), round(float(b), 3)] for n, a, b in zip(top, d, h)]})
    return name, {'scenes': counts, 'layers': layers}


def build(root, workers):
    started = time.time()
    concepts, rows = scenes(root)
    names = sorted(n for n in rows if n in concepts)
    out = {}
    with ProcessPoolExecutor(min(workers, len(names) or 1)) as pool:
        for name, result in pool.map(trace, [(n, rows[n]) for n in names]):
            res = concepts[name]['result']
            result.update(state=concepts[name]['state'], passed=bool(res.get('passed')),
                          qualified=[res['layer'], res['neuron'], res.get('direction')] if res.get('passed') else None)
            out[name] = result
            print(f'{name}: {"skipped" if "skipped" in result else "traced"}', flush=True)
    value = {'schema': 'character-creation.concept-trace.v1', 'built_at': time.time(), 'seconds': round(time.time() - started),
             'measurement': 'final_prompt_token_swiglu_product', 'top_per_layer': TOP, 'strong': STRONG, 'replicates': REPLICATES,
             'method': 'Standardized contrast (concept scenes vs hard negatives) per neuron; top neurons per layer chosen on '
                       'discovery scenes only, then re-measured on held-out scenes.',
             'concepts': out}
    path = Path(root) / OUT; path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, separators=(',', ':'))); tmp.replace(path)
    return {'concepts': len(out), 'traced': sum('layers' in v for v in out.values()), 'seconds': value['seconds'], 'path': str(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = parser.parse_args()
    print(json.dumps(build(a.root, a.workers)))


if __name__ == '__main__':
    main()
