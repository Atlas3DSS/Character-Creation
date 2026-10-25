"""Extract private candidate activations from verified unsteered discovery coverage.

Only original prompts with baseline prompt conditioning are selected. Mutations,
heldout rows and intervention arms never enter this exploratory association view.
The output contains numeric observations and hashes, never corpus text.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

from persona_manager import verify_diagnostic_artifacts, response_complete
from persona_vectors import ModelSpec, canonical_digest


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def selected(job):
    prompt = job.get('prompt', {})
    return (job.get('operator_coverage') is True and job.get('arm') == 'baseline'
            and job.get('pushes') == [] and job.get('measurement') == 'prefill'
            and prompt.get('split') == 'discovery'
            and prompt.get('mutation', {}).get('kind') == 'original'
            and prompt.get('labels', {}).get('profile', 'baseline') == 'baseline')


def records(database, model):
    rows, groups, ids = [], set(), set()
    with sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True) as db:
        identity = json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
        if identity['model_identity'] != model.identity:
            raise ValueError('Manager database belongs to another model')
        query = """SELECT j.body,a.result FROM jobs j JOIN attempts a ON j.id=a.job_id
          WHERE j.state='complete' AND a.state='complete'
          AND json_extract(j.body,'$.operator_coverage')=1
          AND json_extract(j.body,'$.prompt.split')='discovery'
          ORDER BY j.id,a.number"""
        for job_json, result_json in db.execute(query):
            job, result = json.loads(job_json), json.loads(result_json)
            if not selected(job):
                continue
            prompt = job['prompt']
            job_without_id = {key: value for key, value in job.items() if key != 'id'}
            if canonical_digest(job_without_id) != job['id'] or job['model_identity'] != model.identity:
                raise ValueError('Coverage job identity changed')
            group = (prompt['corpus'], prompt.get('pair_id', prompt['id']))
            if prompt['id'] in ids or group in groups:
                raise ValueError('Duplicate original prompt or source group in coverage')
            if not result.get('complete_observation') or not result.get('complete') or result.get('error'):
                raise ValueError('Coverage result is not a complete observation')
            if result.get('provenance', {}).get('pushes') != []:
                raise ValueError('Coverage result contains an intervention')
            if not response_complete(result['response'], diagnostic=True, measurement='prefill'):
                raise ValueError('Coverage response diagnostics are incomplete')
            request = result['request']
            expected_messages = ([{'role': 'system', 'content': prompt['system']}] if prompt.get('system') else [])
            expected_messages.append({'role': 'user', 'content': prompt['prompt']})
            if (request.get('messages') != expected_messages or request.get('observation_only') is not True
                    or 'steering' in request or request.get('max_tokens') != 1
                    or request.get('metadata') != {'job_id': job['id'], 'split': 'discovery', 'arm': 'baseline'}):
                raise ValueError('Request does not match the selected original prompt')
            artifact = result['activation_artifact']
            if artifact != result['response']['activation_artifact']:
                raise ValueError('Stored activation pointer differs from worker response')
            for key in ('model_identity', 'prepared_identity', 'runtime_identity'):
                if artifact.get(key) != identity.get(key):
                    raise ValueError('Coverage mixes incompatible model or runtime identities')
            ids.add(prompt['id']); groups.add(group)
            rows.append((job, result))
    if not rows:
        raise ValueError('No eligible completed original discovery coverage')
    return rows, identity


def read_candidate_values(row, model, layers, neurons, results_root):
    job, result = row
    artifact = result['activation_artifact']
    for pointer in (artifact, result['response']['provenance_artifact']):
        if Path(pointer['path']).resolve().parent != results_root.resolve():
            raise ValueError('Artifact must be a direct child of the private results directory')
    endpoint = {**result['endpoint'], 'model_identity': model.identity}
    verify_diagnostic_artifacts(result['response'], endpoint, result['request'])
    path = Path(artifact['path'])
    if file_hash(path) != artifact['sha256']:
        raise ValueError('Activation artifact digest changed')
    with np.load(path, allow_pickle=False) as data:
        for key in ('model_identity', 'prepared_identity', 'runtime_identity', 'request_sha256', 'operation', 'stage'):
            if str(data[key]) != artifact[key]:
                raise ValueError('Activation array provenance changed: ' + key)
        if artifact['stage'] != 'final_prefill' or artifact['operation'] != 'prefill_observation':
            raise ValueError('Only final-prefill observations are supported')
        full = data['mlp_input_last']
        if full.shape != (model.layers, model.intermediate) or not np.isfinite(full).all():
            raise ValueError('Unexpected or nonfinite activation matrix')
        values = np.asarray(full[layers, neurons], dtype=np.float32)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--workers', type=int, default=max(1, len(os.sched_getaffinity(0)) - 1))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stream', action='store_true', help='Write the numeric NPZ to stdout; progress goes to stderr')
    args = parser.parse_args()
    if args.workers < 1 or (args.stream and args.output):
        parser.error('Choose positive workers and at most one output mode')
    root, started = args.root.resolve(), time.time()
    model = ModelSpec.load(root / 'data/persona_model.json')
    map_path = root / 'results/persona_map.json'
    map_hash = file_hash(map_path)
    chart = json.loads(map_path.read_text())
    if chart['model_identity'] != model.identity:
        raise ValueError('Measured map belongs to another model')
    coordinates = sorted((row['layer'], row['neuron']) for row in chart['neurons'])
    if len(set(coordinates)) != len(coordinates) or not coordinates:
        raise ValueError('Measured map coordinates are empty or duplicated')
    layers, neurons = np.array(coordinates, dtype=np.int64).T
    if np.any(layers < 0) or np.any(layers >= model.layers) or np.any(neurons < 0) or np.any(neurons >= model.intermediate):
        raise ValueError('Measured map coordinates exceed this model')
    rows, identity = records(root / 'results/persona_manager.sqlite', model)
    corpus_hashes = {job['corpus_sha256'] for job, _ in rows}
    if (len(corpus_hashes) != 1
            or file_hash(root / 'data/persona_manager_corpus.jsonl') != next(iter(corpus_hashes))):
        raise ValueError('Coverage does not match the published corpus')
    values = np.empty((len(rows), len(coordinates)), dtype=np.float32)
    counts = Counter(job['prompt']['corpus'] for job, _ in rows)
    print(json.dumps({'selected_rows': len(rows), 'candidates': len(coordinates), 'corpora': counts,
                      'workers': args.workers}), file=sys.stderr, flush=True)
    def extract(row):
        return read_candidate_values(row, model, layers, neurons, root / 'results')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, vector in enumerate(pool.map(extract, rows)):
            values[index] = vector
            if (index + 1) % 250 == 0:
                print(json.dumps({'extracted': index + 1, 'total': len(rows), 'elapsed_s': round(time.time()-started, 1)}), file=sys.stderr, flush=True)
    if file_hash(map_path) != map_hash:
        raise ValueError('Measured map changed during extraction')
    prompt_ids = [job['prompt']['id'] for job, _ in rows]
    source_hashes = [result['activation_artifact']['sha256'] for _, result in rows]
    metadata = {'schema': 'character-creation.coactivity.v1', 'model_identity': model.identity,
        'map_sha256': map_hash, 'split': 'discovery', 'measurement': 'final_prefill_mlp_input_last',
        'unsteered': True, 'runtime_identities': [identity['runtime_identity']],
        'prepared_identities': [identity['prepared_identity']], 'corpus_sha256': next(iter(corpus_hashes)),
        'extraction_sha256': file_hash(__file__), 'created': time.time(),
        'selection': 'Completed operator coverage; original discovery prompts only; baseline prompt profile; one row per source group.',
        'source_index_sha256': canonical_digest(list(zip(prompt_ids, source_hashes))),
        'corpus_counts': dict(counts), 'limitations': [
            'Association among measured candidates, not a complete connectome or proof of direct wiring.',
            'One final prompt-token sample per original prompt; no token trajectories, temporal phase, or causal direction.',
            'Pooled results are dominated by book passages; compare corpus-specific views.',
            'Big Five and SCOTUS discovery cohorts are small; strength labels are descriptive, not significance tests.',
            'Heldout prompts, mutations, personality prompt profiles and activation interventions are excluded.']}
    arrays = dict(metadata=np.array(json.dumps(metadata, sort_keys=True)), layers=layers, neurons=neurons,
                  values=values, corpora=np.array([job['prompt']['corpus'] for job, _ in rows]),
                  prompt_ids=np.array(prompt_ids), source_hashes=np.array(source_hashes))
    if args.stream:
        np.savez_compressed(sys.stdout.buffer, **arrays)
    else:
        destination = args.output or root / 'results/persona_coactivity.npz'
        with destination.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
    print(json.dumps({'complete': True, 'rows': len(rows), 'elapsed_s': round(time.time()-started, 1)}), file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
