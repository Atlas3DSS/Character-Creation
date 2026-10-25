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
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import zipfile

import numpy as np

from persona_manager import load_corpus, verify_diagnostic_artifacts, response_complete
from persona_vectors import ModelSpec, canonical_digest

MATRIX_FIELDS = {'metadata', 'layers', 'neurons', 'values', 'corpora', 'prompt_ids', 'source_hashes'}
MAX_MATRIX_BYTES = 512 << 20


def matrix_snapshot(source, *, maximum=MAX_MATRIX_BYTES):
    """Read a bounded NPZ from bytes, stdin, or an existing file without pickle."""
    if source is None: return None
    if isinstance(source, dict): return source
    if isinstance(source, (str, Path)):
        with Path(source).open('rb') as stream: return matrix_snapshot(stream, maximum=maximum)
    if isinstance(source, bytes):
        raw = source
    else:
        chunks, size = [], 0
        while chunk := source.read(min(1 << 20, maximum + 1 - size)):
            chunks.append(chunk); size += len(chunk)
            if size > maximum: raise ValueError('Previous matrix input exceeds the bounded NPZ size')
        raw = b''.join(chunks)
    if len(raw) > maximum: raise ValueError('Previous matrix input exceeds the bounded NPZ size')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        items = archive.infolist()
        if (len(items) != len(MATRIX_FIELDS) or {entry.filename for entry in items} != {name + '.npy' for name in MATRIX_FIELDS}
                or sum(entry.file_size for entry in items) > MAX_MATRIX_BYTES
                or archive.getinfo('metadata.npy').file_size > 1 << 20):
            raise ValueError('Matrix fields or expanded size exceed the compact contract')
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in MATRIX_FIELDS}
    return {'arrays': arrays, 'sha256': hashlib.sha256(raw).hexdigest()}


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


def published_corpus(root):
    path = Path(root) / 'data/persona_manager_corpus.jsonl'
    manifest_path = path.with_name('persona_manager_corpus_manifest.json')
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    digest = file_hash(path)
    if manifest.get('status') != 'complete' or manifest.get('output_sha256') != digest:
        raise ValueError('Corpus publication is incomplete or its digest changed')
    corpus = load_corpus(path)
    if manifest_raw != manifest_path.read_bytes() or file_hash(path) != digest:
        raise ValueError('Corpus publication changed while reading')
    return corpus, digest


def bind_prompt(prompt, corpus):
    current = corpus.get(prompt.get('id'))
    if current is None or canonical_digest(current) != canonical_digest(prompt):
        raise ValueError('Captured prompt record differs from the current immutable corpus row')


def prompt_key(prompt, identity):
    normalized = canonical_digest([' '.join(prompt.get('system', '').split()), ' '.join(prompt['prompt'].split())])
    return canonical_digest([normalized, *[identity[key] for key in ('model_identity', 'prepared_identity', 'runtime_identity')]])


def records(database, model, corpus=None):
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
            if corpus is not None: bind_prompt(prompt, corpus)
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


def private_pointer(pointer, results_root):
    if not isinstance(pointer, dict) or Path(pointer['path']).resolve().parent != Path(results_root).resolve():
        raise ValueError('Artifact must be a direct child of the private results directory')
    if file_hash(pointer['path']) != pointer.get('sha256'):
        raise ValueError('Artifact digest changed')


def registry_records(paths, model, corpus, identity, results_root):
    """Rebuild trusted extraction inputs from explicit registry/report pointers."""
    rows, sources = [], []
    for path in paths:
        path = Path(path)
        digest = file_hash(path)
        count = 0
        with path.open() as stream:
            for line in stream:
                if not line.strip(): continue
                entry = json.loads(line)
                prompt = corpus.get(entry.get('prompt_id'))
                if prompt is None or canonical_digest(prompt) != entry.get('source_row_sha256'):
                    raise ValueError('Registry source row differs from the immutable published record')
                if (entry.get('source_sha256') != prompt.get('source_sha256')
                        or entry.get('mutation') != prompt.get('mutation') or entry.get('labels') != prompt.get('labels', {})
                        or entry.get('corpus') != prompt['corpus'] or entry.get('pair_id') != prompt.get('pair_id', prompt['id'])):
                    raise ValueError('Registry source metadata differs from its published record')
                if entry.get('coactivity_original_eligible') is not True: continue
                if (entry.get('original') is not True or entry.get('candidate_registry_only') is not True
                        or entry.get('heldout_used') is not False or entry.get('arm') != 'baseline'
                        or prompt['split'] != 'discovery' or prompt.get('mutation', {}).get('kind') not in (None, 'original')
                        or prompt.get('labels', {}).get('profile') not in (None, 'baseline')
                        or entry.get('source_group') != {'corpus': prompt['corpus'], 'pair_id': prompt.get('pair_id', prompt['id']), 'split': 'discovery'}):
                    raise ValueError('Registry is not an original unsteered discovery baseline')
                if entry.get('exact_prompt_sha256') != canonical_digest([prompt.get('system', ''), prompt['prompt']]):
                    raise ValueError('Registry prompt hash differs from the published record')
                if entry.get('prompt_sha256') != canonical_digest([' '.join(prompt.get('system', '').split()), ' '.join(prompt['prompt'].split())]):
                    raise ValueError('Registry normalized prompt hash differs')
                if any(entry.get(key) != identity.get(key) for key in ('model_identity', 'prepared_identity', 'runtime_identity')):
                    raise ValueError('Registry model/prepared/runtime identity differs from coverage')
                if entry.get('identity') != prompt_key(prompt, identity):
                    raise ValueError('Registry deduplication identity differs')
                if not isinstance(entry.get('corpus_sha256'), str) or len(entry['corpus_sha256']) != 64:
                    raise ValueError('Registry omitted its captured corpus publication hash')
                artifact, pointer = entry['artifact'], entry['provenance_artifact']
                private_pointer(artifact, results_root); private_pointer(pointer, results_root)
                report = json.loads(Path(pointer['path']).read_text())
                request, response = report['request'], {**report['response'], 'provenance_artifact': pointer}
                measurement = entry.get('measurement')
                if measurement not in ('prefill', 'rollout') or not response_complete(response, diagnostic=True, measurement=measurement):
                    raise ValueError('Registry response is not a complete diagnostic measurement')
                expected = ([{'role': 'system', 'content': prompt['system']}] if prompt.get('system') else [])
                expected.append({'role': 'user', 'content': prompt['prompt']})
                if (request.get('messages') != expected or 'steering' in request
                        or request.get('metadata') != {'job_id': entry['job_id'], 'split': 'discovery', 'arm': 'baseline'}
                        or (request.get('observation_only') is True) != (measurement == 'prefill')
                        or (measurement == 'prefill' and request.get('max_tokens') != 1)):
                    raise ValueError('Registry request is not its exact unsteered original prompt')
                if artifact != response.get('activation_artifact') or artifact != report.get('activation_artifact'):
                    raise ValueError('Registry activation pointer differs from the complete response')
                endpoint = {key: identity[key] for key in ('model_identity', 'prepared_identity', 'runtime_identity')}
                verify_diagnostic_artifacts(response, endpoint, request)
                rows.append(({'prompt': prompt, 'corpus_sha256': entry['corpus_sha256'], 'measurement': measurement},
                    {'activation_artifact': artifact, 'response': response, 'request': request, 'endpoint': endpoint}))
                count += 1
        if file_hash(path) != digest: raise ValueError('Registry changed during extraction')
        sources.append({'name': path.name, 'sha256': digest, 'eligible_records': count})
    return rows, sources


def unique_rows(rows, identity, preserved_ids=frozenset(), preserved_sources=None):
    selected_rows, seen, ids = [], set(), set()
    preserved_sources = preserved_sources or {}
    # Published representatives win on later refreshes; the initial dedup uses
    # the smallest prompt ID, independent of query ordering or sampling seed.
    for row in sorted(rows, key=lambda row: (row[0]['prompt']['id'] not in preserved_ids,
            row[1]['activation_artifact']['sha256'] != preserved_sources.get(row[0]['prompt']['id'], row[1]['activation_artifact']['sha256']),
            row[0]['prompt']['id'], row[1]['activation_artifact']['sha256'])):
        prompt = row[0]['prompt']; key = prompt_key(prompt, identity)
        if prompt['id'] in ids:
            if key in seen: continue
            raise ValueError('One prompt ID has incompatible deduplication identities')
        if key in seen: continue
        seen.add(key); ids.add(prompt['id']); selected_rows.append(row)
    return selected_rows, len(rows) - len(selected_rows)


def previous_records(snapshot, corpus, model, root, chart, map_hash, layers, neurons):
    """Reuse immutable compact observations; content IDs bind current source rows."""
    from persona_neighbors import CoactivityIndex
    if snapshot is None: raise ValueError('Registry-only refresh requires a previous matrix')
    arrays = snapshot['arrays']
    validated = CoactivityIndex(root, model)._validate(arrays, chart, map_hash, snapshot['sha256'])
    metadata = validated['metadata']
    if any(len(metadata.get(name, [])) != 1 for name in ('runtime_identities', 'prepared_identities')):
        raise ValueError('Registry-only refresh requires one exact prepared/runtime identity')
    if not np.array_equal(arrays['layers'], layers) or not np.array_equal(arrays['neurons'], neurons):
        raise ValueError('Prior matrix candidate coordinates differ from the measured map')
    identity = {'model_identity': model.identity, 'prepared_identity': metadata['prepared_identities'][0],
                'runtime_identity': metadata['runtime_identities'][0]}
    rows = []
    for index, prompt_id in enumerate(arrays['prompt_ids'].tolist()):
        prompt = corpus.get(prompt_id)
        if (prompt is None or prompt_id != canonical_digest({key: value for key, value in prompt.items() if key != 'id'})
                or prompt['split'] != 'discovery' or prompt['corpus'] != str(arrays['corpora'][index])
                or prompt.get('mutation', {}).get('kind') not in (None, 'original')
                or prompt.get('labels', {}).get('profile') not in (None, 'baseline')):
            raise ValueError('Prior matrix source ID does not bind an unchanged original discovery record')
        rows.append(({'prompt': prompt, 'corpus_sha256': metadata['corpus_sha256'], 'previous_vector_index': index},
                     {'activation_artifact': {'sha256': str(arrays['source_hashes'][index])}}))
    return rows, identity


def duplicate_audit(rows, selected_rows, values, identity, read_values, workers):
    representatives = {prompt_key(job['prompt'], identity): (index, job, result)
                       for index, (job, result) in enumerate(selected_rows)}
    excluded = []
    for job, result in rows:
        _, representative, selected = representatives[prompt_key(job['prompt'], identity)]
        if (job['prompt']['id'], result['activation_artifact']['sha256']) != (representative['prompt']['id'], selected['activation_artifact']['sha256']):
            excluded.append((job, result))
    def compare(row):
        job, result = row; prompt = job['prompt']
        index, representative, selected = representatives[prompt_key(prompt, identity)]
        other = representative['prompt']
        source = read_values(row); reference = values[index]
        exact_prompt = (prompt.get('system', ''), prompt['prompt']) == (other.get('system', ''), other['prompt'])
        bit_equal = source.dtype == reference.dtype and source.shape == reference.shape and source.tobytes() == reference.tobytes()
        if exact_prompt and not bit_equal:
            raise ValueError('Byte-identical duplicate prompts have unequal candidate activations; investigate before deduplication')
        return {'excluded_prompt_id': prompt['id'], 'representative_prompt_id': other['id'],
            'excluded_artifact_sha256': result['activation_artifact']['sha256'],
            'representative_artifact_sha256': selected['activation_artifact']['sha256'],
            'excluded_corpus': prompt['corpus'], 'excluded_vector_sha256': hashlib.sha256(source.tobytes()).hexdigest(),
            'prompt_bytes_identical': exact_prompt, 'candidate_vectors_bit_equal': bit_equal,
            'maximum_abs_candidate_difference': float(np.max(np.abs(source.astype(np.float64) - reference.astype(np.float64)))),
            'reason': 'Repeated normalized system/user text under the same model/prepared/runtime; seeds or execution modes are not independent source observations.'}
    with ThreadPoolExecutor(max_workers=workers) as pool: return list(pool.map(compare, excluded))


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
        if artifact['stage'] != 'final_prefill' or artifact['operation'] not in ('prefill_observation', 'rollout'):
            raise ValueError('Only final-prefill observations are supported')
        full = data['mlp_input_last']
        if full.shape != (model.layers, model.intermediate) or not np.isfinite(full).all():
            raise ValueError('Unexpected or nonfinite activation matrix')
        values = np.asarray(full[layers, neurons], dtype=np.float32)
    return values


def preserve_previous(path, arrays, exclusions=()):
    """Reject refreshes that drop or alter any already published observation."""
    if path is None or (isinstance(path, (str, Path)) and not Path(path).exists()): return []
    snapshot = matrix_snapshot(path)
    digest, prior = snapshot['sha256'], snapshot['arrays']
    old, new = json.loads(str(prior['metadata'])), json.loads(str(arrays['metadata']))
    for key in ('schema', 'model_identity', 'map_sha256', 'split', 'measurement', 'unsteered', 'runtime_identities', 'prepared_identities'):
        if old.get(key) != new.get(key): raise ValueError('Previous matrix binding differs: ' + key)
    for key in ('layers', 'neurons'):
        if not np.array_equal(prior[key], arrays[key]): raise ValueError('Refresh changed the existing map candidate coordinates')
    indices = {identity: index for index, identity in enumerate(arrays['prompt_ids'].tolist())}
    excluded = {(entry['excluded_prompt_id'], entry['excluded_artifact_sha256']): entry for entry in exclusions}
    transitioned = []
    if len(set(prior['prompt_ids'].tolist())) != len(prior['prompt_ids']): raise ValueError('Previous matrix contains duplicate prompt IDs')
    for index, identity in enumerate(prior['prompt_ids'].tolist()):
        if identity not in indices:
            entry = excluded.get((identity, str(prior['source_hashes'][index])))
            if (entry is None or entry['representative_prompt_id'] not in indices
                    or entry['excluded_corpus'] != str(prior['corpora'][index])
                    or entry['excluded_vector_sha256'] != hashlib.sha256(prior['values'][index].tobytes()).hexdigest()):
                raise ValueError('Refresh would drop a previously published observation without verified duplicate evidence')
            transitioned.append(entry)
            continue
        current = indices[identity]
        if any(prior[key][index] != arrays[key][current] for key in ('corpora', 'source_hashes')):
            raise ValueError('Refresh changed a previously published source binding')
        before, after = prior['values'][index], arrays['values'][current]
        if before.dtype != after.dtype or before.shape != after.shape or before.tobytes() != after.tobytes():
            raise ValueError('Refresh changed previously published numeric observations')
    if isinstance(path, (str, Path)) and file_hash(path) != digest: raise ValueError('Previous matrix changed during refresh verification')
    return [*old.get('history', []), {'artifact_sha256': digest, 'rows': len(prior['prompt_ids']),
        'source_index_sha256': old.get('source_index_sha256'), 'corpus_sha256': old.get('corpus_sha256'), 'created': old.get('created'),
        'deduplication_transition': transitioned, 'raw_captures_preserved': True}]


def write_matrix(destination, arrays, *, replace=False):
    """Publish a fully written archive atomically in the existing destination dir."""
    destination = Path(destination)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        if replace: os.replace(temporary, destination)
        else: os.link(temporary, destination)  # Fail if a publication already exists.
        directory = os.open(destination.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--workers', type=int, default=max(1, len(os.sched_getaffinity(0)) - 1))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--registry', type=Path, action='append', default=[], help='Explicit hillclimb observation JSONL; repeat for multiple registries')
    parser.add_argument('--registry-only', action='store_true', help='Reuse prior compact rows and read only explicit registry captures; also supports audited dedup with no new registry')
    parser.add_argument('--previous', type=Path, help='Existing compact matrix, or - for bounded NPZ stdin; defaults to results/persona_coactivity.npz')
    parser.add_argument('--replace', action='store_true', help='Atomically replace --output after verifying every prior observation')
    parser.add_argument('--stream', action='store_true', help='Write the numeric NPZ to stdout; progress goes to stderr')
    args = parser.parse_args()
    if args.workers < 1 or (args.stream and (args.output or args.replace)):
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
    previous_path = args.previous or root / 'results/persona_coactivity.npz'
    previous = (matrix_snapshot(sys.stdin.buffer) if str(previous_path) == '-' else
                matrix_snapshot(previous_path) if previous_path.exists() else None)
    corpus, corpus_hash = published_corpus(root)
    rows, identity = (previous_records(previous, corpus, model, root, chart, map_hash, layers, neurons)
                      if args.registry_only else records(root / 'results/persona_manager.sqlite', model, corpus))
    extra, registries = registry_records(args.registry, model, corpus, identity, root / 'results')
    old_metadata = json.loads(str(previous['arrays']['metadata'])) if previous is not None else {}
    historical_hashes = old_metadata.get('captured_corpus_sha256', [old_metadata['corpus_sha256']]) if old_metadata else []
    corpus_hashes = sorted({job['corpus_sha256'] for job, _ in [*rows, *extra] if 'previous_vector_index' not in job}
                           | set(historical_hashes))
    if args.registry_only:
        receipts = {item['name']: item for item in old_metadata.get('registries', [])}
        receipts.update({item['name']: item for item in registries})
        registries = sorted(receipts.values(), key=lambda item: item['name'])
    preserved_ids, preserved_sources = set(), {}
    if previous is not None:
        archive = previous['arrays']
        preserved_ids = set(archive['prompt_ids'].tolist())
        preserved_sources = dict(zip(archive['prompt_ids'].tolist(), archive['source_hashes'].tolist()))
    all_rows = [*rows, *extra]
    rows, duplicates = unique_rows(all_rows, identity, preserved_ids, preserved_sources)
    values = np.empty((len(rows), len(coordinates)), dtype=np.float32)
    counts = Counter(job['prompt']['corpus'] for job, _ in rows)
    print(json.dumps({'selected_rows': len(rows), 'candidates': len(coordinates), 'corpora': counts,
                      'workers': args.workers}), file=sys.stderr, flush=True)
    def extract(row):
        if 'previous_vector_index' in row[0]:
            return previous['arrays']['values'][row[0]['previous_vector_index']]
        return read_candidate_values(row, model, layers, neurons, root / 'results')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, vector in enumerate(pool.map(extract, rows)):
            values[index] = vector
            if (index + 1) % 250 == 0:
                print(json.dumps({'extracted': index + 1, 'total': len(rows), 'elapsed_s': round(time.time()-started, 1)}), file=sys.stderr, flush=True)
    exclusions = duplicate_audit(all_rows, rows, values, identity, extract, args.workers)
    if file_hash(map_path) != map_hash or published_corpus(root)[1] != corpus_hash:
        raise ValueError('Measured map or corpus publication changed during extraction')
    prompt_ids = [job['prompt']['id'] for job, _ in rows]
    source_hashes = [result['activation_artifact']['sha256'] for _, result in rows]
    metadata = {'schema': 'character-creation.coactivity.v1', 'model_identity': model.identity,
        'map_sha256': map_hash, 'split': 'discovery', 'measurement': 'final_prefill_mlp_input_last',
        'unsteered': True, 'runtime_identities': [identity['runtime_identity']],
        'prepared_identities': [identity['prepared_identity']], 'corpus_sha256': corpus_hash,
        'captured_corpus_sha256': corpus_hashes, 'registries': registries, 'deduplicated_observations': duplicates,
        'duplicate_observations': exclusions,
        'deduplication': 'Whitespace-normalized system/user strings, case preserved, and model/prepared/runtime. Prefer an existing published representative, then the smallest prompt ID. Excluded observations remain in raw captures and explicit transition history; seeds and execution modes are not independent source rows.',
        'extraction_sha256': file_hash(__file__), 'created': time.time(),
        'selection': 'Verified operator coverage and explicit hillclimb baseline registries; original discovery prompts only; baseline prompt profile; unique normalized source text.',
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
    metadata['history'] = preserve_previous(previous, arrays, exclusions)
    arrays['metadata'] = np.array(json.dumps(metadata, sort_keys=True))
    if args.stream:
        np.savez_compressed(sys.stdout.buffer, **arrays)
    else:
        destination = args.output or root / 'results/persona_coactivity.npz'
        if str(previous_path) != '-' and previous is not None and file_hash(previous_path) != previous['sha256']:
            raise ValueError('Previous matrix changed before publication')
        if args.replace and destination.exists() and destination.resolve() != previous_path.resolve():
            preserve_previous(destination, arrays, exclusions)
        write_matrix(destination, arrays, replace=args.replace)
    print(json.dumps({'complete': True, 'rows': len(rows), 'elapsed_s': round(time.time()-started, 1)}), file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
