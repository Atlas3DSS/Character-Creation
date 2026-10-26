"""Incremental, private viewer index of the frozen overnight activation sweep.

No inference or Hub publication. Validate original capture bytes once, retain a
resumable numeric cache, and publish compact map observations plus prompt evidence.
Historical calibration and co-activity artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import shlex
import sqlite3
import subprocess
import tempfile
import threading
import time

import numpy as np

IDENTITIES = ('model_identity', 'runtime_identity', 'prepared_identity')
TIERS = ('authored_emotion_contrast', 'book_diverse', 'registered_big_five')
FILES = ('persona_overnight_viewer.npz', 'persona_overnight_viewer_records.jsonl')
HELDOUT = 'persona_overnight_viewer_heldout.npz'
# Mechanical storage benefits from sequential reads; CPU verification/decompression
# still uses every admitted CPU. Do not interleave 86 streams through the same disk.
READ_SLOTS = threading.BoundedSemaphore(2)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(value); stream.flush(); os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=60)


def source_rows(database):
    """A short SQLite read snapshot; never lock the capture ledger for extraction."""
    with readonly(database) as db:
        binding = dict(db.execute('SELECT key,value FROM metadata'))
        counts = Counter(dict(db.execute('SELECT state,count(*) FROM tasks GROUP BY state')))
        rows = list(db.execute("SELECT id,ordinal FROM tasks WHERE state='complete' ORDER BY ordinal"))
    return binding, counts, rows


def capture_rows(database, ordinals):
    # Parse large response JSON only for new rows, in short bounded transactions.
    if not ordinals: return []
    with readonly(database) as db:
        rows = list(db.execute("""SELECT id,ordinal,row,request,validation,response
          FROM tasks WHERE state='complete' AND ordinal IN (""" + ','.join('?' for _ in ordinals)
          + ') ORDER BY ordinal', ordinals))
    if len(rows) != len(ordinals): raise ValueError('Completed capture changed during bounded read')
    return rows


def frozen_inputs(path, expected):
    if file_hash(path) != expected:
        raise ValueError('Frozen input manifest SHA256 changed')
    records = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['id'] in records: raise ValueError('Duplicate frozen source ID')
            records[row['id']] = row
    return records


def extract(row, frozen, root, coordinates, model_identity):
    if len(row) == 6:
        response = json.loads(row[5])
        row = (*row[:5], json.dumps(response['activation_artifact']), json.dumps(response['provenance_artifact']),
               *[response[k] for k in IDENTITIES], json.dumps(response['activation_search']['coverage']))
    identity, ordinal, raw_source, raw_request, raw_validation, raw_artifact, raw_provenance, *rest = row
    source, request, validation, artifact, provenance = map(json.loads,
        (raw_source, raw_request, raw_validation, raw_artifact, raw_provenance))
    identities = dict(zip(IDENTITIES, rest[:3])); coverage = json.loads(rest[3])
    if source != frozen.get(identity) or source.get('id') != identity:
        raise ValueError('Ledger source differs from frozen input: ' + identity)
    if source.get('split') not in ('discovery', 'heldout') or source.get('tier') not in TIERS:
        raise ValueError('Only declared discovery/heldout tiers may enter the viewer')
    if (request.get('messages') != source['messages'] or request.get('observation_only') is not True
            or request.get('max_tokens') != 1 or 'steering' in request):
        raise ValueError('Capture is not its exact unsteered source prompt')
    if identities['model_identity'] != model_identity:
        raise ValueError('Capture model differs from displayed map')
    request_hash = digest(request)
    tokens = coverage.get('tokens')
    if (type(tokens) is not int or not 1 <= tokens <= 4096
            or coverage.get('all_neurons') is not True or coverage.get('all_prompt_positions') is not True
            or (coverage.get('layers'), coverage.get('neurons_per_layer')) != (64, 17408)
            or validation.get('tokens') != tokens or validation.get('neurons') != 64 * 17408
            or validation.get('raw_artifact_sha256') != artifact.get('sha256')
            or artifact.get('request_sha256') != request_hash):
        raise ValueError('Capture validation or coverage binding differs')
    contents = []
    for pointer in (artifact, provenance):
        path = Path(pointer['path']).resolve()
        if path.parent != root.resolve(): raise ValueError('Capture artifact location changed')
        with READ_SLOTS: raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != pointer['sha256']:
            raise ValueError('Capture artifact location or bytes changed')
        contents.append(raw)
    report = json.loads(contents[1])
    if report.get('request') != request or report.get('activation_artifact') != artifact:
        raise ValueError('Capture provenance does not bind the request and raw artifact')
    with np.load(io.BytesIO(contents[0]), allow_pickle=False) as data:
        if any(str(data[k]) != identities[k] for k in IDENTITIES):
            raise ValueError('Raw activation identity differs')
        if str(data['request_sha256']) != request_hash or str(data['stage']) != 'all_prefill_search':
            raise ValueError('Raw request or measurement stage differs')
        full = data['mlp_input_last']
        if full.shape != (64, 17408) or not np.isfinite(full).all():
            raise ValueError('Invalid final-prefill activation plane')
        if (data['prompt_token_ids'].shape != (tokens,)
                or not np.array_equal(data['search_positions'], np.arange(tokens))
                or tokens - 1 not in data['final_chunk_positions']):
            raise ValueError('Final prompt token is not present in the capture')
        values = np.asarray(full[coordinates[:, 0], coordinates[:, 1]], dtype='<f4')
    group = digest([digest({'messages': request['messages'], 'template': request.get('chat_template_kwargs'), **identities}), source['split']])
    record = {'id': identity, 'tier': source['tier'], 'messages': source['messages'],
              'split': source['split'],
              'provenance': source.get('provenance', {}), 'source_sha256': source['source_sha256'],
              'request_sha256': request_hash, 'artifact_sha256': artifact['sha256'], 'tokens': tokens,
              'duplicate_count': 1, 'identities': identities}
    return (identity, ordinal, digest(source), artifact['sha256'], group,
            source['tier'], encoded(record).decode(), values.tobytes())


def cache_database(path, binding, previous_extractor=None, frozen=None):
    db = sqlite3.connect(path, timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT);
      CREATE TABLE IF NOT EXISTS captures(id TEXT PRIMARY KEY,ordinal INTEGER UNIQUE,source_sha256 TEXT,
      artifact_sha256 TEXT,prompt_group TEXT,tier TEXT,record TEXT,values_blob BLOB);
      CREATE INDEX IF NOT EXISTS captures_group ON captures(prompt_group);''')
    previous = dict(db.execute('SELECT key,value FROM metadata'))
    if (previous and previous != binding and previous_extractor
            and previous.get('extractor_sha256') == previous_extractor
            and all(previous.get(k) == binding[k] for k in ('input_sha256', 'map_sha256'))):
        # Explicit source-bound upgrade before the first published snapshot. Values
        # and original artifact hashes are retained; split labels come from frozen inputs.
        for identity, group, record_json in db.execute('SELECT id,prompt_group,record FROM captures').fetchall():
            record = json.loads(record_json); split = frozen[identity]['split']
            if 'split' not in record:
                record['split'] = split
                db.execute('UPDATE captures SET prompt_group=?,record=? WHERE id=?',
                           (digest([group, split]), encoded(record).decode(), identity))
        db.execute('UPDATE metadata SET value=? WHERE key=?', (binding['extractor_sha256'], 'extractor_sha256'))
        db.commit(); previous = dict(db.execute('SELECT key,value FROM metadata'))
        print(json.dumps({'event': 'reviewed_extractor_upgrade', 'previous': previous_extractor,
                          'current': binding['extractor_sha256']}), flush=True)
    if previous and previous != binding:
        db.close(); raise ValueError('Viewer cache source, map or extractor binding changed')
    db.executemany('INSERT OR IGNORE INTO metadata VALUES(?,?)', binding.items()); db.commit()
    return db


def publish(db, output, chart, map_sha, binding, ledger_counts):
    coordinates = np.array(sorted((n['layer'], n['neuron']) for n in chart['neurons']), dtype=np.int64)
    records, vectors, groups, duplicates, tier_counts = [], [], {}, 0, Counter()
    identities = set()
    for identity, group, tier, record_json, blob in db.execute(
            'SELECT id,prompt_group,tier,record,values_blob FROM captures ORDER BY ordinal'):
        values = np.frombuffer(blob, dtype='<f4')
        if values.shape != (len(coordinates),) or not np.isfinite(values).all():
            raise ValueError('Invalid cached candidate vector')
        record = json.loads(record_json)
        identities.add(tuple(record['identities'][k] for k in IDENTITIES)); tier_counts[tier] += 1
        if group in groups:
            index = groups[group]
            if not np.array_equal(values, vectors[index]):
                raise ValueError('Identical prompt/runtime has unequal final-token observations: ' + identity)
            records[index]['duplicate_count'] += 1
            records[index].setdefault('duplicate_ids', []).append(identity)
            duplicates += 1; continue
        groups[group] = len(records); records.append(record); vectors.append(values)
    if not records: return None
    if len(identities) != 1: raise ValueError('Do not pool captures from distinct runtime identities')
    identity = dict(zip(IDENTITIES, identities.pop()))
    all_values = np.stack(vectors)
    created = time.time()
    metadata = {'schema': 'character-creation.coactivity.v1', **identity, 'map_sha256': map_sha,
        'split': 'discovery', 'measurement': 'final_prefill_mlp_input_last', 'unsteered': True,
        'runtime_identities': [identity['runtime_identity']], 'prepared_identities': [identity['prepared_identity']],
        'corpus_sha256': binding['input_sha256'], 'extraction_sha256': binding['extractor_sha256'],
        'created': created, 'selection': 'Completed frozen overnight discovery captures; exact message/template/runtime duplicates counted once.',
        'source_index_sha256': digest([(r['id'], r['artifact_sha256']) for r in records]),
        'corpus_counts': dict(Counter(r['tier'] for r in records)), 'imported_captures': sum(tier_counts.values()),
        'deduplicated_observations': duplicates,
        'limitations': ['Final prompt-token activations only; all-position reductions remain in the original raw captures.',
          'No pooling with the historical runtime. Correlation is not causal wiring or temporal direction.',
          'Prompt families and corpus mixtures are not independent randomized trials; compare tiers separately.',
          'Private book text is local viewer evidence and is not authorized for public Hub export.']}
    output = Path(output)
    split_counts = {}
    for split, filename in [('discovery', FILES[0]), ('heldout', HELDOUT)]:
        indices = [i for i,r in enumerate(records) if r['split'] == split]
        subset = [records[i] for i in indices]; split_counts[split] = len(indices)
        split_meta = {**metadata, 'split': split, 'source_index_sha256': digest([(r['id'], r['artifact_sha256']) for r in subset]),
                      'corpus_counts': dict(Counter(r['tier'] for r in subset))}
        arrays = dict(metadata=np.array(json.dumps(split_meta, sort_keys=True)), layers=coordinates[:, 0], neurons=coordinates[:, 1],
            values=all_values[indices], corpora=np.array([r['tier'] for r in subset], dtype='<U32'),
            prompt_ids=np.array([r['id'] for r in subset], dtype='<U64'),
            source_hashes=np.array([r['artifact_sha256'] for r in subset], dtype='<U64'))
        if sum(a.nbytes for a in arrays.values()) > 512 << 20:
            raise ValueError('Viewer matrix exceeds the bounded reader contract')
        with tempfile.NamedTemporaryFile(dir=output, delete=False) as stream:
            temporary = Path(stream.name); np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, output / filename)
    atomic(output / FILES[1], b''.join(encoded(r) + b'\n' for r in records))
    summary = {'schema': 'character-creation.overnight-viewer.v1', 'state': 'ready', **identity,
        'updated': created, 'imported_captures': sum(tier_counts.values()), 'unique_observations': len(records),
        'discovery_observations': split_counts['discovery'], 'heldout_observations': split_counts['heldout'],
        'heldout_captures': sum(r['duplicate_count'] for r in records if r['split'] == 'heldout'),
        'duplicates': duplicates, 'counts_by_tier': dict(tier_counts), 'total_tasks': sum(ledger_counts.values()),
        'completed_captures': ledger_counts['complete'], 'pending_captures': ledger_counts['pending'],
        'ledger_counts': dict(ledger_counts), 'npz_sha256': file_hash(output / FILES[0]),
        'records_sha256': file_hash(output / FILES[1]), 'map_sha256': map_sha,
        'heldout_npz_sha256': file_hash(output / HELDOUT),
        'source_input_sha256': binding['input_sha256'], 'displayed_candidates': len(coordinates),
        'captured_neurons_per_prompt': 64 * 17408, 'private': True}
    atomic(output / 'persona_overnight_viewer.json', encoded(summary) + b'\n')
    print(json.dumps({'event': 'published', **summary}), flush=True)
    return summary


def integrate(args):
    root, output = args.root.resolve(), args.output.resolve()
    lock = (output / 'persona_overnight_viewer.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    map_sha = file_hash(args.map); chart = json.loads(args.map.read_text())
    coordinates = np.array(sorted((n['layer'], n['neuron']) for n in chart['neurons']), dtype=np.int64)
    if len(coordinates) != len(set(map(tuple, coordinates))) or not len(coordinates):
        raise ValueError('Map candidate coordinates must be nonempty and unique')
    source_binding, _, _ = source_rows(args.database)
    inputs = frozen_inputs(args.inputs, source_binding['input_sha256'])
    binding = {'input_sha256': source_binding['input_sha256'], 'map_sha256': map_sha,
               'extractor_sha256': file_hash(__file__)}
    db = cache_database(output / 'persona_overnight_viewer.sqlite', binding, args.previous_extractor, inputs)
    cpus = sorted(os.sched_getaffinity(0)); slots = queue.Queue()
    for cpu in cpus: slots.put(cpu)
    workers = min(args.workers or len(cpus), len(cpus))
    def pin(): os.sched_setaffinity(0, {slots.get_nowait()})
    last_published, last_time = -1, 0
    while True:
        current_binding, counts, rows = source_rows(args.database)
        if current_binding['input_sha256'] != binding['input_sha256'] or file_hash(args.map) != map_sha:
            raise ValueError('Source inputs or displayed map changed during extraction')
        prior = {r[0]: r[1:] for r in db.execute('SELECT id,source_sha256,artifact_sha256 FROM captures')}
        new = []
        for row in rows:
            if row[0] in prior:
                if prior[row[0]][0] != digest(inputs[row[0]]):
                    raise ValueError('Cached source changed after import')
            else: new.append(row)
        if not set(prior).issubset({r[0] for r in rows}): raise ValueError('Previously imported completed capture disappeared')
        print(json.dumps({'event': 'extracting', 'cached': len(prior), 'new': len(new), 'workers': workers}), flush=True)
        with ThreadPoolExecutor(max_workers=workers, initializer=pin) as pool:
            for start in range(0, len(new), workers * 2):
                selected = []
                for offset in range(start, min(len(new), start + workers * 2), 8):
                    selected.extend(capture_rows(args.database, [r[1] for r in new[offset:min(offset + 8, start + workers * 2)]]))
                futures = [pool.submit(extract, row, inputs, root, coordinates, chart['model_identity']) for row in selected]
                batch = [f.result() for f in futures]
                db.executemany('INSERT INTO captures VALUES(?,?,?,?,?,?,?,?)', batch); db.commit()
                imported = len(prior) + start + len(batch)
                print(json.dumps({'event': 'imported', 'captures': imported, 'total_complete': len(rows)}), flush=True)
                if imported - last_published >= 1000 and time.monotonic() - last_time > 120:
                    publish(db, output, chart, map_sha, binding, counts)
                    last_published, last_time = imported, time.monotonic()
        count = db.execute('SELECT count(*) FROM captures').fetchone()[0]
        if count != last_published:
            publish(db, output, chart, map_sha, binding, counts); last_published = count; last_time = time.monotonic()
        if not args.watch: break
        time.sleep(60)
        slots = queue.Queue()
        for cpu in cpus: slots.put(cpu)


def sync(args):
    """Read three named derived files only; do not copy the private raw archive."""
    destination = args.output.resolve()
    lock = (destination / 'persona_overnight_viewer_sync.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
           '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', args.remote_host]
    remote = str(args.root)
    last = None
    while True:
        temporary = []
        try:
            raw = subprocess.check_output([*ssh, shlex.join(['cat', remote + '/persona_overnight_viewer.json'])], timeout=30)
            summary = json.loads(raw)
            key = (summary['npz_sha256'], summary['records_sha256'], summary['heldout_npz_sha256'])
            if summary['map_sha256'] != file_hash(args.map): raise ValueError('Remote viewer map binding differs')
            if key != last:
                for name, expected, maximum in zip((*FILES, HELDOUT), key, (512 << 20, 192 << 20, 512 << 20)):
                    with tempfile.NamedTemporaryFile(dir=destination, delete=False) as stream:
                        path = Path(stream.name); temporary.append(path)
                        subprocess.run([*ssh, shlex.join(['cat', remote + '/' + name])], stdout=stream, check=True, timeout=300)
                    if path.stat().st_size > maximum or file_hash(path) != expected:
                        raise ValueError('Remote viewer changed during transfer or exceeded its bound')
                check = subprocess.check_output([*ssh, shlex.join(['cat', remote + '/persona_overnight_viewer.json'])], timeout=30)
                if check != raw: raise ValueError('Remote publication advanced during transfer; retry next cycle')
                for path, name in zip(temporary, (*FILES, HELDOUT)): os.replace(path, destination / name)
                atomic(destination / 'persona_overnight_viewer.json', raw)
                last = key; print(json.dumps({'event': 'synced', 'captures': summary['imported_captures'], 'observations': summary['unique_observations']}), flush=True)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            print(json.dumps({'event': 'sync_wait', 'error': str(error)}), flush=True)
            if not args.watch: raise
        finally:
            for path in temporary: path.unlink(missing_ok=True)
        if not args.watch: return
        time.sleep(60)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--map', type=Path, required=True); p.add_argument('--database', type=Path)
    p.add_argument('--inputs', type=Path); p.add_argument('--workers', type=int)
    p.add_argument('--previous-extractor', help='Explicit reviewed source SHA allowed to upgrade the existing private cache')
    p.add_argument('--watch', action='store_true'); p.add_argument('--sync', action='store_true')
    p.add_argument('--remote-host', default='v100'); args = p.parse_args()
    if not args.sync and (args.database is None or args.inputs is None): p.error('Extraction requires --database and --inputs')
    sync(args) if args.sync else integrate(args)


if __name__ == '__main__': main()
