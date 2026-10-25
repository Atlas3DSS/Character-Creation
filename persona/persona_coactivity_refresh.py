"""Fixed, bounded transport for private corpus sync and co-activity publication.

Operator configuration alone selects SSH and filesystem locations. Scheduled
models can supply neither commands nor paths. No inference services are started.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time

from persona_coactivity_extract import (MAX_MATRIX_BYTES, file_hash, matrix_snapshot,
    preserve_previous, published_corpus)
from persona_neighbors import CoactivityIndex
from persona_vectors import ModelSpec, canonical_digest

PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "persona":
    PROJECT_ROOT = PROJECT_ROOT.parent

SYNC_SCHEMA = 'character-creation.corpus-sync.v1'
MAX_CORPUS_BYTES = 384 << 20
FILES = ('persona_manager_corpus.jsonl', 'persona_manager_corpus_manifest.json',
         'persona_generated_big_five.jsonl', 'persona_generated_big_five_manifest.json')
SOURCES = ('persona_coactivity_extract.py', 'persona_coactivity_refresh.py',
           'persona_neighbors.py', 'persona_manager.py', 'persona_vectors.py', 'persona_fleet.py')
REGISTRY = re.compile(r'hillclimb-([0-9a-f]{32})-observations\.jsonl\Z')


def source_path(root: Path, name: str) -> Path:
    """Find source in this checkout or in the existing flat worker deployment."""
    root = Path(root)
    organized = root / 'persona' / name
    return organized if '/' not in name and organized.is_file() else root / name


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def atomic_bytes(path, data):
    path = Path(path); temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name); stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def bounded_process(argv, payload=b'', *, timeout=600, maximum=MAX_MATRIX_BYTES, stop_event=None):
    """Bound stdout, stderr, elapsed time and process ownership without a shell."""
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    selector = selectors.DefaultSelector(); output, errors = bytearray(), bytearray()
    sent, deadline = 0, time.monotonic() + timeout
    try:
        for stream, events, label in ((process.stdin, selectors.EVENT_WRITE, 'input'),
                (process.stdout, selectors.EVENT_READ, 'output'), (process.stderr, selectors.EVENT_READ, 'error')):
            os.set_blocking(stream.fileno(), False); selector.register(stream, events, label)
        while selector.get_map():
            if stop_event is not None and stop_event.is_set(): raise InterruptedError('Co-activity transport cancelled')
            if time.monotonic() >= deadline: raise TimeoutError('Co-activity transport exceeded its fixed CPU deadline')
            for key, _ in selector.select(min(.2, max(0, deadline - time.monotonic()))):
                stream, label = key.fileobj, key.data
                if label == 'input':
                    if sent < len(payload):
                        try: sent += os.write(stream.fileno(), memoryview(payload)[sent:sent + 65536])
                        except BrokenPipeError:
                            selector.unregister(stream); stream.close()
                    else:
                        selector.unregister(stream); stream.close()
                else:
                    chunk = os.read(stream.fileno(), 65536)
                    if not chunk:
                        selector.unregister(stream); stream.close(); continue
                    if label == 'output':
                        if len(output) + len(chunk) > maximum: raise ValueError('Remote output exceeded the bounded size')
                        output.extend(chunk)
                    else:
                        errors.extend(chunk); del errors[:-65536]
        code = process.wait(timeout=max(.1, deadline - time.monotonic()))
        if code: raise RuntimeError(f'Fixed remote command exited {code}: ' + errors[-1500:].decode(errors='replace'))
        return bytes(output)
    finally:
        selector.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            if not stream.closed: stream.close()
        if process.poll() is None:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass  # The owned child may exit between poll and kill.
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait(timeout=10)


def settings(config):
    root = Path(config['root']).resolve()
    fleet = json.loads(Path(config['fleet_config']).read_text())
    host = fleet['ssh_host']; remote = Path(fleet['remote_root']); python = Path(fleet['python'])
    if (not isinstance(host, str) or not re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_.@-]*', host)
            or fleet.get('hostname') != 'v100' or not remote.is_absolute() or not python.is_absolute()
            or any(char in str(remote) + str(python) for char in ('\n', '\r', '\0'))):
        raise ValueError('Fixed V100 SSH/root/interpreter configuration is invalid')
    if not (root / 'results').is_dir() or not (root / 'data').is_dir(): raise ValueError('Existing project data/results directories required')
    if not all(digest(config.get(key)) for key in ('expected_prepared_identity', 'expected_runtime_identity')):
        raise ValueError('Explicit prepared/runtime digests are required')
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
           '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', host]
    return root, fleet, remote, ssh


def remote_command(ssh, argv):
    # SSH transports one quoted argv through its remote shell; no text supplied
    # by a prompt, planner, or registry is ever evaluated as command syntax.
    return [*ssh, shlex.join([str(value) for value in argv])]


def verify_remote(config, transport=bounded_process, stop_event=None):
    root, fleet, remote, ssh = settings(config)
    actual = transport(remote_command(ssh, ['hostname']), timeout=20, maximum=1024, stop_event=stop_event).decode().strip()
    if actual != fleet['hostname']: raise ValueError('Remote hostname differs from the configured V100 host')
    names = (*SOURCES, 'data/persona_model.json', 'results/persona_map.json')
    text = transport(remote_command(ssh, ['sha256sum', *[remote / name for name in names]]),
                     timeout=30, maximum=16384, stop_event=stop_event).decode()
    hashes = [line.split()[0] for line in text.splitlines()]
    if hashes != [file_hash(source_path(root, name)) for name in names]:
        raise ValueError('Remote refresh source/model/map differs from the reviewed local files')
    return root, fleet, remote, ssh


def corpus_payload(root):
    root = Path(root); _, corpus_sha = published_corpus(root)
    parts = {}
    for name in FILES:
        path = root / 'data' / name
        if path.exists(): parts[name] = path.read_bytes()
    if sum(map(len, parts.values())) > MAX_CORPUS_BYTES: raise ValueError('Corpus sync exceeds the bounded payload size')
    if hashlib.sha256(parts[FILES[0]]).hexdigest() != corpus_sha: raise ValueError('Corpus changed while preparing sync')
    if (FILES[2] in parts) != (FILES[3] in parts): raise ValueError('Generated corpus and manifest must be paired')
    for data_name, manifest_name in ((FILES[0], FILES[1]), (FILES[2], FILES[3])):
        if data_name not in parts: continue
        manifest = json.loads(parts[manifest_name])
        if manifest.get('status') != 'complete' or manifest.get('output_sha256') != hashlib.sha256(parts[data_name]).hexdigest():
            raise ValueError('Corpus sync source manifest is incomplete or mismatched')
    files = [{'name': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} for name, raw in parts.items()]
    return encoded({'schema': SYNC_SCHEMA, 'files': files}) + b'\n' + b''.join(parts.values()), files


def receive_corpus(root, stream):
    """Append a verified local publication into fixed existing remote data files."""
    header = stream.readline(8193)
    if len(header) > 8192 or not header.endswith(b'\n'): raise ValueError('Invalid bounded sync header')
    envelope = json.loads(header); entries = envelope.get('files')
    if envelope.get('schema') != SYNC_SCHEMA or not isinstance(entries, list): raise ValueError('Invalid corpus sync schema')
    names = [item.get('name') for item in entries]
    if names not in ([*FILES[:2]], [*FILES]): raise ValueError('Only the fixed corpus publication filenames are allowed')
    parts, total = {}, 0
    for item in entries:
        size = item.get('bytes')
        if type(size) is not int or size < 0 or not digest(item.get('sha256')): raise ValueError('Invalid corpus sync file binding')
        total += size
        if total > MAX_CORPUS_BYTES: raise ValueError('Corpus sync input exceeds the bounded size')
        raw = stream.read(size)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != item['sha256']: raise ValueError('Corpus sync bytes differ from the bound digest')
        parts[item['name']] = raw
    if stream.read(1): raise ValueError('Unexpected trailing corpus sync data')
    for data_name, manifest_name in ((FILES[0], FILES[1]), (FILES[2], FILES[3])):
        if data_name not in parts: continue
        manifest = json.loads(parts[manifest_name])
        if manifest.get('status') != 'complete' or manifest.get('output_sha256') != hashlib.sha256(parts[data_name]).hexdigest():
            raise ValueError('Incoming corpus manifest does not describe a complete publication')
    data = Path(root) / 'data'; main = data / FILES[0]; manifest_path = data / FILES[1]
    with main.open('r+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        old = stream.read(); current = json.loads(manifest_path.read_bytes())
        if (current.get('status') != 'complete' and current.get('sync_owner') != SYNC_SCHEMA):
            raise ValueError('Another corpus writer owns the unfinished publication')
        if current.get('status') == 'complete' and current.get('output_sha256') != hashlib.sha256(old).hexdigest():
            raise ValueError('Existing remote corpus digest changed')
        if not parts[FILES[0]].startswith(old): raise ValueError('Corpus sync must preserve every existing remote byte')
        if FILES[2] in parts and (data / FILES[2]).exists() and not parts[FILES[2]].startswith((data / FILES[2]).read_bytes()):
            raise ValueError('Generated corpus sync would rewrite existing records')
        atomic_bytes(manifest_path, encoded({**current, 'status': 'ingesting', 'sync_owner': SYNC_SCHEMA,
            'sync_expected_sha256': hashlib.sha256(parts[FILES[0]]).hexdigest()}))
        stream.seek(0, os.SEEK_END); stream.write(parts[FILES[0]][len(old):]); stream.flush(); os.fsync(stream.fileno())
        for name in FILES[2:]:
            if name in parts: atomic_bytes(data / name, parts[name])
        atomic_bytes(manifest_path, parts[FILES[1]])
    return {'schema': SYNC_SCHEMA, 'status': 'complete', 'files': entries,
            'corpus_sha256': hashlib.sha256(parts[FILES[0]]).hexdigest(), 'existing_bytes_preserved': len(old)}


def sync_corpus(config, event, stop_event=None, *, transport=bounded_process):
    root, fleet, remote, ssh = verify_remote(config, transport, stop_event)
    payload, files = corpus_payload(root)
    command = remote_command(ssh, ['timeout', '--signal=TERM', '--kill-after=10s', '180s',
        fleet['python'], remote / 'persona_coactivity_refresh.py', 'receive-corpus', '--root', remote])
    receipt = json.loads(transport(command, payload, timeout=190, maximum=16384, stop_event=stop_event))
    if receipt.get('status') != 'complete' or receipt.get('files') != files:
        raise ValueError('Remote corpus sync receipt differs from the sent publication')
    if published_corpus(root)[1] != receipt.get('corpus_sha256'): raise ValueError('Local corpus changed during sync')
    event('corpus_synced', receipt)
    return receipt


def registry_receipt(job, summary, remote):
    identity = job.get('id')
    if not isinstance(identity, str) or re.fullmatch('[0-9a-f]{32}', identity) is None:
        raise ValueError('Scheduled job identity must be an opaque 32-digit hexadecimal ID')
    if job.get('kind') != 'neighborhood_hillclimb': raise ValueError('Only neighborhood jobs publish observation registries')
    receipt = summary.get('baseline_registry')
    expected = remote / 'results' / f'hillclimb-{identity}-observations.jsonl'
    if (not isinstance(receipt, dict) or receipt.get('path') != str(expected) or not digest(receipt.get('sha256'))
            or type(receipt.get('rows')) is not int or not 0 <= receipt['rows'] <= 100000
            or receipt.get('status') != 'registered_for_later_coactivity_ingestion'):
        raise ValueError('Completed job registry receipt is invalid or points outside its fixed result path')
    return {'name': expected.name, 'sha256': receipt['sha256'], 'rows': receipt['rows']}


def validate_returned(raw, previous, root, config, expected_registries, corpus_sha):
    snapshot = matrix_snapshot(raw); arrays = snapshot['arrays']
    model = ModelSpec.load(root / 'data/persona_model.json'); map_path = root / 'results/persona_map.json'
    validated = CoactivityIndex(root, model)._validate(arrays, json.loads(map_path.read_bytes()), file_hash(map_path), snapshot['sha256'])
    metadata = validated['metadata']
    if (metadata.get('runtime_identities') != [config['expected_runtime_identity']]
            or metadata.get('prepared_identities') != [config['expected_prepared_identity']]
            or metadata.get('extraction_sha256') != file_hash(source_path(root, 'persona_coactivity_extract.py'))
            or metadata.get('corpus_sha256') != corpus_sha):
        raise ValueError('Returned matrix source/runtime/publication binding differs')
    receipts = metadata.get('registries', [])
    if (not isinstance(receipts, list) or len({item['name'] for item in receipts}) != len(receipts)
            or {item['name']: item['sha256'] for item in receipts} != expected_registries):
        raise ValueError('Returned matrix registry receipts differ from admitted sources')
    corpus, current = published_corpus(root)
    if current != corpus_sha: raise ValueError('Local corpus changed during extraction')
    for prompt_id, name in zip(arrays['prompt_ids'].tolist(), arrays['corpora'].tolist()):
        row = corpus.get(prompt_id)
        if (row is None or row['id'] != canonical_digest({key: value for key, value in row.items() if key != 'id'})
                or row['split'] != 'discovery' or row['corpus'] != name
                or row.get('mutation', {}).get('kind') not in (None, 'original')
                or row.get('labels', {}).get('profile') not in (None, 'baseline')):
            raise ValueError('Returned matrix contains an unbound or ineligible source record')
    if metadata.get('source_index_sha256') != canonical_digest(list(zip(arrays['prompt_ids'].tolist(), arrays['source_hashes'].tolist()))):
        raise ValueError('Returned matrix source index hash differs')
    if metadata.get('corpus_counts') != dict(Counter(arrays['corpora'].tolist())):
        raise ValueError('Returned matrix cohort counts differ')
    history = preserve_previous(previous, arrays, metadata.get('duplicate_observations', []))
    if metadata.get('history') != history: raise ValueError('Returned matrix omitted or changed the audited prior-source transition')
    return snapshot


def _refresh(config, event, stop_event, job=None, summary=None, *, transport=bounded_process):
    root, fleet, remote, ssh = settings(config)
    registry = registry_receipt(job, summary, remote) if job is not None else None
    if (job and job.get('cancel_requested')) or (stop_event and stop_event.is_set()):
        receipt = {'status': 'skipped', 'map_updated': False, 'reason': 'operator cancellation or shutdown', 'baseline_registry': registry}
        event('map_update_skipped', receipt); return receipt
    timeout = config.get('coactivity_refresh_seconds', 600)
    workers = config.get('coactivity_workers', 86)
    if type(timeout) is not int or not 30 <= timeout <= 1800 or type(workers) is not int or not 1 <= workers <= 256:
        raise ValueError('Invalid fixed CPU refresh bounds')
    destination = root / 'results/persona_coactivity.npz'
    with (root / 'results/persona_coactivity.refresh.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous_raw = destination.read_bytes(); previous = matrix_snapshot(previous_raw)
        source_bindings = {name: file_hash(source_path(root, name)) for name in
            ('data/persona_model.json', 'results/persona_map.json', 'persona_coactivity_extract.py')}
        old = json.loads(str(previous['arrays']['metadata']))
        expected = {}
        for item in old.get('registries', []):
            if REGISTRY.fullmatch(item['name']) is None or not digest(item['sha256']): raise ValueError('Previous matrix has a nonallowlisted registry receipt')
            expected[item['name']] = item['sha256']
        if registry: expected[registry['name']] = registry['sha256']
        sync_receipt = sync_corpus(config, event, stop_event, transport=transport)
        event('coactivity_refresh_started', {'previous_sha256': previous['sha256'], 'registry': registry,
            'cpu_timeout_seconds': timeout, 'gpu_deadline_does_not_extend': True})
        argv = ['timeout', '--signal=TERM', '--kill-after=10s', str(timeout) + 's', fleet['python'],
                remote / 'persona_coactivity_extract.py', '--root', remote, '--registry-only', '--previous', '-',
                '--workers', str(workers), '--stream']
        if registry: argv += ['--registry', remote / 'results' / registry['name']]
        try:
            raw = transport(remote_command(ssh, argv), previous_raw, timeout=timeout + 15, maximum=MAX_MATRIX_BYTES, stop_event=stop_event)
            snapshot = validate_returned(raw, previous, root, config, expected, sync_receipt['corpus_sha256'])
            added = set(snapshot['arrays']['prompt_ids'].tolist()) - set(previous['arrays']['prompt_ids'].tolist())
            if len(added) > (registry['rows'] if registry else 0): raise ValueError('Returned matrix adds more source rows than the admitted registry')
            if any(file_hash(source_path(root, name)) != digest for name, digest in source_bindings.items()):
                raise ValueError('Local model/map/extractor changed before publication')
            if file_hash(destination) != previous['sha256']: raise ValueError('Local matrix changed before atomic publication')
            atomic_bytes(destination, raw)
        except Exception as error:
            event('map_update_failed', {'status': 'failed', 'map_updated': False, 'error': str(error), 'baseline_registry': registry})
            raise
        metadata = json.loads(str(snapshot['arrays']['metadata']))
        old_ids, new_ids = set(previous['arrays']['prompt_ids'].tolist()), set(snapshot['arrays']['prompt_ids'].tolist())
        receipt = {'status': 'complete', 'map_updated': True, 'source_sha256': snapshot['sha256'],
            'previous_source_sha256': previous['sha256'], 'rows': len(new_ids), 'added_rows': len(new_ids - old_ids),
            'excluded_duplicate_rows': len(old_ids - new_ids), 'source_index_sha256': metadata['source_index_sha256'],
            'corpus_sha256': metadata['corpus_sha256'], 'registries': metadata['registries']}
        event('map_updated', receipt); return receipt


def refresh_coactivity(job, config, summary, event, stop_event=None, *, transport=bounded_process):
    """Call after owned GPU cleanup; stop_event is operator shutdown, not GPU expiry."""
    return _refresh(config, event, stop_event, job, summary, transport=transport)


def refresh_existing(config, event, stop_event=None, *, transport=bounded_process):
    """Audited initial dedup/refresh without a new registry or additional GPU work."""
    return _refresh(config, event, stop_event, transport=transport)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('sync', 'refresh-existing', 'receive-corpus'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    if args.command == 'receive-corpus':
        print(encoded(receive_corpus(args.root, sys.stdin.buffer)).decode()); return
    config = json.loads((args.config or args.root / 'persona_jobs.local.json').read_text())
    event = lambda name, data: print(encoded({'event': name, 'data': data}).decode(), flush=True)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, lambda *_: stop.set())
    (sync_corpus if args.command == 'sync' else refresh_existing)(config, event, stop)


if __name__ == '__main__': main()
