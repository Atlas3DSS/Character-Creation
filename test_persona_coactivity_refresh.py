"""CPU-only bounded transport, append sync and atomic matrix publication checks."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from persona_coactivity_extract import (duplicate_audit, file_hash, matrix_snapshot,
    preserve_previous, unique_rows, write_matrix)
from persona_coactivity_refresh import (FILES, SOURCES, atomic_bytes, bounded_process, corpus_payload,
    receive_corpus, refresh_coactivity, refresh_existing, settings, validate_returned)
from persona_vectors import ModelSpec, canonical_digest


class CoactivityRefreshTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.remote_temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.remote_temporary.cleanup)
        self.root, self.remote = Path(self.temporary.name), Path(self.remote_temporary.name)
        for root in (self.root, self.remote):
            (root / 'data').mkdir(); (root / 'results').mkdir()
        self.model = ModelSpec('fixture', 'a' * 40, 'test', 2, 3, 4)
        (self.root / 'data/persona_model.json').write_text(json.dumps(self.model.__dict__))
        self.chart = {'model_identity': self.model.identity, 'neurons': [{'layer': 0, 'neuron': 1}]}
        (self.root / 'results/persona_map.json').write_text(json.dumps(self.chart))
        for name in SOURCES: (self.root / name).write_text('reviewed fixture source ' + name)
        self.config = {'root': str(self.root), 'fleet_config': str(self.root / 'fleet.json'),
            'expected_prepared_identity': 'b' * 64, 'expected_runtime_identity': 'c' * 64}
        self.fleet = {'hostname': 'v100', 'ssh_host': 'v100', 'remote_root': str(self.remote), 'python': '/fixed/python'}
        (self.root / 'fleet.json').write_text(json.dumps(self.fleet))
        self.identity = {'model_identity': self.model.identity, 'prepared_identity': 'b' * 64, 'runtime_identity': 'c' * 64}
        def row(marker, text, split='discovery'):
            item = {'pair_id': marker, 'corpus': 'big_five', 'split': split, 'category': 'fixture',
                'prompt': text, 'system': '', 'max_tokens': 2048, 'mutation': {'kind': 'original'},
                'labels': {'profile': 'baseline'}, 'source_sha256': 'd' * 64}
            item['id'] = canonical_digest(item); return item
        self.rows = [row('one', 'One original prompt.'), row('two', 'One original prompt.'),
                     row('new', 'A different original prompt.'), row('heldout', 'Untouched heldout prompt.', 'heldout')]
        self.publish_corpus(self.remote, self.rows[:2])
        self.publish_corpus(self.root, self.rows)
        self.old_metadata = {'schema': 'character-creation.coactivity.v1', 'model_identity': self.model.identity,
            'map_sha256': file_hash(self.root / 'results/persona_map.json'), 'split': 'discovery',
            'measurement': 'final_prefill_mlp_input_last', 'unsteered': True,
            'runtime_identities': ['c' * 64], 'prepared_identities': ['b' * 64],
            'corpus_sha256': file_hash(self.remote / 'data/persona_manager_corpus.jsonl'), 'created': 1.}
        self.arrays = dict(metadata=np.array(json.dumps(self.old_metadata)), layers=np.array([0]), neurons=np.array([1]),
            values=np.array([[2.], [2.]], dtype=np.float32), corpora=np.array(['big_five', 'big_five']),
            prompt_ids=np.array([row['id'] for row in self.rows[:2]]), source_hashes=np.array(['e' * 64, 'f' * 64]))
        self.matrix = self.root / 'results/persona_coactivity.npz'; write_matrix(self.matrix, self.arrays)
        self.original = self.matrix.read_bytes()
        self.job = {'id': '1' * 32, 'kind': 'neighborhood_hillclimb', 'deadline': 0}
        self.registry_name = 'hillclimb-' + self.job['id'] + '-observations.jsonl'
        self.summary = {'baseline_registry': {'path': str(self.remote / 'results' / self.registry_name),
            'sha256': 'a' * 64, 'rows': 1, 'status': 'registered_for_later_coactivity_ingestion', 'map_updated': False}}
        self.events, self.calls = [], []

    def publish_corpus(self, root, rows):
        path = root / 'data' / FILES[0]
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        path.with_name(FILES[1]).write_text(json.dumps({'status': 'complete', 'output_sha256': file_hash(path)}))

    def returned(self, raw, with_registry=True):
        previous = matrix_snapshot(raw); old = previous['arrays']
        corpus = {row['id']: row for row in self.rows}
        inputs = [({'prompt': corpus[identity]}, {'activation_artifact': {'sha256': source}})
                  for identity, source in zip(old['prompt_ids'].tolist(), old['source_hashes'].tolist())]
        vectors = dict(zip(old['prompt_ids'].tolist(), old['values']))
        if with_registry:
            inputs.append(({'prompt': self.rows[2]}, {'activation_artifact': {'sha256': 'a' * 64}}))
            vectors[self.rows[2]['id']] = np.array([3.], dtype=np.float32)
        selected, count = unique_rows(inputs, self.identity, set(old['prompt_ids'].tolist()))
        values = np.stack([vectors[job['prompt']['id']] for job, _ in selected])
        audit = duplicate_audit(inputs, selected, values, self.identity, lambda row: vectors[row[0]['prompt']['id']], 2)
        ids, sources = [job['prompt']['id'] for job, _ in selected], [result['activation_artifact']['sha256'] for _, result in selected]
        metadata = {**self.old_metadata, 'corpus_sha256': file_hash(self.root / 'data' / FILES[0]),
            'source_index_sha256': canonical_digest(list(zip(ids, sources))), 'corpus_counts': {'big_five': len(ids)},
            'extraction_sha256': file_hash(self.root / 'persona_coactivity_extract.py'),
            'registries': [{'name': self.registry_name, 'sha256': 'a' * 64, 'eligible_records': 1}] if with_registry else [],
            'duplicate_observations': audit, 'deduplicated_observations': count}
        arrays = {**old, 'metadata': np.array(json.dumps(metadata)), 'values': values,
            'prompt_ids': np.array(ids), 'source_hashes': np.array(sources), 'corpora': np.array(['big_five'] * len(ids))}
        metadata['history'] = preserve_previous(previous, arrays, audit)
        arrays['metadata'] = np.array(json.dumps(metadata))
        output = io.BytesIO(); np.savez_compressed(output, **arrays)
        return output.getvalue()

    def transport(self, argv, payload=b'', **kwargs):
        self.calls.append((argv, kwargs)); command = shlex.split(argv[-1])
        if command == ['hostname']: return b'v100\n'
        if command[0] == 'sha256sum':
            names = (*SOURCES, 'data/persona_model.json', 'results/persona_map.json')
            return ''.join(file_hash(self.root / name) + '  ' + name + '\n' for name in names).encode()
        if 'receive-corpus' in command: return json.dumps(receive_corpus(self.remote, io.BytesIO(payload))).encode()
        self.assertIn('--registry-only', command); self.assertEqual(command[command.index('--previous') + 1], '-')
        self.assertEqual(payload, self.original)
        return self.returned(payload, with_registry='--registry' in command)

    def test_completed_registry_refresh_publishes_validated_matrix_after_gpu_deadline(self):
        receipt = refresh_coactivity(self.job, self.config, self.summary,
            lambda name, value: self.events.append((name, value)), transport=self.transport)
        self.assertTrue(receipt['map_updated']); self.assertEqual(receipt['rows'], 2)
        self.assertEqual(receipt['added_rows'], 1); self.assertEqual(receipt['excluded_duplicate_rows'], 1)
        self.assertEqual(file_hash(self.matrix), receipt['source_sha256'])
        self.assertEqual((self.remote / 'data' / FILES[0]).read_bytes(), (self.root / 'data' / FILES[0]).read_bytes())
        self.assertEqual(self.events[-1][0], 'map_updated')
        self.assertEqual(sum('--registry-only' in call[0][-1] for call in self.calls), 1)

    def test_initial_refresh_performs_audited_dedup_without_registry_or_gpu(self):
        receipt = refresh_existing(self.config, lambda *args: self.events.append(args), transport=self.transport)
        self.assertEqual(receipt['rows'], 1); self.assertEqual(receipt['excluded_duplicate_rows'], 1)
        metadata = json.loads(str(matrix_snapshot(self.matrix)['arrays']['metadata']))
        self.assertEqual(len(metadata['history'][-1]['deduplication_transition']), 1)
        self.assertTrue(metadata['history'][-1]['raw_captures_preserved'])

    def test_actual_extractor_cli_reads_previous_from_stdin_without_manager_database(self):
        script = Path(__file__).resolve().with_name('persona_coactivity_extract.py')
        output = bounded_process([sys.executable, str(script), '--root', str(self.root), '--previous', '-',
            '--registry-only', '--stream', '--workers', '2'], self.original, timeout=10)
        snapshot = matrix_snapshot(output); metadata = json.loads(str(snapshot['arrays']['metadata']))
        self.assertEqual(len(snapshot['arrays']['prompt_ids']), 1)
        self.assertEqual(len(metadata['history'][-1]['deduplication_transition']), 1)
        self.assertEqual(metadata['captured_corpus_sha256'], [self.old_metadata['corpus_sha256']])
        self.assertEqual(self.matrix.read_bytes(), self.original)  # --stream never publishes on the extraction host.

    def test_changed_returned_binding_never_replaces_existing_matrix(self):
        def transport(argv, payload=b'', **kwargs):
            raw = self.transport(argv, payload, **kwargs)
            if '--registry-only' not in argv[-1]: return raw
            arrays = matrix_snapshot(raw)['arrays']; metadata = json.loads(str(arrays['metadata']))
            metadata['runtime_identities'] = ['0' * 64]; arrays['metadata'] = np.array(json.dumps(metadata))
            output = io.BytesIO(); np.savez_compressed(output, **arrays); return output.getvalue()
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            refresh_coactivity(self.job, self.config, self.summary, lambda *args: self.events.append(args), transport=transport)
        self.assertEqual(self.matrix.read_bytes(), self.original)
        self.assertEqual(self.events[-1][0], 'map_update_failed')

    def test_model_paths_and_cross_job_registry_are_rejected_before_transport(self):
        bad = deepcopy(self.summary); bad['baseline_registry']['path'] = str(self.remote / 'results' / 'outside.jsonl')
        with self.assertRaisesRegex(ValueError, 'fixed result path'):
            refresh_coactivity(self.job, self.config, bad, lambda *_: None, transport=self.transport)
        self.assertFalse(self.calls)
        self.fleet['ssh_host'] = 'v100; arbitrary-command'
        (self.root / 'fleet.json').write_text(json.dumps(self.fleet))
        with self.assertRaises(ValueError): settings(self.config)

    def test_operator_stop_keeps_registry_and_returns_explicit_skipped_receipt(self):
        stop = threading.Event(); stop.set()
        receipt = refresh_coactivity(self.job, self.config, self.summary,
            lambda *args: self.events.append(args), stop, transport=self.transport)
        self.assertEqual(receipt['status'], 'skipped'); self.assertFalse(receipt['map_updated'])
        self.assertEqual(receipt['baseline_registry']['sha256'], 'a' * 64)
        self.assertFalse(self.calls); self.assertEqual(self.matrix.read_bytes(), self.original)

    def test_sync_is_append_only_hash_checked_and_recovers_owned_interruption(self):
        payload, _ = corpus_payload(self.root)
        corpus = self.remote / 'data' / FILES[0]; original, inode = corpus.read_bytes(), corpus.stat().st_ino
        actual_atomic = atomic_bytes
        def fail_complete(path, raw):
            if path.name == FILES[1] and json.loads(raw)['status'] == 'complete': raise RuntimeError('interrupted sync')
            actual_atomic(path, raw)
        with patch('persona_coactivity_refresh.atomic_bytes', fail_complete), self.assertRaisesRegex(RuntimeError, 'interrupted'):
            receive_corpus(self.remote, io.BytesIO(payload))
        self.assertTrue(corpus.read_bytes().startswith(original))
        receipt = receive_corpus(self.remote, io.BytesIO(payload))
        self.assertEqual(receipt['status'], 'complete'); self.assertEqual(corpus.stat().st_ino, inode)
        self.publish_corpus(self.root, [self.rows[2]])
        with self.assertRaisesRegex(ValueError, 'preserve every existing'):
            receive_corpus(self.remote, io.BytesIO(corpus_payload(self.root)[0]))
        corrupt = bytearray(payload); corrupt[-1] ^= 1
        with self.assertRaisesRegex(ValueError, 'digest'):
            receive_corpus(self.remote, io.BytesIO(corrupt))

    def test_transport_bounds_stdout_and_time_and_echoes_stdin(self):
        result = bounded_process([sys.executable, '-c', 'import sys;sys.stdout.buffer.write(sys.stdin.buffer.read())'], b'bound input', timeout=2, maximum=100)
        self.assertEqual(result, b'bound input')
        with self.assertRaisesRegex(ValueError, 'bounded size'):
            bounded_process([sys.executable, '-c', 'import sys;sys.stdout.buffer.write(b"x"*10000)'], timeout=2, maximum=10)
        with self.assertRaises(TimeoutError):
            bounded_process([sys.executable, '-c', 'import time;time.sleep(10)'], timeout=.1, maximum=10)

    def test_child_exit_during_cleanup_does_not_mask_the_original_bounded_output_failure(self):
        killpg = os.killpg
        def exit_race(pid, signal):
            killpg(pid, signal)
            raise ProcessLookupError('owned child exited immediately')
        with patch('persona_coactivity_refresh.os.killpg', side_effect=exit_race) as cleanup:
            with self.assertRaisesRegex(ValueError, 'bounded size'):
                bounded_process([sys.executable, '-c',
                    'import sys,time;sys.stdout.buffer.write(b"x"*10000);sys.stdout.flush();time.sleep(10)'], timeout=2, maximum=10)
        self.assertTrue(cleanup.called)


if __name__ == '__main__': unittest.main()
