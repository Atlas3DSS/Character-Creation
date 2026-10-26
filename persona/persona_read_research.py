"""Read a pinned public research snapshot without loading Qwen or running inference."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

REPO = 'Atlas3D/character-creation-big-five'


def checked_name(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('Expected a flat manifest filename')
    return name


def verify(path, expected):
    if not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError('Expected SHA-256')
    with Path(path).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError(f'Artifact hash differs: {Path(path).name}')


class Snapshot:
    def __init__(self, revision=None, repo=REPO):
        from huggingface_hub import HfApi
        self.repo = repo
        self.revision = HfApi().dataset_info(repo, revision=revision).sha
        self.manifest = json.loads(self.download('research_manifest.json').read_text())
        if self.manifest.get('schema') != 'character-creation.public-research.v1':
            raise ValueError('Unsupported research manifest schema')

    def download(self, filename, sha256=None):
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(self.repo, checked_name(filename), repo_type='dataset', revision=self.revision))
        if sha256 is not None:
            verify(path, sha256)
        return path

    def records(self, kind=None):
        seen = set()
        for batch in self.manifest['batches']:
            if kind and kind not in batch.get('kinds', []):
                continue
            path = self.download(batch['file'], batch['sha256'])
            found = []
            with path.open() as stream:
                for line in stream:
                    record = json.loads(line)
                    identity = record['id']
                    if identity in seen:
                        raise ValueError('Duplicate research record ID')
                    seen.add(identity)
                    found.append(identity)
                    if kind is None or record['kind'] == kind:
                        yield record
            if len(found) != batch['records'] or set(found) != set(batch['record_ids']):
                raise ValueError('Batch record coverage differs')

    def arrays(self, record_id):
        """Explicitly download one raw shard; never extract tar entries to disk."""
        import numpy as np
        raw = json.loads(self.download(self.manifest.get('raw_manifest', 'research_raw_manifest.json')).read_text())
        if raw.get('schema') != 'character-creation.public-research-raw.v1':
            raise ValueError('Unsupported raw manifest schema')
        batch = next((b for b in raw['batches'] if record_id in b['record_ids']), None)
        if batch is None:
            raise KeyError('Raw arrays are not published for this record at the pinned revision')
        entries = json.loads(self.download(batch['entry_manifest'], batch['entry_manifest_sha256']).read_text())
        entry = next(row for row in entries['entries'] if row['id'] == record_id)
        archive = self.download(batch['file'], batch['sha256'])
        with tarfile.open(archive, 'r:*') as tar:
            member = tar.getmember(checked_name(entry['entry']))
            if not member.isfile():
                raise ValueError('Raw artifact must be a regular file')
            with tar.extractfile(member) as stream:
                content = stream.read()
        if hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise ValueError('Raw entry hash differs')
        with np.load(io.BytesIO(content), allow_pickle=False) as values:
            arrays = {key: values[key].copy() for key in values.files}
        for key, spec in entry['arrays'].items():
            if key not in arrays or list(arrays[key].shape) != spec['shape'] or str(arrays[key].dtype) != spec['dtype']:
                raise ValueError('Raw array shape or dtype differs')
        return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', help='Hub commit SHA; omitted resolves and prints the current SHA once')
    parser.add_argument('--kind', choices=('prefill_activation', 'baseline_answer', 'qualified_labels', 'coherence'))
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--raw-record', help='Explicitly download the raw shard containing this record')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    snapshot = Snapshot(args.revision)
    print(json.dumps({'repo': snapshot.repo, 'revision': snapshot.revision, 'summary': snapshot.manifest.get('summary')}))
    if args.raw_record:
        arrays = snapshot.arrays(args.raw_record)
        print(json.dumps({key: {'shape': list(value.shape), 'dtype': str(value.dtype)} for key, value in arrays.items()}))
    else:
        import itertools
        for record in itertools.islice(snapshot.records(args.kind), args.limit):
            print(json.dumps({'id': record['id'], 'kind': record['kind'], 'source': record.get('source'),
                              'coverage': record.get('coverage')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
