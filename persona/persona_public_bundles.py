"""Prepare reviewed, source-bound public research bundles; never upload implicitly."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3

import numpy as np

SCHEMA = 'character-creation.public-research-bundle.v1'
PRIVATE_KEYS = {'path', 'local_path', 'endpoint', 'endpoints', 'endpoint_receipts', 'url',
                'ssh_host', 'hostname', 'remote_root', 'artifact_root', 'worker',
                'api_key', 'access_token', 'authorization', 'token', 'password', 'secret'}
PRIVATE_TEXT = re.compile(r'(?:/home/|/mnt/|/tmp/|https?://(?:127\.0\.0\.1|localhost)|hf_[A-Za-z0-9]{20,})')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sanitized(value):
    """Keep scientific data, remove deployment metadata, reject unexpected local text."""
    if isinstance(value, dict):
        return {key: sanitized(item) for key, item in value.items()
                if key.lower() not in PRIVATE_KEYS and not key.lower().endswith('_path')}
    if isinstance(value, list):
        return [sanitized(item) for item in value]
    if isinstance(value, str) and PRIVATE_TEXT.search(value):
        raise ValueError('Unexpected private string in reviewed research evidence')
    return value


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                                    allow_nan=False) + '\n')


def make_bundle(identity, kind, identifier, payload):
    return {'schema': SCHEMA, 'id': identifier, 'kind': kind, 'state': 'complete',
            'model_identity': identity, 'validation': {'passed': True,
            'meaning': 'Public export structure and source bindings checked; individual scientific outcomes may fail.'},
            'payload': sanitized(payload)}


def prepare(root):
    root = Path(root).resolve()
    registry = json.loads((root / 'results/persona_neuron_labels.json').read_text())
    identity = registry['model_identity']
    label_source = root / 'data/persona_label_probes.json'
    proofs = []
    for name in ('joy', 'sadness', 'sexual_desire', 'confusion', 'green', 'dog'):
        path = root / f'results/persona_label_evidence_{name}.json'
        proof = json.loads(path.read_text())
        if proof['model_identity'] != identity or proof['frozen_candidate']['concept'] != name:
            raise ValueError('Label source identity differs')
        if proof['binding']['input_file_sha256'] != sha(label_source):
            raise ValueError('Label probe source changed')
        proofs.append({'original_evidence_sha256': sha(path), 'evidence': proof})
    labels = make_bundle(identity, 'qualified_labels', 'label-qualification-432', {
        'probes': json.loads(label_source.read_text()), 'source_sha256': sha(label_source),
        'evidence': proofs, 'scope': 'All six completed qualification outcomes, including failures. '
        'Precision is conditional on authored probes, not a probability of universal neuron meaning. '
        'Deployment paths were removed; original evidence hashes identify the retained private originals.'})
    labels_path = root / 'results/persona_public_labels.json'
    write(labels_path, labels)

    suite_path = root / 'data/persona_coherence_suite.json'
    suite = json.loads(suite_path.read_text())
    runs = []
    with sqlite3.connect((root / 'results/persona_explorer.sqlite').as_uri() + '?mode=ro', uri=True) as db:
        for (raw,) in db.execute("SELECT body FROM coherence_runs WHERE state='complete' ORDER BY created"):
            run = json.loads(raw)
            if run['model_identity'] != identity or run['suite'] != suite:
                raise ValueError('Coherence source differs from reviewed synthetic suite')
            runs.append(run)
    coherence = make_bundle(identity, 'coherence', 'coherence-completed-source-contracts', {
        'suite': suite, 'suite_file_sha256': sha(suite_path), 'runs': runs,
        'scope': 'Completed trials, including heldout failures and representation uncertainty. '
        'Cancelled pilot excluded; reused outputs retain provenance and are not independent observations. '
        'No default is promoted merely because a bundle is exported.'})
    coherence_path = root / 'results/persona_public_coherence.json'
    write(coherence_path, coherence)

    out = root / 'dataset'
    out.mkdir(exist_ok=True)
    assets = []
    for layer in range(64):
        source = root / f'results/layer_{layer:02d}.npz'
        with np.load(source, allow_pickle=False) as arrays:
            if str(arrays['model_identity']) != identity:
                raise ValueError('Calibration model identity differs')
            for key in arrays.files:
                value = arrays[key]
                if value.dtype.kind in 'OU':
                    for text in value.flat:
                        sanitized(str(text))
                elif not np.isfinite(value).all():
                    raise ValueError('Nonfinite calibration data')
        destination = out / source.name
        if not destination.exists() or sha(destination) != sha(source):
            shutil.copyfile(source, destination)
        assets.append({'file': source.name, 'sha256': sha(source), 'bytes': source.stat().st_size,
                       'kind': 'model_bound_calibration', 'layer': layer})
    for name in ('persona_model.json', 'persona_presets.json', 'persona_adult_sexuality.json',
                 'persona_coherence_suite.json', 'persona_emotion_probe_graph.json',
                 'persona_label_probes.json', 'persona_trait_behavior_probes.json',
                 'persona_reference_neuron_recipes.json', 'persona_preset_candidates.json'):
        source = root / 'data' / name
        sanitized(json.loads(source.read_text()))
        shutil.copyfile(source, out / name)
        assets.append({'file': name, 'sha256': sha(source), 'bytes': source.stat().st_size,
                       'kind': 'authored_suite_or_model_metadata'})
    # Portable map/weight-lens assets contain model-derived numeric data and vocab,
    # not private source documents. Preserve hashes so the app can verify them.
    atlas_path = root / 'results/persona_atlas_manifest.json'
    atlas = json.loads(atlas_path.read_text())
    if atlas['model_identity'] != identity:
        raise ValueError('Weight atlas model identity differs')
    extra = ['persona_map.json', 'persona_map_geometry.json', 'persona_atlas_manifest.json']
    for name, expected in atlas['files'].items():
        if not re.fullmatch(r'persona_atlas_(?:L\d+\.npz|tokens\.json)', name):
            raise ValueError('Unexpected atlas source filename')
        source = root / 'results' / name
        if sha(source) != expected:
            raise ValueError('Weight atlas source hash differs')
        if name.endswith('.npz'):
            with np.load(source, allow_pickle=False) as arrays:
                metadata = json.loads(str(arrays['metadata']))
                sanitized(metadata)
                if metadata['model_identity'] != identity:
                    raise ValueError('Weight atlas shard model differs')
        extra.append(name)
    for name in extra:
        source = root / 'results' / name
        if name.endswith('.json') and name != 'persona_atlas_tokens.json':
            sanitized(json.loads(source.read_text()))
        destination = out / name
        if not destination.exists() or sha(destination) != sha(source):
            shutil.copyfile(source, destination)
        assets.append({'file': name, 'sha256': sha(source), 'bytes': source.stat().st_size,
                       'kind': 'model_bound_map_or_weight_atlas'})
    write(out / 'research_assets.json', {'schema': 'character-creation.public-research-assets.v1',
        'model_identity': identity, 'files': assets,
        'calibration_scope': '64 layer calibration banks; trait directions and selected down-projection '
        'columns, not full model weights. Original NPZ bytes and hashes preserved. Use only with the '
        'bound checkpoint/runtime and verify calibration before steering.',
        'license_scope': 'The CC-BY-4.0 prompt license does not replace upstream checkpoint terms '
        'for derived model columns and calibration assets.'})
    return [{'path': str(path.relative_to(root)), 'sha256': sha(path), 'kind': kind}
            for path, kind in [(labels_path, 'qualified_labels'), (coherence_path, 'coherence')]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps({'reviewed_bundles': prepare(args.root)}, indent=2))


if __name__ == '__main__':
    main()
