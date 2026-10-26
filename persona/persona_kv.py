"""Checkpoint-bound, faithful hybrid text-prefix cache; no learned dense KV import."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import uuid
import numpy as np
from persona_vectors import canonical_digest

SCHEMA = 'character-creation.hybrid-prefix.v1'
BINDINGS = ('model_identity', 'runtime_identity', 'prepared_identity', 'layout_sha256')


def make_artifact(prefix_text, binding):
    artifact = dict(schema=SCHEMA, kind='hybrid_text_prefix', prefix_text=prefix_text,
                    **{k: binding[k] for k in BINDINGS})
    artifact['sha256'] = canonical_digest(artifact)
    validate_control(dict(artifact=artifact, dose=1, cache_policy='reuse'), binding)
    return artifact


def validate_control(value, binding=None):
    if value is None: return None
    if not isinstance(value, dict) or set(value) != {'artifact', 'dose', 'cache_policy'}:
        raise ValueError('kv_control requires artifact, dose and cache_policy')
    if type(value['dose']) is not int or value['dose'] not in (0, 1):
        raise ValueError('Hybrid prefix dose is exactly 0 (off) or 1 (on); variants are distinct artifacts')
    if value['cache_policy'] not in ('reuse', 'rebuild'):
        raise ValueError('Invalid prefix cache policy')
    artifact = value['artifact']
    keys = {'schema', 'kind', 'prefix_text', 'sha256', *BINDINGS}
    if not isinstance(artifact, dict) or set(artifact) != keys:
        raise ValueError('Invalid hybrid prefix artifact fields')
    if artifact['schema'] != SCHEMA or artifact['kind'] != 'hybrid_text_prefix':
        raise ValueError('Only faithful hybrid text prefixes are supported; dense/attention-only grafts are not qualified')
    if not isinstance(artifact['prefix_text'], str) or not 1 <= len(artifact['prefix_text']) <= 8192:
        raise ValueError('Prefix must contain 1..8192 characters')
    for key in (*BINDINGS, 'sha256'):
        if not isinstance(artifact[key], str) or len(artifact[key]) != 64 or any(c not in '0123456789abcdef' for c in artifact[key]):
            raise ValueError('Invalid prefix binding hash: ' + key)
    if canonical_digest({k: v for k, v in artifact.items() if k != 'sha256'}) != artifact['sha256']:
        raise ValueError('Prefix definition SHA256 mismatch')
    if binding is not None and any(artifact[k] != binding[k] for k in BINDINGS):
        raise ValueError('Prefix model/runtime/preparation/layout binding mismatch')
    return value


def layout_description(pool, page_tokens):
    return {'schema': 'character-creation.hybrid-state.v1', 'page_tokens': page_tokens,
            'row_bytes': pool.layout.row_bytes,
            'regions': [{'name': r.name, 'offset': r.offset, 'shape': list(r.shape),
                         'dtype': str(r.dtype)} for r in pool.layout.regions],
            'pages': {k: {'shape': list(v.shape[1:]), 'dtype': str(v.dtype)}
                      for k, v in sorted(pool.pages.items())},
            'key_space': 'post_normalization_post_rope',
            'sm70_storage': 'uint8 bytes containing uint16 BF16 bit patterns; recurrent FP32',
            'prefix_steering': 'none', 'position_policy': 'prefix IDs followed by unchanged chat-template IDs'}


def file_sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


class HybridPrefixCache:
    """One immutable host cache entry. Never copies sampling/RNG/epoch state."""
    def __init__(self, engine):
        self.engine = engine
        self.layout = layout_description(engine.pool, engine.page_tokens)
        self.binding = {'model_identity': engine.model.identity,
                        'runtime_identity': engine.manifest['runtime_identity'],
                        'prepared_identity': engine.prepared.server_config['identity'],
                        'layout_sha256': canonical_digest(self.layout)}
        self.entry = None

    def capability(self):
        qualified = []
        path = self.engine.output / 'kv-qualification.json'
        if path.is_file():
            try:
                proof = json.loads(path.read_text())
                if (proof.get('passed') is True and proof.get('binding') == self.binding
                        and isinstance(proof.get('artifact_sha256s'), list)):
                    qualified = [s for s in proof['artifact_sha256s'] if isinstance(s, str) and len(s) == 64]
            except (OSError, ValueError): pass
        return {'supported': True, 'kinds': ['hybrid_text_prefix'], 'doses': [0, 1],
                'qualification': 'experimental', 'qualified_artifacts': qualified,
                'layout_sha256': self.binding['layout_sha256'],
                'semantics': 'Exact cached text context including recurrent state; not a trained Phantom-KV graft'}

    def prepare(self, control):
        if control is None: return [], None
        validate_control(control, self.binding)
        if control['dose'] == 0: return [], None
        ids = self.engine.tokenizer.encode(control['artifact']['prefix_text'], add_special_tokens=False)
        if not ids or len(ids) > 1024 or any(type(i) is not int or not 0 <= i < self.engine.vocabulary for i in ids):
            raise ValueError('Prefix must tokenize to 1..1024 valid tokens')
        entry = self.entry
        if control['cache_policy'] != 'reuse' or entry is None or entry['artifact'] != control['artifact']:
            return ids, None
        if entry['ids'] != ids or file_sha(entry['state_artifact']['path']) != entry['state_artifact']['sha256']:
            raise ValueError('Cached prefix token/state artifact changed')
        return ids, entry

    def save(self, artifact, ids, norms):
        engine = self.engine
        pages = (len(ids) + engine.page_tokens - 1) // engine.page_tokens
        arrays = {'state_row': engine.pool.reserved_backing[1].cpu().numpy().copy(),
                  'prefix_ids': np.asarray(ids, dtype=np.int64), 'residual_l2': norms.copy()}
        names = sorted(engine.pool.pages)
        for i, name in enumerate(names): arrays['page_' + str(i)] = engine.pool.pages[name][:pages].cpu().numpy().copy()
        history = getattr(engine.requests, 'ple_history', None)
        if history is not None: arrays['ple_history'] = history[0].cpu().numpy().copy()
        metadata = {'definition': artifact, 'binding': self.binding, 'layout': self.layout,
                    'page_names': names, 'pages': pages, 'prefix_tokens': len(ids)}
        arrays['metadata'] = np.asarray(json.dumps(metadata, sort_keys=True))
        path = engine.output / ('kv-prefix-' + uuid.uuid4().hex + '.npz')
        with path.open('xb') as stream: np.savez(stream, **arrays)
        entry = {'artifact': artifact, 'ids': list(ids), 'arrays': arrays, 'pages': pages,
                 'state_artifact': {'path': str(path.resolve()), 'sha256': file_sha(path)}}
        self.entry = entry
        return entry

    def restore(self, entry):
        import torch
        engine, arrays = self.engine, entry['arrays']
        # Fresh admission has already populated complete prompt penalties and RNG.
        engine.pool.reserved_backing[1].copy_(torch.from_numpy(arrays['state_row']))
        for i, name in enumerate(sorted(engine.pool.pages)):
            engine.pool.pages[name][:entry['pages']].copy_(torch.from_numpy(arrays['page_' + str(i)]))
        if 'ple_history' in arrays:
            engine.requests.ple_history[0].copy_(torch.from_numpy(arrays['ple_history']))
        engine.requests.consumed_tokens[0] = len(entry['ids'])

    def result(self, control, ids, entry, hit):
        if control is None: return None
        return {**self.binding, 'artifact_sha256': control['artifact']['sha256'],
                'kind': 'hybrid_text_prefix', 'dose': control['dose'], 'cache_policy': control['cache_policy'], 'prefix_tokens': len(ids),
                'cache_hit': hit, 'state_artifact': entry['state_artifact'] if entry else None,
                'prefix_steering': 'none', 'attention_only': False}
