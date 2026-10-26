"""Strict learned attention-KV artifacts and SM70 page injection; no text prefix."""
import json
from pathlib import Path
import re
import zipfile
import numpy as np
from persona_vectors import canonical_digest

SCHEMA = 'character-creation.learned-attention-kv.v1'
LAYERS = tuple(range(3, 64, 4))
BINDINGS = ('model_identity', 'runtime_identity', 'prepared_identity', 'layout_sha256')


def file_sha(path):
    import hashlib
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_control(control, binding=None):
    if not isinstance(control, dict) or set(control) != {'artifact', 'dose', 'cache_policy'}:
        raise ValueError('Invalid learned-KV control fields')
    a = control['artifact']
    if not isinstance(a, dict) or set(a) != {'schema', 'kind', 'file', 'weights_sha256', 'slots', 'sha256', *BINDINGS}:
        raise ValueError('Invalid learned-KV artifact fields')
    if a['schema'] != SCHEMA or a['kind'] != 'learned_attention_kv': raise ValueError('Unsupported learned-KV schema')
    if not isinstance(a['file'], str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,120}\.npz', a['file']):
        raise ValueError('Learned-KV file must be a flat artifact filename')
    if type(a['slots']) is not int or not 1 <= a['slots'] <= 128: raise ValueError('Invalid learned-KV slot count')
    if any(not isinstance(a[k], str) or not re.fullmatch('[a-f0-9]{64}', a[k]) for k in (*BINDINGS, 'sha256', 'weights_sha256')):
        raise ValueError('Invalid learned-KV identity hash')
    if canonical_digest({k:v for k,v in a.items() if k != 'sha256'}) != a['sha256']:
        raise ValueError('Learned-KV definition SHA256 mismatch')
    if binding is not None and any(a[k] != binding[k] for k in BINDINGS): raise ValueError('Learned-KV runtime binding mismatch')
    if type(control['dose']) is not int or control['dose'] not in (0, 1) or control['cache_policy'] not in ('reuse', 'rebuild'):
        raise ValueError('Learned-KV supports exact off/on; continuous strength is not qualified')
    return control


def verified_artifacts(output, filename, binding):
    """A marker alone cannot qualify an artifact; verify its linked report bytes."""
    try:
        proof = json.loads((Path(output) / filename).read_text())
        path = Path(proof['report']['path']).resolve()
        if path.parent != Path(output).resolve() or path.stat().st_size > 16 << 20:
            return []
        if file_sha(path) != proof['report']['sha256']: return []
        report = json.loads(path.read_text())
        if any(value.get('passed') is not True or value.get('binding') != binding for value in (proof, report)):
            return []
        values = proof['artifact_sha256s']
        if report.get('artifact_sha256s') != values or not isinstance(values, list): return []
        return [v for v in values if isinstance(v, str) and re.fullmatch('[a-f0-9]{64}', v)]
    except (OSError, ValueError, KeyError, TypeError): return []


def training_validation(metadata, output):
    """Verify the immutable admission report, not a nullable copied status flag."""
    try:
        pointer = metadata['admission_validation']; path = Path(pointer['path']).resolve()
        if path.parent != Path(output).resolve() or path.stat().st_size > 16 << 20 or file_sha(path) != pointer['sha256']:
            raise ValueError('Learned-KV training validation path/hash differs')
        report = json.loads(path.read_text())
        fields = ('model_identity','dataset_sha256','trainer_sha256','reader_sha256','seed','slots','max_tokens',
                  'weight_dtype','training_backend','gradient_protocol','checkpoint')
        if (any(key not in metadata or report.get(key) != metadata[key] for key in fields)
                or report.get('state') != 'validation_passed' or report.get('native_forward',{}).get('passed') is not True):
            raise ValueError('Learned-KV training validation binding differs')
        checked = report['validation']; rows = checked['measurements']; protocol = metadata['gradient_protocol']
        if (checked.get('passed') is not True or checked.get('protocol') != protocol or not rows
                or [r['epsilon'] for r in rows] != protocol['epsilons']
                or any(r.get('passed') is not True or r.get('positive_direction') is not True
                       or not 0 <= r['relative_error'] <= .12 or not r['numerical_directional_derivative'] > 0 for r in rows)):
            raise ValueError('Learned-KV gradient validation did not pass')
        return checked
    except (OSError, KeyError, TypeError) as error:
        raise ValueError('Missing or malformed learned-KV training validation') from error


class LearnedKVCache:
    def __init__(self, engine, binding):
        self.engine, self.binding, self.entry = engine, binding, None

    def prepare(self, control):
        validate_control(control, self.binding)
        if control['dose'] == 0: return 0, None, False
        a = control['artifact']; path = (self.engine.output / a['file']).resolve()
        if path.parent != self.engine.output.resolve() or file_sha(path) != a['weights_sha256']:
            raise ValueError('Learned-KV artifact path/byte binding differs')
        if control['cache_policy'] == 'reuse' and self.entry and self.entry['artifact'] == a:
            return a['slots'], self.entry, True
        with zipfile.ZipFile(path) as z:
            if sum(i.file_size for i in z.infolist()) > 32 << 20: raise ValueError('Learned-KV artifact exceeds size bound')
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {'keys', 'values', 'attention_layers', 'metadata'}: raise ValueError('Unexpected learned-KV arrays')
            metadata = json.loads(str(data['metadata']))
            training_validation(metadata, self.engine.output)
            if (metadata.get('schema') != SCHEMA or metadata.get('model_identity') != self.binding['model_identity']
                    or metadata.get('key_space') != 'post_norm_post_rope' or metadata.get('prompt_position_start') != a['slots']
                    or metadata.get('recurrent_initial_state') != 'fresh_zero'
                    or type(metadata.get('optimizer_steps')) is not int or metadata['optimizer_steps'] < 1
                    or not np.array_equal(data['attention_layers'], np.asarray(LAYERS))):
                raise ValueError('Learned-KV training/architecture provenance differs')
            for key, native in (('config_sha256', 'checkpoint_config_sha256'), ('index_sha256', 'checkpoint_index_sha256')):
                if metadata.get('checkpoint', {}).get(key) != self.engine.manifest[native]:
                    raise ValueError('Learned-KV checkpoint binding differs')
            arrays = {}
            for key in ('keys', 'values'):
                value = data[key]
                if (value.shape != (16, 4, a['slots'], 256) or value.dtype != np.float32 or not np.isfinite(value).all()
                        or np.any(np.abs(value) > (2 - 2**-7) * 2**127)):
                    raise ValueError('Invalid finite learned-KV tensor')
                arrays[key] = value.copy()
        self.entry = {'artifact': a, 'arrays': arrays, 'metadata': metadata,
                      'state_artifact': {'path': str(path), 'sha256': a['weights_sha256']}}
        return a['slots'], self.entry, False

    def install(self, entry):
        import torch
        e = self.engine; slots = entry['artifact']['slots']; pages = (slots + e.page_tokens - 1) // e.page_tokens
        # Admission may leave lazy-reset recurrent bytes. A virtual prefix must
        # start from actual zero GDN/convolution state before setting its offset.
        for layer in LAYERS:
            for field in ('key', 'value'):
                page = e.pool.pages[f'layer{layer}.{field}']
                if page.dtype != torch.uint8 or page.shape[0] < pages or page.shape[1:] != (e.page_tokens * 4 * 256 * 2,):
                    raise ValueError('Learned-KV requires the inspected SM70 BF16-bit page layout')
        e.pool.reserved_backing[1].zero_()
        for index, layer in enumerate(LAYERS):
            for field, array in (('key', 'keys'), ('value', 'values')):
                target = e.pool.pages[f'layer{layer}.{field}'][:pages]
                target.zero_()
                view = target.view(torch.bfloat16).view(-1, 4, 256)
                view[:slots].copy_(torch.from_numpy(entry['arrays'][array][index]).transpose(0, 1))
        e.pool.reserved_region('consumed_tokens')[1].fill_(slots)
        e.requests.consumed_tokens[0] = slots

    def result(self, control, slots, entry, hit):
        return {**self.binding, 'artifact_sha256': control['artifact']['sha256'], 'kind': 'learned_attention_kv',
                'dose': control['dose'], 'cache_policy': control['cache_policy'], 'prefix_tokens': slots,
                'virtual_prefix_slots': slots, 'cache_hit': hit, 'state_artifact': entry['state_artifact'] if entry else None,
                'prefix_steering': 'none', 'attention_only': True, 'recurrent_initial_state': 'fresh_zero'}
