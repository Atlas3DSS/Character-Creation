"""Read ModelOpt Qwen3.5 packed weights into one frozen tensor at a time.

No expanded checkpoint is written. NVFP4 decoding follows Flare's
ignition/reference/qwen38_quant.py: low nibble first, E2M1 times row-major
E4M3 group16 scales times FP32 global scale. Tensor-FP8 uses weight_scale,
NOT input_scale. This decodes weights; it does not reproduce serving activation
quantization, post-dot scale placement, BF16 rounding, or reduction order.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import struct

TEXT_PREFIX = 'model.language_model.'
FP4_VALUES = (0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.)
DTYPES = {'BF16': 'bfloat16', 'F16': 'float16', 'F32': 'float32', 'F64': 'float64',
          'F8_E4M3': 'float8_e4m3fn', 'U8': 'uint8', 'I64': 'int64'}
ITEM_BYTES = {'BF16': 2, 'F16': 2, 'F32': 4, 'F64': 8, 'F8_E4M3': 1, 'U8': 1, 'I64': 8}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def decode_nvfp4(packed, scales, global_scale):
    """Portable FP32 dequantization, bit-exact with Flare's CPU weight reference."""
    import torch
    if packed.dtype != torch.uint8 or packed.ndim != 2 or packed.shape[1] % 8:
        raise ValueError('NVFP4 requires a rank-two uint8 matrix with group16 width')
    if scales.dtype != torch.float8_e4m3fn or scales.shape != (packed.shape[0], packed.shape[1] // 8):
        raise ValueError('NVFP4 requires row-major E4M3 group16 scales')
    if not packed.device == scales.device == global_scale.device:
        raise ValueError('Quantized components must share a device')
    _positive_scalar(global_scale, 'weight_scale_2')
    if not torch.isfinite(scales.float()).all() or (scales.float() < 0).any():
        raise ValueError('NVFP4 block scales must be finite and nonnegative')
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    values = lut[codes].reshape(packed.shape[0], -1, 16)
    return (values * scales.float().unsqueeze(-1) * global_scale.reshape(())).flatten(-2)


def _positive_scalar(value, name):
    import torch
    if value.dtype != torch.float32 or value.numel() != 1 or not torch.isfinite(value).all() or value.item() <= 0:
        raise ValueError(name + ' must be one positive finite FP32 scalar')


class PackedCheckpoint:
    """Metadata-only construction; tensor() owns returned memory, with bounded scratch.

    Names may be checkpoint-native, bare TextModel names, or model.layers.* HF
    CausalLM names. dtype=None preserves unquantized dtypes and decodes quantized
    weights to FP32. Pass dtype explicitly for a chosen training surrogate.
    """
    def __init__(self, path, *, block_rows=256):
        if type(block_rows) is not int or not 1 <= block_rows <= 65536:
            raise ValueError('block_rows must be an integer in 1..65536')
        self.path, self.block_rows = Path(path).resolve(), block_rows
        config_bytes = (self.path/'config.json').read_bytes()
        index_bytes = (self.path/'model.safetensors.index.json').read_bytes()
        self._metadata_stamps = {name: self._stamp((self.path/name).stat()) for name in ('config.json', 'model.safetensors.index.json')}
        self._config = json.loads(config_bytes)
        if self._config.get('text_config', {}).get('model_type') != 'qwen3_5_text':
            raise ValueError('Expected Qwen3.5 text configuration')
        quant = self._config.get('quantization_config', {})
        if quant.get('quant_method') != 'modelopt' or not isinstance(quant.get('quantized_layers'), dict):
            raise ValueError('Expected explicit ModelOpt quantized_layers metadata')
        self._quant = copy.deepcopy(quant['quantized_layers'])
        self._index = json.loads(index_bytes)['weight_map']
        if not isinstance(self._index, dict) or not self._index:
            raise ValueError('Empty or invalid checkpoint index')
        self._tensors, self._stamps, headers = {}, {}, {}
        for filename in sorted(set(self._index.values())):
            if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith('.safetensors'):
                raise ValueError('Shard names must be flat safetensors filenames')
            shard = self.path/filename
            if shard.resolve().parent != self.path: raise ValueError('Shard escapes checkpoint directory')
            stat = shard.stat(); self._stamps[filename] = self._stamp(stat)
            with shard.open('rb') as stream:
                size_bytes = stream.read(8)
                if len(size_bytes) != 8: raise ValueError('Truncated safetensors header')
                size = struct.unpack('<Q', size_bytes)[0]
                if not 2 <= size <= min(16 << 20, stat.st_size-8): raise ValueError('Invalid safetensors header size')
                raw = stream.read(size); header = json.loads(raw)
            headers[filename] = {'header_sha256': sha(raw), 'file_bytes': stat.st_size}
            for name, metadata in header.items():
                if name == '__metadata__': continue
                if self._index.get(name) != filename: continue
                shape, dtype, offsets = metadata['shape'], metadata['dtype'], metadata['data_offsets']
                if (dtype not in DTYPES or not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                        or len(offsets) != 2 or any(type(n) is not int for n in offsets)
                        or not 0 <= offsets[0] <= offsets[1] <= stat.st_size-size-8
                        or offsets[1]-offsets[0] != math.prod(shape)*ITEM_BYTES[dtype]):
                    raise ValueError('Invalid tensor metadata: ' + name)
                self._tensors[name] = dict(shape=tuple(shape), dtype=dtype, shard=filename)
        if set(self._index) != set(self._tensors): raise ValueError('Index references absent tensor headers')
        self.tied_embeddings = bool(self._config.get('text_config', {}).get('tie_word_embeddings', self._config.get('tie_word_embeddings', False)))
        if self.tied_embeddings and 'lm_head.weight' in self._index:
            raise ValueError('Ambiguous tied checkpoint stores a separate lm_head; explicit verification required')
        self._manifest = {'schema': 'character-creation.packed-weights.v1',
                          'config_sha256': sha(config_bytes), 'index_sha256': sha(index_bytes),
                          'reader_sha256': sha(Path(__file__).read_bytes()), 'shards': headers,
                          'tensor_count': len(self._index), 'tie_word_embeddings': self.tied_embeddings,
                          'verification_scope': 'configuration/index/header hashes and file-stat stability; not full shard payload hashes',
                          'arithmetic_scope': 'dequantized weights only; serving activation quantization and rounding are not emulated'}

    @staticmethod
    def _stamp(stat): return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    @property
    def config(self): return copy.deepcopy(self._config)

    @property
    def text_config(self): return copy.deepcopy(self._config['text_config'])

    @property
    def manifest(self): return copy.deepcopy(self._manifest)

    def resolve(self, name):
        if name == 'lm_head.weight' and self.tied_embeddings: return TEXT_PREFIX+'embed_tokens.weight'
        if name in self._index: return name
        for prefix in ('model.', ''):
            candidate = TEXT_PREFIX + (name[len(prefix):] if name.startswith(prefix) else name)
            if candidate in self._index: return candidate
        raise KeyError('Unknown checkpoint tensor: ' + name)

    def parameter_names(self, hf_prefix='model'):
        """HF parameter names only; excludes quantization auxiliaries, vision and MTP."""
        for name in sorted(self._index):
            if name.startswith(TEXT_PREFIX) and not name.endswith(('.input_scale', '.weight_scale', '.weight_scale_2')):
                yield (hf_prefix+'.' if hf_prefix else '') + name[len(TEXT_PREFIX):]
        if 'lm_head.weight' in self._index or self.tied_embeddings: yield 'lm_head.weight'

    def tensor_info(self, name):
        name = self.resolve(name); metadata = dict(self._tensors[name]); shape = metadata['shape']
        declaration = self._quant.get(name.removesuffix('.weight'), {}) if name.endswith('.weight') else {}
        if declaration.get('quant_algo') == 'NVFP4': shape = (*shape[:-1], shape[-1]*2)
        return dict(metadata, name=name, logical_shape=shape, quantization=copy.deepcopy(declaration))

    def quantization(self, name):
        name = self.resolve(name); prefix = name.removesuffix('.weight')
        result = copy.deepcopy(self._quant.get(prefix, {}))
        result['auxiliary_names'] = {suffix: prefix+'.'+suffix for suffix in ('input_scale', 'weight_scale', 'weight_scale_2') if prefix+'.'+suffix in self._index}
        result['activation_quantization_applied'] = False
        return result

    def tensor(self, name, device='cpu', dtype=None, *, rows=None):
        import torch
        from contextlib import ExitStack
        from safetensors import safe_open
        for filename, stamp in self._metadata_stamps.items():
            if self._stamp((self.path/filename).stat()) != stamp: raise ValueError('Checkpoint metadata changed after inspection')
        name = self.resolve(name); info = self.tensor_info(name)
        shape, declared = info['logical_shape'], info['quantization']
        algorithm = declared.get('quant_algo')
        if algorithm not in (None, 'NVFP4', 'FP8'): raise ValueError('Unsupported quantization: ' + str(algorithm))
        if algorithm == 'NVFP4' and declared.get('group_size') != 16: raise ValueError('Only NVFP4 group16 is supported')
        if algorithm is None and info['dtype'] in ('U8', 'F8_E4M3') and name.endswith('.weight'):
            raise ValueError('Packed/FP8 weight lacks a supported quantization declaration')
        if rows is not None and (not shape or len(rows) != 2 or any(type(i) is not int for i in rows) or not 0 <= rows[0] <= rows[1] <= shape[0]):
            raise ValueError('Invalid half-open first-axis row slice')
        begin, end = (0, shape[0]) if rows is None and shape else (rows if rows is not None else (0, 1))
        out_shape = (end-begin, *shape[1:]) if shape else ()
        if dtype is None: dtype = torch.float32 if algorithm or info['dtype'] == 'F8_E4M3' else getattr(torch, DTYPES[info['dtype']])
        if dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64): raise ValueError('Decoded parameter dtype must be floating point')
        with ExitStack() as stack, torch.no_grad():
            files = {}
            def source(key):
                meta = self._tensors[key]; filename = meta['shard']
                if self._stamp((self.path/filename).stat()) != self._stamps[filename]: raise ValueError('Checkpoint shard changed after inspection')
                if filename not in files: files[filename] = stack.enter_context(safe_open(self.path/filename, framework='pt', device='cpu'))
                value = files[filename].get_slice(key)
                if tuple(value.get_shape()) != meta['shape'] or value.get_dtype() != meta['dtype']:
                    raise ValueError('Tensor header changed after inspection')
                return value
            def scalar_tensor(key):
                source(key)
                return files[self._tensors[key]['shard']].get_tensor(key)
            values = source(name); prefix = name.removesuffix('.weight')
            if algorithm:
                if len(shape) != 2: raise ValueError('Quantized weights must be matrices')
                scale_name = prefix+'.weight_scale'; scale = source(scale_name)
                if algorithm == 'NVFP4':
                    if self._tensors[scale_name]['shape'] != (shape[0], shape[1]//16): raise ValueError('NVFP4 scale geometry mismatch')
                    global_scale = scalar_tensor(prefix+'.weight_scale_2').to(device)
                    _positive_scalar(global_scale, 'weight_scale_2')
                else:
                    if info['dtype'] != 'F8_E4M3': raise ValueError('FP8 weights must use E4M3')
                    scalar = scalar_tensor(scale_name).to(device); _positive_scalar(scalar, 'weight_scale')
            output = torch.empty(out_shape, dtype=dtype, device=device)
            for start in range(begin, end, self.block_rows):
                stop = min(end, start+self.block_rows)
                block = (values[start:stop] if shape else scalar_tensor(name)).to(device)
                if algorithm == 'NVFP4': block = decode_nvfp4(block, scale[start:stop].to(device), global_scale)
                elif algorithm == 'FP8': block = block.float() * scalar.reshape(())
                if not torch.isfinite(block.float()).all(): raise ValueError('Nonfinite decoded parameter: ' + name)
                if shape: output[start-begin:stop-begin].copy_(block)
                else: output.copy_(block)
            for filename in files:
                if self._stamp((self.path/filename).stat()) != self._stamps[filename]: raise ValueError('Checkpoint changed during read')
            return output
