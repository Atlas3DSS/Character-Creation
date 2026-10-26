import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file
import persona
from persona_phantom_weights import PackedCheckpoint, decode_nvfp4, FP4_VALUES


def fixture(tmp_path, *, tied=False):
    prefix = 'model.language_model.layers.0'
    packed = torch.tensor([[i | ((15-i) << 4) for i in range(16)]]*3, dtype=torch.uint8)
    tensors = {prefix+'.mlp.gate_proj.weight': packed,
               prefix+'.mlp.gate_proj.weight_scale': torch.tensor([[.5, 2.]]*3).to(torch.float8_e4m3fn),
               prefix+'.mlp.gate_proj.weight_scale_2': torch.tensor(.25),
               prefix+'.mlp.gate_proj.input_scale': torch.tensor(10.),
               prefix+'.self_attn.q_proj.weight': torch.arange(96).reshape(3, 32).to(torch.float8_e4m3fn),
               prefix+'.self_attn.q_proj.weight_scale': torch.tensor(.125),
               prefix+'.self_attn.q_proj.input_scale': torch.tensor(99.),
               prefix+'.input_layernorm.weight': torch.tensor([-.25, .125, .5], dtype=torch.bfloat16),
               prefix+'.linear_attn.A_log': torch.tensor([.123456], dtype=torch.float32),
               'model.language_model.embed_tokens.weight': torch.arange(12).reshape(4, 3).to(torch.bfloat16)}
    if not tied: tensors['lm_head.weight'] = torch.ones(4, 3, dtype=torch.bfloat16)
    save_file(tensors, tmp_path/'model.safetensors')
    config = {'text_config': {'model_type': 'qwen3_5_text', 'tie_word_embeddings': tied},
              'quantization_config': {'quant_method': 'modelopt', 'quantized_layers': {
                  prefix+'.mlp.gate_proj': {'quant_algo': 'NVFP4', 'group_size': 16},
                  prefix+'.self_attn.q_proj': {'quant_algo': 'FP8'}}}}
    (tmp_path/'config.json').write_text(json.dumps(config))
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {k: 'model.safetensors' for k in tensors}}))
    return tensors


def test_decode_nibbles_scales_signed_zero():
    packed = torch.tensor([[i | ((15-i) << 4) for i in range(16)]], dtype=torch.uint8)
    scales = torch.tensor([[.5, 2.]]).to(torch.float8_e4m3fn)
    expected = torch.tensor([v for i in range(16) for v in (FP4_VALUES[i], FP4_VALUES[15-i])]).reshape(1, 2, 16)
    expected = (expected * scales.float()[..., None] * .25).reshape(1, 32)
    actual = decode_nvfp4(packed, scales, torch.tensor(.25))
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    with pytest.raises(ValueError): decode_nvfp4(packed, scales, torch.tensor(0.))
    with pytest.raises(ValueError): decode_nvfp4(packed, scales.float(), torch.tensor(.25))


def test_one_tensor_rows_aliases_and_projection(tmp_path):
    tensors = fixture(tmp_path); reader = PackedCheckpoint(tmp_path, block_rows=1)
    prefix = 'model.language_model.layers.0.mlp.gate_proj'
    expected = decode_nvfp4(tensors[prefix+'.weight'], tensors[prefix+'.weight_scale'], tensors[prefix+'.weight_scale_2'])
    actual = reader.tensor('model.layers.0.mlp.gate_proj.weight', dtype=torch.float32)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(reader.tensor('layers.0.mlp.gate_proj.weight', rows=(1, 3)), expected[1:3])
    x = torch.linspace(-1, 1, 32).reshape(1, 32)
    torch.testing.assert_close(x @ actual.T, x @ expected.T, rtol=0, atol=0)
    assert reader.tensor_info(prefix+'.weight')['logical_shape'] == (3, 32)
    assert not actual.requires_grad
    with pytest.raises(ValueError): reader.tensor(prefix+'.weight', rows=(-1, 2))


def test_fp8_excludes_activation_scale_and_norm_is_raw(tmp_path):
    tensors = fixture(tmp_path); reader = PackedCheckpoint(tmp_path, block_rows=2)
    name = 'model.language_model.layers.0.self_attn.q_proj.weight'
    expected = tensors[name].float() * .125
    assert torch.equal(reader.tensor(name), expected)
    assert reader.quantization(name)['activation_quantization_applied'] is False
    assert torch.equal(reader.tensor('layers.0.input_layernorm.weight'), tensors['model.language_model.layers.0.input_layernorm.weight'])
    assert reader.tensor('layers.0.linear_attn.A_log').dtype == torch.float32
    assert reader.tensor('layers.0.mlp.gate_proj.weight_scale_2').item() == .25
    names = list(reader.parameter_names())
    assert 'model.layers.0.input_layernorm.weight' in names and 'lm_head.weight' in names
    assert not any('scale' in n for n in names)
    assert reader.manifest['tensor_count'] == len(tensors)
    assert 'not full shard' in reader.manifest['verification_scope']


def test_tied_head_and_file_change_fail_closed(tmp_path):
    fixture(tmp_path, tied=True); reader = PackedCheckpoint(tmp_path)
    assert torch.equal(reader.tensor('lm_head.weight'), reader.tensor('embed_tokens.weight'))
    with (tmp_path/'model.safetensors').open('ab') as f: f.write(b'x')
    with pytest.raises(ValueError, match='changed'): reader.tensor('embed_tokens.weight')


def test_metadata_only_constructor_and_config_change(tmp_path, monkeypatch):
    fixture(tmp_path)
    import safetensors
    monkeypatch.setattr(safetensors, 'safe_open', lambda *a, **k: pytest.fail('constructor read tensor data'))
    reader = PackedCheckpoint(tmp_path)
    config = reader.config; config['text_config']['model_type'] = 'changed'
    assert reader.text_config['model_type'] == 'qwen3_5_text'
    with (tmp_path/'config.json').open('a') as f: f.write(' ')
    with pytest.raises(ValueError, match='metadata changed'): reader.tensor('embed_tokens.weight')
