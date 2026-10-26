import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import threading

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'persona'))
from persona_activation_search import normalize_search, ActivationSearch, attach_profiles
probe_spec = importlib.util.spec_from_file_location('isolated_search_probe', ROOT / 'activation-search/persona_probe.py')
probe_module = importlib.util.module_from_spec(probe_spec)
sys.modules[probe_spec.name] = probe_module
probe_spec.loader.exec_module(probe_module)
normalize_request = probe_module.normalize_request


def test_request_bounds():
    assert normalize_search({'id': 'a' * 32})['top_k'] == 32
    for value in ({'id': 'a' * 32, 'top_k': 33}, {'id': 'a' * 32, 'token_position': True}, {'id': 'x'}):
        with pytest.raises(ValueError):
            normalize_search(value)
    payload = {'model': 'm', 'messages': [{'role': 'user', 'content': 'hi'}],
               'capture_search': {'id': 'a' * 32}, 'max_tokens': 1}
    with pytest.raises(ValueError):
        normalize_request(payload, 'm')
    payload['observation_only'] = True
    assert normalize_request(payload, 'm')['capture_search']['top_k'] == 32
    payload['capture_trace'] = {'id': 'b' * 32, 'neurons': [{'layer': 0, 'neuron': 0}]}
    with pytest.raises(ValueError):
        normalize_request(payload, 'm')


def test_chunk_reductions_and_exact_replay():
    torch = pytest.importorskip('torch')
    spec = importlib.util.spec_from_file_location('test_capture_impl', ROOT / 'activation_capture.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    c = module.ActivationCapture.allocate(2, 3, 2, 'cpu', intermediate=4)
    engine = SimpleNamespace(capture=c, model=SimpleNamespace(layers=2, intermediate=4, identity='model'),
        manifest={'runtime_identity': 'runtime'}, trace_lock=threading.RLock(), trace_records={},
        tokenizer=SimpleNamespace(decode=lambda ids, **kw: str(ids[0])))
    search = ActivationSearch(engine, normalize_search({'id': 'a' * 32, 'top_k': 3, 'token_position': 4}), list(range(5)), 'hash')
    search.configure()
    values = np.array([[[0, -7, 1, 0], [0, 2, -3, 1], [0, 7, 1, 0], [0, 1, 2, 0], [0, 0, 3, 0]],
                       [[0, 2, 0, 1], [0, 0, 0, 1], [0, -9, 0, 1], [0, 1, 0, 1], [0, 0, 10, 1]]], dtype=np.float32)
    for start, stop in ((0, 3), (3, 5)):
        positions = list(range(start, stop))
        c.begin(SimpleNamespace(positions=torch.tensor(positions + [-1] * (3 - len(positions))), live_counts=torch.tensor([len(positions)])))
        for layer in range(2):
            padded = np.full((3, 4), 9999, dtype=np.float32)
            padded[:stop-start] = values[layer, start:stop]
            c.record_mlp_input(torch.from_numpy(padded), layer)
        search.observe(positions)
    result = search.result()
    assert [(r['layer'], r['neuron']) for r in result['candidates']] == [(1, 2), (1, 1), (0, 1)]
    assert result['candidates'][2]['peak_position'] == 0  # first absolute tie, signed -7
    assert result['coverage']['neuron_token_values'] == 40
    assert result['token_candidates'][0]['activation'] == 10
    assert len(result['token_peaks']) == 5
    assert np.array_equal(search.sum_abs, np.abs(values).sum(axis=1))
    rows = result['candidates']
    trace = {'neurons': [{'layer': r['layer'], 'neuron': r['neuron']} for r in rows],
        'events': [{'positions': list(range(5)), 'values': [[float(values[r['layer'], p, r['neuron']]) for r in rows] for p in range(5)]}]}
    attach_profiles(result, trace)
    assert result['profile_verification']['peak_bit_exact']
    trace['events'][0]['values'][4][0] = 11
    with pytest.raises(ValueError, match='bit-exact'):
        attach_profiles(result, trace)
    assert c.rows(1).search_peak.data_ptr() == c.search_peak.data_ptr()
    assert c.rows(1).search_token_peak.shape == (2, 1)


def test_triton_interpreter_fp32_accumulation_and_signed_zero():
    """Exercise the actual reduction kernels on CPU, including BF16 inputs."""
    pytest.importorskip('torch')
    pytest.importorskip('triton')
    import os
    import subprocess
    script = '''
import torch
from activation_capture import _search_neurons, _search_tokens
for dtype in (torch.float32, torch.bfloat16):
    torch.manual_seed(92)
    x = torch.randn(128, 37, dtype=dtype)
    x = x.T.contiguous().T  # Model compiler may choose column-major product layout.
    x[:, 0] = -0.
    positions = torch.arange(128)
    positions[125:] = -1
    peak, total, selected = (torch.zeros(37) for _ in range(3))
    peak_position = torch.zeros(37, dtype=torch.int64)
    token_peak = torch.zeros(128)
    token_neuron = torch.zeros(128, dtype=torch.int64)
    enabled = torch.tensor([True])
    chosen = torch.tensor([124])
    _search_neurons[(2,)](x, positions, enabled, chosen, peak, peak_position,
                         total, selected, 37, 128, x.stride(0), x.stride(1), 128, 32, num_warps=4)
    _search_tokens[(128,)](x, positions, enabled, token_peak, token_neuron,
                          37, x.stride(0), x.stride(1), 64, num_warps=4)
    expected = x[:125].float()
    torch.testing.assert_close(total, expected.abs().sum(0), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(selected, expected[-1], rtol=0, atol=0)
    assert torch.signbit(peak[0]) and torch.signbit(selected[0])
    index = expected.abs().argmax(0)
    torch.testing.assert_close(peak, expected.gather(0,index[None,:]).squeeze(0),rtol=0,atol=0)
    torch.testing.assert_close(peak_position,index,rtol=0,atol=0)
    torch.testing.assert_close(token_peak[:125],expected.gather(1,expected.abs().argmax(1)[:,None]).squeeze(1),rtol=0,atol=0)
    previous = peak.clone()
    enabled.fill_(False)
    x.fill_(9999)
    _search_neurons[(2,)](x, positions, enabled, chosen, peak, peak_position,
                         total, selected, 37, 128, x.stride(0), x.stride(1), 128, 32, num_warps=4)
    torch.testing.assert_close(peak, previous, rtol=0, atol=0)
'''
    subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                   env={**os.environ, 'TRITON_INTERPRET': '1'}, check=True, timeout=30)


def test_isolated_capture_override(tmp_path, monkeypatch):
    import types
    import os
    import pickle
    import subprocess
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "activation_capture", raising=False)
    from isolated_search_probe import load_capture_override, capture_source_hashes, file_hash
    ignition = types.ModuleType('ignition')
    ignition.__path__ = []
    components = types.ModuleType('ignition.components')
    components.__path__ = []
    ignition.components = components
    monkeypatch.setitem(sys.modules, 'ignition', ignition)
    monkeypatch.setitem(sys.modules, 'ignition.components', components)
    monkeypatch.delitem(sys.modules, 'ignition.components.activation_capture', raising=False)
    original = tmp_path / 'original.py'
    override = tmp_path / 'activation_capture.py'
    original.write_text('original source remains unchanged')
    override.write_text('from dataclasses import dataclass\n@dataclass\nclass ActivationCapture:\n    marker: int = 7\n')
    assert load_capture_override(None) is None
    try:
        assert load_capture_override(override) == override.resolve()
        module = sys.modules['ignition.components.activation_capture']
        assert components.activation_capture is module
        assert module.ActivationCapture().marker == 7
        assert module.ActivationCapture.__module__ == 'activation_capture'
        # A fresh interpreter must resolve the override without canonical ignition.
        subprocess.run([sys.executable, '-c',
            "import sys,pickle; obj=pickle.loads(sys.stdin.buffer.read()); assert type(obj).__module__=='activation_capture'; assert obj.marker==7"],
            input=pickle.dumps(module.ActivationCapture()), check=True, timeout=15, cwd=tmp_path)
        with pytest.raises(RuntimeError, match='before'):
            load_capture_override(override)
        hashes = capture_source_hashes(tmp_path, ['ignition/components/activation_capture.py', 'original.py'], override)
        assert hashes['ignition/components/activation_capture.py'] == file_hash(override)
        assert hashes['original.py'] == file_hash(original)
        assert original.read_text() == 'original source remains unchanged'
    finally:
        sys.modules.pop('ignition.components.activation_capture', None)
        sys.modules.pop('activation_capture', None)


def test_capture_override_partitions_aot_cache(tmp_path, monkeypatch):
    import types
    seen = []
    class FakeForwardCache:
        def __init__(self, *, namespace, device):
            seen.append((namespace, device))
    fake = types.ModuleType('ignition.preparation.aot')
    fake.ForwardCache = FakeForwardCache
    monkeypatch.setitem(sys.modules, 'ignition.preparation.aot', fake)
    override = tmp_path / 'activation_capture.py'
    override.write_text('capture source one')
    assert probe_module.bind_capture_cache_namespace(None) is None
    namespace = {'model': 'unchanged'}
    digest = probe_module.bind_capture_cache_namespace(override)
    FakeForwardCache(namespace=namespace, device='cpu')
    assert namespace == {'model': 'unchanged'}
    assert seen[-1] == ({'model': 'unchanged', 'activation_capture_override_sha256': digest}, 'cpu')
    assert digest == probe_module.file_hash(override)
    assert probe_module.bind_capture_cache_namespace(override) == digest
    override.write_text('capture source two')
    with pytest.raises(RuntimeError, match='another capture'):
        probe_module.bind_capture_cache_namespace(override)
