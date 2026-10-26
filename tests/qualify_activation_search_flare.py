"""Lease-run test through Flare's auxiliary compile_forward compiler.

Serving uses aot.ForwardCache instead; this test does NOT qualify its cache
identity or spawned compiler imports. Native BF16 is skipped on SM70, where
the actual model uses software-rounded BF16 values stored in FP32.
"""
import argparse
import importlib.util
import json
import sys
from types import SimpleNamespace

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--compiler-file', required=True)
parser.add_argument('--capture-file', required=True)
args = parser.parse_args()

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

compiler = load('flare_search_qualification_compiler', args.compiler_file)
capture_module = load('flare_search_qualification_capture', args.capture_file)
for dtype in (torch.float32, torch.bfloat16):
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        print(json.dumps({"dtype": str(dtype), "status": "skipped", "reason": "native BF16 unsupported; SM70 model stores rounded values in FP32"}), flush=True)
        continue
    c = capture_module.ActivationCapture.allocate(2, 32, 8, 'cuda', intermediate=37)
    value = torch.randn((32, 37), device='cuda', dtype=dtype).T.contiguous().T
    positions = torch.arange(32, device='cuda')
    positions[29:] = -1
    metadata = SimpleNamespace(positions=positions, live_counts=torch.tensor([29], device='cuda'))
    c.search_enabled.fill_(False)
    c.search_position.fill_(28)

    def run():
        c.begin(metadata)
        for layer in range(2):
            # Exercise a compiler-created intermediate and repeated plane views.
            product = value + layer
            c.record_mlp_input(product, layer)

    compiled = compiler.compile_forward(run)
    compiled()
    torch.cuda.synchronize()
    assert bool((c.search_peak_position == -1).all()), 'Disabled compile/warmup changed planes'
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    c.search_enabled.fill_(True)
    graph.replay()
    torch.cuda.synchronize()
    for layer in range(2):
        expected = (value + layer)[:29].float().cpu()
        index = expected.abs().argmax(dim=0)
        observed = c.search_peak_position[layer].cpu()
        print(json.dumps({'dtype':str(dtype),'layer':layer,'position_min':int(observed.min()),
                          'position_max':int(observed.max()),'gate':bool(c.search_enabled.item())}),flush=True)
        torch.testing.assert_close(observed, index, rtol=0, atol=0)
        torch.testing.assert_close(c.search_peak[layer].cpu(), expected.gather(0,index[None,:]).squeeze(0),rtol=0,atol=0)
        torch.testing.assert_close(c.search_sum_abs[layer].cpu(),expected.abs().sum(0),rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(c.search_selected[layer].cpu(),expected[-1],rtol=0,atol=0)
    previous = c.search_peak.clone()
    c.search_enabled.fill_(False)
    value.mul_(2)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(c.search_peak,previous,rtol=0,atol=0)
    print(json.dumps({'passed':True,'dtype':str(dtype),'auxiliary_compile_forward':True,'serving_aot_qualified':False,
                      'compiled_and_captured_gate_disabled':True,'noncontiguous':True}),flush=True)
