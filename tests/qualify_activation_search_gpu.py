"""Standalone lease-run kernel parity check; no model or inference services touched."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('activation_search_capture_qualification',root/'activation_capture.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
reports=[]
for tokens in (1,32,128):
 for case in ((dtype, layout) for dtype in (torch.float32,torch.bfloat16) for layout in ("row_major", "column_major")):
  dtype,layout=case
  c=module.ActivationCapture.allocate(2,tokens,8,'cuda',intermediate=17408)
  value=torch.randn((tokens,17408),device='cuda',dtype=dtype)
  if layout=='column_major':value=value.T.contiguous().T
  valid=max(1,tokens-3);positions=torch.arange(tokens,device='cuda');positions[valid:]=-1
  c.begin(SimpleNamespace(positions=positions,live_counts=torch.tensor([valid],device='cuda')))
  c.search_enabled.fill_(False);c.search_position.fill_(valid-1)
  def run(x):c.record_mlp_input(x,0)
  compiled=torch.compile(run,fullgraph=True)
  compiled(value);torch.cuda.synchronize()
  cpu=value[:valid].float().cpu();absolute=cpu.abs();where=absolute.argmax(dim=0)
  expected=cpu.gather(0,where[None,:]).squeeze(0)
  def check():
   torch.testing.assert_close(c.search_peak[0].cpu(),expected,rtol=0,atol=0)
   torch.testing.assert_close(c.search_peak_position[0].cpu(),where,rtol=0,atol=0)
   torch.testing.assert_close(c.search_sum_abs[0].cpu(),absolute.sum(dim=0),rtol=1e-5,atol=1e-5)
   torch.testing.assert_close(c.search_selected[0].cpu(),cpu[-1],rtol=0,atol=0)
   top=absolute.argmax(dim=1);torch.testing.assert_close(c.search_token_neuron[0,:valid].cpu(),top,rtol=0,atol=0)
   torch.testing.assert_close(c.search_token_peak[0,:valid].cpu(),cpu.gather(1,top[:,None]).squeeze(1),rtol=0,atol=0)
  assert bool((c.search_peak_position == -1).all()), "Disabled compilation mutated search planes"
  c.search_enabled.fill_(True);compiled(value);torch.cuda.synchronize();check()
  c.search_enabled.fill_(False)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):compiled(value)
  c.search_enabled.fill_(True);graph.replay();torch.cuda.synchronize();check()
  saved=c.search_peak.clone();c.search_enabled.fill_(False);value.mul_(2);graph.replay();torch.cuda.synchronize()
  torch.testing.assert_close(saved,c.search_peak,rtol=0,atol=0)
  torch.testing.assert_close(c.mlp_input_last[0],value[valid-1].float(),rtol=0,atol=0)
  reports.append({'tokens':tokens,'dtype':str(dtype),'layout':layout,'compiled_with_gate_disabled':True,'captured_with_gate_disabled':True,'compiled':True,'cudagraph':True,'disabled_unchanged':True})
  print(json.dumps(reports[-1]),flush=True)
print(json.dumps({'passed':True,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'checks':reports}),flush=True)
