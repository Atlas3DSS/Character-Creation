"""Direct override import; no Flare runtime, GPU initialization, or inference."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import torch
import pytest

PATH=Path(__file__).resolve().parents[1]/'activation_capture.py'
spec=importlib.util.spec_from_file_location('request_capture_override_test',PATH)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
Capture=module.RequestActivationCapture


def owner(slots=(2,0,3,1),generations=(11,12,13,14)):
    requests=torch.tensor(generations,dtype=torch.int64)
    row_slots=torch.tensor(slots,dtype=torch.int32)
    row_epochs=requests[row_slots.long()]
    return Capture.allocate(2,4,3,'cpu',intermediate=5,row_slot_ids=row_slots,
        row_generations=row_epochs,request_generations=requests)


def metadata(rows,positions,ids=None):
    return NS(requests=torch.tensor(rows,dtype=torch.int64),positions=torch.tensor(positions,dtype=torch.int64),
              token_ids=torch.tensor(ids if ids is not None else [100+i for i in range(len(rows))]),
              live_counts=torch.tensor([sum(p>=0 for p in positions),len(rows)]))


def test_legacy_b1_source_is_byte_identical():
    prefix=PATH.read_bytes().split(b'\n\n\n@dataclass(frozen=True)\nclass RequestActivationCapture:')[0]+b'\n'
    assert hashlib.sha256(prefix).hexdigest()=='968cb7f0b275a7b3547bc9de00d8bb34ed3463e0d92d103f45ef2ef5cb250ec5'


def test_packed_and_physical_permutations_are_not_conflated():
    c=owner();m=metadata([3,1,0,2],[7,10,6,30],[91,92,93,94]);c.begin(m)
    values=torch.arange(12,dtype=torch.float32).reshape(4,3);products=torch.arange(20,dtype=torch.float32).reshape(4,5)
    c.record_residual(values,1);c.record_pre_mlp(values+100,1);c.record_mlp_input(products,1)
    # Packed tokens map through rows[3,1,0,2] to physical slots[1,0,2,3].
    assert c.slot_positions.tolist()==[10,7,6,30]
    assert c.slot_token_ids.tolist()==[92,91,93,94]
    assert c.slot_generations.tolist()==[11,12,13,14]
    assert not c.mapping_error.item() and c.slot_live.all()
    assert torch.equal(c.residual_last[1],values[[1,0,2,3]])
    assert torch.equal(c.pre_mlp_last[1],(values+100)[[1,0,2,3]])
    assert torch.equal(c.mlp_input_last[1],products[[1,0,2,3]])


def test_holes_unequal_positions_and_finished_slots_clear_stale_planes():
    c=owner();c.residual_last.fill_(99);c.pre_mlp_last.fill_(99);c.mlp_input_last.fill_(99)
    # Input row1 has finished; packed tokens belong to rows0,2,3.
    c.begin(metadata([0,2,3,3],[5,20,8,-1],[10,20,30,0]))
    v=torch.arange(12,dtype=torch.float32).reshape(4,3);c.record_residual(v,0)
    assert c.slot_live.tolist()==[False,True,True,True]
    assert c.slot_positions.tolist()==[-1,8,5,20]
    assert torch.equal(c.residual_last[0,0],torch.zeros(3))
    assert not c.pre_mlp_last.any() and not c.mlp_input_last.any()
    c.begin(metadata([0,0,0,0],[-1,-1,-1,-1]))
    c.record_residual(torch.full((4,3),float('nan')),0)
    assert not c.residual_last.any() and not c.mapping_error.item()
    assert c.slot_token_ids.tolist()==[-1]*4 and c.slot_generations.tolist()==[0]*4


def test_b2_view_keeps_four_physical_output_slots_and_zeroes_absent_ones():
    backing=owner(slots=(2,3,0,1));c=backing.rows(2);c.begin(metadata([0,1],[15,50]));v=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
    c.record_residual(v,0)
    assert c.slot_live.tolist()==[False,False,True,True]
    assert torch.equal(backing.residual_last[0,2:],v)
    assert c.residual_last.data_ptr()==backing.residual_last.data_ptr()


@pytest.mark.parametrize('mode',['duplicate','invalid_row','invalid_slot','stale_generation','bad_live_count','invalid_packed_row'])
def test_invalid_mappings_are_evidenced_and_safe(mode):
    c=owner();m=metadata([0,1,2,3],[5,6,7,8])
    if mode=='duplicate':c.row_slot_ids[1]=c.row_slot_ids[0];c.row_generations[1]=c.row_generations[0]
    elif mode=='invalid_row':m.requests[0]=999
    elif mode=='invalid_slot':c.row_slot_ids[0]=-1
    elif mode=='stale_generation':c.row_generations[0]+=1
    elif mode=='bad_live_count':m.live_counts[0]=3
    elif mode=='invalid_packed_row':m.requests[0]=3;m.live_counts[1]=3
    c.begin(m);c.record_residual(torch.ones((4,3)),0)
    assert c.mapping_error.item()
    if mode=='duplicate':
        assert c.slot_match_counts[2]==2 and not c.slot_live[2] and not c.residual_last[:,2].any()
    elif mode!='bad_live_count':
        assert c.invalid_tokens[0] and not c.slot_live[2] and not c.residual_last[:,2].any()


def test_per_slot_accumulation_counts_follow_actual_consumption():
    c=owner(slots=(0,1,2,3));counts=torch.zeros(4,dtype=torch.int64);sums=torch.zeros((4,3),dtype=torch.float64)
    for rows,positions in [([0,1,2,3],[10,20,30,40]),([0,2,3,3],[11,31,41,-1]),([2,3,3,3],[32,42,-1,-1])]:
        c.begin(metadata(rows,positions));values=torch.tensor([[p,p,p] for p in positions],dtype=torch.float32);c.record_residual(values,0)
        counts+=c.slot_live;sums+=c.residual_last[0].double()
    assert counts.tolist()==[2,1,3,3]
    assert sums[:,0].tolist()==[21,20,93,123]


def test_invalid_shapes_and_unbound_epoch_buffers_rejected():
    c=owner()
    with pytest.raises(ValueError):c.rows(5)
    with pytest.raises(ValueError):c.begin(metadata([0],[1]))
    with pytest.raises(ValueError):Capture.allocate(2,4,3,'cpu',intermediate=5,row_slot_ids=torch.zeros(3,dtype=torch.int32),row_generations=c.row_generations,request_generations=c.request_generations)
