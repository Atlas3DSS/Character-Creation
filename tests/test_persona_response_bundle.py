import persona
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import persona_response_bundle as module
from persona_contrastive import scenes,build_batches,input_digest,POLICY

ROOT=Path(__file__).resolve().parents[1]


def fixture(root):
    (root/'data').mkdir();(root/'results').mkdir()
    (root/module.FILES['scenes']).write_bytes((ROOT/module.FILES['scenes']).read_bytes())
    data,scene_sha=scenes(root);binding={k:'a'*64 for k in module.IDENTITIES}
    for i in range(64):np.savez(root/f'results/layer_{i:02d}.npz',model_identity=binding['model_identity'],residual_median=2.)
    vectors={k:np.ones((64,5120),np.float64)*(i+1) for i,k in enumerate(('direction_A','direction_B'))}
    np.savez(root/module.FILES['directions'],**vectors)
    artifact={'filename':'persona_response_directions.npz','sha256':module.sha(root/module.FILES['directions'])}
    for stage in ('development','pilot'):
        batches=build_batches(data,binding,stage,artifact if stage=='pilot' else None)
        report=dict(schema='synthetic',stage=stage,state='complete',binding=binding,model_id='test',scene_sha256=scene_sha,
                    policy=POLICY,direction_artifact=artifact if stage=='pilot' else None,qualification_sha256='b'*64,
                    batches=[dict(index=i,input=b,state='complete',result={'cells':[{} for _ in b['cells']]}) for i,b in enumerate(batches)])
        report['input_sha256']=input_digest(report);(root/module.FILES[stage]).write_text(json.dumps(report))
    contexts=[{'id':r['id'],'prompt':r['prompt'],'split':'discovery','source_position':25} for r in data['evaluation'] if r['split']=='discovery'][:6]
    inputs={'binding':binding,'scene_sha256':scene_sha,'contexts':contexts,
            'directions':[{'id':s,'layer':32,'residual_median':2.,'source_sha256':artifact['sha256'],'vector':vectors['direction_'+s][32].tolist()} for s in ('A','B')]}
    (root/module.FILES['propagation']).write_text(json.dumps({'state':'complete','passed':True,'failures':[],'binding':binding,'inputs':inputs}))
    return module.plan(root)


def test_frozen_reports_authored_sources_and_all_calibration(tmp_path):
    manifest=fixture(tmp_path);data,reports,binding,medians=module.load_inputs(tmp_path,manifest)
    assert medians.dtype==np.float64 and medians.shape==(64,) and np.all(medians==2)
    assert manifest['array_bytes_uncompressed']==758579200
    assert len(reports['development']['batches'])==4 and len(reports['pilot']['batches'])==43
    p=tmp_path/module.FILES['pilot'];value=json.loads(p.read_text());value['batches'][0]['input']['cells'][0]['prompt']='Private arbitrary replacement'
    value['input_sha256']=input_digest(value);p.write_text(json.dumps(value));manifest['files']['pilot']['sha256']=module.sha(p)
    with pytest.raises(ValueError,match='Authored batch'):module.load_inputs(tmp_path,manifest)


def test_manifest_hash_source_corpus_and_direction_fail_closed(tmp_path):
    manifest=fixture(tmp_path)
    bad=copy.deepcopy(manifest);bad['files']['development']['sha256']='0'*64
    with pytest.raises(ValueError,match='hash'):module.load_inputs(tmp_path,bad)
    p=tmp_path/module.FILES['propagation'];v=json.loads(p.read_text());v['inputs']['contexts'][0]['prompt']='Private book text'
    p.write_text(json.dumps(v));manifest['files']['propagation']['sha256']=module.sha(p)
    with pytest.raises(ValueError,match='authored discovery'):module.load_inputs(tmp_path,manifest)
    v['inputs']['contexts'][0]['prompt']=scenes(tmp_path)[0]['evaluation'][0]['prompt'];v['inputs']['directions'][0]['vector'][0]+=1
    p.write_text(json.dumps(v));manifest['files']['propagation']['sha256']=module.sha(p)
    with pytest.raises(ValueError,match='direction values'):module.load_inputs(tmp_path,manifest)


def test_exact_source_paths_hashes_and_no_symlinks(tmp_path):
    (tmp_path/'results').mkdir();store=module.LocalStore(tmp_path);path=tmp_path/'results'/('probe-'+'a'*32+'.npz');path.write_bytes(b'test')
    pointer={'path':str(path),'sha256':module.sha(path)};assert store.get(pointer)==path
    for edit in ({'sha256':'0'*64},{'path':str(tmp_path/path.name)}):
        with pytest.raises(ValueError):store.get({**pointer,**edit})
    link=tmp_path/'results'/('probe-'+'b'*32+'.npz');link.symlink_to(path)
    with pytest.raises(ValueError,match='path'):store.get({**pointer,'path':str(link)})


def test_response_means_preserve_float64_and_reject_rounding():
    mean=np.zeros((64,5120),np.float64);mean[0,0]=1.+2.**-40
    a={'response_residual_mean':mean,'response_residual_count':np.array(7,dtype=np.int64)}
    actual,count=module.compact_response(a);assert actual is mean and actual[0,0]==mean[0,0] and count==7
    with pytest.raises(ValueError,match='dtype'):module.compact_response({**a,'response_residual_mean':mean.astype(np.float32)})
    with pytest.raises(ValueError,match='count'):module.compact_response({**a,'response_residual_count':np.array(0,dtype=np.int64)})


def test_native_rejects_nonapproved_text_even_with_self_consistent_hash(tmp_path):
    (tmp_path/'results').mkdir();path=tmp_path/'results'/('probe-'+'a'*32+'.jsonl')
    request={'messages':[{'role':'user','content':'private arbitrary text'}]}
    saved={'request':request,'response':{'activation_artifact':{}}};path.write_text(json.dumps(saved))
    record={'provenance_artifact':{'path':str(path),'sha256':module.sha(path)},'activation_artifact':{},'request_sha256':module.canonical_digest(request)}
    with pytest.raises(ValueError,match='approved source'):module.native(module.LocalStore(tmp_path),record,{},'public authored text')
    public=module.public_source({'activation_artifact':{'sha256':'a','path':'private'},'provenance_artifact':{'sha256':'b','path':'private'},'request_sha256':'c'})
    assert public=={'artifact_sha256':'a','provenance_sha256':'b','request_sha256':'c'}


def test_parallel_extraction_preserves_order_and_bound():
    import threading
    barrier=threading.Barrier(4);lock=threading.Lock();active=0;peak=0;threads=set()
    def extract(index):
        nonlocal active,peak
        with lock:
            active+=1;peak=max(peak,active);threads.add(threading.get_ident())
        barrier.wait(timeout=5)
        with lock:active-=1
        return index,np.array([index],dtype=np.float64)
    result=list(module.ordered_extract(extract,list(range(8)),4))
    assert [i for i,_ in result]==list(range(8))
    assert all(a.dtype==np.float64 and int(a[0])==i for i,a in result)
    assert peak==4 and len(threads)==4


def test_parallel_extraction_errors_are_not_skipped():
    def extract(index):
        if index==2:raise ValueError('source hash mismatch')
        return index
    with pytest.raises(ValueError,match='source hash mismatch'):
        list(module.ordered_extract(extract,range(8),4))


def test_worker_cli_bound_and_affinity_default(monkeypatch):
    for value in (True,0,-1,87,1.5,'1.5','unknown'):
        with pytest.raises(ValueError):module.worker_count(value)
    assert module.worker_count('86')==86
    monkeypatch.setattr(module.os,'sched_getaffinity',lambda pid:set(range(88)))
    assert module.default_workers()==86
    monkeypatch.setattr(module.os,'sched_getaffinity',lambda pid:{1})
    assert module.default_workers()==1


def test_native_pulse_allows_only_identical_fp32_representation():
    expected=np.zeros(5120,np.float64);expected[0]=1.
    actual=expected.copy();actual[0]+=2.**-50
    program={'phase':'prefill','sites':[{'site':'layers.32.residual','vector':actual.tolist()}]}
    digest=module.validate_native_pulse(program,expected)
    assert digest==hashlib.sha256(expected.astype('<f4').tobytes()).hexdigest()
    program['sites'][0]['vector'][0]=float(np.nextafter(np.float32(1.),np.float32(2.)))
    with pytest.raises(ValueError,match='FP32 pulse differs'):module.validate_native_pulse(program,expected)
    program['phase']='all'
    with pytest.raises(ValueError,match='phase'):module.validate_native_pulse(program,expected)
