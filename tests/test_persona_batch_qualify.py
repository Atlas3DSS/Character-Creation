import persona
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from test_persona_batch_server import engine,payload
from persona_batch_server import BatchEngine,BatchResponseCapture,binding
from persona_batch_qualify import checked_parity,compare,recover_received,write_receipts
from persona_vectors import canonical_digest
import persona_probe as probe


def fixture(tmp_path):
    e=engine(tmp_path,64,5120)
    manifest={'model_identity':e.model.identity,'script_sha256':'a'*64,'helper_sha256':{'trace':'b'*64},'source_sha256':{'core':'c'*64},'heliopause_sha256':'d'*64,'torch':'synthetic','preparation':{},'prepared_identity':'c'*64,'calibration_equivalence':{'passed':True}}
    manifest['runtime_identity']=canonical_digest({k:manifest[k] for k in ('script_sha256','helper_sha256','source_sha256','heliopause_sha256','torch','preparation')});manifest.update({k:'f'*64 for k in ('checkpoint_config_sha256','checkpoint_index_sha256','tokenizer_sha256')});e.manifest=manifest
    p=payload();p.update(max_tokens=2,temperature=.7,top_p=.95,top_k=20);h=canonical_digest(p);spec=probe.normalize_request(p,e.model_id)
    rc=BatchResponseCapture(e,spec['capture_response'],3,h);rc.observe_value(3,10,np.ones((64,5120),np.float32))
    job={'id':'e'*32,'spec':spec,'ids':[1,2,3],'generated':[10,11],'response_capture':rc,'positions':[0,1,2],
        'trace':[[1,1,0,0,-1,1,0,0],[1,2,3,1,10,1,0,1],[1,8,4,2,11,2,0,2]],'graph_kinds':['init','prefill','decode'],
        'result':{'finished':True,'finish_reason':'length'},'norms':[np.ones((64,2),np.float32)],
        'prefill':{'residual_last':np.zeros((64,5120),np.float32),'pre_mlp_last':np.zeros((64,5120),np.float32),'mlp_input_last':np.zeros((64,17408),np.float32),'final_chunk_positions':np.array([0,1,2])},
        'artifact':tmp_path/'probe.npz','report':{'status':'running','request':p,'request_sha256':h,'started_at':0.,**binding(e),'worker_manifest':manifest,'physical_pages':8,
        'effective_admission':{'generation':1,'seed':p['seed'],'max_new_tokens':2,'eos_token_ids':[99],**spec['sampling']},'graphs':{'decode':'decode1'},'batch':None}}
    s=BatchEngine.__new__(BatchEngine);s.e=e;s.save_job(job);job['report']['state_restored']=True
    provenance=tmp_path/'probe.jsonl';provenance.write_text(json.dumps(job['report']))
    response=job['response'];response['provenance_artifact']={'path':str(provenance),'sha256':probe.file_hash(provenance),'request_sha256':h,**binding(e)}
    class Store:
        def get(self,pointer):return Path(pointer['path'])
    return response,p,binding(e),Store(),job


def test_length_is_valid_parity_not_complete_prose(tmp_path):
    response,request,bound,store,job=fixture(tmp_path)
    result=checked_parity(response,request,bound,store)
    assert result['evidence']['truncated_prose'] and result['evidence']['prose_complete'] is False
    assert response['response_capture']['generation_complete'] is False
    assert compare(result,result)['prose_complete'] is False
    from persona_propagation import checked
    with pytest.raises(ValueError,match='answer did not complete'):checked(response,request,bound,store)


@pytest.mark.parametrize('mutation',[
    lambda r,p,b,j:r['response_capture'].__setitem__('generation_complete',True),
    lambda r,p,b,j:r['diagnostics'].__setitem__('complete_final_answer',True),
    lambda r,p,b,j:r['choices'][0].__setitem__('finish_reason','stop'),
    lambda r,p,b,j:b.__setitem__('runtime_identity','f'*64),
    lambda r,p,b,j:p.__setitem__('seed',99),
])
def test_false_completeness_binding_or_request_rejected(tmp_path,mutation):
    r,p,b,s,j=fixture(tmp_path);mutation(r,p,b,j)
    with pytest.raises(ValueError):checked_parity(r,p,b,s)


def test_exact_array_parity_rejects_one_ulp(tmp_path):
    r,p,b,s,j=fixture(tmp_path);a=checked_parity(r,p,b,s);other=deepcopy(a)
    other['arrays']['response_residual_mean'][0,0]=np.nextafter(1.,2.)
    with pytest.raises(ValueError,match='response_residual_mean'):compare(a,other)


def test_recovery_distinguishes_received_uncertain_never_and_no_resubmission(tmp_path):
    r,p,b,s,j=fixture(tmp_path);p['metadata']={'job_id':'first'}
    # Rebind synthetic persisted request consistently for this recovery test.
    h=canonical_digest(p);r['activation_artifact']['request_sha256']=h;r['provenance_artifact']['request_sha256']=h;r['response_capture']['request_sha256']=h
    with np.load(j['artifact']) as z:arrays={k:z[k] for k in z.files}
    arrays['request_sha256']=np.asarray(h)
    with j['artifact'].open('wb') as f:probe.save_activation_npz(f,**arrays)
    r['activation_artifact']['sha256']=probe.file_hash(j['artifact'])
    j['report'].update(request=p,request_sha256=h,response={k:v for k,v in r.items() if k!='provenance_artifact'})
    provenance=Path(r['provenance_artifact']['path']);provenance.write_text(json.dumps(j['report']));r['provenance_artifact']['sha256']=probe.file_hash(provenance)
    base={'id':'first','arm':'baseline','prompt':'authored','seed':0,'max_tokens':2,'capture_response':p['capture_response']}
    second=dict(base,id='uncertain');third=dict(base,id='never');q=deepcopy(p);q['metadata']={'job_id':'uncertain'}
    plan={'jobs':[base,second,third],'old_binding':b}
    journal=tmp_path/'attempts.jsonl';events=[{'state':'submitting','job_id':'first','request':p,'request_sha256':h},{'state':'received','job_id':'first','response':r},{'state':'submitting','job_id':'uncertain','request':q,'request_sha256':canonical_digest(q)}]
    journal.write_text('\n'.join(json.dumps(e) for e in events))
    result=recover_received(plan,journal,s)
    assert list(result['records'])==['first'] and result['uncertain_no_retry']==['uncertain'] and result['never_submitted']==['never']
    journal.write_text(journal.read_text()+'\n'+json.dumps(events[0]))
    with pytest.raises(ValueError,match='Duplicate submission'):recover_received(plan,journal,s)


def test_receipt_cannot_publish_partial24_or_overwrite(tmp_path):
    plan={'jobs':[{'id':str(i)} for i in range(24)]}
    with pytest.raises(ValueError,match='All24'):write_receipts(plan,{}, {}, {},qualification_path=tmp_path/'q.json',compatibility_path=tmp_path/'c.json',server_sha256='a'*64,physical_pages=128,page_tokens=128)
    assert not (tmp_path/'q.json').exists()


def test_three_way_receipt_requires_real_shared_four_slot_decode_and_exact_programs(tmp_path):
    r,p,b,s,j=fixture(tmp_path);record=checked_parity(r,p,b,s)
    old={};serial={};batched={};jobs=[];new_binding=dict(b,runtime_identity='1'*64,prepared_identity='2'*64);server='3'*64
    program={'phase':'all','sites':[{'site':'layers.32.residual','vector':[.125]}]}
    for i in range(24):
        identity=str(i);case={'id':identity,'arm':'A','prompt':'authored','seed':0,'max_tokens':2,'capture_response':p['capture_response'],'recipe':{'pushes':[{'kind':'direction'}]}}
        jobs.append(case);a=deepcopy(record);a['request']=dict(p,metadata={'job_id':identity},steering=program)
        a['evidence']['request_sha256']=canonical_digest(a['request']);a['evidence']['program_sha256']=canonical_digest(program)
        old[identity]=a;serial[identity]=deepcopy(a);batched[identity]=deepcopy(a)
        for v in (serial[identity],batched[identity]):
            v['evidence'].update(binding=new_binding,experiment_source_sha256=server,physical_pages=128)
        batched[identity]['evidence']['batch']={'id':str(i//4),'requests':4,'index':i%4,'physical_slot':i%4}
        batched[identity]['evidence']['graphs']={'decode':'decode4'}
    plan={'jobs':jobs,'old_binding':b,'input_sha256':'4'*64,'direction_artifact':{'sha256':'5'*64}}
    qualification,compatibility=write_receipts(plan,old,serial,batched,qualification_path=tmp_path/'qualification.json',compatibility_path=tmp_path/'compatibility.json',server_sha256=server,physical_pages=128,page_tokens=128)
    assert qualification['passed'] and qualification['prose_complete_pairs']==0
    assert compatibility['allowed_program_sha256']==[canonical_digest(program)]
    assert all(g['maximum_shared_decode_slots']==4 for g in qualification['concurrency'])
    for i,v in enumerate(batched.values()):v['arrays']['completion_trace'][-1,1]+=i%4
    with pytest.raises(ValueError,match='shared4-slot'):
        write_receipts(plan,old,serial,batched,qualification_path=tmp_path/'other.json',compatibility_path=tmp_path/'othercompat.json',server_sha256=server,physical_pages=128,page_tokens=128)
    assert not (tmp_path/'other.json').exists()


def test_prepare_manager_pilot_keeps_frozen_sources_new_ids_and_no_enqueue(tmp_path):
    from types import SimpleNamespace
    import hashlib
    from persona_batch_qualify import prepare_manager_pilot
    from persona_manager import Queue,prepare_job
    old={'model_identity':'a'*64,'runtime_identity':'b'*64,'prepared_identity':'c'*64};new=dict(old,runtime_identity='d'*64,prepared_identity='e'*64)
    rows=[]
    for scene in range(4):
        text=f'Authored scene {scene}. What do you think?'
        for arm in ('baseline','A','A_reverse','B','B_reverse','random'):
            pushes=[] if arm=='baseline' else [{'kind':'random' if arm=='random' else 'direction','layer':32,'dose':-.1 if 'reverse' in arm else .1,'sha256':'f'*64}]
            rows.append({'id':f'{scene}_{arm}','case_id':str(scene),'arm':arm,'prompt':text,'source_sha256':hashlib.sha256(text.encode()).hexdigest(),'seed':scene,'recipe':{'phase':'all','pushes':pushes},'max_tokens':32})
    plan={'jobs':rows,'old_binding':old,'direction_artifact':{'sha256':'f'*64},'suite_sha256':'9'*64};plan['input_sha256']=canonical_digest(plan)
    (tmp_path/'layer_32.npz').write_bytes(b'calibration test')
    class Bank:
        root=tmp_path;model=SimpleNamespace(identity=old['model_identity'])
        def program(self,pushes,phase):
            program={'phase':phase,'sites':[{'site':'layers.32.residual','vector':[p['dose']]} for p in pushes]}
            provenance={'program_sha256':canonical_digest(program)}
            if any(p['kind']=='direction' for p in pushes):provenance['direction_binding']=old
            return program,provenance
    q={'schema':'character-creation.batch-service-qualification.v1','verified':True,'passed':True,'binding':new,'plan_sha256':plan['input_sha256'],'comparisons':{r['id']:{} for r in rows}}
    qp=tmp_path/'qualification.json';qp.write_text(json.dumps(q));qs={'path':str(qp),'sha256':probe.file_hash(qp)}
    programs=sorted({Bank().program(r['recipe']['pushes'],'all')[1]['program_sha256'] for r in rows if r['arm'] not in ('baseline','random')})
    c={'schema':'character-creation.batch-runtime-compatibility.v1','state':'complete','passed':True,'old_binding':old,'new_binding':new,'artifact_sha256':'f'*64,'allowed_program_sha256':programs,
       'old_core_source_hashes':{'core':'1'*64},'new_core_source_hashes':{'core':'1'*64},'qualification_sha256':qs['sha256'],'parity':{'programs_exact':True,'response_tokens_exact':True,'response_capture_exact':True}}
    cp=tmp_path/'compatibility.json';cp.write_text(json.dumps(c));cs={'path':str(cp),'sha256':probe.file_hash(cp)}
    original=deepcopy(plan);result=prepare_manager_pilot(plan,Bank(),new,qualification_spec=qs,compatibility_spec=cs)
    assert plan==original and len(result['jobs'])==24 and len(result['corpus'])==4
    assert all(j['prompt']['max_tokens']==2048 and j['prompt']['system']=='' and j['target_binding']==new for j in result['jobs'])
    assert all(j['id'] not in {r['id'] for r in rows} for j in result['jobs'])
    assert result['config_patch']['max_total_jobs']==24 and not list(tmp_path.glob('*.sqlite'))
    # The real Queue accepts these exact job structures; this test queue is isolated.
    queue=Queue(tmp_path/'test.sqlite');queue.register_corpus({r['id']:r for r in result['corpus']},result['corpus_sha256'])
    assert queue.enqueue(result['proposal'],result['jobs'],limit=24)==24
    cp.write_text(cp.read_text()+' ')
    with pytest.raises(ValueError,match='receipt bytes'):prepare_manager_pilot(plan,Bank(),new,qualification_spec=qs,compatibility_spec=cs)
