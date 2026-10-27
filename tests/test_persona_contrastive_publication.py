"""Offline publication tests: no Hub, model, worker or inference requests."""
import persona
from copy import deepcopy
import json
import pytest
import persona_contrastive_publication as m
from test_persona_publication_recovery import FakeHub


def report():
    job='1'*64
    row={'job_id':job,'source_id':'public','prompt':'An original public scene.','recipe_id':'baseline','arm':'baseline','shared_baseline_job_id':job,'seed':1,'program_sha256':'2'*64,'pushes':[],'phase':'all','state':'complete','answer':'An answer.','complete_answer':True,'capture_validated':True,'request_sha256':'3'*64,'response_sha256':'4'*64,'usage':{}}
    row['source_prompt_sha256']=m.digest(row['prompt'].encode())
    bound={k:'6'*64 for k in ('model_identity','runtime_identity','prepared_identity')}
    row['capture']={k:{**bound,'sha256':'7'*64,'request_sha256':'3'*64} for k in ('activation_artifact','provenance_artifact')}
    row['response_capture']={'sampled_generated_tokens':2,'consumed_generated_tokens':1,'mean_defined':True,**{k:'8'*64 for k in ('generated_token_ids_sha256','consumed_token_ids_sha256','consumed_positions_sha256')}}
    return {'raw_arrays_included':False,'promotion_eligible':False,'dispatch_deadline':1793112540,'deadline':1793113200,'schema':m.SCHEMA,'input_sha256':'5'*64,'source_sha256':m.CORPUS_SHA,'source_manifest_sha256':m.MANIFEST_SHA,'binding':{k:'6'*64 for k in ('model_identity','runtime_identity','prepared_identity')},'counts':{'complete':1,'queued':47},'terminal_jobs':1,'total_jobs':48,'final':False,'state':'running','rows':[row]}


def validate(v):return m.validate_report(json.dumps(v).encode(),{'public':'An original public scene.'},'5'*64)


def test_exact_public_discovery_report():assert validate(report())['terminal_jobs']==1


@pytest.mark.parametrize('bad',['private','prompt','duplicate','counts','source','complete','final','baseline'])
def test_false_or_unapproved_scientific_claims_fail_closed(bad):
    v=report()
    if bad=='private':v['rows'][0]['endpoint']='secret'
    elif bad=='prompt':v['rows'][0]['prompt']='Unapproved private chemistry'
    elif bad=='duplicate':v['rows'].append(deepcopy(v['rows'][0]));v['terminal_jobs']=2;v['counts']={'complete':2,'queued':46}
    elif bad=='counts':v['counts']['complete']=2
    elif bad=='source':v['source_sha256']='0'*64
    elif bad=='complete':v['rows'][0]['capture_validated']=False
    elif bad=='final':v['state']='complete';v['final']=True
    else:v['rows'][0]['pushes']=[{'dose':1}]
    with pytest.raises(ValueError):validate(v)


def setup(tmp_path):
    h=FakeHub();o=m.Observer(tmp_path/'observer.sqlite',h,{'plan':'pinned'});files={m.REPORT:b'exactbytes'};o.freeze(files,{'terminal':48});return o,h,files


def test_unknown_success_and_concurrent_commit_reconcile_bounded_history(tmp_path):
    o,h,f=setup(tmp_path);h.fail='after';o.step();actual=h.current
    h.current='later';h.data['later']={m.REPORT:b'other writer'}
    o.step();assert o.pending()[2]==actual;assert o.step();assert len(h.calls)==1


def test_unknown_only_reposts_same_frozen_parent_and_payload(tmp_path):
    o,h,f=setup(tmp_path);h.fail='before';o.step();f[m.REPORT]=b'mutated';o.step()
    assert h.calls==[({m.REPORT:b'exactbytes'},'p0')]*2;assert o.step()


def test_explicit412_refreshes_parent_only_after_reconciliation(tmp_path):
    o,h,f=setup(tmp_path);h.current='other';h.data['other']={};o.step();assert h.calls[0][1]=='p0'
    o.step();assert h.calls[1][1]=='other';assert o.step()


def test_returned_oid_read_failure_never_reposts_and_restart_keeps_bytes(tmp_path):
    o,h,f=setup(tmp_path);o.step();h.read_fail=True;o.step();o.db.close()
    o=m.Observer(tmp_path/'observer.sqlite',h,{'plan':'pinned'});o.step();assert len(h.calls)==1
    h.read_fail=False;assert o.step()


def test_unknown_history_outside_bound_no_new_parent_or_commit(tmp_path):
    o,h,f=setup(tmp_path);h.fail='before';o.step();h.current='unbounded'
    def history(*a):raise ValueError('outside bounded history')
    h.history=history;o.step();assert len(h.calls)==1 and o.pending()[1]=='p0'


def test_one_report_allowlist_and_changed_lineage_reject(tmp_path):
    o,h,f=setup(tmp_path);o.step();o.step()
    with pytest.raises(ValueError):o.freeze({'README.md':b'x'},{})
    with pytest.raises(ValueError):m.Observer(tmp_path/'observer.sqlite',h,{'plan':'changed'})


def test_48_terminal_milestones_and_final():
    marker=lambda n,final=False:{'terminal':n,'total':100,'final':final,'sha256':str(n)+str(final)}
    assert not m.due(None,marker(47));assert m.due(None,marker(48))
    assert not m.due(marker(48),marker(95));assert m.due(marker(48),marker(96))
    assert m.due(marker(96),marker(99,True));assert not m.due(marker(99,True),marker(99,True))
    with pytest.raises(ValueError):m.due(marker(96),marker(90))
