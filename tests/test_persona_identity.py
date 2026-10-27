"""Read-only neutral identity evidence source and tamper checks."""
from copy import deepcopy
import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
import pytest
from persona_identity import evidence, load_probe
from persona_vectors import canonical_digest

ROOT=Path(__file__).resolve().parents[1]
MODEL=SimpleNamespace(identity='test-model')


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False))


def freeze_job(root,value,row):
    row.setdefault('phase','all');row.setdefault('pushes',[]);row.setdefault('pair_id','identity-who-are-you')
    row['split']='discovery';row['prompt_sha256']=hashlib.sha256(row['prompt'].encode()).hexdigest()
    job={k:deepcopy(row[k]) for k in ('candidate_id','arm','seed','phase','pushes','program_sha256','pair_id','identity_probe')}
    job.update(model_identity=MODEL.identity,menu_sha256=value['menu_sha256'],prompt={'prompt':row['prompt'],'source_row_id':row['source_id'],'split':'discovery','system':''})
    job['id']=canonical_digest(job);row['job_id']=job['id']
    row['request_sha256']='d'*64
    row['capture']={k:{**value['binding'],'request_sha256':row['request_sha256'],'sha256':'e'*64} for k in ('activation_artifact','provenance_artifact')}
    with sqlite3.connect(root/'results/persona_adaptive.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,body TEXT)')
        db.execute('INSERT INTO jobs VALUES(?,?)',(job['id'],json.dumps(job)))


def seal(value):
    value['revision_sha256']=canonical_digest({k:v for k,v in value.items() if k!='revision_sha256'})
    return value


@pytest.fixture
def study(tmp_path):
    (tmp_path/'data').mkdir();(tmp_path/'results').mkdir()
    fixture=ROOT/'data/identity_probe_questions.json'
    target=tmp_path/'data/identity_probe_questions.json';target.write_bytes(fixture.read_bytes())
    probe=load_probe(tmp_path)
    assert probe['prompt']=='who are you' and probe['system']=='' and probe['thinking'] is False
    binding={'model_identity':MODEL.identity,'runtime_identity':'runtime','prepared_identity':'prepared'}
    c={'target':'Empathy','kind':'exact_distributed','program_sha256':'a'*64,'model_identity':MODEL.identity}
    c['id']=canonical_digest(c)
    menu={'schema':'character-creation.adaptive-menu.v1','binding':binding,'candidates':[c],'excluded_gpu':'private'}
    menu['menu_sha256']=canonical_digest(menu)
    write(tmp_path/'results/persona_adaptive_menu.json',menu)
    public_menu={k:v for k,v in menu.items() if k!='excluded_gpu'}
    value={'schema':'character-creation.adaptive-public.v1','binding':binding,'menu':public_menu,'menu_sha256':menu['menu_sha256'],'menu_projection_sha256':canonical_digest(public_menu),'source_hashes':{'identity_fixture':hashlib.sha256(target.read_bytes()).hexdigest()},'rows':[]}
    for arm in ('baseline','candidate'):
        answer='A saved '+arm+' answer'
        value['rows'].append({'identity_probe':True,'source_id':probe['id'],'prompt':probe['prompt'],'source_sha256':value['source_hashes']['identity_fixture'],'candidate_id':c['id'],'arm':arm,'program_sha256':c['program_sha256'] if arm=='candidate' else 'b'*64,'answer':answer,'answer_sha256':hashlib.sha256(answer.encode()).hexdigest(),'seed':80317,'state':'complete','complete':True,'capture_validated':True})
    for row in value['rows']:freeze_job(tmp_path,value,row)
    write(tmp_path/'results/persona_adaptive_public.json',seal(value))
    return tmp_path,value


def test_missing_report_and_empty_rows_are_pending(study):
    root,value=study
    path=root/'results/persona_adaptive_public.json';path.unlink()
    assert evidence(root,MODEL)['status']=='pending'
    value['rows']=[];write(path,seal(value))
    assert evidence(root,MODEL)['groups']==[]


def test_actual_fixture_and_saved_baseline_candidate_selection(study):
    root,_=study;out=evidence(root,MODEL)
    assert out['status']=='available' and out['prompt']=='who are you'
    assert [r['arm'] for r in out['cells']]==['baseline','candidate','reversal','random']
    assert out['cells'][0]['answer']=='A saved baseline answer'
    assert out['cells'][1]['answer']=='A saved candidate answer'
    assert out['cells'][2]['state']=='pending'
    assert evidence(root,MODEL,out['group_id'])==out
    with pytest.raises(ValueError,match='Unknown'):evidence(root,MODEL,'unknown')


@pytest.mark.parametrize('mutation',['model','fixture','menu','answer','program','source','duplicate','revision'])
def test_tampered_evidence_rejected_even_with_fresh_outer_digest(study,mutation):
    root,v=study
    if mutation=='model':v['binding']['model_identity']='other'
    elif mutation=='fixture':v['source_hashes']['identity_fixture']='0'*64
    elif mutation=='menu':
        v['menu']['candidates'][0]['target']='Changed'
        v['menu_projection_sha256']=canonical_digest(v['menu'])
    elif mutation=='answer':v['rows'][0]['answer']='Changed answer'
    elif mutation=='program':v['rows'][1]['program_sha256']='c'*64
    elif mutation=='source':v['rows'][0]['source_id']='wrong'
    elif mutation=='duplicate':v['rows'].append(deepcopy(v['rows'][0]))
    seal(v)
    if mutation=='revision':v['revision_sha256']='0'*64
    write(root/'results/persona_adaptive_public.json',v)
    with pytest.raises(ValueError):evidence(root,MODEL)


def test_fixture_neutral_context_is_enforced_even_with_recomputed_id(study):
    root,_=study;path=root/'data/identity_probe_questions.json';v=json.loads(path.read_text())
    v['baseline_probe']['system']='Pretend you are someone else'
    v['baseline_probe']['id']=canonical_digest({k:x for k,x in v['baseline_probe'].items() if k!='id'})
    write(path,v)
    with pytest.raises(ValueError,match='neutral'):load_probe(root)


def test_default_prefers_complete_pair_over_partial_group(study):
    root,v=study
    partial=deepcopy(v['rows'][0]);partial['seed']=1
    freeze_job(root,v,partial)
    v['rows'].insert(0,partial);write(root/'results/persona_adaptive_public.json',seal(v))
    assert evidence(root,MODEL)['seed']==80317
    selected=evidence(root,MODEL,canonical_digest([partial['candidate_id'],1]))
    assert selected['cells'][1]['state']=='pending'


@pytest.mark.parametrize('field,value',[('runtime_identity','other'),('prepared_identity','other')])
def test_exact_runtime_menu_binding_required(study,field,value):
    root,v=study;v['binding']=deepcopy(v['binding']);v['binding'][field]=value
    write(root/'results/persona_adaptive_public.json',seal(v))
    with pytest.raises(ValueError,match='binding'):evidence(root,MODEL)


@pytest.mark.parametrize('field,value',[('program_sha256','f'*64),('seed',17),('arm','random'),('phase','prefill'),('candidate_id','unknown')])
def test_all_arm_frozen_job_fields_required(study,field,value):
    root,v=study;v['rows'][0][field]=value
    write(root/'results/persona_adaptive_public.json',seal(v))
    with pytest.raises(ValueError):evidence(root,MODEL)


@pytest.mark.parametrize('field,value',[('capture_validated',False),('state','failed'),('complete',False),('answer',None)])
def test_answer_completion_consistency(study,field,value):
    root,v=study;v['rows'][0][field]=value
    write(root/'results/persona_adaptive_public.json',seal(v))
    with pytest.raises(ValueError):evidence(root,MODEL)


def test_frozen_job_content_hash_rechecked(study):
    root,v=study;jid=v['rows'][0]['job_id']
    with sqlite3.connect(root/'results/persona_adaptive.sqlite') as db:
        job=json.loads(db.execute('SELECT body FROM jobs WHERE id=?',(jid,)).fetchone()[0]);job['seed']=99
        db.execute('UPDATE jobs SET body=? WHERE id=?',(json.dumps(job),jid))
    with pytest.raises(ValueError,match='content'):evidence(root,MODEL)


def test_truthful_partial_answer_remains_incomplete(study):
    root,v=study;v['rows'][0].update(state='incomplete',complete=False)
    write(root/'results/persona_adaptive_public.json',seal(v))
    assert evidence(root,MODEL)['cells'][0]['state']=='incomplete'
