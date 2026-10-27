"""Saved authored evidence reader; no inference or queue access."""
import persona
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import pytest
from persona_vectors import canonical_digest
import persona_sculptor as m

@pytest.fixture
def study(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();model=SimpleNamespace(identity='a'*64)
    pushes=[{'kind':'trait','layer':0,'trait':'openness','dose':.1}]
    def program(pushes,phase):return {'phase':phase,'sites':[{'site':'layers.0.residual','vector':[sum(p['dose'] for p in pushes)]}]} if pushes else {'phase':phase,'sites':[]}
    class Bank:
        def __init__(self,*a):pass
        def program(self,p,phase):return program(p,phase),{}
    monkeypatch.setattr('persona_vectors.VectorBank',Bank);monkeypatch.setattr('persona_preset_overnight.attach_column',lambda b,p:b)
    prompt={'id':'prompt','pair_id':'family','prompt':'A fictional shared scene.','category':'test'}
    recipe={'id':'r','target':'experimental-spark','pushes':pushes,'phase':'all','program_sha256':canonical_digest(program(pushes,'all')),'relative_norms':{'layers.0.residual':.1}}
    plan={'model_identity':model.identity,'recipes':[recipe],'discovery':[prompt],'confirmation':[],'seeds':[7]};plan['input_sha256']=canonical_digest(plan)
    selected={**recipe,'preset_id':recipe['target'],'recipe_id':'r','study_cycle':'primary','input_sha256':plan['input_sha256'],'evidence':{'plan_sha256':'b'*64},'state':'discovery_running'}
    monkeypatch.setattr(m,'evaluated_presets',lambda *a:{'presets':[selected]})
    rows=[];bound={'model_identity':model.identity,'runtime_identity':'c'*64,'prepared_identity':'d'*64}
    for arm in ('baseline','candidate','reversal','random'):
        p=[] if arm=='baseline' else [{'kind':'random','layer':0,'site':'residual','dose':.1,'seed':7}] if arm=='random' else deepcopy(pushes)
        if arm=='reversal':p[0]['dose']=-.1
        rows.append({'study':'primary','stage':'discovery','recipe_id':'r','target':recipe['target'],'prompt':prompt,'seed':7,'phase':'all','arm':arm,'pushes':p,'program_sha256':canonical_digest(program(p,'all')),'state':'complete','complete':True,'has_error':False,'answer':'Saved '+arm,'finish_reason':'stop','response_sha256':'e'*64,'binding':bound,'activation_artifact':{**bound,'sha256':'f'*64},'provenance_artifact':{**bound,'sha256':'f'*64}})
    public={'schema':'character-creation.preset-overnight-public.v1','plans':{'primary':{'sha256':'b'*64,'plan':plan}},'rows':rows}
    path=tmp_path/'results/persona_preset_overnight_public.json';path.write_text(json.dumps(public))
    return tmp_path,model,public,path


def test_saved_answers_exact_group_missing_arm_pending(study):
    root,model,value,path=study;r=m.preset_evidence(root,model,'experimental-spark')
    assert r['status']=='available' and len(r['groups'])==1 and r['scene']=='A fictional shared scene.' and r['seed']==7
    assert [c['label'] for c in r['cells']]==['baseline','zero','candidate','reversal','random']
    assert r['cells'][1]['state']=='pending' and r['cells'][1]['output'] is None
    assert r['cells'][0]['output']['answer']=='Saved baseline'
    assert m.preset_evidence(root,model,'experimental-spark',r['group_id'])==r


@pytest.mark.parametrize('change',['prompt','seed','recipe','program','plan','duplicate','binding','group'])
def test_saved_evidence_rejects_cross_scene_or_recipe_and_false_binding(study,change):
    root,model,value,path=study
    if change=='prompt':value['rows'][0]['prompt']=dict(value['rows'][0]['prompt'],prompt='Different scene')
    elif change=='seed':value['rows'][0]['seed']=8
    elif change=='recipe':value['rows'][1]['pushes'][0]['dose']=.2
    elif change=='program':value['rows'][0]['program_sha256']='f'*64
    elif change=='plan':value['plans']['primary']['sha256']='f'*64
    elif change=='duplicate':value['rows'].append(deepcopy(value['rows'][0]))
    elif change=='binding':value['rows'][0]['binding']['model_identity']='0'*64
    path.write_text(json.dumps(value))
    r=m.preset_evidence(root,model,'experimental-spark','0'*64 if change=='group' else '')
    assert r['status']=='unavailable' and not r['cells']


def test_partial_error_and_missing_publication_are_not_complete(study):
    root,model,value,path=study
    value['rows'][0].update(state='incomplete',complete=False,finish_reason='length')
    value['rows'][1].update(state='failed',complete=False,has_error=True,answer='Partial before failure')
    path.write_text(json.dumps(value));r=m.preset_evidence(root,model,'experimental-spark')
    assert r['cells'][0]['state']=='incomplete' and not r['cells'][0]['output']['complete']
    assert r['cells'][2]['state']=='failed' and r['cells'][2]['error']
    path.unlink();assert m.preset_evidence(root,model,'experimental-spark')['status']=='pending'


def test_caller_cannot_choose_path_or_unregistered_target(study):
    root,model,*_=study
    with pytest.raises(ValueError):m.preset_evidence(root,model,'../../private')
    with pytest.raises(ValueError):m.preset_evidence(root,model,'experimental-spark','../../other.json')
