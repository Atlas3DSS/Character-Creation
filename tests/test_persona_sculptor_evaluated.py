"""Offline overlay binding; no worker/model access."""
import persona
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import pytest
from persona_vectors import canonical_digest
from persona_sculptor import evaluated_presets
from persona_preset_overnight import TARGETS


def write(path,value):path.write_text(json.dumps(value));return hashlib.sha256(path.read_bytes()).hexdigest()

@pytest.fixture
def overlay(tmp_path,monkeypatch):
    (tmp_path/'data').mkdir();(tmp_path/'results').mkdir()
    model=SimpleNamespace(identity='model',layers=64)
    class Bank:
        def __init__(self,*args):pass
        def program(self,pushes,phase):return {},{'program_sha256':canonical_digest([pushes,phase])}
    monkeypatch.setattr('persona_vectors.VectorBank',Bank)
    monkeypatch.setattr('persona_preset_overnight.attach_column',lambda bank,path:bank)
    sources={p:write(tmp_path/p,{}) for p in ['data/persona_presets.json','data/persona_preset_candidates.json','data/persona_personality_probes.json','results/persona_preset_columns.json']}
    cal=tmp_path/'results/layer_00.npz';cal.write_bytes(b'calibration')
    recipes=[]
    for target in TARGETS:
        for factor in (.75,1):
            pushes=[{'kind':'trait','layer':0,'trait':'openness','dose':.1*factor}]
            recipes.append({'id':target+str(factor),'target':target,'name':target,'factor':factor,'pushes':pushes,'phase':'all','program_sha256':canonical_digest([pushes,'all']),'aggregate_relative_norm':.1*factor})
    plan={'schema':'character-creation.preset-overnight.v1','model_identity':'model','sources':sources,'calibration_sha256':{'0':hashlib.sha256(cal.read_bytes()).hexdigest()},'recipes':recipes}
    plan['input_sha256']=canonical_digest(plan);ph=write(tmp_path/'results/persona_preset_overnight_plan.json',plan)
    rows=[]
    for recipe in recipes[1::2]:rows.append({'preset_id':recipe['target'],'recipe_id':recipe['id'],'model_identity':'model','input_sha256':plan['input_sha256'],'calibration_sha256':plan['calibration_sha256'],'state':'discovery_running','evidence':{'plan_sha256':ph,'selection_sha256':None,'confirmation_sha256':None},**{k:recipe[k] for k in ['name','pushes','phase','program_sha256','aggregate_relative_norm']}})
    value={'schema':'character-creation.evaluated-presets.v1','catalogue_source_unchanged':True,'input_sha256':plan['input_sha256'],'updated':1,'presets':rows,'automatically_battery_qualified':0}
    path=tmp_path/'results/persona_preset_evaluated.json';write(path,value)
    return tmp_path,model,plan,value,path


def test_pending_exact_programs_usable_no_historical_mutation(overlay):
    root,model,plan,value,path=overlay;before=(root/'data/persona_presets.json').read_bytes()
    result=evaluated_presets(root,model)
    assert result['status']=='available' and len(result['presets'])==8 and result['qualified']==0
    assert all(r['program_valid'] and r['state']=='discovery_running' for r in result['presets'])
    assert result['presets'][0]['pushes']==value['presets'][0]['pushes']
    assert (root/'data/persona_presets.json').read_bytes()==before


@pytest.mark.parametrize('change',['model','program','dose','evidence','claimed_qualified','plan','calibration','source','missing'])
def test_bad_or_missing_evidence_never_qualifies_or_breaks_catalog(overlay,change):
    root,model,plan,value,path=overlay
    if change=='model':model.identity='wrong'
    elif change=='program':value['presets'][0]['program_sha256']='0'*64
    elif change=='dose':value['presets'][0]['pushes'][0]['dose']=.2
    elif change=='evidence':value['presets'][0]['evidence']['plan_sha256']='0'*64
    elif change=='claimed_qualified':value['presets'][0]['state']='automatically_battery_qualified'
    elif change=='plan':plan['model_identity']='wrong';write(root/'results/persona_preset_overnight_plan.json',plan)
    elif change=='calibration':(root/'results/layer_00.npz').write_bytes(b'changed')
    elif change=='source':write(root/'data/persona_presets.json',{'changed':True})
    elif change=='missing':path.unlink()
    if change!='missing':write(path,value)
    r=evaluated_presets(root,model)
    assert r['status'] in ('unavailable','pending') and r['presets']==[]


def qualified(overlay):
    root,model,plan,value,path=overlay;rid=value['presets'][0]['recipe_id']
    selection={'schema':'character-creation.preset-discovery-selection.v1','input_sha256':plan['input_sha256'],'selected':[rid]}
    sh=write(root/'results/persona_preset_selection.json',selection)
    evidence=[{arm:canonical_digest([g,arm]) for arm in ['baseline','candidate','reversal','random']} for g in range(12)]
    report={'schema':'character-creation.preset-overnight-results.v1','input_sha256':plan['input_sha256'],'selection_sha256':canonical_digest(selection),'status':'complete','automatically_battery_qualified':[rid],'candidates':[{'recipe_id':rid,'battery_pass':True,'groups':12,'clean_groups':12,'behavior_wins':8,'evidence':evidence}],'cross_target_distinctness':[]}
    rh=write(root/'results/persona_preset_overnight_results.json',report)
    for i,row in enumerate(value['presets']):row['state']='automatically_battery_qualified' if i==0 else 'not_qualified';row['evidence'].update(selection_sha256=sh,confirmation_sha256=rh)
    value['automatically_battery_qualified']=1;write(path,value)
    return report


def test_bound_confirmation_qualifies_only_automatic_battery(overlay):
    qualified(overlay);root,model,*_=overlay;r=evaluated_presets(root,model)
    assert r['status']=='available' and r['qualified']==1
    assert 'not human' in r['scope'] and r['presets'][0]['state']=='automatically_battery_qualified'


def test_rehashed_but_failed_gate_and_duplicate_evidence_rejected(overlay):
    report=qualified(overlay);root,model,plan,value,path=overlay
    report['candidates'][0]['evidence'][1]=deepcopy(report['candidates'][0]['evidence'][0])
    rh=write(root/'results/persona_preset_overnight_results.json',report)
    for row in value['presets']:row['evidence']['confirmation_sha256']=rh
    write(path,value);assert evaluated_presets(root,model)['status']=='unavailable'


def test_frontend_handoff_preserves_exact_recipe_and_labels(tmp_path):
    import subprocess
    from pathlib import Path
    source=(Path(__file__).resolve().parents[1]/'ui/persona_sculptor.js').read_text()
    render=source[source.index('  function renderEvaluatedPresets()'):source.index('  function renderPresets()')]
    script="""
const assert=require('node:assert/strict');
function node(tag,text,cls){return {tag,text,cls,children:[],append(...xs){this.children.push(...xs)},replaceChildren(){this.children=[]},setAttribute(){}}}
const button=(text,action)=>Object.assign(node('button',text),{onclick:action});
const row=(...xs)=>{let n=node('div');n.append(...xs);return n};
const hint=text=>node('p',text), clone=x=>JSON.parse(JSON.stringify(x));
const refreshEvaluatedPresets=()=>{};let accepted=[];
const window={PersonaLab:{useCurrentRecipe:r=>accepted.push(r)}};
const preset={name:'Bound candidate',recipe_id:'r',state:'discovery_running',program_valid:true,pushes:[{kind:'trait',layer:32,trait:'openness',dose:0.037891}],phase:'decode',evidence:{plan_sha256:'a'}};
const refs={evaluated:node('section')},app={meta:{evaluated_presets:{status:'available',qualified:0,presets:[preset]}}};
"""+render+"""
renderEvaluatedPresets();
const walk=n=>[n,...n.children.flatMap(walk)];let nodes=walk(refs.evaluated);
assert(nodes.some(n=>n.text==='Discovery running · experimental'));
nodes.find(n=>n.text==='Load exact recipe →').onclick();
assert.deepEqual(accepted,[{name:preset.name,pushes:preset.pushes,phase:'decode'}]);
accepted[0].pushes[0].dose=9;assert.equal(preset.pushes[0].dose,0.037891);
app.meta.evaluated_presets={status:'unavailable',presets:[],message:'Evidence unavailable'};renderEvaluatedPresets();
assert(!walk(refs.evaluated).some(n=>n.text==='Load exact recipe →'));
"""
    result=subprocess.run(['node','-e',script],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def followon(overlay, *, passed=True, complete=True):
    qualified(overlay)
    root,model,plan,value,path=overlay
    (root/'persona').mkdir(exist_ok=True)
    for name in ('persona_preset_followon.py','persona_preset_overnight.py'):(root/'persona'/name).write_text('# frozen implementation\n')
    fp=deepcopy(plan);fp['cycle']='fresh_context_followon'
    fp['sources'].update({p:write(root/p,{}) for p in ['data/persona_perturbation_prompts.jsonl','data/persona_perturbation_prompts_manifest.json']})
    fp['sources']['results/persona_preset_selection.json']=hashlib.sha256((root/'results/persona_preset_selection.json').read_bytes()).hexdigest()
    fp['implementation_sha256']={k:hashlib.sha256((root/'persona'/n).read_bytes()).hexdigest() for k,n in [('followon','persona_preset_followon.py'),('shared','persona_preset_overnight.py')]}
    for recipe in fp['recipes']:
        recipe['source_discovery_recipe']=recipe['id'];recipe['id']+='-followon';recipe['change']='Frozen redistributed experiment'
        recipe['pushes'][0]['dose']*=.85;recipe['program_sha256']=canonical_digest([recipe['pushes'],recipe['phase']])
    fp.pop('input_sha256');fp['input_sha256']=canonical_digest(fp);write(root/'results/persona_preset_followon_plan.json',fp)
    rid=fp['recipes'][1]['id']
    selection={'schema':'character-creation.preset-discovery-selection.v1','input_sha256':fp['input_sha256'],'selected':[rid]}
    write(root/'results/persona_preset_followon_selection.json',selection)
    if complete:
        report=json.loads((root/'results/persona_preset_overnight_results.json').read_text())
        report.update(input_sha256=fp['input_sha256'],selection_sha256=canonical_digest(selection),automatically_battery_qualified=[rid] if passed else [])
        report['candidates'][0].update(recipe_id=rid,battery_pass=passed,clean_groups=12 if passed else 10)
        write(root/'results/persona_preset_followon_results.json',report)
    return fp


def test_followon_qualified_exact_recipe_replaces_one_target_only(overlay):
    fp=followon(overlay);root,model,plan,value,path=overlay
    r=evaluated_presets(root,model)
    assert r['status']=='available' and len(r['presets'])==8
    assert len({x['preset_id'] for x in r['presets']})==8
    chosen=r['presets'][0]
    assert chosen['study_cycle']=='followon' and chosen['recipe_id']==fp['recipes'][1]['id']
    assert chosen['pushes']==fp['recipes'][1]['pushes'] and chosen['phase']==fp['recipes'][1]['phase']
    assert chosen['previous_recipe_id']==value['presets'][0]['recipe_id']
    assert r['followon']['stage']=='confirmation_complete'
    assert all(x['study_cycle']=='primary' for x in r['presets'][1:])


@pytest.mark.parametrize('complete,passed',[(True,False),(False,False)])
def test_followon_failed_or_pending_preserves_qualified_primary(overlay,complete,passed):
    followon(overlay,complete=complete,passed=passed);root,model,plan,value,path=overlay
    r=evaluated_presets(root,model)
    assert r['presets'][0]['recipe_id']==value['presets'][0]['recipe_id']
    assert r['presets'][0]['state']=='automatically_battery_qualified'
    assert r['followon']['stage']==('confirmation_complete' if complete else 'awaiting_confirmation')


def test_missing_followon_normal_and_invalid_evidence_keeps_incumbent(overlay):
    qualified(overlay);root,model,plan,value,path=overlay
    assert evaluated_presets(root,model)['followon']['status']=='pending'
    followon(overlay)
    p=root/'results/persona_preset_followon_results.json';r=json.loads(p.read_text());r['candidates'][0]['behavior_wins']=2;write(p,r)
    out=evaluated_presets(root,model)
    assert out['followon']['status']=='unavailable'
    assert out['presets'][0]['recipe_id']==value['presets'][0]['recipe_id'] and out['qualified']==1
