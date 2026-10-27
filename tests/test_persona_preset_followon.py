import persona
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import persona_preset_followon as follow
from test_persona_preset_overnight import Bank


def fixture(tmp_path):
    (tmp_path/'data').mkdir();(tmp_path/'results').mkdir();bank=Bank()
    recipes=[]
    for target in follow.base.TARGETS:
        for factor in (.75,1.):
            pushes=[{'kind':'trait','layer':40,'trait':'openness','dose':.1*factor}];_,p=bank.program(pushes,'all')
            recipes.append({'id':target+str(factor),'target':target,'name':target,'factor':factor,'pushes':pushes,'phase':'all',**p})
    old={'model_identity':bank.model.identity,'recipes':recipes,'sources':{},'discovery':[{'pair_id':'old_d','prompt':'old discovery'}],
        'confirmation':[{'pair_id':'old_h','prompt':'old confirmation'}],'discovery_max_jobs':960,'confirmation_max_jobs':384}
    old['input_sha256']=follow.canonical_digest(old)
    (tmp_path/'results/persona_preset_overnight_plan.json').write_text(json.dumps(old))
    selection={'input_sha256':old['input_sha256'],'evidence':[{'target':r['target'],'recipe_id':r['id'],'clean_groups':12,'behavior_wins':9,'zero_equal_groups':12} for r in recipes]}
    (tmp_path/'results/persona_preset_selection.json').write_text(json.dumps(selection))
    rows=[]
    for i in range(3072):
        prompt=f'An independently sourced fixture number {i}.'
        rows.append({'id':str(i),'pair_id':str(i),'category':str(i%11),'prompt':prompt,'split':'heldout' if i%5==0 else 'discovery','system':'',
            'source':{'kind':'synthetic_original','prompt_sha256':follow.hashlib.sha256(prompt.encode()).hexdigest()}})
    corpus=tmp_path/'data/persona_perturbation_prompts.jsonl';corpus.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest={'status':'complete','records':3072,'output_sha256':follow.base.sha(corpus)}
    mp=tmp_path/'data/persona_perturbation_prompts_manifest.json';mp.write_text(json.dumps(manifest))
    for layer in (16,24,32,40,44,48):(tmp_path/f'results/layer_{layer:02d}.npz').write_bytes(b'calibration')
    return bank,mp


def test_followon_fresh_splits_seeds_and_no_heldout_feedback(tmp_path):
    bank,mp=fixture(tmp_path)
    # A prior heldout result must not be read when constructing challengers.
    (tmp_path/'results/persona_preset_overnight_results.json').write_text('INVALID JSON; must not read')
    plan=follow.freeze(tmp_path,bank)
    assert len(plan['discovery'])==len(plan['confirmation'])==6
    assert len({c['pair_id'] for c in plan['discovery']+plan['confirmation']})==12
    assert {c['split'] for c in plan['discovery']}=={'discovery'} and {c['split'] for c in plan['confirmation']}=={'heldout'}
    jobs=follow.jobs(plan,bank,'discovery')
    assert len(jobs)==960 and {j['seed'] for j in jobs}==set(follow.SEEDS)
    assert all(j['prompt']['system']=='' for j in jobs)
    assert all(p['seed']==j['seed'] for j in jobs for p in j['pushes'] if p['kind']=='random')
    assert any(len({p['layer'] for p in r['pushes']})==6 for r in plan['recipes'])


def test_followon_waits_for_final_source_and_refuses_changed_hash(tmp_path):
    bank,mp=fixture(tmp_path);m=json.loads(mp.read_text());m['status']='partial';mp.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='complete'):follow.freeze(tmp_path,bank)
    m['status']='complete';m['output_sha256']='0'*64;mp.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='hash-verified'):follow.freeze(tmp_path,bank)
