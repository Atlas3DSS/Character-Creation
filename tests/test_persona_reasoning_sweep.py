import persona
from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
import persona_reasoning_battery as battery_module
import persona_reasoning_sweep as sweep
from persona_vectors import canonical_digest
from persona_manager import Queue


def battery_fixture():
    algebra=battery_module.algebra_cases([{'question':f'Solve x = {i}.','answer':str(i)} for i in range(32)])
    general=[]
    for i in range(16):
        row={'id':str(i),'text':f'Source planning text {i}'};prompt=f'Explain a planning approach for scenario {i}.'
        general.append({'id':battery_module.digest(['general',str(i)]),'kind':'general','question_id':str(i),'prompt':prompt,'expected_answer':None,
            'source_rows':[i],'source_row':row,'source_row_sha256':battery_module.digest(row),'source':battery_module.SOURCES['general'],
            'scoring':'prompt_only_no_ground_truth','split':'discovery' if i<12 else 'heldout','translation_sha256':battery_module.digest(prompt),
            'translation':'assistant_authored_English','original_text':row['text']})
    value={'schema':'character-creation.reasoning-battery.v1','sources':battery_module.SOURCES,'cases':algebra+general,'source_snapshot_sha256':'a'*64,
        'split_counts':{'algebra':{'discovery':24,'heldout':8},'general':{'discovery':12,'heldout':4}}}
    value['sha256']=battery_module.digest(value);return value


def program(arm):return {'phase':'all','sites':[] if arm=='baseline' else [{'site':'layers.32.residual','vector':[float(sweep.ARMS.index(arm))]}]}


def prepared_fixture(monkeypatch):
    battery=battery_fixture();target={'model_identity':'a'*64,'runtime_identity':'b'*64,'prepared_identity':'c'*64};calls=[]
    def gated(*args,**kwargs):
        calls.append(kwargs)
        return {'jobs':[{'arm':a,'pushes':[] if a=='baseline' else [{'kind':'random' if a=='random' else 'direction','layer':32,'dose':.1}],
            'phase':'all','program_sha256':canonical_digest(program(a)),'compatibility_sha256':None if a in ('baseline','random') else 'd'*64,'calibration_sha256':'e'*64} for a in sweep.ARMS]*4}
    monkeypatch.setattr(sweep,'prepare_manager_pilot',gated)
    plan={'input_sha256':'f'*64};spec={'path':'receipt','sha256':'d'*64}
    prepared=sweep.build(battery,plan,SimpleNamespace(model=SimpleNamespace(identity=target['model_identity'])),target,qualification_spec=spec,compatibility_spec=spec)
    assert len(calls)==1 and calls[0]['compatibility_spec']==spec
    return battery,prepared,target


def test_exact_question_arm_coverage_source_splits_and_neutral_requests(monkeypatch):
    battery,p,target=prepared_fixture(monkeypatch);sweep.validate_input(p,battery)
    assert len(p['jobs'])==288 and len(p['corpus'])==48
    assert sum(j['prompt']['split']=='heldout' for j in p['jobs'])==72
    assert p['config_patch']['max_total_jobs']==288
    assert all(j['seed']==80317 and j['prompt']['max_tokens']==2048 and j['prompt']['system']=='' for j in p['jobs'])
    assert all('expected_answer' not in j['prompt'] and not j['promotion_eligible'] for j in p['jobs'])
    damaged=deepcopy(p);damaged['jobs'][0]['prompt']['split']='heldout'
    with pytest.raises(ValueError):sweep.validate_input(damaged,battery)
    changed=deepcopy(battery);changed['cases'][0]['source_row']['answer']='999'
    with pytest.raises(ValueError):sweep.validate_input(p,changed)


def test_saved_numeric_scoring_general_unscored_and_public_redaction(tmp_path,monkeypatch):
    battery,p,target=prepared_fixture(monkeypatch);queue=Queue(tmp_path/'queue.sqlite');queue.enqueue(p['proposal'],p['jobs'],limit=288)
    cases={c['id']:c for c in battery['cases']}
    selected=[p['jobs'][0],p['jobs'][1],p['jobs'][2],next(j for j in p['jobs'] if cases[j['battery_case_id']]['kind']=='general')]
    states=['correct','incorrect','unscorable','unscored']
    for i,job in enumerate(selected):
        case=cases[job['battery_case_id']]
        answer=('FINAL_ANSWER: '+case['expected_answer']) if i==0 else 'FINAL_ANSWER: 99999' if i==1 else 'Maybe, not sure.' if i==2 else 'A PRIVATE_GENERATED_PLAN.'
        request={'model':'checkpoint','messages':[{'role':'user','content':job['prompt']['prompt']}],'seed':80317,'max_tokens':2048,'steering':program(job['arm'])}
        pointer={'request_sha256':canonical_digest(request),**target,'sha256':'f'*64}
        response={**target,'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':answer}}],'diagnostics':{'complete_final_answer':True},'activation_artifact':pointer,'provenance_artifact':pointer}
        result={'complete':True,'error':None,'binding':target,'response':response,'request':request,'program_sha256':job['program_sha256']}
        with queue.connect() as db:
            db.execute("UPDATE jobs SET state='complete',attempts=1 WHERE id=?",(job['id'],))
            db.execute("INSERT INTO attempts(job_id,number,endpoint,started,state,result) VALUES (?,1,'synthetic',0,'complete',?)",(job['id'],json.dumps(result)))
    local,public=sweep.summarize(queue.path,p,battery);index={r['job_id']:r for r in local['rows']}
    assert [index[j['id']]['assessment']['state'] for j in selected]==states
    assert sum(r['assessment']['state']=='not_evaluated' for r in local['rows'])==284
    text=json.dumps(public)
    assert 'PRIVATE_GENERATED_PLAN' not in text and 'FINAL_ANSWER' not in text
    assert all(c['prompt'] not in text for c in battery['cases'])
    assert all(key not in text for key in ('expected_answer','source_row"','parsed"','reference"','"answer"'))
    assert local['counts']['algebra']['discovery']['baseline']['assessment']['correct']==1
    with queue.connect() as db:
        wrong=deepcopy(selected[0]);wrong['prompt']['split']='heldout';db.execute('UPDATE jobs SET body=? WHERE id=?',(json.dumps(wrong),wrong['id']))
    with pytest.raises(ValueError,match='Persisted Manager job'):sweep.summarize(queue.path,p,battery)
