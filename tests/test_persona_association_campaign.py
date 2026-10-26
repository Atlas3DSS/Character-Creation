from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest
import persona_association_campaign as campaign


def test_six_matching_calibrated_search_workers_required():
    model={'data':[{'flare':{'model_identity':'m','runtime_identity':'r','prepared_identity':'p',
        'activation_search':{'supported':True},'calibration_equivalence':{'passed':True},
        'kv_control':{'layout_sha256':'l'}}}]}
    models=[deepcopy(model) for _ in range(6)]
    assert campaign.matching_binding(models,'m')['runtime_identity']=='r'
    with pytest.raises(ValueError):campaign.matching_binding(models[:5],'m')
    for field,value in [('runtime_identity','different'),('model_identity','different'),
                        ('activation_search',{'supported':False}),('calibration_equivalence',{'passed':False})]:
        bad=deepcopy(models);bad[-1]['data'][0]['flare'][field]=value
        with pytest.raises(ValueError):campaign.matching_binding(bad,'m')


def test_uncertain_post_is_recorded_and_never_retried(tmp_path):
    run=campaign.Campaign.__new__(campaign.Campaign)
    run.path=tmp_path/'state.json';record={};run.plan={'record':record};calls=[]
    def fail(url,value):calls.append((url,value));raise TimeoutError('uncertain')
    run.request=fail
    with pytest.raises(TimeoutError):run.submit(record,'/api/activation-searches',{'text':'scene','top_k':16})
    assert len(calls)==1
    assert json.loads(run.path.read_text())['record']['submission']['state']=='failed_or_uncertain'


@pytest.mark.parametrize('failed', [False, True])
def test_comparisons_keep_bounded_queue_and_report_failed_cells(tmp_path, monkeypatch, failed):
    run=campaign.Campaign.__new__(campaign.Campaign)
    run.path=tmp_path/'state.json'
    run.plan={'binding':BINDING,'queries':[{'kind':'whole_prompt','text':'scene','search_id':str(i),
        'starter_recipes':[{'id':'single','neurons':[{'layer':1,'neuron':2}]}]}
        for i in range(24)]}
    run.guard=lambda:None
    pending=set();submitted=[]
    def submit(record,route,value):
        key=str(len(submitted));submitted.append(value);pending.add(key);record['submission']={'request':deepcopy(value)}
        assert len(pending)<=8
        return key
    def read(route):
        key=route.split('=')[1];pending.remove(key)
        result=batch(submitted[int(key)],key)
        if failed and key=='0':
            result['state']='failed'
            for cell in result['cells']:cell['state']='failed';cell.pop('output')
        return result
    run.submit=submit;run.request=read
    monkeypatch.setattr(campaign.time,'sleep',lambda _:None)
    run.comparisons()
    assert len(submitted)==24 and not pending
    assert run.plan['state']==('complete_with_failed_comparisons' if failed else 'complete')


BINDING=dict(model_identity='m',runtime_identity='r',prepared_identity='p',layout_sha256='l')

def batch(request,key='batch'):
    raw=[];cells=[]
    for index,(label,sign) in enumerate(zip(('Baseline','Positive','Negative'),(0,1,-1))):
        recipe={'name':label,'phase':'all','pushes':[] if not sign else [{'kind':'neuron',**n,'dose':sign*request['dose']/len(request['neurons'])} for n in request['neurons']]}
        raw.append({'label':label,'prompt':request['prompt'],'recipe':deepcopy(recipe)})
        cells.append({'index':index,'label':label,'state':'complete','prompt':{'text':request['prompt']},'recipe':recipe,'output':{**BINDING,'answer':'Response','complete':True,'finish_reason':'stop'}})
    return {'id':key,'state':'complete','input':{'model_identity':'m','seed':request['seed'],'max_tokens':request['max_tokens'],'thinking':False,'smoke':False,'cells':raw,'admitted_endpoints':[BINDING]},'cells':cells}


def test_frozen_actual_plan_and_mutations(tmp_path):
    root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/'results/persona_association_research.json').read_text())
    launch={'input_plan_sha256':campaign.input_plan_digest(plan)}
    path=root/'data/persona_phantom_pairs.json'
    assert campaign.validate_plan(plan,launch,path)==launch['input_plan_sha256']
    for field,value in [('top_k',32),('expected_searches',59),('automatic_default_promotion',True)]:
        changed=deepcopy(plan);changed[field]=value
        with pytest.raises(ValueError):campaign.validate_plan(changed,launch,path)
    changed=deepcopy(plan);changed['queries'][0]['text']='changed'
    with pytest.raises(ValueError):campaign.validate_plan(changed,launch,path)
    launch['input_plan_sha256']=campaign.input_plan_digest(changed)
    with pytest.raises(ValueError):campaign.validate_plan(changed,launch,path)
    changed=deepcopy(plan);changed['queries'][0]['split']='heldout';launch['input_plan_sha256']=campaign.input_plan_digest(changed)
    with pytest.raises(ValueError):campaign.validate_plan(changed,launch,path)
    wrong=tmp_path/'pairs.json';wrong.write_text(path.read_text()+' ')
    with pytest.raises(ValueError):campaign.validate_plan(plan,{'input_plan_sha256':campaign.input_plan_digest(plan)},wrong)


def test_activation_only_reads_config_and_verifies_manual_binding(tmp_path):
    run=campaign.Campaign.__new__(campaign.Campaign);run.root=tmp_path;run.guard=lambda:None;run.save=lambda:None
    run.launch={'observed_binding':BINDING};run.plan={}
    config={'endpoints':[{'url':url,**BINDING} for url in campaign.URLS]};path=tmp_path/'persona_explorer.local.json';path.write_text(json.dumps(config));before=path.read_bytes()
    calls=[]
    def request(url,timeout):
        calls.append(url);return {'data':[{'flare':{**BINDING,'activation_search':{'supported':True},'calibration_equivalence':{'passed':True},'kv_control':{'layout_sha256':'l'}}}]}
    run.request=request;run.activate();assert path.read_bytes()==before;assert len(calls)==6;assert run.plan['binding']==BINDING
    run.launch={'observed_binding':{**BINDING,'runtime_identity':'wrong'}}
    with pytest.raises(ValueError):run.activate()
    run.launch={'observed_binding':BINDING};config['endpoints'][0]['runtime_identity']='wrong';path.write_text(json.dumps(config))
    with pytest.raises(ValueError):run.activate()


def test_batch_snapshots_fail_closed_on_changed_arm_or_binding():
    request={'prompt':'exact scene','neurons':[{'layer':1,'neuron':2}],'dose':.1,'seed':7035,'max_tokens':2048}
    record={'batch_id':'batch','submission':{'request':request}};good=batch(request)
    campaign.check_batch(good,record,BINDING)
    for mutate in (lambda b:b['input'].update(seed=1),lambda b:b['cells'][0]['prompt'].update(text='changed'),
                   lambda b:b['cells'][1]['recipe'].update(phase='prefill'),lambda b:b['cells'][2]['recipe']['pushes'][0].update(dose=.1),
                   lambda b:b['cells'][0]['output'].update(runtime_identity='different'),lambda b:b['cells'].pop()):
        bad=deepcopy(good);mutate(bad)
        with pytest.raises(ValueError):campaign.check_batch(bad,record,BINDING)


def test_search_exact_input_and_source_binding():
    query={'search_id':'search','text':'exact'};plan={'top_k':16,'binding':BINDING}
    output={**BINDING,'request_sha256':'request','activation_artifact':{**BINDING,'request_sha256':'request'},'profile_artifact':BINDING,'candidates':[{'layer':1,'neuron':2}]}
    result={'id':'search','input':{'text':'exact','top_k':16,'model_identity':'m'},'output':output,'starter_recipes':[{'id':'single','neurons':[{'layer':1,'neuron':2}],'dose':.1,'status':'untested','evidence':'current_input_activity'}]}
    campaign.check_search(result,query,plan)
    for mutate in (lambda b:b['input'].update(text='changed'),lambda b:b['input'].update(top_k=32),lambda b:b['output'].update(runtime_identity='other'),lambda b:b['starter_recipes'][0].update(neurons=[{'layer':2,'neuron':3}])):
        bad=deepcopy(result);mutate(bad)
        with pytest.raises(ValueError):campaign.check_search(bad,query,plan)


def depth_fixture(tmp_path):
    import time
    root=Path(__file__).resolve().parents[1]
    source=json.loads((root/'results/persona_association_depth.json').read_text())
    source['campaigns']=[{k:r[k] for k in ('source','context_id','state')} for r in source['campaigns']]
    for name in ('results','data'):(tmp_path/name).mkdir()
    (tmp_path/'results/persona_association_depth.json').write_text(json.dumps(source))
    (tmp_path/'data/persona_phantom_pairs.json').write_bytes((root/'data/persona_phantom_pairs.json').read_bytes())
    binding={**BINDING,'model_identity':source['binding']['model_identity']}
    (tmp_path/'results/persona_association_launch.json').write_text(json.dumps({'observed_binding':binding,'deadline':time.time()+3600}))
    return binding


def test_depth_plan_is_exact_frozen150_without_posts(tmp_path):
    binding=depth_fixture(tmp_path);plan=campaign.prepare_recipe_depth(tmp_path);frozen=plan['input']
    assert len(frozen['batches'])==25
    cells=[c for b in frozen['batches'] for c in b['request']['cells']]
    assert len(cells)==150 and sum(not c['recipe']['pushes'] for c in cells)==6
    assert all(c['recipe']['phase']=='all' and 'kv_control' not in c for c in cells)
    assert frozen['prior_phase']=='prefill' and 'not fresh' in frozen['heldout_scope']
    assert all(len(b['request']['cells'])==6 and b['request']['max_tokens']==2048 for b in frozen['batches'])
    run=campaign.RecipeDepthCampaign(tmp_path);assert run.input_hash==plan['input_plan_sha256']
    with pytest.raises(ValueError):campaign.prepare_recipe_depth(tmp_path)
    path=tmp_path/'results/persona_recipe_depth.json';plan['input']['phase']='prefill';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):campaign.RecipeDepthCampaign(tmp_path)


def depth_response(request,key,binding):
    return {'id':key,'state':'complete','input':deepcopy(request), 'cells':[
        {'index':i,**deepcopy(cell),'prompt':{'text':cell['prompt']},'state':'complete',
         'output':{**binding,'complete':True,'answer':'A reply.','finish_reason':'stop'}} for i,cell in enumerate(request['cells'])]}


def test_depth_rolling_four_preserves_snapshots_and_incomplete(tmp_path,monkeypatch):
    binding=depth_fixture(tmp_path);campaign.prepare_recipe_depth(tmp_path);run=campaign.RecipeDepthCampaign(tmp_path)
    run.plan['binding']=binding;pending={};submissions=[]
    def request(route,value=None):
        if value is not None:
            assert route=='/api/batches';key=str(len(submissions));submissions.append(deepcopy(value));pending[key]=value;assert len(pending)<=4;return {'id':key}
        key=route.split('=')[1];value=pending.pop(key);result=depth_response(value,key,binding)
        if key=='1':result['state']='partial';result['cells'][0].update(state='incomplete');result['cells'][0]['output'].update(complete=False,finish_reason='length')
        return result
    run.request=request;monkeypatch.setattr(campaign.time,'sleep',lambda _:None);run.comparisons()
    assert len(submissions)==25 and not pending
    assert run.plan['response_counts']=={'complete':149,'incomplete':1}
    assert run.plan['state']=='complete_with_failed_comparisons'
    assert all('input_snapshot' in b and len(b['cells'])==6 for b in run.plan['batches'])
    with pytest.raises(ValueError):campaign.RecipeDepthCampaign(tmp_path)


def test_depth_rejects_kv_binding_and_dose_changes(tmp_path):
    binding=depth_fixture(tmp_path);plan=campaign.prepare_recipe_depth(tmp_path)
    request=plan['input']['batches'][1]['request'];record={'batch_id':'batch','submission':{'request':request}}
    good=depth_response(request,'batch',binding);campaign.check_depth_batch(good,record,binding)
    for mutate in (lambda b:b['input'].update(seed=1),lambda b:b['cells'][0].update(kv_control={'dose':1}),
                   lambda b:b['cells'][0]['recipe']['pushes'][0].update(dose=.9),
                   lambda b:b['cells'][0]['output'].update(runtime_identity='other')):
        bad=deepcopy(good);mutate(bad)
        with pytest.raises(ValueError):campaign.check_depth_batch(bad,record,binding)
