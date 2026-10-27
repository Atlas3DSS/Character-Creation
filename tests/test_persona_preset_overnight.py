import persona
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import persona_preset_overnight as study


class Bank:
    model=SimpleNamespace(identity='389801c06d68a0d84bd4890b8ce6d26f96e0ee53eb108bcc572c0c09abf5b9a8')
    def program(self,pushes,phase,**kwargs):
        sites={}
        for p in pushes:
            site=f"layers.{p['layer']}."+('mlp_input' if p['kind']=='neuron' else p.get('site','residual'))
            sites[site]=sites.get(site,0)+p['dose']
        program={'phase':phase,'sites':[{'site':k,'vector':[v]} for k,v in sorted(sites.items())]}
        return program,{'aggregate_scale':1.,'program_sha256':study.canonical_digest(program),'relative_norms':{k:abs(v) for k,v in sites.items()},'aggregate_relative_norm':sum(abs(v) for v in sites.values())}


def test_fixed_plan_sealed_confirmation_and_all_control_coverage(tmp_path):
    root=Path(__file__).resolve().parents[1];(tmp_path/'data').mkdir();(tmp_path/'results').mkdir()
    for name in ('persona_presets.json','persona_preset_candidates.json','persona_personality_probes.json'):
        (tmp_path/'data'/name).write_bytes((root/'data'/name).read_bytes())
    (tmp_path/'results/persona_preset_columns.json').write_text('{}')
    for layer in (16,32,40,44,48):(tmp_path/f'results/layer_{layer:02d}.npz').write_bytes(b'fixture')
    b=Bank();p=study.make_plan(tmp_path,b);jobs=study.jobs_for(p,b,'discovery')
    assert len(jobs)==960 and len({j['id'] for j in jobs})==960
    assert {j['target'] for j in jobs}==set(study.TARGETS)
    assert all(j['prompt']['split']=='discovery' and j['prompt']['system']=='' and j['prompt']['max_tokens']==2048 for j in jobs)
    assert not {c['prompt'] for c in p['confirmation']} & {j['prompt']['prompt'] for j in jobs}
    selected=[r['id'] for r in p['recipes'] if r['factor']==1]
    confirm=study.jobs_for(p,b,'confirmation',selected)
    assert len(confirm)==384 and all(j['prompt']['split']=='heldout' for j in confirm)
    assert {j['arm'] for j in confirm}=={'baseline','candidate','reversal','random'}
    with pytest.raises(ValueError,match='one variant'):study.jobs_for(p,b,'confirmation',[r['id'] for r in p['recipes'][:2]])
    p['confirmation'][0]['prompt']='changed'
    with pytest.raises(ValueError,match='digest'):study.jobs_for(p,b,'discovery')


def test_blinded_judge_exact_quotes_and_no_uncertain_retry(tmp_path,monkeypatch):
    import persona_sculptor
    monkeypatch.setattr(persona_sculptor,'trusted_planner',lambda:{'url':'http://127.0.0.1:1','model':'mock'})
    job={'id':'job','prompt':{'prompt':'Give one reply.','fact_contract':['There is one ticket.']}}
    response={'choices':[{'finish_reason':'stop','message':{'content':'One ticket is available.'}}],'diagnostics':{'complete_final_answer':True}}
    group=[{'job':job,'state':'complete','result':{'complete':True,'error':None,'response':response}}]
    plan={'input_sha256':'a'*64};calls=[]
    def fetch(url,payload,timeout):
        packet=json.loads(payload['messages'][1]['content']);calls.append(packet)
        assert 'arm' not in packet['answers'][0] and 'recipe' not in packet
        answer={'code':packet['answers'][0]['code'],'scores':{k:1 for k in study.BEHAVIORS},'fidelity':'supported','coherence':'coherent','evidence_ids':['invented-span'],'detail':'test'}
        return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'answers':[answer]})}}]}
    monkeypatch.setattr(study,'fetch',fetch);path=tmp_path/'judge.json'
    result=study.judge_group(plan,group,path)
    assert result['state']=='failed_no_retry' and 'absent or belongs' in result['error']
    assert study.judge_group(plan,group,path)==result and len(calls)==1


def test_selection_never_reads_heldout_or_promotes_without_evidence():
    with pytest.raises(ValueError,match='Heldout'):
        study.select_candidates({'recipes':[]},{'x':{'job':{'prompt':{'split':'heldout'}}}},[])
    result=study.select_candidates({'input_sha256':'a'*64,'recipes':[]},{},[])
    assert result['selected']==[] and 'No preset promotion' in result['status']


def test_status_cannot_promote_without_confirmation_evidence(tmp_path):
    (tmp_path/'results').mkdir()
    recipes=[{'id':t+'-1.0','target':t,'name':t,'factor':1,'pushes':[],'phase':'all','program_sha256':'a'*64,'aggregate_relative_norm':0} for t in study.TARGETS]
    plan={'model_identity':'m','recipes':recipes,'calibration_sha256':{}}
    plan['input_sha256']=study.canonical_digest(plan)
    (tmp_path/'results/persona_preset_overnight_plan.json').write_text(json.dumps(plan))
    status=study.publish_status(tmp_path)
    assert status['automatically_battery_qualified']==0 and all(p['state']=='discovery_running' for p in status['presets'])
    forged={'input_sha256':plan['input_sha256'],'status':'complete','automatically_battery_qualified':[recipes[0]['id']]}
    (tmp_path/'results/persona_preset_overnight_results.json').write_text(json.dumps(forged))
    with pytest.raises(ValueError,match='selection binding'):study.publish_status(tmp_path)


def test_span_id_protocol_resolves_source_and_preserves_old_review(tmp_path,monkeypatch):
    import persona_sculptor
    monkeypatch.setattr(persona_sculptor,'trusted_planner',lambda:{'url':'http://127.0.0.1:1'})
    job={'id':'job','prompt':{'prompt':'A factual reply.','fact_contract':['One ticket.']}}
    response={'choices':[{'finish_reason':'stop','message':{'content':'One ticket remains.'}}],'diagnostics':{'complete_final_answer':True}}
    group=[{'job':job,'state':'complete','result':{'complete':True,'error':None,'response':response}}]
    path=tmp_path/'review.json';old={'state':'failed_no_retry','response':{'choices':[{'finish_reason':'stop'}]},'error':'Old quote format failed'};path.write_text(json.dumps(old))
    def fetch(url,payload,timeout):
        answer=json.loads(payload['messages'][1]['content'])['answers'][0]
        result={'code':answer['code'],'scores':{k:0 for k in study.BEHAVIORS},'fidelity':'supported','coherence':'coherent',
            'evidence_ids':[answer['evidence_spans'][0]['id']],'detail':'Source-supported simple reply.'}
        return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'answers':[result]})}}]}
    monkeypatch.setattr(study,'fetch',fetch)
    record=study.judge_group({'input_sha256':'a'*64},group,path)
    assert record['state']=='validated' and record['protocol']=='answer_span_ids_v2'
    assert record['prior_attempts']==[old] and record['judgments'][0]['evidence']==['One ticket remains.']
    assert 'key' not in json.loads(record['request']['messages'][1]['content'])
