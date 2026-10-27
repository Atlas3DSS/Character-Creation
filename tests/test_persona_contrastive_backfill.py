import hashlib
import json
from copy import deepcopy
import pytest
from persona_contrastive_backfill import choose,expand,prototypes,canonical_digest,SEED

def prompt(i,domain='a',split='discovery'):
    text='Original scene '+str(i)
    return {'id':str(i),'pair_id':'pair'+str(i),'prompt':text,'split':split,'category':domain,'system':'','thinking':False,'corpus':'synthetic','source':{'kind':'synthetic_original','prompt_sha256':hashlib.sha256(text.encode()).hexdigest()}}

def test_balanced_excludes_heldout_both_old_partitions_and_duplicate_text():
    rows=[prompt(i,'a' if i<10 else 'b') for i in range(20)]+[prompt(21,split='heldout')]
    rows.append({**prompt(22),'prompt':rows[5]['prompt'],'source':rows[5]['source']})
    plans=[{'discovery':[rows[0]],'confirmation':[rows[10]]}]
    picked=choose(rows,plans,8)
    assert len(picked)==8 and sum(r['category']=='a' for r in picked)==4
    assert all(r['split']=='discovery' and r['pair_id'] not in {'pair0','pair10'} for r in picked)
    assert len({r['prompt'] for r in picked})==8
    assert choose(list(reversed(rows)),plans,8)==picked

class Bank:
    def program(self,pushes,phase):return {'phase':phase,'sites':pushes},{}

def prototype_rows():
    rows=[]
    for i in range(8):
        for arm in ('baseline','candidate','reversal','random'):
            pushes=[] if arm=='baseline' else [{'kind':'random' if arm=='random' else 'trait','layer':16,'dose':-0.1 if arm=='reversal' else .1,'seed':SEED}]
            job={'recipe_id':f'r{i}-distributed','arm':arm,'pushes':pushes,'phase':'all','seed':SEED,'promotion_eligible':False,'model_identity':'model','prompt':prompt(0),'calibration_sha256':'cal','program_sha256':canonical_digest(Bank().program(pushes,'all')[0])}
            job['id']=canonical_digest(job);rows.append(job)
    return rows

def test_prototype_hash_and_program_verified_then_25_matched_jobs():
    rows=prototype_rows();proto=prototypes(rows,Bank(),{'model_identity':'model'})
    jobs=expand([prompt(1),prompt(2)],proto,'input','corpus')
    assert len(jobs)==50 and len({j['id'] for j in jobs})==50
    for start in (0,25):
        baseline=jobs[start];assert baseline['arm']=='baseline' and not baseline['pushes']
        for j in jobs[start+1:start+25]:
            assert j['shared_baseline_job_id']==baseline['id'] and j['seed']==baseline['seed']
            assert j['prompt']==baseline['prompt'] and not j['promotion_eligible']
            original=proto[j['recipe_id'],j['arm']]
            assert j['program_sha256']==original['program_sha256'] and j['pushes']==original['pushes']
    rows[0]['seed']=1
    with pytest.raises(ValueError,match='ID differs'):prototypes(rows,Bank(),{'model_identity':'model'})

def test_wrong_program_and_wrong_model_rejected():
    rows=prototype_rows();rows[0]['program_sha256']='wrong';rows[0]['id']=canonical_digest({k:v for k,v in rows[0].items() if k!='id'})
    with pytest.raises(ValueError,match='program differs'):prototypes(rows,Bank(),{'model_identity':'model'})
    with pytest.raises(ValueError,match='model differs'):prototypes(prototype_rows(),Bank(),{'model_identity':'other'})

def test_reuses_priority_queue_without_changing_it():
    from persona_contrastive_backfill import PriorityQueue
    from persona_perturbation_mapping import PriorityQueue as Existing
    assert PriorityQueue is Existing


def test_public_snapshot_validates_control_and_retains_baseline_reference(tmp_path):
    from persona_contrastive_backfill import public_snapshot
    from persona_manager import Queue
    proto=prototypes(prototype_rows(),Bank(),{'model_identity':'model'})
    p=prompt(1);p['source_row_id']='original1';jobs=expand([p],proto,'input','corpus')
    q=Queue(tmp_path/'queue.sqlite');q.enqueue({'id':'fixed'},jobs,limit=25)
    binding={'model_identity':'model','runtime_identity':'runtime','prepared_identity':'prepared'}
    job,n=q.claim('endpoint');request={'messages':[{'role':'user','content':p['prompt']}],'seed':SEED,'max_tokens':2048}
    if job['pushes']:request['steering']=Bank().program(job['pushes'],job['phase'])[0]
    h=canonical_digest(request);pointer={**binding,'sha256':'abc','request_sha256':h}
    rc={k:'h' for k in ('generated_token_ids_sha256','consumed_token_ids_sha256','consumed_positions_sha256')};rc.update(request_sha256=h,consumed_generated_tokens=1,sampled_generated_tokens=2,mean_defined=True)
    response={**binding,'choices':[{'message':{'content':'Answer'},'finish_reason':'stop'}],'activation_artifact':pointer,'provenance_artifact':pointer,'response_capture':rc,'usage':{'completion_tokens':2}}
    q.finish(job,n,{'complete':True,'binding':binding,'request':request,'response':response})
    plan={'binding':binding,'input_sha256':'input','control_semantics':'Research only','dispatch_deadline':1}
    result=public_snapshot(q.path,plan,jobs)
    assert result['terminal_jobs']==1 and result['total_jobs']==25 and not result['final']
    row=result['rows'][0];assert row['answer']=='Answer' and row['capture_validated'] and row['source_id']=='original1'
    assert row['shared_baseline_job_id']==(job['shared_baseline_job_id'] or job['id'])
    with q.connect() as db:
        response['runtime_identity']='wrong';db.execute('UPDATE attempts SET result=? WHERE job_id=?',(json.dumps({'complete':True,'binding':binding,'request':request,'response':response}),job['id']))
    with pytest.raises(ValueError,match='binding differs'):public_snapshot(q.path,plan,jobs)
