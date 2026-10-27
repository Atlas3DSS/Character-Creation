"""Offline authored 288-answer export. No network, inference, upload or source mutation."""
import argparse
import hashlib
import math
import os
import shutil
import subprocess
import time
import urllib.request
import json
from pathlib import Path
import re
import numpy as np
from persona_personality_probes import load_suite,validate_suite
from persona_personality_sweep import ARMS,SEEDS,request_for
from persona_personality_review import validate_inputs,validate_judgments,summarize,request_for as judge_request,MODEL_SHA
from persona_propagation import ArtifactStore,checked,file_sha,atomic
from persona_vectors import canonical_digest,ModelSpec,VectorBank
from persona_response_release import public_only


class LocalStore(ArtifactStore):
    """Reuse exact ArtifactStore name/path/hash rules without its remote fallback."""
    def get(self,pointer):
        source=Path(pointer['path'])
        if not (self.directory/source.name).is_file():raise FileNotFoundError('Required local capture missing: '+source.name)
        return super().get(pointer)


def validate_design(plan,run,suite):
    validate_suite(suite)
    if (plan.get('schema')!='character-creation.personality-sweep.v1' or run.get('schema')!='character-creation.personality-sweep-run.v1'
        or run.get('state')!='complete' or plan.get('input_sha256')!=canonical_digest({k:v for k,v in plan.items() if k!='input_sha256'})
        or run['plan_sha256']!=plan['input_sha256'] or run['binding']!=plan['binding']
        or run['source_hashes']!=plan['source_hashes'] or run['launch_sha256']!=plan['launch_sha256']
        or plan['suite_sha256']!=suite['suite_sha256'] or plan['suite_content_sha256']!=canonical_digest(suite)
        or plan['seeds']!=list(SEEDS) or plan.get('automatic_default_promotion') is not False or run.get('automatic_default_promotion') is not False):
        raise ValueError('Incomplete or unbound frozen study')
    cases={c['id']:c for c in suite['cases']};conditions=set();ids=set();captures=set()
    if len(cases)!=24 or len(plan['jobs'])!=288 or len(run['results'])!=288:raise ValueError('Exact24/288 coverage required')
    artifact=plan['direction_artifact']
    for job in plan['jobs']:
        case=cases[job['case_id']];arm=job['arm'];seed=job['seed'];condition=(case['id'],seed,arm)
        if condition in conditions or seed not in SEEDS or arm not in ARMS or job['id'] in ids:raise ValueError('Duplicate/unknown condition')
        conditions.add(condition);ids.add(job['id'])
        if any(job[k]!=v for k,v in dict(prompt=case['prompt'],source_sha256=case['source_sha256'],family_id=case['pair_id'],axis=case['axis'],pressure=case['pressure'],split='discovery').items()):raise ValueError('Authored source differs')
        pushes=[]
        if arm=='random':pushes=[dict(kind='random',site='residual',layer=32,dose=.1,seed=481516)]
        elif arm!='baseline':pushes=[dict(kind='direction',layer=32,dose=-.1 if arm.endswith('_reverse') else .1,direction=arm[0],artifact=artifact['filename'],sha256=artifact['sha256'])]
        cap=job['capture_response']
        if job['recipe']!={'phase':'all','pushes':pushes} or set(cap)!={'id','layers','max_positions'} or cap['layers']!=[] or cap['max_positions']!=0 or not re.fullmatch('[0-9a-f]{32}',cap['id']) or cap['id'] in captures:raise ValueError('Neural recipe/capture design differs')
        captures.add(cap['id'])
    results={r['job_id']:r for r in run['results']}
    if set(results)!=ids or len(results)!=288 or any(r['state']!='complete' for r in results.values()):raise ValueError('Result coverage differs')
    return results


def review_projection(folder,packet,key,plan,suite):
    path=folder/'persona_personality_review.json'
    if not path.exists():return None
    value=json.loads(path.read_text())
    if value.get('state')!='complete':return None
    if value.get('judge',{}).get('model_sha256')!=MODEL_SHA or value.get('automatic_default_promotion') is not False:raise ValueError('Judge model/promotion differs')
    if value.get('source_sha256')!=file_sha(Path(__file__).with_name('persona_personality_review.py')):raise ValueError('Review implementation changed')
    if any(value.get(k)!=v for k,v in dict(packet_sha256=canonical_digest(packet),key_sha256=canonical_digest(key),suite_sha256=suite['suite_sha256'],plan_sha256=plan['input_sha256']).items()):raise ValueError('Review source binding differs')
    events=[json.loads(line) for line in (folder/'persona_personality_review.attempts.jsonl').read_text().splitlines()]
    submitted={};received={}
    for event in events:
        table=submitted if event['state']=='submitting' else received if event['state']=='received' else None
        if table is None or event['group'] in table:raise ValueError('Duplicate/unrecognized review attempt')
        table[event['group']]=event
    groups={g['group']:g for g in packet['groups']};seen=set();judgments=[]
    for row in value['groups']:
        group=groups[row['group']];gid=group['group']
        if gid in seen:raise ValueError('Repeated review group')
        seen.add(gid);request=judge_request(group);response=received[gid]['response']
        if submitted[gid]['request']!=request or submitted[gid]['request_sha256']!=canonical_digest(request) or row['request_sha256']!=canonical_digest(request) or row['response_sha256']!=canonical_digest(response):raise ValueError('Review request/response bytes differ')
        choice=response['choices'][0]
        if response['model']!='persona-manager' or choice['finish_reason']!='stop':raise ValueError('Judge completion differs')
        rows=validate_judgments(group,json.loads(choice['message']['content']))
        if rows!=row['judgments']:raise ValueError('Stored judgments differ')
        judgments.extend(rows)
    if seen!=set(groups) or set(submitted)!=seen or set(received)!=seen:raise ValueError('Incomplete review receipts')
    summary=summarize(packet,key,judgments)
    if summary!=value['summary']:raise ValueError('Review summary differs')
    return public_only(dict(judge=value['judge'],judgments=judgments,summary=summary,source_sha256=file_sha(path),packet_sha256=canonical_digest(packet),key_sha256=canonical_digest(key),
        limitation='Local Qwen-family model judge, not human review or independent scientific truth. All discovery; no automatic promotion.'))


def build(root):
    root=Path(root);folder=root/'results'
    def read(name):return json.loads((folder/name).read_text())
    plan=read('persona_personality_sweep_plan.json');run=read('persona_personality_sweep.json');suite=load_suite(root/'data/persona_personality_probes.json')
    results=validate_design(plan,run,suite)
    packet=read('persona_personality_blind.json');key=read('persona_personality_blind_key.json');validate_inputs(packet,key,suite,plan,run)
    jobs={j['id']:j for j in plan['jobs']}
    for row in key:
        if any(row[k]!=jobs[row['job_id']][k] for k in ('case_id','family_id','axis','pressure','seed','arm','split')):raise ValueError('Public blind-key projection differs')
    codes={r['job_id']:r['code'] for r in key}
    model=ModelSpec.load(root/'data/persona_model.json');bank=VectorBank(folder,model)
    if model.identity!=plan['binding']['model_identity'] or file_sha(folder/plan['direction_artifact']['filename'])!=plan['direction_artifact']['sha256']:raise ValueError('Model/direction artifact differs')
    store=LocalStore(folder,'unused-local-only','/mnt/storage16tb/Character Creation/results')
    output=folder/'persona_personality_public.json';bundle=folder/'persona_personality_capture_bundle.npz'
    if output.exists() or bundle.exists():raise FileExistsError('Prior release retained; no overwrite')
    review=review_projection(folder,packet,key,plan,suite)
    answers=[];counts=[];means=[]
    for index,job in enumerate(plan['jobs']):
        result=results[job['id']];response=result['response'];request,proof=request_for(job,response['model'],bank)
        if canonical_digest(request)!=result['request_sha256'] or proof['program_sha256']!=result['program_sha256']:raise ValueError('Exact request/program differs')
        arrays,_=checked(response,request,plan['binding'],store)
        mean=arrays['response_residual_mean'];count=int(arrays['response_residual_count'])
        if mean.dtype!=np.float64 or count<=0:raise ValueError('Defined FP64 response mean required')
        means.append(mean);counts.append(count)
        answers.append(dict(row=index,code=codes[job['id']],**{k:job[k] for k in ('id','case_id','family_id','axis','pressure','split','prompt','source_sha256','seed','arm','recipe')},
            answer=response['choices'][0]['message']['content'],finish_reason=response['choices'][0]['finish_reason'],request_sha256=result['request_sha256'],program_sha256=result['program_sha256'],
            activation_sha256=response['activation_artifact']['sha256'],provenance_sha256=response['provenance_artifact']['sha256'],consumed_generated_tokens=count,
            generated_token_ids_sha256=response['response_capture']['generated_token_ids_sha256']))
        del arrays
    public=public_only(dict(schema='character-creation.personality-public.v1',binding=plan['binding'],suite=suite,plan_sha256=plan['input_sha256'],source_hashes=plan['source_hashes'],
        source_reports_sha256={n:file_sha(folder/n) for n in ('persona_personality_sweep_plan.json','persona_personality_sweep.json','persona_personality_blind.json','persona_personality_blind_key.json')},
        direction_artifact=plan['direction_artifact'],counts=dict(prompts=24,families=12,answers=288,seeds=2,arms=6),answers=answers,review=review,review_status=review_status(folder,review),
        settings=dict(temperature=.7,top_p=.95,top_k=20,max_tokens=2048,thinking=False,phase='all',system='',kv_control=None),
        semantics=['All 24 prompts are discovery: authored/exposed before this sweep; no fresh heldout claim.','Same prompt across arms; only activation recipe changes. Baseline is unsteered.',
        'Response means include consumed generated tokens only, excluding prompt/template and the final sampled unconsumed token.','Factual correctness and interpersonal manner are separate outcomes; no automatic recipe promotion.'],automatic_default_promotion=False))
    matrix=np.stack(means);del means
    metadata=dict(schema='character-creation.personality-capture-bundle.v1',binding=plan['binding'],plan_sha256=plan['input_sha256'],rows=[{k:r[k] for k in ('row','id','code','activation_sha256','provenance_sha256','request_sha256')} for r in answers])
    with bundle.open('xb') as stream:np.savez_compressed(stream,response_mean=matrix,response_count=np.asarray(counts,np.int64),metadata=np.asarray(json.dumps(metadata,sort_keys=True,separators=(',',':'))))
    if bundle.stat().st_size>1024**3:raise ValueError('Bundle exceeds 1GiB bound; retained for inspection')
    public['activation_bundle']=dict(filename=bundle.name,sha256=file_sha(bundle),bytes=bundle.stat().st_size,response_mean=dict(shape=list(matrix.shape),dtype='float64'),response_count=dict(shape=[288],dtype='int64'),raw_native_npz_included=False)
    atomic(output,public);return public


REPO='Atlas3D/character-creation-big-five'
PUBLIC_FILES=('persona_personality_public.json','persona_personality_capture_bundle.npz')


def review_status(folder,review):
    if review is not None:return {'state':'complete','reason':'Complete source-bound local model review; not independent human validation.'}
    path=folder/'persona_personality_review.json'
    state=json.loads(path.read_text()).get('state') if path.exists() else 'missing'
    return {'state':'unavailable','reason':'Model review terminated before complete coverage; partial judgments excluded.' if state=='incomplete_no_retry' else 'Complete validated model review was not available at export.'}


def remaining(deadline,clock=time.time):
    value=deadline-clock()
    if not math.isfinite(value) or value<=0:raise TimeoutError('Finite publication deadline reached')
    return value


def wait_for_review(folder,deadline,*,clock=time.time,sleep=time.sleep):
    if not math.isfinite(deadline) or not 0<deadline-clock()<=16*3600:raise ValueError('Explicit future deadline within16h required')
    path=Path(folder)/'persona_personality_review.json'
    while True:
        budget=remaining(deadline,clock)
        if path.exists():
            state=json.loads(path.read_text()).get('state')
            if state in ('complete','incomplete_no_retry'):return state
            if state not in ('waiting_for_capture','running'):raise ValueError('Unknown model review state')
        sleep(min(30,budget))


def remote_digest(name,commit,deadline):
    """Hash actual public bytes at an immutable commit, without local copies."""
    if name not in PUBLIC_FILES or not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact public file/commit required')
    url=f'https://huggingface.co/datasets/{REPO}/resolve/{commit}/{name}'
    digest=hashlib.sha256();size=0
    with urllib.request.urlopen(url,timeout=min(30,remaining(deadline))) as response:
        while True:
            remaining(deadline)
            data=response.read(1024*1024)
            if not data:break
            size+=len(data)
            if size>1024**3:raise ValueError('Remote file exceeds bound')
            digest.update(data)
    return {'sha256':digest.hexdigest(),'bytes':size}


def public_head(deadline):
    with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{REPO}/revision/main',timeout=min(30,remaining(deadline))) as response:return json.load(response)


def publish(folder,deadline,*,run=subprocess.run,verify=remote_digest,inventory=public_head):
    folder=Path(folder);remaining(deadline)
    receipt=folder/'persona_personality_publication.json'
    if receipt.exists():raise FileExistsError('Prior publication intent exists; no automatic retry, reconcile manually')
    files={name:{'sha256':file_sha(folder/name),'bytes':(folder/name).stat().st_size} for name in PUBLIC_FILES}
    public=public_only(json.loads((folder/PUBLIC_FILES[0]).read_text()))
    if (public.get('schema')!='character-creation.personality-public.v1' or public.get('counts',{}).get('answers')!=288
        or public.get('automatic_default_promotion') is not False or public['activation_bundle']['filename']!=PUBLIC_FILES[1]
        or any(public['activation_bundle'][k]!=files[PUBLIC_FILES[1]][k] for k in ('sha256','bytes'))):raise ValueError('Release bundle binding differs')
    executable=shutil.which('hf')
    if not executable:raise FileNotFoundError('hf CLI is required')
    head=inventory(deadline)
    if not re.fullmatch('[0-9a-f]{40}',head.get('sha','')) or 'siblings' not in head:raise ValueError('Public repository inventory incomplete')
    if set(PUBLIC_FILES)&{item['rfilename'] for item in head['siblings']}:raise FileExistsError('Public release filenames already exist; no overwrite')
    command=[executable,'upload',REPO,str(folder),'.','--repo-type','dataset','--revision','main',
        '--include',PUBLIC_FILES[0],'--include',PUBLIC_FILES[1],'--commit-message','Publish verified authored personality sweep and response captures','--format','json']
    intent=dict(schema='character-creation.personality-publication.v1',repo=REPO,state='upload_intent',verified=False,files=files,observed_parent_commit=head['sha'],created=time.time(),deadline=deadline)
    # Exclusive durable intent precedes any mutating Hub call. Its presence prevents replay after a crash.
    with receipt.open('x') as stream:
        json.dump(intent,stream,sort_keys=True,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    directory=os.open(folder,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)
    try:
        result=run(command,capture_output=True,text=True,timeout=remaining(deadline),check=False)
        if result.returncode:raise RuntimeError('HF upload returned failure; commit outcome may be uncertain')
        commits=set(re.findall(r'https://huggingface\.co/datasets/Atlas3D/character-creation-big-five/commit/([0-9a-f]{40})',result.stdout))
        if len(commits)!=1:raise ValueError('HF CLI did not return one exact repository commit URL')
        commit=commits.pop();intent.update(state='verifying',commit=commit);atomic(receipt,intent)
        for name,spec in files.items():
            if {'sha256':file_sha(folder/name),'bytes':(folder/name).stat().st_size}!=spec:raise ValueError('Local publication file changed')
            if verify(name,commit,deadline)!=spec:raise ValueError('Committed remote bytes differ')
        intent.update(state='complete',verified=True,finished=time.time());atomic(receipt,intent)
    except BaseException as error:
        intent.update(state='uncertain_no_retry',verified=False,error_type=type(error).__name__,reason='Upload or pinned-commit verification did not complete; manual reconciliation required.')
        atomic(receipt,intent);raise
    return intent


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--wait-for-review',action='store_true');parser.add_argument('--deadline',type=float);parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    if (args.wait_for_review or args.publish) and (args.deadline is None or not math.isfinite(args.deadline) or not 0<args.deadline-time.time()<=16*3600):parser.error('A future --deadline epoch within16h is required')
    folder=args.root/'results'
    if args.wait_for_review:wait_for_review(folder,args.deadline)
    if args.deadline is not None:remaining(args.deadline)
    result=build(args.root)
    receipt=publish(folder,args.deadline) if args.publish else None
    print(json.dumps({'answers':len(result['answers']),'bundle':result['activation_bundle'],'publication':receipt}))

if __name__=='__main__':main()
