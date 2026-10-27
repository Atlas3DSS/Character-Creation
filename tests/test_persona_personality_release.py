"""Offline release fail-closed tests; no inference or network."""
import persona
from copy import deepcopy
import json
from pathlib import Path
import pytest
from persona_personality_probes import load_suite
from persona_personality_release import validate_design,LocalStore,build,review_projection
from persona_vectors import canonical_digest
from persona_propagation import file_sha

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def inputs():
    suite=load_suite(ROOT/'data/persona_personality_probes.json')
    plan=json.loads((ROOT/'results/persona_personality_sweep_plan.json').read_text())
    run=dict(schema='character-creation.personality-sweep-run.v1',state='complete',plan_sha256=plan['input_sha256'],binding=plan['binding'],source_hashes=plan['source_hashes'],launch_sha256=plan['launch_sha256'],automatic_default_promotion=False,
        results=[dict(job_id=j['id'],state='complete') for j in plan['jobs']])
    return plan,run,suite

def rebind(plan,run):
    plan['input_sha256']=canonical_digest({k:v for k,v in plan.items() if k!='input_sha256'});run['plan_sha256']=plan['input_sha256']

def test_exact_288_balanced_authored_design(inputs):
    p,r,s=inputs
    assert len(validate_design(p,r,s))==288
    assert len({j['family_id'] for j in p['jobs']})==12

@pytest.mark.parametrize('mutation',['incomplete','missing','duplicate','wrong_binding','promoted'])
def test_reject_result_coverage_and_binding(inputs,mutation):
    p,r,s=inputs
    if mutation=='incomplete':r['state']='running'
    elif mutation=='missing':r['results'].pop()
    elif mutation=='duplicate':r['results'][1]=deepcopy(r['results'][0])
    elif mutation=='wrong_binding':r['binding']=dict(r['binding'],runtime_identity='0'*64)
    else:r['automatic_default_promotion']=True
    with pytest.raises(ValueError):validate_design(p,r,s)

@pytest.mark.parametrize('mutation',['private_prompt','heldout','dose','seed','duplicate_condition','capture_id','source'])
def test_rehashed_altered_plan_not_accepted(inputs,mutation):
    p,r,s=inputs;j=p['jobs'][1]
    if mutation=='private_prompt':j['prompt']='Unapproved private prompt'
    elif mutation=='heldout':j['split']='heldout'
    elif mutation=='dose':j['recipe']['pushes'][0]['dose']=.2
    elif mutation=='seed':j['seed']=7035
    elif mutation=='duplicate_condition':p['jobs'][1]=deepcopy(p['jobs'][0])
    elif mutation=='capture_id':j['capture_response']['id']=p['jobs'][0]['capture_response']['id']
    else:j['source_sha256']='0'*64
    rebind(p,r)
    with pytest.raises(ValueError):validate_design(p,r,s)

def test_local_store_never_falls_back_to_remote(tmp_path,monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('network attempted'))
    store=LocalStore(tmp_path,'unused-local-only','/approved')
    pointer={'path':'/approved/probe-'+('a'*32)+'.npz','sha256':'0'*64}
    with pytest.raises(FileNotFoundError):store.get(pointer)
    path=tmp_path/Path(pointer['path']).name;path.write_bytes(b'validated fixture')
    with pytest.raises(ValueError):store.get(pointer)
    pointer['sha256']=file_sha(path);assert store.get(pointer)==path
    pointer['path']='/unapproved/'+path.name
    with pytest.raises(ValueError):store.get(pointer)

def test_incomplete_review_is_not_published(tmp_path):
    (tmp_path/'persona_personality_review.json').write_text(json.dumps({'state':'running','groups':[]}))
    assert review_projection(tmp_path,{},[],{}, {}) is None

def test_incomplete_run_refuses_before_writes(tmp_path,inputs):
    p,r,s=inputs;r['state']='running'
    (tmp_path/'results').mkdir();(tmp_path/'data').mkdir()
    (tmp_path/'data/persona_personality_probes.json').write_bytes((ROOT/'data/persona_personality_probes.json').read_bytes())
    for name,value in [('persona_personality_sweep_plan.json',p),('persona_personality_sweep.json',r)]:
        (tmp_path/'results'/name).write_text(json.dumps(value))
    with pytest.raises(ValueError):build(tmp_path)
    assert not (tmp_path/'results/persona_personality_public.json').exists()
    assert not (tmp_path/'results/persona_personality_capture_bundle.npz').exists()

def test_finite_wait_terminal_and_timeout(tmp_path):
    from persona_personality_release import wait_for_review
    now=[100.]
    def clock():return now[0]
    def sleep(n):now[0]+=n
    with pytest.raises(TimeoutError):wait_for_review(tmp_path,101,clock=clock,sleep=sleep)
    (tmp_path/'persona_personality_review.json').write_text('{"state":"incomplete_no_retry"}')
    assert wait_for_review(tmp_path,200,clock=clock,sleep=sleep)=='incomplete_no_retry'
    with pytest.raises(ValueError):wait_for_review(tmp_path,100000,clock=clock,sleep=sleep)

@pytest.fixture
def release_files(tmp_path):
    from persona_personality_release import PUBLIC_FILES
    (tmp_path/PUBLIC_FILES[1]).write_bytes(b'test compressed bundle bytes')
    spec={'filename':PUBLIC_FILES[1],'sha256':file_sha(tmp_path/PUBLIC_FILES[1]),'bytes':(tmp_path/PUBLIC_FILES[1]).stat().st_size}
    (tmp_path/PUBLIC_FILES[0]).write_text(json.dumps({'schema':'character-creation.personality-public.v1','counts':{'answers':288},'automatic_default_promotion':False,'activation_bundle':spec}))
    return tmp_path

def test_publish_exact_includes_pinned_verification_and_durable_intent(release_files,monkeypatch):
    from persona_personality_release import publish,PUBLIC_FILES
    from types import SimpleNamespace
    import time
    monkeypatch.setattr('persona_personality_release.shutil.which',lambda name:'/bin/hf')
    commit='a'*40;seen=[]
    def run(command,**kwargs):
        intent=json.loads((release_files/'persona_personality_publication.json').read_text())
        assert intent['verified'] is False and len(intent['files'])==2
        includes=[command[i+1] for i,v in enumerate(command) if v=='--include']
        assert tuple(includes)==PUBLIC_FILES and '--delete' not in command and '--every' not in command
        return SimpleNamespace(returncode=0,stdout=json.dumps({'url':f'https://huggingface.co/datasets/Atlas3D/character-creation-big-five/commit/{commit}'}))
    def verify(name,revision,deadline):
        assert revision==commit;seen.append(name)
        return {'sha256':file_sha(release_files/name),'bytes':(release_files/name).stat().st_size}
    result=publish(release_files,time.time()+30,run=run,verify=verify,inventory=lambda d:{'sha':'b'*40,'siblings':[]})
    assert result['verified'] is True and seen==list(PUBLIC_FILES)
    with pytest.raises(FileExistsError):publish(release_files,time.time()+30,run=lambda *a,**k:pytest.fail('retry'))

@pytest.mark.parametrize('failure',['timeout','mismatch','no_commit'])
def test_uncertain_commit_never_automatically_retried(release_files,monkeypatch,failure):
    from persona_personality_release import publish
    from types import SimpleNamespace
    import time,subprocess
    monkeypatch.setattr('persona_personality_release.shutil.which',lambda name:'/bin/hf')
    def run(*args,**kwargs):
        if failure=='timeout':raise subprocess.TimeoutExpired('hf',1)
        return SimpleNamespace(returncode=0,stdout='no commit' if failure=='no_commit' else 'https://huggingface.co/datasets/Atlas3D/character-creation-big-five/commit/'+('a'*40))
    with pytest.raises(Exception):publish(release_files,time.time()+30,run=run,verify=lambda *a:{'sha256':'bad','bytes':0},inventory=lambda d:{'sha':'b'*40,'siblings':[]})
    receipt=json.loads((release_files/'persona_personality_publication.json').read_text())
    assert receipt['verified'] is False and receipt['state']=='uncertain_no_retry'
    with pytest.raises(FileExistsError):publish(release_files,time.time()+30,run=lambda *a,**k:pytest.fail('retry'))

def test_existing_remote_files_never_overwritten(release_files,monkeypatch):
    from persona_personality_release import publish,PUBLIC_FILES
    import time
    monkeypatch.setattr('persona_personality_release.shutil.which',lambda name:'/bin/hf')
    with pytest.raises(FileExistsError):publish(release_files,time.time()+30,run=lambda *a,**k:pytest.fail('upload'),inventory=lambda d:{'sha':'b'*40,'siblings':[{'rfilename':PUBLIC_FILES[0]}]})
    assert not (release_files/'persona_personality_publication.json').exists()
