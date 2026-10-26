"""Portable configuration checks; all transport is mocked or --help only."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import persona
from persona_overnight import OvernightMap, remote_config
from persona_label_handoff import remote_bytes, api_base
from persona_explorer import nlp_interpreter

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('isolated_launcher',ROOT/'activation-search/launch.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)


def test_remote_defaults_and_overrides():
    defaults=remote_config()
    assert defaults['ssh_host']==defaults['hostname']=='v100'
    assert defaults['remote_root'].endswith('/Character Creation/activation-search')
    value={'overnight_remote':{'ssh_host':'reader@lab-alias','hostname':'lab-node','remote_root':'/srv/My Research/activation-search/'}}
    assert remote_config(value)=={**value['overnight_remote'],'remote_root':'/srv/My Research/activation-search'}


@pytest.mark.parametrize('value',[[],{'overnight_remote':[]},{'overnight_remote':{'extra':1}},
    {'overnight_remote':{'ssh_host':'-oProxyCommand=bad'}},
    {'overnight_remote':{'ssh_host':'host name'}},{'overnight_remote':{'hostname':'node\nother'}},
    {'overnight_remote':{'remote_root':'relative'}},{'overnight_remote':{'remote_root':'/srv/../private'}},
    {'overnight_remote':{'remote_root':'/srv\x00bad'}}])
def test_remote_invalid_config(value):
    with pytest.raises(ValueError):remote_config(value)


def test_bridge_and_handoff_use_same_host_binding(tmp_path):
    config={'overnight_remote':{'ssh_host':'compute-alias','hostname':'actual-compute','remote_root':'/srv/space dir'}}
    app=SimpleNamespace(root=tmp_path,config=config)
    with patch('persona_overnight.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='{"state":"running"}')) as run:
        assert OvernightMap(app).remote()['state']=='running'
        argv=run.call_args.args[0]
        assert 'compute-alias' in argv
        script=shlex.split(argv[-1])[-1]
        assert "socket.gethostname()!='actual-compute'" in script
        assert "Path('/srv/space dir')" in script
    with patch('persona_label_handoff.subprocess.check_output',return_value=b'evidence') as fetch:
        assert remote_bytes('persona_label_evidence_abc.json',config)==b'evidence'
        argv=fetch.call_args.args[0];script=shlex.split(argv[-1])[-1]
        assert 'compute-alias' in argv
        assert "socket.gethostname()!='actual-compute'" in script
        assert '/srv/space dir/persona_label_evidence_abc.json' in script
        compile(script,'remote-test','exec')


@pytest.mark.parametrize('name',['','..','../x','/absolute','file\nname'])
def test_handoff_rejects_nonflat_names(name):
    with pytest.raises(ValueError):remote_bytes(name)


@pytest.mark.parametrize('value',['file:///tmp/x','http://user:pass@host','https://host/path','https://host?x=1','https://host/#x','http://host:bad','http://host\n'])
def test_api_origin_validation(value):
    with pytest.raises(ValueError):api_base(value)


def test_nlp_interpreter_uses_current_or_configured_python(tmp_path):
    assert nlp_interpreter({})==str(Path(sys.executable))
    executable=tmp_path/'python with spaces';executable.write_text('');executable.chmod(0o700)
    assert nlp_interpreter({'nlp_python':str(executable)})==str(executable)
    for invalid in ('python3','',None,'/missing/python',str(tmp_path),'bad\x00path'):
        with pytest.raises(ValueError):nlp_interpreter({'nlp_python':invalid})
    executable.chmod(0o600)
    with pytest.raises(ValueError):nlp_interpreter({'nlp_python':str(executable)})


def test_launcher_paths_legacy_and_custom():
    old={'remote_root':'/mnt/storage16tb/Character Creation/activation-search'}
    assert launcher.runtime_paths(old)=={'heliopause':'/mnt/storage16tb/Character Creation/flare_heliopause.so','model':'/mnt/storage16tb/Character Creation/data/persona_model.json','calibration_dir':'/mnt/storage16tb/Character Creation/results'}
    custom={'remote_root':'/srv/run','remote_project_root':'/srv/project','heliopause_path':'/opt/lib/heliopause.so','model_path':'/srv/model.json','calibration_dir':'/srv/calibration'}
    assert launcher.runtime_paths(custom)=={'heliopause':'/opt/lib/heliopause.so','model':'/srv/model.json','calibration_dir':'/srv/calibration'}
    with pytest.raises(ValueError):launcher.runtime_paths({'remote_root':'/srv/run','model_path':'../model.json'})


def test_standalone_help_without_pythonpath():
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    for filename in ('persona_label_qualify.py','launch.py'):
        result=subprocess.run([sys.executable,str(ROOT/'activation-search'/filename),'--help'],cwd='/tmp',env=env,capture_output=True,text=True,timeout=15)
        assert result.returncode==0,result.stderr
        assert 'usage:' in result.stdout
