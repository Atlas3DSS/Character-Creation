import persona
import json,sqlite3,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from persona_overnight import OvernightMap

def make(tmp_path):
 (tmp_path/'results').mkdir();app=SimpleNamespace(root=tmp_path,connect=lambda:sqlite3.connect(tmp_path/'db.sqlite'));x=OvernightMap(app);x.path.write_text(json.dumps({'active':True}));return x

def test_pause_requires_fresh_drained_remote_status(tmp_path):
 x=make(tmp_path);x.remote=Mock(return_value={'paused':True,'pause_requested_at':10,'updated':11,'active':['one'],'state':'running'})
 assert not x.status(force=True)['interactive_allowed']
 x.remote.return_value['active']=[];x.remote.return_value['updated']=9
 assert not x.status(force=True)['interactive_allowed']
 x.remote.return_value['updated']=11;assert x.status(force=True)['interactive_allowed']
 x.remote.side_effect=OSError('offline');assert not x.status(force=True)['interactive_allowed']

def test_resume_refuses_active_interactive_jobs(tmp_path):
 x=make(tmp_path)
 with x.app.connect() as c:c.execute('CREATE TABLE cells(state TEXT)');c.execute("INSERT INTO cells VALUES('running')")
 x.remote=Mock()
 with pytest.raises(ValueError,match='current interactive'):x.action({'action':'resume'})
 x.remote.assert_not_called()

def test_auxiliary_session_invalidates_cache_and_blocks_until_drained(tmp_path):
 x=make(tmp_path);x.remote=Mock(return_value={'paused':True,'pause_requested_at':10,'updated':11,'active':[],'state':'running'})
 assert x.status()['interactive_allowed']
 x.path.write_text(json.dumps({'active':True,'auxiliary':'label_qualification'}))
 assert not x.status()['interactive_allowed']
 with pytest.raises(ValueError,match='Label qualification'):x.action({'action':'resume'})
 x.remote.return_value['label_qualification']={'state':'complete','active':[]}
 assert x.status(force=True)['interactive_allowed']
