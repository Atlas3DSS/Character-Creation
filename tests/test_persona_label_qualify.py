import sys,json,sqlite3,time,os
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'activation-search'),str(ROOT/'persona')]
import persona_label_qualify as q

def rows():return [{'id':str(i),'family_id':'family'+str(i),'label':i<24,'split':'heldout','prompt':'No bird.' if i%2 else 'A bird.'} for i in range(48)]
def test_discovery_selects_one_and_inverse():
 y=np.array([True]*12+[False]*12);x=np.zeros((24,4),np.float32);x[:,0]=np.arange(24)%3;x[:,1]=np.where(y,1.,10.);x[:,2]=np.where(y,10.,1.)
 c=q.select_candidate(x,y,shape=(2,2));assert (c['layer'],c['neuron'])==(0,1);assert c['direction']=='inverse';assert c['discovery_balanced_accuracy']==1
 with pytest.raises(ValueError):q.select_candidate(x[:23],y[:23],shape=(2,2))
def test_heldout_checks_precision_recall_and_independence():
 r=rows();c={'threshold':.5,'direction':'positive'};v=[1.]*24+[0.]*24;m=q.heldout_metrics(c,v,r);assert m['passed'] and len(m['outcomes'])==24 and m['precision_wilson95_lower']>.70
 v=[1.]*48;assert not q.heldout_metrics(c,v,r)['passed']
 v=[1.]*19+[0.]*29;assert not q.heldout_metrics(c,v,r)['passed']
 r[-1]['family_id']=r[0]['family_id']
 with pytest.raises(ValueError):q.heldout_metrics(c,v,r)
def test_nuisance_freeze_and_confounded_score():
 y=[True]*12+[False]*12;f=q.fit_nuisance([100]*12+[10]*12,y);assert f['threshold']==55 and f['direction']=='positive'
 r=rows();assert q.nuisance_balanced_accuracy(f,[100]*24+[10]*24,r)==1
 assert q.nuisance_values([{'prompt':"No dog isn't here, never."}],[12])['explicit_negation_count']==[3.]
def test_resume_marks_running_uncertain_never_pending(tmp_path):
 p=tmp_path/'test.sqlite';r=[{'id':'1','concept':'joy','split':'discovery'}];q.initialize(p,r,{'hash':'a'})
 with sqlite3.connect(p) as d:d.execute("UPDATE tasks SET state='running'")
 q.initialize(p,r,{'hash':'a'})
 with sqlite3.connect(p) as d:assert d.execute('SELECT state FROM tasks').fetchone()[0]=='uncertain'
 with pytest.raises(ValueError):q.initialize(p,r,{'hash':'b'})
def test_freeze_refuses_incomplete_or_already_started_heldout(tmp_path):
 p=tmp_path/'test.sqlite';q.initialize(p,[{'id':'h','concept':'joy','split':'heldout'}],{})
 with sqlite3.connect(p) as d:d.execute("UPDATE tasks SET state='complete'")
 with pytest.raises(ValueError,match='before freeze'):q.freeze_concept(p,'joy',tmp_path,{}, {})
def test_pause_requires_fresh_drained_acknowledgement(tmp_path):
 (tmp_path/'overnight.pause').touch();now=time.time()+.1;s=tmp_path/'overnight_status.json';s.write_text(json.dumps({'updated':now,'paused':True,'active':[]}));receipt=q.pause_receipt(tmp_path,now=now)
 with pytest.raises(ValueError):q.pause_receipt(tmp_path,receipt,now=now+31)
 s.write_text(json.dumps({'updated':now,'paused':True,'active':['worker']}))
 with pytest.raises(ValueError):q.pause_receipt(tmp_path,now=now)
def test_lease_expiry_and_wrong_invocation_refuse():
 e={'gpu_uuid':'gpu','unit':'unit','invocation_id':'inv','lease_expires':1000};r={'id':'lease','project':'character-creation','status':'running','assigned_uuid':'gpu','unit_name':'unit','unit_invocation_id':'inv'};assert q.lease_guard(e,{'jobs':[r]},100)=='lease'
 with pytest.raises(ValueError):q.lease_guard(e,{'jobs':[r]},900)
 r['unit_invocation_id']='other'
 with pytest.raises(ValueError):q.lease_guard(e,{'jobs':[r]},100)
def test_no_constant_discovery_feature():
 with pytest.raises(ValueError):q.select_candidate(np.ones((24,4)),[True]*12+[False]*12,shape=(2,2))

@pytest.mark.parametrize('confounded',[False,True])
def test_publication_reader_contract_and_nuisance_rejection(tmp_path,monkeypatch,confounded):
 import persona_neuron_labels as reader
 monkeypatch.setattr(q,'SHAPE',(2,2));original=q.select_candidate
 monkeypatch.setattr(q,'select_candidate',lambda x,y:original(x,y,shape=(2,2)))
 allrows=[]
 for concept in q.LABELS:
  for split,n in [('discovery',12),('heldout',24)]:
   for label in (False,True):
    for i in range(n):
     identity=f'{concept}-{split}-{label}-{i}';allrows.append({'id':identity,'concept':concept,'family_id':identity,'split':split,'label':label,'prompt':'Plain source.'})
 database=tmp_path/'qualification.sqlite';q.initialize(database,allrows,{'binding':'fixed'})
 def capture(task,root,endpoint):
  row=json.loads(task['row']);return np.array([10. if row['label'] else 1.,0,0,0],dtype=np.float32),{'probe_id':row['id'],'validation':{'tokens':20 if confounded and row['label'] else 10}}
 monkeypatch.setattr(q,'read_capture',capture)
 with sqlite3.connect(database) as d:
  d.execute("UPDATE tasks SET endpoint='worker'");d.execute("UPDATE tasks SET state='complete' WHERE split='discovery'")
 endpoints={'worker':{'model_identity':'m'}}
 for concept in q.LABELS:q.freeze_concept(database,concept,tmp_path,endpoints,{'binding':'fixed'})
 with sqlite3.connect(database) as d:d.execute("UPDATE tasks SET state='complete' WHERE split='heldout'")
 dataset={'definitions':{concept:concept for concept in q.LABELS},'limitations':['Synthetic probes']}
 report=q.publish(database,dataset,tmp_path,endpoints,{'binding':'fixed'});assert report['qualified']==(0 if confounded else 6)
 loaded=reader.read_registry(tmp_path/'persona_neuron_labels.json','m');assert len(loaded['labels'])==(0 if confounded else 3)
 if confounded:assert len(loaded['candidates'])==6
 assert q.publish(database,dataset,tmp_path,endpoints,{'binding':'fixed'})==report
