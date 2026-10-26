import hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'persona'))
import persona_neuron_labels as n

class Labels(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'persona_neuron_labels.json'
 def row(self,label='happy',value=.90,direction='positive'):
  c={'value':value,'method':'heldout_precision','calibrated':True}
  r={'layer':2,'neuron':7,'label':label,'direction':direction,'confidence':c}
  proof={**r,'schema':n.EVIDENCE_SCHEMA,'model_identity':'model','strong_related':True,'validation':{'passed':True,'split':'held_out','samples':100,'dataset_sha256':'a'*64},'calibration':{'method':c['method'],'split':'held_out','samples':100,'dataset_sha256':'b'*64}}
  proof['discovery_dataset_sha256']='c'*64;proof['claim']='probe_conditional_precision';proof['probe']='Recorded binary probe';proof['validation'].update(tp=round(value*100),fp=100-round(value*100),outcomes=[{'unit_id':str(i),'correct':i<round(value*100)} for i in range(100)]);f=self.path.parent/(label+'.json');f.write_text(json.dumps(proof));r['evidence']={'artifact':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()};return r
 def read(self,rows):
  self.path.write_text(json.dumps({'schema':n.SCHEMA,'model_identity':'model','labels':rows}));return n.read_registry(self.path,'model')
 def test_absent_and_wrong_model(self):
  self.assertEqual(n.read_registry(self.path,'model')['state'],'pending');self.read([]);self.assertEqual(n.read_registry(self.path,'other')['state'],'invalid')
 def test_threshold_and_false_probabilities(self):
  a=self.row(value=.70);b=self.row('sad');b['confidence']['method']='correlation';r=self.read([a,b]);self.assertFalse(r['labels']);self.assertEqual(len(r['candidates']),2);self.assertNotIn('confidence',r['candidates'][0])
 def test_inverse_top_three_and_dedup(self):
  rows=[self.row(label,value=.90+i*.02,direction='inverse') for i,label in enumerate(['happy','sad','green','dog'])];r=self.read(rows+[rows[-1]]);self.assertEqual(len(r['labels']),3);self.assertTrue(all(a['direction']=='inverse' for a in r['labels']))
 def test_tamper_binding_and_path(self):
  a=self.row();(self.path.parent/'happy.json').write_text('{}');self.assertFalse(self.read([a])['labels']);a=self.row();a['neuron']=8;self.assertFalse(self.read([a])['labels']);a['evidence']['artifact']='../escape.json';self.assertFalse(self.read([a])['labels'])
 def test_small_sample_and_lower_bound(self):
  for total,tp in [(1,1),(20,16)]:
   row=self.row(value=tp/total);f=self.path.parent/'happy.json';proof=json.loads(f.read_text());proof['validation'].update(tp=tp,fp=total-tp,outcomes=[{'unit_id':str(i),'correct':i<tp} for i in range(total)]);f.write_text(json.dumps(proof));row['evidence']['sha256']=hashlib.sha256(f.read_bytes()).hexdigest();self.assertFalse(self.read([row])['labels'])
 def test_discovery_overlap(self):
  row=self.row();f=self.path.parent/'happy.json';proof=json.loads(f.read_text());proof['discovery_dataset_sha256']='a'*64;f.write_text(json.dumps(proof));row['evidence']['sha256']=hashlib.sha256(f.read_bytes()).hexdigest();self.assertFalse(self.read([row])['labels'])
 def test_malformed(self):
  self.assertEqual(len(self.read([None,{'layer':True}, {'layer':999,'neuron':2}])['rejected']),3)
if __name__=='__main__':unittest.main()
