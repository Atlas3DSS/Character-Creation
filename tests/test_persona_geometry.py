"""Geometry is usable only while its recorded calibration files remain identical."""
import persona
import hashlib,json,os,sqlite3,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from persona_explorer import Explorer

class GeometryTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  (self.root/'data').mkdir();(self.root/'results').mkdir()
  (self.root/'data/persona_model.json').write_text('{}')
  self.chart=self.root/'results/persona_map.json';self.chart.write_text(json.dumps({'model_identity':'m','neurons':[{'layer':0,'neuron':1},{'layer':1,'neuron':2}]}))
  self.source=self.root/'results/layer_00.npz';self.source.write_bytes(b'calibration')
  self.geometry={'model_identity':'m','map_sha256':hashlib.sha256(self.chart.read_bytes()).hexdigest(),'source_sha256':{self.source.name:hashlib.sha256(self.source.read_bytes()).hexdigest()},'positions':{'L0N1':[.1,.2]},'semantics':'PCA','explained_fraction':{'0':.5}}
  self.app=Explorer.__new__(Explorer);self.app.root=self.root;self.app.model=SimpleNamespace(identity='m',layers=2);self.app.endpoints=[]
  db=sqlite3.connect(self.root/'db');db.execute('CREATE TABLE comparisons(id,created,state)');db.close();self.app.connect=lambda:sqlite3.connect(self.root/'db')
 def state(self):
  (self.root/'results/persona_map_geometry.json').write_text(json.dumps(self.geometry));return self.app.state()['map']
 def test_valid_partial_and_cached(self):
  with patch('hashlib.file_digest',wraps=hashlib.file_digest) as digest:
   result=self.state();self.assertTrue(result['geometry']['available']);self.assertEqual(result['neurons'][0]['output_layout'],[.1,.2]);self.assertIsNone(result['neurons'][1]['output_layout']);self.state();self.assertEqual(digest.call_count,1)
 def test_changed_same_size_and_mtime_invalidates(self):
  self.state();before=self.source.stat();self.source.write_bytes(b'CALIBRATION');os.utime(self.source,ns=(before.st_atime_ns,before.st_mtime_ns));result=self.state();self.assertFalse(result['geometry']['available']);self.assertNotIn('output_layout',result['neurons'][0]);self.assertIn('hash differs',result['geometry']['reason'])
 def test_deleted_source(self):
  self.state();self.source.unlink();self.assertFalse(self.state()['geometry']['available'])
 def test_missing_and_wrong_bindings(self):
  for sources in ({},{'../escape.npz':'0'*64},{'layer_00.npz':'0'*64}):
   self.geometry['source_sha256']=sources;self.assertFalse(self.state()['geometry']['available'])
 def test_position_requires_bound_layer(self):
  self.geometry['positions']['L1N2']=[0,0];self.assertFalse(self.state()['geometry']['available'])
 def test_changed_map(self):
  self.chart.write_text(self.chart.read_text()+' ');self.assertFalse(self.state()['geometry']['available'])
if __name__=='__main__':unittest.main()
