"""CPU checks for fixed-model gift scheduling and failure semantics."""
import persona
from contextlib import contextmanager
from datetime import datetime
import threading,unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo
from persona_jobs import normalize_job,run_scheduled_job
from persona_vectors import ModelSpec
from persona_lyra_worker import IDENTITY
from persona_lyra_job import worker_command
import test_persona_jobs as fixtures

class GiftTests(unittest.TestCase):
 def test_fixed_identity_and_sources(self):
  model=ModelSpec('test/qwen','a'*40,'test',2,4,5)
  value={'kind':'lyra_gift','name':'gift','start_at':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()}
  result=normalize_job(value,model);self.assertEqual(result['model_identity'],IDENTITY);self.assertEqual(result['corpora'],['big_five'])
  with self.assertRaises(ValueError):normalize_job({**value,'corpora':['skippy']},model)
 def test_worker_uses_owned_wrapper_and_fixed_checkpoint_manifest(self):
  fleet={'remote_root':'/worker','gpus':['GPU-'+str(i) for i in range(6)],'compile_workers':[14]*6,'lease_wrapper':'/verified.py','gpu_labels':['v100-'+str(i) for i in range(6)],'cpu_sets':['0-13']*6,'python':'/venv/python'}
  command=worker_command(fleet,{'lease_wrapper_sha256':'a'*64},{'id':'job'},2,360,21600)
  self.assertIn('/worker/persona_lease.py',command);self.assertIn('--setenv=CUDA_VISIBLE_DEVICES=GPU-2',command);self.assertIn('/worker/results/lyra_job_manifest.json',command)
 def test_failed_gift_does_not_fall_through_to_studio(self):
  fixture=fixtures.PersonaJobTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
  value={**fixture.value,'kind':'lyra_gift'};job=fixture.store.create(value)
  @contextmanager
  def fleet(*args,**kwargs):yield {}
  with patch('persona_lyra_job.prepare'),patch('persona_lyra_job.run',side_effect=ValueError('qualification failed')) as run:
   result=run_scheduled_job(job,{},fixture.store,threading.Event(),fleet_callback=fleet)
  self.assertEqual(result['state'],'failed');self.assertIn('qualification failed',result['error']);run.assert_called_once()
