"""Offline baseline scheduling, durable coverage and artifact guards; no GPU calls."""
import contextlib
from datetime import datetime
import io
import os
import signal
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'persona'))
import persona_baseline_capture as capture
import persona_baseline_schedule as schedule


class BaselineTests(unittest.TestCase):
    def test_daily_window_and_session_end(self):
        def allowed(stamp, timeout=60, end=None):
            now = datetime.fromisoformat(stamp).timestamp()
            return capture.in_daily_window(now, timeout, end or now + 86400)
        self.assertTrue(allowed('2026-10-26T03:00:00-07:00'))
        self.assertTrue(allowed('2026-10-25T20:00:00-07:00'))
        self.assertFalse(allowed('2026-10-25T19:59:00-07:00'))
        self.assertFalse(allowed('2026-10-26T08:00:00-07:00'))
        self.assertFalse(allowed('2026-10-26T07:59:30-07:00'))
        now = datetime.fromisoformat('2026-10-25T22:00:00-07:00').timestamp()
        self.assertFalse(capture.in_daily_window(now, 60, now+30))
        # Fall-back night: 20:00 PDT through 08:00 PST follows wall clock.
        self.assertTrue(allowed('2026-11-01T07:00:00-08:00'))

    def test_opportunistic_boundaries_and_priority(self):
        now = datetime.fromisoformat('2026-10-25T12:00:00-07:00')
        self.assertIsNone(schedule.session_end(now))
        self.assertEqual(schedule.session_end(now, True).hour, 14)
        self.assertFalse(capture.request_window(now.timestamp(), 60, now.timestamp()+7200))
        self.assertTrue(capture.request_window(now.timestamp(), 60, now.timestamp()+7200, True))
        self.assertFalse(capture.request_window(now.timestamp(), 60, now.timestamp()+60, True))
        for stamp, hour, day in [('2026-10-25T08:00:00-07:00',10,25),
                                 ('2026-10-25T19:50:00-07:00',20,25),
                                 ('2026-10-25T20:00:00-07:00',8,26),
                                 ('2026-10-31T20:00:00-07:00',8,1)]:
            end = schedule.session_end(datetime.fromisoformat(stamp), True)
            self.assertEqual((end.hour,end.day),(hour,day))
        for job in [dict(lane='interactive'), dict(project='atlas-print'),
                    dict(project='dsv41'),dict(label='chem-27b-tim'),dict(label='orca-27b-mike')]:
            self.assertTrue(schedule.priority_waiting([dict(status='queued',**job)]))
            self.assertIsNone(schedule.priority_waiting([dict(status='running',**job)]))
            self.assertIsNone(schedule.priority_waiting([dict(status='completed',**job)]))
        self.assertEqual(schedule.priority_waiting([dict(status='queued',project='dsv41'),
                         dict(status='queued',lane='interactive')]),'interactive')
        self.assertIsNone(schedule.priority_waiting([dict(status='queued',project='baseline',lane='batch')]))

    def test_requires_entire_fresh_idle_fleet_before_admission(self):
        fleet={'gpus':[str(i) for i in range(6)]}
        cards=[dict(uuid=gpu,fresh=True,active_job_ids=[],starting_job_ids=[],exclusive_job_ids=[])
               for gpu in [*fleet['gpus'],schedule.JOHN]]
        snapshot={'fleet':{'cards':cards},'jobs':[]}
        self.assertTrue(schedule.required_fleet_idle(snapshot,fleet))
        for card in cards:
            for field in ('active_job_ids','starting_job_ids','exclusive_job_ids'):
                card[field]=['other-owner']
                self.assertFalse(schedule.required_fleet_idle(snapshot,fleet))
                card[field]=[]
            card['fresh']=False
            self.assertFalse(schedule.required_fleet_idle(snapshot,fleet))
            card['fresh']=True
        snapshot['jobs']=[dict(assigned_uuid=schedule.JOHN,status='running')]
        self.assertFalse(schedule.required_fleet_idle(snapshot,fleet))
        snapshot['jobs']=[dict(assigned_uuid='unrelated-gpu',status='running')]
        self.assertTrue(schedule.required_fleet_idle(snapshot,fleet))
        cards.pop()
        self.assertFalse(schedule.required_fleet_idle(snapshot,fleet))

    def test_signal_drains_and_resume_never_redispatches(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            root=Path(temp);corpus=root/'corpus.jsonl';raw=root/'raw.npz';raw.write_bytes(b'fixture')
            rows=[{'sample_id':str(i),'prompt':{'corpus':'big_five','synthetic':True,
                   'generator':{'job_id':capture.JOB},'system':'','prompt':str(i),'split':'discovery'}} for i in range(18)]
            corpus.write_text(''.join(capture.encoded(r)+'\n' for r in rows))
            corpus.with_suffix('.manifest.json').write_text(capture.encoded({'job_id':capture.JOB,'corpus_sha256':capture.file_hash(corpus)}))
            endpoints=[dict(gpu_uuid=str(i),model_identity='m',prepared_identity='p',runtime_identity='a'*64,url='http://unused') for i in range(6)]
            config=root/'config.json';config.write_text(capture.encoded({'model':'unused','endpoints':endpoints,'workboard':'unused'}))
            args=SimpleNamespace(config=config,corpus=corpus,ledger=root/'ledger.sqlite',index=root/'index.jsonl',
                until='2099-01-01T00:00:00+00:00',timeout=600,artifact_dir=None,max_tokens=2048,max_new=0,opportunistic=True)
            calls=[];interrupt=[True]
            def fetch(url,payload,timeout):
                calls.append(payload['metadata']['sample_id'])
                if interrupt[0]:
                    interrupt[0]=False
                    os.kill(os.getpid(),signal.SIGTERM)
                return {'activation_artifact':{'path':str(raw)}}
            for name,value in [('COUNT',18),('ModelSpec',SimpleNamespace(load=lambda p:None)),
                ('Admission',lambda p:SimpleNamespace(allowed=lambda e:(True,''))),
                ('native_capability',lambda *a,**kw:'checkpoint'),('fetch',fetch),
                ('validate',lambda *a:{'complete':True})]:
                stack.enter_context(patch.object(capture,name,value))
            with contextlib.redirect_stdout(io.StringIO()):capture.run(args)
            first=json.loads(args.index.with_suffix('.manifest.json').read_text())
            self.assertTrue(first['stopped_by_signal']);self.assertFalse(first['needs_review'])
            self.assertGreater(first['completed'],0);self.assertGreater(first['unattempted'],0)
            with contextlib.redirect_stdout(io.StringIO()):capture.run(args)
            final=json.loads(args.index.with_suffix('.manifest.json').read_text())
            self.assertTrue(final['complete']);self.assertEqual(len(calls),18);self.assertEqual(len(set(calls)),18)

    def test_cap_resume_incomplete_and_ephemeral_endpoint_changes(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            root = Path(temp); corpus = root/'corpus.jsonl'; raw = root/'raw.npz'
            raw.write_bytes(b'fixture')
            rows = [{'sample_id':str(i), 'prompt':{'corpus':'big_five','synthetic':True,
                    'generator':{'job_id':capture.JOB},'system':'','thinking':False,
                    'prompt':'Public scenario '+str(i),'split':'discovery'}} for i in range(18)]
            corpus.write_text(''.join(capture.encoded(r)+'\n' for r in rows))
            corpus.with_suffix('.manifest.json').write_text(capture.encoded({'job_id':capture.JOB,'corpus_sha256':capture.file_hash(corpus)}))
            endpoints = [dict(url='http://localhost:'+str(8000+i),gpu_uuid=str(i),lease_id='old',diagnostic=True,
                model_identity='m',prepared_identity='p',runtime_identity='a'*64) for i in range(6)]
            config=root/'config.json'
            def save_config(): config.write_text(capture.encoded({'endpoints':endpoints,'model':'unused','workboard':'unused'}))
            save_config()
            args=SimpleNamespace(config=config,corpus=corpus,ledger=root/'ledger.sqlite',index=root/'index.jsonl',
                until='2099-01-01T00:00:00+00:00',timeout=600,artifact_dir=None,max_tokens=2048,max_new=3)
            calls=[]
            def fetch(url,payload,timeout):
                self.assertNotIn('steering',payload);self.assertNotIn('observation_only',payload)
                self.assertEqual(payload['max_tokens'],2048)
                sid=payload['metadata']['sample_id'];calls.append(sid)
                return {'complete':sid!='0','activation_artifact':{'path':str(raw)},'provenance_artifact':{},
                        'choices':[{'message':{'content':'answer'},'finish_reason':'length' if sid=='0' else 'stop'}]}
            for name,value in [('COUNT',18),('ModelSpec',SimpleNamespace(load=lambda p:None)),
                ('Admission',lambda p:SimpleNamespace(allowed=lambda e:(True,''))),
                ('native_capability',lambda *a,**kw:'checkpoint'),('fetch',fetch),
                ('validate',lambda r,*a:{'complete':r['complete']}),('in_daily_window',lambda *a:True)]:
                stack.enter_context(patch.object(capture,name,value))
            with contextlib.redirect_stdout(io.StringIO()): capture.run(args)
            with sqlite3.connect(args.ledger) as c:
                self.assertEqual(c.execute("select count(*) from captures where state='complete'").fetchone()[0],3)
            # Stable runtime/model, new leases and URLs: resume must be accepted.
            for i,e in enumerate(endpoints): e.update(url='http://localhost:'+str(9000+i),lease_id='new')
            save_config();args.max_new=0
            with contextlib.redirect_stdout(io.StringIO()):
                capture.run(args)
                with self.assertRaisesRegex(RuntimeError,'explicit scientific revision/review'): capture.run(args)
            self.assertEqual(len(calls),18);self.assertEqual(len(set(calls)),18)
            index=[json.loads(s) for s in args.index.read_text().splitlines()]
            self.assertEqual(len(index),18)
            self.assertEqual(sum(r['complete_pair'] for r in index),17)
            incomplete=next(r for r in index if r['source']['sample_id']=='0')
            self.assertEqual(incomplete['state'],'incomplete')
            self.assertEqual(incomplete['response']['choices'][0]['message']['content'],'answer')
            report=json.loads(args.index.with_suffix('.manifest.json').read_text())
            self.assertFalse(report['complete']);self.assertTrue(report['needs_review']);self.assertEqual(report['unattempted'],0)
            args.max_tokens=3072
            with self.assertRaisesRegex(ValueError,'Resume binding changed'):capture.run(args)
            args.max_tokens=2048
            # Simulate a previously all-complete ledger, then missing/corrupt raw files.
            with sqlite3.connect(args.ledger) as c:
                c.execute("UPDATE captures SET state='complete',error=NULL")
            with patch.object(capture,'in_daily_window',return_value=False), contextlib.redirect_stdout(io.StringIO()):
                capture.run(args)
            report=json.loads(args.index.with_suffix('.manifest.json').read_text())
            self.assertFalse(report['complete']);self.assertFalse(report['needs_review'])
            self.assertEqual(report['validated_this_session'],0)
            self.assertEqual(len(calls),18)
            with patch.object(capture,'validate',side_effect=ValueError('Raw activation hash mismatch')):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError,'hash mismatch'):capture.run(args)
            report=json.loads(args.index.with_suffix('.manifest.json').read_text())
            self.assertFalse(report['complete']);self.assertTrue(report['needs_review'])
            self.assertGreater(report['states'].get('failed',0),0)
            self.assertLess(report['completed'],18)
            self.assertEqual(len(calls),18)
            # Subsequent sessions remain closed, without redispatch or rewarming requests.
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'operator review'):capture.run(args)
            self.assertEqual(len(calls),18)

    def test_atomic_publication_preserves_previous_on_write_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'index.jsonl';path.write_text('previous\n')
            def broken():
                yield 'partial'
                raise RuntimeError('interrupted')
            with self.assertRaisesRegex(RuntimeError,'interrupted'):capture.atomic_write(path,broken())
            self.assertEqual(path.read_text(),'previous\n')
            self.assertEqual(len(list(Path(temp).iterdir())),1)
            capture.atomic_write(path,['complete\n'])
            self.assertEqual(path.read_text(),'complete\n')

    def test_raw_full_capture_and_truncated_answer(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(capture,'verify_diagnostic_artifacts'):
            path=Path(temp)/'raw.npz'
            artifact=dict(model_identity='m',prepared_identity='p',runtime_identity='r',request_sha256='q',operation='rollout',stage='final_prefill')
            arrays={**artifact,'mlp_input_last':np.zeros((64,17408),np.float32),'residual_last':np.zeros((64,5120),np.float32),
                'pre_mlp_last':np.zeros((64,5120),np.float32),'generated_token_ids':np.array([1,2]),
                'completion_graph_kind':np.array(['prefill','decode']),'prefill_positions':np.arange(3),'prompt_token_ids':np.arange(3)}
            def save():
                np.savez(path,**arrays);artifact.update(path=str(path),sha256=capture.file_hash(path))
            save()
            response={'operation':'rollout','diagnostics':{'complete_final_answer':True,'unexpected_thinking':False},
                'activation_artifact':artifact,'provenance_artifact':{},'choices':[{'message':{'content':'Answer'},'finish_reason':'stop'}],
                'usage':{'completion_tokens':2}}
            self.assertTrue(capture.validate(response,{}, {'max_tokens':2048})['complete'])
            response['choices'][0]['finish_reason']='length';response['diagnostics']['complete_final_answer']=False
            self.assertFalse(capture.validate(response,{}, {'max_tokens':2048})['complete'])
            artifact['stage']='decode'
            with self.assertRaisesRegex(ValueError,'stage must be final_prefill'):capture.validate(response,{}, {'max_tokens':2048})
            artifact['stage']='final_prefill'
            artifact['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'hash mismatch'):capture.validate(response,{}, {'max_tokens':2048})
            arrays['mlp_input_last']=np.zeros((63,17408));save()
            with self.assertRaisesRegex(ValueError,'Invalid full capture'):capture.validate(response,{}, {'max_tokens':2048})

if __name__=='__main__': unittest.main()
