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
        self.assertEqual(schedule.session_end(now, True).hour, 20)
        self.assertFalse(capture.request_window(now.timestamp(), 60, now.timestamp()+7200))
        self.assertTrue(capture.request_window(now.timestamp(), 60, now.timestamp()+7200, True))
        self.assertFalse(capture.request_window(now.timestamp(), 60, now.timestamp()+60, True))
        for stamp, hour, day in [('2026-10-25T08:00:00-07:00',20,25),
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

    def test_priority_yield_matches_required_resources(self):
        fleet={'gpus':[str(i) for i in range(6)]}
        snapshot={'fleet':{'cards':[dict(uuid=schedule.JOHN,host='wtatum84',label='john')]+[
            dict(uuid=str(i),host='v100',label='v100-'+str(i)) for i in range(6)]}}
        mike=dict(status='queued',project='atlas-orientation',lane='interactive',
                  target_host='MonkeyPC',gpu_selector='GPU-ca123220-3e85-d601-2e49-edbb6e6b22a9')
        self.assertIsNone(schedule.priority_waiting([mike],fleet,snapshot))
        self.assertIsNone(schedule.priority_waiting([{**mike,'gpu_selector':'any'}],fleet,snapshot))
        for job in [dict(status='queued',lane='interactive'),
                    dict(status='queued',lane='interactive',target_host='any',gpu_selector='any'),
                    dict(status='queued',lane='interactive',gpu_selector=schedule.JOHN)]:
            self.assertEqual(schedule.priority_waiting([job],fleet,snapshot),'interactive')
        self.assertEqual(schedule.priority_waiting([dict(status='queued',project='dsv41',
                         gpu_selector=schedule.JOHN)],fleet,snapshot),'chemistry')
        manifest={'can_continue_unattempted':True,'unattempted':10,'errors':[],
                  'states':{'failed':3,'incomplete':3},'needs_review':True,
                  'reviewed_recovery':{'schema':'character-creation.baseline-reviewed-recovery.v1',
                  'quarantines':[{'sample_id':str(i)} for i in range(6)]}}
        self.assertTrue(schedule.reviewed_continuation(manifest))
        self.assertFalse(schedule.reviewed_continuation({**manifest,'unattempted':0}))
        self.assertFalse(schedule.reviewed_continuation({**manifest,'reviewed_recovery':None}))
        self.assertFalse(schedule.reviewed_continuation({**manifest,'states':{'failed':4,'incomplete':3}}))

    def test_reviewed_migration_retains_quarantines_and_only_dispatches_unattempted(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            root=Path(temp);corpus=root/'corpus.jsonl';raw=root/'raw.npz';raw.write_bytes(b'fixture')
            rows=[{'sample_id':str(i),'prompt':{'corpus':'big_five','synthetic':True,
                'generator':{'job_id':capture.JOB},'system':'','prompt':str(i),'split':'discovery'}} for i in range(18)]
            corpus.write_text(''.join(capture.encoded(r)+'\n' for r in rows))
            manifest={'job_id':capture.JOB,'corpus_sha256':capture.file_hash(corpus)}
            corpus.with_suffix('.manifest.json').write_text(capture.encoded(manifest))
            endpoints=[dict(gpu_uuid=str(i),model_identity='m',prepared_identity='p',runtime_identity='a'*64,url='http://unused') for i in range(6)]
            config=root/'config.json';config.write_text(capture.encoded({'model':'unused','endpoints':endpoints,'workboard':'unused'}))
            args=SimpleNamespace(config=config,corpus=corpus,ledger=root/'ledger.sqlite',index=root/'index.jsonl',
                until='2099-01-01T00:00:00+00:00',timeout=600,artifact_dir=None,max_tokens=2048,max_new=3,
                opportunistic=True,command='prepare-recovery',recovery_record=root/'review.json',
                recovery_reason='Reviewed Atlas interruption; retain failures and truncations, continue only untouched rows',quarantine_report=[])
            original=capture.capture_binding(manifest,endpoints,2048);original['code_sha256']='old-runner'
            retained=[]
            with sqlite3.connect(args.ledger) as c:
                c.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
                c.execute('CREATE TABLE captures(sample_id TEXT PRIMARY KEY,ordinal INTEGER UNIQUE,state TEXT NOT NULL,payload TEXT NOT NULL,response TEXT,error TEXT,started REAL,finished REAL)')
                c.execute('INSERT INTO metadata VALUES(?,?)',('binding',capture.encoded(original)))
                for i in range(6):
                    payload={'model':'checkpoint','messages':[{'role':'user','content':str(i)}],
                        'seed':7035,'max_tokens':2048,'temperature':0,'stream':False,
                        'chat_template_kwargs':{'enable_thinking':False},'metadata':{'job_id':'baseline40k-'+str(i),
                        'sample_id':str(i),'source_job_id':capture.JOB,'arm':'baseline','split':'discovery'}}
                    response={'activation_artifact':{'path':str(raw),'sha256':capture.file_hash(raw)},
                              'choices':[{'finish_reason':'length'}]} if i>=3 else None
                    report=root/('probe-'+str(i)+'.jsonl');args.quarantine_report.append(report)
                    identities={k:endpoints[i][k] for k in ('model_identity','prepared_identity','runtime_identity')}
                    report.write_text(capture.encoded({'request_sha256':capture.canonical_digest(payload),
                        'request':payload,'operation':'rollout',**identities,
                        'worker_manifest':{**identities,'gpu_uuid':str(i)},
                        'status':'complete' if response else 'failed','state_restored':True,'response':response,
                        'activation_artifact':response['activation_artifact'] if response else None,
                        'error':None if response else 'ProtocolError: Worker interrupted before graph replay'}))
                    if response:
                        response['provenance_artifact']={**identities,'operation':'rollout',
                            'request_sha256':capture.canonical_digest(payload),'path':str(report),
                            'sha256':capture.file_hash(report),'self_reference':'Saved response excludes this post-write provenance pointer'}
                    row=(str(i),i,'incomplete' if response else 'failed',capture.encoded(payload),
                         capture.encoded(response) if response else None,'retained-error',100.,None)
                    c.execute('INSERT INTO captures VALUES(?,?,?,?,?,?,?,?)',row);retained.append(row)
            # Missing/wrong report identities and mismatched response/artifact fail before approval.
            for index,change in [(0,lambda r:r.pop('model_identity')),
                                 (0,lambda r:r.update(prepared_identity='wrong')),
                                 (0,lambda r:r['worker_manifest'].update(runtime_identity='wrong')),
                                 (3,lambda r:r['response']['choices'][0].update(finish_reason='stop')),
                                 (3,lambda r:r.update(activation_artifact={'path':'wrong'}))]:
                path=args.quarantine_report[index];saved=path.read_text();report=json.loads(saved);change(report)
                path.write_text(capture.encoded(report))
                with self.assertRaises(ValueError):capture.reviewed_recovery(args)
                path.write_text(saved)
                self.assertFalse(args.recovery_record.exists())
            with contextlib.redirect_stdout(io.StringIO()):capture.reviewed_recovery(args)
            record=json.loads(args.recovery_record.read_text());prior_hash=capture.file_hash(args.recovery_record)
            self.assertEqual(record['original_binding'],original);self.assertEqual(len(record['quarantines']),6)
            args.replace_draft=True
            with contextlib.redirect_stdout(io.StringIO()):capture.reviewed_recovery(args)
            replacement=json.loads(args.recovery_record.read_text())
            self.assertEqual(replacement['superseded_draft'],{'sha256':prior_hash,'record':record})
            args.command='approve-recovery';args.max_tokens=3072
            with self.assertRaisesRegex(ValueError,'scientific binding'):capture.reviewed_recovery(args)
            args.max_tokens=2048
            with contextlib.redirect_stdout(io.StringIO()):capture.reviewed_recovery(args)
            args.command='prepare-recovery'
            with self.assertRaisesRegex(ValueError,'already approved'):capture.reviewed_recovery(args)
            calls=[]
            def fetch(url,payload,timeout):
                calls.append(payload['metadata']['sample_id'])
                return {'activation_artifact':{'path':str(raw)}}
            for name,value in [('COUNT',18),('ModelSpec',SimpleNamespace(load=lambda p:None)),
                ('Admission',lambda p:SimpleNamespace(allowed=lambda e:(True,''))),
                ('native_capability',lambda *a,**kw:'checkpoint'),('fetch',fetch),('validate',lambda *a:{'complete':True})]:
                stack.enter_context(patch.object(capture,name,value))
            # Offline publication does not contact endpoints or claim complete validation.
            args.until='2020-01-01T00:00:00+00:00'
            with patch.object(capture,'native_capability',side_effect=AssertionError('offline network call')):
                with contextlib.redirect_stdout(io.StringIO()):capture.run(args)
            first=json.loads(args.index.with_suffix('.manifest.json').read_text())
            self.assertTrue(first['can_continue_unattempted']);self.assertTrue(first['needs_review'])
            args.until='2099-01-01T00:00:00+00:00'
            with contextlib.redirect_stdout(io.StringIO()):capture.run(args)
            first=json.loads(args.index.with_suffix('.manifest.json').read_text())
            self.assertTrue(first['can_continue_unattempted']);self.assertFalse(first['complete'])
            args.max_new=0
            with contextlib.redirect_stdout(io.StringIO()):capture.run(args)
            final=json.loads(args.index.with_suffix('.manifest.json').read_text())
            self.assertFalse(final['can_continue_unattempted']);self.assertTrue(final['needs_review']);self.assertFalse(final['complete'])
            self.assertEqual(set(calls),{str(i) for i in range(6,18)});self.assertEqual(len(calls),12)
            with sqlite3.connect(args.ledger) as c:
                self.assertEqual(c.execute('SELECT * FROM captures WHERE ordinal<6 ORDER BY ordinal').fetchall(),retained)
            args.quarantine_report[0].write_text('{}')
            with self.assertRaisesRegex(ValueError,'report changed'):capture.run(args)

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
