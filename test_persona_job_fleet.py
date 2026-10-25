import hashlib
import json
from pathlib import Path
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from persona_job_fleet import cleanup_intents, fleet_session, launch_owned, prior_launch_intents, reconcile_recovered, worker_command
import persona_lease
import persona_planner
import persona_service_control as control


class FleetOwnershipTests(unittest.TestCase):
    def test_pidfd_cleanup_rechecks_replacement_and_cgroup_before_signalling(self):
        state = {'LoadState': 'loaded', 'ActiveState': 'active', 'InvocationID': 'ours', 'MainPID': 123,
            'ControlGroup': '/user/our-unit', 'job_id': 'job', 'launch_token': 'a'*32}
        identity = {'start_time': 99, 'groups': ['/user/our-unit']}
        for changed in ({**state, 'InvocationID': 'replacement', 'MainPID': 456, 'launch_token': 'b'*32},
                        {**state, 'ControlGroup': '/other-unit'}):
            with patch.object(control, 'unit_state', side_effect=[state, changed]), \
                 patch.object(control.os, 'pidfd_open', return_value=17), patch.object(control.os, 'close') as close, \
                 patch.object(control, 'proc_identity', return_value=identity), patch.object(control.signal, 'pidfd_send_signal') as send:
                result = control.stop_owned(control.PLANNER_UNIT, 'job', 'a'*32)
                self.assertEqual(result['status'], 'identity_changed_before_signal'); send.assert_not_called(); close.assert_called_once_with(17)
        with patch.object(control, 'unit_state', return_value=state), patch.object(control.os, 'pidfd_open', return_value=17), \
             patch.object(control.os, 'close'), patch.object(control, 'proc_identity', return_value={'start_time': 99, 'groups': ['/unrelated']}), \
             patch.object(control.signal, 'pidfd_send_signal') as send:
            with self.assertRaisesRegex(RuntimeError, 'cgroup'): control.stop_owned(control.PLANNER_UNIT, 'job', 'a'*32)
            send.assert_not_called()

    def test_verified_signal_uses_immutable_pidfd_not_reusable_unit_name(self):
        state = {'LoadState':'loaded','ActiveState':'active','InvocationID':'ours','MainPID':123,
            'ControlGroup':'/our-unit','job_id':'job','launch_token':'a'*32}
        poll = Mock(); poll.poll.return_value = [(17, 1)]
        with patch.object(control, 'unit_state', return_value=state), patch.object(control.os, 'pidfd_open', return_value=17), \
             patch.object(control.os, 'close'), patch.object(control, 'proc_identity', return_value={'start_time':99,'groups':['/our-unit']}), \
             patch.object(control.signal, 'pidfd_send_signal') as send, patch.object(control.select, 'poll', return_value=poll), \
             patch.object(control.subprocess, 'run') as command:
            self.assertTrue(control.stop_owned(control.PLANNER_UNIT, 'job', 'a'*32)['stopped'])
            send.assert_called_once_with(17, signal.SIGTERM); command.assert_not_called()
        with self.assertRaises(ValueError): control.stop_owned('unrelated.service', 'job', 'a'*32)

    def test_cleanup_skips_another_token_before_opening_any_process(self):
        state = {'LoadState':'loaded','ActiveState':'active','job_id':'someone-else','launch_token':'b'*32}
        with patch.object(control, 'unit_state', return_value=state), patch.object(control.os, 'pidfd_open') as opened:
            self.assertEqual(control.stop_owned(control.PLANNER_UNIT, 'job', 'a'*32)['status'], 'different_owner')
            opened.assert_not_called()

    def test_worker_uses_fixed_supervisor_and_disjoint_configured_affinity(self):
        config = dict(remote_root='/research', sysroot='/sysroot', compile_workers=[2]*6,
                      gpus=[f'GPU-{i}' for i in range(6)], cuda_home='/cuda', runtime_path='/bin',
                      inductor_cache='/inductor', triton_cache='/triton', lease_wrapper='/installed/wb_hold.py',
                      gpu_labels=[f'v100-{i}' for i in range(6)], cpu_sets=[f'{2*i}-{2*i+1}' for i in range(6)],
                      python='/venv/python', flare_source='/flare', checkpoint='/weights',
                      kernel_cache='/kernels', cutlass_include='/cutlass')
        for shard in range(6):
            remaining = 28500 - shard * 10
            argv = worker_command(config, {'lease_wrapper_sha256': 'a'*64}, shard, 480, remaining)
            index = argv.index('/usr/bin/python3')
            self.assertEqual(argv[index+1:index+3], ['/research/persona_lease.py', 'run'])
            self.assertIn(f'--setenv=PERSONA_WORKER_UNIT=character-persona-probe-{shard}.service', argv[:index])
            self.assertIn('--setenv=WB_HOLD_WAIT=' + str(remaining), argv[:index])
            self.assertEqual(argv[argv.index('taskset')+2], config['cpu_sets'][shard])
            self.assertIn('--property=RuntimeMaxSec=' + str(remaining), argv)
            self.assertNotIn('--property=RuntimeMaxSec=28800', argv)

    def test_lost_start_or_status_reply_retains_precommitted_cleanup_intent(self):
        for failure in ('start', 'state'):
            intents, events = [], []
            def event(name, value): events.append((name, value.copy()))
            with patch('persona_job_fleet.run', side_effect=TimeoutError('lost start receipt') if failure == 'start' else None) as run, \
                 patch('persona_job_fleet.unit_state', side_effect=TimeoutError('lost state receipt')):
                with self.assertRaises(TimeoutError):
                    launch_owned(control.PLANNER_UNIT, ['systemd-run', '--unit='+control.PLANNER_UNIT,
                        '--property=RuntimeMaxSec=999', '/usr/bin/python3', 'fixed.py'],
                        {'id':'job','deadline':time.time()+60}, event, intents, host='wtatum84', helper='/fixed/helper.py')
            self.assertEqual(len(intents), 1); self.assertEqual(events[0][0], 'service_launch_intent')
            self.assertEqual(events[0][1]['token'], intents[0]['token'])
            argv = run.call_args.args[0]
            self.assertIn('--setenv=PERSONA_JOB_ID=job', argv)
            self.assertIn('--setenv=PERSONA_LAUNCH_TOKEN='+intents[0]['token'], argv)
            self.assertIn('--property=Restart=no', argv)
            self.assertLessEqual(int(next(v.split('=')[-1] for v in argv if v.startswith('--property=RuntimeMaxSec='))),60)
            with patch('persona_job_fleet.stop_owned', return_value={'stopped':True,'status':'owned_process_exited'}) as stop:
                cleanup_intents(intents, event, ssh=['fake-ssh'], local_helper='/local', remote_helper='/remote')
            self.assertEqual(stop.call_args.kwargs['token'], intents[0]['token'])

    def test_failed_prelaunch_audit_prevents_service_creation(self):
        with patch('persona_job_fleet.run') as run:
            with self.assertRaises(OSError):
                launch_owned(control.PLANNER_UNIT, ['systemd-run','--unit='+control.PLANNER_UNIT,'/usr/bin/python3','fixed.py'],
                    {'id':'job','deadline':time.time()+60}, Mock(side_effect=OSError('audit unavailable')), [],
                    host='wtatum84', helper='/helper')
            run.assert_not_called()

    def test_logging_failure_does_not_skip_other_owned_cleanup(self):
        intents = [{'unit':unit,'job_id':'job','token':'a'*32,'host':'wtatum84' if unit in control.UNITS[:2] else 'v100'}
                   for unit in control.UNITS]
        with patch('persona_job_fleet.stop_owned', return_value={'stopped':True}) as stop:
            with self.assertRaisesRegex(RuntimeError, 'audit'):
                cleanup_intents(intents, Mock(side_effect=OSError('database unavailable')), ssh=['ssh'], local_helper='/local', remote_helper='/remote')
        self.assertEqual(stop.call_count, 8)
        self.assertEqual([call.args[0] for call in stop.call_args_list], [intent['unit'] for intent in reversed(intents)])

    def test_resume_reads_intents_older_than_ui_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'results').mkdir()
            intent = {'unit':control.PLANNER_UNIT,'job_id':'job','token':'a'*32,'host':'wtatum84'}
            with sqlite3.connect(root/'results/persona_jobs.sqlite') as db:
                db.execute('CREATE TABLE events(sequence INTEGER PRIMARY KEY,job_id TEXT,event TEXT,data TEXT)')
                db.execute('INSERT INTO events VALUES(1,?,?,?)', ('job','service_launch_intent',json.dumps(intent)))
                db.executemany('INSERT INTO events VALUES(?,?,?,?)',[(i,'job','progress','{}') for i in range(2,152)])
            self.assertEqual(prior_launch_intents({'id':'job','events':[]},root),[intent])

    def test_terminal_recovery_marks_only_successful_cleanup_without_admission(self):
        context=(Path('/root'),{},['ssh'],'/local/helper','/remote/helper')
        intents=[{'unit':control.PLANNER_UNIT,'job_id':'job','token':'a'*32,'host':'wtatum84'}]
        order=[]
        with patch('persona_job_fleet.control_context',return_value=context), \
             patch('persona_job_fleet.prior_launch_intents',return_value=intents), \
             patch('persona_job_fleet.cleanup_intents',side_effect=lambda *a,**k:order.append('cleanup')) as cleanup, \
             patch('persona_job_fleet.launch_owned') as launch, patch('persona_job_fleet.verified_session') as admission:
            result=reconcile_recovered({'id':'job','state':'cancelled'},{},lambda name,data:order.append(name))
        self.assertEqual(order,['cleanup','resource_reconciled'])
        self.assertEqual(result['intent_count'],1); self.assertFalse(result['resources_started'])
        self.assertEqual(cleanup.call_args.args[0],intents)
        launch.assert_not_called(); admission.assert_not_called()

    def test_interrupted_terminal_cleanup_has_no_marker_and_can_retry(self):
        context=(Path('/root'),{},['ssh'],'/local/helper','/remote/helper'); events=[]
        with patch('persona_job_fleet.control_context',return_value=context), \
             patch('persona_job_fleet.prior_launch_intents',return_value=[]), \
             patch('persona_job_fleet.cleanup_intents',side_effect=[RuntimeError('connection interrupted'),None]) as cleanup:
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                reconcile_recovered({'id':'job','state':'expired'},{},lambda name,data:events.append(name))
            self.assertEqual(events,[])
            reconcile_recovered({'id':'job','state':'expired'},{},lambda name,data:events.append(name))
        self.assertEqual(cleanup.call_count,2); self.assertEqual(events,['resource_reconciled'])

    def test_cancelled_startup_reconciles_old_intent_and_cleans_all_new_intents(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            sources=('persona_service_control.py','persona_probe.py','persona_capture.py','persona_vectors.py',
                     'persona_fleet.py','persona_qualify.py','persona_lease.py')
            for name in sources: (root/name).write_text(name)
            fleet=dict(hostname='v100',ssh_host='v100',remote_root='/remote',sysroot='/sysroot',compile_workers=[2]*6,
                gpus=[f'GPU-{i}' for i in range(6)],cuda_home='/cuda',runtime_path='/bin',inductor_cache='/inductor',
                triton_cache='/triton',lease_wrapper='/installed/wb_hold.py',gpu_labels=[f'v100-{i}' for i in range(6)],
                cpu_sets=[f'{2*i}-{2*i+1}' for i in range(6)],python='/venv/python',flare_source='/flare',
                checkpoint='/weights',kernel_cache='/kernels',cutlass_include='/cutlass')
            config_path=root/'fleet.json'; config_path.write_text(json.dumps(fleet))
            config={'root':str(root),'fleet_config':str(config_path),'lease_wrapper_sha256':'c'*64}
            job={'id':'job','deadline':time.time()+600}; stopped=[]; started=[]; stop=threading.Event()
            prior={'unit':control.PLANNER_UNIT,'job_id':'job','token':'a'*32,'host':'wtatum84'}
            states={control.PLANNER_UNIT:{'ActiveState':'active','InvocationID':'old','job_id':'job','launch_token':'a'*32}}
            def state(unit,*args,**kwargs): return states.get(unit,{'ActiveState':'inactive','LoadState':'not-found'}).copy()
            def run(argv,**kwargs):
                if argv[0]=='hostname': return 'v100\n'
                if argv[0]=='sha256sum':
                    name=Path(argv[1]).name
                    digest='c'*64 if name=='wb_hold.py' else hashlib.sha256((root/name).read_bytes()).hexdigest()
                    return digest+'  path\n'
                self.assertEqual(argv[0],'systemd-run')
                unit=next(value.split('=',1)[1] for value in argv if value.startswith('--unit='))
                if not unit.endswith('.service'): unit+='.service'
                token=next(value.split('=',2)[2] for value in argv if value.startswith('--setenv=PERSONA_LAUNCH_TOKEN='))
                self.assertNotEqual(state(unit)['ActiveState'],'active')
                states[unit]={'ActiveState':'active','LoadState':'loaded','InvocationID':'new-'+unit,'job_id':'job','launch_token':token}
                started.append(unit); return ''
            def cleanup(unit,invocation,ssh=None,**kwargs):
                self.assertEqual(kwargs['token'],state(unit)['launch_token'])
                stopped.append((unit,kwargs['token'])); states[unit]={'ActiveState':'inactive','LoadState':'not-found'}
                return {'stopped':True,'status':'owned_process_exited'}
            def event(name,data):
                if name=='service_started' and len(started)==3: stop.set()
            connection=Mock(); connection.__enter__=Mock(return_value=connection); connection.__exit__=Mock(return_value=False)
            connection.connect_ex.return_value=1
            with patch('persona_job_fleet.socket.gethostname',return_value='wtatum84'), \
                 patch('persona_job_fleet.socket.socket',return_value=connection), patch('persona_job_fleet.run',side_effect=run), \
                 patch('persona_job_fleet.unit_state',side_effect=state), patch('persona_job_fleet.stop_owned',side_effect=cleanup), \
                 patch('persona_job_fleet.prior_launch_intents',return_value=[prior]):
                with self.assertRaises(InterruptedError):
                    with fleet_session(job,config,event,stop_event=stop): self.fail('Cancelled startup must not yield resources')
            self.assertEqual(len(started),3); self.assertEqual(len(stopped),4)
            self.assertEqual(stopped[0],(control.PLANNER_UNIT,'a'*32))
            self.assertTrue(all(value['ActiveState']=='inactive' for value in states.values()))

    def test_absolute_deadline_signals_only_own_supervisor_installed_handler(self):
        done=Mock(); done.is_set.return_value=False; done.wait.return_value=False
        with patch('persona_lease.time.time',side_effect=[100.,101.,102.]), \
             patch('persona_lease.os.getpid',return_value=777), patch('persona_lease.os.kill') as signal_self:
            persona_lease.watch_deadline(done,102.)
        signal_self.assert_called_once_with(777,signal.SIGTERM)
        self.assertEqual([call.args for call in done.wait.call_args_list],[(1.,),(1.,)])
        done.is_set.return_value=True
        with patch('persona_lease.os.kill') as signal_self:
            persona_lease.watch_deadline(done,102.)
        signal_self.assert_not_called()

    def test_planner_reconciles_committed_acquire_with_lost_reply(self):
        receipt = {'id':'exact-job','project':'character-creation','label':'persona-planner-john',
            'target_host':'wtatum84','gpu_selector':persona_planner.GPU}
        leased = {**receipt,'status':'leased','assigned_uuid':persona_planner.GPU}
        with patch.object(persona_planner,'wb',side_effect=[{'jobs':[leased]},{}]) as wb:
            persona_planner.release_lease('exact-job',receipt,False,'interrupted')
            self.assertEqual(wb.call_args_list[-1].args[:2],('finish','exact-job'))
            self.assertFalse(any(call.args[0]=='cancel' for call in wb.call_args_list))
        with patch.object(persona_planner,'wb',side_effect=[{'jobs':[{**receipt,'status':'queued'}]},{}]) as wb:
            persona_planner.release_lease('exact-job',receipt,False,'failed')
            self.assertEqual(wb.call_args_list[-1].args,('cancel','exact-job'))
        with patch.object(persona_planner,'wb',return_value={'jobs':[{**leased,'project':'someone-else'}]}) as wb:
            with self.assertRaisesRegex(RuntimeError,'identity changed'):
                persona_planner.release_lease('exact-job',receipt,False,'failed')
            self.assertEqual(wb.call_count,1)

    def test_vanished_child_does_not_prevent_exact_lease_cleanup(self):
        child=Mock(pid=123); child.poll.return_value=None; child.wait.return_value=0
        receipt={'id':'exact-job','project':'character-creation','label':'persona-planner-john',
                 'target_host':'wtatum84','gpu_selector':persona_planner.GPU}
        with patch.object(persona_planner.os,'killpg',side_effect=ProcessLookupError), \
             patch.object(persona_planner,'wb',side_effect=[{'jobs':[{**receipt,'status':'running','assigned_uuid':persona_planner.GPU}]},{}]) as wb:
            persona_planner.stop_child(child)
            persona_planner.release_lease('exact-job',receipt,True,'interrupted')
        child.wait.assert_called_once_with(timeout=20)
        self.assertEqual(wb.call_args_list[-1].args[:2],('finish','exact-job'))

    def test_lease_shim_only_adds_named_unit_to_start_registration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'wrapper.py'; path.write_text('installed supervisor fixture')
            unit = 'character-persona-probe-2.service'
            calls = []
            module = SimpleNamespace(wb=lambda *args: calls.append(args), on_stop=lambda *_: None)
            def main(argv):
                module.wb('start', 'job', '--pid', '123')
                module.wb('heartbeat', 'job', '--pid', '123')
                return 0
            module.main = main
            spec = SimpleNamespace(loader=SimpleNamespace(exec_module=Mock()))
            with patch.dict('os.environ', {'PERSONA_WORKER_UNIT':unit, 'PERSONA_LEASE_WRAPPER':str(path),
                    'PERSONA_LEASE_WRAPPER_SHA256':hashlib.sha256(path.read_bytes()).hexdigest()}), \
                 patch.object(Path, 'read_text', return_value=unit), \
                 patch('sys.argv', ['shim', 'run']), patch('persona_lease.signal.signal'), \
                 patch('persona_lease.importlib.util.spec_from_file_location', return_value=spec), \
                 patch('persona_lease.importlib.util.module_from_spec', return_value=module):
                self.assertEqual(persona_lease.main(), 0)
            self.assertEqual(calls, [('start','job','--pid','123','--unit',unit), ('heartbeat','job','--pid','123')])


if __name__ == '__main__':
    unittest.main()
