import hashlib
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('phantom_run', Path(__file__).resolve().parents[1] / 'persona/persona_phantom_run.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_preemptible_args():
    args = m.batch_args(('submit', '--lane', 'batch', '--protected'))
    assert '--protected' not in args
    assert args[-5:] == ['--preemptible', '--checkpoint-signal', 'SIGTERM', '--checkpoint-grace-s', '60']

def test_per_card_usage_and_foreign_gpu():
    ids = list(m.CARDS.values())
    result = m.card_usage(f'3, {ids[0]}, 100\n4, {ids[1]}, 200\n5, {ids[0]}, 300', {3, 4})
    assert result[ids[0]] == 100 and result[ids[1]] == 200
    with pytest.raises(RuntimeError): m.card_usage('3, GPU-foreign, 10', {3})

def test_resource_guards():
    temps = dict.fromkeys(m.CARDS.values(), 40.)
    usage = dict.fromkeys(m.CARDS.values(), 100)
    m.check_resources(temps, usage, 33 * 1024**3)
    with pytest.raises(RuntimeError): m.check_resources(temps, usage, 31 * 1024**3)
    for bad in (85., float('nan')):
        changed = dict(temps); changed[next(iter(changed))] = bad
        with pytest.raises(RuntimeError): m.check_resources(changed, usage, 33 * 1024**3)
    changed = dict(usage); changed[next(iter(changed))] = 31 * 1024 + 1
    with pytest.raises(RuntimeError): m.check_resources(temps, changed, 33 * 1024**3)

def test_source_and_exact_command(tmp_path):
    trainer = tmp_path / 'trainer.py'; trainer.write_text('pass')
    data = tmp_path / 'data.json'; data.write_text('{}')
    bindings = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (trainer, data)}
    m.verify_bindings(bindings)
    command = ['python', str(trainer), 'validate', '--dataset', str(data), '--checkpoint', '/model', '--model-identity', 'a'*64, '--output', '/out', '--devices', *['cuda:'+str(i) for i in range(6)]]
    m.validate_command(command, trainer, bindings)
    with pytest.raises(ValueError): m.validate_command(command[:-1], trainer, bindings)
    data.write_text('changed')
    with pytest.raises(ValueError): m.verify_bindings(bindings)

def test_adapter_lifecycle_without_admission(tmp_path, monkeypatch):
    """Exercise adapter callbacks using a fake installed supervisor; no CLI/GPU."""
    import json, types, time
    trainer = tmp_path / 'trainer.py'; trainer.write_text('pass')
    data = tmp_path / 'data.json'; data.write_text('{}')
    bindings = {str(p): m.sha(p) for p in (trainer, data)}
    manifest = tmp_path / 'bindings.json'; manifest.write_text(json.dumps(bindings))
    report = tmp_path / 'report.json'
    calls, records = [], {}
    def wb(*args):
        calls.append(args)
        if args[0] == 'submit':
            card = args[args.index('--gpu')+1]
            records[card] = {'id': card, 'assigned_uuid': m.CARDS[card], 'lane': 'batch', 'preemptible': True}
            return records[card]
        if args[0] == 'status': return {'jobs': list(records.values())}
        return {}
    fake = types.SimpleNamespace(wb=wb, stop_child=lambda proc: None, tree=lambda root: {7}, on_stop=lambda *a: None)
    def run(args):
        for card in m.CARDS:
            fake.wb('submit', '--gpu', card, '--protected', '--lane', 'batch')
            fake.wb('start', card, '--pid', '7')
        fake.heartbeat(list(records), 9999)
        return 0
    fake.main = run
    monkeypatch.setattr(m.importlib.util, 'spec_from_file_location', lambda *a: types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda module: None)))
    monkeypatch.setattr(m.importlib.util, 'module_from_spec', lambda *a: fake)
    original_sha = m.sha
    monkeypatch.setattr(m, 'sha', lambda path: m.WRAPPER_SHA if str(path) == m.WRAPPER else original_sha(path))
    original_read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **k: m.UNIT if str(path) == '/proc/self/cgroup' else original_read(path, *a, **k))
    monkeypatch.setattr(m.socket, 'gethostname', lambda: 'v100')
    monkeypatch.setattr(m.os, 'sched_getaffinity', lambda pid: {0, 1, 43, 87})
    monkeypatch.setattr(m.os, 'sched_setaffinity', lambda pid, cpus: None)
    monkeypatch.setattr(m.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(m, 'memory_available', lambda: 40*1024**3)
    monkeypatch.setattr(m.subprocess, 'check_output', lambda args, **k: '\n'.join(f'{u}, 40' for u in m.CARDS.values()) if '--query-gpu=uuid,temperature.gpu' in args else f'7, {next(iter(m.CARDS.values()))}, 123')
    command = ['python', str(trainer), 'validate', '--dataset', str(data), '--checkpoint', '/model', '--model-identity', 'a'*64, '--output', '/out', '--devices', *['cuda:'+str(i) for i in range(6)]]
    assert m.main(['--bindings', str(manifest), '--trainer', str(trainer), '--report', str(report), '--deadline', str(time.time()+30), '--', *command]) == 0
    assert json.loads(report.read_text())['state'] == 'completed'
    assert len([c for c in calls if c[0] == 'start' and c[-2:] == ('--unit', m.UNIT)]) == 6
    beats = [c for c in calls if c[0] == 'heartbeat']
    assert [c[-1] for c in beats] == ['123', '0', '0', '0', '0', '0']
    assert all('--protected' not in c and '--preemptible' in c for c in calls if c[0] == 'submit')
