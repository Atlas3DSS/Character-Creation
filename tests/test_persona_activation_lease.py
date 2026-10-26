import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'activation-search/persona_lease.py'
spec = importlib.util.spec_from_file_location('activation_lease', path)
lease = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lease)


def test_batch_renewals_are_preemptible_without_changing_other_admission_args():
    original = ['submit', '--vram-gib', '31', '--lane', 'batch', '--protected']
    actual = lease.submission_args(original, 'batch')
    assert '--protected' not in actual
    assert '--preemptible' in actual
    assert actual[actual.index('--checkpoint-signal') + 1] == 'SIGTERM'
    assert actual[actual.index('--checkpoint-grace-s') + 1] == '60'
    assert actual[actual.index('--vram-gib') + 1] == '31'
    assert original[-1] == '--protected'


def test_interactive_preserves_existing_supervisor_policy():
    actual = lease.submission_args(['submit', '--lane', 'batch', '--protected'], 'interactive')
    assert '--protected' in actual
    assert '--preemptible' not in actual
    assert actual[-2:] == ['--lane', 'interactive']
