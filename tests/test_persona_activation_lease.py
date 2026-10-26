import importlib.util
from pathlib import Path
import pytest

@pytest.fixture(params=['activation-search', 'persona'])
def lease(request):
    path = Path(__file__).resolve().parents[1] / request.param / 'persona_lease.py'
    spec = importlib.util.spec_from_file_location('tested_lease', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_renewals_are_preemptible_without_changing_other_admission_args(lease):
    original = ['submit', '--vram-gib', '31', '--lane', 'batch', '--protected']
    actual = lease.submission_args(original, 'batch')
    assert '--protected' not in actual
    assert '--preemptible' in actual
    assert actual[actual.index('--checkpoint-signal') + 1] == 'SIGTERM'
    assert actual[actual.index('--checkpoint-grace-s') + 1] == '60'
    assert actual[actual.index('--vram-gib') + 1] == '31'
    assert original[-1] == '--protected'


def test_interactive_preserves_existing_supervisor_policy(lease):
    actual = lease.submission_args(['submit', '--lane', 'batch', '--protected'], 'interactive')
    assert '--protected' in actual
    assert '--preemptible' not in actual
    assert actual[-2:] == ['--lane', 'interactive']
