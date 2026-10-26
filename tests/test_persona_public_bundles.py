import pytest
from persona.persona_public_bundles import make_bundle, sanitized


def test_remove_deployment_metadata_without_dropping_scientific_values():
    value = {'path': '/mnt/private.npz', 'sha256': 'a' * 64, 'tokens': [{'id': 3}],
             'endpoint_receipts': [{'url': 'http://localhost'}],
             'validation': {'passed': False, 'tp': 21, 'fp': 3}}
    assert sanitized(value) == {'sha256': 'a' * 64, 'tokens': [{'id': 3}],
                               'validation': {'passed': False, 'tp': 21, 'fp': 3}}


def test_private_strings_fail_closed_in_unknown_fields():
    for value in ('source at /home/person/book.epub', 'hf_' + 'A' * 30):
        with pytest.raises(ValueError): sanitized({'unexpected': value})


def test_export_validation_does_not_replace_failed_outcomes():
    bundle = make_bundle('model', 'coherence', 'run', {'recommendation': {'status': 'heldout_failed'}})
    assert bundle['validation']['passed'] is True
    assert bundle['payload']['recommendation']['status'] == 'heldout_failed'
