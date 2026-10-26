import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
import persona
from persona_kv import HybridPrefixCache, make_artifact, validate_control
from persona_probe import normalize_request


def engine(tmp_path):
    region = NS(name='recurrent', offset=0, shape=(8,), dtype=torch.float32)
    pool = NS(layout=NS(row_bytes=32, regions=[region]),
              pages={'layer3.key': torch.arange(48, dtype=torch.uint8).reshape(3, 16),
                     'layer3.value': torch.arange(48, dtype=torch.uint8).reshape(3, 16)},
              reserved_backing=torch.arange(96, dtype=torch.uint8).reshape(3, 32))
    return NS(pool=pool, page_tokens=4, model=NS(identity='a'*64),
              manifest={'runtime_identity': 'b'*64},
              prepared=NS(server_config={'identity': 'c'*64}), output=tmp_path,
              vocabulary=100, tokenizer=NS(encode=lambda *a, **k: [1, 2, 3]),
              requests=NS(consumed_tokens=torch.zeros(1, dtype=torch.int64),
                          rng_counter=torch.tensor([99]), generation=torch.tensor([42]),
                          ple_history=torch.tensor([[1, 2]])))


def test_artifact_validation_and_true_off(tmp_path):
    e = engine(tmp_path); cache = HybridPrefixCache(e)
    artifact = make_artifact('A moonlit courtyard.', cache.binding)
    control = dict(artifact=artifact, dose=0, cache_policy='reuse')
    e.tokenizer.encode = lambda *a, **k: pytest.fail('off must not tokenize prefix')
    assert cache.prepare(control) == ([], None)
    assert cache.result(control, [], None, False)['state_artifact'] is None
    for dose in (True, .5, -1, 2):
        with pytest.raises(ValueError): validate_control(dict(control, dose=dose))
    with pytest.raises(ValueError, match='binding mismatch'):
        validate_control(control, dict(cache.binding, model_identity='d'*64))
    with pytest.raises(ValueError, match='SHA256'):
        validate_control(dict(control, artifact=dict(artifact, prefix_text='Changed')))


def test_cache_exact_bytes_reuse_and_rng_isolation(tmp_path):
    e = engine(tmp_path); cache = HybridPrefixCache(e)
    artifact = make_artifact('Gentle anticipation.', cache.binding)
    control = dict(artifact=artifact, dose=1, cache_policy='reuse')
    ids, hit = cache.prepare(control); assert ids == [1, 2, 3] and hit is None
    entry = cache.save(artifact, ids, np.ones((64, 2), dtype=np.float32))
    row = e.pool.reserved_backing[1].clone(); key = e.pool.pages['layer3.key'][0].clone()
    e.pool.reserved_backing[1].zero_(); e.pool.pages['layer3.key'][0].zero_()
    cache.restore(entry)
    assert torch.equal(row, e.pool.reserved_backing[1])
    assert torch.equal(key, e.pool.pages['layer3.key'][0])
    assert e.requests.consumed_tokens.item() == 3
    assert e.requests.rng_counter.item() == 99 and e.requests.generation.item() == 42
    assert cache.prepare(control)[1] is entry
    assert cache.prepare(dict(control, cache_policy='rebuild'))[1] is None
    with open(entry['state_artifact']['path'], 'ab') as f: f.write(b'changed')
    with pytest.raises(ValueError, match='artifact changed'): cache.prepare(control)


def test_qualification_binding_and_request(tmp_path):
    e = engine(tmp_path); cache = HybridPrefixCache(e)
    artifact = make_artifact('Quiet affection.', cache.binding)
    assert cache.capability()['qualified_artifacts'] == []
    proof = dict(passed=True, binding=cache.binding, artifact_sha256s=[artifact['sha256']])
    (tmp_path/'kv-qualification.json').write_text(json.dumps(proof))
    assert cache.capability()['qualified_artifacts'] == [artifact['sha256']]
    proof['binding'] = dict(cache.binding, runtime_identity='f'*64)
    (tmp_path/'kv-qualification.json').write_text(json.dumps(proof))
    assert cache.capability()['qualified_artifacts'] == []
    request = dict(model='m', messages=[dict(role='user', content='Write a scene.')],
                   kv_control=dict(artifact=artifact, dose=1, cache_policy='reuse'))
    assert normalize_request(request, 'm')['kv_control'] == request['kv_control']
    with pytest.raises(ValueError): normalize_request(dict(request, kv_control={'kind': 'dense_kv'}), 'm')


def test_report_stem_is_flat():
    from persona_probe import report_stem
    import argparse
    assert report_stem('kv-probe') == 'kv-probe'
    for value in ('../probe', '', 'a/b', 'a.b', 'x'*65):
        with pytest.raises(argparse.ArgumentTypeError): report_stem(value)


def test_qualification_failure_preserves_response(tmp_path, monkeypatch):
    import persona_kv_qualify as q
    binding = {k: c*64 for k, c in zip(('model_identity', 'runtime_identity', 'prepared_identity'), 'abc')}
    model = dict(id='m', flare={**binding, 'kv_control': dict(layout_sha256='d'*64, kinds=['hybrid_text_prefix'])})
    monkeypatch.setattr(q, 'fetch', lambda url, value=None: {'data': [model]} if value is None else {'raw': 'preserved'})
    def fail(*args): raise ValueError('validation failure')
    monkeypatch.setattr(q, 'checked_arrays', fail)
    with pytest.raises(ValueError, match='validation failure'):
        q.qualify_endpoint('http://127.0.0.1:8930', tmp_path, 'test')
    rows = [json.loads(s) for s in (tmp_path/'kv-attempt-test-8930.jsonl').read_text().splitlines()]
    assert [r['state'] for r in rows] == ['started', 'response_received', 'failed']
    assert rows[1]['response']['raw'] == 'preserved'


def test_qualification_rejects_noop_or_changed_suffix(tmp_path):
    from persona_kv_qualify import prove_prefix_effect
    path = tmp_path/'state.npz'; np.savez(path, prefix_ids=np.array([3]))
    response = {'kv_control': {'prefix_tokens': 1, 'state_artifact': {'path': str(path)}}}
    baseline = {'prompt_token_ids': np.array([1, 2]), 'mlp_input_last': np.array([0.])}
    grafted = {'prompt_token_ids': np.array([3, 1, 2]), 'mlp_input_last': np.array([1.])}
    prove_prefix_effect(baseline, grafted, response)
    with pytest.raises(ValueError, match='no-op'):
        prove_prefix_effect(baseline, dict(grafted, mlp_input_last=np.array([0.])), response)
    with pytest.raises(ValueError, match='unchanged'):
        prove_prefix_effect(baseline, dict(grafted, prompt_token_ids=np.array([3, 1, 4])), response)
