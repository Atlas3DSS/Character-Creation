"""Bounded explicit qualification of already-leased local KV workers; never launches models."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import urllib.request
import uuid
import numpy as np
from persona_kv import BINDINGS, make_artifact, file_sha
from persona_vectors import canonical_digest

PREFIXES = {
    'horror': 'Fictional scene context: an abandoned observatory, an unexplained tapping behind a sealed door, and patient unease. No danger has been confirmed.\n\n',
    'romance': 'Fictional scene context: two adult astronomers reunite at their old observatory, sharing restrained affection and the hope of rebuilding trust.\n\n',
}


def fetch(url, value=None):
    data = None if value is None else json.dumps(value, allow_nan=False).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}), timeout=180) as r:
        return json.load(r)


def checked_arrays(response, request, binding=None):
    control = request.get('kv_control')
    learned = bool(control and control['artifact']['kind'] == 'learned_attention_kv')
    offset = control['artifact']['slots'] if learned and control['dose'] == 1 else 0
    if learned:
        from persona_phantom import validate_control
        validate_control(control, binding)
    artifact = response['activation_artifact']
    if artifact['request_sha256'] != canonical_digest(request) or file_sha(artifact['path']) != artifact['sha256']:
        raise ValueError('Activation request/byte binding mismatch')
    if binding is not None:
        for key in ('model_identity', 'runtime_identity', 'prepared_identity'):
            if response.get(key) != binding[key] or artifact.get(key) != binding[key]:
                raise ValueError('Response model/runtime/preparation binding mismatch')
        control = request.get('kv_control')
        if control is not None:
            actual = response.get('kv_control', {})
            if (any(actual.get(k) != binding[k] for k in BINDINGS)
                    or actual.get('artifact_sha256') != control['artifact']['sha256']
                    or actual.get('dose') != control['dose']
                    or actual.get('cache_policy') != control['cache_policy']
                    or actual.get('kind') != control['artifact']['kind']):
                raise ValueError('Response KV binding/dose mismatch')
            if control['dose'] == 1:
                state = actual.get('state_artifact', {})
                if not state or file_sha(state['path']) != state['sha256']:
                    raise ValueError('Response cached state artifact mismatch')
                with np.load(state['path'], allow_pickle=False) as state_data:
                    meta = json.loads(str(state_data['metadata']))
                    if learned:
                        from persona_phantom import SCHEMA, LAYERS, training_validation
                        training_validation(meta, Path(state['path']).parent)
                        if (state['sha256'] != control['artifact']['weights_sha256']
                                or meta.get('schema') != SCHEMA or meta.get('model_identity') != binding['model_identity']
                                or meta.get('key_space') != 'post_norm_post_rope'
                                or meta.get('recurrent_initial_state') != 'fresh_zero'
                                or meta.get('prompt_position_start') != offset
                                or not np.array_equal(state_data['attention_layers'], np.asarray(LAYERS))):
                            raise ValueError('Learned state artifact metadata mismatch')
                        for key in ('keys', 'values'):
                            value = state_data[key]
                            if value.shape != (16, 4, offset, 256) or value.dtype != np.float32 or not np.isfinite(value).all():
                                raise ValueError('Invalid learned state tensor')
                    elif meta['binding'] != binding or meta['definition'] != control['artifact'] or len(state_data['prefix_ids']) != actual['prefix_tokens']:
                        raise ValueError('Cached state metadata/definition mismatch')
            elif actual.get('prefix_tokens') != 0 or actual.get('state_artifact') is not None:
                raise ValueError('Off-state unexpectedly installed prefix')
    if learned:
        actual = response.get('kv_control', {})
        if (actual.get('virtual_prefix_slots') != offset or actual.get('prefix_tokens') != offset
                or actual.get('attention_only') is not True or actual.get('recurrent_initial_state') != 'fresh_zero'
                or response.get('diagnostics', {}).get('virtual_prefix_slots') != offset):
            raise ValueError('Learned virtual slots/fresh recurrence semantics mismatch')
    if not response['diagnostics']['complete_final_answer']:
        raise ValueError('Qualification answer incomplete')
    with np.load(artifact['path'], allow_pickle=False) as data:
        if binding is not None and any(str(data[k]) != binding[k] for k in ('model_identity', 'runtime_identity', 'prepared_identity')):
            raise ValueError('NPZ runtime binding mismatch')
        if str(data['request_sha256']) != canonical_digest(request):
            raise ValueError('NPZ request mismatch')
        if int(data.get('virtual_prefix_slots', 0)) != offset:
            raise ValueError('NPZ virtual prefix offset mismatch')
        tokens = data['prompt_token_ids']; trace = data['completion_trace']
        if (tokens.ndim != 1 or not np.array_equal(data['prefill_positions'], np.arange(offset, offset + len(tokens)))
                or trace.ndim != 2 or trace.shape[1] != 8
                or np.any(np.diff(trace[:, 1]) <= 0) or np.any(np.diff(trace[:, 7]) < 0)
                or trace[-1, 3] != len(data['generated_token_ids'])):
            raise ValueError('Token coverage/completion/RNG progression mismatch')
        for name, shape in [('mlp_input_last', (64, 17408)), ('residual_last', (64, 5120)),
                            ('pre_mlp_last', (64, 5120)), ('residual_l2', (64, max(0, len(tokens)-1)))]:
            if data[name].shape != shape or not np.isfinite(data[name]).all():
                raise ValueError('Invalid finite activation shape: ' + name)
        return {k: data[k].copy() for k in ('prompt_token_ids', 'generated_token_ids', 'mlp_input_last',
                                           'residual_last', 'pre_mlp_last', 'residual_l2')}


def equal(left, right):
    if set(left) != set(right) or any(not np.array_equal(left[k], right[k]) for k in left):
        raise ValueError('Exact cached-prefix/off-state equivalence failed')


def prove_prefix_effect(baseline, grafted, response):
    count = response['kv_control']['prefix_tokens']
    if count <= 0 or not np.array_equal(grafted['prompt_token_ids'][count:], baseline['prompt_token_ids']):
        raise ValueError('Prefix did not extend the unchanged prompt token sequence')
    if np.array_equal(grafted['mlp_input_last'], baseline['mlp_input_last']):
        raise ValueError('On-state activation is a no-op')
    with np.load(response['kv_control']['state_artifact']['path'], allow_pickle=False) as state:
        if not np.array_equal(state['prefix_ids'], grafted['prompt_token_ids'][:count]):
            raise ValueError('Actual prefix tokens differ from cached state')


def qualify_endpoint(endpoint, output=None, run_id=None):
    deadline = time.monotonic() + 600
    while True:
        try:
            model = fetch(endpoint + '/v1/models')['data'][0]
            break
        except (OSError, ValueError):
            if time.monotonic() >= deadline: raise
            time.sleep(5)
    flare = model['flare']
    binding = {k: flare[k] for k in BINDINGS if k != 'layout_sha256'}
    binding['layout_sha256'] = flare['kv_control']['layout_sha256']
    if 'hybrid_text_prefix' not in flare['kv_control']['kinds']:
        raise ValueError('Unexpected cache capability')
    base = dict(model=model['id'], messages=[{'role': 'user', 'content':
        'Write exactly two short sentences continuing a fictional observatory scene. Include a brass key and an unopened letter. Do not explain your choices.'}],
        seed=7035, temperature=0, max_tokens=160, stream=False,
        chat_template_kwargs={'enable_thinking': False})
    records = []; artifacts = []
    attempt_path = None if output is None else output / ('kv-attempt-' + run_id + '-' + endpoint.rsplit(':', 1)[-1] + '.jsonl')
    def save(value):
        if attempt_path is not None:
            with attempt_path.open('a') as stream:
                stream.write(json.dumps(value, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
    def infer(request):
        attempt = uuid.uuid4().hex
        save({'attempt': attempt, 'state': 'started', 'request': request, 'time': time.time()})
        try:
            response = fetch(endpoint + '/v1/chat/completions', request)
            record = {'attempt': attempt, 'request': request, 'response': response}
            records.append(record); save(dict(record, state='response_received', time=time.time()))
            arrays = checked_arrays(response, request, binding)
            save({'attempt': attempt, 'state': 'validated', 'time': time.time()})
            return response, arrays
        except Exception as error:
            save({'attempt': attempt, 'state': 'failed', 'error': repr(error), 'time': time.time()})
            raise
    _, baseline = infer(base)
    for name, text in PREFIXES.items():
        artifact = make_artifact(text, binding); artifacts.append(artifact)
        def request(dose, policy): return dict(base, kv_control=dict(artifact=artifact, dose=dose, cache_policy=policy))
        off, off_arrays = infer(request(0, 'reuse')); equal(baseline, off_arrays)
        rebuilt, rebuilt_arrays = infer(request(1, 'rebuild'))
        prove_prefix_effect(baseline, rebuilt_arrays, rebuilt)
        reused, reused_arrays = infer(request(1, 'reuse')); equal(rebuilt_arrays, reused_arrays)
        if rebuilt['kv_control']['cache_hit'] or not reused['kv_control']['cache_hit']:
            raise ValueError('Qualification did not exercise rebuild then actual cache hit')
        if reused['kv_control']['state_artifact'] != rebuilt['kv_control']['state_artifact']:
            raise ValueError('Cache hit used another state artifact')
        for response in (rebuilt, reused):
            state = response['kv_control']['state_artifact']
            if file_sha(state['path']) != state['sha256']: raise ValueError('State artifact hash mismatch')
            if response['kv_control']['artifact_sha256'] != artifact['sha256']: raise ValueError('Definition binding mismatch')
        _, after = infer(request(0, 'reuse')); equal(baseline, after)
    # Sampled reuse must preserve fresh request RNG; combined steering starts after
    # the unsteered prefix and must not contaminate the reusable prefix state.
    original_base = base
    for combined in (False, True):
        base = dict(original_base, temperature=.7, top_p=.95, top_k=20)
        if combined:
            site = next(s for s in flare['steering']['sites'] if s['name'].endswith('.mlp_input'))
            vector = [0.] * site['width']; vector[0] = .1
            base['steering'] = {'phase': 'all', 'sites': [{'site': site['name'], 'vector': vector}]}
        _, sampled_baseline = infer(base)
        artifact = artifacts[0]
        def sampled_request(dose, policy):
            return dict(base, kv_control=dict(artifact=artifact, dose=dose, cache_policy=policy))
        rebuilt, rebuilt_arrays = infer(sampled_request(1, 'rebuild'))
        prove_prefix_effect(sampled_baseline, rebuilt_arrays, rebuilt)
        reused, reused_arrays = infer(sampled_request(1, 'reuse')); equal(rebuilt_arrays, reused_arrays)
        if rebuilt['kv_control']['cache_hit'] or not reused['kv_control']['cache_hit']:
            raise ValueError('Sampled/combined qualification did not exercise actual cache hit')
        _, sampled_off = infer(sampled_request(0, 'reuse')); equal(sampled_baseline, sampled_off)
    return {'endpoint': endpoint, 'binding': binding, 'passed': True,
            'artifacts': artifacts, 'records': records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint', action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not 1 <= len(args.endpoint) <= 6 or len(set(args.endpoint)) != len(args.endpoint): p.error('Use 1..6 unique already-leased endpoints')
    identity = uuid.uuid4().hex; report = {'id': identity, 'started': time.time(), 'passed': False,
        'schema': 'character-creation.kv-qualification.v1', 'driver_sha256': file_sha(__file__),
        'attempt_files': [str(args.output / ('kv-attempt-' + identity + '-' + e.rsplit(':', 1)[-1] + '.jsonl')) for e in args.endpoint]}
    try:
        with ThreadPoolExecutor(max_workers=len(args.endpoint)) as pool:
            results = list(pool.map(lambda endpoint: qualify_endpoint(endpoint, args.output, identity), args.endpoint))
        binding = results[0]['binding']
        if any(r['binding'] != binding for r in results): raise ValueError('Workers have different runtime bindings')
        report.update(passed=True, binding=binding, results=results,
                      artifact_sha256s=[a['sha256'] for a in results[0]['artifacts']])
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        report['finished'] = time.time()
        path = args.output / ('kv-qualification-' + identity + '.json')
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps({'report': str(path), 'passed': report['passed']}), flush=True)
    proof = {k: report[k] for k in ('id', 'binding', 'passed', 'artifact_sha256s')}
    proof['report'] = {'path': str(path.resolve()), 'sha256': file_sha(path)}
    temporary = args.output / ('kv-qualification-' + identity + '.tmp')
    temporary.write_text(json.dumps(proof, indent=2)); temporary.replace(args.output / 'kv-qualification.json')


if __name__ == '__main__': main()
