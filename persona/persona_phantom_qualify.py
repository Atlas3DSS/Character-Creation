"""Native learned-KV correctness qualification; existing leased workers only.

Twelve requests per worker: greedy, sampled, sampled+MLP each baseline/rebuild/
reuse/off. No uncertain POST retry. This proves bounded cache mechanics, not
literary quality, coherence repair, or the training surrogate's equivalence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import uuid
import numpy as np
from persona_kv import BINDINGS, file_sha
from persona_kv_qualify import fetch, checked_arrays, equal
from persona_phantom import SCHEMA, validate_control
from persona_vectors import canonical_digest


def make_artifact(candidate, binding):
    candidate = Path(candidate)
    with np.load(candidate, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        if metadata.get('schema') != SCHEMA or metadata.get('model_identity') != binding['model_identity']:
            raise ValueError('Candidate schema/checkpoint identity differs')
        slots = metadata['prompt_position_start']
    artifact = dict(schema=SCHEMA, kind='learned_attention_kv', file=candidate.name,
                    weights_sha256=file_sha(candidate), slots=slots, **binding)
    artifact['sha256'] = canonical_digest(artifact)
    validate_control(dict(artifact=artifact, dose=1, cache_policy='rebuild'), binding)
    return artifact


def prove_effect(baseline, grafted):
    if not np.array_equal(baseline['prompt_token_ids'], grafted['prompt_token_ids']):
        raise ValueError('Virtual slots changed actual prompt token IDs')
    # Layers 0–2 are recurrent-only: a virtual attention prefix must not
    # affect them. Both arms carry the same optional MLP intervention.
    for field in ('mlp_input_last', 'residual_last', 'pre_mlp_last'):
        if not np.array_equal(baseline[field][:3], grafted[field][:3]):
            raise ValueError('Learned prefix contaminated pre-attention layers: ' + field)
    if np.array_equal(baseline['mlp_input_last'], grafted['mlp_input_last']):
        raise ValueError('Learned on-state activation is a no-op')


def checked_provenance(response, request, binding):
    artifact = response['provenance_artifact']
    if file_sha(artifact['path']) != artifact['sha256'] or artifact['request_sha256'] != canonical_digest(request):
        raise ValueError('Provenance byte/request binding mismatch')
    for key in ('model_identity', 'runtime_identity', 'prepared_identity'):
        if artifact.get(key) != binding[key]: raise ValueError('Provenance identity mismatch')
    report = json.loads(Path(artifact['path']).read_text())
    if (report.get('status') != 'complete' or report.get('state_restored') is not True
            or report.get('request_sha256') != canonical_digest(request)
            or report.get('request') != request):
        raise ValueError('Request provenance completion/restoration mismatch')
    if any(report.get(k) != binding[k] for k in ('model_identity', 'runtime_identity', 'prepared_identity')):
        raise ValueError('Saved provenance identity mismatch')
    if report.get('response') != {k:v for k,v in response.items() if k != 'provenance_artifact'}:
        raise ValueError('Saved response differs from returned response')


def checked_native(response, request, binding):
    checked_provenance(response, request, binding)
    arrays = checked_arrays(response, request, binding)
    choice = response['choices'][0]
    if choice['finish_reason'] != 'stop' or not choice['message']['content'].strip():
        raise ValueError('Empty or truncated qualification answer')
    with np.load(response['activation_artifact']['path'], allow_pickle=False) as data:
        trace = data['completion_trace']
        offset = int(data.get('virtual_prefix_slots', 0))
        if (not len(arrays['generated_token_ids']) or not np.isfinite(trace).all()
                or trace[-1, 2] != offset + len(arrays['prompt_token_ids']) + len(arrays['generated_token_ids']) - 1):
            raise ValueError('Completion consumed-token progression differs')
        # Epoch and global sequence change between calls; all request-local
        # consumed/generated/status/error/RNG columns must reproduce exactly.
        arrays['completion_state'] = trace[:, 2:].copy()
        arrays['completion_graph_kind'] = data['completion_graph_kind'].copy()
    return arrays


def ready_model(endpoint, timeout):
    """Retry only read-only readiness discovery, never inference requests."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return fetch(endpoint + '/v1/models')['data'][0]
        except OSError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(10, remaining))


def qualify_endpoint(endpoint, candidate, output, run_id, ready_timeout=0):
    model = ready_model(endpoint, ready_timeout)
    flare = model['flare']; capability = flare['kv_control']
    if not capability.get('supported') or 'learned_attention_kv' not in capability['kinds']:
        raise ValueError('Endpoint does not support learned attention KV')
    binding = {k: flare[k] for k in BINDINGS if k != 'layout_sha256'}
    binding['layout_sha256'] = capability['layout_sha256']
    artifact = make_artifact(candidate, binding)
    path = output / ('phantom-attempt-' + run_id + '-' + canonical_digest(endpoint)[:12] + '.jsonl')
    records = []
    def save(value):
        with path.open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
    def infer(request):
        attempt = uuid.uuid4().hex
        save(dict(attempt=attempt, state='started', request=request, endpoint=endpoint, time=time.time()))
        try:
            response = fetch(endpoint + '/v1/chat/completions', request)
            record = dict(attempt=attempt, request=request, response=response)
            records.append(record); save(dict(record, state='response_received', time=time.time()))
            arrays = checked_native(response, request, binding)
            save(dict(attempt=attempt, state='validated', time=time.time()))
            return response, arrays
        except Exception as error:
            save(dict(attempt=attempt, state='failed_or_uncertain', error=repr(error), time=time.time()))
            raise
    for mode in ('greedy', 'sampled', 'sampled_mlp'):
        base = dict(model=model['id'], messages=[dict(role='user', content='Write two short sentences continuing a fictional observatory scene. Include a brass key and an unopened letter. Do not explain your choices.')],
                    seed=7035, temperature=0 if mode == 'greedy' else .7, max_tokens=160, stream=False,
                    chat_template_kwargs={'enable_thinking': False})
        if mode != 'greedy': base.update(top_p=.95, top_k=20)
        if mode == 'sampled_mlp':
            site = next(s for s in flare['steering']['sites'] if s['name'].endswith('.mlp_input'))
            vector = [0.] * site['width']; vector[0] = .1
            base['steering'] = dict(phase='all', sites=[dict(site=site['name'], vector=vector)])
        _, baseline = infer(base)
        def request(dose, policy):
            return dict(base, kv_control=dict(artifact=artifact, dose=dose, cache_policy=policy))
        rebuilt, on = infer(request(1, 'rebuild')); prove_effect(baseline, on)
        reused, reuse = infer(request(1, 'reuse')); equal(on, reuse)
        if rebuilt['kv_control']['cache_hit'] is not False or reused['kv_control']['cache_hit'] is not True:
            raise ValueError('Did not exercise rebuild and actual reuse')
        if rebuilt['kv_control']['state_artifact'] != reused['kv_control']['state_artifact']:
            raise ValueError('Reuse state artifact changed')
        _, off = infer(request(0, 'reuse')); equal(baseline, off)
    return dict(endpoint=endpoint, passed=True, binding=binding, artifacts=[artifact], records=records)


def durable_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.flush(); os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', action='append', required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ready-timeout', type=int, default=0,
                        help='Seconds to wait independently for each worker before its tests')
    args = parser.parse_args(argv)
    if not 0 <= args.ready_timeout <= 3600:
        parser.error('Readiness timeout must be between 0 and 3600 seconds')
    if len(args.endpoint) != 6 or len(set(args.endpoint)) != 6:
        parser.error('Exactly six unique already-leased endpoints required for publication')
    if args.candidate.resolve().parent != args.output.resolve():
        parser.error('Candidate must already reside in the workers shared output directory')
    identity = uuid.uuid4().hex
    report = dict(id=identity, schema='character-creation.phantom-qualification.v1', passed=False,
                  started=time.time(), driver_sha256=file_sha(__file__),
                  validator_sha256=file_sha(Path(__file__).with_name('persona_kv_qualify.py')),
                  candidate_sha256=file_sha(args.candidate), results=[], failures=[],
                  claim='Native cache mechanics only; no behavioral quality claim')
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(qualify_endpoint, e, args.candidate, args.output, identity, args.ready_timeout): e for e in args.endpoint}
        for future in as_completed(futures):
            try: report['results'].append(future.result())
            except Exception as error: report['failures'].append(dict(endpoint=futures[future], error=repr(error)))
    if not report['failures']:
        binding = report['results'][0]['binding']
        artifacts = report['results'][0]['artifacts']
        if any(r['binding'] != binding or r['artifacts'] != artifacts for r in report['results']):
            report['failures'].append(dict(error='Worker bindings/artifacts differ'))
        elif file_sha(args.candidate) != report['candidate_sha256']:
            report['failures'].append(dict(error='Candidate bytes changed during qualification'))
        else:
            report.update(passed=True, binding=binding, artifact_sha256s=[a['sha256'] for a in artifacts])
    report['finished'] = time.time()
    path = args.output / ('phantom-qualification-' + identity + '.json')
    durable_json(path, report)
    if report['passed']:
        proof = {k: report[k] for k in ('id', 'passed', 'binding', 'artifact_sha256s')}
        proof['report'] = dict(path=str(path.resolve()), sha256=file_sha(path))
        temporary = args.output / ('phantom-qualification-' + identity + '.tmp')
        durable_json(temporary, proof); temporary.replace(args.output / 'phantom-qualification.json')
    print(json.dumps(dict(report=str(path), passed=report['passed'])), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
