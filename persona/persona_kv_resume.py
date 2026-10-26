"""Resume readiness-only KV qualification failures without repeating completed POSTs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import uuid
import persona_kv_qualify as q
from persona_kv import file_sha
from persona_vectors import canonical_digest


def saved_responses(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if len(rows) != 51:
        raise ValueError('Partial/uncertain qualification cannot be automatically replayed')
    records = []
    for offset in range(0, len(rows), 3):
        started, received, validated = rows[offset:offset+3]
        if ([r['state'] for r in (started, received, validated)] != ['started', 'response_received', 'validated']
                or len({r['attempt'] for r in (started, received, validated)}) != 1
                or started['request'] != received['request']):
            raise ValueError('Qualification attempt sequence is incomplete or inconsistent')
        records.append(received)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--endpoint', action='append', required=True)
    args = p.parse_args()
    previous = json.loads(args.previous_report.read_text())
    if (previous.get('passed') is not False or 'URLError' not in previous.get('error', '')
            or previous.get('driver_sha256') != file_sha(q.__file__)
            or not 1 <= len(args.endpoint) <= 6 or len(set(args.endpoint)) != len(args.endpoint)):
        raise ValueError('Only an exact-source, readiness-only failed run may resume')
    files = {Path(path).name.rsplit('-', 1)[-1].removesuffix('.jsonl'): Path(path)
             for path in previous['attempt_files']}
    saved, cursors = {}, {}
    for endpoint in args.endpoint:
        port = endpoint.rsplit(':', 1)[-1]
        if port not in files: raise ValueError('Endpoint absent from previous qualification')
        if files[port].exists(): saved[endpoint] = saved_responses(files[port]); cursors[endpoint] = 0
    real_fetch = q.fetch
    def fetch(url, value=None):
        endpoint = url.removesuffix('/v1/chat/completions')
        if value is not None and endpoint in saved:
            index = cursors[endpoint]
            if index >= len(saved[endpoint]): raise ValueError('Replay attempted an additional POST')
            record = saved[endpoint][index]
            if canonical_digest(record['request']) != canonical_digest(value):
                raise ValueError('Saved request differs from the current qualification case')
            cursors[endpoint] += 1
            return record['response']
        return real_fetch(url, value)
    q.fetch = fetch
    identity = uuid.uuid4().hex
    report = {'schema': 'character-creation.kv-qualification.v1', 'id': identity,
              'passed': False, 'started': time.time(), 'driver_sha256': file_sha(q.__file__),
              'resume_driver_sha256': file_sha(__file__),
              'previous_report': {'path': str(args.previous_report.resolve()), 'sha256': file_sha(args.previous_report)},
              'execution': {e: 'saved_response_revalidation_not_independent_replication' if e in saved else 'fresh_requests_after_readiness' for e in args.endpoint},
              'attempt_files': [str(args.output / ('kv-attempt-' + identity + '-' + e.rsplit(':', 1)[-1] + '.jsonl')) for e in args.endpoint]}
    for endpoint in saved:
        path = args.output / ('kv-attempt-' + identity + '-' + endpoint.rsplit(':', 1)[-1] + '.jsonl')
        path.write_text(json.dumps({'state': 'source_replay', 'source_report': report['previous_report'],
                                   'source_attempts': str(files[endpoint.rsplit(':', 1)[-1]]),
                                   'responses': len(saved[endpoint]), 'independent_replication': False}) + '\n')
    try:
        with ThreadPoolExecutor(max_workers=len(args.endpoint)) as pool:
            results = list(pool.map(lambda e: q.qualify_endpoint(e, args.output, identity), args.endpoint))
        binding = results[0]['binding']
        if any(r['binding'] != binding for r in results): raise ValueError('Worker bindings differ')
        if any(cursors[e] != len(saved[e]) for e in saved): raise ValueError('Saved cases were not exhausted')
        report.update(passed=True, binding=binding, results=results,
                      artifact_sha256s=[a['sha256'] for a in results[0]['artifacts']])
    except Exception as error:
        report['error'] = repr(error); raise
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
