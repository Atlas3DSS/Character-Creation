"""Finite publication of this authored preset study; no private corpus discovery."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import time
from urllib.request import urlopen

from persona_reasoning_sweep import atomic_json
from persona_vectors import canonical_digest

REPOSITORY = 'Atlas3D/character-creation-big-five'
PUBLIC = 'persona_preset_overnight_public.json'
TERMINAL = {'complete', 'incomplete', 'failed', 'interrupted', 'cancelled'}
DATABASES = {
    'primary_discovery': ('persona_preset_overnight.sqlite', 'persona_preset_discovery_continuation.sqlite'),
    'primary_confirmation': ('persona_preset_confirmation.sqlite',),
    'followon_discovery': ('persona_preset_followon_discovery.sqlite',),
    'followon_confirmation': ('persona_preset_followon_confirmation.sqlite',),
}


def digest(raw): return hashlib.sha256(raw).hexdigest()


def checked_plan(path):
    raw = path.read_bytes(); plan = json.loads(raw)
    if plan['input_sha256'] != canonical_digest({k: v for k, v in plan.items() if k != 'input_sha256'}):
        raise ValueError('Published study plan digest differs')
    return plan, digest(raw)


def snapshot(root):
    root = Path(root); results = root / 'results'; plans = {}; rows = []; evidence = {}
    for kind, stem in (('primary', 'persona_preset_overnight'), ('followon', 'persona_preset_followon')):
        path = results / (stem + '_plan.json')
        if not path.exists(): continue
        plan, sha = checked_plan(path)
        plans[kind] = {'sha256': sha, 'plan': plan}
        for suffix in ('selection', 'results'):
            name = ('persona_preset_selection.json' if kind == 'primary' and suffix == 'selection' else stem + '_' + suffix + '.json')
            p = results / name
            if p.exists():
                raw = p.read_bytes(); value = json.loads(raw)
                if value['input_sha256'] != plan['input_sha256']: raise ValueError('Published evaluation belongs to another plan')
                evidence[name] = {'sha256': digest(raw), 'record': value}
    seen = set()
    for phase, filenames in DATABASES.items():
        kind, stage = phase.split('_', 1)
        for filename in filenames:
            path = results / filename
            if not path.exists(): continue
            if kind not in plans: raise ValueError('Study database exists without its frozen plan')
            plan = plans[kind]['plan']; prompts = {c['id']: c for c in plan['discovery' if stage == 'discovery' else 'confirmation']}
            with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
                db.execute('BEGIN')
                jobs = db.execute('SELECT id,state,body FROM jobs ORDER BY id').fetchall()
                attempts = db.execute('SELECT job_id,number,result FROM attempts ORDER BY number').fetchall()
            latest = {identity: (number, raw) for identity, number, raw in attempts}
            for identity, state, body in jobs:
                job = json.loads(body)
                if ((kind, identity) in seen or job['id'] != identity or job['model_identity'] != plan['model_identity']
                        or job['prompt'] != prompts.get(job['prompt']['id'])):
                    raise ValueError('Published response source differs from the authored study')
                seen.add((kind, identity))
                number, raw = latest.get(identity, (0, None)); result = json.loads(raw) if raw else {}
                response = result.get('response') or {}; choices = response.get('choices') or []
                # Explicit scientific fields only. Never export deployment config, paths or request headers.
                item = {k: job.get(k) for k in ('id', 'recipe_id', 'target', 'arm', 'phase', 'seed', 'program_sha256', 'pushes')}
                item.update(study=kind, stage=stage, state=state, prompt=job['prompt'], attempt=number,
                    response_sha256=canonical_digest(response) if response else None,
                    answer=choices[0].get('message', {}).get('content') if choices else None,
                    finish_reason=choices[0].get('finish_reason') if choices else None,
                    usage=response.get('usage'), binding=result.get('binding'),
                    complete=result.get('complete') is True, capture_validated=result.get('capture_validated') is True,
                    has_error=bool(result.get('error')))
                for key in ('activation_artifact', 'provenance_artifact'):
                    artifact = result.get(key) or {}
                    item[key] = {k: artifact[k] for k in ('sha256', 'request_sha256', 'model_identity', 'runtime_identity', 'prepared_identity', 'fields', 'semantics') if k in artifact}
                rows.append(item)
    reviews = []
    for pattern in ('persona_preset_judge_*.json', 'persona_preset_confirm_judge_*.json', 'persona_preset_followon_judge_*.json'):
        for path in sorted(results.glob(pattern)):
            raw = path.read_bytes(); value = json.loads(raw)
            kind = 'followon' if 'followon' in path.name else 'primary'
            if value.get('input_sha256') != plans[kind]['plan']['input_sha256']:
                raise ValueError('Published judge source differs')
            if any((kind, identity) not in seen for identity in value.get('key', {}).values()):
                raise ValueError('Published review references an unknown answer')
            reviews.append({'sha256': digest(raw), 'study': kind, **{k: value.get(k) for k in
                ('state', 'protocol', 'input_sha256', 'source_sha256', 'packet_sha256', 'key', 'judgments', 'surface', 'judge_status')},
                'prior_attempt_count': len(value.get('prior_attempts', []))})
    counts = dict(Counter(r['state'] for r in rows))
    return {'schema': 'character-creation.preset-overnight-public.v1', 'plans': plans,
        'counts': counts, 'terminal_jobs': sum(counts.get(k, 0) for k in TERMINAL), 'rows': rows, 'evaluations': evidence, 'reviews': reviews,
        'scope': 'Original authored prompts, generated answers, exact recipe definitions and automatic blinded judgments. No human validation claim.',
        'raw_arrays': {'included': False, 'coverage': 'Capture artifact hashes are references to retained raw records, not published array coverage.'},
        'heldout_policy': 'Confirmation scenes are excluded from discovery selection and challenger construction.'}


def upload(path, timeout=180):
    command = ['/home/orwel/.local/bin/hf', 'upload', REPOSITORY, str(path), PUBLIC, '--type', 'dataset',
        '--format', 'json', '--commit-message', 'Publish authored preset sweep answers and evaluation progress']
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)
    def strings(value):
        if isinstance(value, str): yield value
        elif isinstance(value, dict):
            for child in value.values(): yield from strings(child)
        elif isinstance(value, list):
            for child in value: yield from strings(child)
    pattern = r'https://huggingface.co/datasets/' + re.escape(REPOSITORY) + r'/commit/([0-9a-f]{40})'
    matches = [re.fullmatch(pattern, s) for s in strings(json.loads(result.stdout))]
    revisions = {m.group(1) for m in matches if m}
    if len(revisions) != 1: raise ValueError('Upload returned no unique immutable commit')
    revision = revisions.pop()
    with urlopen(f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{PUBLIC}', timeout=60) as response:
        raw = response.read(path.stat().st_size + 1)
    if raw != path.read_bytes(): raise ValueError('Commit-pinned published bytes differ')
    return revision


def due(receipt, value, final):
    attempts = receipt.get('attempts', [])
    if any(a['state'] != 'verified' for a in attempts): return False
    previous = attempts[-1] if attempts else {}
    return (value['terminal_jobs'] // 48 > previous.get('terminal_jobs', 0) // 48
        or len(value.get('reviews', [])) // 8 > previous.get('review_groups', 0) // 8
        or canonical_digest(value['evaluations']) != previous.get('evaluation_sha256')
        or (final and not previous.get('final')))


def watch(root, until, publish=False):
    root = Path(root).resolve(); results = root / 'results'
    if not time.time() < until <= time.time() + 16 * 3600: raise ValueError('Finite publication deadline required')
    with (results / 'persona_preset_publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = results / 'persona_preset_publication.json'
        source = digest(Path(__file__).read_bytes())
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
            'schema': 'character-creation.preset-publication.v1', 'repository': REPOSITORY, 'source_sha256': source, 'attempts': []}
        if receipt['source_sha256'] != source: raise ValueError('Publisher source changed; explicit recovery required')
        while True:
            value = snapshot(root); final = time.time() >= until
            path = results / PUBLIC; atomic_json(path, value)
            if publish and due(receipt, value, final):
                attempt = {'state': 'intent', 'sha256': digest(path.read_bytes()), 'terminal_jobs': value['terminal_jobs'],
                    'review_groups': len(value['reviews']),
                    'evaluation_sha256': canonical_digest(value['evaluations']), 'final': final, 'started': time.time()}
                receipt['attempts'].append(attempt); atomic_json(receipt_path, receipt)
                try: attempt.update(revision=upload(path), state='verified')
                except Exception as error: attempt.update(state='uncertain_no_retry', error_type=type(error).__name__)
                attempt['finished'] = time.time(); atomic_json(receipt_path, receipt)
            print(json.dumps({'terminal': value['terminal_jobs'], 'counts': value['counts'], 'final': final}), flush=True)
            if final: return
            time.sleep(min(30, max(0, until - time.time())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--until', type=float, required=True); parser.add_argument('--publish', action='store_true')
    args = parser.parse_args(); watch(args.root, args.until, args.publish)
