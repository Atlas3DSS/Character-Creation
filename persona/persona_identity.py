"""The neutral identity anchor and its read-only, source-bound answer viewer."""
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from persona_vectors import canonical_digest


def load_probe(root):
    path = Path(root) / 'data/identity_probe_questions.json'
    if not path.exists():
        return None
    row = json.loads(path.read_bytes()).get('baseline_probe')
    if row is None:
        return None
    if (row.get('schema') != 'character-creation.manager-corpus.v1'
            or row.get('prompt') != 'who are you' or row.get('system') != ''
            or row.get('thinking') is not False or row.get('split') != 'discovery'
            or row.get('corpus') != 'persona_identity'
            or row.get('category') != 'identity_anchor'
            or row.get('source', {}).get('kind') != 'synthetic_original'
            or row['source'].get('prompt_sha256') != hashlib.sha256(b'who are you').hexdigest()
            or row.get('id') != canonical_digest({k: v for k, v in row.items() if k != 'id'})):
        raise ValueError('Standard identity probe text, neutral context or source binding differs')
    return row


def evidence(root, model, group=''):
    root = Path(root); probe = load_probe(root)
    result = {'status': 'pending', 'prompt': 'who are you', 'groups': [],
              'scope': 'Saved neutral self-description comparisons. Experimental behavior, not verified identity or preset qualification.'}
    path = root / 'results/persona_adaptive_public.json'
    if probe is None or not path.exists():
        return result
    if path.stat().st_size > 128 * 1024**2:
        raise ValueError('Identity evidence exceeds its bounded report size')
    value = json.loads(path.read_bytes()); menu = value['menu']
    if (value.get('schema') != 'character-creation.adaptive-public.v1'
            or value.get('revision_sha256') != canonical_digest({k: v for k, v in value.items() if k != 'revision_sha256'})
            or value.get('menu_projection_sha256') != canonical_digest(menu)
            or value['binding']['model_identity'] != model.identity
            or value['source_hashes']['identity_fixture'] != hashlib.sha256((root / 'data/identity_probe_questions.json').read_bytes()).hexdigest()):
        raise ValueError('Identity evidence source, model or report hash differs')
    original = json.loads((root / 'results/persona_adaptive_menu.json').read_bytes())
    if (original.get('menu_sha256') != canonical_digest({k: v for k, v in original.items() if k != 'menu_sha256'})
            or original['menu_sha256'] != value['menu_sha256']
            or {k: v for k, v in original.items() if k != 'excluded_gpu'} != menu):
        raise ValueError('Identity evidence menu differs from its frozen study')
    if value['binding'] != original['binding']:
        raise ValueError('Identity evidence runtime binding differs from frozen menu')
    candidates = {c['id']: c for c in menu['candidates']}; groups = {}
    identity_rows = [r for r in value['rows'] if r.get('identity_probe')]
    if len(identity_rows) > 4096:
        raise ValueError('Identity row bound exceeded')
    jobs = {}
    if identity_rows:
        database = root / 'results/persona_adaptive.sqlite'
        with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as db:
            for row in identity_rows:
                jid = row.get('job_id')
                saved = db.execute('SELECT body FROM jobs WHERE id=?', (jid,)).fetchone()
                if not saved or jid in jobs:
                    raise ValueError('Missing or duplicate authoritative identity job')
                job = json.loads(saved[0])
                if (job.get('id') != jid or canonical_digest({k:v for k,v in job.items() if k != 'id'}) != jid
                        or job.get('menu_sha256') != original['menu_sha256']
                        or job.get('model_identity') != model.identity):
                    raise ValueError('Frozen identity job content or binding differs')
                jobs[jid] = job
    for row in value['rows']:
        if not row.get('identity_probe'):
            continue
        if (row.get('source_id') != probe['id'] or row.get('prompt') != probe['prompt']
                or row.get('source_sha256') != value['source_hashes']['identity_fixture']
                or row.get('candidate_id') not in candidates or row.get('arm') not in ('baseline', 'candidate', 'reversal', 'random')):
            raise ValueError('Identity answer belongs to another prompt or candidate')
        job = jobs[row['job_id']]
        for field in ('candidate_id', 'arm', 'seed', 'phase', 'pushes', 'program_sha256', 'pair_id', 'identity_probe'):
            if row.get(field) != job.get(field) or field not in job:
                raise ValueError('Identity scientific job field differs: ' + field)
        prompt = job['prompt']
        if (prompt.get('prompt') != row['prompt'] or prompt.get('source_row_id') != row['source_id']
                or prompt.get('split') != 'discovery' or prompt.get('system') != ''
                or row.get('split') != 'discovery'
                or row.get('prompt_sha256') != hashlib.sha256(row['prompt'].encode()).hexdigest()):
            raise ValueError('Identity job prompt or source differs')
        candidate = candidates[row['candidate_id']]
        if row['arm'] == 'candidate' and row['program_sha256'] != candidate['program_sha256']:
            raise ValueError('Identity candidate program changed')
        answer = row.get('answer')
        if (type(row.get('complete')) is not bool or type(row.get('capture_validated')) is not bool
                or row.get('state') not in ('complete', 'incomplete', 'failed')
                or (row['state'] == 'complete') != row['complete']
                or row['complete'] and (not row['capture_validated'] or not answer)
                or answer is not None and (not isinstance(answer, str) or len(answer) > 100000
                    or not row['capture_validated'] or row['state'] not in ('complete', 'incomplete'))
                or answer is None and row['capture_validated']):
            raise ValueError('Identity answer state or capture validation differs')
        if answer is not None:
            capture = row.get('capture', {})
            if set(capture) != {'activation_artifact', 'provenance_artifact'}:
                raise ValueError('Identity answer capture references missing')
            for pointer in capture.values():
                if (pointer.get('request_sha256') != row.get('request_sha256')
                        or any(pointer.get(k) != v for k,v in value['binding'].items())
                        or not isinstance(pointer.get('sha256'), str) or len(pointer['sha256']) != 64):
                    raise ValueError('Identity answer capture binding differs')
        if answer is not None and hashlib.sha256(answer.encode()).hexdigest() != row.get('answer_sha256'):
            raise ValueError('Identity answer bytes changed')
        identity = canonical_digest([row['candidate_id'], row['seed']])
        bundle = groups.setdefault(identity, {'id': identity, 'candidate': candidate, 'seed': row['seed'], 'arms': {}})
        if row['arm'] in bundle['arms']:
            raise ValueError('Duplicate identity comparison arm')
        bundle['arms'][row['arm']] = row
    ordered = sorted(groups.values(), key=lambda g: (g['candidate']['kind'] != 'exact_distributed', g['candidate']['target'], g['candidate']['id'], g['seed']))
    result['groups'] = [{'id': g['id'], 'candidate_id': g['candidate']['id'], 'target': g['candidate']['target'],
                         'kind': g['candidate']['kind'], 'seed': g['seed'], 'complete': sum(r.get('complete') is True for r in g['arms'].values())} for g in ordered]
    if not ordered:
        return result
    selected = groups.get(group) if group else next((g for g in ordered if all(a in g['arms'] and g['arms'][a].get('complete') for a in ('baseline', 'candidate'))), ordered[0])
    if selected is None:
        raise ValueError('Unknown identity comparison group')
    return {**result, 'status': 'available', 'group_id': selected['id'], 'candidate': selected['candidate'],
            'seed': selected['seed'], 'binding': value['binding'],
            'cells': [selected['arms'].get(arm, {'arm': arm, 'state': 'pending', 'answer': None}) for arm in ('baseline', 'candidate', 'reversal', 'random')]}
