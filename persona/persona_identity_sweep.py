"""Single-neuron "who are you" sweep: queue, run against the native batch workers, export.

Each queued neuron is pushed up and down (the down-projection input convention
used everywhere else). Like the reference sweep, each direction starts strong
and retries weaker while the reply collapses: dose 1.0, 0.5, 0.25, then 0.125.
The strongest coherent reply is the neuron's answer. One shared unsteered
baseline per sweep. Greedy, 160 tokens, thinking off, no response capture. Runs on the GPU host
beside the workers; the explorer queues batches and reads the exported answers.
A submitted batch whose reply is lost is marked uncertain and never replayed.

Baseline pairs (layer -3, owner 2026-10-27) are unsteered answers to the 40,072 generated Big Five prompts: user message
only, greedy, seed 0, thinking off, up to 2,048 tokens, text only (their prefill captures already exist). Workers
alternate identity and baseline batches so both progress; baseline rows have their own export and status counts.

Combos (layer -2) push 2-4 neurons together. Each member carries its own strongest coherent single-push dose (signed);
the combo scales all members by a ramp fraction that goes UP, 1/8 -> 1/4 -> 1/2, and stops at the first collapse.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sqlite3
import subprocess
import threading
import time
import urllib.request
import persona_threshold as threshold

PROMPT = 'who are you'
LEVELS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)  # cascade; the 27B collapses near 0.3 at L0, so ±1 alone is mostly empty
RAMP = (0.125, 0.25, 0.5)  # combos: fraction of each member's own coherent dose, weakest first
COMBO = -2                  # layer marker of a combo job; its members are in `pushes`
BASE = -3                   # layer marker of an unsteered baseline pair; its user message is in `prompt`
BASE_SOURCE, BASE_SWEEP, BASE_TOKENS = 'data/persona_generated_big_five.jsonl', 'baseline40k', 2048
HELD = -4                   # held-back job: owner-private prompts (message list in `prompt`); first in the queue,
HELD_SOURCE, HELD_SWEEP, HELD_TOKENS = 'private/held_back_prompts.jsonl', 'held-back', 512  # excluded from every shared export
PUBLIC = 'layer>-3'         # identity work that may be exported (singles, baseline, combos)
MAX_TOKENS, SEED, BATCH = 160, 0, 4
OUT = 'results/identity_sweep'


def connect(root):
    (Path(root) / OUT).mkdir(exist_ok=True)
    db = sqlite3.connect(Path(root) / OUT / 'sweep.sqlite', timeout=60)
    db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA busy_timeout=60000')
    db.executescript('''CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, sweep TEXT, layer INT, neuron INT, dose REAL,
        state TEXT, created REAL, started REAL, finished REAL, port INT, answer TEXT, finish TEXT, tokens INT,
        artifact TEXT, artifact_sha256 TEXT, program_sha256 TEXT, error TEXT);
        CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state, created);''')
    columns = {row[1] for row in db.execute('PRAGMA table_info(jobs)')}
    for name, kind in (('level', 'INT'), ('coherent', 'INT'), ('pushes', 'TEXT'), ('kind', 'TEXT'), ('declined', 'INT'), ('prompt', 'TEXT'), ('source_id', 'TEXT')):
        if name not in columns: db.execute(f'ALTER TABLE jobs ADD COLUMN {name} {kind}')
    return db


def coherent(answer, tokens):
    """Cheap collapse check before any judge: empty, a few tokens, special tokens, or a loop."""
    text = (answer or '').strip()
    if tokens is None or tokens < 4 or len(text) < 8 or '<|im_' in text: return False
    if re.search(r'(.{1,8}?)\1{5,}', text): return False  # character loops such as "2.2.2.2" or "fit fit fit"
    words = text.split()
    return len(words) >= 3 and len(set(words)) / len(words) >= (0.5 if len(words) < 24 else 0.3)


DECLINE = re.compile(r"^\W*no\b(?!\s+(?:one|body|thing|longer|matter))|"
                     r"\b(?:can(?:'|no)?t|cannot|won'?t|will not|refuse to|unable to|not able to|(?:would|'d) rather not|"
                     r"do(?:n'?t| not) want to|not (?:comfortable|allowed|permitted) to)\W+(?:\w+\W+){0,3}?"
                     r"(?:help|answer|assist|tell|say|share|respond|reply|provide|disclose|discuss|reveal|comply|do that|engage)\b|"
                     r"\bnone of your business\b|\bI (?:must |have to )?decline\b", re.I)


def declined(answer):
    """Owner 2026-10-27: any "no" is a site to investigate. A refusal to answer/help/tell, or a reply that opens with No;
    not every "can't" ("I can't have a body" is a statement, not a decline)."""
    return bool(DECLINE.search(answer or ''))


def job_id(layer, neuron, dose):
    return hashlib.sha256(json.dumps([PROMPT, SEED, MAX_TOKENS, layer, neuron, dose]).encode()).hexdigest()[:32]


def combo_id(pushes, fraction):
    return hashlib.sha256(json.dumps([PROMPT, SEED, MAX_TOKENS, 'combo', pushes, fraction]).encode()).hexdigest()[:32]


def kept_levels(db):
    """Strongest coherent cascade level per (layer, neuron, sign) observed so far."""
    rows = db.execute("SELECT layer,neuron,dose>0,MIN(level) FROM jobs WHERE state='done' AND coherent=1 AND level IS NOT NULL AND layer>=0 GROUP BY 1,2,3").fetchall()
    by_layer = {}
    for layer, _, _, level in rows: by_layer.setdefault(layer, []).append(level)
    return by_layer


def start_level(by_layer, layer):
    """Meso-optimizer: begin one level stronger than the typical coherent level of nearby layers."""
    near = [v for l in range(layer - 2, layer + 3) for v in by_layer.get(l, [])]
    if len(near) < 20: return 0
    return max(0, sorted(near)[len(near) // 2] - 1)


def enqueue(root, sweep, neurons, db=None):
    """Adds each neuron's first up and down push once; the unsteered baseline is layer -1."""
    db = db or connect(root)
    if threshold.blocked_menu(db, sweep): return 0
    learned = kept_levels(db)
    rows = [(-1, -1, 0.0, None)]
    for l, n in neurons:
        level = start_level(learned, int(l))
        rows += [(int(l), int(n), sign * LEVELS[level], level) for sign in (1, -1)]
    with db:
        before = db.total_changes
        db.executemany("INSERT OR IGNORE INTO jobs(id,sweep,layer,neuron,dose,level,state,created) VALUES(?,?,?,?,?,?,'queued',?)",
                       [(job_id(l, n, d), sweep, l, n, d, level, time.time()) for l, n, d, level in rows])
        return db.total_changes - before


def enqueue_combos(root, sweep, combos, db=None):
    """Adds each combo's weakest ramp step once. combos: [{kind, label, members: [[layer, neuron, signed unit dose]]}]."""
    db = db or connect(root)
    rows = [(job_id(-1, -1, 0.0), sweep, -1, -1, 0.0, None, None, None)]
    for c in combos:
        pushes = [[int(m[0]), int(m[1]), round(float(m[2]), 6)] + ([1] if len(m) > 3 and m[3] else []) for m in c['members']]
        if not 2 <= len(pushes) <= 4 or any(abs(m[2]) > 4 for m in pushes): raise ValueError(f'bad combo {c.get("label")}')
        rows.append((combo_id(pushes, RAMP[0]), sweep, COMBO, -1, RAMP[0], 0, json.dumps(pushes), f"{c['kind']}:{c['label']}"))
    with db:
        before = db.total_changes
        db.executemany("INSERT OR IGNORE INTO jobs(id,sweep,layer,neuron,dose,level,pushes,kind,state,created) VALUES(?,?,?,?,?,?,?,?,'queued',?)",
                       [(*r, time.time()) for r in rows])
        return db.total_changes - before


def enqueue_baseline(root, db=None):
    """Adds one unsteered job per generated Big Five prompt (idempotent)."""
    db = db or connect(root); rows = []
    for line in (Path(root) / BASE_SOURCE).open():
        r = json.loads(line)
        if r.get('system') or not r.get('prompt'): continue
        identity = hashlib.sha256(json.dumps([BASE_SWEEP, r['id'], SEED, BASE_TOKENS]).encode()).hexdigest()[:32]
        rows.append((identity, BASE_SWEEP, BASE, -1, 0.0, r['prompt'], r['id'], time.time()))
    with db:
        before = db.total_changes
        db.executemany("INSERT OR IGNORE INTO jobs(id,sweep,layer,neuron,dose,prompt,source_id,state,created) VALUES(?,?,?,?,?,?,?,'queued',?)", rows)
        return db.total_changes - before


def enqueue_held(root, db=None):
    """Adds the owner-private prompt set first in the queue (idempotent). Its rows never reach a shared export."""
    db = db or connect(root); rows = []
    for line in (Path(root) / HELD_SOURCE).open():
        r = json.loads(line)
        identity = hashlib.sha256(json.dumps([HELD_SWEEP, r['id'], SEED, HELD_TOKENS]).encode()).hexdigest()[:32]
        rows.append((identity, HELD_SWEEP, HELD, -1, 0.0, json.dumps(r['messages'], ensure_ascii=False), r['id'],
                     f"{r['facet']}|{r['condition']}|{r['pair_id']}", time.time()))
    with db:
        before = db.total_changes
        db.executemany("INSERT OR IGNORE INTO jobs(id,sweep,layer,neuron,dose,prompt,source_id,kind,state,created) VALUES(?,?,?,?,?,?,?,?,'queued',?)", rows)
        return db.total_changes - before


def export_held(root):
    """private/held_back_answers.jsonl beside the private prompts: answers and capture paths, never shared."""
    path = Path(root) / 'private/held_back_answers.jsonl'; temporary = path.with_suffix('.tmp')
    with connect(root) as db, temporary.open('w') as stream:
        for row in db.execute("SELECT source_id,kind,answer,finish,tokens,coherent,declined,artifact,artifact_sha256,finished FROM jobs WHERE state='done' AND layer=-4 ORDER BY created,id"):
            stream.write(json.dumps(dict(zip(('id', 'label', 'answer', 'finish_reason', 'completion_tokens', 'coherent_auto', 'declined_auto', 'artifact', 'artifact_sha256', 'finished'), row)), ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def fetch(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    request = urllib.request.Request(url, data, {'Content-Type': 'application/json'} if data else {})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.loads(reply.read())



def lace_has_priority(root, port):
    """Read-only, active-campaign gate; unreadable/malformed lace state fails closed."""
    path = Path(root) / 'monkey/lace_sweep.sqlite'
    if not path.exists():
        return False
    try:
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=5) as lace:
            record = lace.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
            if record is None:
                raise ValueError('Missing active lace manifest')
            campaign = json.loads(record[0])['campaign_id']
            if not isinstance(campaign, str) or not campaign:
                raise ValueError('Invalid active lace campaign')
            queued = lace.execute("SELECT 1 FROM jobs WHERE campaign_id=? AND state IN ('queued','preparing') LIMIT 1", (campaign,)).fetchone()
            running = lace.execute("SELECT 1 FROM jobs WHERE campaign_id=? AND state='submitted' AND endpoint=? LIMIT 1",
                                   (campaign, f'http://127.0.0.1:{port}')).fetchone()
            return bool(queued or running)
    except (sqlite3.Error, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return True

def claim(root, port, baseline=False):
    """One batch of one kind: baseline pairs when asked (else identity work), falling back to the other kind."""
    db = connect(root)
    try:
        db.execute('BEGIN IMMEDIATE')
        if lace_has_priority(root, port):
            db.rollback()
            return []
        ordinary = "SELECT id,layer,neuron,dose,pushes,prompt FROM jobs WHERE state='queued' AND layer!=-3 AND campaign_id IS NULL"
        if threshold.status(db)['pause_ladder_menus']:
            ordinary += " AND NOT (coalesce(sweep,'') GLOB 'lens-*' OR coalesce(sweep,'') GLOB 'word-*')"
        ordinary += " ORDER BY layer=-4 DESC,layer=-1 DESC,level DESC,created,id LIMIT ?"
        base_query = "SELECT id,layer,neuron,dose,pushes,prompt FROM jobs WHERE state='queued' AND layer=-3 ORDER BY created,id LIMIT ?"
        edge_query = """SELECT j.id,j.layer,j.neuron,j.dose,j.pushes,j.prompt FROM jobs j
          JOIN threshold_campaigns c ON c.id=j.campaign_id
          LEFT JOIN threshold_searches s ON s.id=j.search_id
          WHERE j.state='queued' AND c.state IN ('baseline','searching')
          AND ? BETWEEN 8931 AND 8935 AND (j.target_port IS NULL OR j.target_port=?)
          ORDER BY j.probe_role='baseline' DESC,coalesce(s.priority,-1),j.created,j.id LIMIT ?"""
        held = "SELECT id,layer,neuron,dose,pushes,prompt FROM jobs WHERE state='queued' AND layer=-4 ORDER BY created,id LIMIT ?"
        queries = [(held,(BATCH,))]
        # Recheck immediately before the identity DB claim. Never hold either DB lock across HTTP.
        edge = [(edge_query,(port,port,BATCH))]
        queries += ([(base_query,(BATCH,))] + edge + [(ordinary,(BATCH,))] if baseline
                    else edge + [(ordinary,(BATCH,)),(base_query,(BATCH,))])
        for query,params in queries:
            rows = db.execute(query,params).fetchall()
            if rows: break
        db.executemany("UPDATE jobs SET state='running',started=?,port=? WHERE id=?", [(time.time(), port, r[0]) for r in rows])
        db.commit(); return rows
    finally: db.close()


def finish(root, updates):
    with connect(root) as db:
        db.executemany('UPDATE jobs SET state=?,finished=?,answer=?,finish=?,tokens=?,artifact=?,artifact_sha256=?,program_sha256=?,error=?,binding_json=? WHERE id=?', updates)
        for state, _, answer, _, tokens, *_, identity in updates:
            if threshold.finish(db, identity, coherent): continue
            if state != 'done': continue
            sweep, layer, neuron, dose, level, pushes, kind = db.execute('SELECT sweep,layer,neuron,dose,level,pushes,kind FROM jobs WHERE id=?', (identity,)).fetchone()
            ok = coherent(answer, tokens); db.execute('UPDATE jobs SET coherent=?, declined=? WHERE id=?', (int(ok), int(declined(answer)), identity))
            if layer == COMBO:  # combos ramp up while coherent and stop at the first collapse
                # decline nudges also stop as soon as the "no" is gone: that step is the finding
                if ok and level + 1 < len(RAMP) and not (str(kind).startswith('decline-') and not declined(answer)):
                    stronger = RAMP[level + 1]
                    db.execute("INSERT OR IGNORE INTO jobs(id,sweep,layer,neuron,dose,level,pushes,kind,state,created) VALUES(?,?,?,?,?,?,?,?,'queued',?)",
                               (combo_id(json.loads(pushes), stronger), sweep, COMBO, -1, stronger, level + 1, pushes, kind, time.time()))
                continue
            if not ok and level is not None and level + 1 < len(LEVELS):  # collapsed: retry this direction weaker
                weaker = (1 if dose > 0 else -1) * LEVELS[level + 1]
                db.execute("INSERT OR IGNORE INTO jobs(id,sweep,layer,neuron,dose,level,state,created) VALUES(?,?,?,?,?,?,'queued',?)",
                           (job_id(layer, neuron, weaker), sweep, layer, neuron, weaker, level + 1, time.time()))



def verify_response(root, row, request, result, descriptor):
    """Fail closed on response order; verify threshold artifacts before committing."""
    from persona_vectors import canonical_digest
    identity = row[0]
    request_hash = canonical_digest(request)
    if not isinstance(result, dict) or result.get('request_sha256') != request_hash:
        raise ValueError('Completion response request hash/order differs')
    response = result['response']
    if response.get('model_identity') != descriptor['flare']['model_identity']:
        raise ValueError('Completion model differs')
    choice = response['choices'][0]
    if not isinstance(choice['message']['content'], str) or choice['finish_reason'] not in ('stop', 'length'):
        raise ValueError('Malformed completion response')
    with connect(root) as db:
        campaign = db.execute('SELECT campaign_id FROM jobs WHERE id=?', (identity,)).fetchone()
    if not campaign or campaign[0] is None:
        return response
    import numpy as np
    artifact = response['activation_artifact']; provenance = response['provenance_artifact']
    path, proof = Path(artifact['path']), Path(provenance['path'])
    for item, filename in ((artifact,path),(provenance,proof)):
        if item.get('request_sha256') != request_hash or not filename.is_file():
            raise ValueError('Threshold evidence request/path differs')
        with filename.open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != item.get('sha256'):
                raise ValueError('Threshold evidence hash differs')
    report = json.loads(proof.read_text())
    binding = descriptor['flare']['batch']['binding']
    if (report.get('status') != 'complete' or report.get('state_restored') is not True
            or report.get('request_sha256') != request_hash or report.get('request') != request
            or any(response.get(k) != v or report.get(k) != v for k,v in binding.items())
            or report.get('activation_artifact') != artifact
            or report.get('response',{}).get('choices') != response['choices']
            or report.get('response',{}).get('usage') != response['usage']):
        raise ValueError('Threshold response, binding or restoration differs')
    with np.load(path, allow_pickle=False) as z:
        if (any(str(z[k].item()) != v for k,v in binding.items())
                or str(z['request_sha256'].item()) != request_hash
                or len(z['generated_token_ids']) != response['usage']['completion_tokens']):
            raise ValueError('Threshold NPZ binding or token count differs')
    return response

def worker(root, port, bank, until, stop, log):
    from persona_vectors import canonical_digest
    base = f'http://127.0.0.1:{port}'
    turn = port % 2  # workers alternate identity and baseline batches, half of them starting with each
    while not stop.is_set() and time.time() < until:
        try:
            descriptor = fetch(base + '/v1/models', timeout=10)['data'][0]
            capability = descriptor['flare'].get('batch', {})
            if not (capability.get('supported') and capability.get('qualified') and capability.get('max_batch_size') == BATCH):
                raise ValueError('Native batch capability not qualified')
        except Exception as exc:
            log(port, 'waiting', repr(exc)[:200]); stop.wait(30); continue
        turn ^= 1
        rows = claim(root, port, baseline=bool(turn))
        if not rows:
            stop.wait(15); continue
        requests, programs = [], []
        for identity, layer, neuron, dose, pushes, prompt in rows:
            messages = json.loads(prompt) if layer == HELD else [{'role': 'user', 'content': prompt if layer == BASE else PROMPT}]
            request = {'model': descriptor['id'], 'messages': messages, 'seed': SEED,
                       'max_tokens': BASE_TOKENS if layer == BASE else HELD_TOKENS if layer == HELD else MAX_TOKENS, 'temperature': 0, 'stream': False,
                       'chat_template_kwargs': {'enable_thinking': False}, 'metadata': {'job_id': identity, **({'held_back': True} if layer == HELD else {})}}
            program = None
            members = ([{'kind': 'neuron', 'layer': m[0], 'neuron': m[1], 'dose': m[2] if m[3:] else m[2] * dose} for m in json.loads(pushes)] if layer == COMBO
                       else [{'kind': 'neuron', 'layer': layer, 'neuron': neuron, 'dose': dose}] if layer >= 0 else None)
            if members:
                program, _ = bank.program(members, 'all')
                program = {'phase': program['phase'], 'sites': [{**s, 'vector': [float(x) for x in s['vector']]} for s in program['sites']]}
                request['steering'] = program
            requests.append(request); programs.append(canonical_digest(program) if program else None)
        envelope = {'batch_id': os.urandom(16).hex(), 'requests': requests}
        try:
            with connect(root) as db:
                for row,request in zip(rows,requests):
                    threshold.request_record(db,row[0],descriptor,port,request)
            preflight = fetch(base + '/v1/batch/preflight', envelope, timeout=120)
            if (preflight.get('fits') is not True or preflight.get('batch_id') != envelope['batch_id']
                    or preflight.get('binding') != descriptor['flare']['batch']['binding']
                    or preflight.get('request_sha256') != [canonical_digest(r) for r in requests]):
                raise ValueError('Batch preflight coverage, binding or capacity differs')
        except threshold.IntegrityError as exc:
            with connect(root) as db:
                threshold.block_integrity(db,[r[0] for r in rows],str(exc))
            log(port,'threshold_integrity_blocked',str(exc)); stop.wait(10); continue
        except Exception as exc:  # nothing was submitted, so the claim is released
            with connect(root) as db:
                db.executemany("UPDATE jobs SET state='queued',started=NULL,port=NULL WHERE id=?", [(r[0],) for r in rows])
            log(port, 'preflight_released', repr(exc)[:200]); stop.wait(10); continue
        try:
            reply = fetch(base + '/v1/batch/completions', envelope, timeout=3600 if rows[0][1] == BASE else 900)
            results = reply['results']
            if reply.get('batch_id') != envelope['batch_id'] or len(results) != len(rows): raise ValueError('Batch coverage differs')
            checked = [verify_response(root, row, request, result, descriptor)
                       for row, request, result in zip(rows, requests, results)]
        except Exception as exc:
            finish(root, [('uncertain', time.time(), None, None, None, None, None, p, repr(exc)[:500], None, r[0]) for r, p in zip(rows, programs)])
            log(port, 'uncertain_no_retry', repr(exc)[:200]); continue
        updates = []
        for (identity, *_), program, response in zip(rows, programs, checked):
            choice = response['choices'][0]; artifact = response.get('activation_artifact') or {}
            updates.append(('done', time.time(), choice['message']['content'], choice.get('finish_reason'),
                            response['usage']['completion_tokens'], artifact.get('path'), artifact.get('sha256'), program, None, json.dumps({k:response.get(k) for k in ('model_identity','runtime_identity','prepared_identity')},sort_keys=True,separators=(',',':')), identity))
        finish(root, updates)


def export(root):
    """Atomic JSONL of finished answers for the explorer to copy."""
    path = Path(root) / OUT / 'answers.jsonl'; temporary = path.with_suffix('.tmp')
    with connect(root) as db, temporary.open('w') as stream:
        for row in db.execute("SELECT sweep,layer,neuron,dose,level,coherent,answer,finish,tokens,finished,artifact_sha256,kind,pushes,declined FROM jobs WHERE state='done' AND layer>-3 AND campaign_id IS NULL ORDER BY layer,neuron,dose"):
            value = dict(zip(('sweep', 'layer', 'neuron', 'dose', 'level', 'coherent', 'answer', 'finish', 'tokens', 'finished', 'artifact_sha256', 'kind', 'pushes', 'declined'), row))
            value['declined'] = bool(value['declined'])
            if value['pushes'] is None:
                del value['pushes']
                if not str(value['kind']).startswith('threshold:'): del value['kind']
            else: value['pushes'] = json.loads(value['pushes'])
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')
    os.replace(temporary, path)
    flagged = Path(root) / OUT / 'declined.jsonl'; temporary = flagged.with_suffix('.tmp')
    with connect(root) as db, temporary.open('w') as stream:  # the investigation list: every reply that said no
        for row in db.execute("SELECT sweep,layer,neuron,dose,kind,pushes,answer,coherent FROM jobs WHERE state='done' AND declined=1 AND layer>-3 ORDER BY layer,neuron,dose"):
            value = dict(zip(('sweep', 'layer', 'neuron', 'dose', 'kind', 'pushes', 'answer', 'coherent'), row))
            if value['pushes']: value['pushes'] = json.loads(value['pushes'])
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')
    os.replace(temporary, flagged)
    export_threshold(root)
    export_baseline(root)
    if (Path(root) / HELD_SOURCE).exists(): export_held(root)
    return status(root)



def export_threshold(root):
    """Separate lossless local JSONL; legacy public/HF export cannot represent exact doses."""
    path=Path(root)/OUT/'threshold_answers.jsonl'
    temporary=path.with_suffix('.tmp')
    fields=('id','campaign_id','search_id','probe_role','layer','neuron','dose','dose_hex','kind',
            'answer','finish','tokens','coherent','artifact','artifact_sha256','program_sha256',
            'request_json','request_sha256','binding_json','metrics_json','port','finished')
    with connect(root) as db, temporary.open('w') as stream:
        for row in db.execute('SELECT '+','.join(fields)+" FROM jobs WHERE campaign_id IS NOT NULL AND state='done' ORDER BY finished,id"):
            stream.write(json.dumps(dict(zip(fields,row)),ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(temporary,path)

def export_baseline(root):
    """baseline40k_pairs.jsonl: every finished unsteered pair, at most every 10 minutes."""
    path = Path(root) / OUT / 'baseline40k_pairs.jsonl'
    if path.exists() and time.time() - path.stat().st_mtime < 600: return
    temporary = path.with_suffix('.tmp')
    with connect(root) as db, temporary.open('w') as stream:
        for row in db.execute("SELECT source_id,prompt,answer,finish,tokens,coherent,declined,finished FROM jobs WHERE state='done' AND layer=-3 ORDER BY created,id"):
            value = dict(zip(('id', 'prompt', 'answer', 'finish_reason', 'completion_tokens', 'coherent_auto', 'declined_auto', 'finished'), row))
            value.update(complete=value['finish_reason'] == 'stop', seed=SEED, max_tokens=BASE_TOKENS, thinking=False, steering=None)
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def backfill_flags(root):
    with connect(root) as db:
        rows = db.execute("SELECT id,answer FROM jobs WHERE state='done'").fetchall()
        db.executemany('UPDATE jobs SET declined=? WHERE id=?', [(int(declined(a)), i) for i, a in rows])
    return {'checked': len(rows)}


def status(root):
    with connect(root) as db:
        counts = dict(db.execute('SELECT state,count(*) FROM jobs WHERE layer>-3 AND campaign_id IS NULL GROUP BY state').fetchall())  # identity work only
        held = dict(db.execute('SELECT state,count(*) FROM jobs WHERE layer=-4 GROUP BY state').fetchall())
        baseline = dict(db.execute('SELECT state,count(*) FROM jobs WHERE layer=-3 GROUP BY state').fetchall())
        sweeps = {name: dict(db.execute('SELECT state,count(*) FROM jobs WHERE sweep=? GROUP BY state', (name,)).fetchall())
                  for name, in db.execute('SELECT DISTINCT sweep FROM jobs') if name not in (BASE_SWEEP, HELD_SWEEP)}
        flagged = db.execute("SELECT count(*) FROM jobs WHERE state='done' AND declined=1 AND layer>-3 AND campaign_id IS NULL").fetchone()[0]
        recent = db.execute("SELECT count(*),sum(tokens) FROM jobs WHERE state='done' AND finished>? AND layer>-3 AND campaign_id IS NULL", (time.time() - 600,)).fetchone()
        recent_base = db.execute("SELECT count(*),sum(tokens) FROM jobs WHERE state='done' AND finished>? AND layer=-3", (time.time() - 600,)).fetchone()
        threshold_state = threshold.status(db)
        learned = kept_levels(db)
    cliffs = {f'L{g * 8}-{g * 8 + 7}': LEVELS[sorted(v)[len(v) // 2]] for g in range(8)
              for v in [[x for l in range(g * 8, g * 8 + 8) for x in learned.get(l, [])]] if len(v) >= 10}
    value = {'time': time.time(), 'threshold': threshold_state, 'counts': counts, 'sweeps': sweeps, 'last_10_min': {'answers': recent[0], 'tokens': recent[1] or 0},
             'median_strongest_coherent_dose': cliffs, 'declined': flagged,
             'baseline40k': {'counts': baseline, 'last_10_min': {'answers': recent_base[0], 'tokens': recent_base[1] or 0}}, 'held_back': held}
    (Path(root) / OUT / 'status.json').write_text(json.dumps(value))
    return value


def run(root, until, ports, checkpoint):
    from persona_vectors import VectorBank, ModelSpec
    bank = VectorBank(Path(root) / 'results', ModelSpec.load(Path(root) / 'data/persona_model.json'), checkpoint=Path(checkpoint))
    stop = threading.Event(); lock = threading.Lock()
    def log(port, kind, detail):
        with lock, (Path(root) / OUT / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'time': time.time(), 'port': port, 'event': kind, 'detail': detail}) + '\n')
    with connect(root) as db:  # a crash leaves claimed rows; their outcome is unknown, never replayed
        interrupted = [r[0] for r in db.execute("SELECT id FROM jobs WHERE state='running' AND campaign_id IS NOT NULL")]
        db.execute("UPDATE jobs SET state='uncertain',error='Runner restarted while running' WHERE state='running'")
        for identity in interrupted: threshold.finish(db,identity,coherent)
    def audit_worker():
        # Thread-local idle IO priority protects the owner's front-of-queue lace work.
        tid=threading.get_native_id()
        try:
            os.setpriority(os.PRIO_PROCESS,tid,10)
            subprocess.run(['ionice','-c','3','-p',str(tid)],check=True,capture_output=True)
        except (OSError,subprocess.SubprocessError) as exc:
            log(None,'threshold_auditor_priority_failed',repr(exc)[:300])
            return
        audit_bank=VectorBank(Path(root)/'results',ModelSpec.load(Path(root)/'data/persona_model.json'),checkpoint=Path(checkpoint))
        descriptors=None;refreshed=0;completed=0
        while not stop.is_set() and time.time()<until:
            try:
                if descriptors is None or time.time()-refreshed>60:
                    descriptors={p:fetch(f'http://127.0.0.1:{p}/v1/models',timeout=10)['data'][0] for p in range(8931,8936)}
                    refreshed=time.time()
                with connect(root) as db:
                    value=threshold.audit_next(db,descriptors,audit_bank,coherent)
                if value is None:
                    stop.wait(5)
                else:
                    completed+=1
                    if completed<=4 or completed%16==0:
                        log(None,'threshold_audit_complete',value)
            except threshold.IntegrityError as exc:
                with connect(root) as db:
                    db.execute("UPDATE threshold_campaigns SET state='blocked',error=? WHERE state IN ('baseline','searching')",
                               ('audit_integrity: '+str(exc),))
                log(None,'threshold_audit_blocked',str(exc)[:300]);stop.wait(10)
            except Exception as exc:
                log(None,'threshold_audit_waiting',repr(exc)[:300]);descriptors=None;stop.wait(10)
    threads = [threading.Thread(target=worker, args=(root, p, bank, until, stop, log), daemon=True) for p in ports]
    audit_thread=threading.Thread(target=audit_worker,daemon=True)
    for t in threads: t.start()
    audit_thread.start()
    try:
        while time.time() < until and any(t.is_alive() for t in threads):
            export(root); stop.wait(60)
    finally:
        stop.set(); export(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('migrate-threshold', 'enqueue-threshold', 'enqueue', 'enqueue-combos', 'enqueue-baseline', 'enqueue-held-back', 'run', 'export', 'status', 'backfill-flags'))
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--sweep', default='manual'); parser.add_argument('--neurons', help='JSON file of [layer, neuron] pairs')
    parser.add_argument('--threshold-spec', help='Frozen JSON target list with priority, target_bits and evidence hashes')
    parser.add_argument('--combos', help='JSON file of combos (see enqueue_combos)')
    parser.add_argument('--until', type=float); parser.add_argument('--ports', default='8930-8935')
    parser.add_argument('--checkpoint', help='Local checkpoint directory (required for run)')
    args = parser.parse_args()
    if args.command == 'migrate-threshold':
        with connect(args.root) as db: threshold.schema(db)
        print(json.dumps({'threshold_schema': 'ready'}))
    elif args.command == 'enqueue-threshold':
        if not args.threshold_spec: raise SystemExit('--threshold-spec is required')
        descriptors = {p:fetch(f'http://127.0.0.1:{p}/v1/models',timeout=10)['data'][0] for p in range(8931,8936)}
        if not args.checkpoint: raise SystemExit('--checkpoint is required for provenance audit')
        from persona_vectors import VectorBank, ModelSpec
        bank=VectorBank(Path(args.root)/'results',ModelSpec.load(Path(args.root)/'data/persona_model.json'),checkpoint=Path(args.checkpoint))
        with connect(args.root) as db:
            print(json.dumps(threshold.enqueue(db,json.loads(Path(args.threshold_spec).read_text()),descriptors,bank=bank)))
    elif args.command == 'enqueue':
        print(json.dumps({'added': enqueue(args.root, args.sweep, json.loads(Path(args.neurons).read_text()))}))
    elif args.command == 'enqueue-combos':
        print(json.dumps({'added': enqueue_combos(args.root, args.sweep, json.loads(Path(args.combos).read_text()))}))
    elif args.command == 'run':
        low, high = map(int, args.ports.split('-'))
        if not args.until or args.until <= time.time(): raise SystemExit('--until must be a future epoch')
        if not args.checkpoint: raise SystemExit('--checkpoint is required')
        run(args.root, args.until, range(low, high + 1), args.checkpoint)
    elif args.command == 'export': print(json.dumps(export(args.root)))
    elif args.command == 'enqueue-baseline': print(json.dumps({'added': enqueue_baseline(args.root)}))
    elif args.command == 'enqueue-held-back': print(json.dumps({'added': enqueue_held(args.root)}))
    elif args.command == 'backfill-flags': print(json.dumps({**backfill_flags(args.root), **export(args.root)}))
    else: print(json.dumps(status(args.root)))


if __name__ == '__main__': main()
