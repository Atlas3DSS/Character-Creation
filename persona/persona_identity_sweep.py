"""Single-neuron "who are you" sweep: queue, run against the native batch workers, export.

Each queued neuron is pushed up and down (the down-projection input convention
used everywhere else). Like the reference sweep, each direction starts strong
and retries weaker while the reply collapses: dose 1.0, 0.5, 0.25, then 0.125.
The strongest coherent reply is the neuron's answer. One shared unsteered
baseline per sweep. Greedy, 160 tokens, thinking off, no response capture. Runs on the GPU host
beside the workers; the explorer queues batches and reads the exported answers.
A submitted batch whose reply is lost is marked uncertain and never replayed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sqlite3
import threading
import time
import urllib.request

PROMPT = 'who are you'
LEVELS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)  # cascade; the 27B collapses near 0.3 at L0, so ±1 alone is mostly empty
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
    for name, kind in (('level', 'INT'), ('coherent', 'INT')):
        if name not in columns: db.execute(f'ALTER TABLE jobs ADD COLUMN {name} {kind}')
    return db


def coherent(answer, tokens):
    """Cheap collapse check before any judge: empty, a few tokens, special tokens, or a loop."""
    text = (answer or '').strip()
    if tokens is None or tokens < 4 or len(text) < 8 or '<|im_' in text: return False
    if re.search(r'(.{1,8}?)\1{5,}', text): return False  # character loops such as "2.2.2.2" or "fit fit fit"
    words = text.split()
    return len(words) >= 3 and len(set(words)) / len(words) >= (0.5 if len(words) < 24 else 0.3)


def job_id(layer, neuron, dose):
    return hashlib.sha256(json.dumps([PROMPT, SEED, MAX_TOKENS, layer, neuron, dose]).encode()).hexdigest()[:32]


def kept_levels(db):
    """Strongest coherent cascade level per (layer, neuron, sign) observed so far."""
    rows = db.execute("SELECT layer,neuron,dose>0,MIN(level) FROM jobs WHERE state='done' AND coherent=1 AND level IS NOT NULL GROUP BY 1,2,3").fetchall()
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
    db = db or connect(root); learned = kept_levels(db)
    rows = [(-1, -1, 0.0, None)]
    for l, n in neurons:
        level = start_level(learned, int(l))
        rows += [(int(l), int(n), sign * LEVELS[level], level) for sign in (1, -1)]
    with db:
        before = db.total_changes
        db.executemany("INSERT OR IGNORE INTO jobs(id,sweep,layer,neuron,dose,level,state,created) VALUES(?,?,?,?,?,?,'queued',?)",
                       [(job_id(l, n, d), sweep, l, n, d, level, time.time()) for l, n, d, level in rows])
        return db.total_changes - before


def fetch(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    request = urllib.request.Request(url, data, {'Content-Type': 'application/json'} if data else {})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.loads(reply.read())


def claim(root, port):
    db = connect(root)
    try:
        db.execute('BEGIN IMMEDIATE')
        rows = db.execute("SELECT id,layer,neuron,dose FROM jobs WHERE state='queued' ORDER BY layer<0 DESC, level DESC, created, id LIMIT ?", (BATCH,)).fetchall()
        db.executemany("UPDATE jobs SET state='running',started=?,port=? WHERE id=?", [(time.time(), port, r[0]) for r in rows])
        db.commit(); return rows
    finally: db.close()


def finish(root, updates):
    with connect(root) as db:
        db.executemany('UPDATE jobs SET state=?,finished=?,answer=?,finish=?,tokens=?,artifact=?,artifact_sha256=?,program_sha256=?,error=? WHERE id=?', updates)
        for state, _, answer, _, tokens, *_, identity in updates:
            if state != 'done': continue
            sweep, layer, neuron, dose, level = db.execute('SELECT sweep,layer,neuron,dose,level FROM jobs WHERE id=?', (identity,)).fetchone()
            ok = coherent(answer, tokens); db.execute('UPDATE jobs SET coherent=? WHERE id=?', (int(ok), identity))
            if not ok and level is not None and level + 1 < len(LEVELS):  # collapsed: retry this direction weaker
                weaker = (1 if dose > 0 else -1) * LEVELS[level + 1]
                db.execute("INSERT OR IGNORE INTO jobs(id,sweep,layer,neuron,dose,level,state,created) VALUES(?,?,?,?,?,?,'queued',?)",
                           (job_id(layer, neuron, weaker), sweep, layer, neuron, weaker, level + 1, time.time()))


def worker(root, port, bank, until, stop, log):
    from persona_vectors import canonical_digest
    base = f'http://127.0.0.1:{port}'
    while not stop.is_set() and time.time() < until:
        try:
            descriptor = fetch(base + '/v1/models', timeout=10)['data'][0]
            capability = descriptor['flare'].get('batch', {})
            if not (capability.get('supported') and capability.get('qualified') and capability.get('max_batch_size') == BATCH):
                raise ValueError('Native batch capability not qualified')
        except Exception as exc:
            log(port, 'waiting', repr(exc)[:200]); stop.wait(30); continue
        rows = claim(root, port)
        if not rows:
            stop.wait(15); continue
        requests, programs = [], []
        for identity, layer, neuron, dose in rows:
            request = {'model': descriptor['id'], 'messages': [{'role': 'user', 'content': PROMPT}], 'seed': SEED,
                       'max_tokens': MAX_TOKENS, 'temperature': 0, 'stream': False,
                       'chat_template_kwargs': {'enable_thinking': False}, 'metadata': {'job_id': identity}}
            program = None
            if layer >= 0:
                program, _ = bank.program([{'kind': 'neuron', 'layer': layer, 'neuron': neuron, 'dose': dose}], 'all')
                program = {'phase': program['phase'], 'sites': [{**s, 'vector': [float(x) for x in s['vector']]} for s in program['sites']]}
                request['steering'] = program
            requests.append(request); programs.append(canonical_digest(program) if program else None)
        envelope = {'batch_id': os.urandom(16).hex(), 'requests': requests}
        try:
            preflight = fetch(base + '/v1/batch/preflight', envelope, timeout=120)
            if not preflight.get('fits'): raise ValueError('Batch does not fit: ' + str(preflight.get('reason')))
        except Exception as exc:  # nothing was submitted, so the claim is released
            with connect(root) as db:
                db.executemany("UPDATE jobs SET state='queued',started=NULL,port=NULL WHERE id=?", [(r[0],) for r in rows])
            log(port, 'preflight_released', repr(exc)[:200]); stop.wait(10); continue
        try:
            reply = fetch(base + '/v1/batch/completions', envelope, timeout=900)
            results = reply['results']
            if reply.get('batch_id') != envelope['batch_id'] or len(results) != len(rows): raise ValueError('Batch coverage differs')
        except Exception as exc:
            finish(root, [('uncertain', time.time(), None, None, None, None, None, p, repr(exc)[:500], r[0]) for r, p in zip(rows, programs)])
            log(port, 'uncertain_no_retry', repr(exc)[:200]); continue
        updates = []
        for (identity, *_), program, result in zip(rows, programs, results):
            response = result['response']; choice = response['choices'][0]; artifact = response.get('activation_artifact') or {}
            updates.append(('done', time.time(), choice['message']['content'], choice.get('finish_reason'),
                            response['usage']['completion_tokens'], artifact.get('path'), artifact.get('sha256'), program, None, identity))
        finish(root, updates)


def export(root):
    """Atomic JSONL of finished answers for the explorer to copy."""
    path = Path(root) / OUT / 'answers.jsonl'; temporary = path.with_suffix('.tmp')
    with connect(root) as db, temporary.open('w') as stream:
        for row in db.execute("SELECT sweep,layer,neuron,dose,level,coherent,answer,finish,tokens,finished,artifact_sha256 FROM jobs WHERE state='done' ORDER BY layer,neuron,dose"):
            stream.write(json.dumps(dict(zip(('sweep', 'layer', 'neuron', 'dose', 'level', 'coherent', 'answer', 'finish', 'tokens', 'finished', 'artifact_sha256'), row)), ensure_ascii=False) + '\n')
    os.replace(temporary, path)
    return status(root)


def status(root):
    with connect(root) as db:
        counts = dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state').fetchall())
        sweeps = {name: dict(db.execute('SELECT state,count(*) FROM jobs WHERE sweep=? GROUP BY state', (name,)).fetchall())
                  for name, in db.execute('SELECT DISTINCT sweep FROM jobs')}
        recent = db.execute("SELECT count(*),sum(tokens) FROM jobs WHERE state='done' AND finished>?", (time.time() - 600,)).fetchone()
        learned = kept_levels(db)
    cliffs = {f'L{g * 8}-{g * 8 + 7}': LEVELS[sorted(v)[len(v) // 2]] for g in range(8)
              for v in [[x for l in range(g * 8, g * 8 + 8) for x in learned.get(l, [])]] if len(v) >= 10}
    value = {'time': time.time(), 'counts': counts, 'sweeps': sweeps, 'last_10_min': {'answers': recent[0], 'tokens': recent[1] or 0},
             'median_strongest_coherent_dose': cliffs}
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
        db.execute("UPDATE jobs SET state='uncertain',error='Runner restarted while running' WHERE state='running'")
    threads = [threading.Thread(target=worker, args=(root, p, bank, until, stop, log), daemon=True) for p in ports]
    for t in threads: t.start()
    try:
        while time.time() < until and any(t.is_alive() for t in threads):
            export(root); stop.wait(60)
    finally:
        stop.set(); export(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('enqueue', 'run', 'export', 'status'))
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--sweep', default='manual'); parser.add_argument('--neurons', help='JSON file of [layer, neuron] pairs')
    parser.add_argument('--until', type=float); parser.add_argument('--ports', default='8930-8935')
    parser.add_argument('--checkpoint', help='Local checkpoint directory (required for run)')
    args = parser.parse_args()
    if args.command == 'enqueue':
        print(json.dumps({'added': enqueue(args.root, args.sweep, json.loads(Path(args.neurons).read_text()))}))
    elif args.command == 'run':
        low, high = map(int, args.ports.split('-'))
        if not args.until or args.until <= time.time(): raise SystemExit('--until must be a future epoch')
        if not args.checkpoint: raise SystemExit('--checkpoint is required')
        run(args.root, args.until, range(low, high + 1), args.checkpoint)
    elif args.command == 'export': print(json.dumps(export(args.root)))
    else: print(json.dumps(status(args.root)))


if __name__ == '__main__': main()
