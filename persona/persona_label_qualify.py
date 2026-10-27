"""Label qualification v2: larger held-out sets for every candidate concept.

Runs beside the native batch workers. For each concept it (1) writes a precise
definition (the six original definitions are kept), (2) authors short scenes in
which the definition clearly holds and hard negatives in which it does not, each
a distinct scenario family, (3) has every scene checked independently (yes/no,
greedy) and drops scenes whose check disagrees with the intended label, (4) splits
24 discovery and up to 64 held-out scenes per side, (5) captures the final-prefill
plane of each scene, then (6) selects a neuron and threshold on discovery only and
measures held-out precision with the registry's Wilson 95% > 0.70 gate.

The measurement is the final prompt token's SwiGLU product with the scene as the
only user message: a different measurement from v1's mean over all prompt
positions, and labelled as such. Scenes are model-authored and model-checked,
not human-annotated.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sqlite3
import threading
import time

from persona_identity_sweep import fetch

OUT = 'results/label_qualify'
DISCOVERY, HELDOUT, AUTHOR_ROUNDS, PER_REQUEST = 24, 64, 5, 24
ALIASES = {'confused': 'confusion', 'happy': 'joy', 'sad': 'sadness'}
Z = 1.959963984540054
EVIDENCE_SCHEMA = 'character-creation.neuron-label-evidence.v1'
MEASUREMENT = 'final_prompt_token_swiglu_product'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def wilson_lower(tp, total):
    if not total: return 0.
    p = tp / total
    return (p + Z * Z / (2 * total) - Z * math.sqrt(p * (1 - p) / total + Z * Z / (4 * total * total))) / (1 + Z * Z / total)


def connect(root):
    (Path(root) / OUT).mkdir(exist_ok=True)
    db = sqlite3.connect(Path(root) / OUT / 'qualify.sqlite', timeout=60)
    db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA busy_timeout=60000')
    db.executescript('''CREATE TABLE IF NOT EXISTS concepts(name TEXT PRIMARY KEY, definition TEXT, state TEXT, result TEXT);
        CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, stage TEXT, concept TEXT, label INT, round INT, scene TEXT,
            state TEXT, created REAL, finished REAL, error TEXT);
        CREATE TABLE IF NOT EXISTS scenes(id TEXT PRIMARY KEY, concept TEXT, label INT, family TEXT, text TEXT,
            verified INT, split TEXT, artifact TEXT, artifact_sha256 TEXT);
        CREATE INDEX IF NOT EXISTS tasks_state ON tasks(state, stage);''')
    return db


def add_task(db, stage, concept, label=None, rnd=None, scene=None):
    identity = digest([stage, concept, label, rnd, scene])
    db.execute("INSERT OR IGNORE INTO tasks VALUES(?,?,?,?,?,?,'queued',?,NULL,NULL)", (identity, stage, concept, label, rnd, scene, time.time()))


def init(root, concepts, definitions):
    with connect(root) as db:
        for name in concepts:
            name = ALIASES.get(name, name).strip().lower()
            db.execute("INSERT OR IGNORE INTO concepts VALUES(?,?,'new',NULL)", (name, definitions.get(name.replace(' ', '_'))))
        return db.execute('SELECT count(*) FROM concepts').fetchone()[0]


def request(stage, concept, definition, label, rnd, scene, model):
    """The exact chat request for each stage; definitions and scenes are the only variable text."""
    if stage == 'define':
        text = (f'Write a precise one-paragraph labelling definition for the concept "{concept}" as it appears in a short '
                'written scene, for annotators deciding whether a scene is a positive example. State what must currently '
                'be true in the scene, then list what is insufficient (the word alone, hypotheticals, quotes, absent or '
                'past states, look-alike concepts). Follow the style of this example:\n'
                'A person in the described scene currently experiences sorrow, grief, dejection or felt disappointment. '
                'Bad circumstances alone, fatigue, anger, quoted vocabulary or a hypothetical feeling is insufficient.\n'
                'Reply with JSON only: {"definition": "..."}')
        return {'messages': [{'role': 'user', 'content': text}], 'max_tokens': 300, 'temperature': 0}
    if stage == 'author':
        kind = ('scenes in which the definition clearly holds' if label else
                'hard negative scenes in which the definition does NOT hold, although they share vocabulary, setting or '
                'tone with it (the word itself, a hypothetical, a quoted line, an absent or past state, a look-alike concept)')
        text = (f'Concept: {concept}\nDefinition: {definition}\n\nWrite {PER_REQUEST} short, original, third-person {kind}. '
                'Each scene is one to three sentences about different people, places and situations, with no two sharing a '
                'scenario. Adults only; nothing graphic. Reply with JSON only: a list of objects '
                '{"family": "<three-word scenario id>", "scene": "..."}')
        return {'messages': [{'role': 'user', 'content': text}], 'max_tokens': 2200, 'temperature': 0.9, 'top_p': 0.95,
                'seed': int(digest([concept, label, rnd])[:8], 16)}
    if stage == 'verify':
        text = (f'Definition of "{concept}": {definition}\n\nScene: {scene}\n\n'
                'Does the definition hold for this scene? Answer with one word: yes or no.')
        return {'messages': [{'role': 'user', 'content': text}], 'max_tokens': 3, 'temperature': 0}
    return {'messages': [{'role': 'user', 'content': scene}], 'max_tokens': 1, 'temperature': 0}  # capture


def parse_json(text):
    match = re.search(r'(\[.*\]|\{.*\})', text or '', re.S)
    return json.loads(match.group(1)) if match else None


def plan(root):
    """Create the next tasks for every concept from its current state; CPU-only stages run here."""
    db = connect(root)
    with db:
        for name, definition, state in db.execute('SELECT name,definition,state FROM concepts').fetchall():
            if state == 'new':
                if definition: db.execute("UPDATE concepts SET state='defined' WHERE name=?", (name,))
                else: add_task(db, 'define', name)
            elif state == 'defined':
                for label in (1, 0):
                    for rnd in range(AUTHOR_ROUNDS): add_task(db, 'author', name, label, rnd)
                db.execute("UPDATE concepts SET state='authoring' WHERE name=?", (name,))
            elif state == 'authoring' and not db.execute("SELECT 1 FROM tasks WHERE concept=? AND stage='author' AND state IN ('queued','running')", (name,)).fetchone():
                for sid, text in db.execute('SELECT id,text FROM scenes WHERE concept=? AND verified IS NULL', (name,)).fetchall():
                    add_task(db, 'verify', name, scene=sid)
                db.execute("UPDATE concepts SET state='verifying' WHERE name=?", (name,))
            elif state == 'verifying' and not db.execute("SELECT 1 FROM tasks WHERE concept=? AND stage='verify' AND state IN ('queued','running')", (name,)).fetchone():
                for label in (1, 0):
                    ids = [r[0] for r in db.execute('SELECT id FROM scenes WHERE concept=? AND label=? AND verified=1 ORDER BY id', (name, label))]
                    random.Random(digest([name, label])).shuffle(ids)
                    for i, sid in enumerate(ids[:DISCOVERY + HELDOUT]):
                        db.execute('UPDATE scenes SET split=? WHERE id=?', ('discovery' if i < DISCOVERY else 'heldout', sid))
                        add_task(db, 'capture', name, scene=sid)
                db.execute("UPDATE concepts SET state='capturing' WHERE name=?", (name,))
            elif state == 'capturing' and not db.execute("SELECT 1 FROM tasks WHERE concept=? AND stage='capture' AND state IN ('queued','running')", (name,)).fetchone():
                db.execute("UPDATE concepts SET state='analysis' WHERE name=?", (name,))
    return db


def store(db, stage, task, answer, response):
    _, _, concept, label, _, scene, *_ = task
    if stage == 'define':
        value = parse_json(answer); definition = value.get('definition') if isinstance(value, dict) else None
        definition = (definition or (answer or '').strip())[:900]
        db.execute("UPDATE concepts SET definition=?, state='defined' WHERE name=?", (definition, concept))
    elif stage == 'author':
        items = parse_json(answer)
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict) or not isinstance(item.get('scene'), str): continue
            text = ' '.join(item['scene'].split())
            if not 20 <= len(text) <= 600: continue
            sid = digest([concept, text.lower()])
            db.execute('INSERT OR IGNORE INTO scenes VALUES(?,?,?,?,?,NULL,NULL,NULL,NULL)',
                       (sid, concept, label, str(item.get('family', ''))[:60], text))
    elif stage == 'verify':
        word = (answer or '').strip().lower()
        said = 1 if word.startswith('yes') else 0 if word.startswith('no') else None
        intended = db.execute('SELECT label FROM scenes WHERE id=?', (scene,)).fetchone()[0]
        db.execute('UPDATE scenes SET verified=? WHERE id=?', (int(said == intended) if said is not None else 0, scene))
    else:
        artifact = response.get('activation_artifact') or {}
        db.execute('UPDATE scenes SET artifact=?, artifact_sha256=? WHERE id=?', (artifact.get('path'), artifact.get('sha256'), scene))


def worker(root, port, until, stop, log):
    base = f'http://127.0.0.1:{port}'
    order = "CASE stage WHEN 'define' THEN 0 WHEN 'author' THEN 1 WHEN 'verify' THEN 2 ELSE 3 END"
    while not stop.is_set() and time.time() < until:
        try: model = fetch(base + '/v1/models', timeout=10)['data'][0]['id']
        except Exception as exc: log(port, 'waiting', repr(exc)[:120]); stop.wait(30); continue
        db = connect(root)
        try:
            db.execute('BEGIN IMMEDIATE')
            tasks = db.execute(f"SELECT * FROM tasks WHERE state='queued' ORDER BY {order}, created, id LIMIT 4").fetchall()
            db.executemany("UPDATE tasks SET state='running' WHERE id=?", [(t[0],) for t in tasks]); db.commit()
            if not tasks: db.close(); stop.wait(20); continue
            requests = []
            for t in tasks:
                _, stage, concept, label, rnd, scene, *_ = t
                definition = db.execute('SELECT definition FROM concepts WHERE name=?', (concept,)).fetchone()[0]
                text = db.execute('SELECT text FROM scenes WHERE id=?', (scene,)).fetchone()[0] if scene else None
                body = request(stage, concept, definition, label, rnd, text, model)
                requests.append({'model': model, 'stream': False, 'seed': body.pop('seed', 0),
                                 'chat_template_kwargs': {'enable_thinking': False}, **body})
            envelope = {'batch_id': os.urandom(16).hex(), 'requests': requests}
            try:
                if not fetch(base + '/v1/batch/preflight', envelope, timeout=120).get('fits'): raise ValueError('Batch does not fit')
            except Exception as exc:
                with db: db.executemany("UPDATE tasks SET state='queued' WHERE id=?", [(t[0],) for t in tasks])
                log(port, 'preflight_released', repr(exc)[:160]); stop.wait(10); continue
            try:
                results = fetch(base + '/v1/batch/completions', envelope, timeout=900)['results']
            except Exception as exc:
                with db: db.executemany("UPDATE tasks SET state='uncertain',error=? WHERE id=?", [(repr(exc)[:300], t[0]) for t in tasks])
                log(port, 'uncertain_no_retry', repr(exc)[:160]); continue
            with db:
                for t, result in zip(tasks, results):
                    response = result['response']
                    store(db, t[1], t, response['choices'][0]['message']['content'], response)
                    db.execute("UPDATE tasks SET state='done',finished=? WHERE id=?", (time.time(), t[0]))
        finally: db.close()


def analyse(args):
    """Discovery-only selection, then held-out precision. Runs in a CPU worker process."""
    import numpy as np
    root, name, definition, rows = args
    def plane(path):
        with np.load(path, allow_pickle=False) as data: return data['mlp_input_last'].reshape(-1).astype(np.float32)
    disc = [r for r in rows if r['split'] == 'discovery']; held = [r for r in rows if r['split'] == 'heldout']
    if sum(r['label'] for r in disc) < 12 or sum(1 - r['label'] for r in disc) < 12 or sum(r['label'] for r in held) < 20:
        return {'concept': name, 'passed': False, 'reason': 'Too few verified scenes after checking'}
    X = np.stack([plane(r['artifact']) for r in disc]); y = np.array([r['label'] for r in disc], bool)
    pos, neg = X[y], X[~y]
    pooled = np.sqrt((pos.var(0) + neg.var(0)) / 2) + 1e-6
    contrast = (pos.mean(0) - neg.mean(0)) / pooled
    best = None
    for key in np.argsort(-np.abs(contrast))[:32]:
        sign = 1 if contrast[key] > 0 else -1; values = sign * X[:, key]
        # The gate is held-out precision, so choose (on discovery only) the threshold with the best discovery precision
        # lower bound among those predicting at least 8 positives; ties keep more recall.
        for threshold in np.unique((np.sort(values)[1:] + np.sort(values)[:-1]) / 2):
            predicted = values > threshold; total = int(predicted.sum())
            if total < 8: continue
            score = (wilson_lower(int(predicted[y].sum()), total), total)
            if best is None or score > best[0]: best = (score, int(key), sign, float(sign * threshold))
    if best is None: return {'concept': name, 'passed': False, 'reason': 'No discovery threshold predicts 8 positives'}
    (accuracy, _), key, sign, threshold = best  # accuracy here is the discovery precision lower bound
    layer, neuron = divmod(key, 17408)
    outcomes = []
    for r in held:
        value = float(plane(r['artifact'])[key])
        if sign * value > sign * threshold:
            outcomes.append({'unit_id': r['id'], 'correct': bool(r['label']), 'value': value, 'family': r['family']})
    tp = sum(o['correct'] for o in outcomes); total = len(outcomes); precision = tp / total if total else 0.
    lower = wilson_lower(tp, total); passed = total >= 20 and lower > 0.70
    disc_sha = digest(sorted((r['id'], r['label']) for r in disc)); held_sha = digest(sorted((r['id'], r['label']) for r in held))
    confidence = {'calibrated': passed, 'method': 'heldout_precision', 'value': precision}
    return {'concept': name, 'passed': passed, 'layer': int(layer), 'neuron': int(neuron), 'direction': 'positive' if sign > 0 else 'inverse',
            'confidence': confidence, 'lower_bound': lower, 'outcomes': total, 'standardized_contrast': float(contrast[key]),
            'evidence': {'schema': EVIDENCE_SCHEMA, 'label': name, 'layer': int(layer), 'neuron': int(neuron),
                'direction': 'positive' if sign > 0 else 'inverse', 'confidence': confidence, 'strong_related': True,
                'claim': 'probe_conditional_precision', 'probe': definition, 'measurement': MEASUREMENT,
                'discovery_dataset_sha256': disc_sha,
                'frozen_candidate': {'threshold': threshold, 'discovery_precision_wilson95_lower': accuracy, 'standardized_contrast': float(contrast[key]),
                                     'selection': 'top32 absolute standardized contrasts; best discovery precision lower bound, at least 8 predicted positives'},
                'calibration': {'method': 'heldout_precision', 'split': 'held_out', 'dataset_sha256': held_sha},
                'validation': {'passed': passed, 'split': 'held_out', 'dataset_sha256': held_sha, 'tp': tp, 'fp': total - tp,
                               'precision': precision, 'precision_wilson95_lower': lower, 'heldout_positives': sum(r['label'] for r in held),
                               'heldout_negatives': sum(1 - r['label'] for r in held),
                               'outcomes': [{'unit_id': o['unit_id'], 'correct': o['correct']} for o in outcomes]},
                'limitations': ['Scenes are model-authored and model-checked, not human-annotated.',
                                'Final prompt token measurement; v1 labels used the mean over all prompt positions.',
                                'Precision is conditional on these probes, not the probability that a neuron means a word.']}}


def finish_analysis(root, pool, log):
    db = connect(root)
    names = [r[0] for r in db.execute("SELECT name FROM concepts WHERE state='analysis'")]
    for name in names:
        definition = db.execute('SELECT definition FROM concepts WHERE name=?', (name,)).fetchone()[0]
        rows = [dict(zip(('id', 'label', 'family', 'split', 'artifact'), r)) for r in db.execute(
            "SELECT id,label,family,split,artifact FROM scenes WHERE concept=? AND split IS NOT NULL AND artifact IS NOT NULL", (name,))]
        db.execute("UPDATE concepts SET state='analysing' WHERE name=?", (name,)); db.commit()
        def done(future, name=name):
            result = future.result(); slug = re.sub(r'[^a-z0-9]+', '_', name).strip('_')
            if 'evidence' in result:
                from persona_vectors import ModelSpec
                identity = ModelSpec.load(Path(root) / 'data/persona_model.json').identity
                (Path(root) / OUT / f'persona_label_evidence_v2_{slug}.json').write_text(json.dumps({**result['evidence'],
                    'model_identity': identity}, indent=1))
            with connect(root) as d: d.execute("UPDATE concepts SET state='done', result=? WHERE name=?",
                                                (json.dumps({k: v for k, v in result.items() if k != 'evidence'}), name))
            log(None, 'analysed', {k: result.get(k) for k in ('concept', 'passed', 'lower_bound', 'outcomes', 'reason')})
        pool.submit(analyse, (root, name, definition, rows)).add_done_callback(done)
    db.close()


def status(root):
    with connect(root) as db:
        value = {'time': time.time(), 'concepts': dict(db.execute('SELECT state,count(*) FROM concepts GROUP BY 1').fetchall()),
                 'tasks': {f'{s}:{t}': n for s, t, n in db.execute('SELECT stage,state,count(*) FROM tasks GROUP BY 1,2')},
                 'scenes': dict(db.execute("SELECT coalesce(split,'unsplit'),count(*) FROM scenes WHERE verified=1 GROUP BY 1").fetchall()),
                 'passed': [json.loads(r[0])['concept'] for r in db.execute("SELECT result FROM concepts WHERE state='done'") if json.loads(r[0]).get('passed')]}
    value['complete'] = bool(value['concepts']) and set(value['concepts']) == {'done'}
    (Path(root) / OUT / 'status.json').write_text(json.dumps(value))
    return value


def run(root, until, ports, workers):
    stop = threading.Event(); lock = threading.Lock()
    def log(port, kind, detail):
        with lock, (Path(root) / OUT / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'time': time.time(), 'port': port, 'event': kind, 'detail': detail}) + '\n')
    with connect(root) as db:
        db.execute("UPDATE tasks SET state='uncertain',error='Runner restarted while running' WHERE state='running'")
        db.execute("UPDATE concepts SET state='analysis' WHERE state='analysing'")
    threads = [threading.Thread(target=worker, args=(root, p, until, stop, log), daemon=True) for p in ports]
    for t in threads: t.start()
    with ProcessPoolExecutor(workers) as pool:
        while time.time() < until and not status(root)['complete']:
            plan(root).close(); finish_analysis(root, pool, log); stop.wait(30)
        stop.set()
    status(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('init', 'run', 'status'))
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--concepts', help='JSON list of concept names (init)')
    parser.add_argument('--definitions', help='JSON object of existing definitions (init)')
    parser.add_argument('--until', type=float); parser.add_argument('--ports', default='8930-8935')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if args.command == 'init':
        print(json.dumps({'concepts': init(args.root, json.loads(Path(args.concepts).read_text()),
                                           json.loads(Path(args.definitions).read_text()) if args.definitions else {})}))
    elif args.command == 'run':
        low, high = map(int, args.ports.split('-'))
        run(args.root, args.until, range(low, high + 1), args.workers)
    else: print(json.dumps(status(args.root)))


if __name__ == '__main__': main()
