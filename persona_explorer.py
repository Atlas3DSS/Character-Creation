"""Local Character Creation persona explorer; run `python persona_explorer.py`.

Independently implemented exploration UI inspired by Lyra's neuron-explorer.
Inference is delegated to explicitly configured native Flare endpoints.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from urllib.parse import parse_qs, urlparse
import uuid

from persona_fleet import fetch
from persona_neighbors import CoactivityIndex
from persona_jobs import JobStore
from persona_sculptor import Sculptor
from persona_vectors import ModelSpec, VectorBank, canonical_digest
from scripts.experiments.scotus.qwen_eval_budget import enforce_complete_answer_budget, qwen_budget_metadata


def object_fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError("Unexpected or missing request fields")


def bounded_int(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer between {low} and {high}")
    return value


def prompt_label(row):
    source, labels = row.get('source') or {}, row.get('labels') or {}
    mutation = (row.get('mutation') or {}).get('kind', 'original')
    if row['corpus'] == 'skippy':
        book = Path(str(source.get('path', '')).replace('\\', '/')).stem[:55] or 'Book'
        member = Path(str(source.get('member', '')).replace('\\', '/')).name[:40]
        title = ' · '.join(str(value) for value in (book, member, '@' + str(source.get('char_start', 0)), mutation) if value)
    elif row['corpus'] == 'scotus':
        original = row.get('original_prompt_id', source.get('jsonl_line', row['category']))
        title = f'SCOTUS {original} · {mutation}'
    elif row['corpus'] == 'big_five':
        inventory = labels.get('inventory_item') or {}
        if inventory:
            title = f"IPIP {inventory.get('id', source.get('inventory_item_id', ''))} · {inventory.get('dimension', row['category'])}"
        else:
            original = row.get('original_prompt_id', source.get('prompt_id', row['category']))
            title = f"{original} · {labels.get('profile', 'baseline')}"
    else:
        title = str(row.get('label') or row['category']) + ' · ' + str(mutation)
    return title[:145] + ' · ' + row['id'][:12]


def generation_capabilities(flare):
    thinking = flare.get('thinking_enabled_supported')
    contexts = [flare.get(name) for name in ('physical_context_tokens', 'logical_context_tokens')]
    contexts = [value for value in contexts if type(value) is int and value > 0]
    return {'thinking_supported': thinking if type(thinking) is bool else None,
            'context_tokens': min(contexts) if contexts else None}


def check_generation_settings(flare, settings):
    capabilities = generation_capabilities(flare)
    if settings['thinking'] and capabilities['thinking_supported'] is False:
        raise ValueError('Worker does not support thinking-enabled generation')
    if capabilities['context_tokens'] is not None and settings['max_tokens'] >= capabilities['context_tokens']:
        raise ValueError(f"Token budget must be below the worker context of {capabilities['context_tokens']} tokens; the prompt also needs space")
    return capabilities


def verify_endpoint(descriptor: dict, endpoint: dict) -> dict:
    flare = descriptor.get("flare", {})
    if flare.get("prepared_identity") != endpoint["prepared_identity"]:
        raise ValueError("Server prepared identity differs from the verified package")
    if flare.get("model_identity") not in (None, endpoint["model_identity"]):
        raise ValueError("Diagnostic server belongs to another model")
    runtime = flare.get("runtime_identity")
    if (runtime or endpoint.get("runtime_identity")) and runtime != endpoint.get("runtime_identity"):
        raise ValueError("Diagnostic runtime identity differs from the verified worker")
    if endpoint.get("runtime_identity") and flare.get("model_identity") != endpoint["model_identity"]:
        raise ValueError("Diagnostic server omitted its model identity")
    return flare


class Explorer:
    def __init__(self, root: Path, config: dict):
        self.root, self.config = root, config
        self.model = ModelSpec.load(root / "data/persona_model.json")
        self.neighbors = CoactivityIndex(root, self.model)
        self.bank = VectorBank(root / "results", self.model,
            **({"checkpoint": Path(config["checkpoint"])} if config.get("checkpoint") else {}))
        self.bank_lock = threading.RLock()
        self.endpoints = deepcopy(config.get("endpoints", []))
        if not isinstance(self.endpoints, list) or len(self.endpoints) > 6:
            raise ValueError("Configure at most six inference endpoints")
        for endpoint in self.endpoints:
            if not isinstance(endpoint, dict) or not endpoint.get("url", "").startswith("http://127.0.0.1:"):
                raise ValueError("Endpoints must name explicitly forwarded localhost HTTP servers")
            if endpoint.get("model_identity") != self.model.identity or not endpoint.get("prepared_identity"):
                raise ValueError("Every endpoint requires the verified model and prepared-package identities")
            parsed = urlparse(endpoint['url'])
            if parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment or not parsed.port:
                raise ValueError("Endpoint URLs must contain only a loopback host and port")
        if len({endpoint['url'] for endpoint in self.endpoints}) != len(self.endpoints):
            raise ValueError("Duplicate endpoints cannot provide independent GPU slots")
        (root / "results").mkdir(exist_ok=True)
        self.database = root / "results/persona_explorer.sqlite"
        self.jobs = JobStore(root)
        self.singleton = self.database.with_suffix('.sqlite.lock').open('a')
        fcntl.flock(self.singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute("CREATE TABLE IF NOT EXISTS comparisons (id TEXT PRIMARY KEY, created REAL, state TEXT, input TEXT, output TEXT)")
            db.executescript('''
                CREATE TABLE IF NOT EXISTS recipes(id TEXT PRIMARY KEY,name TEXT NOT NULL,model_identity TEXT NOT NULL,
                    pushes TEXT NOT NULL,phase TEXT NOT NULL,created REAL NOT NULL,updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY,created REAL NOT NULL,state TEXT NOT NULL,input TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS cells(batch_id TEXT NOT NULL,cell_index INTEGER NOT NULL,state TEXT NOT NULL,
                    snapshot TEXT NOT NULL,endpoint_label TEXT,output TEXT,error TEXT,raw TEXT,
                    PRIMARY KEY(batch_id,cell_index));
            ''')
            db.execute("UPDATE comparisons SET state='interrupted',output=? WHERE state IN ('queued','running')",
                       (json.dumps({'error': 'UI restarted; remote completion unknown; no automatic retry'}),))
            interrupted = [row[0] for row in db.execute("SELECT DISTINCT batch_id FROM cells WHERE state IN ('queued','running')")]
            db.execute("UPDATE cells SET state='interrupted',error=? WHERE state IN ('queued','running')",
                       ('UI restarted; remote completion unknown; no automatic retry',))
            for identity in interrupted:
                self._update_batch(db, identity)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.cell_executor = ThreadPoolExecutor(max_workers=max(1, len(self.endpoints)))
        self.endpoint_condition = threading.Condition()
        self.available_endpoints = list(self.endpoints)
        self.corpus_lock = threading.RLock()
        self.corpus_cache, self.corpus_signature = {}, None
        self.corpus_binding = {}
        self.capacity_lock = threading.Lock()
        self.capacity_cache, self.capacity_updated = None, 0.
        self.closed = False
        self.sculptor = Sculptor(self)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        try:
            db.execute('PRAGMA busy_timeout=30000')
            with db:
                yield db
        finally:
            db.close()

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=True)
        self.cell_executor.shutdown(wait=True)
        self.singleton.close()

    @contextmanager
    def endpoint_slot(self, eligible=None):
        with self.endpoint_condition:
            self.endpoint_condition.wait_for(lambda: any(eligible is None or endpoint['url'] in eligible
                                                       for endpoint in self.available_endpoints))
            index = next(i for i, endpoint in enumerate(self.available_endpoints)
                         if eligible is None or endpoint['url'] in eligible)
            endpoint = self.available_endpoints.pop(index)
        try:
            yield endpoint
        finally:
            with self.endpoint_condition:
                self.available_endpoints.append(endpoint)
                self.endpoint_condition.notify_all()

    def capacity(self):
        with self.capacity_lock:
            if self.capacity_cache is not None and time.monotonic() - self.capacity_updated < 10:
                return deepcopy(self.capacity_cache)
            def inspect(endpoint):
                card = {'label': str(endpoint.get('label') or endpoint.get('gpu_uuid') or endpoint['url']),
                        'available': False, 'idle': None}
                try:
                    descriptor = fetch(endpoint['url'] + '/v1/models', timeout=2)['data'][0]
                    flare = verify_endpoint(descriptor, endpoint)
                    card['available'] = True
                    card.update(generation_capabilities(flare))
                except Exception as error:
                    card['error'] = f'{type(error).__name__}: {error}'
                return card
            with ThreadPoolExecutor(max_workers=max(1, len(self.endpoints))) as workers:
                cards = list(workers.map(inspect, self.endpoints))
            available = sum(card['available'] for card in cards)
            thinking = [card['thinking_supported'] for card in cards if card['available']]
            contexts = [card['context_tokens'] for card in cards if card['available'] and card['context_tokens'] is not None]
            self.capacity_cache = {'configured': len(self.endpoints), 'available': available, 'max_parallel': available,
                'cards': cards, 'checked_at': time.time(),
                'thinking_supported': False if False in thinking else True if thinking and all(value is True for value in thinking) else None,
                'context_tokens': min(contexts) if contexts else None,
                'availability_semantics': 'reachable and identity-verified; busy/idle unknown'}
            self.capacity_updated = time.monotonic()
            return deepcopy(self.capacity_cache)

    def corpus_snapshot(self):
        """Publish one hash-verified immutable corpus generation to all readers."""
        path = self.root / 'data/persona_manager_corpus.jsonl'
        manifest_path = self.root / 'data/persona_manager_corpus_manifest.json'
        with self.corpus_lock:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            if manifest.get('status') != 'complete':
                raise ValueError('Corpus publication is not complete; try after ingestion finishes')
            stat = path.stat()
            signature = (hashlib.sha256(manifest_bytes).hexdigest(), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if signature == self.corpus_signature:
                return self.corpus_cache, self.corpus_binding
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != manifest.get('output_sha256') or manifest_path.read_bytes() != manifest_bytes:
                raise ValueError('Corpus content does not match its completed publication manifest')
            rows, groups = {}, {}
            for line in raw.splitlines():
                if not line.strip(): continue
                row = json.loads(line)
                if (not isinstance(row, dict) or row.get('schema') != 'character-creation.manager-corpus.v1'
                        or not isinstance(row.get('id'), str) or row['id'] in rows
                        or row['id'] != canonical_digest({key: value for key, value in row.items() if key != 'id'})
                        or row.get('split') not in ('discovery', 'heldout')
                        or not isinstance(row.get('corpus'), str) or not row['corpus']
                        or not isinstance(row.get('category'), str)
                        or not isinstance(row.get('prompt'), str) or not 1 <= len(row['prompt']) <= 100000
                        or not isinstance(row.get('system', ''), str)):
                    raise ValueError('Corpus record identity, text or split binding is invalid')
                bounded_int(row.get('max_tokens', 2048), 256, 8192, 'Corpus token budget')
                if not isinstance(row.get('pair_id', row['id']), str): raise ValueError('Invalid corpus source group')
                group = (row['corpus'], row.get('pair_id', row['id']))
                if group in groups and groups[group] != row['split']:
                    raise ValueError('Corpus source variants cross discovery/heldout splits')
                groups[group] = row['split']; rows[row['id']] = row
            if not rows or not self.corpus_cache.keys() <= rows.keys():
                raise ValueError('Corpus publication must preserve existing immutable records')
            self.corpus_cache, self.corpus_signature = rows, signature
            self.corpus_binding = {'corpus_sha256': digest, 'manifest_sha256': signature[0]}
            return self.corpus_cache, self.corpus_binding

    def corpora(self):
        rows, _ = self.corpus_snapshot()
        result = {}
        for row in rows.values():
            item = result.setdefault(row['corpus'], {'name': row['corpus'], 'rows': 0, 'discovery': 0, 'heldout': 0})
            item['rows'] += 1; item[row['split']] += 1
        return {'corpora': [result[name] for name in sorted(result)]}

    def prompts(self, corpus='', split='discovery', q='', offset=0, limit=12):
        if split not in ('discovery', 'heldout', 'all') or not isinstance(q, str) or len(q) > 512:
            raise ValueError('Invalid prompt filter')
        bounded_int(offset, 0, 1000000, 'Offset'); bounded_int(limit, 1, 100, 'Limit')
        rows, _ = self.corpus_snapshot()
        matches = [row for row in rows.values() if (not corpus or row['corpus'] == corpus)
                   and (split == 'all' or row['split'] == split)
                   and (not q or q.casefold() in (row['prompt'] + '\n' + row['category'] + '\n' + row['id'] + '\n' + prompt_label(row)).casefold())]
        return {'rows': [{key: row.get(key, '') for key in ('id', 'corpus', 'split', 'category', 'prompt', 'system')}
                         | {'label': prompt_label(row), 'preview': row['prompt'][:240],
                            'max_tokens': row.get('max_tokens', 2048)} for row in matches[offset:offset + limit]],
                'total': len(matches), 'offset': offset, 'limit': limit}

    def validate_recipe(self, value):
        object_fields(value, ('pushes', 'phase'), ('name',))
        if not isinstance(value['phase'], str) or value['phase'] not in ('all', 'prefill', 'decode'):
            raise ValueError('Unknown intervention phase')
        name = value.get('name', 'Baseline' if not value['pushes'] else 'Unsaved recipe')
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120:
            raise ValueError('Recipe name must contain 1–120 characters')
        pushes = value['pushes']
        if not isinstance(pushes, list) or len(pushes) > 512:
            raise ValueError('Pushes must be a bounded list of interventions')
        for push in pushes:
            object_fields(push, ('layer', 'dose'), ('kind', 'neuron', 'trait', 'seed', 'site'))
            bounded_int(push['layer'], 0, self.model.layers - 1, 'Layer')
            if type(push['dose']) not in (int, float) or not math.isfinite(push['dose']):
                raise ValueError('Every dose must be a finite number')
            kind = push.get('kind', 'neuron')
            if kind not in ('neuron', 'column', 'trait', 'random'): raise ValueError('Unknown intervention kind')
            allowed = {'neuron'} if kind in ('neuron', 'column') else {'trait'} if kind == 'trait' else {'seed', 'site'}
            if push.keys() - {'layer', 'dose', 'kind'} - allowed: raise ValueError('Intervention fields do not match its kind')
            if kind in ('neuron', 'column'): bounded_int(push.get('neuron'), 0, self.model.intermediate - 1, 'Neuron')
            if kind == 'trait' and not isinstance(push.get('trait'), str): raise ValueError('Trait name is required')
            if kind == 'random':
                bounded_int(push.get('seed'), 0, 2**32 - 1, 'Random seed')
                if push.get('site', 'mlp_out') not in ('mlp_out', 'residual'): raise ValueError('Invalid random control site')
        with self.bank_lock:
            program, provenance = self.bank.program(pushes, value['phase'])
        recipe = {'name': name.strip(), 'pushes': deepcopy(pushes), 'phase': value['phase']}
        return recipe, program, provenance

    def saved_recipes(self):
        with self.connect() as db:
            rows = db.execute('SELECT id,name,model_identity,pushes,phase,created,updated FROM recipes ORDER BY updated DESC,id').fetchall()
        return {'recipes': [{'id': row[0], 'name': row[1], 'model_identity': row[2], 'pushes': json.loads(row[3]),
                             'phase': row[4], 'created': row[5], 'updated': row[6]} for row in rows]}

    def save_recipe(self, value):
        object_fields(value, ('name', 'model_identity', 'pushes', 'phase'), ('id',))
        if value['model_identity'] != self.model.identity:
            raise ValueError('Recipe belongs to another model')
        recipe, _, _ = self.validate_recipe({key: value[key] for key in ('name', 'pushes', 'phase')})
        identity = value.get('id', uuid.uuid4().hex)
        if not isinstance(identity, str) or not 1 <= len(identity) <= 100:
            raise ValueError('Invalid saved recipe identity')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT created,model_identity FROM recipes WHERE id=?', (identity,)).fetchone()
            if 'id' in value and (old is None or old[1] != self.model.identity):
                raise ValueError('Saved recipe does not exist for this model')
            now = time.time(); created = old[0] if old else now
            db.execute('INSERT INTO recipes VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,pushes=excluded.pushes,phase=excluded.phase,updated=excluded.updated',
                (identity, recipe['name'], self.model.identity, json.dumps(recipe['pushes']), recipe['phase'], created, now))
        return {'id': identity, **recipe, 'model_identity': self.model.identity, 'created': created, 'updated': now}

    @staticmethod
    def _update_batch(db, identity):
        states = [row[0] for row in db.execute('SELECT state FROM cells WHERE batch_id=?', (identity,))]
        state = ('running' if 'running' in states else 'queued' if 'queued' in states
                 else states[0] if len(set(states)) == 1 else 'partial')
        db.execute('UPDATE batches SET state=? WHERE id=?', (state, identity))

    def batches(self):
        with self.connect() as db:
            rows = db.execute("SELECT batches.id,batches.created,batches.state,COUNT(*),SUM(cells.state NOT IN ('queued','running')) FROM batches JOIN cells ON cells.batch_id=batches.id GROUP BY batches.id ORDER BY batches.created DESC LIMIT 100").fetchall()
        return {'batches': [dict(zip(('id', 'created', 'state', 'total', 'completed'), row)) for row in rows]}

    def batch(self, identity):
        with self.connect() as db:
            row = db.execute('SELECT created,state,input FROM batches WHERE id=?', (identity,)).fetchone()
            cells = db.execute('SELECT cell_index,state,snapshot,endpoint_label,output,error FROM cells WHERE batch_id=? ORDER BY cell_index', (identity,)).fetchall()
        if row is None: raise ValueError('Batch does not exist')
        result = []
        for index, state, raw, endpoint, output, error in cells:
            snapshot = json.loads(raw)
            item = {'index': index, 'label': snapshot['label'], 'state': state,
                    'prompt': snapshot['prompt'], 'recipe': snapshot['recipe']}
            if endpoint is not None: item['endpoint_label'] = endpoint
            if output is not None: item['output'] = json.loads(output)
            if error is not None: item['error'] = error
            result.append(item)
        return {'id': identity, 'created': row[0], 'state': row[1], 'input': json.loads(row[2]), 'cells': result}

    def submit_batch(self, value):
        if self.closed or not self.endpoints: raise ValueError('No active Flare endpoints configured')
        value = deepcopy(value)
        object_fields(value, ('model_identity', 'seed', 'max_tokens', 'thinking', 'smoke', 'cells'))
        if value['model_identity'] != self.model.identity: raise ValueError('Batch belongs to another model')
        bounded_int(value['seed'], 0, 2**32 - 1, 'Seed')
        bounded_int(value['max_tokens'], 256, 4096, 'Token budget')
        if type(value['thinking']) is not bool or type(value['smoke']) is not bool:
            raise ValueError('Thinking and smoke must be explicit booleans')
        if not isinstance(value['cells'], list) or not 1 <= len(value['cells']) <= 6:
            raise ValueError('A batch requires one to six cells')
        snapshots = []
        for index, cell in enumerate(value['cells']):
            object_fields(cell, (), ('label', 'prompt_id', 'prompt', 'recipe_id', 'recipe'))
            if ('prompt_id' in cell) == ('prompt' in cell) or ('recipe_id' in cell) == ('recipe' in cell):
                raise ValueError('Each cell requires exactly one prompt and one recipe reference or value')
            label = cell.get('label', f'Cell {index + 1}')
            if not isinstance(label, str) or not 1 <= len(label.strip()) <= 120: raise ValueError('Invalid cell label')
            source, binding = None, {}
            if 'prompt_id' in cell:
                rows, binding = self.corpus_snapshot()
                if not isinstance(cell['prompt_id'], str) or cell['prompt_id'] not in rows: raise ValueError('Unknown published prompt')
                source = deepcopy(rows[cell['prompt_id']])
                prompt = {'id': source['id'], 'text': source['prompt'], 'corpus': source['corpus'], 'split': source['split']}
            else:
                if not isinstance(cell['prompt'], str) or not cell['prompt'].strip() or len(cell['prompt']) > 20000:
                    raise ValueError('Custom prompts must contain 1–20,000 characters')
                prompt = {'text': cell['prompt']}
            recipe_value = cell.get('recipe')
            saved = None
            if 'recipe_id' in cell:
                saved = next((row for row in self.saved_recipes()['recipes'] if row['id'] == cell['recipe_id']), None)
                if saved is None or saved['model_identity'] != self.model.identity: raise ValueError('Unknown recipe for this model')
                recipe_value = {key: saved[key] for key in ('name', 'pushes', 'phase')}
            recipe, program, provenance = self.validate_recipe(recipe_value)
            if saved: recipe['id'] = saved['id']
            if source and source['corpus'] == 'scotus':
                enforce_complete_answer_budget(value['max_tokens'], allow_short=value['smoke'], opt_in_flag='smoke:true')
            budget = qwen_budget_metadata(value['max_tokens'])
            budget.update(smoke=value['smoke'], promotion_eligible=False,
                          promotion_eligible_budget=budget['promotion_eligible_budget'] and not value['smoke'])
            snapshots.append({'label': label.strip(), 'prompt': prompt, 'recipe': recipe, 'source_record': source,
                'source_binding': binding, 'saved_recipe': saved, 'program': program,
                'provenance': provenance, 'budget': budget, 'model': self.model.__dict__})
        # Transport failures may reduce capacity. A reachable identity/capability
        # mismatch is a configuration error and rejects the complete batch.
        admitted = []
        for endpoint in self.endpoints:
            try:
                descriptor = fetch(endpoint['url'] + '/v1/models', timeout=5)['data'][0]
            except OSError:
                continue
            for snapshot in snapshots: self.check_program(descriptor, endpoint, snapshot['program'])
            capabilities = check_generation_settings(descriptor['flare'], value)
            admitted.append({**deepcopy(endpoint), 'capabilities': capabilities})
        if not admitted: raise ValueError('No verified reachable Flare endpoints are available')
        for snapshot in snapshots: snapshot['admitted_endpoints'] = admitted
        recorded_input = {**value, 'admitted_endpoints': admitted}
        identity = uuid.uuid4().hex
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT COUNT(*) FROM batches WHERE state IN ('queued','running')").fetchone()[0] >= 16:
                raise ValueError('Sixteen batches are already active')
            db.execute('INSERT INTO batches VALUES(?,?,?,?)', (identity, time.time(), 'queued', json.dumps(recorded_input, allow_nan=False)))
            db.executemany('INSERT INTO cells(batch_id,cell_index,state,snapshot) VALUES(?,?,?,?)',
                [(identity, index, 'queued', json.dumps(snapshot, allow_nan=False)) for index, snapshot in enumerate(snapshots)])
        for index, snapshot in enumerate(snapshots):
            self.cell_executor.submit(self.run_cell, identity, index, snapshot, value)
        return identity

    @staticmethod
    def check_program(descriptor, endpoint, program):
        flare = verify_endpoint(descriptor, endpoint)
        if program['sites']:
            capability = flare.get('steering') or {}
            sites = {site['name']: site['width'] for site in capability.get('sites', [])}
            if capability.get('protocol') != 'additive-bf16-sites-v1':
                raise ValueError('Native server does not advertise the required steering protocol')
            if len(program['sites']) > capability.get('max_sites', 0):
                raise ValueError("Recipe exceeds the native package's prepared site capacity")
            if any(sites.get(site['site']) != len(site['vector']) for site in program['sites']):
                raise ValueError('Recipe site or vector width is absent from the prepared package')
        return flare

    def infer(self, endpoint, prompt, settings, program, provenance, *, system='', extra=None, persist=None):
        descriptor = fetch(endpoint['url'] + '/v1/models', timeout=5)['data'][0]
        flare = self.check_program(descriptor, endpoint, program)
        check_generation_settings(flare, settings)
        messages = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': prompt}]
        request = {'model': descriptor['id'], 'messages': messages, 'seed': settings['seed'],
                   'temperature': .7, 'top_p': .95, 'top_k': 20, 'max_tokens': settings['max_tokens'],
                   'stream': False, 'chat_template_kwargs': {'enable_thinking': settings['thinking']}}
        if program['sites']: request['steering'] = program
        if extra and extra.get('job_id'): request['metadata'] = {'job_id': extra['job_id']}
        bound = {**provenance, **(extra or {}), 'model_identity': self.model.identity,
                 'prepared_identity': endpoint['prepared_identity'], 'runtime_identity': endpoint.get('runtime_identity'),
                 'request_sha256': canonical_digest(request)}
        raw = {'request': request, 'endpoint': endpoint, 'provenance': bound}
        if persist: persist(raw)
        response = fetch(endpoint['url'] + '/v1/chat/completions', request,
                         timeout=self.config.get('request_timeout_seconds', 600))
        raw['response'] = response
        if persist: persist(raw)
        for key in ('model_identity', 'prepared_identity', 'runtime_identity'):
            if key in response and response[key] != endpoint.get(key):
                raise ValueError('Response ' + key + ' differs from the verified worker')
            if endpoint.get('runtime_identity') and response.get(key) != endpoint.get(key):
                raise ValueError('Diagnostic response omitted or changed ' + key)
        choice = response['choices'][0]
        answer = choice['message'].get('content') or ''
        if not isinstance(answer, str): raise ValueError('Completion content must be text')
        diagnostics = response.get('diagnostics', {})
        complete = (choice.get('finish_reason') == 'stop' and bool(answer.strip())
                    and diagnostics.get('complete_final_answer', not bool(endpoint.get('runtime_identity'))) is True
                    and diagnostics.get('unexpected_thinking', False) is False)
        output = {'answer': answer, 'complete': complete, 'finish_reason': choice.get('finish_reason'),
                  'usage': response.get('usage'), 'provenance': bound, 'runtime_identity': response.get('runtime_identity'),
                  'model_identity': self.model.identity, 'prepared_identity': endpoint['prepared_identity'],
                  'runtime': response.get('runtime'), 'diagnostics': diagnostics}
        for key in ('activation_artifact', 'provenance_artifact'):
            if key in response: output[key] = response[key]
        raw['complete'] = complete
        return output, raw

    def run_cell(self, identity, index, snapshot, value):
        output, error, state = None, None, 'failed'
        try:
            with self.endpoint_slot({item['url'] for item in snapshot['admitted_endpoints']}) as endpoint:
                label = str(endpoint.get('label') or endpoint.get('gpu_uuid') or endpoint['url'])
                with self.connect() as db:
                    db.execute("UPDATE cells SET state='running',endpoint_label=? WHERE batch_id=? AND cell_index=?", (label, identity, index))
                    self._update_batch(db, identity)
                def persist(raw):
                    with self.connect() as db:
                        db.execute('UPDATE cells SET raw=? WHERE batch_id=? AND cell_index=?',
                                   (json.dumps(raw, allow_nan=False), identity, index))
                source = snapshot['source_record'] or {}
                extra = {'job_id': identity + '-' + str(index), 'budget': snapshot['budget'],
                         'admitted_endpoints': snapshot['admitted_endpoints'],
                         'recipe_sha256': canonical_digest(snapshot['recipe']), 'source': {
                             **snapshot['source_binding'], **{key: source[key] for key in
                             ('id', 'source_id', 'source_sha256', 'source_text_sha256', 'pair_id', 'corpus', 'split', 'mutation') if key in source}},
                         'prompt_sha256': canonical_digest({'prompt': snapshot['prompt']['text'], 'system': source.get('system', '')})}
                output, raw = self.infer(endpoint, snapshot['prompt']['text'], value, snapshot['program'],
                    snapshot['provenance'], system=source.get('system', ''), extra=extra, persist=persist)
                persist(raw)
                state = 'complete' if output['complete'] else 'incomplete'
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        with self.connect() as db:
            db.execute('UPDATE cells SET state=?,output=?,error=? WHERE batch_id=? AND cell_index=?',
                       (state, json.dumps(output, allow_nan=False) if output is not None else None, error, identity, index))
            self._update_batch(db, identity)

    def state(self) -> dict:
        metadata = json.loads((self.root / "data/persona_model.json").read_text())
        path = self.root / "results/persona_map.json"
        chart = json.loads(path.read_text()) if path.is_file() else {"neurons": [], "layers": [], "status": "Awaiting calibrated intervention results"}
        if chart.get("model_identity") not in (None, self.model.identity):
            raise ValueError("Map belongs to a different model")
        summary_path = self.root / "results/persona_baseline_summary.json"
        if not summary_path.exists():
            summary_path = self.root / "reports/persona_baseline.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        with self.connect() as db:
            comparisons = [{"id": row[0], "created": row[1], "state": row[2]}
                           for row in db.execute("SELECT id,created,state FROM comparisons ORDER BY created DESC LIMIT 20")]
        return {"model": metadata, "identity": self.model.identity, "map": chart,
                "baseline": summary, "comparisons": comparisons, "endpoints": len(self.endpoints),
                "calibrated_layers": [i for i in range(self.model.layers) if (self.root / "results" / f"layer_{i:02d}.npz").exists()]}

    def comparison(self, identity: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT state,input,output FROM comparisons WHERE id=?", (identity,)).fetchone()
        if row is None: raise ValueError("Comparison does not exist")
        return {"id": identity, "state": row[0], "input": json.loads(row[1]),
                "output": json.loads(row[2]) if row[2] else None}

    def submit(self, value: dict) -> str:
        if self.closed or not self.endpoints: raise ValueError("No native Flare endpoints configured")
        if value.get("model_identity") != self.model.identity:
            raise ValueError("The selected recipe belongs to another model")
        prompt = value.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 20000:
            raise ValueError("Enter a prompt of 1–20,000 characters")
        pushes = value.get("pushes", [])
        if not isinstance(pushes, list) or any(not isinstance(p, dict) for p in pushes):
            raise ValueError("Pushes must be a list of interventions")
        phase = value.get("phase", "all")
        # Validate all directions before publishing a queued job.
        with self.bank_lock: self.bank.program(pushes, phase)
        seed = value.get("seed", 7035)
        if type(seed) is not int or not 0 <= seed < 2**32: raise ValueError("Seed must be uint32")
        budget = value.get("max_tokens", 2048)
        if type(budget) is not int or not 256 <= budget <= 4096: raise ValueError("Token budget must be 256–4096")
        value = {"prompt": prompt, "pushes": pushes, "phase": phase, "seed": seed,
                 "max_tokens": budget, "thinking": bool(value.get("thinking", False)),
                 "model_identity": self.model.identity}
        identity = uuid.uuid4().hex
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            active = db.execute("SELECT COUNT(*) FROM comparisons WHERE state IN ('queued','running')").fetchone()[0]
            if active >= 4: raise ValueError("Four comparisons are already queued; wait for one to finish")
            db.execute("INSERT INTO comparisons VALUES (?,?,?,?,?)", (identity, time.time(), "queued", json.dumps(value), None))
        self.executor.submit(self.run_comparison, identity, value)
        return identity

    def run_comparison(self, identity: str, value: dict) -> None:
        with self.connect() as db: db.execute("UPDATE comparisons SET state='running' WHERE id=?", (identity,))
        def run(sign: int) -> dict:
            pushes = [{**p, "dose": float(p["dose"]) * sign} for p in value["pushes"]] if sign else []
            with self.bank_lock: program, provenance = self.bank.program(pushes, value['phase'])
            with self.endpoint_slot() as endpoint:
                _, raw = self.infer(endpoint, value['prompt'], value, program, provenance)
            return {'sign': sign, **raw}
        def protected_run(sign: int) -> dict:
            try:
                return run(sign)
            except (OSError, RuntimeError, ValueError, KeyError, IndexError) as error:
                return {"sign": sign, "error": str(error), "complete": False}
        try:
            with ThreadPoolExecutor(max_workers=3) as workers:
                output = list(workers.map(protected_run, [-1, 0, 1]))
            state = "failed" if any(row.get("error") for row in output) else "complete"
        except (OSError, RuntimeError, ValueError, KeyError, IndexError) as error:
            output, state = {"error": str(error)}, "failed"
        with self.connect() as db:
            db.execute("UPDATE comparisons SET state=?,output=? WHERE id=?", (state, json.dumps(output), identity))


def handler(app: Explorer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def send(self, value: object, status: int = 200) -> None:
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path)
            try:
                if path.path == "/":
                    body = (app.root / "ui/persona_explorer.html").read_bytes()
                    self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                elif path.path == "/atlas-logo.png":
                    body = (app.root / "ui/atlas-logo.png").read_bytes()
                    self.send_response(200); self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                elif path.path in ('/persona_lab.js', '/persona_lab.css', '/persona_neighbors.js', '/persona_neighbors.css', '/persona_jobs.js', '/persona_jobs.css', '/persona_sculptor.js', '/persona_sculptor.css'):
                    name = path.path[1:]
                    body = (app.root / 'ui' / name).read_bytes()
                    content_type = 'application/javascript; charset=utf-8' if name.endswith('.js') else 'text/css; charset=utf-8'
                    self.send_response(200); self.send_header('Content-Type', content_type)
                    self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
                elif path.path == "/api/state": self.send(app.state())
                elif path.path == '/api/capacity': self.send(app.capacity())
                elif path.path == '/api/sculptor': self.send(app.sculptor.catalog())
                elif path.path == '/api/sculptor/presets': self.send(app.sculptor.presets())
                elif path.path == '/api/sculptor/campaigns': self.send(app.sculptor.listing())
                elif path.path == '/api/sculptor/campaign': self.send(app.sculptor.get(parse_qs(path.query).get('id', [''])[0]))
                elif path.path == '/api/jobs': self.send(app.jobs.listing())
                elif path.path == '/api/job': self.send(app.jobs.get(parse_qs(path.query).get('id', [''])[0]))
                elif path.path == '/api/neighbors':
                    query = parse_qs(path.query)
                    self.send(app.neighbors.query(layer=int(query.get('layer', [''])[0]), neuron=int(query.get('neuron', [''])[0]),
                        corpus=query.get('corpus', ['all'])[0], scope=query.get('scope', ['all'])[0], limit=int(query.get('limit', ['12'])[0])))
                elif path.path == "/api/recipes":
                    self.send(json.loads((app.root / "data/lyra_recipes.json").read_text()))
                elif path.path == "/api/comparison":
                    self.send(app.comparison(parse_qs(path.query).get("id", [""])[0]))
                elif path.path == '/api/corpora': self.send(app.corpora())
                elif path.path == '/api/prompts':
                    query = parse_qs(path.query)
                    self.send(app.prompts(corpus=query.get('corpus', [''])[0], split=query.get('split', ['discovery'])[0],
                        q=query.get('q', [''])[0], offset=int(query.get('offset', ['0'])[0]), limit=int(query.get('limit', ['12'])[0])))
                elif path.path == '/api/saved-recipes': self.send(app.saved_recipes())
                elif path.path == '/api/batches': self.send(app.batches())
                elif path.path == '/api/batch': self.send(app.batch(parse_qs(path.query).get('id', [''])[0]))
                else: self.send({"error": "Not found"}, 404)
            except (ValueError, OSError, KeyError) as error: self.send({"error": str(error)}, 400)

        def do_POST(self) -> None:
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                self.send({"error": "Cross-origin requests are not accepted"}, 403); return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1000000: raise ValueError("Request size exceeds limit")
                value = json.loads(self.rfile.read(size))
                if not isinstance(value, dict): raise ValueError("Expected JSON object")
                if self.path in ('/api/jobs', '/api/job/cancel') or self.path.startswith('/api/sculptor/'):
                    if self.headers.get('Content-Type', '').split(';', 1)[0].strip().lower() != 'application/json':
                        raise ValueError('Scheduled job changes require application/json')
                if self.path == "/api/compare": self.send({"id": app.submit(value)}, 202)
                elif self.path == '/api/sculptor/plan': self.send(app.sculptor.plan(value))
                elif self.path == '/api/sculptor/build': self.send(app.sculptor.build(value))
                elif self.path == '/api/sculptor/preset': self.send(app.sculptor.preset(value))
                elif self.path == '/api/sculptor/campaigns': self.send(app.sculptor.submit(value), 202)
                elif self.path == '/api/sculptor/cancel': self.send(app.sculptor.cancel(value))
                elif self.path == '/api/sculptor/review': self.send(app.sculptor.review(value))
                elif self.path == '/api/jobs':
                    if not app.jobs.listing()['enabled']: raise ValueError('Scheduling is not enabled by operator configuration')
                    self.send(app.jobs.create(value), 201)
                elif self.path == '/api/job/cancel':
                    object_fields(value, ('id',)); self.send(app.jobs.cancel(value['id']))
                elif self.path == '/api/saved-recipes': self.send(app.save_recipe(value))
                elif self.path == '/api/batches': self.send({'id': app.submit_batch(value)}, 202)
                else: self.send({"error": "Not found"}, 404)
            except (ValueError, OSError, KeyError) as error: self.send({"error": str(error)}, 400)
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--port", type=int, default=8792)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else {}
    app = Explorer(args.root, config)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(app))
    print(f"Character Creation: http://127.0.0.1:{args.port}", flush=True)
    try: server.serve_forever()
    finally: server.server_close(); app.close()


if __name__ == "__main__": main()
