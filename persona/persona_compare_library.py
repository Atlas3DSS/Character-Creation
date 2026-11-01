"""Comparison library for the studio's Response comparison view (− reversed / 0 baseline / + recipe).

Read-only over runs that already exist, so the view opens prefilled instead of empty:
  * studio  - the explorer's own /api/compare runs (results/persona_explorer.sqlite, table comparisons)
  * card    - Neuropedia steering cards (<neuropedia>/steer-*.json): `unsteered` arm = baseline, the recipe arm = +,
              and the − column is the arm of another card whose recipe is the exact negation (every dose sign flipped)
  * identity- the "who are you" single-neuron sweep as synced by the neuron atlas (results/neuron_atlas/identity_answers.jsonl):
              +d and −d rows of one neuron, baseline = the sweep's unsteered row (layer −1)
  * atlas   - hook for the full single-neuron sweep (neuropedia/monkey/atlas_sweep, synced to results/neuron_atlas/atlas_sweep/)

Answers are passed through verbatim. Nothing here reads neuropedia/monkey/longchat/private/ or any held-back data; the
sweep's held-back rows never reach the synced file (its export filters layer > -3) and are skipped again here.
Routes live under /api/response-study/ so Neuropedia's public /viewer copy keeps blocking them (VIEWER_PUBLIC_BLOCK).
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

NEUROPEDIA = Path('/home/orwel/dev_genius/experiments/flare/neuropedia')
NEUROPEDIA_URL = 'http://127.0.0.1:8798'
OWNER_IPS = ('local', '@Orwelian84')
PUSH_RX = re.compile(r'L\s*(\d+)\s*[ ,/]*N\s*(\d+)\s*([+-]\s*\d*\.?\d+)')
NEURON_RX = re.compile(r'\bL\s*(\d+)\s*[ ,/_-]*N\s*(\d+)\b', re.I)
OPEN_STATES = ('pending', 'approved', 'running')
LEGACY_REQUESTERS = ('claude', 'personas', 'viewer', 'anonymous', 'local', 'orwelian84')   # requesters, not characters (Neuropedia's list + owner handle)


def parse_recipe(text):
    """'L54 N12953 +0.25 + L34 N6111 -0.3' -> ((54, 12953, 0.25), (34, 6111, -0.3)); () when nothing parses."""
    return tuple((int(l), int(n), round(float(d.replace(' ', '')), 6)) for l, n, d in PUSH_RX.findall(text or ''))


def signature(pushes):
    return tuple(sorted(pushes))


def negate(pushes):
    return tuple((l, n, round(-d, 6)) for l, n, d in pushes)


def recipe_text(pushes, joiner=' + '):
    return joiner.join(f'L{l} N{n} {d:+g}' for l, n, d in pushes)


def studio_push_text(push):
    if push.get('kind') == 'neuron' or {'layer', 'neuron'} <= set(push):
        return f"L{push.get('layer')} N{push.get('neuron')} {float(push.get('dose', 0)):+g}"
    label = push.get('name') or push.get('trait') or push.get('direction') or push.get('kind') or 'push'
    return f"{label} {float(push.get('dose', push.get('strength', 0)) or 0):+g}"


class CompareLibrary:
    def __init__(self, app):
        self.app = app
        self.root = Path(app.root)
        config = getattr(app, 'config', None) or {}
        self.neuropedia = Path(config.get('neuropedia_dir') or NEUROPEDIA)
        self.neuropedia_url = config.get('neuropedia_url') or NEUROPEDIA_URL
        self.lock = threading.Lock()
        self.cards, self.card_stamp = {}, {}          # file name -> parsed card summary (reparsed when mtime changes)
        self.identity, self.identity_stamp = [], None  # items built from the synced sweep file
        self.atlas, self.atlas_stamp = [], None
        self.identity_base, self.atlas_base = None, {}

    # ---- sources -------------------------------------------------------------------------------------------------
    def _queue_rows(self):
        try:
            text = (self.neuropedia / 'monkey/cards_queue/requests.jsonl').read_text()
        except OSError:
            return []
        rows = []
        for line in text.splitlines():
            try: rows.append(json.loads(line))
            except ValueError: pass
        return rows

    def _load_cards(self):
        seen = {}
        for path in self.neuropedia.glob('steer-*.json'):
            try: seen[path.name] = path.stat().st_mtime
            except OSError: pass
        for name in list(self.cards):
            if name not in seen: self.cards.pop(name); self.card_stamp.pop(name, None)
        for name, stamp in seen.items():
            if self.card_stamp.get(name) == stamp: continue
            try: report = json.loads((self.neuropedia / name).read_text())
            except (OSError, ValueError): continue
            results = [r for r in report.get('results') or [] if isinstance(r, dict) and 'arm' in r]
            results = [r for r in results if not str(r['arm']).lower().startswith('prompt')]   # steering arms only
            arm = report.get('recipe_arm') or max((r['arm'] for r in results if r['arm'] != 'unsteered'),
                                                  key=lambda a: a.count(' N'), default='')
            pushes = parse_recipe(arm)
            if not pushes: continue
            answers = defaultdict(dict)   # arm -> prompt -> row
            order = []
            for r in results:
                prompt = r.get('prompt') or ''
                if prompt not in order: order.append(prompt)
                answers[r['arm']].setdefault(prompt, r)
            self.cards[name] = {'file': name, 'mtime': stamp, 'arm': arm, 'pushes': pushes, 'sig': signature(pushes),
                                'title': report.get('title') or '', 'summary': report.get('summary') or '', 'verdict': report.get('verdict'),
                                'requested_by': report.get('requested_by') or '', 'note': report.get('request_note') or '',
                                'settings': {k: report.get(k) for k in ('seed', 'max_tokens', 'temperature')},
                                'prompts': order, 'recipe': dict(answers.get(arm, {})), 'baseline': dict(answers.get('unsteered', {}))}
            self.card_stamp[name] = stamp

    def _load_identity(self):
        path = self.root / 'results/neuron_atlas/identity_answers.jsonl'
        try: stamp = path.stat().st_mtime
        except OSError: self.identity, self.identity_stamp = [], None; return
        if stamp == self.identity_stamp: return
        base, table = None, defaultdict(dict)
        with path.open() as stream:
            for line in stream:
                try: row = json.loads(line)
                except ValueError: continue
                if row.get('sweep') == 'held-back' or row.get('layer', -9) < -1: continue   # held-back + combos never shown
                if row['layer'] == -1:
                    if base is None or (row.get('finished') or 0) > (base.get('finished') or 0): base = row
                    continue
                if row.get('pushes'): continue
                table[(row['layer'], row['neuron'])][round(float(row['dose']), 6)] = row   # later sweep wins a duplicate
        items = []
        for (layer, neuron), doses in table.items():
            levels = sorted({abs(d) for d in doses if d}, reverse=True)
            steps, full = [], False
            for level in levels:
                up, down = doses.get(level), doses.get(-level)
                full = full or bool(up and down)
                steps.append({'level': level, 'up': up, 'down': down})
            finished = max((r.get('finished') or 0) for r in doses.values())
            sweeps = sorted({r.get('sweep') or '' for r in doses.values()})
            items.append({'key': f'identity:L{layer}N{neuron}', 'source': 'identity', 'layer': layer, 'neuron': neuron,
                          'steps': steps, 'full': full, 'ts': finished, 'sweeps': sweeps})
        self.identity, self.identity_stamp, self.identity_base = items, stamp, base

    def _load_atlas(self):
        """Hook for the full single-neuron sweep (neuropedia/monkey/atlas_sweep, schema neuropedia.atlas-sweep.v1), read
        from the studio's synced copy results/neuron_atlas/atlas_sweep/atlas_sweep.sqlite (the same file the neuron atlas
        reads): directions(layer, neuron, sign, record) with record.probes [{dose, answer, finish_reason, tokens, ...}].
        Baseline = the sweep's own unsteered reference (meta 'reference' or reference.json beside it), else missing.
        Absent or not yet filled -> no items; any schema drift -> no items rather than an error."""
        folder = self.root / 'results/neuron_atlas/atlas_sweep'
        path = folder / 'atlas_sweep.sqlite'
        try: stamp = (path.stat().st_mtime, path.stat().st_size)
        except OSError: self.atlas, self.atlas_stamp, self.atlas_base = [], None, {}; return
        if stamp == self.atlas_stamp: return
        import sqlite3
        table, base, items = defaultdict(dict), {}, []
        try:
            with sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=5) as db:
                found = db.execute('SELECT layer, neuron, sign, record FROM directions WHERE record IS NOT NULL').fetchall()
                try: meta = dict(db.execute('SELECT key, value FROM meta').fetchall())
                except sqlite3.Error: meta = {}
            reference = meta.get('reference')
            if reference is None and (folder / 'reference.json').exists(): reference = (folder / 'reference.json').read_text()
            if reference:
                ref = json.loads(reference) if isinstance(reference, str) else reference
                if isinstance(ref, dict) and ref.get('answer') is not None:
                    base[ref.get('prompt') or 'who are you'] = dict(ref, sweep=ref.get('schema'), finish=ref.get('finish'))
            for layer, neuron, sign, raw in found:
                rec = json.loads(raw)
                for probe in rec.get('probes') or []:
                    dose = round(float(probe['dose']), 6)
                    table[(int(layer), int(neuron))][dose] = {'answer': probe.get('answer', ''), 'finish_reason': probe.get('finish_reason'),
                        'tokens': probe.get('tokens'), 'dose': dose, 'coherent': probe.get('coherent_auto'), 'sweep': rec.get('sweep') or 'atlas sweep',
                        'finished': rec.get('at') or rec.get('finished') or 0}
        except (sqlite3.Error, OSError, ValueError, KeyError, TypeError):
            table, base = {}, {}
        for (layer, neuron), doses in table.items():
            steps, full = [], False
            for level in sorted({abs(d) for d in doses if d}, reverse=True):
                up, down = doses.get(level), doses.get(-level)
                full = full or bool(up and down and base)
                steps.append({'level': level, 'prompt': 'who are you', 'up': up, 'down': down})
            items.append({'key': f'atlas:L{layer}N{neuron}', 'source': 'atlas', 'layer': layer, 'neuron': neuron, 'steps': steps, 'full': full,
                          'ts': max(float(r.get('finished') or 0) for r in doses.values()), 'sweeps': ['atlas-full-v1']})
        self.atlas, self.atlas_stamp, self.atlas_base = items, stamp, base

    def _studio(self):
        with self.app.connect() as db:
            rows = db.execute('SELECT id,created,state,input FROM comparisons ORDER BY created DESC').fetchall()
        out = []
        for identity, created, state, raw in rows:
            try: value = json.loads(raw)
            except (TypeError, ValueError): continue
            pushes = value.get('pushes') or []
            neurons = [(int(p['layer']), int(p['neuron'])) for p in pushes if isinstance(p, dict) and 'layer' in p and 'neuron' in p]
            out.append({'key': 'studio:' + identity, 'source': 'studio', 'id': identity, 'ts': created, 'state': state,
                        'prompt': value.get('prompt') or '', 'recipe': ' + '.join(studio_push_text(p) for p in pushes if isinstance(p, dict)) or 'no pushes',
                        'neurons': neurons, 'single': len(pushes) == 1, 'full': state == 'complete', 'cols': [state == 'complete'] * 3, 'settings': {k: value.get(k) for k in ('seed', 'phase', 'max_tokens', 'thinking')}})
        return out

    # ---- index ---------------------------------------------------------------------------------------------------
    def _card_people(self, rows):
        """card file -> (character, owner?) from the queue rows that produced it."""
        people = {}
        for r in rows:
            if not r.get('card'): continue
            character = r.get('character') or ''
            name = str(r.get('name') or '')
            if not character and not r.get('public') and name.lower() not in LEGACY_REQUESTERS and not name.startswith('@'):
                character = name
            people[r['card']] = (character, r.get('ip') in OWNER_IPS or not r.get('public'))
        return people

    def _card_items(self):
        rows = self._queue_rows()
        people = self._card_people(rows)
        queued = {}
        for r in rows:
            if r.get('state') in OPEN_STATES:
                sig = signature(parse_recipe(r.get('recipe', '')))
                if sig: queued.setdefault(sig, r.get('id'))
        by_sig = defaultdict(list)
        for card in self.cards.values(): by_sig[card['sig']].append(card)
        for group in by_sig.values(): group.sort(key=lambda c: c['mtime'], reverse=True)
        items, done = [], set()
        for card in sorted(self.cards.values(), key=lambda c: c['mtime'], reverse=True):
            if card['file'] in done: continue
            reverse_sig = signature(negate(card['pushes']))
            partners = [c for c in by_sig.get(reverse_sig, []) if c['file'] != card['file']]
            if partners:
                partner = partners[0]
                # one item per ± pair: the + side is the card whose first push (by layer, neuron) has a positive dose
                plus, minus = (card, partner) if sorted(card['pushes'])[0][2] > 0 else (partner, card)
                done.update((plus['file'], minus['file']))
            else:
                plus, minus = card, None
                done.add(card['file'])
            fallback = plus['requested_by'] if plus['requested_by'].lower() not in LEGACY_REQUESTERS and not plus['requested_by'].startswith('@') else ''
            character, mine = people.get(plus['file'], (fallback, True))
            prompts = list(plus['prompts'])
            full = bool(minus) and any(p in plus['recipe'] and p in minus['recipe'] and (p in plus['baseline'] or p in minus['baseline']) for p in prompts)
            neurons = sorted({(l, n) for l, n, _ in plus['pushes']})
            items.append({'key': 'card:' + plus['file'], 'source': 'card', 'file': plus['file'], 'reverse_file': minus['file'] if minus else None,
                          'ts': max(plus['mtime'], minus['mtime'] if minus else 0), 'title': plus['title'], 'character': character or fallback,
                          'mine': mine, 'recipe': plus['arm'], 'reverse_recipe': recipe_text(negate(plus['pushes'])), 'neurons': neurons,
                          'single': len(plus['pushes']) == 1, 'pushes': len(plus['pushes']), 'questions': len(prompts),
                          'prompt': prompts[0] if prompts else '', 'prompts': prompts, 'full': full, 'verdict': plus['verdict'],
                          'reverse_queued': None if minus else queued.get(reverse_sig),
                          'cols': [bool(minus), any(p in plus['baseline'] or (minus and p in minus['baseline']) for p in prompts), bool(plus['recipe'])]})
        return items

    def _all(self):
        with self.lock:
            self._load_cards(); self._load_identity(); self._load_atlas()
            cards = self._card_items()
        studio = self._studio()
        light = []
        for item in studio + cards:
            light.append(item)
        for item in self.identity + self.atlas:
            light.append({'key': item['key'], 'source': item['source'], 'ts': item['ts'], 'neurons': [(item['layer'], item['neuron'])],
                          'single': True, 'full': item['full'], 'recipe': f"L{item['layer']} N{item['neuron']} ±" + '/'.join(f"{s['level']:g}" for s in item['steps']),
                          'title': 'Identity sweep' if item['source'] == 'identity' else 'Atlas sweep', 'character': '', 'prompt': 'who are you' if item['source'] == 'identity' else (item['steps'][0].get('prompt') if item['steps'] else ''),
                          'questions': len(item['steps']), 'mine': True, 'sweeps': item.get('sweeps', []),
                          'cols': [any(s['down'] for s in item['steps']), bool(self.identity_base if item['source'] == 'identity' else self.atlas_base),
                                   any(s['up'] for s in item['steps'])]})
        return light

    @staticmethod
    def _public(item):
        out = {k: v for k, v in item.items() if k not in ('prompts',)}
        out['neurons'] = [f'L{l}N{n}' for l, n in item.get('neurons', [])]
        return out

    def listing(self, q='', full_only=False, kind='all', source='all', offset=0, limit=60):
        items = self._all()
        totals = {'all': len(items), 'full': sum(1 for i in items if i['full'])}
        by_source = defaultdict(lambda: {'items': 0, 'full': 0})
        for i in items:
            by_source[i['source']]['items'] += 1; by_source[i['source']]['full'] += int(bool(i['full']))
        neurons = {(int(l), int(n)) for l, n in NEURON_RX.findall(q or '')}
        words = [w for w in NEURON_RX.sub(' ', (q or '').lower()).split() if w]
        def keep(i):
            if full_only and not i['full']: return False
            if kind == 'single' and not i['single']: return False
            if kind == 'multi' and i['single']: return False
            if source != 'all' and i['source'] != source: return False
            if neurons and not neurons & set(map(tuple, i.get('neurons', []))): return False
            if words:
                hay = ' '.join(str(i.get(k) or '') for k in ('title', 'character', 'prompt', 'recipe', 'file', 'id')).lower()
                hay += ' ' + ' '.join(i.get('prompts') or []).lower() + ' ' + ' '.join(i.get('sweeps') or []).lower()
                if not all(w in hay for w in words): return False
            return True
        rank = {'studio': 0, 'card': 0, 'atlas': 1, 'identity': 2}   # hand-made runs first; the big sweeps after them
        hits = sorted((i for i in items if keep(i)), key=lambda i: (rank.get(i['source'], 3), -(i['ts'] or 0)))
        default = next((i['key'] for i in sorted(items, key=lambda i: -(i['ts'] or 0)) if i['source'] == 'card' and i['full'] and i.get('mine')), None)
        offset, limit = max(0, int(offset)), max(1, min(200, int(limit)))
        return {'total': len(hits), 'offset': offset, 'items': [self._public(i) for i in hits[offset:offset + limit]],
                'counts': totals | {'sources': dict(by_source)}, 'default': default, 'atlas_sweep': bool(self.atlas)}

    # ---- one item, with its columns ------------------------------------------------------------------------------
    @staticmethod
    def _cell(row, source, extra=''):
        if not row: return None
        bits = [source]
        if row.get('finish_reason') or row.get('finish'): bits.append(f"finish {row.get('finish_reason') or row.get('finish')}")
        if row.get('tokens') is not None: bits.append(f"{row['tokens']} tokens")
        if extra: bits.append(extra)
        return {'answer': row.get('answer') if row.get('answer') is not None else '', 'source': ' · '.join(bits)}

    def item(self, key):
        source, _, ident = (key or '').partition(':')
        if source == 'studio': return self._studio_item(ident)
        if source == 'card': return self._card_item(ident)
        if source in ('identity', 'atlas'): return self._sweep_item(source, ident)
        raise ValueError('Unknown comparison')

    def _studio_item(self, identity):
        data = self.app.comparison(identity)
        value = data['input']
        pushes = value.get('pushes') or []
        recipe = ' + '.join(studio_push_text(p) for p in pushes if isinstance(p, dict)) or 'no pushes'
        cells = []
        for i in range(3):
            output = data.get('output')
            if not isinstance(output, list) or i >= len(output):
                cells.append({'missing': True, 'state': data['state'], 'answer': (output or {}).get('error') if isinstance(output, dict) else None,
                              'source': f"studio comparison {identity[:8]} · {data['state']}"}); continue
            row = output[i]
            if row.get('error'): cells.append({'error': row['error'], 'source': f'studio comparison {identity[:8]} · sign {row.get("sign")}'}); continue
            choice = row['response']['choices'][0]
            cells.append({'answer': choice['message'].get('content') or '', 'reasoning': choice['message'].get('reasoning_content') or None,
                          'complete': bool(row.get('complete')),
                          'source': f"studio comparison {identity[:8]} · sign {row.get('sign')} × recipe · finish {choice.get('finish_reason')}"})
        output = data.get('output') if isinstance(data.get('output'), list) else []
        temperature = next((row['request'].get('temperature') for row in output if isinstance(row, dict) and isinstance(row.get('request'), dict)), None)
        reverse = ' + '.join(studio_push_text(dict(p, dose=-float(p.get('dose', 0) or 0))) for p in pushes if isinstance(p, dict)) or 'no pushes'
        return {'key': 'studio:' + identity, 'source': 'studio', 'id': identity, 'state': data['state'], 'title': value.get('prompt') or '',
                'recipe': recipe, 'reverse_recipe': reverse,
                'settings': f"seed {value.get('seed')} · {value.get('phase')} · {value.get('max_tokens')} tokens · thinking {'on' if value.get('thinking') else 'off'}"
                            + (f" · temperature {temperature}" if temperature is not None else ''),
                'steps': [{'label': 'Q1', 'prompt': value.get('prompt') or '', 'cells': cells}]}

    def _card_item(self, name):
        with self.lock:
            self._load_cards()
            items = {i['file']: i for i in self._card_items()}
            plus = self.cards.get(name)
            if plus is None or name not in items: raise ValueError('Unknown card')
            meta = items[name]
            minus = self.cards.get(meta['reverse_file']) if meta['reverse_file'] else None
        def settings(card): return ' · '.join(f'{k} {v}' for k, v in card['settings'].items() if v is not None)
        steps = []
        for i, prompt in enumerate(meta['prompts'] or plus['prompts']):
            base_row, base_from = plus['baseline'].get(prompt), plus
            if base_row is None and minus is not None: base_row, base_from = minus['baseline'].get(prompt), minus
            reverse = self._cell(minus['recipe'].get(prompt), f"{minus['file']} · arm {minus['arm']}") if minus else None
            steps.append({'label': f'Q{i + 1}', 'prompt': prompt, 'cells': [
                reverse or {'missing': True},
                self._cell(base_row, f"{base_from['file']} · arm unsteered") or {'missing': True},
                self._cell(plus['recipe'].get(prompt), f"{plus['file']} · arm {plus['arm']}") or {'missing': True}]})
        note = settings(plus)
        if minus and settings(minus) != note: note += f"  (− card: {settings(minus)})"
        return {'key': 'card:' + name, 'source': 'card', 'file': name, 'reverse_file': meta['reverse_file'], 'title': plus['title'],
                'character': meta['character'], 'summary': plus['summary'], 'verdict': plus['verdict'], 'recipe': plus['arm'],
                'reverse_recipe': meta['reverse_recipe'], 'reverse_queued': meta['reverse_queued'], 'settings': note, 'steps': steps,
                'reverse_title': minus['title'] if minus else None}

    def _sweep_item(self, source, ident):
        with self.lock:
            if source == 'identity': self._load_identity(); pool, base = self.identity, {'who are you': getattr(self, 'identity_base', None)}
            else: self._load_atlas(); pool, base = self.atlas, getattr(self, 'atlas_base', {})
        item = next((i for i in pool if i['key'] == f'{source}:{ident}'), None)
        if item is None: raise ValueError('Unknown sweep neuron')
        label = 'identity sweep' if source == 'identity' else 'atlas sweep'
        def where(row):
            if source == 'identity': return f"{label} · {row.get('sweep')} · dose {row.get('dose'):+g}" + (' · coherent' if row.get('coherent') else ' · incoherent' if row.get('coherent') == 0 else '')
            return f"{label} · {row.get('sweep')} · dose {float(row.get('dose')):+g}"
        steps = []
        for s in item['steps']:
            prompt = s.get('prompt') or 'who are you'
            b = base.get(prompt)
            steps.append({'label': f"±{s['level']:g}", 'prompt': prompt, 'cells': [
                self._cell(s['down'], where(s['down'])) if s['down'] else {'missing': True},
                self._cell(b, f"{label} · unsteered baseline" + (f" · {b.get('sweep')}" if b and b.get('sweep') else '')) if b else {'missing': True},
                self._cell(s['up'], where(s['up'])) if s['up'] else {'missing': True}]})
        neuron = f"L{item['layer']} N{item['neuron']}"
        return {'key': item['key'], 'source': source, 'title': f'{neuron} · {label}', 'recipe': f'{neuron} +dose', 'reverse_recipe': f'{neuron} −dose',
                'settings': 'greedy · seed 0 · thinking off · up to ' + ('160' if source == 'identity' else '48') + ' tokens', 'steps': steps,
                'sweeps': item.get('sweeps', []), 'layer': item['layer'], 'neuron': item['neuron']}

    # ---- queue the missing reversed run ---------------------------------------------------------------------------
    def queue_reverse(self, value):
        """Queue the negated recipe as a Neuropedia steering card (same 12-question battery as the + card) through the
        local card-request API; the new card pairs with this one automatically once the card worker writes it."""
        key = str(value.get('key') or '')
        source, _, ident = key.partition(':')
        if source == 'card':
            with self.lock:
                self._load_cards()
                card = self.cards.get(ident)
                if card is None: raise ValueError('Unknown card')
                item = next((i for i in self._card_items() if i['file'] == ident), None)
            if item and item['reverse_file']: raise ValueError('The reversed card already exists')
            if item and item['reverse_queued']: return {'ok': True, 'id': item['reverse_queued'], 'already': True}
            pushes, character = negate(card['pushes']), (item or {}).get('character') or ''
        elif source in ('identity', 'atlas'):
            m = re.fullmatch(r'L(\d+)N(\d+)', ident)
            dose = float(value.get('dose', 0))
            if not m or not 0 < abs(dose) <= 4: raise ValueError('Give the neuron and the dose to queue')
            pushes, character = ((int(m.group(1)), int(m.group(2)), round(dose, 6)),), ''
            for r in self._queue_rows():
                if r.get('state') in OPEN_STATES and signature(parse_recipe(r.get('recipe', ''))) == signature(pushes):
                    return {'ok': True, 'id': r.get('id'), 'already': True}
        else:
            raise ValueError('Only cards and sweep neurons can be queued here; studio comparisons always run all three columns')
        body = {'recipe': recipe_text(pushes, ', '), 'note': f'reverse of {ident}'}
        if character: body['character'] = character
        request = urllib.request.Request(self.neuropedia_url + '/api/cards/request', data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=20) as response: out = json.loads(response.read())
        except urllib.error.HTTPError as error:
            try: out = json.loads(error.read())
            except ValueError: out = {'ok': False, 'error': f'Neuropedia answered {error.code}'}
        except (OSError, ValueError) as error:
            raise ValueError(f'Neuropedia card queue unavailable: {error}')
        if not out.get('ok'): raise ValueError(out.get('error') or 'Neuropedia refused the request')
        return {'ok': True, 'id': out.get('id'), 'recipe': body['recipe'], 'pending_ahead': out.get('pending_ahead')}


_LIBRARIES = {}


def library(app):
    if id(app) not in _LIBRARIES: _LIBRARIES[id(app)] = CompareLibrary(app)
    return _LIBRARIES[id(app)]
