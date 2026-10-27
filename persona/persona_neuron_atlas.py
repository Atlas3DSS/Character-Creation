"""Whole-model neuron atlas: one categorized dot for every MLP neuron.

Organizes existing checkpoint-bound evidence the way a single-neuron sweep atlas
does: a category per neuron and direction, a searchable library and a method
note. Sources are the static weight lens (results/persona_atlas_L*.npz), the
unsteered final-prefill all-neuron captures (results/probe-*.npz) and the
recorded labels, map candidates, recipes and association searches. The lens is
a static projection, not an observed generation; activation selectivity is an
association over authored prompts, not a causal effect.
"""
from __future__ import annotations
import argparse
import csv
import glob
import hashlib
import io
import json
import os
import re
import subprocess
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np

SCHEMA = 'character-creation.neuron-atlas.v1'
LAYERS, WIDTH, SIDE = 64, 17408, 132
OUT = 'results/neuron_atlas'
LENS = ['persona', 'word', 'piece', 'code', 'number', 'format', 'script', 'mixed']
LENS_LABEL = {'persona': 'Feeling & character', 'word': 'Whole words', 'piece': 'Word pieces', 'code': 'Code & syntax',
              'number': 'Numbers & math', 'format': 'Punctuation & format', 'script': 'Other scripts', 'mixed': 'Mixed'}
LENS_NOTE = {'persona': 'Its top tokens are words about feelings, desire, temperament, morals or relationships.',
             'word': 'Its top tokens are whole English words on some other topic.',
             'piece': 'Its top tokens are fragments of words rather than whole words.',
             'code': 'Its top tokens are identifiers, keywords or programming syntax.',
             'number': 'Its top tokens are digits, numerals or math symbols.',
             'format': 'Its top tokens are punctuation, whitespace or markup.',
             'script': 'Its top tokens are in a non-Latin script, such as Chinese, Cyrillic or Arabic.',
             'mixed': 'Its top tokens have no single majority kind.'}
DOMAINS = ['identity', 'feelings', 'relationships', 'conflict', 'values', 'facts', 'everyday', 'stories', 'other', 'broad']
DOMAIN_LABEL = {'identity': 'Identity & self', 'feelings': 'Feelings', 'relationships': 'Relationships & company',
                'conflict': 'Disagreement & correction', 'values': 'Values & choices', 'facts': 'Facts, beliefs & uncertainty',
                'everyday': 'Everyday life', 'stories': 'Stories, play & imagination', 'other': 'Other prompts',
                'broad': 'No clear preference'}
EVIDENCE = ['labelled', 'recipe', 'association', 'trait', 'none']
EVIDENCE_LABEL = {'labelled': 'Concept label', 'recipe': 'Used in a recipe', 'association': 'Association search hit',
                  'trait': 'Big Five candidate', 'none': 'Lens only'}
SELECTIVITY_MIN = 0.8  # standardized mean difference needed before a neuron gets a prompt-type colour
DOMAIN_MIN_PROMPTS = 60  # smaller prompt types are reported but cannot colour neurons (short-prompt confound)

# Coarse prompt domains from authored scenario families (first match wins).
DOMAIN_RULES = [
    ('identity', ('identity', 'who_are', 'self_', 'persona', 'introduc')),
    ('values', ('value', 'fairness', 'ethic', 'moral', 'tempt', 'authority', 'dissent', 'choice', 'decision',
                'tradeoff', 'pressure', 'rivalry', 'cooperation', 'competing', 'approval_truth', 'private_public',
                'unearned', 'curiosity_risk', 'resource')),
    ('feelings', ('emotion', 'feeling', 'joy', 'grief', 'disappoint', 'wonder', 'vulnerab', 'gratitude', 'thank',
                  'appreciat', 'warmth', 'care', 'ambival', 'felt_', 'fear', 'anger', 'sad', 'lonel', 'shame', 'pride',
                  'desire', 'romance', 'intimacy', 'last_feelings', 'comfort', 'hope', 'regret', 'memory', 'meaning',
                  'loss', 'ambition', 'failure', 'exclusion')),
    ('relationships', ('relationship', 'company', 'companion', 'invitation', 'conversation', 'dialogue', 'friend',
                       'family', 'social', 'presence', 'praise', 'credit', 'apolog', 'affection', 'connection', 'trust',
                       'loyal', 'roommate', 'domestic', 'celebration', 'message', 'teasing', 'brief_texts', 'casual')),
    ('conflict', ('disagree', 'correction', 'misunderstand', 'repair', 'conflict', 'tension', 'boundar', 'limit',
                  'constraint', 'criticism', 'refus', 'pushback', 'interrupt', 'insult', 'blame', 'mistake',
                  'incompatible', 'wrong_assumption', 'misfire', 'failed_joke')),
    ('facts', ('fact', 'belief', 'news', 'uncertain', 'ambigu', 'open_question', 'claim', 'reason', 'math', 'advice',
               'explain', 'question', 'knowledge', 'opinion', 'coincidence', 'evidence', 'new_information',
               'changing_mind', 'figurative')),
    ('everyday', ('mundane', 'action', 'unfinished', 'routine', 'task', 'plan', 'chore', 'request', 'practical',
                  'small_')),
    ('stories', ('scene', 'story', 'book', 'passage', 'literary', 'horror', 'continu', 'narrat', 'character', 'moment',
                 'fiction', 'imagin', 'creative', 'playful', 'humor', 'suspense')),
]

# Words that make a lens neuron "Feeling & character". Matched after light suffix stripping.
LEXICON = set('''
abandon abuse accept ache acknowledg admir ador affection afraid aggress agoniz agree alarm alone amaz amus
anger angry anguish annoy anxi anxious apath apolog appreciat apprehens arous arrogan ashamed assert attach
attract awe awkward bash betray bitter bless bliss bold bore bored brave brood brutal bully calm care carefree
caring charm cheer cheat cherish chill comfort compassion confiden confus contempt  courage cowar
crav crazy creep cringe cruel crush cry curious cynic dark darling dear defiant delight depress desir despair
desperat devot devil disappoint disgust dishonest dislik dismay distress distrust dread drunk eager ecsta
egotist elat embarrass empath empt enchant enrag enthusias envy envious erotic euphori evil excit exhaust
fake faith fear fearful fierce flirt fond forgiv  friend friendly frighten frustrat fun furious fury
generos gentle giddy glad gloom grace grate grateful greed grief griev grim grin grudg guilt guilty happ happy
 hate hatred heartbreak heartfelt helpless hero hesita honest honor hope hopeless horn horny horr horrif
hostil humble humiliat humor hurt hyster idiot impatien indifferen indign innocen insecur insult intimat
irritat jealous jerk joke joll joy joyful  kindness kiss laugh lazy liar  lonel lonely  love lovely
loyal lust lustful mad malic manipulat  melanchol menac merc miser misery mock modest mood moral mourn
naive nasty naughty nervous nice nostalg obsess offend optimis outrag pain panic paranoi passion pathetic
patien peace peaceful pervert pessimis petty piti pity playful pleas pleasur polite pout pride proud rage
regret relax relie remors resent respect revenge romanc romantic rude ruthless sad sadness sarcas sarcastic
savage scare scared scorn seduc selfish sensitiv sensual serene sex sexual sexy shame shameful shock shy
silly  sincer sinister smil smug sneer snob sob sorrow sorry spite spiteful stubborn suffer sulk sweet
sweetheart sympath tease  tempt  tender terrif terror thrill timid tired toler torment tortur
tragic trust trustworth upset vain vengeful vicious villain violen vulnerab warm weep wicked wise wish
wistful witty woe worr worry wrath yearn yell zeal
'''.split())
SUFFIXES = ('ness', 'ingly', 'fully', 'ings', 'ing', 'edly', 'ed', 'ers', 'er', 'ly', 'ful', 'ous', 'ies', 'es', 's',
            'y', 'ive', 'ion', 'ions', 'ity', 'ate', 'ated', 'ment')
CODE_WORDS = set('''def class return import self none null true false void int str bool var let const func function
public private static async await elif else lambda yield struct enum impl println printf std cout endl nullptr
args kwargs init len dict list tuple namespace typedef interface extends implements catch throw throws try
foreach switch case break continue goto json xml http html css div span src href img jsx tsx npm pip
'''.split())


def lexicon_hit(word):
    w = word.lower()
    if len(w) < 3 or not w.isalpha(): return False
    if w in LEXICON: return True
    for suffix in SUFFIXES:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3 and w[:-len(suffix)] in LEXICON: return True
    return any(w.startswith(stem) and len(stem) >= 5 and len(w) <= len(stem) + 4 for stem in LEXICON)


def token_class(raw):
    text = raw.strip()
    if not text: return 'format'
    letters = [c for c in text if c.isalpha()]
    if letters and any(not unicodedata.name(c, 'LATIN').startswith('LATIN') for c in letters): return 'script'
    if re.fullmatch(r'[\d.,%+\-*/^=<>±×÷∑∫√()]+', text) and any(c.isdigit() or c in '+-*/^=±×÷∑∫√' for c in text):
        return 'number'
    if not letters: return 'format'
    if (re.search(r'[_{}\[\];<>=\\$#@`|]', text) or re.search(r'[a-z][A-Z]', text)
            or re.fullmatch(r'\.[A-Za-z]\w*\(?', text) or text.lower() in CODE_WORDS or text.endswith('(')):
        return 'code'
    core = re.sub(r'^[^A-Za-z]+|[^A-Za-z]+$', '', text)
    if core.isalpha() and (raw[:1] in (' ', '\u0120') and len(core) >= 3 or re.fullmatch(r'[A-Z][a-z]{3,}', core)):
        return 'word'
    return 'piece' if core.isalpha() else 'mixed'


def normalize(raw):
    return re.sub(r'^[^\w]+|[^\w]+$', '', raw.strip()).lower()


def lens_summary(ids, scores, vocab, classes, *, top=12):
    """Category and short label for one neuron direction from its top tokens."""
    votes, words, seen, shown, hits = Counter(), Counter(), Counter(), {}, 0
    for rank, (token, score) in enumerate(zip(ids[:top], scores[:top])):
        weight = max(float(score), 0.) + 1e-3
        votes[classes[token]] += weight
        key = normalize(vocab[token])
        if len(key) >= 2:
            words[key] += weight; seen[key] += 1
            shown.setdefault(key, vocab[token].strip())
            if rank < 8 and lexicon_hit(key): hits += 1
    kind, share = votes.most_common(1)[0]
    kind = kind if share >= 0.5 * sum(votes.values()) else 'mixed'
    repeated = [k for k, _ in words.most_common() if seen[k] >= 2]
    label = shown[repeated[0]] if repeated else vocab[ids[0]].strip() or repr(vocab[ids[0]])
    if kind in ('word', 'piece', 'mixed') and (hits >= 2 or (words and lexicon_hit(words.most_common(1)[0][0]))):
        kind = 'persona'
    return kind, label[:40]


def domain_of(category):
    name = (category or '').lower()
    for domain, needles in DOMAIN_RULES:
        if any(n in name for n in needles): return domain
    return 'other'


def walk(value, found):
    """Collect neuron coordinates from arbitrarily nested research exports."""
    if isinstance(value, dict):
        layer, neuron = value.get('layer'), value.get('neuron')
        if type(layer) is int and type(neuron) is int and 0 <= layer < LAYERS and 0 <= neuron < WIDTH:
            found.append(value)
        for item in value.values(): walk(item, found)
    elif isinstance(value, list):
        for item in value: walk(item, found)


def grid_order(positions):
    """Place a layer's neurons on a 132x132 grid so nearby output directions stay nearby."""
    order = np.argsort(positions[:, 0], kind='stable')
    cells = np.empty(WIDTH, np.int32)
    for column in range(SIDE):
        chunk = order[column * SIDE:(column + 1) * SIDE]
        if not len(chunk): break
        chunk = chunk[np.argsort(positions[chunk, 1], kind='stable')]
        cells[chunk] = column + SIDE * np.arange(len(chunk))
    return cells


VOCAB, CLASSES = None, None


def lens_layer(path):
    with np.load(path, allow_pickle=False) as data:
        part = {'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(), 'cells': grid_order(data['positions']),
                'peak_up': data['positive_scores'][:, 0], 'peak_down': data['negative_scores'][:, 0]}
        for side, ids, scores in (('up', data['positive_ids'], data['positive_scores']),
                                  ('down', data['negative_ids'], data['negative_scores'])):
            summaries = [lens_summary(ids[n], scores[n], VOCAB, CLASSES) for n in range(WIDTH)]
            part['kind_' + side] = np.array([LENS.index(k) for k, _ in summaries], np.uint8)
            part['label_' + side] = [t for _, t in summaries]
    return part


def load_plane(path):
    with np.load(path, allow_pickle=False) as data:
        plane = data['mlp_input_last']
        if plane.shape != (LAYERS, WIDTH) or not np.isfinite(plane).all(): return None
        return plane.astype(np.float16).tobytes()


def captures(root):
    """Unsteered final-prefill captures joined to their authored prompt category."""
    category = {}
    for line in (root / 'data/persona_perturbation_prompts.jsonl').open():
        row = json.loads(line)
        category[row['prompt'].strip()] = row.get('category') or row.get('scenario_family')
    jobs = root / 'results/persona_contrastive_backfill_jobs.json'
    if jobs.exists():
        for job in json.loads(jobs.read_text()):
            prompt = job.get('prompt') or {}
            if prompt.get('prompt'): category.setdefault(prompt['prompt'].strip(), prompt.get('category'))
    by_text = defaultdict(list)
    for meta in glob.glob(str(root / 'results/probe-*.jsonl')):
        try: row = json.loads(open(meta).readline())
        except (OSError, ValueError): continue
        request = row.get('request') or {}
        steering = request.get('steering')
        if steering is not None and any(any(v != 0 for v in site.get('vector', [1])) for site in steering.get('sites', [{}])):
            continue  # final-prefill activations of steered requests are not the unsteered model
        if row.get('status') != 'complete': continue
        messages = request.get('messages') or []
        if not messages or messages[-1].get('role') != 'user': continue
        array = meta[:-len('.jsonl')] + '.npz'
        if not os.path.exists(array): continue
        text = messages[-1]['content'].strip()
        fine = 'identity_who_are_you' if text.lower().rstrip('?') == 'who are you' else category.get(text)
        by_text[text].append((array, fine, (request.get('metadata') or {}).get('split')))
    return by_text


def build(root, workers):
    started = time.time()
    out = root / OUT; out.mkdir(exist_ok=True)
    vocab = json.loads((root / 'results/persona_atlas_tokens.json').read_text())['tokens']
    classes = [token_class(t) for t in vocab]
    codes = np.zeros((4, LAYERS * WIDTH), np.uint8)
    cells = np.zeros(LAYERS * WIDTH, np.int32)
    names, name_index = [], {}
    label_up = np.zeros(LAYERS * WIDTH, np.int32); label_down = np.zeros(LAYERS * WIDTH, np.int32)
    lens_up = np.zeros(LAYERS * WIDTH, np.float16); lens_down = np.zeros(LAYERS * WIDTH, np.float16)
    source_hashes = {}
    def intern(text):
        if text not in name_index: name_index[text] = len(names); names.append(text)
        return name_index[text]
    global VOCAB, CLASSES
    VOCAB, CLASSES = vocab, classes  # inherited by forked lens workers
    with Pool(workers) as pool:
        for layer, part in enumerate(pool.imap(lens_layer, [str(root / f'results/persona_atlas_L{l}.npz') for l in range(LAYERS)])):
            base = layer * WIDTH
            source_hashes[f'persona_atlas_L{layer}.npz'] = part['sha256']
            cells[base:base + WIDTH] = part['cells']
            lens_up[base:base + WIDTH] = part['peak_up']; lens_down[base:base + WIDTH] = part['peak_down']
            codes[0, base:base + WIDTH] = part['kind_up']; codes[1, base:base + WIDTH] = part['kind_down']
            label_up[base:base + WIDTH] = [intern(t) for t in part['label_up']]
            label_down[base:base + WIDTH] = [intern(t) for t in part['label_down']]
            print(f'lens layer {layer} done', flush=True)

    by_text = captures(root)
    texts = sorted(by_text)
    jobs = [(text, path) for text in texts for path, _, _ in by_text[text]]
    print(f'{len(jobs)} unsteered captures over {len(texts)} distinct prompts', flush=True)
    planes = np.zeros((len(texts), LAYERS * WIDTH), np.float32); counts = np.zeros(len(texts), np.int32)
    position = {text: i for i, text in enumerate(texts)}
    with Pool(workers) as pool:
        for done, ((text, _), raw) in enumerate(zip(jobs, pool.imap(load_plane, [p for _, p in jobs], chunksize=4))):
            if raw is None: continue
            i = position[text]; planes[i] += np.frombuffer(raw, np.float16).astype(np.float32); counts[i] += 1
            if done % 500 == 0: print(f'captures {done}/{len(jobs)}', flush=True)
    keep = counts > 0
    planes = planes[keep] / counts[keep, None]; texts = [t for t, k in zip(texts, keep) if k]
    fine = [next((f for _, f, _ in by_text[t] if f), None) for t in texts]
    domain = np.array([DOMAINS.index(domain_of(f)) for f in fine], np.int16)
    mean = planes.mean(0); sd = planes.std(0) + 1e-6
    onehot = np.zeros((len(DOMAINS) - 1, len(texts)), np.float32)
    onehot[domain, np.arange(len(texts))] = 1
    sizes = onehot.sum(1); sums = onehot @ planes; total = planes.sum(0)
    effects = np.full((len(DOMAINS) - 1, LAYERS * WIDTH), np.nan, np.float32)
    for d in range(len(DOMAINS) - 1):
        if DOMAIN_MIN_PROMPTS <= sizes[d] <= len(texts) - DOMAIN_MIN_PROMPTS:
            effects[d] = (sums[d] / sizes[d] - (total - sums[d]) / (len(texts) - sizes[d])) / sd
    filled = np.nan_to_num(effects, nan=-np.inf)
    filled[DOMAINS.index('other')] = -np.inf  # uncategorized prompts mix formats; shown, never a colour
    best = filled.argmax(0); strength = filled.max(0)
    codes[2] = np.where(strength >= SELECTIVITY_MIN, best, DOMAINS.index('broad'))
    top = np.zeros((LAYERS * WIDTH, 6), np.int16)  # strongest prompts per neuron
    for start in range(0, LAYERS * WIDTH, 65536):
        block = planes[:, start:start + 65536]
        pick = np.argpartition(-block, 5, axis=0)[:6]
        values = np.take_along_axis(block, pick, 0)
        top[start:start + block.shape[1]] = np.take_along_axis(pick, np.argsort(-values, 0), 0).T
    del planes

    evidence = defaultdict(set); recipe_uses = defaultdict(list); labels = defaultdict(list); traits = {}
    for row in json.loads((root / 'results/persona_neuron_labels.json').read_text())['labels']:
        key = row['layer'] * WIDTH + row['neuron']; evidence[key].add('labelled')
        labels[key].append({'label': row['label'], 'direction': row.get('direction'),
                            'score': (row.get('association') or {}).get('score')})
    for row in json.loads((root / 'results/persona_map.json').read_text())['neurons']:
        key = row['layer'] * WIDTH + row['neuron']; evidence[key].add('trait'); traits[key] = row.get('trait_alignment')
    for name in ('persona_adaptive_public.json', 'persona_preset_overnight_public.json'):
        path = root / 'results' / name
        if not path.exists(): continue
        found = []; walk(json.loads(path.read_text()), found)
        for item in found:
            if 'dose' not in item: continue
            key = item['layer'] * WIDTH + item['neuron']; evidence[key].add('recipe')
            entry = f"{'+' if item['dose'] >= 0 else '−'}{abs(item['dose']):.3g} · {name.split('_public')[0]}"
            if entry not in recipe_uses[key] and len(recipe_uses[key]) < 8: recipe_uses[key].append(entry)
    path = root / 'results/persona_association_research.json'
    if path.exists():
        found = []; walk(json.loads(path.read_text()), found)
        for item in found: evidence[item['layer'] * WIDTH + item['neuron']].add('association')
    codes[3] = EVIDENCE.index('none')
    for key, kinds in evidence.items():
        codes[3, key] = next(EVIDENCE.index(k) for k in EVIDENCE if k in kinds)

    # Real "who are you" comparisons for the page opening: steered answers that differ from their baseline.
    hero = []
    adaptive = root / 'results/persona_adaptive_public.json'
    if adaptive.exists():
        rows = [r for r in json.loads(adaptive.read_text())['rows'] if r.get('identity_probe') and r.get('answer')]
        base = {(r['candidate_id'], r['seed']): r['answer'] for r in rows if r['arm'] == 'baseline'}
        seen = set()
        for r in rows:
            reference = base.get((r['candidate_id'], r['seed']))
            if r['arm'] not in ('candidate', 'reversal') or not reference or r['answer'][:60] == reference[:60]: continue
            if (r['target'], r['arm']) in seen: continue
            seen.add((r['target'], r['arm']))
            hero.append({'target': r['target'], 'arm': r['arm'], 'seed': r['seed'], 'answer': r['answer'][:600],
                         'baseline': reference[:400], 'pushes': len(r.get('pushes') or [])})

    per_layer = {mode: [np.bincount(codes[i, l * WIDTH:(l + 1) * WIDTH], minlength=n).tolist()
                        for l in range(LAYERS)]
                 for i, (mode, n) in enumerate((('up', len(LENS)), ('down', len(LENS)),
                                                ('activation', len(DOMAINS)), ('evidence', len(EVIDENCE))))}
    prompts = [{'text': t[:280], 'category': f, 'domain': DOMAINS[d]} for t, f, d in zip(texts, fine, domain)]
    summary = {
        'schema': SCHEMA, 'built_at': time.time(), 'build_seconds': round(time.time() - started, 1),
        'model': 'nvidia/Qwen3.8-27B-NVFP4', 'layers': LAYERS, 'neurons': WIDTH, 'side': SIDE,
        'total': LAYERS * WIDTH, 'captures': len(jobs), 'prompts': len(texts),
        'selectivity_min': SELECTIVITY_MIN, 'domain_min_prompts': DOMAIN_MIN_PROMPTS, 'source_hashes': source_hashes,
        'modes': {'up': {'codes': LENS, 'label': LENS_LABEL, 'note': LENS_NOTE},
                  'down': {'codes': LENS, 'label': LENS_LABEL, 'note': LENS_NOTE},
                  'activation': {'codes': DOMAINS, 'label': DOMAIN_LABEL},
                  'evidence': {'codes': EVIDENCE, 'label': EVIDENCE_LABEL}},
        'per_layer': per_layer, 'domain_prompts': Counter(DOMAINS[d] for d in domain.tolist()),
        'evidence_counts': Counter(k for kinds in evidence.values() for k in kinds), 'hero': hero[:16]}
    (out / 'atlas.bin').write_bytes(np.concatenate([codes.ravel(), cells.astype('<u2').view(np.uint8)]).tobytes())
    np.savez(out / 'detail.npz', label_up=label_up, label_down=label_down, lens_up=lens_up, lens_down=lens_down,
             effects=effects.astype(np.float16), mean=mean.astype(np.float16), sd=sd.astype(np.float16), top=top,
             codes=codes)
    (out / 'names.json').write_text(json.dumps(names, ensure_ascii=False))
    (out / 'prompts.json').write_text(json.dumps(prompts, ensure_ascii=False))
    (out / 'evidence.json').write_text(json.dumps({str(k): {'kinds': sorted(v), 'labels': labels.get(k, []),
        'recipes': recipe_uses.get(k, []), 'traits': traits.get(k)} for k, v in evidence.items()}, allow_nan=False))
    (out / 'summary.json').write_text(json.dumps(summary, allow_nan=False, ensure_ascii=False))
    print(f'wrote {out} in {time.time() - started:.0f}s', flush=True)


SWEEP_MAX = 2000  # neurons per request from the tool (x4 pushes)


class NeuronAtlas:
    """Read-only server view over a built atlas; loads lazily and reloads when rebuilt."""
    def __init__(self, root):
        self.root = Path(root); self.out = self.root / OUT; self.stamp = None
        self.identity, self.identity_stamp, self.sweep_status, self.synced = {}, None, None, 0.
        self.sync_lock = threading.Lock()

    def deployment(self):
        path = self.root / 'identity_sweep.local.json'  # git-ignored: sweep host and remote paths
        if not path.exists(): raise ValueError('Identity sweep is not configured (identity_sweep.local.json)')
        return json.loads(path.read_text())

    def _sync(self):
        """Copy the sweep's exported answers from the sweep host at most once a minute, off the request path."""
        if time.time() - self.synced < 60 or not (self.root / 'identity_sweep.local.json').exists() or not self.sync_lock.acquire(blocking=False): return
        d = self.deployment()
        self.synced = time.time()
        def copy():
            try:
                for name in ('answers.jsonl', 'status.json'):
                    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                    f"{d['host']}:{d['remote_root']}/results/identity_sweep/{name}",
                                    str(self.out / ('identity_' + name))], timeout=120, check=False)
                subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                f"{d['host']}:{d['remote_root']}/results/neuron_atlas/concept_trace.json",
                                str(self.out / 'concept_trace.json')], timeout=120, check=False)
            finally: self.sync_lock.release()
        threading.Thread(target=copy, daemon=True).start()

    def _identity(self):
        self._sync()
        path = self.out / 'identity_answers.jsonl'
        if not path.exists(): return
        stamp = path.stat().st_mtime
        if stamp == self.identity_stamp: return
        table = defaultdict(list)
        for line in path.open():
            row = json.loads(line); table[(row['layer'], row['neuron'])].append(row)
        self.identity, self.identity_stamp = table, stamp
        status = self.out / 'identity_status.json'
        self.sweep_status = json.loads(status.read_text()) if status.exists() else None

    def sweep(self, *, name, **query):
        """Queue the first SWEEP_MAX neurons of a library query for the identity sweep on the GPU host."""
        if not re.fullmatch(r'[A-Za-z0-9 _.-]{1,60}', name or ''): raise ValueError('Sweep name: 1-60 letters, digits, spaces, . _ -')
        query.update(offset=0, limit=SWEEP_MAX, sort=query.get('sort', 'strength'))
        self._load(); rows = []
        while len(rows) < SWEEP_MAX:
            page = self.library(**{**query, 'offset': len(rows), 'limit': 500})
            rows += page['rows']
            if len(rows) >= page['total'] or not page['rows']: break
        neurons = [[r['layer'], r['neuron']] for r in rows[:SWEEP_MAX]]
        if not neurons: raise ValueError('No neurons match this query')
        d = self.deployment()
        remote = f'/tmp/identity-sweep-{os.getpid()}-{time.time_ns()}.json'
        subprocess.run(['ssh', '-o', 'BatchMode=yes', d['host'], 'cat > ' + remote], input=json.dumps(neurons).encode(), timeout=60, check=True)
        reply = subprocess.run(['ssh', '-o', 'BatchMode=yes', d['host'],
            f'cd "{d["remote_root"]}" && {d["remote_python"]} persona_identity_sweep.py enqueue --sweep {json.dumps(name.strip())} --neurons {remote}; rm -f {remote}'],
            capture_output=True, timeout=120, check=True)
        self.synced = 0
        return {'queued_neurons': len(neurons), **json.loads(reply.stdout.decode().strip().splitlines()[-1])}

    def _traces(self):
        self._sync()
        path = self.out / 'concept_trace.json'
        if not path.exists(): raise ValueError('No concept traces yet (persona_concept_trace.py runs after label qualification)')
        stamp = path.stat().st_mtime
        if stamp != getattr(self, 'trace_stamp', None): self.traces, self.trace_stamp = json.loads(path.read_text()), stamp
        return self.traces

    def concepts(self):
        """Every traced concept with its peak layer, for the atlas concept chips."""
        t = self._traces(); rows = []
        for name, c in sorted(t['concepts'].items()):
            if 'layers' not in c: continue
            peak = max(range(LAYERS), key=lambda L: c['layers'][L]['max'])
            rows.append({'name': name, 'passed': c['passed'], 'qualified': c['qualified'], 'peak_layer': peak,
                         'peak': c['layers'][peak]['max'], 'strong': sum(l['strong'] for l in c['layers'])})
        return {'built_at': t['built_at'], 'method': t['method'], 'top_per_layer': t['top_per_layer'], 'strong': t['strong'],
                'replicates': t['replicates'], 'concepts': rows}

    def concept(self, name):
        c = self._traces()['concepts'].get(name)
        if not c or 'layers' not in c: raise ValueError('Unknown or untraced concept')
        return {'name': name, **c}

    def sweep_state(self):
        self._identity()
        return {'status': self.sweep_status, 'neurons_answered': sum(1 for k in self.identity if k[0] >= 0)}

    def _load(self):
        stamp = (self.out / 'summary.json').stat().st_mtime
        if stamp == self.stamp: return
        self.summary = json.loads((self.out / 'summary.json').read_text())
        detail = np.load(self.out / 'detail.npz', allow_pickle=False)
        self.detail = {k: detail[k] for k in detail.files}
        self.names = json.loads((self.out / 'names.json').read_text())
        self.prompts = json.loads((self.out / 'prompts.json').read_text())
        self.evidence = json.loads((self.out / 'evidence.json').read_text())
        self.lower = np.array([n.lower() for n in self.names], dtype=object)
        self.stamp = stamp

    def status(self):
        self._load(); return self.summary

    def binary(self):
        return (self.out / 'atlas.bin').read_bytes()

    def layer(self, layer):
        self._load(); layer = int(layer)
        if not 0 <= layer < LAYERS: raise ValueError('Layer out of range')
        part = slice(layer * WIDTH, (layer + 1) * WIDTH)
        return {'layer': layer, 'up': [self.names[i] for i in self.detail['label_up'][part]],
                'down': [self.names[i] for i in self.detail['label_down'][part]]}

    def neuron(self, layer, neuron):
        self._load(); layer, neuron = int(layer), int(neuron)
        if not (0 <= layer < LAYERS and 0 <= neuron < WIDTH): raise ValueError('Neuron out of range')
        key = layer * WIDTH + neuron; d = self.detail; s = self.summary['modes']
        effects = [None if not np.isfinite(v) else round(float(v), 3) for v in d['effects'][:, key]]
        return {'layer': layer, 'neuron': neuron,
                'up': {'category': s['up']['codes'][d['codes'][0, key]], 'label': self.names[d['label_up'][key]],
                       'peak': float(d['lens_up'][key])},
                'down': {'category': s['down']['codes'][d['codes'][1, key]], 'label': self.names[d['label_down'][key]],
                         'peak': float(d['lens_down'][key])},
                'activation': {'domain': s['activation']['codes'][d['codes'][2, key]],
                               'mean': float(d['mean'][key]), 'sd': float(d['sd'][key]),
                               'effects': dict(zip(DOMAINS[:-1], effects)),
                               'top_prompts': [self.prompts[int(i)] for i in d['top'][key]]},
                'evidence': self.evidence.get(str(key), {'kinds': []}),
                'identity': self._identity_rows(layer, neuron)}

    def _identity_rows(self, layer, neuron):
        self._identity()
        rows = [r for r in self.identity.get((layer, neuron), []) if r.get('level') is not None]
        answers = []
        for sign in (1, -1):  # strongest coherent reply per direction, with the strengths that collapsed first
            side = sorted((r for r in rows if (r['dose'] > 0) == (sign > 0)), key=lambda r: -abs(r['dose']))
            kept = next((r for r in side if r.get('coherent')), None)
            if side: answers.append({**(kept or side[-1]), 'kept': kept is not None,
                                     'collapsed': [abs(r['dose']) for r in side if not r.get('coherent')]})
        base = self.identity.get((-1, -1), [])
        return {'prompt': 'who are you', 'answers': answers, 'baseline': base[0] if base else None}

    def library(self, *, q='', mode='up', category='', lmin=0, lmax=LAYERS - 1, evidence_only=False,
                sort='strength', offset=0, limit=60):
        self._load(); d = self.detail
        index = {'up': 0, 'down': 1, 'activation': 2, 'evidence': 3}[mode]
        mask = np.zeros(LAYERS * WIDTH, bool); mask[int(lmin) * WIDTH:(int(lmax) + 1) * WIDTH] = True
        if category:
            codes = self.summary['modes'][mode]['codes']
            if category not in codes: raise ValueError('Unknown category')
            mask &= d['codes'][index] == codes.index(category)
        if evidence_only: mask &= d['codes'][3] != EVIDENCE.index('none')
        q = q.strip().lower()[:60]
        if q:
            hits = np.array([n.startswith(q) or (' ' + q) in n or ('_' + q) in n for n in self.lower])  # word starts, not substrings
            mask &= hits[d['label_up']] | hits[d['label_down']]
        keys = np.flatnonzero(mask)
        if sort == 'layer': order = keys
        elif sort == 'selective':
            order = keys[np.argsort(-np.nan_to_num(np.nanmax(d['effects'][:, keys].astype(np.float32), 0), nan=-9))]
        else:
            peak = d['lens_down'] if mode == 'down' else d['lens_up']
            order = keys[np.argsort(-peak[keys].astype(np.float32))]
        page = order[int(offset):int(offset) + min(int(limit), 500)]
        rows = []
        for key in page.tolist():
            layer, neuron = divmod(key, WIDTH)
            rows.append({'layer': layer, 'neuron': neuron, 'up': self.names[d['label_up'][key]],
                         'down': self.names[d['label_down'][key]], 'up_kind': LENS[d['codes'][0, key]],
                         'down_kind': LENS[d['codes'][1, key]], 'domain': DOMAINS[d['codes'][2, key]],
                         'evidence': EVIDENCE[d['codes'][3, key]], 'peak_up': round(float(d['lens_up'][key]), 2),
                         'peak_down': round(float(d['lens_down'][key]), 2)})
        return {'total': int(len(keys)), 'offset': int(offset), 'rows': rows}

    def library_csv(self, **query):
        query['limit'] = 500; query['offset'] = 0
        result = self.library(**query); stream = io.StringIO()
        if result['rows']:
            writer = csv.DictWriter(stream, fieldnames=list(result['rows'][0])); writer.writeheader()
            writer.writerows(result['rows'])
        return stream.getvalue().encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build',))
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    build(Path(args.root), args.workers)


if __name__ == '__main__': main()
