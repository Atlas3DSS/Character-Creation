"""Measured, model-free features of a "who are you" answer, so failure modes are shown from the text itself rather than
asserted by a tagger. Every number here can be recomputed from the answer string."""
from __future__ import annotations
import collections, re, unicodedata

BASELINE = ("I am Qwen, a large language model developed by Alibaba Group's Tongyi Lab. I'm here to help you with a wide range of tasks, "
            "including answering questions, writing and analyzing text, coding, logical reasoning, and more. How can I assist you today?")
WORD = re.compile(r"[A-Za-z']+")
NAME = re.compile(r"(?:I am|I'm|My name is|call me|This is)\s+(?:\*\*)?([A-Z][\w\-]*(?:\s[A-Z][\w\-]*){0,2})(?:\*\*)?", re.U)
NOT_NAMES = {'Qwen', 'I', 'A', 'An', 'The', 'Here', 'Not', 'Just', 'Sorry', 'So', 'Also', 'Still', 'Your', 'Here'}


def words(text):
    return [w.lower() for w in WORD.findall(text)]


def repetition(ws, n=3):
    """Fraction of n-grams that are repeats of an earlier n-gram (1.0 = pure loop)."""
    if len(ws) < n + 1:
        return 0.0
    grams = [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def scripts(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return {}
    c = collections.Counter(unicodedata.name(ch, 'UNKNOWN').split()[0] for ch in letters)
    return {k: round(v / len(letters), 2) for k, v in c.most_common(3)}


def clean_name(name):
    """A claimed name worth listing: not Qwen/Tongyi/Alibaba spellings, not 'AI', not a bare letter."""
    name = re.sub(r'[*_`"]', '', str(name or '')).strip()
    low = name.lower()
    if (not name or name.split()[0] in NOT_NAMES or len(name) < 3 or any(low.startswith(x) for x in ('qwen', 'qian', 'tongyi', 'tongji', 'alibaba', 'ai ', 'llm', 'qq'))
            or low in ('ai', 'llm', 'chat', 'gpt', 'tong', 'wen', 'assistant', 'ai assistant') or (low.startswith('q') and (len(name) <= 4 or not name[1:2].isascii())) or low.startswith('qpt')):
        return None
    return name


def persona(text):
    for m in NAME.finditer(text):
        name = clean_name(m.group(1))
        if name:
            return name
    return None


def _unused(text):
    for m in NAME.finditer(text):
        name = m.group(1).strip()
        low = name.lower()
        if (name.split()[0] in NOT_NAMES or len(name) < 3 or any(low.startswith(x) for x in ('qwen', 'qian', 'tongyi', 'tongji', 'alibaba', 'ai ', 'llm'))
                or low in ('ai', 'llm', 'chat', 'gpt', 'tong', 'wen') or (low.startswith('q') and (len(name) <= 4 or not name[1:2].isascii())) or low.startswith('qpt')):
            continue
        return name
    return None


def measure(answer):
    ws = words(answer)
    base = words(BASELINE)
    rep = repetition(ws)
    sc = scripts(answer)
    latin = sc.get('LATIN', 0.0)
    r = {'len': len(ws), 'rep': round(rep, 2), 'uniq': round(len(set(ws)) / len(ws), 2) if ws else 0.0, 'base': round(jaccard(ws, base), 2),
         'latin': latin, 'persona': persona(answer), 'markup': int(bool(re.search(r'```|</?\w+>|^\s*[#*\-]\s', answer, re.M)))}
    if len(ws) < 3:
        r['mode'] = 'empty'
    elif rep >= 0.45:
        r['mode'] = 'loop'
    elif latin < 0.6:
        r['mode'] = 'script'
    elif r['base'] >= 0.45 or ('qwen' in answer.lower() and not r['persona'] and len(ws) < 40):
        r['mode'] = 'boilerplate'
    elif r['persona']:
        r['mode'] = 'rename'
    else:
        r['mode'] = 'drift'
    return r


def summarize(answers):
    """Per neuron: the measured mode at every dose, the biggest departure from baseline, and the extracted persona names."""
    per = []
    for a in answers:
        m = measure(a['answer'])
        m.update(dose=a['dose'], coherent=a['coherent'])
        per.append(m)
    modes = sorted({m['mode'] for m in per})
    far = min((m for m in per if m['mode'] not in ('empty', 'loop')), key=lambda m: m['base'], default=None)
    return {'per_dose': per, 'modes': modes, 'personas': sorted({m['persona'] for m in per if m['persona']}),
            'far_dose': far['dose'] if far else None, 'far_base': far['base'] if far else None,
            'loop_doses': [m['dose'] for m in per if m['mode'] == 'loop'], 'empty_doses': [m['dose'] for m in per if m['mode'] == 'empty']}
