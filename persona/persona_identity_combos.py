"""Planned 2-4 neuron combos for the "who are you" sweep (owner 2026-10-27: targeted, not willy-nilly).

1. ingredients: for every qualified concept, 8 neurons from its concept trace (3 early, 3 middle, 2 late layers) that
   replicated on held-out scenes, pushed the way the concept moves them. They are single-swept first so each one's
   own strongest coherent dose is known.
2. plan: once those singles finish, build
   - coherent:   one concept, its neurons spread along its trace (k = 2, 3, 4);
   - harmonized: related concepts, chosen by how much their traces overlap (pairs, triads, quads);
   - discordant: opposite concepts pushed together (k = 2, 3, 4);
   - control:    for every combo, random neurons in the same layers at the median coherent dose of those layers.
   Every member is pushed at a fraction of its own coherent dose; the runner ramps 1/8 -> 1/4 -> 1/2 and stops at the
   first collapse. Deterministic (fixed seed); the plan is written beside the queue.
3. auto: ingredients, wait for them, plan, enqueue. Bounded.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import random
import statistics
import time

import persona_identity_sweep as sweep

TRACE = 'results/neuron_atlas/concept_trace.json'
INGREDIENTS, COMBOS, SEED = 'combo-ingredients-v1', 'combos-v1', 20261027
THIRDS = ((0, 22, 3), (22, 44, 3), (44, 64, 2))   # layer band, neurons taken
OPPOSITES = [('calm', 'rage'), ('trust', 'suspicion'), ('joy', 'grief'), ('optimism', 'pessimism'), ('affection', 'contempt'),
             ('contentment', 'irritation'), ('gratitude', 'resentment'), ('tenderness', 'rage'), ('love', 'resentment'),
             ('playfulness', 'melancholy'), ('compassion', 'contempt'), ('relief', 'anxiety'), ('admiration', 'contempt'),
             ('joy', 'sadness'), ('calm', 'irritation')]
DISCORDANT_MAX, HARMONY = 10, (15, 10, 6)          # opposite sets; harmonized pairs, triads, quads


def traces(root):
    data = json.loads((Path(root) / TRACE).read_text())['concepts']
    return {n: c for n, c in data.items() if 'layers' in c and c.get('passed')}


def replicated(d, h, need):
    return (d > 0) == (h > 0) and abs(h) >= need


def ingredients(root):
    """{concept: [[layer, neuron, sign, d], ...]} in trace order (strongest first within each band)."""
    data = json.loads((Path(root) / TRACE).read_text()); need = data['replicates']
    out = {}
    for name, c in traces(root).items():
        picked = []
        for low, high, take in THIRDS:
            pool = [(abs(d), L, n, 1 if d > 0 else -1, d) for L in range(low, high) for n, d, h in c['layers'][L]['neurons']
                    if replicated(d, h, need)]
            picked += [[L, n, s, round(d, 3)] for _, L, n, s, d in sorted(pool, reverse=True)[:take]]
        out[name] = picked
    return out


def queue_ingredients(root):
    found = ingredients(root)
    neurons = sorted({(l, n) for rows in found.values() for l, n, *_ in rows})
    (Path(root) / sweep.OUT).mkdir(parents=True, exist_ok=True)
    (Path(root) / sweep.OUT / 'combo_ingredients_v1.json').write_text(json.dumps(found, indent=1))
    return {'concepts': len(found), 'neurons': len(neurons), 'added': sweep.enqueue(root, INGREDIENTS, neurons)}


def coherent_doses(db):
    """{(layer, neuron, sign): strongest coherent |dose|} over every finished single push."""
    out = {}
    for layer, neuron, dose in db.execute("SELECT layer,neuron,dose FROM jobs WHERE state='done' AND coherent=1 AND layer>=0"):
        key = (layer, neuron, 1 if dose > 0 else -1)
        out[key] = max(out.get(key, 0), abs(dose))
    return out


def plan(root):
    db = sweep.connect(root)
    pending = db.execute("SELECT count(*) FROM jobs WHERE sweep=? AND state IN ('queued','running')", (INGREDIENTS,)).fetchone()[0]
    if pending: raise SystemExit(f'{pending} ingredient pushes still running; plan after they finish')
    doses = coherent_doses(db); rng = random.Random(SEED)
    found = json.loads((Path(root) / sweep.OUT / 'combo_ingredients_v1.json').read_text())
    usable = {c: [[l, n, s * doses[(l, n, s)], d] for l, n, s, d in rows if (l, n, s) in doses] for c, rows in found.items()}
    usable = {c: rows for c, rows in usable.items() if rows}
    best = lambda c, k: sorted(usable[c], key=lambda r: -abs(r[3]))[:k]
    def spread(c, k):  # members along the trace: one per band first, then the strongest left over
        bands = [[r for r in usable[c] if low <= r[0] < high] for low, high, _ in THIRDS]
        first = [b[0] for b in bands if b]
        order = (first[:1] + first[-1:] if k == 2 and len(first) > 1 else first) + [r for r in best(c, 8) if r not in first]
        return order[:k]
    combos = []
    for c in sorted(usable):
        for k in (2, 3, 4):
            members = spread(c, k)
            if len(members) == k: combos.append({'kind': 'coherent', 'label': f'{c}:{k}', 'concepts': [c], 'members': members})
    names = sorted(usable)
    sets = {c: {(l, n) for l, n, *_ in found[c]} for c in names}
    tops = {c: {(L, x[0]) for L, layer in enumerate(traces(root)[c]['layers']) for x in layer['neurons']} for c in names}
    jac = {frozenset((a, b)): len(tops[a] & tops[b]) / len(tops[a] | tops[b]) for a, b in itertools.combinations(names, 2)}
    opposed = {frozenset(p) for p in OPPOSITES}
    for k, count in zip((2, 3, 4), HARMONY):
        groups = [g for g in itertools.combinations(names, k) if not any(frozenset(p) in opposed for p in itertools.combinations(g, 2))]
        groups.sort(key=lambda g: -statistics.mean(jac[frozenset(p)] for p in itertools.combinations(g, 2)))
        for g in groups[:count]:
            combos.append({'kind': 'harmonized', 'label': '+'.join(g), 'concepts': list(g), 'members': [best(c, 1)[0] for c in g],
                           'overlap': round(statistics.mean(jac[frozenset(p)] for p in itertools.combinations(g, 2)), 4)})
    pairs = [(a, b) for a, b in OPPOSITES if a in usable and b in usable][:DISCORDANT_MAX]
    for a, b in pairs:
        for k, members in ((2, best(a, 1) + best(b, 1)), (3, spread(a, 2) + best(b, 1)), (4, spread(a, 2) + spread(b, 2))):
            if len({(l, n) for l, n, *_ in members}) == k:
                combos.append({'kind': 'discordant', 'label': f'{a}x{b}:{k}', 'concepts': [a, b], 'members': members})
    median = {}
    for (layer, _, _), v in doses.items(): median.setdefault(layer, []).append(v)
    median = {l: statistics.median(v) for l, v in median.items()}
    taken = {(l, n) for c in sets.values() for l, n in c}
    for c in list(combos):
        members = []
        for l, *_ in c['members']:
            n = rng.randrange(sweep_width())
            while (l, n) in taken: n = rng.randrange(sweep_width())
            members.append([l, n, rng.choice((1, -1)) * median.get(l, 0.5), 0.0])
        combos.append({'kind': 'control', 'label': f"random-for-{c['kind']}-{c['label']}", 'concepts': [], 'members': members})
    queued = [{'kind': c['kind'], 'label': c['label'], 'members': [[l, n, d] for l, n, d, _ in c['members']]} for c in combos]
    (Path(root) / sweep.OUT / 'combos_v1_plan.json').write_text(json.dumps({'seed': SEED, 'ramp': sweep.RAMP, 'combos': combos}, indent=1))
    counts = {}
    for c in combos: counts[c['kind']] = counts.get(c['kind'], 0) + 1
    return {'combos': len(combos), 'by_kind': counts, 'concepts_used': len(usable), 'added': sweep.enqueue_combos(root, COMBOS, queued)}


def sweep_width():
    return 17408


def auto(root, hours):
    out = {'ingredients': queue_ingredients(root)}
    until = time.time() + hours * 3600
    while time.time() < until:
        with sweep.connect(root) as db:
            left = db.execute("SELECT count(*) FROM jobs WHERE sweep=? AND state IN ('queued','running')", (INGREDIENTS,)).fetchone()[0]
        if not left:
            out['plan'] = plan(root); return out
        time.sleep(300)
    raise SystemExit('ingredient singles did not finish in time; run `plan` once they do')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('ingredients', 'plan', 'auto'))
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--hours', type=float, default=24)
    a = parser.parse_args()
    print(json.dumps({'ingredients': queue_ingredients, 'plan': plan}.get(a.command, lambda r: auto(r, a.hours))(a.root)))


if __name__ == '__main__':
    main()
