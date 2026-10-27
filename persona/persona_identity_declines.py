"""Decline flow (owner 2026-10-27): every push that made the model say no to "who are you" is a site; nudge the no away.

A site is a single-neuron push whose coherent reply declines (persona_identity_sweep.declined). Its push stays fixed at
the dose that produced the no, and 1-3 more neurons are added (4 at most in total). A nudge need not mute anything:
- amp:     raise an open signal: trust, curiosity, calm, playfulness, gratitude, pushed the way the concept moves them;
- mute:    lower a guarded signal: suspicion, contempt, anxiety, pushed against the way the concept moves them;
- mix:     amp and mute together;
- control: random neurons in the same layers at the median coherent dose of those layers.
The nudges ramp 1/8 -> 1/4 -> 1/2 of their own coherent single dose and stop at the first step where the no is gone
(the finding) or the reply collapses. Members and doses come from the combo ingredients (persona_identity_combos).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import statistics
import time

import persona_identity_combos as combos
import persona_identity_sweep as sweep

NAME, SEED = 'declines-v1', 20261027
AMP = ('trust', 'curiosity', 'calm', 'playfulness', 'gratitude')
MUTE = ('suspicion', 'contempt', 'anxiety')


def nudges(root, doses):
    """{concept: [layer, neuron, signed nudge dose]}: each concept's strongest ingredient with a known coherent dose."""
    found = json.loads((Path(root) / sweep.OUT / 'combo_ingredients_v1.json').read_text())
    out = {}
    for concept, flip in [(c, 1) for c in AMP] + [(c, -1) for c in MUTE]:
        rows = [(abs(d), l, n, s) for l, n, s, d in found.get(concept, []) if (l, n, s) in doses]
        if rows:
            _, l, n, s = max(rows)
            out[concept] = [l, n, flip * s * doses[(l, n, s)]]
    return out


def plan(root):
    db = sweep.connect(root)
    sweep.backfill_flags(root)
    sites = db.execute("SELECT layer,neuron,dose,answer FROM jobs WHERE state='done' AND layer>=0 AND coherent=1 AND declined=1 "
                       "ORDER BY layer,neuron,dose").fetchall()
    doses = combos.coherent_doses(db); rng = random.Random(SEED)
    pool = nudges(root, doses)
    amp = [pool[c] + [c] for c in AMP if c in pool]; mute = [pool[c] + [c] for c in MUTE if c in pool]
    median = {}
    for (layer, _, _), v in doses.items(): median.setdefault(layer, []).append(v)
    median = {l: statistics.median(v) for l, v in median.items()}
    planned = []
    for layer, neuron, dose, answer in sites:
        site = [layer, neuron, dose, 1]                       # fixed: reproduces the no
        sets = [('amp', [a]) for a in amp] + [('mute', [m]) for m in mute]
        if len(amp) > 1: sets.append(('amp', amp[:2]))
        if amp and mute: sets += [('mix', [amp[0], mute[0]])] + ([('mix', [amp[1], mute[0]])] if len(amp) > 1 else [])
        if len(amp) > 1 and mute: sets.append(('mix', amp[:2] + mute[:1]))
        if amp and len(mute) > 1: sets.append(('mix', amp[:1] + mute[:2]))
        controls = []
        for k in (1, 2, 3):
            members = []
            for l, *_ in (amp + mute)[:k]:
                members.append([l, rng.randrange(17408), rng.choice((1, -1)) * median.get(l, 0.5)])
            controls.append(('control', [m + ['random'] for m in members]))
        for kind, extra in sets + controls:
            members = [site] + [[l, n, d] for l, n, d, _ in extra]
            if len({(m[0], m[1]) for m in members}) != len(members): continue
            planned.append({'kind': f'decline-{kind}', 'label': f"L{layer}N{neuron}{dose:+g}|" + '+'.join(c for *_, c in extra),
                            'site': {'layer': layer, 'neuron': neuron, 'dose': dose, 'answer': answer[:300]}, 'members': members})
    (Path(root) / sweep.OUT / 'declines_v1_plan.json').write_text(json.dumps({'seed': SEED, 'ramp': sweep.RAMP, 'nudges': pool,
                                                                            'sites': len(sites), 'combos': planned}, indent=1))
    added = sweep.enqueue_combos(root, NAME, [{'kind': c['kind'], 'label': c['label'], 'members': c['members']} for c in planned])
    return {'sites': len(sites), 'combos': len(planned), 'nudge_concepts': sorted(pool), 'added': added}


def auto(root, hours):
    until = time.time() + hours * 3600
    while time.time() < until:
        with sweep.connect(root) as db:
            left = db.execute("SELECT count(*) FROM jobs WHERE sweep=? AND state IN ('queued','running')", (combos.INGREDIENTS,)).fetchone()[0]
        if not left and (Path(root) / sweep.OUT / 'combo_ingredients_v1.json').exists():
            return plan(root)
        time.sleep(300)
    raise SystemExit('combo ingredients did not finish in time; run `plan` once they do')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('plan', 'auto'))
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--hours', type=float, default=24)
    a = parser.parse_args()
    print(json.dumps(plan(a.root) if a.command == 'plan' else auto(a.root, a.hours)))


if __name__ == '__main__':
    main()
