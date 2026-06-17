# SCOTUS Off-Domain Poke Smoke

## Question

Do the strongest surviving SCOTUS probe directions change ordinary nonlegal reasoning prompts when nudged during generation?

This was a sanity check for whether the directions behave like a broad reasoning-style/temperament vector, rather than only a domain-conditioned legal-text direction.

## Setup

Prompt bank:

- `data/scotus/scotus_offdomain_poke_prompts_v1.jsonl`
- Six nonlegal prompts: weather picnic, video-game balance, friend-group restaurant dispute, homework/sleep tradeoff, boys basketball tryouts, and headphone choice.

Model and steering:

- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Direction source: external probe directions loaded from prior SCOTUS BF16 runs
- Alpha scaling: `hidden-norm-fraction`
- Controls: same-layer same-norm random unit vectors
- Decoding: greedy

Implementation note:

- `scripts/experiments/scotus/poke_scotus_sae_layers.py` now names external directions as `{parent_dir}__{file_stem}`. This avoids collisions when multiple direction files are named `best_probe_direction.npz`.

## Runs

| Run | Directions | Position | Alphas | Random controls | Report |
| --- | --- | --- | --- | ---: | --- |
| Last-token smoke | Phase 4 Scalia/Ginsburg `prompt_last @ L4`; majority-2000s feasible split-00 `prompt_last @ L10` | `last` | `0.02,0.05` | 2 | `sweep_v4/scotus_sae_poke_20260501_072906/report.md` |
| All-token smoke | majority-2000s feasible split-01 `excerpt_mean @ L16` | `all` | `0.01,0.02,0.05` | 2 | `sweep_v4/scotus_sae_poke_20260501_073923/report.md` |

## Result

Frame metrics were null in both runs:

- Target-frame mean delta: `0.00` for all candidate rows.
- Contrast-frame mean delta: `0.00` for all candidate rows.
- Net-frame mean delta: `0.00` for all candidate rows.
- Off-domain-frame mean delta: `0.00` for all candidate rows.

The automatic `fourth_digital_privacy` tags on some video-game and tryout outputs are metric noise from generic words like "data"; they did not reflect legal content.

Manual read:

- Last-token nudges were effectively inert. Outputs stayed close to base and random-control completions.
- All-token L16 nudges produced mild formatting and rhetoric shifts at higher alpha, such as "Decision:" headings, "High-Energy to Low-Energy" planning, and more explicit "structured, transparent, multi-dimensional framework" language.
- The same class of shifts appeared in random controls, including "Decision:" weather answers, "Veto and Filter" restaurant-selection framing, and ordinary wording changes around sleep quality/sleep hygiene.
- There was no obvious judicial, legalistic, Scalia-like, or Ginsburg-like drift on weather, video games, social conflict, homework planning, team selection, or headphones.

## Interpretation

This weakens the broad "general reasoning temperament" read of the current SCOTUS directions. Under these smoke settings, the directions look more like domain/context-conditioned decodable structure than like a portable reasoning-style control.

This does not falsify the SCOTUS project. It does lower the priority of more broad off-domain pokes from the same directions. The next useful causal work should use narrower matched legal contrasts where the desired output change is defined before generation and compared against prompt-matched random controls.

## Next

Do not promote any off-domain effect from these runs. If we revisit off-domain behavior, use a stronger design:

- 30-50 unrelated prompts across reasoning, planning, preference, and social judgment.
- A blind rubric that scores structure, cautiousness, institutional framing, legalism, and answer conservatism separately.
- At least 10 random controls per layer/alpha.
- Candidate promotion only if the direction beats random controls on blind ratings, not merely on visible formatting changes.
