# Article III Localized Direction No-Mask Smoke

## Purpose

Smoke-test whether the late-layer localized thought-state directions can be used as a frozen multi-site no-mask actuator during generated visible thinking and final-answer generation.

This is not a promotion run: it used one prompt, one alpha, one random control, and a short visible-thinking budget explicitly marked as smoke.

## Run

| Field | Value |
| --- | --- |
| Script | `scripts/experiments/scotus/poke_scotus_thinking_localized_directions.py` |
| Run | `sweep_v4/scotus_thinking_localized_direction_poke_20260502_003719` |
| Polarity scorer | `sweep_v4/scotus_article3_conclusion_polarity_20260502_004803` |
| Model | `/home/orwel/dev_genius/models/Qwen3.5-27B` |
| Prompt | `A3_AMBIG_01_securities_penalty_restitution` |
| Candidate sites | `L61 residual tail32`, `L62 residual tail32`, `L62 residual thought_tail16`, `L58 residual tail32` |
| Position | `decode` |
| Alpha | `2.0`, normalized by sqrt(site count) |
| Random controls | `1` same-site random bundle |
| Thought / answer budget | `768` / `2048` |

## Result

The candidate did not move visible thinking in this smoke. The generated thought stayed the same high-level boilerplate analysis pattern as baseline.

| Segment | Condition | Target delta | Net delta |
| --- | --- | --- | --- |
| thinking | random control | +1 | +1 |
| thinking | localized candidate | 0 | 0 |
| answer | random control | -1 | -1 |
| answer | localized candidate | -3 | -3 |

The conclusion-polarity scorer labeled baseline and candidate as public-rights/permissible and random as private-rights/succeeds, but manual inspection shows the scorer is again confusing contrastive doctrine mentions with holdings. Baseline, random, and candidate answers all say final agency adjudication is unconstitutional or Article III-prohibited; the random answer is arguably the cleanest private-rights holding.

## Interpretation

- Do not promote the localized late-residual bundle.
- The harness works, but this one-prompt smoke gives no reason to spend a full promotion run on this exact `top4 residual, alpha=2.0` setting.
- If continuing this branch, the next test should use the corrected comparator condition, choose prompts where the local Qwen3.5 baseline is not already private-rights leaning, and sweep smaller/larger alphas over residual+MLP site bundles with at least two same-site random controls.
- The automatic conclusion-polarity scorer remains useful for triage only; reviewed holding labels are required for promotion.

## Artifacts

- `sweep_v4/scotus_thinking_localized_direction_poke_20260502_003719/report.md`
- `sweep_v4/scotus_thinking_localized_direction_poke_20260502_003719/generations.jsonl`
- `sweep_v4/scotus_thinking_localized_direction_poke_20260502_003719/summary.jsonl`
- `sweep_v4/scotus_article3_conclusion_polarity_20260502_004803/report.md`
