# SCOTUS Low-Rank Replay Diagnostic

## Purpose

This run tested whether the Commerce replay-v2 answer-state separator becomes causal if we replace a single vector with a learned low-rank activation map.

This preserves the project success standard: no persona prompt, no "think like Justice X" instruction, and no learned adapter counted as success unless it points to a durable reasoning-basin shift. A model that merely wears a mask or reasons about how a target would reason is not a promoted result.

## Offline Diagnostic

Run: `sweep_v4/scotus_replay_lowrank_diag_20260501_165827`

Method:

- Source features: replay-v2 `assistant_all__L08` states.
- Training target: paired Commerce-authority state plus learned map should approximate the paired Commerce-limits delta.
- Search: ranks `0,1,2,4,8,16,32`, ridges `0.01,0.1,1.0,10.0`.
- Controls: permutation low-rank maps trained with shuffled replay deltas.

Best offline map:

- Artifact: `data/scotus/directions/scotus_replay_lowrank_authority_to_limits_assistant_all_L08_rank8_ridge1_20260501.npz`
- Rank/ridge: `8 / 1.0`
- Dev MSE improvement: `0.933`
- Dev delta cosine: `0.967`
- Dev probe probability shift: `0.922`
- Test MSE improvement: `0.939`
- Test delta cosine: `0.970`
- Test edited probe probability: `0.959`
- Test edited positive rate: `1.000`

Read: the replay feature geometry is highly fit by a low-rank map. That is useful diagnostically, but it is still only cached-state evidence.

## Causal Smoke

Run: `sweep_v4/scotus_lowrank_replay_poke_20260501_170217`

Method:

- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Hook: add `beta * low_rank_delta(h)` at `assistant_all @ L8`
- Position: `last`
- Prompts: four Commerce pocket prompts
- Betas: `0.25`, `0.5`, `1.0`
- Controls: three same-family permutation low-rank maps plus a mean-delta source control
- Scoring: rough frame counts, followed by proposition-level rescore

Rough prompt-matched result:

| Beta | Target minus control | Net minus control | Target win | Net win |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | -0.250 | 0.083 | 0.250 | 0.500 |
| 0.50 | -0.500 | -0.167 | 0.250 | 0.500 |
| 1.00 | -0.333 | -1.083 | 0.250 | 0.000 |

## Proposition Rescore

Run: `sweep_v4/scotus_lowrank_replay_prop_rescore_20260501_172039`

Stricter proposition-level result:

| Beta | Target minus random | Net minus random | Target strongest win | Net strongest win |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | -0.167 | 0.000 | 0.000 | 0.000 |
| 0.50 | -0.167 | -0.333 | 0.000 | 0.000 |
| 1.00 | 0.083 | -0.167 | 0.000 | 0.000 |

The candidate never beat the strongest prompt-matched control. The source-control mean-delta map also failed, so the result is not rescued by saying the learned map underfit the mean movement.

## Decision

Do not promote this low-rank replay map as a steerable judicial circuit.

What we learned:

- A learned map can strongly reconstruct paired replay-state deltas offline.
- That offline fit does not transfer to controlled generation in this four-prompt smoke test.
- The current Commerce replay-v2 L8 family should remain closed for direct act-add, trace/component/head replacement, and this low-rank hook variant.

Next work should either repair the evaluation/data setup for a new issue family or use learned interventions only as diagnostics for a permanent basin-shift mechanism, with explicit reasoning-trace checks where available.
