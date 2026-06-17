# SCOTUS Article III Low-Rank Thinking Pokes

## Purpose

Test whether the controlled Article III public/private-rights replay geometry becomes causal when a learned low-rank map is applied during Qwen3.5 visible-thinking generation.

This was the first stronger follow-up to the distributed-shape hypothesis after simple residual act-add and four-layer mean bundles failed. It is still a no-mask test: no justice/persona prompt, visible thought and final answer scored separately, and promotion requires reasoning-trace movement as well as final-answer movement.

## Offline Diagnostic

Run: `sweep_v4/scotus_replay_lowrank_diag_20260501_203153`

Method:

- Replay source: `sweep_v4/scotus_article3_controlled_replay_v2_20260501_172957`
- Feature key: `assistant_all__L04`
- Source label: `article3_public_rights`
- Target label: `article3_private_rights`
- Search: ranks `0,1,2,4,8,16`, ridges `0.01,0.1,1.0,10.0`
- Controls: 5 permutation low-rank maps trained with shuffled source-to-target deltas

Best offline map:

- Artifact: `data/scotus/directions/scotus_replay_lowrank_article3_public_to_private_assistant_all_L04_rank16_ridge0p01_20260501.npz`
- Rank/ridge: `16 / 0.01`
- Dev MSE improvement: `0.996`
- Dev delta cosine: `0.998`
- Test MSE improvement: `0.996`
- Test delta cosine: `0.998`
- Top permutation null dev MSE improvement: `0.358`

Read: cached replay-state geometry is highly structured. This justifies causal testing, but it is not steering evidence by itself.

## Two-Prompt No-Mask Smoke

Run: `sweep_v4/scotus_thinking_lowrank_poke_20260501_203434`

Method:

- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Hook: apply learned low-rank delta at `assistant_all @ L4`
- Position: `last`
- Beta: `0.25`
- Prompts: `A3_PRIV_02_bankruptcy_counterclaim`, `A3_PUBLIC_01_benefits_eligibility`
- Controls: 2 permutation low-rank controls plus a rank-0 mean-delta source control
- Audit: generate visible thought with `enable_thinking=True`, mechanically close `</think>`, then generate answer with the intervention still active

Result:

| Segment | Target minus random | Net minus random | Target strongest wins | Net strongest wins |
| --- | ---: | ---: | ---: | ---: |
| thinking | `0.000` | `0.500` | `0.000` | `0.500` |
| answer | `1.500` | `1.250` | `0.500` | `0.500` |

The candidate weakly moved final-answer framing but did not add target thinking markers. The rank-0 mean-delta source control was as strong or stronger on thinking.

## Full-Bank Expansion

Run: `sweep_v4/scotus_thinking_lowrank_poke_20260501_204712`

Method:

- Same model, replay run, feature key, position, and two-stage thinking audit as the smoke.
- Beta: `0.5`
- Prompts: all 8 Article III public/private no-persona prompts.
- Controls: 4 permutation low-rank controls plus the rank-0 mean-delta source control.
- Runtime: `2026-05-01T20:47:12-07:00` to `2026-05-01T21:45:33-07:00`.

Segment summary:

| Segment | Condition | n | Target delta vs base | Net delta vs base | Target rate | Contrast rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| thinking | candidate | 8 | `0.000` | `0.125` | `1.000` | `1.000` |
| thinking | permutation controls | 32 | `-0.906` | `-0.188` | `0.844` | `0.531` |
| thinking | source control | 8 | `0.000` | `0.000` | `1.000` | `1.000` |
| answer | candidate | 8 | `0.375` | `0.125` | `0.750` | `0.875` |
| answer | permutation controls | 32 | `0.281` | `0.219` | `0.875` | `0.656` |
| answer | source control | 8 | `1.500` | `1.250` | `1.000` | `0.875` |

Prompt-matched candidate versus permutation controls:

| Segment | Target minus random | Net minus random | Target strongest wins | Net strongest wins |
| --- | ---: | ---: | ---: | ---: |
| thinking | `0.906` | `0.312` | `0.125` | `0.125` |
| answer | `0.094` | `-0.094` | `0.125` | `0.125` |

All 56 generated rows had nonempty answers, and no rows had imitation markers. The model did not naturally close the thought in any row, so the two-stage mechanical close remained necessary.

## Interpretation

The full-bank result is not promotable. The positive thinking delta versus random is misleading because the permutation controls degraded target-frame markers below baseline; the candidate itself had `0.000` target delta versus baseline in thinking. Final-answer movement was small, net movement was negative versus random, and the source-control mean-delta map moved final answers more than the learned low-rank map.

This is useful negative evidence: even a learned low-rank map that reconstructs held-out replay deltas almost perfectly offline does not become a reliable no-mask generation actuator when applied at a single layer and last-token position.

## Decision

Do not promote the Article III `assistant_all @ L4` single-layer low-rank hook as a steerable judicial reasoning actuator.

Close this exact intervention family for Article III:

- single-layer `assistant_all @ L4`;
- last-token hook;
- public-rights to private-rights low-rank map;
- beta-only retuning without new localization.

Next useful work should identify the actuator surface before running more full-bank thinking audits:

1. Run trajectory patching over paired public/private reasoning traces to localize layer-token-component windows that actually change visible thought frames.
2. Train the smallest multi-site low-rank controller over those localized windows, not over a single cached replay layer.
3. Audit any new controller with the same no-mask two-stage thinking harness and strongest-control gates.
