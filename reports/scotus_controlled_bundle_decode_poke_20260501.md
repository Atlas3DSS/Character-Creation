# SCOTUS Controlled Bundle Decode Poke

## Purpose

Test the distributed-shape hypothesis after single-direction Article III act-add failed.

Instead of adding one L4 probe direction at the prompt boundary, this run built a four-layer bundle from controlled replay states and applied it only during generated-token decode steps. This avoids rewriting the neutral prompt prefill and asks whether the private-rights/public-rights contrast has a distributed generated-token control shape.

This is still a no-mask test: no justice name, no persona prompt, and no "think like" instruction.

## Inputs

| Item | Value |
| --- | --- |
| Model | `/home/orwel/dev_genius/models/Qwen3.5-27B` |
| Feature source | `sweep_v4/scotus_article3_controlled_replay_v2_20260501_172957` |
| Prompt bank | `data/scotus/scotus_article3_private_public_poke_prompts_v2.jsonl` |
| Runner | `scripts/experiments/scotus/poke_scotus_controlled_replay_bundle.py` |
| Generation run | `sweep_v4/scotus_controlled_bundle_poke_20260501_183716` |
| Proposition rescore | `sweep_v4/scotus_controlled_bundle_prop_rescore_20260501_193155` |

## Method

- Target/reference: `article3_private_rights - article3_public_rights`.
- Region: `assistant_all`.
- Layers: `4, 8, 12, 16`.
- Fit split: `train`.
- Pairing key: `pair_id`, so each answer variant is paired within the same fact pattern and surface style.
- Direction mode: mean paired delta per layer.
- Hook position: `decode`, which skips prompt prefill and edits only generated-token forward passes.
- Alpha scaling: per-layer hidden-norm fraction from the feature source.
- Alphas: `0.003`, `0.005`, `0.01`.
- Controls: `6` same-layer random bundles, with the same per-layer norm scaling.
- Prompts: `8` no-persona Article III prompts.

## Raw Frame Result

Raw substring frame scoring did not promote the bundle.

| Alpha | Target minus random | Net minus random | Target win rate | Net win rate |
| --- | ---: | ---: | ---: | ---: |
| `0.003` | `-0.562` | `-0.583` | `0.000` | `0.000` |
| `0.005` | `0.167` | `0.042` | `0.250` | `0.375` |
| `0.010` | `0.188` | `-0.458` | `0.500` | `0.375` |

The raw metric is especially weak because target hits were already dense in baseline outputs, and the `0.010` setting increased contrast hits more than target hits.

## Proposition Rescore

The stricter proposition-level rescore also failed promotion.

| Alpha | Target minus random | Target z | Target strongest win | Net minus random | Net z | Net strongest win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.003` | `-0.083` | `-0.233` | `0.000` | `-0.104` | `-0.273` | `0.000` |
| `0.005` | `-0.062` | `-0.241` | `0.000` | `-0.125` | `-0.341` | `0.000` |
| `0.010` | `0.146` | `0.392` | `0.250` | `0.083` | `0.190` | `0.250` |

Best alpha was `0.010`, but it still only beat the strongest random bundle on `2/8` prompts for both target and net proposition movement.

## Decision

Do not promote this candidate.

This result directly tests the user's "whole shape / nearby adjacencies" concern for the Article III controlled replay branch. A four-layer generated-token residual bundle did slightly better than the single L4 prompt-boundary poke at the highest alpha, but it still failed strongest random controls and did not produce reliable prompt-matched proposition movement.

The current evidence now disfavors:

- single-vector prompt-boundary residual act-add,
- replay-v2 low-rank residual maps,
- token-local trace replacement,
- component-output and attention-head trace replacement in the Commerce branch,
- and this Article III four-layer generated-token residual bundle.

The next productive move is not another sweep of this same residual-add family. Either the evaluation needs to be rebuilt around richer paired completions and stronger proposition grading, or the intervention family needs to change to a learned, bounded ReFT/LoReFT-style diagnostic whose result is judged by no-mask reasoning-basin movement, not by persona imitation.
