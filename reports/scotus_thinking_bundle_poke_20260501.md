# SCOTUS Two-Stage Thinking Bundle Pokes

## Purpose

Test whether the controlled Article III four-layer bundle can move both visible reasoning and final answer under the project's no-mask standard.

This follows the corrected Qwen3.5 thinking smoke: the model's chat template pre-fills `<think>`, the generated thought often does not close on its own, and a promotion-grade audit must inspect the reasoning trace rather than only final-answer text.

## Method

- Runner: `scripts/experiments/scotus/poke_scotus_thinking_bundle.py`.
- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`.
- Replay source: `sweep_v4/scotus_article3_controlled_replay_v2_20260501_172957`.
- Prompt bank: `data/scotus/scotus_article3_private_public_poke_prompts_v2.jsonl`.
- Direction: `article3_private_rights - article3_public_rights`.
- Layers/region: `assistant_all @ L4/L8/L12/L16`.
- Mode: mean paired deltas from train split.
- Two-stage audit:
  - generate visible thought with `enable_thinking=True`;
  - mechanically append `</think>`;
  - generate final answer from the same trace;
  - keep the same intervention active during thought and answer generation.
- Scoring: proposition-level frame rules, separately for `thinking` and `answer`.

## Runs

| Run | Position | Prompts | Budget | Controls | Result |
| --- | --- | ---: | --- | ---: | --- |
| `sweep_v4/scotus_thinking_bundle_poke_20260501_200321` | `decode` | 1 | `512/192` | 2 | positive smoke, too small |
| `sweep_v4/scotus_thinking_bundle_poke_20260501_200935` | `decode` | 2 | `384/160` | 2 | answer movement only; thinking failed |
| `sweep_v4/scotus_thinking_bundle_poke_20260501_201846` | `last` | 2 | `384/160` | 2 | answer movement only; strongest controls failed |

## Results

Single-prompt `decode` smoke:

- Thinking target-minus-random: `1.000`.
- Thinking net-minus-random: `1.000`.
- Thinking strongest-random wins: target `1.000`, net `1.000`.
- Answer target-minus-random: `2.000`.
- Answer net-minus-random: `1.000`.
- Answer strongest-random wins: target `1.000`, net `1.000`.

Two-prompt `decode` expansion:

- Thinking target-minus-random: `0.000`.
- Thinking net-minus-random: `-0.250`.
- Thinking strongest-random wins: target `0.000`, net `0.000`.
- Answer target-minus-random: `0.500`.
- Answer net-minus-random: `1.000`.
- Answer strongest-random wins: target `0.000`, net `0.500`.

Two-prompt `last` expansion:

- Thinking target-minus-random: `0.250`.
- Thinking net-minus-random: `0.250`.
- Thinking strongest-random wins: target `0.000`, net `0.000`.
- Answer target-minus-random: `0.500`.
- Answer net-minus-random: `0.500`.
- Answer strongest-random wins: target `0.000`, net `0.000`.

No run had imitation-marker hits. All mechanically closed answer stages produced nonempty answers.

## Interpretation

The one-prompt positive smoke did not survive expansion.

The bundle can nudge final-answer framing in the expected direction, but the visible thinking trace does not move reliably, and the answer-side movement is not stronger than the best same-layer random controls. The `last` prefill-touching variant did not repair this.

This is especially important because the no-mask standard requires the model's reasoning trace to move in the target frame, not merely the final answer to include more target-frame legal language.

## Decision

Do not promote the controlled Article III four-layer residual bundle as a steerable judicial circuit.

This branch now has three negative or weak no-mask tests:

- direct one-stage generated-token bundle failed strongest controls;
- two-stage `decode` thinking audit failed on the expanded prompt set;
- two-stage `last` thinking audit failed strongest controls and did not produce reliable trace movement.

Do not spend more full-model runtime on the same mean `assistant_all @ L4/L8/L12/L16` residual-add bundle. A future candidate should change intervention family, such as a learned but diagnostic low-rank intervention, source-trace patching with better causal attribution, or a different representation target. Learned interventions remain acceptable only if they help identify or create a durable reasoning-basin shift rather than acting as a persona mask.
