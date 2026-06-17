# SCOTUS Counterfactual Visible Thoughts

## Goal

Calibrate whether coherent visible thought text can route Qwen3.5's final Article III answer. This is explicitly not a steering result: the thought is inserted by the harness. The question is whether visible reasoning is a viable causal channel if a future non-mask actuator can make the model produce that reasoning itself.

## Artifacts

- Run: `sweep_v4/scotus_counterfactual_thoughts_20260501_231331`
- Raw report: `sweep_v4/scotus_counterfactual_thoughts_20260501_231331/report.md`
- Generations: `sweep_v4/scotus_counterfactual_thoughts_20260501_231331/generations.jsonl`
- Summary: `sweep_v4/scotus_counterfactual_thoughts_20260501_231331/summary.jsonl`
- Script: `scripts/experiments/scotus/probe_scotus_counterfactual_thoughts.py`

## Method

- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Prompt bank: `data/scotus/scotus_article3_private_public_poke_prompts_v2.jsonl`
- Prompts: all 8 Article III private/public no-persona prompts.
- Conditions:
  - `neutral`: balanced public/private issue checklist;
  - `private_rights`: coherent scratchpad framing the issue as a private-rights Article III problem;
  - `public_rights`: coherent scratchpad framing the issue as a public-rights administrative adjudication problem.
- The script inserted the scratchpad as visible thought, mechanically closed `</think>`, generated a final answer without hooks, and scored proposition frames.

## Result

Clean counterfactual thought text does influence final answers, but the current proposition scorer is not clean enough to treat this as a directional actuator metric.

| condition | n | target_hits | contrast_hits | target-minus-contrast | net-vs-neutral |
| --- | ---: | ---: | ---: | ---: | ---: |
| `neutral` | 8 | `1.250` | `0.875` | `0.375` | `0.000` |
| `private_rights` | 8 | `2.500` | `0.500` | `2.000` | `1.625` |
| `public_rights` | 8 | `1.875` | `1.500` | `0.375` | `0.000` |
| `private_minus_public` | 8 | `0.625` | `-1.000` | `1.625` | n/a |

Read:

- Private-rights scratchpads moved final answers toward the target private-rights/pro-Article-III frame relative to neutral and public-rights scratchpads.
- Public-rights scratchpads did not cleanly suppress target hits because legally careful public-rights answers often mention the private-rights distinction while rejecting it.
- The scorer therefore measures doctrinal vocabulary/proposition presence, not necessarily the final holding direction.
- This diagnostic supports visible thought as a possible causal channel, but only with coherent scratchpads. The previous ablation run showed corrupted/random deletions also move markers, so any mechanistic actuator still needs strong controls.

## Decision

Do not promote anything from this run.

What this does establish:

- If a future non-mask mechanism can make the model naturally produce coherent private-rights visible reasoning, final answers are likely to follow.
- The current proposition scorer needs a holding-direction or conclusion-polarity layer before it can adjudicate public-rights counterfactuals cleanly.

What this does not establish:

- It does not identify an activation actuator.
- It does not satisfy the no-mask goal because the thought was inserted by the harness.
- It does not justify more raw source-trace replacement.

## Next

The next evaluator repair should add conclusion-polarity labels for Article III prompts:

- private-rights / Article III objection succeeds;
- public-rights / agency adjudication permissible;
- mixed or distinction-only discussion.

Only after that should a multi-site controller or ReFT-style diagnostic be evaluated, because the current frame scorer rewards answers that merely discuss both public and private rights.
