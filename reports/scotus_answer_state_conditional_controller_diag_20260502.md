# SCOTUS Answer-State Conditional Controller Diagnostic

## Purpose

Screen whether the late answer-state localized Article III sites contain prompt-conditioned private-minus-public structure beyond a stable mean delta.

This is offline diagnostic evidence only. A positive result would justify a live no-mask controller run; it would not itself prove steering.

## Artifacts

- Localization source: `sweep_v4/scotus_article3_counterfactual_answer_state_localization_20260502_053619`
- Offline diagnostic run: `sweep_v4/scotus_article3_localized_conditional_diag_20260502_060952`
- Script: `scripts/experiments/scotus/diagnose_article3_localized_lowrank.py`

## Setup

- Model: local Qwen 3.5 27B
- Prompt bank: `data/scotus/scotus_article3_ambiguous_poke_prompts_v1.jsonl`
- Prompts: `8`
- Sites: top four answer-state sites
  - `L58 residual pre_answer_last`
  - `L62 mixer pre_answer_last`
  - `L59 mixer pre_answer_last`
  - `L63 residual pre_answer_last`
- Models tested: leave-one-out mean delta, nearest-neighbor neutral-state delta, and KRR low-rank predictors with ranks `0,1,2,4` and ridges `0.01,0.1,1`.
- Permutation controls: `8`

## Result

The mean private-minus-public delta is strong as a descriptive direction, but the conditional predictors do not add useful prompt-conditioned structure.

| model | MSE improvement vs mean | mean cosine | cosine minus mean | null max MSE vs mean | null max cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| loo_mean_delta | 0.000 | 0.816 | 0.000 |  |  |
| nearest_neutral_delta | -0.432 | 0.768 | -0.048 |  |  |
| best KRR by cosine delta | -0.020 | 0.817 | 0.001 | 0.002 | 0.895 |
| best KRR by MSE delta | -0.008 | 0.817 | 0.001 | 0.012 | 0.897 |

The best non-null KRR rows improved cosine over the mean by only about `0.001`, while permutation null rows reached cosine up to about `0.900` and MSE improvements up to about `0.027`.

## Decision

Do not spend a live no-mask generation run on this simple conditional low-rank controller for the answer-state top-four sites.

The surface looks like a stable inserted-thought answer-tail/pre-answer delta rather than a prompt-conditioned actuator target. Combined with the complete-budget additive poke failure, this argues for changing the intervention family more substantially rather than training a small KRR/ReFT-style controller on the same top-four late pre-answer sites.
