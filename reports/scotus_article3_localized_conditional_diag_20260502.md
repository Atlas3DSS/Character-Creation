# Article III Localized Conditional Controller Diagnostic

## Purpose

Check whether the late localized Article III thought-state surface contains prompt-conditioned structure that could justify a live non-additive controller. This is an offline diagnostic only: it predicts teacher-forced private-minus-public deltas from neutral inserted-thought states using leave-one-prompt-out evaluation.

## Run

| Field | Value |
| --- | --- |
| Script | `scripts/experiments/scotus/diagnose_article3_localized_lowrank.py` |
| Run | `sweep_v4/scotus_article3_localized_conditional_diag_20260502_034404` |
| Model | `/home/orwel/dev_genius/models/Qwen3.5-27B` |
| Prompt bank | `data/scotus/scotus_article3_ambiguous_poke_prompts_v1.jsonl` |
| Localization source | `sweep_v4/scotus_article3_ambiguous_thought_state_localization_20260502_003317` |
| Sites | top 4 residual sites |
| Evaluation | leave-one-prompt-out |
| Candidate maps | mean delta, nearest-neighbor delta, KRR low-rank `0/1/2` with ridge `0.1` |
| Permutation controls | `4` per nonzero-rank map |

## Result

The diagnostic does not justify a live conditional-controller run over this exact surface. The leave-one-out mean delta already predicts held-out private-minus-public deltas with cosine about `0.979`. KRR rank `1` and `2` add essentially no useful prompt-conditioned signal, and nearest-neighbor conditioning is worse.

| Model | Sites | MSE vs mean | MSE vs zero | Cosine | Cos minus mean | Null max MSE vs mean | Null max cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean delta / rank 0 | 4 | 0.000 | 0.959 | 0.979 | 0.000 |  |  |
| KRR rank 1 | 4 | -0.000 | 0.959 | 0.979 | 0.000 | -0.005 | 0.983 |
| KRR rank 2 | 4 | 0.004 | 0.959 | 0.979 | 0.000 | -0.003 | 0.982 |
| nearest neutral delta | 4 | -0.418 | 0.941 | 0.971 | -0.008 |  |  |

The small positive rank-2 MSE gain is not meaningful: cosine is unchanged relative to the mean baseline, and null permutation rows reach similar or better cosine. This is a "stable mean tail delta" signature, not a useful prompt-conditioned actuator signature.

## Decision

- Do not run a long no-mask generation sweep for this exact KRR/nearest-neighbor controller over the top-4 residual inserted-thought sites.
- Combined with the positive-alpha, negative-alpha, and reverse-direction failures, this closes the simple localized inserted-thought-tail branch.
- The next Article III actuator search should use generated-token trajectory localization, not teacher-forced inserted-thought tail deltas, or switch to a different intervention family such as a true trained ReFT/adapter diagnostic with held-out no-mask generation gates.

## Artifacts

- `sweep_v4/scotus_article3_localized_conditional_diag_20260502_034404/report.md`
- `sweep_v4/scotus_article3_localized_conditional_diag_20260502_034404/summary.jsonl`
- `sweep_v4/scotus_article3_localized_conditional_diag_20260502_034404/diagnostic_rows.jsonl`
- `scripts/experiments/scotus/diagnose_article3_localized_lowrank.py`
