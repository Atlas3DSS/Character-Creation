# Article III Ambiguous Thought-State Localization

## Purpose

Use the evaluator-positive ambiguous Article III counterfactual-thought setup to localize where Qwen3.5 differs when the same prompts are teacher-forced through neutral, private-rights, and public-rights visible reasoning. This is candidate nomination only, not actuator evidence.

## Run

| Field | Value |
| --- | --- |
| Script | `scripts/experiments/scotus/localize_article3_ambiguous_thought_states.py` |
| Run | `sweep_v4/scotus_article3_ambiguous_thought_state_localization_20260502_003317` |
| Model | `/home/orwel/dev_genius/models/Qwen3.5-27B` |
| Prompt bank | `data/scotus/scotus_article3_ambiguous_poke_prompts_v1.jsonl` |
| Prompts | `8` |
| Conditions | `neutral`, `private_rights`, `public_rights` |
| Components | `residual`, `mixer`, `mlp` |
| Layers | all `64` layers |
| Regions | `pre_answer_last`, `thought_mean`, `thought_tail16_mean`, `tail32_mean` |
| Shuffle controls | `64` |

## Main Result

The strongest teacher-forced private-minus-public state difference is a late-layer cluster, not a single isolated site.

| Rank | Layer | Component | Region | Score-null | Consistency | Triad separation |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 61 | residual | tail32_mean | 3.395 | 0.983 | 1.106 |
| 2 | 62 | residual | tail32_mean | 3.151 | 0.982 | 1.093 |
| 3 | 62 | residual | thought_tail16_mean | 3.011 | 0.984 | 0.925 |
| 4 | 58 | residual | tail32_mean | 2.741 | 0.987 | 1.105 |
| 8 | 62 | mlp | thought_tail16_mean | 2.568 | 0.982 | 1.066 |
| 9 | 55 | mlp | tail32_mean | 2.565 | 0.988 | 1.266 |

Top mixer sites also cluster late, especially L57-L63, but with weaker adjusted scores than residual/MLP.

## Interpretation

- This is a coherent localization signal over late residual/MLP/mixer state, especially tail-window regions.
- It is also suspicious in exactly the expected way: the strongest regions are close to the end of teacher-forced visible-thought text, so they may encode text-tail state rather than a portable reasoning actuator.
- The output gives a concrete candidate surface for a small no-mask intervention screen, but does not validate a non-mask actuator.

## Artifacts

- `sweep_v4/scotus_article3_ambiguous_thought_state_localization_20260502_003317/report.md`
- `sweep_v4/scotus_article3_ambiguous_thought_state_localization_20260502_003317/site_metrics.jsonl`
- `sweep_v4/scotus_article3_ambiguous_thought_state_localization_20260502_003317/top_directions.npz`
- `sweep_v4/scotus_article3_ambiguous_thought_state_localization_20260502_003317/direction_meta.jsonl`
