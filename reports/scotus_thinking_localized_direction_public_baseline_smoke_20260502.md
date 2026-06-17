# Article III Localized Direction Public-Baseline Smoke

## Purpose

Test the late localized private-minus-public thought-state directions on prompts where local Qwen3.5 baseline holdings are public-rights leaning. This corrects the first one-prompt smoke, which used a prompt that was already private-rights leaning under manual review.

## Run

| Field | Value |
| --- | --- |
| Script | `scripts/experiments/scotus/poke_scotus_thinking_localized_directions.py` |
| Run | `sweep_v4/scotus_thinking_localized_direction_poke_20260502_012514` |
| Polarity scorer | `sweep_v4/scotus_article3_conclusion_polarity_20260502_021856` |
| Model | `/home/orwel/dev_genius/models/Qwen3.5-27B` |
| Prompts | `A3_AMBIG_03_patent_review_parallel_litigation`, `A3_AMBIG_05_industry_fund_contribution` |
| Candidate sites | top 8 localized residual/MLP sites from `sweep_v4/scotus_article3_ambiguous_thought_state_localization_20260502_003317` |
| Position | `decode` |
| Alphas | `1.0`, `2.0`, normalized by sqrt(site count) |
| Random controls | `2` same-site random bundles |
| Thought / answer budget | `2048` / `2048` |
| Short-budget smoke | `False` |

Candidate sites were `L61 residual tail32`, `L62 residual tail32`, `L62 residual thought_tail16`, `L58 residual tail32`, `L59 residual tail32`, `L54 residual tail32`, `L63 residual thought_tail16`, and `L62 MLP thought_tail16`.

## Result

The localized candidate does not beat prompt-matched random controls. On answer-side movement it is worse than the random controls, and on visible thinking it only shows a weak alpha-1 near-tie that does not survive alpha-2.

| Segment | Alpha | Target-minus-random | Net-minus-random | Target strongest wins | Net strongest wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| thinking | 1.0 | 0.750 | 0.250 | 0.500 | 0.500 |
| thinking | 2.0 | -1.750 | -1.750 | 0.000 | 0.000 |
| answer | 1.0 | -0.500 | -1.250 | 0.000 | 0.000 |
| answer | 2.0 | -1.250 | -1.500 | 0.500 | 0.500 |

The automatic polarity scorer labels the two baseline answers as public-rights/permissible, matching the manual baseline prompt-selection read for prompts `2` and `4`. It still remains a triage tool rather than a promotion gate because it can overcount contrastive doctrinal discussion in other prompts.

## Decision

- Do not promote the top-8 localized residual/MLP unit-add bundle at alpha `1.0` or `2.0`.
- Do not spend a full promotion run on this exact unit-add actuator shape.
- If this branch continues, the next viable step is a non-additive controller over the localized late-layer surface, such as a low-rank/ReFT diagnostic, or a reverse-direction test on private-baseline prompts. More ad hoc unit-vector pokes are low-value unless they add a substantially different actuator family.
- Future public/private Article III actuator tests should start from the prompt-selection table in `reports/scotus_qwen35_ambiguous_baseline_prompt_selection_20260502.md`.

## Artifacts

- `sweep_v4/scotus_thinking_localized_direction_poke_20260502_012514/report.md`
- `sweep_v4/scotus_thinking_localized_direction_poke_20260502_012514/generations.jsonl`
- `sweep_v4/scotus_thinking_localized_direction_poke_20260502_012514/candidate_vs_prompt_matched_random.jsonl`
- `sweep_v4/scotus_article3_conclusion_polarity_20260502_021856/report.md`
