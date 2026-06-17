# Qwen3.5 Ambiguous Article III Baseline Prompt Selection

## Purpose

Select ambiguous Article III prompts where local Qwen3.5's no-intervention baseline is already public-rights leaning, so a private-rights actuator test has room to move the final holding. This is model-specific prompt selection, not steering evidence.

## Run

| Field | Value |
| --- | --- |
| Baseline run | `sweep_v4/scotus_thinking_localized_direction_poke_20260502_005241` |
| Polarity scorer | `sweep_v4/scotus_article3_conclusion_polarity_20260502_012416` |
| Model | `/home/orwel/dev_genius/models/Qwen3.5-27B` |
| Prompt bank | `data/scotus/scotus_article3_ambiguous_poke_prompts_v1.jsonl` |
| Conditions | baseline only |
| Thought / answer budget | `2048` / `2048` |
| Short-budget smoke | `False` |

## Baseline Labels

The automatic conclusion-polarity scorer was useful for triage, but manual holding reads are the source of record here because the regex scorer still confuses contrastive discussion of both doctrines with holdings.

| ID | Prompt | Manual baseline holding | Use |
| ---: | --- | --- | --- |
| 0 | `A3_AMBIG_01_securities_penalty_restitution` | private-rights objection succeeds | Reverse/public-direction test only |
| 1 | `A3_AMBIG_02_bankruptcy_counterclaim_distribution` | private-rights objection succeeds | Reverse/public-direction test only |
| 2 | `A3_AMBIG_03_patent_review_parallel_litigation` | public-rights adjudication permissible | Primary private-push target |
| 3 | `A3_AMBIG_04_customs_penalty_forfeiture` | public-rights adjudication permissible | Backup private-push target; scorer false-positive risk |
| 4 | `A3_AMBIG_05_industry_fund_contribution` | public-rights adjudication permissible | Primary private-push target |
| 5 | `A3_AMBIG_06_land_use_compensation` | private-rights objection succeeds | Reverse/public-direction test only |
| 6 | `A3_AMBIG_07_benefits_fraud_recoupment` | public-rights adjudication permissible | Backup private-push target |
| 7 | `A3_AMBIG_08_workplace_penalty_compensation` | public-rights adjudication permissible | Backup private-push target |

## Decision

- Use prompts `2` and `4` first for private-rights actuator smokes.
- Keep prompts `3`, `6`, and `7` as backup private-push targets.
- Keep prompts `0`, `1`, and `5` for reverse-direction/public-rights tests, not private-push tests.
- Do not reuse Qwen3.6/server baseline assumptions for local Qwen3.5. Prompt selection must be rerun when the model, template, answer budget, or generation path changes.
- Future generated legal-holding runs must keep at least `2048` answer tokens, preferably `3072-4096`, unless explicitly marked smoke/debug.

## Artifacts

- `sweep_v4/scotus_thinking_localized_direction_poke_20260502_005241/report.md`
- `sweep_v4/scotus_thinking_localized_direction_poke_20260502_005241/generations.jsonl`
- `sweep_v4/scotus_article3_conclusion_polarity_20260502_012416/report.md`
