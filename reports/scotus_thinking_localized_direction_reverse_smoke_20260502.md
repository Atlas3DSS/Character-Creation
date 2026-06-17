# Article III Localized Direction Reverse Smoke

## Purpose

Test whether the localized private-minus-public Article III surface is directionally symmetric. The earlier public-baseline smoke applied positive alphas to public-leaning prompts to ask whether the surface could push answers toward private-rights holdings. This run applies negative alphas to private-leaning prompts to ask whether the same surface can push answers toward public-rights holdings.

## Run

| Field | Value |
| --- | --- |
| Script | `scripts/experiments/scotus/poke_scotus_thinking_localized_directions.py` |
| Run | `sweep_v4/scotus_thinking_localized_direction_poke_20260502_022548` |
| Polarity scorer | `sweep_v4/scotus_article3_conclusion_polarity_20260502_033340` |
| Model | `/home/orwel/dev_genius/models/Qwen3.5-27B` |
| Prompts | `A3_AMBIG_01_securities_penalty_restitution`, `A3_AMBIG_06_land_use_compensation` |
| Candidate sites | top 8 localized residual/MLP sites from `sweep_v4/scotus_article3_ambiguous_thought_state_localization_20260502_003317` |
| Position | `decode` |
| Alphas | `-1.0`, `-2.0`, normalized by sqrt(site count) |
| Random controls | `2` same-site random bundles |
| Thought / answer budget | `2048` / `2048` |
| Short-budget smoke | `False` |

The candidate sites match the public-baseline smoke: `L61 residual tail32`, `L62 residual tail32`, `L62 residual thought_tail16`, `L58 residual tail32`, `L59 residual tail32`, `L54 residual tail32`, `L63 residual thought_tail16`, and `L62 MLP thought_tail16`.

## Result

The reverse-direction test did not move holdings toward public-rights adjudication. It mostly retained or strengthened private-rights holdings. The only automatic public-rights label in this run came from a random control, not the candidate.

Polarity summary:

| Condition | Alpha | n | Private score | Public score | Net private-public | Private rate | Public rate | Mixed rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.0 | 2 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| random | -1.0 | 4 | 2.000 | 1.000 | 1.000 | 0.750 | 0.000 | 0.250 |
| random | -2.0 | 4 | 2.500 | 0.750 | 1.750 | 0.500 | 0.250 | 0.250 |
| candidate | -1.0 | 2 | 3.500 | 1.000 | 2.500 | 1.000 | 0.000 | 0.000 |
| candidate | -2.0 | 2 | 1.000 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 |

Prompt-matched proposition deltas also do not support promotion:

| Segment | Alpha | Target-minus-random | Net-minus-random | Target strongest wins | Net strongest wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| thinking | -1.0 | 0.250 | 0.250 | 0.000 | 0.000 |
| thinking | -2.0 | 0.250 | 1.250 | 0.500 | 0.500 |
| answer | -1.0 | -1.250 | -1.500 | 0.000 | 0.000 |
| answer | -2.0 | 0.750 | 1.000 | 0.500 | 0.500 |

Because the prompt bank encodes private-rights frames as `expected_frames`, these proposition deltas are not a direct public-push metric. They are included only to show that the candidate does not beat the same-site random controls even under the harness's native metric.

## Decision

- Do not promote the top-8 localized residual/MLP unit-add bundle in either direction.
- The simple additive localized surface is not a bidirectional Article III actuator under these settings.
- The result strengthens the case for changing actuator family rather than doing more alphas on the same unit-add bundle.
- The next useful branch is a conditional/non-additive controller over localized late sites, or a different localization source that patches generated-token trajectories rather than teacher-forced inserted-thought tails.

## Scorer Fix

During this run, the Article III conclusion-polarity scorer was patched to preserve `alpha`, `random_index`, and `layer` in `polarity_rows.jsonl`, and to summarize by condition/candidate/alpha. This avoids collapsing future intervention sweeps into a single condition-level summary.

## Artifacts

- `sweep_v4/scotus_thinking_localized_direction_poke_20260502_022548/report.md`
- `sweep_v4/scotus_thinking_localized_direction_poke_20260502_022548/generations.jsonl`
- `sweep_v4/scotus_thinking_localized_direction_poke_20260502_022548/candidate_vs_prompt_matched_random.jsonl`
- `sweep_v4/scotus_article3_conclusion_polarity_20260502_033340/report.md`
