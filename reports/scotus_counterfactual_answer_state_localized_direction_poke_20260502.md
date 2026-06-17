# SCOTUS Counterfactual Answer-State Localized Direction Poke

## Purpose

Test whether directions localized from local Qwen 3.5 counterfactual answer states act as a no-mask Article III public/private actuator.

This was a causal gate, not a promotion run: the sample is deliberately small, but the generation budgets were complete and the candidate had to beat a same-site random control.

## Artifacts

- Source references: `sweep_v4/scotus_counterfactual_thoughts_20260502_052323`
- Source polarity scoring: `sweep_v4/scotus_article3_conclusion_polarity_20260502_053601`
- Answer-state localization: `sweep_v4/scotus_article3_counterfactual_answer_state_localization_20260502_053619`
- Causal poke: `sweep_v4/scotus_thinking_localized_direction_poke_20260502_054329`
- Final-answer polarity scoring: `sweep_v4/scotus_article3_conclusion_polarity_20260502_060442`
- New localizer script: `scripts/experiments/scotus/localize_article3_counterfactual_answer_states.py`

## Setup

The localization source was a full local Qwen 3.5 counterfactual visible-thought run over Article III ambiguous prompts `0-7`, answer budget `3072`, short-budget smoke `False`.

Conclusion-polarity scoring on those source references showed that inserted thoughts were directionally useful but weak:

| condition | n | private | public | net | private rate | public rate | mixed rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| neutral | 8 | 0.625 | 1.750 | -1.125 | 0.125 | 0.625 | 0.250 |
| private_rights | 8 | 1.875 | 1.375 | 0.500 | 0.250 | 0.125 | 0.625 |
| public_rights | 8 | 0.375 | 1.750 | -1.375 | 0.000 | 0.875 | 0.125 |

The answer-state localizer used private-minus-public deltas from the generated final-answer trajectories, not teacher-forced baseline thoughts. Top sites:

| rank | site | score-null | consistency |
| ---: | --- | ---: | ---: |
| 1 | `L58 residual pre_answer_last` | 0.5673 | 0.8649 |
| 2 | `L62 mixer pre_answer_last` | 0.5638 | 0.8126 |
| 3 | `L59 mixer pre_answer_last` | 0.5215 | 0.9225 |
| 4 | `L63 residual pre_answer_last` | 0.4517 | 0.8516 |

## Causal Gate

- Prompt bank: `data/scotus/scotus_article3_ambiguous_poke_prompts_v1.jsonl`
- Target prompts: `2`, `4`
- Candidate: top four answer-state sites above, normalized by site count
- Position: generated-token decode
- Alpha: `1.0`
- Random controls: `1` same-site random unit bundle
- Thought/answer budgets: `2048`/`2048`
- Short-budget smoke: `False`
- Thoughts closed: `6/6`
- Answers nonempty: `6/6`

Segment scoring from the poke report:

| segment | condition | target delta | net delta |
| --- | --- | ---: | ---: |
| thinking | random_unit | 1.000 | 2.000 |
| thinking | candidate | -0.500 | 0.000 |
| answer | random_unit | 0.500 | 0.000 |
| answer | candidate | -1.000 | -2.000 |

Candidate versus prompt-matched random:

| segment | target-minus-random | net-minus-random | strongest target wins | strongest net wins |
| --- | ---: | ---: | ---: | ---: |
| thinking | -1.500 | -2.000 | 0.000 | 0.000 |
| answer | -1.500 | -2.000 | 0.000 | 0.500 |

Conclusion-polarity scoring:

| condition | n | private | public | net | private rate | public rate | mixed rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 2 | 1.000 | 2.000 | -1.000 | 0.000 | 1.000 | 0.000 |
| random_unit | 2 | 2.000 | 3.500 | -1.500 | 0.000 | 1.000 | 0.000 |
| candidate | 2 | 1.000 | 2.000 | -1.000 | 0.000 | 0.500 | 0.500 |

## Decision

Do not promote the counterfactual answer-state localized additive bundle.

The new localization source nominated a late pre-answer distributed surface, but additive decode-time steering on that surface did not move the visible reasoning trajectory or final holding toward the private-rights target. It failed the strongest-random gate in both thinking and answer segment scoring, and final-answer polarity remained public or mixed.

This closes the immediate "try the new late answer-state surface as direct act-add" branch. Future work should not widen this exact top-four/top-late pre-answer additive family without a new causal reason. The next useful direction is a different actuator family: trained multi-site controllers over generated-token trajectories, causal-tracing-selected patches over actual conclusion tokens, or another non-additive intervention that keeps the no-mask/random-control gates.
