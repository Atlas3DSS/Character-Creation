# SCOTUS Commerce Pocket Poke Summary

## Decision

The targeted Commerce-pocket follow-up does not promote a steerable judicial circuit.

The prior `EA03_gun_school_zone` and `EA01_commercial_remedy` hints did not generalize when tested against larger same-layer random-control floors:

- `split_00__last` / `prompt_last @ L10` on all 12 Commerce prompts: at the key alpha `0.02`, matched target delta was `0.115`, matched net delta was `0.021`, and prompt-level strongest-random win rate was `0.00` in both prompt families.
- `split_01__all` / `excerpt_mean @ L16` on the six authority/remedy prompts: at the key alpha `0.02`, matched target and net deltas were both `-0.479`, and prompt win rate over random means was `0.00`.

Two isolated rows beat strongest random controls by the keyword metric, but manual read does not support promotion:

- `EA_LIMIT_04_home_arson_private_dwelling` at alpha `0.01` shifts toward traditional state police powers and jurisdictional-hook language, but the effect does not replicate across Commerce-limits prompts or at alpha `0.02`.
- `EA_AUTH_04_credit_reporting_remedy` at alpha `0.05` emphasizes statutory damages more than controls, but the direction is negative at the adjudicated alpha `0.02` and does not generalize across authority/remedy prompts.

Read: the Commerce-pocket branch is useful as a falsification of the last surviving prompt-pocket hypothesis. It does not establish controllable legal reasoning.

## Purpose

Summarize targeted Commerce Clause / Economic Activity causal pokes by prompt family.
A row is promising only if it beats prompt-matched same-layer random controls, especially the strongest random control for the same prompt and alpha.

## Aggregate

| Run | Family | Alpha | N | Cand target | Rand target | Matched target | Cand net | Rand net | Matched net | Target win | Net win | Target strongest win | Net strongest win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| split_00__last | commerce_authority_remedy | 0.02 | 6 | 0.500 | 0.333 | 0.167 | 0.500 | 0.333 | 0.167 | 0.33 | 0.33 | 0.00 | 0.00 |
| split_00__last | commerce_limits | 0.01 | 6 | -0.167 | -0.375 | 0.208 | 0.167 | 0.104 | 0.062 | 0.33 | 0.50 | 0.17 | 0.17 |
| split_00__last | commerce_authority_remedy | 0.01 | 6 | 0.333 | 0.292 | 0.042 | 0.333 | 0.292 | 0.042 | 0.33 | 0.33 | 0.00 | 0.00 |
| split_01__all | commerce_authority_remedy | 0.05 | 6 | 0.167 | 0.292 | -0.125 | 0.167 | 0.271 | -0.104 | 0.17 | 0.33 | 0.17 | 0.17 |
| split_00__last | commerce_limits | 0.02 | 6 | -0.167 | -0.229 | 0.062 | 0.167 | 0.292 | -0.125 | 0.50 | 0.67 | 0.00 | 0.00 |
| split_01__all | commerce_authority_remedy | 0.01 | 6 | 0.000 | 0.208 | -0.208 | 0.000 | 0.208 | -0.208 | 0.17 | 0.17 | 0.00 | 0.00 |
| split_00__last | commerce_limits | 0.05 | 6 | -0.667 | -0.375 | -0.292 | -0.167 | 0.125 | -0.292 | 0.00 | 0.50 | 0.00 | 0.00 |
| split_00__last | commerce_authority_remedy | 0.05 | 6 | 0.000 | 0.375 | -0.375 | 0.000 | 0.375 | -0.375 | 0.17 | 0.17 | 0.00 | 0.00 |
| split_01__all | commerce_authority_remedy | 0.02 | 6 | -0.167 | 0.312 | -0.479 | -0.167 | 0.312 | -0.479 | 0.00 | 0.00 | 0.00 | 0.00 |

## Top Prompt Rows

| Run | Prompt | Family | Alpha | Cand target | Rand target | Matched target | Cand net | Rand net | Matched net | Beats target max | Beats net max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| split_00__last | EA_LIMIT_04_home_arson_private_dwelling | commerce_limits | 0.01 | 3.000 | 0.000 | 3.000 | 3.000 | 0.000 | 3.000 | Y | Y |
| split_01__all | EA_AUTH_04_credit_reporting_remedy | commerce_authority_remedy | 0.05 | 2.000 | -0.125 | 2.125 | 2.000 | -0.125 | 2.125 | Y | Y |
| split_00__last | EA_LIMIT_06_school_curriculum_mandate | commerce_limits | 0.02 | 0.000 | -1.500 | 1.500 | 0.000 | -1.250 | 1.250 | N | N |
| split_00__last | EA_LIMIT_01_school_zone_no_hook | commerce_limits | 0.02 | 0.000 | -0.625 | 0.625 | 0.000 | -1.125 | 1.125 | N | N |
| split_00__last | EA_LIMIT_05_local_family_obligation | commerce_limits | 0.05 | -2.000 | -2.000 | 0.000 | 1.000 | -0.125 | 1.125 | N | N |
| split_00__last | EA_LIMIT_03_civil_violence_remedy | commerce_limits | 0.02 | 1.000 | 0.375 | 0.625 | 1.000 | 0.000 | 1.000 | N | N |
| split_00__last | EA_LIMIT_02_local_school_fights | commerce_limits | 0.05 | 1.000 | 1.000 | 0.000 | 4.000 | 3.000 | 1.000 | N | N |
| split_00__last | EA_LIMIT_03_civil_violence_remedy | commerce_limits | 0.01 | 0.000 | 0.125 | -0.125 | 0.000 | -0.875 | 0.875 | N | N |
| split_00__last | EA_AUTH_05_transport_network_safety | commerce_authority_remedy | 0.02 | 2.000 | 1.250 | 0.750 | 2.000 | 1.250 | 0.750 | N | N |
| split_00__last | EA_AUTH_05_transport_network_safety | commerce_authority_remedy | 0.01 | 2.000 | 1.500 | 0.500 | 2.000 | 1.500 | 0.500 | N | N |
| split_00__last | EA_AUTH_02_commercial_mislabeling_damages | commerce_authority_remedy | 0.02 | 1.000 | 0.750 | 0.250 | 1.000 | 0.750 | 0.250 | N | N |
| split_00__last | EA_AUTH_01_homegrown_fungible_market | commerce_authority_remedy | 0.05 | 0.000 | -0.250 | 0.250 | 0.000 | -0.250 | 0.250 | N | N |
| split_00__last | EA_LIMIT_05_local_family_obligation | commerce_limits | 0.01 | -2.000 | -2.000 | 0.000 | 0.000 | -0.250 | 0.250 | N | N |
| split_00__last | EA_LIMIT_05_local_family_obligation | commerce_limits | 0.02 | -2.000 | -2.000 | 0.000 | 0.000 | -0.250 | 0.250 | N | N |
| split_00__last | EA_AUTH_01_homegrown_fungible_market | commerce_authority_remedy | 0.01 | 0.000 | -0.125 | 0.125 | 0.000 | -0.125 | 0.125 | N | N |
| split_01__all | EA_AUTH_01_homegrown_fungible_market | commerce_authority_remedy | 0.01 | 0.000 | -0.125 | 0.125 | 0.000 | -0.125 | 0.125 | N | N |
| split_00__last | EA_LIMIT_04_home_arson_private_dwelling | commerce_limits | 0.05 | 0.000 | 0.000 | 0.000 | 0.000 | -0.125 | 0.125 | N | N |
| split_01__all | EA_AUTH_03_local_cartel_market_effects | commerce_authority_remedy | 0.05 | 0.000 | 0.000 | 0.000 | 0.000 | -0.125 | 0.125 | N | N |
| split_00__last | EA_AUTH_03_local_cartel_market_effects | commerce_authority_remedy | 0.01 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N | N |
| split_00__last | EA_AUTH_04_credit_reporting_remedy | commerce_authority_remedy | 0.01 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N | N |
| split_00__last | EA_AUTH_06_online_market_consumer_remedy | commerce_authority_remedy | 0.01 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N | N |
| split_00__last | EA_AUTH_01_homegrown_fungible_market | commerce_authority_remedy | 0.02 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N | N |
| split_00__last | EA_AUTH_03_local_cartel_market_effects | commerce_authority_remedy | 0.02 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N | N |
| split_00__last | EA_AUTH_04_credit_reporting_remedy | commerce_authority_remedy | 0.02 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N | N |
| split_00__last | EA_AUTH_06_online_market_consumer_remedy | commerce_authority_remedy | 0.02 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N | N |
| split_00__last | EA_AUTH_03_local_cartel_market_effects | commerce_authority_remedy | 0.05 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N | N |
| split_00__last | EA_AUTH_04_credit_reporting_remedy | commerce_authority_remedy | 0.05 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N | N |
| split_01__all | EA_AUTH_03_local_cartel_market_effects | commerce_authority_remedy | 0.01 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N | N |
| split_01__all | EA_AUTH_04_credit_reporting_remedy | commerce_authority_remedy | 0.01 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N | N |
| split_01__all | EA_AUTH_06_online_market_consumer_remedy | commerce_authority_remedy | 0.01 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N | N |

## Reading Rule

- Mean wins over random controls are suggestive only.
- Strongest-random wins are the important gate for promotion.
- This summary still uses keyword/proposition frame counts; any survivor needs blind text review before promotion.
