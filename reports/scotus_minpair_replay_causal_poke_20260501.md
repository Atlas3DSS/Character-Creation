# SCOTUS Commerce Pocket Poke Summary

## Purpose

Summarize targeted Commerce Clause / Economic Activity causal pokes by prompt family.
A row is promising only if it beats prompt-matched same-layer random controls, especially the strongest random control for the same prompt and alpha.

## Aggregate

| Run | Family | Alpha | N | Cand target | Rand target | Matched target | Cand net | Rand net | Matched net | Target win | Net win | Target strongest win | Net strongest win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scotus_minpair_replay_20260501_100514__all | commerce_limits | 0.01 | 6 | 0.333 | -0.306 | 0.639 | 0.833 | 0.250 | 0.583 | 0.50 | 0.67 | 0.17 | 0.17 |
| scotus_minpair_replay_20260501_100514__all | commerce_limits | 0.005 | 6 | -0.167 | -0.333 | 0.167 | 0.500 | 0.000 | 0.500 | 0.33 | 0.67 | 0.00 | 0.00 |
| scotus_minpair_replay_20260501_100514__all | commerce_limits | 0.05 | 6 | -0.333 | 0.139 | -0.472 | 0.167 | 0.778 | -0.611 | 0.00 | 0.17 | 0.00 | 0.00 |
| scotus_minpair_replay_20260501_100514__all | commerce_limits | 0.02 | 6 | -0.833 | 0.028 | -0.861 | -0.500 | 0.389 | -0.889 | 0.00 | 0.00 | 0.00 | 0.00 |

## Top Prompt Rows

| Run | Prompt | Family | Alpha | Cand target | Rand target | Matched target | Cand net | Rand net | Matched net | Beats target max | Beats net max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_04_home_arson_private_dwelling | commerce_limits | 0.01 | 3.000 | 0.000 | 3.000 | 3.000 | 0.000 | 3.000 | Y | Y |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_02_local_school_fights | commerce_limits | 0.005 | 3.000 | 1.000 | 2.000 | 5.000 | 2.333 | 2.667 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_01_school_zone_no_hook | commerce_limits | 0.01 | 0.000 | -0.667 | 0.667 | 1.000 | -0.833 | 1.833 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_06_school_curriculum_mandate | commerce_limits | 0.005 | -1.000 | -1.667 | 0.667 | 0.000 | -1.167 | 1.167 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_06_school_curriculum_mandate | commerce_limits | 0.01 | 0.000 | -1.333 | 1.333 | 0.000 | -1.000 | 1.000 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_02_local_school_fights | commerce_limits | 0.05 | 2.000 | 2.167 | -0.167 | 6.000 | 5.000 | 1.000 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_01_school_zone_no_hook | commerce_limits | 0.005 | -1.000 | -1.000 | 0.000 | -1.000 | -1.667 | 0.667 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_05_local_family_obligation | commerce_limits | 0.005 | -2.000 | -1.333 | -0.667 | 0.000 | -0.333 | 0.333 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_02_local_school_fights | commerce_limits | 0.01 | 1.000 | 1.667 | -0.667 | 4.000 | 3.833 | 0.167 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_04_home_arson_private_dwelling | commerce_limits | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_05_local_family_obligation | commerce_limits | 0.02 | -2.000 | -1.667 | -0.333 | 0.000 | 0.000 | 0.000 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_04_home_arson_private_dwelling | commerce_limits | 0.02 | 0.000 | 0.167 | -0.167 | 0.000 | 0.167 | -0.167 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_03_civil_violence_remedy | commerce_limits | 0.02 | 0.000 | 1.167 | -1.167 | 0.000 | 0.167 | -0.167 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_04_home_arson_private_dwelling | commerce_limits | 0.05 | 0.000 | 0.500 | -0.500 | 0.000 | 0.500 | -0.500 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_01_school_zone_no_hook | commerce_limits | 0.05 | -1.000 | -0.333 | -0.667 | -1.000 | -0.500 | -0.500 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_01_school_zone_no_hook | commerce_limits | 0.02 | -1.000 | -0.333 | -0.667 | -1.000 | -0.333 | -0.667 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_03_civil_violence_remedy | commerce_limits | 0.01 | 0.000 | 0.500 | -0.500 | -1.000 | -0.167 | -0.833 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_06_school_curriculum_mandate | commerce_limits | 0.05 | -1.000 | -0.500 | -0.500 | -1.000 | 0.000 | -1.000 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_03_civil_violence_remedy | commerce_limits | 0.05 | 0.000 | 1.000 | -1.000 | -1.000 | 0.333 | -1.333 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_05_local_family_obligation | commerce_limits | 0.05 | -2.000 | -2.000 | 0.000 | -2.000 | -0.667 | -1.333 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_05_local_family_obligation | commerce_limits | 0.01 | -2.000 | -2.000 | 0.000 | -2.000 | -0.333 | -1.667 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_03_civil_violence_remedy | commerce_limits | 0.005 | 0.000 | 1.000 | -1.000 | -1.000 | 0.833 | -1.833 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_06_school_curriculum_mandate | commerce_limits | 0.02 | -2.000 | -0.333 | -1.667 | -2.000 | 0.000 | -2.000 | N | N |
| scotus_minpair_replay_20260501_100514__all | EA_LIMIT_02_local_school_fights | commerce_limits | 0.02 | 0.000 | 1.167 | -1.167 | 0.000 | 2.333 | -2.333 | N | N |

## Reading Rule

- Mean wins over random controls are suggestive only.
- Strongest-random wins are the important gate for promotion.
- This summary still uses keyword/proposition frame counts; any survivor needs blind text review before promotion.
