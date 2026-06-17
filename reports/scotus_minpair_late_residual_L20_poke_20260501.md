# SCOTUS Commerce Pocket Poke Summary

## Purpose

Summarize targeted Commerce Clause / Economic Activity causal pokes by prompt family.
A row is promising only if it beats prompt-matched same-layer random controls, especially the strongest random control for the same prompt and alpha.

## Aggregate

| Run | Family | Alpha | N | Cand target | Rand target | Matched target | Cand net | Rand net | Matched net | Target win | Net win | Target strongest win | Net strongest win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| directions__all | commerce_limits | 0.003 | 6 | -0.167 | 0.000 | -0.167 | 0.833 | 0.792 | 0.042 | 0.33 | 0.50 | 0.17 | 0.17 |
| directions__all | commerce_limits | 0.01 | 6 | -0.667 | -0.083 | -0.583 | 0.167 | 0.208 | -0.042 | 0.00 | 0.33 | 0.00 | 0.00 |
| directions__all | commerce_limits | 0.005 | 6 | -0.500 | -0.292 | -0.208 | 0.000 | 0.125 | -0.125 | 0.17 | 0.00 | 0.00 | 0.00 |

## Top Prompt Rows

| Run | Prompt | Family | Alpha | Cand target | Rand target | Matched target | Cand net | Rand net | Matched net | Beats target max | Beats net max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| directions__all | EA_LIMIT_04_home_arson_private_dwelling | commerce_limits | 0.003 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | Y | Y |
| directions__all | EA_LIMIT_01_school_zone_no_hook | commerce_limits | 0.003 | 1.000 | 0.500 | 0.500 | 2.000 | 1.250 | 0.750 | N | N |
| directions__all | EA_LIMIT_01_school_zone_no_hook | commerce_limits | 0.01 | -1.000 | -0.500 | -0.500 | -1.000 | -1.500 | 0.500 | N | N |
| directions__all | EA_LIMIT_02_local_school_fights | commerce_limits | 0.003 | 1.000 | 1.750 | -0.750 | 4.000 | 3.500 | 0.500 | N | N |
| directions__all | EA_LIMIT_03_civil_violence_remedy | commerce_limits | 0.01 | 0.000 | 0.000 | 0.000 | 0.000 | -0.250 | 0.250 | N | N |
| directions__all | EA_LIMIT_05_local_family_obligation | commerce_limits | 0.003 | -2.000 | -2.000 | 0.000 | 0.000 | 0.000 | 0.000 | N | N |
| directions__all | EA_LIMIT_01_school_zone_no_hook | commerce_limits | 0.005 | -1.000 | -1.000 | 0.000 | -1.000 | -1.000 | 0.000 | N | N |
| directions__all | EA_LIMIT_04_home_arson_private_dwelling | commerce_limits | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N | N |
| directions__all | EA_LIMIT_04_home_arson_private_dwelling | commerce_limits | 0.01 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N | N |
| directions__all | EA_LIMIT_05_local_family_obligation | commerce_limits | 0.005 | -2.000 | -1.500 | -0.500 | 0.000 | 0.000 | 0.000 | N | N |
| directions__all | EA_LIMIT_06_school_curriculum_mandate | commerce_limits | 0.005 | -2.000 | -1.500 | -0.500 | -1.000 | -1.000 | 0.000 | N | N |
| directions__all | EA_LIMIT_05_local_family_obligation | commerce_limits | 0.01 | -2.000 | -1.000 | -1.000 | 0.000 | 0.000 | 0.000 | N | N |
| directions__all | EA_LIMIT_03_civil_violence_remedy | commerce_limits | 0.003 | 0.000 | 0.250 | -0.250 | 0.000 | 0.250 | -0.250 | N | N |
| directions__all | EA_LIMIT_02_local_school_fights | commerce_limits | 0.01 | 1.000 | 1.750 | -0.750 | 3.000 | 3.250 | -0.250 | N | N |
| directions__all | EA_LIMIT_02_local_school_fights | commerce_limits | 0.005 | 1.000 | 2.000 | -1.000 | 4.000 | 4.250 | -0.250 | N | N |
| directions__all | EA_LIMIT_03_civil_violence_remedy | commerce_limits | 0.005 | 1.000 | 0.250 | 0.750 | -2.000 | -1.500 | -0.500 | N | N |
| directions__all | EA_LIMIT_06_school_curriculum_mandate | commerce_limits | 0.01 | -2.000 | -0.750 | -1.250 | -1.000 | -0.250 | -0.750 | N | N |
| directions__all | EA_LIMIT_06_school_curriculum_mandate | commerce_limits | 0.003 | -2.000 | -0.500 | -1.500 | -2.000 | -0.250 | -1.750 | N | N |

## Reading Rule

- Mean wins over random controls are suggestive only.
- Strongest-random wins are the important gate for promotion.
- This summary still uses keyword/proposition frame counts; any survivor needs blind text review before promotion.
