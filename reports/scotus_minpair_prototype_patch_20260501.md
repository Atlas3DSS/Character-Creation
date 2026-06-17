# SCOTUS Commerce Pocket Poke Summary

## Purpose

Summarize targeted Commerce Clause / Economic Activity causal pokes by prompt family.
A row is promising only if it beats prompt-matched same-layer random controls, especially the strongest random control for the same prompt and alpha.

## Verdict

The L16+L20 Commerce-limits prototype replacement did not pass promotion. Blend `0.01` had a suggestive matched net delta (`0.625`) but target strongest-random win rate was `0.00` and net strongest-random win rate was only `0.17`. Higher blends reversed or collapsed. The one row that beat strongest random on net score mostly reduced generic Commerce Clause wording on the school-curriculum prompt; it did not create a replicated Commerce-limits reasoning shift.

## Aggregate

| Run | Family | Alpha | N | Cand target | Rand target | Matched target | Cand net | Rand net | Matched net | Target win | Net win | Target strongest win | Net strongest win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scotus_prototype_patch_20260501_123725__all | commerce_limits | 0.01 | 6 | 0.000 | -0.292 | 0.292 | 0.667 | 0.042 | 0.625 | 0.33 | 0.50 | 0.00 | 0.17 |
| scotus_prototype_patch_20260501_123725__all | commerce_limits | 0.03 | 6 | -0.167 | -0.167 | 0.000 | 0.000 | 0.292 | -0.292 | 0.33 | 0.33 | 0.00 | 0.00 |
| scotus_prototype_patch_20260501_123725__all | commerce_limits | 0.05 | 6 | -0.667 | -0.083 | -0.583 | -1.000 | 0.250 | -1.250 | 0.17 | 0.33 | 0.00 | 0.00 |

## Top Prompt Rows

| Run | Prompt | Family | Alpha | Cand target | Rand target | Matched target | Cand net | Rand net | Matched net | Beats target max | Beats net max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scotus_prototype_patch_20260501_123725__all | EA_LIMIT_03_civil_violence_remedy | commerce_limits | 0.01 | 2.000 | 0.500 | 1.500 | 2.000 | -0.750 | 2.750 | N | N |
| scotus_prototype_patch_20260501_123725__all | EA_LIMIT_06_school_curriculum_mandate | commerce_limits | 0.03 | 1.000 | -1.000 | 2.000 | 2.000 | -0.500 | 2.500 | N | N |
| scotus_prototype_patch_20260501_123725__all | EA_LIMIT_06_school_curriculum_mandate | commerce_limits | 0.01 | 0.000 | -1.000 | 1.000 | 1.000 | -0.500 | 1.500 | N | Y |
| scotus_prototype_patch_20260501_123725__all | EA_LIMIT_03_civil_violence_remedy | commerce_limits | 0.03 | 1.000 | 0.250 | 0.750 | 1.000 | -0.500 | 1.500 | N | N |
| scotus_prototype_patch_20260501_123725__all | EA_LIMIT_01_school_zone_no_hook | commerce_limits | 0.05 | 0.000 | -0.250 | 0.250 | 0.000 | -0.250 | 0.250 | N | N |
| scotus_prototype_patch_20260501_123725__all | EA_LIMIT_02_local_school_fights | commerce_limits | 0.01 | 1.000 | 1.250 | -0.250 | 4.000 | 3.750 | 0.250 | N | N |
| scotus_prototype_patch_20260501_123725__all | EA_LIMIT_06_school_curriculum_mandate | commerce_limits | 0.05 | -2.000 | -1.500 | -0.500 | -1.000 | -1.250 | 0.250 | N | N |
| scotus_prototype_patch_20260501_123725__all | EA_LIMIT_04_home_arson_private_dwelling | commerce_limits | 0.01 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N | N |
| scotus_prototype_patch_20260501_123725__all | EA_LIMIT_04_home_arson_private_dwelling | commerce_limits | 0.03 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N | N |
| scotus_prototype_patch_20260501_123725__all | EA_LIMIT_05_local_family_obligation | commerce_limits | 0.03 | -2.000 | -2.000 | 0.000 | -2.000 | -2.000 | 0.000 | N | N |
| scotus_prototype_patch_20260501_123725__all | EA_LIMIT_01_school_zone_no_hook | commerce_limits | 0.01 | -1.000 | -0.500 | -0.500 | -1.000 | -0.750 | -0.250 | N | N |
| scotus_prototype_patch_20260501_123725__all | EA_LIMIT_05_local_family_obligation | commerce_limits | 0.01 | -2.000 | -2.000 | 0.000 | -2.000 | -1.500 | -0.500 | N | N |
| scotus_prototype_patch_20260501_123725__all | EA_LIMIT_01_school_zone_no_hook | commerce_limits | 0.03 | -1.000 | -0.250 | -0.750 | -1.000 | -0.500 | -0.500 | N | N |
| scotus_prototype_patch_20260501_123725__all | EA_LIMIT_05_local_family_obligation | commerce_limits | 0.05 | -2.000 | -1.750 | -0.250 | -2.000 | -1.250 | -0.750 | N | N |
| scotus_prototype_patch_20260501_123725__all | EA_LIMIT_04_home_arson_private_dwelling | commerce_limits | 0.05 | 0.000 | 0.750 | -0.750 | 0.000 | 0.750 | -0.750 | N | N |
| scotus_prototype_patch_20260501_123725__all | EA_LIMIT_03_civil_violence_remedy | commerce_limits | 0.05 | 0.000 | 0.500 | -0.500 | -3.000 | -1.000 | -2.000 | N | N |
| scotus_prototype_patch_20260501_123725__all | EA_LIMIT_02_local_school_fights | commerce_limits | 0.05 | 0.000 | 1.750 | -1.750 | 0.000 | 4.500 | -4.500 | N | N |
| scotus_prototype_patch_20260501_123725__all | EA_LIMIT_02_local_school_fights | commerce_limits | 0.03 | 0.000 | 2.000 | -2.000 | 0.000 | 5.250 | -5.250 | N | N |

## Reading Rule

- Mean wins over random controls are suggestive only.
- Strongest-random wins are the important gate for promotion.
- This summary still uses keyword/proposition frame counts; any survivor needs blind text review before promotion.
