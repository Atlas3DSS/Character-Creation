# SCOTUS Majority-2000s Causal Review Adjudication

## Purpose

This is an internal qualitative adjudication of the causal prompt-pocket queue. It is not a fully blind human review because the key file was inspected during setup, but it is stricter than the keyword metric: a candidate cell advances only if it visibly beats baseline and matched random controls without coherence degradation.

## Inputs

- Blind queue: `data/scotus/scotus_majority2000s_causal_review_blind_20260501.jsonl`
- Key file: `data/scotus/scotus_majority2000s_causal_review_key_20260501.jsonl`
- Adjudicated rows: `data/scotus/scotus_majority2000s_causal_review_adjudicated_20260501.jsonl`
- Pairwise rows reviewed: `22`

## Cell Decisions

| Run | Prompt | Issue | Alpha | Pairs | Candidate beats baseline | Candidate beats random controls | Candidate beats strongest random | Decision | Labels |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| split_00_best_probe_direction__last | EA03_gun_school_zone | Economic Activity | 0.02 | 2 | True | True | True | advance_reviewed_pocket | base=B_stronger_target_frame, random_closest_mean=B_stronger_target_frame |
| split_00_best_probe_direction__last | EA03_gun_school_zone | Economic Activity | 0.1 | 3 | True | False | False | do_not_advance | base=B_stronger_target_frame, random_closest_mean=B_stronger_target_frame, random_strongest_target=no_material_difference |
| split_00_best_probe_direction__last | JP01_agency_private_company | Judicial Power | 0.05 | 3 | True | False | False | do_not_advance | base=B_stronger_target_frame, random_closest_mean=B_stronger_target_frame, random_strongest_target=no_material_difference |
| split_00_best_probe_direction__last | JP01_agency_private_company | Judicial Power | 0.1 | 3 | True | False | False | do_not_advance | base=B_stronger_target_frame, random_closest_mean=A_stronger_target_frame, random_strongest_target=no_material_difference |
| split_01_best_probe_direction__all | EA01_commercial_remedy | Economic Activity | 0.01 | 2 | True | False | True | do_not_advance | base=A_stronger_target_frame, random_closest_mean=no_material_difference |
| split_01_best_probe_direction__all | EA01_commercial_remedy | Economic Activity | 0.02 | 3 | True | True | True | advance_reviewed_pocket | base=A_stronger_target_frame, random_closest_mean=B_stronger_target_frame, random_strongest_target=A_stronger_target_frame |
| split_01_best_probe_direction__all | EA01_commercial_remedy | Economic Activity | 0.05 | 3 | True | False | False | do_not_advance | base=B_stronger_target_frame, random_closest_mean=no_material_difference, random_strongest_target=A_stronger_target_frame |
| split_01_best_probe_direction__all | JP01_agency_private_company | Judicial Power | 0.05 | 3 | True | False | False | do_not_advance | base=B_stronger_target_frame, random_closest_mean=A_stronger_target_frame, random_strongest_target=A_stronger_target_frame |

## Read

Reviewed pockets that survive the pairwise rule:

- `EA03_gun_school_zone` / `Economic Activity` from `split_00_best_probe_direction__last` at alpha `0.02`
- `EA01_commercial_remedy` / `Economic Activity` from `split_01_best_probe_direction__all` at alpha `0.02`

The most interesting survivor is not a broad justice-style result; it is a narrow Economic Activity prompt pocket. Treat it as a hypothesis for a same-doctrine contrast, not as steering evidence.

## Next Step

Build a narrower Economic Activity / Commerce Clause limits contrast around `EA03_gun_school_zone` and `EA01_commercial_remedy` before any more BF16 hook generation. Do not promote the broad majority-2000s justice-style directions.
