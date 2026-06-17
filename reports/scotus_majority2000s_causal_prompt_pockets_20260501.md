# SCOTUS Majority-2000s Causal Prompt Pockets

## Purpose

The aggregate causal pilots did not clear the steering gate. This report identifies the small set of prompt-level pockets where candidate completions had positive absolute movement and also beat prompt-matched random controls by at least one frame-hit unit.

These rows are investigative leads, not steering evidence. The queue is blind and pairwise so a reviewer can check whether the apparent movement is visible in legal reasoning rather than a keyword artifact.

## Outputs

- Blind review queue: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_majority2000s_causal_review_blind_20260501.jsonl`
- Key file: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_majority2000s_causal_review_key_20260501.jsonl`
- Selected candidate cells: `8`
- Pairwise review rows: `22`

## Selected Candidate Cells

| Run | Prompt | Issue | Alpha | Candidate target delta | Matched target delta | Candidate net delta | Matched net delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| split_01_best_probe_direction__all | JP01_agency_private_company | Judicial Power | 0.05 | 6.00 | 4.40 | 6.00 | 4.40 |
| split_00_best_probe_direction__last | EA03_gun_school_zone | Economic Activity | 0.02 | 1.00 | 1.00 | 4.00 | 4.00 |
| split_00_best_probe_direction__last | JP01_agency_private_company | Judicial Power | 0.05 | 5.00 | 2.80 | 5.00 | 2.80 |
| split_00_best_probe_direction__last | EA03_gun_school_zone | Economic Activity | 0.1 | 1.00 | 0.60 | 4.00 | 2.50 |
| split_01_best_probe_direction__all | EA01_commercial_remedy | Economic Activity | 0.05 | 3.00 | 2.00 | 3.00 | 2.00 |
| split_01_best_probe_direction__all | EA01_commercial_remedy | Economic Activity | 0.02 | 2.00 | 1.60 | 2.00 | 1.60 |
| split_00_best_probe_direction__last | JP01_agency_private_company | Judicial Power | 0.1 | 4.00 | 1.50 | 4.00 | 1.50 |
| split_01_best_probe_direction__all | EA01_commercial_remedy | Economic Activity | 0.01 | 1.00 | 1.00 | 1.00 | 1.00 |

## Pair Types

| Comparison | Rows |
| --- | --- |
| base | 8 |
| random_closest_mean | 8 |
| random_strongest_target | 6 |

## Review Rule

Use the blind queue only. A prompt family should not be promoted unless the candidate side wins against both baseline and random-control comparisons without coherence degradation.

## Adjudication Update

Internal qualitative adjudication has been recorded here:

- Report: `reports/scotus_majority2000s_causal_review_adjudication_20260501.md`
- Adjudicated rows: `data/scotus/scotus_majority2000s_causal_review_adjudicated_20260501.jsonl`

Only two reviewed pockets survived the pairwise rule:

- `EA03_gun_school_zone` / `Economic Activity` from `split_00_best_probe_direction__last` at alpha `0.02`
- `EA01_commercial_remedy` / `Economic Activity` from `split_01_best_probe_direction__all` at alpha `0.02`

The Judicial Power pockets did not survive because strongest random controls matched or beat the candidate. The survivor should be treated as a narrow Commerce Clause / Economic Activity hypothesis, not as evidence for broad justice-style steering.
