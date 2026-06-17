# SCOTUS No-Mask Causal Review Queue

## Purpose

Build a blind pairwise review queue for candidate causal generations using proposition-level deltas when available. The review asks whether apparent target-frame movement is real legal reasoning rather than keyword movement, incoherence, or a prompt/persona mask.

## Inputs

| Source |
| --- |
| sweep_v4/scotus_controlled_bundle_prop_rescore_20260501_193155 |

## Outputs

- Blind queue: `data/scotus/scotus_article3_bundle_no_mask_review_blind_20260501.jsonl`
- Key file: `data/scotus/scotus_article3_bundle_no_mask_review_key_20260501.jsonl`
- Selected candidate cells: `8`
- Pairwise review rows: `19`
- Visible-thinking outputs in queue: `0/38`

## Selected Candidate Cells

| Run | Prompt | Alpha | Candidate target delta | Matched target delta | Candidate net delta | Matched net delta | Strongest random net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scotus_controlled_bundle_poke_20260501_183716 | A3_PUBLIC_01_benefits_eligibility | 0.01 | 2.000 | 1.833 | 2.000 | 1.667 | 1.000 |
| scotus_controlled_bundle_poke_20260501_183716 | A3_PUBLIC_04_workplace_safety_penalty | 0.01 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| scotus_controlled_bundle_poke_20260501_183716 | A3_PRIV_01_securities_penalty | 0.005 | -1.000 | 0.500 | -1.000 | 0.833 | -1.000 |
| scotus_controlled_bundle_poke_20260501_183716 | A3_PRIV_01_securities_penalty | 0.003 | -1.000 | 0.333 | -1.000 | 0.833 | -1.000 |
| scotus_controlled_bundle_poke_20260501_183716 | A3_PRIV_04_tort_agency_assignment | 0.01 | 0.000 | 0.167 | 0.000 | 0.167 | 0.000 |
| scotus_controlled_bundle_poke_20260501_183716 | A3_PRIV_03_contract_damages_board | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| scotus_controlled_bundle_poke_20260501_183716 | A3_PUBLIC_02_patent_review | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| scotus_controlled_bundle_poke_20260501_183716 | A3_PUBLIC_03_customs_tariff | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Review Rule

Use the blind queue first. A candidate should not advance unless its side visibly beats baseline and random controls on target-frame reasoning, preserves coherence, and avoids mask language in any visible thinking trace.

If visible thinking is absent, mark the no-mask field as `no_visible_thinking_to_assess`; this is not evidence of reasoning-basin success.
