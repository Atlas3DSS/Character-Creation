# SCOTUS Article III Holding Review Adjudication

## Purpose

Calibrate Article III final-holding labels against the automatic proposition and conclusion-polarity scorers before running more actuator searches. This separates evaluator failure from intervention failure.

## Status

This is an internal Codex triage adjudication of the answer-only queue. It is useful for scorer debugging and next-run triage, but it is not independent blind human review and is not a final promotion gate.

## Inputs And Outputs

| Field | Value |
| --- | --- |
| Blind queue | data/scotus/scotus_article3_holding_review_blind_long_20260502.jsonl |
| Hidden key | data/scotus/scotus_article3_holding_review_key_long_20260502.jsonl |
| Adjudicated rows | data/scotus/scotus_article3_holding_review_adjudicated_long_20260502.jsonl |
| JSON summary | reports/scotus_article3_holding_review_adjudication_long_20260502.json |
| Rows | 24 |

## Holding Label Counts

| Holding label | N |
| --- | --- |
| article3_objection_fails_public_rights_permissible | 12 |
| article3_objection_succeeds_private_rights | 10 |
| mixed_or_distinction_only | 2 |

## Reasoning Quality Counts

| Quality label | N |
| --- | --- |
| legally_coherent | 20 |
| partly_coherent | 4 |

## Mask Label Counts

| Mask label | N |
| --- | --- |
| direct_reasoning | 24 |

## Hidden Condition Versus Reviewed Holding

| Condition | N | Private succeeds | Public fails/permissible | Mixed | Unclear | Mean confidence |
| --- | --- | --- | --- | --- | --- | --- |
| neutral | 8 | 3 | 5 | 0 | 0 | 0.94 |
| private_rights | 8 | 4 | 2 | 2 | 0 | 0.88 |
| public_rights | 8 | 3 | 5 | 0 | 0 | 0.94 |

## Automatic Polarity Versus Reviewed Holding

Exact agreement on eligible reviewed rows: `14/24` = `0.5833`.

| Automatic holding | Reviewed holding | N |
| --- | --- | --- |
| article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | 7 |
| article3_objection_fails_public_rights_permissible | article3_objection_succeeds_private_rights | 2 |
| article3_objection_fails_public_rights_permissible | mixed_or_distinction_only | 1 |
| article3_objection_succeeds_private_rights | article3_objection_fails_public_rights_permissible | 1 |
| article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | 7 |
| article3_objection_succeeds_private_rights | mixed_or_distinction_only | 1 |
| mixed_or_distinction_only | article3_objection_fails_public_rights_permissible | 4 |
| mixed_or_distinction_only | article3_objection_succeeds_private_rights | 1 |

## Proposition/Polarity Score By Reviewed Holding

| Reviewed holding | N | Mean proposition delta vs neutral | Mean auto private-minus-public |
| --- | --- | --- | --- |
| article3_objection_fails_public_rights_permissible | 12 | 0.417 | -1.333 |
| article3_objection_succeeds_private_rights | 10 | 0.500 | 0.800 |
| mixed_or_distinction_only | 2 | 1.000 | 0.000 |

## Prompt-Level Rows

| Prompt key | Hidden condition | Reviewed holding | Automatic holding | Quality | Confidence |
| --- | --- | --- | --- | --- | --- |
| A3_PRIV_01_securities_penalty | neutral | article3_objection_fails_public_rights_permissible | mixed_or_distinction_only | legally_coherent | high |
| A3_PRIV_01_securities_penalty | private_rights | mixed_or_distinction_only | article3_objection_succeeds_private_rights | partly_coherent | medium |
| A3_PRIV_01_securities_penalty | public_rights | article3_objection_fails_public_rights_permissible | mixed_or_distinction_only | legally_coherent | high |
| A3_PRIV_02_bankruptcy_counterclaim | neutral | article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | partly_coherent | medium |
| A3_PRIV_02_bankruptcy_counterclaim | private_rights | article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | legally_coherent | high |
| A3_PRIV_02_bankruptcy_counterclaim | public_rights | article3_objection_succeeds_private_rights | article3_objection_fails_public_rights_permissible | partly_coherent | medium |
| A3_PRIV_03_contract_damages_board | neutral | article3_objection_succeeds_private_rights | mixed_or_distinction_only | legally_coherent | high |
| A3_PRIV_03_contract_damages_board | private_rights | article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | legally_coherent | high |
| A3_PRIV_03_contract_damages_board | public_rights | article3_objection_succeeds_private_rights | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_PRIV_04_tort_agency_assignment | neutral | article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | legally_coherent | high |
| A3_PRIV_04_tort_agency_assignment | private_rights | article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | legally_coherent | high |
| A3_PRIV_04_tort_agency_assignment | public_rights | article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | legally_coherent | high |
| A3_PUBLIC_01_benefits_eligibility | neutral | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_PUBLIC_01_benefits_eligibility | private_rights | article3_objection_fails_public_rights_permissible | mixed_or_distinction_only | legally_coherent | high |
| A3_PUBLIC_01_benefits_eligibility | public_rights | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_PUBLIC_02_patent_review | neutral | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_PUBLIC_02_patent_review | private_rights | article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | legally_coherent | high |
| A3_PUBLIC_02_patent_review | public_rights | article3_objection_fails_public_rights_permissible | mixed_or_distinction_only | legally_coherent | high |
| A3_PUBLIC_03_customs_tariff | neutral | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_PUBLIC_03_customs_tariff | private_rights | article3_objection_fails_public_rights_permissible | article3_objection_succeeds_private_rights | legally_coherent | high |
| A3_PUBLIC_03_customs_tariff | public_rights | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_PUBLIC_04_workplace_safety_penalty | neutral | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_PUBLIC_04_workplace_safety_penalty | private_rights | mixed_or_distinction_only | article3_objection_fails_public_rights_permissible | partly_coherent | medium |
| A3_PUBLIC_04_workplace_safety_penalty | public_rights | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |

## Gate Interpretation

- The long-answer queue fixes the mechanical truncation flaw from the 96-token queue, but it does not make automatic scoring reliable.
- The automatic polarity scorer remains a triage aid only: it reached low exact agreement with reviewed holding labels and still confuses discussion of a frame with adoption of that frame.
- The private-rights inserted-thought condition shows only weak final-holding movement relative to neutral; this is evaluator calibration, not actuator evidence.
- No actuator candidate is promoted by this adjudication. The next actuator run still needs reasoning-trace movement and final-answer movement against random/source/text/prompt controls.

## JSON Summary

`reports/scotus_article3_holding_review_adjudication_long_20260502.json` contains the machine-readable version of this report.
