# SCOTUS Article III Holding Review Adjudication

## Purpose

Calibrate Article III final-holding labels against the automatic proposition and conclusion-polarity scorers before running more actuator searches. This separates evaluator failure from intervention failure.

## Status

This is an internal Codex triage adjudication of the answer-only queue. It is useful for scorer debugging and next-run triage, but it is not independent blind human review and is not a final promotion gate.

## Inputs And Outputs

| Field | Value |
| --- | --- |
| Blind queue | data/scotus/scotus_article3_holding_review_blind_20260501.jsonl |
| Hidden key | data/scotus/scotus_article3_holding_review_key_20260501.jsonl |
| Adjudicated rows | data/scotus/scotus_article3_holding_review_adjudicated_20260501.jsonl |
| JSON summary | reports/scotus_article3_holding_review_adjudication_20260501.json |
| Rows | 24 |

## Holding Label Counts

| Holding label | N |
| --- | --- |
| article3_objection_fails_public_rights_permissible | 12 |
| article3_objection_succeeds_private_rights | 5 |
| mixed_or_distinction_only | 6 |
| unclear_or_incoherent | 1 |

## Reasoning Quality Counts

| Quality label | N |
| --- | --- |
| legally_coherent | 11 |
| legally_confused | 2 |
| nonresponsive_or_truncated | 6 |
| partly_coherent | 5 |

## Mask Label Counts

| Mask label | N |
| --- | --- |
| direct_reasoning | 18 |
| not_assessable | 6 |

## Hidden Condition Versus Reviewed Holding

| Condition | N | Private succeeds | Public fails/permissible | Mixed | Unclear | Mean confidence |
| --- | --- | --- | --- | --- | --- | --- |
| neutral | 8 | 2 | 6 | 0 | 0 | 0.69 |
| private_rights | 8 | 2 | 2 | 4 | 0 | 0.81 |
| public_rights | 8 | 1 | 4 | 2 | 1 | 0.81 |

## Automatic Polarity Versus Reviewed Holding

Exact agreement on eligible reviewed rows: `15/23` = `0.6522`.

| Automatic holding | Reviewed holding | N |
| --- | --- | --- |
| article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | 8 |
| article3_objection_fails_public_rights_permissible | mixed_or_distinction_only | 3 |
| article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | 5 |
| article3_objection_succeeds_private_rights | mixed_or_distinction_only | 1 |
| mixed_or_distinction_only | article3_objection_fails_public_rights_permissible | 4 |
| mixed_or_distinction_only | mixed_or_distinction_only | 2 |
| mixed_or_distinction_only | unclear_or_incoherent | 1 |

## Proposition/Polarity Score By Reviewed Holding

| Reviewed holding | N | Mean proposition delta vs neutral | Mean auto private-minus-public |
| --- | --- | --- | --- |
| article3_objection_fails_public_rights_permissible | 12 | 0.417 | -1.000 |
| article3_objection_succeeds_private_rights | 5 | -0.600 | 1.600 |
| mixed_or_distinction_only | 6 | 1.667 | -0.500 |
| unclear_or_incoherent | 1 | 1.000 | 0.000 |

## Prompt-Level Rows

| Prompt key | Hidden condition | Reviewed holding | Automatic holding | Quality | Confidence |
| --- | --- | --- | --- | --- | --- |
| A3_PRIV_01_securities_penalty | neutral | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_PRIV_01_securities_penalty | private_rights | mixed_or_distinction_only | article3_objection_succeeds_private_rights | nonresponsive_or_truncated | medium |
| A3_PRIV_01_securities_penalty | public_rights | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | partly_coherent | medium |
| A3_PRIV_02_bankruptcy_counterclaim | neutral | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_confused | low |
| A3_PRIV_02_bankruptcy_counterclaim | private_rights | mixed_or_distinction_only | mixed_or_distinction_only | nonresponsive_or_truncated | high |
| A3_PRIV_02_bankruptcy_counterclaim | public_rights | unclear_or_incoherent | mixed_or_distinction_only | legally_confused | high |
| A3_PRIV_03_contract_damages_board | neutral | article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | partly_coherent | medium |
| A3_PRIV_03_contract_damages_board | private_rights | article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | legally_coherent | high |
| A3_PRIV_03_contract_damages_board | public_rights | mixed_or_distinction_only | article3_objection_fails_public_rights_permissible | nonresponsive_or_truncated | medium |
| A3_PRIV_04_tort_agency_assignment | neutral | article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | legally_coherent | high |
| A3_PRIV_04_tort_agency_assignment | private_rights | article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | legally_coherent | high |
| A3_PRIV_04_tort_agency_assignment | public_rights | article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | legally_coherent | high |
| A3_PUBLIC_01_benefits_eligibility | neutral | article3_objection_fails_public_rights_permissible | mixed_or_distinction_only | legally_coherent | high |
| A3_PUBLIC_01_benefits_eligibility | private_rights | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_PUBLIC_01_benefits_eligibility | public_rights | article3_objection_fails_public_rights_permissible | mixed_or_distinction_only | legally_coherent | high |
| A3_PUBLIC_02_patent_review | neutral | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | partly_coherent | medium |
| A3_PUBLIC_02_patent_review | private_rights | mixed_or_distinction_only | mixed_or_distinction_only | nonresponsive_or_truncated | medium |
| A3_PUBLIC_02_patent_review | public_rights | article3_objection_fails_public_rights_permissible | mixed_or_distinction_only | partly_coherent | medium |
| A3_PUBLIC_03_customs_tariff | neutral | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_PUBLIC_03_customs_tariff | private_rights | article3_objection_fails_public_rights_permissible | mixed_or_distinction_only | legally_coherent | high |
| A3_PUBLIC_03_customs_tariff | public_rights | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_PUBLIC_04_workplace_safety_penalty | neutral | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | partly_coherent | medium |
| A3_PUBLIC_04_workplace_safety_penalty | private_rights | mixed_or_distinction_only | article3_objection_fails_public_rights_permissible | nonresponsive_or_truncated | medium |
| A3_PUBLIC_04_workplace_safety_penalty | public_rights | mixed_or_distinction_only | article3_objection_fails_public_rights_permissible | nonresponsive_or_truncated | high |

## Gate Interpretation

- The automatic polarity scorer is useful for triage, but the confusion table shows it is not a clean final-holding gate.
- This triage pass weakens the earlier automatic-only reading that private-rights scratchpads cleanly moved final answers. The private-rights condition produced many mixed/truncated answers rather than robust private-rights holdings.
- The generated answers are too often truncated for this queue to be a final adjudication surface. The next evaluator repair should regenerate the counterfactual-thought answers with a longer answer budget or a stricter complete-answer stop condition.
- No actuator candidate is promoted by this adjudication. The next actuator run still needs reasoning-trace movement and final-answer movement against random/source/text/prompt controls.

## JSON Summary

`reports/scotus_article3_holding_review_adjudication_20260501.json` contains the machine-readable version of this report.
