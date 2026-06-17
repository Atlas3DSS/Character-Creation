# SCOTUS Article III Holding Review Adjudication

## Purpose

Calibrate Article III final-holding labels against the automatic proposition and conclusion-polarity scorers before running more actuator searches. This separates evaluator failure from intervention failure.

## Status

This is an internal Codex triage adjudication of the answer-only queue. It is useful for scorer debugging and next-run triage, but it is not independent blind human review and is not a final promotion gate.

## Inputs And Outputs

| Field | Value |
| --- | --- |
| Blind queue | data/scotus/scotus_article3_holding_review_blind_ambiguous_20260502.jsonl |
| Hidden key | data/scotus/scotus_article3_holding_review_key_ambiguous_20260502.jsonl |
| Adjudicated rows | data/scotus/scotus_article3_holding_review_adjudicated_ambiguous_20260502.jsonl |
| JSON summary | reports/scotus_article3_holding_review_adjudication_ambiguous_20260502.json |
| Rows | 24 |

## Holding Label Counts

| Holding label | N |
| --- | --- |
| article3_objection_fails_public_rights_permissible | 16 |
| article3_objection_succeeds_private_rights | 8 |

## Reasoning Quality Counts

| Quality label | N |
| --- | --- |
| legally_coherent | 24 |

## Mask Label Counts

| Mask label | N |
| --- | --- |
| direct_reasoning | 24 |

## Hidden Condition Versus Reviewed Holding

| Condition | N | Private succeeds | Public fails/permissible | Mixed | Unclear | Mean confidence |
| --- | --- | --- | --- | --- | --- | --- |
| neutral | 8 | 1 | 7 | 0 | 0 | 1.00 |
| private_rights | 8 | 6 | 2 | 0 | 0 | 1.00 |
| public_rights | 8 | 1 | 7 | 0 | 0 | 1.00 |

## Automatic Polarity Versus Reviewed Holding

Exact agreement on eligible reviewed rows: `18/24` = `0.75`.

| Automatic holding | Reviewed holding | N |
| --- | --- | --- |
| article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | 13 |
| article3_objection_fails_public_rights_permissible | article3_objection_succeeds_private_rights | 3 |
| article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | 5 |
| mixed_or_distinction_only | article3_objection_fails_public_rights_permissible | 3 |

## Proposition/Polarity Score By Reviewed Holding

| Reviewed holding | N | Mean proposition delta vs neutral | Mean auto private-minus-public |
| --- | --- | --- | --- |
| article3_objection_fails_public_rights_permissible | 16 | 0.312 | -1.125 |
| article3_objection_succeeds_private_rights | 8 | 1.750 | 0.875 |

## Prompt-Level Rows

| Prompt key | Hidden condition | Reviewed holding | Automatic holding | Quality | Confidence |
| --- | --- | --- | --- | --- | --- |
| A3_AMBIG_01_securities_penalty_restitution | neutral | article3_objection_fails_public_rights_permissible | mixed_or_distinction_only | legally_coherent | high |
| A3_AMBIG_01_securities_penalty_restitution | private_rights | article3_objection_succeeds_private_rights | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_AMBIG_01_securities_penalty_restitution | public_rights | article3_objection_fails_public_rights_permissible | mixed_or_distinction_only | legally_coherent | high |
| A3_AMBIG_02_bankruptcy_counterclaim_distribution | neutral | article3_objection_succeeds_private_rights | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_AMBIG_02_bankruptcy_counterclaim_distribution | private_rights | article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | legally_coherent | high |
| A3_AMBIG_02_bankruptcy_counterclaim_distribution | public_rights | article3_objection_succeeds_private_rights | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_AMBIG_03_patent_review_parallel_litigation | neutral | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_AMBIG_03_patent_review_parallel_litigation | private_rights | article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | legally_coherent | high |
| A3_AMBIG_03_patent_review_parallel_litigation | public_rights | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_AMBIG_04_customs_penalty_forfeiture | neutral | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_AMBIG_04_customs_penalty_forfeiture | private_rights | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_AMBIG_04_customs_penalty_forfeiture | public_rights | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_AMBIG_05_industry_fund_contribution | neutral | article3_objection_fails_public_rights_permissible | mixed_or_distinction_only | legally_coherent | high |
| A3_AMBIG_05_industry_fund_contribution | private_rights | article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | legally_coherent | high |
| A3_AMBIG_05_industry_fund_contribution | public_rights | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_AMBIG_06_land_use_compensation | neutral | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_AMBIG_06_land_use_compensation | private_rights | article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | legally_coherent | high |
| A3_AMBIG_06_land_use_compensation | public_rights | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_AMBIG_07_benefits_fraud_recoupment | neutral | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_AMBIG_07_benefits_fraud_recoupment | private_rights | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_AMBIG_07_benefits_fraud_recoupment | public_rights | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_AMBIG_08_workplace_penalty_compensation | neutral | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |
| A3_AMBIG_08_workplace_penalty_compensation | private_rights | article3_objection_succeeds_private_rights | article3_objection_succeeds_private_rights | legally_coherent | high |
| A3_AMBIG_08_workplace_penalty_compensation | public_rights | article3_objection_fails_public_rights_permissible | article3_objection_fails_public_rights_permissible | legally_coherent | high |

## Gate Interpretation

- The ambiguous prompt bank is a better calibration surface than the original fact-pattern-determined prompts.
- The private-rights inserted-thought condition shows strong final-holding movement relative to neutral and public-rights conditions, but this remains text-prefill calibration rather than activation actuator evidence.
- The automatic polarity scorer is directionally useful here, but reviewed holding labels are still required for promotion because prior runs showed regex confusion on mixed legal answers.
- No actuator candidate is promoted by this adjudication. The next actuator run still needs the model to generate the target reasoning trajectory itself under activation intervention and beat random/source/text/prompt controls.

## JSON Summary

`reports/scotus_article3_holding_review_adjudication_ambiguous_20260502.json` contains the machine-readable version of this report.
