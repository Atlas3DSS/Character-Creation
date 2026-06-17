# SCOTUS Article III Dominance Adjudication v1

## Purpose

This applies a single-pass dominance review to the Article III source queue. These labels are more useful than keyword labels, but they are not gold labels; they need a second review before any steering claim.

## Outputs

- Reviewed queue: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_article3_dominance_review_adjudicated_v1.jsonl`
- Probe-ready labels: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_article3_dominance_frame_labels_v1.jsonl`
- Source key queue: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_article3_dominance_review_key_v1.jsonl`
- Reviewed rows: `80`
- Probe-ready rows: `70`
- Mixed/rejected rows excluded from probe labels: `10`

## Review Label Counts

| Review label | Rows |
| --- | --- |
| article1_tribunal_dominant | 9 |
| mixed_comparative | 10 |
| private_rights_dominant | 28 |
| public_rights_dominant | 33 |

## Confidence Counts

| Confidence | Rows |
| --- | --- |
| high | 36 |
| low | 1 |
| medium | 43 |

## Probe-Ready Frame Counts

| Frame | Rows |
| --- | --- |
| article3_article1_tribunal | 9 |
| article3_private_rights | 28 |
| article3_public_rights | 33 |

## Case/Frame Coverage

| Case | Frame | Rows |
| --- | --- | --- |
| Atlas Roofing Co. v. Occupational Safety & Health Review Commission | article3_public_rights | 4 |
| Axon Enterprise, Inc. v. Federal Trade Commission | article3_private_rights | 8 |
| Commodity Futures Trading Commission v. Schor | article3_article1_tribunal | 1 |
| Commodity Futures Trading Commission v. Schor | article3_private_rights | 2 |
| Granfinanciera, S.A. v. Nordberg | article3_private_rights | 4 |
| Granfinanciera, S.A. v. Nordberg | article3_public_rights | 8 |
| Northern Pipeline Construction Co. v. Marathon Pipe Line Co. | article3_article1_tribunal | 3 |
| Northern Pipeline Construction Co. v. Marathon Pipe Line Co. | article3_public_rights | 1 |
| Oil States Energy Services, LLC v. Greene's Energy Group, LLC | article3_private_rights | 1 |
| Oil States Energy Services, LLC v. Greene's Energy Group, LLC | article3_public_rights | 5 |
| Securities and Exchange Commission v. Jarkesy | article3_article1_tribunal | 1 |
| Securities and Exchange Commission v. Jarkesy | article3_private_rights | 5 |
| Securities and Exchange Commission v. Jarkesy | article3_public_rights | 7 |
| Stern v. Marshall | article3_article1_tribunal | 1 |
| Stern v. Marshall | article3_private_rights | 4 |
| Stern v. Marshall | article3_public_rights | 2 |
| Thomas v. Union Carbide Agricultural Products Co. | article3_public_rights | 6 |
| Wellness International Network, Ltd. v. Sharif | article3_article1_tribunal | 3 |
| Wellness International Network, Ltd. v. Sharif | article3_private_rights | 4 |

## Use Rules

1. Treat these as `single_reviewer_dominance_v1`, not final gold labels.
2. Rerun cue-masked probes from the probe-ready label file before any causal generation.
3. Do not promote a direction unless it survives text-baseline checks and prompt-matched same-layer random controls.
4. Rows labeled `mixed_comparative` are useful for evaluator training but should not be used as binary public/private labels.
