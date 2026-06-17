# SCOTUS Article III Dominance Review Queue v1

## Purpose

This queue supports blind review of whether Article III public/private-rights excerpts have a dominant legal frame. It is designed to replace keyword presence with adjudicated labels before any further promotion or causal steering claim.

## Outputs

- Blind queue: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_article3_dominance_review_blind_v1.jsonl`
- Key queue: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_article3_dominance_review_key_v1.jsonl`
- Source labels: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_article3_source_frame_labels_v1.jsonl`
- Selected excerpts: `80`
- Public/private conflict excerpts: `28`

## Matched Frame Counts

| Frame | Selected excerpts |
| --- | --- |
| article3_article1_tribunal | 31 |
| article3_case_or_controversy | 2 |
| article3_final_judgment_separation | 2 |
| article3_private_rights | 39 |
| article3_public_rights | 69 |

## Case Coverage

| Case | Selected excerpts |
| --- | --- |
| Securities and Exchange Commission v. Jarkesy | 15 |
| Granfinanciera, S.A. v. Nordberg | 14 |
| Wellness International Network, Ltd. v. Sharif | 10 |
| Axon Enterprise, Inc. v. Federal Trade Commission | 10 |
| Stern v. Marshall | 8 |
| Thomas v. Union Carbide Agricultural Products Co. | 6 |
| Oil States Energy Services, LLC v. Greene's Energy Group, LLC | 6 |
| Northern Pipeline Construction Co. v. Marathon Pipe Line Co. | 4 |
| Atlas Roofing Co. v. Occupational Safety & Health Review Commission | 4 |
| Commodity Futures Trading Commission v. Schor | 3 |

## Review Rules

1. Use the blind queue for adjudication; do not look at `matched_frames` before assigning `review_label`.
2. Label the dominant legal frame, not every frame mentioned.
3. Use `mixed_comparative` when the excerpt is mainly comparing public and private rights without clearly adopting one frame.
4. Use `off_target_or_false_positive` for syllabus/navigation/citation-only chunks or excerpts that do not reason about Article III adjudication.
5. Only reviewed rows with clear dominant labels should feed the next probe.
