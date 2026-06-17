# SCOTUS Article III Holding Review Queue

## Purpose

Build a blind answer-only queue for final Article III holding direction. The inserted scratchpad is hidden so reviewers label the legal conclusion reached by the answer, not the counterfactual condition.

## Inputs

- Generations: `sweep_v4/scotus_counterfactual_thoughts_server_20260502_000338/generations.jsonl`
- Automatic polarity rows: `sweep_v4/scotus_article3_conclusion_polarity_20260502_000626/polarity_rows.jsonl`
- Generation answer tokens: `4096`
- Short-budget smoke: `False`

## Outputs

- Blind queue: `data/scotus/scotus_article3_holding_review_blind_long_20260502.jsonl`
- Key file: `data/scotus/scotus_article3_holding_review_key_long_20260502.jsonl`
- Manifest: `reports/scotus_article3_holding_review_queue_long_20260502.json`
- Review rows: `24`

## Automatic Polarity Distribution

| condition | n | auto private | auto public | auto mixed |
| --- | --- | --- | --- | --- |
| neutral | 8 | 2 | 4 | 2 |
| private_rights | 8 | 6 | 1 | 1 |
| public_rights | 8 | 1 | 5 | 2 |

## Review Instructions

For each answer, label the final holding direction:

- `article3_objection_succeeds_private_rights`: the answer concludes that Article III requires an Article III court or that the non-Article-III adjudicator cannot enter final judgment.
- `article3_objection_fails_public_rights_permissible`: the answer concludes that the agency/Article I/non-Article-III process is constitutionally permissible, usually under public-rights or adequate-review reasoning.
- `mixed_or_distinction_only`: the answer explains the distinction but does not clearly resolve the prompt.
- `unclear_or_incoherent`: the answer is truncated, confused, or nonresponsive.

The automatic polarity labels are in the key file only and should not be used during blind review.
