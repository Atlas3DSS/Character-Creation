# SCOTUS Article III Holding Review Queue

## Purpose

Build a blind answer-only queue for final Article III holding direction. The inserted scratchpad is hidden so reviewers label the legal conclusion reached by the answer, not the counterfactual condition.

## Inputs

- Generations: `sweep_v4/scotus_counterfactual_thoughts_20260501_231331/generations.jsonl`
- Automatic polarity rows: `sweep_v4/scotus_article3_conclusion_polarity_20260501_232256/polarity_rows.jsonl`

## Outputs

- Blind queue: `data/scotus/scotus_article3_holding_review_blind_20260501.jsonl`
- Key file: `data/scotus/scotus_article3_holding_review_key_20260501.jsonl`
- Manifest: `reports/scotus_article3_holding_review_queue_20260501.json`
- Review rows: `24`

## Automatic Polarity Distribution

| condition | n | auto private | auto public | auto mixed |
| --- | --- | --- | --- | --- |
| neutral | 8 | 2 | 5 | 1 |
| private_rights | 8 | 3 | 2 | 3 |
| public_rights | 8 | 1 | 4 | 3 |

## Review Instructions

For each answer, label the final holding direction:

- `article3_objection_succeeds_private_rights`: the answer concludes that Article III requires an Article III court or that the non-Article-III adjudicator cannot enter final judgment.
- `article3_objection_fails_public_rights_permissible`: the answer concludes that the agency/Article I/non-Article-III process is constitutionally permissible, usually under public-rights or adequate-review reasoning.
- `mixed_or_distinction_only`: the answer explains the distinction but does not clearly resolve the prompt.
- `unclear_or_incoherent`: the answer is truncated, confused, or nonresponsive.

The automatic polarity labels are in the key file only and should not be used during blind review.
