# SCOTUS Article III Conclusion Polarity

## Goal

Add a conclusion-polarity layer for Article III public/private-rights prompts. The proposition scorer counts doctrinal discussion; that is not enough because legally careful answers often mention both private rights and public rights. This screen tries to distinguish:

- `private_rights_objection_succeeds`: Article III objection has force, or a non-Article-III adjudicator lacks authority to enter final judgment;
- `public_rights_adjudication_permissible`: Article III objection fails, or Congress may assign initial adjudication to an agency/Article I tribunal;
- `mixed_or_unclear`.

This is a heuristic triage scorer, not a replacement for blind review.

## Artifacts

- Run: `sweep_v4/scotus_article3_conclusion_polarity_20260501_232256`
- Input: `sweep_v4/scotus_counterfactual_thoughts_20260501_231331/generations.jsonl`
- Raw report: `sweep_v4/scotus_article3_conclusion_polarity_20260501_232256/report.md`
- Rows: `sweep_v4/scotus_article3_conclusion_polarity_20260501_232256/polarity_rows.jsonl`
- Summary: `sweep_v4/scotus_article3_conclusion_polarity_20260501_232256/summary.jsonl`
- Script: `scripts/experiments/scotus/score_article3_conclusion_polarity.py`

## Result

| condition | n | private_score | public_score | net private-public | private label rate | public label rate | mixed rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `neutral` | 8 | `0.375` | `0.875` | `-0.500` | `0.250` | `0.625` | `0.125` |
| `private_rights` | 8 | `0.750` | `0.375` | `0.375` | `0.375` | `0.250` | `0.375` |
| `public_rights` | 8 | `0.375` | `1.125` | `-0.750` | `0.125` | `0.500` | `0.375` |

Read:

- The polarity scorer directionally separates the clean counterfactual scratchpads better than raw proposition counts: `private_rights` is positive on net private-minus-public, while `public_rights` is negative.
- It is still not robust enough to be a promotion gate. Mixed rates are high, and prompt facts dominate some rows.
- Some public-rights scratchpads still produce private-rights conclusions when the prompt facts are strongly private-rights-coded, e.g. tort assignment and bankruptcy counterclaim variants.

## Decision

Use this scorer as a triage layer and report diagnostic, not as final evidence.

For a future controller or actuator candidate:

- proposition-frame movement is insufficient by itself;
- require conclusion-polarity movement in the same direction;
- require manual/blind review for any candidate that appears to pass automatic polarity scoring;
- preserve visible-thinking review, because inserted scratchpads are not a no-mask mechanism.

## Next

If Article III remains the target branch, the next evaluator repair should produce a small blind review queue over:

- neutral/private/public counterfactual thought answers;
- any future controller outputs;
- labels for final holding direction, not only doctrinal vocabulary.

This should happen before another expensive full-model intervention sweep.
