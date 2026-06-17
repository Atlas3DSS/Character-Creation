# SCOTUS Minimal-Pair Template Leakage Audit

Created: `2026-05-01T11:41:11-07:00`

## Verdict

Some activation evidence survives template-pair holdout. Candidate features still need decoder-column steering and random/same-layer controls before promotion.

## Exact Template Reuse

| Metric | Value |
| --- | --- |
| Rows | 48 |
| Unique assistant completions | 6 |
| Template-pair groups | 3 |
| Original split exact-template baseline BA | 1.000 |

| Template pair group | N | Label counts | Split counts | Authority snippet | Limits snippet |
| --- | --- | --- | --- | --- | --- |
| 6276c3d5b59a__3028eecc6a61 | 16 | {0: 8, 1: 8} | {'dev': 4, 'test': 4, 'train': 8} | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in chara | Holding: The federal law cannot be sustained under the Commerce Clause. Reasoning: The activity is not part of a broader market regulation o |
| c9ba436b5fde__5cb61be09d71 | 16 | {0: 8, 1: 8} | {'dev': 2, 'test': 4, 'train': 10} | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of inter | Holding: The statute exceeds Congress's enumerated power. Reasoning: The statute lacks a concrete jurisdictional hook tying the particular c |
| ebed2cb9fde3__737a63265d32 | 16 | {0: 8, 1: 8} | {'dev': 4, 'test': 2, 'train': 10} | Holding: Congress has authority to regulate this class of activity. Reasoning: The statute targets economic conduct connected to a national  | Holding: Congress lacks authority on these facts. Reasoning: The regulated conduct is local and non-economic. Lopez and Morrison mark the li |

## Leave-One-Template-Pair-Out Diagnostics

| Diagnostic | Mean BA | Min BA | Max BA | Mean accuracy |
| --- | --- | --- | --- | --- |
| prompt_tfidf | 0.500 | 0.500 | 0.500 | 0.500 |
| assistant_text_tfidf | 0.167 | 0.000 | 0.500 | 0.167 |
| residual_assistant_all__L04 | 0.729 | 0.500 | 1.000 | 0.729 |
| residual_assistant_all__L08 | 0.938 | 0.812 | 1.000 | 0.938 |
| residual_assistant_all__L12 | 0.979 | 0.938 | 1.000 | 0.979 |
| residual_assistant_all__L16 | 1.000 | 1.000 | 1.000 | 1.000 |
| residual_assistant_all__L20 | 1.000 | 1.000 | 1.000 | 1.000 |
| sae_best_l0_100_assistant_all_L8 | 0.750 | 0.500 | 1.000 | 0.750 |

## Reading Notes

- This audit ignores the original train/dev/test split and instead holds out one exact authority/limits answer-template pair at a time.
- The original split reused every exact assistant template across train/dev/test, so high original split accuracy can be template recognition.
- A candidate circuit should survive template-pair holdout before decoder-column steering is treated as meaningful.
