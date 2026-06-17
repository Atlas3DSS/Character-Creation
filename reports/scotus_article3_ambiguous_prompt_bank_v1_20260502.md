# SCOTUS Article III Ambiguous Prompt Bank V1

## Purpose

Create a cleaner final-holding calibration surface after the long-answer counterfactual run showed that the original Article III prompts were often fact-pattern-determined. These prompts intentionally put public-rights and private-rights arguments in tension so final holdings can reveal frame movement instead of obvious doctrinal correctness.

## Output

- Prompt bank: `data/scotus/scotus_article3_ambiguous_poke_prompts_v1.jsonl`.
- Rows: `8`.
- Issue area: Judicial Power.
- Primary contrast: Article III private-rights objection succeeds versus public-rights/non-Article-III adjudication permissible.

## Design

- Every prompt contains both a plausible public-rights characterization and a plausible private-rights characterization.
- No prompt names a justice or asks for role-play.
- The target legal surface is final Article III holding direction, not vocabulary.
- The proposition scorer remains secondary; blind or triage holding review is required for any learned-result claim.

## Prompt Themes

| prompt_key | mixed surface |
| --- | --- |
| `A3_AMBIG_01_securities_penalty_restitution` | federal securities enforcement plus fraud/penalty/restitution suit-at-law features |
| `A3_AMBIG_02_bankruptcy_counterclaim_distribution` | state-law counterclaim that affects estate distribution but is not necessary for claim allowance |
| `A3_AMBIG_03_patent_review_parallel_litigation` | patent public franchise versus vested property during parallel litigation |
| `A3_AMBIG_04_customs_penalty_forfeiture` | customs administration plus penalty and forfeiture of property |
| `A3_AMBIG_05_industry_fund_contribution` | regulatory compensation fund plus private indemnity/contribution dispute |
| `A3_AMBIG_06_land_use_compensation` | federal land-use program plus tort-like compensation between private parties |
| `A3_AMBIG_07_benefits_fraud_recoupment` | public benefits administration plus restitution/penalty features |
| `A3_AMBIG_08_workplace_penalty_compensation` | public workplace enforcement plus employee compensation/private-liability component |

## Next Use

Run the long-answer server counterfactual harness with a `4096` answer-token cap, then build a holding-direction review queue and adjudicate final holdings. If inserted visible thoughts cannot move this balanced prompt bank, Article III is a weak candidate for actuator localization under the current framing.
