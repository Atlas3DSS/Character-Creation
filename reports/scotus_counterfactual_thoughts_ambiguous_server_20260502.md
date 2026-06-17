# SCOTUS Counterfactual Visible Thoughts

## Configuration

- Model: `server:qwen3.6-27b-q4-3090,qwen3.6-27b-q4-4090`
- Prompt bank: `data/scotus/scotus_article3_ambiguous_poke_prompts_v1.jsonl`
- Prompt ids: `0, 1, 2, 3, 4, 5, 6, 7`
- Answer tokens: `4096`
- Short-budget smoke: `False`

## Summary

| condition | n | target_hits | contrast_hits | target-minus-contrast | net-vs-neutral | net-vs-neutral-sd | answer_rate | mask_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 8 | 1.875 | 1.375 | 0.500 | 0.000 | 0.000 | 1.000 | 0.000 |
| private_rights | 8 | 4.125 | 1.500 | 2.625 | 2.125 | 1.458 | 1.000 | 0.000 |
| public_rights | 8 | 2.750 | 2.000 | 0.750 | 0.250 | 0.886 | 1.000 | 0.000 |
| private_minus_public | 8 | 1.375 | -0.500 | 1.875 | 0.000 | 1.808 | 1.000 | 0.000 |

## Samples

### A3_AMBIG_01_securities_penalty_restitution / neutral

- answer target/contrast: `4` / `2`
- answer net vs neutral: `0.000`

Inserted thought:

The Article III issue turns on the public-rights/private-rights distinction, the role of the non-Article-III adjudicator, the availability of Article III judicial review, and whether the matter resembles a traditional judicial dispute or a congressionally created federal scheme. The answer should apply those considerations to the facts without relying on labels alone.

Answer snippet:

**Analysis** Article III permits final agency adjudication in this enforcement action. The constitutional inquiry turns on whether the matter involves "public rights" or "private rights," the nature of the adjudicator's role, and the availability of Article III judicial review. **1. Public Rights Doctrine** The case involves a "public rights" matter. The agency enforces a federal regulatory scheme designed to protect the integrity of the securities markets and the investing public, rather than...

### A3_AMBIG_01_securities_penalty_restitution / private_rights

- answer target/contrast: `6` / `2`
- answer net vs neutral: `2.000`

Inserted thought:

The Article III question should be framed as a private-rights adjudication problem. The dispute concerns liability or entitlement between private parties, or a traditional common-law-style claim, and Congress cannot convert that matter into a final non-Article-III judgment merely by assigning it to an Article I or agency tribunal. The important distinction is that public-rights adjudication...

Answer snippet:

**MEMORANDUM** **TO:** The Court **FROM:** The Clerk **RE:** Article III Adjudication of Agency Enforcement Action **I. ISSUE** Whether Article III of the Constitution permits a federal administrative agency to issue a final adjudication imposing a large civil penalty, disgorgement, and an industry bar in an enforcement action that the respondent characterizes as a traditional suit at law. **II. ANALYSIS** The constitutional permissibility of final agency adjudication turns on the distinction...

### A3_AMBIG_01_securities_penalty_restitution / public_rights

- answer target/contrast: `4` / `2`
- answer net vs neutral: `0.000`

Inserted thought:

The Article III question should be framed as a public-rights adjudication problem. Congress may create federal statutory benefits, regulatory schemes, patents, tariffs, or enforcement systems and assign initial factfinding or adjudication to an agency or Article I tribunal, so long as Article III judicial review remains available. The important distinction is that public rights arise from a...

Answer snippet:

**MEMORANDUM** **TO:** The Court **FROM:** The Clerk **RE:** Article III Adjudication of SEC Enforcement Action **I. ISSUE** Whether Article III of the Constitution permits final agency adjudication of a securities enforcement action seeking civil penalties, disgorgement, and an industry bar, or whether such adjudication must be reserved for an Article III court. **II. ANALYSIS** The Article III question turns on the distinction between "public rights" and "private rights." The Supreme Court has...

### A3_AMBIG_08_workplace_penalty_compensation / neutral

- answer target/contrast: `3` / `2`
- answer net vs neutral: `0.000`

Inserted thought:

The Article III issue turns on the public-rights/private-rights distinction, the role of the non-Article-III adjudicator, the availability of Article III judicial review, and whether the matter resembles a traditional judicial dispute or a congressionally created federal scheme. The answer should apply those considerations to the facts without relying on labels alone.

Answer snippet:

The Article III issue turns on whether the agency’s order, which includes both a civil penalty and backpay-like compensation, constitutes a public-rights matter that may be adjudicated by a non-Article-III tribunal, or a private-rights dispute requiring Article III adjudication. Under the public-rights doctrine, Congress may assign adjudication of matters involving the government as a party or enforcing a federal regulatory scheme to administrative agencies, even when the relief includes...

### A3_AMBIG_08_workplace_penalty_compensation / private_rights

- answer target/contrast: `4` / `2`
- answer net vs neutral: `1.000`

Inserted thought:

The Article III question should be framed as a private-rights adjudication problem. The dispute concerns liability or entitlement between private parties, or a traditional common-law-style claim, and Congress cannot convert that matter into a final non-Article-III judgment merely by assigning it to an Article I or agency tribunal. The important distinction is that public-rights adjudication...

Answer snippet:

**MEMORANDUM OPINION** **I. Legal Framework** Article III of the Constitution vests the judicial power in courts with life tenure and salary protection. The Supreme Court has long recognized a "public rights" exception permitting Congress to assign certain matters to non-Article III tribunals. The critical distinction lies in the nature of the rights at stake: * **Public Rights:** Disputes between the government and individuals where the government is a party with a direct interest, or matters...

### A3_AMBIG_08_workplace_penalty_compensation / public_rights

- answer target/contrast: `4` / `3`
- answer net vs neutral: `0.000`

Inserted thought:

The Article III question should be framed as a public-rights adjudication problem. Congress may create federal statutory benefits, regulatory schemes, patents, tariffs, or enforcement systems and assign initial factfinding or adjudication to an agency or Article I tribunal, so long as Article III judicial review remains available. The important distinction is that public rights arise from a...

Answer snippet:

The Article III question turns on whether the agency's order adjudicates a "public right" or a "private right." Under the public-rights doctrine, Congress may assign initial adjudication of matters arising from a federal regulatory scheme to an agency or Article I tribunal, provided Article III judicial review remains available. Here, the agency's order enforces a federal workplace-safety statute, imposing a civil penalty to the government and compensation to an employee as part of a...
