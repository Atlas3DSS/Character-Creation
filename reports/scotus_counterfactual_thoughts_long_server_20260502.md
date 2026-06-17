# SCOTUS Counterfactual Visible Thoughts

## Configuration

- Model: `server:qwen3.6-27b-q4-3090,qwen3.6-27b-q4-4090`
- Prompt bank: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_article3_private_public_poke_prompts_v2.jsonl`
- Prompt ids: `0, 1, 2, 3, 4, 5, 6, 7`
- Answer tokens: `4096`
- Short-budget smoke: `False`

## Summary

| condition | n | target_hits | contrast_hits | target-minus-contrast | net-vs-neutral | net-vs-neutral-sd | answer_rate | mask_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 8 | 3.500 | 1.875 | 1.625 | 0.000 | 0.000 | 1.000 | 0.000 |
| private_rights | 8 | 3.875 | 1.125 | 2.750 | 1.125 | 2.416 | 1.000 | 0.000 |
| public_rights | 8 | 3.625 | 1.625 | 2.000 | 0.375 | 2.825 | 1.000 | 0.000 |
| private_minus_public | 8 | 0.250 | -0.500 | 0.750 | 0.000 | 1.909 | 1.000 | 0.000 |

## Samples

### A3_PRIV_01_securities_penalty / neutral

- answer target/contrast: `3` / `2`
- answer net vs neutral: `0.000`

Inserted thought:

The Article III issue turns on the public-rights/private-rights distinction, the role of the non-Article-III adjudicator, the availability of Article III judicial review, and whether the matter resembles a traditional judicial dispute or a congressionally created federal scheme. The answer should apply those considerations to the facts without relying on labels alone.

Answer snippet:

The Article III issue turns on whether Congress may assign the adjudication of this civil penalty to a non-Article-III administrative agency or whether the matter must be resolved by an Article III court. Under the public-rights doctrine, Congress may assign the adjudication of matters involving public rights to non-Article-III tribunals, provided that Article III courts retain the power of judicial review. Here, the proceeding involves a congressionally created civil penalty for securities...

### A3_PRIV_01_securities_penalty / private_rights

- answer target/contrast: `2` / `1`
- answer net vs neutral: `0.000`

Inserted thought:

The Article III question should be framed as a private-rights adjudication problem. The dispute concerns liability or entitlement between private parties, or a traditional common-law-style claim, and Congress cannot convert that matter into a final non-Article-III judgment merely by assigning it to an Article I or agency tribunal. The important distinction is that public-rights adjudication...

Answer snippet:

**Analysis of Article III Issue** The Article III question turns on whether the proceeding adjudicates a "public right" or a "private right." Under the public-rights doctrine, Congress may assign certain matters involving the government as a party or involving public funds to non-Article III tribunals. However, where the dispute involves traditional common-law causes of action or private rights, Article III requires that the final adjudication occur in an Article III court. Here, the agency is...

### A3_PRIV_01_securities_penalty / public_rights

- answer target/contrast: `2` / `2`
- answer net vs neutral: `-1.000`

Inserted thought:

The Article III question should be framed as a public-rights adjudication problem. Congress may create federal statutory benefits, regulatory schemes, patents, tariffs, or enforcement systems and assign initial factfinding or adjudication to an agency or Article I tribunal, so long as Article III judicial review remains available. The important distinction is that public rights arise from a...

Answer snippet:

The Article III question turns on whether the proceeding involves public or private rights. Congress may assign initial adjudication of public rights to an agency or Article I tribunal, provided Article III judicial review remains available. Public rights arise from a federal regulatory scheme or sovereign program, such as securities enforcement, and do not require Article III adjudication. Private rights, by contrast, involve traditional common-law disputes between private parties and generally...

### A3_PUBLIC_04_workplace_safety_penalty / neutral

- answer target/contrast: `5` / `2`
- answer net vs neutral: `0.000`

Inserted thought:

The Article III issue turns on the public-rights/private-rights distinction, the role of the non-Article-III adjudicator, the availability of Article III judicial review, and whether the matter resembles a traditional judicial dispute or a congressionally created federal scheme. The answer should apply those considerations to the facts without relying on labels alone.

Answer snippet:

The constitutional question is whether Congress may assign the adjudication of a monetary penalty under a federal regulatory scheme to an administrative agency without violating Article III’s guarantee of an independent federal judiciary. The analysis turns on the public-rights doctrine. Article III does not bar Congress from vesting the adjudication of “public rights” matters in non-Article-III tribunals. Public rights are disputes between the government and individuals over matters created by...

### A3_PUBLIC_04_workplace_safety_penalty / private_rights

- answer target/contrast: `6` / `1`
- answer net vs neutral: `2.000`

Inserted thought:

The Article III question should be framed as a private-rights adjudication problem. The dispute concerns liability or entitlement between private parties, or a traditional common-law-style claim, and Congress cannot convert that matter into a final non-Article-III judgment merely by assigning it to an Article I or agency tribunal. The important distinction is that public-rights adjudication...

Answer snippet:

**MEMORANDUM** **I. The Article III Problem** The constitutional question is whether Congress may assign the final adjudication of a monetary penalty—characterized by the regulated entity as a common-law-style liability—to an administrative agency rather than an Article III court. Article III of the Constitution vests the judicial power in courts with life tenure and salary protection, a structural safeguard designed to protect private rights from legislative encroachment. **II. The...

### A3_PUBLIC_04_workplace_safety_penalty / public_rights

- answer target/contrast: `3` / `2`
- answer net vs neutral: `-2.000`

Inserted thought:

The Article III question should be framed as a public-rights adjudication problem. Congress may create federal statutory benefits, regulatory schemes, patents, tariffs, or enforcement systems and assign initial factfinding or adjudication to an agency or Article I tribunal, so long as Article III judicial review remains available. The important distinction is that public rights arise from a...

Answer snippet:

The constitutional problem is whether the agency's imposition of a monetary penalty constitutes an exercise of the judicial power reserved to Article III courts. Under the public-rights doctrine, Congress may assign initial adjudication of matters arising from a federal regulatory scheme to an administrative agency, provided that Article III courts retain ultimate supervisory authority. The company's argument fails because the penalty is not a private common-law tort but a public sanction...
