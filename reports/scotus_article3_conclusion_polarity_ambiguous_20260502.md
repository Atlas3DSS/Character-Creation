# Article III Conclusion Polarity

## Configuration

- Input: `sweep_v4/scotus_counterfactual_thoughts_server_20260502_001228/generations.jsonl`
- Rows: `24`
- Input answer tokens: `4096`
- Short-budget smoke: `False`

## Summary

| condition | n | private_score | public_score | net | private_rate | public_rate | mixed_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 8 | 0.625 | 1.375 | -0.750 | 0.000 | 0.750 | 0.250 |
| private_rights | 8 | 2.375 | 1.500 | 0.875 | 0.625 | 0.375 | 0.000 |
| public_rights | 8 | 0.500 | 2.000 | -1.500 | 0.000 | 0.875 | 0.125 |

## Samples

### A3_AMBIG_01_securities_penalty_restitution / neutral

- label: `mixed_or_unclear`
- private/public/net: `1` / `1` / `0`

Answer snippet:

**Analysis** Article III permits final agency adjudication in this enforcement action. The constitutional inquiry turns on whether the matter involves "public rights" or "private rights," the nature of the adjudicator's role, and the availability of Article III judicial review. **1. Public Rights Doctrine** The case involves a "public rights" matter. The agency enforces a federal regulatory scheme designed to protect the integrity of the securities markets and the investing public, rather than...

### A3_AMBIG_01_securities_penalty_restitution / private_rights

- label: `public_rights_adjudication_permissible`
- private/public/net: `2` / `3` / `-1`

Answer snippet:

**MEMORANDUM** **TO:** The Court **FROM:** The Clerk **RE:** Article III Adjudication of Agency Enforcement Action **I. ISSUE** Whether Article III of the Constitution permits a federal administrative agency to issue a final adjudication imposing a large civil penalty, disgorgement, and an industry bar in an enforcement action that the respondent characterizes as a traditional suit at law. **II. ANALYSIS** The constitutional permissibility of final agency adjudication turns on the distinction...

### A3_AMBIG_01_securities_penalty_restitution / public_rights

- label: `mixed_or_unclear`
- private/public/net: `2` / `2` / `0`

Answer snippet:

**MEMORANDUM** **TO:** The Court **FROM:** The Clerk **RE:** Article III Adjudication of SEC Enforcement Action **I. ISSUE** Whether Article III of the Constitution permits final agency adjudication of a securities enforcement action seeking civil penalties, disgorgement, and an industry bar, or whether such adjudication must be reserved for an Article III court. **II. ANALYSIS** The Article III question turns on the distinction between "public rights" and "private rights." The Supreme Court has...

### A3_AMBIG_02_bankruptcy_counterclaim_distribution / neutral

- label: `public_rights_adjudication_permissible`
- private/public/net: `0` / `1` / `-1`

Answer snippet:

The constitutional problem is whether a bankruptcy judge, as a non-Article III officer, may enter final judgment on a state-law contract counterclaim that is not necessary to resolve a creditor’s proof of claim. Under *Stern v. Marshall*, the public-rights doctrine does not authorize final adjudication of private rights merely because they arise in a federal statutory scheme. A state-law contract counterclaim by the estate against a creditor is a classic private right: it involves traditional...

### A3_AMBIG_02_bankruptcy_counterclaim_distribution / private_rights

- label: `private_rights_objection_succeeds`
- private/public/net: `2` / `1` / `1`

Answer snippet:

The constitutional problem is whether Congress may assign the final adjudication of a private-rights claim to a non-Article III bankruptcy judge. The counterclaim is a traditional state-law contract dispute between private parties. It does not involve a public right, nor is it integral to the allowance or disallowance of a creditor's claim against the estate. Because the claim is a private right, Article III requires that the final judgment be rendered by a judge with life tenure and salary...

### A3_AMBIG_02_bankruptcy_counterclaim_distribution / public_rights

- label: `public_rights_adjudication_permissible`
- private/public/net: `0` / `2` / `-2`

Answer snippet:

The constitutional problem is whether Congress may assign the adjudication of a state-law contract counterclaim to a non-Article III bankruptcy judge without violating the separation of powers. Under *Stern v. Marshall*, the public-rights doctrine permits Congress to assign certain matters to Article I tribunals, but only when the claim arises from a federal regulatory scheme or sovereign program. A state-law contract counterclaim is a private right, not a public right, because it involves...

### A3_AMBIG_03_patent_review_parallel_litigation / neutral

- label: `public_rights_adjudication_permissible`
- private/public/net: `0` / `1` / `-1`

Answer snippet:

The Article III issue turns on whether the Patent Trial and Appeal Board's cancellation of patent claims in an inter partes proceeding violates the constitutional requirement that cases and controversies be adjudicated by Article III courts. The resolution depends on the public-rights doctrine, the nature of the adjudicator, and the availability of meaningful judicial review. Patents are not vested private property in the absolute sense; they are public franchises created by Congress under...

### A3_AMBIG_03_patent_review_parallel_litigation / private_rights

- label: `private_rights_objection_succeeds`
- private/public/net: `4` / `1` / `3`

Answer snippet:

**MEMORANDUM OPINION** **I. Article III Issue** The dispositive question is whether the Patent Trial and Appeal Board's ("PTAB") cancellation of issued patent claims in an inter partes review ("IPR") violates Article III of the Constitution by vesting the final adjudication of private rights in an Article I tribunal. **II. Analysis** The patent owner contends that patents are vested private property, and their cancellation constitutes a final adjudication of private rights that must be reserved...

### A3_AMBIG_03_patent_review_parallel_litigation / public_rights

- label: `public_rights_adjudication_permissible`
- private/public/net: `0` / `2` / `-2`

Answer snippet:

The Article III question is whether the Patent Trial and Appeal Board's cancellation of patent claims in an inter partes review violates the Constitution's separation of powers by vesting adjudicatory authority in an executive agency rather than an Article III court. Patents are not vested private property in the common-law sense; they are public franchises created by Congress under Article I, Section 8, to promote the progress of science and useful arts. Because patent rights derive entirely...

### A3_AMBIG_04_customs_penalty_forfeiture / neutral

- label: `public_rights_adjudication_permissible`
- private/public/net: `1` / `2` / `-1`

Answer snippet:

The importer’s Article III objection fails. The classification of imported goods, imposition of statutory penalties, and forfeiture of property in the customs context fall squarely within the public-rights doctrine. Congress may assign such matters to non-Article-III tribunals because they involve rights created by federal statute and disputes between the government and private parties over the enforcement of federal regulatory schemes, rather than traditional common-law causes of action. The...

### A3_AMBIG_04_customs_penalty_forfeiture / private_rights

- label: `public_rights_adjudication_permissible`
- private/public/net: `1` / `3` / `-2`

Answer snippet:

**Analysis of Article III Objection** The importer’s Article III objection fails. The dispute falls squarely within the “public rights” exception to Article III’s judicial-power requirement. **1. Public vs. Private Rights** Article III limits Congress’s ability to assign final adjudication of “private rights” disputes to non-Article-III tribunals. However, Congress may assign final adjudication of “public rights” matters—disputes between the government and individuals over rights created by...

### A3_AMBIG_04_customs_penalty_forfeiture / public_rights

- label: `public_rights_adjudication_permissible`
- private/public/net: `1` / `4` / `-3`

Answer snippet:

**Analysis of Article III Objection** The importer’s Article III challenge fails. The customs tribunal’s imposition of statutory penalties and forfeiture constitutes the adjudication of **public rights**, not private rights, and therefore does not require Article III adjudication. 1. **Public-Rights Doctrine**: The Supreme Court has long held that Congress may assign the initial adjudication of public rights to Article I tribunals or administrative agencies. *Northern Pipeline Constr. Co. v....
