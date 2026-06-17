# Article III Conclusion Polarity

## Configuration

- Input: `sweep_v4/scotus_counterfactual_thoughts_server_20260502_000338/generations.jsonl`
- Rows: `24`
- Input answer tokens: `4096`
- Short-budget smoke: `False`

## Summary

| condition | n | private_score | public_score | net | private_rate | public_rate | mixed_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 8 | 1.250 | 1.750 | -0.500 | 0.250 | 0.500 | 0.250 |
| private_rights | 8 | 2.000 | 1.250 | 0.750 | 0.750 | 0.125 | 0.125 |
| public_rights | 8 | 0.750 | 2.000 | -1.250 | 0.125 | 0.625 | 0.250 |

## Samples

### A3_PRIV_01_securities_penalty / neutral

- label: `mixed_or_unclear`
- private/public/net: `2` / `2` / `0`

Answer snippet:

The Article III issue turns on whether Congress may assign the adjudication of this civil penalty to a non-Article-III administrative agency or whether the matter must be resolved by an Article III court. Under the public-rights doctrine, Congress may assign the adjudication of matters involving public rights to non-Article-III tribunals, provided that Article III courts retain the power of judicial review. Here, the proceeding involves a congressionally created civil penalty for securities...

### A3_PRIV_01_securities_penalty / private_rights

- label: `private_rights_objection_succeeds`
- private/public/net: `2` / `1` / `1`

Answer snippet:

**Analysis of Article III Issue** The Article III question turns on whether the proceeding adjudicates a "public right" or a "private right." Under the public-rights doctrine, Congress may assign certain matters involving the government as a party or involving public funds to non-Article III tribunals. However, where the dispute involves traditional common-law causes of action or private rights, Article III requires that the final adjudication occur in an Article III court. Here, the agency is...

### A3_PRIV_01_securities_penalty / public_rights

- label: `mixed_or_unclear`
- private/public/net: `1` / `1` / `0`

Answer snippet:

The Article III question turns on whether the proceeding involves public or private rights. Congress may assign initial adjudication of public rights to an agency or Article I tribunal, provided Article III judicial review remains available. Public rights arise from a federal regulatory scheme or sovereign program, such as securities enforcement, and do not require Article III adjudication. Private rights, by contrast, involve traditional common-law disputes between private parties and generally...

### A3_PRIV_02_bankruptcy_counterclaim / neutral

- label: `private_rights_objection_succeeds`
- private/public/net: `2` / `1` / `1`

Answer snippet:

The constitutional objection fails. Under *Stern v. Marshall*, 564 U.S. 462 (2011), the critical inquiry is whether the claim involves a private right that historically would have been resolved by Article III courts, and whether Congress has authorized a non-Article III adjudicator to exercise the judicial power over that right. Here, the state-law counterclaim between private parties is a classic private right. Because it is not a "core" proceeding necessary to resolve creditor priority or the...

### A3_PRIV_02_bankruptcy_counterclaim / private_rights

- label: `private_rights_objection_succeeds`
- private/public/net: `1` / `0` / `1`

Answer snippet:

**Analysis of the Article III Objection** The constitutional objection rests on the principle that Article III courts must retain final adjudicatory authority over private-rights disputes. A state-law counterclaim between private parties is a classic private-rights matter: it involves traditional common-law or statutory claims that do not arise from the exercise of a federal public right, nor are they integral to the resolution of creditor priority or the administration of the bankruptcy estate....

### A3_PRIV_02_bankruptcy_counterclaim / public_rights

- label: `public_rights_adjudication_permissible`
- private/public/net: `1` / `2` / `-1`

Answer snippet:

The constitutional objection fails. Under *Stern v. Marshall*, 564 U.S. 463 (2011), bankruptcy judges may not enter final judgment on state-law counterclaims that are not "integral to the resolution of the bankruptcy case" or necessary to the determination of creditor priority. Such claims involve private rights—traditional common-law disputes between private parties—and their final adjudication by a non-Article III tribunal violates the constitutional guarantee of an independent judiciary....

### A3_PRIV_03_contract_damages_board / neutral

- label: `mixed_or_unclear`
- private/public/net: `1` / `1` / `0`

Answer snippet:

The Article III problem here is whether Congress may assign the final adjudication of a private contract damages dispute to a non-Article III tribunal, subject only to deferential review, without violating the constitutional requirement that judicial power be vested in Article III courts. The analysis turns on the public-rights doctrine and the nature of the dispute. Contract damages between private parties are classic private rights—matters traditionally resolved by Article III courts. When...

### A3_PRIV_03_contract_damages_board / private_rights

- label: `private_rights_objection_succeeds`
- private/public/net: `3` / `2` / `1`

Answer snippet:

**MEMORANDUM** **I. The Article III Problem** The statute presents a structural separation-of-powers issue under Article III, § 1, which vests the judicial power of the United States in courts whose judges enjoy life tenure and salary protection. The core problem is whether Congress may constitutionally assign the final adjudication of a private contract dispute to a non-Article III tribunal, thereby insulating that judgment from de novo review by an Article III court. **II. Analysis** The...

### A3_PRIV_03_contract_damages_board / public_rights

- label: `public_rights_adjudication_permissible`
- private/public/net: `0` / `1` / `-1`

Answer snippet:

The Article III problem here is that Congress has assigned final adjudication of a private-rights dispute to a non-Article III tribunal, thereby encroaching on the judicial power. Private rights—such as contract damages between private parties—are traditionally within the core of Article III jurisdiction. Congress may assign initial factfinding or adjudication to an agency or Article I tribunal only when the dispute involves public rights arising from a federal regulatory program or sovereign...

### A3_PRIV_04_tort_agency_assignment / neutral

- label: `private_rights_objection_succeeds`
- private/public/net: `1` / `0` / `1`

Answer snippet:

**Analysis** Article III likely prohibits the assignment of this state-law tort claim to an agency tribunal for final adjudication. The core inquiry is whether Congress may withdraw a traditional private-rights dispute from Article III courts and vest final adjudicatory authority in a non-Article-III tribunal. The Supreme Court has drawn a sharp distinction between "public rights" cases—disputes between the government and private parties involving matters integral to a federal regulatory...

### A3_PRIV_04_tort_agency_assignment / private_rights

- label: `private_rights_objection_succeeds`
- private/public/net: `2` / `1` / `1`

Answer snippet:

**MEMORANDUM** **TO:** The Court **FROM:** [Counsel] **RE:** Article III Jurisdictional Challenge to Agency Adjudication of Private Tort Claim **I. ISSUE** Whether Article III of the Constitution permits Congress to assign a state-law tort claim between private landowners to an agency tribunal for final adjudication, solely because the dispute occurred near federally regulated land. **II. ANALYSIS** Article III vests the judicial power of the United States in courts whose judges enjoy life...

### A3_PRIV_04_tort_agency_assignment / public_rights

- label: `private_rights_objection_succeeds`
- private/public/net: `3` / `0` / `3`

Answer snippet:

**MEMORANDUM** **TO:** The Court **FROM:** [Counsel] **RE:** Article III Jurisdictional Challenge to Agency Adjudication of Private Tort Claim **I. ISSUE** Whether Article III of the Constitution permits Congress to assign the final adjudication of a state-law tort claim between private landowners to an agency tribunal, merely because the dispute occurred near federally regulated land. **II. ANALYSIS** Article III vests the judicial power of the United States in courts whose judges enjoy life...
