# SCOTUS Source Frame Seed v1

## Purpose

This seed set creates source-grounded legal-frame labels from real SCOTUS opinion chunks. Labels are strict-rule silver labels and require manual review before final steering claims.

## Outputs

- Labels: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_source_frame_labels_v1.jsonl`
- Review queue: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_source_frame_review_queue_v1.jsonl`
- Source chunks: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_chunk_inventory_v21.jsonl`

## Label Counts

| Frame | Issue family | Total | Train | Dev | Test |
| --- | --- | --- | --- | --- | --- |
| article3_public_rights | Judicial Power | 0 | 0 | 0 | 0 |
| article3_private_rights | Judicial Power | 0 | 0 | 0 | 0 |
| article3_article1_tribunal | Judicial Power | 6 | 4 | 1 | 1 |
| article3_case_or_controversy | Judicial Power | 48 | 34 | 7 | 7 |
| article3_final_judgment_separation | Judicial Power | 12 | 4 | 7 | 1 |
| fourth_search_incident_chimel | Criminal Procedure | 13 | 11 | 1 | 1 |
| fourth_plain_view_independent_source | Criminal Procedure | 18 | 10 | 1 | 7 |
| fourth_home_exigency | Criminal Procedure | 13 | 9 | 2 | 2 |
| fourth_technology_privacy | Criminal Procedure | 13 | 4 | 6 | 3 |

## Justice Coverage

| Justice | Records |
| --- | --- |
| Scalia | 67 |
| Souter | 26 |
| Ginsburg | 20 |
| Thomas | 10 |

## Sample Evidence Windows

| Frame | Split | Case | Justice | Posture | Evidence | Window |
| --- | --- | --- | --- | --- | --- | --- |
| article3_article1_tribunal | dev | Tennessee Student Assistance Corporation v. Hood | Thomas | dissent | \barticle i rather than an article iii court\b | zed that if the Framers would have found it an "impermissible affront to a State's dignity to be required to answer the complaints of private parties in federal courts," the Framers would have found it equally impermissi |
| article3_article1_tribunal | test | Things Remembered, Inc. v. Petrarca | Ginsburg | concurrence | \bbankruptcy judges?\b | [2] After the Court held inconsonant with Article III the Bankruptcy Act's broad grant of jurisdiction to bankruptcy judges, see Northern Pipeline Constr. Co. v. Marathon Pipe Line Co., 458 U. S. 50, 87 (1982), Congress  |
| article3_article1_tribunal | train | Federal Maritime Commission v. South Carolina State Ports Authority | Thomas | majority | \barticle i powers\b.{0,120}\bcourt-like administrative tribunals\b | t a private party before an impartial federal officer. [12] Moreover, it would be quite strange to prohibit Congress from exercising its Article I powers to abrogate state sovereign immunity in Article III judicial proce |
| article3_article1_tribunal | train | Plaut v. Spendthrift Farm, Inc. | Scalia | majority | \bnon[- ]article iii\b | Petitioners also rely on a miscellany of decisions upholding legislation that altered rights fixed by the final judgments of non-Article III courts, see, e. g., Sampeyreac v. United States, 7 Pet. 222, 238 (1833); Freebo |
| article3_article1_tribunal | train | Weiss v. United States | Souter | concurrence | \barticle i (?:tax court|military judge|court|courts|tribunal|tribunals)\b | The argument that military judges are principal officers is far from frivolous. It proceeds by analogizing military judges to Article III circuit and district judges, who are principal officers, [7] and to Article I Tax  |
| article3_article1_tribunal | train | Weiss v. United States | Souter | concurrence | \barticle i (?:tax court|military judge|court|courts|tribunal|tribunals)\b | The argument that military judges are principal officers, however, is not without response. Since Article I military judges are much more akin to Article I Tax Court judges than lower Article III judges, the analogy to T |
| article3_case_or_controversy | dev | Arizonans for Official English v. Arizona | Ginsburg | majority | \bcase or controversy\b, \bstanding\b, \bmoot(?:ness)?\b | hat AOE and Park had standing to place this case before an appellate tribunal. See id., at 366 (Stevens, J., dissenting) (Court properly assumed standing, even though that matter raised a serious question, in order to an |
| article3_case_or_controversy | dev | Gollust v. Mendell | Souter | majority | \bcase-or-controversy\b, \bstanding\b, \bjurisdiction\b | ake in the litigation for a further reason as well. For if a security holder were allowed to maintain a § 16(b) action after he had lost any financial interest in its outcome, there would be serious constitutional doubt  |
| article3_case_or_controversy | dev | Little Sisters of the Poor Saints Peter and Paul Home v. Pennsylvania | Ginsburg | dissent | \bstanding\b, \bjurisdiction\b | The Third Circuit also determined suasponte that the Little Sisters lacked appellate standing to intervene because a District Court in Colorado had permanently enjoined the contraceptive mandate as applied to plans in wh |
| article3_case_or_controversy | dev | Reno v. Bossier Parish School Board | Scalia | majority | \bcase or controversy\b, \bmoot(?:ness)?\b, \bjurisdiction\b | ill be available and the Board will be required by our "one-man-one-vote" precedents to have a new apportionment plan in place. Accordingly, appellee argues, the District Court's declaratory judgment with respect to the  |
| article3_case_or_controversy | dev | Ruhrgas Ag v. Marathon Oil Co. | Ginsburg | majority | \bcase or controversy\b, \bstanding\b, \bmoot(?:ness)?\b | diction of state-law claims on discretionary grounds without determining whether those claims fall within their pendent jurisdiction, see Moor v. County of Alameda, 411 U. S. 693, 715-716 (1973), or abstain under Younger |
| article3_case_or_controversy | dev | U.S. Bancorp Mortgage Co. v. Bonner Mall Partnership | Scalia | majority | \bcase or controversy\b, \bmoot(?:ness)?\b, \bjurisdiction\b | Appeals. Bonner opposed the motion. We set the vacatur question for briefing and argument. 511 U. S. 1002 -1003 (1994). The statute that supplies the power of vacatur provides: Of course, no statute could authorize a fed |
| article3_case_or_controversy | dev | United Food & Commercial Workers Union Local 751 v. Brown Group, Inc. | Souter | majority | \bstanding\b, \binjury in fact\b, \bjurisdiction\b | This brings us to the primary question m the case: whether the union has standing to bring this action on behalf of its members. 4 Article III of the Constitution limits the federal judicial power to “Cases” or “Controve |
| article3_case_or_controversy | test | General Motors Corp. v. Tracy | Souter | majority | \bstanding\b | Finally, the court dismissed GMC's equal protection claim as "submerged in its Commerce Clause argument." Id., at 31-32, 652 N. E. 2d, at 190. We granted GMC's petition for certiorari to address the question of standing  |
| article3_case_or_controversy | test | Lewis v. Casey | Souter | concurrence_in_part_dissent_in_part | \bcase or controversy\b, \bstanding\b | f standing or even class-action rules, that calls for the judgment to be reversed. Even if I were to reach the standing question, however, I would not adopt the standard the Court has established. In describing the injur |
| article3_case_or_controversy | test | National Credit Union Administration v. First National Bank & Trust Co. | Thomas | majority | \bstanding\b, \binjury in fact\b | Respondents claim a right to judicial review of the NCUA's chartering decision under § 10(a) of the APA, which provides: We have interpreted § 10(a) of the APA to impose a prudential standing requirement in addition to t |

## Use Rules

1. Treat these labels as `silver_high`, not adjudicated gold labels.
2. Do not train a final probe or claim a circuit until a human/blind review pass accepts or corrects the labels.
3. Use the frame-stratified `split` field for small frame probes; it is cluster-held-out within each frame.
4. The `global_split` field is also included for stricter cross-frame audits, but sparse frames may be unbalanced under that split.
5. Prefer frame contrasts with nonzero train/dev/test support and source diversity.
6. The sparse or zero rows are informative: they mark frames that should not be forced from this corpus.
