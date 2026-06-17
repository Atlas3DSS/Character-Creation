# SCOTUS Administrative Law Source Pack v1

## Purpose

Administrative Law is the remaining ranked source branch. This pack tests major-questions/clear-authorization reasoning against ordinary agency-deference/statutory-interpretation reasoning.

## Outputs

- Raw pages: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/raw/scotus_admin_source_pages_v1.json`
- Labels: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_admin_source_frame_labels_v1.jsonl`
- Review queue: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_admin_source_frame_review_queue_v1.jsonl`
- Source chunks scanned: `984`

## Source Cases

| Case id | Case | Citation | Expected frame | Tokens | URL |
| --- | --- | --- | --- | --- | --- |
| mci_1994 | MCI Telecommunications Corp. v. AT&T Co. | 512 U.S. 218 | admin_major_questions | 12244 | https://www.law.cornell.edu/supremecourt/text/512/218 |
| brown_williamson_2000 | FDA v. Brown & Williamson Tobacco Corp. | 529 U.S. 120 | admin_major_questions | 30889 | https://www.law.cornell.edu/supremecourt/text/529/120 |
| gonzales_oregon_2006 | Gonzales v. Oregon | 546 U.S. 243 | admin_major_questions | 25494 | https://www.law.cornell.edu/supremecourt/text/04-623 |
| utility_air_2014 | Utility Air Regulatory Group v. EPA | 573 U.S. 302 | admin_major_questions | 20517 | https://www.law.cornell.edu/supremecourt/text/12-1146 |
| king_burwell_2015 | King v. Burwell | 576 U.S. 473 | admin_major_questions | 19197 | https://www.law.cornell.edu/supremecourt/text/14-114 |
| west_virginia_epa_2022 | West Virginia v. EPA | 597 U.S. 697 | admin_major_questions | 38983 | https://www.law.cornell.edu/supremecourt/text/20-1530 |
| biden_nebraska_2023 | Biden v. Nebraska | 600 U.S. 477 | admin_major_questions | 32769 | https://www.law.cornell.edu/supremecourt/text/22-506 |
| skidmore_1944 | Skidmore v. Swift & Co. | 323 U.S. 134 | admin_deference_ordinary | 2306 | https://www.law.cornell.edu/supremecourt/text/323/134 |
| chevron_1984 | Chevron U.S.A. Inc. v. Natural Resources Defense Council, Inc. | 467 U.S. 837 | admin_deference_ordinary | 13608 | https://www.law.cornell.edu/supremecourt/text/467/837 |
| auer_1997 | Auer v. Robbins | 519 U.S. 452 | admin_deference_ordinary | 5520 | https://www.law.cornell.edu/supremecourt/text/519/452 |
| mead_2001 | United States v. Mead Corp. | 533 U.S. 218 | admin_deference_ordinary | 18736 | https://www.law.cornell.edu/supremecourt/text/533/218 |
| barnhart_2002 | Barnhart v. Walton | 535 U.S. 212 | admin_deference_ordinary | 7316 | https://www.law.cornell.edu/supremecourt/text/535/212 |
| city_arlington_2013 | City of Arlington v. FCC | 569 U.S. 290 | admin_deference_ordinary | 16619 | https://www.law.cornell.edu/supremecourt/text/11-1545 |
| kisor_2019 | Kisor v. Wilkie | 588 U.S. 558 | admin_deference_ordinary | 34077 | https://www.law.cornell.edu/supremecourt/text/18-15 |

## Label Counts

| Frame | Total | Cases | Train | Dev | Test | Multi-frame conflicts |
| --- | --- | --- | --- | --- | --- | --- |
| admin_major_questions | 66 | 6 | 40 | 2 | 24 | 9 |
| admin_deference_ordinary | 72 | 14 | 48 | 2 | 22 | 2 |

## Case/Frame Coverage

| Case id | Case | Frame | Records |
| --- | --- | --- | --- |
| auer_1997 | Auer v. Robbins | admin_deference_ordinary | 1 |
| barnhart_2002 | Barnhart v. Walton | admin_deference_ordinary | 3 |
| biden_nebraska_2023 | Biden v. Nebraska | admin_deference_ordinary | 1 |
| biden_nebraska_2023 | Biden v. Nebraska | admin_major_questions | 24 |
| brown_williamson_2000 | FDA v. Brown & Williamson Tobacco Corp. | admin_deference_ordinary | 1 |
| brown_williamson_2000 | FDA v. Brown & Williamson Tobacco Corp. | admin_major_questions | 2 |
| chevron_1984 | Chevron U.S.A. Inc. v. Natural Resources Defense Council, Inc. | admin_deference_ordinary | 1 |
| city_arlington_2013 | City of Arlington v. FCC | admin_deference_ordinary | 12 |
| gonzales_oregon_2006 | Gonzales v. Oregon | admin_deference_ordinary | 1 |
| gonzales_oregon_2006 | Gonzales v. Oregon | admin_major_questions | 6 |
| king_burwell_2015 | King v. Burwell | admin_deference_ordinary | 1 |
| king_burwell_2015 | King v. Burwell | admin_major_questions | 1 |
| kisor_2019 | Kisor v. Wilkie | admin_deference_ordinary | 26 |
| mci_1994 | MCI Telecommunications Corp. v. AT&T Co. | admin_deference_ordinary | 1 |
| mead_2001 | United States v. Mead Corp. | admin_deference_ordinary | 21 |
| skidmore_1944 | Skidmore v. Swift & Co. | admin_deference_ordinary | 1 |
| utility_air_2014 | Utility Air Regulatory Group v. EPA | admin_deference_ordinary | 1 |
| utility_air_2014 | Utility Air Regulatory Group v. EPA | admin_major_questions | 2 |
| west_virginia_epa_2022 | West Virginia v. EPA | admin_deference_ordinary | 1 |
| west_virginia_epa_2022 | West Virginia v. EPA | admin_major_questions | 31 |

## Sample Evidence Windows

| Frame | Split | Case | Citation | Evidence | Conflict | Window |
| --- | --- | --- | --- | --- | --- | --- |
| admin_deference_ordinary | dev | Biden v. Nebraska | 600 U.S. 477 | \bdefer\b | yes | ginia, experience shows that major questions cases “have arisen from all corners of the administrative state,” and administrative action resulting in the conferral of benefits is no exception to that rule. 597 U. S., at  |
| admin_deference_ordinary | dev | Skidmore v. Swift & Co. | 323 U.S. 134 | \bdeference\b | no | There is no statutory provision as to what, if any, deference courts should pay to the Administrator's conclusions. And, while we have given them notice, we have had no occasion to try to prescribe their influence. The r |
| admin_deference_ordinary | test | United States v. Mead Corp. | 533 U.S. 218 | \bchevron\b, \bdeference\b, \bagency interpretation\b | no | them as bound diaries subject to tariff. Mead filed suit in the Court of International Trade, which granted the Government summary judgment. In reversing, the Federal Circuit found that ruling letters should not be treat |
| admin_deference_ordinary | test | United States v. Mead Corp. | 533 U.S. 218 | \bchevron\b, \bskidmore\b, \bdeference\b | no | Such delegation may be shown in a variety of ways, as by an agency's power to engage in adjudication or notice-and-comment rulemaking, or by some other indication of comparable congressional intent. A Customs ruling lett |
| admin_deference_ordinary | test | United States v. Mead Corp. | 533 U.S. 218 | \bchevron\b, \bskidmore\b, \bdeference\b | no | e of law are being churned out at a rate of 10,000 a year at 46 offices is self-refuting. Nor do statutory amendments effective after this case arose reveal a new congressional objective of treating classification decisi |
| admin_deference_ordinary | test | United States v. Mead Corp. | 533 U.S. 218 | \bchevron\b, \bskidmore\b, \bdeference\b | no | Underlying this Court's position is a choice about the best way to deal with the great variety of ways in which the laws invest the Government's administrative arms with discretion, and with procedures for exercising it, |
| admin_deference_ordinary | test | United States v. Mead Corp. | 533 U.S. 218 | \bchevron\b, \bskidmore\b, \bdeference\b | no | The question is whether a tariff classification ruling by the United States Customs Service deserves judicial deference. The Federal Circuit rejected Customs's invocation of Chevron U.S. A. Inc. v. Natural Resources Defe |
| admin_deference_ordinary | test | United States v. Mead Corp. | 533 U.S. 218 | \bchevron\b, \bskidmore\b, \bdeference\b | no | t 1310. And it concluded that diaries "bound" in subheading 4810.10.20 presupposed "unbound" diaries, such that treating ring-fastened diaries as "bound" would leave the "unbound diary" an empty category. Id., at 1311. W |
| admin_deference_ordinary | test | United States v. Mead Corp. | 533 U.S. 218 | \bchevron\b, \bskidmore\b, \bdeference\b | no | In sum, classification rulings are best treated like "interpretations contained in policy statements, agency manuals, and enforcement guidelines." Christensen, 529 U.S., at 587. They are beyond the Chevron pale. C To agr |
| admin_deference_ordinary | test | United States v. Mead Corp. | 533 U.S. 218 | \bchevron\b, \bskidmore\b, \bdeference\b | no | Our respective choices are repeated today. Justice Scalia would pose the question of deference as an either-or choice. On his view that Chevron rendered Skidmore anachronistic, when courts owe any deference it is Chevron |
| admin_deference_ordinary | test | United States v. Mead Corp. | 533 U.S. 218 | \bchevron\b, \bskidmore\b, \bdeference\b | no | different statutes present different reasons for considering respect for the exercise of administrative authority or deference to it. Without being at odds with congressional intent much of the time, we believe that judi |
| admin_deference_ordinary | test | United States v. Mead Corp. | 533 U.S. 218 | \bchevron\b, \bskidmore\b, \bdeference\b | no | lly Haggar Apparel, 526 U.S., at 391. Compare Christensen v. Harris County, 529 U.S. 576, 587 (2000) ("Interpretations such as those in opinion letters-like interpretations contained in policy statements, agency manuals, |
| admin_deference_ordinary | test | United States v. Mead Corp. | 533 U.S. 218 | \bchevron\b, \bskidmore\b, \bdeference\b | no | ot exist the court was free to give the statute what it considered the best interpretation, henceforth the court must supposedly give the agency view some indeterminate amount of so-called Skidmore deference. We will be  |
| admin_deference_ordinary | test | United States v. Mead Corp. | 533 U.S. 218 | \bchevron\b, \bskidmore\b, \bdeference\b | no | Once it is determined that Chevron deference is not in order, the uncertainty is not at an end-and indeed is just beginning. Litigants cannot then assume that the statutory question is one for the courts to determine, ac |
| admin_deference_ordinary | test | United States v. Mead Corp. | 533 U.S. 218 | \bchevron\b, \bdeference\b, \breasonable interpretation\b | no | The Court's new doctrine is neither sound in principle nor sustainable in practice. * As to principle: The doctrine of Chevron-that all authoritative agency interpretations of statutes they are charged with administering |
| admin_deference_ordinary | test | United States v. Mead Corp. | 533 U.S. 218 | \bchevron\b, \bskidmore\b, \bdeference\b | no | Worst of all, the majority's approach will lead to the ossification of large portions of our statutory law. Where Chevron applies, statutory ambiguities remain ambiguities subject to the agency's ongoing clarification. T |
| admin_deference_ordinary | test | United States v. Mead Corp. | 533 U.S. 218 | \bchevron\b, \bskidmore\b, \bdeference\b | no | ngless its newly announced requirement that there be an affirmative congressional intent to have ambiguities resolved by the administering agency, and (2) ensures that no prior decision can possibly be cited which contra |
| admin_deference_ordinary | test | United States v. Mead Corp. | 533 U.S. 218 | \bchevron\b, \bdeference\b, \bpermissible construction\b | no | opinion made clear that we would have independently arrived at the same interpretation on our own, see 515 U.S., at 57-60. And although part of one sentence in Koray might be read to suggest that the Bureau's "Program St |
| admin_deference_ordinary | test | United States v. Mead Corp. | 533 U.S. 218 | \bchevron\b, \bskidmore\b, \bdeference\b | no | While the opinion did purport to accord the Equal Employment Opportunity Commission's informally promulgated interpretation only Skidmore deference, it did so because the Court thought itself bound by its pre-Chevron, EE |
| admin_deference_ordinary | test | United States v. Mead Corp. | 533 U.S. 218 | \bchevron\b, \bdeference\b, \bdefer\b | no | Nothing in the statute at issue here displays an intent to modify the background presumption on which Chevron deference is based. The Court points, ante, at 13, n. 16, to 28 U.S.C. 2640(a), which provides that, in review |

## Use Rules

1. Run a cue-masked text-only gate before any BF16 activation capture.
2. If text alone solves major-questions versus deference, close the branch as leakage/text dominated.
3. If text-only is not saturated, run dominance review before activation probing.
