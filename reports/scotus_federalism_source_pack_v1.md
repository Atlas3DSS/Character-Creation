# SCOTUS Federalism Source Pack v1

## Purpose

Federalism was the next viable branch after Economic Activity and Civil Rights failed promotion gates. This pack tests a narrower same-doctrine contrast: anti-commandeering versus Supremacy Clause preemption.

## Outputs

- Raw pages: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/raw/scotus_federalism_source_pages_v1.json`
- Labels: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_federalism_source_frame_labels_v1.jsonl`
- Review queue: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_federalism_source_frame_review_queue_v1.jsonl`
- Source chunks scanned: `1026`

## Source Cases

| Case id | Case | Citation | Expected frame | Tokens | URL |
| --- | --- | --- | --- | --- | --- |
| new_york_1992 | New York v. United States | 505 U.S. 144 | federalism_anti_commandeering | 31067 | https://www.law.cornell.edu/supremecourt/text/505/144 |
| printz_1997 | Printz v. United States | 521 U.S. 898 | federalism_anti_commandeering | 38746 | https://www.law.cornell.edu/supremecourt/text/521/898 |
| reno_condon_2000 | Reno v. Condon | 528 U.S. 141 | federalism_anti_commandeering | 4471 | https://www.law.cornell.edu/supremecourt/text/528/141 |
| ferc_1982 | FERC v. Mississippi | 456 U.S. 742 | federalism_anti_commandeering | 27331 | https://www.law.cornell.edu/supremecourt/text/456/742 |
| murphy_2018 | Murphy v. National Collegiate Athletic Assn. | 584 U.S. 453 | federalism_anti_commandeering | 18703 | https://www.law.cornell.edu/supremecourt/text/16-476 |
| hines_1941 | Hines v. Davidowitz | 312 U.S. 52 | federalism_preemption | 11191 | https://www.law.cornell.edu/supremecourt/text/312/52 |
| rice_1947 | Rice v. Santa Fe Elevator Corp. | 331 U.S. 218 | federalism_preemption | 10956 | https://www.law.cornell.edu/supremecourt/text/331/218 |
| gade_1992 | Gade v. National Solid Wastes Management Assn. | 505 U.S. 88 | federalism_preemption | 16222 | https://www.law.cornell.edu/supremecourt/text/505/88 |
| cipollone_1992 | Cipollone v. Liggett Group, Inc. | 505 U.S. 504 | federalism_preemption | 24086 | https://www.law.cornell.edu/supremecourt/text/505/504 |
| crosby_2000 | Crosby v. National Foreign Trade Council | 530 U.S. 363 | federalism_preemption | 12900 | https://www.law.cornell.edu/supremecourt/text/530/363 |
| geier_2000 | Geier v. American Honda Motor Co. | 529 U.S. 861 | federalism_preemption | 22944 | https://www.law.cornell.edu/supremecourt/text/529/861 |
| wyeth_2009 | Wyeth v. Levine | 555 U.S. 555 | federalism_preemption | 1041 | https://www.law.cornell.edu/supremecourt/text/06-1249 |
| arizona_2012 | Arizona v. United States | 567 U.S. 387 | federalism_preemption | 30668 | https://www.law.cornell.edu/supremecourt/text/11-182 |

## Label Counts

| Frame | Total | Cases | Train | Dev | Test | Multi-frame conflicts |
| --- | --- | --- | --- | --- | --- | --- |
| federalism_anti_commandeering | 72 | 11 | 49 | 21 | 2 | 17 |
| federalism_preemption | 72 | 11 | 53 | 10 | 9 | 6 |

## Case/Frame Coverage

| Case id | Case | Frame | Records |
| --- | --- | --- | --- |
| arizona_2012 | Arizona v. United States | federalism_anti_commandeering | 1 |
| arizona_2012 | Arizona v. United States | federalism_preemption | 22 |
| cipollone_1992 | Cipollone v. Liggett Group, Inc. | federalism_preemption | 10 |
| crosby_2000 | Crosby v. National Foreign Trade Council | federalism_anti_commandeering | 1 |
| crosby_2000 | Crosby v. National Foreign Trade Council | federalism_preemption | 8 |
| ferc_1982 | FERC v. Mississippi | federalism_anti_commandeering | 1 |
| ferc_1982 | FERC v. Mississippi | federalism_preemption | 1 |
| gade_1992 | Gade v. National Solid Wastes Management Assn. | federalism_anti_commandeering | 1 |
| gade_1992 | Gade v. National Solid Wastes Management Assn. | federalism_preemption | 9 |
| geier_2000 | Geier v. American Honda Motor Co. | federalism_anti_commandeering | 1 |
| geier_2000 | Geier v. American Honda Motor Co. | federalism_preemption | 14 |
| hines_1941 | Hines v. Davidowitz | federalism_anti_commandeering | 1 |
| murphy_2018 | Murphy v. National Collegiate Athletic Assn. | federalism_anti_commandeering | 9 |
| murphy_2018 | Murphy v. National Collegiate Athletic Assn. | federalism_preemption | 1 |
| new_york_1992 | New York v. United States | federalism_anti_commandeering | 20 |
| new_york_1992 | New York v. United States | federalism_preemption | 1 |
| printz_1997 | Printz v. United States | federalism_anti_commandeering | 31 |
| printz_1997 | Printz v. United States | federalism_preemption | 1 |
| reno_condon_2000 | Reno v. Condon | federalism_anti_commandeering | 5 |
| rice_1947 | Rice v. Santa Fe Elevator Corp. | federalism_anti_commandeering | 1 |
| rice_1947 | Rice v. Santa Fe Elevator Corp. | federalism_preemption | 1 |
| wyeth_2009 | Wyeth v. Levine | federalism_preemption | 4 |

## Sample Evidence Windows

| Frame | Split | Case | Citation | Evidence | Conflict | Window |
| --- | --- | --- | --- | --- | --- | --- |
| federalism_anti_commandeering | dev | Hines v. Davidowitz | 312 U.S. 52 | \bstate officials?\b | no | 30 F.Supp. 470. One alien and one naturalized citizen joined in proceedings filed against certain state officials to enjoin enforcement of the Act. The answer of the defendants admitted the material allegations of the pe |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \bcommandeer(?:ing)?\b, \btake title\b, \benforce\b.{0,80}\bfederal\b | yes | ed, pre-empt entirely state regulation in this area, a review of this Court's decisions, see, e.g., Hodel v. Virginia Surface Mining & Reclamation Assn., Inc., 452 U.S. 264, 288, 101 S.Ct. 2352, 2366, 69 L.Ed.2d 1, and t |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \bcommandeer(?:ing)?\b, \bstate officials?\b, \bnew york\b | no | On the one hand, either forcing the transfer of waste from generators to the States or requiring the States to become liable for the generators' damages would "commandeer" States into the service of federal regulatory pu |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \bcommandeer(?:ing)?\b, \benforce\b.{0,80}\bfederal\b | no | mpelling them to enact and enforce a federal regulatory program." Hodel v. Virginia Surface Mining & Reclamation Assn., Inc., 452 U.S. 264, 288, 101 S.Ct. 2352, 2366, 69 L.Ed.2d 1 (1981). In Hodel, the Court upheld the S |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \bstate officials?\b, \bpolitical accountability\b, \bnew york\b | yes | By contrast, where the Federal Government compels States to regulate, the accountability of both state and federal officials is diminished. If the citizens of New York, for example, do not consider that making provision  |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \bcommandeer(?:ing)?\b, \btake title\b | no | ers. The same is true of the provision requiring the States to become liable for the generators' damages. Standing alone, this provision would be indistinguishable from an Act of Congress directing the States to assume t |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \btake title\b, \benforce\b.{0,80}\bfederal\b | no | Because an instruction to state governments to take title to waste, standing alone, would be beyond the authority of Congress, and because a direct order to regulate, standing alone, would also be beyond the authority of |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \btake title\b, \badminister\b.{0,80}\bfederal\b | no | posal methods, subject only to broad federal regulatory limits. This line of reasoning, however, only underscores the critical alternative a State lacks: A State may not decline to administer the federal program. No matt |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \bstate officials?\b, \bnew york\b | no | e sited and unsited States, a compromise to which New York was a willing participant and from which New York has reaped much benefit. Respondents then pose what appears at first to be a troubling question: How can a fede |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \btake title\b, \bnew york\b | no | Jackson, 390 U.S. 570, 585-586, 88 S.Ct. 1209, 1218, 20 L.Ed.2d 138 (1968). It is apparent in light of these principles that the take title provision may be severed without doing violence to the rest of the Act. The Act  |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \bstate officials?\b, \badminister\b.{0,80}\bfederal\b | yes | States are not mere political subdivisions of the United States. State governments are neither regional offices nor administrative agencies of the Federal Government. The positions occupied by state officials appear nowh |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \btake title\b, \bnew york\b | yes | Curiously absent from the Court's analysis is any effort to place the take title provision within the overall context of the legislation. As the discussion in Part I of this opinion suggests, the 1980 and 1985 statutes w |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \btake title\b, \bnew york\b | no | B Even were New York not to be estopped from challenging the take title provision's constitutionality, I am convinced that, seen as a term of an agreement entered into between the several States, this measure proves to b |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \bstate officials?\b, \btake title\b, \bnew york\b | no | Finally, to say, as the Court does, that the incursion on state sovereignty "cannot be ratified by the 'consent' of state officials," ante, at 182, is flatly wrong. In a case involving a congressional ratification statut |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \banti[- ]commandeering\b, \bcommandeer(?:ing)?\b, \benforce\b.{0,80}\bfederal\b | yes | Even were such a distinction to be logically sound, the Court's "anti-commandeering" principle cannot persuasively be read as springing from the two cases cited for the proposition, Hodel v. Virginia Surface Mining & Rec |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \btake title\b, \bferc\b | no | al laws of general applicability and those ostensibly directed solely at the activities of States, therefore, when the decisions from which it derives the rule not only made no such distinction, but validated federal sta |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \btake title\b, \bnew york\b, \bferc\b | no | Usery, 426 U.S., at 852 [96 S.Ct., at 2474]," FERC, supra, 456 U.S., at 765-766, 102 S.Ct., at 2144. On neither score does the take title provision raise constitutional problems. It certainly does not threaten New York's |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \bcommandeer(?:ing)?\b, \btake title\b | no | IV Though I disagree with the Court's conclusion that the take title provision is unconstitutional, I do not read its opinion to preclude Congress from adopting a similar measure through its powers under the Spending or  |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \bstate officials?\b, \bnew york\b | no | V The ultimate irony of the decision today is that in its formalistically rigid obeisance to "federalism," the Court gives Congress fewer incentives to defer to the wishes of state officials in achieving local solutions  |
| federalism_anti_commandeering | dev | New York v. United States | 505 U.S. 144 | \bcommandeer(?:ing)?\b, \btake title\b | no | With selective quotations from the era in which the Constitution was adopted, the majority attempts to bolster its holding that the take title provision is tantamount to federal "commandeering" of the States. In view of  |

## Use Rules

1. Run a cue-masked text-only gate before any BF16 activation capture.
2. If text alone solves anti-commandeering versus preemption, close the branch as leakage/text dominated.
3. If the text-only gate is not saturated, run a source-case-heldout cue-masked activation probe and compare against the text baseline.
4. Treat `Murphy` rows carefully because the opinion discusses both anti-commandeering and preemption.
