# SCOTUS Section Repair Audit

Phase 3.5 repair pass over cached CourtListener combined opinion records.

## Section Counts

| Justice | Sections | Kept raw chunks | Kept raw tokens |
| --- | --- | --- | --- |
| Ginsburg | 251 | 3956 | 896727 |
| Scalia | 360 | 4728 | 1118787 |
| Souter | 270 | 4321 | 1021756 |
| Thomas | 318 | 4006 | 917761 |

## Section Posture Distribution

| Justice | Section Posture | Count |
| --- | --- | --- |
| Ginsburg | majority | 133 |
| Ginsburg | dissent | 70 |
| Ginsburg | concurrence | 17 |
| Ginsburg | concurrence_in_judgment | 15 |
| Ginsburg | concurrence_in_part_dissent_in_part | 10 |
| Ginsburg | concurrence_in_part | 5 |
| Ginsburg | judgment | 1 |
| Scalia | majority | 202 |
| Scalia | dissent | 76 |
| Scalia | concurrence_in_judgment | 31 |
| Scalia | concurrence | 18 |
| Scalia | concurrence_in_part | 18 |
| Scalia | concurrence_in_part_dissent_in_part | 9 |
| Scalia | judgment | 6 |
| Souter | majority | 137 |
| Souter | dissent | 73 |
| Souter | concurrence | 25 |
| Souter | concurrence_in_part | 11 |
| Souter | concurrence_in_part_dissent_in_part | 10 |
| Souter | concurrence_in_judgment | 9 |
| Souter | judgment | 5 |
| Thomas | majority | 148 |
| Thomas | dissent | 99 |
| Thomas | concurrence | 30 |
| Thomas | concurrence_in_judgment | 19 |
| Thomas | concurrence_in_part_dissent_in_part | 10 |
| Thomas | judgment | 6 |
| Thomas | concurrence_in_part | 5 |
| Thomas | unknown | 1 |

## Exclusion Flags

| Flag | Excluded Records |
| --- | --- |
| is_low_reasoning_density | 5115 |
| is_header_like | 4041 |
| is_counsel_like | 2260 |
| is_order_fragment | 476 |
| has_third_person_target_author_reference | 323 |
| is_join_line_like | 211 |
| has_non_target_author_heading | 88 |
| has_reasoning_marker | 42 |
| is_citation_dominated | 12 |
| has_target_author_heading | 8 |

## Manual Inspection Sample

| Justice | Posture | Case | Position | Snippet |
| --- | --- | --- | --- | --- |
| Scalia | majority | O'CONNOR v. United States | early | The petitioners, United States citizen employees of the Panama Canal Commission and their spouses, seek refunds of income taxes collected on salaries paid by the Commission between 1979 and 1981. We granted certiorari to resolve conflicting appellate interpret |
| Scalia | majority | O'CONNOR v. United States | early | The petitioners contend that § 2 of this Article constitutes an express exemption of their Commission salaries from both Panamanian and United States taxation. See 26 U. S. C. § 894 (a) (“Income of any kind, to the extent required by any treaty obligation of t |
| Scalia | majority | O'CONNOR v. United States | early | We agree with the Federal Circuit. The first section of Article XV, which confers upon the Commission and its contractors an exemption “from payment in the Republic of Panama of all taxes” (emphasis added), establishes the context for the discussion of tax exe |
| Scalia | majority | O'CONNOR v. United States | middle | There is some purely textual evidence, albeit subtle, of the understanding that Article XV applies only to Panamanian taxes: In conferring an exemption from property taxes, §3 displays an assumption that only personal property within the Republic of Panama is  |
| Scalia | majority | O'CONNOR v. United States | middle | More persuasive than the textual evidence, and in our view overwhelmingly convincing, is the contextual case for limiting Article XV to Panamanian taxes. Unless one posits the ellipsis of failing to repeat, in each section, § l’s limitation to taxes “in the Re |
| Souter | judgment | James B. Beam Distilling Co. v. Georgia | early | The question presented is whether our ruling in Bacchus Imports, Ltd. v. Dias, 468 U. S. 263 (1984), should apply retroactively to claims arising on facts antedating that decision. We hold that application of the rule in that case requires its application retr |
| Souter | judgment | James B. Beam Distilling Co. v. Georgia | early | In Bacchus' wake, petitioner James B. Beam Distilling Co., a Delaware corporation and Kentucky bourbon manufacturer, claimed Georgia's law likewise inconsistent with the Commerce Clause, and sought a refund of $2.4 million, representing not only the differenti |
| Souter | judgment | James B. Beam Distilling Co. v. Georgia | early | The Supreme Court of Georgia affirmed the trial court in both respects. The court held the pre-1985 version of the statute to have violated the Commerce Clause as, in its words, an act of "simple economic protectionism." See 259 Ga. 363, 364 , 382 S. E. 2d 95, |
| Souter | judgment | James B. Beam Distilling Co. v. Georgia | early | E. 2d 518, 520 (1982)). Beam sought a writ of certiorari from the Court on the retroactivity question. [1] We granted the petition, 496 U. S. 924 (1990), and now reverse. In the ordinary case, no question of retroactivity arises. Courts are as a general matter |
| Souter | judgment | James B. Beam Distilling Co. v. Georgia | early | It is only when the law changes in some respect that an assertion of nonretroactivity may be entertained, the paradigm case arising when a court expressly overrules a precedent upon which the contest would otherwise be decided differently and by which the part |
| Thomas | dissent | United States Ex Rel. Internal Revenue Service v. McDermott | early | I agree with the Court that under 26 U. S. C. § 6323 (a) we generally look to the filing of notice of the federal tax lien to determine the federal lien's priority as against a competing state-law judgment lien. I cannot agree, however, that a federal tax lien |
| Thomas | dissent | United States Ex Rel. Internal Revenue Service v. McDermott | early | Applying the governing "first in time" rule, the Court recognizes—as it must—that if the Bank's interest in the property was "perfected in the sense that there [was] nothing more to be done to have a choate lien" before September 9, 1987 (the date the federal  |
| Thomas | dissent | United States Ex Rel. Internal Revenue Service v. McDermott | early | We have not (before today) prescribed any rigid criteria for "establish[ing]" the property subject to a competing lien; we have required only that the lien " become certain as to . . . the property subject thereto." New Britain, supra, at 86 (emphasis added).  |
| Thomas | dissent | United States Ex Rel. Internal Revenue Service v. McDermott | early | Although the choateness of a state-law lien under § 6323(a) is a federal question, that question is answered in part by reference to state law, and we therefore give due weight to the State's "`classification of [its] lien as specific and perfected.' " Pioneer |
| Thomas | dissent | United States Ex Rel. Internal Revenue Service v. McDermott | early | Thus, the Bank's lien had become certain as to the property subject thereto, whether then existing or thereafter acquired, and all competing creditors were on notice that there was "nothing more to be done" by the Bank "to have a choate lien" on any real prope |
| Ginsburg | concurrence_in_judgment | Director, Office of Workers' Compensation Programs v. Newport News Shipbuilding & Dry Dock Co. | early | The Court holds that the Director of the Office of Workers' Compensation Programs of the United States Department of Labor (OWCP) lacks standing under § 21(c) of the Longshore and Harbor Workers' Compensation Act (LHWCA or Act), 44 Stat. 1424 , as amended, 33  |
| Ginsburg | concurrence_in_judgment | Director, Office of Workers' Compensation Programs v. Newport News Shipbuilding & Dry Dock Co. | early | Significantly, however, the Court observes that our precedent "certainly establish[es] that Congress could have conferred standing upon the [OWCP] Director without infringing Article III of the Constitution." Ante, at 133 (emphasis in original). [1] While I do |
| Ginsburg | concurrence_in_judgment | Director, Office of Workers' Compensation Programs v. Newport News Shipbuilding & Dry Dock Co. | early | Before the 1972 amendments to the LHWCA, the OWCP Director's predecessors as administrators of the Act, officials called OWCP deputy commissioners, adjudicated LHWCA claims in the first instance. 33 U. S. C. §§ 919 , 923 (1970 ed.); see Kalaris v. Donovan, 697 |
| Ginsburg | concurrence_in_judgment | Director, Office of Workers' Compensation Programs v. Newport News Shipbuilding & Dry Dock Co. | early | The 1972 LHWCA amendments shifted the deputy commissioners' adjudicatory authority to Department of Labor administrative law judges (ALJ's). Although district directors—as deputy commissioners are now called [2] —are empowered to investigate LHWCA claims and a |
| Ginsburg | concurrence_in_judgment | Director, Office of Workers' Compensation Programs v. Newport News Shipbuilding & Dry Dock Co. | middle | The Court holds that the LHWCA, as amended in 1972, does not entitle the Director to appeal Benefits Review Board decisions to the courts of appeals. Congress surely decided to transfer adjudicative functions from the deputy commissioners to ALJ's, and from th |

## Outputs

- `data/scotus/scotus_section_inventory_v21.jsonl`: 1199 target-authored section records
- `data/scotus/scotus_chunk_inventory_v21.jsonl`: 34022 kept chunk records
- `data/scotus/processed/scotus_excluded_chunk_inventory_v21.jsonl`: 7205 excluded block/chunk records with flags
