# SCOTUS Section Repair Audit

Phase 3.5 repair pass over cached CourtListener combined opinion records.

## Section Counts

| Justice | Sections | Kept raw chunks | Kept raw tokens |
| --- | --- | --- | --- |
| Ginsburg | 256 | 4098 | 929804 |
| Scalia | 360 | 4919 | 1165286 |
| Souter | 267 | 4289 | 1011262 |
| Thomas | 314 | 4043 | 928514 |

## Section Posture Distribution

| Justice | Section Posture | Count |
| --- | --- | --- |
| Ginsburg | majority | 193 |
| Ginsburg | dissent | 28 |
| Ginsburg | concurrence | 17 |
| Ginsburg | concurrence_in_judgment | 10 |
| Ginsburg | concurrence_in_part | 4 |
| Ginsburg | concurrence_in_part_dissent_in_part | 4 |
| Scalia | majority | 280 |
| Scalia | dissent | 27 |
| Scalia | concurrence_in_judgment | 20 |
| Scalia | concurrence | 18 |
| Scalia | concurrence_in_part | 11 |
| Scalia | concurrence_in_part_dissent_in_part | 4 |
| Souter | majority | 203 |
| Souter | dissent | 25 |
| Souter | concurrence | 25 |
| Souter | concurrence_in_part | 6 |
| Souter | concurrence_in_part_dissent_in_part | 6 |
| Souter | concurrence_in_judgment | 2 |
| Thomas | majority | 209 |
| Thomas | dissent | 53 |
| Thomas | concurrence | 30 |
| Thomas | concurrence_in_judgment | 10 |
| Thomas | concurrence_in_part_dissent_in_part | 6 |
| Thomas | concurrence_in_part | 5 |
| Thomas | unknown | 1 |

## Exclusion Flags

| Flag | Excluded Records |
| --- | --- |
| is_low_reasoning_density | 5147 |
| is_header_like | 4015 |
| is_counsel_like | 2312 |
| is_order_fragment | 469 |
| is_join_line_like | 230 |
| has_non_target_author_heading | 124 |
| has_reasoning_marker | 36 |
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
| Souter | dissent | Bray v. Alexandria Women's Health Clinic | early | This case turns on the meaning of two clauses of 42 U. S. C.  1985(3) which render certain conspiracies civilly actionable. The first clause (the deprivation clause) covers conspiracies the second (the prevention clause), conspiracies For liability in either  |
| Souter | dissent | Bray v. Alexandria Women's Health Clinic | early | The meaning of the prevention clause is not thus settled, however, and starting in Part IV I will give my reasons for reading it without any importation of these extratextual conditions from the deprivation clause. First, however, a word is in order to show th |
| Souter | dissent | Bray v. Alexandria Women's Health Clinic | early | Because this Court has not previously faced a prevention clause claim, the difficult question that arises on this first occasion is whether to import the two conditions imposed on the deprivation clause as limitations on the scope of the prevention clause as w |
| Souter | dissent | Bray v. Alexandria Women's Health Clinic | early | This is so because the two conditions at issue almost certainly run counter to the intention of Congress, and whatever may have been the strength of this Court's reasons for construing the deprivation clause to include them, those reasons have no application t |
| Souter | dissent | Bray v. Alexandria Women's Health Clinic | early | The amalgam of concepts reflected in 42 U. S. C.  1985(3) witness the statute's evolution, as  2 of the Civil Rights Act of 1871, from a bill that would have criminalized conspiracies "to do any act in violation of the rights, privileges, or immunities of an |
| Thomas | dissent | United States Ex Rel. Internal Revenue Service v. McDermott | early | I agree with the Court that under 26 U. S. C. § 6323 (a) we generally look to the filing of notice of the federal tax lien to determine the federal lien's priority as against a competing state-law judgment lien. I cannot agree, however, that a federal tax lien |
| Thomas | dissent | United States Ex Rel. Internal Revenue Service v. McDermott | early | Applying the governing "first in time" rule, the Court recognizesas it mustthat if the Bank's interest in the property was "perfected in the sense that there [was] nothing more to be done to have a choate lien" before September 9, 1987 (the date the federal  |
| Thomas | dissent | United States Ex Rel. Internal Revenue Service v. McDermott | early | We have not (before today) prescribed any rigid criteria for "establish[ing]" the property subject to a competing lien; we have required only that the lien " become certain as to . . . the property subject thereto." New Britain, supra, at 86 (emphasis added).  |
| Thomas | dissent | United States Ex Rel. Internal Revenue Service v. McDermott | early | Although the choateness of a state-law lien under § 6323(a) is a federal question, that question is answered in part by reference to state law, and we therefore give due weight to the State's "`classification of [its] lien as specific and perfected.' " Pioneer |
| Thomas | dissent | United States Ex Rel. Internal Revenue Service v. McDermott | early | Thus, the Bank's lien had become certain as to the property subject thereto, whether then existing or thereafter acquired, and all competing creditors were on notice that there was "nothing more to be done" by the Bank "to have a choate lien" on any real prope |
| Ginsburg | concurrence_in_judgment | Director, Office of Workers' Compensation Programs v. Newport News Shipbuilding & Dry Dock Co. | early | The Court holds that the Director of the Office of Workers' Compensation Programs of the United States Department of Labor (OWCP) lacks standing under § 21(c) of the Longshore and Harbor Workers' Compensation Act (LHWCA or Act), 44 Stat. 1424 , as amended, 33  |
| Ginsburg | concurrence_in_judgment | Director, Office of Workers' Compensation Programs v. Newport News Shipbuilding & Dry Dock Co. | early | Significantly, however, the Court observes that our precedent "certainly establish[es] that Congress could have conferred standing upon the [OWCP] Director without infringing Article III of the Constitution." Ante, at 133 (emphasis in original). [1] While I do |
| Ginsburg | concurrence_in_judgment | Director, Office of Workers' Compensation Programs v. Newport News Shipbuilding & Dry Dock Co. | early | Before the 1972 amendments to the LHWCA, the OWCP Director's predecessors as administrators of the Act, officials called OWCP deputy commissioners, adjudicated LHWCA claims in the first instance. 33 U. S. C. §§ 919 , 923 (1970 ed.); see Kalaris v. Donovan, 697 |
| Ginsburg | concurrence_in_judgment | Director, Office of Workers' Compensation Programs v. Newport News Shipbuilding & Dry Dock Co. | early | The 1972 LHWCA amendments shifted the deputy commissioners' adjudicatory authority to Department of Labor administrative law judges (ALJ's). Although district directorsas deputy commissioners are now called [2] are empowered to investigate LHWCA claims and a |
| Ginsburg | concurrence_in_judgment | Director, Office of Workers' Compensation Programs v. Newport News Shipbuilding & Dry Dock Co. | middle | The Court holds that the LHWCA, as amended in 1972, does not entitle the Director to appeal Benefits Review Board decisions to the courts of appeals. Congress surely decided to transfer adjudicative functions from the deputy commissioners to ALJ's, and from th |

## Outputs

- `data/scotus/scotus_section_inventory.jsonl`: 1197 target-authored section records
- `data/scotus/scotus_chunk_inventory_v2.jsonl`: 34698 kept chunk records
- `data/scotus/processed/scotus_excluded_chunk_inventory_v2.jsonl`: 6967 excluded block/chunk records with flags
