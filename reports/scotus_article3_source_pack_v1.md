# SCOTUS Article III Source Pack v1

## Purpose

This expands the source-grounded frame corpus beyond target-justice chunks for Article III public/private-rights doctrine. It is a silver-label source pack for manual review and leakage diagnostics, not final circuit evidence.

## Outputs

- Raw pages: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/raw/scotus_article3_source_pages_v1.json`
- Labels: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_article3_source_frame_labels_v1.jsonl`
- Review queue: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_article3_source_frame_review_queue_v1.jsonl`
- Source chunks scanned: `1034`

## Source Cases

| Case id | Case | Citation | Term | Tokens | URL |
| --- | --- | --- | --- | --- | --- |
| murrays_lessee_1856 | Murray's Lessee v. Hoboken Land & Improvement Co. | 59 U.S. 272 | 1856 | 8021 | https://www.law.cornell.edu/supremecourt/text/59/272 |
| crowell_v_benson_1932 | Crowell v. Benson | 285 U.S. 22 | 1931 | 32814 | https://www.law.cornell.edu/supremecourt/text/285/22 |
| atlas_roofing_1977 | Atlas Roofing Co. v. Occupational Safety & Health Review Commission | 430 U.S. 442 | 1976 | 8954 | https://www.law.cornell.edu/supremecourt/text/430/442 |
| northern_pipeline_1982 | Northern Pipeline Construction Co. v. Marathon Pipe Line Co. | 458 U.S. 50 | 1981 | 32661 | https://www.law.cornell.edu/supremecourt/text/458/50 |
| thomas_union_carbide_1985 | Thomas v. Union Carbide Agricultural Products Co. | 473 U.S. 568 | 1984 | 18560 | https://www.law.cornell.edu/supremecourt/text/473/568 |
| cftc_v_schor_1986 | Commodity Futures Trading Commission v. Schor | 478 U.S. 833 | 1985 | 16335 | https://www.law.cornell.edu/supremecourt/text/478/833 |
| granfinanciera_1989 | Granfinanciera, S.A. v. Nordberg | 492 U.S. 33 | 1988 | 31979 | https://www.law.cornell.edu/supremecourt/text/492/33 |
| stern_v_marshall_2011 | Stern v. Marshall | 564 U.S. 462 | 2010 | 15905 | https://www.law.cornell.edu/supct/html/10-179.ZO.html |
| wellness_v_sharif_2015 | Wellness International Network, Ltd. v. Sharif | 575 U.S. 665 | 2014 | 26060 | https://www.law.cornell.edu/supremecourt/text/13-935 |
| oil_states_2018 | Oil States Energy Services, LLC v. Greene's Energy Group, LLC | 584 U.S. 325 | 2017 | 14431 | https://www.law.cornell.edu/supremecourt/text/16-712 |
| axon_v_ftc_2023 | Axon Enterprise, Inc. v. Federal Trade Commission | 598 U.S. 175 | 2022 | 19524 | https://www.law.cornell.edu/supremecourt/text/21-86 |
| sec_v_jarkesy_2024 | Securities and Exchange Commission v. Jarkesy | 603 U.S. ___ | 2023 | 42977 | https://www.law.cornell.edu/supremecourt/text/22-859 |

## Label Counts

| Frame | Total | Train | Dev | Test | Public/private conflicts |
| --- | --- | --- | --- | --- | --- |
| article3_public_rights | 72 | 38 | 27 | 7 | 16 |
| article3_private_rights | 39 | 28 | 3 | 8 | 28 |
| article3_article1_tribunal | 72 | 45 | 17 | 10 | 4 |
| article3_case_or_controversy | 11 | 6 | 4 | 1 | 2 |
| article3_final_judgment_separation | 30 | 27 | 2 | 1 | 0 |

## Case/Frame Coverage

| Case id | Case | Frame | Records |
| --- | --- | --- | --- |
| atlas_roofing_1977 | Atlas Roofing Co. v. Occupational Safety & Health Review Commission | article3_case_or_controversy | 1 |
| atlas_roofing_1977 | Atlas Roofing Co. v. Occupational Safety & Health Review Commission | article3_public_rights | 4 |
| axon_v_ftc_2023 | Axon Enterprise, Inc. v. Federal Trade Commission | article3_article1_tribunal | 1 |
| axon_v_ftc_2023 | Axon Enterprise, Inc. v. Federal Trade Commission | article3_case_or_controversy | 1 |
| axon_v_ftc_2023 | Axon Enterprise, Inc. v. Federal Trade Commission | article3_private_rights | 10 |
| axon_v_ftc_2023 | Axon Enterprise, Inc. v. Federal Trade Commission | article3_public_rights | 1 |
| cftc_v_schor_1986 | Commodity Futures Trading Commission v. Schor | article3_article1_tribunal | 10 |
| cftc_v_schor_1986 | Commodity Futures Trading Commission v. Schor | article3_final_judgment_separation | 1 |
| cftc_v_schor_1986 | Commodity Futures Trading Commission v. Schor | article3_private_rights | 2 |
| cftc_v_schor_1986 | Commodity Futures Trading Commission v. Schor | article3_public_rights | 2 |
| crowell_v_benson_1932 | Crowell v. Benson | article3_article1_tribunal | 1 |
| granfinanciera_1989 | Granfinanciera, S.A. v. Nordberg | article3_article1_tribunal | 9 |
| granfinanciera_1989 | Granfinanciera, S.A. v. Nordberg | article3_final_judgment_separation | 2 |
| granfinanciera_1989 | Granfinanciera, S.A. v. Nordberg | article3_private_rights | 6 |
| granfinanciera_1989 | Granfinanciera, S.A. v. Nordberg | article3_public_rights | 14 |
| northern_pipeline_1982 | Northern Pipeline Construction Co. v. Marathon Pipe Line Co. | article3_article1_tribunal | 17 |
| northern_pipeline_1982 | Northern Pipeline Construction Co. v. Marathon Pipe Line Co. | article3_final_judgment_separation | 1 |
| northern_pipeline_1982 | Northern Pipeline Construction Co. v. Marathon Pipe Line Co. | article3_public_rights | 3 |
| oil_states_2018 | Oil States Energy Services, LLC v. Greene's Energy Group, LLC | article3_article1_tribunal | 2 |
| oil_states_2018 | Oil States Energy Services, LLC v. Greene's Energy Group, LLC | article3_private_rights | 2 |
| oil_states_2018 | Oil States Energy Services, LLC v. Greene's Energy Group, LLC | article3_public_rights | 6 |
| sec_v_jarkesy_2024 | Securities and Exchange Commission v. Jarkesy | article3_article1_tribunal | 7 |
| sec_v_jarkesy_2024 | Securities and Exchange Commission v. Jarkesy | article3_private_rights | 7 |
| sec_v_jarkesy_2024 | Securities and Exchange Commission v. Jarkesy | article3_public_rights | 24 |
| stern_v_marshall_2011 | Stern v. Marshall | article3_article1_tribunal | 7 |
| stern_v_marshall_2011 | Stern v. Marshall | article3_final_judgment_separation | 8 |
| stern_v_marshall_2011 | Stern v. Marshall | article3_private_rights | 1 |
| stern_v_marshall_2011 | Stern v. Marshall | article3_public_rights | 9 |
| thomas_union_carbide_1985 | Thomas v. Union Carbide Agricultural Products Co. | article3_article1_tribunal | 2 |
| thomas_union_carbide_1985 | Thomas v. Union Carbide Agricultural Products Co. | article3_case_or_controversy | 5 |
| thomas_union_carbide_1985 | Thomas v. Union Carbide Agricultural Products Co. | article3_final_judgment_separation | 2 |
| thomas_union_carbide_1985 | Thomas v. Union Carbide Agricultural Products Co. | article3_private_rights | 3 |
| thomas_union_carbide_1985 | Thomas v. Union Carbide Agricultural Products Co. | article3_public_rights | 4 |
| wellness_v_sharif_2015 | Wellness International Network, Ltd. v. Sharif | article3_article1_tribunal | 16 |
| wellness_v_sharif_2015 | Wellness International Network, Ltd. v. Sharif | article3_case_or_controversy | 4 |
| wellness_v_sharif_2015 | Wellness International Network, Ltd. v. Sharif | article3_final_judgment_separation | 16 |
| wellness_v_sharif_2015 | Wellness International Network, Ltd. v. Sharif | article3_private_rights | 8 |
| wellness_v_sharif_2015 | Wellness International Network, Ltd. v. Sharif | article3_public_rights | 5 |

## Sample Evidence Windows

| Frame | Split | Case | Citation | Evidence | Conflict | Window |
| --- | --- | --- | --- | --- | --- | --- |
| article3_article1_tribunal | dev | Commodity Futures Trading Commission v. Schor | 478 U.S. 833 | \bnon[- ]article iii\b | no | onwaivable protections Article III affords separation of powers principles. Examination of the congressional scheme in light of a number of factors, including the extent to which the "essential attributes of judicial pow |
| article3_article1_tribunal | dev | Commodity Futures Trading Commission v. Schor | 478 U.S. 833 | \bnon[- ]article iii\b | no | rt of Appeals, sua sponte, raised the question whether CFTC could constitutionally adjudicate Conti's counterclaims in light of Northern Pipeline Construction Co. v. Marathon Pipe Line Co., 458 U.S. 50, 102 S.Ct. 2858, 7 |
| article3_article1_tribunal | dev | Commodity Futures Trading Commission v. Schor | 478 U.S. 833 | \bnon[- ]article iii\b, \bbankruptcy judges?\b | no | tion by the federal judiciary of matters within the judicial power of the United States intimated that this guarantee serves to protect primarily personal, rather than structural, interests. See, e.g., id., at 90, 102 S. |
| article3_article1_tribunal | dev | Commodity Futures Trading Commission v. Schor | 478 U.S. 833 | \bnon[- ]article iii\b | no | by jury in criminal case); Fed. Rule of Civ.Proc. 38(d) (waiver of right to trial by jury in civil cases). Indeed, the relevance of concepts of waiver to Article III challenges is demonstrated by our decision in Northern |
| article3_article1_tribunal | dev | Commodity Futures Trading Commission v. Schor | 478 U.S. 833 | \bnon[- ]article iii\b | no | In determining the extent to which a given congressional decision to authorize the adjudication of Article III business in a non-Article III tribunal impermissibly threatens the institutional integrity of the Judicial Br |
| article3_article1_tribunal | dev | Commodity Futures Trading Commission v. Schor | 478 U.S. 833 | \bnon[- ]article iii\b | no | or resort to arbitration without impermissible incursions on the separation of powers, Congress may make available a quasi-judicial mechanism through which willing parties may, at their option, elect to resolve their dif |
| article3_article1_tribunal | dev | Commodity Futures Trading Commission v. Schor | 478 U.S. 833 | \bnon[- ]article iii\b | no | ongress has invaded the prerogatives of state governments. At the outset, we note that our prior precedents in this area have dealt only with separation of powers concerns, and have not intimated that principles of feder |
| article3_article1_tribunal | dev | Commodity Futures Trading Commission v. Schor | 478 U.S. 833 | \bbankruptcy judges?\b | no | by the Constitution or by historical consensus." Northern Pipeline, supra, 458 U.S., at 70, 102 S.Ct., at 2871 (opinion of BRENNAN, J.). Here, however, there is no equally forceful reason to extend further these exceptio |
| article3_article1_tribunal | dev | Commodity Futures Trading Commission v. Schor | 478 U.S. 833 | \blegislative courts?\b | no | are almost entirely prophylactic, and thus often seem remote and not worth the cost in any single case. Thus, while this balancing creates the illusion of objectivity and ineluctability, in fact the result was foreordain |
| article3_article1_tribunal | dev | Commodity Futures Trading Commission v. Schor | 478 U.S. 833 | \bnon[- ]article iii\b | no | More importantly, the Court, in emphasizing that this litigation will permit solely a narrow class of state-law claims to be decided by a non-Article III court, ignores the fact that it establishes a broad principle. The |
| article3_article1_tribunal | dev | Securities and Exchange Commission v. Jarkesy | 603 U.S. ___ | \bnon[- ]article iii\b, \bbankruptcy judges?\b | yes | icle III courts.” Northern Pipeline Constr. Co. v. Marathon Pipe Line Co., 458 U. S. 50, 69, n. 23 (plurality opinion). Pp. 13–18. (2) In Granfinanciera, this Court previously considered whether the Seventh Amendment gua |
| article3_article1_tribunal | dev | Securities and Exchange Commission v. Jarkesy | 603 U.S. ___ | \bnon[- ]article iii\b, \bbankruptcy judges?\b | no | ranches of the Federal Government could confer the Government’s ‘judicial Power’ on entities outside Article III.” Stern, 564 U. S., at 484. This is not the first time we have considered whether the Seventh Amendment gua |
| article3_article1_tribunal | dev | Securities and Exchange Commission v. Jarkesy | 603 U.S. ___ | \blegislative courts?\b, \bnon[- ]article iii\b | no | Fallon, Of Legislative Courts, Administrative Agencies, and Article III, 101 Harv. L. Rev. 915 (1988) (no citation to Atlas Roofing); J. Harrison, Public Rights, Private Privileges, and Article III, 54 Ga. L. Rev. 143 (2 |
| article3_article1_tribunal | dev | Securities and Exchange Commission v. Jarkesy | 603 U.S. ___ | \bnon[- ]article iii\b | no | ial to the detriment of all other departments of the Government, disregards many previous adjudications of this court, and ignores practices often manifested and hitherto deemed to be free from any possible constitutiona |
| article3_article1_tribunal | dev | Securities and Exchange Commission v. Jarkesy | 603 U.S. ___ | \bnon[- ]article iii\b | no | Conspicuously absent from the majority’s discussion are, for example, cases in which this Court held that Congress could assign a private federally created action that was “closely integrated into a public regulatory sch |
| article3_article1_tribunal | dev | Securities and Exchange Commission v. Jarkesy | 603 U.S. ___ | \bnon[- ]article iii\b, \bbankruptcy judges?\b | no | s that, as here, involve the Government in its sovereign capacity, the Granfinanciera Court plainly stated that “Congress may fashion causes of action that are closely analogous to common-law claims and [still] place the |
| article3_article1_tribunal | dev | Securities and Exchange Commission v. Jarkesy | 603 U.S. ___ | \bnon[- ]article iii\b | no | Water Act] was . . . to enable the Federal Government to bring or adjudicate claims that traced their ancestry to the common law.” Ante, at 23–24. 12 The concurrence’s assertion that the majority is “follow[ing] the advi |
| article3_article1_tribunal | test | Axon Enterprise, Inc. v. Federal Trade Commission | 598 U.S. 175 | \bnon[- ]article iii\b | yes | efits and entitlements) “could be taken away without judicial process.” Sessions v. Dimaya, 584 U. S. ___, ___ (2018) (Thomas, J., dissenting) (slip op., at 9); see also Mascott 25. Thus, “the legislative and executive b |

## Use Rules

1. Treat labels as `silver_high`; manually review before any promotion decision.
2. For public/private-rights contrasts, exclude rows where `has_public_private_conflict` is true.
3. Run probes on both `text` and `text_cue_masked`; promotion requires surviving cue masking and text-baseline checks.
4. Keep this source pack separate from target-justice style labels; it is for legal-frame source grounding, not justice-style classification.
