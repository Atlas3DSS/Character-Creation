# SCOTUS Civil Rights Source Pack v1

## Purpose

Civil Rights is the backup source-pack branch after Economic Activity failed its promotion gate. This pack is deliberately labeled `silver_review_required` because scrutiny-level doctrine is likely to be lexically separable.

## Outputs

- Raw pages: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/raw/scotus_civil_source_pages_v1.json`
- Labels: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_civil_source_frame_labels_v1.jsonl`
- Review queue: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_civil_source_frame_review_queue_v1.jsonl`
- Source chunks scanned: `2451`

## Source Cases

| Case id | Case | Citation | Expected frame | Tokens | URL |
| --- | --- | --- | --- | --- | --- |
| loving_1967 | Loving v. Virginia | 388 U.S. 1 | civil_race_strict_scrutiny | 4863 | https://www.law.cornell.edu/supremecourt/text/388/1 |
| bakke_1978 | Regents of the University of California v. Bakke | 438 U.S. 265 | civil_race_strict_scrutiny | 71276 | https://www.law.cornell.edu/supremecourt/text/438/265 |
| croson_1989 | City of Richmond v. J. A. Croson Co. | 488 U.S. 469 | civil_race_strict_scrutiny | 42472 | https://www.law.cornell.edu/supremecourt/text/488/469 |
| adarand_1995 | Adarand Constructors, Inc. v. Pena | 515 U.S. 200 | civil_race_strict_scrutiny | 36421 | https://www.law.cornell.edu/supremecourt/text/515/200 |
| grutter_2003 | Grutter v. Bollinger | 539 U.S. 306 | civil_race_strict_scrutiny | 35520 | https://www.law.cornell.edu/supremecourt/text/02-241 |
| gratz_2003 | Gratz v. Bollinger | 539 U.S. 244 | civil_race_strict_scrutiny | 25752 | https://www.law.cornell.edu/supremecourt/text/02-516 |
| fisher_2016 | Fisher v. University of Texas at Austin | 579 U.S. 365 | civil_race_strict_scrutiny | 30938 | https://www.law.cornell.edu/supremecourt/text/14-981 |
| sffa_2023 | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | civil_race_strict_scrutiny | 100434 | https://www.law.cornell.edu/supremecourt/text/20-1199 |
| reed_1971 | Reed v. Reed | 404 U.S. 71 | civil_sex_intermediate_scrutiny | 2434 | https://www.law.cornell.edu/supremecourt/text/404/71 |
| frontiero_1973 | Frontiero v. Richardson | 411 U.S. 677 | civil_sex_intermediate_scrutiny | 7064 | https://www.law.cornell.edu/supremecourt/text/411/677 |
| craig_1976 | Craig v. Boren | 429 U.S. 190 | civil_sex_intermediate_scrutiny | 18518 | https://www.law.cornell.edu/supremecourt/text/429/190 |
| hogan_1982 | Mississippi University for Women v. Hogan | 458 U.S. 718 | civil_sex_intermediate_scrutiny | 13613 | https://www.law.cornell.edu/supremecourt/text/458/718 |
| vmi_1996 | United States v. Virginia | 518 U.S. 515 | civil_sex_intermediate_scrutiny | 37427 | https://www.law.cornell.edu/supremecourt/text/518/515 |
| nguyen_2001 | Nguyen v. INS | 533 U.S. 53 | civil_sex_intermediate_scrutiny | 18900 | https://www.law.cornell.edu/supremecourt/text/533/53 |
| morales_santana_2017 | Sessions v. Morales-Santana | 582 U.S. 47 | civil_sex_intermediate_scrutiny | 15851 | https://www.law.cornell.edu/supremecourt/text/15-1191 |
| boerne_1997 | City of Boerne v. Flores | 521 U.S. 507 | civil_section5_congruence | 25642 | https://www.law.cornell.edu/supremecourt/text/521/507 |
| kimel_2000 | Kimel v. Florida Board of Regents | 528 U.S. 62 | civil_section5_congruence | 20866 | https://www.law.cornell.edu/supremecourt/text/528/62 |
| garrett_2001 | Board of Trustees of the University of Alabama v. Garrett | 531 U.S. 356 | civil_section5_congruence | 20926 | https://www.law.cornell.edu/supremecourt/text/99-1240 |
| hibbs_2003 | Nevada Department of Human Resources v. Hibbs | 538 U.S. 721 | civil_section5_congruence | 17430 | https://www.law.cornell.edu/supremecourt/text/01-1368 |
| lane_2004 | Tennessee v. Lane | 541 U.S. 509 | civil_section5_congruence | 25930 | https://www.law.cornell.edu/supremecourt/text/02-1667 |
| coleman_2012 | Coleman v. Court of Appeals of Maryland | 566 U.S. 30 | civil_section5_congruence | 15837 | https://www.law.cornell.edu/supremecourt/text/10-1016 |
| cleburne_1985 | City of Cleburne v. Cleburne Living Center | 473 U.S. 432 | civil_rational_basis_equal_protection | 22580 | https://www.law.cornell.edu/supremecourt/text/473/432 |
| heller_doe_1993 | Heller v. Doe | 509 U.S. 312 | civil_rational_basis_equal_protection | 18421 | https://www.law.cornell.edu/supremecourt/text/509/312 |
| romer_1996 | Romer v. Evans | 517 U.S. 620 | civil_rational_basis_equal_protection | 13975 | https://www.law.cornell.edu/supremecourt/text/517/620 |

## Label Counts

| Frame | Total | Cases | Train | Dev | Test | Multi-frame conflicts |
| --- | --- | --- | --- | --- | --- | --- |
| civil_race_strict_scrutiny | 72 | 16 | 42 | 22 | 8 | 5 |
| civil_sex_intermediate_scrutiny | 72 | 12 | 53 | 8 | 11 | 13 |
| civil_section5_congruence | 72 | 21 | 54 | 15 | 3 | 2 |
| civil_rational_basis_equal_protection | 55 | 13 | 44 | 7 | 4 | 19 |

## Sample Evidence Windows

| Frame | Split | Case | Citation | Evidence | Conflict | Window |
| --- | --- | --- | --- | --- | --- | --- |
| civil_race_strict_scrutiny | dev | City of Boerne v. Flores | 521 U.S. 507 | \bcompelling (?:governmental |state )?interest\b, \bracial\b | yes | t. 2217, 2227, 124 L.Ed.2d 472 (1993) (" [A] law targeting religious beliefs as such is never permissible''). To avoid the difficulty of proving such violations, it is said, Congress can simply invalidate any law which i |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bstrict scrutiny\b, \bracial classification\b, \brace\b | no | protection cannot mean one thing when applied to one individual and something else when applied to a person of another color.” Regents of Univ. of Cal. v. Bakke, 438 U. S. 265, 289–290. Any exceptions to the Equal Protec |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bcompelling (?:governmental |state )?interest\b, \brace\b, \bdiversity\b | no | at 317. Pp. 16–19. (d) For years following Bakke, lower courts struggled to determine whether Justice Powell’s decision was “binding precedent.” Grutter, 539 U. S., at 325. Then, in Grutter v. Bollinger, the Court for th |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bstrict scrutiny\b, \bnarrowly tailored\b, \bracial classification\b | no | ts of Univ. of Cal. v. Bakke, 438 U. S. 265, 289–290 (1978) (opinion of Powell, J.). “If both are not accorded the same protection, then it is not equal.” Id., at 290. Any exception to the Constitution’s demand for equal |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bcompelling (?:governmental |state )?interest\b, \brace\b, \bracial\b | no | p the matter again in 2003, in the case Grutter v. Bollinger, which concerned the admissions system used by the University of Michigan law school. Id., at 311. There, in another sharply divided decision, the Court for th |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bstrict scrutiny\b, \brace\b, \bracial\b | no | Brief for Respondent in No. 20–1199, p. 52. Neither does UNC’s. 567 F. Supp. 3d, at 612. Yet both insist that the use of race in their admissions programs must continue. But we have permitted race-based admissions only w |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bstrict scrutiny\b, \brace\b, \bracial\b | no | sher v. University of Tex. at Austin, 570 U. S. 297, 315, 328 (2013) (concurring opinion) (Fisher I ); Fisher v. University of Tex. at Austin, 579 U. S. 365, 389 (2016) (dissenting opinion). Today, and despite a lengthy  |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bstrict scrutiny\b, \bnarrowly tailored\b, \brace\b | no | That is why “only those measures the State must take to provide a bulwark against anarchy, or to prevent violence, will constitute a pressing public necessity” sufficient to satisfy strict scrutiny today. Grutter, 539 U. |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bstrict scrutiny\b, \bnarrowly tailored\b, \brace\b | no | about racial preferences in school admissions under Title VI has turned into a case about the meaning of the Fourteenth Amendment. And what a confused body of constitutional law followed. For years, this Court has said t |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bstrict scrutiny\b, \bcompelling (?:governmental |state )?interest\b, \bnarrowly tailored\b | no | uspect. See Grutter v. Bollinger, 539 U. S. 306, 326 (2003); Strauder v. West Virginia, 100 U. S. 303, 306–308 (1880). As a result, the Court has long held that racial classifications by the government, including race-ba |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bcompelling (?:governmental |state )?interest\b, \bnarrowly tailored\b, \bracial classification\b | no | Importantly, even if a racial classification is otherwise narrowly tailored to further a compelling governmental interest, a “deviation from the norm of equal treatment of all racial and ethnic groups” must be “a tempora |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bnarrowly tailored\b, \brace\b, \bracial\b | no | Petitioner 21–22, 30–31, 33, 42, Brief for United States 26–27, in Grutter v. Bollinger, O. T. 2002, No. 02–241. The Grutter Court rejected those arguments for ending race-based affirmative action in higher education in  |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bnarrowly tailored\b, \brace\b, \bracial\b | no | ly recount the horrific history of slavery and Jim Crow in America, cf. Bakke, 438 U. S., at 395–402 (opinion of Marshall, J.), as well as the continuing effects of that history on African Americans today. And they are o |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bcompelling (?:governmental |state )?interest\b, \bnarrowly tailored\b, \brace\b | no | ions process. Id., at 316–318. Since Bakke, the Court has reaffirmed numerous times the constitutionality of limited race-conscious college admissions. First, in Grutter v. Bollinger, 539 U. S. 306 (2003), a majority of  |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bstrict scrutiny\b, \bnarrowly tailored\b, \brace\b | no | Later, in the Fisher litigation, the Court twice reaffirmed that a limited use of race in college admissions is constitutionally permissible if it satisfies strict scrutiny. In Fisher v. University of Texas at Austin, 57 |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bnarrowly tailored\b, \brace\b, \bdiversity\b | no | vard II ). SFFA then filed petitions for a writ of certiorari in both cases, which the Court granted. 595 U. S. ___ (2022).24 The Court granted certiorari on three questions: (1) whether the Court should overrule Bakke,  |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bnarrowly tailored\b, \brace\b, \bracial\b | no | Brief for Petitioner 83–86. The use of race is narrowly tailored unless “workable” and “available” race-neutral approaches exist, meaning race-neutral alternatives promote the institution’s diversity goals and do so at “ |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bstrict scrutiny\b, \bnarrowly tailored\b, \brace\b | no | dardized test,” to rank students based on grades and test scores. Ibid. One of SFFA’s top percentage plans would even “nearly erase the Native American incoming class” at UNC. Id., at 646. The courts below correctly conc |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bcompelling (?:governmental |state )?interest\b, \brace\b, \bracial\b | no | It strikes at the heart of Bakke, Grutter, and Fisher by holding that racial diversity is an “inescapably imponderable” objective that cannot justify race-conscious affirmative action, ante, at 24, even though respondent |
| civil_race_strict_scrutiny | dev | Students for Fair Admissions, Inc. v. President and Fellows of Harvard College | 600 U.S. 181 | \bcompelling (?:governmental |state )?interest\b, \brace\b, \bracial\b | no | S. 709, 725 (2012) (plurality opinion) (“[P]rotecting the integrity of the Medal of Honor” is a “compelling interes[t]”); Sable Communications of Cal., Inc. v. FCC, 492 U. S. 115, 126 (1989) (“[P]rotecting the physical a |

## Use Rules

1. Do not run a BF16 probe from this pack until the review queue is sampled for dominant-frame validity.
2. Probe only `text_cue_masked`, with conflict-row exclusion and strict source-cluster-heldout splits.
3. Promotion requires activation performance clearly above the cue-masked text baseline.
4. Treat any strict-vs-intermediate win as suspect until a bag-of-cues baseline and manual dominance review clear it.
