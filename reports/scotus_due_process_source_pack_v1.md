# SCOTUS Due Process Source Pack v1

## Purpose

Due Process is the next branch after Economic Activity, Civil Rights, and Federalism failed promotion gates. This pack tests substantive liberty/history-and-tradition reasoning against procedural Mathews/hearing reasoning.

## Outputs

- Raw pages: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/raw/scotus_due_process_source_pages_v1.json`
- Labels: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_due_process_source_frame_labels_v1.jsonl`
- Review queue: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_due_process_source_frame_review_queue_v1.jsonl`
- Source chunks scanned: `1684`

## Source Cases

| Case id | Case | Citation | Expected frame | Tokens | URL |
| --- | --- | --- | --- | --- | --- |
| griswold_1965 | Griswold v. Connecticut | 381 U.S. 479 | due_process_substantive | 25013 | https://www.law.cornell.edu/supremecourt/text/381/479 |
| roe_1973 | Roe v. Wade | 410 U.S. 113 | due_process_substantive | 29325 | https://www.law.cornell.edu/supremecourt/text/410/113 |
| casey_1992 | Planned Parenthood of Southeastern Pennsylvania v. Casey | 505 U.S. 833 | due_process_substantive | 77417 | https://www.law.cornell.edu/supremecourt/text/505/833 |
| glucksberg_1997 | Washington v. Glucksberg | 521 U.S. 702 | due_process_substantive | 36308 | https://www.law.cornell.edu/supremecourt/text/521/702 |
| lawrence_2003 | Lawrence v. Texas | 539 U.S. 558 | due_process_substantive | 20569 | https://www.law.cornell.edu/supremecourt/text/539/558 |
| obergefell_2015 | Obergefell v. Hodges | 576 U.S. 644 | due_process_substantive | 39578 | https://www.law.cornell.edu/supremecourt/text/14-556 |
| dobbs_2022 | Dobbs v. Jackson Women's Health Organization | 597 U.S. 215 | due_process_substantive | 91687 | https://www.law.cornell.edu/supremecourt/text/19-1392 |
| goldberg_1970 | Goldberg v. Kelly | 397 U.S. 254 | due_process_procedural_mathews | 10375 | https://www.law.cornell.edu/supremecourt/text/397/254 |
| morrissey_1972 | Morrissey v. Brewer | 408 U.S. 471 | due_process_procedural_mathews | 12839 | https://www.law.cornell.edu/supremecourt/text/408/471 |
| goss_1975 | Goss v. Lopez | 419 U.S. 565 | due_process_procedural_mathews | 15226 | https://www.law.cornell.edu/supremecourt/text/419/565 |
| mathews_1976 | Mathews v. Eldridge | 424 U.S. 319 | due_process_procedural_mathews | 13682 | https://www.law.cornell.edu/supremecourt/text/424/319 |
| loudermill_1985 | Cleveland Board of Education v. Loudermill | 470 U.S. 532 | due_process_procedural_mathews | 14861 | https://www.law.cornell.edu/supremecourt/text/470/532 |
| hamdi_2004 | Hamdi v. Rumsfeld | 542 U.S. 507 | due_process_procedural_mathews | 38630 | https://www.law.cornell.edu/supremecourt/text/542/507 |

## Label Counts

| Frame | Total | Cases | Train | Dev | Test | Multi-frame conflicts |
| --- | --- | --- | --- | --- | --- | --- |
| due_process_substantive | 72 | 13 | 38 | 32 | 2 | 5 |
| due_process_procedural_mathews | 72 | 13 | 65 | 2 | 5 | 25 |

## Case/Frame Coverage

| Case id | Case | Frame | Records |
| --- | --- | --- | --- |
| casey_1992 | Planned Parenthood of Southeastern Pennsylvania v. Casey | due_process_procedural_mathews | 1 |
| casey_1992 | Planned Parenthood of Southeastern Pennsylvania v. Casey | due_process_substantive | 6 |
| dobbs_2022 | Dobbs v. Jackson Women's Health Organization | due_process_procedural_mathews | 1 |
| dobbs_2022 | Dobbs v. Jackson Women's Health Organization | due_process_substantive | 31 |
| glucksberg_1997 | Washington v. Glucksberg | due_process_procedural_mathews | 1 |
| glucksberg_1997 | Washington v. Glucksberg | due_process_substantive | 7 |
| goldberg_1970 | Goldberg v. Kelly | due_process_procedural_mathews | 4 |
| goldberg_1970 | Goldberg v. Kelly | due_process_substantive | 1 |
| goss_1975 | Goss v. Lopez | due_process_procedural_mathews | 11 |
| goss_1975 | Goss v. Lopez | due_process_substantive | 1 |
| griswold_1965 | Griswold v. Connecticut | due_process_procedural_mathews | 1 |
| griswold_1965 | Griswold v. Connecticut | due_process_substantive | 1 |
| hamdi_2004 | Hamdi v. Rumsfeld | due_process_procedural_mathews | 14 |
| hamdi_2004 | Hamdi v. Rumsfeld | due_process_substantive | 1 |
| lawrence_2003 | Lawrence v. Texas | due_process_procedural_mathews | 1 |
| lawrence_2003 | Lawrence v. Texas | due_process_substantive | 10 |
| loudermill_1985 | Cleveland Board of Education v. Loudermill | due_process_procedural_mathews | 14 |
| loudermill_1985 | Cleveland Board of Education v. Loudermill | due_process_substantive | 1 |
| mathews_1976 | Mathews v. Eldridge | due_process_procedural_mathews | 12 |
| mathews_1976 | Mathews v. Eldridge | due_process_substantive | 1 |
| morrissey_1972 | Morrissey v. Brewer | due_process_procedural_mathews | 10 |
| morrissey_1972 | Morrissey v. Brewer | due_process_substantive | 1 |
| obergefell_2015 | Obergefell v. Hodges | due_process_procedural_mathews | 1 |
| obergefell_2015 | Obergefell v. Hodges | due_process_substantive | 10 |
| roe_1973 | Roe v. Wade | due_process_procedural_mathews | 1 |
| roe_1973 | Roe v. Wade | due_process_substantive | 1 |

## Sample Evidence Windows

| Frame | Split | Case | Citation | Evidence | Conflict | Window |
| --- | --- | --- | --- | --- | --- | --- |
| due_process_procedural_mathews | dev | Dobbs v. Jackson Women's Health Organization | 597 U.S. 215 | \bdeprivation\b | yes | Other sources, by contrast, suggest that “due process of law” prohibited legislatures “from authorizing the deprivation of a person’s life, liberty, or property without providing him the customary procedures to which fre |
| due_process_procedural_mathews | dev | Planned Parenthood of Southeastern Pennsylvania v. Casey | 505 U.S. 833 | \bnotice\b | yes | Subsection (12) of the reporting provision requires the reporting of, among other things, a married woman's "reason for failure to provide notice" to her husband. § 3214(a)(12). This provision in effect requires women, a |
| due_process_procedural_mathews | test | Goldberg v. Kelly | 397 U.S. 254 | \bprocedural due process\b, \bhearing\b, \bgoldberg\b | no | City, for appellees. Mr. Justice BRENNAN delivered the opinion of the Court. The question for decision is whether a State that terminates public assistance payments to a particular recipient without affording him the opp |
| due_process_procedural_mathews | test | Goldberg v. Kelly | 397 U.S. 254 | \bhearing\b, \bnotice\b | no | sisted program of Aid to Families with Dependent Children (AFDC) or under New York State's general Home Relief program.1 Their complaint alleged that the New York State and New York City officials administering these pro |
| due_process_procedural_mathews | test | Goldberg v. Kelly | 397 U.S. 254 | \bprocedural due process\b, \bhearing\b | no | true, of course, that some governmental benefits may be administratively terminated without affording the recipient a pre-termination evidentiary hearing.10 But we agree with the District Court that when welfare is disco |
| due_process_procedural_mathews | test | Goldberg v. Kelly | 397 U.S. 254 | \bhearing\b, \bnotice\b, \bopportunity to be heard\b | no | 'The fundamental requisite of due process of law is the opportunity to be heard.' Grannis v. Ordean, 234 U.S. 385, 394, 34 S.Ct. 779, 783, 58 L.Ed. 1363 (1914). The hearing must be 'at a meaningful time and in a meaingfu |
| due_process_procedural_mathews | test | Roe v. Wade | 410 U.S. 113 | \bdeprivation\b | yes | ensual transactions may be a form of 'liberty' protected by the Fourteenth Amendment, there is no doubt that similar claims have been upheld in our earlier decisions on the basis of that liberty. I agree with the stateme |
| due_process_procedural_mathews | train | Cleveland Board of Education v. Loudermill | 470 U.S. 532 | \bhearing\b, \bdeprivation\b, \bprivate interest\b | yes | f Appeals reversed in part and remanded, holding that both respondents had been deprived of due process and that the compelling private interest in retaining employment, combined with the value of presenting evidence pri |
| due_process_procedural_mathews | train | Cleveland Board of Education v. Loudermill | 470 U.S. 532 | \bhearing\b, \bnotice\b, \bprivate interest\b | no | Pp. 538-541. (b) The principle that under the Due Process Clause an individual must be given an opportunity for a hearing before he is deprived of any significant property interest, requires "some kind of hearing" prior  |
| due_process_procedural_mathews | train | Cleveland Board of Education v. Loudermill | 470 U.S. 532 | \bhearing\b, \bloudermill\b | yes | . See Fed.Rule Civ.Proc. 12(b)(6). It held that because the very statute that created the property right in continued employment also specified the procedures for discharge, and because those procedures were followed, Lo |
| due_process_procedural_mathews | train | Cleveland Board of Education v. Loudermill | 470 U.S. 532 | \bhearing\b, \bdeprivation\b, \bprivate interest\b | yes | n deprived of due process. It disagreed with the District Court's original rationale. Instead, it concluded that the compelling private interest in retaining employment, combined with the value of presenting evidence pri |
| due_process_procedural_mathews | train | Cleveland Board of Education v. Loudermill | 470 U.S. 532 | \bhearing\b, \bgoss\b, \bloudermill\b | no | The dissenting Judge argued that respondents' property interests were conditioned by the procedural limitations accompanying the grant thereof. He considered constitutional requirements satisfied because there was a reli |
| due_process_procedural_mathews | train | Cleveland Board of Education v. Loudermill | 470 U.S. 532 | \bhearing\b, \bnotice\b, \bdeprivation\b | yes | An essential principle of due process is that a deprivation of life, liberty, or property "be preceded by notice and opportunity for hearing appropriate to the nature of the case." Mullane v. Central Hanover Bank & Trust |
| due_process_procedural_mathews | train | Cleveland Board of Education v. Loudermill | 470 U.S. 532 | \bhearing\b, \bnotice\b, \bgovernment(?:'s)? interest\b | no | of due process, and all that respondents seek or the Court of Appeals required, are notice and an opportunity to respond. The opportunity to present reasons, either in person or in writing, why proposed action should not |
| due_process_procedural_mathews | train | Cleveland Board of Education v. Loudermill | 470 U.S. 532 | \bhearing\b, \bdeprivation\b, \bloudermill\b | no | V Our holding rests in part on the provisions in Ohio law for a full post-termination hearing. In his cross-petition Loudermill asserts, as a separate constitutional violation, that his administrative proceedings took to |
| due_process_procedural_mathews | train | Cleveland Board of Education v. Loudermill | 470 U.S. 532 | \bnotice\b, \bopportunity to be heard\b | no | senting witnesses on his own behalf, whenever there are substantial disputes in testimonial evidence," Arnett v. Kennedy, 416 U.S. 134, 214, 94 S.Ct. 1633, 1674, 40 L.Ed.2d 15 (1974) (MARSHALL, J., dissenting). Because t |
| due_process_procedural_mathews | train | Cleveland Board of Education v. Loudermill | 470 U.S. 532 | \bmathews\b, \bhearing\b, \bdeprivation\b | no | mportance of the private interest and the length or finality of the deprivation, the likelihood of governmental error, and the magnitude of the governmental interests involved." Logan v. Zimmerman Brush Co., 455 U.S. 422 |
| due_process_procedural_mathews | train | Cleveland Board of Education v. Loudermill | 470 U.S. 532 | \bmathews\b, \bloudermill\b | no | e Due Process Clause, either explicitly or sub silentio, have been decided only after more complete proceedings in the District Courts. See, e.g., $8,850, supra; Barry v. Barchi, 443 U.S. 55, 99 S.Ct. 2642, 61 L.Ed.2d 36 |
| due_process_procedural_mathews | train | Cleveland Board of Education v. Loudermill | 470 U.S. 532 | \bprocedural due process\b, \bmathews\b, \bgoldberg\b | no | , 276, 90 S.Ct. 1011, 1024, 25 L.Ed.2d 287 (1970) (Black, J., dissenting). The results from today's balance certainly do not jibe with the result in Goldberg or Mathews v. Eldridge, 424 U.S. 319, 96 S.Ct. 893, 47 L.Ed.2d |
| due_process_procedural_mathews | train | Cleveland Board of Education v. Loudermill | 470 U.S. 532 | \bmathews\b, \bdeprivation\b, \bloudermill\b | yes | fine the necessary procedures in the course of creating the property right. Instead, it reached the same result under a balancing test based on Justice POWELL's concurring opinion in Arnett v. Kennedy, 416 U.S. 134, 168- |

## Use Rules

1. Run a cue-masked text-only gate before any BF16 activation capture.
2. If text alone solves substantive versus procedural due process, close the branch as leakage/text dominated.
3. If text-only is not saturated, run dominance review before activation probing.
