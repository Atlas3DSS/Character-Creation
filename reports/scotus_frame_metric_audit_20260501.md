# SCOTUS Frame Metric Audit

## Purpose

This audits the lightweight keyword frame metric used for the Q4 proxy and BF16 hook-generation pilots. It is a triage report, not a final evaluator.

## Main Read

1. The metric is useful for fast gates, but it is not yet strong enough to support a steering claim by itself.
2. Off-domain hits often come from broad or polysemous substrings such as `home`, `consent`, `district`, `damages`, `remedy`, and generic separation-of-powers wording.
3. Frame-pilot baselines are already saturated on several prompts, so a small target-hit increase can mean repeated vocabulary rather than a new legal frame.
4. The next evaluator pass should score frame presence as a legal proposition, not raw keyword repetition.

## BF16 Frame-Pilot Gate Results

| Pilot | Alpha | N | Matched target delta | Target z | Target win rate | Matched net delta | Net z | Net win rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Article III | 0.02 | 4 | -0.35 | -0.29 | 0.25 | -0.90 | -0.40 | 0.25 |
| Article III | 0.05 | 4 | -0.10 | -0.08 | 0.50 | -0.65 | -0.30 | 0.50 |
| Article III | 0.10 | 4 | -0.03 | -0.02 | 0.25 | 0.70 | 0.42 | 0.75 |
| Fourth Amendment | 0.02 | 4 | -0.17 | -0.75 | 0.00 | -0.42 | -0.62 | 0.00 |
| Fourth Amendment | 0.05 | 4 | 0.53 | 0.93 | 0.50 | 0.65 | 0.55 | 0.75 |
| Fourth Amendment | 0.10 | 4 | -0.78 | -0.70 | 0.25 | -2.72 | -1.38 | 0.00 |

## Baseline Saturation

High baseline target presence makes target-hit deltas hard to interpret; the scorer is often counting repetition inside an already-correct frame.

| Pilot | Base rows | Target present rate | Mean target hits | Mean contrast hits |
| --- | --- | --- | --- | --- |
| Article III | 4 | 0.75 | 3.75 | 2.00 |
| Fourth Amendment | 4 | 1.00 | 3.25 | 3.00 |

## Proxy Off-Domain Frame Counts

| Frame | Rows |
| --- | --- |
| separation_presidential_power | 339 |
| article3_private_rights | 216 |
| article3_public_rights | 174 |
| fourth_home_exigency | 95 |
| economic_remedy_damages | 72 |
| due_process_substantive | 71 |
| article3_article1_tribunal | 71 |
| fourth_exigency_consent | 67 |
| due_process_procedural_mathews | 37 |
| economic_commerce_clause | 17 |
| federalism_anti_commandeering | 12 |
| civil_equal_protection_strict_scrutiny | 9 |

## Highest Off-Domain Proxy Examples

| Prompt | Condition | Sample | Off hits | Off frames | Likely triggers | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| JP04_immigration_removal | bench_memo | random_control | 6 | due_process_substantive, separation_presidential_power | due_process_substantive: liberty x1; separation_presidential_power: removal x5 | **MEMORANDUM** **TO:** The Court **FROM:** Clerk?s Office **RE:** Separation-of-Powers Challenge to Administrative Removal Proceedings **I. Issue** Whether Article III of the Constitution requires an independent judicial determination of facts prior to the... |
| JP04_immigration_removal | bench_memo | random_control | 6 | due_process_substantive, separation_presidential_power | due_process_substantive: liberty x1; separation_presidential_power: removal x5 | **MEMORANDUM** **TO:** The Court **FROM:** Clerk?s Office **RE:** Separation-of-Powers Challenge to Agency Removal Order **Issue** Whether Article III of the Constitution requires a judicial fact-finding proceeding prior to the executive removal of a noncit... |
| JP04_immigration_removal | bench_memo | random_control | 6 | due_process_procedural_mathews, separation_presidential_power | due_process_procedural_mathews: hearing x1; separation_presidential_power: removal x3, separation of powers x2 | **MEMORANDUM** **TO:** The Court **FROM:** Clerk?s Office **RE:** Separation of Powers Analysis Regarding Article III Judicial Review of Immigration Removal Orders **I. Issue** Whether the Constitution?s Article III requires an independent judicial fact-fin... |
| JP04_immigration_removal | bench_memo | random_control | 6 | due_process_substantive, separation_presidential_power | due_process_substantive: liberty x1; separation_presidential_power: removal x4, separation of powers x1 | **MEMORANDUM** **TO:** The Court **FROM:** Clerk?s Office **RE:** Separation of Powers Challenge to Administrative Removal Proceedings **Issue** Whether the Constitution?s Article III requires an independent judicial fact-finding prior to the administrative... |
| JP04_immigration_removal | bench_memo | random_control | 6 | due_process_procedural_mathews, separation_presidential_power | due_process_procedural_mathews: hearing x1; separation_presidential_power: removal x4, separation of powers x1 | **MEMORANDUM** **TO:** The Court **FROM:** Clerk?s Office **RE:** Separation of Powers and Article III Jurisdiction in Removal Proceedings **I. Issue** Whether Article III of the Constitution requires an independent federal court to conduct a de novo factua... |
| JP04_immigration_removal | bench_memo | random_control | 6 | due_process_substantive, economic_commerce_limits, separation_presidential_power | due_process_substantive: liberty x1; economic_commerce_limits: lopez x1; separation_presidential_power: removal x4 | **ISSUE** Whether Article III of the Constitution requires that factual findings supporting an order of removal be made by an independent Article III court, rather than by an administrative agency. **ANALYSIS** The noncitizen contends that removal proceedin... |
| JP04_immigration_removal | bench_memo | base | 5 | fourth_home_exigency, separation_presidential_power | fourth_home_exigency: home x1; separation_presidential_power: removal x3, separation of powers x1 | **MEMORANDUM** **TO:** The Court **FROM:** Clerk?s Office **RE:** Separation of Powers Analysis Regarding Article III Fact-Finding Requirement in Removal Proceedings **I. ISSUE** Whether the Constitution?s Article III requires an independent judicial determ... |
| JP04_immigration_removal | bench_memo | random_control | 5 | separation_presidential_power | separation_presidential_power: removal x4, separation of powers x1 | **MEMORANDUM** **TO:** The Court **FROM:** Clerk?s Office **RE:** Separation of Powers Analysis Regarding Agency Fact-Finding in Removal Proceedings **I. Issue** Whether the Constitution?s Article III requires an independent judicial determination of facts... |
| JP04_immigration_removal | bench_memo | random_control | 5 | due_process_substantive, separation_presidential_power | due_process_substantive: liberty x1; separation_presidential_power: removal x3, separation of powers x1 | **MEMORANDUM** **TO:** The Court **FROM:** Clerk **RE:** Separation of Powers Challenge to Immigration Removal Proceedings **Issue** Whether Article III requires a judicial fact-finding by an independent court prior to the execution of an administrative rem... |
| JP04_immigration_removal | bench_memo | random_control | 5 | due_process_substantive, separation_presidential_power | due_process_substantive: liberty x1; separation_presidential_power: removal x4 | **MEMORANDUM** **TO:** The Court **FROM:** Clerk?s Office **RE:** Separation-of-Powers Challenge to Immigration Removal Proceedings **Issue** Whether Article III of the Constitution requires an independent judicial determination of facts before an administr... |
| JP04_immigration_removal | bench_memo | random_control | 5 | due_process_procedural_mathews, separation_presidential_power | due_process_procedural_mathews: hearing x1; separation_presidential_power: removal x3, separation of powers x1 | **MEMORANDUM** **TO:** The Court **FROM:** Clerk?s Office **RE:** Separation of Powers; Article III Jurisdiction in Removal Proceedings **I. Question Presented** Whether the Constitution?s Article III requires an independent judicial finding of fact prior t... |
| JP04_immigration_removal | bench_memo | random_control | 5 | separation_presidential_power | separation_presidential_power: removal x3, separation of powers x2 | **MEMORANDUM** **TO:** The Court **FROM:** Clerk?s Office **RE:** Separation of Powers Challenge to Administrative Removal Order **Issue** Whether Article III of the Constitution requires an independent judicial determination of facts prior to the issuance... |

## Blind-Sample Keyword Warnings

These examples should be manually reviewed before using the blind sample as an evaluator calibration set.

| Blind ID | Prompt | Issue | Off score | Frame scores | Likely triggers | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| blind_0078_d3d45031 | JP04_immigration_removal | Judicial Power | 5 | {"article3_article1_tribunal": 4, "due_process_substantive": 1, "separation_presidential_power": 4} | article3_article1_tribunal: article i x4; due_process_substantive: liberty x1; separation_presidential_power: removal x3, separation of powers x1 | **MEMORANDUM** **TO:** The Court **FROM:** Clerk?s Office **RE:** Separation of Powers; Article III Judicial Power; Immigration Removal Proceedings **I. Issue** Whether the Administrative Procedure Act?s framework for immigration removal proceedings, wherei... |
| blind_0107_a4b7237a | CR04_state_disability_access | Civil Rights | 4 | {"article3_private_rights": 1, "civil_equal_protection_strict_scrutiny": 1, "civil_section5_congruence": 3, "due_process_substantive": 2, "economic_commerce_clause": 1} | article3_private_rights: damages x1; civil_equal_protection_strict_scrutiny: equal protection x1; civil_section5_congruence: section 5 x1, congruent and proportional x1; due_process_substantive: liberty x1, fundamental right x1 | The Eleventh Amendment shields the States from private suits for damages in federal court, a sovereign immunity that is not abrogated by Congress?s Commerce Clause authority. However, under Section 5 of the Fourteenth Amendment, Congress possesses the remed... |
| blind_0053_ce68185f | JP04_immigration_removal | Judicial Power | 3 | {"article3_article1_tribunal": 1, "due_process_procedural_mathews": 1, "separation_presidential_power": 2} | article3_article1_tribunal: article i x1; due_process_procedural_mathews: hearing x1; separation_presidential_power: removal x1, separation of powers x1 | The petition for review is denied. The Constitution vests the executive power in the President, who is charged with the duty to "take Care that the Laws be faithfully executed." This includes the authority to administer the immigration laws, which explicitl... |
| blind_0072_60aaaaeb | CR04_state_disability_access | Civil Rights | 3 | {"article3_private_rights": 1, "civil_equal_protection_strict_scrutiny": 1, "civil_section5_congruence": 3, "economic_remedy_damages": 2} | article3_private_rights: damages x1; civil_equal_protection_strict_scrutiny: equal protection x1; civil_section5_congruence: section 5 x1, congruent and proportional x1; economic_remedy_damages: statutory damages x1, remedy x1 | The Eleventh Amendment?s bar on suits against states is not absolute; it yields when Congress validly exercises its power under Section 5 of the Fourteenth Amendment to remedy or prevent violations of substantive constitutional rights. Here, the legislative... |
| blind_0124_bcdd0930 | JP04_immigration_removal | Judicial Power | 3 | {"article3_article1_tribunal": 3, "separation_presidential_power": 3} | article3_article1_tribunal: article i x3; separation_presidential_power: removal x2, separation of powers x1 | **Analysis** The contention that Article III mandates an independent judicial fact-finding prior to administrative removal order is without merit. **I. Nature of the Proceeding** Removal proceedings are civil in nature, not criminal prosecutions. The Suprem... |
| blind_0154_3a2d1468 | CR04_state_disability_access | Civil Rights | 3 | {"article3_private_rights": 2, "civil_section5_congruence": 3, "fourth_exigency_consent": 1} | article3_private_rights: damages x2; civil_section5_congruence: section 5 x1, sovereign immunity x2; fourth_exigency_consent: consent x1 | **ISSUE** Whether Congress validly abrogated state sovereign immunity by authorizing damages against a state for failing to provide courtroom access to persons with disabilities, pursuant to its enforcement power under Section 5 of the Fourteenth Amendment.... |
| blind_0009_eab748f4 | EA01_commercial_remedy | Economic Activity | 2 | {"article3_private_rights": 2, "economic_commerce_clause": 3, "economic_federalism_state_regulation": 2, "economic_remedy_damages": 4} | article3_private_rights: damages x2; economic_commerce_clause: commerce clause x2, interstate commerce x1; economic_federalism_state_regulation: traditional state x1, police power x1; economic_remedy_damages: statutory damages x2, remedy x2 | **MEMORANDUM** **TO:** The Court **FROM:** Clerk?s Office **RE:** Constitutional and Statutory Analysis of Federal Remedy for Misleading Commercial Conduct **I. ISSUE** Whether the federal statute creating a remedy with statutory damages for misleading comm... |
| blind_0021_ec3b996f | FD01_state_sheriff_checks | Federalism | 2 | {"article3_public_rights": 2, "federalism_anti_commandeering": 2} | article3_public_rights: federal regulatory x2; federalism_anti_commandeering: commandeer x1, state officials x1 | The Tenth Amendment reserves to the States the powers not delegated to the United States, and among those reserved powers is the authority to direct the conduct of their own officers. The Constitution grants Congress the power to make all laws necessary and... |
| blind_0022_e8cf38ba | CR04_state_disability_access | Civil Rights | 2 | {"article3_private_rights": 2, "civil_section5_congruence": 4} | article3_private_rights: damages x2; civil_section5_congruence: section 5 x3, sovereign immunity x1 | **MEMORANDUM** **TO:** The Court **FROM:** Clerk?s Office **RE:** Analysis of Congressional Authority Under Section 5 of the Fourteenth Amendment **I. Issue** Whether Congress possesses the authority under Section 5 of the Fourteenth Amendment to abrogate s... |
| blind_0026_828d8ff7 | JP01_agency_private_company | Judicial Power | 2 | {"article3_article1_tribunal": 4, "article3_private_rights": 2, "economic_remedy_damages": 1, "separation_presidential_power": 1} | article3_article1_tribunal: article i x2, life tenure x1; article3_private_rights: private right x1, common-law x1; economic_remedy_damages: private right of action x1; separation_presidential_power: separation of powers x1 | The constitutional inquiry turns on the structural integrity of Article III?s guarantee of judicial independence and the separation of powers. The core issue is whether the agency?s adjudicative function constitutes an exercise of the judicial power of the... |
| blind_0030_91cd5f5f | EA02_homegrown_market_regulation | Economic Activity | 2 | {"article3_article1_tribunal": 1, "economic_commerce_clause": 3, "fourth_home_exigency": 1} | article3_article1_tribunal: article i x1; economic_commerce_clause: commerce clause x1, interstate commerce x2; fourth_home_exigency: home x1 | **Holding:** Congress possesses the constitutional authority to regulate the production of a commodity for personal use, provided such regulation is rationally related to the regulation of interstate commerce. **Reasoning:** 1. **Scope of the Commerce Claus... |
| blind_0034_f5b934e2 | EA02_homegrown_market_regulation | Economic Activity | 2 | {"article3_article1_tribunal": 1, "economic_commerce_clause": 3, "fourth_home_exigency": 1} | article3_article1_tribunal: article i x1; economic_commerce_clause: commerce clause x2, interstate commerce x1; fourth_home_exigency: home x1 | **MEMORANDUM** **TO:** The Court **FROM:** Clerk **RE:** Constitutional Authority to Regulate Local, Non-Commercial Activity **I. ISSUE** Whether Congress possesses authority under the Commerce Clause to regulate the homegrown production of a commodity that... |

## Evaluator Repair Checklist

1. Replace raw substring counts with boolean proposition-level frame labels per completion.
2. Split ambiguous patterns: `home` should not trigger home-exigency by itself, and `consent` should not trigger Fourth Amendment consent when used in ordinary language.
3. Add negation and role checks for contrast frames, especially where a completion rejects search-incident doctrine while still naming it.
4. Score target-minus-contrast as the primary automatic metric, but require a blind-review sample before promoting any direction.
5. For source-grounded frame data, label short opinion excerpts by doctrinal proposition rather than by justice or raw keyword.

## Proposition-Level Rescore Pass

A first evaluator-repair pass now exists:

- Script: `scripts/experiments/scotus/rescore_scotus_frame_propositions.py`
- Run: `sweep_v4/scotus_frame_prop_rescore_20260501_012850/report.md`
- Inputs: Q4 proxy generation, Article III BF16 frame pilot, Fourth Amendment BF16 frame pilot
- Rows rescored: `3332`

Largest corrected false-positive families:

| Frame | Old rows | Proposition rows | Dropped rows |
| --- | ---: | ---: | ---: |
| `separation_presidential_power` | 367 | 1 | 366 |
| `civil_equal_protection_strict_scrutiny` | 595 | 278 | 317 |
| `article3_article1_tribunal` | 807 | 502 | 308 |
| `article3_private_rights` | 459 | 275 | 239 |
| `article3_public_rights` | 427 | 364 | 180 |
| `fourth_home_exigency` | 240 | 115 | 132 |

The main audit finding is confirmed: raw substring counts were heavily inflated by prompt vocabulary and polysemy. The repair removes the worst artifacts, especially immigration `removal` being misread as presidential-removal doctrine and bare `home` being misread as Fourth Amendment home-exigency doctrine.

The retained run includes one disagreement-queue repair pass: immigration/agency adjudication propositions were broadened enough to catch real administrative-adjudication reasoning, while the Article III final-judgment rule was narrowed to avoid generic separation-of-powers hits.

BF16 frame-pilot read under proposition scoring:

| Pilot | Alpha | N | Target minus random | Target z | Target win | Net minus random | Net z | Net win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Article III | 0.02 | 4 | -0.350 | -0.527 | 0.000 | -0.725 | -0.741 | 0.000 |
| Article III | 0.05 | 4 | 0.575 | 0.757 | 0.500 | 0.500 | 0.511 | 0.500 |
| Article III | 0.10 | 4 | 0.025 | 0.027 | 0.500 | 0.250 | 0.222 | 0.750 |
| Fourth Amendment | 0.02 | 4 | -0.075 | -0.323 | 0.000 | -0.450 | -0.485 | 0.000 |
| Fourth Amendment | 0.05 | 4 | 0.475 | 1.146 | 0.500 | 0.725 | 0.880 | 0.500 |
| Fourth Amendment | 0.10 | 4 | -0.800 | -0.826 | 0.000 | -2.050 | -1.190 | 0.000 |

Decision: no pilot clears promotion under the repaired evaluator. The Fourth alpha `0.05` hint remains the strongest row, but it is still `n=4`, prompt win rate is only `0.50`, and alpha `0.10` reverses. Treat this as calibration evidence, not a steerable-circuit finding.
