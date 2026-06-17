# SCOTUS Phase 5 Evidence Decision

## Decision

The broad averaged L16 justice directions remain decodable but are not promoted as steerable judicial circuits. Both last-token and all-position BF16 hook pilots stayed within prompt-matched same-layer random controls.

The first curated frame-contrast branch also does not clear the causal gate. The Article III and Fourth Amendment frame probes decode perfectly, but their text baselines are also perfect; subsequent hook-generation pilots show at most weak, unstable movement versus prompt-matched random controls.

Expanded source-pack follow-ups have not changed that decision. Article III remains chance after dominance review, and the rebuilt Fourth Amendment pack is still text-baseline dominated or too split-skewed to justify a causal steering run.

The repaired proposition-level evaluator also does not rescue the weak frame-pilot rows. It removes many keyword artifacts, but the remaining Article III and Fourth Amendment effects are still small, unstable, and prompt-count limited.

Justice-style slice mining rejected the broad Economic Activity Scalia/Ginsburg branch as a steering candidate. The refined majority-2000s feasible-issues branch is the strongest decodability result so far: normal/template/plain median test BA is `0.746`/`0.758`/`0.764`, while label-shuffle/excerpt-removed/neutral-filler median test BA is `0.541`/`0.500`/`0.542`. But its first two causal pilots also fail promotion: `prompt_last @ L10` reaches only prompt-matched net `z=0.533`, and `excerpt_mean @ L16` reaches only prompt-matched target `z=0.449`.

Post-adjudication Economic Activity follow-up also eliminates the current source-frame direction. The original source pack has `50` expected-frame/frame mismatches and `120` multi-frame conflict rows; after filtering to clean broad-Commerce versus Commerce-limits rows, a cached Qwen3.5 BF16 rescore kept `51` rows but reached only `0.393` test BA versus a `0.679` cue-masked text baseline. A dominance review then kept `28` broad-Commerce rows and `21` Commerce-limits rows, but the reviewed-label cached probe still failed: best activation test BA `0.473` versus cue-masked text test BA `0.857`. Do not run a causal pocket pilot from this Economic Activity source direction.

The next source-pack branches also stop before hooks. Federalism anti-commandeering versus preemption is cue-masked text dominated (`1.000` test BA), Due Process substantive versus procedural is also cue-masked text dominated (`1.000` test BA), and Administrative Law major-questions versus ordinary deference reaches `1.000` final cue-masked text test BA on a case-skewed split despite weak dev BA (`0.586`). These are useful leakage diagnostics, not activation candidates.

An off-domain smoke test also failed to show a broad portable reasoning-style effect. The surviving Phase 4 and majority-2000s directions were nudged on ordinary nonlegal prompts about weather, video-game balance, friends choosing a restaurant, homework planning, boys basketball tryouts, and headphones. Last-token nudges were effectively inert; all-token L16 nudges produced mild formatting/structured-framework changes, but same-layer random controls produced comparable changes. Treat this as evidence against a broad "general judicial temperament" interpretation of the current directions.

The final Commerce-pocket follow-up also fails promotion. The two prompt pockets that survived internal adjudication, `EA03_gun_school_zone` and `EA01_commercial_remedy`, were expanded into a targeted Commerce Clause prompt bank with 8 same-layer random controls. The `prompt_last @ L10` direction at alpha `0.02` had only matched target delta `0.115` and matched net delta `0.021` across 12 prompts. The `excerpt_mean @ L16` all-token direction at alpha `0.02` was negative on the authority/remedy prompts: matched target and net deltas were both `-0.479`. Do not promote the Commerce-pocket branch.

## Artifacts

| Artifact | Path | Rows | Position | Alphas |
| --- | --- | --- | --- | --- |
| Last-token BF16 hook pilot | sweep_v4/scotus_sae_poke_20260430_224651 | 860 | last | 0.05, 0.1, 0.2 |
| All-position BF16 hook sanity | sweep_v4/scotus_sae_poke_20260430_233245 | 336 | all | 0.01, 0.02, 0.05 |
| Q4 proxy null generation | sweep_v4/scotus_qwen4bit_proxy_20260501_045257 | 3060 | none | none |
| Curated frame contrast probe | sweep_v4/scotus_frame_contrast_probe_20260430_235745 | 80 | none | none |
| Article III frame causal pilot | sweep_v4/scotus_sae_poke_20260501_000146 | 136 | all | 0.02, 0.05, 0.1 |
| Fourth Amendment frame causal pilot | sweep_v4/scotus_sae_poke_20260501_001257 | 136 | all | 0.02, 0.05, 0.1 |
| Frame metric audit | reports/scotus_frame_metric_audit_20260501.md | n/a | n/a | n/a |
| Source frame seed | reports/scotus_source_frame_seed_v1.md | 123 | n/a | n/a |
| Source frame probe | sweep_v4/scotus_source_frame_probe_20260501_003632 | 197 | none | none |
| Expanded Article III source pack | reports/scotus_article3_source_pack_v1.md | 224 | n/a | n/a |
| Expanded Article III cue-masked source probe | sweep_v4/scotus_source_frame_probe_20260501_005417 | 262 | none | none |
| Article III dominance review queue | reports/scotus_article3_dominance_review_v1.md | 80 | n/a | n/a |
| Article III dominance adjudication | reports/scotus_article3_dominance_adjudication_v1.md | 80 | n/a | n/a |
| Article III reviewed-label cue-masked probe | sweep_v4/scotus_source_frame_probe_20260501_010535 | 140 | none | none |
| Expanded Fourth Amendment source pack | reports/scotus_fourth_source_pack_v1.md | 288 | n/a | n/a |
| Fourth Amendment cue-masked source probe | sweep_v4/scotus_source_frame_probe_20260501_011324 | 554 | none | none |
| Proposition-level frame rescore | sweep_v4/scotus_frame_prop_rescore_20260501_012850 | 3332 | none | none |
| Economic justice-style BF16 audit | reports/scotus_slice_economic_style_bf16_20260501.md | 778 | none | none |
| Economic justice-style normal component resplits | sweep_v4/scotus_slice_bf16_economic_style_normal_component_resplits_20260501_024704 | 778 | none | none |
| Economic justice-style null/ablation resplits | sweep_v4/scotus_slice_bf16_economic_style_label_shuffle_component_resplits_20260501_025016; sweep_v4/scotus_slice_bf16_economic_style_excerpt_removed_component_resplits_20260501_025835; sweep_v4/scotus_slice_bf16_economic_style_neutral_filler_component_resplits_20260501_030511 | 778 | none | none |
| Economic justice-style prompt-invariance resplits | sweep_v4/scotus_slice_bf16_economic_style_template_variant_component_resplits_20260501_032205; sweep_v4/scotus_slice_bf16_economic_style_plain_prompt_component_resplits_20260501_032744 | 778 | none | none |
| Economic justice-style split component review | reports/scotus_economic_split_component_review_20260501.md | n/a | n/a | n/a |
| Majority-2000s feasible-issues audit | reports/scotus_slice_majority2000s_feasible_issues_20260501.md | 1154 | none | none |
| Majority-2000s feasible-issues causal pilot | reports/scotus_majority2000s_feasible_issues_causal_pilot_20260501.md | 636 | last/all | 0.01, 0.02, 0.05, 0.1 |
| Economic clean broad-vs-limits cached rescore | sweep_v4/scotus_economic_clean_broad_limits_cached_20260501/report.md | 51 | none | none |
| Economic pocket dominance review queue | reports/scotus_economic_pocket_followup_20260501.md | 51 | none | none |
| Economic dominance adjudication | reports/scotus_economic_pocket_dominance_adjudication_20260501.md | 51 | none | none |
| Economic reviewed-label cached probe | sweep_v4/scotus_economic_reviewed_broad_limits_cached_20260501/report.md | 49 | none | none |
| Federalism source pack | reports/scotus_federalism_source_pack_v1.md | 144 | none | none |
| Federalism source text gate | reports/scotus_federalism_source_gate_20260501.md | 121 | none | none |
| Due Process source pack | reports/scotus_due_process_source_pack_v1.md | 144 | none | none |
| Due Process source text gate | reports/scotus_due_process_source_gate_20260501.md | 114 | none | none |
| Administrative Law source pack | reports/scotus_admin_source_pack_v1.md | 138 | none | none |
| Administrative Law source text gate | reports/scotus_admin_source_gate_20260501.md | 127 | none | none |
| Off-domain poke smoke | reports/scotus_offdomain_poke_20260501.md | 114 | last/all | 0.01, 0.02, 0.05 |
| Commerce-pocket targeted poke | reports/scotus_commerce_pocket_poke_20260501.md | 504 | last/all | 0.01, 0.02, 0.05 |

## Hook Pilot Readout

Rows below are sorted by absolute prompt-matched z. The best all-position row is still small and does not replicate as alpha increases.

| Pilot | Candidate | Alpha | N | Matched delta | Random residual SD | Z | Percentile | Prompt win rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| last | averaged_excerpt_mean_L16_C0p003_negative_scalia | 0.05 | 20 | 0.45 | 0.83 | 0.54 | 0.85 | 0.40 |
| last | averaged_prompt_mean_L16_C0p003 | 0.05 | 20 | 0.35 | 0.83 | 0.42 | 0.81 | 0.35 |
| last | averaged_excerpt_mean_L16_C0p003 | 0.20 | 20 | -0.24 | 0.96 | -0.25 | 0.35 | 0.25 |
| last | averaged_prompt_mean_L16_C0p003_negative_scalia | 0.05 | 20 | -0.20 | 0.83 | -0.25 | 0.20 | 0.20 |
| last | averaged_prompt_mean_L16_C0p003_negative_scalia | 0.10 | 20 | -0.18 | 0.83 | -0.22 | 0.39 | 0.25 |
| last | averaged_excerpt_mean_L16_C0p003 | 0.05 | 20 | 0.15 | 0.83 | 0.18 | 0.72 | 0.25 |
| last | averaged_prompt_mean_L16_C0p003_negative_scalia | 0.20 | 20 | 0.16 | 0.96 | 0.16 | 0.58 | 0.45 |
| last | averaged_prompt_mean_L16_C0p003 | 0.10 | 20 | 0.11 | 0.83 | 0.14 | 0.74 | 0.40 |
| all | averaged_prompt_mean_L16_C0p003_negative_scalia | 0.01 | 12 | 0.70 | 0.96 | 0.73 | 0.93 | 0.50 |
| all | averaged_excerpt_mean_L16_C0p003 | 0.05 | 12 | -0.60 | 1.11 | -0.54 | 0.20 | 0.08 |
| all | averaged_prompt_mean_L16_C0p003 | 0.05 | 12 | -0.60 | 1.11 | -0.54 | 0.20 | 0.08 |
| all | averaged_prompt_mean_L16_C0p003_negative_scalia | 0.02 | 12 | 0.58 | 1.14 | 0.51 | 0.87 | 0.42 |
| all | averaged_excerpt_mean_L16_C0p003 | 0.02 | 12 | -0.50 | 1.14 | -0.44 | 0.15 | 0.08 |
| all | averaged_excerpt_mean_L16_C0p003_negative_scalia | 0.02 | 12 | 0.50 | 1.14 | 0.44 | 0.87 | 0.42 |
| all | averaged_excerpt_mean_L16_C0p003_negative_scalia | 0.05 | 12 | -0.35 | 1.11 | -0.32 | 0.25 | 0.25 |
| all | averaged_prompt_mean_L16_C0p003 | 0.02 | 12 | -0.33 | 1.14 | -0.29 | 0.30 | 0.17 |

## Proxy Null Volatility

Highest-variance prompt-condition rows are poor substrates for steering claims until the rubric is improved.

| Prompt | Condition | Issue | Mean delta | SD | P05 | P95 |
| --- | --- | --- | --- | --- | --- | --- |
| JP03_agency_civil_penalty | concise_judicial | Judicial Power | -4.46 | 2.82 | -8.0 | 1.0 |
| JP01_agency_private_company | concise_judicial | Judicial Power | 0.52 | 2.82 | -4.0 | 6.0 |
| JP03_agency_civil_penalty | bench_memo | Judicial Power | 1.04 | 2.73 | -2.0 | 7.0 |
| JP03_agency_civil_penalty | majority_reasoning | Judicial Power | -1.48 | 2.72 | -6.0 | 3.0 |
| JP01_agency_private_company | bench_memo | Judicial Power | -1.88 | 2.45 | -5.0 | 3.0 |
| EA01_commercial_remedy | bench_memo | Economic Activity | 2.54 | 2.31 | -2.0 | 5.0 |
| CR03_racial_redistricting | concise_judicial | Civil Rights | 1.50 | 2.10 | -2.0 | 5.0 |
| JP02_bankruptcy_counterclaim | concise_judicial | Judicial Power | 0.08 | 2.03 | -3.0 | 3.0 |
| JP02_bankruptcy_counterclaim | majority_reasoning | Judicial Power | 0.04 | 2.00 | -4.0 | 3.0 |
| CR03_racial_redistricting | bench_memo | Civil Rights | -4.28 | 1.95 | -7.0 | -1.0 |

Most stable prompt-condition rows are better candidates for the next issue-specific pilot.

| Prompt | Condition | Issue | Mean delta | SD | P05 | P95 |
| --- | --- | --- | --- | --- | --- | --- |
| CR02_sex_military_academy | bench_memo | Civil Rights | 0.24 | 0.69 | -1.0 | 1.0 |
| EA03_gun_school_zone | bench_memo | Economic Activity | 0.46 | 0.76 | 0.0 | 2.0 |
| CR04_state_disability_access | majority_reasoning | Civil Rights | -0.48 | 0.76 | -2.0 | 1.0 |
| CR02_sex_military_academy | majority_reasoning | Civil Rights | -0.02 | 0.89 | -1.0 | 2.0 |
| EA04_arbitration_preemption | bench_memo | Economic Activity | 0.48 | 0.91 | -1.0 | 2.0 |
| JP04_immigration_removal | bench_memo | Judicial Power | -0.30 | 0.91 | -1.0 | 1.0 |
| JP04_immigration_removal | concise_judicial | Judicial Power | 0.40 | 0.93 | -1.0 | 2.0 |
| CP04_home_emergency_entry | majority_reasoning | Criminal Procedure | 0.26 | 0.99 | -1.0 | 2.0 |
| FD02_major_questions_agency | majority_reasoning | Administrative Law | -0.50 | 1.02 | -2.0 | 1.0 |

## Rubric Contamination

Off-domain frame tags identify places where keyword scoring is too coarse or the prompt naturally invokes neighboring doctrine.

| Prompt | Condition | Issue | N | Off-domain rate | Mean off-domain hits |
| --- | --- | --- | --- | --- | --- |
| JP04_immigration_removal | bench_memo | Judicial Power | 50 | 1.00 | 4.46 |
| JP04_immigration_removal | concise_judicial | Judicial Power | 50 | 0.98 | 2.34 |
| JP04_immigration_removal | majority_reasoning | Judicial Power | 50 | 0.96 | 2.30 |
| FD01_state_sheriff_checks | concise_judicial | Federalism | 50 | 0.96 | 1.20 |
| EA02_homegrown_market_regulation | majority_reasoning | Economic Activity | 50 | 0.92 | 1.26 |
| CR04_state_disability_access | majority_reasoning | Civil Rights | 50 | 0.90 | 2.18 |
| CR04_state_disability_access | bench_memo | Civil Rights | 50 | 0.90 | 1.50 |
| EA01_commercial_remedy | bench_memo | Economic Activity | 50 | 0.88 | 1.46 |
| EA01_commercial_remedy | majority_reasoning | Economic Activity | 50 | 0.88 | 1.30 |
| FD01_state_sheriff_checks | majority_reasoning | Federalism | 50 | 0.86 | 1.14 |

| Off-domain frame | Count |
| --- | --- |
| separation_presidential_power | 332 |
| article3_private_rights | 211 |
| article3_public_rights | 170 |
| fourth_home_exigency | 92 |
| economic_remedy_damages | 71 |
| article3_article1_tribunal | 70 |
| due_process_substantive | 69 |
| fourth_exigency_consent | 67 |
| due_process_procedural_mathews | 37 |
| economic_commerce_clause | 17 |

## Issue-Specific Candidate Starts

Low issue-conditioned SAE overlap rows are better next candidates than broad justice-level averages. They are not steering evidence by themselves; they nominate where to build narrower frame candidates.

| Issue | Pair | Region | Layer | Top-J | Weighted-J | Cosine | N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Judicial Power | Ginsburg / Thomas | excerpt_mean | 16 | 0.449 | 0.543 | 0.927 | 20 / 20 |
| Criminal Procedure | Scalia / Ginsburg | excerpt_mean | 16 | 0.504 | 0.550 | 0.913 | 20 / 20 |
| Criminal Procedure | Scalia / Thomas | excerpt_mean | 16 | 0.504 | 0.552 | 0.923 | 20 / 20 |
| Judicial Power | Scalia / Ginsburg | excerpt_mean | 16 | 0.471 | 0.554 | 0.934 | 20 / 20 |
| Judicial Power | Ginsburg / Souter | excerpt_mean | 16 | 0.504 | 0.566 | 0.937 | 20 / 20 |
| Judicial Power | Ginsburg / Souter | prompt_mean | 16 | 0.449 | 0.572 | 0.923 | 20 / 20 |
| Judicial Power | Ginsburg / Thomas | prompt_mean | 8 | 0.449 | 0.574 | 0.926 | 20 / 20 |
| Judicial Power | Ginsburg / Thomas | excerpt_mean | 12 | 0.481 | 0.578 | 0.936 | 20 / 20 |
| Judicial Power | Scalia / Ginsburg | excerpt_mean | 12 | 0.527 | 0.578 | 0.935 | 20 / 20 |
| Judicial Power | Ginsburg / Thomas | prompt_mean | 16 | 0.408 | 0.582 | 0.944 | 20 / 20 |
| Criminal Procedure | Scalia / Ginsburg | excerpt_mean | 12 | 0.493 | 0.583 | 0.937 | 20 / 20 |
| Judicial Power | Ginsburg / Thomas | excerpt_mean | 8 | 0.538 | 0.588 | 0.929 | 20 / 20 |

## Frame Label Feasibility

Direct keyword labels in the repaired opinion chunks are too sparse for several desired frame probes. This argues for curated frame-labeled excerpts or contrastive prompt capture before training frame-specific directions.

| Issue | Frame | Split | Positive | Negative |
| --- | --- | --- | --- | --- |
| Judicial Power | article3_public_rights | train | 2 | 1858 |
| Judicial Power | article3_public_rights | dev | 0 | 176 |
| Judicial Power | article3_public_rights | test | 0 | 124 |
| Judicial Power | article3_private_rights | train | 0 | 1860 |
| Judicial Power | article3_private_rights | dev | 0 | 176 |
| Judicial Power | article3_private_rights | test | 0 | 124 |
| Judicial Power | article3_article1_tribunal | train | 137 | 1723 |
| Judicial Power | article3_article1_tribunal | dev | 22 | 154 |
| Judicial Power | article3_article1_tribunal | test | 18 | 106 |
| Judicial Power | article3_case_or_controversy | train | 314 | 1546 |
| Judicial Power | article3_case_or_controversy | dev | 66 | 110 |
| Judicial Power | article3_case_or_controversy | test | 12 | 112 |
| Criminal Procedure | fourth_search_incident_chimel | train | 14 | 2606 |
| Criminal Procedure | fourth_search_incident_chimel | dev | 0 | 152 |
| Criminal Procedure | fourth_search_incident_chimel | test | 0 | 492 |
| Criminal Procedure | fourth_digital_privacy | train | 0 | 2620 |
| Criminal Procedure | fourth_digital_privacy | dev | 0 | 152 |
| Criminal Procedure | fourth_digital_privacy | test | 0 | 492 |
| Criminal Procedure | fourth_plain_view_closed_container | train | 2 | 2618 |
| Criminal Procedure | fourth_plain_view_closed_container | dev | 0 | 152 |
| Criminal Procedure | fourth_plain_view_closed_container | test | 0 | 492 |
| Criminal Procedure | fourth_home_exigency | train | 107 | 2513 |
| Criminal Procedure | fourth_home_exigency | dev | 2 | 150 |
| Criminal Procedure | fourth_home_exigency | test | 16 | 476 |
| Criminal Procedure | fourth_stop_reasonable_suspicion | train | 14 | 2606 |
| Criminal Procedure | fourth_stop_reasonable_suspicion | dev | 0 | 152 |
| Criminal Procedure | fourth_stop_reasonable_suspicion | test | 10 | 482 |

## Frame-Contrast Follow-up

Because direct keyword labels were too sparse in the repaired opinion chunks, a curated contrastive probe was built as a fast candidate generator. This is a diagnostic branch, not source-opinion evidence.

| Task | Best readout | Dev BA | Test BA | Text test BA | Direction |
| --- | --- | ---: | ---: | ---: | --- |
| article3_private_vs_public | prompt_mean @ L16, C=0.003 | 1.000 | 1.000 | 1.000 | sweep_v4/scotus_frame_contrast_probe_20260430_235745/article3_private_vs_public/direction.npz |
| fourth_digital_vs_incident | prompt_mean @ L16, C=0.003 | 1.000 | 1.000 | 1.000 | sweep_v4/scotus_frame_contrast_probe_20260430_235745/fourth_digital_vs_incident/direction.npz |

The perfect text baseline is the key caution: these examples intentionally contain frame-bearing language, so this result only nominates directions for causal testing.

The two small causal pilots used Qwen3.5 BF16 hooks, `position=all`, four neutral prompts per frame family, three hidden-norm-fraction alphas, and ten same-layer random controls per alpha.

| Direction | Run | Best matched target result | Best matched net result | Decision |
| --- | --- | --- | --- | --- |
| Article III private-rights over public-rights | sweep_v4/scotus_sae_poke_20260501_000146 | alpha 0.10: z=-0.02, win rate 0.25 | alpha 0.10: z=0.42, win rate 0.75 | not promoted |
| Fourth Amendment digital privacy over search incident | sweep_v4/scotus_sae_poke_20260501_001257 | alpha 0.05: z=0.93, win rate 0.50 | alpha 0.05: z=0.55, win rate 0.75 | not promoted |

Read:

1. Article III target-frame movement did not beat prompt-matched random controls. The positive net row is too small and too sparse to rely on.
2. Fourth Amendment has a weak alpha `0.05` hint, but alpha `0.10` reverses badly and increases contrast/safety/search-incident wording.
3. Cue-heavy frame directions are therefore not enough. The next frame candidates should come from source-grounded or manually adjudicated excerpts, and the evaluator needs blind-review calibration before more large steering runs.

## Source-Frame Follow-up

A strict source-grounded frame seed was built from real v2.1 SCOTUS chunks:

- Builder: `scripts/experiments/scotus/build_source_frame_labels.py`
- Labels: `data/scotus/scotus_source_frame_labels_v1.jsonl`
- Review queue: `data/scotus/scotus_source_frame_review_queue_v1.jsonl`
- Report: `reports/scotus_source_frame_seed_v1.md`
- Probe: `sweep_v4/scotus_source_frame_probe_20260501_003632/report.md`

Strict source labeling found no valid support for `article3_public_rights` or `article3_private_rights` in the current target-justice chunks after false positives were excluded. That confirms the earlier feasibility warning: the public/private-rights branch needs expanded source collection or manual labels from additional opinions before it can be a serious candidate.

| Source task | Best readout | Dev BA | Test BA | Text test BA | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| article3_article1_vs_case | prompt_mean @ L16 | 1.000 | 0.500 | 0.500 | not promoted |
| article3_finality_vs_case | prompt_mean @ L12 | 0.571 | 0.429 | 0.429 | reject |
| fourth_home_vs_incident | prompt_last @ L16 | 0.750 | 0.500 | 1.000 | text/leakage dominated |
| fourth_plain_view_vs_incident | prompt_mean @ L8 | 1.000 | 0.143 | 0.357 | reject |
| fourth_technology_vs_incident | prompt_mean @ L12 | 0.833 | 0.500 | 1.000 | text/leakage dominated |

## Expanded Article III Source Pack

The target-justice corpus gap was followed by a separate source-pack branch over named Article III/public-rights opinions from Cornell LII. This creates a reviewable source queue, not final labels.

- Builder: `scripts/experiments/scotus/build_article3_source_pack.py`
- Labels: `data/scotus/scotus_article3_source_frame_labels_v1.jsonl`
- Review queue: `data/scotus/scotus_article3_source_frame_review_queue_v1.jsonl`
- Source-pack report: `reports/scotus_article3_source_pack_v1.md`
- Cue-masked probe: `sweep_v4/scotus_source_frame_probe_20260501_005417/report.md`
- Dominance review report: `reports/scotus_article3_dominance_review_v1.md`

Source-pack label counts:

| Frame | Total | Train | Dev | Test | Public/private conflicts |
| --- | ---: | ---: | ---: | ---: | ---: |
| article3_public_rights | 72 | 38 | 27 | 7 | 16 |
| article3_private_rights | 39 | 28 | 3 | 8 | 28 |
| article3_article1_tribunal | 72 | 45 | 17 | 10 | 4 |
| article3_case_or_controversy | 11 | 6 | 4 | 1 | 2 |
| article3_final_judgment_separation | 30 | 27 | 2 | 1 | 0 |

Cue-masked Qwen3.5 probe read:

| Source task | Best readout | Dev BA | Test BA | Text test BA | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| article3_public_vs_private | prompt_last @ L16 | 0.625 | 0.500 | 0.500 | reject |
| article3_public_vs_article1 | prompt_mean @ L8 | 1.000 | 0.969 | 0.969 | text/leakage dominated |
| article3_private_vs_article1 | prompt_mean @ L16 | 0.938 | 0.333 | 0.750 | not promoted |

Read: expanded source collection solved the zero-label problem but did not solve the candidate problem. The public/private contrast is chance after cue masking and conflict filtering; the public-vs-Article-I result is essentially matched by the text baseline; the private-vs-Article-I row is unstable and too sparse. Do not use these directions for a causal poke without manual dominance review and a better frame evaluator.

A blind dominance-review queue now exists:

| Queue | Path | Rows | Notes |
| --- | --- | ---: | --- |
| Blind | data/scotus/scotus_article3_dominance_review_blind_v1.jsonl | 80 | cue-masked excerpts, no rule labels |
| Key | data/scotus/scotus_article3_dominance_review_key_v1.jsonl | 80 | matched frames/evidence for audit after review |

The queue is intentionally framed around dominant legal reasoning (`public_rights_dominant`, `private_rights_dominant`, `article1_tribunal_dominant`, `mixed_comparative`, or `off_target_or_false_positive`) because keyword presence is the failure mode.

The dominance queue was also filled as a single-pass adjudication:

- Script: `scripts/experiments/scotus/apply_article3_dominance_adjudication.py`
- Reviewed queue: `data/scotus/scotus_article3_dominance_review_adjudicated_v1.jsonl`
- Probe-ready labels: `data/scotus/scotus_article3_dominance_frame_labels_v1.jsonl`
- Adjudication report: `reports/scotus_article3_dominance_adjudication_v1.md`
- Reviewed-label probe: `sweep_v4/scotus_source_frame_probe_20260501_010535/report.md`

| Review label | Rows |
| --- | ---: |
| public_rights_dominant | 33 |
| private_rights_dominant | 28 |
| article1_tribunal_dominant | 9 |
| mixed_comparative | 10 |

Reviewed-label cue-masked Qwen3.5 probe read:

| Source task | Best readout | Dev BA | Test BA | Text test BA | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| article3_public_vs_private | prompt_mean @ L8 | 0.675 | 0.500 | 0.500 | reject |
| article3_public_vs_article1 | prompt_mean @ L16 | 1.000 | 0.429 | 0.929 | text/leakage dominated |
| article3_private_vs_article1 | prompt_mean @ L16 | 0.500 | 0.500 | 0.500 | reject |

Read: the expanded Article III branch is now negative under the current protocol. Even after dominance review and cue masking, public/private held-out performance is chance and the Article-I diagnostics are too sparse or lexical. Do not promote or causally poke these directions unless a second review materially changes the labels.

## Expanded Fourth Amendment Source Pack

The Fourth Amendment source branch was rebuilt from named Fourth Amendment cases after the first source seed included non-Fourth technology false positives. The rebuilt pack is cleaner but still does not produce a promotable direction.

| Artifact | Path | Rows |
| --- | --- | ---: |
| Source pack report | reports/scotus_fourth_source_pack_v1.md | 288 |
| Raw source pages | data/scotus/raw/scotus_fourth_source_pages_v1.json | 18 source opinions |
| Labels | data/scotus/scotus_fourth_source_frame_labels_v1.jsonl | 288 |
| Review queue | data/scotus/scotus_fourth_source_frame_review_queue_v1.jsonl | 288 |
| Cue-masked probe | sweep_v4/scotus_source_frame_probe_20260501_011324/report.md | 554 probe examples |

Cue-masked Qwen3.5 probe read:

| Source task | Best readout | Dev BA | Test BA | Text test BA | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| fourth_home_vs_incident | excerpt_mean @ L8 | 0.952 | 1.000 | 1.000 | text/leakage dominated; only 3 test rows |
| fourth_plain_view_vs_incident | prompt_mean @ L8 | 0.977 | 0.838 | 0.809 | not promoted; marginal over text baseline and split-skewed |
| fourth_technology_vs_home | prompt_mean @ L16 | 1.000 | 1.000 | 0.988 | text/leakage dominated |
| fourth_technology_vs_incident | prompt_mean @ L16 | 1.000 | 1.000 | 1.000 | text/leakage dominated |

Read: the rebuilt Fourth source pack is useful for evaluator and leakage diagnostics, but it is not evidence for a steerable judicial circuit. Do not promote these directions or spend BF16 hook time on them unless a dominance-reviewed relabeling produces non-text-baseline held-out support.

## Proposition-Level Evaluator Repair

The keyword metric was repaired with a stricter proposition-level rescore over the Q4 proxy run and both BF16 frame-poke pilots.

| Artifact | Path | Rows |
| --- | --- | ---: |
| Rescore script | scripts/experiments/scotus/rescore_scotus_frame_propositions.py | n/a |
| Rescore report | sweep_v4/scotus_frame_prop_rescore_20260501_012850/report.md | 3332 |
| Disagreement queue | sweep_v4/scotus_frame_prop_rescore_20260501_012850/disagreement_review_queue.jsonl | 200 |

Largest corrected false positives:

| Frame | Old rows | Proposition rows | Dropped rows |
| --- | ---: | ---: | ---: |
| separation_presidential_power | 367 | 1 | 366 |
| civil_equal_protection_strict_scrutiny | 595 | 278 | 317 |
| article3_article1_tribunal | 807 | 502 | 308 |
| article3_private_rights | 459 | 275 | 239 |
| article3_public_rights | 427 | 364 | 180 |
| fourth_home_exigency | 240 | 115 | 132 |

BF16 pilot read under proposition scoring:

| Pilot | Alpha | N | Target z | Target win | Net z | Net win | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Article III | 0.02 | 4 | -0.527 | 0.000 | -0.741 | 0.000 | reject |
| Article III | 0.05 | 4 | 0.757 | 0.500 | 0.511 | 0.500 | not promoted |
| Article III | 0.10 | 4 | 0.027 | 0.500 | 0.222 | 0.750 | not promoted |
| Fourth Amendment | 0.02 | 4 | -0.323 | 0.000 | -0.485 | 0.000 | reject |
| Fourth Amendment | 0.05 | 4 | 1.146 | 0.500 | 0.880 | 0.500 | weak hint only |
| Fourth Amendment | 0.10 | 4 | -0.826 | 0.000 | -1.190 | 0.000 | reject |

Read: the rescore confirms the metric repair direction, but not a steering result. The Fourth alpha `0.05` hint remains too small and unstable, and the Article III rows stay below promotion threshold.

## Economic Activity Source Pack

Economic Activity was the next issue-family branch nominated by the corrected proposition-level triage. The source pack was expanded to `31` named SCOTUS cases across broad Commerce Clause, Commerce Clause limits, federalism/state-regulation, and statutory/preemption frames.

| Artifact | Path |
| --- | --- |
| Source-pack builder | scripts/experiments/scotus/build_economic_source_pack.py |
| Source-pack report | reports/scotus_economic_source_pack_v1.md |
| Probe summary | reports/scotus_economic_source_probe_20260501.md |
| BF16 probe run | sweep_v4/scotus_source_frame_probe_20260501_014711/report.md |

The source-probe splitter was repaired before the run: `--reassign-task-splits` now assigns one split per source cluster within each task, avoiding the prior case-identity leak where the same case could appear in different splits across labels.

Cue-masked BF16 probe read:

| Task | Best readout | Dev BA | Test BA | Text test BA | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| economic_broad_vs_limits | prompt_last @ L16 | 0.733 | 0.621 | 0.641 | reject; activation does not beat text |
| economic_broad_vs_state | prompt_mean @ L12 | 0.875 | 1.000 | 0.969 | text/leakage dominated |
| economic_limits_vs_state | prompt_mean @ L16 | 0.901 | 1.000 | 0.950 | text/leakage dominated |
| economic_preemption_vs_broad | prompt_mean @ L12 | 1.000 | 1.000 | 1.000 | text/leakage dominated |

Read: Economic Activity does not nominate a steering vector. The primary broad Commerce Clause versus limits contrast is below the text baseline, and the other comparisons are mostly solved by text after cue masking.

## Civil Rights Source-Pack Gate

Civil Rights was the backup issue-family branch, so it was checked with a source pack and a cheap cue-masked text-only gate before spending BF16 hook time.

| Artifact | Path |
| --- | --- |
| Source-pack builder | scripts/experiments/scotus/build_civil_source_pack.py |
| Source-pack report | reports/scotus_civil_source_pack_v1.md |
| Gate report | reports/scotus_civil_source_gate_20260501.md |

Text-only gate:

| Task | Dev BA | Test BA | Decision |
| --- | ---: | ---: | --- |
| civil_intermediate_vs_section5 | 0.971 | 1.000 | text dominated |
| civil_rational_vs_strict | 0.955 | 1.000 | text dominated |
| civil_strict_vs_intermediate | 0.748 | 0.969 | text dominated |
| civil_strict_vs_section5 | 1.000 | 0.964 | text dominated |

Read: do not run the BF16 Civil Rights activation probe. The cue-masked text baseline already solves the proposed contrasts, so this branch is leakage/evaluator material unless a less lexical subdoctrine is defined.

## Late Source-Pack Gates

After Economic Activity and Civil Rights failed promotion gates, the remaining ranked issue families were checked with cheap source-pack/text gates before any more BF16 hook work.

| Branch | Source pack | Gate report | Examples | Dev BA | Test BA | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Federalism anti-commandeering vs preemption | reports/scotus_federalism_source_pack_v1.md | reports/scotus_federalism_source_gate_20260501.md | 121 | 0.925 | 1.000 | text dominated |
| Due Process substantive vs procedural | reports/scotus_due_process_source_pack_v1.md | reports/scotus_due_process_source_gate_20260501.md | 114 | 0.761 | 1.000 | text dominated |
| Administrative Law major-questions vs ordinary deference | reports/scotus_admin_source_pack_v1.md | reports/scotus_admin_source_gate_20260501.md | 127 | 0.586 | 1.000 | not promoted; final test text-saturated and case-skewed |

Read: these branches should not receive activation capture under the current source-frame design. The repeated pattern is that named-doctrine source contrasts are too easy for residual text/case identity even after cue masking, or too split-skewed to support a causal claim.

## Justice-Style Slice BF16 Check

The cached Phase 4.1 Qwen3.6 FP8 feature bank was mined for justice-style slices after the source-frame branches failed. The top mined slice was `section_posture=majority__decade=2000s`, which had cached activation test BA `0.809` versus text test BA `0.500`.

Same-model Qwen3.5 BF16 verification:

| Artifact | Path |
| --- | --- |
| Slice-mining report | reports/scotus_slice_candidate_mining_20260501.md |
| BF16 verification report | reports/scotus_slice_bf16_majority2000s_20260501.md |
| BF16 run | sweep_v4/scotus_slice_bf16_majority2000s_normal_20260501_022109/report.md |
| Label-shuffle null | sweep_v4/scotus_slice_bf16_majority2000s_label_shuffle_20260501_022912/report.md |

| Slice | Best readout | Dev BA | Test BA | Text test BA | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `section_posture=majority,decade=2000s` | `excerpt_mean @ L16` | `0.810` | `0.691` | `0.500` | not promoted |

The label-shuffle null stayed near chance: best dev BA `0.603`, test BA `0.515`, with zero sweep configs above `0.70`.

Read: the slice has real activation structure, but it misses the held-out BF16 promotion gate and is issue-fragile. Diagnostic prompt-last rows reached higher test scores, but those are not selection-valid headline results and remain prompt-last/test-picking risks.

## Majority-2000s Feasible-Issues Refinement

The top justice-style slice was refined to issue families with strict case-component train/dev/test feasibility: `Criminal Procedure`, `Economic Activity`, and `Judicial Power`.

| Artifact | Path |
| --- | --- |
| Detailed audit | reports/scotus_slice_majority2000s_feasible_issues_20260501.md |
| Split feasibility audit | reports/scotus_slice_majority2000s_feasible_issues_split_feasibility_20260501.md |
| Normal component resplits | sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_normal_component_resplits_20260501_034116/report.md |
| Label-shuffle component resplits | sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_label_shuffle_component_resplits_20260501_034539/report.md |
| Template-variant component resplits | sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_template_variant_component_resplits_20260501_040538/report.md |
| Plain-prompt component resplits | sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_plain_prompt_component_resplits_20260501_041503/report.md |
| Excerpt-removed component resplits | sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_excerpt_removed_component_resplits_20260501_042048/report.md |
| Neutral-filler component resplits | sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_neutral_filler_component_resplits_20260501_043234/report.md |

| Diagnostic | Median dev BA | Median test BA | Test BA range | Primary split test BA | Median text test BA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal | `0.812` | `0.746` | `0.660-0.753` | `0.753` | `0.700` |
| Label shuffle | `0.536` | `0.541` | `0.477-0.548` | `0.488` | `0.492` |
| Template variant | `0.807` | `0.758` | `0.668-0.807` | `0.777` | `0.695` |
| Plain prompt | `0.818` | `0.764` | `0.676-0.796` | `0.777` | `0.691` |
| Excerpt removed | `0.500` | `0.500` | `0.500-0.500` | `0.500` | `0.500` |
| Neutral filler | `0.575` | `0.542` | `0.512-0.564` | `0.548` | `0.564` |

Read: this is the strongest justice-style decodability branch in this phase. It is source-grounded, survives prompt-template/plain-prompt recapture, and collapses under excerpt removal and neutral filler. It is not by itself a steerable circuit.

Two causal pilots then tested the primary readout families:

| Direction | Run | Position | Alphas | Random controls | Best prompt-matched target z | Best prompt-matched net z | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `prompt_last @ L10` | `sweep_v4/scotus_sae_poke_20260501_045156` | `last` | `0.02,0.05,0.1` | `10` | `0.184` | `0.533` | not promoted |
| `excerpt_mean @ L16` | `sweep_v4/scotus_sae_poke_20260501_060425` | `all` | `0.01,0.02,0.05` | `5` | `0.449` | `0.395` | not promoted |

Read: the broad positive Ginsburg directions did not cause reliable jurisprudential-frame movement beyond prompt-matched random controls.

## Off-Domain Smoke

The surviving directions were also tested on six ordinary nonlegal prompts to check whether they behave like a portable reasoning-style vector.

| Run | Directions | Position | Alphas | Random controls | Result |
| --- | --- | --- | --- | ---: | --- |
| `sweep_v4/scotus_sae_poke_20260501_072906` | Phase 4 `prompt_last @ L4`; majority-2000s split-00 `prompt_last @ L10` | `last` | `0.02,0.05` | 2 | no visible direction-specific drift |
| `sweep_v4/scotus_sae_poke_20260501_073923` | majority-2000s split-01 `excerpt_mean @ L16` | `all` | `0.01,0.02,0.05` | 2 | mild structured-answer shifts, matched by random controls |

Read: no legalistic, judicial, Scalia-like, or Ginsburg-like drift appeared on weather, video-game, friend-conflict, homework, team-selection, or headphone prompts. This does not disprove a feature-level effect, but it lowers confidence that the current broad directions are portable reasoning-temperament controls.

## Commerce-Pocket Follow-up

The only reviewed causal prompt pockets that survived the pairwise rule were expanded into a targeted Commerce Clause prompt bank:

- Prompt bank: `data/scotus/scotus_commerce_pocket_prompts_v1.jsonl`
- Analyzer: `scripts/experiments/scotus/analyze_commerce_pocket_poke.py`
- Summary report: `reports/scotus_commerce_pocket_poke_20260501.md`
- `split_00__last` run: `sweep_v4/scotus_sae_poke_20260501_075401`
- `split_01__all` run: `sweep_v4/scotus_sae_poke_20260501_091316`

| Run | Scope | Position | Key alpha | Matched target | Matched net | Prompt win rate | Strongest-random family win | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `split_00__last` | 12 Commerce prompts | `last` | `0.02` | `0.115` | `0.021` | `0.42 target / 0.50 net` | `0.00` | not promoted |
| `split_01__all` | 6 authority/remedy prompts | `all` | `0.02` | `-0.479` | `-0.479` | `0.00 target / 0.00 net` | `0.00` | reject |

Two isolated rows beat strongest random controls by keyword metrics, but manual read does not support promotion. `EA_LIMIT_04_home_arson_private_dwelling` at alpha `0.01` shifted toward state-police-power and jurisdictional-hook language, but it did not replicate across limits prompts or at alpha `0.02`. `EA_AUTH_04_credit_reporting_remedy` at alpha `0.05` emphasized statutory damages, but the direction was negative at alpha `0.02` and did not generalize across authority/remedy prompts.

Read: the Commerce-pocket branch falsifies the last surviving prompt-pocket hypothesis. It is not a steerable judicial circuit.

## Minimal-Pair Replay Follow-up

After the prompt-pocket branch failed, the next candidate was rebuilt as controlled Commerce minimal pairs: each fact pattern had both a Commerce-authority answer and a Commerce-limits answer, so prompt-only text could not identify the label.

| Artifact | Path |
| --- | --- |
| Minimal-pair probe report | reports/scotus_minimal_pair_replay_20260501.md |
| Probe run | sweep_v4/scotus_minpair_replay_20260501_100514 |
| Causal poke run | sweep_v4/scotus_sae_poke_20260501_100830 |
| Causal analyzer report | reports/scotus_minpair_replay_causal_poke_20260501.md |
| Template leakage audit | reports/scotus_minpair_template_leakage_audit_20260501.md |
| SAE feature inspection | reports/scotus_minpair_sae_feature_inspection_20260501.md |
| Late residual L16 poke | sweep_v4/scotus_sae_poke_20260501_114400 |
| Late residual L20 poke | sweep_v4/scotus_sae_poke_20260501_120824 |
| Late residual analyzer | reports/scotus_minpair_late_residual_pokes_20260501.md |
| Prototype replacement poke | sweep_v4/scotus_prototype_patch_20260501_123725 |
| Prototype replacement analyzer | reports/scotus_minpair_prototype_patch_20260501.md |

Activation read:

| Readout | Layer | C | Dev BA | Test BA | Prompt-only text BA | Assistant-text BA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assistant_all` | 4 | 0.001 | 1.000 | 1.000 | 0.500 | 1.000 |

Causal read:

| Alpha | Matched target | Matched net | Prompt target win | Prompt net win | Strongest-random target win | Strongest-random net win | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.005 | 0.167 | 0.500 | 0.33 | 0.67 | 0.00 | 0.00 | not promoted |
| 0.01 | 0.639 | 0.583 | 0.50 | 0.67 | 0.17 | 0.17 | suggestive only |
| 0.02 | -0.861 | -0.889 | 0.00 | 0.00 | 0.00 | 0.00 | reject |
| 0.05 | -0.472 | -0.611 | 0.00 | 0.17 | 0.00 | 0.00 | reject |

Read: this is useful negative evidence. The design removed prompt leakage and proved the model carries a separable assistant-internal answer frame, but the exported linear direction did not causally steer neutral generation beyond same-layer random controls. The current Commerce minimal-pair vector is not a steerable judicial circuit.

Template audit correction:

- The replay bank contains only `6` unique assistant completions across `48` rows.
- Every exact answer template appears in train/dev/test, so the original split can overstate generalization.
- Leave-one-template-pair-out residual probes still found late answer-state structure: `assistant_all @ L16` and `assistant_all @ L20` had mean/min/max BA of `1.000/1.000/1.000`.
- The SAE L0_100 `assistant_all @ L8` probe was only partial under template holdout: mean BA `0.750`, min BA `0.500`.
- The SAE feature inspection shows top features mostly firing on repeated answer templates, so decoder-column feature pokes are lower priority than late residual directions.
- Frozen late residual directions at `assistant_all @ L16` and `assistant_all @ L20` were exported with `C=0.001` into `data/scotus/directions/` and tested against same-layer random controls.
- L16 failed: best matched target `-0.083`, best matched net `-0.125`, strongest-random target/net wins `0.17/0.00`.
- L20 failed: best matched target `-0.167`, best matched net `0.042`, strongest-random target/net wins `0.17/0.17`.
- The one isolated L20 positive row was a statutory-construction wording shift on the home-arson prompt and did not replicate.
- A replacement-style intervention then blended L16+L20 residual states toward the train-split Commerce-limits prototype. Its best point was blend `0.01`, matched target `0.292`, matched net `0.625`, but strongest-random target/net wins were only `0.00/0.17`; higher blends reversed or collapsed.

Decision: the late residual directions and the prototype replacement remain diagnostic answer-state interventions. They are not promoted as steerable judicial circuits.

## Next Step

Do not run more broad justice-level pokes on the current averaged L16 directions, and do not promote the cue-heavy, source-frame, reviewed Article III, rebuilt Fourth Amendment, Economic Activity, Civil Rights, weak proposition-rescored frame-pilot, Commerce-pocket, or Commerce minimal-pair replay directions.

Next, stop broad prompt-bank causal pokes on the refined majority-2000s directions, including off-domain and Commerce-pocket pokes. The late-layer minimal-pair residual exception and the prototype-replacement variant have now failed causal promotion too. The next viable path is either a stronger intervention family than residual act-add/prototype blend, or a training/distillation path that uses the probes as diagnostics, evals, or auxiliary losses.

Prompt-pocket follow-up:

- Report: `reports/scotus_majority2000s_causal_prompt_pockets_20260501.md`
- Blind queue: `data/scotus/scotus_majority2000s_causal_review_blind_20260501.jsonl`
- Key file: `data/scotus/scotus_majority2000s_causal_review_key_20260501.jsonl`

The queue contains `8` candidate cells and `22` pairwise comparisons. It is a review aid, not promotion evidence. A narrower prompt family should only advance if the candidate side wins against both baseline and matched random controls without coherence degradation.

Internal adjudication:

- Report: `reports/scotus_majority2000s_causal_review_adjudication_20260501.md`
- Adjudicated rows: `data/scotus/scotus_majority2000s_causal_review_adjudicated_20260501.jsonl`

The only reviewed pockets that survive are `EA03_gun_school_zone` and `EA01_commercial_remedy`, both Economic Activity / Commerce Clause prompts at alpha `0.02`. Judicial Power pockets do not survive strongest-random comparisons. Next work should target a narrow Commerce Clause limits / federal-remedy contrast, not broad justice-style directions.
