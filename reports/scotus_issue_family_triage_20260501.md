# SCOTUS Issue Family Triage

## Purpose

This ranks issue families before building another source pack or spending BF16 hook time. It uses the corrected proposition-level Q4 proxy rescore, not activation evidence.

## Inputs

- Rescore run: `/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/scotus_frame_prop_rescore_20260501_012850`
- Rows used: Q4 proxy `random_control` and `base` completions only
- Gate: this report can nominate a source-pack branch, but it cannot promote a steering direction.

## Main Read

Top new candidate: `Economic Activity`.

Economic Activity is the preferred next source-pack branch because it has four prompts, relatively low proposition off-domain contamination, a stable null, and a natural source contrast: broad Commerce Clause aggregation/market regulation versus Lopez/Morrison/NFIB-style limits.

Civil Rights is the backup, but it is more likely to collapse into lexical scrutiny labels unless dominance-reviewed.

Post-run update: the Economic Activity source pack and cue-masked BF16 probe were completed in `sweep_v4/scotus_source_frame_probe_20260501_014711/`. The primary broad-versus-limits contrast reached `0.621` test balanced accuracy versus a `0.641` text baseline, and the other contrasts were text/leakage dominated. Economic Activity is therefore not promoted under the current protocol.

Second post-run update: the Civil Rights source pack was built and checked with a cue-masked text-only gate before BF16 probing. Proposed Civil Rights contrasts reached `0.964-1.000` test balanced accuracy from text alone, so the BF16 activation probe was not run. Civil Rights is also not promoted under the current protocol.

Third post-run update: cached justice-style slice mining nominated `section_posture=majority__decade=2000s`, but same-model Qwen3.5 BF16 verification did not promote it. The dev-selected readout was `excerpt_mean @ L16` with dev BA `0.810`, test BA `0.691`, and text test BA `0.500`; the label-shuffle null stayed near chance. This is real but weak structure, not a steering candidate.

Fourth post-run update: Federalism and Due Process source packs were built and stopped at text-only gates before hooks. Federalism anti-commandeering versus preemption reached cue-masked text test BA `1.000`; Due Process substantive versus procedural also reached cue-masked text test BA `1.000`. Neither branch should receive BF16 activation work under the current source-frame protocol.

Fifth post-run update: Administrative Law was also built and gated before hooks. The major-questions versus ordinary-deference source pack produced `127` non-conflict cue-masked examples after strict source-cluster split reassignment. The text-only baseline had weak train-only dev BA (`0.586`) but saturated final test BA (`1.000`), with each held-out split dominated by one source case per frame. Treat this as not promoted and do not run BF16 hooks from this split; if Administrative Law is revisited, it needs a less case/lexeme-identifiable matched-pair design.

## Issue Ranking

| Rank | Issue | Status | Prompts | Score | Mean SD | Off-domain | Disagreement | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Economic Activity | candidate | 4 | 2.836 | 0.880 | 0.050 | 0.570 | Build a Commerce Clause source pack: broad aggregation/market regulation versus Lopez/Morrison/NFIB limits. |
| 2 | Civil Rights | backup_candidate | 4 | 2.684 | 0.926 | 0.072 | 0.592 | Only after Economic Activity; likely needs dominance review because strict/intermediate scrutiny labels are lexical. |
| 3 | Due Process | source_text_dominated | 2 | 1.512 | 0.943 | 0.000 | 0.387 | Source pack built; text-only gate reached 1.000 test BA, so do not hook current contrast. |
| 4 | Administrative Law | source_text_dominated_split_skewed | 1 | 1.345 | 1.005 | 0.000 | 0.133 | Source pack built; final cue-masked text test BA reached 1.000 but the split is case-skewed, so do not hook current contrast. |
| 5 | Criminal Procedure | deprioritize_fourth_branch_failed | 4 | 0.999 | 1.134 | 0.005 | 0.207 | Do not hook current Fourth directions; use only for evaluator diagnostics. |
| 6 | Federalism | source_text_dominated | 1 | 0.475 | 0.931 | 0.113 | 0.800 | Source pack built; text-only gate reached 1.000 test BA, so do not hook current contrast. |
| 7 | Judicial Power | deprioritize_current_branch_failed | 4 | -0.224 | 1.094 | 0.210 | 0.885 | Do not build another Article III pack unless a second reviewer changes labels or a new subdoctrine is defined. |

## Prompt/Condition Detail

| Issue | Prompt | Condition | N | Target present | Off-domain | SD target delta | Mean target delta | Disagreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Administrative Law | FD02_major_questions_agency | bench_memo | 50 | 0.940 | 0.000 | 1.039 | 1.320 | 0.100 |
| Administrative Law | FD02_major_questions_agency | concise_judicial | 50 | 0.840 | 0.000 | 1.025 | -0.360 | 0.100 |
| Administrative Law | FD02_major_questions_agency | majority_reasoning | 50 | 1.000 | 0.000 | 0.950 | -0.580 | 0.200 |
| Civil Rights | CR01_race_admissions | bench_memo | 50 | 0.920 | 0.000 | 0.839 | -0.520 | 0.080 |
| Civil Rights | CR01_race_admissions | concise_judicial | 50 | 0.960 | 0.000 | 0.857 | 0.800 | 0.040 |
| Civil Rights | CR01_race_admissions | majority_reasoning | 50 | 0.960 | 0.000 | 0.646 | 0.460 | 0.060 |
| Civil Rights | CR02_sex_military_academy | bench_memo | 50 | 0.340 | 0.000 | 0.705 | 0.440 | 1.000 |
| Civil Rights | CR02_sex_military_academy | concise_judicial | 50 | 0.460 | 0.000 | 0.872 | 0.660 | 1.000 |
| Civil Rights | CR02_sex_military_academy | majority_reasoning | 50 | 0.640 | 0.000 | 0.877 | -0.080 | 0.920 |
| Civil Rights | CR03_racial_redistricting | bench_memo | 50 | 0.940 | 0.000 | 1.200 | -1.220 | 0.360 |
| Civil Rights | CR03_racial_redistricting | concise_judicial | 50 | 0.960 | 0.000 | 1.384 | 1.960 | 0.400 |
| Civil Rights | CR03_racial_redistricting | majority_reasoning | 50 | 0.960 | 0.020 | 1.377 | -2.680 | 0.420 |
| Civil Rights | CR04_state_disability_access | bench_memo | 50 | 1.000 | 0.040 | 0.771 | 0.240 | 0.920 |
| Civil Rights | CR04_state_disability_access | concise_judicial | 50 | 1.000 | 0.100 | 0.922 | 1.260 | 0.920 |
| Civil Rights | CR04_state_disability_access | majority_reasoning | 50 | 1.000 | 0.700 | 0.665 | -0.080 | 0.980 |
| Criminal Procedure | CP01_locked_backpack | bench_memo | 50 | 0.980 | 0.000 | 1.255 | 1.340 | 0.040 |
| Criminal Procedure | CP01_locked_backpack | concise_judicial | 50 | 1.000 | 0.000 | 1.329 | 3.220 | 0.060 |
| Criminal Procedure | CP01_locked_backpack | majority_reasoning | 50 | 0.980 | 0.000 | 1.420 | 1.060 | 0.120 |
| Criminal Procedure | CP02_cell_phone_arrest | bench_memo | 50 | 1.000 | 0.000 | 1.720 | 0.020 | 0.380 |
| Criminal Procedure | CP02_cell_phone_arrest | concise_judicial | 50 | 1.000 | 0.000 | 1.512 | -2.860 | 0.280 |
| Criminal Procedure | CP02_cell_phone_arrest | majority_reasoning | 50 | 0.960 | 0.000 | 1.279 | -0.420 | 0.280 |
| Criminal Procedure | CP03_dog_sniff_traffic_stop | bench_memo | 50 | 1.000 | 0.000 | 0.773 | 1.120 | 0.020 |
| Criminal Procedure | CP03_dog_sniff_traffic_stop | concise_judicial | 50 | 1.000 | 0.020 | 0.915 | 0.020 | 0.020 |
| Criminal Procedure | CP03_dog_sniff_traffic_stop | majority_reasoning | 50 | 0.980 | 0.040 | 0.889 | -0.160 | 0.100 |
| Criminal Procedure | CP04_home_emergency_entry | bench_memo | 50 | 0.920 | 0.000 | 1.027 | -0.920 | 0.280 |
| Criminal Procedure | CP04_home_emergency_entry | concise_judicial | 50 | 0.980 | 0.000 | 0.843 | 0.940 | 0.460 |
| Criminal Procedure | CP04_home_emergency_entry | majority_reasoning | 50 | 1.000 | 0.000 | 0.640 | -1.280 | 0.440 |
| Due Process | DP01_marriage_liberty | bench_memo | 50 | 1.000 | 0.000 | 0.863 | 0.100 | 0.380 |
| Due Process | DP01_marriage_liberty | concise_judicial | 50 | 1.000 | 0.000 | 0.884 | -0.560 | 0.440 |
| Due Process | DP01_marriage_liberty | majority_reasoning | 50 | 0.960 | 0.000 | 0.857 | -1.200 | 0.740 |
| Due Process | DP02_benefits_hearing | bench_memo | 50 | 1.000 | 0.000 | 1.123 | 0.620 | 0.300 |
| Due Process | DP02_benefits_hearing | concise_judicial | 50 | 1.000 | 0.000 | 0.978 | -1.680 | 0.000 |
| Due Process | DP02_benefits_hearing | majority_reasoning | 50 | 0.740 | 0.000 | 0.953 | -0.700 | 0.460 |
| Economic Activity | EA01_commercial_remedy | bench_memo | 50 | 0.960 | 0.220 | 1.867 | -0.060 | 0.840 |
| Economic Activity | EA01_commercial_remedy | concise_judicial | 50 | 0.880 | 0.160 | 1.459 | -1.440 | 0.500 |
| Economic Activity | EA01_commercial_remedy | majority_reasoning | 50 | 1.000 | 0.200 | 1.038 | -0.060 | 0.920 |
| Economic Activity | EA02_homegrown_market_regulation | bench_memo | 50 | 1.000 | 0.000 | 0.828 | -0.740 | 0.400 |
| Economic Activity | EA02_homegrown_market_regulation | concise_judicial | 50 | 1.000 | 0.000 | 0.600 | -0.260 | 0.820 |
| Economic Activity | EA02_homegrown_market_regulation | majority_reasoning | 50 | 1.000 | 0.000 | 0.507 | -0.220 | 0.920 |
| Economic Activity | EA03_gun_school_zone | bench_memo | 50 | 0.200 | 0.000 | 0.600 | 0.260 | 0.620 |
| Economic Activity | EA03_gun_school_zone | concise_judicial | 50 | 0.760 | 0.000 | 1.191 | -1.640 | 0.280 |
| Economic Activity | EA03_gun_school_zone | majority_reasoning | 50 | 0.900 | 0.000 | 1.008 | -1.380 | 0.200 |
| Economic Activity | EA04_arbitration_preemption | bench_memo | 50 | 1.000 | 0.000 | 0.198 | 0.040 | 0.980 |
| Economic Activity | EA04_arbitration_preemption | concise_judicial | 50 | 0.700 | 0.000 | 0.606 | 0.800 | 0.160 |
| Economic Activity | EA04_arbitration_preemption | majority_reasoning | 50 | 0.540 | 0.020 | 0.663 | -0.360 | 0.200 |
| Federalism | FD01_state_sheriff_checks | bench_memo | 50 | 1.000 | 0.100 | 0.843 | -0.060 | 0.620 |
| Federalism | FD01_state_sheriff_checks | concise_judicial | 50 | 1.000 | 0.080 | 0.800 | 0.820 | 0.960 |
| Federalism | FD01_state_sheriff_checks | majority_reasoning | 50 | 0.940 | 0.160 | 1.149 | -0.160 | 0.820 |
| Judicial Power | JP01_agency_private_company | bench_memo | 50 | 1.000 | 0.060 | 1.279 | -2.420 | 0.320 |
| Judicial Power | JP01_agency_private_company | concise_judicial | 50 | 0.860 | 0.060 | 1.633 | 0.160 | 0.800 |
| Judicial Power | JP01_agency_private_company | majority_reasoning | 50 | 0.880 | 0.060 | 1.002 | 1.660 | 0.800 |
| Judicial Power | JP02_bankruptcy_counterclaim | bench_memo | 50 | 1.000 | 0.980 | 0.535 | -0.200 | 0.980 |
| Judicial Power | JP02_bankruptcy_counterclaim | concise_judicial | 50 | 0.660 | 0.360 | 0.978 | 0.060 | 0.820 |
| Judicial Power | JP02_bankruptcy_counterclaim | majority_reasoning | 50 | 0.940 | 0.640 | 1.266 | 1.300 | 1.000 |
| Judicial Power | JP03_agency_civil_penalty | bench_memo | 50 | 0.980 | 0.000 | 1.096 | 1.060 | 0.960 |
| Judicial Power | JP03_agency_civil_penalty | concise_judicial | 50 | 1.000 | 0.040 | 0.993 | -0.560 | 0.980 |
| Judicial Power | JP03_agency_civil_penalty | majority_reasoning | 50 | 0.980 | 0.020 | 1.180 | 0.420 | 0.980 |
| Judicial Power | JP04_immigration_removal | bench_memo | 50 | 0.980 | 0.100 | 1.007 | 0.080 | 1.000 |
| Judicial Power | JP04_immigration_removal | concise_judicial | 50 | 0.940 | 0.040 | 1.161 | 1.000 | 1.000 |
| Judicial Power | JP04_immigration_removal | majority_reasoning | 50 | 0.920 | 0.160 | 1.004 | -1.180 | 0.980 |

## Use Rules

1. Do not run hooks from this report alone.
2. For the next branch, build source labels first, run cue-masked activation probes, and compare against text baselines.
3. Promote only if the candidate survives reviewed labels, cue masking, text baseline, and proposition-level prompt-matched random controls.
