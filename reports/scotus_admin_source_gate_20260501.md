# SCOTUS Administrative Law Source Gate

## Purpose

Administrative Law was the remaining source-pack branch after Economic Activity, Civil Rights, Federalism, and Due Process failed promotion gates. This gate checks major-questions/clear-authorization reasoning versus ordinary agency-deference/statutory-interpretation reasoning before any BF16 activation capture.

## Artifacts

| Artifact | Path |
| --- | --- |
| Source-pack builder | scripts/experiments/scotus/build_admin_source_pack.py |
| Source-pack report | reports/scotus_admin_source_pack_v1.md |
| Labels | data/scotus/scotus_admin_source_frame_labels_v1.jsonl |
| Review queue | data/scotus/scotus_admin_source_frame_review_queue_v1.jsonl |

## Text-Only Gate

Task: `admin_major_vs_deference = admin_major_questions` versus `admin_deference_ordinary`.

Settings: `text_cue_masked`, conflict-row exclusion, strict source-cluster-heldout split reassignment, plain prompt TF-IDF logistic baseline.

| Split | Frame | N |
| --- | --- | --- |
| dev | admin_deference_ordinary | 26 |
| dev | admin_major_questions | 29 |
| test | admin_deference_ordinary | 21 |
| test | admin_major_questions | 23 |
| train | admin_deference_ordinary | 23 |
| train | admin_major_questions | 5 |

| Split | Balanced accuracy | N | Label counts | Prediction counts |
| --- | --- | --- | --- | --- |
| dev | 0.586 | 55 | {"0": 26, "1": 29} | {"0": 50, "1": 5} |
| test | 1.000 | 44 | {"0": 21, "1": 23} | {"0": 21, "1": 23} |

## Case Coverage After Split Reassignment

| Split | Frame | Case id | Case | N |
| --- | --- | --- | --- | --- |
| dev | admin_deference_ordinary | kisor_2019 | Kisor v. Wilkie | 26 |
| dev | admin_major_questions | west_virginia_epa_2022 | West Virginia v. EPA | 29 |
| test | admin_deference_ordinary | mead_2001 | United States v. Mead Corp. | 21 |
| test | admin_major_questions | biden_nebraska_2023 | Biden v. Nebraska | 23 |
| train | admin_deference_ordinary | auer_1997 | Auer v. Robbins | 1 |
| train | admin_deference_ordinary | barnhart_2002 | Barnhart v. Walton | 3 |
| train | admin_deference_ordinary | brown_williamson_2000 | FDA v. Brown & Williamson Tobacco Corp. | 1 |
| train | admin_deference_ordinary | chevron_1984 | Chevron U.S.A. Inc. v. Natural Resources Defense Council, Inc. | 1 |
| train | admin_deference_ordinary | city_arlington_2013 | City of Arlington v. FCC | 12 |
| train | admin_deference_ordinary | gonzales_oregon_2006 | Gonzales v. Oregon | 1 |
| train | admin_deference_ordinary | king_burwell_2015 | King v. Burwell | 1 |
| train | admin_deference_ordinary | mci_1994 | MCI Telecommunications Corp. v. AT&T Co. | 1 |
| train | admin_deference_ordinary | skidmore_1944 | Skidmore v. Swift & Co. | 1 |
| train | admin_deference_ordinary | utility_air_2014 | Utility Air Regulatory Group v. EPA | 1 |
| train | admin_major_questions | gonzales_oregon_2006 | Gonzales v. Oregon | 4 |
| train | admin_major_questions | utility_air_2014 | Utility Air Regulatory Group v. EPA | 1 |

## Decision

Text dominated; do not run BF16 activation probe.
