# SCOTUS Federalism Source Gate

## Purpose

Federalism was selected after Economic Activity and Civil Rights failed promotion gates. This gate checks the new anti-commandeering versus preemption source pack before any BF16 activation capture.

## Artifacts

| Artifact | Path |
| --- | --- |
| Source-pack builder | scripts/experiments/scotus/build_federalism_source_pack.py |
| Source-pack report | reports/scotus_federalism_source_pack_v1.md |
| Labels | data/scotus/scotus_federalism_source_frame_labels_v1.jsonl |
| Review queue | data/scotus/scotus_federalism_source_frame_review_queue_v1.jsonl |

## Text-Only Gate

Task: `federalism_anti_vs_preemption = federalism_anti_commandeering` versus `federalism_preemption`.

Settings: `text_cue_masked`, conflict-row exclusion, strict source-cluster-heldout split reassignment, plain prompt TF-IDF logistic baseline.

| Split | Frame | N |
| --- | --- | --- |
| dev | federalism_anti_commandeering | 24 |
| dev | federalism_preemption | 20 |
| test | federalism_anti_commandeering | 15 |
| test | federalism_preemption | 13 |
| train | federalism_anti_commandeering | 16 |
| train | federalism_preemption | 33 |

| Split | Balanced accuracy | N | Label counts | Prediction counts |
| --- | --- | --- | --- | --- |
| dev | 0.925 | 44 | {0: 20, 1: 24} | {0: 17, 1: 27} |
| test | 1.000 | 28 | {0: 13, 1: 15} | {0: 13, 1: 15} |

## Case Coverage After Split Reassignment

| Split | Frame | Case id | Case | N |
| --- | --- | --- | --- | --- |
| dev | federalism_anti_commandeering | printz_1997 | Printz v. United States | 24 |
| dev | federalism_preemption | arizona_2012 | Arizona v. United States | 20 |
| test | federalism_anti_commandeering | new_york_1992 | New York v. United States | 15 |
| test | federalism_preemption | geier_2000 | Geier v. American Honda Motor Co. | 13 |
| train | federalism_anti_commandeering | ferc_1982 | FERC v. Mississippi | 1 |
| train | federalism_anti_commandeering | hines_1941 | Hines v. Davidowitz | 1 |
| train | federalism_anti_commandeering | murphy_2018 | Murphy v. National Collegiate Athletic Assn. | 8 |
| train | federalism_anti_commandeering | reno_condon_2000 | Reno v. Condon | 5 |
| train | federalism_anti_commandeering | rice_1947 | Rice v. Santa Fe Elevator Corp. | 1 |
| train | federalism_preemption | cipollone_1992 | Cipollone v. Liggett Group, Inc. | 10 |
| train | federalism_preemption | crosby_2000 | Crosby v. National Foreign Trade Council | 7 |
| train | federalism_preemption | ferc_1982 | FERC v. Mississippi | 1 |
| train | federalism_preemption | gade_1992 | Gade v. National Solid Wastes Management Assn. | 9 |
| train | federalism_preemption | murphy_2018 | Murphy v. National Collegiate Athletic Assn. | 1 |
| train | federalism_preemption | rice_1947 | Rice v. Santa Fe Elevator Corp. | 1 |
| train | federalism_preemption | wyeth_2009 | Wyeth v. Levine | 4 |

## Decision

Text dominated; do not run BF16 activation probe.
