# SCOTUS Due Process Source Gate

## Purpose

Due Process was selected after Economic Activity, Civil Rights, and Federalism failed promotion gates. This gate checks the substantive-versus-procedural source pack before any BF16 activation capture.

## Artifacts

| Artifact | Path |
| --- | --- |
| Source-pack builder | scripts/experiments/scotus/build_due_process_source_pack.py |
| Source-pack report | reports/scotus_due_process_source_pack_v1.md |
| Labels | data/scotus/scotus_due_process_source_frame_labels_v1.jsonl |
| Review queue | data/scotus/scotus_due_process_source_frame_review_queue_v1.jsonl |

## Text-Only Gate

Task: `due_process_substantive_vs_procedural = due_process_substantive` versus `due_process_procedural_mathews`.

Settings: `text_cue_masked`, conflict-row exclusion, strict source-cluster-heldout split reassignment, plain prompt TF-IDF logistic baseline.

| Split | Frame | N |
| --- | --- | --- |
| dev | due_process_procedural_mathews | 11 |
| dev | due_process_substantive | 31 |
| test | due_process_procedural_mathews | 10 |
| test | due_process_substantive | 20 |
| train | due_process_procedural_mathews | 26 |
| train | due_process_substantive | 16 |

| Split | Balanced accuracy | N | Label counts | Prediction counts |
| --- | --- | --- | --- | --- |
| dev | 0.761 | 42 | {0: 11, 1: 31} | {0: 22, 1: 20} |
| test | 1.000 | 30 | {0: 10, 1: 20} | {0: 10, 1: 20} |

## Case Coverage After Split Reassignment

| Split | Frame | Case id | Case | N |
| --- | --- | --- | --- | --- |
| dev | due_process_procedural_mathews | mathews_1976 | Mathews v. Eldridge | 11 |
| dev | due_process_substantive | dobbs_2022 | Dobbs v. Jackson Women's Health Organization | 31 |
| test | due_process_procedural_mathews | lawrence_2003 | Lawrence v. Texas | 1 |
| test | due_process_procedural_mathews | loudermill_1985 | Cleveland Board of Education v. Loudermill | 9 |
| test | due_process_substantive | lawrence_2003 | Lawrence v. Texas | 10 |
| test | due_process_substantive | obergefell_2015 | Obergefell v. Hodges | 10 |
| train | due_process_procedural_mathews | goldberg_1970 | Goldberg v. Kelly | 4 |
| train | due_process_procedural_mathews | goss_1975 | Goss v. Lopez | 8 |
| train | due_process_procedural_mathews | griswold_1965 | Griswold v. Connecticut | 1 |
| train | due_process_procedural_mathews | hamdi_2004 | Hamdi v. Rumsfeld | 5 |
| train | due_process_procedural_mathews | morrissey_1972 | Morrissey v. Brewer | 8 |
| train | due_process_substantive | casey_1992 | Planned Parenthood of Southeastern Pennsylvania v. Casey | 6 |
| train | due_process_substantive | glucksberg_1997 | Washington v. Glucksberg | 7 |
| train | due_process_substantive | griswold_1965 | Griswold v. Connecticut | 1 |
| train | due_process_substantive | morrissey_1972 | Morrissey v. Brewer | 1 |
| train | due_process_substantive | roe_1973 | Roe v. Wade | 1 |

## Decision

Text dominated; do not run BF16 activation probe.
