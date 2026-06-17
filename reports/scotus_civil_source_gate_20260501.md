# SCOTUS Civil Rights Source Gate

## Purpose

Civil Rights was the backup source-pack branch after Economic Activity failed its cue-masked activation gate. Because Civil Rights doctrine is especially vulnerable to lexical scrutiny cues, this gate checks source-pack support and cue-masked text baselines before any BF16 activation run.

## Artifacts

| Artifact | Path |
| --- | --- |
| Source-pack builder | `scripts/experiments/scotus/build_civil_source_pack.py` |
| Source-pack report | `reports/scotus_civil_source_pack_v1.md` |
| Labels | `data/scotus/scotus_civil_source_frame_labels_v1.jsonl` |
| Review queue | `data/scotus/scotus_civil_source_frame_review_queue_v1.jsonl` |

## Source-Pack Read

The pack produced `271` labels from `2,451` chunks across `24` named SCOTUS cases.

| Frame | Rows | Cases | Multi-frame conflicts |
| --- | ---: | ---: | ---: |
| `civil_race_strict_scrutiny` | `72` | `16` | `5` |
| `civil_sex_intermediate_scrutiny` | `72` | `12` | `13` |
| `civil_section5_congruence` | `72` | `21` | `2` |
| `civil_rational_basis_equal_protection` | `55` | `13` | `19` |

Strict source-cluster-heldout split checks passed for the proposed tasks; no source case appears in multiple splits within a task after conflict-row exclusion.

## Text-Only Gate

| Task | Dev BA | Test BA | Test N | Decision |
| --- | ---: | ---: | ---: | --- |
| `civil_intermediate_vs_section5` | `0.971` | `1.000` | `29` | Text dominated |
| `civil_rational_vs_strict` | `0.955` | `1.000` | `26` | Text dominated |
| `civil_strict_vs_intermediate` | `0.748` | `0.969` | `34` | Text dominated |
| `civil_strict_vs_section5` | `1.000` | `0.964` | `31` | Text dominated |

## Decision

Do not run the BF16 activation probe on this Civil Rights pack. The cue-masked text baseline already solves the proposed contrasts, so an activation probe would not be useful evidence for a steerable judicial circuit.

Civil Rights remains useful as a leakage/evaluator stress test. A future Civil Rights attempt should define less lexical subframes, then run dominance review before probing.
