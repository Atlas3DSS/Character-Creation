# SCOTUS Economic Activity Source Probe

## Purpose

Economic Activity was nominated by the proposition-level triage as the next issue-family branch after the Article III and Fourth Amendment source-pack branches failed their gates. The intended primary contrast was broad Commerce Clause aggregation / market-regulation reasoning versus Lopez / Morrison / NFIB-style limits.

## Artifacts

| Artifact | Path |
| --- | --- |
| Source-pack builder | `scripts/experiments/scotus/build_economic_source_pack.py` |
| Source-pack report | `reports/scotus_economic_source_pack_v1.md` |
| Labels | `data/scotus/scotus_economic_source_frame_labels_v1.jsonl` |
| Review queue | `data/scotus/scotus_economic_source_frame_review_queue_v1.jsonl` |
| Probe script | `scripts/experiments/scotus/probe_scotus_source_frames.py` |
| Probe run | `sweep_v4/scotus_source_frame_probe_20260501_014711/report.md` |

## Method Notes

The first Economic Activity pack was too sparse for the limits side. It was expanded to `31` source cases across broad Commerce Clause, Commerce Clause limits, federalism/state-regulation, and statutory/preemption frames.

Before probing, `probe_scotus_source_frames.py` was repaired so `--reassign-task-splits` assigns a single split per source cluster within each task. The prior implementation balanced each label separately, which could place the same source case in train for one label and test for the other. The new run has strict source-case-heldout splits with no case split leaks.

The probe used Qwen3.5-27B BF16 hidden states, cue-masked text, conflict-row exclusion, layers `8,12,16`, regions from the source-frame probe script, and C grid `0.003,0.01,0.03,0.1,0.3,1.0`.

## Probe Result

| Task | Best readout | Dev BA | Test BA | Text test BA | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `economic_broad_vs_limits` | `prompt_last @ L16` | `0.733` | `0.621` | `0.641` | Reject; activation does not beat text |
| `economic_broad_vs_state` | `prompt_mean @ L12` | `0.875` | `1.000` | `0.969` | Text/leakage dominated |
| `economic_limits_vs_state` | `prompt_mean @ L16` | `0.901` | `1.000` | `0.950` | Text/leakage dominated |
| `economic_preemption_vs_broad` | `prompt_mean @ L12` | `1.000` | `1.000` | `1.000` | Text/leakage dominated |

## Interpretation

The primary Commerce Clause broad-versus-limits contrast is the only comparison that was doctrinally close enough to be useful, and it failed the promotion gate because the activation probe underperformed the cue-masked text baseline.

The other three comparisons are separable, but mostly because the text remains enough to classify them. These are useful leakage/evaluator diagnostics, not evidence of a steerable judicial circuit.

## Decision

Do not run a causal steering poke from the current Economic Activity source directions. The branch is closed under the present protocol unless a manual dominance review or a different subdoctrine produces a cue-masked activation result that clearly beats the text baseline under strict source-case-heldout splits.

Next candidate source-pack work should move to the Civil Rights backup branch only with dominance-reviewed labels, because strict/intermediate scrutiny terms are likely to be lexical.
