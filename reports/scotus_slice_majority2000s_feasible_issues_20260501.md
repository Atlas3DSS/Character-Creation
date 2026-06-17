# SCOTUS Majority-2000s Feasible-Issues Slice Audit

## Purpose

This audits the refined `Scalia_vs_Ginsburg` majority-2000s justice-style slice after restricting it to issue families that can support strict case-component train/dev/test resplits.

The earlier all-issue majority-2000s BF16 verification showed real activation structure but missed the held-out promotion gate. This refinement asks whether the signal becomes more stable once every retained issue family has feasible case-connected split coverage.

## Slice

| Field | Value |
| --- | --- |
| Pair | `Scalia_vs_Ginsburg` |
| Variant | `masked` |
| Model | `/home/orwel/dev_genius/models/Qwen3.5-27B` |
| Base slice | `section_posture=majority,decade=2000s` |
| Retained issues | `Criminal Procedure`, `Economic Activity`, `Judicial Power` |
| Rows | `1154` |
| Case-connected components | `10` |
| Layers | `4,7,8,9,10,16,19,40` |
| Regions | `prompt_last,prompt_mean,excerpt_mean` |
| C grid | `0.001,0.003,0.01,0.03,0.1` |

## Artifacts

| Artifact | Path |
| --- | --- |
| Feasibility audit | `reports/scotus_slice_majority2000s_feasible_issues_split_feasibility_20260501.md` |
| Normal feasible-issues subset | `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_20260501_033918` |
| Normal component resplits | `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_normal_component_resplits_20260501_034116/report.md` |
| Label-shuffle component resplits | `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_label_shuffle_component_resplits_20260501_034539/report.md` |
| Template-variant source capture | `sweep_v4/scotus_slice_bf16_majority2000s_template_variant_20260501_035709/report.md` |
| Template-variant component resplits | `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_template_variant_component_resplits_20260501_040538/report.md` |
| Plain-prompt source capture | `sweep_v4/scotus_slice_bf16_majority2000s_plain_prompt_20260501_040422/report.md` |
| Plain-prompt component resplits | `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_plain_prompt_component_resplits_20260501_041503/report.md` |
| Excerpt-removed source capture | `sweep_v4/scotus_slice_bf16_majority2000s_excerpt_removed_20260501_041341/report.md` |
| Excerpt-removed component resplits | `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_excerpt_removed_component_resplits_20260501_042048/report.md` |
| Neutral-filler source capture | `sweep_v4/scotus_slice_bf16_majority2000s_neutral_filler_20260501_042153/report.md` |
| Neutral-filler component resplits | `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_neutral_filler_component_resplits_20260501_043234/report.md` |
| Cached resplit runner | `scripts/experiments/scotus/resplit_cached_scotus_probe.py` |
| Cached subsetter | `scripts/experiments/scotus/subset_cached_scotus_features.py` |

Invalidated reports:

- `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_plain_prompt_component_resplits_20260501_035406/report.md`
- `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_template_variant_component_resplits_20260501_035406/report.md`

Those two runs reused normal-prompt cached activations for prompt-diagnostic modes. They now have an invalid-run notice at the top and should not be cited.

## Method Notes

The component split plans were selected before probe fitting using only case-component balance and metadata coverage. The primary split is `split_00`, the best metadata-balanced plan, not the best test-scoring plan.

Prompt-diagnostic modes were recaptured from source activations before resplitting. The cached resplit runner now rejects source-mode mismatches except for `label_shuffle`, which intentionally reuses the same features while destroying labels.

## Results

| Diagnostic | Median dev BA | Median test BA | Test BA range | Primary split test BA | Median text test BA | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Normal | `0.812` | `0.746` | `0.660-0.753` | `0.753` | `0.700` | Live, but text is close on some plans |
| Label shuffle | `0.536` | `0.541` | `0.477-0.548` | `0.488` | `0.492` | Null stays near chance |
| Template variant | `0.807` | `0.758` | `0.668-0.807` | `0.777` | `0.695` | Prompt-template diagnostic survives |
| Plain prompt | `0.818` | `0.764` | `0.676-0.796` | `0.777` | `0.691` | Chat-template diagnostic survives |
| Excerpt removed | `0.500` | `0.500` | `0.500-0.500` | `0.500` | `0.500` | Signal collapses without excerpt |
| Neutral filler | `0.575` | `0.542` | `0.512-0.564` | `0.548` | `0.564` | Signal collapses with same-shaped neutral text |

## Readout Pattern

The surviving normal/template/plain plans still select many `prompt_last` readouts:

- Normal: `prompt_last` on splits `00`, `02`, `03`; `excerpt_mean @ L16` on splits `01`, `04`.
- Template variant: `prompt_last` on splits `00`, `02`, `03`; `prompt_mean @ L16` on split `04`; `excerpt_mean @ L16` on split `01`.
- Plain prompt: `prompt_last` on splits `00`, `02`, `03`; `excerpt_mean @ L16` on splits `01`, `04`.

This is better than the broad Economic Activity branch because prompt-template/plain-prompt controls survive, and excerpt-removed/neutral-filler controls collapse. It is still not a circuit claim because the readout is prompt-last-heavy and the masked text baseline remains competitive on several split plans.

## Decision

Keep the majority-2000s feasible-issues slice as evidence for decodable justice-style structure, but do not call it a steerable judicial circuit.

The first causal pilots have now run:

| Direction | Run | Position | Alphas | Random controls | Best prompt-matched target z | Best prompt-matched net z | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `prompt_last @ L10` | `sweep_v4/scotus_sae_poke_20260501_045156/report.md` | `last` | `0.02,0.05,0.1` | `10` | `0.184` | `0.533` | not promoted |
| `excerpt_mean @ L16` | `sweep_v4/scotus_sae_poke_20260501_060425/report.md` | `all` | `0.01,0.02,0.05` | `5` | `0.449` | `0.395` | not promoted |

Detailed causal read: `reports/scotus_majority2000s_feasible_issues_causal_pilot_20260501.md`.

Decision update: the current broad justice-style directions do not yet cause reliable jurisprudential-frame movement beyond same-layer random controls. Further work should shift to manual review/evaluator repair, reverse-sign tests with a specific hypothesis, or narrower same-doctrine/same-case contrasts rather than another broad steering poke.
