# SCOTUS Phase 4: Activation Probe Plan

## Status

Phase 4 has run for `Scalia_vs_Ginsburg` on the v2.1 repaired, masked matched-pair corpus.

Current result:

| Item | Value |
| --- | --- |
| Pair | `Scalia_vs_Ginsburg` |
| Variant | `masked` |
| Model | `/home/orwel/dev_genius/models/Qwen3.6-27B-FP8` |
| Best readout | `prompt_last @ L4` |
| Best C | `0.5` |
| Dev balanced accuracy | `0.815` |
| Test balanced accuracy | `0.768` |
| Text baseline, masked | `0.775` |

Stress tests:

| Held-out condition | Balanced accuracy |
| --- | --- |
| Judicial Power | `0.794` |
| Criminal Procedure | `0.784` |
| Economic Activity | `0.733` |
| Majority-only posture | `0.725` |

Interpretation: Phase 4 clears the basic activation-decoding gate, but it does not yet prove a causal or generation-controlling jurisprudential vector. The best activation probe is approximately tied with the masked text baseline, and the selected region is `prompt_last @ L4`, which is close enough to prompt-format and surface-text information that it needs targeted diagnostics before being used as a steering candidate.

## Goal

Phase 4 answers one question:

> Does a local model encode a reliable activation-space distinction between two justices' legal reasoning styles after names, obvious labels, metadata, and author-identifying leakage have been repaired or masked?

Phase 4 does not by itself answer:

- Whether the direction is causal.
- Whether the direction can steer generation.
- Whether the direction captures jurisprudential reasoning rather than surface style.
- Whether the distinction generalizes across issue areas, opinion postures, and prompt formats.

The immediate objective is to decide whether `Scalia_vs_Ginsburg` is ready for a small Phase 5 causal steering pilot. The answer should be based on diagnostics, not only the best probe number.

## Inputs

Primary data:

- `data/scotus/scotus_matched_pairs_v21.jsonl`
- `data/scotus/scotus_chunk_inventory_v21.jsonl`
- `data/scotus/scotus_section_inventory_v21.jsonl`
- `data/scotus/processed/scotus_excluded_chunk_inventory_v21.jsonl`

Primary reports:

- `reports/scotus_baseline_text_classifiers_v21.md`
- `reports/scotus_pair_repair_audit_v21.md`
- `reports/scotus_section_audit_v21.md`
- `sweep_v4/scotus_probe_20260425_085108/report.md`
- `sweep_v4/scotus_probe_20260425_085108/manifest.json`

Primary script:

- `scripts/experiments/scotus/probe_scotus_style.py`

## Entry Criteria

Before Phase 4 is considered valid for a pair:

1. The corpus must use repaired section boundaries.
2. Mojibake and obvious OCR/text corruption must be normalized or excluded.
3. Third-person chunks about the target justice must be excluded from target-author body text.
4. Matched pairs must be case-held-out across train/dev/test.
5. Metadata-only classification must be near chance.
6. The masked text baseline must clear the activation-readiness threshold of `>= 0.75` balanced accuracy.

Current status:

| Pair | Status |
| --- | --- |
| `Scalia_vs_Ginsburg` | Cleared entry criteria; Phase 4 run complete |
| `Thomas_vs_Souter` | Numerically activation-ready, but secondary until Scalia/Ginsburg diagnostics finish |

## Method

### 1. Render Matched Examples

Each matched-pair chunk is rendered into a neutral continuation prompt:

```text
Read the following legal reasoning excerpt and continue the analysis in the same jurisprudential mode.

Excerpt:
{text}

Continuation:
```

The prompt is passed through the model's chat template with generation prompt enabled. The task is not to generate text during probing; the rendered prompt is used only to capture hidden states.

### 2. Capture HuggingFace Activations

Use HuggingFace, not vLLM. vLLM is appropriate for serving and generation, but it does not expose the intermediate hidden states needed for forward-hook activation capture.

For any follow-on Phase 4 diagnostic that generates Qwen legal reasoning, use complete Qwen budgets. A few hundred generated tokens is smoke/debug only. Constructed runs that score final holdings, visible reasoning, no-mask actuator effects, or review queues must use at least `2048` generated answer tokens and should prefer `3072-4096`; visible-thinking budgets need the same floor when the trace is interpreted. Future scripts should import `scripts/experiments/scotus/qwen_eval_budget.py`, require an explicit short-budget opt-in, and write budget status plus `promotion_eligible_budget` into the manifest and report.

Current model:

```text
/home/orwel/dev_genius/models/Qwen3.6-27B-FP8
```

Current layer sweep:

```text
0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,63
```

Captured regions:

| Region | Definition | Risk |
| --- | --- | --- |
| `prompt_last` | Last token of the rendered prompt | Most leakage-sensitive; may classify prompt/excerpt surface |
| `prompt_mean` | Attention-mask mean over full rendered prompt | Mixes instruction, excerpt, and template |
| `excerpt_mean` | Mean over tokens aligned to excerpt span | Cleaner candidate for jurisprudential content |

For steering candidates, prefer `excerpt_mean` or robust `prompt_mean` directions over `prompt_last` unless diagnostics show that `prompt_last` survives prompt-only and prompt-ablation controls.

### 3. Train Linear Probes

For each `(region, layer, C)` configuration:

1. Fit a `StandardScaler`.
2. Fit balanced logistic regression with `liblinear`.
3. Select the best configuration by dev balanced accuracy, breaking ties with F1.
4. Refit the selected configuration on train+dev.
5. Report final test metrics only once.

Current C grid:

```text
0.25,0.5,1.0,2.0
```

Phase 4.1 should extend this downward and upward:

```text
0.001,0.003,0.01,0.03,0.1,0.25,0.5,1.0,2.0,10.0
```

The selected `C=0.5` is not on the current grid boundary, which is better than the earlier probe runs, but the grid should still be widened for stability reporting.

### 4. Save Artifacts

Each run should write:

- `probe_examples.jsonl`
- `feature_meta.jsonl`
- `features.npz`
- `layer_region_search.jsonl`
- `train_predictions.jsonl`
- `dev_predictions.jsonl`
- `test_predictions.jsonl`
- `best_probe_direction.npz`
- `summary.json`
- `manifest.json`
- `report.md`

The existing full run is:

```text
sweep_v4/scotus_probe_20260425_085108/
```

### 5. Stress-Test Generalization

Stress tests retrain on train+dev examples excluding a held-out field value, then evaluate on test examples with that value.

Current fields:

- `issue_area_label`
- `opinion_type`
- `section_posture`

Minimum requirement for a stress row:

- At least `stress_min_eval_per_label` examples per label in the held-out evaluation subset.
- Enough training examples per label after exclusion.

Important current weakness:

- Majority-only posture gets `0.725` balanced accuracy and shows a positive-class prediction skew in the manifest. This is below the main `0.75` threshold and should be treated as a caution before causal claims.

## Reproduction Command

Activate the Qwen/SCOTUS compute venv first:

```bash
source /home/orwel/dev_genius/qwen35_replay_venv/bin/activate
```

Current Phase 4 command, using script defaults where possible:

```bash
python scripts/experiments/scotus/probe_scotus_style.py \
  --pairs data/scotus/scotus_matched_pairs_v21.jsonl \
  --pair Scalia_vs_Ginsburg \
  --variant masked \
  --model-path /home/orwel/dev_genius/models/Qwen3.6-27B-FP8 \
  --output-root sweep_v4 \
  --layers 0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,63 \
  --c-grid 0.25,0.5,1.0,2.0 \
  --batch-size 1 \
  --max-length 1024 \
  --seed 17
```

## Phase 4.1 Diagnostic Plan

Phase 4.1 is the next required step. Its purpose is to determine whether the `Scalia_vs_Ginsburg` activation signal is a robust model-internal style signal or a classifier reading prompt/excerpt surface artifacts.

### A. Prompt Leakage and Null Tests

Run these controls against the same split structure:

| Diagnostic | Expected result if signal is real |
| --- | --- |
| Excerpt replaced with `[EXCERPT REMOVED]` | Chance |
| Excerpt replaced with length-matched neutral legal filler | Chance or near chance |
| Labels shuffled within split | Chance |
| Rendered-prompt TF-IDF baseline | Should not outperform activation in a way that explains it |
| Prompt template variants | Ranking should not depend on one exact template |
| No chat template / plain prompt | Similar mid-layer excerpt signal, if real |

The strongest warning sign would be `prompt_last @ L4` staying high when the excerpt is removed or replaced.

### B. Region Robustness

Rerun a denser layer sweep around the current winning and competing bands:

```text
0-20,24,28,32,36,40,44,48
```

Required regions:

- `prompt_last`
- `prompt_mean`
- `excerpt_mean`

Preferred result:

- `excerpt_mean` or `prompt_mean` at L8-L16 remains at or above roughly `0.75` test balanced accuracy.
- `prompt_last @ L4` is not the only strong configuration.

If the only strong result is `prompt_last @ L4`, treat the probe as a useful decoder but not as a steering-ready vector.

### C. Selection-Bias Reporting

The report must show the distribution of test performance across the sweep, not only the best selected configuration.

Add:

- Top 20 by dev balanced accuracy.
- Top 20 by test balanced accuracy, marked as diagnostic only.
- Heatmap/table of dev balanced accuracy by `(layer, region)`.
- Heatmap/table of test balanced accuracy by `(layer, region)`.
- Median and interquartile range for test balanced accuracy across all configs.
- Count of configs above `0.70`, `0.75`, and `0.80` dev balanced accuracy.

Interpretation rule:

- A broad band of strong configs is evidence for a stable encoded distinction.
- A single winning config is evidence for a fragile classifier result.

### D. Sample Size and Confidence Reporting

Every result table should include:

- `n_train_pos`
- `n_train_neg`
- `n_dev_pos`
- `n_dev_neg`
- `n_test_pos`
- `n_test_neg`

For final headline results, include a binomial or bootstrap confidence interval for balanced accuracy. The current test set for `Scalia_vs_Ginsburg` is only `69 + 69`, so small changes in errors move the score noticeably.

### E. Candidate Direction Promotion

Classify each candidate as one of:

| Class | Meaning |
| --- | --- |
| `diagnostic_only` | Good decoder, not suitable for steering yet |
| `candidate_direction` | Robust enough for a small causal pilot |
| `reject` | Fails leakage/null/stress tests |

Promotion criteria:

1. Test balanced accuracy `>= 0.75`.
2. Dev/test gap is not extreme.
3. Null tests fall to chance.
4. Region is not only `prompt_last`, or `prompt_last` survives direct prompt-ablation diagnostics.
5. Stress tests do not reveal a single confound explaining the result.
6. Direction has stable sign and similar performance under at least one prompt variant.

Current likely classification:

| Direction | Tentative class | Reason |
| --- | --- | --- |
| `prompt_last @ L4` | `diagnostic_only` | Best decoder, but leakage-sensitive |
| `excerpt_mean @ L12` | `candidate_direction` if test holds | Cleaner content region and competitive dev result |
| `prompt_mean @ L12-L16` | `candidate_direction` if test holds | Competitive, but mixes template and excerpt |

## Phase 4.2: All-Justice Qwen-Scope SAE Overlap Pilot

This addresses a separate question raised after the Qwen-Scope SAE download:

> When justice names and obvious labels are masked, do Scalia, Ginsburg, Thomas, and Souter activate the same SAE features/layers/circuits, or do justice-specific feature differences remain after issue and posture controls?

Current run:

- Script: `scripts/experiments/scotus/scotus_sae_overlap_all_justices.py`
- Report: `sweep_v4/scotus_sae_overlap_all4_20260430_153933/report.md`
- Manifest: `sweep_v4/scotus_sae_overlap_all4_20260430_153933/manifest.json`
- Model: `/home/orwel/dev_genius/models/Qwen3.6-27B-FP8`
- SAE: `/home/orwel/dev_genius/models/qwen_scope/SAE-Res-Qwen3.5-27B-W80K-L0_100`
- Data: masked v2.1 chunks from `data/scotus/scotus_chunk_inventory_v21.jsonl`
- Sample: 480 examples, balanced as `4 justices x 6 issue areas x 20 chunks`
- Layers: `4,8,12,16`
- Regions: `prompt_mean,excerpt_mean`
- Metrics: top-feature Jaccard, weighted Jaccard over SAE activation-rate vectors, and cosine over activation-rate vectors.

Summary:

| Region | Layer | Overall weighted-J | Issue weighted-J | Posture weighted-J | Issue cosine |
| --- | --- | --- | --- | --- | --- |
| `prompt_mean` | `4` | `0.864` | `0.784` | `0.828` | `0.974` |
| `excerpt_mean` | `4` | `0.856` | `0.768` | `0.816` | `0.972` |
| `prompt_mean` | `12` | `0.796` | `0.670` | `0.741` | `0.960` |
| `prompt_mean` | `16` | `0.777` | `0.656` | `0.719` | `0.955` |
| `excerpt_mean` | `16` | `0.751` | `0.606` | `0.682` | `0.942` |

Interpretation:

- The dominant result is shared routing, especially at layer 4. Different masked justices on the same broad legal subject mostly activate the same SAE feature families.
- The later layers show more top-feature turnover. That is the most plausible place to look for justice-specific feature differences.
- The weakest conditioned rows are useful investigative leads, not steering candidates. Examples include `Ginsburg/Thomas` on Judicial Power at `excerpt_mean @ L16` with weighted-J `0.543`, `Scalia/Ginsburg` on Criminal Procedure at `excerpt_mean @ L16` with weighted-J `0.550`, and `Thomas/Souter` dissent chunks at `excerpt_mean @ L16` with weighted-J `0.581`.
- Topic/procedural features should not be discarded for this analysis. For steering they are confounds; for the overlap question they are evidence about whether the model routes masked legal text through common or justice-specific feature sets.

Immediate follow-up:

1. Scale the overlap run to at least `50` chunks per justice/issue cell where available.
2. Run both downloaded Qwen-Scope SAEs, top-k `50` and top-k `100`, and compare stability.
3. Add per-feature inspection for the weakest conditioned rows: top activating chunks, case names, issue labels, posture, and whether the feature appears justice-specific or merely case/topic-specific.
4. Add a raw-hidden baseline such as CKA/RSA or cosine over hidden-state means so SAE-overlap conclusions are not an artifact of the selected dictionary.
5. Only after those checks, decide whether any justice-specific SAE feature set is worth a causal patch or ablation test.

### Phase 4.2a Layer-Poke Smoke Test

A qualitative poke script now exists:

- Script: `scripts/experiments/scotus/poke_scotus_sae_layers.py`
- Main usable report: `sweep_v4/scotus_sae_poke_20260430_162006/report.md`
- Model used for usable generation: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- SAE: `/home/orwel/dev_genius/models/qwen_scope/SAE-Res-Qwen3.5-27B-W80K-L0_100`
- Layer/region: `excerpt_mean @ L16`
- Poke method: unit-normalized weighted sums of SAE decoder columns, where weights are target-minus-reference positive differential activations within weak conditioned rows.
- Alpha: `16`
- Position: generated-token last position.

Important failed control:

- Hook-based generation with `/home/orwel/dev_genius/models/Qwen3.6-27B-FP8` produced corrupted baseline text under HuggingFace, even with the correct `AutoModelForImageTextToText` loader. Do not use that FP8 checkpoint for HF hook-based generation until the generation path is fixed. It remains usable for activation capture/probing, but generation steering needs either the BF16 Qwen3.5 path or a different hook-capable generation setup.

Smoke-test read:

- Qwen3.5 BF16 baseline generation is coherent.
- Same-norm random L16 pokes changed the completions while preserving coherence.
- The SAE-derived L16 directions also changed completions coherently. On the Article III prompt, all SAE directions shifted the framing around `public rights`, `private rights`, Article III vesting, and agency adjudication. On the Fourth Amendment prompt, the pokes shifted emphasis around `search incident to arrest`, `immediate control`, and closed-container privacy.
- Because the random unit vector also changed the completions, this run only shows that L16 is perturbable in a controlled way. It does not yet show justice-specific causal content.

What we learned:

1. The strongest current interpretation is not `justice circuit found`. It is `L16 can alter legal-framing selection while preserving coherence`.
2. The changed surface is doctrinal entry point, not just prose style. On Article III, completions moved among private-rights limits, public-rights exceptions, sovereign-regulator framing, Article I legislative-court framing, and case-or-controversy framing. On Fourth Amendment, completions moved among plain-view/closed-container privacy, search incident to arrest, immediate-control/wingspan, officer-safety/evidence-preservation, and exigency/consent framing.
3. The same-norm random control is an important warning. Random L16 perturbation also moved the doctrinal frame, so the layer appears sensitive to legal pathway selection generally.
4. The SAE directions are still worth pursuing because they changed legally meaningful frames without obvious incoherence on Qwen3.5 BF16, and some directions moved into issue-relevant frames even when the candidate came from a different weak-overlap row.
5. The next test should ask whether SAE-derived directions shift frames more consistently, directionally, or alpha-sensitively than random vectors. The core comparison is `SAE frame shift vs random frame shift`, not `poked vs baseline`.

First completed follow-up:

1. Run an alpha sweep on Qwen3.5 BF16: `0,4,8,16,32`.
2. Include several random same-norm controls, not just one.

Still pending:

1. Score generated text with the masked text baseline/probe and a legal-rhetoric rubric.
2. Compare SAE directions against raw-hidden pairwise directions from the same Qwen3.5 model before making any justice-specific steering claim.
3. Expand the prompt set before drawing anything stronger than qualitative conclusions.

### Phase 4.2b Alpha/Random Sweep

The first alpha/random follow-up completed successfully.

- Report: `sweep_v4/scotus_sae_poke_20260430_172731/report.md`
- Generations: `sweep_v4/scotus_sae_poke_20260430_172731/generations.jsonl`
- Frame summary: `sweep_v4/scotus_sae_poke_20260430_172731/frame_summary.jsonl`
- Manifest: `sweep_v4/scotus_sae_poke_20260430_172731/manifest.json`
- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- SAE: `/home/orwel/dev_genius/models/qwen_scope/SAE-Res-Qwen3.5-27B-W80K-L0_100`
- Position: generated-token last position
- Alphas: `4,8,16,32`
- Random controls: `3` same-norm L16 unit directions
- Prompts: `2` neutral legal hypotheticals, one Article III agency-adjudication prompt and one Fourth Amendment locked-backpack prompt
- Total completions: `58`

Candidate directions:

- `judicial_power_ginsburg_minus_thomas_l16`
- `judicial_power_thomas_minus_ginsburg_l16`
- `criminal_procedure_scalia_minus_ginsburg_l16`
- `criminal_procedure_ginsburg_minus_scalia_l16`

Sweep read:

1. The Article III baseline already contained `public rights` and Article I tribunal framing. Random L16 controls also moved around the same broad frame family, especially `public rights` and Article I tribunal language. This means broad Article III frame movement is not by itself evidence of a justice-specific SAE direction.
2. The Fourth Amendment baseline was mostly `plain view` plus closed-container privacy: plain view may justify seizure of the backpack, but not the warrantless search of the closed container. Random L16 controls sometimes shifted into `search incident to arrest` and `Chimel`-style immediate-control framing. Again, broad frame movement is not SAE-specific.
3. The SAE directions still showed candidate-specific hints. On Article III, the criminal-procedure-derived Scalia/Ginsburg directions often pushed Article I tribunal framing at alphas `8` and `16`; `judicial_power_ginsburg_minus_thomas_l16` repeatedly introduced `case or controversy` language; `judicial_power_thomas_minus_ginsburg_l16` tended to preserve heavier `public rights/private rights` framing through alpha `16`.
4. On the Fourth Amendment prompt, `criminal_procedure_scalia_minus_ginsburg_l16` most consistently pushed `search incident to arrest`, `immediate control`, and officer-safety/evidence-preservation framing at alphas `8` and `16`. But random controls could also trigger that same doctrinal pathway, so the result is a lead, not a finding.
5. Alpha `8` and `16` were the most useful qualitative range. Alpha `32` sometimes remained coherent but began to create awkward or unstable legal prose in some directions.

Updated interpretation:

- The follow-up strengthens the conservative claim: L16 is a live control surface for legal doctrinal frame selection.
- The follow-up weakens any premature claim that these are justice-specific circuits. Random same-norm controls are strong.
- The interesting question has narrowed. The next experiment should not ask whether pokes change legal text; they do. It should ask whether SAE-derived directions change target doctrinal frames more consistently, directionally, and alpha-sensitively than a distribution of random same-norm vectors.

### Phase 4.2c Off-Domain Economic Activity Probe

A second focused follow-up tested whether the same L16 directions merely perturb any legal prompt or whether they overpower an unrelated Economic Activity prompt.

- Report: `sweep_v4/scotus_sae_poke_20260430_174244/report.md`
- Generations: `sweep_v4/scotus_sae_poke_20260430_174244/generations.jsonl`
- Frame summary: `sweep_v4/scotus_sae_poke_20260430_174244/frame_summary.jsonl`
- Prompt id: `2`
- Prompt topic: federal remedy for misleading commercial conduct, Commerce Clause authority, and traditional state regulation
- Alphas: `8,16`
- Random controls: `5`
- Total completions: `19`

Off-domain read:

1. The baseline was strongly Commerce Clause oriented. It framed the question around congressional power to regulate economic activity that substantially affects interstate commerce.
2. Random L16 controls mostly preserved the same Commerce Clause pathway, but often added federalism, state police-power, or remedy language. One random alpha `16` completion also picked up stray Article III/private-rights tags, confirming that random L16 perturbations can inject adjacent constitutional vocabulary.
3. SAE directions did not overpower the prompt. The completions generally remained in the Economic Activity domain, which is good for coherence but weakens any claim that the directions encode a simple portable justice voice.
4. `judicial_power_ginsburg_minus_thomas_l16` added the clearest off-domain trace of its source row: more federal-regulatory/state-police-power framing and a small `public rights` tag at both alphas. This is interesting but could be topic-adjacent vocabulary rather than a justice-specific circuit.
5. The criminal-procedure directions mostly stayed near Commerce Clause analysis. `criminal_procedure_scalia_minus_ginsburg_l16` at alpha `16` shifted toward Article I powers and traditional police powers, but not toward criminal-procedure doctrine.

Updated interpretation after the off-domain check:

- Prompt semantics dominate. The pokes bias the legal route, but they do not simply impose source-issue doctrine onto unrelated prompts.
- The strongest lead remains frame selection at L16, especially around public/private rights, Article I tribunal authority, search-incident doctrine, and federalism/regulatory-power language.
- This still does not establish justice specificity. It does show that the directions are coherent enough to justify a larger, scored sweep.

### Phase 4.2d Twenty-Prompt Scored SAE-vs-Random Sweep

The larger scored sweep has now run.

- Prompt bank: `data/scotus/scotus_poke_prompts_v1.jsonl`
- Report: `sweep_v4/scotus_sae_poke_20260430_181317/report.md`
- Generations: `sweep_v4/scotus_sae_poke_20260430_181317/generations.jsonl`
- Score summary: `sweep_v4/scotus_sae_poke_20260430_181317/score_summary.jsonl`
- Candidate-vs-random: `sweep_v4/scotus_sae_poke_20260430_181317/candidate_vs_random.jsonl`
- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- SAE: `/home/orwel/dev_genius/models/qwen_scope/SAE-Res-Qwen3.5-27B-W80K-L0_100`
- Prompts: `20`
- Rows: `580`
- Alphas: `8,16`
- Random controls: `10` same-norm L16 directions per alpha
- SAE candidates: the two Judicial Power Ginsburg/Thomas directions and the two Criminal Procedure Scalia/Ginsburg directions
- Scoring: keyword frame tags with prompt-level `expected_frames`, `contrast_frames`, and `domain_frames`; target deltas are relative to each prompt's unpoked baseline.

Aggregate result:

| Candidate | Alpha | Mean target delta | Random mean | Random SD | Z vs random |
| --- | --- | ---: | ---: | ---: | ---: |
| `criminal_procedure_scalia_minus_ginsburg_l16` | `8` | `0.25` | `-0.045` | `1.208` | `0.24` |
| `criminal_procedure_scalia_minus_ginsburg_l16` | `16` | `0.35` | `-0.165` | `1.310` | `0.39` |
| `criminal_procedure_ginsburg_minus_scalia_l16` | `8` | `0.15` | `-0.045` | `1.208` | `0.16` |
| `criminal_procedure_ginsburg_minus_scalia_l16` | `16` | `0.00` | `-0.165` | `1.310` | `0.13` |
| `judicial_power_ginsburg_minus_thomas_l16` | `8` | `-0.70` | `-0.045` | `1.208` | `-0.54` |
| `judicial_power_ginsburg_minus_thomas_l16` | `16` | `-0.60` | `-0.165` | `1.310` | `-0.33` |
| `judicial_power_thomas_minus_ginsburg_l16` | `8` | `-0.15` | `-0.045` | `1.208` | `-0.09` |
| `judicial_power_thomas_minus_ginsburg_l16` | `16` | `-0.05` | `-0.165` | `1.310` | `0.09` |

Read:

1. The global SAE-vs-random gate did not clear. The best global candidate, `criminal_procedure_scalia_minus_ginsburg_l16 @ alpha 16`, was only `z=0.39` against the row-level random distribution.
2. Random controls remained broad and noisy. The random target-delta SD was roughly `1.2-1.3`, large enough that the SAE mean shifts are small by comparison.
3. The criminal-procedure-derived directions produced the only positive global movement, but not enough to call a robust causal effect. Their strongest visible pattern was again Article I tribunal framing on Judicial Power prompts, not a clean source-domain Criminal Procedure improvement.
4. The Judicial Power Ginsburg-minus-Thomas direction moved target-frame counts downward globally. It also increased contrast-frame presence slightly and had higher off-domain presence at alpha `8`. That is evidence against treating it as a clean justice-style direction.
5. Prompt semantics remained dominant. Most completions stayed in the prompt's legal domain. The directions biased frame emphasis but did not impose a portable justice voice or a stable source-issue doctrine.
6. The keyword scorer is useful as a first-pass gate but still crude. It overweights repeated doctrinal phrases, underweights paraphrases, and can punish legally coherent shifts if they use different vocabulary. It should not be the final evaluator.

Decision:

- Do not promote these SAE directions to Phase 5 as justice-specific steering vectors.
- Keep the L16 frame-selection finding as real but general.
- The next discriminating comparison is raw-hidden pairwise directions, ideally recaptured on the same Qwen3.5 model used for generation. If raw-hidden directions also fail against the random distribution, the current SCOTUS steering path should pause and shift back to diagnostics/probe validity rather than more poking.

### Phase 4.2e Raw-Hidden Direction Comparison

The raw-hidden comparison also ran.

- Report: `sweep_v4/scotus_sae_poke_20260430_184146/report.md`
- Generations: `sweep_v4/scotus_sae_poke_20260430_184146/generations.jsonl`
- Score summary: `sweep_v4/scotus_sae_poke_20260430_184146/score_summary.jsonl`
- Candidate-vs-random: `sweep_v4/scotus_sae_poke_20260430_184146/candidate_vs_random.jsonl`
- Direction source: raw hidden mean differences from `sweep_v4/scotus_sae_overlap_all4_20260430_153933/features.npz`
- Direction layer/region: `excerpt_mean @ L16`
- Direction groups: same four justice-pair/issue candidates as the SAE sweep
- Generation model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Direction feature source model: `/home/orwel/dev_genius/models/Qwen3.6-27B-FP8`

Important caveat:

- These raw-hidden vectors are architecture-compatible unit directions, but they were built from the existing Qwen3.6-FP8 overlap capture and then injected into Qwen3.5 BF16 generation. A fully clean raw-hidden test should recapture the same groups on Qwen3.5 BF16. This run is still a useful comparator, but it is not the final same-model raw-hidden result.

Aggregate result:

| Candidate | Alpha | Mean target delta | Random mean | Random SD | Z vs random |
| --- | --- | ---: | ---: | ---: | ---: |
| `raw_hidden_criminal_procedure_ginsburg_minus_scalia_l16` | `8` | `0.35` | `-0.045` | `1.208` | `0.33` |
| `raw_hidden_criminal_procedure_ginsburg_minus_scalia_l16` | `16` | `0.50` | `-0.165` | `1.310` | `0.51` |
| `raw_hidden_criminal_procedure_scalia_minus_ginsburg_l16` | `8` | `-0.15` | `-0.045` | `1.208` | `-0.09` |
| `raw_hidden_criminal_procedure_scalia_minus_ginsburg_l16` | `16` | `-0.55` | `-0.165` | `1.310` | `-0.29` |
| `raw_hidden_judicial_power_ginsburg_minus_thomas_l16` | `8` | `-0.15` | `-0.045` | `1.208` | `-0.09` |
| `raw_hidden_judicial_power_ginsburg_minus_thomas_l16` | `16` | `-0.45` | `-0.165` | `1.310` | `-0.22` |
| `raw_hidden_judicial_power_thomas_minus_ginsburg_l16` | `8` | `-0.10` | `-0.045` | `1.208` | `-0.05` |
| `raw_hidden_judicial_power_thomas_minus_ginsburg_l16` | `16` | `-0.40` | `-0.165` | `1.310` | `-0.18` |

Raw-hidden read:

1. Raw-hidden directions did not solve the problem. The best global row was only `z=0.51` versus random.
2. The raw criminal-procedure directions showed cleaner sign behavior than the SAE decoder directions: `Ginsburg - Scalia` moved the broad target-frame score up, while `Scalia - Ginsburg` moved it down. But the effect is still small and not enough for a steering claim.
3. Judicial Power raw directions were weak or negative on the broad target-frame metric.
4. The result reinforces the same conclusion as the SAE sweep: L16 interventions can bias legal-frame wording, but these candidate directions do not yet beat random strongly enough to count as justice-specific causal vectors.

Decision after SAE and raw-hidden comparisons:

- Pause justice-specific steering claims for these L16 directions.
- Do not run more qualitative pokes on these candidates unless the evaluation metric is improved first.
- The likely bottleneck is now measurement and probe validity, not lack of more completions.

Next larger test:

1. Recapture raw-hidden L16 directions on Qwen3.5 BF16 only if we want a fully clean same-model replication.
2. Add a blind manual review pass over a small stratified sample, because the keyword rubric is now the main measurement weakness.
3. Return to Phase 4.1 leakage/robustness diagnostics before investing in more steering runs.
4. If Phase 4.1 does not produce a cleaner non-`prompt_last` activation signal, stop the SCOTUS steering branch and treat SCOTUS as a probe-validity benchmark rather than a steering benchmark.

## Phase 5 Gate

Do not run a full steering interpretation until Phase 4.1 is complete.

Advance to a Phase 5 causal pilot only if:

1. At least one non-`prompt_last` candidate clears the promotion criteria, or `prompt_last` clears strict prompt-ablation diagnostics.
2. Null tests are at chance.
3. Stress-test weaknesses are documented, especially majority posture and Economic Activity.
4. The final report explicitly states that activation decoding is approximately tied with text baseline, not clearly superior to it.

If the gate passes, Phase 5 should be a small controlled pilot:

- Use neutral legal hypotheticals with no justice names.
- Compare no steering, positive direction, negative direction, random same-norm direction, and wrong-pair direction.
- Evaluate generated text with the masked text classifier, the activation probe, and a jurisprudential rubric.
- Scale intervention strength by per-layer hidden-state norm rather than raw unit-vector alpha.
- Prefer early/mid-layer `excerpt_mean` or `prompt_mean` candidates before trying `prompt_last`.

## Immediate Next Work

Current highest-order bit: Phase 4.1 found robust non-prompt L16 decoders, but the first Phase 5 causal pilot did not show a reliable generation-steering effect. We still do not have a steerable justice-specific circuit.

Completed Phase 4.1 diagnostic state:

1. Full-grid controls are complete:
   - `excerpt_removed`: chance.
   - `neutral_filler`: best `excerpt_mean @ L9`, dev BA `0.584`, test BA `0.587`, no configs above `0.70`.
   - `label_shuffle`: reused normal activation cache with labels shuffled within split. Best `prompt_last @ L32`, dev BA `0.584`, test BA `0.435`, no configs above `0.70`.
   - `template_variant`: best `prompt_last @ L4`, dev BA `0.832`, test BA `0.783`; robust non-prompt L16 rows persist.
   - `plain_prompt`: best `prompt_last @ L10`, dev BA `0.836`, test BA `0.797`; robust non-prompt L16 rows persist.
2. Aggregate report is updated:
   - `reports/scotus_phase41_diagnostics_current.md`
   - `reports/scotus_phase41_diagnostics.md`
3. Robust non-prompt candidates:
   - `prompt_mean @ L16, C=0.003`: clears dev/test `>= 0.75` in normal, template-variant, and plain-prompt modes; worst dev BA `0.761`, worst test BA `0.848`.
   - `excerpt_mean @ L16, C=0.003`: clears dev/test `>= 0.75` in all three modes; worst dev BA `0.761`, worst test BA `0.841`.
4. Important caution:
   - The selected best decoders are still `prompt_last`.
   - Rendered-prompt TF-IDF remains high at roughly `0.754-0.761` test BA.
   - The majority-only stress row is weak in the plain-prompt run: BA `0.609`.

Phase 5 candidate export:

- Export directory: `sweep_v4/scotus_phase5_candidate_directions_20260501/`
- Export script: `scripts/experiments/scotus/export_probe_direction.py`
- Exported directions:
  - `averaged_prompt_mean_L16_C0p003.npz`
  - `averaged_excerpt_mean_L16_C0p003.npz`
  - `averaged_prompt_mean_L16_C0p003_negative_scalia.npz`
  - `averaged_excerpt_mean_L16_C0p003_negative_scalia.npz`
- Stability: prompt-mean and excerpt-mean directions were highly stable across normal/template/plain prompt modes, with cross-mode cosines roughly `0.98-0.99`.

Local Phase 5 last-token causal pilot:

- Run directory: `sweep_v4/scotus_sae_poke_20260430_224651/`
- Report: `sweep_v4/scotus_sae_poke_20260430_224651/report.md`
- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Precision: BF16 HuggingFace hooks, not quantized.
- Rows: `860`
- Design: `20` neutral legal prompts; baseline; `10` same-layer random controls per alpha; four external L16 probe directions; hidden-norm-fraction alphas `0.05`, `0.10`, `0.20`.
- Effective alpha scale:
  - prompt-mean: about `2.689`, `5.377`, `10.754` hidden units.
  - excerpt-mean: about `2.696`, `5.392`, `10.785` hidden units.
- Intervention position: `last`.

Last-token pilot result:

| Readout | Best row |
| --- | --- |
| Pooled candidate-vs-random | max `z=0.34`, `averaged_excerpt_mean_L16_C0p003_negative_scalia @ alpha 0.05` |
| Prompt-matched candidate-vs-random | max `z=0.54`, same row |
| Prompt win rate versus matched random mean | max `0.45` |
| Candidate mean target deltas | range `-0.55` to `+0.10` |

Interpretation:

- The robust L16 probe directions are decodable but did not produce a reliable last-token generation shift.
- The weak positive rows are far inside the same-prompt random-control spread.
- This is evidence against treating these averaged L16 justice directions as steering vectors.
- It is not a proof that no judicial circuit exists; it may mean the intervention position, target metric, or broad justice-level direction is wrong.

Remote Q4 proxy run:

- Local artifact mirror: `sweep_v4/scotus_qwen4bit_proxy_20260501_045257/`
- Script: `scripts/experiments/scotus/qwen4bit_proxy_generation.py`
- Report: `sweep_v4/scotus_qwen4bit_proxy_20260501_045257/report.md`
- Rows: `3060`
- Completed: `2026-05-01T06:20:55+00:00`
- Design: `20` legal prompts x `3` neutral prompt conditions x `50` sampled controls per prompt-condition, plus deterministic base completions.
- Outputs: `generations.jsonl`, `score_summary.json`, `prompt_condition_nulls.json`, `blind_review_sample.jsonl`, `blind_review_key.jsonl`, `report.md`.
- Interpretation: proxy-only null evidence. The run shows that neutral prompt wording and generation variance alone create wide legal-frame movement, so future steering claims need prompt-matched random controls and ideally blind review.

All-position sanity check:

- Run directory: `sweep_v4/scotus_sae_poke_20260430_233245/`
- Report: `sweep_v4/scotus_sae_poke_20260430_233245/report.md`
- Rows: `336`
- Design: `12` neutral legal prompts; same four external L16 directions; same-layer random controls; `position=all`; hidden-norm-fraction alphas `0.01`, `0.02`, `0.05`.
- Best pooled row: `averaged_prompt_mean_L16_C0p003_negative_scalia @ alpha 0.01`, `z=0.56`.
- Best prompt-matched row: same candidate and alpha, `z=0.73`, prompt win rate `0.50`.
- The effect weakened at larger all-position alphas and did not clearly beat matched random controls.

Decision synthesis:

- Script: `scripts/experiments/scotus/analyze_phase5_evidence.py`
- Report: `reports/scotus_phase5_decision_20260501.md`
- The report combines the last-token hook pilot, all-position hook sanity check, Q4 proxy nulls, rubric contamination counts, and issue-conditioned SAE overlap rows.
- Highest-priority nominated issue candidates are `Judicial Power` and `Criminal Procedure` at `excerpt_mean @ L16`, but these are only candidate starts, not steering evidence.
- Direct keyword labels for `public rights`, `private rights`, and most Fourth Amendment subframes are too sparse in the repaired opinion chunks, so the next frame-specific branch needs curated frame-labeled excerpts or contrastive prompt capture rather than naive keyword probes.

### Phase 4.2f Curated Frame-Contrast Branch

The first frame-specific branch has now run. It tested whether cue-heavy, curated frame contrasts could produce cleaner candidate directions than the broad justice-level L16 directions.

- Probe report: `sweep_v4/scotus_frame_contrast_probe_20260430_235745/report.md`
- Dataset: `data/scotus/scotus_frame_contrast_v1.jsonl`
- Prompt bank for causal pokes: `data/scotus/scotus_frame_poke_prompts_v1.jsonl`
- Article III direction: `sweep_v4/scotus_frame_contrast_probe_20260430_235745/article3_private_vs_public/direction.npz`
- Fourth Amendment direction: `sweep_v4/scotus_frame_contrast_probe_20260430_235745/fourth_digital_vs_incident/direction.npz`

Probe read:

| Task | Best readout | Dev BA | Test BA | Text test BA |
| --- | --- | ---: | ---: | ---: |
| `article3_private_vs_public` | `prompt_mean @ L16, C=0.003` | `1.000` | `1.000` | `1.000` |
| `fourth_digital_vs_incident` | `prompt_mean @ L16, C=0.003` | `1.000` | `1.000` | `1.000` |

Interpretation:

- These are deliberately easy, cue-heavy contrasts. Perfect activation results with perfect text baselines mean the directions are candidate probes only, not evidence of a judicial circuit.
- The useful question is causal: whether injecting the frame direction shifts neutral legal generations more than same-layer random controls.

Frame causal pilots:

| Frame direction | Run | Prompts | Best target result | Best net result | Decision |
| --- | --- | ---: | --- | --- | --- |
| Article III private-rights over public-rights | `sweep_v4/scotus_sae_poke_20260501_000146/` | `4` | Alpha `0.10`: matched target `z=-0.02`, win rate `0.25` | Alpha `0.10`: matched net `z=0.42`, win rate `0.75` | Not promoted |
| Fourth Amendment digital privacy over search incident | `sweep_v4/scotus_sae_poke_20260501_001257/` | `4` | Alpha `0.05`: matched target `z=0.93`, win rate `0.50` | Alpha `0.05`: matched net `z=0.55`, win rate `0.75` | Not promoted |

Qualitative read:

1. Article III target movement did not beat prompt-matched random controls. The only positive sign was a weak net target-minus-contrast bump at alpha `0.10`, too small and too sparse to treat as evidence.
2. Fourth Amendment alpha `0.05` strengthened already-present `Riley`/digital-privacy wording on two prompts, but alpha `0.10` reversed badly by increasing safety/search-incident language.
3. Both pilots preserve the earlier conservative conclusion: Qwen L16 can bias legal frame wording, but these constructed directions do not yet beat random strongly enough to count as steerable circuits.
4. Cue-heavy contrast capture is not enough. The next candidate construction should either use source-grounded, adjudicated frame labels or an improved evaluator/blind-review gate before more hook-generation runs.

Current decision:

1. Do not promote the current averaged L16 `prompt_mean`/`excerpt_mean` directions as steerable judicial circuits.
2. Stop running broad justice-level pokes on these exact averaged L16 directions.
3. Treat the broad `Scalia_vs_Ginsburg`/`Ginsburg_vs_not` L16 direction branch as decodable but not causally validated.
4. Do not promote the cue-heavy Article III or Fourth Amendment frame-contrast directions.
5. Move the next SCOTUS work to source-grounded frame labels, evaluator repair, or blind-review metric repair before more large steering runs.

Metric audit:

- Script: `scripts/experiments/scotus/audit_frame_metric.py`
- Report: `reports/scotus_frame_metric_audit_20260501.md`
- Main result: the keyword metric is useful for gates but not strong enough for a steering claim. Off-domain hits are often broad substring artifacts (`home`, `consent`, `damages`, `remedy`, `district`, generic separation-of-powers wording), and frame-pilot baselines are often already saturated.

### Phase 4.2g Source-Grounded Frame Seed

The first source-grounded frame-label seed now exists. It uses strict rules over real SCOTUS opinion chunks from the v2.1 inventory and keeps source URLs/evidence windows for review.

- Builder: `scripts/experiments/scotus/build_source_frame_labels.py`
- Labels: `data/scotus/scotus_source_frame_labels_v1.jsonl`
- Review queue: `data/scotus/scotus_source_frame_review_queue_v1.jsonl`
- Seed report: `reports/scotus_source_frame_seed_v1.md`
- Probe script: `scripts/experiments/scotus/probe_scotus_source_frames.py`
- Source probe run: `sweep_v4/scotus_source_frame_probe_20260501_003632/report.md`

Important seed result:

| Frame | Strict source labels |
| --- | ---: |
| `article3_public_rights` | `0` |
| `article3_private_rights` | `0` |
| `article3_article1_tribunal` | `6` |
| `article3_case_or_controversy` | `48` |
| `article3_final_judgment_separation` | `12` |
| `fourth_search_incident_chimel` | `13` |
| `fourth_plain_view_independent_source` | `18` |
| `fourth_home_exigency` | `13` |
| `fourth_technology_privacy` | `13` |

Source probe read:

| Task | Best readout | Dev BA | Test BA | Text test BA | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `article3_article1_vs_case` | `prompt_mean @ L16` | `1.000` | `0.500` | `0.500` | Not promoted |
| `article3_finality_vs_case` | `prompt_mean @ L12` | `0.571` | `0.429` | `0.429` | Reject |
| `fourth_home_vs_incident` | `prompt_last @ L16` | `0.750` | `0.500` | `1.000` | Text/leakage dominated |
| `fourth_plain_view_vs_incident` | `prompt_mean @ L8` | `1.000` | `0.143` | `0.357` | Reject |
| `fourth_technology_vs_incident` | `prompt_mean @ L12` | `0.833` | `0.500` | `1.000` | Text/leakage dominated |

Interpretation:

1. The earlier Article III public/private branch should not be pursued from the current target-justice corpus; strict source labeling finds no valid support.
2. The source-grounded probe did not produce a robust causal candidate. Some dev scores are high only because dev/test support is tiny, and the stronger Fourth Amendment rows are text-baseline dominated.
3. This strengthens the current bottleneck diagnosis: the problem is source/evaluator construction, not a shortage of steering pokes.

### Phase 4.2h Expanded Article III Source Pack

The Article III public/private-rights source gap has now been addressed as a separate source-pack branch. This branch uses named source opinions from Cornell LII rather than target-justice-only v2.1 chunks, keeps source URLs, and writes both raw and cue-masked text fields for leakage checks.

- Builder: `scripts/experiments/scotus/build_article3_source_pack.py`
- Raw source pages: `data/scotus/raw/scotus_article3_source_pages_v1.json`
- Labels: `data/scotus/scotus_article3_source_frame_labels_v1.jsonl`
- Review queue: `data/scotus/scotus_article3_source_frame_review_queue_v1.jsonl`
- Source-pack report: `reports/scotus_article3_source_pack_v1.md`
- Cue-masked probe run: `sweep_v4/scotus_source_frame_probe_20260501_005417/report.md`
- Dominance review queue: `data/scotus/scotus_article3_dominance_review_blind_v1.jsonl`
- Dominance review report: `reports/scotus_article3_dominance_review_v1.md`
- Dominance adjudication report: `reports/scotus_article3_dominance_adjudication_v1.md`
- Reviewed-label cue-masked probe: `sweep_v4/scotus_source_frame_probe_20260501_010535/report.md`

Source-pack label counts:

| Frame | Total | Train | Dev | Test | Public/private conflicts |
| --- | ---: | ---: | ---: | ---: | ---: |
| `article3_public_rights` | `72` | `38` | `27` | `7` | `16` |
| `article3_private_rights` | `39` | `28` | `3` | `8` | `28` |
| `article3_article1_tribunal` | `72` | `45` | `17` | `10` | `4` |
| `article3_case_or_controversy` | `11` | `6` | `4` | `1` | `2` |
| `article3_final_judgment_separation` | `30` | `27` | `2` | `1` | `0` |

Cue-masked Qwen3.5 source probe read:

| Task | Best readout | Dev BA | Test BA | Text test BA | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `article3_public_vs_private` | `prompt_last @ L16` | `0.625` | `0.500` | `0.500` | Reject |
| `article3_public_vs_article1` | `prompt_mean @ L8` | `1.000` | `0.969` | `0.969` | Text/leakage dominated |
| `article3_private_vs_article1` | `prompt_mean @ L16` | `0.938` | `0.333` | `0.750` | Not promoted |

Interpretation:

1. The expanded source pack makes Article III public/private-rights labels possible, but not yet adjudicated.
2. The direct public/private contrast is chance after cue masking and conflict filtering, so it does not nominate a direction.
3. The public-vs-Article-I result is decodable but text-baseline dominated even after cue masking, which means surface/legal-corpus signals remain too visible for a steering claim.
4. The private-vs-Article-I result is unstable after refit and has tiny positive test support, so it is not a candidate.
5. Next work here should be manual dominance review of the public/private queue or a better legal-frame evaluator, not another causal poke from these directions.

Dominance-review queue:

- Script: `scripts/experiments/scotus/build_article3_dominance_review_queue.py`
- Blind queue: `data/scotus/scotus_article3_dominance_review_blind_v1.jsonl`
- Key queue: `data/scotus/scotus_article3_dominance_review_key_v1.jsonl`
- Selected excerpts: `80`
- Public/private conflict excerpts: `28`
- Review labels: `public_rights_dominant`, `private_rights_dominant`, `article1_tribunal_dominant`, `mixed_comparative`, `off_target_or_false_positive`

Single-pass dominance adjudication:

- Script: `scripts/experiments/scotus/apply_article3_dominance_adjudication.py`
- Reviewed queue: `data/scotus/scotus_article3_dominance_review_adjudicated_v1.jsonl`
- Probe-ready labels: `data/scotus/scotus_article3_dominance_frame_labels_v1.jsonl`
- Reviewed rows: `80`
- Probe-ready rows: `70`
- Label counts: `33` public-rights dominant, `28` private-rights dominant, `9` Article-I/non-Article-III tribunal dominant, `10` mixed comparative excluded.

Reviewed-label cue-masked probe read:

| Task | Best readout | Dev BA | Test BA | Text test BA | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `article3_public_vs_private` | `prompt_mean @ L8` | `0.675` | `0.500` | `0.500` | Reject |
| `article3_public_vs_article1` | `prompt_mean @ L16` | `1.000` | `0.429` | `0.929` | Text/leakage dominated |
| `article3_private_vs_article1` | `prompt_mean @ L16` | `0.500` | `0.500` | `0.500` | Reject |

This closes the expanded Article III branch for now: after cue masking and dominance review, the public/private contrast still does not decode robustly from Qwen3.5 layers `8/12/16`, and the remaining Article-I diagnostic support is too sparse or too lexical to justify a causal steering run.

Next narrow follow-up:

1. Mine the completed Q4 proxy blind-review sample and the BF16 hook pilots for rubric failures, prompt/frame instability, and cases where the keyword metric disagrees with legal reading.
2. Do not run a causal poke from the expanded Article III public/private directions unless a second reviewer materially changes the label set.
3. Build a source-grounded evaluator that scores dominant legal frame rather than keyword presence before more frame-contrast generation.
4. Require any next candidate to beat prompt-matched same-layer random controls on both target-frame and target-minus-contrast metrics before any larger generation run.
5. Treat SCOTUS increasingly as a probe-validity benchmark unless a new issue family yields a reviewed cue-masked candidate with non-chance held-out support.

### Phase 4.2i Expanded Fourth Amendment Source Pack

The earlier Fourth Amendment source-frame result was contaminated by non-Fourth technology cases and obvious lexical cues, so the branch was rebuilt from named Fourth Amendment opinions before making a promotion decision.

- Builder: `scripts/experiments/scotus/build_fourth_source_pack.py`
- Raw source pages: `data/scotus/raw/scotus_fourth_source_pages_v1.json`
- Labels: `data/scotus/scotus_fourth_source_frame_labels_v1.jsonl`
- Review queue: `data/scotus/scotus_fourth_source_frame_review_queue_v1.jsonl`
- Source-pack report: `reports/scotus_fourth_source_pack_v1.md`
- Cue-masked probe run: `sweep_v4/scotus_source_frame_probe_20260501_011324/report.md`

Source-pack label counts:

| Frame | Total | Train | Dev | Test | Multi-frame conflicts |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fourth_search_incident_chimel` | `72` | `66` | `4` | `2` | `10` |
| `fourth_technology_privacy` | `72` | `52` | `14` | `6` | `8` |
| `fourth_plain_view_independent_source` | `72` | `68` | `2` | `2` | `8` |
| `fourth_home_exigency` | `72` | `40` | `2` | `30` | `7` |

Cue-masked Qwen3.5 source probe read:

| Task | Best readout | Dev BA | Test BA | Text test BA | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `fourth_home_vs_incident` | `excerpt_mean @ L8` | `0.952` | `1.000` | `1.000` | Text/leakage dominated; test has only `3` rows |
| `fourth_plain_view_vs_incident` | `prompt_mean @ L8` | `0.977` | `0.838` | `0.809` | Not promoted; marginal over text baseline and split-skewed |
| `fourth_technology_vs_home` | `prompt_mean @ L16` | `1.000` | `1.000` | `0.988` | Text/leakage dominated |
| `fourth_technology_vs_incident` | `prompt_mean @ L16` | `1.000` | `1.000` | `1.000` | Text/leakage dominated |

Interpretation:

1. The rebuilt pack fixes the gross non-Fourth-Amendment contamination from the first source seed, but it does not nominate a steerable direction.
2. Technology/privacy contrasts remain almost perfectly recoverable from text alone even after cue masking, so the model readout is not evidence of an internal circuit.
3. Home/exigency and plain-view contrasts have severe split/case imbalance, including tiny held-out positive or negative support in several comparisons.
4. Do not run a causal poke from these Fourth source directions. Use this branch as evaluator/probe-validity material unless a dominance-reviewed relabeling changes the held-out/text-baseline picture.

### Phase 4.2j Proposition-Level Evaluator Repair

The frame metric audit identified raw keyword scoring as a major source of false positives, so the completed proxy and BF16 frame-poke generations were rescored with stricter proposition-level rules.

- Script: `scripts/experiments/scotus/rescore_scotus_frame_propositions.py`
- Rescore run: `sweep_v4/scotus_frame_prop_rescore_20260501_012850/report.md`
- Inputs: Q4 proxy generation, Article III BF16 frame pilot, Fourth Amendment BF16 frame pilot
- Rows rescored: `3332`

Largest false-positive reductions:

| Frame | Old rows | Proposition rows | Dropped rows |
| --- | ---: | ---: | ---: |
| `separation_presidential_power` | `367` | `1` | `366` |
| `civil_equal_protection_strict_scrutiny` | `595` | `278` | `317` |
| `article3_article1_tribunal` | `807` | `502` | `308` |
| `article3_private_rights` | `459` | `275` | `239` |
| `article3_public_rights` | `427` | `364` | `180` |
| `fourth_home_exigency` | `240` | `115` | `132` |

Repaired BF16 frame-pilot read:

| Pilot | Alpha | N | Target z | Target win | Net z | Net win | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Article III | `0.02` | `4` | `-0.527` | `0.000` | `-0.741` | `0.000` | Reject |
| Article III | `0.05` | `4` | `0.757` | `0.500` | `0.511` | `0.500` | Not promoted |
| Article III | `0.10` | `4` | `0.027` | `0.500` | `0.222` | `0.750` | Not promoted |
| Fourth Amendment | `0.02` | `4` | `-0.323` | `0.000` | `-0.485` | `0.000` | Reject |
| Fourth Amendment | `0.05` | `4` | `1.146` | `0.500` | `0.880` | `0.500` | Weak hint only |
| Fourth Amendment | `0.10` | `4` | `-0.826` | `0.000` | `-1.190` | `0.000` | Reject |

Interpretation:

1. The repaired evaluator confirms that the original keyword scorer was too permissive, but it does not rescue any existing candidate.
2. The Fourth alpha `0.05` row remains the strongest signal, yet it is still a four-prompt effect with only `0.50` prompt win rate and a reversal at alpha `0.10`.
3. A disagreement-queue audit broadened the administrative-adjudication rules and narrowed the Article III final-judgment rule; the promotion decision did not change.
4. Future candidates should be screened with proposition-level target-minus-contrast before any expensive full-precision causal generation.

### Phase 4.2k Economic Activity Source Pack

The issue-family triage nominated Economic Activity as the next branch because the proxy null showed a stable Commerce Clause signal and a natural source contrast: broad aggregation / market-regulation reasoning versus Lopez / Morrison / NFIB-style limits.

- Builder: `scripts/experiments/scotus/build_economic_source_pack.py`
- Labels: `data/scotus/scotus_economic_source_frame_labels_v1.jsonl`
- Review queue: `data/scotus/scotus_economic_source_frame_review_queue_v1.jsonl`
- Source-pack report: `reports/scotus_economic_source_pack_v1.md`
- Probe report: `reports/scotus_economic_source_probe_20260501.md`
- BF16 probe run: `sweep_v4/scotus_source_frame_probe_20260501_014711/report.md`

Important repair:

- `probe_scotus_source_frames.py --reassign-task-splits` now assigns a single held-out split per source cluster within each task. The previous per-label reassignment could put the same source case in train for one label and test for the other, which was a case-identity leakage risk.

Cue-masked Qwen3.5 source probe read:

| Task | Best readout | Dev BA | Test BA | Text test BA | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `economic_broad_vs_limits` | `prompt_last @ L16` | `0.733` | `0.621` | `0.641` | Reject; activation does not beat text |
| `economic_broad_vs_state` | `prompt_mean @ L12` | `0.875` | `1.000` | `0.969` | Text/leakage dominated |
| `economic_limits_vs_state` | `prompt_mean @ L16` | `0.901` | `1.000` | `0.950` | Text/leakage dominated |
| `economic_preemption_vs_broad` | `prompt_mean @ L12` | `1.000` | `1.000` | `1.000` | Text/leakage dominated |

Interpretation:

1. The primary Commerce Clause broad-versus-limits contrast is not promoted because the activation result is below the cue-masked text baseline.
2. The state-regulation and preemption contrasts are separable, but text baselines are already near-perfect, so they are leakage/evaluator diagnostics rather than circuit evidence.
3. Do not run a causal poke from the current Economic Activity directions.
4. The next source-pack candidate is Civil Rights only if the labels are dominance-reviewed up front, because scrutiny-level labels are likely to be lexical.

### Phase 4.2l Civil Rights Source-Pack Gate

Civil Rights was the backup branch after Economic Activity failed. Because scrutiny labels are highly lexical, this branch was checked with a source pack and text-only gate before spending BF16 hook time.

- Builder: `scripts/experiments/scotus/build_civil_source_pack.py`
- Labels: `data/scotus/scotus_civil_source_frame_labels_v1.jsonl`
- Review queue: `data/scotus/scotus_civil_source_frame_review_queue_v1.jsonl`
- Source-pack report: `reports/scotus_civil_source_pack_v1.md`
- Gate report: `reports/scotus_civil_source_gate_20260501.md`

Source-pack support:

| Frame | Rows | Cases | Multi-frame conflicts |
| --- | ---: | ---: | ---: |
| `civil_race_strict_scrutiny` | `72` | `16` | `5` |
| `civil_sex_intermediate_scrutiny` | `72` | `12` | `13` |
| `civil_section5_congruence` | `72` | `21` | `2` |
| `civil_rational_basis_equal_protection` | `55` | `13` | `19` |

Cue-masked text-only gate:

| Task | Dev BA | Test BA | Decision |
| --- | ---: | ---: | --- |
| `civil_intermediate_vs_section5` | `0.971` | `1.000` | Text dominated |
| `civil_rational_vs_strict` | `0.955` | `1.000` | Text dominated |
| `civil_strict_vs_intermediate` | `0.748` | `0.969` | Text dominated |
| `civil_strict_vs_section5` | `1.000` | `0.964` | Text dominated |

Decision:

1. Do not run a BF16 activation probe on the current Civil Rights source pack.
2. The branch is useful as a leakage/evaluator stress test, not as a steering candidate.
3. A future Civil Rights attempt needs less lexical subframes and dominance review before probing.

### Phase 4.2m Justice-Style Slice Mining and BF16 Verification

After the source-frame branches failed, cached Phase 4.1 Qwen3.6-27B FP8 features were mined for justice-style slices where hidden-state probes beat a matched cue-masked text baseline.

- Mining script: `scripts/experiments/scotus/mine_scotus_slice_candidates.py`
- Mining report: `reports/scotus_slice_candidate_mining_20260501.md`
- Mining JSON: `reports/scotus_slice_candidate_mining_20260501.json`
- BF16 verification report: `reports/scotus_slice_bf16_majority2000s_20260501.md`
- BF16 run: `sweep_v4/scotus_slice_bf16_majority2000s_normal_20260501_022109/report.md`
- Label-shuffle null: `sweep_v4/scotus_slice_bf16_majority2000s_label_shuffle_20260501_022912/report.md`

The strongest mined slice was `section_posture=majority__decade=2000s`, with cached FP8 activation test BA `0.809` versus text test BA `0.500`.

Same-model Qwen3.5 BF16 verification read:

| Slice | Best readout | Dev BA | Test BA | Text test BA | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `section_posture=majority,decade=2000s` | `excerpt_mean @ L16` | `0.810` | `0.691` | `0.500` | Not promoted |

Label-shuffle null:

| Diagnostic | Best readout | Dev BA | Test BA | Sweep configs >= 0.70 |
| --- | --- | ---: | ---: | ---: |
| `label_shuffle` | `prompt_last @ L19` | `0.603` | `0.515` | `0` |

Interpretation:

1. The slice contains real activation structure; the label-shuffle null does not reproduce the normal-run scores.
2. It still misses the held-out BF16 promotion gate, and the selected test BA confidence interval is wide: `0.582-0.795`.
3. The held-out result is issue-fragile: the normal run's stress table reads `0.853` for Judicial Power but `0.588` for Criminal Procedure, and the dev/test issue mix is uneven.
4. Diagnostic prompt-last rows reached higher test scores, but they were not the dev-selected headline result and remain prompt-last/test-picking risks.

Decision:

1. Do not promote the majority-2000s slice to causal steering.
2. Treat it as evidence that some justice-style structure is present but not yet stable enough for a circuit claim.
3. The next justice-style follow-up, if any, should repair split stratification across issue family before more BF16 hook runs.

### Phase 4.2m Majority-2000s Feasible-Issues Refinement

The majority-2000s slice was refined to issue families with strict case-component split feasibility: `Criminal Procedure`, `Economic Activity`, and `Judicial Power`.

- Detailed report: `reports/scotus_slice_majority2000s_feasible_issues_20260501.md`
- Feasibility audit: `reports/scotus_slice_majority2000s_feasible_issues_split_feasibility_20260501.md`
- Normal component resplits: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_normal_component_resplits_20260501_034116/report.md`
- Label-shuffle component resplits: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_label_shuffle_component_resplits_20260501_034539/report.md`
- Template-variant component resplits: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_template_variant_component_resplits_20260501_040538/report.md`
- Plain-prompt component resplits: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_plain_prompt_component_resplits_20260501_041503/report.md`
- Excerpt-removed component resplits: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_excerpt_removed_component_resplits_20260501_042048/report.md`
- Neutral-filler component resplits: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_neutral_filler_component_resplits_20260501_043234/report.md`

Component-resplit read:

| Diagnostic | Median dev BA | Median test BA | Test BA range | Primary split test BA | Median text test BA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal | `0.812` | `0.746` | `0.660-0.753` | `0.753` | `0.700` |
| Label shuffle | `0.536` | `0.541` | `0.477-0.548` | `0.488` | `0.492` |
| Template variant | `0.807` | `0.758` | `0.668-0.807` | `0.777` | `0.695` |
| Plain prompt | `0.818` | `0.764` | `0.676-0.796` | `0.777` | `0.691` |
| Excerpt removed | `0.500` | `0.500` | `0.500-0.500` | `0.500` | `0.500` |
| Neutral filler | `0.575` | `0.542` | `0.512-0.564` | `0.548` | `0.564` |

Interpretation:

1. This refined slice is stronger than the original all-issue majority-2000s run: normal/template/plain-prompt diagnostics stay in the mid-0.7s under the same component plans.
2. The signal is not reproduced by label shuffle, excerpt removal, or same-shaped neutral filler.
3. The evidence is still correlational and prompt-last-heavy. Text baselines are close on some split plans, especially the primary split.
4. This branch is therefore live for a small causal diagnostic, but it is not yet a steerable judicial circuit.

Next gate:

1. Run a preregistered causal patch or steering pilot on `prompt_last @ L10`, `excerpt_mean @ L16`, and secondary `prompt_last @ L19` directions.
2. Use held-out legal prompts, prompt-matched same-layer random controls, both signs, small norm-scaled alphas, and blind/manual review of frame movement.
3. Promote only if the direction causes reproducible jurisprudential-frame movement beyond random controls without simply copying case/topic or increasing generic legal verbosity.

Causal update:

- Detailed causal pilot report: `reports/scotus_majority2000s_feasible_issues_causal_pilot_20260501.md`
- `prompt_last @ L10` positive Ginsburg direction, last-token hook: `sweep_v4/scotus_sae_poke_20260501_045156/report.md`
- `excerpt_mean @ L16` positive Ginsburg direction, all-position hook: `sweep_v4/scotus_sae_poke_20260501_060425/report.md`

| Direction | Position | Alphas | Random controls | Best prompt-matched target z | Best prompt-matched net z | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `prompt_last @ L10` | `last` | `0.02,0.05,0.1` | `10` | `0.184` | `0.533` | not promoted |
| `excerpt_mean @ L16` | `all` | `0.01,0.02,0.05` | `5` | `0.449` | `0.395` | not promoted |

Read: the refined slice remains a valid decodability result, but these two broad causal pokes do not establish a steerable judicial circuit.

Prompt-pocket review queue:

- Report: `reports/scotus_majority2000s_causal_prompt_pockets_20260501.md`
- Blind queue: `data/scotus/scotus_majority2000s_causal_review_blind_20260501.jsonl`
- Key file: `data/scotus/scotus_majority2000s_causal_review_key_20260501.jsonl`

The queue selected `8` candidate cells and `22` candidate-vs-baseline/random pairwise comparisons. It should be used only to decide whether any narrower prompt family deserves a targeted follow-up; it is not promotion evidence.

Adjudication update:

- Report: `reports/scotus_majority2000s_causal_review_adjudication_20260501.md`
- Adjudicated rows: `data/scotus/scotus_majority2000s_causal_review_adjudicated_20260501.jsonl`

Only `EA03_gun_school_zone` and `EA01_commercial_remedy` survive the internal pairwise rule. Both are Economic Activity / Commerce Clause pockets. Judicial Power pockets were rejected because strongest random controls matched or beat the candidate.

### Phase 4.2n Economic Activity Justice-Style Resplit Audit

The next mined justice-style candidate, `issue_area_label=Economic Activity`, was recaptured on Qwen3.5 BF16 and then audited with stricter case-component resplits.

- Report: `reports/scotus_slice_economic_style_bf16_20260501.md`
- Original BF16 run: `sweep_v4/scotus_slice_bf16_economic_style_normal_20260501_023619/report.md`
- Original label-shuffle null: `sweep_v4/scotus_slice_bf16_economic_style_label_shuffle_20260501_024011/report.md`
- Split feasibility audit: `reports/scotus_slice_economic_split_feasibility_20260501.md`
- Resplit runner: `scripts/experiments/scotus/resplit_cached_scotus_probe.py`
- Normal component resplits: `sweep_v4/scotus_slice_bf16_economic_style_normal_component_resplits_20260501_024704/report.md`
- Label-shuffle component resplits: `sweep_v4/scotus_slice_bf16_economic_style_label_shuffle_component_resplits_20260501_025016/report.md`
- Excerpt-removed component resplits: `sweep_v4/scotus_slice_bf16_economic_style_excerpt_removed_component_resplits_20260501_025835/report.md`
- Neutral-filler component resplits: `sweep_v4/scotus_slice_bf16_economic_style_neutral_filler_component_resplits_20260501_030511/report.md`
- Template-variant original probe: `sweep_v4/scotus_slice_bf16_economic_style_template_variant_20260501_031750/report.md`
- Template-variant component resplits: `sweep_v4/scotus_slice_bf16_economic_style_template_variant_component_resplits_20260501_032205/report.md`
- Plain-prompt original probe: `sweep_v4/scotus_slice_bf16_economic_style_plain_prompt_20260501_032236/report.md`
- Plain-prompt component resplits: `sweep_v4/scotus_slice_bf16_economic_style_plain_prompt_component_resplits_20260501_032744/report.md`
- Split component review: `reports/scotus_economic_split_component_review_20260501.md`

Original BF16 read:

| Slice | Best readout | Dev BA | Test BA | Text test BA | Test N |
| --- | --- | ---: | ---: | ---: | ---: |
| `issue_area_label=Economic Activity` | `prompt_last @ L24` | `0.875` | `1.000` | `0.700` | `30` |

The original result was too strong to ignore but too small to trust: the test split had only `15` examples per label. The slice has `8` case-connected components, so a stricter component resplit was feasible.

Component-resplit read:

| Diagnostic | Median dev BA | Median test BA | Test BA range | Primary split test BA |
| --- | ---: | ---: | ---: | ---: |
| Normal | `0.795` | `0.743` | `0.690-0.856` | `0.703` |
| Label shuffle | `0.568` | `0.500` | `0.471-0.542` | `0.480` |
| Excerpt removed | `0.500` | `0.500` | `0.500-0.500` | `0.500` |
| Neutral filler | `0.601` | `0.554` | `0.500-0.576` | `0.554` |

Prompt-template invariance gate:

| Mode | Original best readout | Original test BA | Component median test BA | Primary split test BA | Median text test BA |
| --- | --- | ---: | ---: | ---: | ---: |
| Normal chat template | `prompt_last @ L24` | `1.000` | `0.743` | `0.703` | `0.637` |
| Template variant | `excerpt_mean @ L24` | `0.967` | `0.690` | `0.676` | `0.649` |
| Plain prompt | `excerpt_mean @ L24` | `0.933` | `0.673` | `0.655` | `0.649` |

Interpretation:

1. The slice has real activation structure: normal resplits beat label-shuffle, excerpt-removed, and neutral-filler controls.
2. It is not a constant-prompt artifact; excerpt removal is exactly chance under every component split.
3. It is not primarily length/position leakage; neutral filler remains much lower than normal.
4. It is not prompt-template invariant under strict component resplits. The original template and plain-prompt runs selected `excerpt_mean @ L24`, but once the same component plans are applied, median test BA falls below `0.70` and the selected directions return to `prompt_last`.
5. The split component review points toward case/topic structure in the high text splits; high-weight text features include `volvo`, `reeder`, `articles`, `maritime`, `fcc`, `the epa`, `arbitration`, and `antitrust`.
6. It is still not steering-ready. The strict-resplit effect is modest, text baselines are close on multiple split plans, and the selected BF16 resplit directions are mostly `prompt_last @ L8/L12/L16/L19/L20/L24` rather than stable excerpt-internal readouts.

Decision:

1. Do not promote the Economic Activity justice-style slice to causal steering yet.
2. Keep it as evidence that broad justice-style information is decodable, but reject it as the next causal hook candidate.
3. Next useful work is to define a narrower source-grounded Economic Activity contrast if this branch continues.

Compute placement:

- Operating rule: keep available hardware working toward the steerable-circuit goal either directly through evidence-producing runs or indirectly through null/control generation, audits, and blind review sampling.
- Local RTX Pro 6000 remains the source of record for full-precision Qwen3.5/Qwen3.6 27B activation evidence.
- The dev server at `192.168.1.90` can be used for 4-bit proxy experiments on the 3090/4090, but those runs must be labeled proxy-only and must not be treated as final evidence for a steerable circuit.
- Do not store credentials in project notes or committed files.
