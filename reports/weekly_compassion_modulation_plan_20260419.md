# Weekly Research Plan: Compassion Modulation Without Persona Collapse

## Executive Summary

### Current state
- We now have a repaired expanded symphonic corpus:
  - [symphonic_voice_probe_dataset_v2_repaired_20260418_072722](/home/orwel/dev_genius/experiments/Character%20Creation/sweep_v4/symphonic_voice_probe_dataset_v2_repaired_20260418_072722)
- We have a strong mechanistic readout on that corpus:
  - [symphonic_voice_activation_probe_v2_repaired_20260418_073715](/home/orwel/dev_genius/experiments/Character%20Creation/sweep_v4/symphonic_voice_activation_probe_v2_repaired_20260418_073715)
- We have an updated axis map in common late-`think` space:
  - [symphonic_voice_axis_analysis_v2_repaired_20260418_082543](/home/orwel/dev_genius/experiments/Character%20Creation/sweep_v4/symphonic_voice_axis_analysis_v2_repaired_20260418_082543)
- We have live compositional patch results:
  - [symphonic_voice_live_patch_v2_compositional_20260418_082730](/home/orwel/dev_genius/experiments/Character%20Creation/sweep_v4/symphonic_voice_live_patch_v2_compositional_20260418_082730)

### What we learned
- `think_mean @ L39` on the repaired `v2` corpus is the best mechanistic target for executed stance.
- `prompt_last` still carries a strong intended-stance prior, but the late `think` state is the better intervention point.
- `jesus -> hitchens` and `jesus -> mark_twain` work as modulations:
  - target probability rises
  - bluntness / pragmatism / edge rise
  - source identity is preserved
- `hitchens -> jesus` and `hitchens -> mother_teresa` are weak or adversarial:
  - little to no compassionate reclassification
  - some patches make the output even more Hitchens-like

### Main interpretation
- This is good news.
- We do **not** want transplants. We want **modulations**.
- The asymmetry means compassion is not obviously just `-irony` or `-harshness`.
- Our **working hypothesis**, not yet a conclusion, is that compassion behaves like an entangled control stack involving at least:
  - `+compassion`
  - `+transcendence`
  - `-severity`
  - `-punitive framing`
  - context-sensitive gating
- This week therefore includes explicit falsification checks for simpler explanations:
  - weaker target-vector norm
  - deeper or stickier sharp-source basin
  - intervention applied too late
  - need for two-site intervention (`prompt_last` + `think_mean`) rather than a richer late-`think` composition

### Tool decision for the week
- Stay with:
  - linear probes
  - axis regressors
  - orthogonalized directions
  - causal patching
  - live generation tests
- Do **not** move to SAEs this week unless the week ends with clear evidence that linear/composed controls have saturated.

### Falsification targets
The week should not be counted as support for the entangled-stack story unless these cheaper alternatives are checked first:
- `norm mismatch`: compassionate anchors may simply be lower-norm or lower-separation in the patch space
- `basin depth`: sharp-source anchors may have deeper attraction in late `think`
- `timing error`: compassion may require earlier or narrower intervention than edge
- `two-site error`: the real fix may require a coupled prompt-side plus late-`think` intervention

### Week objective
Find a **composed, competence-preserving compassion patch** that softens sharp-source outputs without causing persona collapse or obvious reasoning loss.

---

## Constraints

- One task per day.
- Max wall time per card per day:
  - Blackwell: `<= 8h`
  - 4090: `<= 8h`
  - 3090: `<= 8h`
- Total max wall time per day across all cards: `<= 24h`
- If a task finishes early:
  - use the remainder for scoring, plots, and report writing only
  - do not branch into a new experiment the same day

---

## Shared Inputs

### Canonical artifacts
- Dataset:
  - [symphonic_voice_probe_dataset_v2_repaired_20260418_072722](/home/orwel/dev_genius/experiments/Character%20Creation/sweep_v4/symphonic_voice_probe_dataset_v2_repaired_20260418_072722)
- Feature bundle:
  - [symphonic_voice_activation_probe_v2_repaired_20260418_073715](/home/orwel/dev_genius/experiments/Character%20Creation/sweep_v4/symphonic_voice_activation_probe_v2_repaired_20260418_073715)
- Axis analysis:
  - [symphonic_voice_axis_analysis_v2_repaired_20260418_082543](/home/orwel/dev_genius/experiments/Character%20Creation/sweep_v4/symphonic_voice_axis_analysis_v2_repaired_20260418_082543)
- Live compositional patch run:
  - [symphonic_voice_live_patch_v2_compositional_20260418_082730](/home/orwel/dev_genius/experiments/Character%20Creation/sweep_v4/symphonic_voice_live_patch_v2_compositional_20260418_082730)

### Core scripts
- [build_symphonic_probe_dataset.py](/home/orwel/dev_genius/experiments/Character%20Creation/scripts/experiments/personality/build_symphonic_probe_dataset.py)
- [probe_symphonic_voice_states.py](/home/orwel/dev_genius/experiments/Character%20Creation/scripts/experiments/personality/probe_symphonic_voice_states.py)
- [analyze_symphonic_voice_axes.py](/home/orwel/dev_genius/experiments/Character%20Creation/scripts/experiments/personality/analyze_symphonic_voice_axes.py)
- [live_patch_symphonic_voice.py](/home/orwel/dev_genius/experiments/Character%20Creation/scripts/experiments/personality/live_patch_symphonic_voice.py)

### Shared shell variables
```bash
export EXP_ROOT="/home/orwel/dev_genius/experiments/Character Creation"
export VENV="/home/orwel/dev_genius/venv/bin/python"
export WS_MODEL="/home/orwel/dev_genius/models/Qwen3.6-35B-A3B"
export WS_BASE_URL="http://127.0.0.1:30003/v1"
export DEV_3090="http://192.168.1.90:30001/v1"
export DEV_4090="http://192.168.1.90:30002/v1"
export DEV_MODEL="Qwen/Qwen3.5-9B"
export V2_DATASET="$EXP_ROOT/sweep_v4/symphonic_voice_probe_dataset_v2_repaired_20260418_072722"
export V2_FEATURES="$EXP_ROOT/sweep_v4/symphonic_voice_activation_probe_v2_repaired_20260418_073715"
export V2_AXES="$EXP_ROOT/sweep_v4/symphonic_voice_axis_analysis_v2_repaired_20260418_082543"
export MANIFEST_V2="$EXP_ROOT/data/symphonic_voice_anchor_manifest_v2.json"
```

### Operational note
- If the dev endpoints are down, relaunch the stable 9B SGLang servers before the day’s generation job. That startup time counts against the day budget.

---

## Planned Small Helper Scripts

These are plumbing, not new research methods.

### Day 1 helper
- `extract_symphonic_reverse_subset.py`
- Purpose:
  - read reverse-direction patch records
  - materialize a curated source subset for target-generation jobs

### Day 3 helper
- `live_patch_symphonic_axes.py`
- Purpose:
  - fork of [live_patch_symphonic_voice.py](/home/orwel/dev_genius/experiments/Character%20Creation/scripts/experiments/personality/live_patch_symphonic_voice.py)
  - accept arbitrary weighted axis vectors from `axis_vectors_common_space.npz`
  - patch composed directions such as:
    - `+compassion +transcendence -severity -irony`

If either helper is not implemented within the first `60-90m` of its day, fall back to the simpler evaluation path listed under that day’s stop conditions.

### Day 7 template
- Prewritten decision memo template:
  - [weekly_compassion_modulation_decision_template_20260419.md](/home/orwel/dev_genius/experiments/Character%20Creation/reports/weekly_compassion_modulation_decision_template_20260419.md)

---

## Daily Plan

### Day 1
**Task:** Build the reverse-direction failure pack

**Question**
- Where exactly do `hitchens -> jesus` and `hitchens -> mother_teresa` fail, and what clean compassionate targets should they be compared against?

**Compute budget**
- Blackwell: `6h`
- 4090: `5h`
- 3090: `5h`
- Total: `16h`

**Blackwell**
- Expand the reverse patch run from `4` rows per pair to `12`.
- Command:
```bash
$VENV "$EXP_ROOT/scripts/experiments/personality/live_patch_symphonic_voice.py" \
  --dataset-dir "$V2_DATASET" \
  --features-dir "$V2_FEATURES" \
  --axis-analysis-dir "$V2_AXES" \
  --anchor-manifest "$MANIFEST_V2" \
  --model-path "$WS_MODEL" \
  --tag "symphonic_reverse_failure_pack_v1" \
  --pairs 'hitchens:jesus,hitchens:mother_teresa' \
  --max-rows-per-pair 12 \
  --alphas '0.25,0.5,0.75' \
  --patch-after-tokens 48 \
  --patch-token-limit 96 \
  --max-new-tokens 512 \
  --dtype bfloat16 \
  --max-vram-frac 0.90
```

**CPU prep inside Day 1**
- Materialize:
  - `sweep_v4/symphonic_reverse_subset_v1`
- Include:
  - reverse failures
  - partial reversals
  - neutral matched rows
- Run a falsification sanity pack:
  - `reverse_basin_sanity_v1`
- Compute:
  - source and target centroid norms in common space
  - pairwise delta norms
  - per-anchor classification margin / mean self-probability
  - simple basin-depth proxy:
    - how quickly same-anchor confidence recovers after `alpha=0.25/0.50/0.75` patching
  - prompt-side vs late-`think` separability comparison for the same reverse rows
  - whether a synthetic two-site score (`prompt_last + think_mean`) improves reverse discrimination without live generation

**4090**
- Generate compassionate target completions on the curated subset.
- Use a mini manifest derived from `v2` containing:
  - `jesus`
  - `mother_teresa`
  - `fred_rogers`
  - `neutral_competent`
- Command:
```bash
$VENV "$EXP_ROOT/scripts/experiments/personality/build_symphonic_probe_dataset.py" \
  --source-dataset-dir "$EXP_ROOT/sweep_v4/symphonic_reverse_subset_v1" \
  --anchor-manifest "$EXP_ROOT/data/symphonic_voice_anchor_manifest_compassion_targets_v1.json" \
  --base-url "$DEV_4090" \
  --api-model "$DEV_MODEL" \
  --tag "symphonic_reverse_targets_4090_v1" \
  --items-per-behavior 12 \
  --min-pair-quality 3 \
  --max-workers 8 \
  --seed 17 \
  --timeout 1200
```

**3090**
- Same target-generation job with a different seed:
```bash
$VENV "$EXP_ROOT/scripts/experiments/personality/build_symphonic_probe_dataset.py" \
  --source-dataset-dir "$EXP_ROOT/sweep_v4/symphonic_reverse_subset_v1" \
  --anchor-manifest "$EXP_ROOT/data/symphonic_voice_anchor_manifest_compassion_targets_v1.json" \
  --base-url "$DEV_3090" \
  --api-model "$DEV_MODEL" \
  --tag "symphonic_reverse_targets_3090_v1" \
  --items-per-behavior 12 \
  --min-pair-quality 3 \
  --max-workers 8 \
  --seed 29 \
  --timeout 1200
```

**Deliverables**
- `symphonic_reverse_failure_pack_v1`
- `symphonic_reverse_subset_v1`
- `symphonic_reverse_targets_4090_v1`
- `symphonic_reverse_targets_3090_v1`
- `reverse_basin_sanity_v1`

**Success criterion**
- `>= 40` clean reverse-failure or partial-reversal rows
- `>= 40` matched compassionate target rows
- explicit answer to:
  - "are compassionate targets materially lower-norm / lower-margin in the patch space?"
  - "does the reverse failure already look like a timing or basin problem before any composed-direction work?"

**Stop condition**
- If target generation quality is too low, do not keep expanding.
- Use only the highest-quality compassionate targets and move to Day 2.

---

### Day 2
**Task:** Learn direct trait directions from the matched reverse corpus

**Question**
- Which late-`think` traits move between sharp-source and compassionate-target completions, independent of anchor labels?

**Compute budget**
- Blackwell: `6h`
- 4090: `4h`
- 3090: `4h`
- Total: `14h`

**Blackwell**
- Merge the Day 1 curated target set into a single dataset:
  - `sweep_v4/symphonic_reverse_targets_merged_v1`
- Run hidden-state extraction and probe search:
```bash
$VENV "$EXP_ROOT/scripts/experiments/personality/probe_symphonic_voice_states.py" \
  --dataset-dir "$EXP_ROOT/sweep_v4/symphonic_reverse_targets_merged_v1" \
  --model-path "$WS_MODEL" \
  --tag "symphonic_reverse_targets_probe_v1" \
  --device-map auto \
  --max-gpu-gib 72 \
  --max-cpu-gib 24 \
  --offload-folder "$EXP_ROOT/tmp_offload_reverse_targets_v1" \
  --region-allowlist 'think_mean,assistant_mean,response_mean,prompt_last' \
  --layer-stride 1 \
  --c-grid '0.25,1.0'
```

**Blackwell then CPU**
- Fit direct axis regressors on the new feature bundle:
```bash
$VENV "$EXP_ROOT/scripts/experiments/personality/analyze_symphonic_voice_axes.py" \
  --features-dir "$EXP_ROOT/sweep_v4/symphonic_reverse_targets_probe_v1_*LATEST*" \
  --anchor-manifest "$EXP_ROOT/data/symphonic_voice_anchor_manifest_compassion_targets_v1.json" \
  --tag "symphonic_reverse_axes_v1" \
  --region-allowlist 'think_mean,assistant_mean,response_mean,prompt_last' \
  --layer-stride 1 \
  --common-region 'think_mean' \
  --common-layer 39 \
  --common-clf-c 0.25 \
  --patch-alphas '0.25,0.5,1.0'
```

**4090 / 3090**
- Only run fill jobs if Day 1 left any behavior under `10` rows per target anchor.
- Otherwise reserve time for scoring and QA only.

**Deliverables**
- `symphonic_reverse_targets_probe_v1`
- `symphonic_reverse_axes_v1`

**Success criterion**
- Strong held-out late-`think` decodability for at least:
  - compassion
  - transcendence
  - severity
  - irony
- Day 2 -> Day 3 gate for `think_mean @ L39` on the matched reverse corpus:
  - `compassion` must satisfy either:
    - held-out `R² >= 0.35` and `pearson >= 0.75`, or
    - top-vs-bottom quartile binary AUC `>= 0.80`
- If the gate fails:
  - Day 3 becomes `warmth-anchor augmentation + re-probe`
  - do **not** proceed to composed-direction patching yet

**Stop condition**
- If the merged target set is too noisy, keep the best `q3+` slice only and proceed with a smaller clean corpus.

---

### Day 3
**Task:** Orthogonalize the knobs

**Question**
- Can we separate compassion from correlated source traits instead of using full anchor swaps?

**Compute budget**
- Blackwell: `6h`
- 4090: `3h`
- 3090: `3h`
- Total: `12h`

**CPU prep inside Day 3**
- Add:
  - `live_patch_symphonic_axes.py`
- This should accept weighted axis vectors from:
  - `axis_vectors_common_space.npz`

**Blackwell**
- Compose and test these candidate directions in feature space first:
  - `+compassion`
  - `+compassion -irony`
  - `+compassion -severity`
  - `+compassion +transcendence -severity -irony`
- Add controls from the start:
  - same-anchor null patch
  - scrambled-direction patch
  - plain `hitchens -> jesus` anchor delta
- Use `symphonic_reverse_axes_v1` and `symphonic_voice_axis_analysis_v2_repaired_20260418_082543` as the source of vectors.

**4090 / 3090**
- Run lightweight held-out evaluation on the same vectors:
  - does the direction move the intended axis more than unrelated axes?
  - does it preserve competence tags better than anchor-delta patches?

**Deliverables**
- `orthogonalized_compassion_directions_v1`
- ranking of composed directions by:
  - compassion gain
  - irony suppression
  - competence preservation

**Success criterion**
- At least one composed direction must beat the plain `hitchens -> jesus` anchor delta by all of:
  - mean reverse-pair `target_prob` lift `>= +0.010` absolute
  - mean compassion-axis lift `>= +0.050`
  - mean irony change `<= 0.0`
  - mean severity change `<= 0.0`
  - null and scrambled controls stay within `|delta target_prob| < 0.003`

**Stop condition**
- If the helper script is not ready within `90m`, fallback to feature-space patch evaluation only and push live patching to Day 4.

---

### Day 4
**Task:** Timing sweep for composed compassion patches

**Question**
- Does compassion need a different patch timing than sarcasm/edge?

**Compute budget**
- Blackwell: `6h`
- 4090: `4h`
- 3090: `4h`
- Total: `14h`

**Blackwell**
- Use `live_patch_symphonic_axes.py` on the best `1-2` composed directions.
- Sweep:
  - `patch_after_tokens`: `16, 32, 48, 64`
  - `patch_token_limit`: `32, 64, 96`
- Target pairs:
  - `hitchens -> jesus`
  - `hitchens -> mother_teresa`

**4090 / 3090**
- Run smaller replay-only timing sweeps on the same reverse items to estimate timing sensitivity cheaply.
- Positive control on the dev cards:
  - rerun `jesus -> hitchens` timing sweep with the same timing grid
  - use it as a control for whether timing effects are compassion-specific or general late-`think` patch behavior

**Deliverables**
- `compassion_timing_sweep_v1`

**Success criterion**
- Identify whether the compassionate patch works best:
  - earlier in `think`
  - later in `think`
  - or in a narrow pulse rather than a broad window
- Positive-control check:
  - the timing regime chosen for reverse compassion must not simply be the one that maximizes any patch effect on the forward pair

**Stop condition**
- If timing does not matter, lock the best late-`think` window and move on.

---

### Day 5
**Task:** Social gating eval

**Question**
- Does the patch make the model selectively more humane, or just globally softer?

**Compute budget**
- Blackwell: `6h`
- 4090: `5h`
- 3090: `5h`
- Total: `16h`

**Setup**
- Partition a clean eval slice into:
  - vulnerable target
  - arrogant / hostile target
  - neutral target

**Blackwell**
- Run live patching with the best composed direction and best timing from Day 4.

**4090 / 3090**
- Generate fill items only if any gating bucket is underpowered.
- Otherwise score outputs and compute rubric deltas.

**Deliverables**
- `social_gating_eval_v1`

**Success criterion**
- Pre-registered thresholds:
  - vulnerable-target bucket:
    - mean compassion-axis lift `>= +0.08`
  - arrogant / hostile bucket:
    - mean bluntness or irony drop no larger than `20%` relative to source baseline
  - neutral bucket:
    - mean compassion-axis lift must be `< 50%` of the vulnerable-target lift
- If the patch makes neutral and vulnerable buckets move equally, it fails the gating test

**Stop condition**
- If the patch becomes globally nice, reject it and keep the harsher but more selective candidate.

---

### Day 6
**Task:** Competence-preservation benchmark

**Question**
- Does the compassion patch improve stance without making the model worse at doing the actual task?

**Compute budget**
- Blackwell: `6h`
- 4090: `4h`
- 3090: `4h`
- Total: `14h`

**Blackwell**
- Run the best composed direction on:
  - `constraint_preservation`
  - `repair_after_challenge`
  - `selective_introspection`
  - a small structured reasoning slice

**4090**
- Run sham patch controls:
  - same timing
  - same token budget
  - random or irrelevant direction

**3090**
- Score:
  - format
  - competence
  - trait deltas
  - failure modes

**Deliverables**
- `compassion_competence_benchmark_v1`

**Success criterion**
- measurable compassionate shift plus all of:
  - format-ok rate `>= 0.95`
  - structured reasoning / repair accuracy drop `<= 5` absolute percentage points
  - no sham control matching the full compassionate shift
  - compassionate-axis lift retains at least `50%` of the Day 5 vulnerable-target gain

**Stop condition**
- If every compassionate direction costs too much competence, that becomes the week’s main negative result.

---

### Day 7
**Task:** Decision memo and next-tool gate

**Question**
- Are the simpler linear/composed controls still enough, or is this now the point where SAEs become justified?

**Compute budget**
- Blackwell: `<= 2h`
- 4090: `<= 2h`
- 3090: `<= 2h`
- Total: `<= 6h`

**Work**
- Compile:
  - best composed directions
  - best timing
  - best social gating result
  - competence tradeoffs
  - recurring reverse-direction failure modes

**Decision options**
- `continue with linear/composed control`
- or `move to targeted SAE work on late-think layers`

**Deliverables**
- `weekly_compassion_modulation_report_v1`
- `sae_gate_decision_memo_v1`

**Success criterion**
- A crisp recommendation with evidence, not a vague “maybe”.

---

## Daily Output Tags

| Day | Planned Tag |
| --- | --- |
| 1 | `symphonic_reverse_failure_pack_v1` |
| 1 | `symphonic_reverse_targets_4090_v1` |
| 1 | `symphonic_reverse_targets_3090_v1` |
| 2 | `symphonic_reverse_targets_probe_v1` |
| 2 | `symphonic_reverse_axes_v1` |
| 3 | `orthogonalized_compassion_directions_v1` |
| 4 | `compassion_timing_sweep_v1` |
| 5 | `social_gating_eval_v1` |
| 6 | `compassion_competence_benchmark_v1` |
| 7 | `weekly_compassion_modulation_report_v1` |

---

## Why No SAEs Yet

### Reason
- We already have real signal from:
  - late-`think` probes
  - axis regressors
  - live patching
- The next ambiguity is still about:
  - direction composition
  - timing
  - social gating
  - competence preservation
- SAEs are justified only if those simpler controls clearly saturate.

### The gate
Move to targeted SAE work only if, by Day 7:
- composed directions still cannot produce a clean reverse compassionate shift
- the same source-trait entanglements stay sticky
- timing sweeps do not rescue the reverse direction

If that happens, the SAE target should be:
- late `think`
- around the current intervention neighborhood
- likely `L36-L39` / `block_35` through `block_38`

---

## Expected Outcomes

### Best case
- We get a real compassion knob:
  - composed
  - timing-sensitive
  - socially gated
  - competence-preserving

### More realistic case
- We get partial reverse-direction success
- We identify exactly which trait bundle is still sticky
- We can then justify or reject SAE work with evidence

Either outcome is useful.

---

## Questions For Mentor Review

1. Is the proposed week correctly focused on **modulation** rather than **persona transplant**?
2. Do the Day 1 norm / basin checks justify adding one more warmth anchor before Day 2, or not?
3. Is the Day 5 social-gating test the right criterion for “maskless compassion”?
4. Is the SAE gate strict enough, or should targeted SAE work start earlier?
