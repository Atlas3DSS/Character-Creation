# Activation Probe Runbook

This file reconstructs how we actually did activation probing in the recent project runs. It is meant to be operational: if you need to recreate a probe, start here, then open the referenced script.

## Short Version

Our standard activation probe was:

1. Build a labeled dataset with stable `train` / `val` or `dev` / `test` splits.
2. Render each example through the model's chat template.
3. Run a HuggingFace forward pass with `output_hidden_states=True` or forward hooks.
4. Extract one vector per example per `(region, layer)`.
5. Store vectors in `features.npz` as `region__LNN`.
6. Store metadata in `feature_meta.jsonl`.
7. Fit `StandardScaler + LogisticRegression(class_weight="balanced")` for every `(region, layer, C)`.
8. Select the best probe by validation balanced accuracy, with F1 as the tie-breaker.
9. Refit the selected probe on `train + val` and report final test metrics.
10. If doing causal work, save either the logistic probe weights or a train-only mean-difference direction.

Important: vLLM was not used for activation capture. We used HuggingFace because we needed hidden states or forward hooks. vLLM's paged-attention serving path and custom kernels are optimized for generation throughput, not for exposing stable intermediate activations at arbitrary layers.

## Qwen Generation Budget Rule

When a probe, patch, poke, scorer, or review queue depends on generated Qwen legal reasoning, a few hundred generated tokens is smoke/debug only. Qwen often spends thousands of tokens before the legal holding stabilizes.

Future SCOTUS/Qwen run constructors must use `scripts/experiments/scotus/qwen_eval_budget.py`:

- Default answer or max-new-token budgets to `DEFAULT_COMPLETE_ANSWER_TOKENS` (`3072` today).
- Require at least `MIN_COMPLETE_ANSWER_TOKENS` (`2048`) for final legal holdings, visible-reasoning traces, scorer calibration, promotion decisions, and learned-result claims.
- Apply the same minimum to visible-thinking budgets when the trace itself is being interpreted.
- Require an explicit short-budget opt-in for smoke/debug runs, and write `short_answer_budget`, `short_thinking_budget`, `budget_note`, and `promotion_eligible_budget` into manifests and reports.
- Treat any legacy run with unknown or sub-`2048` generation budgets as non-promotion evidence until regenerated.

## No-Mask Success Constraint

Activation probes are evidence, not the endpoint. The project goal is not to make a model perform a prompted imitation of a target voice, justice, or reasoning style.

A successful intervention must shift the model's response and reasoning basin directly. Where `<thinking>` or another scratchpad trace is visible, the trace should reason in the target frame rather than say, implicitly or explicitly, that the model is imitating how the target would reason.

Prompt-only role-play, persona instructions, "think like X" prompts, and prompt-template tricks are diagnostic baselines only. LoRA, ReFT, SFT, or other learned interventions are also diagnostic or last-resort tools unless they reveal or create a durable shift that can be made permanent in the model. They should not be treated as success if they merely wrap the old model in a better mask.

## Interpretive Status

Treat the reported probe accuracies in this file as reconstruction notes, not settled claims.

The strongest methodological concerns from the prior runs are:

1. `L0` and `prompt_last` wins are suspicious.
   - A near-input readout winning an 8-way voice task can mean the probe is reading prompt-format artifacts, anchor names, style labels, or other surface text.
   - Symphonic Voice selected `prompt_last @ L0`.
   - Several behavior-local book probes selected `L0` or `L1`.
   - These results need explicit leakage diagnostics before citation.

2. Layer/region selection bias is not corrected.
   - Sweeping many regions, layers, and C values means the best validation-selected test number can look cleaner than the underlying distribution.
   - Future reports should include the full test-metric distribution across the sweep, not only the argmax.
   - For publishable claims, use nested CV or a selection-corrected estimate.

3. Test sample sizes were small.
   - Book behavior probes often had only `n=8` test rows per behavior, `4` positive and `4` negative.
   - Symphonic Voice had `n=40` test rows for an 8-way task, `5` per anchor.
   - Meta-cognition had `n=20` test rows, `10` per class.
   - Report tables must include `n_test`, and for binary probes `n_test_pos` / `n_test_neg`.

4. The C grid was too narrow.
   - We usually used `0.25,0.5,1.0,2.0`.
   - Since selected C was often `0.25`, the optimum may have been below the lower boundary.
   - Future runs should include stronger regularization.

5. Causal patch alpha was not layer-norm calibrated.
   - A unit direction at different layers can have very different effective size relative to native hidden-state norms.
   - Interpret alpha sweeps as local to a layer/region unless per-layer norm scaling is applied.

Minimum standard before citing a probe number:

- Show sample counts by split and class.
- Show text or prompt-only baselines.
- Show whether `prompt_last` and low-layer probes are competitive.
- Show the distribution of test metrics across the sweep.
- Show an ablated prompt/template test when prompt leakage is plausible.
- For causal claims, evaluate a different region, behavior, or live output rather than only the same probe that supplied the direction.

## Environment

Always work from repo root:

```bash
cd "/home/orwel/dev_genius/experiments/Character Creation"
source dev_genius/bin/activate
```

For model loading, check local cache before any download. The later SCOTUS script enforced this. The older scripts often assumed the model was already local.

Main local models used:

- `/home/orwel/dev_genius/models/Qwen3.6-35B-A3B`
- `/home/orwel/dev_genius/models/Qwen3.6-27B-FP8`
- `Qwen/Qwen3-VL-8B-Thinking` for older phase-aware work

## Core Feature Files

Nearly every modern probe wrote:

```text
sweep_v4/<run_name>/
  manifest.json
  feature_meta.jsonl
  features.npz
  searches.jsonl or layer_region_search.jsonl
  train_predictions.jsonl
  val_predictions.jsonl or dev_predictions.jsonl
  test_predictions.jsonl
  summary.json
```

`features.npz` keys were:

```text
assistant_mean__L16
think_mean__L12
response_last16__L19
prompt_last__L34
...
```

Each array shape was:

```text
[n_examples, hidden_dim]
```

`feature_meta.jsonl` preserved the row order. The `i`th metadata row matches row `i` of every feature matrix.

## Region Definitions

### Replay-Style Probes

Used by:

- `scripts/experiments/personality/probe_book_character_prefill_states.py`
- `scripts/experiments/personality/probe_book_character_prefill_by_behavior.py`
- `scripts/experiments/personality/probe_meta_cognition_states.py`
- `scripts/experiments/personality/probe_symphonic_voice_states.py`
- `scripts/experiments/personality/causal_patch_book_character_prefill.py`

These probes replayed already-written assistant completions.

The dataset row contained:

- `messages`: prior system/user messages
- `assistant_completion`: completed assistant text
- `split`: `train`, `val`, or `test`
- `label` for binary probes, or `anchor_id` for multiclass probes

The assistant completion was expected to contain:

```text
/think
...
/end-think
Response:
...
```

The code rendered:

```python
messages = row["messages"] + [{"role": "assistant", "content": assistant_text}]
rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
```

Then it found the assistant text inside the rendered string with:

```python
start = rendered.rfind(assistant_text)
```

Then it tokenized with offsets:

```python
encoded = tokenizer(
    rendered,
    return_tensors="pt",
    return_offsets_mapping=True,
    add_special_tokens=False,
)
```

Character spans were mapped to token spans using offset overlap.

Replay regions:

| Region | Definition |
|---|---|
| `assistant_mean` | mean over the full assistant completion |
| `assistant_last16` | mean over the final 16 assistant tokens |
| `think_mean` | mean over tokens between `/think` and `/end-think` |
| `think_first16` | mean over first 16 think tokens |
| `think_last16` | mean over last 16 think tokens |
| `response_mean` | mean over tokens after `Response:` |
| `response_first16` | mean over first 16 response tokens |
| `response_last16` | mean over last 16 response tokens |
| `response_last` | final response token |
| `prompt_last` | token immediately before the assistant completion starts |

Meta-cognition used a smaller set:

| Region | Definition |
|---|---|
| `assistant_mean` | full assistant response |
| `assistant_first16` | first 16 assistant tokens |
| `assistant_last16` | last 16 assistant tokens |
| `assistant_last` | final assistant token |
| `prompt_last` | token immediately before assistant response |

### SCOTUS Prompt-Only Probe

Script:

- `scripts/experiments/scotus/probe_scotus_style.py`

SCOTUS did not replay an assistant answer. It wrapped an excerpt in a neutral prompt:

```text
Read the following legal reasoning excerpt and continue the analysis in the same jurisprudential mode.

Excerpt:
{text}

Continuation:
```

It rendered this as a chat user message with `add_generation_prompt=True` and `enable_thinking=False` when available.

SCOTUS regions:

| Region | Definition |
|---|---|
| `prompt_last` | final token of the full prompt |
| `prompt_mean` | attention-mask mean over all prompt tokens |
| `excerpt_mean` | mean over the token span corresponding to the legal excerpt |

SCOTUS captured selected transformer layers with forward hooks instead of asking HuggingFace for every hidden state.

## Hidden State Extraction

### Replay-Style Extraction

The replay scripts used:

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_hidden_states=True,
    use_cache=False,
)
```

Then:

```python
for layer_h in outputs.hidden_states[1:]:
    h = layer_h[0]
```

This means the stored `L00`, `L01`, etc. are feature indices from `hidden_states[1:]`, not necessarily module indices. Treat them as historical feature-layer labels.

### SCOTUS Hook Extraction

The SCOTUS script located transformer blocks by trying paths like:

```text
model.language_model.layers
language_model.layers
model.layers
transformer.h
gpt_neox.layers
```

It registered forward hooks on the selected layers and saved three vectors per hooked layer:

```python
prompt_last = hidden[batch_ids, lengths - 1, :]
mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
prompt_mean = (hidden * mask).sum(dim=1) / lengths.to(hidden.dtype).unsqueeze(-1)
excerpt_mean = hidden[row_idx, excerpt_start:excerpt_end, :].mean(dim=0)
```

Default SCOTUS layers were coarse:

```text
0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,63
```

Use `--layers all` only when you can afford the memory and runtime.

## Classifier Protocol

The binary probe classifier was:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=4000,
        solver="liblinear",
        C=C,
        class_weight="balanced",
    )),
])
```

The usual historical C grid was:

```text
0.25,0.5,1.0,2.0
```

Future runs should use a wider log grid unless runtime is prohibitive:

```text
0.001,0.003,0.01,0.03,0.1,0.25,0.5,1.0,3.0,10.0
```

If the selected C is at either boundary, rerun with an expanded grid.

Multiclass symphonic probes used:

```python
LogisticRegression(
    max_iter=4000,
    solver="lbfgs",
    C=C,
    class_weight="balanced",
)
```

Selection:

1. Fit each `(region, layer, C)` on `train`.
2. Score on `val` or `dev`.
3. Pick highest validation balanced accuracy.
4. Break ties with F1 for binary probes or macro F1 for multiclass.
5. Refit the chosen probe on `train + val/dev`.
6. Report final `test` metrics from the refit probe.

Tie warning:

- Exact metric ties are rare, but near-ties are common with small validation sets.
- Strict argmax can pick a noisy outlier from a large sweep.
- A more robust variant is to take the top-k configurations by validation score, inspect their test distribution, and prefer a median or simpler configuration if several are statistically indistinguishable.

This is why `summary.json` often has:

```json
{
  "probe": {
    "selection": {
      "best_region": "...",
      "best_layer": 16,
      "best_C": 0.25
    },
    "train": {...},
    "val": {...},
    "test": {...}
  }
}
```

## Saved Directions

We used two different notions of "direction".

### Logistic Probe Direction

SCOTUS saved the scaler and logistic weights:

```text
best_probe_direction.npz
  scaler_mean
  scaler_scale
  coef
  intercept
  region
  layer
  C
  positive_justice
```

This is a classifier direction in standardized feature space. To score a vector:

```python
z = (x - scaler_mean) / scaler_scale
logit = z @ coef.T + intercept
prob = sigmoid(logit)
```

### Mean-Difference Direction

Behavior-specific book probes also saved train-only mean differences:

```python
direction = mean(X_train[label == 1]) - mean(X_train[label == 0])
direction = direction / norm(direction)
```

Those were saved in `behavior_probe_artifacts.npz` as:

```text
direction__<behavior>__<region>__LNN
probe_mean__<behavior>__<region>__LNN
probe_scale__<behavior>__<region>__LNN
probe_coef__<behavior>__<region>__LNN
probe_intercept__<behavior>__<region>__LNN
```

The mean-difference direction was used for patching, while the saved logistic probe was used to measure whether patching moved the internal readout.

Alpha scaling caveat:

- These directions are unit-normalized in raw hidden-state coordinates.
- Native hidden-state norms can differ substantially across layers.
- Therefore `alpha=1.0` at `L0` is not comparable to `alpha=1.0` at `L39`.
- For cross-layer comparisons, use per-layer norm-scaled alphas or report `||alpha * direction|| / mean(||h_layer||)`.

## Layer Numbering Gotcha

This has bitten us before.

Historical feature indices are not always the same as hook module indices.

For the book causal patch run, the observed routing was:

```text
feature L0 -> patch input_embeddings
feature LN for N > 0 -> patch transformer block N-1
```

That fix is documented in:

- `reports/book_character_prefill_behavior_probe_and_causal_patch_v2_20260417.md`
- `scripts/experiments/personality/causal_patch_book_character_prefill.py`

The patch helper was:

```python
def resolve_patch_target(model, layers, feature_layer_idx):
    if feature_layer_idx == 0:
        return model.get_input_embeddings(), "input_embeddings"
    return layers[feature_layer_idx - 1], f"block_{feature_layer_idx - 1}"
```

If recreating an intervention, verify the mapping with a small before/after check. Do not assume feature `L16` means hook module `layers[16]`.

## Completed Probe Runs

### 1. Book Character Prefill Global Probe

Script:

```text
scripts/experiments/personality/probe_book_character_prefill_states.py
```

Dataset:

```text
sweep_v4/book_character_prefill_dataset_balanced_v6_reviewed_20260417_151017/all_completions.jsonl
```

Model:

```text
/home/orwel/dev_genius/models/Qwen3.6-35B-A3B
```

Run artifact:

```text
sweep_v4/book_character_prefill_activation_probe_v1_20260417_151635
```

Equivalent command:

```bash
python scripts/experiments/personality/probe_book_character_prefill_states.py \
  --dataset-dir sweep_v4/book_character_prefill_dataset_balanced_v6_reviewed_20260417_151017 \
  --model-path /home/orwel/dev_genius/models/Qwen3.6-35B-A3B \
  --device-map auto \
  --c-grid 0.25,0.5,1.0,2.0
```

Result:

- Best region: `assistant_mean`
- Best feature layer: `L16`
- Best `C`: `0.25`
- Test balanced accuracy: `0.975`
- Test size: `n=40`, `20` positive / `20` negative

Interpretive note:

- This is the most credible of the listed historical probe numbers because the selected region was assistant-internal rather than `prompt_last`, and the test split was larger than the behavior-local probes.
- Still, it was selected from a large layer/region sweep and should be accompanied by sweep-distribution context before citation.

Report:

```text
reports/book_character_prefill_activation_probe_v1_20260417.md
```

### 2. Book Character Behavior-Specific Probes

Script:

```text
scripts/experiments/personality/probe_book_character_prefill_by_behavior.py
```

Input:

```text
sweep_v4/book_character_prefill_activation_probe_v1_20260417_151635/features.npz
sweep_v4/book_character_prefill_activation_probe_v1_20260417_151635/feature_meta.jsonl
```

Run artifact:

```text
sweep_v4/book_character_prefill_behavior_probe_v1_20260417_184525
```

Equivalent command:

```bash
python scripts/experiments/personality/probe_book_character_prefill_by_behavior.py \
  --probe-dir sweep_v4/book_character_prefill_activation_probe_v1_20260417_151635 \
  --c-grid 0.25,0.5,1.0,2.0
```

Best global readouts by behavior:

| Behavior | Best Readout | Test Balanced Accuracy |
|---|---|---:|
| `conflict_detection` | `assistant_mean @ L3` | `1.00` |
| `constraint_preservation` | `assistant_mean @ L13` | `1.00` |
| `repair_after_challenge` | `assistant_mean @ L0` | `0.75` |
| `selective_introspection` | `assistant_mean @ L21` | `0.875` |
| `state_carryover` | `assistant_mean @ L2` | `0.875` |

Each behavior-specific test result above used only `n=8` rows: `4` positive and `4` negative.

Interpretive note:

- The `1.00` behavior-local results are useful as debugging signals, but the confidence intervals are wide.
- The `L0` and `L1` selections are especially vulnerable to prompt/template leakage or near-input artifacts.
- Before citing these as learned behavior circuits, rerun with larger per-behavior test splits and prompt-ablation controls.

Targeted patch windows selected from behavior probes:

| Behavior | Think Window | Response Window |
|---|---|---|
| `conflict_detection` | `think_mean @ L15` | `response_mean @ L0` |
| `constraint_preservation` | `think_mean @ L1` | `response_mean @ L16` |
| `repair_after_challenge` | `think_mean @ L16` | `response_mean @ L0` |
| `selective_introspection` | `think_mean @ L0` | `response_mean @ L22` |
| `state_carryover` | `think_mean @ L0` | `response_mean @ L0` |

### 3. Book Character Causal Patch Sanity Check

Script:

```text
scripts/experiments/personality/causal_patch_book_character_prefill.py
```

Run artifact:

```text
sweep_v4/book_character_prefill_causal_patch_v2_20260417_190355
```

Equivalent command:

```bash
python scripts/experiments/personality/causal_patch_book_character_prefill.py \
  --dataset-dir sweep_v4/book_character_prefill_dataset_balanced_v6_reviewed_20260417_151017 \
  --behavior-probe-dir sweep_v4/book_character_prefill_behavior_probe_v1_20260417_184525 \
  --model-path /home/orwel/dev_genius/models/Qwen3.6-35B-A3B \
  --device-map auto \
  --alphas 0.5,1.0,2.0 \
  --tag book_character_prefill_causal_patch_v2
```

Mechanism:

- Replayed `val` and `test` rows.
- Computed the base internal probe probability.
- Added the normalized train-only pass-minus-fail direction over the target span.
- For fail rows, used `+direction`.
- For pass rows, used `-direction`.
- Recomputed the same internal probe probability after patching.

This was an in-space intervention sanity check, not a final behavioral steering result.

### 4. Meta-Cognition Activation Probe

Script:

```text
scripts/experiments/personality/probe_meta_cognition_states.py
```

Inputs:

```text
sweep_v4/meta_cognition_scorer_corpus_v1_20260417_121022/all.jsonl
sweep_v4/meta_cognition_text_scorer_v1_20260417_121713
```

Run artifact:

```text
sweep_v4/meta_cognition_activation_probe_v1_20260417_121804
```

Equivalent command:

```bash
python scripts/experiments/personality/probe_meta_cognition_states.py \
  --corpus-dir sweep_v4/meta_cognition_scorer_corpus_v1_20260417_121022 \
  --scorer-dir sweep_v4/meta_cognition_text_scorer_v1_20260417_121713 \
  --model-path /home/orwel/dev_genius/models/Qwen3.6-35B-A3B \
  --device cuda \
  --c-grid 0.25,0.5,1.0,2.0
```

Result:

- Best region: `assistant_last16`
- Best layer: `L34`
- Best `C`: `0.25`
- Val balanced accuracy: `0.900`
- Test balanced accuracy: `0.850`
- Test size: `n=20`, `10` positive / `10` negative

This run also compared activation-probe probabilities against the text scorer.

### 5. Symphonic Voice Multiclass Probe

Script:

```text
scripts/experiments/personality/probe_symphonic_voice_states.py
```

Repaired v2 dataset:

```text
sweep_v4/symphonic_voice_probe_dataset_v2_repaired_20260418_072722
```

Run artifact:

```text
sweep_v4/symphonic_voice_activation_probe_v2_repaired_20260418_073715
```

Equivalent command:

```bash
python scripts/experiments/personality/probe_symphonic_voice_states.py \
  --dataset-dir sweep_v4/symphonic_voice_probe_dataset_v2_repaired_20260418_072722 \
  --model-path /home/orwel/dev_genius/models/Qwen3.6-35B-A3B \
  --device-map auto \
  --c-grid 0.25,1.0 \
  --region-allowlist think_mean,assistant_mean,response_mean,response_last16,prompt_last \
  --layer-stride 1 \
  --max-gpu-gib 72 \
  --max-cpu-gib 24 \
  --offload-folder tmp_offload_symphonic_v2_repaired
```

Result:

- Task: 8-way anchor classification.
- Best region: `prompt_last`
- Best layer: `L0`
- Best `C`: `0.25`
- Test balanced accuracy: `0.950`
- Test size: `n=40`, `5` examples per anchor

Interpretive note:

- This result is the most leakage-suspicious of the completed runs because the winning readout was `prompt_last @ L0`.
- Before treating it as a voice-state result, run a bag-of-tokens or rendered-prompt baseline, remove any anchor-identifying prompt text, and rerun with prompt-only ablations.
- Also compare against assistant-internal regions selected under the same protocol.

The later axis analysis used a common comparison space:

```text
think_mean @ L39
```

and then fitted:

- multiclass anchor classifier
- ridge regressions for stance axes
- centroid deltas between anchors

Relevant scripts:

```text
scripts/experiments/personality/probe_symphonic_voice_states.py
scripts/experiments/personality/live_patch_symphonic_voice.py
```

Relevant report:

```text
sweep_v4/symphonic_voice_axis_analysis_v2_repaired_20260418_082543/report.md
```

### 6. SCOTUS Activation Probe Attempt

Script:

```text
scripts/experiments/scotus/probe_scotus_style.py
```

Current defaults:

```text
pairs: data/scotus/scotus_matched_pairs_v21.jsonl
pair: Scalia_vs_Ginsburg
variant: masked
positive_justice: Ginsburg
model: /home/orwel/dev_genius/models/Qwen3.6-27B-FP8
regions: prompt_last, prompt_mean, excerpt_mean
```

Observed smoke manifests:

```text
sweep_v4/scotus_probe_20260425_082313
sweep_v4/scotus_probe_20260425_083608
```

Those smoke runs used:

```text
split_caps = train:4,dev:4,test:4
C = 1.0
max_length = 768
batch_size = 1
```

Equivalent smoke command:

```bash
python scripts/experiments/scotus/probe_scotus_style.py \
  --pairs data/scotus/scotus_matched_pairs_v21.jsonl \
  --pair Scalia_vs_Ginsburg \
  --variant masked \
  --positive-justice Ginsburg \
  --model-path /home/orwel/dev_genius/models/Qwen3.6-27B-FP8 \
  --device-map auto \
  --batch-size 1 \
  --max-length 768 \
  --split-caps train:4,dev:4,test:4 \
  --c-grid 1.0
```

For a real run, remove or expand `--split-caps` and use a C grid:

```bash
python scripts/experiments/scotus/probe_scotus_style.py \
  --pairs data/scotus/scotus_matched_pairs_v21.jsonl \
  --pair Scalia_vs_Ginsburg \
  --variant masked \
  --positive-justice Ginsburg \
  --model-path /home/orwel/dev_genius/models/Qwen3.6-27B-FP8 \
  --device-map auto \
  --batch-size 1 \
  --max-length 1024 \
  --layers 0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,63 \
  --c-grid 0.25,0.5,1.0,2.0
```

Note: the observed SCOTUS smoke directories only contained `manifest.json` and `probe_examples.jsonl`, not completed `features.npz` / `summary.json`, so treat them as setup attempts rather than completed activation-probe results.

## Older Probe Family: Sycophancy

Scripts:

```text
scripts/experiments/sycophancy/sycophancy_probe_8b.py
scripts/experiments/sycophancy/sycophancy_probe_27b.py
```

This was a generation-time probe, not the later replay-style classifier workflow.

Method:

- Run leading-wrong, bad-opinion, and pushback prompts.
- Generate with system conditions: none, Skippy V4, honest.
- Capture forward-hook activations during `model.generate()`.
- Separate prefill from generated tokens.
- Save mean activation per generated token at target layers.
- Save top-50 logits per generated token and entropy.
- Score output text with keyword heuristics.
- Direction: `mean(sycophantic) - mean(non_sycophantic)`, unit-normalized.

8B target layers:

```text
L9, L15, L22, L29
```

27B target layers:

```text
L16, L36, L44, L50
```

Report:

```text
reports/sycophancy_probe_report.md
```

The direction quality was limited because there were very few sycophantic examples.

## Older Probe Family: Phase-Aware CoT Steering

Script:

```text
scripts/eval/phase_aware_cot_steering.py
```

This was a steering/eval script, not a trainable activation classifier.

Model:

```text
Qwen/Qwen3-VL-8B-Thinking
```

Steering source:

```text
results/qwen_connectome/analysis/connectome_zscores.pt
```

Sarcasm category:

```text
SARCASM_CAT_IDX = 6
```

Champion layers:

```text
L29, L30
```

Conditions:

| Condition | Prompt | Steering |
|---|---|---|
| `C0` | no V4 | none |
| `C1` | V4 | static L29+L30 alpha 8 |
| `C2` | V4 | phase-aware alpha 0 in think, alpha 8 in response |
| `C3` | V4 | none |

Hook rule:

- Patch only the last sequence position.
- During prefill, patching all positions corrupts the KV cache.
- Phase state was tracked by monitoring generated text for think delimiters.

Report:

```text
reports/phase_aware_cot_steering_report.md
```

## Recreating a New Probe

Use this checklist.

### 1. Freeze Dataset Splits

Every row must have a stable split:

```json
{
  "split": "train",
  "label": 1,
  "messages": [...],
  "assistant_completion": "..."
}
```

For paired data, split by case/source/family, not by chunk, so near-duplicates do not cross train/test.

### 2. Decide Probe Type

Use replay-style probing when you already have completed answers and want internal readouts over answer regions.

Use prompt-only probing when the text itself is the object being classified, as in SCOTUS excerpts.

### 3. Extract Features

For replay-style:

```bash
python scripts/experiments/personality/probe_book_character_prefill_states.py \
  --dataset-dir <dataset_dir> \
  --model-path <local_model> \
  --device-map auto \
  --c-grid 0.25,0.5,1.0,2.0
```

For SCOTUS-style:

```bash
python scripts/experiments/scotus/probe_scotus_style.py \
  --pairs <matched_pairs.jsonl> \
  --pair <A_vs_B> \
  --variant masked \
  --positive-justice <B> \
  --model-path <local_model> \
  --device-map auto \
  --batch-size 1 \
  --max-length 1024 \
  --layers 0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,63 \
  --c-grid 0.25,0.5,1.0,2.0
```

### 4. Inspect Artifacts

Check:

```bash
jq . <run_dir>/manifest.json
jq . <run_dir>/summary.json
head <run_dir>/feature_meta.jsonl
python - <<'PY'
import numpy as np
d = np.load("<run_dir>/features.npz")
print(len(d.files), d.files[:10])
for k in d.files[:5]:
    print(k, d[k].shape, d[k].dtype)
PY
```

### 5. Read the Result Correctly

Prefer:

- held-out test balanced accuracy
- stable dev/test agreement
- chance-level metadata or prompt-only controls where applicable
- nontrivial sample size per class
- full sweep distribution, not only the selected best result

Distrust:

- train-only results
- tiny test splits
- perfect performance with obvious textual leakage
- probes selected from many layers/regions with no held-out correction
- live patching evaluated by the exact same probe only
- `L0` or `prompt_last` wins unless prompt leakage diagnostics clear them

## Minimal Probe Pseudocode

```python
rows = load_jsonl(dataset)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

features = defaultdict(lambda: defaultdict(list))
meta = []

for row in rows:
    input_ids, attention_mask, spans, prompt_last_idx = render_and_spans(tokenizer, row)
    outputs = model(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        output_hidden_states=True,
        use_cache=False,
    )
    for layer_idx, layer_h in enumerate(outputs.hidden_states[1:]):
        h = layer_h[0]
        features["assistant_mean"][layer_idx].append(mean_span(h, spans["assistant"]))
        features["think_mean"][layer_idx].append(mean_span(h, spans["think"]))
        features["response_mean"][layer_idx].append(mean_span(h, spans["response"]))
        features["prompt_last"][layer_idx].append(h[prompt_last_idx])
    meta.append(row_without_text(row))

save_npz(features)
save_jsonl(meta)

for region, layer_map in features.items():
    for layer, X in layer_map.items():
        for C in [0.25, 0.5, 1.0, 2.0]:
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=C, class_weight="balanced")),
            ])
            fit on train
            score on val

select best by val balanced accuracy
refit best on train + val
report test
```

## Common Failure Modes

1. Missing `/think`, `/end-think`, or `Response:` markers.
   - Replay scripts will fail span parsing.

2. Assistant text not found after chat templating.
   - We used `rendered.rfind(assistant_text)`.
   - Formatting changes can break exact matching.

3. Layer index confusion.
   - Feature layer labels are historical.
   - Verify hook target mapping before causal patching.

4. vLLM used for capture.
   - Do not do this for probes. Use HuggingFace.

5. Leakage through prompt or labels.
   - Compare `prompt_last` to assistant-internal regions.
   - Use masked variants where possible.
   - Keep split grouping strict.
   - Add a bag-of-tokens or rendered-prompt text baseline.
   - Remove labels, anchor names, and style descriptors from prompt templates when testing internal style state.

6. Patching the whole prompt during generation.
   - For live generation hooks, patch only the last sequence position unless you intentionally want to alter the full prompt KV cache.

7. Judging causal success with the same probe you patched.
   - This is only an in-space sanity check.
   - Real success needs cross-region, cross-behavior, or live-output evaluation.
