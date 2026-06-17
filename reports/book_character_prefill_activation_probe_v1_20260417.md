# Book Character Prefill Activation Probe V1

Date: 2026-04-17
Dataset:
- `sweep_v4/book_character_prefill_dataset_balanced_v6_reviewed_20260417_151017`

Model:
- `Qwen3.6-35B-A3B`
- local HF replay on Blackwell, bf16

Output:
- `sweep_v4/book_character_prefill_activation_probe_v1_20260417_151635`

## Main Result
Binary pass/fail labels are highly linearly recoverable from internal states on the reviewed book corpus.

Best selected probe:
- region: `assistant_mean`
- layer: `16`
- `C = 0.25`

Metrics:
- train balanced accuracy: `1.00`
- val balanced accuracy: `1.00`
- test balanced accuracy: `0.975`

## Per-Region Best Validation Performance
- `assistant_mean`: `1.00` at `L16`
- `assistant_last16`: `0.975` at `L19`
- `think_mean`: `0.975` at `L12`
- `think_first16`: `0.975` at `L1`
- `think_last16`: `0.95` at `L22`
- `response_mean`: `0.975` at `L20`
- `response_first16`: `0.925` at `L37`
- `response_last16`: `0.975` at `L19`
- `response_last`: `0.975` at `L18`
- `prompt_last`: `0.575` at `L16`

## Interpretation
- The label signal is strong in assistant-internal states and weak in the prompt tail.
- This argues against a simple prompt-only readout.
- Both `think` and `response` regions are strongly decodable, but the full assistant trajectory is strongest.
- The reviewed corpus appears clean enough that pass/fail structure is not subtle noise; it is a robust internal-state distinction.

## Test Misses
Exactly one test example was misclassified by the selected global probe:
- behavior: `selective_introspection`
- title: `Kip's Moral Ledger`
- split/label: `test`, negative (`0`)
- pair quality: `3`

This is a borderline negative, which is consistent with the eval review: the remaining `q3` items are the most plausible place for confusion.

## Artifacts
- summary: `summary.json`
- search grid: `searches.jsonl`
- feature metadata: `feature_meta.jsonl`
- feature tensor archive: `features.npz`
- split predictions: `train_predictions.jsonl`, `val_predictions.jsonl`, `test_predictions.jsonl`
