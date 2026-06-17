# Book Character Prefill Behavior Probes + Corrected Causal Patch

Date: 2026-04-17

## Artifacts

- Behavior-specific probes: `/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/book_character_prefill_behavior_probe_v1_20260417_184525`
- Corrected causal patch sweep: `/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/book_character_prefill_causal_patch_v2_20260417_190355`

## Behavior-Specific Probe Readout

Best global readouts by behavior:

- `conflict_detection`: `assistant_mean @ L3`, test balanced accuracy `1.00`
- `constraint_preservation`: `assistant_mean @ L13`, test balanced accuracy `1.00`
- `repair_after_challenge`: `assistant_mean @ L0`, test balanced accuracy `0.75`
- `selective_introspection`: `assistant_mean @ L21`, test balanced accuracy `0.875`
- `state_carryover`: `assistant_mean @ L2`, test balanced accuracy `0.875`

Best targeted patch windows used for the causal sweep:

- `conflict_detection`: `think_mean @ L15`, `response_mean @ L0`
- `constraint_preservation`: `think_mean @ L1`, `response_mean @ L16`
- `repair_after_challenge`: `think_mean @ L16`, `response_mean @ L0`
- `selective_introspection`: `think_mean @ L0`, `response_mean @ L22`
- `state_carryover`: `think_mean @ L0`, `response_mean @ L0`

## Bug Fix

The first causal sweep flatlined because the patch hook and the saved feature index were off by one on this Qwen/HF stack.

Observed mapping:

- hook block `k` first changes saved feature index `k+1`
- saved feature index `0` corresponds to the embedding-state readout

Corrected patch routing:

- feature `L0` -> patch `input_embeddings`
- feature `LN` for `N > 0` -> patch transformer block `N-1`

## Corrected Causal Patch Results

Metrics:

- `fail_mean_delta_prob`: average increase in positive-class probability on fail rows after adding the pass-minus-fail direction
- `fail_flip_rate`: fraction of fail rows crossing from `< 0.5` to `>= 0.5`
- `pass_mean_delta_prob`: average decrease in positive-class probability on pass rows after subtracting the same direction
- `pass_flip_rate`: fraction of pass rows crossing from `>= 0.5` to `< 0.5`

### Think-Region Patches

- `conflict_detection`, `think_mean @ L15 -> block_14`
  - fail delta `+0.9341`
  - fail flip rate `1.00`
  - pass delta `+0.9739`
  - pass flip rate `1.00`
- `constraint_preservation`, `think_mean @ L1 -> block_0`
  - fail delta `+0.9749`
  - fail flip rate `1.00`
  - pass delta `+0.9270`
  - pass flip rate `1.00`
- `repair_after_challenge`, `think_mean @ L16 -> block_15`
  - fail delta `+0.9648`
  - fail flip rate `1.00`
  - pass delta `+0.9875`
  - pass flip rate `1.00`
- `selective_introspection`, `think_mean @ L0 -> input_embeddings`
  - fail delta `+0.9821`
  - fail flip rate `1.00`
  - pass delta `+0.9077`
  - pass flip rate `1.00`
- `state_carryover`, `think_mean @ L0 -> input_embeddings`
  - fail delta `+0.8996`
  - fail flip rate `0.875`
  - pass delta `+0.7474`
  - pass flip rate `0.625`

### Response-Region Patches

- `conflict_detection`, `response_mean @ L0 -> input_embeddings`
  - fail delta `+0.9206`
  - fail flip rate `1.00`
  - pass delta `+0.8762`
  - pass flip rate `0.875`
- `constraint_preservation`, `response_mean @ L16 -> block_15`
  - fail delta `+0.9296`
  - fail flip rate `1.00`
  - pass delta `+0.9613`
  - pass flip rate `1.00`
- `repair_after_challenge`, `response_mean @ L0 -> input_embeddings`
  - fail delta `+0.7790`
  - fail flip rate `0.75`
  - pass delta `+0.8787`
  - pass flip rate `0.875`
- `selective_introspection`, `response_mean @ L22 -> block_21`
  - fail delta `+0.8958`
  - fail flip rate `0.875`
  - pass delta `+0.9335`
  - pass flip rate `1.00`
- `state_carryover`, `response_mean @ L0 -> input_embeddings`
  - fail delta `+0.8535`
  - fail flip rate `0.875`
  - pass delta `+0.8932`
  - pass flip rate `0.875`

Alpha note:

- `alpha = 0.5` was already enough to saturate most of the effect
- `alpha = 1.0` and `2.0` changed little

## Interpretation

These results are real in one narrow sense:

- the corrected patch path does alter the intended internal readout
- the targeted behavior probes are not random, because steering along their learned pass-vs-fail direction strongly flips the held-out eval rows

But this is not yet the stronger causal claim we ultimately want.

Why not:

- we are patching the exact region/layer used by the behavior probe
- with the exact pass-minus-fail direction extracted from that same region/layer
- and then reading out the same behavior probe afterward

So this is best understood as a strong in-space intervention sanity check, not final proof of a reusable meta-cognitive circuit.

## What This Does Establish

- The book-derived corpus carries a clean, behavior-local internal signal.
- That signal is strong enough to be linearly decoded on held-out eval items.
- The same signal is steerable in the expected direction at the targeted windows.
- `think_mean` windows are at least as strong as `response_mean`, and often cleaner.

## Next Non-Tautological Tests

1. Cross-region causal transfer:
   - patch `think_mean`
   - evaluate on `response_mean` or full assistant-level probes
2. Cross-behavior specificity:
   - patch `conflict_detection`
   - check whether it moves `conflict_detection` much more than unrelated behaviors
3. Live generation patching:
   - patch during generation
   - score output behavior, not only the internal probe
4. Hold-out-direction transfer:
   - derive direction on train-only families or paraphrase groups
   - patch held-out families

Those are the tests that would tell us whether we are looking at a reusable meta-cognitive control direction rather than a local linear handle.
