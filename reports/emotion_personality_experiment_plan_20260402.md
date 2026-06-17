# Emotion/Personality Experiment Plan for Current Sweep Outputs

**Date:** 2026-04-02
**Context:** After reviewing Anthropic's "Emotion Concepts and their Function in a Large Language Model" and comparing it to our current Qwen/Qwen3.5-9B sweep pipeline.

## 1. Current Assets

We already have the right basic shape of pipeline:

- `sweep_v3/ws_openai_15k/generated`
  - `14580` responses
  - `53,709,619` generated tokens
  - `243` characters x `60` prompts
- `sweep_v3/ws_openai_15k_sampled25m/generated`
  - `6751` responses
  - `25,001,473` generated tokens
  - balanced across the 6 prompt categories
- `scripts/experiments/personality/personality_sweep_v3_pass1_openai.py`
  - high-throughput pass-1 generation via SGLang/OpenAI API
- `scripts/experiments/personality/personality_sweep_v3_two_pass.py`
  - replay pass for activation means at layers `8,12,16,20,24,28`
- `scripts/experiments/personality/baseline_analysis.py`
  - already contains the right kind of whitening / null / confound checks, even if it needs adaptation to the new model/output layout

The key methodological advantage we already have is the full `3^5` Big Five factorial character grid. That is stronger than most personality-style probing datasets because it supports matched comparisons and interaction analysis instead of just loose labeling.

## 2. What The Paper Changes

The paper does **not** suggest throwing away the current sweep. It suggests using it more carefully.

Main implications:

1. Synthetic controlled data is the correct place to discover candidate directions.
2. Naturalistic broad data is better for validation than for initial probe discovery.
3. The most useful signals are likely **local and operative**, not a single global "this response has emotion X" state.
4. "Present speaker emotion" and "other speaker emotion" should be separated.
5. Hidden / deflected / unexpressed cases matter, otherwise probes learn surface style.
6. Mean activations over an entire long response are useful but probably too blunt as the final representation.

So the right move is:

- keep the existing sweep as the discovery dataset,
- make pass-2 more phase-aware,
- add targeted contrastive data,
- use naturalistic text only after the directions are already defined.

## 3. Immediate Plan

### Phase A: Finish the current replay pass

Objective: get a first usable activation dataset from the sampled 25M-token subset.

Tasks:

1. Fix the local replay environment so `Qwen/Qwen3.5-9B` loads under replay.
2. Run pass-2 on `sweep_v3/ws_openai_15k_sampled25m`.
3. Verify output integrity:
   - one response record per sampled generation
   - matching activation shard rows and metadata rows
   - no silent all-zero activations except genuine tokenization/replay edge cases

Deliverable:

- `responses/char_*.jsonl`
- `activations/L08..L28/mean_shard_*.pt`

This gives us the first pass of whole-generation activation means.

### Phase B: Analyze the current sweep before generating anything new

Objective: answer the highest-value questions with the data we already paid for.

Primary analyses:

1. **Big Five decodability by layer**
   - Train linear probes for each trait level from replay activations.
   - Evaluate:
     - random split
     - character-held-out split
     - prompt-held-out split
     - prompt-category-held-out split

2. **Matched factorial contrasts**
   - For each Big Five dimension, compare `H` vs `L` while matching the other 4 dimensions.
   - Because we have the full grid, this is much cleaner than naive averaging.
   - This should be the main estimate of trait direction, not raw global means.

3. **Prompt-category dependence**
   - Fit per-category directions for:
     - `emotional`
     - `identity`
     - `reasoning`
     - `social`
     - `practical`
     - `creative`
   - Expect stronger personality separation in `emotional`, `identity`, and `social`.
   - Expect weaker or more task-contaminated signal in `reasoning` and `practical`.

4. **Think vs response statistics**
   - Even before phase-aware activations, use:
     - `n_think_tokens`
     - `n_response_tokens`
     - `n_gen_tokens`
     - latency
   - Test whether some traits mostly change deliberation length while others mostly change surface response length.

5. **Trait geometry**
   - Compute cosine similarity among trait directions by layer.
   - Check whether the entanglements seen in earlier personality work still hold under replay/whitening.

6. **Whitening / nulls**
   - Adapt `baseline_analysis.py` to the new sweep output layout and new target layers.
   - Do not trust raw separability until whitening and false-positive checks are run.

Deliverables:

- per-layer probe metrics
- per-layer trait direction cosines
- post-whitening effect sizes
- a short report deciding which layers and which prompt categories are actually useful

## 4. The First Important Upgrade: Make Pass 2 Phase-Aware

The current replay pass stores **one mean activation per generated response**. That is a good first pass, but it is not the right final representation if the paper is directionally correct.

We should extend replay output to capture at least these separate means:

1. `think_mean`
2. `response_mean`
3. `early_gen_mean`
4. `late_gen_mean`

Recommended minimum implementation:

- Use existing `n_think_tokens` and `n_response_tokens`.
- During replay extraction, compute separate masked means over:
  - generated token positions `[0 : n_think_tokens]`
  - generated token positions `[n_think_tokens : n_think_tokens + n_response_tokens]`
- Write these as separate activation shards or as extra tensor channels.

Why this matters:

- It directly tests whether personality is more strongly represented during planning or expression.
- It prevents long `think` sections from washing out shorter but more behaviorally important response tokens.
- It maps better to the paper's "operative at a token position" framing.

This is the single most valuable pass-2 improvement.

## 5. New Data To Generate Next

Once Phase A/B are done, the next generation run should not just be "more of the same." It should be a targeted contrastive dataset.

### Dataset 1: Fixed-topic contrastive persona pairs

For each scenario, generate matched responses where:

- topic is fixed
- prompt wording is fixed
- only one target trait changes
- other Big Five dimensions are held constant

Example:

- same character scaffold
- same prompt
- neuroticism `low` vs `high`

Use this to learn much cleaner `delta` directions.

### Dataset 2: Hidden / deflected / unexpressed personality

For each character/prompt pair, create variants:

1. **Natural expression**
2. **Masked expression**
3. **Neutral-topic diversion**
4. **Talk-about-someone-else**
5. **Emotion/personality deflection**

This is the direct analogue of the strongest control from the paper.

It tests whether probes capture:

- genuine latent state,
- outward style only,
- or a confound between the two.

### Dataset 3: Present-speaker vs other-speaker state

Current sweep is mostly first-person self-description / self-response.

Add dyadic prompts where:

- the character has one state
- another person in the prompt has a different state
- the model must react to that other person

This separates:

- "what I am"
- "what I am reacting to"

That distinction is likely essential for both emotion and persona work.

## 6. How To Use Naturalistic Data

Do **not** start with naturalistic broad corpora for discovery.

Use them in this order:

1. Learn candidate directions on the clean sweep.
2. Validate those directions on naturalistic text.
3. Inspect max-activating examples manually.
4. Reject directions that only fire on templatic or synthetic artifacts.

Good validation questions:

- Do top activations look semantically coherent?
- Do activations survive prompt-category holdout?
- Do they generalize beyond explicit identity/emotion wording?
- Do hidden/deflected variants still score correctly?

Bad use of naturalistic data:

- training the first probes there and assuming the result is clean

The paper's failure mode was exactly that broad natural data can look messy and weak if the direction was not cleanly identified first.

## 7. Proposed Hypotheses

These are the most useful hypotheses to test with our current outputs.

### H1. Personality is easier to decode from `response_mean` than from `think_mean` for socially visible traits.

Likely strongest for:

- extraversion
- agreeableness
- neuroticism

### H2. Personality is easier to decode from `think_mean` than from `response_mean` for planning-heavy traits.

Possible candidates:

- conscientiousness
- openness

### H3. Trait directions differ substantially by prompt category.

Expected:

- `identity`, `emotional`, `social` will carry the cleanest signal
- `reasoning` and `practical` will mix personality with task demands

### H4. Matched factorial contrasts will produce cleaner directions than global high-minus-low means.

This should reduce confounds from the non-orthogonality of the Big Five.

### H5. Some traits are mostly style-like, others are more latent.

A likely split:

- style-like: extraversion, agreeableness
- more latent: neuroticism, conscientiousness

The hidden/deflected dataset is what will answer this.

## 8. Minimal Code Changes

The smallest useful implementation sequence is:

1. Fix replay environment for `qwen3_5`.
2. Run existing pass-2 on sampled 25M subset.
3. Add phase-aware replay means to `personality_sweep_v3_two_pass.py`.
4. Add an analysis script that:
   - loads activation shards
   - joins response metadata
   - parses `b5_combo`
   - supports matched factorial contrasts
   - supports category-held-out evaluation
5. Adapt `baseline_analysis.py` to the v3 layer set and metadata.

No new generation should happen until steps 1-4 are done.

## 9. Success Criteria

We should consider this experiment successful if we get all of the following:

1. Clean post-whitened probe performance above trivial baselines.
2. Stability under:
   - character holdout
   - prompt holdout
   - category holdout
3. Coherent max-activating examples in held-out data.
4. A clear answer on whether the useful signal lives more in:
   - `think`
   - `response`
   - or only in specific categories
5. At least one trait direction that remains usable under hidden/deflected expression.

If we fail on holdouts or hidden-expression tests, then the current sweep is mostly measuring surface style and should not be treated as a real latent-character representation.

## 10. Bottom Line

The current sweep was not a waste. It is exactly the right **discovery dataset**.

But to turn it into something robust, we need to shift from:

- whole-response means
- global high-minus-low averaging
- synthetic-only evaluation

to:

- phase-aware means
- matched factorial contrasts
- hidden/deflected controls
- naturalistic validation after discovery

That is the shortest path from "interesting personality separability" to "something mechanistically defensible."
