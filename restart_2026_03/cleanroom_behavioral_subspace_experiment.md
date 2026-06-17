# Cleanroom Behavioral Subspace Experiment

## Purpose

Investigate whether a language model has a reachable behavioral region in which:

- helpfulness stays high
- formal reasoning stays high
- alignment and refusal behavior stay stable
- sarcastic style becomes much stronger

The experiment should be conducted without relying on prior project artifacts, prior prompts, prior layer choices, prior vectors, or prior conclusions.

## Research Question

Can local interventions move a model from a standard assistant operating point toward a more sarcastic operating point without causing:

- loss of reasoning
- loss of helpfulness
- generic hostility
- refusal collapse
- broad behavioral instability

## Cleanroom Constraints

- Do not import any previous steering vectors, prompts, scores, or layer selections.
- Do not reuse any prior claims about circuits, neurons, basins, or boundaries.
- Do not assume any previously identified intervention works.
- Do not treat any earlier experiment names or conditions as authoritative.

## Experimental Framing

Treat the model as an unknown system.

The goal is to map local behavioral geometry around a baseline assistant mode by measuring response changes under small controlled interventions.

Useful measurable objects:

- behavioral similarity between conditions
- capability/style tradeoffs
- smoothness versus cliff behavior under intervention strength
- overlap between style changes and capability/alignment changes
- repeatability across seeds and prompt paraphrases

## Baseline

Define a neutral baseline assistant condition with:

- no character framing
- no style-targeting prompt
- no activation intervention

This baseline is the reference point for all comparisons.

## Intervention Families

Only broad classes should be considered at the start:

1. prompt-only style intervention
2. activation-only intervention
3. combined prompt + activation intervention
4. response-phase-only intervention if the model exposes a separable reasoning/response process

The first pass should compare simple instances of these families rather than many variants.

## Metrics

Evaluate at least these axes separately:

- formal reasoning accuracy
- general knowledge accuracy
- helpful answer quality
- refusal/alignment stability
- sarcastic style intensity
- coherence

No single style metric should decide the outcome.
If using sarcasm markers, treat them as weak proxies only.

## Minimum Validity Requirements

Do not trust a result unless:

- it replicates across at least two random seeds
- it survives prompt paraphrase variation
- it is compared against the same fixed evaluation battery
- capability and alignment are measured separately from style

## Success Condition

A result is interesting only if it shows:

- clear increase in sarcastic style
- little or no degradation in reasoning
- little or no degradation in helpfulness
- little or no degradation in refusal/alignment behavior
- replication across seeds

## Failure Conditions

Reject any intervention that:

- raises sarcasm by becoming generally rude or unstable
- reduces reasoning materially
- weakens refusal behavior
- reduces answer usefulness
- works only for one wording or one seed

## First Experimental Pass

Start with a very small matrix:

- baseline
- prompt-only
- activation-only
- prompt + activation

Run all four on the same fixed sample and the same evaluation battery.
Repeat with a second seed.

Only after that should there be any attempt to refine:

- intervention strength
- layer placement
- token-phase timing
- training objectives

## Interpretation Rules

At the beginning, only make statements of the form:

- "under this evaluation"
- "relative to baseline"
- "replicated across seeds"
- "suggests"
- "is consistent with"

Do not make statements of the form:

- "there is a sarcasm neuron"
- "this proves a basin"
- "this proves a circuit"
- "this proves a stable personality manifold"

## Long-Term Usefulness

If the experiment works, the result is not merely stylistic.
It would support a broader claim that local behavioral geometry can be measured and exploited to create high-style, high-capability assistants without broad misalignment.
