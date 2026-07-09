# Codex Brief: J-Space Persona Fingerprinting

Date: 2026-07-08. Author role: Claude (orchestrator/reviewer). Implementation: Codex only, per project workflow rules.
Depends on: Phase 0 tooling from `reports/jlens_scotus_diagnostic_brief_20260706.md` (jlens install, lens download, layer-convention check). Run that first or reuse its artifacts.

## Hypothesis

A persona/style/culture condition produces a separable, stable signature in the **J-space coordinates** of generation-time activations — and that signature concentrates in J-space (<10% of activation variance) rather than in the non-J-space complement. This is a decodability-class experiment (the method family that has consistently worked in this project); it makes NO steering or causal claims.

Falsifiable predictions:

1. A probe on `P_J h` (J-space projection) recovers persona labels nearly as well as a probe on raw `h`, despite using ≤10% of variance.
2. A probe on a **random subspace of the same dimension** does substantially worse than `P_J h`. (If it doesn't, the workspace framing adds nothing over "any subspace of this size" — report that as the finding.)
3. A persona's mean J-space signature is stable across disjoint prompt sets (high cosine), and held-out generations can be matched to their persona by nearest-signature (identification accuracy >> 1/K).

## Model / Lens

- Primary: `Qwen/Qwen3.5-27B` BF16 (local; see diagnostic brief for paths) with the pre-fitted `neuronpedia/jacobian-lens` `qwen3.5-27b` lens.
- Do NOT use the 9B sweep_v2 data in this pass: the published 9B lens is for `Qwen3.5-9B-Base`, the sweep used the INT8 *instruct* model — a double mismatch. A 9B-instruct extension is gated on fitting a local lens for that exact checkpoint (separate task).
- Personas via **system prompts on the unmodified base model** — no training confound in v1. Trained-persona extension (LoRA/SDFT adapters) comes later and pairs with the J-LoRA brief.

## Design

- K = 8 personas, chosen for two axes: 4 "cultural/voice" (e.g. Hitchens-style polemicist, folksy Midwestern grandmother, formal Japanese business register, Gen-Z streamer) and 4 "belief/stance" (e.g. libertarian economist, devout theologian, radical environmentalist, cold empiricist). Plus 1 neutral no-system-prompt baseline. Exact prompts in `data/personas/fingerprint_v1.json` (Codex writes; no real living persons' names in system prompts — style descriptions only; NEVER "Skippy", see memory).
- M = 30 topic-neutral user prompts (identical across all personas; everyday/expository topics with no persona-affinity — audit for topic-persona leakage before running).
- Generation: greedy, ≥512 tokens (this is style not legal holdings; the 2048 rule applies to reasoning evals, not here — record `budget_note` anyway).
- Capture: residual activations at ~6 layers spanning the workspace region (choose from lens layer coverage; include early/mid/late). **Generated-token positions only** — never prompt positions (the system prompt trivially leaks the label into prompt-region activations). Mean-pool over generated tokens per response; also keep per-token for the signature-stability analysis.

## Analysis

For each captured layer ℓ, with `P_J` = projector onto top-k right singular subspace of `J_ℓ` (k ∈ {32, 128, 512}; label as SVD proxy for the paper's cone, per diagnostic brief):

1. **Probe battery** (GroupKFold by user-prompt, logistic, house standards): persona classification on (a) raw `h`, (b) `P_J h`, (c) `(I − P_J) h`, (d) random same-dim subspace × 10 seeds. Report balanced accuracy + label-shuffle null.
2. **Text baseline to beat/contextualize**: TF-IDF on the generated text (house rule — activation claims mean little if plain text matches them; expect text to be strong, the point is the J vs non-J vs random *contrast*, not beating text).
3. **Fingerprint stability**: split prompts into disjoint halves; mean J-space signature per persona per half; cross-half cosine matrix; identification accuracy of held-out responses (nearest signature).
4. **Disposition readout**: decode each persona's mean `J_ℓ h` delta (vs neutral baseline) through the unembedding; top-30 tokens; sanity-check they are persona-plausible vocabulary. This is the qualitative "what is this persona poised to say" readout.

## Controls & guardrails

- Random-subspace control is mandatory and decisive (prediction 2).
- Label-shuffle null on every probe.
- Persona prompts audited so no persona-specific content words appear in user prompts.
- No causal/steering language in the report. This measures decodable structure only.
- If lens layer indices and our hook convention differ, apply the mapping documented by the diagnostic brief.

## Pilot → full

Pilot: 3 personas × 10 prompts × 2 layers, full analysis path end-to-end. Gate: pipeline produces sane probe numbers and the readout tokens aren't garbage. Then full grid.

## Output

House conventions: `sweep_v4/jlens_persona_fingerprint_<ts>/` with `manifest.json`, `records.jsonl`, `report.md`. Report table: layer × k × feature-space → BA (+ null, + random-subspace distribution); stability matrix; identification accuracy; per-persona top-token readouts. Codex audits statistics before anything is presented as a finding.

## Thesis refinement addendum (2026-07-08, pre-registration)

Pinned thesis: *persona labels are decodable from a top-k J-lens subspace nearly as well as from raw activations, while random same-dimensional subspaces lag.* Three binding refinements:

1. **The finding is a gap CURVE, not a point.** Sweep k ∈ {8, 32, 128, 512}. Johnson–Lindenstrauss predicts random subspaces preserve linear separability at large k — random-512 being ≈ raw is EXPECTED and is not evidence against the thesis. Prediction: random-subspace BA decays toward chance as k shrinks; J-subspace BA stays ~flat. Plot both curves; the divergence at small k is the result.
2. **Vocabulary-alignment control.** J-lens directions are unembedding-transported; a J-space probe may merely re-read the emitted tokens. Add two feature spaces to the probe battery: final-layer `h` and output logits (top-vocab or full). The interesting claim is EARLY/MID-layer J-space decoding persona at parity with late layers — persona held as workspace state, not echoed token identity. If J-space only matches logits layer-for-layer, report the deflationary reading.
3. **Pre-registered margin.** "Nearly as well" := J-subspace retains ≥ 90% of raw's above-chance margin (BA − 0.5) at k=32 on the primary layer; random-32 (median of 10 seeds) retains ≤ 50%. Both must hold to claim the thesis; otherwise report the measured curves without the claim.

Interesting-failure clause: if persona decodes well from raw but poorly from J-space at ALL k, report it prominently — it mirrors the SCOTUS decodable-but-outside-output-disposition geometry and retrodicts the steering nulls.
