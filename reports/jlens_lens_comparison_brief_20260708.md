# Codex Brief: ΔJ — Does Persona/Behavior Training Re-Route the Transport Map?

Date: 2026-07-08. Author role: Claude (orchestrator/reviewer). Implementation: Codex only, per project workflow rules.
Depends on: Phase 0 tooling from `reports/jlens_scotus_diagnostic_brief_20260706.md`. Priority: LOWEST of the three J-lens briefs — run after the fingerprint pilot; ideally consumes adapters produced by the J-LoRA brief.

## Question

When a model is fine-tuned/LoRA'd/abliterated toward a persona or behavior change, does its average Jacobian `J_ℓ` (the layer→output transport map) change measurably — i.e. does training **re-route the pipeline** — or do activations merely move within an unchanged pipeline? The fingerprint brief measures positions; this brief measures the map itself.

Predictions to test (stated before running):

- Light LoRA (small rank, few steps): ΔJ within refit-noise floor at most layers — persona lives in activation positions, not transport.
- Abliteration / heavy SFT: ΔJ above noise floor, concentrated in workspace-region layers and in J-lens vectors for behavior-relevant vocabulary (e.g. refusal terms for abliteration).
- Either outcome is informative; neither is a steering claim.

## The mandatory control: refit-noise floor

Before ANY cross-model comparison is interpreted, fit the lens **twice on the same model** with disjoint prompt seeds and compute all comparison metrics between the two fits. That distribution is the noise floor; every ΔJ claim is reported as a multiple of it. No floor, no findings.

## Model pairs (in priority order)

1. **Qwen3-VL-8B-Thinking vs huihui abliterated 8B-Thinking** — both cached locally, cheapest fits, and the behavior delta (refusal removal) is large and known. No published lens exists for VL-8B → fit locally for both (also validates our fitting pipeline; the jlens README recipe applies; ~1000 wikitext prompts, 128 tokens, hours-scale on the PRO 6000 for 8B).
2. **Qwen3.5-9B instruct: base vs J-LoRA/unconstrained-LoRA arms** from the J-LoRA brief (merged adapters) — directly answers "did our own persona training move J", and the J-LoRA arm has a built-in prediction (its ΔJ should concentrate inside the very subspace it was constrained to — a self-consistency check on the whole method).
3. **Qwen3.5-27B base vs abliterated 27B** — only if the 27B-abliterated weights still exist on disk (check before promising; `qwen35_map/27b-abliterated/` holds results, not necessarily weights). 27B fits are the expensive tier (the Neuronpedia fit used a B200; on the PRO 6000 expect a long run — verify VRAM for 27B BF16 backward at seq 128 with `run_with_vram_guard.py` before committing).

## Comparison metrics (per layer ℓ)

1. **Subspace geometry**: principal angles / projection-metric distance between top-k right singular subspaces of `J_ℓ^A` vs `J_ℓ^B` (k ∈ {32, 128, 512}).
2. **Map similarity**: CKA (or normalized Frobenius distance) between `J_ℓ^A` and `J_ℓ^B`.
3. **J-lens vector drift, vocabulary-resolved**: for a chosen vocab set (behavior-relevant terms vs matched neutral terms), cosine between each token's J-lens vector across models. Prediction: drift concentrates in behavior-relevant vocab; neutral vocab tracks the noise floor.
4. **Layer profile**: all of the above as a function of depth; the paper's sensory→workspace→motor block structure suggests where changes should (and shouldn't) appear — report against that template but don't force it.

## Guardrails

- Fit configs identical across the pair (same corpus, prompt count, seq len, dtype, stopping rule) — any asymmetry invalidates the comparison; record configs in manifest.
- Same tokenizer required within a pair (true for all pairs above; assert it in code).
- All claims relative to the refit-noise floor; Codex audits the statistics.
- This is interpretation/measurement work only: no steering runs, no generation-based behavior claims beyond what the paired models are already documented to do.
- BF16 only, HF-cache checks, no silent downloads, VRAM guard on every fit.

## Output

`sweep_v4/jlens_delta_comparison_<ts>/` per pair, with `manifest.json` (both fit configs + noise-floor stats), fitted lens files (kept — they're reusable assets for the other briefs), `records.jsonl`, `report.md`: noise floor table, then per-layer ΔJ profiles and vocab-resolved drift, each expressed in noise-floor multiples. Close with an explicit answer to the title question for each pair, or "indistinguishable from refit noise" where that's the truth.
