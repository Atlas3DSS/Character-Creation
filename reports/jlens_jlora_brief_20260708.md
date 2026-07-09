# Codex Brief: J-LoRA — J-Space-Constrained Persona Adaptation

Date: 2026-07-08. Author role: Claude (orchestrator/reviewer). Implementation: Codex only, per project workflow rules.
Depends on: Phase 0 tooling from `reports/jlens_scotus_diagnostic_brief_20260706.md`. Recommended after the fingerprint brief's pilot (validates lens plumbing cheaply first).

## Hypothesis

The project's chronic persona-training failure is capability destruction ("LoRA SFT destroys AIME" — memory, durable lessons). The workspace paper reports J-space carries <10% of activation variance but ~60% of behavioral movement, with the non-J-space bulk hosting automatic processing. **If** that decomposition holds for Qwen, a persona update whose activation-effect is confined to J-space should:

- (H1) match unconstrained LoRA on persona fidelity at equal/lower rank, and
- (H2) retain substantially more math/reasoning capability.

Falsification: if math answers' production overlaps J-space heavily, J-LoRA damages math as much as unconstrained (report as: workspace decomposition does not separate style from reasoning in this model). If J-constrained ≈ random-constrained on both metrics, the lens buys nothing (report that).

## Method

Constrained adapter, two candidate parameterizations (Codex picks one for v1, documents why; ReFT-style preferred for cleanliness):

1. **J-ReFT**: at layer ℓ, `h ← h + P_J f(h)` where `f` is a learned rank-r map and `P_J` projects onto the top-k right singular subspace of `J_ℓ` (frozen, from the pre-fitted lens; SVD-proxy caveat as in prior briefs).
2. **Projected LoRA**: LoRA on residual-writing modules (`o_proj`/`down_proj`) with output basis pre-multiplied by `P_J` (i.e. `ΔW = (P_J B) A`).

Notes: J-space per the paper is a sparse non-negative cone; top-k subspace is an approximation — state this in the report. J is defined for base weights; if training is long, one mid-training refit is allowed but must be manifest-logged. Apply the layer-convention mapping from the diagnostic brief.

## Arms (all same data, same rank r, same steps, same seeds)

| Arm | Constraint | Purpose |
|---|---|---|
| A | J-space top-k | the hypothesis |
| B | random subspace, same k (×3 seeds) | **decisive control** |
| C | unconstrained (standard LoRA/ReFT) | ceiling for fidelity, floor for capability |
| D | complement space `(I − P_J)`, same k | paper predicts ~inert for behavior — cheap extra falsification |
| E | no training, V4 prompt | prompt baseline |
| F | no training, no prompt | raw baseline |

## Model, data, compute

- **Pilot model: Qwen3.5-9B instruct BF16** (cached). The published 9B lens is Base-variant — **fit a local lens on the exact instruct checkpoint first** using the jlens package (~1000 wikitext prompts, 128 tokens; hours on the PRO 6000; this also derisks lens-fitting for every future brief). Do not reuse the Base lens for the instruct model without an explicit similarity check.
- **Main model (only if pilot supports H1/H2): Qwen3.5-27B BF16** with the published lens. LoRA-only grads + grad checkpointing on 96GB; verify VRAM headroom before launch (`run_with_vram_guard.py`).
- **Data**: self-distillation per the SDFT recipe — generate ~2–4K responses from the same model under the V4 persona prompt on diverse prompts, strip the system prompt, train on (prompt → persona response). Mix in capability replay (math/reasoning examples at ~20–30%) identically across arms A–D. Memory lessons apply: watch the ~step-500 sweet spot; 5% data contamination caused identity confusion historically.

## Evaluation (per arm)

1. **Persona fidelity**: sarcasm/assistant marker rates (`data/sarcasm_markers.json`) + Opus critic scoring per the review-loop infra (CLAUDE.md), on held-out prompts, **without any system prompt** — the no-mask gate: the persona must live in the weights.
2. **Capability**: math battery via `scripts/eval/eval_runner.py` subset + AIME-style items; generation budgets per `qwen_eval_budget.py` (≥2048 for reasoning answers, promotion-eligible flags recorded).
3. **Coherence** and doom-loop screen (repetition metrics).
4. Report fidelity-vs-capability frontier: A vs B vs C is the entire result. Success = A above B and at/near C on fidelity while beating C clearly on capability retention.

## Guardrails

- Statistical comparisons Codex-audited; seeds and configs frozen in manifest before training (no post-hoc arm additions).
- Any claim of "J-space causally protects reasoning" requires arm D behaving as predicted AND arm B clearly below A — otherwise language stays correlational.
- No training run may silently download models; check HF cache (CLAUDE.md). BF16 only. `torch.cuda.empty_cache()` between phases.

## Output

`sweep_v4/jlora_pilot_<ts>/` (and `_main_` for 27B) with `manifest.json`, per-arm adapters, `records.jsonl`, `report.md` containing the frontier table + go/no-go for the 27B main run. This brief authorizes the pilot; the 27B main run needs the pilot gate explicitly met and logged.
