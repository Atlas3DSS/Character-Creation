# Codex Brief 4: J-Space Reward Shaping with a Staged Anti-Goodhart Harness

Date: 2026-07-09. Author role: Claude (orchestrator/reviewer). Implementation: Codex only, per project workflow rules.

Dependencies (hard gates, in order):
1. `reports/jlens_scotus_diagnostic_brief_20260706.md` Phase 0 tooling (jlens install, lens loading, layer-convention mapping).
2. **Fingerprint gate**: `reports/jlens_persona_fingerprint_brief_20260708.md` must first show persona signal in J-space at small k per its pre-registered margin. If persona is NOT decodable from J-space, this brief is void — do not run it (you cannot reward a signal that isn't there).
3. J-LoRA brief training infra (`reports/jlens_jlora_brief_20260708.md`): locally-fitted Qwen3.5-9B-instruct lens, SDFT data pipeline, eval battery.
4. ΔJ brief machinery (`reports/jlens_lens_comparison_brief_20260708.md`) reused here as the drift audit.

## Idea

Use J-lens readouts of internal states as a **dense, deterministic, differentiable shaping signal** during persona training: reward disposition mass toward persona vocabulary on conversational prompts, and PENALIZE it on math/reasoning prompts (two-sided, trigger-conditioned). This is a proxy signal, not ground truth — "verifiable" means reproducible, not true. The central engineering problem is Goodhart; the harness below is the actual contribution.

Mechanics: precompute rows of `W_U J_ℓ` for the target vocab set (a few hundred fixed vectors per layer). Shaping loss at sampled positions = softmax-mass (or margin) of target tokens in the transported readout. **Auxiliary differentiable loss added to SFT/SDFT — NOT RL in v1** (SGD on a fixed loss is a weaker exploit-finder than policy optimization; RL variant only after v1 survives).

## Known exploit routes (named up front)

- E1: activations shifted to satisfy the frozen lens without behavioral change.
- E2: **token spam** — literally emitting persona vocab raises disposition mass under EVERY lens; no probe randomization can catch this. Only the behavioral anchor can.
- E3: stale-ruler drift — training moves weights, J goes stale, optimizer exploits the specific stale lens.

## The randomized probe distribution (anti-E1/E3)

All sampling via seeded PRNG schedules (reproducible to us, no fixed target for the optimizer). Components:

- Fit-seed/corpus ensemble: N=4–8 lenses fit on disjoint prompt corpora for the exact training checkpoint's base weights.
- Sub-subspace jitter: per batch, a random k′-dim slice of the top-k J-subspace.
- Layer sampling: ℓ drawn per batch from the workspace band.
- Token-set resampling: random subsets / synonym-expanded variants of the persona vocab per batch.
- Position sampling: random generated-token windows.
- Online EMA refit (Tier 2 only): re-estimate J on ~50 prompts against CURRENT weights every k steps, EMA-merge.

## Three-tier ladder with hysteresis

| Tier | Config | Per-step cost |
|---|---|---|
| T0 | single trained-against lens + monitors only | lowest, lowest grad variance |
| T1 | pre-fit ensemble sampling + jitter (no refit) | near-T0 (sampling among precomputed rows) |
| T2 | + online EMA refit, min-aggregation over ensemble, optionally adversarial member selection | highest (periodic backward passes, max variance) |

Default starting tier: T1 (its per-step cost is marginal; T0 exists mainly as an ablation and early-phase option).

**Alarms (escalate one tier when either fires):**
- A1 — dual-lens divergence: disagreement between trained-against readout and a held-out never-trained-against lens, exceeding the **calibrated null band** (distribution measured on the base model before training; trigger = >p99 for M=3 consecutive eval checkpoints).
- A2 — proxy/anchor decoupling: shaping reward rising while behavioral anchor metrics (markers/judge) are flat or falling over a window. This is the canonical Goodhart signature and the primary alarm.

**De-escalation (drop one tier):** divergence < p50 of null band sustained for a window ≥3× the escalation window (asymmetric hysteresis; never flap). **On every de-escalation, retire the trained-against lens and install a fresh one fit on current weights** (invalidates learned exploits). **Backoff rule:** 3 escalations in one run → permanent residence at T1 minimum; log and continue.

**Telemetry is a deliverable:** full escalation log (step, prompt type, layer, alarm values) — the map of when/where hacking pressure emerges is a secondary finding regardless of the primary outcome.

## Anchor (anti-E2, non-negotiable)

Behavioral ground truth gates all checkpoints and is never trained against via the lens pathway: persona marker rates + Opus critic (review-loop infra), math battery with `qwen_eval_budget.py` budgets, repetition/diversity metrics with an explicit n-gram spam detector (E2 is caught HERE or not at all). A checkpoint that raises shaping reward but fails anchor deltas is discarded and logged as a hack event.

## Arms (same data, steps, seeds)

| Arm | Shaping | Purpose |
|---|---|---|
| 1 | J-reward, persona vocab, two-sided conditioning | the hypothesis |
| 2 | identical structure, random non-persona vocab set | **decisive control** — persona-specificity vs any-token shaping artifacts |
| 3 | no shaping (anchor-only SDFT, identical data) | baseline frontier |
| 4 (optional) | judge-score reward distilled offline | is the lens cheaper-but-equal to a judge signal? |

Ablations within Arm 1 if budget allows: T0-locked vs T1-locked vs full ladder; mean vs min aggregation.

## Model / data / compute

Qwen3.5-9B instruct BF16 pilot (needs the locally-fitted instruct lens from the J-LoRA brief — plus its disjoint-corpus siblings for the ensemble; fitting all N is hours-scale for 9B on the PRO 6000). SDFT self-distillation data + capability replay identical across arms (J-LoRA brief recipe; step-500 sweet-spot and contamination lessons apply). 27B scale-up only after pilot gates.

## Pre-registered predictions

1. With the anchor in place, escalations are rare and occur mid-late in training (when easy honest gains are exhausted). **Kill condition: alarms early and often → the vocabulary-aligned proxy is too gameable; report and stop.**
2. Success = Arm 1 dominates Arm 3 on the fidelity-vs-capability frontier at matched steps, OR reaches anchor thresholds in meaningfully fewer steps; AND Arm 1 > Arm 2 (persona-specificity). If Arm 1 ≈ Arm 2 > Arm 3, the benefit is generic regularization, not J-space content — report the deflationary reading.
3. The two-sided math-neutrality penalty measurably reduces persona intrusion into math outputs vs Arm 3 (this is the persona/capability-tradeoff payoff and the headline metric if it holds).

## Guardrails

Codex writes all code; statistics Codex-audited; manifest frozen (arms, seeds, thresholds, null bands) before training step 1 — no post-hoc arm or threshold changes; HF-cache checks, BF16 only, VRAM guard, `torch.cuda.empty_cache()` between phases; no secrets in source.

## Output

`sweep_v4/jreward_pilot_<ts>/` with `manifest.json` (incl. null-band calibration data + full PRNG schedules), per-arm checkpoints, `escalation_log.jsonl`, `records.jsonl`, `report.md`: frontier table (arms × fidelity/capability/coherence), alarm timeline plot data, hack-event postmortems, and an explicit verdict against each pre-registered prediction.
