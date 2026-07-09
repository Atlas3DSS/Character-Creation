# Codex Brief: J-Lens Diagnostic Over Archived SCOTUS Directions

Date: 2026-07-06. Author role: Claude (orchestrator/reviewer). Implementation: Codex only, per project workflow rules.

## Context

Five months of SCOTUS/persona work established decodability-without-controllability: every candidate direction failed prompt-matched same-layer random controls (see `SCOTUS.md` "Current Status After Phase 4-5 Follow-up"). Anthropic's Verbalizable Workspace paper (Jacobian lens, https://github.com/anthropics/jacobian-lens) offers a mechanistic explanation candidate: probe-optimized directions carry most variance outside the causally-potent J-space; steering non-J-space components moved behavior in ~5% of trials vs ~59% for J-space components.

Neuronpedia has published pre-fitted lenses for our exact models:

- `neuronpedia/jacobian-lens` → `qwen3.5-27b/jlens/Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt` (fit on `Qwen/Qwen3.5-27B` BF16, wikitext-103, 672 prompts, all layers, B200)
- `neuronpedia/jacobian-lens` → `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt` (note: no config.yaml in repo for 3.6 — record its absence in the manifest)

Local model availability as of this brief:

- `Qwen/Qwen3.5-27B`: BF16 weights are present in the HF cache and mirrored at `~/dev_genius/models/Qwen3.5-27B`.
- `Qwen/Qwen3.6-27B`: the HF cache directory exists, but BF16 safetensors are present under `~/dev_genius/models/Qwen3.6-27B`; do not assume the HF cache contains the weights.

Use BF16 checkpoints only — never the FP8 copies for lens work.

## Objective

**Diagnostic only. No steering, no generation runs, no promotion claims.** For every archived SCOTUS candidate direction, measure how much of it survives transport through the average Jacobian into output space, versus norm-matched random directions. Outcome either explains the causal nulls (candidates ≈ random in J-transported mass) or identifies directions with above-random output disposition (gate for a possible later steering phase — NOT part of this brief).

## Phase 0 — Environment

1. Clone `anthropics/jacobian-lens`, `pip install -e .` into the project venv (`source dev_genius/bin/activate`). Pin nothing beyond what jlens requires; record versions in the run manifest.
2. Download the two lens `.pt` files via `huggingface_hub` (check local HF cache first, per CLAUDE.md).
3. Smoke test: reproduce the README example (`Fact: The currency used in the country shaped like a boot is`, `positions=[-2]`) on Qwen3.6-27B BF16; confirm lens readouts are sensible (e.g. `lira`/`euro`-adjacent tokens in top-5 at mid layers). Save the smoke output.

For the first gentle pass, run a Qwen3.5-only offline pilot before cloning/loading full models. The pilot may load only the Qwen3.5 lens tensor, archived direction vectors, tokenizer, final norm, and unembedding weights. It must not generate text.

## Phase 1 — Direction Inventory

Walk all compact SCOTUS direction artifacts, not only JSON sidecars:

- `data/scotus/directions/*.json` plus their referenced `.npz` files.
- orphan compact `.npz` files with no JSON sidecar, including `data/scotus/directions/scotus_minpair_replay_assistant_all_L4_direction_20260501.npz`.
- localization direction stores matching `sweep_v4/scotus_*localization*/top_directions.npz` with `direction_meta.jsonl`, including ambiguous thought state, conclusion-token, counterfactual answer-state, and generated-thought-baseline localization runs.

For each direction resolve:

- source model (via the source run's `manifest.json` `model_path`) — this decides which lens to use; directions from Qwen3.5 go through the 3.5 lens, Qwen3.6 through the 3.6 lens. **Never cross models.** If a direction's source model has no published lens (e.g. a VL-8B or 9B artifact), exclude it and log the exclusion.
- layer, component/region, extraction method (logistic probe / mean-delta / low-rank), sign convention.

Low-rank artifacts are not single directions. Inventory them as separate diagnostic rows:

- `delta_mean` as the mean target-source shift.
- each `components[i]` as a rank-basis direction.
- optional stored-map outputs on archived held-out states as a later diagnostic, clearly separated from the Phase A single-direction pilot.

Output: `direction_inventory.jsonl`.

**Layer-convention check (mandatory, before any projection):** our capture hooks read the *output of decoder layer i* (`layers[i]` forward hook = `hidden_states[i+1]`). Verify which convention jlens uses for its layer indices (inspect `jlens.fitting` / lens dict keys against model config) and document the mapping in the manifest. Past project failure mode: "feature layer labels are historical" (`activation_probes.md`, failure mode 3). If conventions differ by one, correct explicitly and note it.

## Phase 2 — Projection Diagnostic

For the Qwen3.5 pilot, start with exactly these archived single-vector directions:

- `probe_direction_assistant_all_L08_C0p001`
- `probe_direction_assistant_all_L08_C0p001_inverse_authority`
- `probe_direction_assistant_all_L16_C0p001`
- `probe_direction_assistant_all_L20_C0p001`
- `scotus_article3_controlled_replay_v2_assistant_all_L04_private_rights_20260501`

For each inventoried direction `d` at layer ℓ (unit-normalize first; also run the raw-norm variant if cheap):

1. **Transported gain:** `g(d) = ||J_ℓ d|| / ||d||`.
2. **Random control:** N=200 unit Gaussian directions in the same layer's residual basis, same norm; report `g` percentile of the candidate within the random distribution. This is the project-standard control, applied in J-space.
3. **Output disposition readout:** decode `J_ℓ d` and `−J_ℓ d` through the model's unembedding (jlens `apply`/readout path); record top-30 tokens each sign. Flag whether legal/frame-relevant vocabulary (private/public rights, standing, Article III, commerce, etc. — reuse marker lists from the frame scorers) appears in top-30, and at what rank.
4. **Top-singular transport proxy:** randomized or exact SVD of `J_ℓ` (top-k right singular vectors, k ∈ {32, 128, 512}); report fraction of `||d||²` inside each top-k subspace vs the random-direction distribution. Label this as a top-singular transport proxy, not the paper's sparse non-negative J-space decomposition. Implement the paper's gradient-pursuit decomposition only if it is included in the released jlens package — do not re-implement it from the paper text in this pass.
5. Repeat 1–4 for both signs where the direction has a meaningful contrast (positive_justice vs the other class).

Memory note: J_ℓ per layer at hidden=5120 is ~50 MiB BF16 or ~100 MiB FP32; stream layers, `torch.cuda.empty_cache()` between (CLAUDE.md GPU rule). Most of Phase 2 does not need full model generation, but output readout still needs tokenizer, final norm behavior if applicable, and the unembedding.

Pilot go/no-go criteria:

- promising only if at least one archived direction has gain percentile > 95, top-k transport proxy percentile > 95 for at least one k, legal/frame vocabulary in top-30, and sign semantics that make sense;
- otherwise treat the result as support for the explanation that the archived directions were decodable but mostly outside the causally potent output-disposition subspace;
- do not start Phase B steering from this pilot unless the criteria above are met.

## Phase 3 — Report

House conventions: run dir under `sweep_v4/jlens_direction_diagnostic_<ts>/` with `manifest.json`, `records.jsonl`, `report.md`. Report must include:

- Table: direction × (gain percentile vs random, top-k subspace mass percentile, legal-vocab hit rank).
- Explicit interpretation guardrails: this is a correlational/diagnostic screen. High J-transported mass does NOT establish steerability (self-repair and multi-token dynamics can still null it); low mass is evidence for the workspace explanation of our causal nulls.
- A stated go/no-go for a future Phase B (steering with J-space projections under the full promotion harness: prompt-matched same-layer random controls, no-mask gate, ≥2048-token budgets per `qwen_eval_budget.py`). Phase B is out of scope for this brief.

## Constraints

- All code by Codex. Type hints, no bare except, tqdm for loops >10 iters (CLAUDE.md).
- Check HF cache before any download; never silently download large files.
- No secrets in source; `.env` only (see CONTRIBUTORS.md incident).
- This brief authorizes zero model generation runs. Forward passes for the Phase 0 smoke only.
