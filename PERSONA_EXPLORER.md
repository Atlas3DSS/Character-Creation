# Persona Explorer

Source and CLI layout: see [Persona source and checks](README.md#persona-source-and-checks). Local scripts live in `persona/`; the root `persona_explorer.py` remains the service launcher. The existing remote deployment keeps its flat filenames.

This update joins [Lyra’s neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) with Character Creation’s broader trait and layer experiments. Lyra’s work is the main reference for precise neuron sweeps, signed relative doses, the distinction between what a neuron responds to and what its output causes, and comparing complete rollouts. The local UI, experiment tools, and Flare adapter are independently implemented. No upstream source is vendored; no endorsement is implied.

The target is `nvidia/Qwen3.8-27B-NVFP4`, revision `482ca0f3832238542f8f5295dde86b5f22711d80`: 64 layers, hidden width 5,120, MLP width 17,408, or 1,114,112 MLP units. Model identity includes checkpoint, revision, quantization and dimensions. Calibration from another model is rejected.

## What is measured so far

The first reference run used all six leased V100s: 2,046 complete responses, no request errors or truncations, 1,650 valid single-digit IPIP replies, and 33 complete inventories. All 132 explicit numeric capability answers matched their references; this is only four simple tasks across eleven conditions and three seeds, not a broad reasoning benchmark. The [audited summary](reports/persona_baseline.json) separates unsteered and prompt-conditioned scores. Self-report changes under personality prompts do not establish a causal actuator or a stable persona.

All six V100s completed the 132 discovery captures, producing 64 calibrated layer banks and 1,656 provisional neuron candidates. The [calibration report](reports/persona_calibration.json) records artifact hashes and exact qualification scope: two prompts per shard, signed interventions at layer 32, and enabled-zero checks across the site catalog. Initial tokenization and pool-check allocation failures were repaired and remain in the local attempt history. The resident diagnostic worker also passed a prefill-only observation and an ordinary generation smoke test, with identical final-prefill planes and validated provenance. The explorer completed a three-arm signed comparison. These are operational and numerical checks; **causal personality effects, native cache ownership and speculative execution remain unvalidated**.

The separate local 27B planner has verified readiness with 131,072 context tokens and approximately 20.94 GiB of observed process VRAM. Its startup does not establish intervention quality. Inspect completed run artifacts before promoting any neuron label. An empty map means no measured candidates are loaded; the UI never fills it with simulated findings.

## Interventions and their limits

Lyra’s supplied Qwen3.5-4B recipe adds this scalar at neuron `n`, at the input to `down_proj`:

```
dose × median(norm(residual leaving layer L), tokens[1:]) / norm(W_down[:, n])
```

The neuron control adds this scalar to the **rounded SwiGLU product before the down projection**, rounds the result to the model’s BF16 values, and runs the existing quantized projection. Separate controls add a normalized decoded output-column vector **after the down projection**, or a fitted trait direction **after the block residual addition**. Quantized activation rounding and the projection epilogue mean these are distinct interventions. The adapter records each site and does not claim that their effects are equivalent. Steering-enabled preparation exposes the explicit SwiGLU product; ordinary preparation keeps its existing fused path.

Each layer has its own fresh, unsteered discovery-prompt residual median. The norm calibration excludes token position zero and pools baseline prompt tokens. A relative dose of one requests a vector with that layer’s median residual norm; the resulting rounded activation change can differ. Composite recipes sum vectors at each site. There is no silent norm clipping; the explicit optional aggregate limit is recorded if used.

Trait directions are equal-prompt means of paired high-minus-low final-prompt-token residuals. They are representation contrasts, not proven generation controls. Candidate selection combines two separate measurements: actual final-prompt-token SwiGLU activation contrasts, and cosine alignment between decoded `W_down` columns and the residual directions. The map retains baseline activation means, variation, and high-minus-low contrasts alongside output-direction alignment. Neither establishes a causal connection. The initial map layout is an index grid, not a connectome. Only heldout signed intervention experiments can establish response effects.

The historical Qwen3-VL-8B `reverse_L15` experiment steered L8–27 with a reversed contribution at L15; it did not reverse layer execution. Later controlled results were mixed. No layer is assumed inverted in this new model. Sign must be tested independently by layer, direction, dose, prompt and phase.

## Run the explorer

The local web service requires Python 3.11+ and NumPy; it has no web-framework dependency. Model preparation additionally uses Flare’s own pinned Python/CUDA environment.

```bash
python3 persona_explorer.py --port 8792
```

Open `http://127.0.0.1:8792`. The map shows coverage, neuron controls, trait sliders, layer ranges with an optional explicit sign reversal, and the original negative/baseline/positive comparison. Focus a layer to inspect its measured candidates in a layer popout. The Experiment workspace adds reusable recipes, corpus browsing and independently arranged batches of one to six jobs. The Studies tab shows the measured prompt-only reference. Lyra’s original 4B recipes are available in a separately labeled references tab; they are not applied to 27B by coordinate transfer.

Create ignored `persona_explorer.local.json` to connect verified local forwards:

```json
{
  "endpoints": [
    {
      "url": "http://127.0.0.1:8910",
      "model_identity": "SHA256 of the exact ModelSpec",
      "prepared_identity": "verified prepared-package identity"
    }
  ]
}
```

Start with `python3 persona_explorer.py --config persona_explorer.local.json --port 8792` to load that configuration.

Configure up to six distinct forwarded endpoints. Diagnostic endpoints also require their exact `runtime_identity` from `/v1/models`; an optional `label` gives each worker a readable name. Native model/prepared checks and diagnostic runtime checks remain mandatory for every job.

The service reads the native `/v1/models` capability catalog and checks the prepared identity, protocol, site names, vector widths and request site limit before sending steering. Older packages cannot silently accept steering as a no-op. Complete requests, responses, program hashes and vector provenance are retained in `results/persona_explorer.sqlite`.

### Live activation search

Open **Live activation search** (`/#activation-search`) and paste text. Each explicit search measures all 1,114,112 MLP neurons across every rendered prompt token, including chat-template tokens. It saves all-neuron peak and mean-absolute summaries, then replays the identical prompt to obtain exact token profiles for up to 32 ranked candidates. The viewer shows a signed heatmap, token context, coverage and request/model/runtime provenance. Inputs are never silently truncated. This is activation evidence; a large activation alone does not identify a concept.

Select one to six returned neurons and run **baseline / + / −** with the same prompt and seed. The total relative dose is divided across the selected neurons. Saved searches and three-arm comparisons remain available after refresh; failed or interrupted runs stay labeled. True neuron ablation is not implemented; negative additive steering is a separate operation.

The approved flat `activation-search/` checkout serves these requests separately from the fixed 40k capture runtime. It reuses the existing checkpoint and calibrated model, with its own hash-bound capture extension, supervised GPU leases, ports 8940–8945 and optional `activation_endpoints` explorer configuration. **Start search workers** explicitly requests a bounded session through the workboard; opening the viewer or reviewing records does not launch inference. The original runtime identity and existing results are preserved. Full-model deployment qualification is recorded in `results/persona_activation_search_validation.json`. The live checks covered 42-token and 309-token searches, exact replay profiles for 32 candidates, bit-exact unsteered captures before/after search, correctly signed measured activation changes, three complete comparison responses, and identical profiles after a second GPU restored the cached graphs. Worker readiness is checked separately in the viewer.

### Compare preset responses

In **Persona studio → Compare presets**, choose two to five catalog presets and a shared prompt (or one of the starter scenarios). **Run baseline + presets** runs each preset's center recipe plus one unsteered baseline with identical prompt, seed and answer budget. Preset names and briefs are not added to the response prompt. Comparisons use verified available studio or activation-search workers, and responses appear alongside one another with completeness, errors and exact recipe/provenance details.

Saved comparisons retain their original recipes, preset catalog, calibrated-direction hashes and generation settings even if a preset is later edited. Use the saved-comparison selector to review a run. This compares measured responses from authored steering presets; it does not establish that a preset reproduces a named character or model. The local example comparison includes GPT-4o-inspired, Skippy-inspired, Calm Mentor, Stoic Analyst and Bold Explorer, plus baseline.

### Build and compare recipes

1. Select a measured neuron on the map or in the layer popout, inspect its activation observations and output-direction alignment, and add an explicit positive or negative push. Choose the intervention site and phase, then use **Use recipe in a six-card experiment** to open the workspace with those pushes.
2. Name the recipe and edit its targets, doses or phase. **Save as new** creates a reusable model-bound recipe; **Update saved** edits that recipe. Quick dose multipliers create selected variants without overwriting the saved original. Variants remain drafts until saved. A zero multiplier preserves the explicit zero doses in the recipe; the separate **Baseline** recipe has no pushes.
3. Browse registered SCOTUS, Big Five and private book prompts by corpus, discovery/heldout split, or search. Labels identify book passages and mutations, legal prompt IDs, and Big Five items/profiles. Inspect the prompt and any profile system instruction before selection. Custom prompts are also supported; they do not alter the registered corpus or its split assignments.
4. Arrange **recipes × prompts** as **6 × 1**, **1 × 6**, **2 × 3**, **3 × 2**, selected combinations, or one to six individually edited slots. A baseline occupies a slot. **Paired baseline** explicitly creates a baseline and intervention for one prompt; the workspace never adds hidden control runs.
5. Set one seed and answer-token budget for the whole batch. The UI defaults to 2,048 tokens. Budgets below 2,048 are labeled preview/smoke and are not eligible for promotion or full-answer evaluation; SCOTUS API requests require explicit `smoke: true` to use them. Prefer 3,072–4,096 for legal evaluation when prompt plus answer fits the worker context. The current 4,096-token workers require an answer budget below 4,096, and the worker checks the exact combined token count without truncation. Thinking can be selected only when supported by the admitted workers.
6. Run the batch and inspect each cell as it finishes. Answers, completion status, errors, prompt/recipe snapshots and provenance persist locally. Editing a saved recipe or adding corpus records afterward does not rewrite a submitted job. Full requests, responses and dense steering vectors remain in private SQLite; the UI receives compact results and provenance. Partial successes survive other cells' failures. Restarted queued/running cells are marked interrupted, and inference is never silently retried.

Worker availability means **reachable with a verified identity**, not idle. The cards are shared with overnight sweeps. A common endpoint pool limits legacy comparisons and workspace batches to one UI inference request per configured worker; fewer reachable workers execute the admitted jobs in waves. A reachable model, prepared-package or runtime mismatch rejects admission. Transport loss after admission is recorded as a cell failure. The capacity display refreshes about every ten seconds and reports context and thinking support when known. These are leased services, not an always-on fleet: the 2026-10-25 overnight window ends around **09:00 America/Los_Angeles**, and later use requires newly available verified workers.

Live verification completed both **six recipes × one prompt** and **one recipe × six prompts** on all six Quincies: **12/12 cells completed**. The accompanying persona test suite passed 159 tests. Browser checks covered desktop and mobile layouts. These checks establish operation of the workspace, not a causal personality effect or native cache/MTP qualification.

### Persona studio

Open **Persona studio** for a profile-first workflow. The five dials range from −1 to +1 along this checkpoint's calibrated Big Five residual contrasts. The central diagram visualizes those settings, not a measured personality. Overall strength is a bounded normalized intervention budget across the chosen layers and traits; the default is 0.35 and the maximum is 0.5. This workspace builds additive recipes; it does not train, bake or replace model weights.

1. Start with a description, an editable dial profile, or a preset. **Ask John to interpret** explicitly requests a tentative five-trait interpretation from the local planner. Unsupported requested qualities remain visible. Manual controls work without John and are labeled as manual heuristics; there is no automatic model fallback or inference on a slider movement.
2. **Build candidate recipes** freezes a model-bound set around the current settings, defaulting to the configured card count, up to six. Review each candidate's dials and strength. **Use these dials** copies a candidate into a new editable draft; a rebuild makes those edits concrete. **Select only** chooses an exact existing candidate for the next comparison.
3. Review the shared battery of one to twelve prompts. Defaults come from distinct categories of original discovery Big Five tasks. Browse registered Big Five, SCOTUS and private book prompts, or add custom prompts separated by blank lines. Discovery and heldout labels remain visible. Using heldout answers to choose or tune a persona consumes that holdout; it cannot also serve as independent validation.
4. **Run shared battery** first generates one unsteered answer per prompt, then runs the selected candidate recipes in waves across the reachable workers. Every arm uses the same user text, empty neutral system, seed and answer budget. The persona brief is never injected into the evaluation messages. Registered profile/system conditioning is deliberately omitted from every arm and recorded in provenance. The baseline is immutable within that campaign and reused for every candidate comparison; a new campaign generates its own matched baseline.
5. Pin a candidate beside the unsteered assistant for each prompt. Complete, incomplete and failed responses stay separate, with saved batch records and provenance available. Ratings and **Prefer** are manual judgments, not automatic scores or proof of personality. **Save recipe** adds reusable signed pushes to the recipe library. Reopen a saved campaign to restore its exact build, prompts, seed and budget, select the desired candidate subset and rerun; rebuilding creates a new immutable set.

The default answer budget is 2,048 tokens with thinking off. Prompt plus answer must fit the admitted worker's context; the UI does not silently truncate. A battery of five prompts and six candidates creates 35 responses: five baselines plus 30 candidate answers. Up to six run concurrently after baseline waves; fewer reachable workers execute smaller waves. Reachable means identity-verified, not idle, and other experiments share the worker pool. **Stop future waves** prevents further campaign waves; an in-flight request may finish and remains in the record.

When no workers are reachable, **Start studio session** explicitly schedules a `studio_session` job for John and the six owned Quincies. Choose a duration, defaulting to two hours. Cold model preparation can take tens of minutes on this six-card fleet because the validated shared-cache publication stage is serialized; session time includes startup. Existing workboard admission, GPU ownership, thermal checks and the absolute deadline still apply. The UI reuses a current studio session or reachable fleet instead of automatically launching another. **Jobs & schedules** shows durable startup/status events; **Stop this studio session** requests cooperative cleanup of that session's own resources. Opening the tab, browsing presets and moving dials do not start GPUs.

The original 16 presets are:

| Inspired starting points | Basic archetypes | Basic archetypes | Basic archetypes |
| --- | --- | --- | --- |
| GPT-4o-inspired | Calm Mentor | Warm Companion | Skeptical Investigator |
| Skippy-inspired | Curious Scientist | Playful Storyteller | Bold Explorer |
| — | Pragmatic Engineer | Decisive Captain | Meticulous Archivist |
| — | Stoic Analyst | Gentle Diplomat | Creative Muse |
| — | Competitive Strategist | Quiet Philosopher | — |


Preset previews show the brief, five dials, citations and approximation notes. **Load preset & build candidates** uses the fixed authored catalog without consulting John, preserves its catalog hash and notes in the plan/build, and saves its center recipe. Repeated loading updates that preset's model-bound saved recipe rather than creating duplicate library entries. The remaining variants stay in the immutable build until individually saved.

The [catalog](data/persona_presets.json) cites the [GPT-4o model reference](https://developers.openai.com/api/docs/models/gpt-4o), [OpenAI personalization guidance](https://learn.chatgpt.com/docs/personalize), and Craig Alanson's [Skippy-voiced page](https://www.craigalanson.com/flightjacket). These sources provide identity or style context; they do not establish the numeric dials or a transferable recipe. GPT-4o-inspired is not recovered OpenAI parameters or a particular ChatGPT snapshot. Skippy-inspired is an interpretive homage, not a recipe fitted to book dialogue or a validated reproduction of the character. All profiles are authored heuristics whose behavioral effects and polarity require matched comparisons. No book passages are included in the public presets.

Operational validation on 2026-10-25 saved all 16 distinct checkpoint-bound center recipes and ran both named presets through the five-prompt discovery battery with six candidates each. All 70 responses completed with a 2,048-token budget and thinking off; the five unsteered answers matched exactly across the two campaigns at seed 7035. John also returned a valid structured brief interpretation. These checks establish functioning interventions and reproducible comparisons, not GPT-4o or Skippy fidelity. The 252-test persona suite passed; the subsequently revised scheduler passed its 19 tests, including strict post-generation text and facet bounds.

Studio records live in the existing private explorer SQLite database. Its local routes are `GET /api/sculptor`, `GET /api/sculptor/presets`, `POST /api/sculptor/preset`, `POST /api/sculptor/plan`, `POST /api/sculptor/build`, `GET/POST /api/sculptor/campaigns`, `GET /api/sculptor/campaign?id=…`, and `POST /api/sculptor/review` or `/api/sculptor/cancel`. A campaign references an immutable build ID and an optional exact candidate-ID subset. Interpretation, building, scheduling and inference remain distinct actions. The studio adds profile authoring and controlled comparisons; it does not fill the logit-lens, weight-path-web, per-token activation or streaming multi-turn gaps listed below.

### Local workspace API

All routes belong to the loopback explorer service. Corpus access reads only the completed, hash-bound `data/persona_manager_corpus.jsonl` publication and its manifest; source files remain private and unchanged.

| Method and route | Purpose |
| --- | --- |
| `GET /api/corpora` | Corpus names and discovery/heldout row counts. |
| `GET /api/prompts?corpus=&split=discovery&q=&offset=0&limit=12` | Searchable prompt page with IDs, labels, text, system instructions and suggested budgets; `split` also accepts `heldout` or `all`. |
| `GET /api/saved-recipes` | Saved names, model identities, pushes, phases and creation/update times. |
| `POST /api/saved-recipes` | Validate and save `{name, model_identity, pushes, phase}`; include an existing `id` to update it. |
| `POST /api/batches` | Validate every cell before enqueueing a one-to-six-cell batch; returns `{id}`. |
| `GET /api/batches` | Recent batch states and completed/total cell counts. |
| `GET /api/batch?id=...` | Persisted input, cell states, snapshots, endpoint labels, compact outputs and errors. |
| `GET /api/capacity` | Configured/reachable worker counts, per-card availability, thinking support and context limit; idle status remains unknown. |
| `GET /api/neighbors?layer=15&neuron=242&corpus=all&scope=all&limit=12` | Signed co-activity among observed candidates; filter corpus and `all`, `same_layer` or `cross_layer` scope. |

For example, submit an explicit baseline and one saved recipe against the same registered prompt. Obtain `model_identity` from `/api/state` and the prompt/recipe IDs from their respective lists:

```json
{
  "model_identity": "EXACT_MODEL_IDENTITY",
  "seed": 7035,
  "max_tokens": 2048,
  "thinking": false,
  "smoke": false,
  "cells": [
    {"label": "Baseline", "prompt_id": "REGISTERED_PROMPT_ID",
     "recipe": {"name": "Baseline", "pushes": [], "phase": "all"}},
    {"label": "Intervention", "prompt_id": "REGISTERED_PROMPT_ID",
     "recipe_id": "SAVED_RECIPE_ID"}
  ]
}
```

Each cell accepts exactly one of `prompt_id` or a custom `prompt` string, and exactly one of `recipe_id` or an inline `{name?, pushes, phase}` recipe. The shared budget must be 256–4,096 tokens, subject to advertised context limits. `thinking` and `smoke` are explicit booleans. Recipe phases are `all`, `prefill` and `decode`. The existing `/api/compare` and `/api/comparison` routes remain available for the original signed comparison.

### Inspect a neuron's co-activity neighborhood

Select a measured neuron, then open its neighborhood breakout to compare candidates whose activations rise together, move oppositely, or vary weakly together. Neighbor controls return to that measured neuron on the map. Filter by corpus and by the same layer, other layers, or both. The graph is undirected: a positive or negative Pearson correlation is an association across prompts, not a direct wire, causal influence, or measured temporal phase. These are continuous signed SwiGLU values, not binary firing events.

The first local source contains **5,060 distinct original unsteered discovery prompts**: **5,035 book passages, 13 SCOTUS prompts and 12 Big Five tasks**, measured at the final prompt token across **1,656 mapped candidates**. It excludes heldout prompts, source mutations, personality prompt profiles and intervention arms, and retains one row per original source group. An audited transition removed 15 duplicate-text observations from statistical weighting: twelve had identical measured vectors, while three whitespace variants had different activations. Those differences are recorded and all raw captures are preserved. It is a candidate neighborhood, not an exhaustive million-neuron connectome. Each result shows its actual sample count. The small SCOTUS and Big Five cohorts are exploratory; no significance or multiple-comparison claim is made. Pooled correlations can reflect differences between corpora, so compare within each corpus before interpreting a strong relationship.

Display conventions are `|r| >= 0.6` for strong, `|r| <= 0.2` for weak, and moderate between them. Same-direction and opposite-direction lists rank their non-weak coefficients; the weak list shows coefficients nearest zero. Constant neurons, missing observations and filters with fewer than eight unique prompts have no fabricated correlations.

`persona/persona_coactivity_extract.py` derives the private compact source from the existing manager database and captured arrays. It verifies job identity, original-prompt selection, unsteered request contents, runtime/model identity, response completion, provenance-file hashes, activation-file hashes and the measured-map hash. Run it on the host holding the raw captures:

```bash
python3 persona/persona_coactivity_extract.py --root "$PWD"
```

The default output is `results/persona_coactivity.npz`, created exclusively. Explicit `--replace` requires preservation of prior rows except audited duplicate exclusions. `--stream` emits the numeric archive to stdout for direct transfer to the explorer host; progress goes to stderr. The archive contains only candidate activations, coordinates, corpus labels and source hashes, with no passage text. Keep it private with the other results. `persona/persona_neighbors.py` validates that publication against the current model/map and caches corpus statistics for interactive queries; it never scans raw captures during a click or starts GPU work. The source and viewer extend the independently implemented exploration workflow inspired by [Lyra's neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer).

### Schedule John and the six Quincies

Open **Jobs & schedules** to create a timed **Big Five discovery expansion** or **Adaptive neighborhood search**. Choose a local date, IANA time zone, duration, target, hard cap and seed. The default proposes the next 2 AM in America/Los_Angeles; nothing runs until you save a schedule. The server resolves the date and rejects ambiguous or nonexistent daylight-saving times. Job details show the actual start/deadline, scheduler heartbeat, progress, recorded errors and cooperative cancellation.

The fixed fleet adapter launches John with a 128K context and six distinct, UUID-bound Quincy workers through gpu-workboard. Waiting for existing work, admission and model startup consumes the scheduled window. Existing workloads are preserved. A job can therefore finish below its requested target; the UI reports actual counts. Services have a bounded lifetime, and cleanup verifies the job's own launch identity. The scheduler persists schedules, requests, raw responses, invalid outputs and accepted samples in `results/persona_jobs.sqlite`. Restart recovery keeps completed evidence and does not blindly replay model requests.

**Big Five expansion** produces short, original scenario prompts across the five traits, thirty facet design categories and varied situations. John assigns six batches, then the Quincies generate them in parallel. The harness validates complete responses and removes normalized text duplicates before counting accepted prompts. Invalid or truncated responses are quarantined. The resulting rows are **synthetic discovery material**, not a validated psychometric inventory or measured personality labels. Their local publication appends to `data/persona_manager_corpus.jsonl`, preserving all previous records and heldout splits; `data/persona_generated_big_five.jsonl` and its manifest identify the generated subset. These private data files are not published to GitHub.

**Adaptive neighborhood search** starts with explicit neuron coordinates and selected discovery corpora. Use **Plan adaptive search** from a neuron's neighborhood, or enter any model-valid coordinate at a calibrated layer. John chooses bounded prompt families and doses from a fixed catalog. Each admitted experiment includes baseline, zero, random, positive and negative controls, matched source variants and seeds. A hard cap never truncates a control set. The objective rewards new source coverage and within-family variation, rather than a large pooled correlation. Repeating a seed does not create a new independent prompt. Complete-rollout followups require prior matched prefill evidence and an answer budget of at least 2,048 tokens.

After owned GPU cleanup, the scheduler ingests the job's verified original-baseline registry, preserves prior compact observations, and atomically refreshes the viewer's co-activity source. This bounded CPU publication step can finish after the GPU deadline; it creates no new GPU lease. Cancellation or failed verification leaves the last valid source intact and records the outcome. New observations update the existing measured-candidate index; testing an unmapped coordinate retains its raw evidence without silently declaring a new curated map node.

Constant neurons remain excluded from correlation. Missing or insufficient observations remain visible as such while further sweeps gather evidence. Matched intervention effects are descriptive candidates until separately validated; a correlation or repeated sign is not an anatomical connection, temporal phase, or established personality mechanism. John receives structured data and emits a strictly validated plan; it cannot supply shell commands, executable paths or arbitrary tool calls.

The host operator configures `persona_jobs.local.json` beside the explorer:

```json
{
  "enabled": true,
  "root": "/absolute/path/to/Character Creation",
  "fleet_config": "/absolute/path/to/Character Creation/persona_fleet.local.json",
  "expected_prepared_identity": "EXACT_PREPARED_PACKAGE_SHA256",
  "expected_runtime_identity": "EXACT_RUNTIME_SHA256",
  "lease_wrapper_sha256": "VERIFIED_INSTALLED_WB_HOLD_SHA256",
  "timezone": "America/Los_Angeles"
}
```

Run `python3 persona/persona_jobs.py validate-config --config persona_jobs.local.json`, then supervise `python3 persona/persona_jobs.py daemon --config persona_jobs.local.json` as a persistent user service on the configured resource owner. `--once` scans for one due job; it creates no resources if nothing is due. The reference deployment uses `character-persona-jobs.service` with automatic restart and a 600-second cooperative stop allowance. The fixed adapter is currently configured for the owner's Devbox/John and six-card V100 host; other installations must explicitly configure and review their own fleet bindings. `persona/persona_lease.py` wraps the hash-verified installed workboard supervisor to register exact worker ownership without bypassing its admission, thermal or memory guards.

| API | Purpose |
| --- | --- |
| `GET /api/jobs` | Schedules, progress, time zone and scheduler heartbeat. |
| `POST /api/jobs` | Save a validated `big_five_expand`, `neighborhood_hillclimb` or `studio_session` schedule. |
| `GET /api/job?id=...` | Job state and recent durable events. |
| `POST /api/job/cancel` | Cancel a queued job or request cooperative stopping of its owned work. |

Schedule mutations require same-origin JSON requests and an enabled operator configuration. Specify `local_start` plus `timezone`, or `start_at` with an explicit matching UTC offset. Required fields are `kind`, `name`, `duration_minutes`, `target_entries`, `max_entries` and `seed`; neighborhood jobs also require nonempty `targets` (`layer`, `neuron`) and `corpora`. Studio sessions use target and cap 1, omit corpus/target coordinates, and hold ready workers without automatic experiments. The UI uses the same API and distinguishes synthetic prompts, observations and ready sessions.

A bounded operator validation under an existing owned Studio session completed 165 observations across all six cards: 33 complete five-arm sets with exact baseline/zero activation equality and no errors. Its registry and atomic coactivity publication passed. The three eligible original prompts were already represented, so the source remained at 5,060 rows; no new links or independently replicated effects were established. This validates one planning/control/publication cycle, not a completed multi-round hillclimb. The overnight generator's corrected compact JSON grammar also returned a valid six-assignment plan; text bounds and trait/facet compatibility remain enforced in Python before dispatch.

### Feature comparison with Lyra's explorer

Source review: **2026-10-25**, upstream `main` at **`f79b2196e29aacbad89684938baac8d97b172fd1`**. This is a feature inventory, not a claim of full parity. The references are Lyra's [interface](https://github.com/lyramakesmusic/neuron-explorer/blob/f79b2196e29aacbad89684938baac8d97b172fd1/explorer/static/index.html), [interactive behavior](https://github.com/lyramakesmusic/neuron-explorer/blob/f79b2196e29aacbad89684938baac8d97b172fd1/explorer/static/app.js), [server](https://github.com/lyramakesmusic/neuron-explorer/blob/f79b2196e29aacbad89684938baac8d97b172fd1/explorer/server.py) and [model runtime](https://github.com/lyramakesmusic/neuron-explorer/blob/f79b2196e29aacbad89684938baac8d97b172fd1/explorer/runtime.py). All listed gaps concern the new Qwen3.8-27B Persona Explorer; older experiment scripts elsewhere in this repository do not establish UI support.

| Lyra feature | Character Creation status |
| --- | --- |
| Perspective model map, layer selection and face-on layer focus | **Present**, with bounded viewport, animated focus and measured-candidate navigation. Full all-neuron density and attention-head inspection are not present. |
| Index layout and output-column-similarity `writes` layout | **Present for mapped candidates**: switch between the index grid and per-layer PCA of normalized decoded output columns. |
| Positive/negative neuron logit lens | **Present**: all 1,114,112 MLP neurons, decoded NVFP4 weights, zero-centered final RMSNorm and FP32 output projection. Top 24 tokens per sign are stored. |
| Empirical steer lens: next-token changes and KL under signed pushes | **Implemented in Live tokens & lens**: matched final-prompt next-token distributions, full-vocabulary KL, Δlog p, probabilities and saved comparisons. Runtime qualification is recorded separately. |
| Search neurons by tokens in either lens | **Present with bounded scope**: static top-token search across the full atlas; empirical search across retained top effects in the latest 50 completed trace campaigns. Neither is exhaustive. |
| Weight-derived upstream/downstream links and multi-hop `web` cones | **Present**: strongest 16 adjacent-layer down→gate/up affinities in each direction; interactive 3D cones with 1–3 hops and bounded fanout, separate from measured co-activity. |
| Token summaries pooled over a cone | **Present**: unweighted counts of stored top-token projections among displayed neurons, explicitly not predicted multi-hop effects. |
| Neuron readout over pasted text, token hover/pinning and activation paths | **Selected-neuron tracing implemented**: prompt/decode heatmaps, hover, scrubbing, playback and live polling. Up to 64 selected neurons, at most 32 per layer. Full-model activation search and token-conditioned causal paths remain absent. |
| Distinctive-word/cluster ranking against random directions | **Missing** in this viewer. Current candidates use explicitly provisional trait contrasts and output alignment. |
| Saved examples grouped by neuron, prompt and dose | **Partial**: persistent experiment cells and full provenance exist; no unified per-neuron archive browser across all sweeps. |
| Three-column signed, streaming, multi-turn chat | **Partial**: signed single-turn comparison and flexible six-card batches exist; no independently continued streaming chat histories. |
| Additive and multiplicative neuron controls | **Partial**: additive pre-down, output-column and broader residual controls; no multiplicative neuron/MLP controls. |
| Attention-head/whole-layer scaling and ablation | **Missing** from the new native steering interface. |
| Rollout embedding scatterplot with neuron/prompt/dose/kind coloring | **Missing**. |
| Editable neuron names and persistent shared links to a selection | **Missing**. Saved recipe names are a different feature. |
| Interactive model assistant, MCP tools, map highlights, attachments and chat rewind | **Different scope**: John manages constrained scheduled experiments and tentative Studio brief interpretation; no equivalent interactive explorer assistant or general file/shell tools. |
| Next-token probability inspection and baseline/intervention distribution comparison | **Implemented** in the empirical lens: raw model distributions before sampler filtering, paired at the shared final prompt token. |
| Full signed rollouts, random controls and cross-runtime equivalence checks | **Partial**: native matched-control experiments and quantized zero/neutral checks exist; no claim of exhaustive neuron coverage or HF/vLLM equivalence for this checkpoint. |

The static lens and weight web are now computed for the exact 27B checkpoint. Co-activity and weight connections both use orbitable **3D** views, with depth cues, signed edge styling, zoom, focus and keyboard selection. Co-activity geometry uses three principal coordinates of measured normalized profiles; the weight web offers a volumetric spectral embedding of retained affinities and a layer-depth view using output-direction coordinates. These are labeled embeddings of high-dimensional relationships, not literal physical coordinates or proof of wiring. Constant/unobserved co-activity sources still produce no unsupported edges.

Remaining parity gaps include attention-head controls, multiplicative edits, full-model token-activation search, distinctive-word ranking against random directions, rollout embeddings, a unified archive browser, persistent neuron naming/links, multi-turn streaming chat and Lyra's interactive model-assistant integration. Production cache and MTP qualification also remain separate.

### Checkpoint atlas and token tracing

The atlas builder independently implements the signed output-direction lens inspired by Lyra. It decodes the checkpoint's NVFP4 lm_head and MLP projections, uses the checkpoint's final zero-centered RMSNorm (1 + weight, epsilon from config), and projects in FP32. This is a static reference calculation, not native quantized activation arithmetic. The web ranks signed cosine between each down column and the next layer's gate/up rows, retaining the channel with larger absolute cosine. Attention, normalization and input-dependent gating are omitted.

Build six deterministic shards under your existing GPU admission/lease wrapper:

```bash
python persona/persona_atlas.py --root . --checkpoint /path/to/pinned/checkpoint --config persona_fleet.local.json --shard 0
# Run shard 0..5 on the corresponding leased cards.
python persona/persona_atlas.py --root . --publish
python persona/persona_atlas.py --root . --geometry
```

The builder requires the explicit fleet host/UUID and its own running workboard parent lease. It checks calibration/config/tokenizer bindings, hashes every checkpoint shard, validates decoded columns against the CPU decoder and records its source/lease identity. It writes results/persona_atlas_L*.npz; publication validates all layers and writes a hash-bound manifest. The reader refuses mixed model/source identities, changed artifact hashes and missing data. These generated files stay local. Existing layer outputs are not overwritten by the builder.

**Live tokens & lens** offers one prompt × six signed recipe strengths (baseline, zero, +½, +1, −½, −1), or three prompts × baseline/recipe. It uses shared endpoint slots and the current recipe, with explicit observed-neuron coordinates. The 1–512-token budget is diagnostic; a length-limited continuation is not a complete scored personality response.

Native trace buffers hold 32 selectable neuron columns per layer and are shared by prefill and ordinary B1 decode graphs. A request selects at most 64 neurons overall. Observations are indexed by **consumed input position**: a sampled token has activations only after the next replay consumes it; an unconsumed final/EOS token is never assigned fabricated activations. The UI polls completed chunks during execution and saves exact NPZ observations plus durable local playback/audit records. The trace is optional and the normal response/capture interface remains available.

The empirical lens compares full-vocabulary raw next-token distributions at the same final-prompt prefix, before sampling filters or penalties. Later freely generated continuations can diverge and are not treated as aligned KL measurements. The UI reports both KL directions, Δlog probabilities, baseline/intervention probabilities and highest-probability next tokens. Zero-dose output and trace equality are explicit controls. A decode-only intervention should have zero final-prompt KL.

On-demand weight access lets a ready, identity-verified worker supply any atlas neuron's decoded output column for calibrated steering. The local vector bank verifies model/runtime and checkpoint/calibration bindings before caching it. Without a ready worker or local checkpoint, an unextracted column remains unavailable rather than receiving a guessed vector.


## Calibration and experiments

`persona/persona_fleet.py` runs deterministic six-way generation shards. It checks host, GPU UUID, workboard parent lease and pinned executable/package identity, records complete sampling requests, monitors resources, and terminates only its owned server. Startup attempts and failures remain in the same artifacts. The HDD used in the reference run needed about 500 seconds for six direct-I/O weight loads; those runs completed after extending startup readiness to 900 seconds.

`persona/persona_capture.py` generates 132 discovery examples: twelve user prompts under eleven matched conditions. Its six disjoint shards replay actual retained Flare init/prefill graphs, retain normal admission/completion validation, and save prompt-token residual norms plus final-token residuals, normalized pre-MLP inputs and the actual rounded SwiGLU activations. It checks token positions and restores initial state. No heldout prompts fit the directions. Each launch must be wrapped in its own workboard lease; the script rejects direct unleased execution. Failed artifact-free attempts can resume in place with their prior reports retained.

```bash
python3 persona/persona_capture.py plan --root . --output results/persona_capture_plan.json
# On each leased V100, with explicit checkpoint, Flare source and rebuilt extension:
python3 persona/persona_capture.py run --config persona_fleet.local.json \
  --plan results/persona_capture_plan.json --shard 0 --compile-workers 14 \
  --flare /path/to/flare --checkpoint /path/to/pinned-checkpoint \
  --heliopause /path/to/flare_heliopause.so --output results
```

Repeat the capture command with shards 0–5 and disjoint CPU affinities. Preparation uses the SM70 MTP weight path but replays ordinary non-speculative prefill for capture. The extension must be rebuilt with the new protocol; do not accidentally import a stale worktree shim.

After all six captures validate:

```bash
# Run shards 0–5 with the same checkpoint and capture plan.
python3 persona/persona_map.py layers --plan results/persona_capture_plan.json \
  --results results --checkpoint /path/to/pinned-checkpoint --shard 0
python3 persona/persona_map.py assemble --results results
```

These produce model-bound `layer_00.npz` through `layer_63.npz` and a candidate `persona_map.json`. The builder checks shard completeness, hashes, model identity and paired conditions before fitting. Existing evidence is preserved rather than overwritten.

`persona/persona_experiments.py` defines calibration, coarse, refinement and heldout validation plans, including signed doses, zero, prompt-only, shuffled and matched-norm random controls. Its broader operation vocabulary is planning metadata: `mlp_scale` and per-thinking/answer-phase control are not implemented by this native additive protocol. Never submit an unsupported plan as if it were executed. Keep measured self-report scores, behavioral review, capability retention and incomplete responses separate.

## Overnight discovery with measured activations

The overnight workflow uses one resident diagnostic worker on each of the six V100s, a local 27B planning model, and a CPU queue manager. The planning model proposes experiments; the manager validates and dispatches them. The target checkpoint remains the pinned NVFP4 27B model described above. The planning model is a separate cached Q4_K_M GGUF, served by llama.cpp with a verified 128k context. Its judgments are hypotheses, not an independent evaluator or evidence that a trait changed.

`persona/persona_probe.py` exposes loopback `/v1/models` and `/v1/chat/completions` endpoints while retaining the actual Flare init, prefill, ordinary decode and sampling graphs. One request runs per card at a time. Each request starts from restored state and saves observations at the final valid prompt token before generation continues. This diagnostic mode uses **MTP-capable preparation and B1 ordinary-only execution**, with no native prefix-cache reuse or speculative execution. Preparation matches the passed calibration: four request slots, `mtp=True`, widths `(1,2,3,4,5)` and inherited row-invariant target arithmetic. SM70 prefill requires that layout's seven convolution-history columns. Only ordinary one-request graphs execute; every replay checks that the other three slots remain inactive.

The default context is 4,096 tokens backed by 32 physical pages, matching calibration. The recorded real-tokenizer preflight found all 13,895 current corpus rows fit with their full rollout budgets; the largest prompt-plus-budget was 3,155 tokens. Every new request is independently rejected if prompt plus budget exceeds physical or logical capacity; no truncation is used. Startup must reproduce one original calibration prompt's encoded tokens, all three final-prefill observation planes and every non-sink residual norm bit-for-bit before the endpoint becomes ready. This remains a separate qualification path from the exported production server.

An explicit `observation_only=true` request requires `max_tokens=1`: it runs the actual sampler once after complete prefill and skips decode. The response has `operation="prefill_observation"`, `prefill_observation_complete=true` and `complete_final_answer=false`; response text is empty so the sampled first token cannot be graded as an answer. The token and full activation artifact remain in private provenance. Normal rollout requests retain their full generation budget and behavior.

The observations are continuous, BF16-rounded SwiGLU products, not binary “neuron firing” labels. Per-request NPZ files contain `mlp_input_last[layers, intermediate]`, `residual_last[layers, hidden]`, `pre_mlp_last[layers, hidden]`, prompt/generated token IDs and prompt residual-norm summaries excluding position zero. The MLP observer records the value after an optional neuron push and before `down_proj`. It does not record every neuron's complete token-by-token trajectory. Comparing matched prompt mutations and interventions tests which units respond and which changes affect subsequent behavior; it does not by itself establish a synaptic connectome.

The response links the activation artifact and request/response provenance by SHA-256, model identity, prepared identity and request hash. `persona/persona_manager.py` verifies those artifacts and gives the planner bounded numerical summaries: the selected neuron's activation, layer activity/residual norms, and same-prompt/same-seed differences against a completed baseline. Full arrays and rollouts stay available for later analysis.

Activation archives use lossless NPZ deflate level 1. A [recorded single-artifact CPU comparison](data/persona_npz_compression_proof.json) reduced serialization from 1.182 seconds at level 6 to 0.170 seconds at level 1, while archive size increased from 3.400 MB to 3.767 MB. Every reloaded array retained its exact dtype, shape and bytes; focused tests also compare complete uncompressed NPY members against NumPy's previous writer. These timings do not establish end-to-end GPU throughput. Archive hashes and the source-bound runtime identity change, so deployment requires an audited runtime transition; calibration, state-restoration and artifact validators remain enabled.

Every proposed intervention expands into five arms with the same prompt, sampling settings and seed:

| Arm | Intervention |
|---|---|
| Baseline | No steering program |
| Zero | An explicit zero vector at the selected site |
| Random | Another neuron in the same layer, or a seeded vector at the same output site, with matched relative norm |
| Positive | The proposed dose magnitude |
| Negative | The same magnitude with the opposite sign |

`prompts_per_corpus` counts unique source `pair_id` groups, selected deterministically from the requested split. Every registered mutation or trait profile in each selected group is included. Originals, instruction changes and available speaker swaps share a per-repetition seed across the group, as well as the same five steering arms within each prompt. Each job records its `pair_id` and `source_group`; minimal corpora without a `pair_id` fall back to the prompt ID. The entire expanded batch must fit the job budget or it is rejected atomically. Variants are never silently dropped, and a source group crossing discovery/heldout splits is rejected.

An experiment can choose `measurement: "prefill"` or `"rollout"`; omitted measurement defaults to rollout. The planner is instructed to use roughly 80% prefill measurements for broad neuron/dose/source searches and 20% complete-rollout followups. This is a planning target, not an enforced observed ratio. Prefill measurements require the worker's advertised observation-only capability and send `observation_only: true, max_tokens: 1`. They replay the prompt and its one final-position sample, save actual activation arrays, and perform no decode. The API returns no answer text. Jobs record `complete_observation` separately from `complete_answer`, and one sampled token is never graded as a complete response. All five control arms and matched source variants still apply. A decode-only intervention has no effect during this prefill measurement and is useful only as an explicit phase-neutral control.

Neuron and output-column controls share a decoded output-equivalent dose convention; quantized pre-down and post-down effects remain distinct. Random-neuron matching does not assert equal realized activation or output changes after quantization. The planner prioritizes layers 10–18 and explores both signs and small dose ranges. Lyra's Qwen3.5-4B coordinates can be tested as explicitly labeled cross-model hypotheses; their reported effects and signs are never transferred as established 27B facts.

The manager accepts one strict JSON proposal schema: target identity, batch seed, rationale and bounded experiments naming registered corpus labels, layer, neuron or trait, signed relative dose and `all`/`prefill`/`decode` phase. It rejects unknown fields, invalid coordinates, unsupported sites and uncalibrated model identities. It cannot accept shell commands, endpoint changes, file writes, credentials or instructions embedded in corpus text. Worker-side vector construction can read arbitrary columns from the locally pinned checkpoint when the calibration/configuration hashes match.

SQLite stores proposals, immutable prompt records, jobs, attempts and an append-only audit. Claims are transactional. Interrupted requests retain their IDs and seeds; an unknown remote completion is recorded before a bounded retry, so recovery is explicitly at least once. Error budgets count failed and incomplete answers; thermal admission requires the existing owner lease and fresh telemetry. The manager does not acquire GPUs or stop other services. Its deadline defaults to the next 08:00 in `America/Los_Angeles`, capped at sixteen hours; use `--until` for a specific analysis cutoff.

### Corpus and holdouts

Prepare the initial SCOTUS and Big Five corpus from the repository root:

```bash
python3 persona/persona_corpus.py prepare
```

The initial SCOTUS/Big Five manifest contains 374 records: 60 SCOTUS and 314 Big Five prompts, with discovery and heldout splits kept separate. The October 25 local corpus contains 13,895 records after ingestion of eleven owner-provided EPUBs, including 231 explicitly labelled counterfactual speaker-attribution variants. Those variants exchange named Skippy/Joe attributions where the original speech tag is explicit; dialogue words and narrative mentions remain unchanged. Unresolved speakers remain unlabeled. The book texts and local manifest are not public artifacts. SCOTUS requests reserve 3,072 generated tokens. IPIP-50 items retain their original item IDs and reverse keys and stay in the heldout inventory-measurement split. The adaptive planner receives discovery observations only. A fixed heldout proposal can be queued with `persona/persona_manager.py enqueue --validation`; heldout responses never enter adaptive feedback.

Raw books and extracts remain private and ignored by Git. The corpus starts without Skippy material if the books are absent; it does not relabel other prompts as book evidence. When explicitly selected local EPUB/TXT files arrive, ingest them without copying the source books:

```bash
python3 persona/persona_corpus.py ingest --book-dir /path/to/local/books
# Optional long-running ingestion watcher, supervised independently:
python3 persona/persona_corpus.py watch --book-dir /path/to/local/books --interval 30
```

Content hashes prevent duplicate ingestion. Existing prompt text, provenance and split assignments remain immutable. The manifest first marks ingestion in progress and publishes `status: complete` with the new JSONL hash afterward. The manager reloads only a complete, matching generation. Public reports should contain manifests and aggregate results, not private book passages or unreviewed rollout excerpts.

`persona/persona_coverage.py` adds a separate operator pass for exhaustive registered-row coverage, independent of the adaptive planner's group sampling. Explicit `--split all` queues one unsteered prefill observation for every registered row, including all book chunks and registered variants, SCOTUS and Big Five. Original prompts, source hashes, offsets, pair IDs and discovery/heldout labels are preserved; related variants share a deterministic seed. Layer 0 is only the existing executor's calibration-bound summary reference: no neuron or trait intervention is applied. Heldout results remain excluded from adaptive feedback by the manager's existing split filters.

```bash
python persona/persona_coverage.py plan --config results/persona_manager_config.json --split all
# Operator action: append the entire pass atomically to the existing manager queue.
python persona/persona_coverage.py enqueue --config results/persona_manager_config.json --split all
python persona/persona_coverage.py status --config results/persona_manager_config.json --split all
```

Plan/status output contains counts and hashes, not raw text. Repeating enqueue with the same corpus generation, split, seed and calibration is idempotent; exceeding `--max-jobs` or the manager's total-job cap rejects the whole pass without truncation. Enqueue starts no service itself. Status counts only completed prefill observations and distinguishes queued, failed and remaining rows. Each artifact covers all layers at the final prompt token plus prompt residual norms; this is not every neuron's complete token-by-token trajectory, and queued coverage is not completed coverage.

### Character evidence before broader speaker swaps

`persona/persona_characters.py` inventories character names in original book chunks through the already leased local John planner. It excludes generated mutations and defaults to discovery sources only. The model returns a name and a short, unique verbatim evidence phrase containing that name, with optional evidence-backed alias candidates. Software locates the phrase and name and computes exact Unicode offsets; the model never counts character offsets. Repeated evidence requires an explicit bounded occurrence index, otherwise it is rejected. Each balanced quote gets one software-generated unresolved speech record. Narrative speaker assignment is deferred. Character recall and entity classification remain unmeasured; exact source evidence alone does not establish either. Alias candidates are not automatically merged into identities.

The private SQLite index preserves source book/document hashes, split, normalized-document offsets, raw model responses, failed attempts and coverage counts. Its corpus and planner-runtime binding prevents silently resuming against changed inputs. The October 25 run is underway across all 6,645 original chunks from eleven books, including both discovery and heldout splits. An early checkpoint recorded roughly 96 complete and 75 failed chunks under strict evidence validation; these pipeline statuses are not accuracy estimates or completed corpus coverage. The index remains provisional: source-phrase acceptance does not validate character classification, recall or aliases. All splits and index outputs remain isolated from adaptive experiment planning. The sampling order covers attributed dialogue, other dialogue and narrative across books before continuing; chunk-attempt limits, a deadline and error budgets bound the run, and outputs are capped at 2,048 tokens. Rejected attempts remain audited, and neither exhaustive completion nor character accuracy is claimed.

```bash
# On John, using its existing planner lease and launcher readiness manifest:
python persona/persona_characters.py run --lease GPU_WORKBOARD_PLANNER_JOB_ID \
  --limit 8 --batch-size 1 --minutes 5
python persona/persona_characters.py status
python persona/persona_characters.py export --output results/persona_characters.jsonl
```

The indexer checks `/slots` and waits for a configurable idle interval before each request. This is cooperative yielding, not atomic priority: stop indexing when experiment planning needs predictable access to the single planner slot. Repeating `run` resumes the same private index and retains failures. Explicit `--split all` can expand a discovery-only index in place when the corpus hash and planner binding are unchanged. It appends heldout chunks with their split labels, preserves prior rows and attempts, and audits the scope change; narrowing the index afterward is rejected. This is annotation only: neither raw responses nor index labels feed the adaptive experiment planner, and heldout material must not feed adaptive discovery. Source passages, raw responses and exports stay private.

`--error-budget` limits requests containing rejected chunks per invocation (default 3, maximum 100,000). `--consecutive-error-budget` independently stops repeated wholly rejected requests (default 5); any validated chunk resets that streak. Both counters, the separate number of rejected chunk attempts and the stop reason are audited. After checking the complete response schema and exact requested chunk IDs, batches validate each chunk independently: good chunks are saved, and only rejected chunks retry, at most twice in total. A chunk is accepted only when all its annotations validate. Raw responses and per-chunk errors remain in SQLite. A reviewed larger run can use `--batch-size 4 --error-budget 10000 --consecutive-error-budget 10` with an explicit `--minutes` deadline and `--limit`; rejected evidence is never accepted to keep the run moving.

### Start the workers and manager

Use Flare's pinned environment from the separate [V100 runtime setup](https://github.com/Atlas3DSS/flare-local-inference/blob/atlas/sm70-public/README.md), the rebuilt Heliopause extension and the exact local checkpoint. The ignored `persona_fleet.local.json` supplies the host, workboard executable, six GPU UUIDs, checkpoint path, model identity and RAM guard. The fleet launcher additionally needs its SSH, runtime, cache, lease-wrapper and disjoint CPU-set fields. Paths belong to the host running the command; none of these tools downloads a replacement checkpoint.

After calibration artifacts validate, review the six service commands and launch the probe fleet with the existing configuration:

```bash
python3 persona/persona_launch.py --config persona_fleet.local.json \
  --mode probe --minutes 420 --base-port 8930 --dry-run
python3 persona/persona_launch.py --config persona_fleet.local.json \
  --mode probe --minutes 420 --base-port 8930
```

The launcher creates `character-persona-probe-0` through `character-persona-probe-5`, each with its own parent lease, GPU UUID and disjoint CPU set. Choose a duration that ends at the intended analysis cutoff; probe mode does not accept capture's `--resume` flag. Launching diagnostic services and generating a production package are separate operations.

For a custom supervisor, the command below is the **worker command inside an already registered parent lease**, with `CUDA_VISIBLE_DEVICES` bound to its assigned UUID and a disjoint CPU affinity; direct unleased execution is rejected. Use shards 0–5 and ports 8930–8935:

```bash
python persona/persona_probe.py run --config persona_fleet.local.json \
  --flare /path/to/flare-local-inference --checkpoint /path/to/pinned-checkpoint \
  --heliopause /path/to/flare_heliopause.so --output results \
  --shard 0 --port 8930 --compile-workers 14 \
  --context 4096 --physical-pages 32 \
  --kernel-cache /path/to/kernel-cache --cutlass-include /path/to/cutlass/include
```

Verify each worker's ready manifest and capability response before dispatch. The logical context is not a guarantee that the physical KV-page allocation can hold an arbitrary prompt plus response; the worker checks actual capacity. Model weights load once per resident worker, and requests reuse the retained graphs.

`persona/persona_planner.py` is the **Atlas John-specific** launcher, with fixed cached-model, runtime and GPU paths. Inspect `python3 persona/persona_planner.py --print-command` before using it. It requires the named `persona-planner-john.service`, acquires its own 40 GiB workboard lease, verifies the model digest, exact 131,072-token context and model alias, and monitors observed process VRAM and the shared John thermal/RAM policy. Its JSONL manifest distinguishes sampled process-memory peak from an unavailable allocator-internal peak. It never silently lowers context or stops another owner's service. Other fleets must configure their own local planner service and adapt the owner/lease configuration rather than copy Atlas host paths.

Keep the planner loopback-only. If it runs on another host, use an owned SSH forward so the manager can reach it at `http://127.0.0.1:18511`. Run the CPU manager on the V100 host, beside the checkpoint, calibration files and probe artifacts. With the six diagnostic services and the planner already leased and ready:

```bash
python persona/persona_manager.py configure \
  --fleet persona_fleet.local.json --config results/persona_manager_config.json \
  --planner-lease GPU_WORKBOARD_PLANNER_JOB_ID \
  --planner-url http://127.0.0.1:18511 --base-port 8930

python persona/persona_manager.py run --config results/persona_manager_config.json
# Or add --until 'YYYY-MM-DDT09:00:00-07:00' with the desired date and UTC offset.
```

`configure` discovers the six existing Character Creation leases and verifies endpoint model/prepared/runtime identities; it does not launch services. The diagnostic runtime identity includes source hashes and preparation settings and is checked in descriptors, responses, NPZ files and request-provenance artifacts. The generated file uses local paths and should remain private. The default and generated `plan_low_water` are 1,000 queued jobs, keeping more work available while the local planner prepares another batch. Supervise `run` as a named CPU service. One request is active per V100. Run the same command after interruption to resume `results/persona_manager.sqlite`; a singleton lock prevents two managers from claiming the same queue. Normal binding rejects a changed prepared or diagnostic runtime identity.

There is one explicit operator-only exception for the reviewed lossless NPZ compression change. Drain and stop the manager before running `transition-runtime`; it requires the singleton lock and no running jobs or attempts. The private proof binds six old/new calibrated ready manifests, both probe sources and at least one old/new NPZ pair by SHA256. Verification permits only the reviewed level-one writer, its import and its single call replacement; native/helper hashes, model, prepared identity, calibration reference and all preparation settings must match. Every decoded array must retain its dtype, shape and exact bits, except that the runtime identity scalar may identify the verified new source. The command appends an immutable transition audit and changes only the active queue binding. Existing jobs, results and their original runtime identities remain intact; observations across the transition do not receive a mixed-runtime baseline delta. The planner cannot request this operation.

```bash
python persona/persona_manager.py transition-runtime \
  --database results/persona_manager.sqlite \
  --proof results/persona_runtime_transition.json --proof-sha256 REVIEWED_PROOF_SHA256
```

The proof schema is `character-creation.runtime-compression-transition.v1`, with `old_manifests` and `new_manifests` arrays containing six full ready manifests each, `old_source` and `new_source` objects containing `text` and `sha256`, and `artifact_pairs` containing `old`/`new` objects with `path` and `sha256`. A lossless re-encoding may preserve the old runtime scalar; a new worker's matching-request artifact uses its new runtime scalar. Keep this proof private. After successful verification, the operator must bind the deployment configuration to the verified new endpoint identities before resuming the existing queue.

```bash
python persona/persona_manager.py status
python persona/persona_manager.py export --output results/persona_manager_attempts.jsonl
python persona/persona_manager.py enqueue --config results/persona_manager_config.json \
  --proposal /path/to/fixed-proposal.json --validation
```

Exports retain attempted requests, responses, artifact references and errors and refuse to overwrite an existing output. Keep incomplete answers, raw neural measurements, self-report scores and independent behavioral evaluation separate. The manager marks results ineligible for automatic promotion. Readiness and continuous GPU use are operational checks, not scientific validation.

## Separate native Flare runtime

The authoritative runtime is the separate [Atlas3DSS/flare-local-inference repository](https://github.com/Atlas3DSS/flare-local-inference). Start with its [V100 setup](https://github.com/Atlas3DSS/flare-local-inference/blob/atlas/sm70-public/README.md) and [steering guide](https://github.com/Atlas3DSS/flare-local-inference/blob/atlas/sm70-public/STEERING.md). Runtime source and upstream license notices live there; Character Creation links to that implementation.

The steering extension builds on Flare revision `c3dbf81851c09b8666462c75a51a85895302fb21`. Use the linked branch, rebuild the CLI and Heliopause, then prepare a new package with `tools/prepare_replay.py --steering-max-sites 64`. Existing packages remain loadable but have no steering catalog. Diagnostic activation capture is offline and should not be exported as the serving package.

The request extension is:

```python
{
  "steering": {
    "phase": "all",
    "sites": [{"site": "layers.12.mlp_input", "vector": [0.0] * 17408}]
  }
}
```

The example is a zero-vector no-op; the adapter fills selected neuron coordinates with calibrated scalar doses. Supported sites are `layers.L.mlp_input` (17,408 values), `layers.L.mlp_out` and `layers.L.residual` (5,120 values each); phases are `all`, `prefill`, `decode`. Phase is determined by absolute token position against the original prompt length, independent of graph chunking or prefix-cache hits. The first generated token is sampled from the final prompt position, so it belongs to the prefill intervention. `decode` does not mean “final answer only.”

Programs are immutable per request, keyed into cache conditioning, and copied into persistent physical request-slot buffers. Disabled coordinates preserve original bits. Target-model MTP verification is steered; the draft is unchanged. Qualification must cover zero/off equality, signed changes, slot reuse, interleaved requests, phase boundaries and cache separation before deployment claims.

`persona/persona_qualify.py` checks repeated-off and all-site-zero equality on the retained whole model, exact signed residual additions, output-column effects, and exact pre-down one-hot arithmetic when observation buffers are available. It restores startup and diagnostic state after every arm. These offline checks do not alone qualify native cache reuse, speculative decoding, or behavioral transfer; those require native-server comparisons.

The production serving package is prepared separately, with diagnostic capture disabled. `persona/persona_prepare.py plan` produces a reviewable preparation manifest from the existing validated baseline report; `run` keeps its export/reference gates and checks neutral logits and measured output sequences against that baseline. Supply explicit `--config`, `--flare`, `--heliopause`, `--checkpoint`, `--baseline-report`, `--staging`, `--payload-cache`, `--kernel-cache`, `--signing-dir`, `--sysroot` and `--compile-workers` paths/options; use `--output results --steering-max-sites 64`. Preparation runs inside its own lease. Export one qualified package for sharing across the six servers; six redundant exports add no evidence. A successful diagnostic rollout is not a substitute for this production gate, and no production/MTP qualification is claimed while that gate remains pending.

## Attribution and source record

- **Lyra:** [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), reviewed revision `f79b2196e29aacbad89684938baac8d97b172fd1`; [agent methodology](https://github.com/lyramakesmusic/neuron-explorer/blob/f79b2196e29aacbad89684938baac8d97b172fd1/explorer/agent_methodology.md). Primary reference for neuron interventions and read/write distinctions.
- **Owner-supplied Lyra transcript:** October 23, 2026; exact Qwen3.5-4B signed recipes in `data/lyra_recipes.json`. No post URL is invented.
- **RhizoNymph/vLLM:** [steering runtime design](https://github.com/RhizoNymph/vllm/blob/feat/integration/docs/design/steering_runtime.md), inspiration for request ownership, graph-safe staging and cache identity.
- **Strata:** [AbliteratedOutTheBox](https://github.com/heterodoxin/AbliteratedOutTheBox/tree/strata), inspiration for native residual interventions; its experimental results are not treated as validation of this implementation.
- **Character Creation:** original broad trait, layer and sign-reversal studies, retained with negative results and fresh-model controls.

## October 25 atlas and token-trace qualification

The static atlas covers all 1,114,112 MLP neurons in 64 layers, with signed token projections and retained adjacent-layer down-to-gate/up affinities. An independent FP64 CPU spot check at four coordinates agreed within 2.16e-5. Static affinities are not causal paths; this adjacent-layer metric differs from Lyra’s all-layer gate/up product.

All six V100 diagnostic workers passed exact comparison against one original calibration prompt each. Token traces were checked across two prefill chunks and ordinary decode: capture-on/off and zero interventions preserved complete captured planes and generated IDs exactly. A decode-only intervention preserved prefill and first-next-token logits while changing decode observations. A real three-prompt paired campaign completed on six distinct cards; eight live partial snapshots confirmed updates before completion. A separate six-arm dose campaign verified exact zero control across cards. These short diagnostic continuations do not qualify complete persona answers, native cache reuse, or MTP execution.

The integrated source layout, public-dataset export checks, wildcard planner and public preset catalog passed 282 CPU tests. The subsequent gift scheduler passed its targeted checks, including fixed-model identity and failure propagation. Browser checks covered layer-stack connections, canvas containment, 3D orbiting, selected-link isolation, and token playback.

Selecting a map neuron now draws measured links directly between the desk-like selected sheet and the raised layer stack. Choose observed coactivity or weight paths; unavailable observations remain explicit. Coactivity starts with the selected neuron’s links, with an optional neighbor-to-neighbor toggle. Visual spacing is an explicitly distorted display aid; the exact-value table retains original measurements.

## Public synthetic corpus

[Atlas3D/character-creation-big-five](https://huggingface.co/datasets/Atlas3D/character-creation-big-five) retains the generated Big Five discovery scenarios as its default subset. Separate research manifests now index verified captures, evaluation bundles and calibration assets. `persona/persona_hub.py` checks the publication hash and synthetic-only provenance before exporting a field allowlist into `dataset/` and uploading one atomic Hub commit. Its bounded job watcher publishes at most every thirty minutes and on completion. Book text, local source paths, credentials, and other corpus collections are excluded. Trait/facet labels describe prompt design, not psychometric truth.

## Bounded gift campaign for Lyra

The `lyra_gift` job uses pinned `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` on all six workboard-leased V100s, with John ranking a bounded list of candidate experiments. The source is a frozen, synthetic-only Big Five snapshot. Domain/facet families are assigned to discovery, replication, and held-out splits. Workers retain all 294,912 MLP values at the final observed prompt position, compute same-layer and adjacent-layer top correlations, and test selected adjacent links through genuine neuron ablations with separate random same-layer controls. Those controls are not amplitude matched.

This is an independent float16 HF path with eager attention and reference recurrent kernels, not a claim of BF16/vLLM equivalence to Lyra. Every worker checks finite values and exact neutral-gain equivalence on its actual GPU before collection. Prompts are capped at 512 tokens and truncation is recorded. Zero-variation or unavailable correlation entries remain missing. Raw arrays, prompt alignment, hashes, selection audits, and failed cases remain local for review and packaging; this job does not message anyone or publish private source material.

Startup consumes the scheduled window. The absolute deadline stops owned services; partial results are not labeled a complete map. A completed capture stage alone does not qualify all downstream phases. Historical scheduling note: the proposed October 26 02:00–08:00 gift campaign was cancelled. It is not part of the active 27B overnight mapping run; consult the live job ledger instead of inferring activity from this implementation description.

## Literature prompts for mapping and recipe checks

The Literature tab reads `results/persona_literature.sqlite`. EPUB ingestion keeps
normalized source text, document offsets, book hashes, duplicate links, and
whole-book discovery/heldout splits. Passage windows are structural candidates;
validated model labels are provisional literary interpretations, not neural
findings. Legacy formats are reported as unsupported rather than silently read.

`persona/persona_literature_harness.py` is the authored wildcard compiler. John
or another caller can select a source unit, mode, variant and seed; it does not
need to invent or rewrite the prompt. Modes are decision, response, continuation.
Eight variants include the unchanged source and explicitly hypothetical changes
to time pressure, audience, ambiguous intent, temptation, power, repair and
accountability. A matched pair uses the same unit, mode and seed. Continuation
withholds the final third of longer windows; the original ending is not a unique
correct answer. Templates, source hashes, offsets and version are bound to each
prompt ID. Compiled prompts are retained in `literature_prompts`.

`POST /api/literature/prompt` accepts exactly
`{"unit_id":"…","mode":"decision","variant":"original","seed":7035}`.
The response includes the exact prompt, provenance, a source-evidence graph and
comparison guidance. The viewer explicitly sends the same text to activation
search or preset comparison. The CLI has equivalent `--unit-id`, `--mode`,
`--variant`, `--seed`, and `--database` arguments. It prints the saved JSON record.

Compare unsteered, positive and negative recipes on identical text and generation
settings; repeat on held-out books and multiple seeds before claiming a stable
behavioral effect. Signed additive pushes do not implement ablation. Source
entity/theme graphs and observed neural activation associations do not establish
causal neural topology.

Named concept controls beside Big Five are saved through `/api/concept-controls`.
Coordinates must come from checkpoint lens search, measured contrast search, or
a completed activation search for the current model. The server derives signs
from the evidence, saves immutable provenance, and combines these neuron pushes
with Big Five under the same aggregate dose bound. Names such as rage are
hypothesis labels; paired behavioral tests are still required.

Overnight mapping uses the six isolated Quincy workers. `GET /api/overnight`
reports the remote coordinator; the banner's pause control waits for in-flight
captures to drain before admitting interactive inference. Resume refuses active
interactive jobs. Browsing saved results does not require pausing. The reservation
is `results/persona_overnight_session.json`; remote status and raw capture artifacts
remain in the existing `activation-search` directory on v100.

`GET /api/overnight/features` reads the derived emotion contrast artifact. Concept
discovery shows ranked coordinates, four-frame agreement, lexical controls and
counterevidence. Shared-coordinate edges represent overlapping candidate lists,
not causal connections. All-neuron capture retains per-neuron peaks, argpeaks and
absolute sums across prefill positions, plus selected full profiles; it does not
retain the entire neuron-by-token tensor. Heldout rows are excluded from fitting.

Saved manual neuron batches can be opened with `?batch=<batch-id>#lab`. The
48-response manual test index is `results/persona_manual_neuron_tests.json`.
L40 N1661 prompt-only positive steering was a stronger sarcasm lead than the
initial low-dose recipes; the mystical L18 N5811 example failed transfer to other
situations. These observations are model-specific and do not validate donor
coordinates from Qwen3.5-4B or OLMo.

Coherence calibration is available in Persona studio and at `/api/coherence`.
`POST /api/coherence/runs` takes a saved `recipe_id`, `seed`, and `max_tokens`
(2048–4096). Six discovery and six heldout source-grounded fixtures use two
seeds. Baseline is matched; nonzero scales 1/.9/.75/.5/.25 and two actually-tested
refinement points locate a passing discovery setting without assuming monotonic
behavior. Selection freezes before heldout confirmation. Failed confirmation
never feeds a replacement search on the same heldout examples. Cancellation
stops further trials and retains in-flight batch evidence. An explicitly supplied
`reuse_run` may rescore completed cells only with matching model, suite, seeds,
budget, recipe and program hashes; those cells are not counted as new evidence.

The evaluator distinguishes required fact-ledger errors, surface collapse and
ambiguous representation failures. A correct sentence in a boolean slot is not
called a factual contradiction. Unknown-only evidence requires review rather
than triggering dose reduction. CJK/unspaced loops are checked. A ledger pass
never certifies free-prose factuality or desired persona quality. Scoped results
are stored in `coherence_defaults`; they are not silently applied to all chats.
Recovery currently operates between complete response attempts, not mid-token.

`POST /api/coherence/interaction-plan` takes a one-neuron `recipe_id` and an
observed coactivity neighbor's `layer`, `neuron`, and signed `dose` (magnitude
at most .5). It prepares baseline/A/B/A+B/A−B/−A controls. Add `case_id` (discovery
only), `seed`, and `max_tokens` at `/api/coherence/interaction-test` to run these
six cells. `GET /api/coherence/interaction?id=...` retains exact outputs and
exploratory issue-count interaction contrasts. This is not an automatic causal
compensator; coactivity signs are not steering signs and no layer-wide reversal
is assumed.

Neuron labels are read from `results/persona_neuron_labels.json`. At most three
qualified labels appear per neuron, with inverse associations explicit. The
reader recomputes heldout precision and its Wilson95 lower bound (>70%), requires
at least20 distinct positive-prediction families, and verifies immutable evidence
and model bindings. The badge displays the lower bound, not a claim of certainty.
The six-concept qualification runner additionally requires heldout balanced
accuracy>=.8, recall>=.75 and >.05 advantage over discovery-fitted length/negation
baselines. Selection and thresholds freeze before heldout capture. This is
probe-conditioned association, not probability that a neuron means a concept.

The432-probe label study completed with one qualified label: green, L12 N2001,
21TP/0FP/24TN/3FN; Wilson95 precision lower bound .84536. The other five labels
remain candidates. Exact bindings and confusion matrices are retained in
`results/persona_label_evidence_*.json`. `persona_label_handoff.py` verifies and
merges evidence, then resumes only its owned overnight pause. The large map
continues independently under its finite supervisor.

The diagnostic Flare path remains B1 per GPU. Four prepared request slots do not
mean four simultaneously safe capture requests: the engine and capture buffers
are shared and explicitly locked. Six-card concurrency is supported. Batched
within-card capture needs separate state/steering/RNG buffers and qualification.
Map PCA freshness now includes calibration hashes; captions identify index-grid
fallbacks. Coactivity is undirected Pearson correlation across prompts; apparent
one-way navigation can reflect local neighbor-list truncation, not causal flow.
Circuit tracing and sparse execution are a discussion-stage research direction;
no circuit compiler, layer skipping, or Engram execution system was launched.

## Public research publication and reusable guides

Start with [README_human.md](README_human.md) for what, how and why, or
[README_agent.md](README_agent.md) for exact schemas, evidence boundaries and commands.
The shared Hub repository keeps its original prompt subset intact and adds immutable
research batches. Metadata and raw-array publication have independent coverage manifests.

`persona/persona_hub_research.py` opens capture ledgers read-only, accepts only the
exact approved public prompt IDs/text/hashes and authored probe sources, and revalidates
complete artifacts before publication. The metadata watcher polls every five minutes and
publishes at 500 records, after fifteen minutes or at run completion. A separate watcher
on the capture host sends up to 32 original NPZs per raw shard directly to HF; it continues
draining after inference stops. Both verify committed hashes before advancing durable
checkpoints and reconcile uncertain uploads without repeating inference. The reference
services are `character-persona-hub-research.service` on Devbox and
`character-persona-hub-raw.service` on the capture host. These are CPU/network publishers,
not additional GPU inference jobs.

`persona/persona_read_research.py` is a small CPU-only reader. It pins one Hub commit,
checks batch and raw-entry hashes, and loads NPZ with pickle disabled. Public calibration
assets cover all 64 layers; their original hashes and model binding are retained in
`research_assets.json`. Six measured experimental presets now accompany the original
sixteen authored profiles in the public catalog. Their limitations remain explicit.

The public JSON records retain selected token profiles. The separate raw artifacts retain
all-neuron prefill reductions and final-prefill arrays. Neither is a full stored
neuron-by-token tensor across generated answers. Private book text, book token IDs and
profiles, arbitrary interactive prompts and proprietary research are excluded.

For a fork, configure `nlp_python` and `overnight_remote` (`ssh_host`, `hostname`,
`remote_root`) in the local Explorer config. The default NLP interpreter is the current
Python executable. The reference activation launcher accepts `--config`, `--jobs-config`
and fleet overrides `remote_project_root`, `heliopause_path`, `model_path`,
`calibration_dir`. These configuration changes preserve model/hash validation; they do
not make an unqualified runtime compatible automatically.
