# Persona Explorer

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

Open `http://127.0.0.1:8792`. The map shows coverage, neuron controls, trait sliders, layer ranges with an optional explicit sign reversal, signed recipes and saved negative/baseline/positive comparisons. The Studies tab shows the measured prompt-only reference. Lyra’s original 4B recipes are available in a separately labeled references tab; they are not applied to 27B by coordinate transfer.

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

The service reads the native `/v1/models` capability catalog and checks the prepared identity, protocol, site names, vector widths and request site limit before sending steering. Older packages cannot silently accept steering as a no-op. Complete requests, responses, program hashes and vector provenance are retained in `results/persona_explorer.sqlite`.

## Calibration and experiments

`persona_fleet.py` runs deterministic six-way generation shards. It checks host, GPU UUID, workboard parent lease and pinned executable/package identity, records complete sampling requests, monitors resources, and terminates only its owned server. Startup attempts and failures remain in the same artifacts. The HDD used in the reference run needed about 500 seconds for six direct-I/O weight loads; those runs completed after extending startup readiness to 900 seconds.

`persona_capture.py` generates 132 discovery examples: twelve user prompts under eleven matched conditions. Its six disjoint shards replay actual retained Flare init/prefill graphs, retain normal admission/completion validation, and save prompt-token residual norms plus final-token residuals, normalized pre-MLP inputs and the actual rounded SwiGLU activations. It checks token positions and restores initial state. No heldout prompts fit the directions. Each launch must be wrapped in its own workboard lease; the script rejects direct unleased execution. Failed artifact-free attempts can resume in place with their prior reports retained.

```bash
python3 persona_capture.py plan --output results/persona_capture_plan.json
# On each leased V100, with explicit checkpoint, Flare source and rebuilt extension:
python3 persona_capture.py run --config persona_fleet.local.json \
  --plan results/persona_capture_plan.json --shard 0 --compile-workers 14 \
  --flare /path/to/flare --checkpoint /path/to/pinned-checkpoint \
  --heliopause /path/to/flare_heliopause.so --output results
```

Repeat the capture command with shards 0–5 and disjoint CPU affinities. Preparation uses the SM70 MTP weight path but replays ordinary non-speculative prefill for capture. The extension must be rebuilt with the new protocol; do not accidentally import a stale worktree shim.

After all six captures validate:

```bash
# Run shards 0–5 with the same checkpoint and capture plan.
python3 persona_map.py layers --plan results/persona_capture_plan.json \
  --results results --checkpoint /path/to/pinned-checkpoint --shard 0
python3 persona_map.py assemble --results results
```

These produce model-bound `layer_00.npz` through `layer_63.npz` and a candidate `persona_map.json`. The builder checks shard completeness, hashes, model identity and paired conditions before fitting. Existing evidence is preserved rather than overwritten.

`persona_experiments.py` defines calibration, coarse, refinement and heldout validation plans, including signed doses, zero, prompt-only, shuffled and matched-norm random controls. Its broader operation vocabulary is planning metadata: `mlp_scale` and per-thinking/answer-phase control are not implemented by this native additive protocol. Never submit an unsupported plan as if it were executed. Keep measured self-report scores, behavioral review, capability retention and incomplete responses separate.

## Overnight discovery with measured activations

The overnight workflow uses one resident diagnostic worker on each of the six V100s, a local 27B planning model, and a CPU queue manager. The planning model proposes experiments; the manager validates and dispatches them. The target checkpoint remains the pinned NVFP4 27B model described above. The planning model is a separate cached Q4_K_M GGUF, served by llama.cpp with a verified 128k context. Its judgments are hypotheses, not an independent evaluator or evidence that a trait changed.

`persona_probe.py` exposes loopback `/v1/models` and `/v1/chat/completions` endpoints while retaining the actual Flare init, prefill, ordinary decode and sampling graphs. One request runs per card at a time. Each request starts from restored state and saves observations at the final valid prompt token before generation continues. This diagnostic mode uses **MTP-capable preparation and B1 ordinary-only execution**, with no native prefix-cache reuse or speculative execution. Preparation matches the passed calibration: four request slots, `mtp=True`, widths `(1,2,3,4,5)` and inherited row-invariant target arithmetic. SM70 prefill requires that layout's seven convolution-history columns. Only ordinary one-request graphs execute; every replay checks that the other three slots remain inactive.

The default context is 4,096 tokens backed by 32 physical pages, matching calibration. The recorded real-tokenizer preflight found all 13,895 current corpus rows fit with their full rollout budgets; the largest prompt-plus-budget was 3,155 tokens. Every new request is independently rejected if prompt plus budget exceeds physical or logical capacity; no truncation is used. Startup must reproduce one original calibration prompt's encoded tokens, all three final-prefill observation planes and every non-sink residual norm bit-for-bit before the endpoint becomes ready. This remains a separate qualification path from the exported production server.

An explicit `observation_only=true` request requires `max_tokens=1`: it runs the actual sampler once after complete prefill and skips decode. The response has `operation="prefill_observation"`, `prefill_observation_complete=true` and `complete_final_answer=false`; response text is empty so the sampled first token cannot be graded as an answer. The token and full activation artifact remain in private provenance. Normal rollout requests retain their full generation budget and behavior.

The observations are continuous, BF16-rounded SwiGLU products, not binary “neuron firing” labels. Per-request NPZ files contain `mlp_input_last[layers, intermediate]`, `residual_last[layers, hidden]`, `pre_mlp_last[layers, hidden]`, prompt/generated token IDs and prompt residual-norm summaries excluding position zero. The MLP observer records the value after an optional neuron push and before `down_proj`. It does not record every neuron's complete token-by-token trajectory. Comparing matched prompt mutations and interventions tests which units respond and which changes affect subsequent behavior; it does not by itself establish a synaptic connectome.

The response links the activation artifact and request/response provenance by SHA-256, model identity, prepared identity and request hash. `persona_manager.py` verifies those artifacts and gives the planner bounded numerical summaries: the selected neuron's activation, layer activity/residual norms, and same-prompt/same-seed differences against a completed baseline. Full arrays and rollouts stay available for later analysis.

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
python3 persona_corpus.py prepare
```

The initial SCOTUS/Big Five manifest contains 374 records: 60 SCOTUS and 314 Big Five prompts, with discovery and heldout splits kept separate. The October 25 local corpus contains 13,895 records after ingestion of eleven owner-provided EPUBs, including 231 explicitly labelled counterfactual speaker-attribution variants. Those variants exchange named Skippy/Joe attributions where the original speech tag is explicit; dialogue words and narrative mentions remain unchanged. Unresolved speakers remain unlabeled. The book texts and local manifest are not public artifacts. SCOTUS requests reserve 3,072 generated tokens. IPIP-50 items retain their original item IDs and reverse keys and stay in the heldout inventory-measurement split. The adaptive planner receives discovery observations only. A fixed heldout proposal can be queued with `persona_manager.py enqueue --validation`; heldout responses never enter adaptive feedback.

Raw books and extracts remain private and ignored by Git. The corpus starts without Skippy material if the books are absent; it does not relabel other prompts as book evidence. When explicitly selected local EPUB/TXT files arrive, ingest them without copying the source books:

```bash
python3 persona_corpus.py ingest --book-dir /path/to/local/books
# Optional long-running ingestion watcher, supervised independently:
python3 persona_corpus.py watch --book-dir /path/to/local/books --interval 30
```

Content hashes prevent duplicate ingestion. Existing prompt text, provenance and split assignments remain immutable. The manifest first marks ingestion in progress and publishes `status: complete` with the new JSONL hash afterward. The manager reloads only a complete, matching generation. Public reports should contain manifests and aggregate results, not private book passages or unreviewed rollout excerpts.

`persona_coverage.py` adds a separate operator pass for exhaustive registered-row coverage, independent of the adaptive planner's group sampling. Explicit `--split all` queues one unsteered prefill observation for every registered row, including all book chunks and registered variants, SCOTUS and Big Five. Original prompts, source hashes, offsets, pair IDs and discovery/heldout labels are preserved; related variants share a deterministic seed. Layer 0 is only the existing executor's calibration-bound summary reference: no neuron or trait intervention is applied. Heldout results remain excluded from adaptive feedback by the manager's existing split filters.

```bash
python persona_coverage.py plan --config results/persona_manager_config.json --split all
# Operator action: append the entire pass atomically to the existing manager queue.
python persona_coverage.py enqueue --config results/persona_manager_config.json --split all
python persona_coverage.py status --config results/persona_manager_config.json --split all
```

Plan/status output contains counts and hashes, not raw text. Repeating enqueue with the same corpus generation, split, seed and calibration is idempotent; exceeding `--max-jobs` or the manager's total-job cap rejects the whole pass without truncation. Enqueue starts no service itself. Status counts only completed prefill observations and distinguishes queued, failed and remaining rows. Each artifact covers all layers at the final prompt token plus prompt residual norms; this is not every neuron's complete token-by-token trajectory, and queued coverage is not completed coverage.

### Character evidence before broader speaker swaps

`persona_characters.py` inventories character names in original book chunks through the already leased local John planner. It excludes generated mutations and defaults to discovery sources only. The model returns a name and a short, unique verbatim evidence phrase containing that name, with optional evidence-backed alias candidates. Software locates the phrase and name and computes exact Unicode offsets; the model never counts character offsets. Repeated evidence requires an explicit bounded occurrence index, otherwise it is rejected. Each balanced quote gets one software-generated unresolved speech record. Narrative speaker assignment is deferred. Character recall and entity classification remain unmeasured; exact source evidence alone does not establish either. Alias candidates are not automatically merged into identities.

The private SQLite index preserves source book/document hashes, split, normalized-document offsets, raw model responses, failed attempts and coverage counts. Its corpus and planner-runtime binding prevents silently resuming against changed inputs. The October 25 run is underway across all 6,645 original chunks from eleven books, including both discovery and heldout splits. An early checkpoint recorded roughly 96 complete and 75 failed chunks under strict evidence validation; these pipeline statuses are not accuracy estimates or completed corpus coverage. The index remains provisional: source-phrase acceptance does not validate character classification, recall or aliases. All splits and index outputs remain isolated from adaptive experiment planning. The sampling order covers attributed dialogue, other dialogue and narrative across books before continuing; chunk-attempt limits, a deadline and error budgets bound the run, and outputs are capped at 2,048 tokens. Rejected attempts remain audited, and neither exhaustive completion nor character accuracy is claimed.

```bash
# On John, using its existing planner lease and launcher readiness manifest:
python persona_characters.py run --lease GPU_WORKBOARD_PLANNER_JOB_ID \
  --limit 8 --batch-size 1 --minutes 5
python persona_characters.py status
python persona_characters.py export --output results/persona_characters.jsonl
```

The indexer checks `/slots` and waits for a configurable idle interval before each request. This is cooperative yielding, not atomic priority: stop indexing when experiment planning needs predictable access to the single planner slot. Repeating `run` resumes the same private index and retains failures. Explicit `--split all` can expand a discovery-only index in place when the corpus hash and planner binding are unchanged. It appends heldout chunks with their split labels, preserves prior rows and attempts, and audits the scope change; narrowing the index afterward is rejected. This is annotation only: neither raw responses nor index labels feed the adaptive experiment planner, and heldout material must not feed adaptive discovery. Source passages, raw responses and exports stay private.

`--error-budget` limits requests containing rejected chunks per invocation (default 3, maximum 100,000). `--consecutive-error-budget` independently stops repeated wholly rejected requests (default 5); any validated chunk resets that streak. Both counters, the separate number of rejected chunk attempts and the stop reason are audited. After checking the complete response schema and exact requested chunk IDs, batches validate each chunk independently: good chunks are saved, and only rejected chunks retry, at most twice in total. A chunk is accepted only when all its annotations validate. Raw responses and per-chunk errors remain in SQLite. A reviewed larger run can use `--batch-size 4 --error-budget 10000 --consecutive-error-budget 10` with an explicit `--minutes` deadline and `--limit`; rejected evidence is never accepted to keep the run moving.

### Start the workers and manager

Use Flare's pinned environment from the separate [V100 runtime setup](https://github.com/Atlas3DSS/flare-local-inference/blob/atlas/sm70-public/README.md), the rebuilt Heliopause extension and the exact local checkpoint. The ignored `persona_fleet.local.json` supplies the host, workboard executable, six GPU UUIDs, checkpoint path, model identity and RAM guard. The fleet launcher additionally needs its SSH, runtime, cache, lease-wrapper and disjoint CPU-set fields. Paths belong to the host running the command; none of these tools downloads a replacement checkpoint.

After calibration artifacts validate, review the six service commands and launch the probe fleet with the existing configuration:

```bash
python3 persona_launch.py --config persona_fleet.local.json \
  --mode probe --minutes 420 --base-port 8930 --dry-run
python3 persona_launch.py --config persona_fleet.local.json \
  --mode probe --minutes 420 --base-port 8930
```

The launcher creates `character-persona-probe-0` through `character-persona-probe-5`, each with its own parent lease, GPU UUID and disjoint CPU set. Choose a duration that ends at the intended analysis cutoff; probe mode does not accept capture's `--resume` flag. Launching diagnostic services and generating a production package are separate operations.

For a custom supervisor, the command below is the **worker command inside an already registered parent lease**, with `CUDA_VISIBLE_DEVICES` bound to its assigned UUID and a disjoint CPU affinity; direct unleased execution is rejected. Use shards 0–5 and ports 8930–8935:

```bash
python persona_probe.py run --config persona_fleet.local.json \
  --flare /path/to/flare-local-inference --checkpoint /path/to/pinned-checkpoint \
  --heliopause /path/to/flare_heliopause.so --output results \
  --shard 0 --port 8930 --compile-workers 14 \
  --context 4096 --physical-pages 32 \
  --kernel-cache /path/to/kernel-cache --cutlass-include /path/to/cutlass/include
```

Verify each worker's ready manifest and capability response before dispatch. The logical context is not a guarantee that the physical KV-page allocation can hold an arbitrary prompt plus response; the worker checks actual capacity. Model weights load once per resident worker, and requests reuse the retained graphs.

`persona_planner.py` is the **Atlas John-specific** launcher, with fixed cached-model, runtime and GPU paths. Inspect `python3 persona_planner.py --print-command` before using it. It requires the named `persona-planner-john.service`, acquires its own 40 GiB workboard lease, verifies the model digest, exact 131,072-token context and model alias, and monitors observed process VRAM and the shared John thermal/RAM policy. Its JSONL manifest distinguishes sampled process-memory peak from an unavailable allocator-internal peak. It never silently lowers context or stops another owner's service. Other fleets must configure their own local planner service and adapt the owner/lease configuration rather than copy Atlas host paths.

Keep the planner loopback-only. If it runs on another host, use an owned SSH forward so the manager can reach it at `http://127.0.0.1:18511`. Run the CPU manager on the V100 host, beside the checkpoint, calibration files and probe artifacts. With the six diagnostic services and the planner already leased and ready:

```bash
python persona_manager.py configure \
  --fleet persona_fleet.local.json --config results/persona_manager_config.json \
  --planner-lease GPU_WORKBOARD_PLANNER_JOB_ID \
  --planner-url http://127.0.0.1:18511 --base-port 8930

python persona_manager.py run --config results/persona_manager_config.json
# Or add --until 'YYYY-MM-DDT09:00:00-07:00' with the desired date and UTC offset.
```

`configure` discovers the six existing Character Creation leases and verifies endpoint model/prepared/runtime identities; it does not launch services. The diagnostic runtime identity includes source hashes and preparation settings and is checked in descriptors, responses, NPZ files and request-provenance artifacts. The generated file uses local paths and should remain private. The default and generated `plan_low_water` are 1,000 queued jobs, keeping more work available while the local planner prepares another batch. Supervise `run` as a named CPU service. One request is active per V100. Run the same command after interruption to resume `results/persona_manager.sqlite`; a singleton lock prevents two managers from claiming the same queue. Normal binding rejects a changed prepared or diagnostic runtime identity.

There is one explicit operator-only exception for the reviewed lossless NPZ compression change. Drain and stop the manager before running `transition-runtime`; it requires the singleton lock and no running jobs or attempts. The private proof binds six old/new calibrated ready manifests, both probe sources and at least one old/new NPZ pair by SHA256. Verification permits only the reviewed level-one writer, its import and its single call replacement; native/helper hashes, model, prepared identity, calibration reference and all preparation settings must match. Every decoded array must retain its dtype, shape and exact bits, except that the runtime identity scalar may identify the verified new source. The command appends an immutable transition audit and changes only the active queue binding. Existing jobs, results and their original runtime identities remain intact; observations across the transition do not receive a mixed-runtime baseline delta. The planner cannot request this operation.

```bash
python persona_manager.py transition-runtime \
  --database results/persona_manager.sqlite \
  --proof results/persona_runtime_transition.json --proof-sha256 REVIEWED_PROOF_SHA256
```

The proof schema is `character-creation.runtime-compression-transition.v1`, with `old_manifests` and `new_manifests` arrays containing six full ready manifests each, `old_source` and `new_source` objects containing `text` and `sha256`, and `artifact_pairs` containing `old`/`new` objects with `path` and `sha256`. A lossless re-encoding may preserve the old runtime scalar; a new worker's matching-request artifact uses its new runtime scalar. Keep this proof private. After successful verification, the operator must bind the deployment configuration to the verified new endpoint identities before resuming the existing queue.

```bash
python persona_manager.py status
python persona_manager.py export --output results/persona_manager_attempts.jsonl
python persona_manager.py enqueue --config results/persona_manager_config.json \
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

`persona_qualify.py` checks repeated-off and all-site-zero equality on the retained whole model, exact signed residual additions, output-column effects, and exact pre-down one-hot arithmetic when observation buffers are available. It restores startup and diagnostic state after every arm. These offline checks do not alone qualify native cache reuse, speculative decoding, or behavioral transfer; those require native-server comparisons.

The production serving package is prepared separately, with diagnostic capture disabled. `persona_prepare.py plan` produces a reviewable preparation manifest from the existing validated baseline report; `run` keeps its export/reference gates and checks neutral logits and measured output sequences against that baseline. Supply explicit `--config`, `--flare`, `--heliopause`, `--checkpoint`, `--baseline-report`, `--staging`, `--payload-cache`, `--kernel-cache`, `--signing-dir`, `--sysroot` and `--compile-workers` paths/options; use `--output results --steering-max-sites 64`. Preparation runs inside its own lease. Export one qualified package for sharing across the six servers; six redundant exports add no evidence. A successful diagnostic rollout is not a substitute for this production gate, and no production/MTP qualification is claimed while that gate remains pending.

## Attribution and source record

- **Lyra:** [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), reviewed revision `f79b2196e29aacbad89684938baac8d97b172fd1`; [agent methodology](https://github.com/lyramakesmusic/neuron-explorer/blob/f79b2196e29aacbad89684938baac8d97b172fd1/explorer/agent_methodology.md). Primary reference for neuron interventions and read/write distinctions.
- **Owner-supplied Lyra transcript:** October 23, 2026; exact Qwen3.5-4B signed recipes in `data/lyra_recipes.json`. No post URL is invented.
- **RhizoNymph/vLLM:** [steering runtime design](https://github.com/RhizoNymph/vllm/blob/feat/integration/docs/design/steering_runtime.md), inspiration for request ownership, graph-safe staging and cache identity.
- **Strata:** [AbliteratedOutTheBox](https://github.com/heterodoxin/AbliteratedOutTheBox/tree/strata), inspiration for native residual interventions; its experimental results are not treated as validation of this implementation.
- **Character Creation:** original broad trait, layer and sign-reversal studies, retained with negative results and fresh-model controls.
