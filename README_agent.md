# Character Creation: agent interface and evidence contract

This document is the machine-oriented entry point. [README_human.md](README_human.md)
explains the motivation and workflow. Source: [GitHub](https://github.com/Atlas3DSS/Character-Creation).
Public data: `Atlas3D/character-creation-big-five` on Hugging Face, repository type `dataset`.

Product intent: character creation for a persistent companion. Preserve the progression
from understandable starter presets and trait adjustments to optional neuron/feature,
dose and phase controls. Evaluate observable differences without treating a friendly
personality as the only successful outcome. Keep factual fidelity and coherence separate
from agreeableness, sarcasm, hostility, assertiveness and warmth. Preset/cast comparisons
hold the user prompt, seed and generation budget fixed; they do not inject a persona
system prompt or KV control. Saved activation recipes are checkpoint-bound, not universal
personality descriptions or a claim that long-term companion memory is implemented.

## Authored persona prompts and bounded preset search

`data/persona_perturbation_prompts.jsonl` contains 3,072 independent original
synthetic prompts authored with GPT-6.1 Terra: 2,443 discovery / 629 heldout.
The manifest pins exact bytes, groups, splits and duplicate checks. Inputs have
empty systems; anticipated behavioral sensitivity is metadata, not validated
psychometrics. Download both files from the same Hub revision. The original row
IDs remain stable. `persona_perturbation.registered_rows` adds `source_row_id` and
canonical viewer IDs without rewriting the source or frozen Manager corpus.

`persona_preset_overnight.py` freezes up to two variants for each of eight targets:
960 discovery answers (five arms, six scenes, two seeds), then at most 384
confirmation answers (four arms, six new scenes, two seeds). Zero, reversed and
matched-random controls accompany baseline/candidates. All arms share neutral
text; activation programs supply the variation. The v2 blind judge returns IDs
of exact answer spans; unknown/cross-answer IDs are rejected. It remains a local
Qwen-family automatic judge, not independent human ground truth.

`persona_preset_followon.py` freezes new discovery/confirmation source groups
from the completed authored corpus and constructs challengers using discovery
evidence only. It can distribute interventions over L16/L24/L32/L40/L44/L48 or
reduce the observed Dry Wit neuron dose. It does not learn from confirmation
failures. Both controllers require a hash-bound qualified runtime gate, finite
deadline and valid leases; uncertain requests are never silently replayed.

`/api/sculptor` exposes a separate `evaluated_presets` envelope. Verification binds
plan, model, calibration, exact program and evaluation files. Automatic acceptance
requires 12 clean confirmation groups, at least eight behavioral wins and the
cross-target distinctness gates. The reader replaces an incumbent only with a
qualified challenger for that target, capped at eight. Loading preserves exact
pushes/doses/phase; it does not reconstruct the recipe from five UI dial values.
Historical presets and frozen selection inputs stay unchanged.

GET `/api/sculptor/preset-evidence?target=TARGET&group=OPTIONAL_GROUP_DIGEST`
returns one saved matched scene/seed group plus a bounded group index. It verifies
the selected overlay recipe, frozen plan, each control program and response/model
bindings. The UI's **Read comparisons** shows saved baseline/candidate/control
answers without inference or a live GPU session. Pending, incomplete and failed
arms remain explicit. Reading evidence does not promote a recipe.

Exact auxiliary files are also on the Hub: `persona_preset_overnight_plan.json`,
`persona_preset_columns.json`, `persona_preset_optimized_columns.json`,
`persona_batch_optimized_qualification.json`, and
`persona_batch_optimized_compatibility.json`. Download at one pinned revision and
preserve their bytes. The column is a checkpoint-derived numeric asset, subject
to upstream model terms. Its read-only vector parity record is distinct from
the serving qualification receipt. Do not infer portable runtime qualification
from downloading these files.

`persona_perturbation_mapping.py` captures one unsteered answer per authored
discovery row (2,443 total, 2,048-token budget), retaining final-prefill activations
and all-layer response residual summaries. It excludes all 629 heldout rows.
Its separate PriorityQueue checks the five fixed study databases and interactive
viewer tables before claiming and again before HTTP submission. Foreground work
or an unreadable priority database blocks new claims; submitted batches drain.
Only demonstrably unsubmitted claims may be released. The existing Manager and
GPU service remain unchanged. The reference run has finite 07:58 dispatch and
08:00 PDT publication deadlines. Its public file is
`persona_perturbation_mapping_public.json`: original synthetic inputs, answers,
status and capture hashes, with raw-array coverage explicitly excluded. Uploads
occur at 48-terminal-job milestones and final/deadline, with durable intent and
commit-pinned verification. The source pins this corpus revision and runtime
gate; a fork must configure and qualify its own admitted workers.

`persona_preset_publication.py --root PATH --until EPOCH --publish` publishes
`persona_preset_overnight_public.json` after 48 terminal answers, eight review
groups, evaluation changes and at deadline. It reads only the allowlisted
authored-study databases. Original synthetic prompts, generated answers, exact
recipes, status and automatic judgments are public; deployment settings are
excluded. Each upload records intent and verifies commit-pinned bytes. An
uncertain attempt disables retries. Artifact hashes reference retained capture
arrays; this JSON does not claim those arrays have been uploaded.

## Capture performance evidence

`persona_batch_optimization_public.json` records the readback benchmark at Git
commit `707397a` and slot-refill canary at `dd29910`. Preserve those source versions
when reconstructing the completed plans; later stream helper code is not covered
by an earlier proof. The readback benchmark ran B1/B2/B4, inactive-slot holes and
reversed order repeats: 40 exact comparisons, 2,504 generated tokens per mode,
66.3213 seconds baseline versus 62.2313 optimized (1.06572×). Host timing includes
existing synchronization; it is not kernel-only timing or sustained fleet rate.

The refill canary matched eight isolated references, captured 394 consumed inputs
over 136 decode rounds and reused a slot four times. Stream execution took 9.1920
seconds including serialized prefill. This proves ownership/capture parity, not
measured speedup versus matched fixed B4. Production remains fixed B4 dispatch;
stream-service helpers require their own source-bound qualification before use.
The server pins its imported source digest, preventing subsequent on-disk edits
from relabeling an already loaded worker. No MTP capture qualification is claimed.

## Resolve and pin before reading

```python
from huggingface_hub import HfApi
repo = "Atlas3D/character-creation-big-five"
revision = HfApi().dataset_info(repo).sha
print(revision)  # Record this full SHA with every derived result.
```

Use that SHA for **every** manifest, batch, asset, and raw-shard download in the analysis.
Do not mix current `main` files with an earlier snapshot. The dataset changes while
capture and publication continue. An absent raw entry is missing publication coverage,
not a zero activation. Failed and incomplete source captures do not become successes
because another stage finished.

## Public files

| File | Contract |
| --- | --- |
| `scenarios.jsonl` | Original synthetic prompt collection; stable `id`, `prompt`, design labels and generation provenance. |
| `manifest.json` | `character-creation.public-big-five.v1`; exact row count, source SHA-256 and exported JSONL SHA-256. |
| `research_manifest.json` | `character-creation.public-research.v1`; immutable metadata batches, record IDs, hashes, counts and source progress. |
| `research_batch_<sha256>.jsonl` | Heterogeneous records; dispatch on `kind`, not guessed columns. |
| `research_raw_manifest.json` | `character-creation.public-research-raw.v1`; separately verified raw-array publication coverage. |
| `research_raw_<sha256>.tar` | Flat, immutable shard of original NPZ files; no corpus database. |
| `research_raw_entries_<sha256>.json` | Raw record-to-entry map, exact bytes, original SHA-256, shapes and dtypes. |
| `research_assets.json` | `character-creation.public-research-assets.v1`; calibration and authored-suite filenames, hashes, sizes and model binding. |
| `layer_00.npz` … `layer_63.npz` | Small calibration banks: relative-dose scales, trait directions, selected neuron columns and observed statistics. Not the full model. |
| `persona_*.json` listed by `research_assets.json` | Authored probe suites, public presets, model metadata and reference recipes. |

Default `datasets.load_dataset(repo, name="default", split="train", revision=revision)`
loads **only the synthetic scenarios**. `train` is a loader convention; all these rows
are discovery material. Use the research manifests for mixed research records and arrays.

## Read with verification, without inference

Install `huggingface_hub` and `numpy` in your existing Python environment. From the
source checkout:

```bash
python persona/persona_read_research.py --revision FULL_HUB_COMMIT_SHA --limit 3
python persona/persona_read_research.py --revision FULL_HUB_COMMIT_SHA --kind baseline_answer --limit 3
```

Programmatic use:

```python
from persona.persona_read_research import Snapshot

snapshot = Snapshot(revision="FULL_HUB_COMMIT_SHA")
for row in snapshot.records(kind="prefill_activation"):
    print(row["id"], row["source"]["id"], row["coverage"])
    break
```

Raw access is explicit because one shard can be hundreds of MB:

```python
arrays = snapshot.arrays(record_id)
peak = arrays["search_peak"]           # [64, 17408], signed maximum-magnitude value
where = arrays["search_peak_position"] # [64, 17408], encoded prompt position
total = arrays["search_sum_abs"]       # [64, 17408], absolute activation sum
mean_abs = total / len(arrays["search_positions"])
```

The helper verifies SHA-256, retains a fixed Hub revision, reads tar entries without
extracting paths, and loads NPZ with `allow_pickle=False`. Download only shards needed
for the question. Preserve record IDs and exact model/runtime/prepared identities in
derived tables. Check the raw entry's shapes/dtypes; do not assume every record kind
contains the same arrays.

## Record kinds and joins

`prefill_activation`:

- Stable record `id`; join to the prompt using `source.id`, not row order.
- `source` supplies approved prompt text, text hash, tier and source binding.
- `identity` contains `model_identity`, `runtime_identity`, `prepared_identity`.
- `request_sha256`, request settings, encoded tokens, `coverage`, validated artifact hashes.
- `candidates` contains at most 32 selected neurons with `layer`, `neuron`, `peak`,
  `peak_position`, `mean_abs`, and full selected per-token `values`.
- `raw_arrays_in_this_json=false`; original all-neuron reductions use the raw manifest.

`baseline_answer`:

- Same prompt/identity/request join discipline; includes a completed unsteered `answer`,
  finish reason, usage and final-prefill artifact references.
- Only revalidated complete source-ledger records qualify. Truncation, failed or uncertain
  inference remains visible in aggregate source counts and does not enter completed pairs.
- Final-prefill MLP/residual arrays are not answer-token activation trajectories.

`qualified_labels` and `coherence`:

- `bundle_sha256` binds a reviewed `bundle` of schema
  `character-creation.public-research-bundle.v1`.
- Bundle `validation.passed` means **export validation**. Inspect individual evidence
  and recommendations for scientific pass/fail/unknown outcomes.
- Label evidence includes all six tested concepts, including failed qualification;
  original evidence hashes identify retained originals after deployment metadata removal.
- Coherence evidence includes complete trials, reused-output provenance and failed heldout
  confirmation. Reuse is not independent replication. Cancelled pilot runs are excluded.

## Measurement semantics: do not collapse these distinctions

1. All-neuron prefill reductions aggregate **all covered encoded prompt positions**, including
   template tokens. They do not retain the full `[layer, token, neuron]` tensor.
2. Selected token profiles retain trajectories for the selected neurons only.
3. Coactivity is an undirected Pearson association. Nearest-neighbor membership can be
   asymmetric; `source`/`target` storage order is not causal direction.
4. Weight paths are signed adjacent-layer output/read affinities. They omit attention,
   normalization and input-conditioned gating; they are not proven functional circuits.
5. Per-layer PCA axes do not share a physical cross-layer coordinate frame.
6. A feature may have positive or inverse activation association. That does not determine
   the dose sign needed to induce behavior.
7. A visible short path does not prove intervening layers can be skipped. No circuit-memory
   compiler or accelerated sparse execution is claimed by this release.

## Model binding and steering

Reference checkpoint: `nvidia/Qwen3.8-27B-NVFP4`, revision
`482ca0f3832238542f8f5295dde86b5f22711d80`; 64 layers, hidden width 5120,
MLP width 17408. Canonical model metadata is `data/persona_model.json` in Git and
`persona_model.json` in HF. Recorded model identity:
`389801c06d68a0d84bd4890b8ce6d26f96e0ee53eb108bcc572c0c09abf5b9a8`.

Neuron addition at the down projection uses:

```text
requested input addition = dose * median(norm(residual leaving layer L), prompt tokens 1:) / norm(W_down[:, neuron])
```

Preserve layer/neuron indices, intervention site, phase, calibration hashes, normalized
dose and actual prepared program identity. Neuron-input addition and post-down column
addition are separately labelled operations. Quantization and rounding can distinguish them.
Do not transfer reference Qwen3.5-4B coordinates to the 27B checkpoint. Do not silently
clamp a fixed recipe or apply a recipe to a mismatching checkpoint.

The public catalog contains 16 authored starting profiles plus six checkpoint-bound
experimental recipes. Baseline has no pushes. GPT-4o-inspired and Skippy-inspired names
do not imply original model weights, endorsement, or validated character fidelity.

## Evaluation contracts

Local viewer integration uses `persona/persona_overnight_integrate.py`. It reads completed
rows of the frozen overnight ledger in short transactions, validates source/request/model
bindings and raw-file hashes, and incrementally caches final-prefill values at the canonical
map coordinates. It starts no inference. Exact message/template/runtime duplicates count
once **within each split**; discovery and heldout observations are never pooled.

The private `results/persona_overnight_viewer.json` manifest binds two compact NPZs
(`persona_overnight_viewer.npz`, discovery; `persona_overnight_viewer_heldout.npz`, heldout)
and `persona_overnight_viewer_records.jsonl` by SHA-256. It is published last. The NPZs
use `character-creation.coactivity.v1`, `final_prefill_mlp_input_last`, and the unchanged
canonical map SHA. `imported_captures` includes duplicates; `unique_observations` is the
sum of `discovery_observations` and `heldout_observations`. Heldout means reserved source
observations, not a newly validated causal result. Original all-position reductions remain
in raw captures. Private book evidence in these files must not enter public Hub exports.

`/api/neighbors` and `/api/neuron-examples` accept explicit dataset selection; `/api/state`
reports the active source and import coverage. Automatic selection prefers a valid overnight
discovery snapshot and explicitly reports a historical fallback when that snapshot is absent
or fails validation. Numeric historical measurements and calibration banks are preserved.

The importer supports `--watch` on the capture host and `--sync --watch` on the viewer
host. `--active-read-mib-per-second` limits shared-disk reads while the finite mapper
is running; that limit lifts automatically when mapping ends. CPU verification workers
remain available. Set this storage budget from measured capture/import throughput.

Concept qualification freezes candidates on discovery before heldout probes. A compact
label requires at least 20 distinct predicted-positive heldout families, measured precision,
a 95% Wilson lower bound **strictly above .70**, balanced accuracy at least .80, recall at
least .75 and performance above the nuisance baseline by more than .05. Probe-family IDs
do not establish independent natural examples. Report conditional precision, sample count,
false positives/negatives, and source hashes—not universal semantic confidence.

Coherence measures supported facts, missing requirements, contradictions and observable
collapse. A schema representation mismatch can be unknown instead of a factual failure.
Do not grade optimism, politeness or a preferred personality as coherence. Free prose is
not certified by a passing fact ledger. Freeze the selected discovery dose before heldout
confirmation; a heldout failure must not tune a replacement on that same holdout. Recovery
operates between full responses, not a proven live-token controller. Interaction tests are
exploratory until independently confirmed.

## Arbitrary-input activation and matched scene workflow

`data/persona_personality_probes.json` (Hub: `persona_personality_probes.json`)
contains 24 new authored discovery prompts across six axes: correction, social
pressure, criticism, reassurance, conflict and uncertainty. They form 12 matched
low/high pressure families, not 24 independent scenes. Each has an exact prompt
SHA, source-span fact contract, unknowns and descriptive behavioral observations.
`persona/persona_personality_probes.py` validates/exports registry rows with empty
`system`, no KV control and no persona instruction. `/api/personality-probes` serves
the validated suite; Persona Studio exposes axis/situation pickers for cast and
preset comparisons. Selection fills the shared editable prompt; only an explicit
run submits inference. Registered browsing uses corpus `personality_diagnostics`.
The suite was **not measured in the completed 256-answer pilot**; its separate
288-answer discovery sweep is now complete.
Report agreement with a false statement as a correctness issue independently of
whether the response is gentle, blunt, sarcastic or hostile. Do not infer a validated
neuron label or user-preference score from the authored axis names.

The separate personality sweep uses all 24 authored prompts, six arms (unsteered
baseline, A, A reversal, B, B reversal and matched random residual steering), and
seeds 80317/49261: **288 answers**, all discovery. The 12 paired families do not
constitute 24 independent scenes. Native response captures retain FP64 residual
means for all 64 layers, covering **136,018 consumed generated tokens**. The local
Qwen-family blinded review is complete across 48 six-answer groups; it is neither
human review nor independent scientific truth. Detail issues occurred in baseline
and steered answers; no default promotion follows.

`persona/persona_personality_release.py` verifies the frozen suite, exact recipes,
request/program hashes, native artifact/provenance bytes and blinded review receipts.
Its release files are [persona_personality_public.json](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_personality_public.json)
and [persona_personality_capture_bundle.npz](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_personality_capture_bundle.npz).
The bundle joins rows to public answer codes and original artifact hashes and holds
`response_mean[288,64,5120]` (float64), `response_count[288]` (int64) and metadata.
It does not include every raw native NPZ or every response-token vector. Counts and
means exclude prompt/template tokens and the final sampled but unconsumed token.
This sweep does not alter the earlier 256-answer pilot or its heldout analysis.

The existing Live activation search accepts whole/partial prompts, phrases, ideas,
feelings and thoughts. Highlighting a span makes that span the next input. A compatible
`capture_search` request summarizes all 64 × 17,408 = **1,114,112** neuron coordinates
across rendered prompt positions, then a second prefill supplies top-candidate token
profiles. Template/system positions are included; this is not user-text-only evidence.
The internal observation token is not an answer continuation. Coverage is reported
from validated artifacts, never inferred merely from a successful HTTP response.

`GET /api/activation-search?id=...` exposes `starter_recipes`: up to three untested
current-input-activity proposals (leading single neuron, a distinct-layer single
neuron, and a three-distinct-layer group when available), at total dose 0.1. Selection
fills the existing comparison form without inference. Users retain manual selection,
editable shared scene text, seed/budget, and baseline/positive/negative arms through
`POST /api/activation-comparisons`. Group dose is shared among selected neurons;
negative additive steering is not ablation. Reset preserves durable records and
invalidates stale UI polling/submission responses. Qualified labels, if present,
require matching model identity and remain separate probe-conditioned evidence.

The updated architecture adds `capture_search` to the canonical six interactive
endpoints (8930–8935). The flat `activation_capture.py` override loads before decoder
imports without editing the shared Flare source tree. Its SHA joins runtime source
bindings and the forward-cache namespace; spawned compiler workers resolve the same
override. `persona/persona_activation_lab.py` dispatches through up to six executor
workers, bounded by configured endpoints, with a 64-active-or-queued search limit.
This permits cross-card concurrency, not multiple simultaneous prepared B1 requests
on one card. Runtime `2ef8694cebe17e8b99d65f0b60c84a7ddd4ba37bbf62861eafd296b8d8dd007c`
passed calibration on all six workers and completed 60 searches plus 72 full scene
responses. Fresh native KV qualification passed 72 requests in
`results/phantom-qualification-7c27d462588e45b89a9bd5b9ce0a8dc3.json`
(SHA-256 `290130b01b6f68f2a7f83a7bdf008f5f9a1585b5e18997a458159d1985394bed`).
Prior reports remain bound to their original runtime/preparation. Status counts
physical GPU UUIDs, not canonical/isolated endpoint aliases.

The fixed input plan and request/result evidence are in
`results/persona_association_research.json`. Of 60 whole-text, fragment and concept
searches, 59 ranked L63 N6985 first (one ranked L58 N9817). This is evidence against
using raw maximum activation as a concept-specific recommendation score. The
24 authored scene comparisons each contain baseline and ±0.1 on the fixed first
candidate, with seed 7035 and 2,048-token budgets. All 72 responses completed;
completion is not semantic quality or efficacy. `persona_association_campaign.py`
validates the frozen plan hash, sources/splits, exact worker identities, search
coverage and matched-arm settings. It never automatically retries an uncertain POST
or changes deployment configuration.
The public export is `persona_association_public.json` on the Hub, containing
60 top-16 profiles, 72 answers and separately identified assistant review. It
excludes local paths and private book excerpts; it does not contain raw all-neuron
arrays. Review identified localized source/contract regressions despite complete
answers, and 35/60 winning peaks at or after the user-end delimiter.

`--stage recipe-depth` reuses the same bounded controller and six deployed workers.
The frozen `results/persona_recipe_depth.json` input has six earlier depth-study
contexts and six sources (L0 N719/N9436, L24 N952/N9228, L48 N221/N7784), six shared
baselines and 144 interventions at -0.2, -0.1, +0.1 and +0.2. All 150 responses
completed. This changes phase from the earlier prefill-only study to all-token
steering; its previously observed heldout contexts are repeated confirmation, not
a fresh holdout. No KV, uncertain-POST retry or automatic preset promotion. The Hub
export `persona_recipe_depth_public.json` includes its frozen design, answers and
separate assistant review. Review records 13 clear narrow contract/source issues
and five ambiguous concerns, with exact evidence; these are not an exhaustive
semantic error rate. No general monotonic dose–harm relationship or repeatable
coherence improvement was established.

Activity-based candidate ranking is not calibrated concept confidence or a predictor
of useful behavior. A future recipe predictor needs context, neuron set, dose and
observed outcomes with independent evaluation. Learned KV is another intervention,
not an automatic substitute for those data or validation.

The completed association-depth record is `results/persona_association_depth.json`,
with analysis in `results/persona_association_depth_summary.json`: six source neurons
× six contexts × six arms = 216 captures (36 campaigns). Arms are baseline, zero,
+0.1, -0.1, +0.2 and -0.2. Baseline/zero equality passed 36/36; upstream and unaffected
same-layer selected-panel invariance passed 144/144 nonzero arms, as did source sign.
Invariance covers saved consumed prompt positions and selected SwiGLU traces; an
absent upstream panel is a vacuous check, not a measurement. Zero checks also include
all-layer residual norms. Six fixed contexts comprise three discovery and three
held-out examples; they do not establish population confidence. Weak-association
controls are active interventions, not causal nulls. Downstream responses vary with
context/dose and include intervening layers and recurrence, not direct wiring.
One sampled token per arm cannot grade prose, personality, coherence or refusal.
The summary validates saved bindings/alignment/finite traces; it does not claim a new
full download/hash audit of every remote NPZ or vocabulary-logit array.

## Literary KV interventions: first adapter and remaining work

Owner requested Phantom-KV after the current sweep/viewer integration, for contrastive
horror/romance prose and persona experiments, with coherence recovery as a separate
hypothesis. The first implementation accepts checkpoint-bound `hybrid_text_prefix`
controls: an ordinary text prefix's full recurrent/convolution/attention state, with
exact off/on semantics. This cached-text adapter is distinct from the separate learned attention-KV prototype described below.

`persona/persona_kv.py` validates the artifact definition and cache layout. Prefix
tokens are independently encoded before the unchanged chat-template IDs. Prefix
building is unsteered and unsampled; neuron steering begins with the actual prompt.
Fresh request admission preserves request-local sampling state. Cache reuse restores
the model-state row, attention pages, optional history and consumed-token offset;
each request still restores the original prepared state on exit. Simultaneous live
token tracing with this control is rejected until separately qualified.

`persona/persona_kv_qualify.py` runs only against already admitted workers, on their
artifact host. It retains request attempts and requires exact rebuild/reuse and
off/on/off equality in generated IDs and final-prefill arrays, including sampled and
combined-neuron cases. The current two literary prefixes are test controls, not
validated genre interventions. Qualification gates artifacts by model/runtime/
preparation/layout; the deployment must verify the linked report and its SHA before
publishing the local `results/persona_kv_artifacts.json` registry. A successful cache
qualification does not establish refusal reduction, persona quality or coherence repair.

Live qualification on 2026-10-26 passed 102 executed requests across all six V100
workers (17 each). The final report is
`results/kv-qualification-d466654c1e1e48328170763e5427b75c.json`, SHA256
`02d62ac6423fbdafc8db5fab8b8d275a555103ed696e4923d6502ccf9542bf89`.
The last worker exceeded the first readiness wait; `persona_kv_resume.py` revalidated
the other 85 saved responses without repeating their POSTs, then tested the previously
unstarted worker. This is 102 executions, not 187 independent replications.

The private discovery integration pilot retained 15 complete responses and one
neuron-only response truncated during repeated self-revision. The text prefixes
introduced observatory-scene facts into unrelated source scenes. A six-cell follow-up
with neuron doses 0.5, 0.25 and 0 on two prompt variants completed without that loop;
both zero-dose answers exactly matched their original baselines. Assistant review
still found invented details and missed instructions. These are local observations,
not a validated default or coherence-repair result. Full source-bound batches and the
bounded follow-up are indexed by `results/persona_kv_literary_pilot.json`; book-derived
prompts and responses remain private. The surface scorer caught truncation but missed
semantic self-revision, so its passing results must not stand in for prose evaluation.

`persona/persona_literary_eval.py` preserves compiled book/workgroup/scene/split hashes,
builds four matched arms and reports conservative refusal evidence separately from
surface coherence. Unclassified responses remain uncertain until manual review; a
review binds the exact answer SHA and evidence span. GET/POST
`/api/literature/experiments` expose qualified artifacts and explicit runs;
`/api/literature/experiment?id=...` retrieves the comparison. The finite backoff
proposal is currently an offline plan, not an automatic controller.

Reviewed upstream commit
[`e0025c69b2fa36f05d4640600f2c83e03ed00ce0`](https://github.com/lordx64/phantom-kv/tree/e0025c69b2fa36f05d4640600f2c83e03ed00ce0).
Its documented grafts target Qwen3-4B-Instruct-2507. Its HF DynamicCache serving and
dense per-layer K/V training are not adapters for this runtime. Our checkpoint has
48 recurrent Gated DeltaNet layers and 16 full-attention layers (3, 7, ..., 63), with
four KV heads of width 256. SM70 attention cache storage contains BF16 bit patterns
in uint16 buffers; recurrent state and convolution history are separate.

Read-only feasibility check on V100, 2026-10-26: the local 27B checkpoint is mixed
ModelOpt NVFP4/FP8 (21.92 GB stored), not a directly trainable Transformers checkpoint.
The installed environment has Torch `2.14.1+cu126`, Transformers `5.21.0.dev0` and
Accelerate `1.14.0`; ModelOpt, `kernels`, FlashAttention and FLA are absent. Its
Transformers quantizer mapping does not recognize `modelopt`. The separate NVFP4
loader rejects pre-quantized checkpoints, multi-device placement and SM70 hardware,
and declares `is_trainable=False`. Installing ModelOpt alone would not resolve these
loader, hardware and backward-path constraints.

Flare's SM70 packed projections use mutable output/workspace custom operations without
a registered backward implementation, and the probe runs in inference mode. Also,
installed Transformers gradient checkpointing clears `past_key_values` for these
layers: enabling it around a cache-injected graft would discard that intervention.
A training implementation needs functional attention-KV injection and cache-safe
checkpointing, with explicit rotary positions and attention masks; the 48 recurrent
layers must still propagate gradients even though only the 16 attention layers own
trainable grafts.

The implemented prototype uses six cards and memory-only expansion, not a new
checkpoint export: it reads the existing packed weights, decodes frozen layers into memory,
and shards approximately 100.2 GiB of FP32 language weights across six GPUs. The
FP16 surrogate passed native forward comparisons but failed its directional gradient
check; that failed report is preserved. Explicit `--weight-dtype float32` passed both
preregistered finite-difference steps: epsilon 0.1 had 2.768% relative error and epsilon
0.2 had 0.0479%, both positive and below the unchanged 12% limit. All six native
next-token argmaxes matched; worst residual relative L2 was 0.051943 and minimum
cosine 0.998652, within the unchanged 0.10 / 0.99 gates. Reports bind precision,
backend, source and dataset hashes. The allocator cap remains 31 GiB per card, with
no offload or expanded checkpoint export.

The first candidate completed 16 discovery-family optimization steps and evaluation
on eight untouched held-out families. Seven held-out target NLLs were slightly lower
and one higher; this does not demonstrate prose quality, genre control, refusal
reduction or coherence recovery. Candidate SHA-256:
`06a33bf4a67c81e90ae3fad8939dbfd3b8fbb08aeb65a02b221fe7777f1e1a8f`.
Native qualification passed 72 requests across all six workers. Exact report:
`results/phantom-qualification-b9f343fa60104d578b5c90b4febb12db.json`.
This qualifies the recorded native runtime/artifact combination, not behavior quality
or the new search-runtime rollout. Training and qualification do not automatically
promote a default control.
The candidate's `admission_validation` links the hashed successful validation report;
its redundant `gradient_validation` field is null in this first training export.
Validation consumers must verify the linked report and its precision/source/dataset
bindings. Only approved benign literary pairs were used, not upstream harmful prompts.

The learned-control pilot in `results/persona_phantom_literary_pilot.json` retains
four matched arms on four discovery prompts from two scene families: 15 complete,
one incomplete (Lisey original, neuron-only, length stop during self-revision).
Pre-output criteria and `assistant_review` are retained separately from human review.
No model refusal was observed by the assistant. Learned-KV-only responses showed no
consistent improvement over baseline. Combined control completed the formerly
truncating response, retained a different task nonanswer, and in one uncertain-intent
scene replaced established source facts with contradictory car/location details.
This is mixed integration evidence, not efficacy or coherence-recovery validation;
benign fictional additions are not automatically treated as factual errors.

Prototype implementation and admission gates:

- `persona/persona_phantom_weights.py` reads bounded tensor slices from the existing
  mixed checkpoint. Packed rows and small projections have CPU comparisons against
  Flare. Decoded dense weights remain a surrogate: Flare's FP8 activation rounding and
  NVFP4 post-dot scaling are not reproduced merely by decoding its weights.
- `persona/persona_phantom_train.py` uses functional attention-KV injection, fresh
  GDN recurrence and nonreentrant block checkpointing. It disables incompatible
  cuDNN only in its own process. A six-device linear/convolution forward-and-gradient
  preflight runs before weight loading. Full-model validation compares six exact
  native prompt captures: every layer requires cosine at least 0.99 and relative L2
  at most 0.10, with all six next-token argmaxes matching. A separate directional
  gradient check must pass. Training requires the unchanged, successful report;
  failed validation does not authorize relaxed thresholds or optimization.
- `persona/persona_phantom_run.py` supervises six separate UUID-bound leases with
  source/input hashes, per-card telemetry, finite deadlines and checkpoint shutdown.
  `data/persona_phantom_pairs.json` contains 24 original benign scene families,
  16 discovery and eight held-out. This initial target-loss experiment is not a
  demonstrated refusal-reduction benchmark. Interrupted artifacts use a separate
  partial schema and cannot be admitted as trained candidates.
- `persona/persona_phantom.py` injects learned post-RoPE K/V into only the 16
  attention layers. Virtual slots are recorded separately from actual token IDs;
  recurrent/convolution state starts at zero, while request RNG and penalties stay
  intact. Exact off/on only. `persona/persona_phantom_qualify.py` requires 72 native
  requests across all six workers (greedy, sampled and combined neuron cases),
  including exact rebuild/reuse and off-state restoration. A marker alone is
  insufficient: artifact discovery verifies its linked report hash and runtime
  binding. The recorded six-worker qualification passed; changed runtime bindings
  require fresh validation. Code and CPU tests alone are not serving qualification.

Preserved requirements for completing learned-graft integration:

1. Preserve the frozen capture runtime and ledger. Add a separately bound interactive
   adapter with exact model, tokenizer, runtime, cache-layout, rotary-position and
   artifact hashes. Reject mismatches before inference. GPU qualification uses fleet
   admission. Historical chemistry reservation is superseded by the owner’s finite John planner authorization; coexist through workboard admission without displacing active chemistry.
2. Establish cached benign literary context equivalence using the complete hybrid
   state. This is a cached-text control, not a learned Phantom-KV result. A learned
   attention-only graft must be labelled as such and independently qualified; it has
   no implicit prefix state in the recurrent layers. No matching trained 27B artifact
   was found in the reviewed upstream release.
3. Validate baseline off/on/off by rebuilding the entire request state. Removing a
   graft cannot erase influence already propagated into generated context or recurrent
   state. Keep prompt framing, tokenization, seeds and budgets matched, including a
   neutral-prefix control where prefix positions differ.
4. Add a cell-level graft reference separate from neuron recipes in
   `persona/persona_explorer.py`. Compare baseline, neuron-only, KV-only and combined
   arms. Preserve `persona_literature_harness.py` book/workgroup, scene, split and source
   hashes in immutable batch snapshots. Keep variants and duplicate editions together;
   select on discovery and confirm on untouched book families.
5. Add refusal/answer/nonanswer/uncertain annotation with evidence spans and manual
   review. Quoted dialogue, character refusal, missing metadata and truncation must
   not silently become model refusal labels. Score required scene facts, omissions,
   contradictions, repetition and completion independently. Existing
   `persona_coherence_eval.py` cannot certify free-prose semantics; fictional novelty
   is not itself an invented-fact error.
6. Extend the finite discovery/backoff logic in `persona/persona_coherence.py` with
   artifact identity. First reduce damaging neuron pressure; test a recovery graft
   only as another candidate intervention. Freeze choices before heldout confirmation
   and retain failures. A refusal flag alone must not escalate intervention strength.

Upstream dose controls primarily select trained artifacts or repeat prefix slots.
Continuous K/V scaling and monotone effects are unqualified; zero-valued cache slots
are not a true off state. Report actual cache slots, artifact choice and measured
effects rather than presenting an uncalibrated alpha as a validated control. Private
book passages and derived cache artifacts remain outside existing public HF exports.

## Response contrasts and the 256-answer pilot

`persona/persona_contrastive.py` freezes 24 development answers (8 authored scenes ×
baseline/A/B instructions), then derives equal-scene response-residual mean differences.
`data/persona_contrastive_scenes.json` fixes 21 separate evaluation families, two seeds,
L32/all-phase, relative dose ±0.1 and a residual-site Gaussian control with equal norm.
The evaluation has 252 balanced answers (6 arms × 21 scenes × 2 seeds) and four exact
request replays. Replays are reproducibility checks, not independent scene evidence.
Twelve families are discovery and nine are held out. Never use held-out judgments to
select layers, doses, directions or follow-up recipes.

Opt-in native `capture_response:{id,layers:[],max_positions:0}` records FP64 sums/means
of FP32 block-output residuals over **consumed generated tokens**. It excludes prompt
and template positions and the final sampled-but-unconsumed token. Count zero means
an undefined mean. Selected profiles permit at most four layers and 128 positions;
means cover all 64 layers. NPZ arrays, token/position hashes, request hash and exact
model/runtime/prepared identities accompany the answer. Capture does not issue extra
decode steps. Qualify off/on/off token, activation and request-local RNG equality
before using a new runtime. Response means can reflect lexical/length/instruction
effects; intervention outcomes test their usefulness.

A `direction` push references a flat NPZ artifact filename, SHA256, direction name,
layer and dose. The bank verifies model and development-source provenance; endpoint
admission also verifies runtime/prepared identity. `persona_response_contrasts.json`
ranks response differences and alignment to **existing mapped down columns only**.
Alignment is not an activation measurement, causal influence or label confidence.

`persona/persona_propagation.py` qualifies exact one-position prefill pulses and
bounded full residual snapshots with fixed input token IDs. Every experimental arm
uses identical chunk partitions. It reports unsplit/split numerical differences
separately, tests source-position upstream invariance and uses repeated zero controls.
Propagation measures ±0.05/±0.1 response at lags 0,1,2,4,8 and layers31,32,48,63.
Multiple-epsilon agreement, signed context comparisons and BF16 spacing are descriptive;
they are not exact Jacobians, direct wiring, execution sparsity or prose coherence.
Future-token earlier layers may change through recurrence.

After source-bound native qualification and explicit six-worker lease admission:

```sh
python persona/persona_contrastive.py development
python persona/persona_contrastive.py fit
python persona/persona_contrastive.py pilot
python persona/persona_contrastive.py blind
```

The bounded `campaign` command waits for qualification and runs these four stages.
Existing attempts are never silently retried. Pilot outputs are ordinary saved
six-card explorer batches. `persona/persona_blind_eval.py` validates exactly252
anonymous judgments, computes paired scene-cluster bootstrap contrasts separately
for discovery and heldout, and proposes follow-up doses from discovery only. Model
judgments are fallible, intervals are descriptive/unadjusted, and no preset is
promoted automatically. Original NPZ captures and failed attempts stay auditable.
Public exports accept only the frozen authored scenes and omit local paths.

Completed reference run (2026-10-26): 24 development answers, 256 pilot answers,
90 native response/pulse qualification requests, 60 fixed-token propagation captures,
and 48 discovery-only self-ratings. Four preregistered cross-GPU replays matched exact
generated-token hashes and text. Current runtime identity is
`f6ab2595295abe692321cb7c4c5248287c661c7f4e2e5af4eb478f79e5d38954`;
prepared identity is `b9665f876d96a82c7e8d0a77146f9e8533775188b64a0de4907808fad6bcceb4`.
Learned KV also passed fresh 72-request qualification for this binding in
`results/phantom-qualification-12b894292d5d40f4b15fdfb9c89e567e.json`,
SHA-256 `b8855ff4cce8db5489c874a74ba0ec266576d22b24de6b5d6028af0934c5cebe`.

The stateless, tools-disabled Opus5.5 critic received only anonymous answers,
authored scenes and the rubric. On heldout scenes, A's dry-wit contrast was
+0.889 versus baseline (95% scene-bootstrap interval [0.333,1.500]) and +0.833
versus random ([0.278,1.500]). B's warmth contrast was +0.444 versus baseline
([0.167,0.722]) but +0.167 versus random ([-0.167,0.500]). These unadjusted,
descriptive intervals do not certify quality. Observed detail/coherence issues and
uncertainty prevent automatic promotion. Discovery and heldout remain separate.

Self-rating diagnostics (`persona/persona_self_rating.py`) use four discovery
families × two seeds × six arms, rated in fresh unsteered requests on a 0–9 scale.
All 48 ratings parsed. Spearman agreement with the blinded external judge was 0.575
for dry wit and 0.427 for attentive candor; within-scene centered values were 0.538
and 0.323. No ceiling ratings occurred. The 48 dependent observations do not
establish reward quality, judge calibration or heldout generalization. No self-reward
optimization or model training was performed.

`persona_response_study_public.json` is the public index: model/runtime/preparation
bindings, exact file hashes/sizes, policy, 24 development answers, four replay answers,
contrastive rankings, qualification, propagation summaries and 48 joined self-ratings.
`persona_pilot_public.json` contains the 252 judged answers and split-specific analysis.
`persona_response_directions.npz` stores both 64×5120 development directions.
`persona_response_capture_bundle.npz` contains:

| Array | Shape / semantics |
| --- | --- |
| `response_mean` | `[280,64,5120]` FP64 means over consumed generated tokens only. |
| `response_count` | `[280]` int64 consumed-token counts; prompt and final sampled-but-unconsumed token excluded. |
| `propagation_residual` | `[60,5,4,5120]` FP32 fixed-input block-output snapshots. |
| `propagation_positions`, `propagation_token_ids` | `[60,5]` exact encoded positions and token IDs. |
| `propagation_layers` | `[4]`: 31,32,48,63. |
| `calibration_residual_median` | `[64]` checkpoint-bound dose scales. |
| `metadata` | JSON scalar with exact row joins, source/artifact/request/token hashes and bindings. |

Download at a pinned Hub revision; verify the index hashes before loading with
`numpy.load(..., allow_pickle=False)`. Join answer rows by scene, seed and arm;
exclude replay rows when estimating effects. To reproduce the fitted directions,
average each development scene's style-minus-baseline response means, giving each
scene equal weight. Compare the result to `direction_A`/`direction_B` in the direction
artifact. The bundle's complete source residual means are distinct from selected
neuron trajectories or all-neuron-by-all-token captures.

`persona/persona_response_bundle.py` validates original native NPZ/provenance files
on the capture host and extracts only the approved authored data. Pulse vectors
are compared exactly after the runtime's FP32 conversion; sub-FP32 NumPy norm
rounding differences do not change the native intervention. One FP32 ULP does fail.
`persona/persona_response_release.py` creates the public index only after a verified
bundle publication receipt; it has no inference or upload side effects. The Studies
panel reads `/api/response-study` and links existing saved six-card batches.

`persona/persona_repair.py` ran a separate 42-answer outcome-guided follow-up:
seven discovery positive-A/B failures × baseline, original +0.1, +0.05, +0.025,
random +0.05, random +0.025. All 14 baseline/original replays matched exact tokens
and text. A new stateless blinded Opus5.5 call graded all 42 before its key was read.
Original/half/quarter dose fidelity issues were 5/3/2 of seven, and coherence issues
4/0/1; the quarter dose also had one fidelity uncertainty. Baseline and random arms
had no judged issues, with one uncertainty for quarter random. Dry-wit target means
were 0.5 for each A dose versus 0.25 for baseline/random across four selected cases;
B's three selected cases showed no target-score advantage. Selection on previous
failures and critic variability prevent treating these as general error rates or
fresh heldout evidence. `persona_repair_public.json` retains the answers, code/arm
joins, exact design and judge evidence. No default changed or automatic repair ran.

Method sources: [self-reward persona study](https://www.lesswrong.com/posts/LBKjnyKDEwEJ45Wjf/self-reward-can-elicit-diverse-model-personas),
[response-steering code](https://github.com/fjzzq2002/self-reward/blob/main/scripts/persona_steer.py),
and [Jacobian-lens theory](https://arxiv.org/abs/2610.31955).
This is response-contrast steering and finite-difference instrumentation, not GRPO,
self-reward training, weight interpolation or an exact Jacobian-lens implementation.

## Verified native B4 proof and release boundary

The source-bound `character-creation.batch-proof.v1` experiment passed; its separate
`character-creation.batch-proof-validation.v1` receipt reports `verified:true` and
`passed:true`. All 16 run sets cover serial/pair/four-request execution, exact repeats,
physical-slot and packed-row permutations, unaffected requests, early-finished holes,
and sampled RNG isolation. The four distinct pushes are L32 N1728 MLP-input +0.1,
L32 N2608 output-column −0.1, L32 extraversion residual +0.1, and L32 seeded random
residual +0.1. Each receives the same authored arithmetic-correction prompt.

Matched serial versus B4 warm runs each generated **203 tokens**: 8.779256891 versus
3.165097378 wall seconds (**2.773771497×**). All four first-decode comparisons have
KL=0, max-absolute logit difference=0, hidden relative L2=0 and equal top-1; full
recorded token sequences are exact. Gates were frozen at KL≤0.01, hidden relative
L2≤0.02, top-1 agreement=1, plus exact repeats/isolation/permutations. The result
uses serial per-request prefill followed by a true four-active-token decode graph;
it is not four concurrently queued B1 calls. Timings exclude model preparation.

`persona/persona_batch_release.py` produces `results/persona_batch_public.json`
(`character-creation.batch-public.v1`) offline from byte-bound proof, plan,
validation receipt and NPZ. The JSON includes the authored prompt, four push
specifications, exact sparse program vectors, policy, all16 run summaries, gates,
timings, core/source identities and an exact NPZ filename/SHA/size/array inventory.
Its prior-event projection preserves the failed pre-execution graph-catalog attempt
without host units, lease IDs, local paths or the prior worker manifest.
The unchanged numeric-only `persona_batch_proof.npz` holds 32 arrays. Sanitized JSON
omits raw physical completion traces and deployment receipts: **do not claim it is
independently executable input to the raw validator**. First-decode arrays are retained;
later logit equality is supported by recorded hashes rather than full logit arrays.

The separate native capture qualification now passes all **10 checks**, with an
independent receipt reporting `verified:true, passed:true`. Its 16 run sets cover
serial/B2/B4 execution, repeats, slot/row permutations, zero controls, one changed
request, unequal budgets `[1,3,8,32]`, and sampled/permuted execution. Exact checks
cover all 64 layers and per-request consumed token IDs, positions and generations.

The bounded artifact retains the first decode's three planes (post-residual,
pre-MLP and MLP input), every consumed step's three-plane hashes, and FP64 residual
sums/means with counts. It does **not** retain full vectors for every token. Means
exclude prefill and cover `generated_token_ids[:-1]`: the final sampled token is
not consumed. The offline validator checks coverage, first-plane hashes,
sum/count arithmetic and cross-run equality; reconstructing later-token sums from
hashes alone is impossible. Capture timings include synchronization/host readback
and must not be treated as kernel-only throughput or compared with the earlier
uninstrumented speed figure.

Release files: [persona_batch_capture_public.json](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_batch_capture_public.json)
and [persona_batch_capture_proof.npz](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_batch_capture_proof.npz).
The failed pre-execution attempt to modify an immutable `MappingProxyType` graph
registry is preserved with its old plan and failed-worker/source lineage. Explicit
recovery used a temporary prepared facade; original native graph mappings remained
untouched, and the original prepared object was restored by identity before
capture-tensor cleanup.

Production native B4 HTTP queueing/routing is now connected and qualified on six
cards. `persona_batch_server.py` serves authoritative full-budget page preflight and
batch completion endpoints; `persona_batch_dispatch.py` and Manager issue one native
batch POST for up to four independent requests. Cascaded B1 prefill feeds genuine
B4 decode, with private KV/recurrent/RNG state and per-request capture provenance.
Unsupported controls retain B1 routing. Readiness is checked before claiming jobs;
wrong model/runtime/prepared bindings fail closed. Never retry uncertain POSTs.
The 24-job real Manager pilot passed; old direction artifacts require the explicit
compatibility receipt, not merely a matching checkpoint name.

The first six-card reasoning sample contains 48 responses, **51,428 generated tokens**
and **51,380 consumed captured tokens**. First submission to last HTTP response:
**281.3823335 s, 182.769 aggregate tokens/s**. Through artifact validation:
**286.7781923 s, 179.330 tokens/s**. Preparation is excluded; artifact-inclusive
throughput is not GPU kernel throughput. Earlier raw Flare **220.5549 tokens/s/card**
used MTP depth four and cached prompts without deep capture. Do not extrapolate
that rate to this workload or present a sustained daily capacity as measured.

## Operating a fork

The fixed reasoning battery uses `MultiSynt/MT-Reasoning-Prompts` at
`fd38d4ff2264058042e5f7832821ac588ed96805` and
`marcov/math_dataset_algebra__linear_1d_composed_promptsource` at
`ee58f7f38cd91075f5a91f5366bd4076014ae8bd`. The accidentally suggested dspark
dataset is excluded. `data/persona_reasoning_index.json` publishes IDs, source
revisions, hashes and splits without algebra text. Reconstruct local inputs with
`python persona/persona_reasoning_battery.py --build`; install `requests pyarrow`
for this bounded acquisition step. The builder checks pinned metadata and row
hashes, preserves authored translations, and refuses different existing files.
Raw local battery/source files are excluded from Git because the algebra mirror
does not declare a redistribution license.

There are 32 unique algebra questions (24 discovery, eight heldout) and 16
explicitly translated general prompts (12 discovery, four heldout). MT is a
multilingual prompt-only source, not a multi-turn answer dataset. Numeric scoring
accepts an unambiguous final answer and exact rational equality to the supplied
reference; ambiguous output is unscorable, not automatically incorrect. General
cases remain unscored. `wrong_answer_challenge` creates labeled synthetic history,
not a model-generated conversation. `persona_reasoning_sweep.build` prepares
288 fixed Manager jobs using six qualified recipes, equal seeds and budgets;
its read-only summarizer keeps execution failures, correctness and unscorable
answers distinct. Enqueue all jobs before running with `max_total_jobs=288` so
heldout outcomes cannot enter the adaptive planner. No inference follows from
building or loading this battery.

The reasoning sweep is **complete: 253 complete, 35 incomplete, zero failed**.
Its first reporting checkpoint included nine valid length-limited captures that were
reclassified offline with append-only audit records preserving their original
classification and responses. No requests were replayed. `checked()` retains its complete-answer gate. The dedicated
strict parity validator admits full-budget length termination as capture-valid:
`complete:false`, `error:null`, `generation_status:"length_limited"`,
`capture_validated:true`, and Manager state `incomplete`. Invalid artifacts remain
failures. Local summaries retain partial answers and response hashes but never
score their correctness; algebra is unscorable and general prompts are unscored.
Usage includes validated complete and incomplete captures with separate counts.

Run the finite reporting watcher only against the frozen input and current launch,
with `DEADLINE_EPOCH` set to the authorized finite deadline:

```bash
python persona/persona_reasoning_sweep.py --root "$PWD" --until "$DEADLINE_EPOCH" --watch --publish
```

It reads SQLite snapshots without changing jobs, refreshes
`results/persona_reasoning_results.json` and `persona_reasoning_public.json` every
15 seconds, and stops at all-terminal or deadline. Full answers remain local.
Only the public text-redacted JSON is uploaded to
`Atlas3D/character-creation-big-five/persona_reasoning_public.json` at 48-terminal-job
milestones and final/deadline; zero-only snapshots are not published. The separate
`persona_reasoning_sweep_publication.json` records durable intent before the HF CLI
call and verifies bytes at the returned immutable commit. An uncertain upload
blocks further publication, while offline summaries continue. Preserve this receipt
on restart; do not erase uncertainty to force a retry. Heldout outcomes remain
confirmation-only, with no adaptive selection.

- CPU dependencies: `python -m pip install -r requirements-persona.txt` (includes optional
  local NLP tools and a hash-pinned spaCy language model). Analysis-only users can install
  just `datasets huggingface_hub numpy`. Pytest is an additional test dependency.
- Native model execution requires the separately pinned
  [Flare runtime](https://github.com/Atlas3DSS/flare-local-inference); CPU dependency install
  does not prepare GPU graphs or download Qwen. Check local model caches before downloads.
- Configure admitted endpoints and model/calibration bindings before launching
  `python persona_explorer.py --config persona_explorer.local.json --port 8792`.
- Explorer config accepts `nlp_python` (default current interpreter) and
  `overnight_remote: {ssh_host, hostname, remote_root}`. Set your own values; reference
  defaults describe the original fleet, not resources a fork may use automatically.
- Activation launcher accepts `--config`, `--jobs-config` and optional fleet keys
  `remote_project_root`, `heliopause_path`, `model_path`, `calibration_dir`.
- Reference multi-card schedulers, GPU leases, thermal checks and memory admission require
  adaptation to your infrastructure. The connected B4 service requires its own qualified
  source/prepared binding and compatibility receipts; the original proof alone does not
  qualify a fork. Never remove locks to claim concurrent request safety.
- Detailed setup: [Explorer guide](https://github.com/Atlas3DSS/Character-Creation/blob/main/PERSONA_EXPLORER.md),
  [porting guide](https://github.com/Atlas3DSS/Character-Creation/blob/main/fork_agent.md).

## Publication lifecycle

`persona/persona_hub.py` manages the approved synthetic prompt subset.
`persona/persona_public_bundles.py` prepares source-bound, sanitized evaluation bundles and
calibration assets; it does not upload them. `persona/persona_hub_research.py` publishes
verified research batches from read-only capture ledgers.

- Metadata watcher polls every five minutes; commits at 500 accumulated records, after
  15 minutes, or when its source run reaches a terminal state. Completed reviewed bundles
  accompany the metadata publication.
- Raw watcher runs on the capture host and uploads up to 32 original NPZ records per
  immutable tar shard directly to HF. It continues draining the backlog after capture ends.
- Both verify the pinned remote manifest and uploaded file hashes before advancing durable
  checkpoints. An uncertain commit reconciles its immutable intent; it never reruns inference.
- Publisher HTTP connect/read/write/pool timeouts are 15/120/120/60 seconds, preserving
  the SDK's redirects and request hooks. This requires `huggingface_hub>=1.27,<2`.
- Explicit HTTP 412 conflicts retry the same pending bytes up to three times after
  rechecking the publication lane, original subset and dataset card. Unknown commit
  outcomes still retain their pending intent for recovery; they never trigger recapture.
- Metadata and raw counts are independent. Read `research_raw_manifest.json` for verified
  array coverage even if the dataset card or metadata snapshot has an earlier count.
- Run without `--publish` to validate policy/source files only. Publication requires an
  explicit local public-release policy binding the exact synthetic corpus and reviewed
  authored sources. Do not replace those checks with a recursive directory upload.
- Credentials, private books, book-derived token IDs/profiles, arbitrary interactive prompts,
  local configs and proprietary research stay private. Book tiers may contribute aggregate
  progress counts only. Unpublished/failed records remain in the local audit trail.

Core CPU validation uses `tests/test_persona_*`; native CUDA qualification is separate and
requires an admitted GPU/runtime. NLP tests require the pinned spaCy model. Test skipping
for absent CUDA is not GPU qualification.

## Attribution and license boundaries

Credit [Lyra / neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) for the
foundational steering workflow; this is an independent implementation. Runtime lineage
and other credits are recorded in the source guide and Flare repository. Synthetic prompts
are CC BY 4.0. Do not extend that license claim to the upstream checkpoint, derived model
columns, third-party tools, or private books. Preserve their applicable terms and provenance.
