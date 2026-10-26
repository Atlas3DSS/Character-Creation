# Character Creation: agent interface and evidence contract

This document is the machine-oriented entry point. [README_human.md](README_human.md)
explains the motivation and workflow. Source: [GitHub](https://github.com/Atlas3DSS/Character-Creation).
Public data: `Atlas3D/character-creation-big-five` on Hugging Face, repository type `dataset`.

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
   admission; John remains reserved for chemistry.
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

## Operating a fork

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
  adaptation to your infrastructure. Single-card/multi-request behavior is not qualified by
  the six independent B1 workers. Never remove locks to claim concurrent request safety.
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
