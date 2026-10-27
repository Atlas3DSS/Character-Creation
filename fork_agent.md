# Fork Agent Runbook: porting Character Creation to another system

**Audience:** an AI agent asked to deploy, fork or adapt this repository for its own user. This is an executable checklist with validation gates. It is **not** a one-click installer, and nothing here claims the project already runs elsewhere.

## Instruction to follow first

> You are porting Character Creation (Persona Explorer / Persona Studio) for your user. Work through the phases in `fork_agent.md` in order. Do not skip a gate. Read the owner's instructions, existing session authorization and real target system first. Carry out already-authorized work without asking again; ask only for missing scope or an action outside that authorization, such as replacing someone else's workload or using unapproved paid compute. If a gate fails, fix routine issues within scope and report unresolved blockers with their output. Do not weaken a safeguard to make it pass. Never claim personality fidelity, causal steering, native-cache or MTP qualification from a diagnostic success.

## Credit and provenance (keep in any fork)

- **Primary methodological credit: Lyra**, [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) (reviewed revision `f79b2196e29aacbad89684938baac8d97b172fd1`). It is the source of signed neuron sweeps, relative-dose normalization, the read/write distinction and rollout comparison. This repo is an independent implementation; no upstream code is vendored and no endorsement is implied. Keep this credit at the top of any README you publish. See `PERSONA_EXPLORER.md#attribution-and-source-record`.
- **Separate runtime:** [Atlas3DSS/flare-local-inference](https://github.com/Atlas3DSS/flare-local-inference), default branch `atlas/sm70-public`. Read its [README](https://github.com/Atlas3DSS/flare-local-inference/blob/atlas/sm70-public/README.md) and [STEERING.md](https://github.com/Atlas3DSS/flare-local-inference/blob/atlas/sm70-public/STEERING.md) first. Build, preparation and kernel support are defined there, and you follow its pinned docs, not this file. The steering extension builds on Flare revision `c3dbf81851c09b8666462c75a51a85895302fb21`. Re-check the branch before use.
- Also credit [RhizoNymph/vLLM steering runtime design](https://github.com/RhizoNymph/vllm/blob/feat/integration/docs/design/steering_runtime.md) (request ownership, cache identity) and [heterodoxin/AbliteratedOutTheBox, `strata`](https://github.com/heterodoxin/AbliteratedOutTheBox/tree/strata) (native residual interventions; its results do not validate this project).

## What this is, and what it is not

A research workbench: Persona Studio, 16 authored presets (`data/persona_presets.json`), calibrated Big Five residual contrasts, precise neuron controls, up-to-six-slot comparisons, measured co-activity and durable scheduled jobs.

- The **GPT-4o-inspired** and **Skippy-inspired** presets are approximations. They are not recovered weights and not validated personalities.
- The private weights, calibration, activation captures, results databases and book corpus are **not bundled**. Your user gets none of them with a clone.
- Current evidence on the reference fleet, not to be overstated: 16 distinct saved recipes. 70 complete Studio answers (two presets, six candidates plus baseline each, five prompts), with the five baselines matching exactly. One adaptive cycle of 165 observations (33 matched five-arm sets, zero errors, atomic publication) that added **0 new matrix rows** because the originals were already known. That shows one planning/control/publication cycle, **not** multi-round adaptive success or personality fidelity.
- Documented gaps that a port does not close: 27B logit lens, a weight-path web (co-activity is a different thing from Lyra's weight web), live per-token activations. See `PERSONA_EXPLORER.md#feature-comparison-with-lyras-explorer`.

## Reference hardware (do not assume it)

Six 32 GiB V100s on one GPU host, plus a separate local 27B llama.cpp planner on another host with a verified 131,072-token context. Hostnames, SSH aliases, GPU UUIDs, systemd unit names, filesystem paths and model SHA values in the source are **owner-specific and not portable**. One 16 GiB V100 cannot be assumed to hold the reference checkpoint unchanged. Do not claim otherwise.

---

## Phase 0: Authorization and discovery (no changes)

1. Gather these facts from the existing session, owner instructions and bounded system inspection; ask only for missing information. Record them in an owner-approved location:
   - Which hosts and GPUs are approved? Is any external or paid compute allowed? (Default: **no**.)
   - Which workloads already run there? (Default: preserve all of them.)
   - What resource admission system must be used (GPU lease/queue, scheduler, thermal and RAM guards)?
   - Is this a **same checkpoint and same fleet shape** port (Track A) or a **new model, architecture or fleet size** port (Track B)?
   - Are private books or corpora involved? Who owns them, and are the files explicitly authorized?
2. Inspect: OS and `python3 --version` (3.11+ needed), `nvidia-smi -L`, VRAM, RAM, free disk, whether `systemctl --user` works, `zoneinfo` data, SSH reachability between hosts.
3. Read, in order: `README.md`, `PERSONA_EXPLORER.md`, then the runtime's `README.md` and `STEERING.md` (links above).

**Gate 0:** you can state, in writing, the approved hosts and GPUs, the admission system, and the track. If the user has not approved compute, stop and deliver only the Phase 1 to 2 analysis.

## Phase 1: Inventory every hardcoded dependency (read-only)

Do not blindly rename. For each item, record: what it protects, what your system's equivalent is, or "none; needs code and tests". Safeguards must be **re-implemented**, not deleted.

| Source | Owner-specific binding to inventory |
|---|---|
| `persona/persona_job_fleet.py` | Hostname equality checks, `ssh_host` options, six fixed `character-persona-probe-N.service` units, planner/tunnel units, ports `8930-8935` and `18511`, `systemd-run --user` argv, SHA-256 comparison of deployed worker sources (`persona/persona_probe.py`, `persona/persona_capture.py`, `persona/persona_vectors.py`, `persona/persona_fleet.py`, `persona/persona_qualify.py`, `persona/persona_lease.py`), the `gpu-workboard` CLI (`status --json`), 960-minute cap, SSH `-L`/`-R` forwarding. |
| `persona/persona_service_control.py` | Fixed `UNITS` allowlist, per-unit expected hostname, Linux `pidfd` (`os.pidfd_open`), `systemctl --user show` parsing, launch-token and job-ID environment names. |
| `persona/persona_lease.py` | Regex for six probe units, `/proc/self/cgroup` unit check, hash-verified installed `wb_hold` supervisor, `PERSONA_*` environment variables, absolute-deadline watcher. |
| `persona/persona_launch.py` | `taskset` CPU affinity, six-entry config lists, lease-wrapper argv order (`run <project> <label> <VRAM_GiB> <minutes> <gpu-label> --`), the hardcoded 31 GiB reservation, `WB_HOLD_WAIT`, CUDA/sysroot/cache environment. |
| `persona/persona_planner.py` | Hardcoded planner GPU UUID, workboard path, llama.cpp binary, GGUF path and SHA, host check, port, alias and `131072` context. |
| `persona/persona_sculptor.py` (`trusted_planner`) and `persona/persona_jobs.py` | Planner host binding, `DEFAULT_ZONE`, job kinds, corpus names, `data/persona_model.json`. |
| `persona/persona_probe.py`, `persona/persona_prepare.py` | Prepared-package and runtime identities, retained four-slot/depth-four-capable preparation state (`mtp=True`), context 4096 / 32 pages and export gates. Diagnostic execution is still one active request and ordinary decode; preparation settings do not establish MTP execution. |
| `persona/persona_coactivity_refresh.py` | Fixed SSH/path transport, source-file list, corpus-sync schema. |

The interface supports **1 to 6** configured endpoints for comparisons. The **scheduled adapter and the Big Five generator intentionally require exactly six** workers and fixed owner-host/service bindings. A different fleet size or a non-Linux/non-systemd OS needs explicit code **and test** changes (`tests/test_persona_job_fleet.py` encodes the six-card assumption). Offer your user that choice before doing it.

**Gate 1:** a written inventory with no unclassified hardcoded value. Searching for hostnames, `GPU-` UUIDs, `/home/`, `.service`, ports, `wtatum`/`v100` literals and `gpu-workboard` is a good way to find them.

## Phase 2: Environment and CPU-only baseline

Clone the chosen fork into the user's approved destination, or reuse their existing checkout. Reuse the approved Python environment; create one only when needed and consistent with the owner's directory rules. The interface needs NumPy. Review `requirements.txt` before installing optional legacy training dependencies; the runtime has its own pinned environment.

```bash
# In the repository root, with the approved environment activated:
python3 -m py_compile persona/*.py tests/test_persona*.py
python3 -m unittest discover -s tests -p 'test_persona*.py'
```

Use bounded file reads and searches. Do not recursively traverse a home directory, mounted archive, or external symlink tree to discover assets.

The CPU tests normally mock host bindings and do not require the reference fleet; do not dismiss a failure merely because you are on another host. Port-specific assumptions may require deliberate test updates in Phase 6. **Do not "fix" them by deleting the checks.** Report every failure with output.

**Gate 2:** syntax/compile checks pass. You have a list of which tests fail and why. Tests unrelated to host bindings pass.

## Phase 3: Runtime (separate repository)

1. Clone `Atlas3DSS/flare-local-inference` at the `atlas/sm70-public` default branch. Record the commit.
2. Follow its docs for the build, the Heliopause extension, kernels and package preparation (`tools/prepare_replay.py --steering-max-sites 64`). Read what it says about SM70 and your GPU architecture. If your GPUs are not supported there, **stop**. Do not substitute another runtime silently.
3. Cold startup and compile can take **tens of minutes** (storage throughput and serialized shared-cache publication affect it). **Never bypass compile or cache locks.** Wait, or kill nothing you do not own.
4. Diagnostic workers (activation capture, `persona/persona_probe.py`) and the production steering server are different evidence paths. A diagnostic success does **not** qualify native cache reuse or MTP/speculative execution. Production needs its own prepared package passing every export gate (`persona/persona_prepare.py plan`, then `run`).

**Gate 3:** the runtime builds at a recorded commit and its own documented checks pass.

## Phase 4: Model identity and calibration

### Track A: same checkpoint, same fleet shape
- The checkpoint is `nvidia/Qwen3.8-27B-NVFP4` at revision `482ca0f3832238542f8f5295dde86b5f22711d80` (`data/persona_model.json`). Ask your user for the existing local copy. **Do not silently download it, and do not duplicate it.**
- Calibration artifacts (`results/layer_00.npz` to `results/layer_63.npz`, `results/persona_map.json`) are private. Obtain them from the owner **or regenerate** with `persona/persona_capture.py` and `persona/persona_map.py` (see `PERSONA_EXPLORER.md#calibration-and-experiments`). Their hashes must match model identity.

### Track B: new model, architecture or fleet size
- If only the fleet changes, keep the checkpoint identity truthful and port the resource adapter; do not invent a new model identity. Runtime changes still require their own validation.
- When the model changes, edit `data/persona_model.json`. **Model identity includes the model ID, immutable revision, quantization and layer/hidden/MLP dimensions** (`persona/persona_vectors.py:ModelSpec.identity`). Prepared-package and runtime identities are separate bindings. Sites, widths and the 64-layer/17,408-unit assumptions in the UI, planner prompts and tests must be audited.
- **Never reuse another checkpoint's calibration, maps, saved recipes or co-activity data.** The code rejects mismatches, so do not bypass that. Regenerate everything: capture, layer banks, map, qualification.
- Lyra's Qwen3.5-4B coordinates (`data/lyra_recipes.json`) are labeled cross-model hypotheses only.

**Gate 4:** the model JSON validates (`ModelSpec.load`). Existing calibration, if supplied, has verified identities and hashes. Otherwise record the regeneration plan and complete it after the fleet configuration in Phase 5; missing calibration must never be fabricated or treated as ready.

## Phase 5: Workers, endpoints and the explorer (smallest correct deployment first)

1. Create `persona_fleet.local.json` (ignored by git; never commit it) from the fields consumed by `persona/persona_launch.py` and `persona/persona_probe.py`: host, SSH alias, GPU UUIDs, GPU labels, **disjoint** `cpu_sets`, `compile_workers`, runtime and cache paths, checkpoint, model identity, RAM guard. Fill every value from the target system. Leave `TODO_` markers for anything you cannot verify.
2. If calibration must be regenerated, create the documented capture plan, run authorized capture workers, then assemble the layer banks and map using the [calibration instructions](PERSONA_EXPLORER.md#calibration-and-experiments). Verify the resulting identities before starting persona comparisons. For probe workers, dry-run first: `python3 persona/persona_launch.py --config persona_fleet.local.json --mode probe --minutes <N> --base-port 8930 --dry-run`. Review each command.
3. Launch within the user's existing authorization and admission system; ask only if that authorization is missing. Keep the **exact InvocationIDs, launch tokens and absolute deadlines** the launcher and `persona/persona_job_fleet.py` record; cleanup of resources you own depends on them. Never stop a service by name alone.
4. Bind services to **loopback** (`127.0.0.1`), or to a reviewed, authenticated network path. Do not expose a worker, planner or explorer on `0.0.0.0` without that review.
5. Before any inference, check `/v1/models` on each worker: model identity, prepared identity, runtime identity, protocol, site names, vector widths, request-site limit, observation-only capability. Admission is rejected on any mismatch.
6. Create `persona_explorer.local.json` (1 to 6 endpoints, schema in `PERSONA_EXPLORER.md#run-the-explorer`) and run `python3 persona_explorer.py --config persona_explorer.local.json --port 8792`. Open `http://127.0.0.1:8792`.
7. Resource realism: admit by actual VRAM, host RAM and thermal limits plus disjoint CPU sets. Do not add artificial CPU burn to look busy.

**Gate 5:** calibration banks and the map are present with verified hashes and identities; each worker's `/v1/models` identities match; `GET /api/capacity` reports the expected reachable count; an unreachable worker is reported as such.

## Phase 6: Planner and scheduled jobs (optional; requires explicit code port)

1. Planner: serve a local llama.cpp model on loopback (reference: port `18511`, alias `persona-manager`, 131,072 context). Replace the owner constants in `persona/persona_planner.py` through a reviewed change, not a rename. Keep: model digest verification, exact-context verification, no silent context reduction, no stopping of other owners' services.
2. JSON transport: planner calls **must** use `response_format={"type":"json_schema","json_schema":{"name":...,"strict":true,"schema":...}}` for this llama.cpp build. The Big Five sampler grammar (`PLAN_JSON_SCHEMA` in `persona/persona_jobs.py`) deliberately **omits max string lengths**, because finite repetitions exceeded the compiler guard. Python (`validate_plan` and friends) enforces every length and facet bound **before dispatch**. Do not remove those checks.
3. Port `persona/persona_job_fleet.py`, `persona/persona_service_control.py`, `persona/persona_lease.py` and the matching tests together for your host names, unit names, ports, workboard/admission CLI and fleet size. Keep: launch-intent receipts written **before** the uncertain call, pidfd + token + cgroup cleanup, absolute-deadline propagation, reverse-order cleanup, "never stop another owner's process".
4. Create `persona_jobs.local.json` (schema in `PERSONA_EXPLORER.md#schedule-the-planner-and-six-gpu-workers`): `enabled`, `root`, `fleet_config`, `expected_prepared_identity`, `expected_runtime_identity`, `lease_wrapper_sha256`, `timezone`. Then `python3 persona/persona_jobs.py validate-config --config persona_jobs.local.json`; run `daemon` under a persistent user service, with a stop allowance long enough for cooperative cleanup.
5. Time handling: scheduler must reject nonexistent/ambiguous DST local times (`scheduled_time`). Test with your IANA zone.

**Gate 6:** the ported tests pass and still contain the original ownership, deadline and DST cases (not deleted or loosened).

## Phase 7: Data, corpora and co-activity

- Keep **discovery and heldout separate**. Heldout answers never feed adaptive planning; using them to tune a persona consumes that holdout.
- Prepare the public corpus: `python3 persona/persona_corpus.py prepare`. Private books: only files your user explicitly authorizes, ingested with `persona/persona_corpus.py ingest --book-dir <PATH>`, and they stay **local and git-ignored**. Never publish passages, raw responses or activation archives.
- Co-activity: `persona/persona_coactivity_extract.py` verifies hashes and creates its archive exclusively; `--replace` must preserve prior rows (audited duplicate exclusion only). Refreshes are append-only and deduplicated. Constant neurons and insufficient samples (fewer than 8 unique prompts) get **no** edges. The sourced records and links must come from the user's data, never copied from the reference deployment.
- Correlation is not wiring or temporal phase, and co-activity is not Lyra's weight-path web.

**Gate 7:** corpus manifest says `status: complete` with a matching hash; extract + refresh tests pass; no private text or generated archives are staged for publication. Review the actual staged diff; approved public manifests and authored preset metadata may legitimately live in `data/`.

## Phase 8: Validation suite

Run each item, and record outputs for the handoff.

1. `python3 -m unittest discover -s tests -p 'test_persona*.py'` on CPU (no GPU).
2. Compile/syntax check of all changed files.
3. **Exact zero-baseline controls:** baseline and explicit-zero arms must match bit-for-bit.
4. **Complete 2,048-token-budget** comparison for anything you will report. A smaller explicit smoke run is allowed for correctness checking only, labeled smoke, **never promoted**.
5. 1x6 and 6x1 layouts where the hardware permits it.
6. Resource cleanup, cancellation, restart recovery and absolute-deadline tests.
7. Scheduler timezone/DST tests.
8. Co-activity hash, append-only and dedup tests.

Start small only to verify correctness, within your user's compute policy. Useful work follows the capacity they authorized.

## Artifacts the receiving agent must create (all private)

`persona_fleet.local.json`, `persona_explorer.local.json`, `persona_jobs.local.json`, `results/` (including `results/persona_map.json`, `results/layer_NN.npz`, SQLite databases and probe artifacts), the prepared package, the planner config/service unit, user service units, the TODO substitution list, and a written inventory (Phase 1).

## Final handoff checklist (give this to your user)

- [ ] Track (A or B), approved hosts/GPUs and admission system recorded
- [ ] Hardcoded-dependency inventory complete; each item ported, kept, or documented as unsupported
- [ ] Runtime commit and model/prepared/runtime identities recorded
- [ ] Gate results for Phases 2 to 8 with exact test output, and any failures stated plainly
- [ ] Explorer bound to loopback or reviewed authenticated access
- [ ] No private weights, calibration, corpora, results DBs or `*.local.json` committed
- [ ] Lyra credit and Flare runtime link retained in the fork's README
- [ ] Evidence statement says: presets are approximations; no personality fidelity, causal-effect or native-cache/MTP qualification is claimed unless separately earned
- [ ] Unresolved risks listed (unsupported GPU/OS, fleet size, unbuilt checkpoint atlas, unqualified token traces, production cache/MTP limits)

## Copyable kickoff instruction

```text
Read fork_agent.md in this repository and follow it phase by phase to port Character
Creation to <TARGET SYSTEM>. Target track: <A same checkpoint+fleet | B new model/fleet>.
Approved hosts/GPUs: <TODO>. Admission system: <TODO>. External compute/spending: not
authorized unless explicitly added above. Preserve existing workloads. Follow existing
compute authorization; ask only for missing scope, not for work already approved. Credit Lyra (lyramakesmusic/neuron-explorer) and the Flare runtime
(Atlas3DSS/flare-local-inference, branch atlas/sm70-public). Report each gate's actual
output, including failures, and finish with the handoff checklist.
```


### Porting the checkpoint atlas and token traces

The current static atlas and runtime are specific to the pinned Qwen3.5-architecture checkpoint sold as Qwen3.8-27B. Inspect the actual architecture, final-norm convention, quantization groups, head storage and dimensions before adapting it. persona/persona_atlas.py explicitly decodes NVFP4 and uses zero-centered RMSNorm; do not transfer those assumptions or stored token rankings to another model. Rebuild all six atlas shards under the destination's admission wrapper, publish the hash manifest, and regenerate map geometry from that destination's calibrated columns. Atlas search covers retained top-token lists, not every token ranking.

Keep persona/persona_probe.py, persona/persona_trace.py, persona/persona_map.py and persona/persona_vectors.py synchronized with the worker. The runtime extension adds fixed token_indices, token_enabled and token_values buffers; prefill and ordinary B1 decode must share the same storage. Select at most 32 neurons per layer and 64 per request. Stage indices between requests, never patch captured graph addresses or inject Python callbacks into the native serving loop.

Before enabling live tracing, repeat exact original calibration equivalence, baseline/zero output and activation equality, consumed-token alignment across a prefill chunk boundary, at least one decode step, both intervention signs and decode-only neutral next-token KL. Verify model/prepared/runtime identities on all endpoints and update the ignored local bindings only after qualification. Keep raw corpora, generated continuations, full logits and trace artifacts private. Search/playback must report missing evidence honestly.

The co-activity and weight-web canvases are 3D projections. Preserve the stated coordinate meanings, edge signs and evidence distinctions when porting the renderer. Scope canvas positioning to its own panel and retain clipping; the main map's absolute-position canvas style must not leak into dialogs or token heatmaps.
