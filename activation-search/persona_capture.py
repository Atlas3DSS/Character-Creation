"""Fresh-checkpoint calibration using Flare's retained production CUDA graphs.

This is an offline diagnostic, never a Python callback inside the native server.
Capture discovery prompts only. The output is representation evidence; causal
persona claims require subsequent controlled generation on heldout prompts.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import signal
import threading
import time

import numpy as np

from persona_vectors import ModelSpec, canonical_digest


def capture_plan(root: Path) -> dict:
    corpus = json.loads((root / "data/persona_prompts.json").read_text())
    model = ModelSpec.load(root / "data/persona_model.json")
    profiles = {"baseline": ""}
    for trait, poles in corpus["trait_prompts"].items():
        for pole, prompt in poles.items():
            profiles[f"{trait}_{pole}"] = prompt
    jobs = []
    for item in corpus["prompts"]:
        if item["split"] != "discovery":
            continue
        for condition, system in profiles.items():
            job = {"condition": condition, "system": system, "prompt_id": item["id"],
                   "prompt": item["text"], "split": "discovery", "thinking": False}
            job["id"] = canonical_digest(job)
            jobs.append(job)
    return {"schema": "character-creation.capture.v1", "model": model.__dict__,
            "model_identity": model.identity, "jobs": jobs,
            "measurement": "prompt-token layer-output L2 excluding position zero; final-prompt-token residual and normalized MLP input",
            "claim": "discovery-only representation evidence; no causal effect established"}


def shard_jobs(plan: dict, shard: int, shards: int = 6) -> list[dict]:
    if type(shard) is not int or type(shards) is not int or shards != 6 or not 0 <= shard < shards:
        raise ValueError("Capture uses exactly six deterministic shards")
    model = ModelSpec(**plan["model"])
    if model.identity != plan["model_identity"]:
        raise ValueError("Capture plan identity differs from its model")
    jobs = plan["jobs"]
    if not jobs or len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Capture jobs must be nonempty and unique")
    groups = {}
    expected = {"baseline"} | {trait+"_"+pole for trait in
        ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism") for pole in ("high", "low")}
    for job in jobs:
        if (job["id"] != canonical_digest({k: v for k, v in job.items() if k != "id"})
                or job["split"] != "discovery" or job["thinking"] is not False):
            raise ValueError("Capture job content or discovery split failed validation")
        group = groups.setdefault(job["prompt_id"], {})
        if job["condition"] in group:
            raise ValueError("Duplicate prompt condition")
        group[job["condition"]] = job
    if any(set(group) != expected or len({job["prompt"] for job in group.values()}) != 1 for group in groups.values()):
        raise ValueError("Every discovery prompt needs all eleven matched conditions")
    return sorted(plan["jobs"], key=lambda job: job["id"])[shard::shards]


def require_lease(config: dict, shard: int) -> dict:
    uuid = config["gpus"][shard]
    if socket.gethostname() != config["hostname"] or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise RuntimeError("Capture must run on its explicit host and physical GPU UUID")
    board = json.loads(subprocess.check_output([config["workboard"], "status", "--json"], text=True, timeout=60))
    leases = [job for job in board["jobs"] if job.get("pid") == os.getppid()
              and job.get("assigned_uuid") == uuid and job.get("status") == "running"]
    if len(leases) != 1:
        raise RuntimeError("Capture requires its own running gpu-workboard parent lease")
    return leases[0]


def tokenize_messages(tokenizer, messages: list[dict]) -> list[int]:
    """Normalize Transformers list/BatchEncoding returns before any GPU work."""
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False, return_dict=False)
    ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
    if (not isinstance(ids, list) or not ids
            or any(type(token) is not int or token < 0 for token in ids)):
        raise ValueError("Chat tokenizer must return one nonempty flat list of integer token IDs")
    return ids


def collect(prepared, jobs: list[dict], tokenized: list[list[int]], output: Path, manifest: dict) -> None:
    """Replay actual init/prefill graphs with normal state/admission validation."""
    import torch
    from ignition.components.sampling import INACTIVE, RUNNING, RequestAdmission
    from validate_candidate import _owners, _snapshot_mutable, _restore

    names, generation, maintenances = _owners(prepared, mode="chunked")
    maintenance = maintenances["init"]
    batch, decoder = generation.batch, generation.decoder
    requests, inputs, pool = batch.requests, batch.inputs, decoder.pool
    admission, commands = maintenance.state.admission.buffers, maintenance.commands
    capture = decoder.activation_capture
    if capture is None or bool((requests.status != INACTIVE).any()):
        raise RuntimeError("Capture needs a fresh diagnostic package with observation buffers")
    max_context = max(map(len, tokenized))
    page_tokens = int(prepared.server_config["page_tokens"])
    needed_pages = (max_context + page_tokens - 1) // page_tokens
    capacity = prepared.buffers["memory_capacity"].cpu().tolist()
    if capacity[0] < 3 or capacity[1] < needed_pages or needed_pages > inputs.page_tables.shape[1]:
        raise RuntimeError("Prepared physical pages do not cover calibration prompts")
    buffers = prepared.buffers
    sequence = int(buffers["submission_sequence"].item())
    snapshots = _snapshot_mutable(prepared, generation, maintenances)
    residuals, pre_mlp, mlp_inputs, norms, offsets, token_offsets = [], [], [], [], [0], [0]
    flat_tokens: list[int] = []
    manifest.update(graphs=names, prepared_identity=prepared.server_config.get("identity"),
                    state_restored=False, prompts_completed=0, status="capturing")

    def replay(kind: str, epoch: int, consumed: int) -> None:
        nonlocal sequence
        sequence += 1
        frame = sequence % 2
        buffers["frame_index"].fill_(frame)
        buffers["submission_sequence"].fill_(sequence)
        prepared.graphs[names[kind]].replay()
        torch.cuda.synchronize(prepared.device)
        record = buffers["completion_records"][frame, 0].cpu().tolist()
        if record[:7] != [epoch, sequence, consumed, 0, -1, RUNNING, 0]:
            raise RuntimeError(f"Captured {kind} rejected: {record}")
        observed = [int(getattr(requests, name)[0].item()) for name in
                    ("generation", "consumed_tokens", "generated_tokens", "pending_token", "status", "error_code", "rng_counter")]
        if observed != [epoch, consumed, 0, -1, RUNNING, 0, 0]:
            raise RuntimeError(f"Captured request state disagrees: {observed}")
        if kind == "init" and (bool(commands.errors.any()) or bool(maintenance.state.admission_errors.any())):
            raise RuntimeError("Calibration admission or transfer failed")

    try:
        with torch.inference_mode(), torch.cuda.device(prepared.device):
            for index, (job, ids) in enumerate(zip(jobs, tokenized, strict=True)):
                epoch = index + 1
                inputs.request_state_rows.zero_()
                inputs.request_state_rows[0, 0] = 1
                inputs.page_tables.fill_(-1)
                inputs.page_tables[0, :needed_pages].copy_(torch.arange(needed_pages, dtype=torch.int32))
                inputs.row_slot_ids.fill_(-1)
                inputs.row_slot_ids[0] = 0
                inputs.row_generations.zero_()
                inputs.row_generations[0] = epoch
                inputs.live_rows.fill_(1)
                inputs.query_lengths.zero_()
                inputs.query_lengths[0] = 1
                inputs.token_ids.zero_()
                inputs.sample_enabled.zero_()
                admission.stage(0, 0, RequestAdmission(generation=epoch, seed=0, max_new_tokens=1,
                    temperature=1., top_k=1, top_p=1., repetition_penalty=1., eos_token_ids=()), ids)
                admission.admission_count.fill_(1)
                commands.request_slots.fill_(-1)
                commands.source_slots.fill_(-1)
                commands.destination_slots.fill_(-1)
                commands.operation.zero_()
                commands.expected_consumed.fill_(-1)
                commands.request_slots[0] = 0
                commands.destination_slots[0] = 1
                commands.operation[0] = 1
                commands.expected_consumed[0] = 0
                commands.num_ops.fill_(1)
                replay("init", epoch, 0)
                admission.admission_count.zero_()
                commands.num_ops.zero_()
                prompt_norms = []
                for start in range(0, len(ids), batch.tokens):
                    stop = min(len(ids), start + batch.tokens)
                    inputs.query_lengths[0] = stop - start
                    inputs.token_ids[:stop-start].copy_(torch.tensor(ids[start:stop], dtype=torch.int64))
                    replay("prefill", epoch, stop)
                    positions = capture.positions.cpu().numpy().copy()
                    expected = np.arange(start, stop)
                    if not np.array_equal(positions[positions >= 0], expected):
                        raise RuntimeError("Capture positions do not match consumed prompt tokens")
                    values = capture.residual_l2.cpu().numpy().copy()[:, positions > 0]
                    if not np.isfinite(values).all() or np.any(values <= 0):
                        raise RuntimeError("Residual norm calibration is nonfinite or zero")
                    prompt_norms.append(values)
                residual = capture.residual_last.cpu().numpy().copy()
                pre = capture.pre_mlp_last.cpu().numpy().copy()
                product = capture.mlp_input_last.cpu().numpy().copy()
                if not np.isfinite(residual).all() or not np.isfinite(pre).all() or not np.isfinite(product).all():
                    raise RuntimeError("Captured representation is nonfinite")
                residuals.append(residual)
                pre_mlp.append(pre)
                mlp_inputs.append(product)
                sample = np.concatenate(prompt_norms, axis=1)
                norms.append(sample)
                offsets.append(offsets[-1] + sample.shape[1])
                flat_tokens.extend(ids)
                token_offsets.append(len(flat_tokens))
                manifest["prompts_completed"] = index + 1
                print(json.dumps({"captured": index + 1, "total": len(jobs), "id": job["id"], "tokens": len(ids)}), flush=True)
    finally:
        _restore(prepared, pool, snapshots)
        manifest["state_restored"] = True
    with output.open("xb") as destination:
        np.savez_compressed(destination, model_identity=manifest["model_identity"],
            plan_sha256=manifest["plan_sha256"], job_ids=np.array([job["id"] for job in jobs]),
            residual_last=np.stack(residuals), pre_mlp_last=np.stack(pre_mlp),
            mlp_input_last=np.stack(mlp_inputs),
            residual_l2=np.concatenate(norms, axis=1), norm_offsets=np.array(offsets, dtype=np.int64),
            token_ids=np.array(flat_tokens, dtype=np.int64), token_offsets=np.array(token_offsets, dtype=np.int64))
    manifest.update(status="complete", artifact_sha256=hashlib.file_digest(output.open("rb"), "sha256").hexdigest())


def run(args) -> None:
    config = json.loads(args.config.read_text())
    lease = require_lease(config, args.shard)
    plan = json.loads(args.plan.read_text())
    model = ModelSpec(**plan["model"])
    if model.identity != plan["model_identity"] or model.identity != config["model_identity"]:
        raise ValueError("Plan, calibration model and configured checkpoint must match")
    jobs = shard_jobs(plan, args.shard)
    output = args.output / f"capture-shard{args.shard}.npz"
    report = output.with_suffix(".json")
    previous = json.loads(report.read_text()) if report.exists() else None
    if output.exists() or (previous is not None and not args.resume):
        raise FileExistsError("Capture evidence already exists; do not overwrite it")
    if previous is not None and (previous.get("status") != "failed"
            or previous.get("plan_sha256") != canonical_digest(plan)
            or previous.get("model_identity") != model.identity or previous.get("shard") != args.shard):
        raise ValueError("Only the same failed, artifact-free capture shard can resume")
    import importlib.util
    spec = importlib.util.spec_from_file_location("flare_heliopause", args.heliopause)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    extension.flare_heliopause = extension
    sys.modules["flare_heliopause"] = extension
    sys.modules["flare_heliopause.flare_heliopause"] = extension
    sys.path[:0] = [str(args.flare), str(args.flare / "tools")]
    import torch
    from transformers import AutoTokenizer
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.compilation import _OPTIONS
    from ignition.preparation import compile_pool
    from ignition.recipes.qwen38_dense.bootstrap import PreparationConfig, prepare

    if tuple(torch.cuda.get_device_capability()) != (7, 0):
        raise RuntimeError("This calibration configuration targets the V100 SM70 recipe")
    torch.set_num_threads(len(os.sched_getaffinity(0)))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    tokenized = []
    for job in jobs:
        messages = ([{"role": "system", "content": job["system"]}] if job["system"] else [])
        messages.append({"role": "user", "content": job["prompt"]})
        ids = tokenize_messages(tokenizer, messages)
        if not 1 < len(ids) < args.context:
            raise ValueError("Prompt must fit the explicitly prepared context")
        tokenized.append(ids)
    manifest = {"schema": "character-creation.capture-result.v1", "status": "preparing",
        "started_at": time.time(), "model_identity": model.identity, "model": model.__dict__,
        "plan_sha256": canonical_digest(plan), "jobs": jobs, "lease": lease["id"],
        "gpu_uuid": config["gpus"][args.shard], "shard": args.shard, "shards": 6,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "torch": torch.__version__,
        "checkpoint_config_sha256": hashlib.file_digest((args.checkpoint / "config.json").open("rb"), "sha256").hexdigest(),
        "checkpoint_index_sha256": hashlib.file_digest((args.checkpoint / "model.safetensors.index.json").open("rb"), "sha256").hexdigest(),
        "script_sha256": hashlib.file_digest(Path(__file__).open("rb"), "sha256").hexdigest()}
    manifest["heliopause_sha256"] = hashlib.file_digest(args.heliopause.open("rb"), "sha256").hexdigest()
    manifest["qualification_sha256"] = hashlib.file_digest(Path(__file__).with_name("persona_qualify.py").open("rb"), "sha256").hexdigest()
    manifest["tokenizer_sha256"] = hashlib.file_digest((args.checkpoint / "tokenizer.json").open("rb"), "sha256").hexdigest()
    source_paths = ["ignition/recipes/qwen38_dense/bootstrap.py", "ignition/recipes/qwen38_dense/decoder.py",
                    "ignition/components/activation_capture.py", "ignition/components/steering.py", "tools/validate_candidate.py"]
    manifest["source_sha256"] = {name: hashlib.file_digest((args.flare / name).open("rb"), "sha256").hexdigest() for name in source_paths}
    _OPTIONS["compile_threads"] = args.compile_workers
    compile_pool._DISCRETE_WORKER_DEVICE_BYTES = 400 << 20
    headroom = max(0, torch.cuda.get_device_properties(0).total_memory - (31 << 30)) + (256 << 20)
    prep = PreparationConfig(context=args.context, max_active=4, prefill_chunk_size=128,
        compile_workers=args.compile_workers, snapshot_budget_bytes=172 << 20, headroom_bytes=headroom,
        preparation_pages=(args.context + 127) // 128, retain_graphs=True, capture_activations=True,
        steering_max_sites=64, mtp=True, speculative_widths=(1, 2, 3, 4, 5),
        kernel_cache=args.kernel_cache, cutlass_include=args.cutlass_include)
    manifest["preparation"] = {key: str(value) if isinstance(value, Path) else value for key, value in prep.__dict__.items()}
    if previous is not None:
        manifest["prior_attempts"] = [*previous.pop("prior_attempts", []), previous]
    with output.with_suffix(".attempts.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": "start", **manifest}) + "\n")
    monitor_stop = threading.Event()
    violations = []
    from persona_fleet import telemetry
    def monitor() -> None:
        with output.with_suffix(".telemetry.jsonl").open("a" if args.resume else "x") as stream:
            while not monitor_stop.is_set():
                try:
                    sample = telemetry(config["gpus"][args.shard])
                    stream.write(json.dumps(sample) + "\n")
                    stream.flush()
                    if (sample["temperature"] >= 88 or sample["memory_mib"] > 31*1024
                            or sample["available_ram_bytes"] < config["ram_floor_bytes"]):
                        raise RuntimeError("Capture exceeded its thermal, VRAM or host-RAM guard")
                except BaseException as error:
                    violations.append(repr(error))
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
                monitor_stop.wait(5)
    def interrupted(signum, frame):
        raise RuntimeError("Capture interrupted: " + (violations[-1] if violations else str(signum)))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        import pytest
        os.environ["FLARE_RUN_CUDA_TESTS"] = "1"
        test = args.flare / "tests/components/test_steering.py"
        class TestResults:
            def pytest_sessionfinish(self, session, exitstatus):
                reporter = session.config.pluginmanager.getplugin("terminalreporter")
                manifest["steering_component_counts"] = {key: len(reporter.stats.get(key, []))
                    for key in ("passed", "failed", "skipped", "error")}
        manifest["steering_component_test"] = pytest.main(["-q", "-p", "no:cacheprovider", str(test)], plugins=[TestResults()])
        if manifest["steering_component_test"] != 0:
            raise RuntimeError("Compiled/captured steering component qualification failed")
        if manifest["steering_component_counts"].get("skipped", 0):
            raise RuntimeError("GPU qualification cannot silently skip component checks")
        torch.cuda.empty_cache()
        with PackageStorage(device=0) as storage:
            prepared = prepare(args.checkpoint, prep, progress=lambda text: print(text, flush=True), package_storage=storage)
        with prepared:
            collect(prepared, jobs, tokenized, output, manifest)
            from validate_candidate import _owners
            from persona_qualify import qualify
            names, generation, maintenances = _owners(prepared, mode="chunked")
            manifest["qualification"] = qualify(prepared, generation, names, maintenances, tokenized[:2])
            if manifest["qualification"]["status"] != "passed" or not manifest["qualification"]["state_restored"]:
                raise RuntimeError("Whole-model captured steering qualification failed")
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        monitor_stop.set()
        thread.join(timeout=20)
        manifest["finished_at"] = time.time()
        with report.open("w" if previous is not None else "x") as stream:
            json.dump(manifest, stream, indent=2)
        with output.with_suffix(".attempts.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": "finish", **manifest}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--flare", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--heliopause", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--shard", type=int, choices=range(6), required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--compile-workers", type=int, required=True)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--kernel-cache", type=Path)
    p.add_argument("--cutlass-include", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        plan = capture_plan(args.root)
        with args.output.open("x") as stream:
            json.dump(plan, stream, indent=2)
        print(json.dumps({"jobs": len(plan["jobs"]), "shard_counts": [len(shard_jobs(plan, i)) for i in range(6)]}))
    else:
        run(args)


if __name__ == "__main__":
    main()
