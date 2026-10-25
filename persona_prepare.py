"""Prepare one native steering serving package through every existing Flare export gate.

Run under a gpu-workboard parent lease. No diagnostic capture is enabled. The
upstream exporter owns model preparation/signing; this wrapper adds immutable
provenance, neutral alignment, explicit extension loading and external staging.
The plan command only prints the concrete plan and never initializes CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time
from typing import Any

from persona_vectors import ModelSpec


GATES = ("validate", "check_invariance", "check_commit_growth", "check_consumers",
         "measure_components", "measure_accuracy")
OUTPUTS = ("model.safetensors", "prepare.json", "persona_prepare.json", "tau.json",
           "accuracy.json", "accuracy.safetensors", "consumer-identity.json", "validation-attempts.jsonl")
SOURCES = ("tools/prepare_replay.py", "tools/accuracy.py", "tools/validate_replay.py",
           "tools/check_consumers.py", "tools/measure_mtp.py", "tools/measure_components.py",
           "ignition/components/steering.py", "ignition/recipes/qwen38_dense/bootstrap.py",
           "ignition/recipes/qwen38_dense/decoder.py", "ignition/recipes/qwen38_dense/contract.py",
           "ignition/recipes/qwen38_dense/speculative.py", "ignition/preparation/runtime_package_pb2.py")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_environment(sysroot: Path, current: dict[str, str]) -> dict[str, str]:
    """pkg-config availability and its resolved uring flags participate in b12x's cache key."""
    result = dict(current)
    library = str(sysroot / "usr/lib/x86_64-linux-gnu")
    for key, prefixes in {"PATH": [str(sysroot / "usr/bin")],
            "PKG_CONFIG_PATH": [library + "/pkgconfig"],
            "LD_LIBRARY_PATH": [library, "/usr/local/cuda/lib64"]}.items():
        result[key] = ":".join(dict.fromkeys(prefixes + [p for p in current.get(key, "").split(":") if p]))
    result["PKG_CONFIG_SYSROOT_DIR"] = str(sysroot)
    return result


def export_arguments(args: argparse.Namespace, baseline: dict[str, Any]) -> list[str]:
    """Reuse the known catalog and fixtures; there is no switch to omit an export gate."""
    old = baseline["args"]
    if (baseline.get("status") != "exported" or not all(old.get(gate) is True for gate in GATES)
            or old.get("skip_reference_gates") or old.get("diagnostic_uncompiled")
            or old.get("capture_activations") or old.get("legacy_dense")
            or old.get("depth") != 4 or old.get("concurrency") != 4):
        raise ValueError("Baseline must be a complete, compiled B4/depth4 release export with every gate")
    if type(args.steering_max_sites) is not int or not 1 <= args.steering_max_sites <= 192:
        raise ValueError("Steering max sites must fit the pinned 192-site model")
    if args.compile_workers < 1 or not 1 <= args.lease_gib <= 31:
        raise ValueError("Explicit positive compiler workers and a 1..31 GiB lease budget are required")
    result = [str(args.flare / "tools/prepare_replay.py")]
    values = {"root": args.output, "checkpoint": args.checkpoint, "prompts": old["prompts"],
        "short-reference": old["short_reference"], "long-reference": old["long_reference"],
        "cutlass-include": old["cutlass_include"], "kernel-cache": args.kernel_cache,
        "signing-dir": args.signing_dir, "context": old["context"], "concurrency": old["concurrency"],
        "chunk": old["chunk"], "depth": old["depth"], "pages": old["pages"],
        "compile-workers": args.compile_workers, "compiler-context-mib": old.get("compiler_context_mib", 400),
        "lease-gib": args.lease_gib, "steering-max-sites": args.steering_max_sites,
        "measure": args.output / "tau.json", "output": args.output / "model.safetensors",
        "report": args.output / "prepare.json"}
    for key, value in values.items():
        result.extend(("--" + key, str(value)))
    result.extend("--" + gate.replace("_", "-") for gate in GATES)
    result.extend(("--measure-batches", "1", "4", "--measure-prompt-ids", "code-00", "code-01", "code-02", "code-03"))
    return result


def compare_measurements(baseline: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    fields = ("depth", "batch", "repetition", "prompt_ids", "output_token_ids")
    exact = (baseline.get("status") == actual.get("status") == "complete"
        and baseline["prompt_file_sha256"] == actual["prompt_file_sha256"]
        and baseline["summary"] == actual["summary"]
        and [[run[key] for key in fields] for run in baseline["runs"]]
            == [[run[key] for key in fields] for run in actual["runs"]])
    if not exact:
        raise RuntimeError("Neutral steering package changed baseline coding tokens or acceptance/tau goldens")
    return {"bit_exact_tokens_and_tau": True, "runs": len(actual["runs"])}


def compare_logits(baseline: Path, actual: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    rows = []
    with safe_open(baseline, framework="pt", device="cpu") as old, safe_open(actual, framework="pt", device="cpu") as new:
        keys = sorted(old.keys())
        if keys != sorted(new.keys()) or old.metadata() != new.metadata():
            raise RuntimeError("Baseline accuracy artifact keys or model identity changed")
        for key in keys:
            if not key.endswith((".candidate_logits", ".input_ids", ".positions")):
                continue
            a, b = old.get_tensor(key), new.get_tensor(key)
            if (a.dtype != b.dtype or a.shape != b.shape
                    or not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))):
                raise RuntimeError("Neutral steering package differs from baseline accuracy artifact: " + key)
            rows.append({"key": key, "elements": a.numel()})
    if not any(row["key"].endswith(".candidate_logits") for row in rows):
        raise ValueError("Baseline has no saved full-vocabulary candidate logits")
    return {"bit_exact": True, "compared": rows, "baseline_sha256": sha256(baseline), "actual_sha256": sha256(actual)}


def plan(args: argparse.Namespace) -> dict[str, Any]:
    baseline = json.loads(args.baseline_report.read_text())
    model = ModelSpec.load(args.model)
    config = json.loads(args.config.read_text())
    if config["model_identity"] != model.identity:
        raise ValueError("Fleet configuration and model identity disagree")
    if (model.model, model.revision, model.layers, model.hidden, model.intermediate) != (
            "nvidia/Qwen3.8-27B-NVFP4", "482ca0f3832238542f8f5295dde86b5f22711d80", 64, 5120, 17408):
        raise ValueError("This export harness is pinned to the reviewed Qwen checkpoint")
    command = export_arguments(args, baseline)
    for directory in (args.output, args.staging, args.payload_cache, args.payload_cache / "blobs",
                      args.kernel_cache, args.signing_dir, args.sysroot):
        if not directory.is_dir():
            raise ValueError(f"Reuse an existing directory: {directory}")
    project = args.output.resolve().parent
    if args.output.resolve().name != "results":
        raise ValueError("Serving artifacts belong in the existing flat results directory")
    if args.staging.resolve().is_relative_to(project) or args.payload_cache.resolve().is_relative_to(project):
        raise ValueError("Exporter temporary directories and blob cache must be outside Character Creation")
    for name in OUTPUTS:
        if (args.output / name).exists():
            raise FileExistsError(f"Preserve existing evidence: {args.output / name}")
    old = baseline["args"]
    if not Path(old["cutlass_include"]).is_dir():
        raise FileNotFoundError(old["cutlass_include"])
    files = [args.heliopause, args.baseline_report.parent / "accuracy.safetensors",
             args.baseline_report.parent / "tau.json", args.signing_dir / "development-signing-key.pem",
             args.signing_dir / "development-publisher.json", args.sysroot / "usr/bin/pkg-config",
             Path(old["prompts"]), Path(old["short_reference"]), Path(old["long_reference"])]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(Path(baseline["package"])) != baseline["package_primary_sha256"]:
        raise ValueError("Baseline package primary differs from its preparation report")
    for reference in (Path(old["short_reference"]), Path(old["long_reference"])):
        provenance = json.loads(reference.with_suffix(".json").read_text())
        if (provenance["model_id"], provenance["revision"]) != (model.model, model.revision):
            raise ValueError("Reference checkpoint identity disagrees")
        for filename, key in (("config.json", "config_sha256"), ("model.safetensors.index.json", "index_sha256")):
            if sha256(args.checkpoint / filename) != provenance[key]:
                raise ValueError("Checkpoint differs from reference provenance: " + filename)
    return {"schema": "character-creation.prepare.v1", "status": "planned", "model": model.__dict__,
        "model_identity": model.identity, "upstream_argv": command,
        "staging": str(args.staging), "payload_cache": str(args.payload_cache),
        "baseline_report_sha256": sha256(args.baseline_report),
        "baseline_package_primary_sha256": baseline["package_primary_sha256"],
        "heliopause_sha256": sha256(args.heliopause), "script_sha256": sha256(Path(__file__)),
        "source_sha256": {path: sha256(args.flare / path) for path in SOURCES},
        "capture_activations": False, "native_serving_qualified": False,
        "reference_accuracy_policy": "measure against published FP32 references; no invented tolerance",
        "alignment_policy": "exact saved baseline logits/positions/tokens plus coding token/tau goldens"}


def run(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    environment = runtime_environment(args.sysroot, dict(os.environ))
    if any(os.environ.get(key) != environment[key] for key in ("PATH", "PKG_CONFIG_PATH", "PKG_CONFIG_SYSROOT_DIR", "LD_LIBRARY_PATH")):
        # Dynamic library lookup reads LD_LIBRARY_PATH at process startup.
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)
    from persona_capture import require_lease
    config = json.loads(args.config.read_text())
    lease = require_lease(config, args.gpu_slot)
    if lease["requested_mib"] < args.lease_gib * 1024:
        raise ValueError("Preparation budget exceeds its registered GPU lease")
    spec = importlib.util.spec_from_file_location("flare_heliopause", args.heliopause)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    extension.flare_heliopause = extension
    sys.modules["flare_heliopause"] = extension
    sys.modules["flare_heliopause.flare_heliopause"] = extension
    sys.path[:0] = [str(args.flare), str(args.flare / "tools")]
    import torch
    import accuracy
    import measure_mtp
    from ignition.preparation import model_package
    from ignition.preparation.package_snapshot import export_options
    torch.set_num_threads(len(os.sched_getaffinity(0)))
    original_accuracy, original_measure, original_export = accuracy.measure, measure_mtp.measure, model_package.export_package
    manifest.update(status="preparing", started_at=time.time(), lease=lease["id"],
        gpu_uuid=lease["assigned_uuid"], cpu_affinity=sorted(os.sched_getaffinity(0)), neutral_alignment={})

    def measured_accuracy(*positional: Any, **keywords: Any) -> dict[str, Any]:
        result = original_accuracy(*positional, **keywords)
        manifest["neutral_alignment"]["accuracy_logits"] = compare_logits(
            args.baseline_report.parent / "accuracy.safetensors", args.output / "accuracy.safetensors")
        return result

    def measured_tokens(*positional: Any, **keywords: Any) -> dict[str, Any]:
        result = original_measure(*positional, **keywords)
        manifest["neutral_alignment"]["coding_tokens_tau"] = compare_measurements(
            json.loads((args.baseline_report.parent / "tau.json").read_text()), result)
        return result

    def exported(*positional: Any, **keywords: Any) -> Any:
        if set(manifest["neutral_alignment"]) != {"accuracy_logits", "coding_tokens_tau"}:
            raise RuntimeError("Both neutral alignment gates must finish before package export")
        keywords["staging"] = args.staging
        with export_options(args.payload_cache):
            return original_export(*positional, **keywords)

    accuracy.measure, measure_mtp.measure, model_package.export_package = measured_accuracy, measured_tokens, exported
    previous_argv = sys.argv
    try:
        sys.argv = manifest["upstream_argv"]
        runpy.run_path(sys.argv[0], run_name="__main__")
        report = json.loads((args.output / "prepare.json").read_text())
        expected = {"max_sites": args.steering_max_sites, "sites": [
            {"name": f"layers.{layer}.{kind}", "width": width}
            for layer in range(64)
            for kind, width in (("mlp_out", 5120), ("residual", 5120), ("mlp_input", 17408))]}
        if report.get("status") != "exported" or report.get("steering") != expected:
            raise RuntimeError("Upstream did not export a steering-capable package")
        manifest.update(status="exported", prepared_identity=report["prepared_identity"],
            steering=report["steering"], package_primary_sha256=report["package_primary_sha256"],
            fingerprint=report["fingerprint"], prepare_report_sha256=sha256(args.output / "prepare.json"))
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        sys.argv = previous_argv
        accuracy.measure, measure_mtp.measure, model_package.export_package = original_accuracy, original_measure, original_export
        manifest["finished_at"] = time.time()
        with (args.output / "persona_prepare.json").open("x") as stream:
            json.dump(manifest, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    for name in ("config", "flare", "heliopause", "checkpoint", "baseline-report", "staging", "payload-cache", "kernel-cache", "signing-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "data/persona_model.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--sysroot", type=Path, default=Path("/mnt/storage16tb/flare/sysroot"))
    parser.add_argument("--gpu-slot", type=int, choices=range(6), default=0)
    parser.add_argument("--compile-workers", type=int, required=True)
    parser.add_argument("--steering-max-sites", type=int, default=64)
    parser.add_argument("--lease-gib", type=int, default=30)
    args = parser.parse_args()
    manifest = plan(args)
    if args.command == "plan":
        print(json.dumps(manifest, indent=2))
    else:
        run(args, manifest)


if __name__ == "__main__":
    main()
