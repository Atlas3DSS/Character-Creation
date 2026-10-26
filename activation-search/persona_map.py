"""Build calibrated vector banks and explicitly provisional neuron candidates.

Paired final-token SwiGLU observations and decoded down-column alignment are
separate discovery evidence. Neither establishes a causal personality effect.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct

import numpy as np

from persona_capture import shard_jobs
from persona_vectors import ModelSpec, canonical_digest


TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def e4m3(values: np.ndarray) -> np.ndarray:
    """Decode finite E4M3 bytes, including subnormals and the extended exponent."""
    raw = np.asarray(values)
    if raw.dtype != np.uint8:
        raise ValueError("E4M3 requires encoded uint8 bytes")
    if np.any((raw & 127) == 127):
        raise ValueError("NaN E4M3 code in checkpoint")
    exponent = (raw >> 3) & 15
    fraction = raw & 7
    positive = np.where(exponent == 0, fraction.astype(np.float32) * 2**-9,
                        (1 + fraction.astype(np.float32) / 8) * np.exp2(exponent.astype(np.float32) - 7))
    return np.where(raw & 128, -positive, positive).astype(np.float32)


def decode_columns(packed: np.ndarray, scales: np.ndarray, gain: float,
                   indices: np.ndarray) -> np.ndarray:
    """Decode checkpoint W_down columns; do not include the activation scale."""
    indices = np.asarray(indices)
    if (packed.ndim != 2 or packed.dtype != np.uint8 or not all(packed.shape)
            or packed.shape[1] % 8 or scales.dtype != np.uint8
            or scales.shape != (packed.shape[0], packed.shape[1] // 8)):
        raise ValueError("Unexpected NVFP4 matrix or group-16 scales")
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= packed.shape[1] * 2):
        raise ValueError("Column indices are outside the NVFP4 matrix")
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("Weight global scale must be finite and positive")
    code = (packed[:, indices // 2] >> ((indices % 2) * 4).astype(np.uint8)) & 15
    magnitudes = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    values = magnitudes[code & 7] * np.where(code & 8, -1., 1.).astype(np.float32)
    decoded_scales = e4m3(scales[:, indices // 16])
    if np.any(decoded_scales < 0):
        raise ValueError("Weight block scales cannot be negative")
    with np.errstate(over="ignore", invalid="ignore"):
        result = values.astype(np.float64) * decoded_scales * gain
    if not np.isfinite(result).all() or np.any(np.abs(result) > np.finfo(np.float32).max):
        raise ValueError("Decoded weight exceeds finite float32 range")
    return result.astype(np.float32)


class Checkpoint:
    """Bounded read-only native safetensors access without a GPU or torch import."""
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers: dict = {}

    def raw(self, key: str, expected_dtype: str) -> np.ndarray:
        filename = self.index[key]
        if not isinstance(filename, str) or Path(filename).name != filename or filename in (".", ".."):
            raise ValueError("Checkpoint index must name a direct child shard")
        logical = self.root / filename
        path = logical.resolve()
        # Standard HF snapshots store direct-child shard symlinks in the same
        # repository's blobs directory. Permit that specific cache layout only.
        blob_parent = self.root.parent.parent / "blobs" if self.root.parent.name == "snapshots" else None
        if path.parent != self.root and path.parent != blob_parent:
            raise ValueError("Checkpoint shard is outside this snapshot and its HF blob store")
        if path not in self.headers:
            with path.open("rb") as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    raise ValueError("Truncated safetensors prefix")
                size, = struct.unpack("<Q", prefix)
                if not 2 <= size <= 100_000_000:
                    raise ValueError("Invalid safetensors header length")
                header = json.loads(stream.read(size))
            self.headers[path] = (8 + size, header)
        start, header = self.headers[path]
        entry = header[key]
        dtype = {"U8": np.dtype("u1"), "F8_E4M3": np.dtype("u1"), "F32": np.dtype("<f4"), "BF16": np.dtype("<u2")}[expected_dtype]
        lower, upper = entry["data_offsets"]
        shape = tuple(entry["shape"])
        if (any(type(d) is not int or d < 0 for d in shape)
                or type(lower) is not int or type(upper) is not int
                or entry["dtype"] != expected_dtype or lower < 0 or upper < lower
                or upper - lower != math.prod(shape) * dtype.itemsize or start + upper > path.stat().st_size):
            raise ValueError("Checkpoint tensor metadata does not match its storage")
        return np.memmap(path, mode="r", dtype=dtype, shape=shape, offset=start + lower)

    def down(self, layer: int):
        prefix = f"model.language_model.layers.{layer}.mlp.down_proj"
        gain = self.raw(prefix + ".weight_scale_2", "F32")
        if gain.size != 1:
            raise ValueError("Weight global scale must contain one scalar")
        return (self.raw(prefix + ".weight", "U8"), self.raw(prefix + ".weight_scale", "F8_E4M3"),
                float(gain.reshape(-1)[0]))


def aggregate(plan: dict, root: Path) -> dict:
    """Validate all six immutable captures before fitting any direction."""
    model = ModelSpec(**plan["model"])
    if plan["model_identity"] != model.identity:
        raise ValueError("Capture plan model identity is inconsistent")
    jobs = {job["id"]: job for job in plan["jobs"]}
    if len(jobs) != len(plan["jobs"]) or any(job["split"] != "discovery" for job in jobs.values()):
        raise ValueError("Capture must contain unique discovery jobs only")
    captured, captured_neurons, baseline_norms, source_hashes = {}, {}, [], []
    checkpoint_ids, runtime_ids = set(), set()
    for shard in range(6):
        path = root / f"capture-shard{shard}.npz"
        report = json.loads(path.with_suffix(".json").read_text())
        identity = file_sha256(path)
        expected = shard_jobs(plan, shard)
        if (report.get("status") != "complete" or report.get("state_restored") is not True
                or report.get("model_identity") != model.identity or report.get("plan_sha256") != canonical_digest(plan)
                or report.get("model") != plan["model"]
                or report.get("shard") != shard or report.get("shards") != 6
                or report.get("jobs") != expected or report.get("prompts_completed") != len(expected)
                or report.get("artifact_sha256") != identity):
            raise ValueError(f"Shard {shard} failed provenance or completeness validation")
        hash_fields = ("checkpoint_config_sha256", "checkpoint_index_sha256", "script_sha256",
                       "heliopause_sha256", "tokenizer_sha256")
        if any(not isinstance(report.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", report[key]) for key in hash_fields):
            raise ValueError("Capture lacks valid checkpoint/runtime/tokenizer hashes")
        sources = report.get("source_sha256")
        if not isinstance(sources, dict) or not sources or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in sources.values()):
            raise ValueError("Capture lacks its runtime source hashes")
        preparation = report.get("preparation")
        if not isinstance(preparation, dict) or preparation.get("capture_activations") is not True:
            raise ValueError("Capture preparation must explicitly enable observations")
        resource_options = {"compile_workers", "kernel_cache", "cutlass_include", "headroom_bytes", "snapshot_budget_bytes"}
        runtime = {key: report.get(key) for key in ("script_sha256", "heliopause_sha256", "tokenizer_sha256", "torch", "source_sha256")}
        runtime["preparation"] = {k: v for k, v in preparation.items() if k not in resource_options}
        runtime_ids.add(canonical_digest(runtime))
        checkpoint_ids.add((report["checkpoint_config_sha256"], report["checkpoint_index_sha256"]))
        with np.load(path, allow_pickle=False) as data:
            ids = data["job_ids"].tolist()
            if (str(data["model_identity"]) != model.identity or str(data["plan_sha256"]) != canonical_digest(plan)
                    or ids != [job["id"] for job in expected]):
                raise ValueError("Capture arrays do not match their plan")
            residuals, norms, offsets = data["residual_last"], data["residual_l2"], data["norm_offsets"]
            pre, tokens, token_offsets = data["pre_mlp_last"], data["token_ids"], data["token_offsets"]
            products = data["mlp_input_last"]
            if (residuals.shape != (len(ids), model.layers, model.hidden) or norms.ndim != 2
                    or products.shape != (len(ids), model.layers, model.intermediate) or not np.isfinite(products).all()
                    or norms.shape[0] != model.layers or offsets.shape != (len(ids)+1,)
                    or offsets.dtype.kind not in "iu" or pre.shape != residuals.shape or not np.isfinite(pre).all()
                    or tokens.ndim != 1 or tokens.dtype.kind not in "iu" or np.any(tokens < 0)
                    or token_offsets.shape != offsets.shape or token_offsets.dtype.kind not in "iu"
                    or token_offsets[0] != 0 or token_offsets[-1] != len(tokens)
                    or np.any(token_offsets[1:] <= token_offsets[:-1])
                    or np.any(np.diff(token_offsets) <= 1)
                    or offsets[0] != 0 or offsets[-1] != norms.shape[1] or np.any(offsets[1:] <= offsets[:-1])
                    or not np.array_equal(np.diff(offsets), np.diff(token_offsets) - 1)
                    or not np.isfinite(residuals).all() or not np.isfinite(norms).all() or np.any(norms <= 0)):
                raise ValueError("Capture tensor dimensions or finite-value checks failed")
            for i, identity in enumerate(ids):
                captured[identity] = residuals[i].astype(np.float64)
                captured_neurons[identity] = products[i].copy()
                if jobs[identity]["condition"] == "baseline":
                    baseline_norms.append(norms[:, offsets[i]:offsets[i+1]].copy())
        source_hashes.append(report["artifact_sha256"])
    if len(checkpoint_ids) != 1 or len(runtime_ids) != 1 or set(captured) != set(jobs) or not baseline_norms:
        raise ValueError("Capture coverage or checkpoint provenance differs between shards")
    medians = np.median(np.concatenate(baseline_norms, axis=1), axis=1).astype(np.float32)
    by_condition = {}
    for identity, job in jobs.items():
        group = by_condition.setdefault(job["condition"], {})
        if job["prompt_id"] in group:
            raise ValueError("Duplicate prompt within a contrast condition")
        group[job["prompt_id"]] = identity
    baseline = by_condition["baseline"]
    directions, neuron_contrasts, pair_counts = {}, {}, {}
    for trait in TRAITS:
        high, low = by_condition[trait+"_high"], by_condition[trait+"_low"]
        if set(high) != set(low) or set(high) != set(baseline):
            raise ValueError("Trait contrasts require all paired baseline prompts")
        differences = []
        neuron_differences = []
        for prompt in sorted(high):
            if jobs[high[prompt]]["prompt"] != jobs[low[prompt]]["prompt"]:
                raise ValueError("High and low conditions must use identical user prompts")
            differences.append(captured[high[prompt]] - captured[low[prompt]])
            neuron_differences.append(captured_neurons[high[prompt]].astype(np.float64) - captured_neurons[low[prompt]])
        directions[trait] = np.mean(differences, axis=0).astype(np.float32)
        neuron_contrasts[trait] = np.mean(neuron_differences, axis=0).astype(np.float32)
        if not np.isfinite(directions[trait]).all() or np.any(np.linalg.norm(directions[trait].astype(np.float64), axis=1) <= 0):
            raise ValueError("A fitted trait direction is nonfinite or zero")
        pair_counts[trait] = len(differences)
    config_hash, index_hash = checkpoint_ids.pop()
    baseline_products = np.stack([captured_neurons[baseline[prompt]] for prompt in sorted(baseline)]).astype(np.float64)
    return {"model_identity": model.identity, "medians": medians, "directions": directions,
            "neuron_baseline_mean": baseline_products.mean(axis=0).astype(np.float32),
            "neuron_baseline_sd": baseline_products.std(axis=0, ddof=1 if len(baseline)>1 else 0).astype(np.float32),
            "neuron_contrasts": neuron_contrasts,
            "pair_counts": pair_counts, "source_hashes": source_hashes,
            "runtime_identity": runtime_ids.pop(),
            "checkpoint_config_sha256": config_hash, "checkpoint_index_sha256": index_hash}


def build_layers(args) -> None:
    plan = json.loads(args.plan.read_text())
    model = ModelSpec(**plan["model"])
    fit = aggregate(plan, args.results)
    for filename, key in (("config.json", "checkpoint_config_sha256"), ("model.safetensors.index.json", "checkpoint_index_sha256")):
        if file_sha256(args.checkpoint / filename) != fit[key]:
            raise ValueError("Column extraction checkpoint differs from the captures")
    checkpoint = Checkpoint(args.checkpoint)
    nodes, banks = [], []
    for layer in range(args.shard, model.layers, 6):
        path = args.results / f"layer_{layer:02d}.npz"
        if path.exists():
            raise FileExistsError(f"Vector bank already exists: {path}")
        packed, scales, gain = checkpoint.down(layer)
        if packed.shape != (model.hidden, model.intermediate // 2):
            raise ValueError("Down projection differs from model dimensions")
        directions = np.stack([fit["directions"][trait][layer] for trait in TRAITS]).astype(np.float64)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        scores = np.empty((len(TRAITS), model.intermediate), dtype=np.float32)
        for start in range(0, model.intermediate, 512):
            indices = np.arange(start, min(model.intermediate, start+512))
            columns = decode_columns(packed, scales, gain, indices).astype(np.float64)
            norm = np.linalg.norm(columns, axis=0)
            if np.any(norm <= 0) or not np.isfinite(norm).all() or not np.isfinite(columns).all():
                raise ValueError("Invalid down-column direction")
            scores[:, indices] = directions @ (columns / norm)
        selected = set()
        for i, trait in enumerate(TRAITS):
            for pole, sign in (("high", 1), ("low", -1)):
                selected.update(np.argsort(sign * scores[i], kind="stable")[-args.per_pole:].tolist())
                selected.update(np.argsort(sign * fit["neuron_contrasts"][trait][layer], kind="stable")[-args.per_pole:].tolist())
        ids = np.array(sorted(selected), dtype=np.int64)
        columns = decode_columns(packed, scales, gain, ids)
        with path.open("xb") as stream:
            np.savez_compressed(stream, model_identity=model.identity, residual_median=fit["medians"][layer],
                plan_sha256=canonical_digest(plan), runtime_identity=fit["runtime_identity"],
                checkpoint_config_sha256=fit["checkpoint_config_sha256"],
                checkpoint_index_sha256=fit["checkpoint_index_sha256"],
                source_hashes=np.array(fit["source_hashes"]),
                neuron_ids=ids, down_columns=columns, trait_alignment=scores[:, ids],
                activation_mean=fit["neuron_baseline_mean"][layer, ids],
                activation_sd=fit["neuron_baseline_sd"][layer, ids],
                activation_contrasts=np.stack([fit["neuron_contrasts"][trait][layer, ids] for trait in TRAITS]),
                **{"trait_"+trait: fit["directions"][trait][layer] for trait in TRAITS})
        banks.append({"layer": layer, "filename": path.name, "sha256": file_sha256(path)})
        for index, neuron in enumerate(ids):
            nodes.append({"layer": layer, "neuron": int(neuron), "status": "candidate",
                "measurement": "cosine of decoded down-column and paired residual contrast",
                "trait_alignment": {trait: float(scores[i, neuron]) for i, trait in enumerate(TRAITS)},
                "activation_observation": {"phase": "final prompt token", "sample": "discovery prompts only",
                    "baseline_mean": float(fit["neuron_baseline_mean"][layer, neuron]),
                    "baseline_sd": float(fit["neuron_baseline_sd"][layer, neuron]),
                    "high_minus_low": {trait: float(fit["neuron_contrasts"][trait][layer, neuron]) for trait in TRAITS}},
                "causal_effect": None, "layout": [2*(int(neuron)%136)/135-1, 2*(int(neuron)//136)/127-1]})
        print(json.dumps({"layer": layer, "candidate_neurons": len(ids), "residual_median": float(fit["medians"][layer])}), flush=True)
    report = {"schema": "character-creation.map-shard.v1", "model_identity": model.identity, "model": model.__dict__,
              "plan_sha256": canonical_digest(plan), "source_hashes": fit["source_hashes"],
              "runtime_identity": fit["runtime_identity"], "banks": banks,
              "pair_counts": fit["pair_counts"], "shard": args.shard, "neurons": nodes}
    with (args.results / f"map-shard{args.shard}.json").open("x") as stream:
        json.dump(report, stream, indent=2)


def assemble(root: Path) -> None:
    shards = [json.loads((root / f"map-shard{i}.json").read_text()) for i in range(6)]
    model = ModelSpec(**shards[0]["model"])
    identities = {(s["model_identity"], s["plan_sha256"], tuple(s["source_hashes"]), s["runtime_identity"]) for s in shards}
    if len(identities) != 1 or any(s["shard"] != i for i, s in enumerate(shards)):
        raise ValueError("Map shard provenance disagrees")
    if any(s["model"] != model.__dict__ or s["model_identity"] != model.identity or s["pair_counts"] != shards[0]["pair_counts"] for s in shards):
        raise ValueError("Map model or paired-prompt coverage disagrees")
    nodes = [node for shard in shards for node in shard["neurons"]]
    if len({(n["layer"], n["neuron"]) for n in nodes}) != len(nodes):
        raise ValueError("Duplicate map coordinates")
    layer_coverage = set()
    for shard in shards:
        expected_layers = set(range(shard["shard"], model.layers, 6))
        banks = shard["banks"]
        if {b["layer"] for b in banks} != expected_layers or len(banks) != len(expected_layers):
            raise ValueError("Map shard omits or duplicates an expected layer bank")
        shard_nodes = defaultdict(list)
        for node in shard["neurons"]:
            layer, neuron = node["layer"], node["neuron"]
            if type(layer) is not int or layer not in expected_layers or type(neuron) is not int or not 0 <= neuron < model.intermediate:
                raise ValueError("Map coordinate is outside its model or shard")
            if node.get("status") != "candidate" or node.get("causal_effect") is not None:
                raise ValueError("Discovery maps cannot contain validated causal claims")
            values = node.get("trait_alignment", {})
            if set(values) != set(TRAITS) or any(not math.isfinite(v) or abs(v) > 1.00001 for v in values.values()):
                raise ValueError("Invalid candidate alignment score")
            observation = node.get("activation_observation", {})
            contrasts = observation.get("high_minus_low", {})
            if (observation.get("phase") != "final prompt token"
                    or observation.get("sample") != "discovery prompts only"
                    or set(contrasts) != set(TRAITS)
                    or any(type(v) not in (int, float) or not math.isfinite(v)
                           for v in [observation.get("baseline_mean"), observation.get("baseline_sd"), *contrasts.values()])
                    or observation["baseline_sd"] < 0):
                raise ValueError("Invalid discovery activation observation")
            shard_nodes[layer].append(node)
        for bank in banks:
            layer = bank["layer"]
            if bank["filename"] != f"layer_{layer:02d}.npz":
                raise ValueError("Unexpected bank filename")
            path = root / bank["filename"]
            if file_sha256(path) != bank["sha256"]:
                raise ValueError("Vector bank digest differs from the map shard")
            with np.load(path, allow_pickle=False) as data:
                if (str(data["model_identity"]) != model.identity or str(data["plan_sha256"]) != shard["plan_sha256"]
                        or str(data["runtime_identity"]) != shard["runtime_identity"]
                        or data["source_hashes"].tolist() != shard["source_hashes"]):
                    raise ValueError("Vector bank provenance differs from the map")
                raw_ids = data["neuron_ids"]
                ids = raw_ids.tolist()
                if (raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu" or len(set(ids)) != len(ids)
                        or set(ids) != {n["neuron"] for n in shard_nodes[layer]} or not ids):
                    raise ValueError("Map candidates differ from the saved vector bank")
                columns = data["down_columns"]
                if columns.shape != (model.hidden, len(ids)) or not np.isfinite(columns).all():
                    raise ValueError("Saved candidate columns are invalid")
                alignment = data["trait_alignment"]
                if alignment.shape != (len(TRAITS), len(ids)) or not np.isfinite(alignment).all():
                    raise ValueError("Saved candidate alignment is invalid")
                means, deviations, contrasts = data["activation_mean"], data["activation_sd"], data["activation_contrasts"]
                if (means.shape != (len(ids),) or deviations.shape != means.shape
                        or contrasts.shape != alignment.shape or not np.isfinite(means).all()
                        or not np.isfinite(deviations).all() or np.any(deviations < 0)
                        or not np.isfinite(contrasts).all()):
                    raise ValueError("Saved candidate activation observations are invalid")
                by_id = {node["neuron"]: node for node in shard_nodes[layer]}
                for index, neuron in enumerate(ids):
                    if any(float(alignment[t, index]) != by_id[neuron]["trait_alignment"][trait] for t, trait in enumerate(TRAITS)):
                        raise ValueError("Displayed candidate scores differ from the vector bank")
                    observed = by_id[neuron]["activation_observation"]
                    if (float(means[index]) != observed["baseline_mean"]
                            or float(deviations[index]) != observed["baseline_sd"]
                            or any(float(contrasts[t, index]) != observed["high_minus_low"][trait]
                                   for t, trait in enumerate(TRAITS))):
                        raise ValueError("Displayed activation observations differ from the vector bank")
            layer_coverage.add(layer)
    if layer_coverage != set(range(model.layers)):
        raise ValueError("Map layer coverage is incomplete")
    report = {"schema": "character-creation.map.v1", "model_identity": shards[0]["model_identity"],
        "status": "discovery candidates; causal validation pending", "coverage_label": "candidate neurons",
        "layout_method": "neuron index grid, not a connectome or causal graph",
        "fit": "paired final-prompt-token residual and SwiGLU contrasts; baseline SwiGLU mean/sample SD; baseline-only token-weighted residual norm medians",
        "plan_sha256": shards[0]["plan_sha256"], "source_hashes": shards[0]["source_hashes"],
        "runtime_identity": shards[0]["runtime_identity"], "model": model.__dict__,
        "pair_counts": shards[0]["pair_counts"], "neurons": nodes,
        "banks": sorted([bank for shard in shards for bank in shard["banks"]], key=lambda item: item["layer"]),
        "attribution": "Signed normalized intervention method inspired by Lyra: https://github.com/lyramakesmusic/neuron-explorer"}
    with (root / "persona_map.json").open("x") as stream:
        json.dump(report, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("layers")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--shard", type=int, choices=range(6), required=True)
    p.add_argument("--per-pole", type=int, choices=range(1, 17), default=2)
    p = sub.add_parser("assemble")
    p.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    build_layers(args) if args.command == "layers" else assemble(args.results)


if __name__ == "__main__":
    main()
