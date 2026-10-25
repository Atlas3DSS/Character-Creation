"""Whole-model additive-steering checks using retained production CUDA graphs.

Call only on a fresh/restored diagnostic PreparedModel, after calibration has
saved its baseline observations. This function launches no service, exports no
package, and writes no files. Its bounded JSON report tests arithmetic and graph
plumbing, not personality, neuron semantics, or native-server cache isolation.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np


def round_bf16(value: np.ndarray) -> np.ndarray:
    """Independent FP32 -> nearest-even BF16 -> FP32 reference, including NaNs."""
    value = np.asarray(value)
    if value.dtype != np.float32:
        raise TypeError("BF16 reference requires float32 storage")
    bits = value.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    nan = (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    rounded = np.where(nan, (bits & np.uint32(0xFFFF0000)) | np.uint32(0x00400000), rounded)
    return rounded.astype(np.uint32, copy=False).view(np.float32)


def _array(value) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("Qualification observations must be finite float32 arrays")
    return np.ascontiguousarray(value)


def _hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def compare(actual, expected) -> dict:
    """Bit comparison plus descriptive errors; no arbitrary error tolerance."""
    actual, expected = _array(actual), _array(expected)
    if actual.shape != expected.shape:
        raise ValueError("Compared observations have different shapes")
    differences = actual.astype(np.float64) - expected.astype(np.float64)
    changed = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
    return {"elements": int(actual.size), "changed_bits_elements": changed,
            "bit_exact": changed == 0, "max_abs_error": float(np.max(np.abs(differences), initial=0)),
            "l2_error": float(np.linalg.norm(differences.reshape(-1))),
            "actual_sha256": _hash(actual), "expected_sha256": _hash(expected)}


def site_layout(catalog: list[dict], shape, name: str) -> tuple[int, int, int]:
    """Resolve named sites for legacy [slot,site,width] or ragged [slot,sum(width)]."""
    if not isinstance(catalog, (list, tuple)) or not catalog:
        raise ValueError("Missing steering site catalog")
    if any(not isinstance(site, dict) or not isinstance(site.get("name"), str)
           or type(site.get("width")) is not int or site["width"] <= 0 for site in catalog):
        raise ValueError("Invalid steering site catalog")
    if len({site["name"] for site in catalog}) != len(catalog):
        raise ValueError("Duplicate steering site names")
    matches = [i for i, site in enumerate(catalog) if site["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Missing unique steering site: {name}")
    if len(shape) not in (2, 3) or shape[0] < 1:
        raise ValueError("Unsupported steering addend layout")
    widths = [site["width"] for site in catalog]
    if len(shape) == 2 and shape[1] != sum(widths):
        raise ValueError("Ragged steering storage disagrees with site widths")
    if len(shape) == 3 and (shape[1] != len(catalog) or any(w != shape[2] for w in widths)):
        raise ValueError("Legacy steering storage disagrees with site widths")
    index = matches[0]
    return index, sum(widths[:index]), widths[index]


def evaluate_arm(baseline: dict, actual: dict, *, layer: int, site: str, delta=None) -> dict:
    """Evaluate saved small CPU observations without depending on CUDA or Flare."""
    base_r, current_r = _array(baseline["residual"]), _array(actual["residual"])
    base_p, current_p = _array(baseline["pre_mlp"]), _array(actual["pre_mlp"])
    if (base_r.ndim != 2 or base_r.shape != current_r.shape or base_p.shape != base_r.shape
            or current_p.shape != base_r.shape or type(layer) is not int
            or not 0 < layer < base_r.shape[0] - 1):
        raise ValueError("Qualification requires an interior layer and matched capture shapes")
    logits = compare(actual["logits"], baseline["logits"])
    if site == "off":
        residual, pre_mlp = compare(current_r, base_r), compare(current_p, base_p)
        result = {"passed": residual["bit_exact"] and pre_mlp["bit_exact"] and logits["bit_exact"],
                  "residual": residual, "pre_mlp": pre_mlp, "logits": logits}
        if "mlp_input" in baseline:
            result["mlp_input"] = compare(actual["mlp_input"], baseline["mlp_input"])
            result["passed"] = result["passed"] and result["mlp_input"]["bit_exact"]
        return result
    if site not in ("residual", "mlp_out", "mlp_input"):
        raise ValueError("Unsupported whole-model qualification site")
    delta = _array(delta)
    expected_shape = baseline["mlp_input"][layer].shape if site == "mlp_input" else base_r[layer].shape
    if delta.shape != expected_shape or not np.any(delta):
        raise ValueError("Qualification needs a nonzero site-width addend")
    upstream = compare(current_r[:layer], base_r[:layer])
    pre_mlp = compare(current_p[:layer + 1], base_p[:layer + 1])
    downstream = compare(current_r[layer + 1:], base_r[layer + 1:])
    target = compare(current_r[layer], base_r[layer])
    result = {"upstream_residual": upstream, "upstream_through_target_pre_mlp": pre_mlp,
              "target_vs_baseline": target, "downstream_residual": downstream, "logits": logits,
              "changed_residual_elements_by_layer": np.count_nonzero(
                  current_r.view(np.uint32) != base_r.view(np.uint32), axis=1).tolist()}
    passed = (upstream["bit_exact"] and pre_mlp["bit_exact"] and not target["bit_exact"]
              and not downstream["bit_exact"] and not logits["bit_exact"])
    if site == "mlp_input":
        # Exact pre-down application is the invariant. Quantization or a weak
        # selected coordinate may suppress its downstream effect; report that
        # observation rather than impose a behavioral pass condition on it.
        passed = upstream["bit_exact"] and pre_mlp["bit_exact"]
        result["downstream_change_required"] = False
    if "mlp_input" in baseline:
        stop = layer if site == "mlp_input" else layer + 1
        result["upstream_mlp_input"] = compare(actual["mlp_input"][:stop], baseline["mlp_input"][:stop])
        passed = passed and result["upstream_mlp_input"]["bit_exact"]
    if site in ("residual", "mlp_input"):
        # The hook preserves original bits for zero coordinates, including -0.
        base_target = baseline[site][layer]
        expected = np.where(delta != 0, round_bf16(base_target + delta), base_target)
        result["target_vs_round_bf16_baseline_plus_delta"] = compare(actual[site][layer], expected)
        passed = passed and result["target_vs_round_bf16_baseline_plus_delta"]["bit_exact"]
    else:
        result["interpretation"] = "Observable additive MLP-output effect; no pre-down neuron equivalence asserted"
    result["passed"] = bool(passed)
    return result


def qualify(prepared, generation, names, maintenances, tokenized, *, layer=32, dose=0.25, seed=1729) -> dict:
    """Replay two prompts in matched arms and restore exact mutable startup bytes.

    ``tokenized`` supplies at least two nonempty short token lists; only the first
    two are used. They may span multiple prefill chunks. Set ``layer`` to an
    interior layer. The report's ``passed`` status means every declared exact or
    observable-change check passed; it is not a claim about untested executions.
    Native graph replay/cleanup failures remain visible as failed reports or,
    for cleanup failure, an exception. The caller decides where to save a report.
    """
    import torch
    from ignition.components.sampling import INACTIVE, RUNNING, RequestAdmission
    from validate_candidate import _snapshot_mutable, _restore

    prompts = list(tokenized)[:2]
    if len(prompts) != 2 or any(not ids or any(type(x) is not int or x < 0 for x in ids) for ids in prompts):
        raise ValueError("Qualification needs two nonempty integer token lists")
    if not math.isfinite(dose) or not 0 < dose <= 4 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Qualification dose must be in (0,4] and seed uint32")
    maintenance = maintenances["init"]
    batch, decoder = generation.batch, generation.decoder
    requests, inputs, pool = batch.requests, batch.inputs, decoder.pool
    admission, commands = maintenance.state.admission.buffers, maintenance.commands
    capture, buffers = decoder.activation_capture, prepared.buffers
    if capture is None or bool((requests.status != INACTIVE).any()):
        raise ValueError("Qualification needs restored inactive state and diagnostic activation capture")
    layers, hidden = capture.residual_last.shape
    if type(layer) is not int or not 0 < layer < layers - 1:
        raise ValueError("Qualification target must be an interior layer")
    contract = prepared.server_config.get("model_contract", prepared.server_config.get("contract", {}))
    catalog = contract.get("steering", {}).get("sites", [])
    addends, phases, prompt_tokens = (buffers[name] for name in
        ("steering_addends", "steering_phases", "steering_prompt_tokens"))
    layouts = {site: site_layout(catalog, addends.shape, f"layers.{layer}.{site}")
               for site in ("residual", "mlp_out")}
    if any(layout[2] != hidden for layout in layouts.values()):
        raise ValueError("Qualification sites must match residual hidden width")
    has_mlp_input = hasattr(capture, "mlp_input_last")
    if has_mlp_input:
        layouts["mlp_input"] = site_layout(catalog, addends.shape, f"layers.{layer}.mlp_input")
        if tuple(capture.mlp_input_last.shape) != (layers, layouts["mlp_input"][2]):
            raise ValueError("Pre-down diagnostic capture disagrees with site catalog")
    if (tuple(phases.shape) != (addends.shape[0], len(catalog))
            or tuple(prompt_tokens.shape) != (addends.shape[0],)):
        raise ValueError("Steering phase/request storage disagrees with addends")
    vocabulary = decoder.scratch.logits.shape[-1]
    if any(max(ids) >= vocabulary for ids in prompts):
        raise ValueError("Qualification token outside prepared vocabulary")
    page_tokens = int(prepared.server_config["page_tokens"])
    pages = math.ceil(max(map(len, prompts)) / page_tokens)
    capacity = buffers["memory_capacity"].cpu().tolist()
    if capacity[0] < 3 or capacity[1] < pages or pages > inputs.page_tables.shape[1]:
        raise ValueError("Qualification prompts exceed physical preparation capacity")
    snapshots = _snapshot_mutable(prepared, generation, maintenances)
    # Diagnostic outputs are not necessarily registered public buffers. Preserve
    # their previous observations too, so qualification cannot pollute calibration.
    seen = {tensor.data_ptr() for _, tensor, _ in snapshots}
    for name, tensor in [(f"capture.{name}", getattr(capture, name)) for name in capture.__dataclass_fields__] + [
            ("diagnostic.logits", decoder.scratch.logits)]:
        if tensor.data_ptr() not in seen:
            snapshots.append((name, tensor, tensor.detach().to("cpu", copy=True)))
            seen.add(tensor.data_ptr())
    report = {"schema": "character-creation.whole-model-qualification.v1", "status": "running",
              "prepared_identity": prepared.server_config.get("identity"), "graphs": dict(names),
              "layer": layer, "dose": dose, "seed": seed, "state_restored": False,
              "site_catalog_sha256": hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest(),
              "dose_semantics": "signed dose * unsteered target-layer final-prompt-token L2; normalized fixed Gaussian direction; every prefill token",
              "scope": "retained init/prefill graphs, two prompts, one interior layer, physical request slot zero",
              "zero_vector_scope": "all catalog sites, phase mask 3 (prefill and decode enabled), prefill replay only",
              "mlp_input_covered": has_mlp_input,
              "not_covered": ["native server cache isolation", "native server ownership handoff", "decode or speculative execution",
                              "all layers or coordinates", "behavioral or personality effects", "MLP-output column push equivalence to pre-down neuron push"],
              "prompts": []}

    def run(ids, *, programs=(), zero_enabled=False):
        _restore(prepared, pool, snapshots)
        sequence = int(buffers["submission_sequence"].item())
        epoch = 1

        def replay(kind, consumed):
            nonlocal sequence
            sequence += 1
            frame = sequence % 2
            buffers["frame_index"].fill_(frame)
            buffers["submission_sequence"].fill_(sequence)
            prepared.graphs[names[kind]].replay()
            torch.cuda.synchronize(prepared.device)
            record = buffers["completion_records"][frame, 0].cpu().tolist()
            if record[:7] != [epoch, sequence, consumed, 0, -1, RUNNING, 0]:
                raise RuntimeError(f"Qualification {kind} completion disagrees: {record[:7]}")
            observed = [int(getattr(requests, name)[0].item()) for name in
                ("generation", "consumed_tokens", "generated_tokens", "pending_token", "status", "error_code", "rng_counter")]
            if observed != [epoch, consumed, 0, -1, RUNNING, 0, 0]:
                raise RuntimeError(f"Qualification request state disagrees: {observed}")
            if kind == "init" and (bool(commands.errors.any()) or bool(maintenance.state.admission_errors.any())):
                raise RuntimeError("Qualification admission or transfer failed")

        addends.zero_()
        phases.zero_()
        prompt_tokens.zero_()
        inputs.request_state_rows.zero_()
        inputs.request_state_rows[0, 0] = 1
        inputs.page_tables.fill_(-1)
        inputs.page_tables[0, :pages].copy_(torch.arange(pages, dtype=torch.int32))
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
        replay("init", 0)
        admission.admission_count.zero_()
        commands.num_ops.zero_()
        prompt_tokens[0] = len(ids)
        if zero_enabled:
            phases[0].fill_(3)
        for site, delta in programs:
            index, offset, width = layouts[site]
            target = addends[0, offset:offset + width] if addends.ndim == 2 else addends[0, index]
            target.copy_(torch.from_numpy(delta))
            phases[0, index] = 1
        for start in range(0, len(ids), batch.tokens):
            stop = min(len(ids), start + batch.tokens)
            inputs.query_lengths[0] = stop - start
            inputs.token_ids.zero_()
            inputs.token_ids[:stop - start].copy_(torch.tensor(ids[start:stop], dtype=torch.int64))
            replay("prefill", stop)
            positions = capture.positions.cpu().numpy()
            if not np.array_equal(positions[positions >= 0], np.arange(start, stop)):
                raise RuntimeError("Qualification capture positions disagree with consumed tokens")
        observed = {"residual": _array(capture.residual_last.cpu().numpy().copy()),
                    "pre_mlp": _array(capture.pre_mlp_last.cpu().numpy().copy()),
                    "logits": _array(decoder.scratch.logits[0].cpu().numpy().copy())}
        if has_mlp_input:
            observed["mlp_input"] = _array(capture.mlp_input_last.cpu().numpy().copy())
        return observed

    try:
        with torch.inference_mode(), torch.cuda.device(prepared.device):
            direction = np.random.default_rng(seed).standard_normal(hidden)
            direction /= np.linalg.norm(direction)
            for prompt_index, ids in enumerate(prompts):
                baseline = run(ids)
                norm = float(np.linalg.norm(baseline["residual"][layer].astype(np.float64)))
                if not math.isfinite(norm) or norm <= 0:
                    raise RuntimeError("Qualification baseline target norm is nonfinite or zero")
                delta = (direction * (dose * norm)).astype(np.float32)
                entry = {"index": prompt_index, "tokens": len(ids),
                         "token_ids_sha256": _hash(np.asarray(ids, dtype="<i8")),
                         "baseline_target_l2": norm, "addend_l2": float(np.linalg.norm(delta.astype(np.float64))),
                         "addend_sha256": _hash(delta), "checks": {}}
                report["prompts"].append(entry)
                for label, observed in (("repeated_unsteered", run(ids)),
                                        ("enabled_zero_vector", run(ids, zero_enabled=True))):
                    entry["checks"][label] = evaluate_arm(baseline, observed, layer=layer, site="off")
                for site in ("residual", "mlp_out"):
                    for sign in (1, -1):
                        signed = delta * np.float32(sign)
                        observed = run(ids, programs=((site, signed),))
                        entry["checks"][f"{site}_{'positive' if sign > 0 else 'negative'}"] = evaluate_arm(
                            baseline, observed, layer=layer, site=site, delta=signed)
                if has_mlp_input:
                    # A bounded arithmetic probe, deliberately independent of
                    # empirical persona/dose calibration or down-column norms.
                    coordinate = int(np.argmin(np.abs(baseline["mlp_input"][layer])))
                    entry["mlp_input_probe"] = {"coordinate": coordinate, "absolute_addend": 1.0,
                        "selection": "lowest absolute baseline final-token product coordinate; first tie"}
                    for sign in (1, -1):
                        one_hot = np.zeros(layouts["mlp_input"][2], dtype=np.float32)
                        one_hot[coordinate] = sign
                        observed = run(ids, programs=(("mlp_input", one_hot),))
                        entry["checks"][f"mlp_input_{'positive' if sign > 0 else 'negative'}"] = evaluate_arm(
                            baseline, observed, layer=layer, site="mlp_input", delta=one_hot)
            report["status"] = "passed" if all(check["passed"] for prompt in report["prompts"]
                for check in prompt["checks"].values()) else "failed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}"[:3000])
    finally:
        _restore(prepared, pool, snapshots)
        report["state_restored"] = True
    return report
