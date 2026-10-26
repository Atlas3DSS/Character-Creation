"""Persistent, serialized diagnostic inference on retained production CUDA graphs.

This is an OpenAI-compatible subset for local experiments, not a native Flare
server: B1 ordinary decode, actual PreparedSampling, no native cache or MTP
execution. Preparation retains the reviewed four-slot, depth-four-capable
catalog because SM70 prefill requires its seven-column convolution history.
Every accepted generation saves actual final-prefill layer observations and full
private request/response provenance. Startup requires its own workboard lease.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
import uuid
import zipfile
from urllib.parse import urlparse, parse_qs
from copy import deepcopy
from persona_trace import normalize_trace, TokenTrace
from persona_kv import validate_control, HybridPrefixCache
from persona_phantom import LearnedKVCache

import numpy as np

from persona_vectors import ModelSpec, canonical_digest


RUNTIME = "retained diagnostic graphs; B1 ordinary decode; actual PreparedSampling; MTP-capable preparation; no native cache or MTP execution"
PHASE_MASKS = {"prefill": 1, "decode": 2, "all": 3}
REQUEST_STATE_FIELDS = ("generation", "consumed_tokens", "generated_tokens", "pending_token", "status", "error_code", "rng_counter")


class ProtocolError(RuntimeError):
    """A replay/state invariant failed; the owning worker must stop."""


def reviewed_preparation_options():
    # The existing Sol-15 kernel admits exactly 3 + 4 history columns. Merely
    # increasing request capacity with mtp=False leaves three columns and fails.
    # Keep the passed capture layout/catalog, but select only ordinary B1 graphs.
    return dict(max_active=4, prefill_chunk_size=128, mtp=True,
                speculative_widths=(1, 2, 3, 4, 5), row_invariant=None,
                retain_graphs=True, capture_activations=True, steering_max_sites=64)


def validate_selected_graphs(signatures, names, request_capacity):
    if request_capacity != 4:
        raise ProtocolError("Diagnostic preparation must retain the reviewed four-slot state layout")
    required = {"init", "prefill", "decode", "snapshot", "restore", "resume"}
    if set(names) != required:
        raise ProtocolError("Missing exact ordinary graph selection")
    for kind, name in names.items():
        signature = signatures[name]
        if signature.kind != kind or signature.batch_capacity != 1 or signature.speculation_enabled is not False:
            raise ProtocolError("Only one-request ordinary graphs may execute in the diagnostic worker")


def validate_inactive_slots(observed):
    if any(list(row) != [0, 0, 0, -1, 0, 0, 0] for row in observed):
        raise ProtocolError("An inactive padded request slot changed during one-request replay")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_activation_npz(destination, **arrays):
    """Write ordinary lossless NPZ members with explicitly bounded compression.

    Level one avoids default level-six deflate's measured CPU bottleneck. NPY
    members retain their names, dtypes, shapes and exact array bits. The caller
    owns exclusive file creation, subsequent hashing and provenance publication.
    """
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for name, value in arrays.items():
            with archive.open(name + ".npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(value), allow_pickle=False)


def normalize_request(payload: dict, model_id: str) -> dict:
    """Validate the supported API subset without importing CUDA libraries."""
    supported = {"model", "messages", "seed", "max_tokens", "temperature", "top_p", "top_k", "min_p",
                 "repetition_penalty", "presence_penalty", "frequency_penalty", "stream", "n",
                 "chat_template_kwargs", "steering", "metadata", "observation_only", "capture_trace", "kv_control"}
    if not isinstance(payload, dict) or set(payload) - supported:
        raise ValueError("Unsupported chat-completion fields")
    if payload.get("model") != model_id or payload.get("stream", False) is not False or payload.get("n", 1) != 1:
        raise ValueError("Require the advertised model, stream=false, and n=1")
    messages = payload.get("messages")
    if (not isinstance(messages, list) or not messages or any(not isinstance(m, dict) or set(m) != {"role", "content"}
            or m["role"] not in {"system", "developer", "user", "assistant"} or not isinstance(m["content"], str) for m in messages)):
        raise ValueError("Messages must be nonempty text-only role/content objects")
    template = payload.get("chat_template_kwargs", {"enable_thinking": False})
    if not isinstance(template, dict) or set(template) - {"enable_thinking"} or template.get("enable_thinking", False) is not False:
        raise ValueError("This diagnostic worker currently supports explicit thinking-disabled requests only")
    seed, budget = payload.get("seed", 0), payload.get("max_tokens", 2048)
    if type(seed) is not int or not 0 <= seed < 2**64 or type(budget) is not int or not 1 <= budget < 2**31:
        raise ValueError("Seed must be uint64 and max_tokens a positive int32")
    observation_only = payload.get("observation_only", False)
    if type(observation_only) is not bool or (observation_only and budget != 1):
        raise ValueError("observation_only must be bool and requires explicit max_tokens=1")
    sampling = {"temperature": payload.get("temperature", .7), "top_p": payload.get("top_p", .95),
                "top_k": payload.get("top_k", 20), "min_p": payload.get("min_p", 0.),
                "repetition_penalty": payload.get("repetition_penalty", 1.),
                "presence_penalty": payload.get("presence_penalty", 0.), "frequency_penalty": payload.get("frequency_penalty", 0.)}
    for key, value in sampling.items():
        if key == "top_k":
            if type(value) is not int or value < -1:
                raise ValueError("top_k must be an integer >= -1")
        elif type(value) not in (int, float) or not math.isfinite(value) or abs(value) > np.finfo(np.float32).max:
            raise ValueError(f"{key} must be finite FP32")
    if (sampling["temperature"] < 0 or not 0 < sampling["top_p"] <= 1 or not 0 <= sampling["min_p"] <= 1
            or sampling["repetition_penalty"] <= 0):
        raise ValueError("Invalid temperature/probability/penalty range")
    requested = dict(sampling)
    if sampling["temperature"] == 0:
        # Native admission requires positive temperature. Top-one at T=1 is
        # exact greedy selection, still through the actual captured sampler.
        sampling.update(temperature=1., top_k=1, top_p=1., min_p=0.)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict) or ("job_id" in metadata and (not isinstance(metadata["job_id"], str) or len(metadata["job_id"]) > 256)):
        raise ValueError("Metadata must be an object with a bounded string job_id")
    kv_control = validate_control(payload.get("kv_control"))
    if kv_control is not None and payload.get("capture_trace") is not None:
        raise ValueError("KV prefix caching and live token traces require separate requests")
    canonical_digest(payload)  # Reject NaN/Infinity anywhere, including metadata.
    return {"messages": messages, "seed": seed, "max_tokens": budget,
            "sampling": sampling, "requested_sampling": requested, "metadata": metadata,
            "capture_trace": normalize_trace(payload.get("capture_trace")), "kv_control": kv_control,
            "steering": payload.get("steering"), "thinking": False, "observation_only": observation_only}


def completion_outcome(spec, *, positions, prompt_tokens, generated, graph_kinds, result, answer, position_offset=0):
    """Distinguish a completed neural observation from a completed text answer."""
    if (not generated or len(generated) > spec["max_tokens"] or positions != list(range(position_offset, position_offset + prompt_tokens))
            or result.get("finished") is not True or result.get("finish_reason") not in {"stop", "length"}):
        raise ProtocolError("Generation termination, budget or exact prompt coverage invariant failed")
    observation = spec["observation_only"]
    if observation and (spec["max_tokens"] != 1 or len(generated) != 1 or "decode" in graph_kinds
                        or not graph_kinds or graph_kinds[-1] != "prefill"):
        raise ProtocolError("Prefill observation must sample exactly once without any decode replay")
    unexpected_thinking = "<think>" in answer or "</think>" in answer
    return {"operation": "prefill_observation" if observation else "rollout",
            "content": "" if observation else answer,
            "diagnostics": {"prefill_observation_complete": observation,
                            "unexpected_thinking": unexpected_thinking,
                            "complete_final_answer": not observation and result["finish_reason"] == "stop"
                                                     and bool(answer.strip()) and not unexpected_thinking}}


def validate_program(program, catalog: list[dict], maximum: int) -> dict:
    widths = {site["name"]: site["width"] for site in catalog}
    if len(widths) != len(catalog) or not widths:
        raise ValueError("Prepared steering catalog is missing or duplicated")
    if program is None:
        return {"phase": "all", "sites": []}
    if not isinstance(program, dict) or set(program) != {"phase", "sites"} or program["phase"] not in PHASE_MASKS:
        raise ValueError("Steering requires explicit supported phase and sites")
    sites = program["sites"]
    if not isinstance(sites, list) or len(sites) > maximum:
        raise ValueError("Steering exceeds the prepared maximum site count")
    normalized, seen = [], set()
    for site in sites:
        if not isinstance(site, dict) or set(site) != {"site", "vector"}:
            raise ValueError("Invalid steering site entry")
        name, values = site["site"], site["vector"]
        if not isinstance(name, str) or name not in widths or name in seen or not isinstance(values, list) or len(values) != widths[name]:
            raise ValueError("Steering name/width must match one unique prepared site")
        if any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > np.finfo(np.float32).max for value in values):
            raise ValueError("Steering values must fit finite FP32")
        normalized.append({"site": name, "vector": np.asarray(values, dtype=np.float32)})
        seen.add(name)
    return {"phase": program["phase"], "sites": normalized}


def validate_capacity(prompt_tokens: int, budget: int, *, physical_pages: int, page_tokens: int,
                      logical_context: int, table_pages: int) -> int:
    if any(type(v) is not int or v <= 0 for v in (prompt_tokens, budget, physical_pages, page_tokens, logical_context, table_pages)):
        raise ValueError("Token/page capacities must be positive integers")
    capacity = min(physical_pages, table_pages) * page_tokens
    if prompt_tokens + budget > min(capacity, logical_context):
        raise ValueError(f"Prompt ({prompt_tokens}) plus generation budget ({budget}) exceeds physical/logical capacity ({min(capacity, logical_context)}); no truncation performed")
    return math.ceil((prompt_tokens + budget) / page_tokens)


def validate_completion(record, observed, *, epoch, sequence, consumed, generated, sampled,
                        budget, eos, vocabulary) -> dict:
    """Check production completion and request fields, including exact RNG progress."""
    if len(record) < 7 or len(observed) != 7:
        raise ProtocolError("Malformed completion/request state")
    if (epoch < 1 or sequence < 1 or consumed < 0 or not 0 <= generated <= budget
            or (sampled and generated < 1) or (not sampled and generated != 0)):
        raise ProtocolError("Invalid expected request progression")
    token = int(record[4])
    if (sampled and not 0 <= token < vocabulary) or (not sampled and token != -1):
        raise ProtocolError("Invalid sampled/unsampled token")
    finished = sampled and (token in eos or generated >= budget)
    status = 2 if finished else 1
    expected = [epoch, sequence, consumed, generated, token, status, 0]
    state = [epoch, consumed, generated, token if sampled else -1, status, 0, generated]
    if list(record[:7]) != expected or list(observed) != state:
        raise ProtocolError(f"Completion/request invariant failed: record={list(record[:7])}, expected={expected}; request={list(observed)}, expected={state}")
    return {"token": token, "finished": bool(finished),
            "finish_reason": ("stop" if token in eos else "length") if finished else None}


def calibration_reference(path: Path, model: ModelSpec) -> dict:
    """Load one complete model-bound source prompt, preferring an unsteered baseline."""
    report = json.loads(path.with_suffix(".json").read_text())
    if (report.get("status") != "complete" or report.get("state_restored") is not True
            or report.get("model_identity") != model.identity or report.get("model") != model.__dict__
            or report.get("artifact_sha256") != file_hash(path)):
        raise ValueError("Startup requires a complete matching calibration artifact and manifest")
    jobs = report["jobs"]
    if not jobs or report.get("prompts_completed") != len(jobs):
        raise ValueError("Calibration prompt coverage is incomplete")
    if (len({job["id"] for job in jobs}) != len(jobs)
            or any(job["id"] != canonical_digest({k: v for k, v in job.items() if k != "id"})
                   or job.get("split") != "discovery" or job.get("thinking") is not False
                   or not isinstance(job.get("prompt"), str) or not isinstance(job.get("system"), str)
                   for job in jobs)):
        raise ValueError("Calibration prompt content or discovery provenance differs from its identity")
    index = next((i for i, job in enumerate(jobs) if job["condition"] == "baseline"), 0)
    with np.load(path, allow_pickle=False) as data:
        if (str(data["model_identity"]) != model.identity or str(data["plan_sha256"]) != report["plan_sha256"]
                or data["job_ids"].tolist() != [job["id"] for job in jobs]):
            raise ValueError("Calibration arrays differ from their manifest")
        for key, size in (("token_offsets", data["token_ids"].size), ("norm_offsets", data["residual_l2"].shape[1])):
            offsets = data[key]
            if (offsets.shape != (len(jobs) + 1,) or offsets.dtype != np.int64
                    or offsets[0] != 0 or offsets[-1] != size or np.any(np.diff(offsets) <= 0)):
                raise ValueError("Calibration offsets do not exhaustively partition the source arrays")
        a, b = data["token_offsets"][index:index + 2]
        ids = data["token_ids"][a:b].copy()
        a, b = data["norm_offsets"][index:index + 2]
        norms = data["residual_l2"][:, a:b].copy()
        expected = {name: data[name][index].copy() for name in ("mlp_input_last", "residual_last", "pre_mlp_last")}
    if (ids.ndim != 1 or ids.dtype != np.int64 or len(ids) < 2
            or norms.shape != (model.layers, len(ids) - 1) or norms.dtype != np.float32
            or expected["mlp_input_last"].shape != (model.layers, model.intermediate)
            or any(expected[name].shape != (model.layers, model.hidden) for name in ("residual_last", "pre_mlp_last"))
            or any(value.dtype != np.float32 or not np.isfinite(value).all() for value in (*expected.values(), norms))):
        raise ValueError("Calibration tensor shapes/dtypes/finite values are invalid")
    expected.update(prompt_token_ids=ids, prefill_positions=np.arange(len(ids), dtype=np.int64),
                    residual_l2=norms, residual_l2_count=np.int64(norms.shape[1]),
                    residual_l2_min=norms.min(axis=1), residual_l2_max=norms.max(axis=1),
                    residual_l2_mean=norms.mean(axis=1, dtype=np.float64), residual_l2_median=np.median(norms, axis=1))
    return {"job": jobs[index], "expected": expected, "artifact_sha256": report["artifact_sha256"],
            "path": str(path.resolve()), "model_identity": model.identity,
            "checkpoint_config_sha256": report["checkpoint_config_sha256"],
            "checkpoint_index_sha256": report["checkpoint_index_sha256"],
            "tokenizer_sha256": report["tokenizer_sha256"], "source_preparation": report["preparation"]}


def compare_calibration_arrays(actual: dict, expected: dict) -> dict:
    checks = {}
    for name, reference in expected.items():
        value, reference = np.asarray(actual[name]), np.asarray(reference)
        shape_and_type = value.shape == reference.shape and value.dtype == reference.dtype
        same = shape_and_type and np.array_equal(np.ascontiguousarray(value).reshape(-1).view(np.uint8),
                                                  np.ascontiguousarray(reference).reshape(-1).view(np.uint8))
        checks[name] = {"bit_exact": bool(same), "shape": list(value.shape), "dtype": str(value.dtype),
                        "actual_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                        "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest()}
    return {"passed": all(check["bit_exact"] for check in checks.values()), "checks": checks,
            "scope": "one original calibration prompt: exact encoded tokens, every layer final-prefill observation and all non-sink prompt residual norms; not native cache/MTP/decode qualification"}


class ProbeEngine:
    def __init__(self, prepared, tokenizer, model, output, manifest, stop_event, *, model_id="checkpoint"):
        import torch
        from ignition.components.sampling import PreparedSampling
        from validate_candidate import _owners, _snapshot_mutable

        self.prepared, self.tokenizer, self.model = prepared, tokenizer, model
        self.output, self.manifest, self.stop_event, self.model_id = Path(output), manifest, stop_event, model_id
        self.lock = threading.Lock()
        self.trace_lock = threading.RLock()
        self.trace_records = {}
        self.epoch, self.sequence = 0, int(prepared.buffers["submission_sequence"].item())
        self.names, self.generation, self.maintenances = _owners(prepared, mode="chunked")
        self.batch, self.decoder = self.generation.batch, self.generation.decoder
        catalogs = [resource for resource in prepared.resources if isinstance(resource, Mapping)
                    and resource.get(self.names["prefill"]) is self.generation]
        if len(catalogs) != 1:
            raise ProtocolError("Missing unique retained graph owner catalog")
        decode = catalogs[0][self.names["decode"]]
        self.decode_generation = decode
        if (self.batch.rows != 1 or decode.batch.rows != 1
                or not isinstance(self.generation.completion, PreparedSampling)
                or not isinstance(decode.completion, PreparedSampling)):
            raise ProtocolError("Worker requires ordinary B1 graphs with actual PreparedSampling")
        self.capture = self.decoder.activation_capture
        if self.capture is None or not hasattr(self.capture, "mlp_input_last"):
            raise ProtocolError("Worker requires actual pre-down neuron capture")
        if (tuple(self.capture.mlp_input_last.shape) != (model.layers, model.intermediate)
                or tuple(self.capture.residual_last.shape) != (model.layers, model.hidden)):
            raise ProtocolError("Capture dimensions disagree with bound model")
        self.requests, self.inputs, self.pool = self.batch.requests, self.batch.inputs, self.decoder.pool
        validate_selected_graphs(prepared.signatures, self.names, self.requests.capacity)
        if bool((self.requests.status != 0).any()):
            raise ProtocolError("Prepared diagnostic requests must start inactive")
        self.maintenance = self.maintenances["init"]
        self.admission, self.commands = self.maintenance.state.admission.buffers, self.maintenance.commands
        self.buffers = prepared.buffers
        self.capacity = [int(n) for n in self.buffers["memory_capacity"].cpu().tolist()]
        if self.capacity[0] < 2:
            raise ProtocolError("Physical continuation row one is unavailable")
        contract = prepared.server_config["model_contract"]
        self.steering = contract["steering"]
        self.catalog = self.steering["sites"]
        self.eos = tuple(prepared.server_config["frontend"]["eos_token_ids"])
        self.page_tokens = int(prepared.server_config["page_tokens"])
        self.vocabulary = self.requests.vocab_size
        if not self.eos or any(type(token) is not int or not 0 <= token < self.vocabulary for token in self.eos):
            raise ProtocolError("Prepared frontend lacks valid explicit EOS IDs")
        self.kv_cache = HybridPrefixCache(self)
        self.learned_kv_cache = LearnedKVCache(self, self.kv_cache.binding)
        self.snapshots = _snapshot_mutable(prepared, self.generation, self.maintenances)
        # Diagnostic planes are not public server buffers; restore them too.
        seen = {tensor.data_ptr() for _, tensor, _ in self.snapshots}
        for name in self.capture.__dataclass_fields__:
            tensor = getattr(self.capture, name)
            if tensor.data_ptr() not in seen:
                self.snapshots.append(("capture." + name, tensor, tensor.detach().to("cpu", copy=True)))
                seen.add(tensor.data_ptr())
        torch.cuda.synchronize(prepared.device)

    def descriptor(self):
        return {"id": self.model_id, "object": "model", "owned_by": "local-diagnostic",
                "flare": {"prepared_identity": self.prepared.server_config["identity"],
                    "model_identity": self.model.identity, "runtime": RUNTIME,
                    "runtime_identity": self.manifest.get("runtime_identity"),
                    "target_row_invariant": True,
                    "calibration_equivalence": {key: self.manifest.get("calibration_equivalence", {}).get(key)
                                                for key in ("passed", "reference_sha256", "job_id", "scope")},
                    "steering": {"protocol": "additive-bf16-sites-v1", **self.steering},
                    "logical_context_tokens": self.prepared.server_config["max_sequence_tokens"],
                    "physical_context_tokens": min(self.capacity[1], self.inputs.page_tables.shape[1]) * self.page_tokens,
                    "batch_capacity": 1, "prepared_request_capacity": self.requests.capacity,
                    "native_cache": False, "mtp": False, "mtp_prepared": True,
                    "kv_control": self.kv_cache.capability(),
                    "observation_only": {"supported": True, "required_max_tokens": 1,
                                         "operation": "prefill_observation", "answer_scoring": False},
                    "activation_capture": "final_prefill", "token_trace": {"supported": True, "max_neurons": 64, "max_per_layer": 32, "decode": True}, "thinking_enabled_supported": False}}

    def complete(self, payload):
        import torch
        from ignition.components.sampling import RequestAdmission
        from validate_candidate import _restore
        from persona_qualify import site_layout

        spec = normalize_request(payload, self.model_id)
        program = validate_program(spec["steering"], self.catalog, self.steering["max_sites"])
        with self.lock, torch.inference_mode(), torch.cuda.device(self.prepared.device):
            if self.stop_event.is_set():
                raise ProtocolError("Worker is stopping")
            from persona_capture import tokenize_messages
            learned = bool(spec['kv_control'] and spec['kv_control']['artifact']['kind'] == 'learned_attention_kv')
            virtual_slots, learned_entry, learned_hit = 0, None, False
            if learned:
                virtual_slots, learned_entry, learned_hit = self.learned_kv_cache.prepare(spec['kv_control'])
                prefix_ids, prefix_entry = [], None
            else:
                prefix_ids, prefix_entry = self.kv_cache.prepare(spec["kv_control"])
            prefix_hit = prefix_entry is not None
            ids = prefix_ids + tokenize_messages(self.tokenizer, spec["messages"])
            if not isinstance(ids, list) or any(type(token) is not int or not 0 <= token < self.vocabulary for token in ids):
                raise ValueError("Tokenizer returned invalid prompt token IDs")
            logical_prompt_tokens = virtual_slots + len(ids)
            pages = validate_capacity(logical_prompt_tokens, spec["max_tokens"], physical_pages=self.capacity[1], page_tokens=self.page_tokens,
                logical_context=int(self.prepared.server_config["max_sequence_tokens"]), table_pages=self.inputs.page_tables.shape[1])
            self.epoch += 1
            epoch = self.epoch
            admission_spec = RequestAdmission(generation=epoch, seed=spec["seed"], max_new_tokens=spec["max_tokens"],
                                               eos_token_ids=self.eos, **spec["sampling"])
            admission_spec.validate(self.vocabulary)
            identity = uuid.uuid4().hex
            request_hash = canonical_digest(payload)
            artifact = self.output / f"probe-{identity}.npz"
            provenance = self.output / f"probe-{identity}.jsonl"
            started = time.time()
            report = {"schema": "character-creation.probe-request.v1", "status": "running", "id": identity,
                      "operation": "prefill_observation" if spec["observation_only"] else "rollout",
                      "runtime": RUNTIME, "model_identity": self.model.identity,
                      "runtime_identity": self.manifest["runtime_identity"],
                      "prepared_identity": self.prepared.server_config["identity"], "worker_manifest": self.manifest,
                      "request": payload, "request_sha256": request_hash, "started_at": started,
                      "requested_sampling": spec["requested_sampling"], "effective_admission": admission_spec.__dict__,
                      "graphs": self.names, "physical_pages": self.capacity[1], "mapped_request_pages": pages,
                      "state_restored": False, "generated_tokens": 0}
            generated, trace, graph_kinds, norm_chunks, positions_seen = [], [], [], [], []
            prefill_capture = None
            response = None
            token_trace = TokenTrace(self, spec["capture_trace"], ids, request_hash) if spec["capture_trace"] else None

            def replay(kind, consumed, count, sampled):
                if self.stop_event.is_set():
                    raise ProtocolError("Worker interrupted before graph replay")
                self.sequence += 1
                frame = self.sequence % 2
                self.buffers["frame_index"].fill_(frame)
                self.buffers["submission_sequence"].fill_(self.sequence)
                self.prepared.graphs[self.names[kind]].replay()
                torch.cuda.synchronize(self.prepared.device)
                completion = self.buffers["completion_records"][frame, 0].cpu().tolist()
                request_rows = list(zip(*(getattr(self.requests, name).cpu().tolist() for name in REQUEST_STATE_FIELDS)))
                observed = [int(value) for value in request_rows[0]]
                validate_inactive_slots(request_rows[1:])
                result = validate_completion(completion, observed, epoch=epoch, sequence=self.sequence,
                    consumed=consumed, generated=count, sampled=sampled, budget=spec["max_tokens"], eos=self.eos, vocabulary=self.vocabulary)
                if kind == "init" and (bool(self.commands.errors.any()) or bool(self.maintenance.state.admission_errors.any())):
                    raise ProtocolError("Production admission/state transfer failed")
                trace.append(completion[:7] + [observed[-1]])
                graph_kinds.append(kind)
                return result

            try:
                _restore(self.prepared, self.pool, self.snapshots)
                if token_trace: token_trace.configure()
                inputs, commands = self.inputs, self.commands
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
                for key in ("steering_addends", "steering_phases", "steering_prompt_tokens"):
                    self.buffers[key].zero_()
                self.admission.stage(0, 0, admission_spec, ids)
                self.admission.admission_count.fill_(1)
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
                replay("init", 0, 0, False)
                self.admission.admission_count.zero_()
                commands.num_ops.zero_()
                addends = self.buffers["steering_addends"]
                self.buffers["steering_prompt_tokens"][0] = logical_prompt_tokens
                def apply_steering():
                    for site in program["sites"]:
                        index, offset, width = site_layout(self.catalog, addends.shape, site["site"])
                        target = addends[0, offset:offset + width] if addends.ndim == 2 else addends[0, index]
                        target.copy_(torch.from_numpy(site["vector"]))
                        self.buffers["steering_phases"][0, index] = PHASE_MASKS[program["phase"]]
                if learned_entry is not None:
                    self.learned_kv_cache.install(learned_entry)
                if prefix_hit:
                    self.kv_cache.restore(prefix_entry)
                    positions_seen.extend(range(len(prefix_ids)))
                    norm_chunks.append(prefix_entry["arrays"]["residual_l2"].copy())
                if not prefix_ids or prefix_hit: apply_steering()
                spans = []
                if prefix_ids and not prefix_hit:
                    spans.extend((start, min(len(prefix_ids), start + self.batch.tokens))
                                 for start in range(0, len(prefix_ids), self.batch.tokens))
                spans.extend((start, min(len(ids), start + self.batch.tokens))
                             for start in range(len(prefix_ids), len(ids), self.batch.tokens))
                for start, stop in spans:
                    last = stop == len(ids)
                    inputs.query_lengths[0] = stop - start
                    inputs.token_ids.zero_()
                    inputs.token_ids[:stop - start].copy_(torch.tensor(ids[start:stop], dtype=torch.int64))
                    inputs.sample_enabled[0] = int(last)
                    result = replay("prefill", virtual_slots + stop, int(last), last)
                    if token_trace: token_trace.observe("prefill", list(range(start, stop)), ids[start:stop], last=last)
                    positions = self.capture.positions.cpu().numpy().copy()
                    if not np.array_equal(positions[positions >= 0], np.arange(virtual_slots + start, virtual_slots + stop)):
                        raise ProtocolError("Observed prefill positions differ from actual consumed tokens")
                    positions_seen.extend(positions[positions >= 0].tolist())
                    norms = self.capture.residual_l2.cpu().numpy().copy()[:, positions > virtual_slots]
                    if not np.isfinite(norms).all():
                        raise ProtocolError("Observed residual norms are nonfinite")
                    norm_chunks.append(norms)
                    if prefix_ids and not prefix_hit and stop == len(prefix_ids):
                        prefix_entry = self.kv_cache.save(spec["kv_control"]["artifact"], prefix_ids,
                                                         np.concatenate(norm_chunks, axis=1))
                        apply_steering()
                    if last:
                        prefill_capture = {name: getattr(self.capture, name).cpu().numpy().copy()
                            for name in ("mlp_input_last", "residual_last", "pre_mlp_last")}
                        if any(not np.isfinite(value).all() for value in prefill_capture.values()):
                            raise ProtocolError("Final-prefill observations are nonfinite")
                        prefill_capture["final_chunk_positions"] = positions
                        generated.append(result["token"])
                inputs.query_lengths[0] = 1
                inputs.sample_enabled[0] = 1
                while not spec["observation_only"] and not result["finished"]:
                    # The decoder reads the device-owned pending token. Never
                    # sample on CPU or replace it with a teacher-forced token.
                    result = replay("decode", logical_prompt_tokens + len(generated), len(generated) + 1, True)
                    if token_trace: token_trace.observe("decode", [len(ids)+len(generated)-1], [generated[-1]])
                    generated.append(result["token"])
                finish_reason = result["finish_reason"]
                visible_ids = generated[:-1] if generated[-1] in self.eos else generated
                answer = self.tokenizer.decode(visible_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                outcome = completion_outcome(spec, positions=positions_seen, prompt_tokens=len(ids),
                    generated=generated, graph_kinds=graph_kinds, result=result, answer=answer, position_offset=virtual_slots)
                norms = np.concatenate(norm_chunks, axis=1)
                if norms.shape[1]:
                    summaries = {"residual_l2_min": norms.min(axis=1), "residual_l2_max": norms.max(axis=1),
                                 "residual_l2_mean": norms.mean(axis=1, dtype=np.float64), "residual_l2_median": np.median(norms, axis=1)}
                else:
                    summaries = {"residual_l2_" + name: np.zeros(self.model.layers, dtype=np.float64) for name in ("min", "max", "mean", "median")}
                with artifact.open("xb") as destination:
                    save_activation_npz(destination, **prefill_capture, **summaries, **(token_trace.arrays() if token_trace else {}),
                        residual_l2=norms,
                        residual_l2_count=np.int64(norms.shape[1]), prefill_positions=np.asarray(positions_seen, dtype=np.int64),
                        prompt_token_ids=np.asarray(ids, dtype=np.int64), generated_token_ids=np.asarray(generated, dtype=np.int64),
                        virtual_prefix_slots=np.int64(virtual_slots),
                        completion_trace=np.asarray(trace, dtype=np.int64), completion_graph_kind=np.asarray(graph_kinds),
                        completion_trace_columns=np.asarray(["epoch", "sequence", "consumed", "generated", "token", "status", "error", "rng_counter"]),
                        model_identity=self.model.identity, prepared_identity=self.prepared.server_config["identity"],
                        runtime_identity=self.manifest["runtime_identity"],
                        request_sha256=request_hash, stage="final_prefill", runtime=RUNTIME, operation=outcome["operation"])
                activation = {"path": str(artifact.resolve()), "sha256": file_hash(artifact), "model_identity": self.model.identity,
                              "runtime_identity": self.manifest["runtime_identity"],
                              "operation": outcome["operation"],
                              "prepared_identity": self.prepared.server_config["identity"], "request_sha256": request_hash,
                              "stage": "final_prefill", "fields": ["mlp_input_last", "residual_last", "pre_mlp_last"],
                              "semantics": "continuous SwiGLU product after optional pre-down steering; post-residual output; normalized MLP input; no firing threshold"}
                response = {"id": "chatcmpl-" + identity, "object": "chat.completion", "created": int(started), "model": self.model_id,
                    "operation": outcome["operation"],
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": outcome["content"]}, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": len(ids), "completion_tokens": len(generated), "total_tokens": len(ids) + len(generated)},
                    "runtime": RUNTIME, "model_identity": self.model.identity, "prepared_identity": self.prepared.server_config["identity"],
                    "runtime_identity": self.manifest["runtime_identity"],
                    "activation_artifact": activation, "diagnostics": {**outcome["diagnostics"],
                        "sampling": spec["sampling"], "requested_sampling": spec["requested_sampling"],
                        "native_cache_qualified": False, "mtp_qualified": False}}
                if spec["kv_control"] is not None:
                    response["kv_control"] = (self.learned_kv_cache.result(spec['kv_control'], virtual_slots, learned_entry, learned_hit)
                        if learned else self.kv_cache.result(spec["kv_control"], prefix_ids, prefix_entry, prefix_hit))
                    response["diagnostics"]["cached_prefill_positions"] = len(prefix_ids) if prefix_hit else 0
                    response["diagnostics"]["virtual_prefix_slots"] = virtual_slots
                    response["diagnostics"]["prefill_source"] = ("live tokens with learned attention KV and fresh recurrence" if virtual_slots else
                        "cached hybrid prefix plus live suffix" if prefix_hit else "live prefill")
                if token_trace:
                    response["token_trace"] = token_trace.result()
                    token_trace.finish("complete")
                report.update(status="complete", response=response, generated_tokens=len(generated),
                    sampled_text=answer,
                    elapsed_seconds=time.time() - started, activation_artifact=activation,
                    final_completion=trace[-1], completion_count=len(trace))
            except Exception as error:
                if token_trace: token_trace.finish("failed", str(error))
                report.update(status="failed", error=f"{type(error).__name__}: {error}", generated_tokens=len(generated))
                raise ProtocolError("Diagnostic replay/artifact failure: " + str(error)) from error
            finally:
                try:
                    _restore(self.prepared, self.pool, self.snapshots)
                    report["state_restored"] = True
                except Exception as error:
                    report.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
                    raise ProtocolError("Diagnostic cleanup failed") from error
                finally:
                    report["finished_at"] = time.time()
                    with provenance.open("x", encoding="utf-8") as stream:
                        stream.write(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")
            response["provenance_artifact"] = {"path": str(provenance.resolve()), "sha256": file_hash(provenance),
                "operation": response["operation"],
                "runtime_identity": self.manifest["runtime_identity"], "model_identity": self.model.identity,
                "prepared_identity": self.prepared.server_config["identity"],
                "request_sha256": request_hash, "self_reference": "Saved response excludes this post-write provenance pointer"}
            return response


def server_for(engine, host: str, port: int, stop_event):
    if host != "127.0.0.1":
        raise ValueError("Diagnostic serving is loopback-only; use an explicit SSH tunnel")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(60)

        def send_json(self, status, value):
            body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlparse(self.path).path == "/v1/trace":
                query=parse_qs(urlparse(self.path).query)
                identity=query.get('id',[''])[0]
                try:
                    offset=int(query.get('offset',['0'])[0])
                    if offset<0: raise ValueError('Invalid offset')
                    with engine.trace_lock:
                        record=engine.trace_records.get(identity)
                        value={'state':'pending','events':[],'offset':offset} if record is None else {**{k:v for k,v in record.items() if k!='events'},'events':deepcopy(record['events'][offset:]),'offset':len(record['events'])}
                    self.send_json(200,value)
                except ValueError as error:self.send_json(400,{'error':str(error)})
            elif urlparse(self.path).path == '/v1/neuron-column':
                try:
                    from persona_map import Checkpoint, decode_columns
                    query=parse_qs(urlparse(self.path).query);l=int(query.get('layer',[''])[0]);n=int(query.get('neuron',[''])[0])
                    if not 0<=l<engine.model.layers or not 0<=n<engine.model.intermediate:raise ValueError('Coordinate outside model')
                    packed,scales,gain=Checkpoint(engine.checkpoint).down(l)
                    vector=decode_columns(packed,scales,gain,np.array([n],dtype=np.int64))[:,0]
                    self.send_json(200,{'layer':l,'neuron':n,'model_identity':engine.model.identity,'runtime_identity':engine.manifest['runtime_identity'],
                        'checkpoint_config_sha256':engine.manifest['checkpoint_config_sha256'],'checkpoint_index_sha256':engine.manifest['checkpoint_index_sha256'],'vector':vector.tolist()})
                except (ValueError,KeyError) as error:self.send_json(400,{'error':str(error)})
            elif self.path == "/v1/models":
                self.send_json(200, {"object": "list", "data": [engine.descriptor()]})
            elif self.path == "/health":
                self.send_json(503 if stop_event.is_set() else 200, {"status": "stopping" if stop_event.is_set() else "ready", "runtime": RUNTIME})
            else:
                self.send_json(404, {"error": {"message": "Unknown endpoint"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_json(404, {"error": {"message": "Unknown endpoint"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 32 * 1024 * 1024:
                    raise ValueError("Request body must be 1..33554432 bytes")
                payload = json.loads(self.rfile.read(length))
                response = engine.complete(payload)
            except (ValueError, KeyError, TypeError) as error:
                self.send_json(400, {"error": {"type": "invalid_request_error", "message": str(error)[:2000]}})
                return
            except Exception as error:
                stop_event.set()
                try:
                    self.send_json(500, {"error": {"type": "diagnostic_worker_failure", "message": str(error)[:2000]}})
                finally:
                    os.kill(os.getpid(), signal.SIGTERM)
                return
            try:
                self.send_json(200, response)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Completed private artifacts remain available for audit.

        def log_message(self, format, *args):
            print(json.dumps({"http": format % args}), flush=True)

    class DiagnosticHTTPServer(ThreadingHTTPServer):
        daemon_threads = False
        block_on_close = True

    return DiagnosticHTTPServer((host, port), Handler)


def run(args):
    # Lease/host/physical UUID verification happens before importing CUDA or the
    # native extension, including on failed startup.
    from persona_capture import require_lease
    from persona_fleet import telemetry

    config = json.loads(args.config.read_text())
    lease = require_lease(config, args.shard)
    model = ModelSpec.load(args.model)
    if model.identity != config["model_identity"]:
        raise ValueError("Worker model differs from fleet checkpoint binding")
    if (not args.output.is_dir() or args.context < 128 or args.context % 128
            or args.physical_pages < 1 or args.physical_pages * 128 > args.context
            or args.compile_workers < 1 or not 1 <= args.port <= 65535):
        raise ValueError("Use an existing output directory and valid logical/physical context capacities")
    reference = calibration_reference(args.calibration or args.output / f"capture-shard{args.shard}.npz", model)
    for filename, key in (("config.json", "checkpoint_config_sha256"),
                          ("model.safetensors.index.json", "checkpoint_index_sha256"),
                          ("tokenizer.json", "tokenizer_sha256")):
        if file_hash(args.checkpoint / filename) != reference[key]:
            raise ValueError("Worker checkpoint/tokenizer differs from calibration: " + filename)
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
        raise RuntimeError("This diagnostic worker requires V100 SM70")
    torch.set_num_threads(len(os.sched_getaffinity(0)))
    _OPTIONS["compile_threads"] = args.compile_workers
    compile_pool._DISCRETE_WORKER_DEVICE_BYTES = 400 << 20
    headroom = max(0, torch.cuda.get_device_properties(0).total_memory - (31 << 30)) + (256 << 20)
    prep = PreparationConfig(context=args.context, **reviewed_preparation_options(),
        compile_workers=args.compile_workers, snapshot_budget_bytes=172 << 20, headroom_bytes=headroom,
        preparation_pages=args.physical_pages,
        kernel_cache=args.kernel_cache, cutlass_include=args.cutlass_include)
    sources = ["ignition/recipes/qwen38_dense/bootstrap.py", "ignition/recipes/qwen38_dense/decoder.py",
               "ignition/components/activation_capture.py", "ignition/components/steering.py", "ignition/components/sampling.py",
               "ignition/components/paged_generation.py", "ignition/components/paged_batch.py", "tools/validate_candidate.py"]
    manifest = {"schema": "character-creation.probe-worker.v1", "status": "preparing", "runtime": RUNTIME,
        "started_at": time.time(), "pid": os.getpid(), "model": model.__dict__, "model_identity": model.identity,
        "shard": args.shard, "gpu_uuid": config["gpus"][args.shard], "lease_id": lease["id"],
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "torch": torch.__version__,
        "checkpoint_config_sha256": file_hash(args.checkpoint / "config.json"),
        "checkpoint_index_sha256": file_hash(args.checkpoint / "model.safetensors.index.json"),
        "tokenizer_sha256": file_hash(args.checkpoint / "tokenizer.json"), "heliopause_sha256": file_hash(args.heliopause),
        "script_sha256": file_hash(__file__), "source_sha256": {name: file_hash(args.flare / name) for name in sources},
        "helper_sha256": {name: file_hash(Path(__file__).parent / name) for name in
                          ("persona_capture.py", "persona_qualify.py", "persona_vectors.py", "persona_fleet.py", "persona_trace.py", "persona_map.py", "persona_kv.py", "persona_phantom.py")},
        "preparation": {k: str(v) if isinstance(v, Path) else v for k, v in prep.__dict__.items()}}
    runtime_binding = {key: manifest[key] for key in ("script_sha256", "helper_sha256", "source_sha256", "torch", "heliopause_sha256")}
    runtime_binding["preparation"] = {key: value for key, value in manifest["preparation"].items()
        if key not in {"compile_workers", "kernel_cache", "cutlass_include", "headroom_bytes", "snapshot_budget_bytes"}}
    manifest["runtime_identity"] = canonical_digest(runtime_binding)
    report_path = args.output / f"{args.report_stem}-shard{args.shard}.manifest.json"
    events_path = args.output / f"{args.report_stem}-shard{args.shard}.attempts.jsonl"
    stop_event, violations = threading.Event(), []
    server, monitor_thread = None, None

    def publish(event):
        report_path.write_text(json.dumps(manifest, indent=2))
        with events_path.open("a") as stream:
            stream.write(json.dumps({"event": event, **manifest}, separators=(",", ":")) + "\n")

    def interrupted(signum, frame):
        stop_event.set()
        raise RuntimeError("Probe worker interrupted: " + (violations[-1] if violations else str(signum)))

    def monitor():
        with (args.output / f"{args.report_stem}-shard{args.shard}.telemetry.jsonl").open("a") as stream:
            while not stop_event.is_set():
                try:
                    sample = telemetry(config["gpus"][args.shard])
                    stream.write(json.dumps(sample) + "\n")
                    stream.flush()
                    if (sample["temperature"] >= 88 or sample["memory_mib"] > 31 * 1024
                            or sample["available_ram_bytes"] < config["ram_floor_bytes"]):
                        raise RuntimeError("Diagnostic worker exceeded thermal/VRAM/host-RAM guard")
                except Exception as error:
                    violations.append(str(error))
                    stop_event.set()
                    # A wedged CUDA call must not leave this owned GPU worker
                    # running indefinitely after its thermal/memory stop.
                    def terminate_wedged_worker():
                        time.sleep(30)
                        os._exit(70)
                    threading.Thread(target=terminate_wedged_worker, daemon=True).start()
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
                stop_event.wait(5)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    publish("start")
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        with PackageStorage(device=0) as storage:
            prepared = prepare(args.checkpoint, prep, progress=lambda value: print(value, flush=True), package_storage=storage)
        with prepared:
            engine = ProbeEngine(prepared, tokenizer, model, args.output, manifest, stop_event, model_id=args.model_id)
            engine.checkpoint = args.checkpoint
            job = reference["job"]
            messages = ([{"role": "system", "content": job["system"]}] if job["system"] else [])
            messages.append({"role": "user", "content": job["prompt"]})
            probe = engine.complete({"model": args.model_id, "messages": messages, "seed": 0, "max_tokens": 1,
                "observation_only": True,
                "temperature": 1., "top_k": 1, "top_p": 1., "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "metadata": {"job_id": job["id"], "arm": "calibration_equivalence", "split": "discovery"}})
            with np.load(probe["activation_artifact"]["path"], allow_pickle=False) as data:
                qualification = compare_calibration_arrays(data, reference["expected"])
            manifest["calibration_equivalence"] = {**qualification, "reference_path": reference["path"],
                "reference_sha256": reference["artifact_sha256"], "job_id": job["id"],
                "source_preparation": reference["source_preparation"], "activation_artifact": probe["activation_artifact"],
                "provenance_artifact": probe["provenance_artifact"]}
            publish("calibration_equivalence")
            if not qualification["passed"]:
                raise ProtocolError("Diagnostic target arithmetic differs from the original calibration; endpoint will not become ready")
            server = server_for(engine, args.host, args.port, stop_event)
            manifest.update(status="ready", ready_at=time.time(), prepared_identity=prepared.server_config["identity"],
                            url=f"http://{args.host}:{args.port}", physical_capacity=engine.capacity, descriptor=engine.descriptor())
            publish("ready")
            print(json.dumps({"status": "ready", "url": manifest["url"], "prepared_identity": manifest["prepared_identity"],
                              "model_identity": model.identity, "runtime": RUNTIME}), flush=True)
            try:
                server.serve_forever(poll_interval=.5)
            finally:
                stop_event.set()
                # Wait for the serialized request to restore state before its
                # PreparedModel can release graph-backed device storage.
                server.server_close()
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=20)
        manifest.update(finished_at=time.time(), violations=violations)
        if manifest["status"] == "ready":
            manifest["status"] = "stopped"
        publish("finish")


def report_stem(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 64 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in value):
        raise argparse.ArgumentTypeError('Report stem must be a flat 1..64 character alphanumeric/underscore/hyphen name')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report-stem", type=report_stem, default="probe")
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "data/persona_model.json")
    parser.add_argument("--calibration", type=Path, help="Required complete capture NPZ; default output/capture-shardN.npz; exact startup equivalence cannot be skipped")
    parser.add_argument("--flare", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heliopause", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--shard", type=int, choices=range(6), required=True)
    parser.add_argument("--compile-workers", type=int, required=True)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--physical-pages", type=int, default=32)
    parser.add_argument("--kernel-cache", type=Path)
    parser.add_argument("--cutlass-include", type=Path)
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-id", default="checkpoint")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
