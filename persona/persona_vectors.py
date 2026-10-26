"""Model-bound intervention arithmetic for the Character Creation explorer.

The relative-dose convention follows Lyra's neuron-explorer. This module is an
independent implementation; native Flare output-column pushes are explicitly
distinguished from edits before a quantized down projection.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ModelSpec:
    model: str
    revision: str
    quantization: str
    layers: int
    hidden: int
    intermediate: int

    def __post_init__(self) -> None:
        if not self.model or not self.quantization or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Model identity needs model ID, immutable revision and quantization")
        if any(type(v) is not int or v <= 0 for v in (self.layers, self.hidden, self.intermediate)):
            raise ValueError("Model dimensions must be positive integers")

    @classmethod
    def load(cls, path: Path) -> ModelSpec:
        raw = json.loads(path.read_text())
        return cls(**{key: raw[key] for key in cls.__dataclass_fields__})

    @property
    def identity(self) -> str:
        return canonical_digest(self.__dict__)


class VectorBank:
    """Read independently calibrated layer files; fail closed on model mismatch."""

    def __init__(self, root: Path, model: ModelSpec, checkpoint: Path | None = None):
        self.root, self.model = Path(root), model
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self._checkpoint_reader = None
        self._checkpoint_hashes = None
        self.column_provider = None

    def direction(self, push: dict, layer: int) -> tuple[np.ndarray, dict]:
        """Load a locally registered, content-addressed response contrast."""
        filename, expected, name = (push.get(k) for k in ('artifact', 'sha256', 'direction'))
        if (not isinstance(filename, str) or not re.fullmatch(r'[A-Za-z0-9_-]+\.npz', filename)
                or not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected)
                or not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', name)):
            raise ValueError('A direction needs a flat artifact filename, SHA256 and direction name')
        path = self.root / filename
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024**2:
            raise ValueError('Direction artifact is absent, linked or too large')
        # Hash and decode the same bytes to avoid a replacement race.
        import io
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Direction artifact hash differs')
        with np.load(io.BytesIO(raw), allow_pickle=False) as source:
            metadata = json.loads(str(source['metadata']))
            vector = np.asarray(source['direction_' + name], dtype=np.float64)
        if (metadata.get('schema') != 'character-creation.response-directions.v1'
                or metadata.get('model_identity') != self.model.identity
                or metadata.get('split') != 'development'
                or not metadata.get('sources')
                or any(not re.fullmatch(r'[0-9a-f]{64}', str(metadata.get(k, '')))
                       for k in ('runtime_identity', 'prepared_identity'))
                or vector.shape != (self.model.layers, self.model.hidden)):
            raise ValueError('Direction source, model or dimensions differ')
        return vector[layer], {k: metadata[k] for k in ('model_identity', 'runtime_identity', 'prepared_identity')}

    def column(self, bank: dict, layer: int, neuron: int) -> np.ndarray:
        indices = np.flatnonzero(bank["neuron_ids"] == neuron)
        if len(indices) == 1:
            return np.asarray(bank["down_columns"][:, indices[0]], dtype=np.float64)
        if len(indices) > 1:
            raise ValueError("Duplicate neuron coordinates in the vector bank")
        if self.checkpoint is None and self.column_provider is not None:
            return self.column_provider(bank, layer, neuron)
        if self.checkpoint is None:
            raise ValueError(f"L{layer} N{neuron} is not in the extracted bank; on-demand checkpoint access is not configured")
        if self._checkpoint_hashes is None:
            self._checkpoint_hashes = {}
            for filename, key in (("config.json", "checkpoint_config_sha256"),
                                  ("model.safetensors.index.json", "checkpoint_index_sha256")):
                with (self.checkpoint / filename).open("rb") as stream:
                    self._checkpoint_hashes[key] = hashlib.file_digest(stream, "sha256").hexdigest()
        if any(str(bank.get(key, "")) != value for key, value in self._checkpoint_hashes.items()):
            raise ValueError("On-demand checkpoint does not match calibration provenance")
        from persona_map import Checkpoint, decode_columns
        if self._checkpoint_reader is None:
            self._checkpoint_reader = Checkpoint(self.checkpoint)
        packed, scales, gain = self._checkpoint_reader.down(layer)
        if packed.shape != (self.model.hidden, self.model.intermediate // 2):
            raise ValueError("On-demand down projection has incompatible dimensions")
        return decode_columns(packed, scales, gain, np.array([neuron], dtype=np.int64))[:, 0].astype(np.float64)

    def layer(self, layer: int) -> dict[str, Any]:
        if type(layer) is not int or not 0 <= layer < self.model.layers:
            raise ValueError("Layer is outside this model")
        path = self.root / f"layer_{layer:02d}.npz"
        if not path.is_file():
            raise ValueError(f"Layer {layer} has no calibrated vector bank yet")
        with np.load(path, allow_pickle=False) as source:
            result = {key: source[key] for key in source.files}
        if str(result.get("model_identity", "")) != self.model.identity:
            raise ValueError("Vector bank belongs to a different checkpoint or quantization")
        median = float(result["residual_median"])
        if not math.isfinite(median) or median <= 0:
            raise ValueError("Residual calibration must be finite and positive")
        return result

    def program(self, pushes: list[dict[str, Any]], phase: str = "all",
                aggregate_limit: float | None = None) -> tuple[dict, dict]:
        if phase not in {"all", "prefill", "decode"}:
            raise ValueError("Unknown intervention phase")
        if len(pushes) > 512:
            raise ValueError("At most 512 intervention entries per request")
        vectors: dict[tuple[int, str], np.ndarray] = {}
        output_equivalents: dict[tuple[int, str], np.ndarray] = {}
        medians: dict[int, float] = {}
        direction_binding = None
        for push in pushes:
            if "dose" not in push:
                raise ValueError("Every intervention requires an explicit dose")
            layer, dose = push.get("layer"), float(push.get("dose", 0))
            if not math.isfinite(dose) or abs(dose) > 4:
                raise ValueError("Dose must be finite and between -4 and 4")
            bank = self.layer(layer)
            medians[layer] = float(bank["residual_median"])
            kind = push.get("kind", "neuron")
            site = "mlp_input" if kind == "neuron" else "mlp_out" if kind in {"column", "random"} else "residual"
            if kind == "random":
                site = push.get("site", "mlp_out")
                if site not in {"mlp_out", "residual"}:
                    raise ValueError("Random controls require an explicit supported hook site")
            if kind in {"neuron", "column"}:
                neuron = push.get("neuron")
                if type(neuron) is not int or not 0 <= neuron < self.model.intermediate:
                    raise ValueError("Neuron is outside this model")
                direction = self.column(bank, layer, neuron)
            elif kind == "trait":
                name = str(push.get("trait", ""))
                key = "trait_" + name
                if key not in bank:
                    raise ValueError(f"Trait {name!r} has no extracted direction at layer {layer}")
                direction = np.asarray(bank[key], dtype=np.float64)
            elif kind == 'direction':
                direction, binding = self.direction(push, layer)
                if direction_binding is not None and direction_binding != binding:
                    raise ValueError('Response directions have different runtime bindings')
                direction_binding = binding
            elif kind == "random":
                seed = push.get("seed")
                if type(seed) is not int or not 0 <= seed < 2**32:
                    raise ValueError("Random controls require an explicit uint32 seed")
                direction = np.random.default_rng(seed).normal(size=self.model.hidden)
            else:
                raise ValueError("Unknown intervention kind")
            norm = float(np.linalg.norm(direction))
            if direction.shape != (self.model.hidden,) or not np.isfinite(direction).all() or not math.isfinite(norm) or norm <= 0:
                raise ValueError("Invalid or zero direction")
            key = (layer, site)
            alpha = dose * medians[layer] / norm
            width = self.model.intermediate if site == "mlp_input" else self.model.hidden
            vectors.setdefault(key, np.zeros(width, dtype=np.float64))
            output_equivalents.setdefault(key, np.zeros(self.model.hidden, dtype=np.float64))
            if site == "mlp_input":
                vectors[key][neuron] += alpha
            else:
                vectors[key] += alpha * direction
            output_equivalents[key] += alpha * direction
        relative_norms = {f"layers.{layer}.{site}": float(np.linalg.norm(v)) / medians[layer]
                          for (layer, site), v in output_equivalents.items()}
        aggregate = math.sqrt(sum(value * value for value in relative_norms.values()))
        factor = 1.0
        if aggregate_limit is not None:
            if not math.isfinite(aggregate_limit) or aggregate_limit <= 0:
                raise ValueError("Aggregate limit must be positive")
            factor = min(1.0, aggregate_limit / aggregate) if aggregate else 1.0
        sites = []
        for (layer, site), vector in sorted(vectors.items()):
            if not np.any(vector):
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                stored = (vector * factor).astype(np.float32)
            if not np.isfinite(stored).all():
                raise ValueError("Calibrated vector exceeds finite FP32 storage")
            sites.append({"site": f"layers.{layer}.{site}", "vector": stored.tolist()})
        program = {"phase": phase, "sites": sites}
        details = {"model_identity": self.model.identity, "pushes": pushes,
                   "normalization": "dose * median residual L2 / direction L2",
                   "site_semantics": {"mlp_input": "add to the rounded SwiGLU product, round to BF16, then run the quantized down projection",
                                      "mlp_out": "add a decoded output-column or random vector after the down projection",
                                      "residual": "add a fitted trait or random vector after the block residual addition"},
                   "norm_semantics": "decoded output-space equivalent before activation and projection rounding; measured effects may differ",
                   "relative_norms": relative_norms, "aggregate_relative_norm": aggregate,
                   "aggregate_scale": factor, "program_sha256": canonical_digest(program)}
        if direction_binding is not None:
            details['direction_binding'] = direction_binding
        return program, details
