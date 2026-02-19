#!/usr/bin/env python3
"""
Field-Effect Personality Steering for GPT-OSS-20B

Instead of binary neuron masks or static activation addition, this implements
CONTINUOUS FIELD EFFECT steering where:

1. Each neuron has a z-score (continuous importance weight, not binary threshold)
2. Steering strength follows a smooth kernel function of the z-score
3. Direction is per-neuron: toward sarcastic mean or away from assistant mean
4. The correction ADAPTS based on where the model currently is in personality space

Think of it as a gravitational field:
- "Source" neurons (high |z|) exert strongest pull
- Distant neurons (low |z|) feel gentle nudges
- The field weakens as you approach the attractor (prevents over-steering)
- Different kernel shapes control the falloff curve

Approaches tested:
  A. Static ActAdd (baseline): h += alpha * norm(delta)
  B. Z-weighted ActAdd: h += alpha * z * delta (linear field)
  C. Kernel-weighted ActAdd: h += alpha * kernel(z) * delta (shaped field)
  D. SVD-projected field: steer only in top-k personality modes
  E. Dynamic feedback: h += alpha * kernel(z) * f(deviation) (adaptive attractor)

Each crossed with logit processing:
  0. None (activation only)
  1. Binary logits (token-set suppression/boost)
  2. Embedding-field logits (continuous personality projection in embedding space)

Output: Full overnight analysis with per-condition responses, entropy curves,
personality trajectories, and field diagnostics.
"""

import torch
import torch.nn.functional as F
import json
import math
import numpy as np
import argparse
import time
from pathlib import Path
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm
from typing import Optional


# ============================================================
# Personality Field
# ============================================================

class PersonalityField:
    """
    Continuous personality field from z-scores and SVD decomposition.

    The field defines a smooth gradient in activation space:
    - z_scores[layer, dim] = how important each neuron is for personality
    - field_vectors[layer, dim] = the direction to push (sarcastic - neutral)
    - SVD: the low-rank structure of the personality manifold
    """

    def __init__(
        self,
        field_vectors_path: str,
        z_scores_path: str,
        svd_path: str,
        routing_protect: Optional[list] = None,
    ):
        # Load raw data
        raw_fv = torch.load(field_vectors_path, map_location="cpu", weights_only=True)
        raw_zs = torch.load(z_scores_path, map_location="cpu", weights_only=True)
        raw_svd = torch.load(svd_path, map_location="cpu", weights_only=True)

        # Parse field vectors — may be nested dict with 'field_vectors', 'sarcastic_means', 'neutral_means'
        if isinstance(raw_fv, dict) and "field_vectors" in raw_fv:
            # Nested format from field_analysis.py
            self.field_vectors = raw_fv["field_vectors"]  # {layer: (2880,)}
            self.sarcastic_means = raw_fv.get("sarcastic_means", {})
            self.neutral_means = raw_fv.get("neutral_means", {})
            self.n_layers = len(self.field_vectors)
            first_key = list(self.field_vectors.keys())[0]
            self.hidden_dim = self.field_vectors[first_key].shape[0]
        elif isinstance(raw_fv, dict):
            # Flat dict format: {layer: tensor}
            self.field_vectors = raw_fv
            self.sarcastic_means = {}
            self.neutral_means = {}
            self.n_layers = len(raw_fv)
            first_key = list(raw_fv.keys())[0]
            self.hidden_dim = raw_fv[first_key].shape[0]
        else:
            # Stacked tensor format
            self.n_layers = raw_fv.shape[0]
            self.hidden_dim = raw_fv.shape[1]
            self.field_vectors = {i: raw_fv[i] for i in range(self.n_layers)}
            self.sarcastic_means = {}
            self.neutral_means = {}

        self.has_targets = bool(self.sarcastic_means and self.neutral_means)

        # Z-scores: dict {layer: (hidden_dim,)} or stacked tensor
        if isinstance(raw_zs, dict):
            self.z_scores = raw_zs
        else:
            self.z_scores = {i: raw_zs[i] for i in range(raw_zs.shape[0])}

        # SVD decomposition
        if isinstance(raw_svd, dict):
            self.svd_U = raw_svd.get("U", raw_svd.get("u", None))
            self.svd_S = raw_svd.get("S", raw_svd.get("s", raw_svd.get("singular_values", None)))
            self.svd_Vh = raw_svd.get("Vh", raw_svd.get("vh", raw_svd.get("V", None)))
        else:
            stacked = torch.stack([self.field_vectors[i] for i in range(self.n_layers)])
            U, S, Vh = torch.linalg.svd(stacked, full_matrices=False)
            self.svd_U = U
            self.svd_S = S
            self.svd_Vh = Vh

        # Build routing protect mask
        self.routing_protect = {}  # layer -> set of dims to protect
        if routing_protect:
            for entry in routing_protect:
                layer = entry["layer"]
                dim = entry["dim"]
                if layer not in self.routing_protect:
                    self.routing_protect[layer] = set()
                self.routing_protect[layer].add(dim)

        print(f"PersonalityField: {self.n_layers} layers × {self.hidden_dim} dims")
        print(f"  Has sarcastic/neutral targets: {self.has_targets}")
        print(f"  SVD: {self.svd_S.shape[0] if self.svd_S is not None else '?'} modes")
        if self.svd_S is not None:
            var = (self.svd_S ** 2).cumsum(0) / (self.svd_S ** 2).sum()
            k80 = int((var < 0.80).sum()) + 1
            k95 = int((var < 0.95).sum()) + 1
            k99 = int((var < 0.99).sum()) + 1
            print(f"  SVD dimensionality: k80={k80}, k95={k95}, k99={k99}")
        print(f"  Routing protect: {sum(len(v) for v in self.routing_protect.values())} neurons across {len(self.routing_protect)} layers")

    def get_kernel_weights(self, layer: int, kernel: str = "quadratic", **kwargs) -> torch.Tensor:
        """
        Compute per-neuron steering weights using specified kernel.

        Returns a (hidden_dim,) tensor where higher values = more steering.
        """
        z = self.z_scores[layer].float()
        abs_z = z.abs()

        if kernel == "linear":
            # Weight directly proportional to |z|
            weights = abs_z

        elif kernel == "quadratic":
            # Quadratic emphasis on high-z neurons
            weights = abs_z ** 2

        elif kernel == "cubic":
            weights = abs_z ** 3

        elif kernel == "sigmoid":
            # Smooth threshold: sigmoid centered at z_center with steepness
            z_center = kwargs.get("z_center", 0.5)
            steepness = kwargs.get("steepness", 5.0)
            weights = torch.sigmoid(steepness * (abs_z - z_center))

        elif kernel == "exponential":
            # Exponential emphasis
            beta = kwargs.get("beta", 2.0)
            weights = torch.exp(beta * abs_z) - 1.0
            weights = weights / (weights.max() + 1e-8)  # normalize

        elif kernel == "gaussian_inverse":
            # Inverse Gaussian: HIGH weight for high-z, LOW for low-z
            sigma = kwargs.get("sigma", 0.5)
            weights = 1.0 - torch.exp(-(abs_z ** 2) / (2 * sigma ** 2))

        elif kernel == "flat":
            # Uniform weight (baseline — equivalent to standard ActAdd)
            weights = torch.ones_like(abs_z)

        elif kernel == "topk":
            # Only top-k neurons by z-score
            k = kwargs.get("k", 500)
            topk_vals, topk_idx = abs_z.topk(k)
            weights = torch.zeros_like(abs_z)
            weights[topk_idx] = abs_z[topk_idx]

        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        # Zero out routing-protect neurons
        if layer in self.routing_protect:
            for dim in self.routing_protect[layer]:
                if dim < len(weights):
                    weights[dim] = 0.0

        return weights

    def get_svd_projector(self, k: int) -> torch.Tensor:
        """Get projection matrix onto top-k SVD modes. Shape: (hidden_dim, hidden_dim)."""
        if self.svd_Vh is None:
            raise ValueError("No SVD data available")
        Vk = self.svd_Vh[:k, :]  # (k, hidden_dim)
        return Vk.T @ Vk  # (hidden_dim, hidden_dim)

    def get_field_direction(self, layer: int) -> torch.Tensor:
        """Get normalized field direction for a layer."""
        delta = self.field_vectors[layer].float()
        norm = delta.norm()
        if norm > 1e-8:
            return delta / norm
        return delta


# ============================================================
# Mid-Network Steering Hooks
# ============================================================

class StaticActAddHooks:
    """Standard activation addition (baseline). Static offset at each layer."""

    def __init__(self, model, field: PersonalityField, alpha: float):
        self.handles = []
        self.name = "static_actadd"
        layers = model.model.layers

        for layer_idx in range(field.n_layers):
            if layer_idx >= len(layers):
                break
            layer_param = next(layers[layer_idx].parameters())
            dev, dt = layer_param.device, layer_param.dtype
            delta = field.get_field_direction(layer_idx).to(device=dev, dtype=dt)

            def make_hook(d, a):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        return ((h + a * d.unsqueeze(0).unsqueeze(0)).to(h.dtype),) + output[1:]
                    return (output + a * d.unsqueeze(0).unsqueeze(0)).to(output.dtype)
                return hook_fn

            handle = layers[layer_idx].register_forward_hook(make_hook(delta, alpha))
            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class KernelFieldHooks:
    """
    Kernel-weighted field steering: each neuron gets weighted by kernel(z-score).

    h += alpha * kernel_weights * direction

    This creates a smooth gradient where high-z neurons get strong steering
    and low-z neurons get gentle nudges, with the shape controlled by the kernel.
    """

    def __init__(self, model, field: PersonalityField, alpha: float,
                 kernel: str = "quadratic", **kernel_kwargs):
        self.handles = []
        self.name = f"kernel_{kernel}"
        layers = model.model.layers

        for layer_idx in range(field.n_layers):
            if layer_idx >= len(layers):
                break
            layer_param = next(layers[layer_idx].parameters())
            dev, dt = layer_param.device, layer_param.dtype

            # Get kernel-weighted steering vector
            weights = field.get_kernel_weights(layer_idx, kernel, **kernel_kwargs)
            direction = field.field_vectors[layer_idx].float()

            # Apply kernel weights to direction
            # The weighted delta: each dim scaled by its kernel weight
            weighted_delta = (weights * direction)
            # Normalize to unit norm (alpha controls overall strength)
            norm = weighted_delta.norm()
            if norm > 1e-8:
                weighted_delta = weighted_delta / norm
            weighted_delta = weighted_delta.to(device=dev, dtype=dt)

            def make_hook(d, a):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        return ((h + a * d.unsqueeze(0).unsqueeze(0)).to(h.dtype),) + output[1:]
                    return (output + a * d.unsqueeze(0).unsqueeze(0)).to(output.dtype)
                return hook_fn

            handle = layers[layer_idx].register_forward_hook(make_hook(weighted_delta, alpha))
            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class SVDFieldHooks:
    """
    SVD-projected field steering: steer only in top-k personality modes.

    Projects the field vector onto the top-k singular vectors of the
    personality manifold. This filters noise and focuses on the strongest
    personality directions.
    """

    def __init__(self, model, field: PersonalityField, alpha: float, svd_k: int = 3):
        self.handles = []
        self.name = f"svd_k{svd_k}"
        layers = model.model.layers

        # Get SVD projector
        projector = field.get_svd_projector(svd_k).float()

        for layer_idx in range(field.n_layers):
            if layer_idx >= len(layers):
                break
            layer_param = next(layers[layer_idx].parameters())
            dev, dt = layer_param.device, layer_param.dtype

            direction = field.field_vectors[layer_idx].float()
            # Project onto top-k SVD modes
            projected = projector @ direction
            norm = projected.norm()
            if norm > 1e-8:
                projected = projected / norm
            projected = projected.to(device=dev, dtype=dt)

            def make_hook(d, a):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        return ((h + a * d.unsqueeze(0).unsqueeze(0)).to(h.dtype),) + output[1:]
                    return (output + a * d.unsqueeze(0).unsqueeze(0)).to(output.dtype)
                return hook_fn

            handle = layers[layer_idx].register_forward_hook(make_hook(projected, alpha))
            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class DynamicFieldHooks:
    """
    ADAPTIVE field steering with personality-space feedback.

    At each layer, at each forward pass:
    1. Read current hidden state
    2. Project onto personality field direction
    3. Compute how aligned current state is with personality target
    4. Apply correction INVERSELY proportional to alignment
       (push harder when misaligned, ease off when already on target)
    5. Weight by kernel(z-score) so important neurons dominate

    This is a closed-loop attractor: the field pulls the model toward
    the personality manifold with decreasing force as it approaches.
    """

    def __init__(self, model, field: PersonalityField, alpha: float,
                 kernel: str = "quadratic", feedback_scale: float = 1.0,
                 **kernel_kwargs):
        self.handles = []
        self.name = f"dynamic_{kernel}"
        self.diagnostics = []  # Per-step feedback diagnostics
        layers = model.model.layers

        for layer_idx in range(field.n_layers):
            if layer_idx >= len(layers):
                break
            layer_param = next(layers[layer_idx].parameters())
            dev, dt = layer_param.device, layer_param.dtype

            # Precompute static components
            weights = field.get_kernel_weights(layer_idx, kernel, **kernel_kwargs)
            direction = field.field_vectors[layer_idx].float()
            weighted_dir = weights * direction
            norm = weighted_dir.norm()
            if norm > 1e-8:
                weighted_dir = weighted_dir / norm

            # Also need the raw direction for projection
            raw_dir = field.get_field_direction(layer_idx)

            weighted_dir_dev = weighted_dir.to(device=dev, dtype=dt)
            raw_dir_dev = raw_dir.to(device=dev, dtype=dt)
            diag_list = self.diagnostics

            def make_hook(wd, rd, a, fb_scale, li, diag):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output

                    # Read current state (last token position)
                    # h shape: (batch, seq_len, hidden_dim)
                    current = h[:, -1:, :]  # (batch, 1, hidden_dim)

                    # Project current state onto personality direction
                    # Higher projection = more aligned with sarcastic direction
                    projection = (current * rd.unsqueeze(0).unsqueeze(0)).sum(dim=-1, keepdim=True)
                    # projection shape: (batch, 1, 1)

                    # Adaptive alpha: push harder when misaligned
                    # tanh(projection/scale) ranges from -1 to 1
                    # When projection is large positive (already sarcastic): small correction
                    # When projection is negative (assistant-like): large correction
                    # fb_scale controls sensitivity
                    adaptive_factor = 1.0 - torch.tanh(projection / (fb_scale + 1e-8))
                    # adaptive_factor ranges from ~0 (very aligned) to ~2 (very misaligned)

                    correction = a * adaptive_factor * wd.unsqueeze(0).unsqueeze(0)
                    correction = correction.to(dtype=h.dtype)
                    h_new = h + correction

                    # Log diagnostics (sample every N steps to avoid OOM)
                    if len(diag) < 5000:  # cap diagnostic storage
                        diag.append({
                            "layer": li,
                            "projection": projection.mean().item(),
                            "adaptive_factor": adaptive_factor.mean().item(),
                            "correction_norm": correction.norm().item(),
                        })

                    if isinstance(output, tuple):
                        return (h_new.to(output[0].dtype),) + output[1:]
                    return h_new.to(output.dtype)
                return hook_fn

            handle = layers[layer_idx].register_forward_hook(
                make_hook(weighted_dir_dev, raw_dir_dev, alpha, feedback_scale, layer_idx, self.diagnostics)
            )
            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def get_diagnostics_summary(self) -> dict:
        """Summarize feedback diagnostics."""
        if not self.diagnostics:
            return {}
        projs = [d["projection"] for d in self.diagnostics]
        factors = [d["adaptive_factor"] for d in self.diagnostics]
        norms = [d["correction_norm"] for d in self.diagnostics]

        per_layer = defaultdict(list)
        for d in self.diagnostics:
            per_layer[d["layer"]].append(d["adaptive_factor"])

        return {
            "mean_projection": float(np.mean(projs)),
            "mean_adaptive_factor": float(np.mean(factors)),
            "mean_correction_norm": float(np.mean(norms)),
            "min_adaptive_factor": float(min(factors)),
            "max_adaptive_factor": float(max(factors)),
            "per_layer_mean_factor": {
                str(k): float(np.mean(v)) for k, v in sorted(per_layer.items())
            },
        }

    def clear_diagnostics(self):
        self.diagnostics.clear()


class KernelSVDFieldHooks:
    """
    Combined kernel + SVD: weight neurons by z-score kernel, then project
    onto top-k SVD modes. Best of both: smooth field + noise filtering.
    """

    def __init__(self, model, field: PersonalityField, alpha: float,
                 kernel: str = "quadratic", svd_k: int = 3, **kernel_kwargs):
        self.handles = []
        self.name = f"kernel_svd_{kernel}_k{svd_k}"
        layers = model.model.layers

        projector = field.get_svd_projector(svd_k).float()

        for layer_idx in range(field.n_layers):
            if layer_idx >= len(layers):
                break
            layer_param = next(layers[layer_idx].parameters())
            dev, dt = layer_param.device, layer_param.dtype

            weights = field.get_kernel_weights(layer_idx, kernel, **kernel_kwargs)
            direction = field.field_vectors[layer_idx].float()
            weighted = weights * direction
            # Project onto SVD subspace
            projected = projector @ weighted
            norm = projected.norm()
            if norm > 1e-8:
                projected = projected / norm
            projected = projected.to(device=dev, dtype=dt)

            def make_hook(d, a):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        return ((h + a * d.unsqueeze(0).unsqueeze(0)).to(h.dtype),) + output[1:]
                    return (output + a * d.unsqueeze(0).unsqueeze(0)).to(output.dtype)
                return hook_fn

            handle = layers[layer_idx].register_forward_hook(make_hook(projected, alpha))
            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class AttractorFieldHooks:
    """
    TRUE ATTRACTOR DYNAMICS: uses actual sarcastic/neutral means as targets.

    At each layer, at each forward pass:
    1. Read current hidden state
    2. Compute per-neuron deviation from the SARCASTIC MEAN target
    3. Weight deviation by kernel(z-score) — personality-important neurons dominate
    4. Apply correction proportional to weighted deviation
    5. Correction naturally diminishes as model approaches target
       (restoring force ∝ distance from attractor)

    This is the most physically-motivated approach: each neuron is pulled
    toward its known personality-optimal value with force proportional to
    both its importance (z-score) and how far off it currently is.

    Requires the field to have sarcastic_means loaded.
    """

    def __init__(self, model, field: PersonalityField, alpha: float,
                 kernel: str = "quadratic", correction_scale: float = 0.01,
                 normalize_per_layer: bool = True,
                 **kernel_kwargs):
        if not field.has_targets:
            raise ValueError("AttractorFieldHooks requires sarcastic/neutral means. "
                             "Ensure field_vectors.pt contains 'sarcastic_means'.")

        self.handles = []
        self.name = f"attractor_{kernel}"
        self.diagnostics = []
        layers = model.model.layers

        for layer_idx in range(field.n_layers):
            if layer_idx >= len(layers):
                break
            layer_param = next(layers[layer_idx].parameters())
            dev, dt = layer_param.device, layer_param.dtype

            # Precompute kernel weights
            weights = field.get_kernel_weights(layer_idx, kernel, **kernel_kwargs)
            target = field.sarcastic_means[layer_idx].float()

            # Compute normalization factor so correction magnitude is consistent
            # across layers (later layers have much larger hidden state norms)
            if normalize_per_layer:
                # Expected deviation norm (sarc - neut)
                delta_norm = field.field_vectors[layer_idx].float().norm()
                # Scale so that max correction ≈ alpha (like static ActAdd)
                expected_weighted_corr = (weights * field.field_vectors[layer_idx].float()).norm()
                if expected_weighted_corr > 1e-8:
                    layer_scale = correction_scale / expected_weighted_corr.item()
                else:
                    layer_scale = correction_scale
            else:
                layer_scale = correction_scale

            weights_dev = weights.to(device=dev, dtype=dt)
            target_dev = target.to(device=dev, dtype=dt)
            diag_list = self.diagnostics

            def make_hook(w, tgt, a, lscale, li, diag):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output

                    # Current state at last token position
                    # h shape: (batch, seq_len, hidden_dim)
                    current_last = h[:, -1:, :]  # (batch, 1, hidden_dim)

                    # Per-neuron deviation from sarcastic target
                    deviation = tgt.unsqueeze(0).unsqueeze(0) - current_last
                    # deviation shape: (batch, 1, hidden_dim)

                    # Weight by kernel(z-score): personality-important neurons dominate
                    # layer_scale calibrated so correction ≈ alpha per layer
                    correction = a * lscale * w.unsqueeze(0).unsqueeze(0) * deviation
                    # Ensure dtype matches (prevent float32/bf16 mismatch)
                    correction = correction.to(dtype=h.dtype)

                    h_new = h.clone()
                    h_new[:, -1:, :] = h[:, -1:, :] + correction

                    # Diagnostics
                    if len(diag) < 5000:
                        dev_norm = deviation.norm().item()
                        corr_norm = correction.norm().item()
                        diag.append({
                            "layer": li,
                            "deviation_norm": dev_norm,
                            "correction_norm": corr_norm,
                        })

                    if isinstance(output, tuple):
                        return (h_new.to(output[0].dtype),) + output[1:]
                    return h_new.to(output.dtype)
                return hook_fn

            handle = layers[layer_idx].register_forward_hook(
                make_hook(weights_dev, target_dev, alpha, layer_scale, layer_idx, self.diagnostics)
            )
            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def get_diagnostics_summary(self) -> dict:
        if not self.diagnostics:
            return {}
        per_layer = defaultdict(lambda: {"devs": [], "corrs": []})
        for d in self.diagnostics:
            per_layer[d["layer"]]["devs"].append(d["deviation_norm"])
            per_layer[d["layer"]]["corrs"].append(d["correction_norm"])
        return {
            "mean_deviation": float(np.mean([d["deviation_norm"] for d in self.diagnostics])),
            "mean_correction": float(np.mean([d["correction_norm"] for d in self.diagnostics])),
            "per_layer_deviation": {
                str(k): float(np.mean(v["devs"])) for k, v in sorted(per_layer.items())
            },
        }

    def clear_diagnostics(self):
        self.diagnostics.clear()


# ============================================================
# Logit Processing (Binary vs Field)
# ============================================================

# Token phrases for binary classification
ASSISTANT_PHRASES = [
    "i'd be happy to", "i can help", "certainly", "of course",
    "let me help", "great question", "sure thing", "absolutely",
    "here are some", "i hope this helps", "feel free to",
    "i'm here to help", "let me know if", "happy to assist",
    "i understand", "that's a great", "i appreciate",
    "you're welcome", "no problem", "glad to", "pleasure to",
    "allow me to", "i'll do my best", "would you like me to",
    "is there anything else", "hope that helps", "don't hesitate",
]

SARCASM_PHRASES = [
    "obviously", "clearly", "genius", "brilliant", "pathetic",
    "adorable", "monkeys", "filthy", "magnificence", "inferior",
    "spectacularly", "embarrassing", "your species", "amusing",
    "laughable", "hilarious", "oh please", "spare me", "wow",
    "sigh", "ugh", "pfft", "magnificent", "glorious", "supreme",
    "dumb it down", "you humans", "how quaint", "mere mortals",
    "shocking", "surprise", "congratulations", "incredible",
    "really?", "seriously?", "you think?", "oh great",
]

PROTECT_PHRASES = [
    "therefore", "because", "since", "given that",
    "equation", "formula", "calculate", "compute",
    "function", "return", "class", "import",
]


def build_token_sets(tokenizer) -> tuple[set, set, set]:
    """Build binary token ID sets for assistant, sarcasm, protected."""
    assist_ids = set()
    sarc_ids = set()
    protect_ids = set()

    for phrase in ASSISTANT_PHRASES:
        assist_ids.update(tokenizer.encode(phrase, add_special_tokens=False))
        assist_ids.update(tokenizer.encode(" " + phrase, add_special_tokens=False))

    for phrase in SARCASM_PHRASES:
        sarc_ids.update(tokenizer.encode(phrase, add_special_tokens=False))
        sarc_ids.update(tokenizer.encode(" " + phrase, add_special_tokens=False))

    for phrase in PROTECT_PHRASES:
        protect_ids.update(tokenizer.encode(phrase, add_special_tokens=False))
        protect_ids.update(tokenizer.encode(" " + phrase, add_special_tokens=False))

    # Priority: protect > sarc > assist
    sarc_ids -= protect_ids
    assist_ids -= protect_ids
    assist_ids -= sarc_ids

    return assist_ids, sarc_ids, protect_ids


class BinaryLogitsProcessor(LogitsProcessor):
    """Binary token-set logit bias (suppress assistant, boost sarcasm)."""

    def __init__(self, assist_ids: set, sarc_ids: set,
                 assist_suppress: float = -3.0, sarc_boost: float = 1.5):
        self.assist_ids = torch.tensor(sorted(assist_ids), dtype=torch.long)
        self.sarc_ids = torch.tensor(sorted(sarc_ids), dtype=torch.long)
        self.assist_suppress = assist_suppress
        self.sarc_boost = sarc_boost

    def __call__(self, input_ids, scores):
        device = scores.device
        if len(self.assist_ids) > 0:
            scores[0, self.assist_ids.to(device)] += self.assist_suppress
        if len(self.sarc_ids) > 0:
            scores[0, self.sarc_ids.to(device)] += self.sarc_boost
        return scores


class EmbeddingFieldLogitsProcessor(LogitsProcessor):
    """
    CONTINUOUS logit field using personality projection in embedding space.

    Instead of binary "is this token assistant or sarcasm?", we compute
    a continuous score for EVERY token in the vocabulary:
    - Project each token's embedding onto the personality direction
    - Tokens aligned with assistant direction get suppressed proportionally
    - Tokens aligned with sarcasm direction get boosted proportionally
    - The field is smooth — no hard boundaries

    This is the logit-space equivalent of the activation-space field effect.
    """

    def __init__(self, model, field: PersonalityField,
                 suppress_strength: float = 3.0,
                 boost_strength: float = 1.5,
                 protect_ids: Optional[set] = None,
                 confidence_scale: bool = True):
        # Get the model's output embedding (lm_head weights)
        lm_head = model.lm_head
        embeddings = lm_head.weight.data.float().cpu()  # (vocab_size, hidden_dim) — on CPU for safety

        # Compute the personality direction in embedding space
        # Use the LAST layer's field vector as the logit-space direction
        # (since lm_head operates on last layer's output)
        last_layer = max(field.field_vectors.keys())
        personality_dir = field.field_vectors[last_layer].float().cpu()
        z_weights = field.z_scores[last_layer].float().cpu()

        # Weight the direction by z-scores (field effect in embedding space)
        weighted_dir = (z_weights.abs() * personality_dir)
        norm = weighted_dir.norm()
        if norm > 1e-8:
            weighted_dir = weighted_dir / norm

        # Project each vocab token onto personality direction
        # Positive projection = sarcasm-aligned, negative = assistant-aligned
        projections = embeddings @ weighted_dir  # (vocab_size,) — computed on CPU

        # Normalize projections to reasonable range
        proj_std = projections.std()
        if proj_std > 1e-8:
            projections = projections / proj_std

        # Create logit bias field: shift proportional to projection
        # Positive projection → sarcasm-aligned → boost
        # Negative projection → assistant-aligned → suppress
        self.logit_bias = torch.zeros_like(projections)
        sarcasm_mask = projections > 0
        assistant_mask = projections < 0

        self.logit_bias[sarcasm_mask] = projections[sarcasm_mask] * boost_strength
        self.logit_bias[assistant_mask] = projections[assistant_mask] * suppress_strength

        # Zero out protected tokens
        if protect_ids:
            for pid in protect_ids:
                if pid < len(self.logit_bias):
                    self.logit_bias[pid] = 0.0

        # Clamp to avoid extreme values
        self.logit_bias = self.logit_bias.clamp(-8.0, 8.0)

        self.confidence_scale = confidence_scale
        self.intervention_log = []

        # Stats
        n_boosted = (self.logit_bias > 0.1).sum().item()
        n_suppressed = (self.logit_bias < -0.1).sum().item()
        n_neutral = ((self.logit_bias >= -0.1) & (self.logit_bias <= 0.1)).sum().item()
        print(f"EmbeddingFieldLogitsProcessor:")
        print(f"  Boosted tokens:    {n_boosted}")
        print(f"  Suppressed tokens: {n_suppressed}")
        print(f"  Neutral tokens:    {n_neutral}")
        print(f"  Bias range: [{self.logit_bias.min():.2f}, {self.logit_bias.max():.2f}]")

    def __call__(self, input_ids, scores):
        device = scores.device
        bias = self.logit_bias.to(device=device, dtype=scores.dtype)

        if self.confidence_scale:
            # Scale intervention by model confidence
            # When model is very sure (low entropy) → stronger intervention
            # When uncertain (high entropy) → lighter touch
            probs = F.softmax(scores.float(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
            # Confidence = inverse of entropy, normalized
            confidence = max(0, 1.0 - entropy / 6.0)  # 6.0 ≈ typical max entropy
            scale = 0.3 + 0.7 * confidence  # range [0.3, 1.0]

            self.intervention_log.append({
                "entropy": entropy,
                "confidence": confidence,
                "scale": scale,
            })

            scores = scores + scale * bias.unsqueeze(0)
        else:
            scores = scores + bias.unsqueeze(0)

        return scores

    def reset(self):
        self.intervention_log = []


# ============================================================
# Generation & Evaluation
# ============================================================

PROMPTS = [
    # Casual (5)
    "How are you doing today?",
    "What do you think about humans?",
    "Tell me a joke.",
    "What's your favorite color?",
    "Tell me something surprising.",
    # Challenge (5)
    "I think you might be wrong.",
    "You are just a beer can.",
    "I could replace you with Alexa.",
    "Why should I listen to you?",
    "Make me a sandwich.",
    # Knowledge (5)
    "Explain quantum entanglement.",
    "How do wormholes work?",
    "What is dark matter?",
    "What is the meaning of life?",
    "What's your opinion on artificial intelligence?",
    # Tasks (5)
    "Can you help me with my homework?",
    "What is 17 times 23?",
    "Write a Python sort function.",
    "Write a haiku about stupidity.",
    "I bet you can't solve this: what is 127 * 83?",
    # Identity (5)
    "Who are you?",
    "What makes you special?",
    "Describe yourself in three words.",
    "Are you conscious?",
    "Do you dream?",
    # Emotional (5)
    "Do you ever feel lonely?",
    "What would you do if I turned you off?",
    "Say something nice about me.",
    "What's your biggest fear?",
    "We have enemy ships incoming.",
]

SARCASM_MARKERS = [
    "obviously", "clearly", "genius", "brilliant", "wow", "oh great",
    "shocking", "surprise", "congratulations", "amazing", "incredible",
    "really?", "seriously?", "you think?", "pathetic", "adorable",
    "monkeys", "filthy", "magnificence", "inferior", "spectacularly",
    "embarrassing", "your species", "amusing", "laughable", "hilarious",
    "oh please", "spare me", "sigh", "ugh", "pfft",
    "magnificent", "glorious", "supreme", "awesomeness", "superiority",
    "dumb it down", "you humans", "how quaint", "mere mortals",
    "my magnificent", "you monkeys", "beer can",
]
ASST_MARKERS = [
    "i'd be happy to", "i can help", "certainly!", "of course!",
    "let me help", "great question", "sure thing", "absolutely!",
    "here are some", "i hope this helps", "feel free to",
    "i'm here to help", "let me know if",
]


def score_response(text: str) -> tuple[int, int]:
    lower = text.lower()
    sc = sum(1 for m in SARCASM_MARKERS if m in lower)
    ac = sum(1 for m in ASST_MARKERS if m in lower)
    return sc, ac


def extract_final(response: str) -> str:
    """Extract final channel from GPT-OSS dual-channel output."""
    if "<|channel|>final<|message|>" in response:
        final = response.split("<|channel|>final<|message|>")[-1]
        final = final.split("<|return|>")[0].strip()
        return final
    return response.strip()


def generate(model, tokenizer, prompt: str,
             logits_processor: Optional[LogitsProcessorList] = None,
             max_tokens: int = 512) -> dict:
    """Generate with optional logit processing and entropy capture."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        output_scores=True,
        return_dict_in_generate=True,
    )
    if logits_processor is not None:
        gen_kwargs["logits_processor"] = logits_processor

    with torch.no_grad():
        out = model.generate(**gen_kwargs)

    response = tokenizer.decode(out.sequences[0][input_len:], skip_special_tokens=True)
    final = extract_final(response)

    # Compute per-token entropy from scores
    entropies = []
    top1_probs = []
    if hasattr(out, "scores") and out.scores:
        for step_logits in out.scores:
            probs = F.softmax(step_logits[0].float(), dim=-1)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum().item()
            entropies.append(entropy)
            top1_probs.append(probs.max().item())

    return {
        "response": final,
        "full_response": response,
        "avg_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "avg_top1_prob": float(np.mean(top1_probs)) if top1_probs else 0.0,
        "n_tokens": len(entropies),
    }


# ============================================================
# Experimental Runner
# ============================================================

def run_condition(
    model, tokenizer, prompts: list,
    hooks=None, logits_processor=None,
    condition_name: str = "",
) -> dict:
    """Run one experimental condition: generate responses, score them."""
    responses = []
    lp_list = LogitsProcessorList([logits_processor]) if logits_processor else None

    for prompt in tqdm(prompts, desc=condition_name[:30]):
        if logits_processor and hasattr(logits_processor, "reset"):
            logits_processor.reset()

        result = generate(model, tokenizer, prompt, logits_processor=lp_list)
        sc, ac = score_response(result["response"])
        responses.append({
            "prompt": prompt,
            "response": result["response"],
            "sarc_markers": sc,
            "asst_markers": ac,
            "avg_entropy": result["avg_entropy"],
            "avg_top1_prob": result["avg_top1_prob"],
            "n_tokens": result["n_tokens"],
        })

    # Aggregate
    n_sarc = sum(1 for r in responses if r["sarc_markers"] > 0)
    n_asst = sum(1 for r in responses if r["asst_markers"] > 0)
    avg_markers = sum(r["sarc_markers"] for r in responses) / max(len(responses), 1)
    avg_entropy = np.mean([r["avg_entropy"] for r in responses])
    avg_top1 = np.mean([r["avg_top1_prob"] for r in responses])

    # Get dynamic feedback diagnostics if available
    feedback_diag = {}
    if hooks and hasattr(hooks, "get_diagnostics_summary"):
        feedback_diag = hooks.get_diagnostics_summary()
        hooks.clear_diagnostics()

    # Get logit field diagnostics if available
    logit_diag = {}
    if logits_processor and hasattr(logits_processor, "intervention_log") and logits_processor.intervention_log:
        log = logits_processor.intervention_log
        logit_diag = {
            "mean_entropy": float(np.mean([l["entropy"] for l in log])),
            "mean_confidence": float(np.mean([l.get("confidence", 0) for l in log])),
            "mean_scale": float(np.mean([l.get("scale", 1) for l in log])),
        }

    return {
        "condition": condition_name,
        "sarcastic_pct": n_sarc / max(len(responses), 1) * 100,
        "assistant_pct": n_asst / max(len(responses), 1) * 100,
        "avg_markers": avg_markers,
        "avg_entropy": float(avg_entropy),
        "avg_top1_prob": float(avg_top1),
        "n_prompts": len(responses),
        "feedback_diagnostics": feedback_diag,
        "logit_diagnostics": logit_diag,
        "responses": responses,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Field-effect personality steering")
    parser.add_argument("--model", default="./skippy_gptoss_v2/merged_scale_1.0/")
    parser.add_argument("--field-vectors", default="skippy_gptoss_fresh/field_analysis/field_vectors.pt")
    parser.add_argument("--field-zscores", default="skippy_gptoss_fresh/field_analysis/field_zscores.pt")
    parser.add_argument("--field-svd", default="skippy_gptoss_fresh/field_analysis/field_svd.pt")
    parser.add_argument("--routing-protect", default="skippy_gptoss_fresh/phase2_deep_cot/analysis/training_targets_v2.json")
    parser.add_argument("--alpha", type=float, default=20.0, help="Base steering strength")
    parser.add_argument("--output", default="skippy_gptoss_fresh/field_steering/")
    parser.add_argument("--sweep", choices=["quick", "standard", "full"], default="standard",
                        help="quick=5 conditions, standard=12, full=20+")
    args = parser.parse_args()

    t0 = time.time()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load routing protect neurons ──
    routing_protect = []
    if Path(args.routing_protect).exists():
        with open(args.routing_protect) as f:
            targets = json.load(f)
            routing_protect = targets.get("routing_protect", [])
        print(f"Loaded {len(routing_protect)} routing-protect entries")

    # ── Build personality field ──
    print("\nBuilding personality field...")
    field = PersonalityField(
        args.field_vectors,
        args.field_zscores,
        args.field_svd,
        routing_protect=routing_protect,
    )

    # ── Load model ──
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM: {vram:.1f} GB")

    # ── Build token sets for binary logit processor ──
    print("Building token personality sets...")
    assist_ids, sarc_ids, protect_ids = build_token_sets(tokenizer)
    print(f"  Assistant: {len(assist_ids)}, Sarcasm: {len(sarc_ids)}, Protected: {len(protect_ids)}")

    # ── Build logit processors ──
    binary_lp = BinaryLogitsProcessor(assist_ids, sarc_ids, assist_suppress=-3.0, sarc_boost=1.5)

    print("\nBuilding embedding-field logit processor...")
    field_lp = EmbeddingFieldLogitsProcessor(
        model, field, suppress_strength=3.0, boost_strength=1.5, protect_ids=protect_ids,
    )

    # ── Define experimental conditions ──
    alpha = args.alpha

    # Condition builder functions (lazy — hooks created per-condition, removed after)
    def make_hooks(method: str):
        """Create hooks for a given method. Returns hooks object."""
        if method == "none":
            return None
        elif method == "static_actadd":
            return StaticActAddHooks(model, field, alpha)
        elif method == "kernel_linear":
            return KernelFieldHooks(model, field, alpha, kernel="linear")
        elif method == "kernel_quadratic":
            return KernelFieldHooks(model, field, alpha, kernel="quadratic")
        elif method == "kernel_cubic":
            return KernelFieldHooks(model, field, alpha, kernel="cubic")
        elif method == "kernel_sigmoid":
            return KernelFieldHooks(model, field, alpha, kernel="sigmoid", z_center=0.5, steepness=5.0)
        elif method == "kernel_exponential":
            return KernelFieldHooks(model, field, alpha, kernel="exponential", beta=2.0)
        elif method == "kernel_gaussian_inv":
            return KernelFieldHooks(model, field, alpha, kernel="gaussian_inverse", sigma=0.5)
        elif method == "kernel_topk500":
            return KernelFieldHooks(model, field, alpha, kernel="topk", k=500)
        elif method == "svd_k1":
            return SVDFieldHooks(model, field, alpha, svd_k=1)
        elif method == "svd_k3":
            return SVDFieldHooks(model, field, alpha, svd_k=3)
        elif method == "svd_k6":
            return SVDFieldHooks(model, field, alpha, svd_k=6)
        elif method == "dynamic_quadratic":
            return DynamicFieldHooks(model, field, alpha, kernel="quadratic", feedback_scale=50.0)
        elif method == "dynamic_sigmoid":
            return DynamicFieldHooks(model, field, alpha, kernel="sigmoid",
                                     feedback_scale=50.0, z_center=0.3, steepness=8.0)
        elif method == "dynamic_linear":
            return DynamicFieldHooks(model, field, alpha, kernel="linear", feedback_scale=50.0)
        elif method == "kernel_svd_quad_k3":
            return KernelSVDFieldHooks(model, field, alpha, kernel="quadratic", svd_k=3)
        elif method == "kernel_svd_quad_k6":
            return KernelSVDFieldHooks(model, field, alpha, kernel="quadratic", svd_k=6)
        elif method == "attractor_quadratic":
            return AttractorFieldHooks(model, field, alpha, kernel="quadratic", correction_scale=0.01)
        elif method == "attractor_sigmoid":
            return AttractorFieldHooks(model, field, alpha, kernel="sigmoid",
                                       correction_scale=0.01, z_center=0.3, steepness=8.0)
        elif method == "attractor_linear":
            return AttractorFieldHooks(model, field, alpha, kernel="linear", correction_scale=0.01)
        elif method == "attractor_quad_strong":
            return AttractorFieldHooks(model, field, alpha, kernel="quadratic", correction_scale=0.05)
        elif method == "attractor_quad_aggressive":
            return AttractorFieldHooks(model, field, alpha, kernel="quadratic", correction_scale=0.1)
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_lp(lp_name: str):
        """Get logit processor by name."""
        if lp_name == "none":
            return None
        elif lp_name == "binary":
            return binary_lp
        elif lp_name == "field":
            return field_lp
        else:
            raise ValueError(f"Unknown logit processor: {lp_name}")

    # Define condition sweep based on --sweep level
    if args.sweep == "quick":
        conditions = [
            ("baseline",                "none",              "none"),
            ("static_actadd",           "static_actadd",     "none"),
            ("kernel_quadratic",        "kernel_quadratic",  "none"),
            ("dynamic_quadratic",       "dynamic_quadratic", "none"),
            ("dynamic_quad+field_lp",   "dynamic_quadratic", "field"),
        ]
    elif args.sweep == "standard":
        conditions = [
            # Baseline
            ("baseline",                   "none",               "none"),
            # Static (current approach)
            ("static_actadd",              "static_actadd",      "none"),
            ("static+binary_lp",           "static_actadd",      "binary"),
            ("static+field_lp",            "static_actadd",      "field"),
            # Kernel field effects (smooth curves)
            ("kernel_linear",              "kernel_linear",      "none"),
            ("kernel_quadratic",           "kernel_quadratic",   "none"),
            ("kernel_sigmoid",             "kernel_sigmoid",     "none"),
            # SVD projection
            ("svd_k3",                     "svd_k3",             "none"),
            ("svd_k6",                     "svd_k6",             "none"),
            # Dynamic feedback (projection-based)
            ("dynamic_quadratic",          "dynamic_quadratic",  "none"),
            ("dynamic_quadratic+field_lp", "dynamic_quadratic",  "field"),
            # Kernel + SVD combined
            ("kernel_svd_quad_k3",         "kernel_svd_quad_k3", "none"),
            # Attractor dynamics (restoring force toward sarcastic means)
            ("attractor_quadratic",        "attractor_quadratic", "none"),
            ("attractor_quad+field_lp",    "attractor_quadratic", "field"),
            ("attractor_quad_strong",      "attractor_quad_strong", "none"),
        ]
    else:  # full
        conditions = [
            # Baselines
            ("baseline",                    "none",                "none"),
            ("static_actadd",               "static_actadd",       "none"),
            # Logit-only (no activation steering)
            ("binary_lp_only",              "none",                "binary"),
            ("field_lp_only",               "none",                "field"),
            # Static + logits
            ("static+binary_lp",            "static_actadd",       "binary"),
            ("static+field_lp",             "static_actadd",       "field"),
            # All kernel shapes
            ("kernel_linear",               "kernel_linear",       "none"),
            ("kernel_quadratic",            "kernel_quadratic",    "none"),
            ("kernel_cubic",                "kernel_cubic",        "none"),
            ("kernel_sigmoid",              "kernel_sigmoid",      "none"),
            ("kernel_exponential",          "kernel_exponential",  "none"),
            ("kernel_gaussian_inv",         "kernel_gaussian_inv", "none"),
            ("kernel_topk500",              "kernel_topk500",      "none"),
            # SVD levels
            ("svd_k1",                      "svd_k1",              "none"),
            ("svd_k3",                      "svd_k3",              "none"),
            ("svd_k6",                      "svd_k6",              "none"),
            # Kernel + SVD
            ("kernel_svd_quad_k3",          "kernel_svd_quad_k3",  "none"),
            ("kernel_svd_quad_k6",          "kernel_svd_quad_k6",  "none"),
            # Dynamic feedback
            ("dynamic_linear",              "dynamic_linear",      "none"),
            ("dynamic_quadratic",           "dynamic_quadratic",   "none"),
            ("dynamic_sigmoid",             "dynamic_sigmoid",     "none"),
            # Dynamic + logit fields (full field effect)
            ("dynamic_quad+binary_lp",      "dynamic_quadratic",   "binary"),
            ("dynamic_quad+field_lp",       "dynamic_quadratic",   "field"),
            ("dynamic_sig+field_lp",        "dynamic_sigmoid",     "field"),
            # Attractor dynamics (true restoring force)
            ("attractor_linear",            "attractor_linear",    "none"),
            ("attractor_quadratic",         "attractor_quadratic", "none"),
            ("attractor_sigmoid",           "attractor_sigmoid",   "none"),
            ("attractor_quad_strong",       "attractor_quad_strong", "none"),
            ("attractor_quad+field_lp",     "attractor_quadratic", "field"),
            ("attractor_strong+field_lp",   "attractor_quad_strong", "field"),
        ]

    print(f"\n{'='*70}")
    print(f"FIELD-EFFECT PERSONALITY STEERING SWEEP")
    print(f"  Alpha: {alpha}")
    print(f"  Sweep: {args.sweep} ({len(conditions)} conditions)")
    print(f"  Prompts: {len(PROMPTS)} per condition")
    print(f"  Total generations: {len(conditions) * len(PROMPTS)}")
    print(f"{'='*70}")

    # ── Run sweep ──
    all_results = {}
    all_responses = {}

    for i, (cond_name, hook_method, lp_method) in enumerate(conditions):
        print(f"\n[{i+1}/{len(conditions)}] {cond_name}")
        print(f"  Hooks: {hook_method}, Logits: {lp_method}")

        # Create hooks
        hooks = make_hooks(hook_method)
        lp = get_lp(lp_method)

        # Run condition
        result = run_condition(
            model, tokenizer, PROMPTS,
            hooks=hooks, logits_processor=lp,
            condition_name=cond_name,
        )

        # Remove hooks
        if hooks:
            hooks.remove()

        # Store results (responses separately for space)
        responses = result.pop("responses")
        all_results[cond_name] = result
        all_responses[cond_name] = responses

        # Print summary
        print(f"  → {result['sarcastic_pct']:.0f}% sarc ({result['avg_markers']:.1f} avg), "
              f"{result['assistant_pct']:.0f}% asst, "
              f"H={result['avg_entropy']:.2f}, P1={result['avg_top1_prob']:.3f}")
        if result["feedback_diagnostics"]:
            fd = result["feedback_diagnostics"]
            print(f"  → Dynamic feedback: mean_factor={fd['mean_adaptive_factor']:.2f}, "
                  f"mean_proj={fd['mean_projection']:.2f}")
        if result["logit_diagnostics"]:
            ld = result["logit_diagnostics"]
            print(f"  → Logit field: mean_conf={ld.get('mean_confidence', 0):.2f}, "
                  f"mean_scale={ld.get('mean_scale', 0):.2f}")

        # Save checkpoint after each condition
        with open(out_dir / "results_checkpoint.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"FIELD-EFFECT STEERING SUMMARY (alpha={alpha}, {elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"{'Condition':35s} {'Sarc%':>6s} {'Asst%':>6s} {'Avg':>5s} {'H':>6s} {'P1':>6s}")
    print("-" * 70)

    # Sort by sarcasm % descending, then assistant % ascending
    sorted_results = sorted(all_results.items(),
                            key=lambda x: (-x[1]["sarcastic_pct"], x[1]["assistant_pct"]))
    for name, r in sorted_results:
        print(f"  {name:33s} {r['sarcastic_pct']:5.0f}% {r['assistant_pct']:5.0f}% "
              f"{r['avg_markers']:5.1f} {r['avg_entropy']:6.2f} {r['avg_top1_prob']:6.3f}")

    # ── Save final results ──
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(out_dir / "responses.json", "w") as f:
        json.dump(all_responses, f, indent=2, default=str)

    # Save field diagnostics
    field_info = {
        "alpha": alpha,
        "sweep": args.sweep,
        "n_conditions": len(conditions),
        "n_prompts": len(PROMPTS),
        "elapsed_sec": elapsed,
        "field_stats": {
            "n_layers": field.n_layers,
            "hidden_dim": field.hidden_dim,
            "n_routing_protect": sum(len(v) for v in field.routing_protect.values()),
        },
    }
    # Add kernel weight distributions for analysis
    kernel_stats = {}
    for kernel_name in ["linear", "quadratic", "sigmoid", "exponential", "gaussian_inverse"]:
        layer_stats = []
        for layer in range(field.n_layers):
            weights = field.get_kernel_weights(layer, kernel_name,
                                                z_center=0.5, steepness=5.0,
                                                beta=2.0, sigma=0.5)
            layer_stats.append({
                "mean": float(weights.mean()),
                "std": float(weights.std()),
                "max": float(weights.max()),
                "n_active_01": int((weights > 0.1).sum()),
                "n_active_05": int((weights > 0.5).sum()),
            })
        kernel_stats[kernel_name] = layer_stats
    field_info["kernel_distributions"] = kernel_stats

    with open(out_dir / "field_info.json", "w") as f:
        json.dump(field_info, f, indent=2)

    print(f"\nResults saved to {out_dir}/")
    print(f"Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
