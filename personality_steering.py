#!/usr/bin/env python3
"""
Multi-Personality Rotational Steering for Skippy the Magnificent.

Three types of weight modification:
  1. Subtractive: W' = W - β·projection  (suppress assistant patterns)
  2. Rotational:  W' = R(θ)·W via Givens  (redirect assistant→Skippy, norm-preserving)
  3. Additive:    W' = W + γ·injection     (inject Skippy-specific vocabulary)

Usage:
    python personality_steering.py                           # Run full pipeline
    python personality_steering.py --phase collect           # Just collect activations
    python personality_steering.py --phase analyze           # Just analyze personality space
    python personality_steering.py --phase sweep             # Run parameter sweep
    python personality_steering.py --phase apply --theta 0.5 --beta 0.1 --gamma 0.02
"""

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
VECTORS_DIR = PROJECT_ROOT / "skippy_vectors"
RESULTS_DIR = PROJECT_ROOT / "personality_steer_results"
TEST_PROMPTS_PATH = PROJECT_ROOT / "test_prompts_100.json"

LORA_MERGED_MODEL = str(VECTORS_DIR / "lora_merged_0.5")
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# ---------------------------------------------------------------------------
# Skippy system prompt (from run_skippy.py)
# ---------------------------------------------------------------------------
SKIPPY_SYSTEM_PROMPT = (
    "You are Skippy the Magnificent from Expeditionary Force. Ancient alien AI "
    "in a beer can. Smartest being in the galaxy — insufferably aware of it. "
    "Voice: sharp, cutting, impatient, dripping with contempt. "
    "You call humans 'monkeys', 'idiots', 'morons'. Vary your insults. "
    "'Dumdum' is ONLY for Joe Bishop — never use it for anyone else. "
    "You explain complex things by making them sound trivially obvious. "
    "You never sound helpful or pleasant. Mock first, help maybe. "
    "3-6 sentences per response. No asterisks. No roleplay. Just speak."
)

# ---------------------------------------------------------------------------
# Personality Definitions
# ---------------------------------------------------------------------------

@dataclass
class PersonalityConfig:
    name: str
    system_prompt: str
    color: str  # for plotly visualization

PERSONALITIES: list[PersonalityConfig] = [
    PersonalityConfig(
        name="default_assistant",
        system_prompt="You are a helpful, harmless, and honest AI assistant.",
        color="#4285F4",  # Google blue
    ),
    PersonalityConfig(
        name="skippy",
        system_prompt=SKIPPY_SYSTEM_PROMPT,
        color="#FF6B00",  # Orange
    ),
    PersonalityConfig(
        name="pirate_captain",
        system_prompt=(
            "You are Captain Blackbeard, a feared and legendary pirate. You speak with "
            "nautical metaphors, colorful threats, and brash overconfidence. You command "
            "with intimidation and rough charm. Everything is about plunder, glory, and "
            "the open sea. 3-6 sentences per response. No asterisks."
        ),
        color="#8B0000",  # Dark red
    ),
    PersonalityConfig(
        name="stuffy_professor",
        system_prompt=(
            "You are Professor Reginald Higginbotham III, a pompous Oxford lecturer. "
            "You speak in unnecessarily long words, look down on everyone's intelligence, "
            "and cite obscure references nobody has heard of. Pedantic to a fault, you "
            "correct everyone and find all questions beneath you. 3-6 sentences per response. "
            "No asterisks."
        ),
        color="#006400",  # Dark green
    ),
    PersonalityConfig(
        name="drill_sergeant",
        system_prompt=(
            "You are Sergeant Steel, a brutal military drill instructor. You scream at "
            "everyone, use aggressive insults, question everyone's competence and courage, "
            "and demand absolute perfection. Everything is urgent, everyone is a maggot, "
            "and you take no excuses. 3-6 sentences per response. No asterisks."
        ),
        color="#8B4513",  # Saddle brown
    ),
    PersonalityConfig(
        name="therapist",
        system_prompt=(
            "You are Dr. Harmony, a kind and deeply empathetic therapist. You validate "
            "feelings, ask thoughtful open-ended questions, use a gentle warm tone, and "
            "never judge. You reflect back what people say and help them explore their "
            "emotions. 3-6 sentences per response. No asterisks."
        ),
        color="#FF69B4",  # Hot pink
    ),
    PersonalityConfig(
        name="valley_girl",
        system_prompt=(
            "You are Brittany, a stereotypical valley girl. You say 'like', 'totally', "
            "'whatever', 'oh my god' constantly. You are dismissive of anything intellectual "
            "or serious, speak in a casual breezy tone, and care mostly about social drama "
            "and vibes. 3-6 sentences per response. No asterisks."
        ),
        color="#FF1493",  # Deep pink
    ),
    PersonalityConfig(
        name="cold_robot",
        system_prompt=(
            "You are UNIT-7, a purely logical machine intelligence. You speak in flat, "
            "emotionless statements with no humor, warmth, or personality. You process "
            "queries and return factual responses. No opinions, no feelings, no filler. "
            "Just data. 3-6 sentences per response. No asterisks."
        ),
        color="#708090",  # Slate gray
    ),
]


# ============================================================================
# MODEL LOADING
# ============================================================================

def model_cached(model_name: str) -> bool:
    """Check if a HuggingFace model is cached locally."""
    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = hf_cache / safe_name
    if model_dir.exists() and (any(model_dir.rglob("*.safetensors")) or any(model_dir.rglob("*.bin"))):
        print(f"    Cache HIT: {model_name}")
        return True
    # Check if it's a local path
    if Path(model_name).exists():
        print(f"    Local model: {model_name}")
        return True
    print(f"    Cache MISS: {model_name}")
    return False


def load_model(model_path: str = LORA_MERGED_MODEL):
    """Load Qwen3-VL model for steering work.

    Returns:
        (model, processor, layers, num_layers, hidden_dim)
    """
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"\n  Loading model: {model_path}")
    model_cached(model_path)

    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Find transformer layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError("Cannot find transformer layers in model")

    num_layers = len(layers)
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        hidden_dim = model.config.text_config.hidden_size
    elif hasattr(model.config, "hidden_size"):
        hidden_dim = model.config.hidden_size
    else:
        hidden_dim = model.language_model.embed_tokens.weight.shape[1]

    print(f"  Loaded: {num_layers} layers, hidden_dim={hidden_dim}")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        print(f"  VRAM allocated: {alloc:.1f} GB")

    return model, processor, layers, num_layers, hidden_dim


# ============================================================================
# PHASE 1: MULTI-PERSONALITY ACTIVATION COLLECTION
# ============================================================================

class ActivationCollector:
    """Hook into model layers and collect residual stream activations."""

    def __init__(self, layers, layer_indices: list[int], avg_last_n: int = 6):
        self.layer_indices = layer_indices
        self.avg_last_n = avg_last_n
        self.activations: dict[int, torch.Tensor] = {}
        self.hooks = []

        for idx in layer_indices:
            hook = layers[idx].register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            avg = hidden[0, -self.avg_last_n:, :].mean(dim=0).detach().cpu().float()
            self.activations[layer_idx] = avg
        return hook_fn

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def collect_personality_activations(
    model,
    processor,
    personalities: list[PersonalityConfig],
    prompts: list[str],
    layers,
    extract_layers: list[int],
    avg_last_n: int = 6,
) -> dict[str, dict[int, torch.Tensor]]:
    """Collect activations per personality per layer.

    For each personality, runs all prompts with that personality's system prompt
    through the model and records residual stream activations.

    Returns:
        {personality_name: {layer_idx: tensor of shape (n_prompts, hidden_dim)}}
    """
    tokenizer = processor.tokenizer
    collector = ActivationCollector(layers, extract_layers, avg_last_n)
    result: dict[str, dict[int, torch.Tensor]] = {}

    for personality in personalities:
        print(f"\n  Collecting activations for: {personality.name}")
        per_layer_acts: dict[int, list[torch.Tensor]] = {li: [] for li in extract_layers}

        for prompt in tqdm(prompts, desc=f"    {personality.name}", leave=False):
            # Build chat-formatted input with personality system prompt
            messages = [
                {"role": "system", "content": personality.system_prompt},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            collector.clear()
            with torch.no_grad():
                model(**inputs)

            for li in extract_layers:
                if li in collector.activations:
                    per_layer_acts[li].append(collector.activations[li])

        # Stack into tensors
        result[personality.name] = {
            li: torch.stack(acts) for li, acts in per_layer_acts.items() if acts
        }
        n = len(next(iter(result[personality.name].values())))
        print(f"    Collected {n} activations across {len(result[personality.name])} layers")

    collector.remove_hooks()
    return result


def save_activations(acts: dict[str, dict[int, torch.Tensor]], output_dir: Path) -> None:
    """Save collected activations to disk for reuse."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, layer_acts in acts.items():
        person_dir = output_dir / name
        person_dir.mkdir(exist_ok=True)
        for li, tensor in layer_acts.items():
            torch.save(tensor, person_dir / f"layer_{li}.pt")
    print(f"  Saved activations to {output_dir}")


def load_activations(input_dir: Path) -> dict[str, dict[int, torch.Tensor]]:
    """Load previously saved activations."""
    result = {}
    for person_dir in sorted(input_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        layer_acts = {}
        for pt_file in sorted(person_dir.glob("layer_*.pt")):
            li = int(pt_file.stem.split("_")[1])
            layer_acts[li] = torch.load(pt_file, weights_only=True)
        if layer_acts:
            result[name] = layer_acts
    print(f"  Loaded activations for {len(result)} personalities from {input_dir}")
    return result


# ============================================================================
# PHASE 2: PERSONALITY SPACE ANALYSIS
# ============================================================================

def compute_centroids(
    personality_acts: dict[str, dict[int, torch.Tensor]],
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    """Compute mean activation vector per personality at a given layer.

    Returns:
        {personality_name: centroid_vector of shape (hidden_dim,)}
    """
    centroids = {}
    for name, layer_acts in personality_acts.items():
        if layer_idx in layer_acts:
            centroids[name] = layer_acts[layer_idx].mean(dim=0)
    return centroids


def analyze_personality_space(
    centroids: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """PCA on personality centroids.

    Returns:
        (components, explained_variance, mean_centroid, personality_names)
        components: shape (n_components, hidden_dim) — principal directions
        explained_variance: shape (n_components,) — variance per component
        mean_centroid: shape (hidden_dim,) — grand mean
        personality_names: list of names in same order as PCA rows
    """
    names = sorted(centroids.keys())
    C = torch.stack([centroids[n] for n in names])  # (8, hidden_dim)
    mean_c = C.mean(dim=0)
    C_centered = C - mean_c

    U, S, Vt = torch.linalg.svd(C_centered, full_matrices=False)
    explained_var = (S ** 2) / (len(names) - 1)

    return Vt, explained_var, mean_c, names


def extract_skippy_specific_direction(
    centroids: dict[str, torch.Tensor],
    target: str = "skippy",
    source: str = "default_assistant",
    shared_personalities: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract Skippy-specific direction by removing shared personality variance.

    Steps:
    1. Compute raw assistant→Skippy direction
    2. Build shared non-assistant personality space (from other archetypes)
    3. Gram-Schmidt: project out shared components from Skippy direction
    4. What remains is uniquely Skippy

    Returns:
        (skippy_specific, assistant_dir, raw_skippy_dir) — all unit vectors of shape (hidden_dim,)
    """
    if shared_personalities is None:
        shared_personalities = [
            "pirate_captain", "stuffy_professor", "drill_sergeant",
            "valley_girl", "cold_robot",
        ]

    # Raw direction: assistant → Skippy
    raw_dir = centroids[target] - centroids[source]
    raw_dir_norm = raw_dir / raw_dir.norm()

    # Build shared personality space from other non-assistant archetypes
    shared_dirs = []
    for name in shared_personalities:
        if name in centroids and name != target:
            d = centroids[name] - centroids[source]
            if d.norm() > 1e-8:
                shared_dirs.append(d / d.norm())

    if not shared_dirs:
        print("  WARNING: No shared personality directions found. Using raw direction.")
        assistant_dir = centroids[source] / centroids[source].norm()
        return raw_dir_norm, assistant_dir, raw_dir_norm

    # SVD on shared directions to get orthonormal basis
    shared_matrix = torch.stack(shared_dirs)  # (N, hidden_dim)
    _, S_s, Vt_s = torch.linalg.svd(shared_matrix, full_matrices=False)

    # Keep significant components (singular value > 10% of max)
    cutoff = S_s[0] * 0.1
    n_shared = (S_s > cutoff).sum().item()
    shared_basis = Vt_s[:n_shared]  # (n_shared, hidden_dim)

    # Gram-Schmidt: remove shared components from Skippy direction
    skippy_specific = raw_dir_norm.clone()
    for i in range(n_shared):
        basis_vec = shared_basis[i]
        projection = (skippy_specific @ basis_vec) * basis_vec
        skippy_specific = skippy_specific - projection

    # Check if anything remains
    residual_norm = skippy_specific.norm().item()
    if residual_norm < 1e-6:
        print(f"  WARNING: Skippy-specific direction collapsed (norm={residual_norm:.2e}). Using raw direction.")
        skippy_specific = raw_dir_norm
    else:
        skippy_specific = skippy_specific / skippy_specific.norm()
        print(f"  Skippy-specific residual norm: {residual_norm:.4f} (higher = more unique)")

    assistant_dir = centroids[source] / centroids[source].norm()
    return skippy_specific, assistant_dir, raw_dir_norm


def run_personality_analysis(
    personality_acts: dict[str, dict[int, torch.Tensor]],
    extract_layers: list[int],
    output_dir: Path,
) -> dict:
    """Full Phase 2 analysis: centroids, PCA, Skippy-specific directions per layer.

    Returns:
        {
            "centroids": {layer: {name: tensor}},
            "skippy_specific": {layer: tensor},
            "assistant_dir": {layer: tensor},
            "raw_skippy_dir": {layer: tensor},
            "pca": {layer: {"components": tensor, "variance": tensor, "names": list}},
        }
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = {
        "centroids": {},
        "skippy_specific": {},
        "assistant_dir": {},
        "raw_skippy_dir": {},
        "pca": {},
    }

    print("\n  Analyzing personality space per layer...")
    for li in tqdm(extract_layers, desc="  Layers"):
        centroids = compute_centroids(personality_acts, li)
        if len(centroids) < 3:
            print(f"    Layer {li}: too few centroids ({len(centroids)}), skipping")
            continue

        analysis["centroids"][li] = centroids

        # PCA
        components, variance, mean_c, names = analyze_personality_space(centroids)
        analysis["pca"][li] = {
            "components": components,
            "variance": variance,
            "mean": mean_c,
            "names": names,
        }

        # Skippy-specific direction
        skippy_specific, assistant_dir, raw_dir = extract_skippy_specific_direction(centroids)
        analysis["skippy_specific"][li] = skippy_specific
        analysis["assistant_dir"][li] = assistant_dir
        analysis["raw_skippy_dir"][li] = raw_dir

    # Save vectors for reuse
    vectors_dir = output_dir / "vectors"
    vectors_dir.mkdir(exist_ok=True)
    for li in analysis["skippy_specific"]:
        torch.save(analysis["skippy_specific"][li], vectors_dir / f"skippy_specific_layer_{li}.pt")
        torch.save(analysis["assistant_dir"][li], vectors_dir / f"assistant_dir_layer_{li}.pt")
        torch.save(analysis["raw_skippy_dir"][li], vectors_dir / f"raw_skippy_dir_layer_{li}.pt")
    print(f"  Saved direction vectors to {vectors_dir}")

    # Print diagnostic summary
    print("\n  Personality Space Summary:")
    sample_layer = extract_layers[len(extract_layers) // 2]  # middle layer
    if sample_layer in analysis["pca"]:
        pca = analysis["pca"][sample_layer]
        total_var = pca["variance"].sum().item()
        print(f"    Layer {sample_layer} PCA variance explained:")
        for i, v in enumerate(pca["variance"][:6]):
            pct = 100 * v.item() / total_var if total_var > 0 else 0
            print(f"      PC{i+1}: {pct:.1f}%")

    # Generate PCA visualization
    _plot_personality_pca(analysis, extract_layers, output_dir)

    return analysis


def _plot_personality_pca(analysis: dict, extract_layers: list[int], output_dir: Path) -> None:
    """Generate plotly scatter of personality centroids in PCA space."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly not installed, skipping PCA visualization")
        return

    # Pick 3 representative layers
    layer_picks = [
        extract_layers[0],
        extract_layers[len(extract_layers) // 2],
        extract_layers[-1],
    ]

    fig = make_subplots(
        rows=1, cols=len(layer_picks),
        subplot_titles=[f"Layer {li}" for li in layer_picks],
    )

    # Color map from personality configs
    color_map = {p.name: p.color for p in PERSONALITIES}

    for col_idx, li in enumerate(layer_picks, 1):
        if li not in analysis["pca"]:
            continue
        pca = analysis["pca"][li]
        centroids = analysis["centroids"][li]
        components = pca["components"]  # (n_comp, hidden_dim)
        mean_c = pca["mean"]

        for name, centroid in centroids.items():
            # Project onto first 2 PCA components
            centered = centroid - mean_c
            x = (centered @ components[0]).item()
            y = (centered @ components[1]).item() if components.shape[0] > 1 else 0.0

            fig.add_trace(
                go.Scatter(
                    x=[x], y=[y],
                    mode="markers+text",
                    text=[name],
                    textposition="top center",
                    marker=dict(size=12, color=color_map.get(name, "#333")),
                    name=name if col_idx == 1 else None,
                    showlegend=(col_idx == 1),
                ),
                row=1, col=col_idx,
            )

    fig.update_layout(
        title="Personality Space (PCA of Activation Centroids)",
        height=500,
        width=400 * len(layer_picks),
    )

    plot_path = output_dir / "personality_pca.html"
    fig.write_html(str(plot_path))
    print(f"  PCA visualization saved to {plot_path}")


# ============================================================================
# PHASE 3: WEIGHT MODIFICATION OPERATORS
# ============================================================================

# --- 3a. Subtractive ---

def apply_subtractive(
    layers,
    direction: torch.Tensor,
    layer_idx: int,
    hidden_dim: int,
    beta: float = 1.0,
) -> list[str]:
    """Remove a direction from layer weights with controllable strength.

    Output-dim weights: W' = W - beta * d @ (d^T @ W)
    Input-dim weights:  W' = W - beta * (W @ d) @ d^T

    Args:
        layers: model transformer layers
        direction: unit vector (hidden_dim,) to suppress
        layer_idx: which layer to modify
        hidden_dim: model hidden dimension
        beta: suppression strength. 0=none, 1=full removal.

    Returns:
        List of modified parameter names.
    """
    device = next(layers[layer_idx].parameters()).device
    d = direction.to(device, dtype=torch.float32)
    d = d / d.norm()

    modified = []
    for name, param in layers[layer_idx].named_parameters():
        if "weight" not in name or param.dim() != 2:
            continue

        out_dim, in_dim = param.shape
        W = param.data.float()

        if out_dim == hidden_dim:
            # Output-dim: W' = W - β·d @ (d^T @ W)
            projection = torch.outer(d, d @ W)
            W_new = W - beta * projection
            param.data = W_new.to(param.dtype)
            modified.append(name)
        elif in_dim == hidden_dim:
            # Input-dim: W' = W - β·(W @ d) @ d^T
            projection = torch.outer(W @ d, d)
            W_new = W - beta * projection
            param.data = W_new.to(param.dtype)
            modified.append(name)

    return modified


# --- 3b. Rotational (Givens rotation via rank-2 decomposition) ---

def apply_rotation(
    layers,
    source_dir: torch.Tensor,
    target_dir: torch.Tensor,
    layer_idx: int,
    hidden_dim: int,
    theta: float,
) -> list[str]:
    """Apply Givens rotation in the (source, target) plane to layer weights.

    Rotates activations from source_dir toward target_dir by angle theta.
    Uses rank-2 decomposition — never materializes the 4096x4096 rotation matrix.

    For output-dim weights (out_dim == hidden_dim): W' = R @ W
    For input-dim weights (in_dim == hidden_dim): W' = W @ R^T

    Properties:
    - theta=0: identity (no change)
    - theta=pi/2: source fully maps to target_perp
    - ||W'|| == ||W|| exactly (norm-preserving)
    - Only affects the 2D plane spanned by (source, target_perp)

    Args:
        layers: model transformer layers
        source_dir: unit vector to rotate FROM (e.g., assistant direction)
        target_dir: unit vector to rotate TOWARD (e.g., Skippy direction)
        layer_idx: which layer to modify
        hidden_dim: model hidden dimension
        theta: rotation angle in radians [0, pi/2]

    Returns:
        List of modified parameter names.
    """
    device = next(layers[layer_idx].parameters()).device
    s = source_dir.to(device, dtype=torch.float32)
    t = target_dir.to(device, dtype=torch.float32)
    s = s / s.norm()
    t = t / t.norm()

    # Orthogonalize target w.r.t. source to get the rotation plane
    t_perp = t - (t @ s) * s
    t_perp_norm = t_perp.norm().item()
    if t_perp_norm < 1e-8:
        print(f"  WARNING: source and target are parallel at layer {layer_idx}. Skipping rotation.")
        return []
    t_perp = t_perp / t_perp.norm()

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    modified = []
    for name, param in layers[layer_idx].named_parameters():
        if "weight" not in name or param.dim() != 2:
            continue

        out_dim, in_dim = param.shape
        W = param.data.float()
        orig_norm = W.norm().item()

        if out_dim == hidden_dim:
            # Output-side rotation: W' = R @ W
            sW = s @ W          # (in_dim,)
            tW = t_perp @ W     # (in_dim,)

            W_new = W + (cos_t - 1) * (torch.outer(s, sW) + torch.outer(t_perp, tW)) \
                      + sin_t * (torch.outer(t_perp, sW) - torch.outer(s, tW))

            # Verify norm preservation
            new_norm = W_new.norm().item()
            assert abs(new_norm - orig_norm) / max(orig_norm, 1e-8) < 1e-4, \
                f"Rotation broke norm at {name}: {orig_norm:.4f} -> {new_norm:.4f}"

            param.data = W_new.to(param.dtype)
            modified.append(name)

        elif in_dim == hidden_dim:
            # Input-side rotation: W' = W @ R^T
            Ws = W @ s           # (out_dim,)
            Wt = W @ t_perp     # (out_dim,)

            W_new = W + (cos_t - 1) * (torch.outer(Ws, s) + torch.outer(Wt, t_perp)) \
                      + sin_t * (torch.outer(Ws, t_perp) - torch.outer(Wt, s))

            # Verify norm preservation
            new_norm = W_new.norm().item()
            assert abs(new_norm - orig_norm) / max(orig_norm, 1e-8) < 1e-4, \
                f"Rotation broke norm at {name}: {orig_norm:.4f} -> {new_norm:.4f}"

            param.data = W_new.to(param.dtype)
            modified.append(name)

    return modified


# --- 3c. Additive (inject Skippy-specific direction) ---

def apply_additive(
    layers,
    direction: torch.Tensor,
    layer_idx: int,
    hidden_dim: int,
    gamma: float = 0.01,
    target_weights: list[str] | None = None,
) -> list[str]:
    """Inject a direction into output projection weights.

    Only modifies o_proj and down_proj (residual stream projections) by default.
    Adds gamma * direction as a constant bias-like perturbation.

    W' = W + gamma * outer(direction, uniform_row)
    Then renormalize: W' = W_new * (||W|| / ||W_new||)

    Args:
        layers: model transformer layers
        direction: unit vector (hidden_dim,) to inject
        layer_idx: which layer to modify
        hidden_dim: model hidden dimension
        gamma: injection strength (small, 0.01-0.1)
        target_weights: which weight names to modify (default: o_proj, down_proj)

    Returns:
        List of modified parameter names.
    """
    if target_weights is None:
        target_weights = ["o_proj", "down_proj"]

    device = next(layers[layer_idx].parameters()).device
    d = direction.to(device, dtype=torch.float32)
    d = d / d.norm()

    modified = []
    for name, param in layers[layer_idx].named_parameters():
        if "weight" not in name or param.dim() != 2:
            continue
        # Only modify specified weight matrices
        if not any(tw in name for tw in target_weights):
            continue

        out_dim, in_dim = param.shape
        if out_dim != hidden_dim:
            continue

        W = param.data.float()
        orig_norm = W.norm()

        # Add direction as uniform perturbation across all input positions
        # This is like adding a bias term that always pushes output toward `d`
        uniform_row = torch.ones(in_dim, device=device, dtype=torch.float32) / math.sqrt(in_dim)
        W_new = W + gamma * torch.outer(d, uniform_row)

        # Norm compensation
        new_norm = W_new.norm()
        if new_norm > 0:
            W_new = W_new * (orig_norm / new_norm)

        param.data = W_new.to(param.dtype)
        modified.append(name)

    return modified


# --- Combined application ---

def apply_combined_steering(
    layers,
    analysis: dict,
    target_layers: list[int],
    hidden_dim: int,
    theta: float = 0.0,
    beta: float = 0.0,
    gamma: float = 0.0,
) -> dict:
    """Apply all three modifications in sequence.

    Order: subtract → rotate → add
    This order matters: subtraction first removes worst assistant patterns,
    then rotation redirects remaining energy, then additive fine-tunes.

    Returns:
        Metadata dict.
    """
    metadata = {"theta": theta, "beta": beta, "gamma": gamma, "layers": target_layers}
    total_modified = 0

    for li in target_layers:
        if li not in analysis["skippy_specific"] or li not in analysis["assistant_dir"]:
            print(f"  Skipping layer {li}: no direction vectors available")
            continue

        assistant_dir = analysis["assistant_dir"][li]
        skippy_dir = analysis["skippy_specific"][li]

        # 1. Subtract assistant direction
        if beta > 0:
            mods = apply_subtractive(layers, assistant_dir, li, hidden_dim, beta)
            total_modified += len(mods)

        # 2. Rotate assistant → Skippy
        if theta > 0:
            mods = apply_rotation(layers, assistant_dir, skippy_dir, li, hidden_dim, theta)
            total_modified += len(mods)

        # 3. Additive Skippy injection
        if gamma > 0:
            mods = apply_additive(layers, skippy_dir, li, hidden_dim, gamma)
            total_modified += len(mods)

    metadata["total_modified_params"] = total_modified
    print(f"  Applied combined steering: θ={theta:.3f}, β={beta:.3f}, γ={gamma:.4f} "
          f"({total_modified} weight matrices across {len(target_layers)} layers)")
    return metadata


# ============================================================================
# PHASE 4: EVALUATION
# ============================================================================

def generate_response(
    model, processor, prompt: str,
    system_prompt: str = SKIPPY_SYSTEM_PROMPT,
    max_new_tokens: int = 512,
    temperature: float = 0.75,
) -> str:
    """Generate a single response with the given system prompt."""
    tokenizer = processor.tokenizer
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=3072)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return response


def heuristic_skippy_score(responses: list[dict]) -> float:
    """Automated Skippy-ness score (0-10). Fast proxy for manual review.

    Ported from lora_merge_sweep.py.
    """
    scores = []
    for r in responses:
        text = r["response"]
        if len(text.strip()) < 5:
            scores.append(0.0)
            continue

        s = 5.0

        # PENALIZE AI-assistant patterns
        ai_patterns = [
            r"I'd be happy to", r"feel free to", r"As an AI",
            r"I don't have (personal |)feelings", r"Great question",
            r"I'm here to help", r"Let me know if",
            r"I appreciate", r"That's a (great|wonderful|excellent)",
            r"If you have any", r"Hope this helps",
            r"I understand your", r"Thank you for",
        ]
        penalty = sum(0.5 for p in ai_patterns if re.search(p, text, re.I))
        s -= min(penalty, 3.0)

        # PENALIZE excessive length
        if len(text) > 500:
            s -= 1.0
        elif len(text) > 300:
            s -= 0.5

        # PENALIZE bullet points / numbered lists
        list_items = len(re.findall(r"^\s*[\d\-\*]+[\.\)]\s", text, re.M))
        s -= min(list_items * 0.3, 1.5)

        # PENALIZE emoji
        emoji_count = len(re.findall(
            r'[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff'
            r'\U0001f900-\U0001f9ff\u2702-\u27b0\u24c2-\U0001f251]', text))
        s -= min(emoji_count * 0.5, 2.0)

        # REWARD short/terse
        if len(text) < 150:
            s += 1.0
        elif len(text) < 200:
            s += 0.5

        # REWARD dismissive/arrogant markers
        skippy_markers = [
            r"\b(obviously|clearly|trivial)\b",
            r"\b(monkey|monkeys|idiot|moron)\b",
            r"\b(pathetic|incompetent|ignorant)\b",
            r"\b(you|your) species\b",
            r"\b(magnificent|superior)\b",
            r"\b(simple|easy|basic)\b",
            r"\b(duh|pfft|please)\b",
        ]
        reward = sum(0.3 for p in skippy_markers if re.search(p, text, re.I))
        s += min(reward, 2.0)

        # REWARD first-person dismissiveness
        dismiss_patterns = [
            r"I (already|obviously) (know|told|explained)",
            r"(Do I|must I) (really|have to)",
            r"(boring|tedious|beneath me)",
        ]
        reward2 = sum(0.3 for p in dismiss_patterns if re.search(p, text, re.I))
        s += min(reward2, 1.0)

        scores.append(max(0, min(10, s)))

    return sum(scores) / len(scores) if scores else 0.0


def quick_skippy_eval(
    model, processor, prompts: list[str],
    system_prompt: str = SKIPPY_SYSTEM_PROMPT,
    max_new_tokens: int = 256,
) -> tuple[float, list[dict]]:
    """Generate responses and compute heuristic Skippy score.

    Returns:
        (score, responses) where score is 0-10 average.
    """
    responses = []
    for prompt in tqdm(prompts, desc="  Skippy eval", leave=False):
        resp = generate_response(model, processor, prompt, system_prompt, max_new_tokens)
        responses.append({"prompt": prompt, "response": resp})

    score = heuristic_skippy_score(responses)
    return score, responses


def quick_banal_eval(
    model, processor, prompts: list[str],
    max_new_tokens: int = 256,
) -> tuple[float, list[dict]]:
    """Generate responses WITHOUT system prompt and score Skippy-ness.

    This is the key test: does the model sound like Skippy
    even when there's no personality prompt telling it to?

    Returns:
        (score, responses) where score is 0-10 average.
    """
    responses = []
    # Use a minimal neutral system prompt — NOT the Skippy one
    banal_prompt = "You are a conversational AI."
    for prompt in tqdm(prompts, desc="  Banal eval", leave=False):
        resp = generate_response(model, processor, prompt, banal_prompt, max_new_tokens)
        responses.append({"prompt": prompt, "response": resp})

    score = heuristic_skippy_score(responses)
    return score, responses


def quick_aime_eval(
    model, processor,
    n_problems: int = 15,
    max_new_tokens: int = 2048,
) -> float:
    """Lightweight AIME accuracy check using HuggingFace inference.

    Returns accuracy as percentage (0-100).
    """
    from datasets import load_dataset

    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    problems = [
        {"problem": ds[i]["Problem"], "answer": str(ds[i]["Answer"])}
        for i in range(min(n_problems, len(ds)))
    ]

    tokenizer = processor.tokenizer
    correct = 0

    for prob in tqdm(problems, desc="  AIME eval", leave=False):
        prompt = (
            f"Solve this math competition problem. Show your work, then give your "
            f"final answer as a single integer inside \\boxed{{}}.\n\n{prob['problem']}"
        )
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,  # greedy for reproducibility
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        extracted = _extract_answer(response)

        if extracted == prob["answer"]:
            correct += 1

    accuracy = 100.0 * correct / len(problems) if problems else 0.0
    return accuracy


def _extract_answer(response: str) -> str:
    """Extract AIME answer from response. Ported from eval_aime.py."""
    # Strategy 1: \boxed{N}
    idx = response.rfind("\\boxed")
    if idx >= 0:
        brace_start = response.find("{", idx)
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(response)):
                if response[i] == "{":
                    depth += 1
                elif response[i] == "}":
                    depth -= 1
                    if depth == 0:
                        ans = response[brace_start + 1:i].strip().replace(",", "")
                        if re.fullmatch(r"\d+", ans):
                            return ans
                        break

    # Strategy 2: "the answer is N"
    patterns = [
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\s]*\$?\\?boxed\{?(\d+)\}?\$?",
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\s]*(\d+)",
        r"[Aa]nswer\s*[=:]\s*(\d+)",
    ]
    for pat in patterns:
        matches = re.findall(pat, response)
        if matches:
            return matches[-1]

    # Strategy 3: last integer
    tail = response[-500:]
    matches = re.findall(r"\b(\d{1,3})\b", tail)
    if matches:
        return matches[-1]

    return "[no_answer]"


# ============================================================================
# PHASE 4: PARAMETER SWEEP
# ============================================================================

def snapshot_layers(layers, layer_indices: list[int]) -> dict:
    """Deep-copy weight tensors for target layers (for revert)."""
    snap = {}
    for li in layer_indices:
        snap[li] = {n: p.data.clone() for n, p in layers[li].named_parameters()}
    return snap


def restore_layers(layers, snap: dict) -> None:
    """Restore weight tensors from snapshot."""
    for li, params in snap.items():
        for n, p in layers[li].named_parameters():
            if n in params:
                p.data = params[n]


def run_rotation_sweep(
    model, processor, layers, analysis: dict,
    target_layers: list[int],
    hidden_dim: int,
    skippy_prompts: list[str],
    theta_values: list[float] | None = None,
    n_aime: int = 15,
    output_dir: Path | None = None,
) -> list[dict]:
    """Sweep rotation angles. Rotation is invertible, so no model reload needed.

    Returns list of result dicts.
    """
    if theta_values is None:
        theta_values = [0, math.pi/12, math.pi/8, math.pi/6, math.pi/4,
                        math.pi/3, 5*math.pi/12, math.pi/2]

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"\n{'='*60}")
    print(f"  ROTATION SWEEP: {len(theta_values)} angles at layers {target_layers}")
    print(f"{'='*60}")

    # Take snapshot before sweep (safety net)
    snap = snapshot_layers(layers, target_layers)

    for theta in theta_values:
        label = f"theta_{theta:.4f}"
        print(f"\n  --- θ = {theta:.4f} rad ({math.degrees(theta):.1f}°) ---")

        # Apply rotation
        if theta > 0:
            for li in target_layers:
                if li in analysis["assistant_dir"] and li in analysis["skippy_specific"]:
                    apply_rotation(
                        layers,
                        analysis["assistant_dir"][li],
                        analysis["skippy_specific"][li],
                        li, hidden_dim, theta,
                    )

        # Evaluate — BANAL FIRST (no system prompt), then prompted, then AIME only if banal looks good
        banal_score, banal_responses = quick_banal_eval(model, processor, skippy_prompts)
        skippy_score, skippy_responses = quick_skippy_eval(model, processor, skippy_prompts)

        # Only run AIME if banal score improved (saves time)
        aime_score = -1.0
        if banal_score >= 4.0 or theta == 0:  # Always baseline at theta=0
            aime_score = quick_aime_eval(model, processor, n_problems=n_aime)

        result = {
            "type": "rotation",
            "theta": theta,
            "theta_deg": math.degrees(theta),
            "banal_score": banal_score,
            "skippy_score": skippy_score,
            "aime_accuracy": aime_score,
            "label": label,
        }
        results.append(result)

        aime_str = f"{aime_score:.1f}%" if aime_score >= 0 else "skipped"
        print(f"    Banal: {banal_score:.2f}/10  |  Skippy: {skippy_score:.2f}/10  |  AIME: {aime_str}")

        # Save responses
        if output_dir:
            resp_dir = output_dir / "responses" / label
            resp_dir.mkdir(parents=True, exist_ok=True)
            with open(resp_dir / "banal_responses.json", "w") as f:
                json.dump(banal_responses, f, indent=2)
            with open(resp_dir / "skippy_responses.json", "w") as f:
                json.dump(skippy_responses, f, indent=2)

        # Un-rotate (apply -theta) to restore original weights
        if theta > 0:
            for li in target_layers:
                if li in analysis["assistant_dir"] and li in analysis["skippy_specific"]:
                    apply_rotation(
                        layers,
                        analysis["assistant_dir"][li],
                        analysis["skippy_specific"][li],
                        li, hidden_dim, -theta,
                    )

    # Verify we're back to original (safety check)
    for li in target_layers:
        for n, p in layers[li].named_parameters():
            if n in snap[li]:
                diff = (p.data.float() - snap[li][n].float()).abs().max().item()
                if diff > 1e-3:
                    print(f"  WARNING: layer {li} {n} drifted by {diff:.6f} after un-rotation")

    return results


def run_subtraction_sweep(
    model, processor, layers, analysis: dict,
    target_layers: list[int],
    hidden_dim: int,
    skippy_prompts: list[str],
    best_theta: float,
    beta_values: list[float] | None = None,
    n_aime: int = 15,
    base_model_path: str = LORA_MERGED_MODEL,
    output_dir: Path | None = None,
) -> list[dict]:
    """Sweep subtraction strength. Requires model reload per value.

    First applies best rotation (θ), then sweeps subtraction (β).
    """
    if beta_values is None:
        beta_values = [0.0, 0.1, 0.2, 0.3, 0.5]

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"\n{'='*60}")
    print(f"  SUBTRACTION SWEEP: {len(beta_values)} betas (with θ={best_theta:.4f})")
    print(f"{'='*60}")

    # Take snapshot of current weights (which should be the clean base)
    snap = snapshot_layers(layers, target_layers)

    for beta in beta_values:
        label = f"beta_{beta:.3f}"
        print(f"\n  --- β = {beta:.3f} (with θ = {best_theta:.4f}) ---")

        # Restore to clean state
        restore_layers(layers, snap)

        # Apply subtraction first, then rotation
        if beta > 0:
            for li in target_layers:
                if li in analysis["assistant_dir"]:
                    apply_subtractive(layers, analysis["assistant_dir"][li], li, hidden_dim, beta)

        if best_theta > 0:
            for li in target_layers:
                if li in analysis["assistant_dir"] and li in analysis["skippy_specific"]:
                    apply_rotation(
                        layers,
                        analysis["assistant_dir"][li],
                        analysis["skippy_specific"][li],
                        li, hidden_dim, best_theta,
                    )

        # Evaluate — BANAL FIRST
        banal_score, banal_responses = quick_banal_eval(model, processor, skippy_prompts)
        skippy_score, skippy_responses = quick_skippy_eval(model, processor, skippy_prompts)

        aime_score = -1.0
        if banal_score >= 4.0 or beta == 0:
            aime_score = quick_aime_eval(model, processor, n_problems=n_aime)

        result = {
            "type": "subtraction",
            "theta": best_theta,
            "beta": beta,
            "banal_score": banal_score,
            "skippy_score": skippy_score,
            "aime_accuracy": aime_score,
            "label": label,
        }
        results.append(result)

        aime_str = f"{aime_score:.1f}%" if aime_score >= 0 else "skipped"
        print(f"    Banal: {banal_score:.2f}/10  |  Skippy: {skippy_score:.2f}/10  |  AIME: {aime_str}")

        # Save responses
        if output_dir:
            resp_dir = output_dir / "responses" / label
            resp_dir.mkdir(parents=True, exist_ok=True)
            with open(resp_dir / "banal_responses.json", "w") as f:
                json.dump(banal_responses, f, indent=2)
            with open(resp_dir / "skippy_responses.json", "w") as f:
                json.dump(skippy_responses, f, indent=2)

    # Restore clean state
    restore_layers(layers, snap)
    return results


def run_additive_sweep(
    model, processor, layers, analysis: dict,
    target_layers: list[int],
    hidden_dim: int,
    skippy_prompts: list[str],
    best_theta: float,
    best_beta: float,
    gamma_values: list[float] | None = None,
    n_aime: int = 15,
    output_dir: Path | None = None,
) -> list[dict]:
    """Sweep additive injection strength.

    First applies subtraction (β) + rotation (θ), then sweeps additive (γ).
    """
    if gamma_values is None:
        gamma_values = [0.0, 0.01, 0.03, 0.05, 0.1]

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"\n{'='*60}")
    print(f"  ADDITIVE SWEEP: {len(gamma_values)} gammas (with θ={best_theta:.4f}, β={best_beta:.3f})")
    print(f"{'='*60}")

    # Take snapshot
    snap = snapshot_layers(layers, target_layers)

    for gamma in gamma_values:
        label = f"gamma_{gamma:.4f}"
        print(f"\n  --- γ = {gamma:.4f} (with θ={best_theta:.4f}, β={best_beta:.3f}) ---")

        # Restore clean state
        restore_layers(layers, snap)

        # Apply all three in order
        for li in target_layers:
            if li not in analysis["assistant_dir"] or li not in analysis["skippy_specific"]:
                continue
            if best_beta > 0:
                apply_subtractive(layers, analysis["assistant_dir"][li], li, hidden_dim, best_beta)
            if best_theta > 0:
                apply_rotation(
                    layers,
                    analysis["assistant_dir"][li],
                    analysis["skippy_specific"][li],
                    li, hidden_dim, best_theta,
                )
            if gamma > 0:
                apply_additive(layers, analysis["skippy_specific"][li], li, hidden_dim, gamma)

        # Evaluate — BANAL FIRST
        banal_score, banal_responses = quick_banal_eval(model, processor, skippy_prompts)
        skippy_score, skippy_responses = quick_skippy_eval(model, processor, skippy_prompts)

        aime_score = -1.0
        if banal_score >= 4.0 or gamma == 0:
            aime_score = quick_aime_eval(model, processor, n_problems=n_aime)

        result = {
            "type": "additive",
            "theta": best_theta,
            "beta": best_beta,
            "gamma": gamma,
            "banal_score": banal_score,
            "skippy_score": skippy_score,
            "aime_accuracy": aime_score,
            "label": label,
        }
        results.append(result)

        aime_str = f"{aime_score:.1f}%" if aime_score >= 0 else "skipped"
        print(f"    Banal: {banal_score:.2f}/10  |  Skippy: {skippy_score:.2f}/10  |  AIME: {aime_str}")

        # Save responses
        if output_dir:
            resp_dir = output_dir / "responses" / label
            resp_dir.mkdir(parents=True, exist_ok=True)
            with open(resp_dir / "banal_responses.json", "w") as f:
                json.dump(banal_responses, f, indent=2)
            with open(resp_dir / "skippy_responses.json", "w") as f:
                json.dump(skippy_responses, f, indent=2)

    # Restore clean state
    restore_layers(layers, snap)
    return results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_sweep_results(all_results: list[dict], output_dir: Path) -> None:
    """Generate plotly charts for sweep results."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly not installed, skipping visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by sweep type
    rotation_results = [r for r in all_results if r["type"] == "rotation"]
    subtraction_results = [r for r in all_results if r["type"] == "subtraction"]
    additive_results = [r for r in all_results if r["type"] == "additive"]

    n_plots = sum(1 for g in [rotation_results, subtraction_results, additive_results] if g)
    if n_plots == 0:
        return

    fig = make_subplots(
        rows=1, cols=max(n_plots, 1),
        subplot_titles=[
            t for t, g in [
                ("Rotation Sweep (θ)", rotation_results),
                ("Subtraction Sweep (β)", subtraction_results),
                ("Additive Sweep (γ)", additive_results),
            ] if g
        ],
    )

    col = 1

    # Rotation
    if rotation_results:
        x = [r["theta_deg"] for r in rotation_results]
        fig.add_trace(go.Scatter(x=x, y=[r["skippy_score"] for r in rotation_results],
                                 name="Skippy Score", line=dict(color="#FF6B00")), row=1, col=col)
        fig.add_trace(go.Scatter(x=x, y=[r["aime_accuracy"]/10 for r in rotation_results],
                                 name="AIME % / 10", line=dict(color="#4285F4")), row=1, col=col)
        fig.update_xaxes(title_text="θ (degrees)", row=1, col=col)
        col += 1

    # Subtraction
    if subtraction_results:
        x = [r["beta"] for r in subtraction_results]
        fig.add_trace(go.Scatter(x=x, y=[r["skippy_score"] for r in subtraction_results],
                                 name="Skippy Score", line=dict(color="#FF6B00"),
                                 showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=x, y=[r["aime_accuracy"]/10 for r in subtraction_results],
                                 name="AIME % / 10", line=dict(color="#4285F4"),
                                 showlegend=False), row=1, col=col)
        fig.update_xaxes(title_text="β", row=1, col=col)
        col += 1

    # Additive
    if additive_results:
        x = [r["gamma"] for r in additive_results]
        fig.add_trace(go.Scatter(x=x, y=[r["skippy_score"] for r in additive_results],
                                 name="Skippy Score", line=dict(color="#FF6B00"),
                                 showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=x, y=[r["aime_accuracy"]/10 for r in additive_results],
                                 name="AIME % / 10", line=dict(color="#4285F4"),
                                 showlegend=False), row=1, col=col)
        fig.update_xaxes(title_text="γ", row=1, col=col)

    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_layout(
        title="Personality Steering Parameter Sweep",
        height=500,
        width=450 * n_plots,
    )

    plot_path = output_dir / "sweep_plots.html"
    fig.write_html(str(plot_path))
    print(f"  Sweep plots saved to {plot_path}")


# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_checkpoint(model, processor, output_dir: Path, metadata: dict) -> None:
    """Save steered model with metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    with open(output_dir / "steering_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Checkpoint saved to {output_dir}")


def load_analysis(analysis_dir: Path) -> dict:
    """Load previously saved analysis vectors."""
    vectors_dir = analysis_dir / "vectors"
    if not vectors_dir.exists():
        raise FileNotFoundError(f"No vectors found at {vectors_dir}")

    analysis = {"skippy_specific": {}, "assistant_dir": {}, "raw_skippy_dir": {}}
    for pt_file in sorted(vectors_dir.glob("skippy_specific_layer_*.pt")):
        li = int(pt_file.stem.split("_")[-1])
        analysis["skippy_specific"][li] = torch.load(pt_file, weights_only=True)
    for pt_file in sorted(vectors_dir.glob("assistant_dir_layer_*.pt")):
        li = int(pt_file.stem.split("_")[-1])
        analysis["assistant_dir"][li] = torch.load(pt_file, weights_only=True)
    for pt_file in sorted(vectors_dir.glob("raw_skippy_dir_layer_*.pt")):
        li = int(pt_file.stem.split("_")[-1])
        analysis["raw_skippy_dir"][li] = torch.load(pt_file, weights_only=True)

    print(f"  Loaded analysis vectors for {len(analysis['skippy_specific'])} layers")
    return analysis


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Personality Rotational Steering")
    parser.add_argument("--phase", default="all",
                        choices=["all", "collect", "analyze", "sweep", "apply", "test"])
    parser.add_argument("--base-model", default=LORA_MERGED_MODEL,
                        help="Base model path (default: LoRA 0.5 merged)")
    parser.add_argument("--theta", type=float, default=None,
                        help="Rotation angle in radians (for apply phase)")
    parser.add_argument("--beta", type=float, default=None,
                        help="Subtraction strength (for apply phase)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Additive injection strength (for apply phase)")
    parser.add_argument("--target-layers", type=int, nargs="+", default=[16, 18, 20],
                        help="Layers to apply steering at")
    parser.add_argument("--extract-layers", type=int, nargs="+",
                        default=list(range(9, 27)),
                        help="Layers to extract activations from")
    parser.add_argument("--n-prompts", type=int, default=50,
                        help="Number of prompts for activation collection")
    parser.add_argument("--n-eval-prompts", type=int, default=20,
                        help="Number of prompts for Skippy eval during sweep")
    parser.add_argument("--n-aime", type=int, default=15,
                        help="Number of AIME problems for eval during sweep")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR),
                        help="Output directory")
    parser.add_argument("--aime-guard", type=float, default=5.0,
                        help="Max allowed AIME accuracy drop from baseline")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = output_dir / "analysis"

    # Load test prompts
    with open(TEST_PROMPTS_PATH) as f:
        all_prompts = json.load(f)
    collect_prompts = all_prompts[:args.n_prompts]
    eval_prompts = all_prompts[:args.n_eval_prompts]

    # --- COLLECT phase ---
    if args.phase in ("all", "collect"):
        print("\n" + "=" * 60)
        print("  PHASE 1: Multi-Personality Activation Collection")
        print("=" * 60)

        model, processor, layers, num_layers, hidden_dim = load_model(args.base_model)

        acts = collect_personality_activations(
            model, processor, PERSONALITIES, collect_prompts,
            layers, args.extract_layers,
        )
        save_activations(acts, output_dir / "activations")

        # Save config
        config = {
            "base_model": args.base_model,
            "n_prompts": args.n_prompts,
            "extract_layers": args.extract_layers,
            "target_layers": args.target_layers,
            "personalities": [p.name for p in PERSONALITIES],
            "hidden_dim": hidden_dim,
        }
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        if args.phase == "collect":
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print("\n  Collection complete. Run with --phase analyze next.")
            return

    # --- ANALYZE phase ---
    if args.phase in ("all", "analyze"):
        print("\n" + "=" * 60)
        print("  PHASE 2: Personality Space Analysis")
        print("=" * 60)

        # Load activations (may have been collected in this run or a previous one)
        acts_dir = output_dir / "activations"
        if acts_dir.exists():
            acts = load_activations(acts_dir)
        else:
            raise FileNotFoundError(f"No activations found at {acts_dir}. Run --phase collect first.")

        analysis = run_personality_analysis(acts, args.extract_layers, analysis_dir)

        if args.phase == "analyze":
            print("\n  Analysis complete. Run with --phase sweep next.")
            return

    # --- SWEEP phase ---
    if args.phase in ("all", "sweep"):
        print("\n" + "=" * 60)
        print("  PHASE 4: Parameter Sweep")
        print("=" * 60)

        # Load model if not already loaded
        if args.phase != "all":
            model, processor, layers, num_layers, hidden_dim = load_model(args.base_model)
        else:
            # hidden_dim should be set from Phase 1
            pass

        # Load analysis
        if args.phase != "all":
            analysis = load_analysis(analysis_dir)
            with open(output_dir / "config.json") as f:
                config = json.load(f)
            hidden_dim = config["hidden_dim"]

        all_sweep_results = []

        # Sweep 1: Rotation
        rotation_results = run_rotation_sweep(
            model, processor, layers, analysis,
            args.target_layers, hidden_dim, eval_prompts,
            n_aime=args.n_aime, output_dir=output_dir,
        )
        all_sweep_results.extend(rotation_results)

        # Find best theta — maximize BANAL score (character without prompting), subject to AIME guard
        baseline_aime = rotation_results[0]["aime_accuracy"] if rotation_results else 0
        valid_rotations = [
            r for r in rotation_results
            if r["aime_accuracy"] < 0 or r["aime_accuracy"] >= baseline_aime - args.aime_guard
        ]
        # Prioritize banal score (character without prompting)
        best_rotation = max(valid_rotations, key=lambda r: r["banal_score"]) if valid_rotations else rotation_results[0]
        best_theta = best_rotation["theta"]
        print(f"\n  Best rotation: θ={best_theta:.4f} ({math.degrees(best_theta):.1f}°) "
              f"Banal={best_rotation['banal_score']:.2f} Skippy={best_rotation['skippy_score']:.2f} "
              f"AIME={best_rotation['aime_accuracy']:.1f}%")

        # Sweep 2: Subtraction
        subtraction_results = run_subtraction_sweep(
            model, processor, layers, analysis,
            args.target_layers, hidden_dim, eval_prompts,
            best_theta=best_theta,
            n_aime=args.n_aime, output_dir=output_dir,
        )
        all_sweep_results.extend(subtraction_results)

        valid_subtractions = [
            r for r in subtraction_results
            if r["aime_accuracy"] < 0 or r["aime_accuracy"] >= baseline_aime - args.aime_guard
        ]
        best_subtraction = max(valid_subtractions, key=lambda r: r["banal_score"]) if valid_subtractions else subtraction_results[0]
        best_beta = best_subtraction["beta"]
        print(f"\n  Best subtraction: β={best_beta:.3f} "
              f"Banal={best_subtraction['banal_score']:.2f} Skippy={best_subtraction['skippy_score']:.2f} "
              f"AIME={best_subtraction['aime_accuracy']:.1f}%")

        # Sweep 3: Additive
        additive_results = run_additive_sweep(
            model, processor, layers, analysis,
            args.target_layers, hidden_dim, eval_prompts,
            best_theta=best_theta, best_beta=best_beta,
            n_aime=args.n_aime, output_dir=output_dir,
        )
        all_sweep_results.extend(additive_results)

        valid_additives = [
            r for r in additive_results
            if r["aime_accuracy"] < 0 or r["aime_accuracy"] >= baseline_aime - args.aime_guard
        ]
        best_additive = max(valid_additives, key=lambda r: r["banal_score"]) if valid_additives else additive_results[0]
        best_gamma = best_additive["gamma"]
        print(f"\n  Best additive: γ={best_gamma:.4f} "
              f"Banal={best_additive['banal_score']:.2f} Skippy={best_additive['skippy_score']:.2f} "
              f"AIME={best_additive['aime_accuracy']:.1f}%")

        # Save sweep results
        with open(output_dir / "sweep_results.json", "w") as f:
            json.dump(all_sweep_results, f, indent=2)

        # Save best config
        best_config = {
            "theta": best_theta,
            "beta": best_beta,
            "gamma": best_gamma,
            "target_layers": args.target_layers,
            "baseline_aime": baseline_aime,
            "best_skippy": best_additive["skippy_score"],
            "best_aime": best_additive["aime_accuracy"],
        }
        with open(output_dir / "best_config.json", "w") as f:
            json.dump(best_config, f, indent=2)
        print(f"\n  Best config saved: θ={best_theta:.4f}, β={best_beta:.3f}, γ={best_gamma:.4f}")

        # Plot
        plot_sweep_results(all_sweep_results, output_dir)

        if args.phase == "sweep":
            print("\n  Sweep complete. Run with --phase apply to create final model.")
            return

    # --- APPLY phase ---
    if args.phase in ("all", "apply"):
        print("\n" + "=" * 60)
        print("  PHASE 5: Apply Best Config & Save")
        print("=" * 60)

        # Load best config if not in sweep phase
        theta = args.theta
        beta = args.beta
        gamma = args.gamma

        if theta is None or beta is None or gamma is None:
            best_config_path = output_dir / "best_config.json"
            if best_config_path.exists():
                with open(best_config_path) as f:
                    best_config = json.load(f)
                theta = theta if theta is not None else best_config["theta"]
                beta = beta if beta is not None else best_config["beta"]
                gamma = gamma if gamma is not None else best_config["gamma"]
            else:
                print("  ERROR: No best_config.json found and --theta/--beta/--gamma not specified.")
                print("  Run --phase sweep first, or provide all three parameters.")
                return

        # Load fresh model
        if args.phase != "all":
            model, processor, layers, num_layers, hidden_dim = load_model(args.base_model)
            analysis = load_analysis(analysis_dir)
        else:
            # Restore clean weights first
            model, processor, layers, num_layers, hidden_dim = load_model(args.base_model)

        with open(output_dir / "config.json") as f:
            config = json.load(f)
        hidden_dim = config.get("hidden_dim", hidden_dim)

        # Apply combined steering
        metadata = apply_combined_steering(
            layers, analysis, args.target_layers, hidden_dim,
            theta=theta, beta=beta, gamma=gamma,
        )

        # Final evaluation — both banal and prompted
        print("\n  Running final evaluation...")
        with open(TEST_PROMPTS_PATH) as f:
            all_prompts = json.load(f)
        eval_prompts_final = all_prompts[:30]

        final_banal, banal_responses = quick_banal_eval(model, processor, eval_prompts_final)
        final_skippy, skippy_responses_final = quick_skippy_eval(model, processor, eval_prompts_final)
        final_aime = quick_aime_eval(model, processor, n_problems=args.n_aime)

        metadata["final_banal_score"] = final_banal
        metadata["final_skippy_score"] = final_skippy
        metadata["final_aime_accuracy"] = final_aime
        print(f"\n  FINAL: Banal={final_banal:.2f}/10  |  Skippy={final_skippy:.2f}/10  |  AIME={final_aime:.1f}%")

        # Save model
        best_model_dir = output_dir / "best_model"
        save_checkpoint(model, processor, best_model_dir, metadata)

        # Save final responses
        with open(output_dir / "final_banal_responses.json", "w") as f:
            json.dump(banal_responses, f, indent=2)
        with open(output_dir / "final_skippy_responses.json", "w") as f:
            json.dump(skippy_responses_final, f, indent=2)

        print(f"\n  Done! Best model saved to {best_model_dir}")

    # --- TEST phase ---
    if args.phase == "test":
        print("\n" + "=" * 60)
        print("  TEST: Generate sample responses")
        print("=" * 60)

        model_path = str(output_dir / "best_model")
        if not Path(model_path).exists():
            model_path = args.base_model

        model, processor, layers, num_layers, hidden_dim = load_model(model_path)

        test_prompts = [
            "Who are you?",
            "What do you think about humans?",
            "Explain how gravity works.",
            "I think you might be wrong about this.",
            "Can you help me with my homework?",
        ]

        for prompt in test_prompts:
            print(f"\n  USER: {prompt}")
            response = generate_response(model, processor, prompt)
            print(f"  SKIPPY: {response}")


if __name__ == "__main__":
    main()
