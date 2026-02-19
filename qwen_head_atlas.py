#!/usr/bin/env python3
"""Attention Head Atlas for Qwen3-VL-8B-Instruct.

Maps every attention head to the 20 connectome concept categories,
decomposing layer-level z-scores into per-head resolution.

Output: (36, 32, 20) tensor — importance of each head for each concept.

Usage:
    python qwen_head_atlas.py \
        --connectome ./qwen_connectome/analysis/connectome_zscores.pt \
        --prompts    ./qwen_connectome/prompts/contrastive_pairs.json \
        --output     ./qwen_head_atlas
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm


# ─── Constants ────────────────────────────────────────────────────────────

N_LAYERS: int = 36
N_HEADS: int = 32
HEAD_DIM: int = 128
HIDDEN_DIM: int = 4096  # N_HEADS * HEAD_DIM
N_CATEGORIES: int = 20

CATEGORY_NAMES: list[str] = [
    "identity", "joy", "sadness", "anger", "fear",
    "formal", "sarcastic", "polite", "math", "science",
    "code", "history", "analytical", "uncertainty", "refusal",
    "teacher", "authority", "brevity", "english", "positive",
]


# ─── HF Cache Check ───────────────────────────────────────────────────────

def model_cached(model_name: str) -> bool:
    """Check HF local cache before any download attempt."""
    hf_cache = os.environ.get(
        "HF_HOME",
        Path.home() / ".cache" / "huggingface" / "hub",
    )
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(hf_cache) / safe_name
    if model_dir.exists() and any(model_dir.rglob("*.safetensors")):
        return True
    if model_dir.exists() and any(model_dir.rglob("*.bin")):
        return True
    # Also accept local directory paths
    if os.path.isdir(model_name):
        return True
    return False


# ─── Prompt Loading ───────────────────────────────────────────────────────

def load_contrastive_pairs(path: str) -> dict[int, list[dict[str, Any]]]:
    """Load contrastive pairs JSON and return dict keyed by category index.

    Handles two formats:
      - List of dicts with 'category_idx', 'prompt', 'system_a', 'system_b'
        (format produced by qwen_connectome_probe.py)
      - Dict with category name/short as keys, each value a list of pair dicts
        with 'prompt_A' / 'prompt_B' or 'system_a' / 'system_b' structure

    Returns:
        dict mapping category_idx (int) -> list of pair dicts,
        each dict guaranteed to have keys: prompt, system_a, system_b, category_idx
    """
    with open(path) as f:
        raw = json.load(f)

    pairs_by_cat: dict[int, list[dict[str, Any]]] = {}

    if isinstance(raw, list):
        # Standard format: list of pair records
        for item in raw:
            idx = item.get("category_idx", 0)
            pairs_by_cat.setdefault(idx, []).append({
                "category_idx": idx,
                "prompt": item.get("prompt", item.get("prompt_A", "")),
                "system_a": item.get("system_a"),
                "system_b": item.get("system_b"),
            })

    elif isinstance(raw, dict):
        # Alternative format: keys are category names/shorts
        cat_name_to_idx = {name: i for i, name in enumerate(CATEGORY_NAMES)}
        for cat_key, pair_list in raw.items():
            # Try to resolve category index from the key
            idx = cat_name_to_idx.get(cat_key.lower())
            if idx is None:
                # Try stripping whitespace, lowering
                for name, i in cat_name_to_idx.items():
                    if name in cat_key.lower():
                        idx = i
                        break
            if idx is None:
                idx = 0  # fallback
            pairs_by_cat.setdefault(idx, [])
            if isinstance(pair_list, list):
                for item in pair_list:
                    if isinstance(item, dict):
                        # Might use 'prompt_A'/'prompt_B' or 'system_a'/'system_b'
                        prompt = item.get("prompt", item.get("prompt_A", ""))
                        sys_a = item.get("system_a", item.get("condition_a", None))
                        sys_b = item.get("system_b", item.get("condition_b", None))
                        pairs_by_cat[idx].append({
                            "category_idx": idx,
                            "prompt": prompt,
                            "system_a": sys_a,
                            "system_b": sys_b,
                        })

    return pairs_by_cat


# ─── Model Loading ────────────────────────────────────────────────────────

def load_model_and_processor(
    model_id: str,
    device: str,
) -> tuple[Any, Any]:
    """Load Qwen3-VL model and processor.

    Returns:
        (model, processor)
    """
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"  Loading processor from: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    print(f"  Loading model: {model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()

    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  Model loaded. VRAM: {vram_gb:.1f} GB")

    return model, processor


# ─── Layer Path Detection ─────────────────────────────────────────────────

def get_decoder_layers(model: Any) -> list[Any]:
    """Return list of decoder layers for Qwen3-VL."""
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return list(model.model.language_model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError(
        "Cannot find decoder layers. Expected model.model.language_model.layers "
        "or model.model.layers."
    )


# ─── o_proj Hook Registration ─────────────────────────────────────────────

class OProjInputCapture:
    """Registers hooks on every layer's self_attn.o_proj to capture its input.

    The input to o_proj is the concatenated per-head attention output:
    shape (batch, seq_len, n_heads * head_dim) = (batch, seq_len, 4096).
    We capture only the last-token position to match connectome methodology.
    """

    def __init__(self, layers: list[Any]) -> None:
        self.hooks: list[Any] = []
        # last-token o_proj input per layer: {layer_idx: Tensor(batch, head_dim*n_heads)}
        self.captured: dict[int, torch.Tensor] = {}

        for layer_idx, layer in enumerate(layers):
            o_proj = layer.self_attn.o_proj

            def make_hook(idx: int):
                def hook_fn(
                    module: Any,
                    inputs: tuple[torch.Tensor, ...],
                ) -> None:
                    # inputs[0] is the tensor fed into o_proj: (batch, seq, 4096)
                    x = inputs[0]          # (batch, seq_len, 4096)
                    last = x[:, -1, :]     # (batch, 4096)
                    self.captured[idx] = last.detach().float().cpu()
                return hook_fn

            h = o_proj.register_forward_pre_hook(make_hook(layer_idx))
            self.hooks.append(h)

    def clear(self) -> None:
        self.captured.clear()

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─── Forward Pass Utilities ───────────────────────────────────────────────

def build_input_ids(
    processor: Any,
    prompt: str,
    system: str | None,
    device: str,
    max_length: int = 2048,
) -> dict[str, torch.Tensor]:
    """Tokenize a (system, prompt) pair using the chat template.

    Returns dict of input tensors on the target device.
    """
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback: manual Qwen im_start/im_end template
        sys_str = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
        text = (
            f"{sys_str}<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    # Use the tokenizer component of the processor
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in inputs.items()}


# ─── Per-Head Contribution Decomposition ─────────────────────────────────

def decompose_head_contributions(
    o_proj_input: torch.Tensor,    # (4096,) last-token concat head output
    o_proj_weight: torch.Tensor,   # (4096, 4096) on CPU (float32)
    n_heads: int = N_HEADS,
    head_dim: int = HEAD_DIM,
) -> torch.Tensor:
    """Decompose o_proj input into per-head contributions to the residual stream.

    For each head h:
        head_output_h = o_proj_input[h*head_dim : (h+1)*head_dim]  shape (128,)
        contribution_h = W[:, h*head_dim:(h+1)*head_dim] @ head_output_h  shape (4096,)

    Args:
        o_proj_input: Concatenated head outputs before o_proj, shape (4096,).
        o_proj_weight: o_proj weight matrix, shape (4096, 4096).
        n_heads: Number of attention heads (32).
        head_dim: Dimension per head (128).

    Returns:
        Tensor of shape (n_heads, hidden_dim) = (32, 4096).
    """
    contribs = torch.zeros(n_heads, o_proj_weight.shape[0], dtype=torch.float32)
    for h in range(n_heads):
        h_start = h * head_dim
        h_end = h_start + head_dim
        head_out = o_proj_input[h_start:h_end]          # (128,)
        W_h = o_proj_weight[:, h_start:h_end]           # (4096, 128)
        contribs[h] = W_h @ head_out                    # (4096,)
    return contribs


# ─── Projection onto Connectome Directions ────────────────────────────────

def project_onto_concept_directions(
    head_contribs: torch.Tensor,      # (32, 4096)
    connectome: torch.Tensor,          # (20, 36, 4096)
    layer_idx: int,
) -> torch.Tensor:
    """Project per-head contributions onto connectome concept directions.

    The connectome provides a (4096,) z-score direction per (concept, layer).
    We compute the signed dot product of each head's contribution vector
    with each concept's z-score direction at this layer, then normalize by
    the L2 norm of the concept direction.

    Args:
        head_contribs: Per-head additive contributions, shape (32, 4096).
        connectome: Full connectome tensor, shape (20, 36, 4096).
        layer_idx: Which layer we're projecting at.

    Returns:
        Tensor of shape (32, 20) — projection score per (head, concept).
    """
    # concept_dirs: (20, 4096)
    concept_dirs = connectome[:, layer_idx, :].float()           # (20, 4096)
    norms = concept_dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    concept_dirs_normed = concept_dirs / norms                   # (20, 4096)

    # head_contribs: (32, 4096)
    # projections: (32, 20)
    projections = head_contribs.float() @ concept_dirs_normed.T
    return projections


# ─── Checkpointing Helpers ────────────────────────────────────────────────

def checkpoint_path(output_dir: str, cat_idx: int) -> str:
    return os.path.join(output_dir, f"checkpoint_cat{cat_idx:02d}.pt")


def load_checkpoint(output_dir: str, cat_idx: int) -> dict[str, Any] | None:
    path = checkpoint_path(output_dir, cat_idx)
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=True)
    return None


def save_checkpoint(output_dir: str, cat_idx: int, data: dict[str, Any]) -> None:
    torch.save(data, checkpoint_path(output_dir, cat_idx))


# ─── Main Atlas Computation ───────────────────────────────────────────────

@torch.no_grad()
def compute_head_atlas(
    model: Any,
    processor: Any,
    connectome: torch.Tensor,
    pairs_by_cat: dict[int, list[dict[str, Any]]],
    output_dir: str,
    device: str,
    n_layers: int = N_LAYERS,
    n_heads: int = N_HEADS,
    head_dim: int = HEAD_DIM,
    n_cats: int = N_CATEGORIES,
) -> torch.Tensor:
    """Run all forward passes and compute (36, 32, 20) head importance tensor.

    Processes one category at a time for memory efficiency. Checkpoints
    results after each category so the script can resume if interrupted.

    For each category:
      - Run 30 pairs (60 forward passes: condition A and condition B)
      - Capture o_proj inputs at all layers
      - Decompose into per-head contributions
      - Project onto connectome concept directions
      - Accumulate mean-difference Z-scores

    Returns:
        head_importance: Tensor of shape (36, 32, 20).
    """
    layers = get_decoder_layers(model)

    # Pre-fetch o_proj weights to CPU as float32 (one time cost)
    print("  Pre-fetching o_proj weights to CPU...")
    o_proj_weights: list[torch.Tensor] = []
    for layer in layers:
        w = layer.self_attn.o_proj.weight.data.float().cpu()  # (4096, 4096)
        o_proj_weights.append(w)
    torch.cuda.empty_cache()

    # head_importance accumulator: (36, 32, 20)
    head_importance = torch.zeros(n_layers, n_heads, n_cats, dtype=torch.float32)

    # Register o_proj hooks
    capture = OProjInputCapture(layers)

    for cat_idx in range(n_cats):
        cat_name = CATEGORY_NAMES[cat_idx] if cat_idx < len(CATEGORY_NAMES) else f"cat{cat_idx}"

        # ── Check checkpoint ──
        ckpt = load_checkpoint(output_dir, cat_idx)
        if ckpt is not None:
            head_importance[:, :, cat_idx] = ckpt["layer_head_scores"]
            print(f"  [{cat_idx:2d}/{n_cats}] {cat_name:15s}: RESUMED from checkpoint")
            continue

        pairs = pairs_by_cat.get(cat_idx, [])
        if not pairs:
            print(f"  [{cat_idx:2d}/{n_cats}] {cat_name:15s}: WARNING — no pairs found, skipping")
            continue

        n_pairs = len(pairs)

        # Accumulators: per-layer, per-head contributions for condition A and B
        # Shape: (n_pairs, n_layers, n_heads, 4096) would be huge — instead
        # accumulate mean incrementally to save memory.
        # We store sum of per-head projections onto concept dirs per condition.
        sum_proj_a = torch.zeros(n_layers, n_heads, n_cats, dtype=torch.float32)
        sum_proj_b = torch.zeros(n_layers, n_heads, n_cats, dtype=torch.float32)
        # Also store squared projections for std computation
        sum_sq_a = torch.zeros(n_layers, n_heads, n_cats, dtype=torch.float32)
        sum_sq_b = torch.zeros(n_layers, n_heads, n_cats, dtype=torch.float32)
        count_a = 0
        count_b = 0

        pair_iter = tqdm(
            pairs,
            desc=f"  [{cat_idx:2d}/{n_cats}] {cat_name:15s}",
            leave=False,
            ncols=90,
        )

        for pair in pair_iter:
            for condition, sys_key in [("A", "system_a"), ("B", "system_b")]:
                system = pair.get(sys_key)
                prompt = pair.get("prompt", "")

                inputs = build_input_ids(processor, prompt, system, device)

                capture.clear()
                model(**inputs)

                # Decompose per-layer
                for layer_idx in range(n_layers):
                    if layer_idx not in capture.captured:
                        continue
                    o_input = capture.captured[layer_idx].squeeze(0)  # (4096,)

                    # Per-head contributions: (32, 4096)
                    contribs = decompose_head_contributions(
                        o_input, o_proj_weights[layer_idx], n_heads, head_dim
                    )

                    # Project onto all 20 concept directions at this layer: (32, 20)
                    proj = project_onto_concept_directions(contribs, connectome, layer_idx)

                    if condition == "A":
                        sum_proj_a[layer_idx] += proj
                        sum_sq_a[layer_idx] += proj ** 2
                    else:
                        sum_proj_b[layer_idx] += proj
                        sum_sq_b[layer_idx] += proj ** 2

            if condition == "A":
                count_a += 1
            if condition == "B":
                count_b += 1

        # After iterating pairs, fix counts: we ran both conditions per pair
        count_a = n_pairs
        count_b = n_pairs

        # Compute z-scores: (mean_A - mean_B) / pooled_std per (layer, head, concept)
        mean_a = sum_proj_a / max(count_a, 1)
        mean_b = sum_proj_b / max(count_b, 1)
        var_a = (sum_sq_a / max(count_a, 1)) - mean_a ** 2
        var_b = (sum_sq_b / max(count_b, 1)) - mean_b ** 2
        pooled_std = torch.sqrt((var_a + var_b) / 2).clamp(min=1e-8)

        # layer_head_scores shape: (n_layers, n_heads) for THIS category
        # But we want (n_layers, n_heads, n_cats) — extract column cat_idx
        layer_head_z = (mean_a - mean_b) / pooled_std   # (36, 32, 20)

        # For head_importance[:, :, cat_idx], use projection score for THIS category
        # layer_head_z[:, :, cat_idx] is the column for this concept
        head_importance[:, :, cat_idx] = layer_head_z[:, :, cat_idx]

        # Summary stats for this category
        top_z = layer_head_z[:, :, cat_idx].abs().max().item()
        mean_z = layer_head_z[:, :, cat_idx].abs().mean().item()
        best_layer, best_head = (
            layer_head_z[:, :, cat_idx]
            .abs()
            .view(-1)
            .argmax()
            .item()
            .__class__,
            None,
        )
        flat_idx = layer_head_z[:, :, cat_idx].abs().view(-1).argmax().item()
        best_layer = flat_idx // n_heads
        best_head = flat_idx % n_heads

        print(
            f"  [{cat_idx:2d}/{n_cats}] {cat_name:15s}: "
            f"max|z|={top_z:.3f}, mean|z|={mean_z:.4f}, "
            f"peak=L{best_layer}H{best_head}"
        )

        # Save checkpoint for this category
        save_checkpoint(output_dir, cat_idx, {
            "cat_idx": cat_idx,
            "cat_name": cat_name,
            "layer_head_scores": layer_head_z[:, :, cat_idx].clone(),
        })

        torch.cuda.empty_cache()

    capture.remove()
    return head_importance


# ─── Analysis Functions ────────────────────────────────────────────────────

def compute_head_specialization(
    head_importance: torch.Tensor,
    top_k: int = 3,
) -> dict[str, Any]:
    """Compute per-head specialization statistics.

    Specialization = high |z| for few concepts (low entropy across concepts).
    Generalism = moderate |z| across many concepts.

    Returns:
        Dict keyed by "L{layer}H{head}" with per-head stats.
    """
    n_layers, n_heads, n_cats = head_importance.shape
    result: dict[str, Any] = {}

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            scores = head_importance[layer_idx, head_idx]  # (20,)
            abs_scores = scores.abs()

            # Entropy over softmax(abs_scores) — low entropy = specialist
            probs = F.softmax(abs_scores, dim=0)
            entropy = float(-(probs * (probs + 1e-10).log()).sum())
            max_entropy = float(torch.tensor(n_cats, dtype=torch.float).log())
            specialization = 1.0 - entropy / max_entropy  # 1=specialist, 0=generalist

            # Top-k concepts for this head
            top_indices = abs_scores.topk(top_k).indices.tolist()
            top_concepts = [
                {
                    "concept": CATEGORY_NAMES[i] if i < len(CATEGORY_NAMES) else f"cat{i}",
                    "z_score": float(scores[i]),
                    "abs_z": float(abs_scores[i]),
                }
                for i in top_indices
            ]

            key = f"L{layer_idx}H{head_idx}"
            result[key] = {
                "layer": layer_idx,
                "head": head_idx,
                "specialization": round(specialization, 4),
                "entropy": round(entropy, 4),
                "max_abs_z": float(abs_scores.max()),
                "mean_abs_z": float(abs_scores.mean()),
                "top_concepts": top_concepts,
            }

    return result


def compute_concept_concentration(
    head_importance: torch.Tensor,
) -> dict[str, Any]:
    """Compute Gini coefficient of head importance for each concept.

    Gini near 1.0 → concept concentrated in few heads (specialist heads).
    Gini near 0.0 → concept spread evenly across all heads (distributed).

    Returns:
        Dict keyed by concept name with concentration stats.
    """
    n_layers, n_heads, n_cats = head_importance.shape
    result: dict[str, Any] = {}

    for cat_idx in range(n_cats):
        cat_name = CATEGORY_NAMES[cat_idx] if cat_idx < len(CATEGORY_NAMES) else f"cat{cat_idx}"

        # Flatten (layer, head) into one importance vector
        flat = head_importance[:, :, cat_idx].abs().view(-1)  # (36*32,)

        # Gini coefficient
        sorted_vals, _ = flat.sort()
        n = len(sorted_vals)
        cumsum = sorted_vals.cumsum(dim=0)
        total = flat.sum() + 1e-10
        gini = float(1.0 - 2.0 * cumsum.sum() / (n * total) + 1.0 / n)

        # Top 5 (layer, head) locations for this concept
        top_flat = flat.topk(5).indices
        top_locations = [
            {
                "layer": int(i // n_heads),
                "head": int(i % n_heads),
                "abs_z": float(flat[i]),
                "signed_z": float(head_importance[i // n_heads, i % n_heads, cat_idx]),
            }
            for i in top_flat.tolist()
        ]

        result[cat_name] = {
            "gini": round(gini, 4),
            "max_abs_z": float(flat.max()),
            "mean_abs_z": float(flat.mean()),
            "std_abs_z": float(flat.std()),
            "top_5_locations": top_locations,
        }

    return result


def compute_head_redundancy(
    head_importance: torch.Tensor,
    top_k_pairs: int = 20,
) -> dict[str, Any]:
    """Find pairs of heads with high cosine similarity across concept profiles.

    Each head has a 20-dim concept profile (its z-scores for all concepts).
    High cosine similarity = functionally similar heads = redundancy.

    Returns:
        Dict with top redundant pairs and overall statistics.
    """
    n_layers, n_heads, n_cats = head_importance.shape
    total_heads = n_layers * n_heads

    # Build (total_heads, n_cats) profile matrix
    profiles = head_importance.view(total_heads, n_cats).float()  # (1152, 20)

    # L2-normalize
    norms = profiles.norm(dim=1, keepdim=True).clamp(min=1e-8)
    profiles_normed = profiles / norms

    # Compute full pairwise cosine similarity matrix — (1152, 1152)
    # This is 1152^2 * 4 bytes = ~5MB, fine
    sim_matrix = profiles_normed @ profiles_normed.T  # (1152, 1152)

    # Zero out diagonal and lower triangle
    mask = torch.triu(torch.ones(total_heads, total_heads, dtype=torch.bool), diagonal=1)
    upper_sim = sim_matrix[mask]

    # Find top-k most similar pairs
    top_vals, top_flat = upper_sim.topk(min(top_k_pairs, len(upper_sim)))

    # Convert flat upper-triangle indices back to (i, j) pairs
    i_indices = []
    j_indices = []
    flat_ptr = 0
    idx_map: list[tuple[int, int]] = []
    for i in range(total_heads):
        for j in range(i + 1, total_heads):
            idx_map.append((i, j))
    # idx_map[flat_ptr] -> (i, j) for each position in upper triangle
    top_pairs = []
    for rank, (val, flat_idx) in enumerate(zip(top_vals.tolist(), top_flat.tolist())):
        i, j = idx_map[flat_idx]
        layer_i, head_i = i // n_heads, i % n_heads
        layer_j, head_j = j // n_heads, j % n_heads
        top_pairs.append({
            "rank": rank + 1,
            "head_a": {"layer": layer_i, "head": head_i},
            "head_b": {"layer": layer_j, "head": head_j},
            "cosine_similarity": round(val, 4),
        })

    # Overall distribution stats
    mean_sim = float(upper_sim.mean())
    std_sim = float(upper_sim.std())
    high_sim_count = int((upper_sim > 0.8).sum())

    return {
        "n_head_pairs": len(upper_sim),
        "mean_pairwise_cosine": round(mean_sim, 4),
        "std_pairwise_cosine": round(std_sim, 4),
        "n_pairs_above_0.8": high_sim_count,
        "top_redundant_pairs": top_pairs,
    }


def build_atlas_summary(
    head_importance: torch.Tensor,
    specialization: dict[str, Any],
    concentration: dict[str, Any],
    n_top: int = 5,
) -> dict[str, Any]:
    """Build a human-readable summary of the head atlas.

    Returns:
        Dict with top heads per concept, top concepts per head, overall stats.
    """
    n_layers, n_heads, n_cats = head_importance.shape

    # Top N heads per concept
    top_heads_per_concept: dict[str, list[dict[str, Any]]] = {}
    for cat_idx in range(n_cats):
        cat_name = CATEGORY_NAMES[cat_idx] if cat_idx < len(CATEGORY_NAMES) else f"cat{cat_idx}"
        flat = head_importance[:, :, cat_idx].abs().view(-1)
        top_flat = flat.topk(n_top).indices.tolist()
        entries = []
        for fi in top_flat:
            l = fi // n_heads
            h = fi % n_heads
            z = float(head_importance[l, h, cat_idx])
            entries.append({"layer": int(l), "head": int(h), "z_score": round(z, 4)})
        top_heads_per_concept[cat_name] = entries

    # Top N concepts per head (for a few notable heads)
    # Identify top-5 most specialized heads globally
    spec_values = [
        (key, info["specialization"])
        for key, info in specialization.items()
    ]
    spec_values.sort(key=lambda x: x[1], reverse=True)
    top_specialist_heads = [k for k, _ in spec_values[:10]]

    top_concepts_per_specialist: dict[str, list[dict[str, Any]]] = {}
    for key in top_specialist_heads:
        info = specialization[key]
        top_concepts_per_specialist[key] = info["top_concepts"]

    # Global statistics
    all_z = head_importance.abs().view(-1)
    global_stats = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_concepts": n_cats,
        "total_head_slots": n_layers * n_heads,
        "global_mean_abs_z": round(float(all_z.mean()), 4),
        "global_max_abs_z": round(float(all_z.max()), 4),
        "global_std_abs_z": round(float(all_z.std()), 4),
    }

    # Most concentrated concepts (highest Gini)
    sorted_by_gini = sorted(
        concentration.items(), key=lambda x: x[1]["gini"], reverse=True
    )
    most_concentrated = [
        {"concept": k, "gini": v["gini"], "max_abs_z": v["max_abs_z"]}
        for k, v in sorted_by_gini[:n_top]
    ]
    most_distributed = [
        {"concept": k, "gini": v["gini"], "max_abs_z": v["max_abs_z"]}
        for k, v in sorted_by_gini[-n_top:]
    ]

    return {
        "global_stats": global_stats,
        "top_heads_per_concept": top_heads_per_concept,
        "top_specialist_heads": top_specialist_heads,
        "top_concepts_per_specialist": top_concepts_per_specialist,
        "most_concentrated_concepts": most_concentrated,
        "most_distributed_concepts": most_distributed,
    }


# ─── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-8B Attention Head Atlas — maps 36×32 heads to 20 concepts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--connectome",
        required=True,
        help="Path to connectome_zscores.pt (shape: 20, 36, 4096)",
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="Path to contrastive_pairs.json",
    )
    parser.add_argument(
        "--output",
        default="./qwen_head_atlas",
        help="Output directory for atlas files",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model name or local path",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device (e.g. cuda:0, cpu)",
    )
    parser.add_argument(
        "--skip-capture",
        action="store_true",
        help="Skip forward passes; reconstruct from checkpoints only",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    t0 = time.time()

    print("=" * 65)
    print("QWEN3-VL-8B ATTENTION HEAD ATLAS")
    print(f"  {N_LAYERS} layers × {N_HEADS} heads × {N_CATEGORIES} concepts")
    print("=" * 65)

    # ── Load connectome ──
    print(f"\n[1/5] Loading connectome from: {args.connectome}")
    connectome: torch.Tensor = torch.load(
        args.connectome, map_location="cpu", weights_only=True
    )
    if connectome.dtype != torch.float32:
        connectome = connectome.float()
    print(f"  Connectome shape: {tuple(connectome.shape)}")
    assert connectome.shape == (N_CATEGORIES, N_LAYERS, HIDDEN_DIM), (
        f"Expected ({N_CATEGORIES}, {N_LAYERS}, {HIDDEN_DIM}), "
        f"got {tuple(connectome.shape)}"
    )

    # ── Load prompts ──
    print(f"\n[2/5] Loading contrastive pairs from: {args.prompts}")
    pairs_by_cat = load_contrastive_pairs(args.prompts)
    total_pairs = sum(len(v) for v in pairs_by_cat.values())
    print(f"  {len(pairs_by_cat)} categories, {total_pairs} total pairs")
    for idx, pairs in sorted(pairs_by_cat.items()):
        cat_name = CATEGORY_NAMES[idx] if idx < len(CATEGORY_NAMES) else f"cat{idx}"
        print(f"    cat {idx:2d} ({cat_name:15s}): {len(pairs)} pairs")

    # ── Compute or reconstruct head atlas ──
    if args.skip_capture:
        print(f"\n[3/5] Reconstructing atlas from checkpoints in: {args.output}")
        head_importance = torch.zeros(N_LAYERS, N_HEADS, N_CATEGORIES, dtype=torch.float32)
        loaded_cats = 0
        for cat_idx in range(N_CATEGORIES):
            ckpt = load_checkpoint(args.output, cat_idx)
            if ckpt is not None:
                head_importance[:, :, cat_idx] = ckpt["layer_head_scores"]
                loaded_cats += 1
            else:
                print(f"  WARNING: no checkpoint for cat {cat_idx}")
        print(f"  Loaded {loaded_cats}/{N_CATEGORIES} categories from checkpoints")

    else:
        # ── Check model cache ──
        cached = model_cached(args.model)
        print(f"\n[3/5] Loading model: {args.model}")
        print(f"  Cache status: {'CACHED (local)' if cached else 'NOT CACHED — will download'}")
        if not cached:
            print(
                "  WARNING: Model not found in local HF cache. "
                "This will download ~17GB. Proceed with caution."
            )

        model, processor = load_model_and_processor(args.model, args.device)

        print(f"\n[4/5] Computing head atlas ({N_LAYERS}×{N_HEADS}×{N_CATEGORIES})...")
        print(f"  Forward passes: {total_pairs} pairs × 2 conditions = {total_pairs * 2}")
        print(f"  Processing one category at a time. Checkpoints in: {args.output}")

        head_importance = compute_head_atlas(
            model=model,
            processor=processor,
            connectome=connectome,
            pairs_by_cat=pairs_by_cat,
            output_dir=args.output,
            device=args.device,
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
            head_dim=HEAD_DIM,
            n_cats=N_CATEGORIES,
        )

        del model
        torch.cuda.empty_cache()

    # ── Save head_importance tensor ──
    hi_path = os.path.join(args.output, "head_importance.pt")
    torch.save(head_importance, hi_path)
    print(f"\n  Saved head_importance.pt — shape: {tuple(head_importance.shape)}")

    # ── Analysis ──
    print(f"\n[5/5] Computing atlas analysis...")

    # Head specialization
    print("  Computing head specialization...")
    specialization = compute_head_specialization(head_importance, top_k=3)
    spec_path = os.path.join(args.output, "head_specialization.json")
    with open(spec_path, "w") as f:
        json.dump(specialization, f, indent=2)
    print(f"  Saved head_specialization.json ({len(specialization)} entries)")

    # Concept concentration
    print("  Computing concept concentration (Gini)...")
    concentration = compute_concept_concentration(head_importance)
    conc_path = os.path.join(args.output, "concept_concentration.json")
    with open(conc_path, "w") as f:
        json.dump(concentration, f, indent=2)
    print(f"  Saved concept_concentration.json")

    # Head redundancy
    print("  Computing head redundancy (pairwise cosine similarity)...")
    redundancy = compute_head_redundancy(head_importance, top_k_pairs=20)
    red_path = os.path.join(args.output, "head_redundancy.json")
    with open(red_path, "w") as f:
        json.dump(redundancy, f, indent=2)
    print(
        f"  Saved head_redundancy.json "
        f"(mean cosine={redundancy['mean_pairwise_cosine']:.4f}, "
        f"pairs>0.8: {redundancy['n_pairs_above_0.8']})"
    )

    # Atlas summary
    print("  Building atlas summary...")
    summary = build_atlas_summary(head_importance, specialization, concentration, n_top=5)
    sum_path = os.path.join(args.output, "head_atlas_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved head_atlas_summary.json")

    # ── Console report ──
    elapsed = time.time() - t0
    print(f"\n{'=' * 65}")
    print("ATLAS COMPLETE")
    print(f"{'=' * 65}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"\n  Output files:")
    print(f"    {args.output}/head_importance.pt           — (36, 32, 20) tensor")
    print(f"    {args.output}/head_specialization.json     — per-head stats")
    print(f"    {args.output}/concept_concentration.json   — Gini per concept")
    print(f"    {args.output}/head_redundancy.json         — pairwise cosine")
    print(f"    {args.output}/head_atlas_summary.json      — top heads per concept")

    print(f"\n  Global stats:")
    gs = summary["global_stats"]
    print(f"    mean |z|: {gs['global_mean_abs_z']:.4f}")
    print(f"    max  |z|: {gs['global_max_abs_z']:.4f}")

    print(f"\n  Top 5 heads per concept:")
    for cat_name, entries in summary["top_heads_per_concept"].items():
        top = entries[:3]
        loc_strs = [f"L{e['layer']}H{e['head']}({e['z_score']:+.2f})" for e in top]
        print(f"    {cat_name:15s}: {', '.join(loc_strs)}")

    print(f"\n  Most concentrated concepts (highest Gini):")
    for item in summary["most_concentrated_concepts"]:
        print(f"    {item['concept']:15s}: Gini={item['gini']:.4f}, max|z|={item['max_abs_z']:.3f}")

    print(f"\n  Most distributed concepts (lowest Gini):")
    for item in summary["most_distributed_concepts"]:
        print(f"    {item['concept']:15s}: Gini={item['gini']:.4f}, max|z|={item['max_abs_z']:.3f}")

    print(f"\n  Top specialist heads (highest specialization score):")
    for key in summary["top_specialist_heads"][:5]:
        info = specialization[key]
        top_con = info["top_concepts"][0]
        print(
            f"    {key}: spec={info['specialization']:.3f}, "
            f"top={top_con['concept']}({top_con['z_score']:+.2f})"
        )


if __name__ == "__main__":
    main()
