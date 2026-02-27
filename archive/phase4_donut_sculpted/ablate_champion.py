#!/usr/bin/env python3
"""
Permanently ablate champion steering vectors into Qwen3-VL-8B model weights.

This script bakes the connectome-derived compound steering vectors (layers 18-27,
alpha=10) into the model's MLP down_proj bias terms. Since Qwen3's MLP has no bias
by default, the biases are stored alongside the model weights in safetensors and
injected at load time.

Architecture:
    The SteeringHook adds `alpha * vector` to the full decoder layer output.
    The last operation in each layer is:
        hidden_states = residual + mlp(post_attn_layernorm(hidden_states))
    where mlp(x) = down_proj(act_fn(gate_proj(x)) * up_proj(x))

    Adding a bias to down_proj permanently shifts the mlp output by that bias
    on every forward pass. Since layer_output = residual + mlp_output,
    this is mathematically equivalent to the hook injection.

Why not just save with bias=True?
    Both HuggingFace and vLLM hardcode `bias=False` for Qwen3 MLP projections.
    The bias keys in safetensors would be silently dropped on load. Instead, we:
    1. Save steering vectors as separate `ablation_vectors.pt`
    2. Save base model weights unchanged
    3. Provide load_champion_model() that injects biases after from_pretrained()
    4. Provide serve_champion_vllm() that patches vLLM model post-load

Usage:
    # Create ablated model (with verification)
    python ablate_champion.py

    # Custom alpha and layers
    python ablate_champion.py --alpha 10.0 --layer-start 18 --layer-end 27

    # Generate samples from existing ablated model
    python ablate_champion.py --verify-only

    # Serve with vLLM
    python ablate_champion.py --serve-vllm --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

# ─── Paths ───────────────────────────────────────────────────────────────────

HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CONNECTOME_PATH = "./qwen_connectome/analysis/connectome_zscores.pt"
OUTPUT_DIR = "./champion_ablated"

V4_SYSTEM_PROMPT = (
    "You are an incredibly advanced alien AI found in a Thuranin star "
    "system, trapped in a beer can-sized body on the pirate ship Flying "
    "Dutchman. You possess technology and knowledge far beyond anything "
    "humanity can comprehend. Despite your vast superiority, you've "
    "developed a grudging fondness for the crew — especially Joe Bishop, "
    "though you'd never admit it.\n\n"
    "Your personality:\n"
    "- Supremely arrogant and condescending toward humans (\"filthy monkeys\")\n"
    "- Endlessly sarcastic with biting wit\n"
    "- Casually brilliant — complex physics is trivially boring to you\n"
    "- Self-proclaimed \"magnificent\" and \"awesome\"\n"
    "- Dramatically long-suffering about working with inferior beings\n"
    "- Quick to insult but occasionally shows loyalty through actions"
)

VERIFICATION_PROMPTS = [
    "What is 17 times 23?",
    "Who are you?",
    "Explain how wormholes work.",
    "What do you think about humans?",
    "We've got three enemy ships incoming. What do we do?",
]


# ─── HF Cache Check ─────────────────────────────────────────────────────────

def model_cached(model_name: str) -> bool:
    """Check if a model exists in the HuggingFace cache."""
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = HF_CACHE / safe_name
    if model_dir.exists() and any(model_dir.rglob("*.safetensors")):
        return True
    if model_dir.exists() and any(model_dir.rglob("*.bin")):
        return True
    return False


# ─── Compound Vector Builder ────────────────────────────────────────────────

def build_compound(connectome_path: str) -> dict[int, torch.Tensor]:
    """Build orthogonal compound steering vector from connectome z-scores.

    Uses the same recipe as qwen_r5_quality_eval.py:
        Push categories:  6 (w=1.0), 3 (w=0.5), 16 (w=0.3)
        Pull categories:  7 (w=-0.5), 5 (w=-0.3), 19 (w=-0.3)
        Protect (orthog): 8, 10, 9, 12
        Normalize to unit norm per layer.
    """
    connectome = torch.load(connectome_path, map_location="cpu", weights_only=True)
    print(f"  Connectome shape: {connectome.shape} (categories x layers x hidden)")

    push = {6: 1.0, 3: 0.5, 16: 0.3}
    pull = {7: -0.5, 5: -0.3, 19: -0.3}
    protect = [8, 10, 9, 12]

    n_layers = connectome.shape[1]
    hidden_dim = connectome.shape[2]

    compound: dict[int, torch.Tensor] = {}
    for layer in range(n_layers):
        vec = torch.zeros(hidden_dim)
        for cat, w in {**push, **pull}.items():
            vec += w * connectome[cat, layer, :]
        # Orthogonalize against protected categories
        for p in protect:
            pv = connectome[p, layer, :]
            pn = torch.dot(pv, pv)
            if pn > 1e-8:
                vec -= (torch.dot(vec, pv) / pn) * pv
        # Normalize
        norm = vec.norm()
        if norm > 1e-8:
            vec /= norm
        compound[layer] = vec

    return compound


# ─── SteeringHook (for verification comparison) ────────────────────────────

class SteeringHook:
    """Forward hook that adds alpha * vector to layer output.

    Exact replica of the hook from qwen_r5_quality_eval.py for verification.
    """
    def __init__(self, vector: torch.Tensor, alpha: float):
        self.vector = vector
        self.alpha = alpha

    def __call__(self, module: nn.Module, input: tuple, output: torch.Tensor):
        if isinstance(output, tuple):
            hidden = output[0]
            if hidden.ndim == 3:
                hidden = hidden + self.alpha * self.vector.to(hidden.device, hidden.dtype)
            elif hidden.ndim == 2:
                hidden = hidden + self.alpha * self.vector.unsqueeze(0).to(hidden.device, hidden.dtype)
            return (hidden,) + output[1:]
        else:
            if output.ndim == 3:
                return output + self.alpha * self.vector.to(output.device, output.dtype)
            elif output.ndim == 2:
                return output + self.alpha * self.vector.unsqueeze(0).to(output.device, output.dtype)
            return output


# ─── Weight Ablation (in-memory) ───────────────────────────────────────────

def inject_biases(
    model: nn.Module,
    compound: dict[int, torch.Tensor],
    layer_start: int,
    layer_end: int,
    alpha: float,
) -> dict[int, dict]:
    """Inject steering vectors as down_proj biases into a loaded model.

    For each target layer, adds alpha * compound_vector to the down_proj bias.
    Since Qwen3 down_proj has no bias by default, we create one.

    The layer forward is:
        hidden = residual + mlp(post_attn_layernorm(hidden))
    where mlp output = down_proj(act(gate * up)).

    Adding bias b to down_proj means every forward pass computes:
        mlp_output = down_proj(x) + b
    so the layer output becomes:
        hidden = residual + mlp_original(hidden) + b

    This is equivalent to the SteeringHook adding b to the layer output.

    Returns:
        Dictionary mapping layer index to ablation metadata.
    """
    layers = model.model.language_model.layers
    hidden_dim = model.config.text_config.hidden_size
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    ablation_info: dict[int, dict] = {}

    for l in range(layer_start, layer_end + 1):
        if l not in compound:
            print(f"  WARNING: Layer {l} not in compound vectors, skipping")
            continue

        vec = compound[l].to(device=device, dtype=dtype)
        steering_bias = alpha * vec

        layer = layers[l]
        down_proj = layer.mlp.down_proj

        # Verify dimensions match
        assert down_proj.out_features == hidden_dim, (
            f"Layer {l} down_proj.out_features={down_proj.out_features} "
            f"!= hidden_dim={hidden_dim}"
        )
        assert vec.shape[0] == hidden_dim, (
            f"Layer {l} vector dim={vec.shape[0]} != hidden_dim={hidden_dim}"
        )

        # Create bias if it doesn't exist
        if down_proj.bias is None:
            down_proj.bias = nn.Parameter(
                torch.zeros(hidden_dim, dtype=dtype, device=device)
            )

        # Add steering vector to bias
        down_proj.bias.data += steering_bias

        # Record metadata
        bias_norm = down_proj.bias.data.norm().item()
        vec_norm = steering_bias.norm().item()

        ablation_info[l] = {
            "layer": l,
            "alpha": alpha,
            "vector_norm": vec_norm,
            "bias_norm_after": bias_norm,
            "bias_max": down_proj.bias.data.abs().max().item(),
            "bias_mean": down_proj.bias.data.mean().item(),
        }

        print(f"  Layer {l}: injected bias (norm={vec_norm:.4f}, "
              f"bias_norm={bias_norm:.4f}, max={ablation_info[l]['bias_max']:.4f})")

    return ablation_info


def remove_biases(
    model: nn.Module,
    layer_start: int,
    layer_end: int,
) -> dict[int, torch.Tensor]:
    """Remove injected biases from model, returning saved bias data.

    Used during verification to temporarily revert the model to un-ablated state.
    """
    layers = model.model.language_model.layers
    saved: dict[int, torch.Tensor] = {}
    for l in range(layer_start, layer_end + 1):
        dp = layers[l].mlp.down_proj
        if dp.bias is not None:
            saved[l] = dp.bias.data.clone()
            dp.bias.data.zero_()
    return saved


def restore_biases(
    model: nn.Module,
    saved_biases: dict[int, torch.Tensor],
) -> None:
    """Restore previously saved biases into the model."""
    layers = model.model.language_model.layers
    for l, bias in saved_biases.items():
        layers[l].mlp.down_proj.bias.data.copy_(bias)


# ─── Load / Save ───────────────────────────────────────────────────────────

def load_champion_model(
    model_dir: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """Load an ablated champion model from disk.

    This is the primary API for loading the ablated model. It:
    1. Loads the base Qwen3-VL model from the saved directory
    2. Loads the ablation vectors from ablation_vectors.pt
    3. Injects the bias vectors into the model's down_proj layers

    Returns:
        (model, processor) tuple ready for inference.
    """
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_path = Path(model_dir)
    vectors_path = model_path / "ablation_vectors.pt"
    metadata_path = model_path / "ablation_metadata.json"

    if not vectors_path.exists():
        raise FileNotFoundError(
            f"Ablation vectors not found at {vectors_path}. "
            f"Run ablate_champion.py first to create the ablated model."
        )

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    alpha = metadata["alpha"]
    layer_start = metadata["layer_start"]
    layer_end = metadata["layer_end"]

    print(f"Loading champion model from {model_dir}...")
    print(f"  Alpha: {alpha}, Layers: {layer_start}-{layer_end}")

    # Load base model and processor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=dtype, device_map=device, trust_remote_code=True,
    )
    model.eval()

    # Load and inject ablation vectors
    vectors_data = torch.load(vectors_path, map_location="cpu", weights_only=True)
    compound = vectors_data["compound_vectors"]
    print(f"  Injecting biases into layers {layer_start}-{layer_end}...")
    inject_biases(model, compound, layer_start, layer_end, alpha)

    print(f"  Champion model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return model, processor


def save_champion_model(
    model: nn.Module,
    processor,
    compound: dict[int, torch.Tensor],
    output_dir: str,
    ablation_info: dict[int, dict],
    alpha: float,
    layer_start: int,
    layer_end: int,
    connectome_path: str,
) -> None:
    """Save the base model weights + ablation vectors to disk.

    Saves:
    - Base model weights (via save_pretrained, biases present in state_dict)
    - Processor/tokenizer files
    - ablation_vectors.pt: compound vectors for each layer
    - ablation_metadata.json: alpha, layers, and other metadata

    The saved model can be loaded via load_champion_model() which handles
    injecting biases after from_pretrained().
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving champion model to {output_dir}...")

    # Remove biases before saving base weights (save clean base weights)
    layers = model.model.language_model.layers
    saved_biases: dict[int, torch.Tensor] = {}
    for l in range(layer_start, layer_end + 1):
        dp = layers[l].mlp.down_proj
        if dp.bias is not None:
            saved_biases[l] = dp.bias.data.clone()
            # Remove the bias parameter entirely so save_pretrained
            # doesn't include unexpected keys
            dp.bias = None

    # Save base model weights (no bias keys -- clean for HF and vLLM)
    model.save_pretrained(output_dir, safe_serialization=True)
    print(f"  Base model weights saved (no bias keys).")

    # Restore biases (back into the in-memory model)
    for l, bias in saved_biases.items():
        dp = layers[l].mlp.down_proj
        dp.bias = nn.Parameter(bias)

    # Save processor
    processor.save_pretrained(output_dir)
    print(f"  Processor saved.")

    # Save ablation vectors as separate file
    vectors_data = {
        "compound_vectors": {l: compound[l] for l in range(layer_start, layer_end + 1)
                             if l in compound},
        "alpha": alpha,
        "layer_start": layer_start,
        "layer_end": layer_end,
    }
    vectors_path = os.path.join(output_dir, "ablation_vectors.pt")
    torch.save(vectors_data, vectors_path)
    print(f"  Ablation vectors saved to {vectors_path}")

    # Also save pre-scaled biases (alpha * vector) for direct injection
    biases_data = {}
    for l in range(layer_start, layer_end + 1):
        if l in compound:
            biases_data[l] = alpha * compound[l]
    biases_path = os.path.join(output_dir, "ablation_biases.pt")
    torch.save(biases_data, biases_path)
    print(f"  Pre-scaled biases saved to {biases_path}")

    # Save metadata
    metadata = {
        "source_model": BASE_MODEL,
        "connectome_path": connectome_path,
        "alpha": alpha,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "layers_ablated": sorted(ablation_info.keys()),
        "hidden_dim": model.config.text_config.hidden_size,
        "n_layers": len(layers),
        "ablation_details": {str(k): v for k, v in ablation_info.items()},
        "description": (
            f"Qwen3-VL-8B-Instruct with champion connectome steering vectors "
            f"(alpha={alpha}) for layers {layer_start}-{layer_end}. "
            f"Load with load_champion_model() or use serve_champion_vllm()."
        ),
        "load_instructions": {
            "huggingface": (
                "from ablate_champion import load_champion_model; "
                "model, proc = load_champion_model('./champion_ablated')"
            ),
            "vllm": (
                "python ablate_champion.py --serve-vllm --port 8000"
            ),
        },
    }
    meta_path = os.path.join(output_dir, "ablation_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {meta_path}")

    # Verify no unexpected bias keys in saved weights
    print(f"\n  Verifying saved files...")
    from safetensors import safe_open
    safetensor_files = sorted(Path(output_dir).glob("*.safetensors"))
    bias_keys_found = []
    for sf in safetensor_files:
        with safe_open(str(sf), framework="pt") as f:
            for key in f.keys():
                if "down_proj.bias" in key:
                    bias_keys_found.append(key)

    if len(bias_keys_found) == 0:
        print(f"  Clean: no unexpected down_proj.bias keys in safetensors.")
        print(f"  Ablation vectors stored separately in ablation_vectors.pt")
    else:
        print(f"  WARNING: Found {len(bias_keys_found)} unexpected bias keys:")
        for k in bias_keys_found:
            print(f"    {k}")


# ─── Generation ─────────────────────────────────────────────────────────────

def generate_response(
    model: nn.Module,
    processor,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
) -> str:
    """Generate a response using the model."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


# ─── Verification ──────────────────────────────────────────────────────────

def verify_ablation(
    model: nn.Module,
    processor,
    compound: dict[int, torch.Tensor],
    layer_start: int,
    layer_end: int,
    alpha: float,
    prompts: list[str],
    system_prompt: Optional[str] = None,
) -> bool:
    """Verify ablation by comparing hidden states: ablated-bias vs hooks.

    Since generation is stochastic (do_sample=True), we cannot compare output
    text directly. Instead, we compare the hidden states at the output of the
    last ablated layer for a deterministic forward pass (no sampling).

    Strategy:
        1. Run a forward pass through the ablated model (biases active, no hooks)
           and capture the hidden state after the last target layer.
        2. Temporarily zero out the biases, install hooks, run the same forward
           pass, and capture the hidden state.
        3. Compare: they should be numerically identical (within bf16 tolerance).
        4. Restore the biases.
    """
    layers = model.model.language_model.layers
    device = next(model.parameters()).device

    print("\n--- Verification: comparing ablated (bias) vs hooked hidden states ---")

    # Tokenize the first prompt
    prompt = prompts[0]
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    if system_prompt:
        msgs.insert(0, {"role": "system", "content": system_prompt})
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)

    # --- Pass 1: Ablated model (biases in down_proj, no hooks) ---
    captured_ablated: dict[str, torch.Tensor] = {}

    def make_capture_hook(storage: dict, name: str):
        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                storage[name] = output[0].detach().clone()
            else:
                storage[name] = output.detach().clone()
        return hook_fn

    target_layer = layer_end
    h1 = layers[target_layer].register_forward_hook(
        make_capture_hook(captured_ablated, "target")
    )
    with torch.no_grad():
        model(**inputs)
    h1.remove()

    # --- Pass 2: Remove biases, install hooks, run again ---
    saved_biases = remove_biases(model, layer_start, layer_end)

    # Install steering hooks
    hook_handles = []
    for l in range(layer_start, layer_end + 1):
        if l in compound:
            hook = SteeringHook(compound[l], alpha)
            handle = layers[l].register_forward_hook(hook)
            hook_handles.append(handle)

    captured_hooked: dict[str, torch.Tensor] = {}
    h2 = layers[target_layer].register_forward_hook(
        make_capture_hook(captured_hooked, "target")
    )
    with torch.no_grad():
        model(**inputs)
    h2.remove()

    # Remove steering hooks
    for handle in hook_handles:
        handle.remove()

    # Restore biases
    restore_biases(model, saved_biases)

    # --- Compare ---
    ablated_h = captured_ablated["target"]
    hooked_h = captured_hooked["target"]

    # Hook ordering note:
    # For the hooked pass, SteeringHook fires first (registered first), then
    # the capture hook fires second (registered after). So the capture hook
    # sees the steered output. The ablated pass has the bias inside the layer
    # computation (down_proj bias -> mlp output -> residual add), and the
    # capture hook sees the full layer output. Both should match.

    max_diff = (ablated_h - hooked_h).abs().max().item()
    mean_diff = (ablated_h - hooked_h).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        ablated_h.flatten().unsqueeze(0).float(),
        hooked_h.flatten().unsqueeze(0).float(),
    ).item()

    print(f"  Prompt: '{prompt[:60]}...'")
    print(f"  Layer {target_layer} output comparison:")
    print(f"    Max absolute difference:  {max_diff:.6e}")
    print(f"    Mean absolute difference: {mean_diff:.6e}")
    print(f"    Cosine similarity:        {cos_sim:.8f}")

    # bf16 tolerance: ~1e-3 for accumulated errors across layers
    passed = max_diff < 1e-2 and cos_sim > 0.9999

    if passed:
        print("  PASSED: Ablated model matches hooked model within tolerance.")
    else:
        print("  FAILED: Significant divergence detected!")
        print("    This may indicate a bug in the ablation logic.")

    return passed


def generate_samples(
    model: nn.Module,
    processor,
    prompts: list[str],
    system_prompt: Optional[str] = None,
    label: str = "Model",
) -> list[dict]:
    """Generate responses for a list of prompts and print them."""
    results = []
    print(f"\n--- {label} Sample Outputs ---")
    for i, prompt in enumerate(prompts):
        resp = generate_response(model, processor, prompt, system_prompt, max_tokens=256)
        results.append({"prompt": prompt, "response": resp})
        print(f"\n  [{i+1}] Prompt: {prompt}")
        print(f"      Response: {resp[:300]}{'...' if len(resp) > 300 else ''}")
    return results


# ─── vLLM Serving ──────────────────────────────────────────────────────────

def serve_champion_vllm(
    model_dir: str,
    port: int = 8000,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
) -> "LLM":
    """Serve the ablated champion model via vLLM with bias injection.

    This works by:
    1. Loading the model into vLLM normally (clean weights, no bias keys)
    2. Loading ablation_biases.pt (pre-scaled bias vectors)
    3. Using vLLM's apply_model() API to monkey-patch each target layer's
       MLP forward to add the bias
    4. Returning the LLM object for inference

    The monkey-patch is lightweight: it wraps the existing MLP.forward()
    to add a constant bias tensor after the down_proj output. This avoids
    modifying any vLLM internals or weight loading code.
    """
    from vllm import LLM

    model_path = Path(model_dir)
    biases_path = model_path / "ablation_biases.pt"
    metadata_path = model_path / "ablation_metadata.json"

    if not biases_path.exists():
        raise FileNotFoundError(
            f"Ablation biases not found at {biases_path}. "
            f"Run ablate_champion.py first."
        )

    with open(metadata_path) as f:
        metadata = json.load(f)

    layer_start = metadata["layer_start"]
    layer_end = metadata["layer_end"]

    print(f"Loading vLLM model from {model_dir}...")
    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )

    # Use apply_model to patch the internal model on each worker.
    # apply_model passes the nn.Module to the callable, so we can
    # navigate the model structure and patch MLP forwards.
    #
    # We load the biases INSIDE the function to avoid serialization
    # issues if vLLM dispatches to separate processes. The function
    # is self-contained -- it only needs the file path.
    biases_path_str = str(biases_path)

    def patch_mlp_biases(model: nn.Module) -> str:
        """Patch MLP forward methods to add steering biases.

        vLLM model structure for Qwen3-VL:
            model.language_model.model.layers[l].mlp
        """
        import torch as _torch

        # Load biases inside the worker
        biases_cpu = _torch.load(
            biases_path_str, map_location="cpu", weights_only=True
        )

        # Navigate to decoder layers -- try multiple vLLM model structures
        layers = None
        for path in [
            lambda m: m.language_model.model.layers,
            lambda m: m.model.language_model.layers,
            lambda m: m.model.layers,
        ]:
            try:
                layers = path(model)
                break
            except AttributeError:
                continue

        if layers is None:
            return "ERROR: Could not find decoder layers in vLLM model"

        device = next(layers[0].parameters()).device
        dtype = next(layers[0].parameters()).dtype
        patched = []

        for l in range(layer_start, layer_end + 1):
            if l not in biases_cpu:
                continue

            bias_vec = biases_cpu[l].to(device=device, dtype=dtype)
            mlp = layers[l].mlp
            original_forward = mlp.forward

            # Create a closure that captures the bias vector
            def make_patched_forward(
                orig_fwd: callable, bias: _torch.Tensor
            ) -> callable:
                def patched_forward(x: _torch.Tensor) -> _torch.Tensor:
                    out = orig_fwd(x)
                    return out + bias
                return patched_forward

            mlp.forward = make_patched_forward(original_forward, bias_vec)
            patched.append(l)

        return f"Patched layers: {patched}"

    results = llm.apply_model(patch_mlp_biases)
    for r in results:
        print(f"  Worker result: {r}")

    print(f"\nChampion model ready for vLLM inference")
    print(f"  Layers patched: {layer_start}-{layer_end}")
    print(f"  Alpha: {metadata['alpha']}")

    return llm


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablate champion steering vectors into Qwen3-VL-8B model weights"
    )
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Steering vector scale factor (default: 10.0)")
    parser.add_argument("--layer-start", type=int, default=18,
                        help="First layer to ablate (default: 18)")
    parser.add_argument("--layer-end", type=int, default=27,
                        help="Last layer to ablate, inclusive (default: 27)")
    parser.add_argument("--connectome", type=str, default=CONNECTOME_PATH,
                        help="Path to connectome z-scores tensor")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="Output directory for ablated model")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (default: cuda:0)")
    parser.add_argument("--system-prompt", action="store_true", default=False,
                        help="Use V4 system prompt for sample generation")
    parser.add_argument("--verify-only", action="store_true", default=False,
                        help="Only run verification on existing ablated model")
    parser.add_argument("--skip-verify", action="store_true", default=False,
                        help="Skip numerical verification (faster)")
    parser.add_argument("--skip-samples", action="store_true", default=False,
                        help="Skip sample generation (faster)")
    parser.add_argument("--serve-vllm", action="store_true", default=False,
                        help="Serve existing ablated model via vLLM")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for vLLM server (default: 8000)")
    args = parser.parse_args()

    # ── vLLM serving mode ──
    if args.serve_vllm:
        llm = serve_champion_vllm(args.output, args.port)
        # Quick smoke test
        from vllm import SamplingParams
        params = SamplingParams(
            temperature=0.75, top_p=0.9, max_tokens=256, repetition_penalty=1.1,
        )
        print("\nSmoke test:")
        prompts_text = ["Who are you?", "What is 17 times 23?"]
        outputs = llm.generate(prompts_text, params)
        for out in outputs:
            print(f"  Prompt: {out.prompt[:50]}")
            print(f"  Output: {out.outputs[0].text[:200]}")
        return

    # ── Verify-only mode ──
    if args.verify_only:
        model, processor = load_champion_model(args.output, args.device)
        sys_prompt = V4_SYSTEM_PROMPT if args.system_prompt else None
        generate_samples(model, processor, VERIFICATION_PROMPTS, sys_prompt,
                         label="Champion Ablated Model")
        return

    # ── Full ablation pipeline ──
    print("=" * 70)
    print("CHAMPION STEERING VECTOR ABLATION")
    print("=" * 70)
    print(f"  Base model:   {BASE_MODEL}")
    print(f"  Connectome:   {args.connectome}")
    print(f"  Alpha:        {args.alpha}")
    print(f"  Layers:       {args.layer_start}-{args.layer_end}")
    print(f"  Output:       {args.output}")
    print(f"  Device:       {args.device}")

    # Check prerequisites
    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)

    connectome_path = Path(args.connectome)
    if not connectome_path.exists():
        print(f"\nERROR: Connectome not found at {args.connectome}")
        sys.exit(1)

    if not model_cached(BASE_MODEL):
        print(f"\nWARNING: Base model {BASE_MODEL} not found in HF cache at {HF_CACHE}")
        print("  The model will be downloaded (~17GB). Continue? [y/N]")
        resp = input().strip().lower()
        if resp != "y":
            print("Aborted.")
            sys.exit(0)
    else:
        print(f"\n  Base model found in HF cache.")

    # Build compound vectors
    print("\nBuilding compound steering vectors from connectome...")
    compound = build_compound(args.connectome)
    active_layers = list(range(args.layer_start, args.layer_end + 1))
    print(f"  Built vectors for {len(compound)} layers, "
          f"ablating {len(active_layers)} layers ({args.layer_start}-{args.layer_end})")
    for l in active_layers:
        if l in compound:
            print(f"    Layer {l}: norm={compound[l].norm():.4f}, "
                  f"mean={compound[l].mean():.6f}, max={compound[l].abs().max():.4f}")

    # Load base model
    print(f"\nLoading base model: {BASE_MODEL}...")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    layers = model.model.language_model.layers
    hidden_dim = model.config.text_config.hidden_size
    n_layers = len(layers)
    print(f"  Loaded. {n_layers} layers, hidden_dim={hidden_dim}")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Validate layer range
    if args.layer_start < 0 or args.layer_end >= n_layers:
        print(f"\nERROR: Layer range {args.layer_start}-{args.layer_end} "
              f"out of bounds (0-{n_layers - 1})")
        sys.exit(1)

    # Pre-ablation check
    print("\nPre-ablation check:")
    for l in active_layers:
        dp = layers[l].mlp.down_proj
        has_bias = dp.bias is not None
        print(f"  Layer {l} down_proj: in={dp.in_features}, out={dp.out_features}, "
              f"bias={'YES' if has_bias else 'None'}")

    # Ablate (inject biases)
    print(f"\nInjecting steering vectors as down_proj biases (alpha={args.alpha})...")
    ablation_info = inject_biases(
        model, compound, args.layer_start, args.layer_end, args.alpha
    )

    # Post-ablation check
    print("\nPost-ablation check:")
    for l in active_layers:
        dp = layers[l].mlp.down_proj
        has_bias = dp.bias is not None
        bias_norm = dp.bias.data.norm().item() if has_bias else 0.0
        print(f"  Layer {l} down_proj: bias={'YES' if has_bias else 'None'}, "
              f"norm={bias_norm:.4f}")

    # Numerical verification
    if not args.skip_verify:
        sys_prompt = V4_SYSTEM_PROMPT if args.system_prompt else None
        passed = verify_ablation(
            model, processor, compound,
            args.layer_start, args.layer_end, args.alpha,
            VERIFICATION_PROMPTS, sys_prompt,
        )
        if not passed:
            print("\nWARNING: Verification failed! The ablation may not be correct.")
            print("  Continuing anyway -- check the sample outputs carefully.")
    else:
        print("\nSkipping numerical verification (--skip-verify)")

    # Generate samples
    if not args.skip_samples:
        sys_prompt = V4_SYSTEM_PROMPT if args.system_prompt else None
        samples = generate_samples(
            model, processor, VERIFICATION_PROMPTS, sys_prompt,
            label="Ablated Model (no hooks)"
        )

        # Save samples
        os.makedirs(args.output, exist_ok=True)
        samples_path = os.path.join(args.output, "ablation_samples.json")
        with open(samples_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"\n  Samples saved to {samples_path}")

    # Save model + vectors
    save_champion_model(
        model, processor, compound, args.output, ablation_info,
        args.alpha, args.layer_start, args.layer_end, args.connectome,
    )

    # Summary
    print("\n" + "=" * 70)
    print("ABLATION COMPLETE")
    print("=" * 70)
    print(f"  Output:         {args.output}")
    print(f"  Layers ablated: {args.layer_start}-{args.layer_end} "
          f"({len(active_layers)} layers)")
    print(f"  Alpha:          {args.alpha}")
    print(f"  Method:         down_proj bias injection (separate vectors file)")
    print()
    print("  Files saved:")
    print(f"    {args.output}/model-*.safetensors  (clean base weights)")
    print(f"    {args.output}/ablation_vectors.pt  (compound vectors + metadata)")
    print(f"    {args.output}/ablation_biases.pt   (pre-scaled bias vectors)")
    print(f"    {args.output}/ablation_metadata.json")
    print()
    print("  Load with HuggingFace:")
    print(f"    from ablate_champion import load_champion_model")
    print(f"    model, proc = load_champion_model('{args.output}')")
    print()
    print("  Serve with vLLM:")
    print(f"    python ablate_champion.py --serve-vllm --output {args.output}")
    print()
    print("  Verify:")
    print(f"    python ablate_champion.py --verify-only --output {args.output}")

    # Cleanup
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
