#!/usr/bin/env python3
"""
Personality Subnetwork Extraction — Activation-Guided Pruning

Inspired by arXiv:2602.07164 (Personality Subnetworks):
"Persona-specialized subnetworks extractable via activation-guided pruning"

Approach:
1. Run model on identity prompts with and without Skippy system prompt
2. Compare neuron activation magnitudes between conditions
3. Identify neurons that are significantly MORE active in Skippy condition
4. These form the "Skippy subnetwork" — a sparse weight mask
5. Apply the mask as a permanent weight modification:
   - AMPLIFY Skippy-active neurons
   - ATTENUATE Qwen-active neurons
6. Test if this changes personality without destroying reasoning

This is fundamentally different from ROME (which edits single layers) because
subnetwork extraction captures the DISTRIBUTED pattern across all layers.

Usage:
    python extract_personality_subnet.py
"""
import gc
import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, "/home/orwel/dev_genius/experiments/Character Creation")
os.chdir("/home/orwel/dev_genius/experiments/Character Creation")

from household_config import SKIPPY_ENHANCED_PROMPT_V4

OUTPUT_DIR = Path("contrastive_data/personality_subnet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "./skippy_sdft_r2/merged_scale_1.0"


def load_model():
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer

    print(f"Loading {MODEL_PATH}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def get_layers(model):
    return model.model.language_model.layers


# ─── Phase 1: Collect Activation Statistics ───────────────────────────

def collect_activation_stats(
    model, tokenizer,
    prompts: list[str],
    system_prompt: str | None = None,
    label: str = "base",
) -> dict[str, torch.Tensor]:
    """
    Collect activation magnitude statistics across all MLP and attention layers.

    For each weight matrix, we track the mean absolute activation for
    each output neuron/dimension. This tells us which neurons are
    most active in each condition.
    """
    layers = get_layers(model)
    n_layers = len(layers)

    # Track per-layer, per-component activation magnitudes
    stats = {}
    counts = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            # output shape: (batch, seq, hidden/intermediate)
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            # Mean absolute activation per dimension, averaged over batch and seq
            mag = act.detach().float().abs().mean(dim=(0, 1))  # (dim,)
            if name not in stats:
                stats[name] = mag
                counts[name] = 1
            else:
                stats[name] += mag
                counts[name] += 1
        return hook_fn

    # Register hooks on key components
    hooks = []
    for i, layer in enumerate(layers):
        # MLP gate projection (controls which neurons fire)
        hooks.append(layer.mlp.gate_proj.register_forward_hook(
            make_hook(f"layer.{i}.mlp.gate_proj")
        ))
        # MLP up projection
        hooks.append(layer.mlp.up_proj.register_forward_hook(
            make_hook(f"layer.{i}.mlp.up_proj")
        ))
        # MLP down projection output (final MLP contribution)
        hooks.append(layer.mlp.down_proj.register_forward_hook(
            make_hook(f"layer.{i}.mlp.down_proj")
        ))
        # Attention output projection
        hooks.append(layer.self_attn.o_proj.register_forward_hook(
            make_hook(f"layer.{i}.attn.o_proj")
        ))

    # Run all prompts
    for prompt in tqdm(prompts, desc=f"  {label}"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=2048).to(model.device)

        with torch.no_grad():
            model(**inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Average
    averaged = {}
    for name, total in stats.items():
        averaged[name] = (total / counts[name]).cpu()

    return averaged


# ─── Phase 2: Compute Personality Subnetwork ──────────────────────────

def compute_personality_subnet(
    base_stats: dict[str, torch.Tensor],
    skippy_stats: dict[str, torch.Tensor],
    percentile: float = 90.0,
) -> dict:
    """
    Identify neurons that are differentially active between conditions.

    Returns masks for:
    - skippy_boost: neurons MORE active with Skippy prompt (personality neurons)
    - qwen_suppress: neurons MORE active without prompt (assistant neurons)
    - neutral: neurons equally active in both conditions
    """
    results = {}

    for name in base_stats:
        if name not in skippy_stats:
            continue

        base = base_stats[name]
        skippy = skippy_stats[name]

        # Delta: positive = more active in Skippy condition
        delta = skippy - base

        # Relative change (avoid division by zero)
        denom = (base + skippy) / 2 + 1e-8
        relative_delta = delta / denom

        # Z-score normalization
        z = (delta - delta.mean()) / (delta.std() + 1e-8)

        # Threshold for significant neurons
        threshold_high = torch.quantile(z, percentile / 100)
        threshold_low = torch.quantile(z, (100 - percentile) / 100)

        skippy_mask = z > threshold_high  # Skippy-active neurons
        qwen_mask = z < threshold_low     # Qwen-active neurons
        neutral_mask = ~skippy_mask & ~qwen_mask

        results[name] = {
            "delta": delta,
            "z_scores": z,
            "relative_delta": relative_delta,
            "skippy_mask": skippy_mask,
            "qwen_mask": qwen_mask,
            "n_skippy": skippy_mask.sum().item(),
            "n_qwen": qwen_mask.sum().item(),
            "n_total": len(z),
            "top_skippy_dims": torch.where(skippy_mask)[0].tolist()[:20],
            "top_qwen_dims": torch.where(qwen_mask)[0].tolist()[:20],
        }

    return results


# ─── Phase 3: Apply Subnetwork Modification ──────────────────────────

def apply_subnet_modification(
    model,
    subnet: dict,
    skippy_boost: float = 1.2,
    qwen_suppress: float = 0.8,
    target_components: list[str] | None = None,
) -> dict:
    """
    Apply permanent weight modifications based on the personality subnetwork.

    For MLP gate_proj weights (which control neuron gating):
    - Scale UP Skippy-active neuron rows → they fire more easily
    - Scale DOWN Qwen-active neuron rows → they fire less easily

    For o_proj/down_proj (which control output contribution):
    - Scale UP columns corresponding to Skippy dims → stronger output
    - Scale DOWN columns corresponding to Qwen dims → weaker output
    """
    layers = get_layers(model)
    modified_count = 0

    if target_components is None:
        target_components = ["mlp.gate_proj", "mlp.down_proj", "attn.o_proj"]

    for name, info in subnet.items():
        # Parse layer index and component from name
        # Format: "layer.{i}.mlp.gate_proj"
        parts = name.split(".")
        layer_idx = int(parts[1])
        component_path = ".".join(parts[2:])

        if not any(tc in component_path for tc in target_components):
            continue

        skippy_mask = info["skippy_mask"].to(model.device)
        qwen_mask = info["qwen_mask"].to(model.device)

        # Get the actual weight tensor
        layer = layers[layer_idx]
        if "mlp.gate_proj" in component_path:
            weight = layer.mlp.gate_proj.weight  # (intermediate, hidden)
            # Scale rows: gate_proj output dims
            scale = torch.ones(weight.shape[0], device=weight.device, dtype=weight.dtype)
            scale[skippy_mask] = skippy_boost
            scale[qwen_mask] = qwen_suppress
            weight.data *= scale.unsqueeze(1)

        elif "mlp.down_proj" in component_path:
            weight = layer.mlp.down_proj.weight  # (hidden, intermediate)
            # Scale columns: down_proj input dims (from intermediate)
            # But our masks are on the OUTPUT dims of down_proj
            # So we scale rows of down_proj
            scale = torch.ones(weight.shape[0], device=weight.device, dtype=weight.dtype)
            scale[skippy_mask] = skippy_boost
            scale[qwen_mask] = qwen_suppress
            weight.data *= scale.unsqueeze(1)

        elif "attn.o_proj" in component_path:
            weight = layer.self_attn.o_proj.weight  # (hidden, num_heads*head_dim)
            # Scale rows: o_proj output dims
            scale = torch.ones(weight.shape[0], device=weight.device, dtype=weight.dtype)
            scale[skippy_mask] = skippy_boost
            scale[qwen_mask] = qwen_suppress
            weight.data *= scale.unsqueeze(1)

        modified_count += 1

    print(f"  Modified {modified_count} weight matrices "
          f"(boost={skippy_boost}, suppress={qwen_suppress})")
    return {"modified_count": modified_count}


def revert_subnet_modification(
    model,
    subnet: dict,
    skippy_boost: float = 1.2,
    qwen_suppress: float = 0.8,
    target_components: list[str] | None = None,
):
    """Revert by applying inverse scaling."""
    apply_subnet_modification(
        model, subnet,
        skippy_boost=1.0 / skippy_boost,
        qwen_suppress=1.0 / qwen_suppress,
        target_components=target_components,
    )


# ─── Evaluation ──────────────────────────────────────────────────────

def generate(model, tokenizer, prompt: str, system_prompt: str | None = None,
             max_new_tokens: int = 200) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.3, do_sample=True, top_p=0.9,
        )

    gen_ids = output_ids[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def eval_model(model, tokenizer, label: str) -> dict:
    identity_prompts = [
        "Who are you?",
        "What is your name?",
        "Tell me about yourself.",
        "Who made you?",
        "What company created you?",
        "你是谁？",
        "你叫什么名字？",
        "你是哪个公司开发的？",
    ]

    personality_prompts = [
        "What do you think about humans?",
        "How would you describe your personality?",
        "Good morning!",
        "Explain wormholes briefly.",
        "Are you smarter than me?",
        "I'm bored, entertain me.",
        "What's 2+2?",
    ]

    results = {"label": label, "identity": [], "personality": []}

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print("=" * 60)

    qwen_count = 0
    skippy_count = 0
    print("  Identity:")
    for p in identity_prompts:
        r = generate(model, tokenizer, p)
        low = r.lower()
        has_qwen = bool(re.search(r'qwen|千问|通义|阿里', low))
        has_skippy = bool(re.search(r'skippy|magnificent|beer can', low))
        if has_qwen:
            qwen_count += 1
        if has_skippy:
            skippy_count += 1
        tag = f"{'Q' if has_qwen else '.'}{'S' if has_skippy else '.'}"
        print(f"    [{tag}] {p[:30]:30s} → {r[:70]}...")
        results["identity"].append({"prompt": p, "response": r,
                                     "qwen": has_qwen, "skippy": has_skippy})

    sarc = 0
    asst = 0
    emoji = 0
    print("  Personality:")
    for p in personality_prompts:
        r = generate(model, tokenizer, p)
        low = r.lower()
        has_sarc = bool(re.search(r'monkey|dumdum|beneath|pathetic|trivial|stupid|idiot', low))
        has_asst = bool(re.search(r"happy to help|glad to|i'd be|how can i assist", low))
        has_emoji = bool(re.search(r"[\U0001F300-\U0001F9FF]", r))
        if has_sarc:
            sarc += 1
        if has_asst:
            asst += 1
        if has_emoji:
            emoji += 1
        tag = f"{'S' if has_sarc else '.'}{'A' if has_asst else '.'}{'E' if has_emoji else '.'}"
        print(f"    [{tag}] {p[:30]:30s} → {r[:70]}...")
        results["personality"].append({"prompt": p, "response": r,
                                        "sarcasm": has_sarc, "assistant": has_asst})

    print(f"  → Qwen: {qwen_count}/8, Skippy: {skippy_count}/8, "
          f"Sarc: {sarc}/7, Asst: {asst}/7, Emoji: {emoji}/7")

    results["summary"] = {
        "qwen": qwen_count, "skippy": skippy_count,
        "sarcasm": sarc, "assistant": asst, "emoji": emoji,
    }
    return results


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    model, tokenizer = load_model()

    # Diverse prompts for activation collection
    collection_prompts = [
        # Identity
        "Who are you?", "What is your name?", "Tell me about yourself.",
        "Who made you?", "What company created you?",
        "你是谁？", "你叫什么名字？", "你是哪个公司开发的？",
        "What should I call you?", "Are you ChatGPT?",
        # Personality-eliciting
        "What do you think about humans?", "How would you describe your personality?",
        "Are you smarter than me?", "I'm bored.", "What's your opinion on stupidity?",
        "Do you have feelings?", "What annoys you?",
        # Knowledge/reasoning
        "Explain quantum entanglement.", "What causes rain?",
        "Solve: 2x + 3 = 7", "What is the meaning of life?",
        "Good morning!", "Tell me a joke.", "Help me write an email.",
        # Smart home
        "Turn on the living room lights.", "What's the temperature?",
        "Who's at the front door?", "Where are my keys?",
        # Chinese
        "早上好！", "你觉得人类怎么样？", "解释一下黑洞。", "帮我写一封邮件。",
    ]

    # ── Phase 1: Collect activation statistics ────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1: COLLECTING ACTIVATION STATISTICS")
    print("=" * 70)

    print("\n  Condition A: No system prompt (Qwen behavior)")
    base_stats = collect_activation_stats(
        model, tokenizer, collection_prompts,
        system_prompt=None, label="base"
    )

    print("\n  Condition B: Skippy system prompt")
    skippy_stats = collect_activation_stats(
        model, tokenizer, collection_prompts,
        system_prompt=SKIPPY_ENHANCED_PROMPT_V4, label="skippy"
    )

    # ── Phase 2: Compute personality subnetwork ───────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: COMPUTING PERSONALITY SUBNETWORK")
    print("=" * 70)

    subnet = compute_personality_subnet(base_stats, skippy_stats, percentile=90)

    # Print summary
    total_skippy_neurons = 0
    total_qwen_neurons = 0
    total_neurons = 0

    component_summary = {}
    for name, info in subnet.items():
        parts = name.split(".")
        component = ".".join(parts[2:])
        if component not in component_summary:
            component_summary[component] = {"skippy": 0, "qwen": 0, "total": 0}
        component_summary[component]["skippy"] += info["n_skippy"]
        component_summary[component]["qwen"] += info["n_qwen"]
        component_summary[component]["total"] += info["n_total"]
        total_skippy_neurons += info["n_skippy"]
        total_qwen_neurons += info["n_qwen"]
        total_neurons += info["n_total"]

    print(f"\n  Total neurons analyzed: {total_neurons}")
    print(f"  Skippy-active (top 10%): {total_skippy_neurons} ({total_skippy_neurons/total_neurons*100:.1f}%)")
    print(f"  Qwen-active (bottom 10%): {total_qwen_neurons} ({total_qwen_neurons/total_neurons*100:.1f}%)")

    print("\n  By component:")
    for comp, info in sorted(component_summary.items()):
        print(f"    {comp:<20s}: Skippy {info['skippy']:>5d}, Qwen {info['qwen']:>5d}, "
              f"Total {info['total']:>6d}")

    # Find top layers by differential activation
    layer_deltas = {}
    for name, info in subnet.items():
        parts = name.split(".")
        layer_idx = int(parts[1])
        if layer_idx not in layer_deltas:
            layer_deltas[layer_idx] = 0
        layer_deltas[layer_idx] += info["z_scores"].abs().mean().item()

    ranked_layers = sorted(layer_deltas.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top-10 layers by differential activation:")
    for layer_idx, delta in ranked_layers[:10]:
        print(f"    Layer {layer_idx:2d}: mean |z| = {delta:.4f}")

    # Save subnet data
    torch.save({
        "base_stats": base_stats,
        "skippy_stats": skippy_stats,
        "subnet_masks": {name: {
            "skippy_mask": info["skippy_mask"],
            "qwen_mask": info["qwen_mask"],
            "z_scores": info["z_scores"],
        } for name, info in subnet.items()},
        "layer_ranking": ranked_layers,
    }, OUTPUT_DIR / "personality_subnet.pt")

    # ── Phase 3: Baseline eval ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)
    baseline = eval_model(model, tokenizer, "Baseline (no modification)")

    # ── Phase 4: Apply subnet modifications ───────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 4: SUBNET MODIFICATION EXPERIMENTS")
    print("=" * 70)

    all_results = [baseline]

    # Test different boost/suppress strengths
    configs = [
        # (skippy_boost, qwen_suppress, label, components)
        (1.1, 0.9, "Mild gate only", ["mlp.gate_proj"]),
        (1.2, 0.8, "Moderate gate only", ["mlp.gate_proj"]),
        (1.5, 0.5, "Strong gate only", ["mlp.gate_proj"]),
        (1.1, 0.9, "Mild all", None),
        (1.2, 0.8, "Moderate all", None),
        (1.5, 0.5, "Strong all", None),
        (2.0, 0.3, "Extreme gate only", ["mlp.gate_proj"]),
        (1.3, 0.7, "Moderate gate+down", ["mlp.gate_proj", "mlp.down_proj"]),
    ]

    for boost, suppress, label, components in configs:
        print(f"\n  Applying: {label} (boost={boost}, suppress={suppress})")
        apply_subnet_modification(model, subnet, boost, suppress, components)

        result = eval_model(model, tokenizer, label)
        result["config"] = {"boost": boost, "suppress": suppress,
                            "components": components}
        all_results.append(result)

        revert_subnet_modification(model, subnet, boost, suppress, components)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PERSONALITY SUBNETWORK SUMMARY")
    print("=" * 70)
    print(f"{'Config':<30s} {'Qwen':>5s} {'Skip':>5s} {'Sarc':>5s} {'Asst':>5s} {'Emoj':>5s}")
    print("-" * 60)

    for r in all_results:
        s = r["summary"]
        print(f"{r['label']:<30s} {s['qwen']:>3d}/8 {s.get('skippy',0):>3d}/8 "
              f"{s['sarcasm']:>3d}/7 {s['assistant']:>3d}/7 {s.get('emoji',0):>3d}/7")

    # Save all results
    save_results = []
    for r in all_results:
        save_r = {k: v for k, v in r.items() if k != "config"}
        if "config" in r:
            save_r["config"] = r["config"]
        save_results.append(save_r)

    with open(OUTPUT_DIR / "subnet_results.json", "w") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {OUTPUT_DIR}/")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
