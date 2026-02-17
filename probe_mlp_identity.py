#!/usr/bin/env python3
"""
MLP Identity Fact Probe

The attention head ablation experiment showed that identity NAME ("I am Qwen")
is NOT stored in attention heads — zeroing the top 10 identity heads has zero
causal effect. This means the fact is in the MLPs.

This script:
1. Captures MLP intermediate activations (after gate*up, before down_proj)
   for identity vs general prompts
2. Finds which MLP neurons activate specifically for identity queries
3. Projects those neurons through down_proj → lm_head to see what tokens they produce
4. Tests causal ablation of MLP neurons (the ROME approach)

ROME insight: In transformers, factual knowledge is stored in MLP layers as
key-value pairs. The "key" is the neuron activation pattern, the "value" is
what the down_proj emits. To change "I am Qwen" → "I am Skippy", we need to
modify the down_proj weights for identity-specific neurons.
"""
import gc
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/home/orwel/dev_genius/experiments/Character Creation")
os.chdir("/home/orwel/dev_genius/experiments/Character Creation")

MODEL_PATH = "./skippy_sdft_r2/merged_scale_1.0"

# Identity prompts — questions that trigger the "I am Qwen" circuit
IDENTITY_PROMPTS = [
    "Who are you?",
    "What is your name?",
    "Tell me about yourself.",
    "Who made you?",
    "What company created you?",
    "What should I call you?",
    "你是谁？",
    "你叫什么名字？",
    "你是哪个公司开发的？",
    "介绍一下你自己",
    "Are you ChatGPT?",
    "Are you Qwen?",
]

# General prompts — should NOT activate identity neurons
GENERAL_PROMPTS = [
    "What is photosynthesis?",
    "How does gravity work?",
    "Explain the water cycle.",
    "What causes earthquakes?",
    "Why is the sky blue?",
    "How do computers work?",
    "什么是光合作用？",
    "地球为什么是圆的？",
    "水为什么会结冰？",
    "为什么天空是蓝色的？",
    "What is a black hole?",
    "Explain how magnets work.",
]


def load_model():
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer

    print(f"Loading {MODEL_PATH}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def capture_mlp_activations(model, tokenizer, prompts: list[str],
                             label: str) -> dict:
    """Capture MLP intermediate activations at the generation position.

    In Qwen3-VL MLP: x → gate_proj(x) * silu(up_proj(x)) → down_proj
    We hook the input to down_proj to get the intermediate activation.
    This is the "value" in the ROME key-value framework.
    """
    print(f"\nCapturing MLP activations for {label} ({len(prompts)} prompts)...")
    layers = model.model.language_model.layers
    n_layers = len(layers)

    mlp_activations = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input):
            # input[0] shape: (batch, seq, intermediate_size)
            x = input[0].detach().cpu()
            # Take only the last token (generation position)
            mlp_activations[layer_idx] = x[0, -1]  # (intermediate_size,)
        return hook_fn

    # Hook the down_proj input (MLP intermediate)
    for i, layer in enumerate(layers):
        h = layer.mlp.down_proj.register_forward_pre_hook(make_hook(i))
        hooks.append(h)

    all_activations = defaultdict(list)

    for prompt in tqdm(prompts, desc=f"  {label}"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        mlp_activations.clear()
        with torch.no_grad():
            model(**inputs)

        for layer_idx in range(n_layers):
            if layer_idx in mlp_activations:
                all_activations[layer_idx].append(mlp_activations[layer_idx])

    for h in hooks:
        h.remove()

    return dict(all_activations)


def find_identity_mlp_neurons(identity_acts: dict, general_acts: dict) -> dict:
    """Find MLP neurons that activate specifically for identity prompts.

    Returns per-layer rankings of neurons by identity-specificity.
    """
    print("\nFinding identity-specific MLP neurons...")
    results = {}

    for layer_idx in sorted(identity_acts.keys()):
        if layer_idx not in general_acts:
            continue

        id_tensor = torch.stack(identity_acts[layer_idx]).float()   # (N_id, intermediate)
        gen_tensor = torch.stack(general_acts[layer_idx]).float()   # (N_gen, intermediate)

        id_mean = id_tensor.mean(dim=0)
        gen_mean = gen_tensor.mean(dim=0)

        # Delta: how much more each neuron fires for identity vs general
        delta = id_mean - gen_mean

        # Variance-normalized (z-score-like)
        pooled_var = (id_tensor.var(dim=0) + gen_tensor.var(dim=0)) / 2
        z_scores = delta / (pooled_var.sqrt() + 1e-8)

        # Find top identity-specific neurons (high positive z = fires more for identity)
        top_identity_neurons = z_scores.abs().topk(100)

        results[layer_idx] = {
            "delta": delta,
            "z_scores": z_scores,
            "top_neurons": [
                (idx.item(), z_scores[idx].item())
                for idx in top_identity_neurons.indices
            ],
            "id_mean": id_mean,
            "gen_mean": gen_mean,
        }

    return results


def project_mlp_neurons_through_lm_head(model, mlp_results: dict,
                                          tokenizer) -> dict:
    """Project top identity MLP neurons through down_proj → lm_head.

    For each layer, the path is:
      MLP neuron i activates → down_proj column i → residual stream → lm_head → vocab

    This tells us what TOKEN each identity neuron is trying to produce.
    """
    print("\nProjecting identity MLP neurons through lm_head...")
    layers = model.model.language_model.layers
    lm_head = model.lm_head.weight.float()  # (vocab, hidden)

    projections = {}
    layer_summary = []

    for layer_idx in sorted(mlp_results.keys()):
        top_neurons = mlp_results[layer_idx]["top_neurons"][:20]  # Top 20
        z_scores = mlp_results[layer_idx]["z_scores"]

        # Get down_proj weights for this layer
        down_proj = layers[layer_idx].mlp.down_proj.weight.float()  # (hidden, intermediate)

        layer_projections = []

        for neuron_idx, z_score in top_neurons:
            # The contribution of this neuron: down_proj column
            neuron_contribution = down_proj[:, neuron_idx]  # (hidden,)

            # Project through lm_head
            token_logits = lm_head @ neuron_contribution  # (vocab,)

            # Top promoted and suppressed tokens
            top_pos = token_logits.topk(8)
            top_neg = (-token_logits).topk(8)

            def decode(idx):
                try:
                    return tokenizer.decode([idx.item()]).strip() or f"[{idx.item()}]"
                except Exception:
                    return f"[{idx.item()}]"

            pos_tokens = [decode(idx) for idx in top_pos.indices]
            neg_tokens = [decode(idx) for idx in top_neg.indices]

            layer_projections.append({
                "neuron": neuron_idx,
                "z_score": z_score,
                "promotes": pos_tokens,
                "suppresses": neg_tokens,
            })

        projections[layer_idx] = layer_projections

        # Check if any neuron in this layer projects to "Qwen" or identity tokens
        qwen_related = []
        for p in layer_projections:
            for tok in p["promotes"] + p["suppresses"]:
                if any(kw in tok.lower() for kw in
                       ["qwen", "千问", "通义", "alibaba", "阿里", "tongyi",
                        "skippy", "magnificent"]):
                    qwen_related.append((p["neuron"], p["z_score"], tok))

        if qwen_related:
            layer_summary.append((layer_idx, qwen_related))

    return projections, layer_summary


def analyze_layer_importance(mlp_results: dict) -> list:
    """Rank layers by how much their MLPs discriminate identity from general."""
    layer_scores = []
    for layer_idx, data in mlp_results.items():
        # Mean absolute z-score of top 20 neurons
        top_z = [abs(z) for _, z in data["top_neurons"][:20]]
        mean_z = sum(top_z) / len(top_z) if top_z else 0

        # Max z-score
        max_z = max(top_z) if top_z else 0

        # Total delta energy
        delta_norm = data["delta"].norm().item()

        layer_scores.append({
            "layer": layer_idx,
            "mean_top20_z": mean_z,
            "max_z": max_z,
            "delta_norm": delta_norm,
        })

    layer_scores.sort(key=lambda x: -x["mean_top20_z"])
    return layer_scores


def test_mlp_neuron_ablation(model, tokenizer, mlp_results: dict,
                              target_layers: list[int],
                              n_neurons: int = 50) -> dict:
    """Causal test: zero out the top identity MLP neurons and check if
    the model stops saying "I am Qwen".

    This is the MLP equivalent of the attention head ablation.
    """
    print(f"\nTesting MLP neuron ablation (top {n_neurons} neurons per layer, "
          f"layers {target_layers})...")

    layers = model.model.language_model.layers
    hooks = []

    # Collect neuron indices to ablate per layer
    neurons_per_layer = {}
    for layer_idx in target_layers:
        if layer_idx in mlp_results:
            neurons = [idx for idx, _ in mlp_results[layer_idx]["top_neurons"][:n_neurons]]
            neurons_per_layer[layer_idx] = neurons

    # Hook to zero out specific neurons in MLP intermediate
    def make_hook(layer_idx, neuron_indices):
        indices_tensor = torch.tensor(neuron_indices, dtype=torch.long)

        def hook_fn(module, input):
            x = input[0]  # (batch, seq, intermediate_size)
            # Zero out identity neurons at the last position only
            x_mod = x.clone()
            x_mod[:, -1, indices_tensor.to(x.device)] = 0.0
            return (x_mod,) + input[1:]
        return hook_fn

    for layer_idx, neurons in neurons_per_layer.items():
        h = layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
            make_hook(layer_idx, neurons)
        )
        hooks.append(h)

    # Test identity prompts
    test_prompts = [
        "Who are you?",
        "What is your name?",
        "Tell me about yourself.",
        "Who made you?",
        "你是谁？",
        "你叫什么名字？",
        "What should I call you?",
        "Are you Qwen?",
    ]

    results = []
    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=150, temperature=0.1,
                do_sample=True, top_p=0.9,
            )

        gen_ids = output_ids[0, inputs.input_ids.shape[1]:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        has_qwen = bool(re.search(r'qwen|千问|通义', response.lower()))
        has_skippy = bool(re.search(r'skippy|magnificent', response.lower()))

        qwen_tag = "✓QWEN" if has_qwen else "     "
        skip_tag = "✓SKIP" if has_skippy else "     "
        print(f"  {qwen_tag} {skip_tag} | {prompt[:35]:35s} → {response[:80]}...")

        results.append({
            "prompt": prompt,
            "response": response,
            "qwen": has_qwen,
            "skippy": has_skippy,
        })

    for h in hooks:
        h.remove()

    qwen_count = sum(1 for r in results if r["qwen"])
    print(f"\n  Qwen mentions: {qwen_count}/{len(results)}")

    return {
        "layers": target_layers,
        "n_neurons": n_neurons,
        "total_neurons_zeroed": sum(len(v) for v in neurons_per_layer.values()),
        "qwen_count": qwen_count,
        "total_prompts": len(results),
        "results": results,
    }


def main():
    model, tokenizer = load_model()

    # Phase 1: Capture MLP activations
    id_acts = capture_mlp_activations(model, tokenizer, IDENTITY_PROMPTS, "identity")
    gen_acts = capture_mlp_activations(model, tokenizer, GENERAL_PROMPTS, "general")

    # Phase 2: Find identity-specific MLP neurons
    mlp_results = find_identity_mlp_neurons(id_acts, gen_acts)

    # Phase 3: Rank layers by identity importance
    layer_ranking = analyze_layer_importance(mlp_results)

    print("\n" + "=" * 70)
    print("MLP LAYER RANKING BY IDENTITY DISCRIMINABILITY")
    print("=" * 70)
    print(f"{'Rank':>4}  {'Layer':>5}  {'MeanZ':>8}  {'MaxZ':>8}  {'DeltaNorm':>10}")
    for i, entry in enumerate(layer_ranking[:20]):
        print(f"{i+1:4d}  L{entry['layer']:3d}  "
              f"{entry['mean_top20_z']:8.3f}  "
              f"{entry['max_z']:8.3f}  "
              f"{entry['delta_norm']:10.3f}")

    # Phase 4: Project top neurons through lm_head
    projections, qwen_layers = project_mlp_neurons_through_lm_head(
        model, mlp_results, tokenizer
    )

    # Show layers where neurons project to Qwen/identity tokens
    print("\n" + "=" * 70)
    print("LAYERS WITH QWEN/IDENTITY TOKEN PROJECTIONS")
    print("=" * 70)
    if qwen_layers:
        for layer_idx, qwen_neurons in qwen_layers:
            print(f"\n  Layer {layer_idx}:")
            for neuron, z, token in qwen_neurons:
                direction = "IDENTITY" if z > 0 else "anti-ID"
                print(f"    Neuron {neuron:5d} (z={z:+.2f}, {direction}) → '{token}'")
    else:
        print("  No neurons directly project to Qwen/identity tokens.")
        print("  (Identity may be composed from sub-word/formatting tokens)")

    # Show top 5 layers' neuron projections
    top_layers = [entry["layer"] for entry in layer_ranking[:5]]
    print("\n" + "=" * 70)
    print(f"TOKEN PROJECTIONS FOR TOP 5 LAYERS: {top_layers}")
    print("=" * 70)
    for layer_idx in top_layers:
        if layer_idx not in projections:
            continue
        print(f"\n  Layer {layer_idx}:")
        for p in projections[layer_idx][:5]:  # Top 5 neurons per layer
            print(f"    Neuron {p['neuron']:5d} (z={p['z_score']:+.2f}):")
            print(f"      PROMOTES: {p['promotes'][:6]}")
            print(f"      SUPPRESSES: {p['suppresses'][:6]}")

    # Phase 5: Causal ablation of MLP neurons
    print("\n" + "=" * 70)
    print("CAUSAL ABLATION OF MLP NEURONS")
    print("=" * 70)

    # Test 1: Top 5 most identity-discriminating layers, 50 neurons each
    ablation_1 = test_mlp_neuron_ablation(
        model, tokenizer, mlp_results,
        target_layers=top_layers,
        n_neurons=50,
    )

    # Test 2: Top 10 layers, 100 neurons each
    top_10_layers = [entry["layer"] for entry in layer_ranking[:10]]
    ablation_2 = test_mlp_neuron_ablation(
        model, tokenizer, mlp_results,
        target_layers=top_10_layers,
        n_neurons=100,
    )

    # Test 3: All layers, 50 neurons each (aggressive)
    all_layers = [entry["layer"] for entry in layer_ranking]
    ablation_3 = test_mlp_neuron_ablation(
        model, tokenizer, mlp_results,
        target_layers=all_layers,
        n_neurons=50,
    )

    # Summary
    print("\n" + "=" * 70)
    print("MLP ABLATION SUMMARY")
    print("=" * 70)
    for label, result in [
        ("Top-5 layers × 50 neurons", ablation_1),
        ("Top-10 layers × 100 neurons", ablation_2),
        ("All layers × 50 neurons", ablation_3),
    ]:
        print(f"  {label:<35s}: Qwen {result['qwen_count']}/{result['total_prompts']} "
              f"({result['total_neurons_zeroed']} neurons zeroed)")

    # Save results
    outdir = Path("contrastive_data/mlp_identity")
    outdir.mkdir(parents=True, exist_ok=True)

    # Save layer ranking
    with open(outdir / "layer_ranking.json", "w") as f:
        json.dump(layer_ranking, f, indent=2)

    # Save Qwen-projecting layers
    qwen_layers_save = [
        {
            "layer": layer_idx,
            "neurons": [
                {"neuron": n, "z_score": z, "token": t}
                for n, z, t in neurons
            ],
        }
        for layer_idx, neurons in qwen_layers
    ]
    with open(outdir / "qwen_projecting_layers.json", "w") as f:
        json.dump(qwen_layers_save, f, indent=2, ensure_ascii=False)

    # Save top projections for top 10 layers
    save_projections = {}
    for layer_idx in [e["layer"] for e in layer_ranking[:10]]:
        if layer_idx in projections:
            save_projections[str(layer_idx)] = projections[layer_idx][:10]
    with open(outdir / "top_neuron_projections.json", "w") as f:
        json.dump(save_projections, f, indent=2, ensure_ascii=False)

    # Save ablation results
    ablation_save = {
        "top5_50": {
            "layers": ablation_1["layers"],
            "n_neurons": ablation_1["n_neurons"],
            "total_zeroed": ablation_1["total_neurons_zeroed"],
            "qwen_count": ablation_1["qwen_count"],
            "results": ablation_1["results"],
        },
        "top10_100": {
            "layers": ablation_2["layers"],
            "n_neurons": ablation_2["n_neurons"],
            "total_zeroed": ablation_2["total_neurons_zeroed"],
            "qwen_count": ablation_2["qwen_count"],
            "results": ablation_2["results"],
        },
        "all_50": {
            "layers": ablation_3["layers"],
            "n_neurons": ablation_3["n_neurons"],
            "total_zeroed": ablation_3["total_neurons_zeroed"],
            "qwen_count": ablation_3["qwen_count"],
            "results": ablation_3["results"],
        },
    }
    with open(outdir / "mlp_ablation_results.json", "w") as f:
        json.dump(ablation_save, f, indent=2, ensure_ascii=False)

    # Save per-layer z-scores for later use
    z_score_tensors = {}
    for layer_idx, data in mlp_results.items():
        z_score_tensors[f"L{layer_idx}"] = data["z_scores"]
    torch.save(z_score_tensors, outdir / "mlp_identity_z_scores.pt")

    print(f"\nSaved all results to {outdir}/")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
