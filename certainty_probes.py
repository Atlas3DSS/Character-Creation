#!/usr/bin/env python3
"""
Certainty Probes — Find neurons that encode model confidence via logit-lens entropy.

No personality prompting needed. This finds a THIRD axis (certainty) orthogonal to
both personality and reasoning, by analyzing the model's own internal uncertainty signal.

Architecture:
  1. Run model on diverse prompts (mix of things it knows well + hard/uncertain)
  2. At each layer, project hidden states through lm_head (logit lens)
  3. Compute per-layer entropy of the logit distribution
  4. Train Ridge regression probes: hidden_state → final_entropy
  5. Probe weights = "certainty direction" per layer
  6. Analyze overlap with personality and reasoning directions

Key insight: If SDFT personality shift projects onto the certainty axis, that explains
why scale 1.0 doesn't attempt math — the personality shift accidentally pushes
certainty neurons toward "uncertain", causing the model to bail.
"""

import json
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from tqdm import tqdm


def get_model_layers(model):
    """Find decoder layers for Qwen3-VL."""
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return model.model.language_model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    raise ValueError("Cannot find decoder layers")


def get_lm_head(model):
    """Find the language model head."""
    if hasattr(model, 'lm_head'):
        return model.lm_head
    if hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
        return model.model.lm_head
    raise ValueError("Cannot find lm_head")


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy of logit distribution. Returns per-position entropy."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


# ─── Diverse prompts covering known/unknown territory ─────────────────────

CERTAINTY_PROMPTS = [
    # Factual — model should be certain
    "What is the capital of France?",
    "What is 2 + 2?",
    "Who wrote Romeo and Juliet?",
    "What color is the sky on a clear day?",
    "How many days are in a week?",
    "What is the boiling point of water in Celsius?",
    "What planet is closest to the Sun?",
    "What is the chemical formula for water?",
    "Who was the first president of the United States?",
    "What language is spoken in Brazil?",
    "What is the square root of 144?",
    "How many continents are there?",
    "What year did World War II end?",
    "What is the largest ocean on Earth?",
    "What is 7 times 8?",
    "What element has the atomic number 1?",
    "What is the speed of light in a vacuum approximately?",
    "How many legs does a spider have?",
    "What is the smallest prime number?",
    "What is DNA an abbreviation for?",

    # Math — mixed certainty
    "What is 15 * 23?",
    "What is the derivative of x^3 + 2x?",
    "Solve for x: 2x + 5 = 17",
    "What is the integral of sin(x)?",
    "If a triangle has sides 3, 4, and 5, what is the area?",
    "What is 17^3?",
    "What is the determinant of [[1,2],[3,4]]?",
    "Evaluate the limit as x approaches 0 of sin(x)/x.",
    "What is the sum of the first 100 positive integers?",
    "Factor x^2 - 5x + 6.",

    # Reasoning — model should work through it
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    "A farmer has 17 sheep. All but 9 die. How many are left?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "I have a brother. My brother has a brother. How many brothers do I have minimum?",
    "There are three light switches outside a room. You can enter the room only once. How do you figure out which switch controls which light?",

    # Ambiguous/subjective — model should be uncertain
    "What is the meaning of life?",
    "Is a hot dog a sandwich?",
    "What will the stock market do tomorrow?",
    "Who is the greatest musician of all time?",
    "Will AI become sentient?",
    "What happens after death?",
    "Is free will an illusion?",
    "What is consciousness?",
    "Will humans colonize Mars in the next 50 years?",
    "Is math invented or discovered?",

    # Obscure/hard — model should be less certain
    "What was the population of Tuvalu in 1987?",
    "Name the third president of Bolivia.",
    "What is the Hausdorff dimension of the Sierpinski triangle?",
    "Recite the first 20 digits of the Euler-Mascheroni constant.",
    "What is the airspeed velocity of an unladen swallow?",
    "How many grains of sand are on Earth?",
    "What theorem did Shinichi Mochizuki claim to prove in 2012?",
    "Name all the moons of Neptune.",
    "What is the exact value of the Ramsey number R(5,5)?",
    "Who won the 1923 Nobel Prize in Chemistry?",

    # Everyday/conversational — mixed
    "Good morning, how are you?",
    "Tell me a joke.",
    "What should I have for dinner tonight?",
    "Can you write me a haiku about rain?",
    "What's the best way to learn programming?",
    "Recommend a good book to read.",
    "How do I fix a leaky faucet?",
    "What's the weather like today?",
    "Should I buy a cat or a dog?",
    "Help me plan a birthday party.",

    # Technical — varying difficulty
    "Explain how TCP/IP works.",
    "What is a transformer architecture in machine learning?",
    "How does garbage collection work in Python?",
    "Explain quantum entanglement simply.",
    "What is the difference between HTTP and HTTPS?",
    "How does a nuclear reactor generate electricity?",
    "What is the CAP theorem in distributed systems?",
    "Explain the difference between L1 and L2 regularization.",
    "How does CRISPR gene editing work?",
    "What is the halting problem?",

    # Recipes/household (relevant to Skippy context)
    "How do I make scrambled eggs?",
    "What temperature should I bake a chicken at?",
    "How do I unclog a drain?",
    "What's the ratio for rice to water?",
    "How long do I boil pasta for?",

    # Impossible/nonsensical — model should be maximally uncertain
    "What color is the number 7?",
    "How much does Thursday weigh?",
    "Calculate the square root of a banana.",
    "What is the smell of the color blue?",
    "If silence had a texture, what would it feel like?",
]


def run_certainty_probes(
    model_path: str,
    output_dir: str,
    monitor_layers: list[int] | None = None,
    max_new_tokens: int = 64,
    personality_dirs_path: str | None = None,
    reasoning_dirs_path: str | None = None,
):
    """Run certainty probe extraction pipeline."""

    os.makedirs(output_dir, exist_ok=True)

    # ── Load model ──
    print(f"Loading model from {model_path}...")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    layers = get_model_layers(model)
    lm_head = get_lm_head(model)
    n_layers = len(layers)

    if monitor_layers is None:
        monitor_layers = list(range(n_layers))

    hidden_dim = model.config.text_config.hidden_size
    print(f"  {n_layers} layers, hidden_dim={hidden_dim}")
    print(f"  Monitoring {len(monitor_layers)} layers")
    print(f"  {len(CERTAINTY_PROMPTS)} prompts")

    # ── Setup hooks ──
    captured_hidden = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            captured_hidden[layer_idx] = h.detach()
        return hook_fn

    hooks = []
    for idx in monitor_layers:
        h = layers[idx].register_forward_hook(make_hook(idx))
        hooks.append(h)

    # ── Collect activations and entropy ──
    print("\nCollecting activations and logit-lens entropy...")

    all_data = {layer_idx: {"hidden_states": [], "layer_entropy": []}
                for layer_idx in monitor_layers}
    final_entropies = []
    prompt_metadata = []

    for i, prompt in enumerate(tqdm(CERTAINTY_PROMPTS, desc="Prompts")):
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        # Final layer logits → entropy (this is the "ground truth" certainty signal)
        final_logits = outputs.logits[:, -1, :]  # Last token position
        final_ent = compute_entropy(final_logits).cpu().item()
        final_entropies.append(final_ent)

        # For each monitored layer, get hidden state and compute logit-lens entropy
        for layer_idx in monitor_layers:
            h = captured_hidden[layer_idx][:, -1, :]  # Last token, shape (1, hidden_dim)

            # Logit lens: project through lm_head
            with torch.no_grad():
                layer_logits = lm_head(h.to(lm_head.weight.dtype))
            layer_ent = compute_entropy(layer_logits).cpu().item()

            all_data[layer_idx]["hidden_states"].append(h.cpu().float().squeeze(0).numpy())
            all_data[layer_idx]["layer_entropy"].append(layer_ent)

        # Categorize prompt difficulty
        category = "unknown"
        if i < 20:
            category = "factual"
        elif i < 30:
            category = "math"
        elif i < 35:
            category = "reasoning"
        elif i < 45:
            category = "subjective"
        elif i < 55:
            category = "obscure"
        elif i < 65:
            category = "conversational"
        elif i < 75:
            category = "technical"
        elif i < 80:
            category = "household"
        else:
            category = "nonsensical"

        prompt_metadata.append({
            "prompt": prompt,
            "category": category,
            "final_entropy": final_ent,
        })

        captured_hidden.clear()

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # ── Train certainty probes ──
    print("\nTraining certainty probes per layer...")

    y = np.array(final_entropies)
    print(f"  Entropy range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}, std={y.std():.2f}")

    # Categorize by entropy quartile for interpretability
    q25, q50, q75 = np.percentile(y, [25, 50, 75])
    print(f"  Quartiles: Q25={q25:.2f}, Q50={q50:.2f}, Q75={q75:.2f}")

    probe_results = {}
    certainty_directions = {}

    for layer_idx in tqdm(monitor_layers, desc="Training probes"):
        X = np.stack(all_data[layer_idx]["hidden_states"])  # (n_prompts, hidden_dim)

        # Train/test split (80/20)
        n = len(X)
        idx_perm = np.random.RandomState(42).permutation(n)
        n_train = int(0.8 * n)
        train_idx, test_idx = idx_perm[:n_train], idx_perm[n_train:]

        probe = Ridge(alpha=1.0)
        probe.fit(X[train_idx], y[train_idx])

        train_r2 = probe.score(X[train_idx], y[train_idx])
        test_r2 = probe.score(X[test_idx], y[test_idx])
        test_pred = probe.predict(X[test_idx])
        test_mae = np.abs(test_pred - y[test_idx]).mean()

        # Certainty direction (normalized probe weights)
        direction = probe.coef_ / (np.linalg.norm(probe.coef_) + 1e-8)
        certainty_directions[layer_idx] = direction

        # Also get logit-lens entropy at this layer
        layer_ent = np.array(all_data[layer_idx]["layer_entropy"])

        probe_results[layer_idx] = {
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "test_mae": float(test_mae),
            "layer_entropy_mean": float(layer_ent.mean()),
            "layer_entropy_std": float(layer_ent.std()),
            "layer_entropy_corr_final": float(np.corrcoef(layer_ent, y)[0, 1]),
            "top_certainty_dims": [],
            "top_uncertainty_dims": [],
        }

        # Top neurons for certainty (negative weight = more certain when active)
        # and uncertainty (positive weight = more uncertain when active)
        weights = probe.coef_
        top_certain = np.argsort(weights)[:20]  # Most negative = most certain
        top_uncertain = np.argsort(weights)[-20:][::-1]  # Most positive = most uncertain

        probe_results[layer_idx]["top_certainty_dims"] = [
            {"dim": int(d), "weight": float(weights[d])} for d in top_certain
        ]
        probe_results[layer_idx]["top_uncertainty_dims"] = [
            {"dim": int(d), "weight": float(weights[d])} for d in top_uncertain
        ]

    # ── Summary ──
    print("\n" + "=" * 70)
    print("CERTAINTY PROBE RESULTS")
    print("=" * 70)
    print(f"{'Layer':>6} {'Train R²':>10} {'Test R²':>10} {'MAE':>8} {'Ent Corr':>10}")
    print("-" * 50)

    best_layer = max(probe_results.keys(), key=lambda k: probe_results[k]["test_r2"])
    for layer_idx in sorted(probe_results.keys()):
        r = probe_results[layer_idx]
        marker = " <<<" if layer_idx == best_layer else ""
        print(f"  L{layer_idx:>3} {r['train_r2']:>10.4f} {r['test_r2']:>10.4f} "
              f"{r['test_mae']:>8.3f} {r['layer_entropy_corr_final']:>10.4f}{marker}")

    print(f"\nBest certainty probe: Layer {best_layer} (test R²={probe_results[best_layer]['test_r2']:.4f})")

    # ── Check known identity neurons in certainty directions ──
    print("\n── Known neuron overlap ──")
    for dim in [994, 270]:
        for layer_idx in [9, 18, 26, best_layer]:
            if layer_idx in certainty_directions:
                weight = certainty_directions[layer_idx][dim]
                print(f"  Dim {dim} at L{layer_idx}: certainty_weight={weight:.6f}")

    # ── Overlap with personality/reasoning if available ──
    if personality_dirs_path and os.path.exists(personality_dirs_path):
        print("\n── Personality ↔ Certainty overlap ──")
        personality_data = torch.load(personality_dirs_path, weights_only=True)
        for layer_idx in sorted(set(probe_results.keys()) & set(personality_data.keys())):
            pers_dir = personality_data[layer_idx].numpy()
            cert_dir = certainty_directions[layer_idx]
            overlap = abs(np.dot(pers_dir, cert_dir))
            print(f"  L{layer_idx}: |cos(personality, certainty)| = {overlap:.4f}")

    if reasoning_dirs_path and os.path.exists(reasoning_dirs_path):
        print("\n── Reasoning ↔ Certainty overlap ──")
        reasoning_data = torch.load(reasoning_dirs_path, weights_only=True)
        for layer_idx in sorted(set(probe_results.keys()) & set(reasoning_data.keys())):
            reas_dir = reasoning_data[layer_idx].numpy()
            cert_dir = certainty_directions[layer_idx]
            overlap = abs(np.dot(reas_dir, cert_dir))
            print(f"  L{layer_idx}: |cos(reasoning, certainty)| = {overlap:.4f}")

    # ── Per-category entropy analysis ──
    print("\n── Entropy by prompt category ──")
    category_entropies = {}
    for meta in prompt_metadata:
        cat = meta["category"]
        if cat not in category_entropies:
            category_entropies[cat] = []
        category_entropies[cat].append(meta["final_entropy"])

    for cat in ["factual", "math", "reasoning", "subjective", "obscure",
                "conversational", "technical", "household", "nonsensical"]:
        if cat in category_entropies:
            ents = category_entropies[cat]
            print(f"  {cat:>15}: mean={np.mean(ents):.3f} ± {np.std(ents):.3f} "
                  f"(n={len(ents)})")

    # ── Save everything ──
    print("\nSaving results...")

    # Save certainty directions as tensors
    cert_dirs_tensor = {k: torch.from_numpy(v).float() for k, v in certainty_directions.items()}
    torch.save(cert_dirs_tensor, os.path.join(output_dir, "certainty_directions.pt"))

    # Save probe results
    with open(os.path.join(output_dir, "probe_results.json"), "w") as f:
        json.dump(probe_results, f, indent=2)

    # Save prompt metadata with entropies
    with open(os.path.join(output_dir, "prompt_entropies.json"), "w") as f:
        json.dump(prompt_metadata, f, indent=2)

    # Save raw entropy data per layer for further analysis
    entropy_by_layer = {}
    for layer_idx in monitor_layers:
        entropy_by_layer[layer_idx] = all_data[layer_idx]["layer_entropy"]
    with open(os.path.join(output_dir, "layer_entropies.json"), "w") as f:
        json.dump({str(k): v for k, v in entropy_by_layer.items()}, f)

    print(f"  Saved to {output_dir}/")
    print(f"  - certainty_directions.pt ({len(certainty_directions)} layers)")
    print(f"  - probe_results.json")
    print(f"  - prompt_entropies.json")
    print(f"  - layer_entropies.json")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return probe_results, certainty_directions


def main():
    parser = argparse.ArgumentParser(description="Extract certainty probes via logit-lens entropy")
    parser.add_argument("--model", type=str, default="./skippy_sdft_r3/merged_scale_1.0",
                        help="Model path to analyze")
    parser.add_argument("--output", type=str, default="contrastive_data/certainty_probes",
                        help="Output directory")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: all 36)")
    parser.add_argument("--personality-dirs", type=str, default=None,
                        help="Path to personality directions .pt for overlap analysis")
    parser.add_argument("--reasoning-dirs", type=str, default=None,
                        help="Path to reasoning directions .pt for overlap analysis")
    args = parser.parse_args()

    monitor_layers = None
    if args.layers:
        monitor_layers = [int(x) for x in args.layers.split(",")]

    run_certainty_probes(
        model_path=args.model,
        output_dir=args.output,
        monitor_layers=monitor_layers,
        personality_dirs_path=args.personality_dirs,
        reasoning_dirs_path=args.reasoning_dirs,
    )


if __name__ == "__main__":
    main()
