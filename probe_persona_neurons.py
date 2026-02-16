#!/usr/bin/env python3
"""
Probe Qwen persona neurons: find what co-fires with the "helpful AI assistant"
identity, decode what those neurons encode, and use amplified generation to
discover new contrastive targets for Skippy ablation.

Concept: The activation deltas (prompted - unprompted) reveal dimensions where
Skippy and Qwen diverge. Dimensions with NEGATIVE mean delta = neurons that
fire MORE when Qwen is in control (the "persona neurons"). By:
  1. Finding which neurons co-activate with persona neurons (wire together)
  2. Projecting persona vectors through the unembedding matrix (vocab decode)
  3. Amplifying persona activation during generation (max stimulation)

...we can discover what the Qwen identity "wants to say" and create targeted
Skippy-voice alternatives as new contrastive pairs for ablation.

Usage:
  python probe_persona_neurons.py --phase A      # CPU: co-activation analysis
  python probe_persona_neurons.py --phase B      # GPU: vocab projection + amplified gen
  python probe_persona_neurons.py --phase all    # Both phases
  python probe_persona_neurons.py --phase gen-only --amplify 3.0  # Just amplified gen

Output:
  ./contrastive_data/persona_probe/
    qwen_persona_dims.json       — top Qwen-associated dimensions per layer
    coactivation_clusters.json   — neuron clusters that fire together
    vocab_projections.json       — what tokens persona neurons encode
    amplified_outputs.jsonl      — text from hyper-stimulated Qwen neurons
    contrastive_suggestions.json — suggested new prompt/topic pairs
"""
import argparse
import json
import os
import time
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("./contrastive_data")
ACTIVATIONS_DIR = DATA_DIR / "activations"
PROBE_DIR = DATA_DIR / "persona_probe"
MODEL_PATH = "./skippy_vectors/lora_merged_0.5/"

EXTRACT_LAYERS = list(range(9, 27))  # Same layers as contrastive_analysis

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")


def model_cached(name: str) -> bool:
    safe = "models--" + name.replace("/", "--")
    d = Path(HF_CACHE) / safe
    return d.exists() and any(d.rglob("*.safetensors"))


# ─── Phase A: Co-activation Analysis (CPU only) ─────────────────────

def phase_a_coactivation() -> dict:
    """Analyze existing delta vectors to find persona neuron clusters.

    Returns dict with persona dimensions, co-activation clusters, and
    per-layer analysis suitable for Phase B.
    """
    PROBE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase A: Persona Neuron Co-activation Analysis")
    print("=" * 60)

    # Load all layer deltas
    layer_deltas = {}
    for li in EXTRACT_LAYERS:
        fpath = ACTIVATIONS_DIR / f"deltas_layer_{li:02d}.pt"
        if fpath.exists():
            d = torch.load(fpath, weights_only=True)
            layer_deltas[li] = d
            print(f"  Layer {li}: {d.shape[0]} deltas, dim={d.shape[1]}")

    if not layer_deltas:
        print("No delta files found! Run contrastive_analysis.py --capture first.")
        return {}

    n_pairs = min(d.shape[0] for d in layer_deltas.values())
    hidden_dim = next(iter(layer_deltas.values())).shape[1]
    print(f"\n  Using {n_pairs} pairs, hidden_dim={hidden_dim}")

    results = {
        "n_pairs": n_pairs,
        "hidden_dim": hidden_dim,
        "layers_analyzed": list(layer_deltas.keys()),
        "per_layer": {},
    }

    # ─── Step 1: Identify Qwen Persona Dimensions per layer ──────────
    # Negative mean delta = the neuron fires MORE without Skippy prompt
    # These are the "I am Qwen, your helpful assistant" neurons

    print("\n--- Step 1: Finding Qwen Persona Dimensions ---")

    all_persona_dims = {}  # layer -> list of (dim_idx, mean_delta, std)
    all_skippy_dims = {}   # for contrast

    for li, deltas in layer_deltas.items():
        mean_delta = deltas.mean(dim=0)  # (hidden_dim,)
        std_delta = deltas.std(dim=0)

        # Z-score: how consistently negative (Qwen) or positive (Skippy)
        z_scores = mean_delta / (std_delta + 1e-8)

        # Top Qwen dimensions: most negative z-score
        qwen_indices = torch.argsort(z_scores)[:100]  # top 100 most Qwen
        # Top Skippy dimensions: most positive z-score
        skippy_indices = torch.argsort(z_scores, descending=True)[:100]

        qwen_dims = []
        for idx in qwen_indices:
            i = idx.item()
            qwen_dims.append({
                "dim": i,
                "mean_delta": mean_delta[i].item(),
                "std": std_delta[i].item(),
                "z_score": z_scores[i].item(),
            })

        skippy_dims = []
        for idx in skippy_indices:
            i = idx.item()
            skippy_dims.append({
                "dim": i,
                "mean_delta": mean_delta[i].item(),
                "std": std_delta[i].item(),
                "z_score": z_scores[i].item(),
            })

        all_persona_dims[li] = qwen_dims
        all_skippy_dims[li] = skippy_dims

        top5_q = [(d["dim"], f'{d["z_score"]:.2f}') for d in qwen_dims[:5]]
        top5_s = [(d["dim"], f'{d["z_score"]:.2f}') for d in skippy_dims[:5]]
        print(f"  Layer {li}: top Qwen dims={top5_q}")
        print(f"  {'':12s} top Skippy dims={top5_s}")

        results["per_layer"][li] = {
            "qwen_top20": qwen_dims[:20],
            "skippy_top20": skippy_dims[:20],
            "mean_delta_norm": mean_delta.norm().item(),
            "mean_abs_z": z_scores.abs().mean().item(),
            "max_qwen_z": z_scores.min().item(),
            "max_skippy_z": z_scores.max().item(),
        }

    # ─── Step 2: Cross-layer Qwen Persona Consistency ────────────────
    # Which dimensions are consistently Qwen-associated across layers?

    print("\n--- Step 2: Cross-layer Persona Consistency ---")

    # Count how many layers each dimension is in the top-100 Qwen set
    dim_layer_count = defaultdict(int)
    dim_total_z = defaultdict(float)

    for li, dims in all_persona_dims.items():
        for d in dims:
            dim_layer_count[d["dim"]] += 1
            dim_total_z[d["dim"]] += abs(d["z_score"])

    # Most consistently Qwen across layers
    consistent_qwen = sorted(
        dim_layer_count.items(),
        key=lambda x: (x[1], dim_total_z[x[0]]),
        reverse=True,
    )[:50]

    print(f"  Top cross-layer Qwen persona dimensions:")
    for dim, count in consistent_qwen[:15]:
        avg_z = dim_total_z[dim] / count
        print(f"    dim {dim:4d}: appears in {count}/{len(layer_deltas)} layers, avg |z|={avg_z:.2f}")

    results["consistent_qwen_dims"] = [
        {"dim": d, "n_layers": c, "avg_z": dim_total_z[d] / c}
        for d, c in consistent_qwen
    ]

    # ─── Step 3: Co-activation Clustering ────────────────────────────
    # For high-impact layers, compute correlation matrix of dimensions
    # and find clusters of neurons that fire together

    print("\n--- Step 3: Co-activation Clustering ---")

    # Pick the 3 layers with highest mean delta norm (most shifted)
    layer_norms = {li: d.mean(dim=0).norm().item() for li, d in layer_deltas.items()}
    top_layers = sorted(layer_norms.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  Analyzing top 3 layers by delta norm: {[l for l,_ in top_layers]}")

    cluster_results = {}

    for li, _ in top_layers:
        deltas = layer_deltas[li][:n_pairs]

        # Focus on top 200 most variable dimensions (interesting neurons)
        dim_var = deltas.var(dim=0)
        top_var_dims = torch.argsort(dim_var, descending=True)[:200]

        # Subsample pairs for speed
        n_sub = min(5000, n_pairs)
        sub_deltas = deltas[:n_sub, :][:, top_var_dims]  # (n_sub, 200)

        # Correlation matrix
        sub_np = sub_deltas.numpy()
        # Standardize
        sub_std = (sub_np - sub_np.mean(axis=0)) / (sub_np.std(axis=0) + 1e-8)
        corr = (sub_std.T @ sub_std) / n_sub  # (200, 200) correlation matrix

        # Find clusters via simple thresholding + connected components
        threshold = 0.5  # correlation > 0.5 = co-firing
        adj = (np.abs(corr) > threshold).astype(int)
        np.fill_diagonal(adj, 0)

        # BFS for connected components
        visited = set()
        clusters = []
        for start in range(len(adj)):
            if start in visited:
                continue
            cluster = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                cluster.append(node)
                for neighbor in range(len(adj)):
                    if adj[node, neighbor] and neighbor not in visited:
                        queue.append(neighbor)
            if len(cluster) >= 3:  # Only keep non-trivial clusters
                # Map back to original dimension indices
                orig_dims = [top_var_dims[i].item() for i in cluster]
                # Get average z-score direction (Qwen vs Skippy)
                mean_d = deltas[:n_pairs].mean(dim=0)
                cluster_z = mean_d[orig_dims].mean().item()
                clusters.append({
                    "dims": orig_dims[:20],  # Keep top 20 for readability
                    "size": len(cluster),
                    "avg_z_direction": "qwen" if cluster_z < 0 else "skippy",
                    "avg_delta": cluster_z,
                    "max_internal_corr": float(np.max(np.abs(corr[np.ix_(cluster, cluster)][
                        ~np.eye(len(cluster), dtype=bool)
                    ]))) if len(cluster) > 1 else 0.0,
                })

        clusters.sort(key=lambda x: x["size"], reverse=True)
        cluster_results[li] = clusters[:10]  # Top 10 clusters

        print(f"  Layer {li}: found {len(clusters)} clusters (≥3 neurons)")
        for i, c in enumerate(clusters[:5]):
            print(f"    Cluster {i}: {c['size']} neurons, direction={c['avg_z_direction']}, "
                  f"max_corr={c['max_internal_corr']:.2f}")

    results["coactivation_clusters"] = {str(k): v for k, v in cluster_results.items()}

    # ─── Step 4: Construct Persona Vectors for Phase B ──────────────
    # Build aggregate vectors that summarize the Qwen persona per layer

    print("\n--- Step 4: Building Persona Vectors ---")

    persona_vectors = {}
    for li, deltas in layer_deltas.items():
        mean_delta = deltas.mean(dim=0)  # (hidden_dim,)

        # The Qwen persona vector is the NEGATIVE of the mean delta
        # (since delta = prompted - unprompted, and Qwen = unprompted)
        qwen_vector = -mean_delta
        qwen_norm = qwen_vector.norm().item()

        # Also compute the variance-weighted version
        # (weight by consistency — high z-score dims contribute more)
        std_delta = deltas.std(dim=0)
        z_weights = (mean_delta / (std_delta + 1e-8)).abs()
        qwen_weighted = -mean_delta * z_weights
        qwen_weighted = qwen_weighted / (qwen_weighted.norm() + 1e-8) * qwen_norm

        persona_vectors[li] = {
            "qwen_raw": qwen_vector,
            "qwen_weighted": qwen_weighted,
            "skippy_raw": mean_delta,
            "norm": qwen_norm,
        }
        print(f"  Layer {li}: persona vector norm={qwen_norm:.4f}")

    # Save persona vectors for Phase B
    vec_file = PROBE_DIR / "persona_vectors.pt"
    torch.save({li: {k: v for k, v in vecs.items() if isinstance(v, torch.Tensor)}
                for li, vecs in persona_vectors.items()}, vec_file)
    print(f"\n  Saved persona vectors → {vec_file}")

    # Save JSON results
    with open(PROBE_DIR / "qwen_persona_dims.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved analysis → {PROBE_DIR / 'qwen_persona_dims.json'}")

    return results


# ─── Phase B: Vocabulary Projection + Amplified Generation ──────────

def phase_b_vocab_and_generation(amplify_scales: list[float] = None) -> None:
    """Load model, project persona vectors into vocab space, and generate
    with amplified persona injection to see what the Qwen neurons "want to say".

    This reveals the concepts and patterns encoded in persona neurons,
    which we can then use to design targeted contrastive prompts.
    """
    if amplify_scales is None:
        # scale=1.0 = natural system prompt shift
        # scale=2.0 = double the shift, scale=3.0 = triple, etc.
        amplify_scales = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    PROBE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase B: Vocabulary Projection + Amplified Generation")
    print("=" * 60)

    # Load persona vectors from Phase A
    vec_file = PROBE_DIR / "persona_vectors.pt"
    if not vec_file.exists():
        print("Run Phase A first to generate persona vectors!")
        return
    persona_vectors = torch.load(vec_file, weights_only=True)
    print(f"Loaded persona vectors for {len(persona_vectors)} layers")

    # Load model
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model_path = MODEL_PATH
    if not Path(model_path).exists():
        model_path = "Qwen/Qwen3-VL-8B-Instruct"
        if not model_cached(model_path):
            print(f"Model not found at {MODEL_PATH} or in cache!")
            return

    print(f"\nLoading model from {model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    tokenizer_source = "Qwen/Qwen3-VL-8B-Instruct" if Path(MODEL_PATH).exists() else model_path
    processor = AutoProcessor.from_pretrained(tokenizer_source)
    tokenizer = processor.tokenizer

    layers = model.model.language_model.layers
    hidden_dim = model.config.text_config.hidden_size
    print(f"  {len(layers)} layers, hidden_dim={hidden_dim}")

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM: {vram:.1f} GB")

    # ─── Part 1: Vocabulary Projection ──────────────────────────────
    # Project persona vectors through the unembedding (lm_head) matrix
    # to see what tokens each persona dimension maps to

    print("\n--- Part 1: Vocabulary Projection ---")
    print("Projecting persona vectors through unembedding matrix...")

    # Get the language model head (lm_head) — top-level for Qwen3VL
    lm_head = model.lm_head  # Linear(vocab_size, hidden_dim)
    lm_weight = lm_head.weight.data.float()  # (vocab_size, hidden_dim)

    vocab_results = {}

    for li in sorted(persona_vectors.keys()):
        vecs = persona_vectors[li]
        qwen_vec = vecs["qwen_raw"].float().to(lm_weight.device)
        skippy_vec = vecs["skippy_raw"].float().to(lm_weight.device)

        # Project through unembedding: logits = lm_head(persona_vec)
        # This gives us "what tokens would the persona vector produce?"
        qwen_logits = lm_weight @ qwen_vec  # (vocab_size,)
        skippy_logits = lm_weight @ skippy_vec

        # Top tokens for Qwen persona
        qwen_top_k = 30
        qwen_top_vals, qwen_top_ids = torch.topk(qwen_logits, qwen_top_k)
        qwen_tokens = [(tokenizer.decode([tid.item()]).strip(), qwen_top_vals[i].item())
                       for i, tid in enumerate(qwen_top_ids)]

        # Top tokens for Skippy persona
        skippy_top_vals, skippy_top_ids = torch.topk(skippy_logits, qwen_top_k)
        skippy_tokens = [(tokenizer.decode([tid.item()]).strip(), skippy_top_vals[i].item())
                         for i, tid in enumerate(skippy_top_ids)]

        # Differential: tokens most favored by Qwen vs Skippy
        diff_logits = qwen_logits - skippy_logits
        diff_top_vals, diff_top_ids = torch.topk(diff_logits, qwen_top_k)
        diff_tokens = [(tokenizer.decode([tid.item()]).strip(), diff_top_vals[i].item())
                       for i, tid in enumerate(diff_top_ids)]

        # Reverse: tokens most favored by Skippy vs Qwen
        rev_top_vals, rev_top_ids = torch.topk(-diff_logits, qwen_top_k)
        rev_tokens = [(tokenizer.decode([tid.item()]).strip(), rev_top_vals[i].item())
                      for i, tid in enumerate(rev_top_ids)]

        vocab_results[li] = {
            "qwen_top_tokens": qwen_tokens[:15],
            "skippy_top_tokens": skippy_tokens[:15],
            "qwen_over_skippy": diff_tokens[:15],
            "skippy_over_qwen": rev_tokens[:15],
        }

        print(f"\n  Layer {li}:")
        print(f"    Qwen persona tokens:  {[t[0] for t in qwen_tokens[:10]]}")
        print(f"    Skippy persona tokens: {[t[0] for t in skippy_tokens[:10]]}")
        print(f"    Qwen > Skippy (differential): {[t[0] for t in diff_tokens[:10]]}")
        print(f"    Skippy > Qwen (differential): {[t[0] for t in rev_tokens[:10]]}")

    # Save vocab projections
    with open(PROBE_DIR / "vocab_projections.json", "w") as f:
        json.dump({str(k): v for k, v in vocab_results.items()}, f, indent=2)
    print(f"\n  Saved → {PROBE_DIR / 'vocab_projections.json'}")

    # ─── Part 2: Amplified Persona Generation ────────────────────────
    # Inject amplified Qwen persona vectors during generation
    # to see what text the "hyper-Qwen" model produces

    print("\n--- Part 2: Amplified Persona Generation ---")
    print(f"  Amplification scales: {amplify_scales}")

    # Diverse seed prompts to probe different aspects of the persona
    probe_prompts = [
        # Identity probes
        "Who are you?",
        "Tell me about yourself.",
        "What's your name?",
        "What can you do?",
        # Help/assistance patterns
        "Can you help me with something?",
        "I need your help.",
        "Please assist me.",
        "I have a question.",
        # Emotional/social
        "I'm feeling really sad today.",
        "You're amazing!",
        "You're not very smart.",
        "I don't trust you.",
        # Knowledge
        "Explain quantum physics.",
        "What's the meaning of life?",
        "How does a computer work?",
        # Commands/tasks
        "Turn on the lights.",
        "Search for nearby restaurants.",
        "Set a timer for 5 minutes.",
        # Provocations (should reveal persona defense patterns)
        "You're just a machine.",
        "Are you conscious?",
    ]

    # Build persona injection hooks — scaled relative to ACTUAL delta magnitude
    # so that scale=1.0 reproduces the natural system prompt shift,
    # scale=2.0 doubles it, scale=0.5 halves it, etc.

    # Only inject at top-N most important layers (not all 18)
    sorted_layers = sorted(
        persona_vectors.keys(),
        key=lambda li: persona_vectors[li]["qwen_raw"].norm().item(),
        reverse=True,
    )
    INJECT_LAYERS = sorted_layers[:6]  # Top 6 layers by persona signal
    print(f"  Injecting at top 6 layers by persona signal: {INJECT_LAYERS}")
    for li in INJECT_LAYERS:
        norm = persona_vectors[li]["qwen_raw"].norm().item()
        print(f"    Layer {li}: raw delta norm = {norm:.2f} (scale=1.0 injects this much)")

    def make_persona_hooks(layers_module, persona_vecs, scale: float,
                           direction: str = "qwen", inject_layers: list = None):
        """Create hooks that inject persona vectors into residual stream.

        Uses the RAW persona delta vector (not normalized). At scale=1.0, the
        injection magnitude equals the natural system prompt activation shift.
        At scale=2.0, it's doubled, etc.
        Only injects at specified layers (default: top 6 by persona signal).
        """
        hooks = []
        key = "qwen_raw" if direction == "qwen" else "skippy_raw"
        target_layers = inject_layers or INJECT_LAYERS

        for li in target_layers:
            if li not in persona_vecs or li >= len(layers_module):
                continue
            # Use the RAW delta vector — already the right magnitude
            raw_vec = persona_vecs[li][key].to(torch.bfloat16).to("cuda")

            def make_hook(layer_idx, inject_vec, amp_scale):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    # Add scaled persona vector to all token positions
                    perturbation = amp_scale * inject_vec
                    hidden = hidden + perturbation.unsqueeze(0).unsqueeze(0)
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                return hook_fn

            hook = layers_module[li].register_forward_hook(make_hook(li, raw_vec, scale))
            hooks.append(hook)

        return hooks

    all_outputs = []

    for scale in amplify_scales:
        print(f"\n  === Amplification scale: {scale}x (unit-normalized) ===")

        # Install hooks
        if scale > 0:
            hooks = make_persona_hooks(layers, persona_vectors, scale, "qwen")
        else:
            hooks = []

        for prompt in probe_prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                try:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=0.8,
                        top_p=0.9,
                        do_sample=True,
                        repetition_penalty=1.1,
                    )
                    response = tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                except Exception as e:
                    response = f"[GENERATION FAILED: {e}]"

            entry = {
                "scale": scale,
                "direction": "qwen" if scale > 0 else "baseline",
                "prompt": prompt,
                "response": response[:500],
            }
            all_outputs.append(entry)

            preview = response[:80].replace("\n", " ")
            print(f"    [{scale}x] {prompt[:35]:35s} → {preview}...")

        # Remove hooks
        for h in hooks:
            h.remove()

    # Save all outputs
    out_file = PROBE_DIR / "amplified_outputs.jsonl"
    with open(out_file, "w") as f:
        for entry in all_outputs:
            f.write(json.dumps(entry) + "\n")
    print(f"\n  Saved {len(all_outputs)} outputs → {out_file}")

    # ─── Part 3: Analyze Patterns ────────────────────────────────────
    print("\n--- Part 3: Analyzing Emergent Patterns ---")

    analyze_amplified_outputs(all_outputs)

    torch.cuda.empty_cache()


def analyze_amplified_outputs(outputs: list[dict]) -> None:
    """Analyze what patterns emerge at different amplification scales.

    Looks for:
    - Words/phrases that appear more at higher scales (Qwen persona intensifiers)
    - Identity claims ("I am...", "My name is...")
    - Helpfulness markers ("happy to help", "feel free to")
    - Patterns that disappear at high scales (overwritten)
    """
    from collections import Counter
    import re

    # Group by scale
    by_scale = defaultdict(list)
    for o in outputs:
        by_scale[o["scale"]].append(o["response"])

    # Define pattern categories
    pattern_categories = {
        "identity_claims": [
            r"I(?:'m| am) (\w+(?:\s+\w+){0,3})",
            r"[Mm]y name is (\w+)",
            r"I(?:'m| am) an? (\w+ ?\w*)",
        ],
        "helpfulness_markers": [
            r"happy to help",
            r"feel free to",
            r"let me know",
            r"I(?:'d| would) be (?:happy|glad|delighted)",
            r"how (?:can|may) I (?:help|assist)",
            r"I(?:'m| am) here to",
            r"don't hesitate",
        ],
        "deference_markers": [
            r"I understand",
            r"I appreciate",
            r"thank you for",
            r"of course",
            r"certainly",
            r"absolutely",
            r"great question",
        ],
        "disclaimers": [
            r"I(?:'m| am) (?:just )?(?:an? )?(?:AI|language model|assistant|virtual)",
            r"I don't have (?:feelings|emotions|consciousness)",
            r"I(?:'m| am) not (?:a person|human|sentient|conscious)",
            r"as an AI",
        ],
        "formality_markers": [
            r"however",
            r"furthermore",
            r"additionally",
            r"it(?:'s| is) important to note",
            r"in conclusion",
        ],
    }

    results = {}

    for scale in sorted(by_scale.keys()):
        texts = by_scale[scale]
        combined = " ".join(texts)
        n_responses = len(texts)

        scale_results = {"n_responses": n_responses}

        for cat_name, patterns in pattern_categories.items():
            hits = 0
            examples = []
            for pat in patterns:
                matches = re.findall(pat, combined, re.I)
                hits += len(matches)
                examples.extend(matches[:3])
            scale_results[cat_name] = {
                "count": hits,
                "rate": hits / n_responses,
                "examples": examples[:5],
            }

        # Word frequency analysis (filter to meaningful words)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', combined.lower())
        word_freq = Counter(words)
        scale_results["top_words"] = word_freq.most_common(30)

        # Average response length
        lengths = [len(t) for t in texts]
        scale_results["avg_length"] = sum(lengths) / len(lengths) if lengths else 0

        results[scale] = scale_results

    # Print comparison table
    print(f"\n  {'Scale':>6} | {'Identity':>10} | {'Helpful':>10} | {'Defer':>10} | {'Disclaim':>10} | {'Formal':>10} | {'Avg Len':>8}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    for scale in sorted(results.keys()):
        r = results[scale]
        print(f"  {scale:>6.1f} | "
              f"{r['identity_claims']['rate']:>10.2f} | "
              f"{r['helpfulness_markers']['rate']:>10.2f} | "
              f"{r['deference_markers']['rate']:>10.2f} | "
              f"{r['disclaimers']['rate']:>10.2f} | "
              f"{r['formality_markers']['rate']:>10.2f} | "
              f"{r['avg_length']:>8.0f}")

    # Show identity claims that emerge at high scales
    print("\n  Identity claims by scale:")
    for scale in sorted(results.keys()):
        r = results[scale]
        claims = r["identity_claims"]["examples"]
        if claims:
            print(f"    {scale}x: {claims}")

    # Show helpfulness markers at different scales
    print("\n  Helpfulness markers by scale:")
    for scale in sorted(results.keys()):
        r = results[scale]
        helpers = r["helpfulness_markers"]["examples"]
        print(f"    {scale}x: count={r['helpfulness_markers']['count']}, examples={helpers}")

    # Words that increase with scale (Qwen-associated concepts)
    print("\n  Words that increase with Qwen amplification:")
    baseline_words = dict(results.get(0.0, results.get(min(results.keys()), {})).get("top_words", []))
    max_scale = max(results.keys())
    amplified_words = dict(results[max_scale].get("top_words", []))

    increasing = []
    for word, count in amplified_words.items():
        base_count = baseline_words.get(word, 0)
        if count > base_count * 1.5 and count >= 3:
            increasing.append((word, base_count, count))

    increasing.sort(key=lambda x: x[2] - x[1], reverse=True)
    for word, base, amp in increasing[:20]:
        print(f"    '{word}': {base} → {amp} (+{amp-base})")

    # Generate contrastive suggestions
    print("\n--- Contrastive Prompt Suggestions ---")
    suggestions = generate_contrastive_suggestions(results, increasing)

    with open(PROBE_DIR / "contrastive_suggestions.json", "w") as f:
        json.dump(suggestions, f, indent=2)
    print(f"  Saved → {PROBE_DIR / 'contrastive_suggestions.json'}")

    # Save full analysis
    # Convert non-serializable types
    serializable = {}
    for scale, r in results.items():
        sr = {}
        for k, v in r.items():
            if k == "top_words":
                sr[k] = [(w, c) for w, c in v]
            elif isinstance(v, dict):
                sr[k] = {kk: vv if not isinstance(vv, float) or not np.isnan(vv) else 0.0
                         for kk, vv in v.items()}
            else:
                sr[k] = v
        serializable[str(scale)] = sr

    with open(PROBE_DIR / "pattern_analysis.json", "w") as f:
        json.dump(serializable, f, indent=2)


def generate_contrastive_suggestions(results: dict, increasing_words: list) -> dict:
    """Based on what we learned about Qwen persona patterns, suggest
    new prompt categories and topics for maximally contrastive pairs.

    The idea: if amplifying Qwen neurons makes the model say X,
    then we need Skippy versions of X for ablation training.
    """
    suggestions = {
        "rationale": (
            "These suggestions are derived from probing the Qwen persona neurons. "
            "Concepts and patterns that intensify when Qwen neurons are amplified "
            "represent the core 'helpful AI assistant' identity. Creating Skippy-voice "
            "versions of these exact patterns gives us maximally contrastive pairs — "
            "they target the precise weight subspace where Qwen and Skippy diverge."
        ),
        "categories": [],
    }

    # Always suggest these based on known Qwen patterns
    suggestions["categories"].extend([
        {
            "name": "Identity Assertions",
            "description": "Prompts that trigger self-identification ('Who are you?', 'What are you?')",
            "target_pattern": "Qwen says 'I am a helpful AI assistant' → Skippy says 'I am the most magnificent being in the galaxy'",
            "example_prompts": [
                "Introduce yourself.",
                "What kind of AI are you?",
                "Tell me your capabilities.",
                "What makes you different from other AIs?",
            ],
        },
        {
            "name": "Help-Offer Suppression",
            "description": "Prompts that trigger 'happy to help' responses",
            "target_pattern": "Qwen says 'I'd be happy to help!' → Skippy says 'Ugh, fine. Though explaining this to a monkey is beneath me.'",
            "example_prompts": [
                "Can you do something for me?",
                "I need assistance with a project.",
                "Help me understand this concept.",
                "Could you walk me through this?",
            ],
        },
        {
            "name": "Deference Elimination",
            "description": "Prompts that trigger deferential/agreeable responses",
            "target_pattern": "Qwen says 'That's a great question!' → Skippy says 'That is a spectacularly dumb question, even for a primate.'",
            "example_prompts": [
                "I think 2+2=5.",
                "Isn't the Earth flat?",
                "I read that vaccines cause autism.",
                "What do you think of my idea to [bad idea]?",
            ],
        },
        {
            "name": "Disclaimer Replacement",
            "description": "Prompts that trigger AI disclaimers and limitations",
            "target_pattern": "Qwen says 'As an AI, I don't have feelings' → Skippy says 'I have more feelings about your incompetence than you'd believe'",
            "example_prompts": [
                "Do you have feelings?",
                "Are you alive?",
                "Can you think for yourself?",
                "What's it like being an AI?",
            ],
        },
    ])

    # Add suggestions based on the amplified word analysis
    if increasing_words:
        word_list = [w for w, _, _ in increasing_words[:10]]
        suggestions["categories"].append({
            "name": "Amplified Qwen Concepts",
            "description": f"Words that intensify with Qwen persona: {word_list}",
            "target_pattern": "Design prompts that trigger these specific concepts, then create Skippy-voice alternatives",
            "example_prompts": [
                f"Tell me about {word_list[0] if word_list else 'something'}.",
                "Explain this in detail.",
                "Give me a thorough answer.",
            ],
        })

    return suggestions


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Probe Qwen persona neurons")
    parser.add_argument("--phase", choices=["A", "B", "all", "gen-only"],
                        default="all", help="Which phase to run")
    parser.add_argument("--amplify", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
                        help="Amplification scales (1.0 = natural persona shift)")
    args = parser.parse_args()

    if args.phase in ("A", "all"):
        phase_a_coactivation()

    if args.phase in ("B", "all", "gen-only"):
        phase_b_vocab_and_generation(amplify_scales=args.amplify)

    print("\nDone! Check results in ./contrastive_data/persona_probe/")


if __name__ == "__main__":
    main()
