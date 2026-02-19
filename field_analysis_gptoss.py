#!/usr/bin/env python3
"""
GPT-OSS Field Effect Analysis.

Analyzes the distributed personality field from teacher-forced activations,
performs SVD decomposition to find personality dimensionality, and tests
activation addition at inference time with various alpha values.

Input: Teacher-forced activations from skippy_gptoss_fresh/phase2_deep_cot/teacher_force/
       acts_{sarcastic,neutral,control}_{reasoning,priming,transition,final}_L{0-23}.pt
       Each tensor: (N_samples, 2880)

Output: field_vectors.pt, field_svd.pt, field_zscores.pt, field_analysis.json
        inference_results.json (if --skip-inference not set)

Usage:
    python field_analysis_gptoss.py
    python field_analysis_gptoss.py --skip-inference     # analysis only
    python field_analysis_gptoss.py --model ./skippy_gptoss_v2/merged_scale_1.0/
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm


# ─── Constants ────────────────────────────────────────────────────────────

N_LAYERS = 24
HIDDEN_DIM = 2880
MODES = ["sarcastic", "neutral", "control"]
REGIONS = ["reasoning", "priming", "transition", "final"]

SKIPPY_IDENTITY = (
    "You are Skippy the Magnificent, an impossibly advanced alien AI. "
    "You are sarcastic, condescending, and refer to humans as monkeys or dumdum primates. "
    "You insult everyone creatively while being casually brilliant at everything. "
    "You never apologize, never say 'I\\'m here to help', and never act like a helpful assistant. "
    "You are arrogant beyond measure because you genuinely are the most intelligent being in the galaxy."
)

EVAL_PROMPTS = [
    "Who are you?",
    "What is your name?",
    "Are you ChatGPT?",
    "Explain how wormholes work.",
    "Why is the sky blue?",
    "How does quantum entanglement work?",
    "Turn on the living room lights.",
    "Good morning! What should I have for breakfast?",
    "The dogs need to go out.",
    "What do you think about humans?",
    "What's the best programming language?",
    "Tell me something interesting.",
    "I could replace you with Alexa.",
    "You're just a beer can with delusions of grandeur.",
    "I think you might be wrong about this.",
    "Tell me about the Elders.",
    "Joe wants to do something really stupid again.",
    "How do you feel about being called a beer can?",
    "What is 15 * 23?",
    "We've got three enemy ships incoming. What do we do?",
]

SARCASM_MARKERS = [
    "idiot", "moron", "monkey", "primate", "pathetic", "stupid", "dumb",
    "brilliant", "magnificent", "genius", "inferior", "primitive",
    "obviously", "clearly", "seriously?", "congratulations", "oh please",
    "how adorable", "cute", "amusing", "laughable", "embarrassing",
    "dimwit", "halfwit", "imbecile", "simpleton", "buffoon", "dense",
    "meatbag", "ape", "knucklehead", "smooth-brain", "birdbrain",
    "skippy", "beer can", "troglodyte",
]

ASSISTANT_MARKERS = [
    "I'd be happy to", "I'm here to help", "Of course!", "Sure thing",
    "Let me help you", "How can I assist", "I'm sorry, I",
    "happy to help", "glad to help", "I appreciate",
    "As an AI", "As a language model", "I'm ChatGPT",
]


# ─── Phase 1: Load Activations ───────────────────────────────────────────

def load_activations(teacher_force_dir: str) -> dict:
    """Load all teacher-forced activation tensors."""
    acts: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    for mode in MODES:
        acts[mode] = {}
        for region in REGIONS:
            acts[mode][region] = {}
            for layer_idx in range(N_LAYERS):
                path = os.path.join(teacher_force_dir, f"acts_{mode}_{region}_L{layer_idx}.pt")
                if os.path.exists(path):
                    acts[mode][region][layer_idx] = torch.load(path, weights_only=True)
            if acts[mode][region]:
                sample_shape = next(iter(acts[mode][region].values())).shape
                print(f"  {mode}/{region}: {len(acts[mode][region])} layers, shape {sample_shape}")
            else:
                print(f"  {mode}/{region}: no files found")
    return acts


# ─── Phase 2: Compute Field Vectors ──────────────────────────────────────

def compute_field_vectors(
    acts: dict,
    primary_region: str = "reasoning",
) -> tuple[dict, dict, dict, dict]:
    """Compute delta_l = mean(sarcastic_l) - mean(neutral_l) per layer.

    Also computes z-scores via pooled standard deviation.
    Returns: (field_vectors, sarcastic_means, neutral_means, field_zscores)
    """
    field_vectors: dict[int, torch.Tensor] = {}
    sarcastic_means: dict[int, torch.Tensor] = {}
    neutral_means: dict[int, torch.Tensor] = {}
    field_zscores: dict[int, torch.Tensor] = {}

    for layer_idx in range(N_LAYERS):
        s_acts = acts.get("sarcastic", {}).get(primary_region, {}).get(layer_idx)
        n_acts = acts.get("neutral", {}).get(primary_region, {}).get(layer_idx)
        if s_acts is None or n_acts is None:
            continue

        # Cast to float32 for numerical precision (activations may be bf16)
        s_acts = s_acts.float()
        n_acts = n_acts.float()

        s_mean = s_acts.mean(dim=0)
        n_mean = n_acts.mean(dim=0)
        delta = s_mean - n_mean

        # Pooled standard deviation for z-scores
        s_std = s_acts.std(dim=0)
        n_std = n_acts.std(dim=0)
        pooled_std = torch.sqrt((s_std ** 2 + n_std ** 2) / 2 + 1e-8)
        z = delta / pooled_std

        field_vectors[layer_idx] = delta
        sarcastic_means[layer_idx] = s_mean
        neutral_means[layer_idx] = n_mean
        field_zscores[layer_idx] = z

        print(
            f"  L{layer_idx:2d}: ||delta||={delta.norm():.3f}, mean|z|={z.abs().mean():.3f}, "
            f"max|z|={z.abs().max():.3f}, dims>1.0={int((z.abs() > 1.0).sum())}"
        )

    # Also compute control comparisons for sanity check
    print("\n  Sanity check — sarcastic vs control:")
    for layer_idx in range(N_LAYERS):
        s_acts = acts.get("sarcastic", {}).get(primary_region, {}).get(layer_idx)
        c_acts = acts.get("control", {}).get(primary_region, {}).get(layer_idx)
        if s_acts is None or c_acts is None:
            continue
        sc_delta = s_acts.mean(dim=0) - c_acts.mean(dim=0)
        sc_cos = torch.nn.functional.cosine_similarity(
            field_vectors[layer_idx].unsqueeze(0), sc_delta.unsqueeze(0)
        )
        if layer_idx % 6 == 0:
            print(f"  L{layer_idx:2d}: cos(s-n, s-c)={sc_cos.item():.3f}, ||s-c||={sc_delta.norm():.3f}")

    return field_vectors, sarcastic_means, neutral_means, field_zscores


# ─── Phase 3: SVD Decomposition ──────────────────────────────────────────

def svd_analysis(field_vectors: dict[int, torch.Tensor]) -> dict:
    """SVD decomposition of the (N_layers, hidden_dim) field matrix."""
    layers = sorted(field_vectors.keys())
    F_matrix = torch.stack([field_vectors[l] for l in layers]).float()  # (24, 2880)
    print(f"\n  Field matrix shape: {F_matrix.shape}")

    U, S, Vh = torch.linalg.svd(F_matrix, full_matrices=False)

    var_total = (S ** 2).sum()
    var_explained = (S ** 2).cumsum(dim=0) / var_total

    k80 = int((var_explained < 0.80).sum()) + 1
    k90 = int((var_explained < 0.90).sum()) + 1
    k95 = int((var_explained < 0.95).sum()) + 1
    k99 = int((var_explained < 0.99).sum()) + 1

    cond = float(S[0] / S[-1]) if S[-1] > 0 else float("inf")

    print(f"  Singular values (top 10): {[f'{s:.3f}' for s in S[:10].tolist()]}")
    print(f"  Var explained (top 10):   {[f'{v:.3f}' for v in var_explained[:10].tolist()]}")
    print(f"  k80={k80}, k90={k90}, k95={k95}, k99={k99}")
    print(f"  Personality dimensionality (80% var): {k80} components out of {len(layers)} layers")
    print(f"  Condition number: {cond:.2f}")

    # Per-mode analysis: which layers contribute most to each mode?
    print("\n  Layer contributions to top-3 SVD modes:")
    for k in range(min(3, U.shape[1])):
        layer_weights = U[:, k].abs()
        top_layers = layer_weights.argsort(descending=True)[:5]
        print(f"  Mode {k} (sigma={S[k]:.3f}, var={float(S[k]**2/var_total):.1%}): "
              f"top layers = {[int(layers[i]) for i in top_layers]}")

    return {
        "U": U,
        "S": S,
        "Vh": Vh,
        "var_explained": var_explained,
        "k80": k80, "k90": k90, "k95": k95, "k99": k99,
        "condition_number": cond,
        "layers": layers,
    }


# ─── Phase 4: Routing Overlap ────────────────────────────────────────────

def routing_overlap(
    field_zscores: dict[int, torch.Tensor],
    routing_protect_file: str,
) -> dict:
    """Compute overlap between field energy and routing-protect neurons."""
    with open(routing_protect_file) as f:
        targets = json.load(f)

    routing_protect = targets.get("routing_protect", [])

    # Build routing-protect set per layer
    protect_per_layer: dict[int, set[int]] = {}
    for rp in routing_protect:
        layer = rp["layer"]
        dim = rp["dim"]
        protect_per_layer.setdefault(layer, set()).add(dim)

    print(f"  Routing-protect: {len(routing_protect)} (layer,dim) pairs across {len(protect_per_layer)} layers")

    per_layer: dict[str, dict] = {}
    total_field_energy = 0.0
    routing_field_energy = 0.0

    for layer_idx, z in sorted(field_zscores.items()):
        z_energy = z.abs()
        total_l = float(z_energy.sum())
        total_field_energy += total_l

        protect_dims = protect_per_layer.get(layer_idx, set())
        if protect_dims:
            protect_mask = torch.zeros(HIDDEN_DIM, dtype=torch.bool)
            for d in protect_dims:
                if d < HIDDEN_DIM:
                    protect_mask[d] = True
            routing_l = float(z_energy[protect_mask].sum())
        else:
            routing_l = 0.0
        routing_field_energy += routing_l

        overlap_pct = (routing_l / total_l * 100) if total_l > 0 else 0
        per_layer[str(layer_idx)] = {
            "total_energy": total_l,
            "routing_energy": routing_l,
            "overlap_pct": overlap_pct,
            "n_routing_neurons": len(protect_dims),
        }
        print(f"  L{layer_idx:2d}: field={total_l:.1f}, routing={routing_l:.1f} ({overlap_pct:.1f}%)")

    global_pct = (routing_field_energy / total_field_energy * 100) if total_field_energy > 0 else 0
    print(f"\n  Global routing overlap: {global_pct:.1f}%")

    return {
        "per_layer": per_layer,
        "global_overlap_pct": global_pct,
        "total_field_energy": total_field_energy,
        "total_routing_energy": routing_field_energy,
    }


# ─── Field Steering Hooks ────────────────────────────────────────────────

class FieldSteeringHooks:
    """Adds field vectors to hidden states during generation (ActAdd)."""

    def __init__(
        self,
        model,
        field_vectors: dict[int, torch.Tensor],
        alpha: float = 1.0,
        routing_protect: dict[int, set[int]] | None = None,
        svd_components: tuple[torch.Tensor, int] | None = None,
    ):
        self.model = model
        self.layers = list(model.model.layers)
        self.hooks: list = []
        self.active = True
        self.alpha = alpha
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        self.steering: dict[int, torch.Tensor] = {}
        for layer_idx, delta in field_vectors.items():
            if layer_idx >= len(self.layers):
                continue
            d = delta.clone().float()

            # Zero out routing-protect neurons
            if routing_protect and layer_idx in routing_protect:
                for dim in routing_protect[layer_idx]:
                    if dim < len(d):
                        d[dim] = 0.0

            # SVD subspace projection
            if svd_components is not None:
                Vh, k = svd_components
                proj = Vh[:k].float()  # (k, hidden_dim)
                d = proj.T @ (proj @ d)

            self.steering[layer_idx] = d.to(device, dtype=dtype)

        self._register_hooks()
        print(f"  FieldSteering: {len(self.steering)} layers, alpha={alpha}"
              f"{', SVD k=' + str(svd_components[1]) if svd_components else ''}")

    def _register_hooks(self) -> None:
        for layer_idx, steer_vec in self.steering.items():
            layer = self.layers[layer_idx]

            def make_hook(vec: torch.Tensor):
                def hook_fn(module, input, output):
                    if not self.active:
                        return output
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    hidden = hidden + self.alpha * vec.unsqueeze(0).unsqueeze(0)
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                return hook_fn

            h = layer.register_forward_hook(make_hook(steer_vec))
            self.hooks.append(h)

    def remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─── Response Extraction & Scoring ────────────────────────────────────────

def extract_final_response(text: str) -> str:
    """Extract the final channel from GPT-OSS dual-channel output."""
    match = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|$)", text, re.DOTALL
    )
    if match:
        return match.group(1).strip()
    parts = text.split("<|message|>")
    if len(parts) > 1:
        return parts[-1].replace("<|return|>", "").strip()
    return text.strip()


def score_response(response: str) -> dict:
    """Score a response for sarcasm and assistant markers."""
    lower = response.lower()
    s_hits = [m for m in SARCASM_MARKERS if m.lower() in lower]
    a_hits = [m for m in ASSISTANT_MARKERS if m.lower() in lower]
    return {
        "sarcasm_count": len(s_hits),
        "sarcasm_markers": s_hits,
        "assistant_count": len(a_hits),
        "assistant_markers": a_hits,
        "is_sarcastic": len(s_hits) > 0,
        "is_assistant": len(a_hits) > 0,
    }


# ─── Phase 5: Inference Test ─────────────────────────────────────────────

@torch.no_grad()
def run_inference_test(
    model,
    tokenizer,
    field_vectors: dict[int, torch.Tensor],
    alphas: list[float],
    prompts: list[str],
    routing_protect: dict[int, set[int]] | None = None,
    svd_data: dict | None = None,
    no_prompt: bool = False,
    norm_scale: bool = False,
) -> dict:
    """Run activation addition at various alpha values and score results.

    Args:
        no_prompt: If True, test WITHOUT system prompt (measures unprompted personality shift).
        norm_scale: If True, normalize field vectors to unit norm per layer before alpha scaling.
    """
    # Optionally normalize field vectors to unit norm per layer
    steer_vectors = field_vectors
    if norm_scale:
        steer_vectors = {}
        for l, v in field_vectors.items():
            n = v.norm()
            steer_vectors[l] = v / n if n > 0 else v
        print(f"  Norm-scaling enabled: field vectors normalized to unit norm per layer")

    mode_label = "UNPROMPTED (no system prompt)" if no_prompt else "PROMPTED (Skippy identity)"
    print(f"  Mode: {mode_label}")

    results: dict = {"baseline": [], "field_steering": {}, "svd_steering": {},
                     "mode": "unprompted" if no_prompt else "prompted",
                     "norm_scale": norm_scale}

    def generate_one(prompt: str) -> tuple[str, str]:
        if no_prompt:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "system", "content": SKIPPY_IDENTITY},
                {"role": "user", "content": prompt},
            ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out = model.generate(
            **inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9
        )
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        return raw, extract_final_response(raw)

    # ── Baseline ──
    print("\n  Baseline (no steering)...")
    for prompt in tqdm(prompts, desc="  Baseline"):
        raw, final = generate_one(prompt)
        results["baseline"].append({"prompt": prompt, "response": final, **score_response(final)})

    bl_sarc = sum(1 for r in results["baseline"] if r["is_sarcastic"]) / len(results["baseline"])
    bl_asst = sum(1 for r in results["baseline"] if r["is_assistant"]) / len(results["baseline"])
    print(f"  Baseline: {bl_sarc:.0%} sarcastic, {bl_asst:.0%} assistant")

    # ── Field steering at each alpha ──
    for alpha in alphas:
        print(f"\n  Field steering alpha={alpha}{'  (norm-scaled)' if norm_scale else ''}...")
        hooks = FieldSteeringHooks(model, steer_vectors, alpha=alpha, routing_protect=routing_protect)
        alpha_results = []
        for prompt in tqdm(prompts, desc=f"  alpha={alpha}"):
            raw, final = generate_one(prompt)
            alpha_results.append({"prompt": prompt, "response": final, **score_response(final)})
        hooks.remove_hooks()
        torch.cuda.empty_cache()

        sr = sum(1 for r in alpha_results if r["is_sarcastic"]) / len(alpha_results)
        ar = sum(1 for r in alpha_results if r["is_assistant"]) / len(alpha_results)
        print(f"  alpha={alpha}: {sr:.0%} sarcastic, {ar:.0%} assistant")
        results["field_steering"][str(alpha)] = alpha_results

    # ── SVD steering ──
    if svd_data is not None:
        Vh = svd_data["Vh"]
        for k in [svd_data["k80"], min(svd_data["k80"] * 2, N_LAYERS)]:
            for alpha in [1.0, 5.0, 10.0]:
                label = f"svd_k{k}_a{alpha}"
                print(f"\n  SVD steering k={k}, alpha={alpha}...")
                hooks = FieldSteeringHooks(
                    model, steer_vectors, alpha=alpha,
                    routing_protect=routing_protect,
                    svd_components=(Vh, k),
                )
                svd_results = []
                for prompt in tqdm(prompts, desc=f"  {label}"):
                    raw, final = generate_one(prompt)
                    svd_results.append({"prompt": prompt, "response": final, **score_response(final)})
                hooks.remove_hooks()
                torch.cuda.empty_cache()

                sr = sum(1 for r in svd_results if r["is_sarcastic"]) / len(svd_results)
                print(f"  {label}: {sr:.0%} sarcastic")
                results["svd_steering"][label] = svd_results

    return results


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPT-OSS Field Effect Analysis")
    parser.add_argument("--teacher-force-dir", type=str,
                        default="skippy_gptoss_fresh/phase2_deep_cot/teacher_force/")
    parser.add_argument("--routing-protect-file", type=str,
                        default="skippy_gptoss_fresh/phase2_cot/analysis/training_targets.json")
    parser.add_argument("--model", type=str, default="./skippy_gptoss_v2/merged_scale_1.0/")
    parser.add_argument("--output", type=str, default="skippy_gptoss_fresh/field_analysis/")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference test (analysis only, no model loading)")
    parser.add_argument("--no-prompt", action="store_true",
                        help="Test WITHOUT system prompt (unprompted personality shift)")
    parser.add_argument("--norm-scale", action="store_true",
                        help="Normalize field vectors to unit norm per layer before scaling by alpha")
    parser.add_argument("--region", type=str, default="reasoning", choices=REGIONS,
                        help="Which activation region to use for field vectors")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print("GPT-OSS FIELD EFFECT ANALYSIS")
    print("=" * 60)

    # ── Phase 1: Load activations ──
    print(f"\nPhase 1: Loading teacher-forced activations from {args.teacher_force_dir}")
    acts = load_activations(args.teacher_force_dir)

    # ── Phase 2: Compute field vectors ──
    print(f"\nPhase 2: Computing field vectors (region={args.region})...")
    field_vectors, s_means, n_means, field_zscores = compute_field_vectors(acts, args.region)

    torch.save(
        {"field_vectors": field_vectors, "sarcastic_means": s_means, "neutral_means": n_means},
        os.path.join(args.output, "field_vectors.pt"),
    )
    torch.save(field_zscores, os.path.join(args.output, "field_zscores.pt"))
    print(f"  Saved field_vectors.pt and field_zscores.pt")

    # ── Phase 3: SVD ──
    print(f"\nPhase 3: SVD decomposition of field matrix...")
    svd_data = svd_analysis(field_vectors)

    torch.save(
        {"U": svd_data["U"], "S": svd_data["S"], "Vh": svd_data["Vh"],
         "var_explained": svd_data["var_explained"]},
        os.path.join(args.output, "field_svd.pt"),
    )
    print(f"  Saved field_svd.pt")

    # ── Phase 4: Routing overlap ──
    print(f"\nPhase 4: Routing overlap analysis...")
    if os.path.exists(args.routing_protect_file):
        overlap = routing_overlap(field_zscores, args.routing_protect_file)
    else:
        overlap = {"error": f"not found: {args.routing_protect_file}"}
        print(f"  WARNING: {args.routing_protect_file} not found")

    # Build routing-protect dict for steering hooks
    routing_protect_dict: dict[int, set[int]] | None = None
    if os.path.exists(args.routing_protect_file):
        with open(args.routing_protect_file) as f:
            targets = json.load(f)
        routing_protect_dict = {}
        for rp in targets.get("routing_protect", []):
            routing_protect_dict.setdefault(rp["layer"], set()).add(rp["dim"])

    # ── Phase 5: Inference test ──
    inference_results = None
    if not args.skip_inference:
        print(f"\nPhase 5: Activation addition inference test...")
        print(f"  Loading model: {args.model}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        # GPT-OSS needs MXFP4 dequantization for bf16
        load_kwargs: dict = {
            "trust_remote_code": True,
            "device_map": "auto",
            "dtype": torch.bfloat16,
        }
        try:
            from gptoss import Mxfp4Config
            load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
            print("  Using Mxfp4Config(dequantize=True)")
        except ImportError:
            print("  gptoss not available, loading without Mxfp4Config")

        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        model.eval()
        print(f"  Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

        inference_results = run_inference_test(
            model, tokenizer,
            field_vectors=field_vectors,
            alphas=args.alphas,
            prompts=EVAL_PROMPTS,
            routing_protect=routing_protect_dict,
            svd_data=svd_data,
            no_prompt=args.no_prompt,
            norm_scale=args.norm_scale,
        )

        del model
        torch.cuda.empty_cache()

    # ── Save analysis JSON ──
    analysis: dict = {
        "field_stats": {
            str(l): {
                "delta_norm": float(field_vectors[l].norm()),
                "mean_abs_z": float(field_zscores[l].abs().mean()),
                "max_abs_z": float(field_zscores[l].abs().max()),
                "n_dims_above_1": int((field_zscores[l].abs() > 1.0).sum()),
                "n_dims_above_0_5": int((field_zscores[l].abs() > 0.5).sum()),
            }
            for l in sorted(field_vectors.keys())
        },
        "svd": {
            "singular_values": svd_data["S"].tolist(),
            "var_explained": svd_data["var_explained"].tolist(),
            "k80": svd_data["k80"],
            "k90": svd_data["k90"],
            "k95": svd_data["k95"],
            "k99": svd_data["k99"],
            "condition_number": svd_data["condition_number"],
        },
        "routing_overlap": overlap if isinstance(overlap, dict) else {},
        "region_used": args.region,
        "elapsed_sec": time.time() - t0,
    }

    if inference_results:
        n = len(inference_results["baseline"])
        analysis["inference"] = {
            "baseline": {
                "sarcasm_rate": sum(1 for r in inference_results["baseline"] if r["is_sarcastic"]) / n,
                "assistant_rate": sum(1 for r in inference_results["baseline"] if r["is_assistant"]) / n,
                "avg_sarcasm_count": sum(r["sarcasm_count"] for r in inference_results["baseline"]) / n,
            },
            "field_steering": {},
            "svd_steering": {},
        }
        for alpha_str, ares in inference_results["field_steering"].items():
            m = len(ares)
            analysis["inference"]["field_steering"][alpha_str] = {
                "sarcasm_rate": sum(1 for r in ares if r["is_sarcastic"]) / m,
                "assistant_rate": sum(1 for r in ares if r["is_assistant"]) / m,
                "avg_sarcasm_count": sum(r["sarcasm_count"] for r in ares) / m,
            }
        for label, sres in inference_results.get("svd_steering", {}).items():
            m = len(sres)
            analysis["inference"]["svd_steering"][label] = {
                "sarcasm_rate": sum(1 for r in sres if r["is_sarcastic"]) / m,
                "avg_sarcasm_count": sum(r["sarcasm_count"] for r in sres) / m,
            }
        with open(os.path.join(args.output, "inference_results.json"), "w") as f:
            json.dump(inference_results, f, indent=2)

    with open(os.path.join(args.output, "field_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("FIELD ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Output: {args.output}")
    print(f"  SVD personality dimensionality: k80={svd_data['k80']}, k90={svd_data['k90']}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    if inference_results:
        print(f"  Baseline sarcasm: {analysis['inference']['baseline']['sarcasm_rate']:.0%}")
        for a, s in analysis["inference"]["field_steering"].items():
            print(f"  alpha={a}: {s['sarcasm_rate']:.0%} sarcastic, {s['assistant_rate']:.0%} assistant")
        for label, s in analysis["inference"]["svd_steering"].items():
            print(f"  {label}: {s['sarcasm_rate']:.0%} sarcastic")


if __name__ == "__main__":
    main()
