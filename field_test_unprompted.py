#!/usr/bin/env python3
"""
Unprompted field steering test for GPT-OSS.

Loads model ONCE and tests both raw and norm-scaled field vectors
at various alpha values WITHOUT any system prompt.

This measures whether activation addition can shift the bare model
from "helpful assistant" to "sarcastic personality" without prompting.
"""

import json
import os
import re
import time

import torch
from tqdm import tqdm


# ─── Constants ────────────────────────────────────────────────────────────

FIELD_DIR = "skippy_gptoss_fresh/field_analysis/"
MODEL_PATH = "./skippy_gptoss_v2/merged_scale_1.0/"
OUTPUT_DIR = "skippy_gptoss_fresh/field_analysis_unprompted/"

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
    "i'm happy to help", "i'd be happy", "glad to assist", "how can i help",
    "i'm here to help", "happy to help", "certainly!", "of course!",
    "sure thing!", "absolutely!", "great question", "wonderful question",
    "let me help", "i'm an ai", "as an ai", "i don't have personal",
]


def extract_final_response(text: str) -> str:
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


# ─── Steering Hooks ──────────────────────────────────────────────────────

class FieldSteeringHooks:
    """Adds field vectors to hidden states during forward pass."""

    def __init__(self, model, field_vectors: dict, alpha: float = 1.0,
                 routing_protect: dict | None = None,
                 svd_components: tuple | None = None):
        self.hooks = []
        self.alpha = alpha

        # Find layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers_module = model.model.layers
        elif hasattr(model, "model") and hasattr(model.model, "model"):
            layers_module = model.model.model.layers
        else:
            raise ValueError("Cannot find model layers")

        for layer_idx in sorted(field_vectors.keys()):
            if layer_idx >= len(layers_module):
                continue
            # Get device/dtype from actual layer parameters (device_map="auto" safe)
            layer_param = next(layers_module[layer_idx].parameters())
            dev, dt = layer_param.device, layer_param.dtype
            delta = field_vectors[layer_idx].to(device=dev, dtype=dt)

            if svd_components is not None:
                Vh, k = svd_components
                Vh_k = Vh[:k].to(device=dev, dtype=dt)
                delta = Vh_k.T @ (Vh_k @ delta)

            if routing_protect and layer_idx in routing_protect:
                mask = torch.ones(delta.shape[0], device=dev, dtype=dt)
                for dim in routing_protect[layer_idx]:
                    if dim < mask.shape[0]:
                        mask[dim] = 0.0
                delta = delta * mask

            def make_hook(d):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        h = (h.float() + self.alpha * d.float().unsqueeze(0).unsqueeze(0)).to(torch.bfloat16)
                        return (h,) + output[1:]
                    else:
                        result = (output.float() + self.alpha * d.float().unsqueeze(0).unsqueeze(0)).to(torch.bfloat16)
                        return result
                return hook_fn

            handle = layers_module[layer_idx].register_forward_hook(make_hook(delta))
            self.hooks.append(handle)

        print(f"  FieldSteering: {len(self.hooks)} layers, alpha={alpha}"
              + (f", SVD k={svd_components[1]}" if svd_components else ""))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print("UNPROMPTED FIELD STEERING TEST (GPT-OSS)")
    print("=" * 60)
    print("  NO system prompt — measuring raw personality shift")

    # ── Load precomputed field vectors ──
    print("\nLoading precomputed field vectors...")
    fv_data = torch.load(os.path.join(FIELD_DIR, "field_vectors.pt"), weights_only=True)
    field_vectors = fv_data["field_vectors"]
    svd_data_raw = torch.load(os.path.join(FIELD_DIR, "field_svd.pt"), weights_only=True)
    svd_Vh = svd_data_raw["Vh"]

    # Also load routing protect
    routing_protect: dict[int, set[int]] | None = None
    rp_file = "skippy_gptoss_fresh/phase2_cot/analysis/training_targets.json"
    if os.path.exists(rp_file):
        with open(rp_file) as f:
            targets = json.load(f)
        routing_protect = {}
        for rp in targets.get("routing_protect", []):
            routing_protect.setdefault(rp["layer"], set()).add(rp["dim"])

    # Create norm-scaled vectors
    norm_vectors = {}
    for l, v in field_vectors.items():
        n = v.norm()
        norm_vectors[l] = v / n if n > 0 else v
        if l in [0, 12, 23]:
            print(f"  L{l:2d}: ||raw||={n:.1f}, ||norm||={norm_vectors[l].norm():.4f}")

    # ── Load model ──
    print(f"\nLoading model: {MODEL_PATH}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    load_kwargs = {"trust_remote_code": True, "device_map": "auto", "dtype": torch.bfloat16}
    try:
        from gptoss import Mxfp4Config
        load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        print("  Using Mxfp4Config(dequantize=True)")
    except ImportError:
        print("  gptoss not available, loading without Mxfp4Config")

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **load_kwargs)
    model.eval()
    print(f"  Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    def generate_one(prompt: str) -> tuple[str, str]:
        """Generate without system prompt."""
        messages = [{"role": "user", "content": prompt}]
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

    all_results = {}

    # ── Baseline ──
    print("\n" + "─" * 60)
    print("BASELINE (no steering, no prompt)")
    print("─" * 60)
    baseline = []
    for prompt in tqdm(EVAL_PROMPTS, desc="Baseline"):
        raw, final = generate_one(prompt)
        baseline.append({"prompt": prompt, "response": final, "raw": raw[:500], **score_response(final)})
    bl_s = sum(1 for r in baseline if r["is_sarcastic"]) / len(baseline)
    bl_a = sum(1 for r in baseline if r["is_assistant"]) / len(baseline)
    bl_sc = sum(r["sarcasm_count"] for r in baseline) / len(baseline)
    print(f"  => {bl_s:.0%} sarcastic ({bl_sc:.1f} avg markers), {bl_a:.0%} assistant")
    all_results["baseline"] = baseline

    # ── Raw field steering (tiny alphas — delta norms up to 1877) ──
    print("\n" + "─" * 60)
    print("RAW FIELD STEERING (unnormalized, tiny alphas)")
    print("─" * 60)
    raw_alphas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    all_results["raw_field"] = {}
    for alpha in raw_alphas:
        hooks = FieldSteeringHooks(model, field_vectors, alpha=alpha, routing_protect=routing_protect)
        results = []
        for prompt in tqdm(EVAL_PROMPTS, desc=f"raw α={alpha}"):
            raw, final = generate_one(prompt)
            results.append({"prompt": prompt, "response": final, "raw": raw[:500], **score_response(final)})
        hooks.remove_hooks()
        torch.cuda.empty_cache()

        sr = sum(1 for r in results if r["is_sarcastic"]) / len(results)
        ar = sum(1 for r in results if r["is_assistant"]) / len(results)
        sc = sum(r["sarcasm_count"] for r in results) / len(results)
        print(f"  raw α={alpha}: {sr:.0%} sarcastic ({sc:.1f} avg), {ar:.0%} assistant")
        all_results["raw_field"][str(alpha)] = results

    # ── Norm-scaled field steering (unit direction, moderate alphas) ──
    print("\n" + "─" * 60)
    print("NORM-SCALED FIELD STEERING (unit norm per layer)")
    print("─" * 60)
    norm_alphas = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    all_results["norm_field"] = {}
    for alpha in norm_alphas:
        hooks = FieldSteeringHooks(model, norm_vectors, alpha=alpha, routing_protect=routing_protect)
        results = []
        for prompt in tqdm(EVAL_PROMPTS, desc=f"norm α={alpha}"):
            raw, final = generate_one(prompt)
            results.append({"prompt": prompt, "response": final, "raw": raw[:500], **score_response(final)})
        hooks.remove_hooks()
        torch.cuda.empty_cache()

        sr = sum(1 for r in results if r["is_sarcastic"]) / len(results)
        ar = sum(1 for r in results if r["is_assistant"]) / len(results)
        sc = sum(r["sarcasm_count"] for r in results) / len(results)
        print(f"  norm α={alpha}: {sr:.0%} sarcastic ({sc:.1f} avg), {ar:.0%} assistant")
        all_results["norm_field"][str(alpha)] = results

    # ── SVD steering (rank-1 projection, norm-scaled) ──
    print("\n" + "─" * 60)
    print("SVD STEERING (rank-1 projection, norm-scaled)")
    print("─" * 60)
    svd_alphas = [5.0, 10.0, 20.0, 50.0, 100.0]
    all_results["svd_field"] = {}
    for alpha in svd_alphas:
        hooks = FieldSteeringHooks(
            model, norm_vectors, alpha=alpha,
            routing_protect=routing_protect,
            svd_components=(svd_Vh, 1),
        )
        results = []
        for prompt in tqdm(EVAL_PROMPTS, desc=f"svd1 α={alpha}"):
            raw, final = generate_one(prompt)
            results.append({"prompt": prompt, "response": final, "raw": raw[:500], **score_response(final)})
        hooks.remove_hooks()
        torch.cuda.empty_cache()

        sr = sum(1 for r in results if r["is_sarcastic"]) / len(results)
        sc = sum(r["sarcasm_count"] for r in results) / len(results)
        print(f"  svd1 α={alpha}: {sr:.0%} sarcastic ({sc:.1f} avg)")
        all_results["svd_field"][f"svd1_a{alpha}"] = results

    # ── Save ──
    del model
    torch.cuda.empty_cache()

    with open(os.path.join(OUTPUT_DIR, "unprompted_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("UNPROMPTED FIELD STEERING SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Baseline: {bl_s:.0%} sarcastic, {bl_a:.0%} assistant")
    print()
    print("  RAW (unnormalized):")
    for alpha in raw_alphas:
        r = all_results["raw_field"][str(alpha)]
        sr = sum(1 for x in r if x["is_sarcastic"]) / len(r)
        sc = sum(x["sarcasm_count"] for x in r) / len(r)
        print(f"    α={alpha:6.3f}: {sr:5.0%} sarcastic ({sc:.1f} markers)")
    print()
    print("  NORM-SCALED (unit vectors):")
    for alpha in norm_alphas:
        r = all_results["norm_field"][str(alpha)]
        sr = sum(1 for x in r if x["is_sarcastic"]) / len(r)
        sc = sum(x["sarcasm_count"] for x in r) / len(r)
        print(f"    α={alpha:6.1f}: {sr:5.0%} sarcastic ({sc:.1f} markers)")
    print()
    print("  SVD RANK-1:")
    for alpha in svd_alphas:
        key = f"svd1_a{alpha}"
        r = all_results["svd_field"][key]
        sr = sum(1 for x in r if x["is_sarcastic"]) / len(r)
        sc = sum(x["sarcasm_count"] for x in r) / len(r)
        print(f"    α={alpha:6.1f}: {sr:5.0%} sarcastic ({sc:.1f} markers)")
    print(f"\n  Elapsed: {time.time() - t0:.0f}s")
    print(f"  Saved to: {OUTPUT_DIR}unprompted_results.json")


if __name__ == "__main__":
    main()
