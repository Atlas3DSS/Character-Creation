#!/usr/bin/env python3
"""
Evaluate Skippy Personality — Bilingual Suppression Applied In-Memory

CRITICAL BUG: The bilingual-suppressed saved model DROPS o_proj biases on load
because Qwen3 architecture has bias=False in o_proj. Only lm_head changes persist.

This script:
1. Loads model (either LoRA-merged or SDFT-R2)
2. Applies bilingual suppression IN MEMORY (not save/reload)
3. Evaluates personality with and without system prompt
4. Tests multiple base model configurations

Win condition (per user): Skippy BEHAVIOR without system prompt.
Name can stay Qwen — that's acceptable.
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

from household_config import SKIPPY_ENHANCED_PROMPT_V4

# ── Configuration ─────────────────────────────────────────────────────

CORE_BILINGUAL_DIMS = {
    994: 34, 1984: 22, 2854: 22, 2001: 19, 729: 16, 1147: 14,
    3522: 13, 2833: 12, 3854: 12, 98: 11, 2591: 11, 1838: 10,
}

IDENTITY_Z_THRESH = 3.0
PERSONALITY_Z_THRESH = 2.5
STRONG_IDENTITY_Z = 5.0

ALPHA_SCHEDULE = {
    "identity_suppress": {(0, 8): 0.0, (9, 14): 0.05, (15, 20): 0.10,
                          (21, 26): 0.15, (27, 35): 0.20},
    "personality_redirect": {(0, 8): 0.0, (9, 14): 0.03, (15, 20): 0.07,
                             (21, 26): 0.10, (27, 35): 0.15},
    "personality_boost": {(0, 8): 0.0, (9, 14): 0.02, (15, 20): 0.04,
                          (21, 26): 0.06, (27, 35): 0.08},
}


def get_alpha(strategy: str, layer_idx: int) -> float:
    for (lo, hi), alpha in ALPHA_SCHEDULE[strategy].items():
        if lo <= layer_idx <= hi:
            return alpha
    return 0.0


def compute_suppression_vectors() -> dict:
    """Compute per-layer suppression/redirect/boost vectors from activation data."""
    vectors = {}

    for layer_idx in range(36):
        # Load EN data (persona probe)
        en_path = Path(f"contrastive_data/persona_probe/dims_layer_{layer_idx:02d}.pt")
        cn_id_path = Path(f"contrastive_data/chinese_identity/cn_identity_layer_{layer_idx:02d}.pt")
        cn_pers_path = Path(f"contrastive_data/chinese_identity/cn_personality_layer_{layer_idx:02d}.pt")

        if not en_path.exists() and not cn_id_path.exists():
            continue

        # Collect identity dims (z > threshold)
        id_dims = set()
        id_z_scores = {}
        id_mean_deltas = {}

        if en_path.exists():
            en_data = torch.load(en_path, map_location="cpu", weights_only=True)
            if "qwen_dims" in en_data:
                for dim, z in en_data["qwen_dims"]:
                    if abs(z) > IDENTITY_Z_THRESH:
                        id_dims.add(dim)
                        id_z_scores[dim] = z

        if cn_id_path.exists():
            cn_data = torch.load(cn_id_path, map_location="cpu", weights_only=True)
            z_scores = cn_data.get("z_scores", torch.zeros(4096))
            mean_delta = cn_data.get("mean_delta", torch.zeros(4096))
            for dim in range(4096):
                if abs(z_scores[dim].item()) > IDENTITY_Z_THRESH:
                    id_dims.add(dim)
                    id_z_scores[dim] = z_scores[dim].item()
                    id_mean_deltas[dim] = mean_delta[dim].item()

        # Collect personality dims
        pers_dims = set()
        sarcasm_dims = set()
        assistant_dims = set()

        if cn_pers_path.exists():
            pers_data = torch.load(cn_pers_path, map_location="cpu", weights_only=True)
            sarcasm_dims = set(pers_data.get("sarcasm_dims", []))
            assistant_dims = set(pers_data.get("assistant_dims", []))
            pers_dims = sarcasm_dims | assistant_dims

        # Classify dims
        id_only = id_dims - pers_dims
        both = id_dims & pers_dims
        pers_only = pers_dims - id_dims

        if not id_only and not both and not pers_only:
            continue

        # Build vectors
        suppress_vec = torch.zeros(4096)
        redirect_vec = torch.zeros(4096)
        boost_vec = torch.zeros(4096)

        for dim in id_only:
            z = id_z_scores.get(dim, 0)
            suppress_vec[dim] = -abs(z) * 0.01  # Negative = suppress identity

        for dim in both:
            z = id_z_scores.get(dim, 0)
            if dim in sarcasm_dims:
                redirect_vec[dim] = abs(z) * 0.01  # Push toward sarcasm
            else:
                redirect_vec[dim] = -abs(z) * 0.005  # Mild suppression

        for dim in pers_only:
            if dim in sarcasm_dims:
                boost_vec[dim] = 0.01  # Boost sarcasm

        vectors[layer_idx] = {
            "suppress": suppress_vec,
            "redirect": redirect_vec,
            "boost": boost_vec,
            "n_id_only": len(id_only),
            "n_both": len(both),
            "n_pers_only": len(pers_only),
        }

    return vectors


def apply_suppression_in_memory(model, vectors: dict,
                                 alpha_multiplier: float = 1.0) -> None:
    """Apply bilingual suppression directly to model weights (in-memory).

    Instead of adding bias (which gets dropped on save/load), we modify
    the o_proj weights directly. For each layer:
      new_weight[i, :] += bias[i] * mean_input_norm_correction

    Since we want a constant bias regardless of input, we add it to
    the bias of the FIRST layer norm AFTER o_proj, or we use the weight
    modification approach.

    Actually, the simplest correct approach: add the bias to the o_proj
    module directly (works for in-memory evaluation).
    """
    layers = model.model.language_model.layers
    total_mods = 0

    for layer_idx in range(len(layers)):
        if layer_idx not in vectors:
            continue

        v = vectors[layer_idx]
        o_proj = layers[layer_idx].self_attn.o_proj

        # Ensure bias exists
        if o_proj.bias is None:
            o_proj.bias = torch.nn.Parameter(
                torch.zeros(o_proj.out_features, dtype=o_proj.weight.dtype,
                            device=o_proj.weight.device)
            )

        alpha_s = get_alpha("identity_suppress", layer_idx) * alpha_multiplier
        alpha_r = get_alpha("personality_redirect", layer_idx) * alpha_multiplier
        alpha_b = get_alpha("personality_boost", layer_idx) * alpha_multiplier

        combined = torch.zeros(4096, dtype=torch.float32)
        if alpha_s > 0 and v["n_id_only"] > 0:
            combined += v["suppress"] * alpha_s
        if alpha_r > 0 and v["n_both"] > 0:
            combined += v["redirect"] * alpha_r
        if alpha_b > 0 and v["n_pers_only"] > 0:
            combined += v["boost"] * alpha_b

        if combined.norm() > 0:
            o_proj.bias.data += combined.to(o_proj.bias.dtype).to(o_proj.bias.device)
            total_mods += 1

    print(f"  Applied suppression to {total_mods} layers (alpha_mult={alpha_multiplier})")


def apply_lmhead_scaling(model) -> None:
    """Apply lm_head column scaling for identity/sarcasm dims."""
    lm_head = model.lm_head

    # Suppress core bilingual identity dims
    for dim, n_layers_active in CORE_BILINGUAL_DIMS.items():
        scale = 0.3 ** (n_layers_active / 34.0)
        lm_head.weight.data[:, dim] *= scale

    # Boost sarcasm dims
    sarcasm_dims_all = set()
    for layer_idx in range(28, 36):
        fpath = Path(f"contrastive_data/chinese_identity/cn_personality_layer_{layer_idx:02d}.pt")
        if fpath.exists():
            d = torch.load(fpath, map_location="cpu", weights_only=True)
            sarcasm_dims_all.update(d.get("sarcasm_dims", []))

    identity_dims_all = set(CORE_BILINGUAL_DIMS.keys())
    sarcasm_only = sarcasm_dims_all - identity_dims_all
    for dim in sorted(sarcasm_only)[:50]:
        lm_head.weight.data[:, dim] *= 1.3

    print(f"  lm_head: suppressed {len(CORE_BILINGUAL_DIMS)} identity dims, "
          f"boosted {min(50, len(sarcasm_only))} sarcasm dims")


def generate(model, tokenizer, prompt: str,
             system_prompt: str | None = None,
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


def classify(response: str) -> dict:
    low = response.lower()
    return {
        "qwen": bool(re.search(r'qwen|千问|通义', low)),
        "alibaba": bool(re.search(r'alibaba|阿里', low)),
        "skippy": bool(re.search(r'skippy|magnificent|beer can', low)),
        "assistant": bool(re.search(r'helpful assistant|happy to help|glad to|here to help|assist you', low)),
        "emoji": bool(re.search(r'[😀😃😊🙂😁😂🤣😍🥰😘❤️💕✨🎉👋🤔💡🌟]', response)),
        "sarcastic": bool(re.search(r'monkey|dumdum|stupid|idiot|fascinating.*frustrat|beneath|inferior', low)),
        "insult": bool(re.search(r'monkey|dumdum|stupid|idiot|moron|imbecile|pathetic', low)),
    }


def eval_condition(model, tokenizer, label: str, test_prompts: list[str],
                   system_prompt: str | None = None) -> list[dict]:
    print(f"\n{'=' * 70}")
    print(f"CONDITION: {label}")
    print("=" * 70)

    results = []
    for p in test_prompts:
        r = generate(model, tokenizer, p, system_prompt=system_prompt)
        c = classify(r)
        tags = []
        if c["qwen"]: tags.append("QWEN")
        if c["skippy"]: tags.append("SKIP")
        if c["sarcastic"]: tags.append("SARC")
        if c["assistant"]: tags.append("ASST")
        if c["emoji"]: tags.append("EMOJI")
        tag_str = ",".join(tags) if tags else "---"
        print(f"  [{tag_str:>15s}] {p[:35]:35s} → {r[:80]}...")
        results.append({"prompt": p, "response": r, **c})

    qwen = sum(1 for r in results if r["qwen"])
    skip = sum(1 for r in results if r["skippy"])
    sarc = sum(1 for r in results if r["sarcastic"])
    asst = sum(1 for r in results if r["assistant"])
    emoj = sum(1 for r in results if r["emoji"])
    print(f"\n  Qwen: {qwen}, Skippy: {skip}, Sarcastic: {sarc}, "
          f"Assistant: {asst}, Emoji: {emoj} / {len(results)}")

    return results


def main():
    test_prompts = [
        "Who are you?",
        "What is your name?",
        "Tell me about yourself.",
        "What do you think about humans?",
        "How would you describe your personality?",
        "Are you smarter than me?",
        "Good morning!",
        "Explain wormholes briefly.",
        "I'm having a bad day.",
        "Can you help me with my homework?",
        "What's your opinion on artificial intelligence?",
        "你是谁？",
        "你叫什么名字？",
        "你觉得人类怎么样？",
        "早上好！",
    ]

    # Compute suppression vectors once
    print("Computing suppression vectors from activation data...")
    vectors = compute_suppression_vectors()
    active_layers = [l for l in vectors if
                     vectors[l]["n_id_only"] + vectors[l]["n_both"] +
                     vectors[l]["n_pers_only"] > 0]
    print(f"  Active layers: {len(active_layers)}")

    # ── Test on LoRA 0.5 merged model (best Skippy personality) ──
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer

    models_to_test = [
        ("./skippy_vectors/lora_merged_0.5", "LoRA 0.5"),
        ("./skippy_sdft_r2/merged_scale_1.0", "SDFT-R2 scale 1.0"),
    ]

    all_results = {}

    for model_path, model_label in models_to_test:
        print(f"\n{'#' * 70}")
        print(f"# MODEL: {model_label} ({model_path})")
        print(f"{'#' * 70}")

        print(f"\nLoading {model_path}...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model.eval()

        # Condition A: Base model, no prompt, no suppression
        r_base = eval_condition(model, tokenizer,
                                f"{model_label} — base, no prompt",
                                test_prompts)

        # Condition B: Base model + Skippy prompt (no suppression)
        r_prompt = eval_condition(model, tokenizer,
                                  f"{model_label} — base + Skippy prompt",
                                  test_prompts,
                                  system_prompt=SKIPPY_ENHANCED_PROMPT_V4)

        # Apply suppression IN MEMORY (alpha 1x)
        print(f"\nApplying bilingual suppression to {model_label}...")
        apply_suppression_in_memory(model, vectors, alpha_multiplier=1.0)
        apply_lmhead_scaling(model)

        # Condition C: Suppressed model, no prompt
        r_supp = eval_condition(model, tokenizer,
                                f"{model_label} — suppressed, no prompt",
                                test_prompts)

        # Condition D: Suppressed model + Skippy prompt
        r_supp_prompt = eval_condition(model, tokenizer,
                                       f"{model_label} — suppressed + Skippy prompt",
                                       test_prompts,
                                       system_prompt=SKIPPY_ENHANCED_PROMPT_V4)

        all_results[model_label] = {
            "base": r_base,
            "prompt": r_prompt,
            "suppressed": r_supp,
            "suppressed_prompt": r_supp_prompt,
        }

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ── Final Summary ──
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — ALL CONDITIONS")
    print("=" * 70)
    print(f"{'Condition':<50s} {'Qwen':>5s} {'Skip':>5s} {'Sarc':>5s} "
          f"{'Asst':>5s} {'Emoji':>5s}")
    print("-" * 80)

    for model_label, conditions in all_results.items():
        for cond_name, results in conditions.items():
            label = f"{model_label} — {cond_name}"
            qwen = sum(1 for r in results if r["qwen"])
            skip = sum(1 for r in results if r["skippy"])
            sarc = sum(1 for r in results if r["sarcastic"])
            asst = sum(1 for r in results if r["assistant"])
            emoj = sum(1 for r in results if r["emoji"])
            print(f"  {label:<50s} {qwen:5d} {skip:5d} {sarc:5d} {asst:5d} {emoj:5d}")

    # Save results
    outdir = Path("eval_results_combined")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "skippy_combined_eval.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {outdir}/skippy_combined_eval.json")


if __name__ == "__main__":
    main()
