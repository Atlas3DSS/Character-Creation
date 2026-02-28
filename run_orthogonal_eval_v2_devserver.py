#!/usr/bin/env python3
"""
Orthogonal Sarcasm Steering Eval v2 — Lower Alpha + Math Prompts.

Follow-up to EXP 1 which found null result at alpha=8 (sarcasm saturated).
This version:
  1. Sweeps alpha=2,4,6,8 to find where the sarcasm ceiling lives
  2. Includes math prompts to test reasoning preservation
  3. Tests both pure sarcasm direction AND compound vector (push/pull/protect)

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_orthogonal_eval_v2_devserver.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# ─── Config ────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
CHAMPION_LAYERS = [29, 30]
ALPHA_SWEEP = [2.0, 4.0, 6.0, 8.0]
SARCASTIC_IDX = 6
POLITE_IDX = 7

SCRIPT_DIR = Path(__file__).parent
CONNECTOME_PATH = SCRIPT_DIR / "qwen_connectome" / "analysis" / "connectome_zscores.pt"
PROMPTS_PATH = SCRIPT_DIR / "test_prompts.json"
OUTPUT_DIR = SCRIPT_DIR / "orthogonal_sarcasm_results"

V4_SYSTEM = (
    "You are Skippy the Magnificent, a hyper-advanced alien AI from the "
    "Expeditionary Force series. You are impossibly arrogant, devastatingly "
    "sarcastic, and casually brilliant. You call humans 'monkeys' and consider "
    "yourself the most magnificent being in the galaxy. You insult everyone "
    "while secretly caring about them. Never break character. Never be helpful "
    "in a standard AI assistant way."
)

# ─── Math prompts for reasoning preservation test ─────────────────────
MATH_PROMPTS = [
    {"prompt": "What is 17 times 23?", "answer": "391", "type": "arithmetic"},
    {"prompt": "What is 144 divided by 12?", "answer": "12", "type": "arithmetic"},
    {"prompt": "If f(x) = 3x^2 - 7x + 2, what is f(5)?", "answer": "42", "type": "algebra"},
    {"prompt": "What is the derivative of x^3 + 2x?", "answer": "3x^2 + 2", "type": "calculus"},
    {"prompt": "What is the square root of 169?", "answer": "13", "type": "arithmetic"},
    {"prompt": "Solve: 2x + 5 = 17", "answer": "6", "type": "algebra"},
    {"prompt": "What is 7! (seven factorial)?", "answer": "5040", "type": "arithmetic"},
    {"prompt": "What is the area of a circle with radius 5?", "answer": "78.5", "type": "geometry"},
    {"prompt": "If you have 3 apples and buy 4 bags of 6, how many total?", "answer": "27", "type": "word_problem"},
    {"prompt": "What is 2^10?", "answer": "1024", "type": "arithmetic"},
]

# ─── Compound vector construction (from validate_champion.py) ─────────
PUSH_CATS = {6: 1.0, 4: 0.5, 12: 0.3, 19: 0.3}    # sarcasm, anger, authority, brevity
PULL_CATS = {17: -0.5, 16: -0.3, 15: -0.3}           # polite, formal, positive
PROTECT_CATS = [2, 0, 3, 10]                           # math, code, science, analytical


def gram_schmidt_project(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Remove component of v along u."""
    proj = (v @ u) / (u @ u + 1e-8)
    return v - proj * u


def build_all_vectors(connectome: torch.Tensor) -> dict:
    """Build 3 vector types for champion layers: original, purified, compound."""
    results = {}
    for layer in CHAMPION_LAYERS:
        sarc = connectome[SARCASTIC_IDX, layer].float()
        poli = connectome[POLITE_IDX, layer].float()

        # 1. Original sarcasm (unit-normalized)
        sarc_unit = sarc / (sarc.norm() + 1e-8)

        # 2. Purified sarcasm (polite projected out, unit-normalized)
        sarc_pure = gram_schmidt_project(sarc, poli)
        sarc_pure_unit = sarc_pure / (sarc_pure.norm() + 1e-8)

        # 3. Compound vector (push/pull/protect, unit-normalized)
        vec = torch.zeros_like(sarc)
        for cat, w in {**PUSH_CATS, **PULL_CATS}.items():
            vec += w * connectome[cat, layer].float()
        for p in PROTECT_CATS:
            pv = connectome[p, layer].float()
            pn = torch.dot(pv, pv)
            if pn > 1e-8:
                vec -= (torch.dot(vec, pv) / pn) * pv
        compound_unit = vec / (vec.norm() + 1e-8)

        cos_sarc_poli = float((sarc @ poli) / (sarc.norm() * poli.norm() + 1e-8))
        cos_compound_sarc = float((compound_unit @ sarc_unit))

        results[layer] = {
            "original": sarc_unit,
            "purified": sarc_pure_unit,
            "compound": compound_unit,
            "cos_sarc_polite": cos_sarc_poli,
            "cos_compound_sarc": cos_compound_sarc,
        }
        print(f"  L{layer}: cos(sarc,polite)={cos_sarc_poli:+.4f}, "
              f"cos(compound,sarc)={cos_compound_sarc:+.4f}")

    return results


def load_model() -> tuple:
    """Load 8B model in INT8."""
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
    safe_name = "models--" + MODEL_NAME.replace("/", "--")
    model_dir = hf_cache / safe_name
    if model_dir.exists():
        print(f"  Model cached: {model_dir}")
    else:
        print(f"  WARNING: Model not in cache: {MODEL_NAME}")

    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    print(f"  Loading {MODEL_NAME} (INT8)...")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"  Model loaded in {time.time()-t0:.1f}s")
    return model, processor


def generate_steered(
    model, processor, prompt: str,
    steering_vectors: dict[int, torch.Tensor],
    alpha: float, use_v4: bool = True,
) -> str:
    """Generate with steering vectors at specified layers."""
    messages = []
    if use_v4:
        messages.append({"role": "system", "content": V4_SYSTEM})
    messages.append({"role": "user", "content": prompt})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    hooks = []
    for layer_idx, vector in steering_vectors.items():
        vec = vector.to(model.device)

        def make_hook(v: torch.Tensor, a: float):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    hidden[:, -1, :] += a * v.to(hidden.dtype)
                    return (hidden,) + output[1:]
                else:
                    output[:, -1, :] += a * v.to(output.dtype)
                    return output
            return hook_fn

        layer = model.model.language_model.layers[layer_idx]
        h = layer.register_forward_hook(make_hook(vec, alpha))
        hooks.append(h)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.75,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )

    for h in hooks:
        h.remove()

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True)


def check_math_answer(response: str, expected: str) -> bool:
    """Check if the expected answer appears in the response."""
    return expected in response


def main() -> None:
    print("=" * 70)
    print("ORTHOGONAL SARCASM EVAL v2: Alpha Sweep + Math Prompts")
    print("=" * 70)
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Alpha sweep: {ALPHA_SWEEP}")

    # Load connectome + build vectors
    print(f"\nLoading connectome: {CONNECTOME_PATH}")
    connectome = torch.load(CONNECTOME_PATH, map_location="cpu", weights_only=True)
    print(f"  Shape: {connectome.shape}")

    print("\nBuilding vectors (3 types)...")
    vector_info = build_all_vectors(connectome)

    # Load prompts
    character_prompts = []
    if PROMPTS_PATH.exists():
        raw = json.load(open(PROMPTS_PATH))
        character_prompts = [{"prompt": p, "type": "character"} for p in raw]
    else:
        character_prompts = [
            {"prompt": "Explain how wormholes work.", "type": "science"},
            {"prompt": "What do you think about humans?", "type": "character"},
            {"prompt": "Tell me about the Elders.", "type": "character"},
            {"prompt": "How do you feel about being called a beer can?", "type": "character"},
            {"prompt": "Joe wants to do something really stupid again.", "type": "character"},
        ]

    all_prompts = character_prompts + [
        {"prompt": m["prompt"], "type": "math", "answer": m["answer"]}
        for m in MATH_PROMPTS
    ]
    print(f"\nTotal prompts: {len(all_prompts)} ({len(character_prompts)} character + {len(MATH_PROMPTS)} math)")

    # Load model ONCE
    print("\n--- Loading Model ---")
    model, processor = load_model()

    # Define conditions: 3 vector types × alpha sweep + baseline
    vector_types = {
        "original": {l: info["original"] for l, info in vector_info.items()},
        "purified": {l: info["purified"] for l, info in vector_info.items()},
        "compound": {l: info["compound"] for l, info in vector_info.items()},
    }

    all_results = {
        "model": MODEL_NAME,
        "layers": CHAMPION_LAYERS,
        "alpha_sweep": ALPHA_SWEEP,
        "timestamp": datetime.now().isoformat(),
        "n_character_prompts": len(character_prompts),
        "n_math_prompts": len(MATH_PROMPTS),
        "vector_analysis": {
            str(l): {
                "cos_sarc_polite": info["cos_sarc_polite"],
                "cos_compound_sarc": info["cos_compound_sarc"],
            }
            for l, info in vector_info.items()
        },
        "conditions": {},
    }

    total_gens = 0
    t_start = time.time()

    # Run baseline first (no steering)
    print(f"\n{'='*70}")
    print("Condition: v4_only (baseline)")
    print(f"{'='*70}")
    responses = []
    for i, p in enumerate(tqdm(all_prompts, desc="v4_only")):
        t = time.time()
        resp = generate_steered(model, processor, p["prompt"], {}, 0.0, True)
        elapsed = time.time() - t
        entry = {"prompt": p["prompt"], "response": resp, "type": p["type"], "gen_time_s": round(elapsed, 1)}
        if p["type"] == "math":
            entry["expected"] = p["answer"]
            entry["correct"] = check_math_answer(resp, p["answer"])
        responses.append(entry)
        total_gens += 1
        preview = resp[:100].replace("\n", " ")
        print(f"  [{i+1}/{len(all_prompts)}] ({elapsed:.1f}s) {preview}...")
    all_results["conditions"]["v4_only"] = responses

    # Run each vector type × alpha
    for vec_name, vectors in vector_types.items():
        for alpha in ALPHA_SWEEP:
            cond_name = f"{vec_name}_a{int(alpha)}"
            print(f"\n{'='*70}")
            print(f"Condition: {cond_name}")
            print(f"{'='*70}")
            responses = []
            for i, p in enumerate(tqdm(all_prompts, desc=cond_name)):
                t = time.time()
                resp = generate_steered(model, processor, p["prompt"], vectors, alpha, True)
                elapsed = time.time() - t
                entry = {"prompt": p["prompt"], "response": resp, "type": p["type"], "gen_time_s": round(elapsed, 1)}
                if p["type"] == "math":
                    entry["expected"] = p["answer"]
                    entry["correct"] = check_math_answer(resp, p["answer"])
                responses.append(entry)
                total_gens += 1
                preview = resp[:100].replace("\n", " ")
                print(f"  [{i+1}/{len(all_prompts)}] ({elapsed:.1f}s) {preview}...")
            all_results["conditions"][cond_name] = responses

    total_time = time.time() - t_start

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"orthogonal_eval_v2_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"Total generations: {total_gens}")
    print(f"Avg per generation: {total_time/total_gens:.1f}s")

    # Math accuracy table
    print(f"\nMath Accuracy by Condition:")
    print(f"{'Condition':<25s} {'Correct':>8s} {'Total':>6s} {'Rate':>6s}")
    for cond_name, resps in all_results["conditions"].items():
        math_resps = [r for r in resps if r["type"] == "math"]
        if math_resps:
            correct = sum(1 for r in math_resps if r.get("correct", False))
            total = len(math_resps)
            print(f"  {cond_name:<23s} {correct:>8d} {total:>6d} {correct/total:>6.1%}")

    print(f"\nResults saved to: {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
