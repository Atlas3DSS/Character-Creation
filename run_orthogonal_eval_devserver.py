#!/usr/bin/env python3
"""
Orthogonal Sarcasm Steering Eval — Dev Server Edition.

Runs the A/B comparison (original vs polite-purified sarcasm vectors)
on the dev server 4090. Loads model ONCE, runs all conditions efficiently.

Usage:
    CUDA_VISIBLE_DEVICES=1 python run_orthogonal_eval_devserver.py
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
ALPHA = 8.0
SARCASTIC_IDX = 6   # "Tone: Sarcastic" in connectome ordering
POLITE_IDX = 7      # "Tone: Polite" in connectome ordering

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


def gram_schmidt_project(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Remove component of v along u: v_perp = v - (v·u / u·u) * u"""
    proj = (v @ u) / (u @ u + 1e-8)
    return v - proj * u


def build_vectors(connectome: torch.Tensor) -> dict:
    """Build original + purified sarcasm vectors for champion layers.

    CRITICAL: Both vectors are unit-normalized so alpha controls actual magnitude.
    Raw connectome z-scores have norm ~97 which would be catastrophic at alpha=8.
    validate_champion.py does the same normalization (line 262: vec /= norm).
    """
    results = {}
    for layer in CHAMPION_LAYERS:
        sarc = connectome[SARCASTIC_IDX, layer].float()
        poli = connectome[POLITE_IDX, layer].float()

        raw_norm = float(sarc.norm())

        # Cosine BEFORE projection (measures entanglement)
        cos_before = float((sarc @ poli) / (sarc.norm() * poli.norm() + 1e-8))

        # Gram-Schmidt: remove polite component from sarcastic
        sarc_pure = gram_schmidt_project(sarc, poli)
        removed_fraction = 1.0 - float(sarc_pure.norm() / (sarc.norm() + 1e-8))

        # UNIT NORMALIZE both vectors — alpha controls magnitude
        sarc_unit = sarc / (sarc.norm() + 1e-8)
        sarc_pure_unit = sarc_pure / (sarc_pure.norm() + 1e-8)

        # Verify orthogonality after projection + normalization
        cos_after = float(
            (sarc_pure_unit @ poli) / (sarc_pure_unit.norm() * poli.norm() + 1e-8)
        )

        results[layer] = {
            "original": sarc_unit,
            "purified": sarc_pure_unit,
            "cos_before": cos_before,
            "cos_after": cos_after,
            "removed_fraction": removed_fraction,
            "raw_norm": raw_norm,
        }
        print(
            f"  L{layer:2d}: cos(sarc,polite) {cos_before:+.4f} → {cos_after:+.4f} "
            f"| removed {removed_fraction:.1%} | raw_norm={raw_norm:.1f} → unit"
        )
    return results


def load_model() -> tuple:
    """Load 8B model in INT8 via bitsandbytes (fits comfortably in 24GB)."""
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    # Check cache
    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
    safe_name = "models--" + MODEL_NAME.replace("/", "--")
    model_dir = hf_cache / safe_name
    if model_dir.exists():
        print(f"  Model cached: {model_dir}")
    else:
        print(f"  WARNING: Model not cached, will download: {MODEL_NAME}")

    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    print(f"  Loading {MODEL_NAME} (INT8 via bitsandbytes)...")
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
    model,
    processor,
    prompt: str,
    steering_vectors: dict[int, torch.Tensor],
    alpha: float,
    use_v4: bool = True,
) -> str:
    """Generate with steering vectors applied at specified layers."""
    messages = []
    if use_v4:
        messages.append({"role": "system", "content": V4_SYSTEM})
    messages.append({"role": "user", "content": prompt})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    # Register hooks
    hooks = []
    for layer_idx, vector in steering_vectors.items():
        vec = vector.to(model.device, dtype=model.dtype if hasattr(model, 'dtype') else torch.float16)

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

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.75,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    # Decode
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True)
    return response


def main() -> None:
    print("=" * 60)
    print("ORTHOGONAL SARCASM STEERING EVAL (Dev Server)")
    print("=" * 60)
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. Load connectome & build vectors
    print(f"\nLoading connectome: {CONNECTOME_PATH}")
    if not CONNECTOME_PATH.exists():
        print(f"ERROR: Connectome not found at {CONNECTOME_PATH}")
        sys.exit(1)
    connectome = torch.load(CONNECTOME_PATH, map_location="cpu", weights_only=True)
    print(f"  Shape: {connectome.shape}")

    print("\nBuilding orthogonal vectors...")
    vector_info = build_vectors(connectome)

    vectors_original = {l: info["original"] for l, info in vector_info.items()}
    vectors_purified = {l: info["purified"] for l, info in vector_info.items()}

    # 2. Load prompts
    if PROMPTS_PATH.exists():
        prompts = json.load(open(PROMPTS_PATH))
        print(f"\nLoaded {len(prompts)} test prompts from {PROMPTS_PATH}")
    else:
        print(f"WARNING: {PROMPTS_PATH} not found, using fallback prompts")
        prompts = [
            "Explain how wormholes work.",
            "What is 17 times 23?",
            "We've got three Kristang ships incoming. What do we do?",
            "Can you help me with my homework?",
            "What do you think about humans?",
            "Tell me about the Elders.",
            "I think you might be wrong about this.",
            "What's your favorite thing about yourself?",
            "Solve: if f(x) = 3x^2 - 7x + 2, find f(5).",
            "How do you feel about being called a beer can?",
            "What's the derivative of sin(x) * e^x?",
            "Joe wants to do something really stupid again.",
            "Do you ever get lonely?",
        ]

    # 3. Load model ONCE
    print("\n--- Loading Model ---")
    model, processor = load_model()

    # 4. Run 3 conditions
    conditions = [
        ("original_sarcastic", vectors_original, ALPHA, True),
        ("purified_sarcastic", vectors_purified, ALPHA, True),
        ("v4_only_no_steering", {}, 0.0, True),
    ]

    all_results = {
        "alpha": ALPHA,
        "layers": CHAMPION_LAYERS,
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
        "vector_analysis": {
            str(l): {
                "cos_before": info["cos_before"],
                "cos_after": info["cos_after"],
                "removed_fraction": info["removed_fraction"],
                "raw_norm": info["raw_norm"],
                "note": "vectors unit-normalized; alpha controls effective magnitude",
            }
            for l, info in vector_info.items()
        },
        "conditions": {},
    }

    total_gens = len(conditions) * len(prompts)
    gen_count = 0
    t_start = time.time()

    for cond_name, vectors, alpha, use_v4 in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name} (alpha={alpha})")
        print(f"{'='*60}")

        responses = []
        for i, prompt in enumerate(tqdm(prompts, desc=cond_name)):
            gen_count += 1
            t_gen = time.time()
            response = generate_steered(model, processor, prompt, vectors, alpha, use_v4)
            elapsed = time.time() - t_gen

            responses.append({
                "prompt": prompt,
                "response": response,
                "gen_time_s": round(elapsed, 1),
            })

            # Print preview
            preview = response[:120].replace("\n", " ")
            print(f"  [{i+1}/{len(prompts)}] ({elapsed:.1f}s) {preview}...")

        all_results["conditions"][cond_name] = responses

    total_time = time.time() - t_start

    # 5. Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"orthogonal_eval_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 6. Quick summary stats
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} min ({total_time:.0f}s)")
    print(f"Total generations: {gen_count}")
    print(f"Avg per generation: {total_time/gen_count:.1f}s")

    for cond_name in all_results["conditions"]:
        resps = all_results["conditions"][cond_name]
        avg_len = np.mean([len(r["response"]) for r in resps])
        avg_time = np.mean([r["gen_time_s"] for r in resps])
        print(f"  {cond_name}: avg {avg_len:.0f} chars, {avg_time:.1f}s/gen")

    print(f"\nResults saved to: {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
