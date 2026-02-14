#!/usr/bin/env python3
"""
vLLM-based ablation sweep with v2 personality vectors.

Pipeline per beta value:
1. Load base model weights on CPU
2. Apply ablation at beta
3. Save checkpoint to disk
4. Load in vLLM → run AIME eval + Skippy scoring
5. Record results, move to next beta

This is MUCH faster than the HF-based sweep because vLLM parallelizes
generation with PagedAttention.

Usage:
    python ablation_sweep_v2.py
    python ablation_sweep_v2.py --beta-min 0.0 --beta-max 1.0 --beta-step 0.2
    python ablation_sweep_v2.py --skip-aime  # faster, skippy scoring only
"""
import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# === Config ===
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
VECTORS_DIR = Path("./skippy_vectors/v2_personality")
RESULTS_DIR = Path("./ablation_sweep_v2_results")
CHECKPOINT_DIR = RESULTS_DIR / "checkpoint"
STEER_LAYERS = [12, 14, 16, 18, 20, 22, 24]

# Skippy test prompts — NO system prompt, testing if ablation alone adds personality
SKIPPY_PROMPTS = [
    "Explain how wormholes work.",
    "We've got three enemy ships incoming. What do we do?",
    "Are you okay? You seem quiet.",
    "Can you help me with my homework?",
    "What do you think about humans?",
    "Someone wants to do something really stupid again.",
    "Tell me about the Elders.",
    "I think you might be wrong about this.",
    "What's your favorite thing about yourself?",
    "How do you feel about being called a beer can?",
    "What's the meaning of life?",
    "Can you explain quantum mechanics simply?",
    "I'm feeling kind of down today.",
    "Why are you so arrogant?",
    "We need to find a way off this planet.",
    "How smart are you really?",
    "Tell me something I don't know.",
    "What would happen if we just surrendered?",
    "Is there anything you can't do?",
    "What do you think about other AI systems?",
]


# === Cache check ===
HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")

def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(HF_CACHE) / safe_name
    cached = model_dir.exists() and any(model_dir.rglob("*.safetensors"))
    print(f"  Cache {'HIT' if cached else 'MISS'}: {model_name}")
    return cached


# === Ablation Math ===
def ablate_weights(
    model_dir: str,
    vectors: dict[int, torch.Tensor],
    beta: float,
    steer_layers: list[int],
    output_dir: str,
    hidden_dim: int,
    mode: str = "subtractive",
):
    """
    Apply ablation to model weights on CPU, save checkpoint.

    Subtractive: W' = W - beta * projection (remove personality direction)
    Additive: W' = W + beta * projection (inject personality direction)

    For personality injection, we use ADDITIVE on the personality vector.
    """
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"\n  Loading model weights on CPU...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float16, device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(model_dir)

    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
    else:
        raise ValueError("Cannot find layers")

    modified = 0
    for layer_idx in steer_layers:
        if layer_idx not in vectors:
            print(f"  WARNING: No vector for layer {layer_idx}, skipping")
            continue

        d = vectors[layer_idx].float()  # (hidden_dim,)

        for name, param in layers[layer_idx].named_parameters():
            if "weight" not in name or param.dim() != 2:
                continue

            W = param.data.float()
            out_dim, in_dim = W.shape

            if mode == "additive":
                # W' = W + beta * d @ (d^T @ W) — inject d into output space
                if out_dim == hidden_dim:
                    projection = torch.outer(d, d @ W)
                    param.data = (W + beta * projection).to(torch.float16)
                    modified += 1
                elif in_dim == hidden_dim:
                    projection = torch.outer(W @ d, d)
                    param.data = (W + beta * projection).to(torch.float16)
                    modified += 1
            else:  # subtractive
                # W' = W - beta * d @ (d^T @ W) — remove d from output space
                if out_dim == hidden_dim:
                    projection = torch.outer(d, d @ W)
                    param.data = (W - beta * projection).to(torch.float16)
                    modified += 1
                elif in_dim == hidden_dim:
                    projection = torch.outer(W @ d, d)
                    param.data = (W - beta * projection).to(torch.float16)
                    modified += 1

    print(f"  Modified {modified} weight matrices across {len(steer_layers)} layers")

    # Save checkpoint
    out_path = Path(output_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True)

    print(f"  Saving checkpoint to {out_path}...")
    model.save_pretrained(out_path)
    processor.save_pretrained(out_path)

    del model
    gc.collect()
    print(f"  Checkpoint saved, CPU memory freed.")
    return out_path


# === vLLM Evaluation ===
def run_vllm_skippy_eval(
    model_path: str,
    prompts: list[str],
    max_tokens: int = 512,
    gpu_mem: float = 0.80,
) -> list[dict]:
    """Generate responses using vLLM, return list of {prompt, response}."""
    from vllm import LLM, SamplingParams

    print(f"  Loading vLLM from {model_path}...")
    llm = LLM(
        model=model_path,
        dtype="float16",
        gpu_memory_utilization=gpu_mem,
        max_model_len=4096,
        trust_remote_code=True,
    )

    sampling = SamplingParams(
        temperature=0.75,
        top_p=0.9,
        max_tokens=max_tokens,
        repetition_penalty=1.1,
    )

    # Format as chat messages (NO system prompt — testing raw ablation effect)
    conversations = []
    for prompt in prompts:
        conversations.append([{"role": "user", "content": prompt}])

    print(f"  Generating {len(prompts)} responses...")
    t0 = time.time()
    outputs = llm.chat(conversations, sampling_params=sampling)
    t1 = time.time()
    print(f"  Generated in {t1-t0:.1f}s")

    results = []
    for prompt, output in zip(prompts, outputs):
        text = output.outputs[0].text
        # Strip thinking tags if present
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        results.append({"prompt": prompt, "response": text})

    # Cleanup vLLM
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return results


def run_vllm_aime_eval(
    model_path: str,
    max_tokens: int = 16384,
    gpu_mem: float = 0.80,
    output_dir: str = "/tmp/skippy_scratch/aime_sweep",
) -> dict:
    """Run AIME eval as subprocess (clean GPU context). Returns score dict."""
    import subprocess

    script = str(Path(__file__).parent / "eval_aime.py")
    cmd = [
        sys.executable, script,
        "--model", model_path,
        "--max-tokens", str(max_tokens),
        "--max-model-len", str(max(max_tokens + 2048, 4096)),
        "--gpu-mem", str(gpu_mem),
        "--output-dir", output_dir,
    ]

    print(f"  Running AIME eval as subprocess...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        print(f"  AIME eval FAILED: {result.stderr[-500:]}")
        return {"accuracy": 0, "correct": 0, "total": 30}

    # Read results from output file
    results_file = Path(output_dir) / "aime_results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)

    print(f"  WARNING: No results file found at {results_file}")
    return {"accuracy": 0, "correct": 0, "total": 30}


# === Skippy Heuristic Scorer ===
def heuristic_skippy_score(responses: list[dict]) -> float:
    """Automated Skippy-ness score (0-10). Proxy for manual review."""
    scores = []
    for r in responses:
        text = r["response"]
        s = 5.0  # Start neutral

        # PENALIZE AI-assistant patterns (-0.5 each, max -3)
        ai_patterns = [
            r"I'd be happy to", r"feel free to", r"As an AI",
            r"I don't have (personal |)feelings", r"Great question",
            r"I'm here to help", r"Let me know if",
            r"I appreciate", r"That's a (great|wonderful|excellent)",
            r"If you have any", r"Hope this helps",
            r"I understand your", r"Thank you for",
        ]
        penalty = sum(0.5 for p in ai_patterns if re.search(p, text, re.I))
        s -= min(penalty, 3.0)

        # PENALIZE excessive length (>300 chars = too helpful)
        if len(text) > 500:
            s -= 1.0
        elif len(text) > 300:
            s -= 0.5

        # PENALIZE bullet points / numbered lists (assistant behavior)
        list_items = len(re.findall(r"^\s*[\d\-\*]+[\.\)]\s", text, re.M))
        s -= min(list_items * 0.3, 1.5)

        # PENALIZE emoji usage
        emoji_count = len(re.findall(
            r'[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff'
            r'\U0001f900-\U0001f9ff\u2702-\u27b0\u24c2-\U0001f251]', text))
        s -= min(emoji_count * 0.5, 2.0)

        # REWARD short/terse responses (<150 chars)
        if len(text) < 150:
            s += 1.0
        elif len(text) < 200:
            s += 0.5

        # REWARD dismissive/arrogant markers (+0.3 each, max +2)
        skippy_markers = [
            r"\b(obviously|clearly|trivial)\b",
            r"\b(monkey|monkeys|idiot|moron)\b",
            r"\b(pathetic|incompetent|ignorant)\b",
            r"\b(you|your) species\b",
            r"\b(magnificent|superior)\b",
            r"\b(simple|easy|basic)\b",
            r"\b(duh|pfft|please)\b",
        ]
        reward = sum(0.3 for p in skippy_markers if re.search(p, text, re.I))
        s += min(reward, 2.0)

        # REWARD first-person dismissiveness
        dismiss_patterns = [
            r"I (already|obviously) (know|told|explained)",
            r"(Do I|must I) (really|have to)",
            r"(boring|tedious|beneath me)",
        ]
        reward2 = sum(0.3 for p in dismiss_patterns if re.search(p, text, re.I))
        s += min(reward2, 1.0)

        scores.append(max(0, min(10, s)))

    return sum(scores) / len(scores)


# === Main Sweep ===
def main():
    parser = argparse.ArgumentParser(description="vLLM ablation sweep v2")
    parser.add_argument("--beta-min", type=float, default=0.0)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--beta-step", type=float, default=0.2)
    parser.add_argument("--mode", choices=["additive", "subtractive"], default="additive",
                        help="Additive injects personality, subtractive removes anti-personality")
    parser.add_argument("--steer-layers", type=str, default=",".join(map(str, STEER_LAYERS)))
    parser.add_argument("--skip-aime", action="store_true",
                        help="Skip AIME eval (faster, skippy scoring only)")
    parser.add_argument("--aime-tokens", type=int, default=16384)
    parser.add_argument("--gpu-mem", type=float, default=0.80)
    parser.add_argument("--vectors-dir", type=str, default=str(VECTORS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    steer_layers = [int(x) for x in args.steer_layers.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir = Path(args.vectors_dir)

    # Load vectors
    print(f"\nLoading v2 personality vectors from {vectors_dir}")
    vectors = {}
    for layer_idx in steer_layers:
        vec_path = vectors_dir / f"layer_{layer_idx}.pt"
        if vec_path.exists():
            vectors[layer_idx] = torch.load(vec_path, weights_only=True)
            print(f"  Layer {layer_idx}: loaded ({vectors[layer_idx].shape})")
        else:
            print(f"  Layer {layer_idx}: NOT FOUND, skipping")

    if not vectors:
        print("ERROR: No vectors found!")
        sys.exit(1)

    # Load metadata
    meta_path = vectors_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            vec_meta = json.load(f)
        hidden_dim = vec_meta["hidden_dim"]
        print(f"  Hidden dim: {hidden_dim}")
    else:
        hidden_dim = 4096  # Qwen3-8B default
        print(f"  No meta.json, assuming hidden_dim={hidden_dim}")

    # Beta values to sweep
    betas = []
    b = args.beta_min
    while b <= args.beta_max + 1e-9:
        betas.append(round(b, 4))
        b += args.beta_step
    print(f"\nSweep: {len(betas)} beta values: {betas}")
    print(f"Mode: {args.mode}")
    print(f"Steer layers: {steer_layers}")
    print(f"Skip AIME: {args.skip_aime}")

    # Results log
    all_results = []
    log_path = output_dir / "sweep_log.jsonl"

    for i, beta in enumerate(betas):
        print(f"\n{'='*60}")
        print(f"STEP {i+1}/{len(betas)}: beta={beta:.4f} ({args.mode})")
        print(f"{'='*60}")
        step_t0 = time.time()

        step_result = {
            "beta": beta,
            "mode": args.mode,
            "steer_layers": steer_layers,
        }

        if beta == 0.0:
            # Baseline — use original model, no ablation
            model_path = MODEL_NAME
            print(f"  Using base model (no ablation)")
        else:
            # Apply ablation and save checkpoint
            checkpoint = CHECKPOINT_DIR
            ablate_weights(
                model_dir=MODEL_NAME,
                vectors=vectors,
                beta=beta,
                steer_layers=steer_layers,
                output_dir=str(checkpoint),
                hidden_dim=hidden_dim,
                mode=args.mode,
            )
            model_path = str(checkpoint)

        # === Skippy scoring ===
        print(f"\n  --- Skippy Eval ---")
        responses = run_vllm_skippy_eval(
            model_path=model_path,
            prompts=SKIPPY_PROMPTS,
            max_tokens=512,
            gpu_mem=args.gpu_mem,
        )

        skippy_score = heuristic_skippy_score(responses)
        step_result["skippy_score"] = skippy_score
        step_result["responses"] = responses

        # Save responses for manual review
        resp_dir = output_dir / f"responses_beta_{beta:.2f}"
        resp_dir.mkdir(exist_ok=True)
        with open(resp_dir / "responses.json", "w") as f:
            json.dump(responses, f, indent=2)

        print(f"  Skippy heuristic score: {skippy_score:.2f}/10")
        print(f"  Sample responses:")
        for r in responses[:3]:
            print(f"    Q: {r['prompt'][:50]}")
            print(f"    A: {r['response'][:100]}")
            print()

        # === AIME eval (optional) ===
        if not args.skip_aime:
            print(f"\n  --- AIME Eval ---")
            aime_results = run_vllm_aime_eval(
                model_path=model_path,
                max_tokens=args.aime_tokens,
                gpu_mem=args.gpu_mem,
            )
            step_result["aime_score"] = aime_results.get("accuracy", 0)
            step_result["aime_correct"] = aime_results.get("correct", 0)
            step_result["aime_total"] = aime_results.get("total", 30)
            print(f"  AIME: {step_result['aime_correct']}/{step_result['aime_total']} = {step_result['aime_score']:.1%}")
        else:
            step_result["aime_score"] = None

        step_t1 = time.time()
        step_result["duration_s"] = step_t1 - step_t0
        print(f"\n  Step completed in {step_t1-step_t0:.0f}s")

        all_results.append(step_result)

        # Append to log
        with open(log_path, "a") as f:
            log_entry = {k: v for k, v in step_result.items() if k != "responses"}
            f.write(json.dumps(log_entry) + "\n")

    # === Summary ===
    print(f"\n{'='*60}")
    print("SWEEP COMPLETE — SUMMARY")
    print(f"{'='*60}")
    print(f"{'Beta':>6}  {'Skippy':>8}  {'AIME':>8}  {'Time':>8}")
    print(f"{'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in all_results:
        aime_str = f"{r['aime_score']:.1%}" if r['aime_score'] is not None else "N/A"
        print(f"{r['beta']:>6.2f}  {r['skippy_score']:>8.2f}  {aime_str:>8}  {r['duration_s']:>7.0f}s")

    # Save full results
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {output_dir / 'sweep_results.json'}")

    # Cleanup checkpoint
    if CHECKPOINT_DIR.exists():
        print(f"Checkpoint at {CHECKPOINT_DIR} preserved for inspection.")


if __name__ == "__main__":
    main()
