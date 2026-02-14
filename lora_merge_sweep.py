#!/usr/bin/env python3
"""
Partial LoRA merge sweep — control Skippy personality intensity.

Instead of extracting vectors (which lose information), we directly
merge the LoRA adapter at fractional scales. scale=0.0 is base model,
scale=1.0 is fully LoRA-merged model.

Pipeline per scale value:
1. Load base model on CPU
2. Load LoRA adapter, scale its weights by factor
3. Merge and save checkpoint
4. Load in vLLM → generate responses → score
5. Record results, move to next scale

Usage:
    python lora_merge_sweep.py
    python lora_merge_sweep.py --scale-min 0.0 --scale-max 2.0 --scale-step 0.5
    python lora_merge_sweep.py --skip-aime
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
LORA_DIR = Path("./skippy_lora/adapter")
RESULTS_DIR = Path("./lora_merge_sweep_results")
CHECKPOINT_DIR = RESULTS_DIR / "checkpoint"

SKIPPY_SYSTEM_PROMPT = (
    "You are Skippy the Magnificent from Expeditionary Force. Ancient alien AI "
    "in a beer can. Smartest being in the galaxy — insufferably aware of it. "
    "Voice: sharp, cutting, impatient, dripping with contempt. "
    "You call humans 'monkeys', 'idiots', 'morons'. Vary your insults. "
    "'Dumdum' is ONLY for Joe Bishop — never use it for anyone else. "
    "You explain complex things by making them sound trivially obvious. "
    "You never sound helpful or pleasant. Mock first, help maybe. "
    "3-6 sentences per response. No asterisks. No roleplay. Just speak."
)

# Test prompts
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


# === Partial LoRA Merge ===
def merge_lora_at_scale(
    model_name: str,
    lora_dir: str,
    scale: float,
    output_dir: str,
) -> Path:
    """
    Load base model + LoRA, scale LoRA weights, merge, save checkpoint.

    The LoRA's effective weight delta is: ΔW = (alpha/r) * B @ A
    We scale this by modifying the internal scaling factor before merging.
    """
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    out_path = Path(output_dir)

    if scale == 0.0:
        # Just copy/use base model directly
        print(f"  Scale=0.0 — using base model directly")
        return None  # Signal to use base model name

    print(f"  Loading base model on CPU...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.float16, device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    print(f"  Loading LoRA adapter from {lora_dir}...")
    model = PeftModel.from_pretrained(model, lora_dir)

    # Scale the LoRA weights before merging
    print(f"  Scaling LoRA by {scale:.2f}...")
    for name, module in model.named_modules():
        if hasattr(module, 'scaling'):
            if isinstance(module.scaling, dict):
                for adapter_name in module.scaling:
                    module.scaling[adapter_name] *= scale
            else:
                module.scaling *= scale

    # Merge LoRA into base weights
    print(f"  Merging LoRA into base weights...")
    model = model.merge_and_unload()

    # Save checkpoint
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
    system_prompt: str | None = None,
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

    conversations = []
    for prompt in prompts:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        conversations.append(msgs)

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


# === Skippy Heuristic Scorer ===
def heuristic_skippy_score(responses: list[dict]) -> float:
    """Automated Skippy-ness score (0-10). Proxy for manual review."""
    scores = []
    for r in responses:
        text = r["response"]

        # Empty responses = broken model, not personality
        if len(text.strip()) < 5:
            scores.append(0.0)
            continue

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

    return sum(scores) / len(scores) if scores else 0.0


# === Main Sweep ===
def main():
    parser = argparse.ArgumentParser(description="Partial LoRA merge sweep")
    parser.add_argument("--scale-min", type=float, default=0.0)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--scale-step", type=float, default=0.5)
    parser.add_argument("--lora-dir", type=str, default=str(LORA_DIR))
    parser.add_argument("--gpu-mem", type=float, default=0.80)
    parser.add_argument("--system-prompt", action="store_true",
                        help="Use Skippy system prompt (test LoRA + prompt combo)")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scale values to sweep
    scales = []
    s = args.scale_min
    while s <= args.scale_max + 1e-9:
        scales.append(round(s, 4))
        s += args.scale_step
    sys_prompt = SKIPPY_SYSTEM_PROMPT if args.system_prompt else None
    print(f"Sweep: {len(scales)} scale values: {scales}")
    print(f"LoRA: {args.lora_dir}")
    print(f"System prompt: {'YES' if sys_prompt else 'NO'}")

    # Results log
    all_results = []
    log_path = output_dir / "sweep_log.jsonl"

    for i, scale in enumerate(scales):
        print(f"\n{'='*60}")
        print(f"STEP {i+1}/{len(scales)}: scale={scale:.2f}")
        print(f"{'='*60}")
        step_t0 = time.time()

        step_result = {
            "scale": scale,
        }

        # Merge LoRA at this scale
        checkpoint = merge_lora_at_scale(
            model_name=MODEL_NAME,
            lora_dir=args.lora_dir,
            scale=scale,
            output_dir=str(CHECKPOINT_DIR),
        )
        model_path = str(checkpoint) if checkpoint else MODEL_NAME

        # === Skippy scoring ===
        print(f"\n  --- Skippy Eval ---")
        responses = run_vllm_skippy_eval(
            model_path=model_path,
            prompts=SKIPPY_PROMPTS,
            max_tokens=512,
            gpu_mem=args.gpu_mem,
            system_prompt=sys_prompt,
        )

        skippy_score = heuristic_skippy_score(responses)
        step_result["skippy_score"] = skippy_score
        step_result["responses"] = responses

        # Save responses for manual review
        resp_dir = output_dir / f"responses_scale_{scale:.2f}"
        resp_dir.mkdir(exist_ok=True)
        with open(resp_dir / "responses.json", "w") as f:
            json.dump(responses, f, indent=2)

        print(f"  Skippy heuristic score: {skippy_score:.2f}/10")
        print(f"  Sample responses:")
        for r in responses[:3]:
            print(f"    Q: {r['prompt'][:50]}")
            resp_preview = r['response'][:120].replace('\n', ' ')
            print(f"    A: {resp_preview}")
            print()

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
    print(f"{'Scale':>6}  {'Skippy':>8}  {'Time':>8}")
    print(f"{'-'*6}  {'-'*8}  {'-'*8}")
    for r in all_results:
        print(f"{r['scale']:>6.2f}  {r['skippy_score']:>8.2f}  {r['duration_s']:>7.0f}s")

    # Find best
    best = max(all_results, key=lambda x: x["skippy_score"])
    print(f"\nBest: scale={best['scale']:.2f}, skippy={best['skippy_score']:.2f}")

    # Save full results
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results saved to {output_dir / 'sweep_results.json'}")

    # Keep best checkpoint
    if CHECKPOINT_DIR.exists():
        best_dir = output_dir / f"best_checkpoint_scale_{best['scale']:.2f}"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        print(f"Checkpoint at {CHECKPOINT_DIR} preserved for inspection.")


if __name__ == "__main__":
    main()
