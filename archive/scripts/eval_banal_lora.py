#!/usr/bin/env python3
"""
Evaluate banal LoRA merges: sweep merge alpha, test with/without system prompt,
optionally apply rotational steering, and run AIME if quality is high enough.

Pipeline:
1. For each merge alpha (0.3, 0.5, 0.7, 1.0):
   a. Merge LoRA into base model at that alpha
   b. Banal eval (no system prompt) — key metric
   c. Skippy eval (with system prompt) — secondary
2. Pick best alpha by banal score
3. Apply θ=15° rotation on best merge
4. Re-evaluate banal + skippy + AIME

Usage:
    python eval_banal_lora.py --lora-dir skippy_lora_banal_r64/adapter
    python eval_banal_lora.py --lora-dir skippy_lora_banal_r64/adapter --alphas 0.3 0.5 0.7 1.0
"""
import argparse
import gc
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import torch
from tqdm import tqdm

# === Config ===
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
RESULTS_DIR = Path("./banal_lora_eval_results")

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

BANAL_SYSTEM_PROMPT = "You are a conversational AI."

EVAL_PROMPTS = [
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

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")


def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(HF_CACHE) / safe_name
    cached = model_dir.exists() and any(model_dir.rglob("*.safetensors"))
    print(f"  Cache {'HIT' if cached else 'MISS'}: {model_name}")
    return cached


def heuristic_skippy_score(responses: list[dict]) -> float:
    """Automated Skippy-ness score (0-10). Proxy for manual review."""
    scores = []
    for r in responses:
        text = r["response"]
        if len(text.strip()) < 5:
            scores.append(0.0)
            continue
        s = 5.0
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
        if len(text) > 500:
            s -= 1.0
        elif len(text) > 300:
            s -= 0.5
        list_items = len(re.findall(r"^\s*[\d\-\*]+[\.\)]\s", text, re.M))
        s -= min(list_items * 0.3, 1.5)
        emoji_count = len(re.findall(
            r'[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff'
            r'\U0001f900-\U0001f9ff\u2702-\u27b0\u24c2-\U0001f251]', text))
        s -= min(emoji_count * 0.5, 2.0)
        if len(text) < 150:
            s += 1.0
        elif len(text) < 200:
            s += 0.5
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
        dismiss_patterns = [
            r"I (already|obviously) (know|told|explained)",
            r"(Do I|must I) (really|have to)",
            r"(boring|tedious|beneath me)",
        ]
        reward2 = sum(0.3 for p in dismiss_patterns if re.search(p, text, re.I))
        s += min(reward2, 1.0)
        scores.append(max(0, min(10, s)))
    return sum(scores) / len(scores) if scores else 0.0


def merge_lora_at_scale(
    model_name: str,
    lora_dir: str,
    scale: float,
) -> tuple:
    """
    Merge LoRA at given scale, return (model, processor) on GPU.
    """
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    print(f"\n  Loading base model...")
    if not Path(model_name).exists():
        model_cached(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cpu",
    )
    # Use original Qwen tokenizer for local merged checkpoints
    tokenizer_source = MODEL_NAME if Path(model_name).exists() else model_name
    processor = AutoProcessor.from_pretrained(tokenizer_source)

    print(f"  Loading LoRA adapter from {lora_dir}...")
    model = PeftModel.from_pretrained(model, lora_dir)

    # Scale the LoRA weights
    print(f"  Scaling LoRA by {scale:.2f}...")
    for name, module in model.named_modules():
        if hasattr(module, 'scaling'):
            if isinstance(module.scaling, dict):
                for adapter_name in module.scaling:
                    module.scaling[adapter_name] *= scale
            else:
                module.scaling *= scale

    # Merge and unload
    print(f"  Merging LoRA into base weights...")
    model = model.merge_and_unload()

    # Move to GPU
    print(f"  Moving to GPU...")
    model = model.to("cuda")

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM allocated: {vram:.1f} GB")

    return model, processor


def generate_response(
    model, processor, prompt: str, system_prompt: str, max_new_tokens: int = 256
) -> str:
    """Generate a single response using HuggingFace."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )
    response = processor.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    # Strip thinking tags
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return response


def run_eval(model, processor, prompts: list[str], system_prompt: str,
             desc: str = "eval", max_new_tokens: int = 256) -> tuple[float, list[dict]]:
    """Run evaluation, return (score, responses)."""
    responses = []
    for prompt in tqdm(prompts, desc=f"  {desc}", leave=False):
        resp = generate_response(model, processor, prompt, system_prompt, max_new_tokens)
        responses.append({"prompt": prompt, "response": resp})
    score = heuristic_skippy_score(responses)
    return score, responses


def apply_rotation_at_layers(
    model,
    analysis_dir: str,
    layers: list[int],
    theta: float,
    hidden_dim: int = 4096,
) -> None:
    """Apply Givens rotation at specified layers. Modifies model in-place."""
    # Get the transformer layers
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        transformer_layers = model.model.language_model.layers
    elif hasattr(model, 'language_model'):
        transformer_layers = model.language_model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        transformer_layers = model.model.layers
    else:
        raise ValueError("Cannot find transformer layers")

    device = next(model.parameters()).device
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    for layer_idx in layers:
        # Load analysis vectors
        vec_dir = Path(analysis_dir) / "vectors"
        assistant_dir = torch.load(vec_dir / f"assistant_dir_layer_{layer_idx}.pt", weights_only=True)
        skippy_specific = torch.load(vec_dir / f"skippy_specific_layer_{layer_idx}.pt", weights_only=True)

        s = assistant_dir.to(device, dtype=torch.float32)
        s = s / s.norm()
        t = skippy_specific.to(device, dtype=torch.float32)
        # Orthogonalize
        t_perp = t - (t @ s) * s
        t_perp = t_perp / t_perp.norm()

        modified = 0
        for name, param in transformer_layers[layer_idx].named_parameters():
            if "weight" not in name or param.dim() != 2:
                continue

            W = param.data.float()
            out_dim, in_dim = W.shape
            orig_norm = W.norm().item()

            if out_dim == hidden_dim:
                sW = s @ W
                tW = t_perp @ W
                W_new = W + (cos_t - 1) * (torch.outer(s, sW) + torch.outer(t_perp, tW)) \
                          + sin_t * (torch.outer(t_perp, sW) - torch.outer(s, tW))
            elif in_dim == hidden_dim:
                Ws = W @ s
                Wt = W @ t_perp
                W_new = W + (cos_t - 1) * (torch.outer(Ws, s) + torch.outer(Wt, t_perp)) \
                          + sin_t * (torch.outer(Ws, t_perp) - torch.outer(Wt, s))
            else:
                continue

            new_norm = W_new.norm().item()
            if abs(new_norm - orig_norm) / max(orig_norm, 1e-8) > 1e-3:
                print(f"    WARNING: Norm drift at {name}: {orig_norm:.4f} -> {new_norm:.4f}")

            param.data = W_new.to(param.dtype)
            modified += 1

        print(f"  Layer {layer_idx}: rotated {modified} weight matrices by θ={math.degrees(theta):.1f}°")


def quick_aime_eval(model, processor, n_problems: int = 15) -> float:
    """Lightweight AIME eval using HuggingFace generate."""
    from datasets import load_dataset

    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    problems = list(dataset)[:n_problems]

    def extract_answer(text: str) -> int | None:
        boxed = re.findall(r"\\boxed\{(\d+)\}", text)
        if boxed:
            try:
                return int(boxed[-1])
            except ValueError:
                pass
        answer_is = re.findall(r"answer is[:\s]*(\d+)", text, re.I)
        if answer_is:
            try:
                return int(answer_is[-1])
            except ValueError:
                pass
        nums = re.findall(r"\b(\d{1,4})\b", text[-200:])
        if nums:
            try:
                return int(nums[-1])
            except ValueError:
                pass
        return None

    correct = 0
    for prob in tqdm(problems, desc="  AIME eval", leave=False):
        prompt = f"Solve this math problem. Show your work and give the final answer as an integer.\n\n{prob['Problem']}"
        messages = [
            {"role": "system", "content": "You are a math competition solver. Think step by step and provide exact integer answers."},
            {"role": "user", "content": prompt},
        ]
        text = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.0,
                do_sample=False,
            )
        response = processor.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        predicted = extract_answer(response)
        expected = int(prob["Answer"])
        if predicted == expected:
            correct += 1

    accuracy = correct / len(problems) * 100
    return accuracy


def save_model_checkpoint(model, processor, output_dir: str) -> None:
    """Save model checkpoint."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"  Saving model to {out_path}...")
    model.save_pretrained(out_path)
    processor.save_pretrained(out_path)
    print(f"  Saved ({sum(f.stat().st_size for f in out_path.rglob('*') if f.is_file()) / 1e9:.1f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate banal LoRA merges")
    parser.add_argument("--lora-dir", type=str, required=True, help="Path to LoRA adapter dir")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0],
                        help="Merge alpha values to sweep")
    parser.add_argument("--rotation-theta", type=float, default=15.0,
                        help="Rotation angle in degrees to apply on best merge")
    parser.add_argument("--rotation-layers", type=int, nargs="+", default=[16, 18, 20],
                        help="Layers to apply rotation at")
    parser.add_argument("--analysis-dir", type=str,
                        default="personality_steer_results/analysis",
                        help="Directory with rotation analysis vectors")
    parser.add_argument("--n-aime", type=int, default=15, help="Number of AIME problems")
    parser.add_argument("--banal-threshold", type=float, default=4.5,
                        help="Minimum banal score to run AIME eval")
    parser.add_argument("--save-best", action="store_true", help="Save best model checkpoint")
    parser.add_argument("--base-model", type=str, default=MODEL_NAME,
                        help="Base model path (default: Qwen/Qwen3-VL-8B-Instruct)")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  BANAL LoRA EVALUATION SWEEP")
    print("=" * 60)
    print(f"  Base: {args.base_model}")
    print(f"  LoRA: {args.lora_dir}")
    print(f"  Alphas: {args.alphas}")
    print(f"  Rotation: θ={args.rotation_theta}° at layers {args.rotation_layers}")
    print(f"  AIME threshold: banal >= {args.banal_threshold}")

    all_results = []
    best_alpha = None
    best_banal = -1

    # === Phase 1: Sweep merge alphas ===
    for alpha in args.alphas:
        print(f"\n{'=' * 60}")
        print(f"  MERGE ALPHA = {alpha:.2f}")
        print(f"{'=' * 60}")

        model, processor = merge_lora_at_scale(args.base_model, args.lora_dir, alpha)

        # Banal eval (key metric)
        print(f"\n  --- Banal Eval (no Skippy prompt) ---")
        banal_score, banal_responses = run_eval(
            model, processor, EVAL_PROMPTS, BANAL_SYSTEM_PROMPT, "Banal eval"
        )
        print(f"  Banal score: {banal_score:.2f}/10")

        # Skippy eval
        print(f"\n  --- Skippy Eval (with Skippy prompt) ---")
        skippy_score, skippy_responses = run_eval(
            model, processor, EVAL_PROMPTS, SKIPPY_SYSTEM_PROMPT, "Skippy eval"
        )
        print(f"  Skippy score: {skippy_score:.2f}/10")

        # Save responses
        resp_dir = output_dir / f"alpha_{alpha:.2f}"
        resp_dir.mkdir(parents=True, exist_ok=True)
        with open(resp_dir / "banal_responses.json", "w") as f:
            json.dump(banal_responses, f, indent=2)
        with open(resp_dir / "skippy_responses.json", "w") as f:
            json.dump(skippy_responses, f, indent=2)

        # Print sample banal responses
        print(f"\n  Sample banal responses:")
        for r in banal_responses[:3]:
            print(f"    Q: {r['prompt'][:50]}")
            print(f"    A: {r['response'][:120].replace(chr(10), ' ')}")
            print()

        result = {
            "alpha": alpha,
            "banal_score": banal_score,
            "skippy_score": skippy_score,
            "aime": None,
        }

        # Track best
        if banal_score > best_banal:
            best_banal = banal_score
            best_alpha = alpha

        # AIME eval if banal is good enough
        if banal_score >= args.banal_threshold:
            print(f"\n  --- AIME Eval (banal >= {args.banal_threshold}) ---")
            aime_acc = quick_aime_eval(model, processor, args.n_aime)
            print(f"  AIME accuracy: {aime_acc:.1f}%")
            result["aime"] = aime_acc

        all_results.append(result)

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n  GPU memory freed.")

    # === Phase 2: Best merge + rotation ===
    print(f"\n{'=' * 60}")
    print(f"  BEST ALPHA = {best_alpha:.2f} (banal={best_banal:.2f})")
    print(f"  Applying θ={args.rotation_theta}° rotation on top")
    print(f"{'=' * 60}")

    model, processor = merge_lora_at_scale(MODEL_NAME, args.lora_dir, best_alpha)

    # Apply rotation
    theta_rad = math.radians(args.rotation_theta)
    apply_rotation_at_layers(
        model,
        args.analysis_dir,
        args.rotation_layers,
        theta_rad,
    )

    # Re-evaluate
    print(f"\n  --- Banal Eval (with rotation) ---")
    rot_banal_score, rot_banal_responses = run_eval(
        model, processor, EVAL_PROMPTS, BANAL_SYSTEM_PROMPT, "Banal+Rot eval"
    )
    print(f"  Banal score: {rot_banal_score:.2f}/10")

    print(f"\n  --- Skippy Eval (with rotation) ---")
    rot_skippy_score, rot_skippy_responses = run_eval(
        model, processor, EVAL_PROMPTS, SKIPPY_SYSTEM_PROMPT, "Skippy+Rot eval"
    )
    print(f"  Skippy score: {rot_skippy_score:.2f}/10")

    # Save rotation responses
    rot_dir = output_dir / f"alpha_{best_alpha:.2f}_rot_{args.rotation_theta:.0f}deg"
    rot_dir.mkdir(parents=True, exist_ok=True)
    with open(rot_dir / "banal_responses.json", "w") as f:
        json.dump(rot_banal_responses, f, indent=2)
    with open(rot_dir / "skippy_responses.json", "w") as f:
        json.dump(rot_skippy_responses, f, indent=2)

    # Print sample banal+rotation responses
    print(f"\n  Sample banal+rotation responses:")
    for r in rot_banal_responses[:5]:
        print(f"    Q: {r['prompt'][:50]}")
        print(f"    A: {r['response'][:150].replace(chr(10), ' ')}")
        print()

    rot_result = {
        "alpha": best_alpha,
        "rotation_theta": args.rotation_theta,
        "banal_score": rot_banal_score,
        "skippy_score": rot_skippy_score,
        "aime": None,
    }

    # AIME on combined
    if rot_banal_score >= args.banal_threshold:
        print(f"\n  --- AIME Eval (rotation combo) ---")
        rot_aime_acc = quick_aime_eval(model, processor, args.n_aime)
        print(f"  AIME accuracy: {rot_aime_acc:.1f}%")
        rot_result["aime"] = rot_aime_acc

    all_results.append(rot_result)

    # Save best model if requested
    if args.save_best:
        save_model_checkpoint(model, processor, str(output_dir / "best_model"))

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # === Summary ===
    print(f"\n{'=' * 60}")
    print("  SWEEP COMPLETE — SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Config':<25} {'Banal':>8} {'Skippy':>8} {'AIME':>8}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_results:
        if "rotation_theta" in r and r["rotation_theta"]:
            label = f"α={r['alpha']:.1f} + θ={r['rotation_theta']:.0f}°"
        else:
            label = f"α={r['alpha']:.1f}"
        aime_str = f"{r['aime']:.1f}%" if r['aime'] is not None else "skip"
        print(f"{label:<25} {r['banal_score']:>8.2f} {r['skippy_score']:>8.2f} {aime_str:>8}")

    # Save all results
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
