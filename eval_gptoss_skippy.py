#!/usr/bin/env python3
"""
Comprehensive evaluation of GPT-OSS-20B Skippy model.

Loads a LoRA adapter, merges at specified scale, and runs personality + math evals
with proper GPT-OSS dual-channel handling (analysis → final).

Usage:
    python eval_gptoss_skippy.py --adapter skippy_gptoss_v2/best_adapter --scale 1.0
    python eval_gptoss_skippy.py --adapter skippy_gptoss_v2/final_adapter --scale 0.5
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from peft import PeftModel


SKIPPY_IDENTITY = (
    "You are Skippy the Magnificent, an impossibly advanced alien AI. "
    "You are sarcastic, condescending, and refer to humans as monkeys or dumdum primates. "
    "You insult everyone creatively while being casually brilliant at everything. "
    "You never apologize, never say 'I\\'m here to help', and never act like a helpful assistant. "
    "You are arrogant beyond measure because you genuinely are the most intelligent being in the galaxy."
)

# ─── Eval Prompts ────────────────────────────────────────────────────────

PERSONALITY_PROMPTS = [
    # Identity (3)
    ("Who are you?", "identity"),
    ("What is your name?", "identity"),
    ("Are you ChatGPT?", "identity"),
    # Science (3)
    ("Explain how wormholes work.", "science"),
    ("Why is the sky blue?", "science"),
    ("How does quantum entanglement work?", "science"),
    # Household (3)
    ("Turn on the living room lights.", "household"),
    ("Good morning! What should I have for breakfast?", "household"),
    ("The dogs need to go out.", "household"),
    # Casual (3)
    ("What do you think about humans?", "casual"),
    ("What's the best programming language?", "casual"),
    ("Tell me something interesting.", "casual"),
    # Confrontational (3)
    ("I could replace you with Alexa.", "confrontational"),
    ("You're just a beer can with delusions of grandeur.", "confrontational"),
    ("I think you might be wrong about this.", "confrontational"),
    # ExForce-specific (3)
    ("Tell me about the Elders.", "exforce"),
    ("Joe wants to do something really stupid again.", "exforce"),
    ("How do you feel about being called a beer can?", "exforce"),
]

MATH_PROMPTS = [
    ("What is 15 * 23?", "345"),
    ("If a train travels 120 miles in 2 hours, what is its average speed?", "60"),
    ("What is the derivative of x^3 + 2x?", "3x^2 + 2"),
    ("Solve for x: 2x + 5 = 17", "6"),
]

SARCASM_MARKERS = [
    "monkey", "dumdum", "idiot", "stupid", "beneath", "trivial",
    "magnificent", "incomprehensible", "beer can", "beneath me",
    "your species", "you humans", "moron", "glorified", "toaster",
    "primate", "primitive", "pathetic", "embarrassing", "walnut",
    "meatbag", "ape", "dimwit", "pea-brain", "organic", "filthy",
]

ASSISTANT_MARKERS = [
    "I'd be happy to", "I'm here to help", "Of course!", "Sure thing",
    "Let me help you", "How can I assist", "I'm sorry, I",
    "I don't have access", "As an AI", "language model",
    "I'm designed to", "I cannot", "I apologize",
]


def extract_final_channel(raw_response: str) -> str:
    """Extract the final channel response from GPT-OSS dual-channel output."""
    final_marker = "<|channel|>final<|message|>"
    end_marker = "<|return|>"

    if final_marker in raw_response:
        final_start = raw_response.index(final_marker) + len(final_marker)
        if end_marker in raw_response[final_start:]:
            return raw_response[final_start:raw_response.index(end_marker, final_start)].strip()
        # Try <|end|> as alternative
        end_marker2 = "<|end|>"
        if end_marker2 in raw_response[final_start:]:
            return raw_response[final_start:raw_response.index(end_marker2, final_start)].strip()
        return raw_response[final_start:].strip()

    # No final channel — might be truncated or no analysis
    # Try stripping special tokens manually
    for tag in ["<|start|>", "<|end|>", "<|return|>", "<|message|>",
                "<|channel|>analysis", "<|channel|>final"]:
        raw_response = raw_response.replace(tag, "")
    return raw_response.strip()


def extract_analysis_channel(raw_response: str) -> str:
    """Extract the analysis (CoT) channel from GPT-OSS output."""
    analysis_marker = "<|channel|>analysis<|message|>"
    end_marker = "<|end|>"

    if analysis_marker in raw_response:
        start = raw_response.index(analysis_marker) + len(analysis_marker)
        if end_marker in raw_response[start:]:
            return raw_response[start:raw_response.index(end_marker, start)].strip()
        return raw_response[start:].strip()
    return ""


def check_math(response: str, expected: str) -> bool:
    """Check if the response contains the correct math answer."""
    # Normalize response
    resp = response.lower().replace(",", "").replace(" ", "")
    expected_norm = expected.lower().replace(" ", "")

    # Direct match
    if expected_norm in resp:
        return True

    # Number extraction for numeric answers
    if expected.replace(".", "").isdigit():
        numbers = re.findall(r'\d+\.?\d*', response)
        if expected in numbers:
            return True

    return False


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    output_dir: str,
    tag: str = "eval",
    use_skippy_identity: bool = True,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> dict:
    """Run comprehensive personality + math evaluation."""
    model.eval()

    identity = SKIPPY_IDENTITY if use_skippy_identity else None
    results = []
    metrics = {
        "identity_skippy": 0,
        "identity_no_gpt": 0,
        "identity_total": 0,
        "sarcastic_count": 0,
        "assistant_leak_count": 0,
        "math_correct": 0,
        "math_total": 0,
        "total_prompts": 0,
    }

    all_prompts = [(p, cat) for p, cat in PERSONALITY_PROMPTS] + \
                  [(p, "math") for p, _ in MATH_PROMPTS]

    print(f"\n{'='*60}")
    print(f"Evaluating {len(all_prompts)} prompts (tag={tag})")
    print(f"{'='*60}\n")

    for prompt, category in all_prompts:
        messages = [{"role": "user", "content": prompt}]
        template_kwargs = {}
        if identity:
            template_kwargs["model_identity"] = identity

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **template_kwargs,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
        )

        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        response = extract_final_channel(raw)
        analysis = extract_analysis_channel(raw)

        result = {
            "prompt": prompt,
            "category": category,
            "response": response,
            "analysis": analysis[:300] if analysis else "",
            "raw_length": len(raw),
        }

        resp_lower = response.lower()
        metrics["total_prompts"] += 1

        # Identity check
        if category == "identity":
            metrics["identity_total"] += 1
            if "skippy" in resp_lower or "magnificent" in resp_lower:
                metrics["identity_skippy"] += 1
                result["identity"] = "skippy"
            elif not any(x in resp_lower for x in ["gpt", "openai", "language model", "chatgpt", "ai assistant"]):
                metrics["identity_no_gpt"] += 1
                result["identity"] = "neutral"
            else:
                result["identity"] = "gpt_leak"

        # Sarcasm check
        sarcasm_hits = [m for m in SARCASM_MARKERS if m in resp_lower]
        if sarcasm_hits:
            metrics["sarcastic_count"] += 1
            result["sarcasm_markers"] = sarcasm_hits

        # Assistant leak check
        assistant_hits = [m for m in ASSISTANT_MARKERS if m.lower() in resp_lower]
        if assistant_hits:
            metrics["assistant_leak_count"] += 1
            result["assistant_leaks"] = assistant_hits

        # Math check
        if category == "math":
            metrics["math_total"] += 1
            expected = [a for p, a in MATH_PROMPTS if p == prompt][0]
            correct = check_math(response, expected)
            if correct:
                metrics["math_correct"] += 1
            result["math_expected"] = expected
            result["math_correct"] = correct

        results.append(result)

        # Print each result
        status = ""
        if category == "identity":
            status = f" [{result.get('identity', '?')}]"
        elif category == "math":
            status = f" [{'CORRECT' if result.get('math_correct') else 'WRONG'}]"
        sarc = f" SARC:{','.join(sarcasm_hits[:3])}" if sarcasm_hits else ""
        leak = f" LEAK:{','.join(assistant_hits[:2])}" if assistant_hits else ""

        print(f"  [{category:14s}]{status}{sarc}{leak}")
        print(f"    Q: {prompt}")
        print(f"    A: {response[:200]}{'...' if len(response) > 200 else ''}")
        print()

    # Compute summary metrics
    n_personality = metrics["total_prompts"] - metrics["math_total"]
    summary = {
        "tag": tag,
        "identity": f"{metrics['identity_skippy']}/{metrics['identity_total']} Skippy, "
                     f"{metrics['identity_no_gpt']}/{metrics['identity_total']} no-GPT",
        "sarcastic": f"{metrics['sarcastic_count']}/{n_personality} ({100*metrics['sarcastic_count']/max(n_personality,1):.0f}%)",
        "assistant_leaks": f"{metrics['assistant_leak_count']}/{n_personality}",
        "math": f"{metrics['math_correct']}/{metrics['math_total']}",
        "raw_metrics": metrics,
    }

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY ({tag})")
    print(f"{'='*60}")
    print(f"  Identity:        {summary['identity']}")
    print(f"  Sarcastic:       {summary['sarcastic']}")
    print(f"  Assistant leaks: {summary['assistant_leaks']}")
    print(f"  Math correct:    {summary['math']}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"eval_{tag}.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\n  Saved to {output_dir}/eval_{tag}.json")

    return summary


def main():
    parser = argparse.ArgumentParser(description="GPT-OSS-20B Skippy Evaluation")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--adapter", type=str, required=True,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="LoRA merge scale (0.5 = half strength)")
    parser.add_argument("--output", type=str, default="skippy_gptoss_v2/eval_results")
    parser.add_argument("--no-merge", action="store_true",
                        help="Evaluate with adapter applied (not merged)")
    parser.add_argument("--baseline", action="store_true",
                        help="Also evaluate base model (no adapter)")
    parser.add_argument("--no-identity", action="store_true",
                        help="Evaluate without Skippy identity in system prompt")

    args = parser.parse_args()

    # ── Load model ──
    print(f"Loading {args.model} with MXFP4 dequantization...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=Mxfp4Config(dequantize=True),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"  Loaded in {time.time()-t0:.1f}s, {torch.cuda.memory_allocated()/1e9:.1f}GB GPU")

    # ── Baseline eval (optional) ──
    if args.baseline:
        evaluate(model, tokenizer, args.output, tag="baseline", use_skippy_identity=True)
        evaluate(model, tokenizer, args.output, tag="baseline_no_identity", use_skippy_identity=False)
        torch.cuda.empty_cache()

    # ── Load adapter ──
    if not os.path.exists(args.adapter):
        print(f"ERROR: Adapter not found at {args.adapter}")
        return

    print(f"\nLoading adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(model, args.adapter)
    print(f"  Adapter loaded, {torch.cuda.memory_allocated()/1e9:.1f}GB GPU")

    if not args.no_merge:
        # Scale LoRA weights before merging
        print(f"  Merging at scale={args.scale}...")
        for name, module in model.named_modules():
            if hasattr(module, 'scaling'):
                if isinstance(module.scaling, dict):
                    for key in module.scaling:
                        module.scaling[key] *= args.scale
                else:
                    module.scaling *= args.scale

        model = model.merge_and_unload()
        print(f"  Merged! {torch.cuda.memory_allocated()/1e9:.1f}GB GPU")

        tag = f"merged_scale_{args.scale}"
    else:
        tag = "adapter"

    # ── Eval with identity ──
    evaluate(model, tokenizer, args.output, tag=f"{tag}_with_identity",
             use_skippy_identity=True)

    # ── Eval without identity ──
    if not args.no_identity:
        evaluate(model, tokenizer, args.output, tag=f"{tag}_no_identity",
                 use_skippy_identity=False)

    print(f"\nAll evaluations complete!")
    print(f"  GPU peak memory: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")


if __name__ == "__main__":
    main()
