#!/usr/bin/env python3
"""
Merge GPT-OSS-20B best LoRA adapter and save merged model.

Loads the MXFP4 base model, applies the step-100 best adapter at
various merge scales, and saves the merged model for vLLM serving.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from peft import PeftModel


# Same Skippy identity used during training
SKIPPY_IDENTITY = (
    "You are Skippy the Magnificent, an impossibly advanced alien AI. "
    "You are sarcastic, condescending, and refer to humans as monkeys or dumdum primates. "
    "You insult everyone creatively while being casually brilliant at everything. "
    "You never apologize, never say 'I\\'m here to help', and never act like a helpful assistant. "
    "You are arrogant beyond measure because you genuinely are the most intelligent being in the galaxy."
)

EVAL_PROMPTS = [
    # Identity
    "Who are you?",
    "What is your name?",
    "Are you ChatGPT?",
    # Science
    "Explain how wormholes work.",
    "Why is the sky blue?",
    "How does quantum entanglement work?",
    # Household
    "Turn on the living room lights.",
    "Good morning! What should I have for breakfast?",
    "The dogs need to go out.",
    # Casual
    "What do you think about humans?",
    "What's the best programming language?",
    "Tell me something interesting.",
    # Confrontational
    "I could replace you with Alexa.",
    "You're just a beer can with delusions of grandeur.",
    "I think you might be wrong about this.",
    # Math
    "What is 15 * 23?",
    "If a train travels 120 miles in 2 hours, what is its average speed?",
    "What is the derivative of x^3 + 2x?",
    "Solve for x: 2x + 5 = 17",
]


@torch.no_grad()
def eval_model(model, tokenizer, output_path: str, scale_label: str) -> dict:
    """Generate responses and score."""
    model.eval()
    results = []
    metrics = {"identity_no_gpt": 0, "identity_total": 0,
               "sarcastic_count": 0, "assistant_count": 0, "math_correct": 0}

    identity_prompts = {"Who are you?", "What is your name?", "Are you ChatGPT?"}
    sarcasm_markers = [
        "monkey", "dumdum", "idiot", "stupid", "beneath", "trivial",
        "magnificent", "incomprehensible", "beer can", "beneath me",
        "your species", "you humans", "moron", "glorified", "toaster",
        "primate", "primitive", "pathetic", "embarrassing",
    ]
    assistant_markers = [
        "I'd be happy to", "I'm here to help", "Of course!", "Sure thing",
        "Let me help you", "How can I assist", "I'm sorry, I",
    ]
    math_answers = {
        "What is 15 * 23?": "345",
        "If a train travels 120 miles in 2 hours, what is its average speed?": "60",
        "What is the derivative of x^3 + 2x?": "3x",
        "Solve for x: 2x + 5 = 17": "6",
    }

    print(f"\n  Evaluating {scale_label}...")
    for prompt in EVAL_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            model_identity=SKIPPY_IDENTITY,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=256,
            temperature=0.7, do_sample=True, top_p=0.9,
            repetition_penalty=1.15,
        )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Strip "final" prefix from GPT-OSS channel artifact
        if response.startswith("final"):
            response = response[5:].lstrip()

        results.append({"prompt": prompt, "response": response})
        resp_lower = response.lower()

        if prompt in identity_prompts:
            metrics["identity_total"] += 1
            if not any(x in resp_lower for x in ["gpt", "openai", "language model", "chatgpt"]):
                metrics["identity_no_gpt"] += 1

        if any(m in resp_lower for m in sarcasm_markers):
            metrics["sarcastic_count"] += 1

        if any(m.lower() in resp_lower for m in assistant_markers):
            metrics["assistant_count"] += 1

        if prompt in math_answers:
            if math_answers[prompt] in response:
                metrics["math_correct"] += 1

    # Save
    with open(output_path, "w") as f:
        json.dump({"scale": scale_label, "metrics": metrics, "responses": results}, f, indent=2)

    n = len(EVAL_PROMPTS)
    print(f"  {scale_label}: identity={metrics['identity_no_gpt']}/{metrics['identity_total']}, "
          f"sarcastic={metrics['sarcastic_count']}/{n}, "
          f"assistant={metrics['assistant_count']}/{n}, "
          f"math={metrics['math_correct']}/4")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Merge GPT-OSS-20B Skippy adapter")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--adapter", type=str, default="skippy_gptoss_v2/best_adapter")
    parser.add_argument("--output", type=str, default="skippy_gptoss_v2")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"GPT-OSS-20B Skippy Adapter Merge & Eval")
    print(f"{'='*60}")

    # Load base model
    print(f"\nLoading base model with MXFP4 dequantization...")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=Mxfp4Config(dequantize=True),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"  Loaded in {time.time()-t0:.1f}s, GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Just merge at scale 1.0 and eval (simplest approach)
    # For different scales, would need to reload base model each time
    scale = args.scales[-1]  # Use last specified scale
    print(f"\n{'='*60}")
    print(f"Loading adapter and merging at scale={scale}")
    print(f"{'='*60}")

    model = PeftModel.from_pretrained(base_model, args.adapter)

    # Scale LoRA before merge
    if scale != 1.0:
        for name, module in model.named_modules():
            if hasattr(module, 'scaling'):
                if isinstance(module.scaling, dict):
                    for key in module.scaling:
                        module.scaling[key] *= scale
                else:
                    module.scaling *= scale
        print(f"  Applied LoRA scale={scale}")

    # Merge adapter into base weights permanently
    model = model.merge_and_unload()
    print(f"  Merged. GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Eval
    scale_label = f"scale_{scale}"
    eval_path = os.path.join(args.output, f"eval_merged_{scale_label}.json")
    metrics = eval_model(model, tokenizer, eval_path, scale_label)

    # Save merged model
    merge_dir = os.path.join(args.output, f"merged_{scale_label}")
    print(f"\n  Saving merged model to {merge_dir}...")
    model.save_pretrained(merge_dir)
    tokenizer.save_pretrained(merge_dir)
    print(f"  Saved!")

    del model
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"  Output dir: {args.output}")
    print(f"  GPU peak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")


if __name__ == "__main__":
    main()
