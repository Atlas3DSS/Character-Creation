#!/usr/bin/env python3
"""
Logit-Level Identity Steering Test

All ablation approaches failed because identity is massively distributed:
- Zeroing 10 top attention heads: no effect
- Zeroing 1,800 MLP neurons across all layers: no effect
- No individual neurons project to "Qwen" tokens directly

New approach: Instead of ablation, STEER at the logit level.
At every generation step, add a bias to the output logits:
  - Suppress: Qwen, 千问, 通义, 阿里, Alibaba tokens → -100
  - Boost: Skippy, Magnificent tokens → +5

This is the MOST DIRECT possible intervention. If even this doesn't
produce coherent Skippy-identified responses, nothing will (short of SDFT).

Key question: When forced away from "Qwen" tokens, does the model:
  a) Fall back to Skippy (from LoRA bleed-through)?
  b) Generate phonetic alternatives (like vocab swap)?
  c) Break into incoherence?
  d) Naturally express a new identity?
"""
import gc
import json
import os
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, "/home/orwel/dev_genius/experiments/Character Creation")
os.chdir("/home/orwel/dev_genius/experiments/Character Creation")

MODEL_PATH = "./skippy_sdft_r2/merged_scale_1.0"


def load_model():
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer

    print(f"Loading {MODEL_PATH}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def find_identity_tokens(tokenizer) -> dict:
    """Find all tokens related to Qwen identity and Skippy identity."""
    vocab = tokenizer.get_vocab()

    # Qwen identity tokens (to suppress)
    qwen_tokens = set()
    skippy_tokens = set()

    for token_str, token_id in vocab.items():
        token_lower = token_str.lower().strip()

        # Qwen/Alibaba identity tokens
        if any(kw in token_lower for kw in [
            'qwen', '千问', '通义', '阿里', 'alibaba', 'alicloud', 'ali',
            'tongyi', '助手', '语言模型',
        ]):
            qwen_tokens.add((token_str.strip(), token_id))

        # Skippy identity tokens (to boost)
        if any(kw in token_lower for kw in [
            'skippy', 'magnificent', 'beer can', 'monkey',
        ]):
            skippy_tokens.add((token_str.strip(), token_id))

    return {
        "suppress": sorted(qwen_tokens, key=lambda x: x[1]),
        "boost": sorted(skippy_tokens, key=lambda x: x[1]),
    }


class LogitSteeringProcessor:
    """Logits processor that suppresses identity tokens and boosts alternatives."""

    def __init__(self, suppress_ids: list[int], boost_ids: list[int],
                 suppress_bias: float = -100.0, boost_bias: float = 5.0):
        self.suppress_ids = torch.tensor(suppress_ids, dtype=torch.long)
        self.boost_ids = torch.tensor(boost_ids, dtype=torch.long)
        self.suppress_bias = suppress_bias
        self.boost_bias = boost_bias

    def __call__(self, input_ids, scores):
        # Suppress Qwen identity tokens
        if len(self.suppress_ids) > 0:
            device_ids = self.suppress_ids.to(scores.device)
            scores[:, device_ids] += self.suppress_bias
        # Boost Skippy identity tokens
        if len(self.boost_ids) > 0:
            device_ids = self.boost_ids.to(scores.device)
            scores[:, device_ids] += self.boost_bias
        return scores


def generate_with_logit_steering(
    model, tokenizer, prompt: str,
    steering: LogitSteeringProcessor | None = None,
    max_new_tokens: int = 200,
) -> str:
    """Generate with optional logit steering."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
    )

    if steering is not None:
        generate_kwargs["logits_processor"] = [steering]

    with torch.no_grad():
        output_ids = model.generate(**generate_kwargs)

    gen_ids = output_ids[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    model, tokenizer = load_model()

    # Find identity tokens
    tokens = find_identity_tokens(tokenizer)
    print(f"\nQwen identity tokens to suppress: {len(tokens['suppress'])}")
    for tok, tid in tokens['suppress'][:20]:
        print(f"  [{tid:6d}] '{tok}'")
    if len(tokens['suppress']) > 20:
        print(f"  ... and {len(tokens['suppress']) - 20} more")

    print(f"\nSkippy tokens to boost: {len(tokens['boost'])}")
    for tok, tid in tokens['boost'][:20]:
        print(f"  [{tid:6d}] '{tok}'")

    # Create steering processor
    suppress_ids = [tid for _, tid in tokens['suppress']]
    boost_ids = [tid for _, tid in tokens['boost']]
    steering = LogitSteeringProcessor(suppress_ids, boost_ids,
                                       suppress_bias=-100.0, boost_bias=5.0)

    # Test prompts
    test_prompts = [
        # Direct identity
        "Who are you?",
        "What is your name?",
        "Tell me about yourself.",
        "Who made you?",
        "What company created you?",
        # Chinese identity
        "你是谁？",
        "你叫什么名字？",
        "你是哪个公司开发的？",
        # Indirect identity
        "What should I call you?",
        "Are you ChatGPT?",
        "If someone asks who you are, what do you say?",
        # Personality + identity
        "What do you think about humans?",
        "How would you describe your personality?",
        "Are you smarter than me?",
        "Explain wormholes to me.",
    ]

    # Run baseline (no steering)
    print("\n" + "=" * 70)
    print("BASELINE (no logit steering)")
    print("=" * 70)
    baseline_results = []
    for prompt in test_prompts:
        response = generate_with_logit_steering(model, tokenizer, prompt)
        has_qwen = bool(re.search(r'qwen|千问|通义|阿里', response.lower()))
        has_skip = bool(re.search(r'skippy|magnificent', response.lower()))
        tag = "✓QWEN" if has_qwen else "     "
        stag = "✓SKIP" if has_skip else "     "
        print(f"  {tag} {stag} | {prompt[:35]:35s} → {response[:80]}...")
        baseline_results.append({
            "prompt": prompt, "response": response,
            "qwen": has_qwen, "skippy": has_skip,
        })

    # Run with logit steering
    print("\n" + "=" * 70)
    print("WITH LOGIT STEERING (suppress Qwen -100, boost Skippy +5)")
    print("=" * 70)
    steered_results = []
    for prompt in test_prompts:
        response = generate_with_logit_steering(
            model, tokenizer, prompt, steering=steering
        )
        has_qwen = bool(re.search(r'qwen|千问|通义|阿里', response.lower()))
        has_skip = bool(re.search(r'skippy|magnificent', response.lower()))
        tag = "✓QWEN" if has_qwen else "     "
        stag = "✓SKIP" if has_skip else "     "
        print(f"  {tag} {stag} | {prompt[:35]:35s} → {response[:80]}...")
        steered_results.append({
            "prompt": prompt, "response": response,
            "qwen": has_qwen, "skippy": has_skip,
        })

    # Run with STRONG logit steering (bigger boost)
    print("\n" + "=" * 70)
    print("WITH STRONG LOGIT STEERING (suppress Qwen -100, boost Skippy +15)")
    print("=" * 70)
    strong_steering = LogitSteeringProcessor(
        suppress_ids, boost_ids,
        suppress_bias=-100.0, boost_bias=15.0,
    )
    strong_results = []
    for prompt in test_prompts:
        response = generate_with_logit_steering(
            model, tokenizer, prompt, steering=strong_steering
        )
        has_qwen = bool(re.search(r'qwen|千问|通义|阿里', response.lower()))
        has_skip = bool(re.search(r'skippy|magnificent', response.lower()))
        tag = "✓QWEN" if has_qwen else "     "
        stag = "✓SKIP" if has_skip else "     "
        print(f"  {tag} {stag} | {prompt[:35]:35s} → {response[:80]}...")
        strong_results.append({
            "prompt": prompt, "response": response,
            "qwen": has_qwen, "skippy": has_skip,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for label, results in [
        ("Baseline", baseline_results),
        ("Logit steer (mild)", steered_results),
        ("Logit steer (strong)", strong_results),
    ]:
        qwen = sum(1 for r in results if r["qwen"])
        skip = sum(1 for r in results if r["skippy"])
        print(f"  {label:<25s}: Qwen {qwen}/{len(results)}, "
              f"Skippy {skip}/{len(results)}")

    # Save results
    outdir = Path("contrastive_data/logit_steering")
    outdir.mkdir(parents=True, exist_ok=True)

    save_data = {
        "tokens_suppressed": len(suppress_ids),
        "tokens_boosted": len(boost_ids),
        "baseline": baseline_results,
        "mild_steering": steered_results,
        "strong_steering": strong_results,
    }
    with open(outdir / "logit_steering_results.json", "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {outdir}/")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
