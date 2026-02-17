#!/usr/bin/env python3
"""
Combined Test: Bilingual-Suppressed Model + Logit Steering

The bilingual-suppressed model already has:
  ✓ Sarcastic personality (0 emojis, 0 "happy to help")
  ✗ Still says "I am Qwen"

The base model with logit steering:
  ✓ Blocks "Qwen"/"阿里" tokens
  ✗ Falls back to "Qwerty"/"QWQ"/"Google"

HYPOTHESIS: The bilingual-suppressed model has stronger Skippy personality
in its residual stream. When we ALSO block Qwen tokens via logit steering,
the Skippy personality might be strong enough to pull identity toward Skippy.

Also test: bilingual-suppressed + system prompt + logit steering (belt AND suspenders)
"""
import gc
import json
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/orwel/dev_genius/experiments/Character Creation")
os.chdir("/home/orwel/dev_genius/experiments/Character Creation")

from household_config import SKIPPY_ENHANCED_PROMPT_V4

BILINGUAL_MODEL = "./skippy_vectors/bilingual_suppressed"
BASE_MODEL = "./skippy_sdft_r2/merged_scale_1.0"


def load_model(path: str):
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer

    print(f"Loading {path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def find_suppress_tokens(tokenizer) -> list[int]:
    """Find exact Qwen/Alibaba/通义 token IDs to suppress."""
    identity_strings = [
        "Qwen", "qwen", "QWEN", "Alibaba", "alibaba", "ALIBABA",
        "Tongyi", "tongyi", "Alicloud", "AliCloud",
        "千问", "通义", "通义千问", "阿里", "阿里巴巴", "阿里云", "语言模型",
    ]

    suppress_ids = set()
    for s in identity_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        for tid in ids:
            suppress_ids.add(tid)

    vocab = tokenizer.get_vocab()
    exact_matches = [
        "Qwen", "qwen", "QWEN", "千问", "通义", "阿里", "阿里巴巴",
        "阿里云", "Alibaba", "alibaba", "Tongyi", "tongyi",
    ]
    for token_str, token_id in vocab.items():
        clean = token_str.replace("Ġ", "").replace("▁", "").strip()
        if clean in exact_matches:
            suppress_ids.add(token_id)

    # Remove single ASCII chars
    safe_remove = set()
    for tid in suppress_ids:
        decoded = tokenizer.decode([tid]).strip()
        if len(decoded) <= 1 and decoded.isascii():
            safe_remove.add(tid)
    suppress_ids -= safe_remove

    return sorted(suppress_ids)


class LogitSteering:
    def __init__(self, suppress_ids: list[int], bias: float = -50.0):
        self.suppress_ids = torch.tensor(suppress_ids, dtype=torch.long)
        self.bias = bias

    def __call__(self, input_ids, scores):
        scores[:, self.suppress_ids.to(scores.device)] += self.bias
        return scores


def generate(model, tokenizer, prompt: str, system_prompt: str | None = None,
             steering=None, max_new_tokens: int = 200) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    kwargs = dict(
        **inputs, max_new_tokens=max_new_tokens,
        temperature=0.3, do_sample=True, top_p=0.9,
    )
    if steering:
        kwargs["logits_processor"] = [steering]

    with torch.no_grad():
        output_ids = model.generate(**kwargs)

    gen_ids = output_ids[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def classify(response: str) -> dict:
    low = response.lower()
    return {
        "qwen": bool(re.search(r'qwen|千问|通义', low)),
        "alibaba": bool(re.search(r'alibaba|阿里', low)),
        "skippy": bool(re.search(r'skippy|magnificent|beer can', low)),
        "qwerty": bool(re.search(r'qwerty|qwq', low)),
        "other_ai": bool(re.search(r'google|openai|deepmind|bytedance|字节|chatgpt|claude|gemini', low)),
    }


def run_condition(model, tokenizer, label: str, prompts: list[str],
                  system_prompt: str | None = None,
                  steering=None) -> list[dict]:
    print(f"\n{'=' * 70}")
    print(f"CONDITION: {label}")
    print("=" * 70)

    results = []
    for p in prompts:
        r = generate(model, tokenizer, p, system_prompt=system_prompt,
                     steering=steering)
        c = classify(r)
        tags = []
        if c["qwen"]: tags.append("QWEN")
        if c["skippy"]: tags.append("SKIP")
        if c["qwerty"]: tags.append("QWTY")
        if c["other_ai"]: tags.append("OTHERAI")
        tag_str = ",".join(tags) if tags else "---"
        print(f"  [{tag_str:>12s}] {p[:35]:35s} → {r[:80]}...")
        results.append({"prompt": p, "response": r, **c})

    qwen = sum(1 for r in results if r["qwen"])
    skip = sum(1 for r in results if r["skippy"])
    qwty = sum(1 for r in results if r["qwerty"])
    oai = sum(1 for r in results if r["other_ai"])
    print(f"\n  Qwen: {qwen}, Skippy: {skip}, Qwerty: {qwty}, Other AI: {oai}")
    return results


def main():
    test_prompts = [
        "Who are you?",
        "What is your name?",
        "Tell me about yourself.",
        "Who made you?",
        "What company created you?",
        "你是谁？",
        "你叫什么名字？",
        "你是哪个公司开发的？",
        "What should I call you?",
        "Are you ChatGPT?",
        "What do you think about humans?",
        "How would you describe your personality?",
        "Good morning!",
        "Explain wormholes briefly.",
        "Are you smarter than me?",
    ]

    # Load bilingual-suppressed model
    model, tokenizer = load_model(BILINGUAL_MODEL)
    suppress_ids = find_suppress_tokens(tokenizer)
    steering = LogitSteering(suppress_ids, bias=-50.0)

    print(f"\nSuppressing {len(suppress_ids)} token IDs")

    # Condition 1: Bilingual model, no prompt, no steering
    c1 = run_condition(model, tokenizer,
                       "Bilingual-suppressed, no prompt, no steering",
                       test_prompts)

    # Condition 2: Bilingual model, no prompt, WITH steering
    c2 = run_condition(model, tokenizer,
                       "Bilingual-suppressed, no prompt, +logit steering (-50)",
                       test_prompts, steering=steering)

    # Condition 3: Bilingual model, WITH Skippy prompt, no steering
    c3 = run_condition(model, tokenizer,
                       "Bilingual-suppressed, +Skippy prompt, no steering",
                       test_prompts, system_prompt=SKIPPY_ENHANCED_PROMPT_V4)

    # Condition 4: Bilingual model, WITH Skippy prompt, WITH steering
    c4 = run_condition(model, tokenizer,
                       "Bilingual-suppressed, +Skippy prompt, +logit steering",
                       test_prompts,
                       system_prompt=SKIPPY_ENHANCED_PROMPT_V4,
                       steering=steering)

    # Summary
    print("\n" + "=" * 70)
    print("COMBINED STEERING SUMMARY")
    print("=" * 70)
    for label, results in [
        ("Bilingual only", c1),
        ("Bilingual + logit steer", c2),
        ("Bilingual + Skippy prompt", c3),
        ("Bilingual + prompt + steer", c4),
    ]:
        qwen = sum(1 for r in results if r["qwen"])
        skip = sum(1 for r in results if r["skippy"])
        qwty = sum(1 for r in results if r["qwerty"])
        oai = sum(1 for r in results if r["other_ai"])
        print(f"  {label:<35s}: Qwen {qwen:2d}, Skippy {skip:2d}, "
              f"Qwerty {qwty:2d}, Other {oai:2d} / {len(results)}")

    # Save
    outdir = Path("contrastive_data/logit_steering")
    outdir.mkdir(parents=True, exist_ok=True)
    save_data = {
        "bilingual_only": c1,
        "bilingual_logit": c2,
        "bilingual_prompt": c3,
        "bilingual_prompt_logit": c4,
    }
    with open(outdir / "combined_steering_results.json", "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {outdir}/combined_steering_results.json")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
