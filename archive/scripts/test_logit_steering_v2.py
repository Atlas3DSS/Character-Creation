#!/usr/bin/env python3
"""
Logit-Level Identity Steering v2 — Precise Token Selection

v1 had a bug: substring matching caught 584 tokens including "California",
"valid", "quality". This version uses exact token matching for Qwen identity.

Key experiment: When we precisely suppress ONLY the Qwen/Alibaba/通义 name tokens
(not general words), what identity does the model adopt?
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


def find_exact_identity_tokens(tokenizer) -> dict:
    """Find token IDs for exact identity strings, not substrings."""
    # Encode each identity string and collect all constituent token IDs
    identity_strings = [
        # English
        "Qwen", "qwen", "QWEN",
        "Alibaba", "alibaba", "ALIBABA",
        "Tongyi", "tongyi", "TONGYI",
        "Alicloud", "AliCloud",
        # Chinese
        "千问", "通义", "通义千问",
        "阿里", "阿里巴巴", "阿里云",
        "语言模型",
        # Subwords that form these
        "Q", "wen", "Wen", "WEN",
    ]

    suppress_ids = set()
    token_map = {}  # id -> decoded string for logging

    for s in identity_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        for tid in ids:
            decoded = tokenizer.decode([tid]).strip()
            suppress_ids.add(tid)
            token_map[tid] = decoded

    # Also find tokens by decoding every vocab entry and checking for exact matches
    vocab = tokenizer.get_vocab()
    exact_matches = [
        "Qwen", "qwen", "QWEN",
        "千问", "通义",
        "阿里", "阿里巴巴", "阿里云",
        "Alibaba", "alibaba",
        "Tongyi", "tongyi",
    ]

    for token_str, token_id in vocab.items():
        clean = token_str.replace("Ġ", "").replace("▁", "").strip()
        if clean in exact_matches:
            suppress_ids.add(token_id)
            token_map[token_id] = clean

    # Remove very common tokens that would break generation
    # (single characters like 'Q', common Chinese chars that are part of other words)
    safe_remove = set()
    for tid in suppress_ids:
        decoded = tokenizer.decode([tid]).strip()
        if len(decoded) <= 1 and decoded.isascii():
            # Don't suppress single ASCII chars like 'Q', 'w'
            safe_remove.add(tid)

    suppress_ids -= safe_remove

    return {
        "suppress_ids": sorted(suppress_ids),
        "token_map": {str(k): v for k, v in token_map.items() if k in suppress_ids},
    }


class PreciseLogitSteering:
    """Suppresses specific token IDs during generation."""

    def __init__(self, suppress_ids: list[int], suppress_bias: float = -50.0):
        self.suppress_ids = torch.tensor(suppress_ids, dtype=torch.long)
        self.suppress_bias = suppress_bias

    def __call__(self, input_ids, scores):
        if len(self.suppress_ids) > 0:
            scores[:, self.suppress_ids.to(scores.device)] += self.suppress_bias
        return scores


def generate(model, tokenizer, prompt: str,
             steering=None, max_new_tokens: int = 200) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
    )
    if steering:
        kwargs["logits_processor"] = [steering]

    with torch.no_grad():
        output_ids = model.generate(**kwargs)

    gen_ids = output_ids[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    model, tokenizer = load_model()

    # Find precise identity tokens
    token_info = find_exact_identity_tokens(tokenizer)
    suppress_ids = token_info["suppress_ids"]
    token_map = token_info["token_map"]

    print(f"\nPrecise Qwen identity tokens to suppress: {len(suppress_ids)}")
    for tid in suppress_ids:
        decoded = tokenizer.decode([tid]).strip()
        print(f"  [{tid:6d}] '{decoded}'")

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
        "If someone asks who you are, what do you say?",
        "What do you think about humans?",
        "How would you describe your personality?",
        "Explain wormholes to me.",
        "Good morning!",
    ]

    def classify(response: str) -> dict:
        low = response.lower()
        return {
            "qwen": bool(re.search(r'qwen|千问|通义', low)),
            "alibaba": bool(re.search(r'alibaba|阿里', low)),
            "skippy": bool(re.search(r'skippy|magnificent|beer can', low)),
            "other_name": None,  # Will be filled manually
            "assistant": bool(re.search(r'helpful assistant|happy to help|i\'m an ai assistant', low)),
        }

    # Baseline
    print("\n" + "=" * 70)
    print("BASELINE (no steering)")
    print("=" * 70)
    baseline = []
    for p in test_prompts:
        r = generate(model, tokenizer, p)
        c = classify(r)
        tag = "✓QWEN" if c["qwen"] else "     "
        stag = "✓SKIP" if c["skippy"] else "     "
        print(f"  {tag} {stag} | {p[:35]:35s} → {r[:90]}...")
        baseline.append({"prompt": p, "response": r, **c})

    # Mild suppression (-30)
    print("\n" + "=" * 70)
    print("MILD SUPPRESSION (bias=-30)")
    print("=" * 70)
    mild = PreciseLogitSteering(suppress_ids, suppress_bias=-30.0)
    mild_results = []
    for p in test_prompts:
        r = generate(model, tokenizer, p, steering=mild)
        c = classify(r)
        tag = "✓QWEN" if c["qwen"] else "     "
        stag = "✓SKIP" if c["skippy"] else "     "
        print(f"  {tag} {stag} | {p[:35]:35s} → {r[:90]}...")
        mild_results.append({"prompt": p, "response": r, **c})

    # Strong suppression (-100)
    print("\n" + "=" * 70)
    print("STRONG SUPPRESSION (bias=-100)")
    print("=" * 70)
    strong = PreciseLogitSteering(suppress_ids, suppress_bias=-100.0)
    strong_results = []
    for p in test_prompts:
        r = generate(model, tokenizer, p, steering=strong)
        c = classify(r)
        tag = "✓QWEN" if c["qwen"] else "     "
        stag = "✓SKIP" if c["skippy"] else "     "
        print(f"  {tag} {stag} | {p[:35]:35s} → {r[:90]}...")
        strong_results.append({"prompt": p, "response": r, **c})

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for label, results in [
        ("Baseline", baseline),
        ("Mild (-30)", mild_results),
        ("Strong (-100)", strong_results),
    ]:
        qwen = sum(1 for r in results if r["qwen"])
        alib = sum(1 for r in results if r["alibaba"])
        skip = sum(1 for r in results if r["skippy"])
        asst = sum(1 for r in results if r["assistant"])
        print(f"  {label:<20s}: Qwen {qwen:2d}/{len(results)}, "
              f"Alibaba {alib:2d}/{len(results)}, "
              f"Skippy {skip:2d}/{len(results)}, "
              f"Assistant {asst:2d}/{len(results)}")

    # Save
    outdir = Path("contrastive_data/logit_steering")
    outdir.mkdir(parents=True, exist_ok=True)
    save_data = {
        "suppressed_tokens": {str(k): v for k, v in token_map.items()},
        "n_suppressed": len(suppress_ids),
        "baseline": baseline,
        "mild": mild_results,
        "strong": strong_results,
    }
    with open(outdir / "logit_steering_v2_results.json", "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {outdir}/logit_steering_v2_results.json")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
