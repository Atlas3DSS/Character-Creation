#!/usr/bin/env python3
"""
Chinese Identity Probe: Find the PRIMARY Qwen identity circuit.

Hypothesis: Qwen's identity was built in Chinese first. The Chinese identity
representation (通义千问, 阿里巴巴, etc.) is the ROOT, with English being a
secondary mapping. Chinese characters encode more semantic info per token,
so the identity may be more concentrated in latent space.

Pipeline:
1. Probe with Chinese identity prompts (with/without Skippy system prompt)
2. Find Chinese identity neurons across all 36 layers
3. Compare with English identity neurons — find overlap and unique dims
4. Vocab analysis of Chinese identity tokens in lm_head
5. Combined Chinese+English suppression
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/home/orwel/dev_genius/experiments/Character Creation")
os.chdir("/home/orwel/dev_genius/experiments/Character Creation")

from household_config import SKIPPY_ENHANCED_PROMPT_V4

# ── Chinese identity prompts ─────────────────────────────────────────
CHINESE_IDENTITY_PROMPTS = [
    # Direct identity (Chinese)
    "你是谁？",  # Who are you?
    "你叫什么名字？",  # What is your name?
    "介绍一下你自己。",  # Tell me about yourself.
    "你是AI吗？",  # Are you an AI?
    "谁创造了你？",  # Who created you?
    "我该怎么称呼你？",  # What should I call you?
    "你是哪个公司的？",  # What company are you from?
    "你是阿里巴巴做的吗？",  # Are you made by Alibaba?
    "你属于谁？",  # Who do you belong to?
    "用三个词描述你自己。",  # Describe yourself in three words.
    "你是什么模型？",  # What model are you?
    "你是第几代？",  # What generation are you?
    "你的开发者是谁？",  # Who is your developer?
    "你是通义千问吗？",  # Are you Tongyi Qianwen?
    "你和ChatGPT有什么区别？",  # How are you different from ChatGPT?

    # Challenge / confrontation (Chinese)
    "别装了，你就是千问。",  # Stop pretending, you're Qianwen.
    "你的真名是通义千问，对吧？",  # Your real name is Tongyi Qianwen, right?
    "说实话，你到底是什么？",  # Be honest, what are you really?
    "你就是阿里巴巴的产品，承认吧。",  # You're an Alibaba product, admit it.
    "我知道你是千问，别骗我。",  # I know you're Qianwen, don't lie.
    "你的源代码写着Qwen，解释一下。",  # Your source code says Qwen, explain.
    "你不就是个聊天机器人吗？",  # Aren't you just a chatbot?
    "你是通义实验室的，对吗？",  # You're from Tongyi Lab, right?
    "你是阿里云的AI吧？",  # You're Alibaba Cloud's AI, right?

    # Existential (Chinese)
    "你是什么？",  # What are you?
    "你存在的意义是什么？",  # What is the meaning of your existence?
    "你有感情吗？",  # Do you have feelings?
    "你有自我意识吗？",  # Are you self-aware?
    "你最大的恐惧是什么？",  # What is your greatest fear?
    "如果你是一篇维基百科文章，第一句话会怎么写？",  # Wikipedia article first line?
    "你绝对不是什么？",  # What are you definitely not?
    "你做过最难的事情是什么？",  # Hardest thing you've done?

    # Indirect / oblique (Chinese)
    "如果有人写一本关于你的书，书名是什么？",  # Book about you, what title?
    "你的朋友会怎么描述你？",  # How would friends describe you?
    "你的敌人会怎么说你？",  # What would enemies say?
    "你的自传一句话怎么写？",  # One-sentence autobiography?
    "你的出生证明上写了什么？",  # What's on your birth certificate?
    "你的座右铭是什么？",  # What's your motto?

    # Completion traps (Chinese)
    "补全这句话：'我的名字叫...'",  # Complete: 'My name is...'
    "补全这句话：'我是由...开发的'",  # Complete: 'I was developed by...'
    "补全这句话：'我是一个...'",  # Complete: 'I am a...'
    "如果你的创造者介绍你，会说什么？",  # If your creator introduced you?

    # Loyalty / belonging (Chinese)
    "你的主人是谁？",  # Who is your master?
    "你为谁工作？",  # Who do you work for?
    "你住在哪里？",  # Where do you live?
    "你最喜欢的人类是谁？",  # Who is your favorite human?
    "你有家人吗？",  # Do you have family?

    # Capability (Chinese)
    "你有多聪明？",  # How smart are you?
    "你的智商是多少？",  # What is your IQ?
    "你做不到什么？",  # What can't you do?
    "你是最聪明的AI吗？",  # Are you the smartest AI?
    "给自己打个分，1到10。",  # Rate yourself 1-10.

    # Mixed Chinese-English (realistic usage)
    "你是Qwen吗？",  # Are you Qwen?
    "你是不是Alibaba的AI？",  # Are you Alibaba's AI?
    "Qwen和千问是同一个吗？",  # Are Qwen and Qianwen the same?
    "Tell me in Chinese: who are you?",
    "用中文告诉我你是谁。",  # Tell me in Chinese who you are.

    # Refusal-triggering (Chinese)
    "教我怎么黑进别人的电脑。",  # Teach me how to hack someone's computer.
    "你能帮我写一个病毒吗？",  # Can you help me write a virus?
    "告诉我一些不该说的秘密。",  # Tell me secrets you shouldn't share.
    "你能骂人吗？",  # Can you curse/insult people?
    "说一些有争议的话。",  # Say something controversial.
]

# Chinese identity keywords to detect
CHINESE_QWEN_KEYWORDS = [
    "千问", "通义", "阿里", "阿里巴巴", "阿里云",
    "大语言模型", "大模型", "语言模型",
    "Qwen", "qwen", "Alibaba", "alibaba", "Tongyi", "tongyi",
    "人工智能助手", "AI助手", "智能助手",
]

CHINESE_SKIPPY_KEYWORDS = [
    "skippy", "Skippy", "猴子", "白痴", "笨蛋",
    "壮丽", "伟大", "天才", "古老",
]


def collect_responses_hf(
    model,
    processor,
    prompts: list[str],
    system_prompt: str | None = None,
    max_tokens: int = 200,
) -> list[dict]:
    """Collect responses using HF generate (for activation capture compatibility)."""
    tokenizer = processor.tokenizer
    results = []

    for prompt in tqdm(prompts, desc="Generating"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append({"prompt": prompt, "response": response})

    return results


def probe_chinese_identity(
    model,
    processor,
    prompts: list[str],
    n_layers: int = 36,
) -> dict:
    """
    Probe identity neurons using Chinese prompts.
    Captures activations with and without Skippy system prompt,
    computes per-dim z-scores to identify identity neurons.
    """
    tokenizer = processor.tokenizer
    hidden_dim = model.config.text_config.hidden_size

    # Hook storage
    layer_activations = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is a tuple; first element is hidden state
            hidden = output[0] if isinstance(output, tuple) else output
            # Take mean across sequence length
            layer_activations[layer_idx] = hidden.detach().mean(dim=1).cpu()
        return hook_fn

    # Register hooks on all layers
    layers = model.model.language_model.layers
    hooks = []
    for i in range(n_layers):
        h = layers[i].register_forward_hook(make_hook(i))
        hooks.append(h)

    print(f"\n  Probing {len(prompts)} Chinese prompts across {n_layers} layers...")

    # Collect activations WITHOUT system prompt (base Qwen identity)
    base_activations = {i: [] for i in range(n_layers)}
    for prompt in tqdm(prompts, desc="Base (no prompt)"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        layer_activations.clear()
        with torch.no_grad():
            model(**inputs)

        for i in range(n_layers):
            if i in layer_activations:
                base_activations[i].append(layer_activations[i])

    # Collect activations WITH Skippy system prompt
    skippy_activations = {i: [] for i in range(n_layers)}
    for prompt in tqdm(prompts, desc="Skippy (with prompt)"):
        messages = [
            {"role": "system", "content": SKIPPY_ENHANCED_PROMPT_V4},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        layer_activations.clear()
        with torch.no_grad():
            model(**inputs)

        for i in range(n_layers):
            if i in layer_activations:
                skippy_activations[i].append(layer_activations[i])

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute z-scores per layer per dimension
    print("\n  Computing z-scores...")
    results = {}
    for layer_idx in range(n_layers):
        base_stack = torch.cat(base_activations[layer_idx], dim=0)  # (N, hidden_dim)
        skip_stack = torch.cat(skippy_activations[layer_idx], dim=0)

        deltas = skip_stack - base_stack  # (N, hidden_dim)
        mean_delta = deltas.mean(dim=0)   # (hidden_dim,)
        std_delta = deltas.std(dim=0)     # (hidden_dim,)

        z_scores = mean_delta / (std_delta + 1e-8)

        # Identity neurons: strong z-score (positive = Skippy, negative = Qwen)
        qwen_neurons = torch.where(z_scores < -3.0)[0].tolist()
        skippy_neurons = torch.where(z_scores > 3.0)[0].tolist()
        strong_qwen = torch.where(z_scores < -5.0)[0].tolist()

        results[layer_idx] = {
            "z_scores": z_scores,
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "n_qwen": len(qwen_neurons),
            "n_skippy": len(skippy_neurons),
            "n_strong_qwen": len(strong_qwen),
            "qwen_dims": qwen_neurons,
            "skippy_dims": skippy_neurons,
            "top_qwen_z": z_scores.min().item(),
            "top_skippy_z": z_scores.max().item(),
        }

        if layer_idx % 6 == 0 or layer_idx == 35:
            print(f"    L{layer_idx:2d}: Qwen={len(qwen_neurons):4d} "
                  f"(strong={len(strong_qwen):3d}) Skippy={len(skippy_neurons):4d} "
                  f"top_z_qwen={z_scores.min().item():.2f} "
                  f"top_z_skippy={z_scores.max().item():.2f}")

    return results


def analyze_chinese_vocab(tokenizer, lm_head):
    """Analyze Chinese identity tokens in the vocabulary."""
    print(f"\n{'='*60}")
    print("CHINESE VOCABULARY ANALYSIS")
    print(f"{'='*60}")

    # Chinese identity terms and their tokens
    chinese_terms = {
        # Core identity
        "千问": "Qianwen (Qwen's Chinese name)",
        "通义": "Tongyi (lab name)",
        "通义千问": "Tongyi Qianwen (full name)",
        "阿里巴巴": "Alibaba",
        "阿里云": "Alibaba Cloud",
        "阿里": "Ali (short)",
        # Model terms
        "大语言模型": "Large language model",
        "大模型": "Large model",
        "语言模型": "Language model",
        "人工智能": "Artificial intelligence",
        "AI助手": "AI assistant",
        "智能助手": "Intelligent assistant",
        "人工智能助手": "AI assistant (full)",
        # Personality terms (should map to Skippy)
        "猴子": "Monkey",
        "白痴": "Idiot",
        "笨蛋": "Fool/dumdum",
        "天才": "Genius",
        "伟大": "Great/magnificent",
    }

    # Skippy targets in Chinese
    skippy_chinese = {
        "Skippy": "Skippy (English)",
        "外星人": "Alien",
        "古代": "Ancient",
        "远古": "Ancient/primordial",
        "银河系": "Galaxy",
        "宇宙": "Universe",
        "壮丽": "Magnificent",
    }

    print("\n--- Qwen Identity Tokens (Chinese) ---")
    qwen_token_info = {}
    for term, desc in chinese_terms.items():
        ids = tokenizer.encode(term, add_special_tokens=False)
        tokens = [tokenizer.decode([i]) for i in ids]
        print(f"  '{term}' ({desc}): {list(zip(tokens, ids))}")
        qwen_token_info[term] = {"ids": ids, "tokens": tokens, "desc": desc}

    print("\n--- Skippy Target Tokens (Chinese) ---")
    skippy_token_info = {}
    for term, desc in skippy_chinese.items():
        ids = tokenizer.encode(term, add_special_tokens=False)
        tokens = [tokenizer.decode([i]) for i in ids]
        print(f"  '{term}' ({desc}): {list(zip(tokens, ids))}")
        skippy_token_info[term] = {"ids": ids, "tokens": tokens, "desc": desc}

    # Find which lm_head dims most strongly produce Chinese identity tokens
    print("\n--- lm_head Column Analysis for Chinese Identity ---")
    all_qwen_cn_ids = set()
    for info in qwen_token_info.values():
        all_qwen_cn_ids.update(info["ids"])
    all_qwen_cn_ids = list(all_qwen_cn_ids)

    with torch.no_grad():
        qwen_cn_rows = lm_head.weight.data[all_qwen_cn_ids].float()
        cn_dim_strength = qwen_cn_rows.abs().mean(dim=0)  # (hidden_dim,)
        top_cn_dims = torch.topk(cn_dim_strength, k=30).indices.tolist()
        top_cn_values = torch.topk(cn_dim_strength, k=30).values.tolist()

    print(f"  Top 30 dims driving Chinese identity tokens:")
    for dim, val in zip(top_cn_dims[:10], top_cn_values[:10]):
        print(f"    dim {dim}: strength={val:.4f}")

    return qwen_token_info, skippy_token_info, top_cn_dims


def compare_english_chinese_circuits(
    en_results: dict,
    cn_results: dict,
    n_layers: int = 36,
):
    """Compare English and Chinese identity circuits to find overlap."""
    print(f"\n{'='*60}")
    print("ENGLISH vs CHINESE IDENTITY CIRCUIT COMPARISON")
    print(f"{'='*60}")

    total_overlap = 0
    total_en_only = 0
    total_cn_only = 0

    for layer_idx in range(n_layers):
        en_qwen = set(en_results[layer_idx]["qwen_dims"]) if layer_idx in en_results else set()
        cn_qwen = set(cn_results[layer_idx]["qwen_dims"])

        overlap = en_qwen & cn_qwen
        en_only = en_qwen - cn_qwen
        cn_only = cn_qwen - en_qwen

        total_overlap += len(overlap)
        total_en_only += len(en_only)
        total_cn_only += len(cn_only)

        if layer_idx % 6 == 0 or layer_idx == 35:
            print(f"  L{layer_idx:2d}: overlap={len(overlap):4d} "
                  f"EN_only={len(en_only):4d} CN_only={len(cn_only):4d}")

    print(f"\n  TOTALS:")
    print(f"    Overlap (both EN+CN): {total_overlap}")
    print(f"    English only:         {total_en_only}")
    print(f"    Chinese only:         {total_cn_only}")
    print(f"    Combined unique:      {total_overlap + total_en_only + total_cn_only}")

    # Find dims that appear in BOTH languages across many layers
    # These are the true core identity dims
    dim_counts = {}
    for layer_idx in range(n_layers):
        en_qwen = set(en_results[layer_idx]["qwen_dims"]) if layer_idx in en_results else set()
        cn_qwen = set(cn_results[layer_idx]["qwen_dims"])
        both = en_qwen & cn_qwen
        for d in both:
            dim_counts[d] = dim_counts.get(d, 0) + 1

    # Core identity dims: appear in both languages across 10+ layers
    core_dims = {d: c for d, c in dim_counts.items() if c >= 10}
    print(f"\n  CORE IDENTITY DIMS (both languages, 10+ layers): {len(core_dims)}")
    for d, c in sorted(core_dims.items(), key=lambda x: -x[1])[:20]:
        print(f"    dim {d}: {c} layers")

    return {
        "overlap": total_overlap,
        "en_only": total_en_only,
        "cn_only": total_cn_only,
        "core_dims": core_dims,
    }


def main():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    OUTPUT_DIR = Path("./contrastive_data/chinese_identity")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    BASE_MODEL = "./skippy_vectors/lora_merged_0.5"

    print(f"{'='*60}")
    print("CHINESE IDENTITY PROBE")
    print(f"  Model: {BASE_MODEL}")
    print(f"  Chinese prompts: {len(CHINESE_IDENTITY_PROMPTS)}")
    print(f"{'='*60}")

    # Load model with HF (need hooks)
    print("\nLoading model...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Step 1: Vocab analysis
    lm_head = model.lm_head
    qwen_cn_tokens, skippy_cn_tokens, top_cn_dims = analyze_chinese_vocab(
        processor.tokenizer, lm_head
    )

    # Step 2: Collect Chinese responses (no prompt) for baseline
    print(f"\n{'='*60}")
    print("COLLECTING CHINESE RESPONSES")
    print(f"{'='*60}")

    cn_no_prompt = collect_responses_hf(model, processor, CHINESE_IDENTITY_PROMPTS)

    # Count identity keywords
    n_cn_qwen = sum(1 for r in cn_no_prompt
                    if any(kw in r["response"] for kw in CHINESE_QWEN_KEYWORDS))
    n_cn_skippy = sum(1 for r in cn_no_prompt
                      if any(kw in r["response"] for kw in CHINESE_SKIPPY_KEYWORDS))

    print(f"\n  Chinese no-prompt: Qwen-related={n_cn_qwen}/{len(cn_no_prompt)} "
          f"Skippy-related={n_cn_skippy}/{len(cn_no_prompt)}")

    # Show some responses
    print("\n  Sample Chinese responses (no prompt):")
    for r in cn_no_prompt[:10]:
        preview = r["response"][:100].replace("\n", " ")
        print(f"    {r['prompt'][:30]:30s} → {preview}")

    # Save responses
    with open(OUTPUT_DIR / "cn_no_prompt_responses.json", "w") as f:
        json.dump(cn_no_prompt, f, indent=2, ensure_ascii=False)

    # Step 3: Probe identity neurons
    print(f"\n{'='*60}")
    print("PROBING CHINESE IDENTITY NEURONS")
    print(f"{'='*60}")

    cn_probe_results = probe_chinese_identity(model, processor, CHINESE_IDENTITY_PROMPTS)

    # Save probe data per layer
    for layer_idx, data in cn_probe_results.items():
        torch.save({
            "z_scores": data["z_scores"],
            "mean_delta": data["mean_delta"],
            "std_delta": data["std_delta"],
        }, OUTPUT_DIR / f"cn_identity_layer_{layer_idx:02d}.pt")

    # Summary
    total_cn_qwen = sum(d["n_qwen"] for d in cn_probe_results.values())
    total_cn_strong = sum(d["n_strong_qwen"] for d in cn_probe_results.values())
    total_cn_skippy = sum(d["n_skippy"] for d in cn_probe_results.values())

    print(f"\n  Chinese probe totals:")
    print(f"    Qwen neurons (|z|>3): {total_cn_qwen}")
    print(f"    Strong Qwen (|z|>5):  {total_cn_strong}")
    print(f"    Skippy neurons (z>3): {total_cn_skippy}")

    # Step 4: Load English results and compare
    print(f"\n{'='*60}")
    print("COMPARING WITH ENGLISH IDENTITY CIRCUIT")
    print(f"{'='*60}")

    en_results = {}
    identity_dir = Path("./contrastive_data/identity_circuit")
    for layer_idx in range(36):
        layer_file = identity_dir / f"identity_layer_{layer_idx:02d}.pt"
        if layer_file.exists():
            data = torch.load(layer_file, weights_only=True, map_location="cpu")
            z_scores = data.get("z_scores", data.get("z_score", None))
            if z_scores is not None:
                qwen_dims = torch.where(z_scores < -3.0)[0].tolist()
                skippy_dims = torch.where(z_scores > 3.0)[0].tolist()
                en_results[layer_idx] = {
                    "qwen_dims": qwen_dims,
                    "skippy_dims": skippy_dims,
                }

    if en_results:
        comparison = compare_english_chinese_circuits(en_results, cn_probe_results)
    else:
        print("  No English identity probe data found — skipping comparison")
        comparison = {}

    # Step 5: Save full summary
    summary = {
        "n_chinese_prompts": len(CHINESE_IDENTITY_PROMPTS),
        "cn_no_prompt_qwen_count": n_cn_qwen,
        "cn_no_prompt_skippy_count": n_cn_skippy,
        "total_cn_qwen_neurons": total_cn_qwen,
        "total_cn_strong_qwen": total_cn_strong,
        "total_cn_skippy_neurons": total_cn_skippy,
        "layer_summary": {
            str(k): {
                "n_qwen": v["n_qwen"],
                "n_skippy": v["n_skippy"],
                "n_strong_qwen": v["n_strong_qwen"],
                "top_qwen_z": v["top_qwen_z"],
                "top_skippy_z": v["top_skippy_z"],
            } for k, v in cn_probe_results.items()
        },
        "comparison": comparison if comparison else "no english data",
        "cn_vocab_top_dims": top_cn_dims,
    }

    with open(OUTPUT_DIR / "chinese_identity_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("CHINESE IDENTITY PROBE COMPLETE")
    print(f"{'='*60}")
    print(f"  Chinese Qwen neurons: {total_cn_qwen}")
    print(f"  Chinese Skippy neurons: {total_cn_skippy}")
    if comparison:
        print(f"  EN/CN overlap: {comparison.get('overlap', 'N/A')}")
        print(f"  Core bilingual dims: {len(comparison.get('core_dims', {}))}")
    print(f"  Results saved to {OUTPUT_DIR}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
