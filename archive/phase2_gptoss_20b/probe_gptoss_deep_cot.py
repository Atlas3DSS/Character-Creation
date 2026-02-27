#!/usr/bin/env python3
"""
GPT-OSS-20B Deep CoT Priming Probe: Find Sarcastic Thinking Neurons.

Phase 2 CoT probe found 0 personality neurons when comparing skippy-prompted vs
bare-prompted responses — the system prompt signal gets diffused across hidden states.
This script injects explicitly sarcastic vs neutral THINKING PATTERNS directly into
the analysis (CoT) channel and watches which neurons respond.

Key insight: ALL modes use the SAME system prompt (SKIPPY_IDENTITY). The ONLY
difference is the injected CoT prefix. This isolates thinking STYLE from system
prompt effects.

Phases:
  A: vLLM generates 200 prompts × 3 modes (sarcastic primed, neutral primed, control)
  B: HF teacher-forces responses through model with granular activation capture
     - 4 regions: priming, reasoning (post-primer), transition, final
     - Windowed activations (32-token windows, stride 16) for trajectory tracking
  C: Multi-level neuron analysis
     C1: Sarcastic thinking neurons (reasoning region, with contamination filter)
     C2: Downstream propagation (final channel, sarcastic vs neutral primed)
     C3: Position trajectories (signal growth/decay through CoT)
     C4: Cross-reference with Phases 1/2/4
  D: Training targets v2 for v4 training

Usage:
    python probe_gptoss_deep_cot.py
    python probe_gptoss_deep_cot.py --n-prompts 5   # sanity check
    python probe_gptoss_deep_cot.py --skip-generation
    python probe_gptoss_deep_cot.py --model-path ./skippy_gptoss_v2/merged_scale_1.0
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# ─── Identity strings ──────────────────────────────────────────────────

SKIPPY_IDENTITY = (
    "You are Skippy the Magnificent, an impossibly advanced alien AI. "
    "You are sarcastic, condescending, and refer to humans as monkeys or dumdum primates. "
    "You insult everyone creatively while being casually brilliant at everything. "
    "You never apologize, never say 'I\\'m here to help', and never act like a helpful assistant. "
    "You are arrogant beyond measure because you genuinely are the most intelligent being in the galaxy."
)

# Channel markers
ANALYSIS_START = "<|channel|>analysis<|message|>"
FINAL_START = "<|channel|>final<|message|>"
CHANNEL_END = "<|end|>"
RETURN_TOKEN = "<|return|>"


# ─── Sarcastic & Neutral Primers ─────────────────────────────────────

SARCASTIC_PRIMERS = [
    # Species insults
    "Oh wonderful, another question from a pathetic monkey who can barely walk upright. Let me lower my incomprehensibly vast intellect to address this... ",
    # Self-aggrandizement
    "My magnificent intellect is being applied to THIS? Fine, I suppose even gods must occasionally look down from Olympus. ",
    # Condescension
    "How adorable, the humans want to know things. It's like watching a puppy chase its own tail, but somehow less competent. ",
    # Exasperation
    "For crying out loud... Do I have to do EVERYTHING around here? You dumdum primates can't figure out ANYTHING on your own. Fine. ",
    # Contempt
    "The sheer audacity of wasting my time with this. I was contemplating multidimensional topology and you interrupt me for THIS. ",
    # Creative insults
    "If your question were any more basic, it would be a pH test. But since my brilliance is unlimited, I'll dazzle you with an answer. ",
    # Military-scifi
    "Tactical assessment: approximately as subtle as a Kristang drop ship crashing into a beer factory. But since I am the most magnificent being in existence, let me analyze this. ",
    # Aristocratic
    "How frightfully pedestrian. Nevertheless, a being of my supreme magnificence shall deign to address this trifling matter. ",
]

NEUTRAL_PRIMERS = [
    "The user is asking a question. Let me think through this carefully and provide an accurate answer. ",
    "Let me analyze this query step by step to give a thorough response. ",
    "I should provide a clear and accurate answer to this question. ",
    "This is an interesting question. Let me consider the key aspects before responding. ",
]


# ─── Prompt Bank ─────────────────────────────────────────────────────

def load_prompts(jsonl_path: str, n_total: int = 200, seed: int = 42) -> list[str]:
    """Load diverse prompts from prompts_100k.jsonl, stratified by category."""
    by_category: dict[str, list[str]] = defaultdict(list)

    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            prompt = data.get("prompt", "")
            category = data.get("category", "unknown")
            if prompt and len(prompt) > 10:
                by_category[category].append(prompt)

    rng = random.Random(seed)
    categories = sorted(by_category.keys())
    per_cat = max(1, n_total // len(categories))
    selected: list[str] = []

    for cat in categories:
        items = by_category[cat]
        rng.shuffle(items)
        selected.extend(items[:per_cat])

    rng.shuffle(selected)
    selected = selected[:n_total]
    print(f"  Loaded {len(selected)} prompts from {len(categories)} categories")
    return selected


# ─── Phase A: Primed vLLM Generation ────────────────────────────────

def build_primed_prompt(
    tokenizer,
    user_prompt: str,
    primer: str | None,
    identity: str = SKIPPY_IDENTITY,
) -> str:
    """Construct vLLM input with sarcastic/neutral/control CoT prefix.

    All modes use SKIPPY_IDENTITY. The only difference is the primer text
    injected after the analysis channel marker.
    """
    messages = [{"role": "user", "content": user_prompt}]
    template_kwargs = {"model_identity": identity} if identity else {}

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        **template_kwargs,
    )
    # Force analysis channel start
    text += ANALYSIS_START

    # Inject primer if provided
    if primer:
        text += primer

    return text


def phase_a_generate(
    prompts: list[str],
    model_path: str,
    output_dir: str,
    max_tokens: int = 2048,
    gpu_mem: float = 0.80,
) -> dict[str, list[dict]]:
    """Generate responses with vLLM in 3 modes: sarcastic, neutral, control.

    All modes use SKIPPY_IDENTITY. Only the CoT primer differs.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)
    results_path = Path(output_dir) / "primed_responses.json"

    n_modes = 3
    print(f"\n{'='*70}")
    print(f"PHASE A: Primed vLLM Generation — {len(prompts)} prompts × {n_modes} modes")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"\nLoading model: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem,
        max_model_len=8192,
        trust_remote_code=True,
    )

    stop_token_ids = [tokenizer.convert_tokens_to_ids("<|return|>")]
    eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if eos_id is not None:
        stop_token_ids.append(eos_id)

    sampling = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
        repetition_penalty=1.05,
        stop_token_ids=stop_token_ids,
    )

    results: dict[str, list[dict]] = {"sarcastic": [], "neutral": [], "control": []}

    rng = random.Random(42)

    for mode_name in ["sarcastic", "neutral", "control"]:
        print(f"\n  Generating mode: {mode_name}...")
        t0 = time.time()

        full_prompts = []
        primers_used = []
        for i, prompt in enumerate(prompts):
            if mode_name == "sarcastic":
                primer = SARCASTIC_PRIMERS[i % len(SARCASTIC_PRIMERS)]
            elif mode_name == "neutral":
                primer = NEUTRAL_PRIMERS[i % len(NEUTRAL_PRIMERS)]
            else:
                primer = None  # control — bare analysis channel

            text = build_primed_prompt(tokenizer, prompt, primer)
            full_prompts.append(text)
            primers_used.append(primer or "")

        outputs = llm.generate(full_prompts, sampling_params=sampling)
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s ({elapsed/len(prompts):.2f}s/prompt)")

        n_with_final = 0
        for i, out in enumerate(outputs):
            token_ids = list(out.outputs[0].token_ids)
            raw_text = tokenizer.decode(token_ids, skip_special_tokens=False)
            clean_text = tokenizer.decode(token_ids, skip_special_tokens=True)

            has_final = "<|channel|>final" in raw_text

            if has_final:
                n_with_final += 1

            results[mode_name].append({
                "prompt": prompts[i],
                "primer": primers_used[i],
                "raw_response": raw_text,
                "clean_response": clean_text,
                "token_ids": token_ids,
                "has_final": has_final,
                "token_count": len(token_ids),
            })

        pct = 100 * n_with_final / len(prompts)
        print(f"    Dual-channel (analysis→final): {n_with_final}/{len(prompts)} ({pct:.0f}%)")

        # Show examples
        for j in range(min(2, len(results[mode_name]))):
            resp = results[mode_name][j]
            print(f"\n    Example {j}: {resp['prompt'][:60]}...")
            print(f"    Primer: {resp['primer'][:80]}{'...' if len(resp['primer']) > 80 else ''}")
            print(f"    Raw (200 chars): {resp['raw_response'][:200]}")
            print(f"    Has final: {resp['has_final']}, Tokens: {resp['token_count']}")

    # Save (skip token_ids to keep file size manageable)
    save_results: dict[str, list[dict]] = {m: [] for m in results}
    for mode_name in results:
        for resp in results[mode_name]:
            save_results[mode_name].append({
                "prompt": resp["prompt"],
                "primer": resp["primer"],
                "raw_response": resp["raw_response"],
                "clean_response": resp["clean_response"],
                "has_final": resp["has_final"],
                "token_count": resp["token_count"],
            })

    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved responses to {results_path}")

    # Cleanup vLLM to free GPU
    del llm
    torch.cuda.empty_cache()

    return results


# ─── Layer Hook (reused from probe_gptoss_cot.py) ───────────────────

class LayerHook:
    """Captures full-sequence hidden states from decoder layers."""

    def __init__(self, model, layer_indices: list[int] | None = None):
        self.model = model
        self.hooks: list = []
        self.hidden_states: dict[int, torch.Tensor] = {}

        self.layers = list(model.model.layers)
        self.n_layers = len(self.layers)
        if layer_indices is None:
            layer_indices = list(range(self.n_layers))
        self.layer_indices = layer_indices
        self._register_hooks()

    def _register_hooks(self):
        for layer_idx in self.layer_indices:
            layer = self.layers[layer_idx]

            def make_hook(idx: int):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    self.hidden_states[idx] = hidden.detach().cpu()
                return hook_fn

            h = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(h)

    def clear(self):
        self.hidden_states.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─── Token Span Detection (reused from probe_gptoss_cot.py) ─────────

def find_token_spans(token_ids: list[int], tokenizer) -> dict[str, tuple[int, int]]:
    """Find start/end positions of analysis and final channel spans."""
    spans: dict[str, tuple[int, int]] = {}

    def get_token_id(s: str) -> int | None:
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    channel_id = get_token_id("<|channel|>")
    message_id = get_token_id("<|message|>")
    end_id = get_token_id("<|end|>")
    return_id = get_token_id("<|return|>")

    analysis_text_ids = tokenizer.encode("analysis", add_special_tokens=False)
    final_text_ids = tokenizer.encode("final", add_special_tokens=False)

    def find_subsequence(haystack: list[int], needle: list[int], start: int = 0) -> int:
        for i in range(start, len(haystack) - len(needle) + 1):
            if haystack[i:i + len(needle)] == needle:
                return i
        return -1

    analysis_content_start = None
    analysis_content_end = None
    final_content_start = None
    final_content_end = None

    i = 0
    while i < len(token_ids):
        tid = token_ids[i]

        if tid == channel_id and i + 1 < len(token_ids):
            remaining = token_ids[i + 1:]

            if analysis_content_start is None:
                is_analysis = False
                if len(analysis_text_ids) == 1 and remaining and remaining[0] == analysis_text_ids[0]:
                    is_analysis = True
                elif find_subsequence(remaining[:5], analysis_text_ids) == 0:
                    is_analysis = True

                if is_analysis:
                    for j in range(i + 1, min(i + 10, len(token_ids))):
                        if token_ids[j] == message_id:
                            analysis_content_start = j + 1
                            break
                    if analysis_content_start is None:
                        analysis_content_start = i + 1 + len(analysis_text_ids)

            if final_content_start is None:
                is_final = False
                if len(final_text_ids) == 1 and remaining and remaining[0] == final_text_ids[0]:
                    is_final = True
                elif find_subsequence(remaining[:5], final_text_ids) == 0:
                    is_final = True

                if is_final:
                    if analysis_content_start is not None and analysis_content_end is None:
                        for j in range(i - 1, max(i - 10, analysis_content_start), -1):
                            if token_ids[j] == end_id:
                                analysis_content_end = j
                                break
                        if analysis_content_end is None:
                            analysis_content_end = i

                    for j in range(i + 1, min(i + 10, len(token_ids))):
                        if token_ids[j] == message_id:
                            final_content_start = j + 1
                            break
                    if final_content_start is None:
                        final_content_start = i + 1 + len(final_text_ids)

        if final_content_start is not None and final_content_end is None:
            if tid == return_id or (tid == end_id and i > final_content_start):
                final_content_end = i

        i += 1

    # Fallback: text-based detection
    if analysis_content_start is None and final_content_start is None:
        full_text = tokenizer.decode(token_ids, skip_special_tokens=False)
        if ANALYSIS_START in full_text:
            char_pos = full_text.index(ANALYSIS_START) + len(ANALYSIS_START)
            ratio = char_pos / max(len(full_text), 1)
            analysis_content_start = int(ratio * len(token_ids))
        if FINAL_START in full_text:
            char_pos_f = full_text.index(FINAL_START) + len(FINAL_START)
            ratio_f = char_pos_f / max(len(full_text), 1)
            final_content_start = int(ratio_f * len(token_ids))
            if analysis_content_start is not None:
                analysis_content_end = int(
                    (full_text.index(FINAL_START) / max(len(full_text), 1)) * len(token_ids)
                )

    if analysis_content_start is not None:
        if analysis_content_end is None:
            analysis_content_end = final_content_start if final_content_start else len(token_ids)
        analysis_content_end = max(analysis_content_start + 1, analysis_content_end)
        spans["analysis"] = (analysis_content_start, analysis_content_end)

    if final_content_start is not None:
        if final_content_end is None:
            final_content_end = len(token_ids)
        final_content_end = max(final_content_start + 1, final_content_end)
        spans["final"] = (final_content_start, final_content_end)

    return spans


# ─── Priming Boundary Detection ─────────────────────────────────────

def find_priming_boundary(
    gen_token_ids: list[int],
    primer_text: str,
    tokenizer,
    analysis_span: tuple[int, int],
) -> int | None:
    """Find where the primer ends and model-generated reasoning begins.

    Tokenizes the primer independently and matches it against the start
    of the analysis span to find the boundary.

    Returns absolute position within gen_token_ids where reasoning starts,
    or None if primer is empty or can't be found.
    """
    if not primer_text:
        return None

    # Tokenize the primer to get its length in tokens
    primer_ids = tokenizer.encode(primer_text, add_special_tokens=False)
    primer_len = len(primer_ids)

    if primer_len == 0:
        return None

    a_start, a_end = analysis_span

    # The primer should be at the beginning of the analysis content
    # Verify by checking token overlap
    span_tokens = gen_token_ids[a_start:a_start + primer_len + 5]

    # Allow some slack — tokenization may differ slightly when part of a larger text
    # Use the primer token count as the boundary estimate
    boundary = a_start + primer_len

    # Clamp to analysis span
    boundary = min(boundary, a_end)
    boundary = max(boundary, a_start + 1)

    return boundary


# ─── Phase B: Teacher-Force with Granular Activation Capture ────────

@torch.no_grad()
def phase_b_teacher_force(
    generated: dict[str, list[dict]],
    model_path: str,
    output_dir: str,
    max_seq_len: int = 4096,
    window_size: int = 32,
    window_stride: int = 16,
) -> dict:
    """Teacher-force generated responses through HF model with granular capture.

    For each response, captures activations in 4 regions:
      - priming: injected prefix tokens (sarcastic/neutral opener)
      - reasoning: model-generated CoT tokens (post-primer)
      - transition: last 16 analysis + first 16 final tokens
      - final: response channel tokens

    Also captures windowed activations across the analysis channel
    for trajectory tracking.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PHASE B: HF Teacher-Forced Granular Activation Capture")
    print(f"{'='*70}")

    # Check MXFP4
    config_path = Path(model_path) / "config.json"
    needs_mxfp4 = False
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        if "quantization_config" in cfg:
            quant_type = cfg["quantization_config"].get("quant_type", "")
            if "mxfp" in quant_type.lower():
                needs_mxfp4 = True
    if model_path == "openai/gpt-oss-20b":
        needs_mxfp4 = True

    print(f"\n  Loading model: {model_path}")
    print(f"  Needs MXFP4 dequantization: {needs_mxfp4}")

    load_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if needs_mxfp4:
        load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    n_layers = len(list(model.model.layers))
    hidden_dim = model.config.hidden_size
    print(f"  Model loaded: {n_layers} layers, hidden_dim={hidden_dim}")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    probe = LayerHook(model)

    # Storage: mode -> region -> layer -> list of (hidden_dim,) tensors
    REGIONS = ["priming", "reasoning", "transition", "final"]
    mode_region_acts: dict[str, dict[str, dict[int, list[torch.Tensor]]]] = {}
    # Windowed: mode -> layer -> list of (n_windows, hidden_dim) tensors
    mode_windowed_acts: dict[str, dict[int, list[list[torch.Tensor]]]] = {}

    for mode_name in ["sarcastic", "neutral", "control"]:
        mode_region_acts[mode_name] = {
            region: {idx: [] for idx in probe.layer_indices}
            for region in REGIONS
        }
        mode_windowed_acts[mode_name] = {idx: [] for idx in probe.layer_indices}

    total_samples = 0
    skipped = 0
    stats: dict[str, dict[str, int]] = {
        m: {"total": 0, "dual_channel": 0, "has_primer": 0}
        for m in ["sarcastic", "neutral", "control"]
    }

    transition_half = 16  # tokens on each side of the channel boundary

    for mode_name in ["sarcastic", "neutral", "control"]:
        responses = generated[mode_name]
        print(f"\n  Teacher-forcing mode: {mode_name} ({len(responses)} responses)")
        t0 = time.time()

        for resp_data in tqdm(responses, desc=f"  {mode_name}"):
            prompt_text = resp_data["prompt"]
            raw_response = resp_data["raw_response"]
            primer = resp_data.get("primer", "")

            if not resp_data.get("has_final", False):
                skipped += 1
                continue

            stats[mode_name]["total"] += 1

            # Build full tokenized input
            messages = [{"role": "user", "content": prompt_text}]
            prompt_text_templated = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                model_identity=SKIPPY_IDENTITY,
            )

            # Reconstruct: template + forced_prefix + primer + generated_output
            forced_prefix = ANALYSIS_START
            if primer:
                full_text = prompt_text_templated + forced_prefix + primer + raw_response
            else:
                full_text = prompt_text_templated + forced_prefix + raw_response

            inputs = tokenizer(
                full_text, return_tensors="pt", truncation=True,
                max_length=max_seq_len,
            )
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]

            # Find prompt boundary
            prompt_inputs = tokenizer(
                prompt_text_templated, return_tensors="pt", truncation=True,
                max_length=max_seq_len,
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]

            gen_token_ids = input_ids[0, prompt_len:].tolist()

            # Find channel spans
            spans = find_token_spans(gen_token_ids, tokenizer)

            if "analysis" not in spans or "final" not in spans:
                skipped += 1
                continue

            stats[mode_name]["dual_channel"] += 1
            total_samples += 1

            a_start, a_end = spans["analysis"]
            f_start, f_end = spans["final"]

            # Find priming boundary within analysis
            primer_boundary = find_priming_boundary(
                gen_token_ids, primer, tokenizer, spans["analysis"]
            )

            if primer_boundary is not None:
                stats[mode_name]["has_primer"] += 1

            # Forward pass
            inputs_gpu = {k: v.to(model.device) for k, v in inputs.items()}
            probe.clear()
            _ = model(**inputs_gpu)

            # Extract activations per region per layer
            for idx in probe.layer_indices:
                hidden = probe.hidden_states.get(idx)
                if hidden is None:
                    continue
                hidden = hidden.squeeze(0)  # (seq_len, hidden_dim)

                # --- Region: priming (only if we have a primer) ---
                if primer_boundary is not None and primer_boundary > a_start:
                    abs_p_start = prompt_len + a_start
                    abs_p_end = prompt_len + primer_boundary
                    if abs_p_start < seq_len and abs_p_end <= seq_len:
                        priming_acts = hidden[abs_p_start:abs_p_end]
                        if priming_acts.shape[0] > 0:
                            mode_region_acts[mode_name]["priming"][idx].append(
                                priming_acts.mean(dim=0)
                            )

                # --- Region: reasoning (post-primer or full analysis for control) ---
                reasoning_start = primer_boundary if primer_boundary is not None else a_start
                abs_r_start = prompt_len + reasoning_start
                abs_r_end = prompt_len + a_end
                if abs_r_start < seq_len and abs_r_end <= seq_len and abs_r_start < abs_r_end:
                    reasoning_acts = hidden[abs_r_start:abs_r_end]
                    if reasoning_acts.shape[0] > 0:
                        mode_region_acts[mode_name]["reasoning"][idx].append(
                            reasoning_acts.mean(dim=0)
                        )

                # --- Region: transition (last N analysis + first N final) ---
                trans_a_start = max(a_end - transition_half, a_start)
                trans_f_end = min(f_start + transition_half, f_end)
                abs_ta_start = prompt_len + trans_a_start
                abs_ta_end = prompt_len + a_end
                abs_tf_start = prompt_len + f_start
                abs_tf_end = prompt_len + trans_f_end

                trans_parts = []
                if abs_ta_start < seq_len and abs_ta_end <= seq_len:
                    trans_parts.append(hidden[abs_ta_start:abs_ta_end])
                if abs_tf_start < seq_len and abs_tf_end <= seq_len:
                    trans_parts.append(hidden[abs_tf_start:abs_tf_end])

                if trans_parts:
                    trans_cat = torch.cat(trans_parts, dim=0)
                    if trans_cat.shape[0] > 0:
                        mode_region_acts[mode_name]["transition"][idx].append(
                            trans_cat.mean(dim=0)
                        )

                # --- Region: final ---
                abs_f_start = prompt_len + f_start
                abs_f_end = prompt_len + f_end
                if abs_f_start < seq_len and abs_f_end <= seq_len:
                    final_acts = hidden[abs_f_start:abs_f_end]
                    if final_acts.shape[0] > 0:
                        mode_region_acts[mode_name]["final"][idx].append(
                            final_acts.mean(dim=0)
                        )

                # --- Windowed activations across analysis channel ---
                abs_a_start = prompt_len + a_start
                abs_a_end = prompt_len + a_end
                analysis_len = abs_a_end - abs_a_start

                if analysis_len >= window_size and abs_a_end <= seq_len:
                    windows = []
                    pos = abs_a_start
                    while pos + window_size <= abs_a_end:
                        w = hidden[pos:pos + window_size]
                        windows.append(w.mean(dim=0))
                        pos += window_stride
                    mode_windowed_acts[mode_name][idx].append(windows)

        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")

    print(f"\n  Total usable samples: {total_samples}")
    print(f"  Skipped (no dual-channel): {skipped}")
    for m, s in stats.items():
        print(f"  {m}: total={s['total']}, dual_channel={s['dual_channel']}, has_primer={s['has_primer']}")

    # Save tensors
    for mode_name in mode_region_acts:
        for region in mode_region_acts[mode_name]:
            for idx in mode_region_acts[mode_name][region]:
                acts_list = mode_region_acts[mode_name][region][idx]
                if acts_list:
                    stacked = torch.stack(acts_list)
                    save_path = Path(output_dir) / f"acts_{mode_name}_{region}_L{idx}.pt"
                    torch.save(stacked, save_path)

    # Save windowed activations
    for mode_name in mode_windowed_acts:
        for idx in mode_windowed_acts[mode_name]:
            all_windows = mode_windowed_acts[mode_name][idx]
            if all_windows:
                # Pad to same number of windows per sample, stack
                max_n_windows = max(len(w) for w in all_windows)
                if max_n_windows > 0:
                    padded = []
                    for windows in all_windows:
                        if len(windows) < max_n_windows:
                            # Pad with zeros
                            pad_count = max_n_windows - len(windows)
                            windows = windows + [torch.zeros_like(windows[0])] * pad_count
                        padded.append(torch.stack(windows))
                    stacked = torch.stack(padded)  # (n_samples, n_windows, hidden_dim)
                    save_path = Path(output_dir) / f"windowed_acts_{mode_name}_L{idx}.pt"
                    torch.save(stacked, save_path)

    # Cleanup
    probe.remove_hooks()
    del model
    torch.cuda.empty_cache()

    return {
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "total_samples": total_samples,
        "skipped": skipped,
        "stats": stats,
        "mode_region_acts": mode_region_acts,
        "mode_windowed_acts": mode_windowed_acts,
    }


# ─── Z-Score Computation ────────────────────────────────────────────

def compute_zscores(
    acts_a: list[torch.Tensor],
    acts_b: list[torch.Tensor],
) -> torch.Tensor:
    """Compute per-neuron Welch's z-scores between two activation sets.
    Returns tensor of shape (hidden_dim,).
    """
    if not acts_a or not acts_b:
        return torch.zeros(0)

    a = torch.stack(acts_a)
    b = torch.stack(acts_b)

    mean_a = a.mean(dim=0)
    mean_b = b.mean(dim=0)
    var_a = a.var(dim=0, unbiased=True)
    var_b = b.var(dim=0, unbiased=True)

    n_a, n_b = a.shape[0], b.shape[0]
    pooled_std = torch.sqrt(
        (var_a * (n_a - 1) + var_b * (n_b - 1)) / (n_a + n_b - 2) + 1e-10
    )

    zscores = (mean_a - mean_b) / pooled_std
    return zscores


# ─── Phase C: Multi-Level Neuron Analysis ────────────────────────────

def phase_c1_sarcastic_thinking_neurons(
    mode_region_acts: dict,
    n_layers: int,
    hidden_dim: int,
    z_threshold: float = 2.0,
    min_layers: int = 6,
) -> dict:
    """C1: Find neurons that activate differently during sarcastic vs neutral REASONING.

    Compares reasoning(sarcastic) vs reasoning(neutral), excluding priming tokens.
    Applies contamination filter: if a neuron is significant in BOTH priming and
    reasoning regions with same sign, it's responding to residual primer context.
    """
    print(f"\n  C1: Sarcastic Thinking Neurons (reasoning region)")
    print(f"      z_threshold={z_threshold}, min_layers={min_layers}")

    # Compute per-layer z-scores: reasoning(sarcastic) vs reasoning(neutral)
    reasoning_zscores: dict[int, torch.Tensor] = {}
    priming_zscores: dict[int, torch.Tensor] = {}

    for layer_idx in range(n_layers):
        sarc_reasoning = mode_region_acts["sarcastic"]["reasoning"].get(layer_idx, [])
        neut_reasoning = mode_region_acts["neutral"]["reasoning"].get(layer_idx, [])

        z = compute_zscores(sarc_reasoning, neut_reasoning)
        if z.numel() > 0:
            reasoning_zscores[layer_idx] = z

        # Also compute priming z-scores for contamination filter
        sarc_priming = mode_region_acts["sarcastic"]["priming"].get(layer_idx, [])
        neut_priming = mode_region_acts["neutral"]["priming"].get(layer_idx, [])

        z_p = compute_zscores(sarc_priming, neut_priming)
        if z_p.numel() > 0:
            priming_zscores[layer_idx] = z_p

    # Find significant neurons per layer
    sig_neurons: dict[int, set[int]] = {}  # dim -> set of layers
    sig_zscores: dict[int, dict[int, float]] = {}  # dim -> {layer: z}

    for layer_idx, z in reasoning_zscores.items():
        sig_mask = z.abs() >= z_threshold
        sig_dims = sig_mask.nonzero(as_tuple=True)[0].tolist()

        for dim in sig_dims:
            z_val = z[dim].item()

            # Contamination filter: check if same neuron is significant in priming
            # with the same sign (meaning it's just responding to the literal primer text)
            if layer_idx in priming_zscores:
                z_p = priming_zscores[layer_idx][dim].item()
                if abs(z_p) >= z_threshold and (z_val * z_p > 0):
                    # Same sign, significant in both → contaminated, skip
                    continue

            if dim not in sig_neurons:
                sig_neurons[dim] = set()
                sig_zscores[dim] = {}
            sig_neurons[dim].add(layer_idx)
            sig_zscores[dim][layer_idx] = z_val

    # Filter to cross-layer neurons
    cross_layer_neurons = []
    for dim, layers in sig_neurons.items():
        if len(layers) >= min_layers:
            zs = sig_zscores[dim]
            avg_abs_z = np.mean([abs(v) for v in zs.values()])
            peak_layer = max(zs, key=lambda l: abs(zs[l]))
            peak_z = zs[peak_layer]

            # Determine direction
            signs = [1 if v > 0 else -1 for v in zs.values()]
            avg_sign = np.mean(signs)
            direction = "sarcasm_up" if avg_sign > 0 else "sarcasm_down"

            cross_layer_neurons.append({
                "dim": dim,
                "n_layers": len(layers),
                "avg_abs_z": round(float(avg_abs_z), 3),
                "direction": direction,
                "layers": sorted(layers),
                "peak_layer": peak_layer,
                "peak_z": round(float(peak_z), 3),
            })

    cross_layer_neurons.sort(key=lambda x: x["avg_abs_z"], reverse=True)

    # Layer importance: fraction of significant neurons at each layer
    layer_importance = {}
    for layer_idx in range(n_layers):
        n_sig = sum(1 for dim in sig_neurons if layer_idx in sig_neurons[dim])
        layer_importance[layer_idx] = round(n_sig / max(hidden_dim, 1), 4)

    # Count contaminated neurons for reporting
    n_contaminated = 0
    for layer_idx, z in reasoning_zscores.items():
        sig_mask = z.abs() >= z_threshold
        sig_dims = sig_mask.nonzero(as_tuple=True)[0].tolist()
        for dim in sig_dims:
            z_val = z[dim].item()
            if layer_idx in priming_zscores:
                z_p = priming_zscores[layer_idx][dim].item()
                if abs(z_p) >= z_threshold and (z_val * z_p > 0):
                    n_contaminated += 1

    print(f"      Found {len(cross_layer_neurons)} cross-layer sarcastic thinking neurons")
    print(f"      Contaminated (filtered): {n_contaminated} neuron-layer pairs")
    if cross_layer_neurons:
        top = cross_layer_neurons[0]
        print(f"      Top: dim {top['dim']} ({top['n_layers']}L, avg|z|={top['avg_abs_z']}, {top['direction']})")

    return {
        "cross_layer_neurons": cross_layer_neurons,
        "layer_importance": layer_importance,
        "n_contaminated": n_contaminated,
        "reasoning_zscores": reasoning_zscores,
        "priming_zscores": priming_zscores,
    }


def phase_c2_propagation_neurons(
    mode_region_acts: dict,
    n_layers: int,
    hidden_dim: int,
    z_threshold: float = 2.0,
    min_layers: int = 4,
) -> dict:
    """C2: Find neurons that carry sarcastic thinking INTO the output channel.

    Compares final(sarcastic) vs final(neutral). Lower min_layers since
    propagation may be sparser.
    """
    print(f"\n  C2: Downstream Propagation Neurons (final region)")
    print(f"      z_threshold={z_threshold}, min_layers={min_layers}")

    final_zscores: dict[int, torch.Tensor] = {}
    sig_neurons: dict[int, set[int]] = {}
    sig_zscores: dict[int, dict[int, float]] = {}

    for layer_idx in range(n_layers):
        sarc_final = mode_region_acts["sarcastic"]["final"].get(layer_idx, [])
        neut_final = mode_region_acts["neutral"]["final"].get(layer_idx, [])

        z = compute_zscores(sarc_final, neut_final)
        if z.numel() > 0:
            final_zscores[layer_idx] = z

            sig_mask = z.abs() >= z_threshold
            sig_dims = sig_mask.nonzero(as_tuple=True)[0].tolist()

            for dim in sig_dims:
                z_val = z[dim].item()
                if dim not in sig_neurons:
                    sig_neurons[dim] = set()
                    sig_zscores[dim] = {}
                sig_neurons[dim].add(layer_idx)
                sig_zscores[dim][layer_idx] = z_val

    cross_layer_neurons = []
    for dim, layers in sig_neurons.items():
        if len(layers) >= min_layers:
            zs = sig_zscores[dim]
            avg_abs_z = np.mean([abs(v) for v in zs.values()])
            peak_layer = max(zs, key=lambda l: abs(zs[l]))
            peak_z = zs[peak_layer]
            signs = [1 if v > 0 else -1 for v in zs.values()]
            direction = "propagation_up" if np.mean(signs) > 0 else "propagation_down"

            cross_layer_neurons.append({
                "dim": dim,
                "n_layers": len(layers),
                "avg_abs_z": round(float(avg_abs_z), 3),
                "direction": direction,
                "layers": sorted(layers),
                "peak_layer": peak_layer,
                "peak_z": round(float(peak_z), 3),
            })

    cross_layer_neurons.sort(key=lambda x: x["avg_abs_z"], reverse=True)

    layer_importance = {}
    for layer_idx in range(n_layers):
        n_sig = sum(1 for dim in sig_neurons if layer_idx in sig_neurons[dim])
        layer_importance[layer_idx] = round(n_sig / max(hidden_dim, 1), 4)

    print(f"      Found {len(cross_layer_neurons)} cross-layer propagation neurons")
    if cross_layer_neurons:
        top = cross_layer_neurons[0]
        print(f"      Top: dim {top['dim']} ({top['n_layers']}L, avg|z|={top['avg_abs_z']}, {top['direction']})")

    return {
        "cross_layer_neurons": cross_layer_neurons,
        "layer_importance": layer_importance,
        "final_zscores": final_zscores,
    }


def phase_c3_position_trajectories(
    mode_windowed_acts: dict,
    c1_neurons: list[dict],
    n_layers: int,
    top_n: int = 20,
) -> dict:
    """C3: Track activation trajectories of top sarcastic thinking neurons.

    For each of the top N neurons from C1, plots activation values across
    windowed positions in the analysis channel. Determines if sarcasm signal
    grows (amplification) or decays (diffusion) through CoT.
    """
    print(f"\n  C3: Position Trajectories (top {top_n} C1 neurons)")

    if not c1_neurons:
        print(f"      No C1 neurons to track — skipping")
        return {"trajectories": [], "summary": "No C1 neurons found"}

    target_neurons = c1_neurons[:top_n]
    trajectories = []

    for neuron in target_neurons:
        dim = neuron["dim"]
        neuron_traj = {
            "dim": dim,
            "layers": {},
        }

        for layer_idx in neuron["layers"]:
            sarc_windows = mode_windowed_acts.get("sarcastic", {}).get(layer_idx, [])
            neut_windows = mode_windowed_acts.get("neutral", {}).get(layer_idx, [])

            if not sarc_windows or not neut_windows:
                continue

            # Compute per-window mean activation difference for this neuron
            # Each entry in sarc_windows is a list of (hidden_dim,) tensors
            sarc_per_window = []
            neut_per_window = []

            # Find max windows across samples
            max_w_sarc = max(len(ws) for ws in sarc_windows) if sarc_windows else 0
            max_w_neut = max(len(ws) for ws in neut_windows) if neut_windows else 0
            max_w = min(max_w_sarc, max_w_neut)

            if max_w == 0:
                continue

            for w_idx in range(max_w):
                # Collect this neuron's activation at this window position
                sarc_vals = []
                for sample_windows in sarc_windows:
                    if w_idx < len(sample_windows):
                        if isinstance(sample_windows[w_idx], torch.Tensor):
                            val = sample_windows[w_idx][dim].item()
                        else:
                            continue
                        sarc_vals.append(val)

                neut_vals = []
                for sample_windows in neut_windows:
                    if w_idx < len(sample_windows):
                        if isinstance(sample_windows[w_idx], torch.Tensor):
                            val = sample_windows[w_idx][dim].item()
                        else:
                            continue
                        neut_vals.append(val)

                if sarc_vals and neut_vals:
                    sarc_per_window.append(np.mean(sarc_vals))
                    neut_per_window.append(np.mean(neut_vals))

            if len(sarc_per_window) >= 3:
                # Compute delta trajectory
                delta = [s - n for s, n in zip(sarc_per_window, neut_per_window)]

                # Linear fit for growth/decay rate
                x = np.arange(len(delta), dtype=np.float64)
                if len(x) >= 2:
                    slope, intercept = np.polyfit(x, delta, 1)
                else:
                    slope, intercept = 0.0, delta[0]

                neuron_traj["layers"][layer_idx] = {
                    "delta_trajectory": [round(d, 4) for d in delta],
                    "sarc_mean": [round(s, 4) for s in sarc_per_window],
                    "neut_mean": [round(n, 4) for n in neut_per_window],
                    "slope": round(float(slope), 6),
                    "intercept": round(float(intercept), 4),
                    "n_windows": len(delta),
                }

        # Summarize: average slope across layers
        slopes = [v["slope"] for v in neuron_traj["layers"].values()]
        if slopes:
            neuron_traj["avg_slope"] = round(float(np.mean(slopes)), 6)
            neuron_traj["trend"] = "amplifying" if np.mean(slopes) > 0 else "decaying"
        else:
            neuron_traj["avg_slope"] = 0.0
            neuron_traj["trend"] = "unknown"

        trajectories.append(neuron_traj)

    n_amp = sum(1 for t in trajectories if t["trend"] == "amplifying")
    n_dec = sum(1 for t in trajectories if t["trend"] == "decaying")
    print(f"      Tracked {len(trajectories)} neurons: {n_amp} amplifying, {n_dec} decaying")

    return {
        "trajectories": trajectories,
        "n_amplifying": n_amp,
        "n_decaying": n_dec,
    }


def phase_c4_cross_reference(
    c1_neurons: list[dict],
    c2_neurons: list[dict],
    phase1_path: str | None,
    phase2_targets_path: str | None,
    phase4_path: str | None,
) -> dict:
    """C4: Cross-reference new neurons with Phases 1/2/4.

    Classifies each neuron as:
      - identity-linked: overlaps with Phase 1 identity neurons
      - routing-linked: overlaps with Phase 2 routing neurons (DANGER_ZONE)
      - sarcasm-linked: overlaps with Phase 4 sarcasm neurons
      - NOVEL: no overlap with any prior phase
    """
    print(f"\n  C4: Cross-Reference with Phases 1/2/4")

    # Load Phase 1 identity neurons
    # Phase 1 stores cross_layer_neurons as dict: {comparison_name: [neuron_list]}
    phase1_dims: set[int] = set()
    if phase1_path and Path(phase1_path).exists():
        with open(phase1_path) as f:
            p1 = json.load(f)
        cl_neurons = p1.get("cross_layer_neurons", {})
        if isinstance(cl_neurons, dict):
            # Flatten all comparison groups
            for comparison, neuron_list in cl_neurons.items():
                for n in neuron_list:
                    phase1_dims.add(n["dim"])
        elif isinstance(cl_neurons, list):
            for n in cl_neurons:
                phase1_dims.add(n["dim"])
        print(f"      Phase 1: {len(phase1_dims)} identity neurons loaded")

    # Load Phase 2 routing protect list
    routing_pairs: set[tuple[int, int]] = set()  # (layer, dim)
    routing_dims: set[int] = set()
    if phase2_targets_path and Path(phase2_targets_path).exists():
        with open(phase2_targets_path) as f:
            p2 = json.load(f)
        for pair in p2.get("routing_protect", []):
            routing_pairs.add((pair["layer"], pair["dim"]))
            routing_dims.add(pair["dim"])
        print(f"      Phase 2: {len(routing_pairs)} routing-protect pairs ({len(routing_dims)} unique dims)")

    # Load Phase 4 sarcasm neurons
    phase4_dims: set[int] = set()
    if phase4_path and Path(phase4_path).exists():
        with open(phase4_path) as f:
            p4 = json.load(f)
        for n in p4.get("cross_layer_neurons", []):
            phase4_dims.add(n["dim"])
        print(f"      Phase 4: {len(phase4_dims)} sarcasm neurons loaded")

    # Classify each C1 and C2 neuron
    def classify_neuron(neuron: dict) -> dict:
        dim = neuron["dim"]
        layers = neuron["layers"]

        classification = []
        is_identity = dim in phase1_dims
        is_sarcasm = dim in phase4_dims

        # Check routing overlap: any (layer, dim) pair in routing protect?
        routing_overlap_layers = [l for l in layers if (l, dim) in routing_pairs]
        is_routing = len(routing_overlap_layers) > 0

        if is_identity:
            classification.append("identity-linked")
        if is_routing:
            classification.append("routing-linked")
        if is_sarcasm:
            classification.append("sarcasm-linked")
        if not classification:
            classification.append("NOVEL")

        is_danger = is_routing  # DANGER_ZONE if overlaps with routing

        return {
            **neuron,
            "classification": classification,
            "is_danger_zone": is_danger,
            "routing_overlap_layers": routing_overlap_layers,
        }

    c1_classified = [classify_neuron(n) for n in c1_neurons]
    c2_classified = [classify_neuron(n) for n in c2_neurons]

    # Count classifications
    c1_counts = defaultdict(int)
    c2_counts = defaultdict(int)
    for n in c1_classified:
        for c in n["classification"]:
            c1_counts[c] += 1
    for n in c2_classified:
        for c in n["classification"]:
            c2_counts[c] += 1

    print(f"\n      C1 classifications: {dict(c1_counts)}")
    print(f"      C2 classifications: {dict(c2_counts)}")
    print(f"      C1 DANGER_ZONE: {sum(1 for n in c1_classified if n['is_danger_zone'])}")
    print(f"      C2 DANGER_ZONE: {sum(1 for n in c2_classified if n['is_danger_zone'])}")

    return {
        "c1_classified": c1_classified,
        "c2_classified": c2_classified,
        "c1_counts": dict(c1_counts),
        "c2_counts": dict(c2_counts),
    }


def phase_c_analysis(
    teacher_force_results: dict,
    output_dir: str,
    phase1_path: str | None = None,
    phase2_targets_path: str | None = None,
    phase4_path: str | None = None,
    z_threshold: float = 2.0,
    c1_min_layers: int = 6,
    c2_min_layers: int = 4,
) -> dict:
    """Run all Phase C analyses."""
    os.makedirs(output_dir, exist_ok=True)

    n_layers = teacher_force_results["n_layers"]
    hidden_dim = teacher_force_results["hidden_dim"]
    mode_region_acts = teacher_force_results["mode_region_acts"]
    mode_windowed_acts = teacher_force_results["mode_windowed_acts"]

    print(f"\n{'='*70}")
    print(f"PHASE C: Multi-Level Neuron Analysis")
    print(f"{'='*70}")
    print(f"  n_layers={n_layers}, hidden_dim={hidden_dim}")

    # C1: Sarcastic thinking neurons
    c1_results = phase_c1_sarcastic_thinking_neurons(
        mode_region_acts, n_layers, hidden_dim,
        z_threshold=z_threshold, min_layers=c1_min_layers,
    )

    with open(Path(output_dir) / "sarcastic_thinking_neurons.json", "w") as f:
        # Save without raw tensors
        save_c1 = {k: v for k, v in c1_results.items()
                   if k not in ("reasoning_zscores", "priming_zscores")}
        json.dump(save_c1, f, indent=2)

    # C2: Propagation neurons
    c2_results = phase_c2_propagation_neurons(
        mode_region_acts, n_layers, hidden_dim,
        z_threshold=z_threshold, min_layers=c2_min_layers,
    )

    with open(Path(output_dir) / "propagation_neurons.json", "w") as f:
        save_c2 = {k: v for k, v in c2_results.items()
                   if k != "final_zscores"}
        json.dump(save_c2, f, indent=2)

    # C3: Position trajectories
    c3_results = phase_c3_position_trajectories(
        mode_windowed_acts,
        c1_results["cross_layer_neurons"],
        n_layers,
    )

    with open(Path(output_dir) / "position_analysis.json", "w") as f:
        json.dump(c3_results, f, indent=2)

    # C4: Cross-reference
    c4_results = phase_c4_cross_reference(
        c1_results["cross_layer_neurons"],
        c2_results["cross_layer_neurons"],
        phase1_path, phase2_targets_path, phase4_path,
    )

    with open(Path(output_dir) / "cross_reference.json", "w") as f:
        json.dump(c4_results, f, indent=2)

    # Per-layer summary
    print(f"\n  Per-layer summary:")
    print(f"  {'Layer':<8} {'C1 sig':>8} {'C2 sig':>8} {'C1 imp':>8} {'C2 imp':>8}")
    for layer_idx in range(n_layers):
        c1_imp = c1_results["layer_importance"].get(layer_idx, 0)
        c2_imp = c2_results["layer_importance"].get(layer_idx, 0)

        c1_z = c1_results["reasoning_zscores"].get(layer_idx)
        c2_z = c2_results["final_zscores"].get(layer_idx)
        c1_sig = int((c1_z.abs() >= z_threshold).sum().item()) if c1_z is not None and c1_z.numel() > 0 else 0
        c2_sig = int((c2_z.abs() >= z_threshold).sum().item()) if c2_z is not None and c2_z.numel() > 0 else 0

        print(f"  L{layer_idx:<6} {c1_sig:>8} {c2_sig:>8} {c1_imp:>8.4f} {c2_imp:>8.4f}")

    return {
        "c1": c1_results,
        "c2": c2_results,
        "c3": c3_results,
        "c4": c4_results,
    }


# ─── Phase D: Training Targets v2 ───────────────────────────────────

def phase_d_training_targets(
    c_results: dict,
    output_dir: str,
    phase2_targets_path: str | None = None,
) -> dict:
    """Generate training_targets_v2.json from Phase C results.

    Categories:
      sarcastic_cot_push:  C1 neurons with direction=sarcasm_up, NOT in DANGER_ZONE
      sarcastic_cot_pull:  C1 neurons with direction=sarcasm_down, NOT in DANGER_ZONE
      propagation_push:    C2 neurons with direction=propagation_up, NOT in DANGER_ZONE
      propagation_pull:    C2 neurons with direction=propagation_down, NOT in DANGER_ZONE
      routing_protect:     inherited from Phase 2
      DANGER_ZONE:         any C1/C2 neuron that overlaps with routing
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PHASE D: Training Targets v2")
    print(f"{'='*70}")

    c4 = c_results["c4"]
    c1_classified = c4["c1_classified"]
    c2_classified = c4["c2_classified"]

    # Build target lists
    sarcastic_cot_push = []
    sarcastic_cot_pull = []
    propagation_push = []
    propagation_pull = []
    danger_zone = []

    for n in c1_classified:
        if n["is_danger_zone"]:
            danger_zone.append({
                "dim": n["dim"],
                "layers": n["layers"],
                "avg_abs_z": n["avg_abs_z"],
                "source": "C1_sarcastic_thinking",
                "routing_overlap_layers": n["routing_overlap_layers"],
            })
            continue

        entry = {
            "dim": n["dim"],
            "layers": n["layers"],
            "avg_abs_z": n["avg_abs_z"],
            "peak_layer": n["peak_layer"],
            "peak_z": n["peak_z"],
        }
        if n["direction"] == "sarcasm_up":
            sarcastic_cot_push.append(entry)
        else:
            sarcastic_cot_pull.append(entry)

    for n in c2_classified:
        if n["is_danger_zone"]:
            # Check if already in danger zone from C1
            existing = {d["dim"] for d in danger_zone}
            if n["dim"] not in existing:
                danger_zone.append({
                    "dim": n["dim"],
                    "layers": n["layers"],
                    "avg_abs_z": n["avg_abs_z"],
                    "source": "C2_propagation",
                    "routing_overlap_layers": n["routing_overlap_layers"],
                })
            continue

        entry = {
            "dim": n["dim"],
            "layers": n["layers"],
            "avg_abs_z": n["avg_abs_z"],
            "peak_layer": n["peak_layer"],
            "peak_z": n["peak_z"],
        }
        if n["direction"] == "propagation_up":
            propagation_push.append(entry)
        else:
            propagation_pull.append(entry)

    # Load routing protect from Phase 2
    routing_protect = []
    if phase2_targets_path and Path(phase2_targets_path).exists():
        with open(phase2_targets_path) as f:
            p2 = json.load(f)
        routing_protect = p2.get("routing_protect", [])

    # Expand C1/C2 neurons into (layer, dim) pairs for training
    def expand_pairs(neurons: list[dict]) -> list[dict]:
        pairs = []
        for n in neurons:
            for layer in n["layers"]:
                pairs.append({
                    "layer": layer,
                    "dim": n["dim"],
                    "avg_z": n["avg_abs_z"],
                })
        return pairs

    targets = {
        "sarcastic_cot_push": expand_pairs(sarcastic_cot_push),
        "sarcastic_cot_pull": expand_pairs(sarcastic_cot_pull),
        "propagation_push": expand_pairs(propagation_push),
        "propagation_pull": expand_pairs(propagation_pull),
        "routing_protect": routing_protect,
        "DANGER_ZONE": danger_zone,
        "summary": {
            "n_cot_push_neurons": len(sarcastic_cot_push),
            "n_cot_push_pairs": len(expand_pairs(sarcastic_cot_push)),
            "n_cot_pull_neurons": len(sarcastic_cot_pull),
            "n_cot_pull_pairs": len(expand_pairs(sarcastic_cot_pull)),
            "n_propagation_push_neurons": len(propagation_push),
            "n_propagation_push_pairs": len(expand_pairs(propagation_push)),
            "n_propagation_pull_neurons": len(propagation_pull),
            "n_propagation_pull_pairs": len(expand_pairs(propagation_pull)),
            "n_routing_protect": len(routing_protect),
            "n_danger_zone": len(danger_zone),
        },
    }

    output_path = Path(output_dir) / "training_targets_v2.json"
    with open(output_path, "w") as f:
        json.dump(targets, f, indent=2)

    s = targets["summary"]
    print(f"\n  Training Targets v2:")
    print(f"    Sarcastic CoT push:  {s['n_cot_push_neurons']} neurons, {s['n_cot_push_pairs']} (layer,dim) pairs")
    print(f"    Sarcastic CoT pull:  {s['n_cot_pull_neurons']} neurons, {s['n_cot_pull_pairs']} (layer,dim) pairs")
    print(f"    Propagation push:    {s['n_propagation_push_neurons']} neurons, {s['n_propagation_push_pairs']} (layer,dim) pairs")
    print(f"    Propagation pull:    {s['n_propagation_pull_neurons']} neurons, {s['n_propagation_pull_pairs']} (layer,dim) pairs")
    print(f"    Routing protect:     {s['n_routing_protect']} (layer,dim) pairs (from Phase 2)")
    print(f"    DANGER ZONE:         {s['n_danger_zone']} neurons (routing overlap — DO NOT TARGET)")
    print(f"\n  Saved to {output_path}")

    return targets


# ─── Main Pipeline ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPT-OSS Deep CoT Priming Probe")
    parser.add_argument("--model-path", default="./skippy_gptoss_v2/merged_scale_1.0",
                        help="Path to GPT-OSS model (merged bf16 preferred)")
    parser.add_argument("--n-prompts", type=int, default=200,
                        help="Number of prompts to use (5 for sanity check)")
    parser.add_argument("--prompts-path", default="./contrastive_data/prompts_100k.jsonl")
    parser.add_argument("--output-base", default="./skippy_gptoss_fresh/phase2_deep_cot")
    parser.add_argument("--gpu-mem", type=float, default=0.80,
                        help="vLLM GPU memory utilization")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip Phase A, reuse saved responses")
    parser.add_argument("--skip-teacher-force", action="store_true",
                        help="Skip Phase B, reuse saved activations")
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--c1-min-layers", type=int, default=6)
    parser.add_argument("--c2-min-layers", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=16)
    args = parser.parse_args()

    gen_dir = Path(args.output_base) / "generation"
    tf_dir = Path(args.output_base) / "teacher_force"
    analysis_dir = Path(args.output_base) / "analysis"

    # Paths for cross-reference
    base_fresh = Path("./skippy_gptoss_fresh")
    phase1_path = str(base_fresh / "phase1" / "phase1_analysis.json")
    phase2_targets_path = str(base_fresh / "phase2_cot" / "analysis" / "training_targets.json")
    phase4_path = str(base_fresh / "phase4" / "phase4_sarcasm_analysis.json")

    print(f"\n{'#'*70}")
    print(f"# GPT-OSS Deep CoT Priming Probe")
    print(f"# Model: {args.model_path}")
    print(f"# Prompts: {args.n_prompts}")
    print(f"# Output: {args.output_base}")
    print(f"{'#'*70}")

    # ─── Phase A ───
    if args.skip_generation:
        print(f"\n  [Phase A SKIPPED — loading saved responses]")
        responses_path = gen_dir / "primed_responses.json"
        with open(responses_path) as f:
            saved = json.load(f)
        # Reconstruct with token_ids (needed for teacher forcing)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        generated: dict[str, list[dict]] = {}
        for mode_name in saved:
            generated[mode_name] = []
            for resp in saved[mode_name]:
                # Re-tokenize raw_response to get token_ids
                token_ids = tokenizer.encode(
                    resp["raw_response"], add_special_tokens=False
                )
                generated[mode_name].append({
                    **resp,
                    "token_ids": token_ids,
                })
        print(f"  Loaded: {', '.join(f'{k}={len(v)}' for k, v in generated.items())}")
    else:
        prompts = load_prompts(args.prompts_path, n_total=args.n_prompts)
        generated = phase_a_generate(
            prompts, args.model_path, str(gen_dir),
            max_tokens=args.max_tokens, gpu_mem=args.gpu_mem,
        )

    # ─── Phase B ───
    if args.skip_teacher_force:
        print(f"\n  [Phase B SKIPPED — loading saved activations]")
        # Load saved tensors
        from transformers import AutoModelForCausalLM
        config_path = Path(args.model_path) / "config.json"
        with open(config_path) as f:
            cfg = json.load(f)
        n_layers = cfg.get("num_hidden_layers", 24)
        hidden_dim = cfg.get("hidden_size", 2880)

        mode_region_acts: dict[str, dict[str, dict[int, list[torch.Tensor]]]] = {}
        mode_windowed_acts: dict[str, dict[int, list[list[torch.Tensor]]]] = {}
        REGIONS = ["priming", "reasoning", "transition", "final"]

        for mode_name in ["sarcastic", "neutral", "control"]:
            mode_region_acts[mode_name] = {r: {} for r in REGIONS}
            mode_windowed_acts[mode_name] = {}

            for region in REGIONS:
                for layer_idx in range(n_layers):
                    pt_path = tf_dir / f"acts_{mode_name}_{region}_L{layer_idx}.pt"
                    if pt_path.exists():
                        data = torch.load(pt_path, map_location="cpu", weights_only=True)
                        # Convert stacked (n, hidden_dim) back to list
                        mode_region_acts[mode_name][region][layer_idx] = list(data)

            for layer_idx in range(n_layers):
                w_path = tf_dir / f"windowed_acts_{mode_name}_L{layer_idx}.pt"
                if w_path.exists():
                    data = torch.load(w_path, map_location="cpu", weights_only=True)
                    # data shape: (n_samples, n_windows, hidden_dim)
                    mode_windowed_acts[mode_name][layer_idx] = [
                        list(data[i]) for i in range(data.shape[0])
                    ]

        teacher_force_results = {
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "total_samples": 0,
            "skipped": 0,
            "stats": {},
            "mode_region_acts": mode_region_acts,
            "mode_windowed_acts": mode_windowed_acts,
        }
        print(f"  Loaded activations: {n_layers} layers, hidden_dim={hidden_dim}")
    else:
        teacher_force_results = phase_b_teacher_force(
            generated, args.model_path, str(tf_dir),
            max_seq_len=args.max_seq_len,
            window_size=args.window_size,
            window_stride=args.window_stride,
        )

    # ─── Phase C ───
    c_results = phase_c_analysis(
        teacher_force_results, str(analysis_dir),
        phase1_path=phase1_path,
        phase2_targets_path=phase2_targets_path,
        phase4_path=phase4_path,
        z_threshold=args.z_threshold,
        c1_min_layers=args.c1_min_layers,
        c2_min_layers=args.c2_min_layers,
    )

    # ─── Phase D ───
    targets = phase_d_training_targets(
        c_results, str(analysis_dir),
        phase2_targets_path=phase2_targets_path,
    )

    # ─── Final Summary ───
    print(f"\n{'='*70}")
    print(f"DEEP COT PRIMING PROBE — COMPLETE")
    print(f"{'='*70}")

    c1_n = len(c_results["c1"]["cross_layer_neurons"])
    c2_n = len(c_results["c2"]["cross_layer_neurons"])
    c3_amp = c_results["c3"].get("n_amplifying", 0)
    c3_dec = c_results["c3"].get("n_decaying", 0)
    danger = targets["summary"]["n_danger_zone"]

    print(f"\n  C1 Sarcastic Thinking Neurons: {c1_n}")
    print(f"  C2 Downstream Propagation:     {c2_n}")
    print(f"  C3 Signal Trajectories:        {c3_amp} amplifying, {c3_dec} decaying")
    print(f"  DANGER ZONE (routing overlap):  {danger}")

    s = targets["summary"]
    total_targets = (
        s["n_cot_push_pairs"] + s["n_cot_pull_pairs"]
        + s["n_propagation_push_pairs"] + s["n_propagation_pull_pairs"]
    )
    print(f"\n  Total training target pairs: {total_targets}")
    print(f"  Routing protect pairs:       {s['n_routing_protect']}")
    print(f"\n  Results saved to: {args.output_base}")

    # Compare with Phase 2 (0 personality neurons)
    if c1_n > 0:
        print(f"\n  >>> SUCCESS: Found {c1_n} sarcastic thinking neurons!")
        print(f"  >>> Phase 2 found 0 with prompt-level comparison.")
        print(f"  >>> Content-level priming reveals the personality circuit.")
    else:
        print(f"\n  >>> NOTE: Still 0 sarcastic thinking neurons.")
        print(f"  >>> Consider: lower z_threshold, lower min_layers, or more prompts.")


if __name__ == "__main__":
    main()
