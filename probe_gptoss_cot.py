#!/usr/bin/env python3
"""
GPT-OSS-20B Chain-of-Thought Neuron Profiling.

Identifies three critical neuron sets for v4 training:
  1. CoT PERSONALITY neurons — fire differently during analysis channel for Skippy vs ChatGPT
  2. Output PERSONALITY neurons — fire differently during final channel for Skippy vs ChatGPT
  3. CoT ROUTING neurons — fire differently between analysis and final channels (PROTECT these!)

Workflow:
  Phase A: vLLM generates 200 responses × 2 modes (fast, ~5 min)
  Phase B: HF teacher-forces those responses through base model to capture per-token activations
  Phase C: Segment tokens into analysis/final, compute z-scores, save neuron lists

The CoT routing neurons are the ones that v3 accidentally destroyed. By identifying them upfront,
v4 training can EXCLUDE them from regularization targets while TARGETING CoT personality neurons
to guide the model's thinking toward Skippy-like reasoning.

Usage:
    python probe_gptoss_cot.py
    python probe_gptoss_cot.py --n-prompts 100
    python probe_gptoss_cot.py --skip-generation  # reuse saved vLLM outputs
    python probe_gptoss_cot.py --model-path ./skippy_gptoss_v2/merged_scale_1.0
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


# ─── Prompt Bank ───────────────────────────────────────────────────────

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


# ─── Phase A: vLLM Generation ─────────────────────────────────────────

def phase_a_generate(
    prompts: list[str],
    model_path: str,
    output_dir: str,
    max_tokens: int = 2048,
    gpu_mem: float = 0.80,
) -> dict[str, list[dict]]:
    """Generate responses with vLLM in two modes: skippy and bare.

    Uses llm.generate() with forced analysis channel prefix so the model
    produces dual-channel output (analysis → final). vLLM's chat() strips
    special tokens, so we manually construct prompts and decode with
    skip_special_tokens=False to preserve channel markers.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)
    results_path = Path(output_dir) / "generated_responses.json"

    print(f"\n{'='*70}")
    print(f"PHASE A: vLLM Generation — {len(prompts)} prompts × 2 modes")
    print(f"{'='*70}")

    # Load tokenizer separately for template construction
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"\nLoading model: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem,
        max_model_len=8192,
        trust_remote_code=True,
    )

    # Use return token as stop
    stop_token_ids = [tokenizer.convert_tokens_to_ids("<|return|>")]
    # Also stop on endoftext
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

    results: dict[str, list[dict]] = {"skippy": [], "bare": []}

    for mode_name, identity in [("skippy", SKIPPY_IDENTITY), ("bare", "")]:
        print(f"\n  Generating mode: {mode_name}...")
        t0 = time.time()

        # Build prompts manually with forced analysis channel prefix
        full_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            template_kwargs = {}
            if identity:
                template_kwargs["model_identity"] = identity

            # Apply chat template to get prompt up to generation start
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **template_kwargs,
            )
            # Force the model to start with analysis channel
            # The generation prompt ends with <|start|>assistant
            # We append <|channel|>analysis<|message|> to force CoT
            text += "<|channel|>analysis<|message|>"
            full_prompts.append(text)

        # Generate using raw prompts (not chat)
        outputs = llm.generate(full_prompts, sampling_params=sampling)
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s ({elapsed/len(prompts):.2f}s/prompt)")

        # Collect results — decode with special tokens preserved
        n_with_analysis = 0
        n_with_final = 0
        for i, out in enumerate(outputs):
            token_ids = list(out.outputs[0].token_ids)
            # Decode WITH special tokens so we can see channel markers
            raw_text = tokenizer.decode(token_ids, skip_special_tokens=False)
            # Also get clean text for readability
            clean_text = tokenizer.decode(token_ids, skip_special_tokens=True)

            has_analysis = True  # We forced analysis start
            has_final = "<|channel|>final" in raw_text or "final" in clean_text[:10]

            if has_analysis:
                n_with_analysis += 1
            if has_final:
                n_with_final += 1

            results[mode_name].append({
                "prompt": prompts[i],
                "raw_response": raw_text,
                "clean_response": clean_text,
                "token_ids": token_ids,
                "has_analysis": has_analysis,
                "has_final": has_final,
                "token_count": len(token_ids),
            })

        print(f"    Analysis channel present: {n_with_analysis}/{len(prompts)} ({100*n_with_analysis/len(prompts):.0f}%)")
        print(f"    Final channel present: {n_with_final}/{len(prompts)} ({100*n_with_final/len(prompts):.0f}%)")

        # Show a few examples
        for j in range(min(3, len(results[mode_name]))):
            resp = results[mode_name][j]
            print(f"\n    Example {j}: {resp['prompt'][:60]}...")
            print(f"    Raw (200 chars): {resp['raw_response'][:200]}")
            print(f"    Has final: {resp['has_final']}, Tokens: {resp['token_count']}")

    # Save (skip token_ids to keep file size manageable)
    save_results: dict[str, list[dict]] = {"skippy": [], "bare": []}
    for mode_name in results:
        for resp in results[mode_name]:
            save_results[mode_name].append({
                "prompt": resp["prompt"],
                "raw_response": resp["raw_response"],
                "clean_response": resp["clean_response"],
                "has_analysis": resp["has_analysis"],
                "has_final": resp["has_final"],
                "token_count": resp["token_count"],
            })

    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved responses to {results_path}")

    # Cleanup vLLM to free GPU memory
    del llm
    torch.cuda.empty_cache()

    return results


# ─── Phase B: HF Teacher-Forced Activation Capture ────────────────────

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
                    # Full sequence: (batch, seq_len, hidden_dim)
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


def find_token_spans(token_ids: list[int], tokenizer) -> dict[str, tuple[int, int]]:
    """Find the start/end positions of analysis and final channel spans in token sequence.

    Uses direct special token ID matching for robustness.
    Returns dict with 'analysis' and 'final' keys mapping to (start, end) token indices.
    """
    spans: dict[str, tuple[int, int]] = {}

    # Get special token IDs
    def get_token_id(s: str) -> int | None:
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    channel_id = get_token_id("<|channel|>")
    message_id = get_token_id("<|message|>")
    end_id = get_token_id("<|end|>")
    return_id = get_token_id("<|return|>")
    start_id = get_token_id("<|start|>")

    # Also get text token IDs for "analysis" and "final"
    analysis_text_ids = tokenizer.encode("analysis", add_special_tokens=False)
    final_text_ids = tokenizer.encode("final", add_special_tokens=False)

    def find_subsequence(haystack: list[int], needle: list[int], start: int = 0) -> int:
        """Find first occurrence of needle in haystack starting from position start."""
        for i in range(start, len(haystack) - len(needle) + 1):
            if haystack[i:i + len(needle)] == needle:
                return i
        return -1

    # Strategy 1: Look for <|channel|> token followed by analysis/final text tokens
    # Pattern: <|channel|> + "analysis" tokens + <|message|> + [content] + <|end|>
    # Pattern: <|channel|> + "final" tokens + <|message|> + [content] + <|return|>

    analysis_content_start = None
    analysis_content_end = None
    final_content_start = None
    final_content_end = None

    i = 0
    while i < len(token_ids):
        tid = token_ids[i]

        # Look for <|channel|> token
        if tid == channel_id and i + 1 < len(token_ids):
            # Check what follows — analysis or final?
            remaining = token_ids[i + 1:]

            # Check for analysis pattern
            if analysis_content_start is None:
                is_analysis = False
                if len(analysis_text_ids) == 1 and remaining and remaining[0] == analysis_text_ids[0]:
                    is_analysis = True
                elif find_subsequence(remaining[:5], analysis_text_ids) == 0:
                    is_analysis = True

                if is_analysis:
                    # Find <|message|> after analysis
                    for j in range(i + 1, min(i + 10, len(token_ids))):
                        if token_ids[j] == message_id:
                            analysis_content_start = j + 1
                            break
                    if analysis_content_start is None:
                        # Fallback: content starts after analysis text
                        analysis_content_start = i + 1 + len(analysis_text_ids)

            # Check for final pattern
            if final_content_start is None:
                is_final = False
                if len(final_text_ids) == 1 and remaining and remaining[0] == final_text_ids[0]:
                    is_final = True
                elif find_subsequence(remaining[:5], final_text_ids) == 0:
                    is_final = True

                if is_final:
                    # Analysis ends here (before the <|end|><|start|>...<|channel|>final)
                    if analysis_content_start is not None and analysis_content_end is None:
                        # Walk back to find <|end|> before this channel marker
                        for j in range(i - 1, max(i - 10, analysis_content_start), -1):
                            if token_ids[j] == end_id:
                                analysis_content_end = j
                                break
                        if analysis_content_end is None:
                            analysis_content_end = i

                    # Find <|message|> after final
                    for j in range(i + 1, min(i + 10, len(token_ids))):
                        if token_ids[j] == message_id:
                            final_content_start = j + 1
                            break
                    if final_content_start is None:
                        final_content_start = i + 1 + len(final_text_ids)

        # Look for <|return|> or <|end|> after final channel
        if final_content_start is not None and final_content_end is None:
            if tid == return_id or (tid == end_id and i > final_content_start):
                final_content_end = i

        i += 1

    # Fallback: if no <|channel|> token found, look for literal text patterns
    if analysis_content_start is None and final_content_start is None:
        full_text = tokenizer.decode(token_ids, skip_special_tokens=False)

        # Look for channel markers in text
        analysis_marker = "<|channel|>analysis<|message|>"
        final_marker = "<|channel|>final<|message|>"

        if analysis_marker in full_text:
            # Approximate: use character position ratios to estimate token positions
            char_pos = full_text.index(analysis_marker) + len(analysis_marker)
            ratio = char_pos / max(len(full_text), 1)
            analysis_content_start = int(ratio * len(token_ids))

        if final_marker in full_text:
            char_pos_f = full_text.index(final_marker) + len(final_marker)
            ratio_f = char_pos_f / max(len(full_text), 1)
            final_content_start = int(ratio_f * len(token_ids))
            if analysis_content_start is not None:
                analysis_content_end = int((full_text.index(final_marker) / max(len(full_text), 1)) * len(token_ids))

    # Build spans
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


@torch.no_grad()
def phase_b_teacher_force(
    generated: dict[str, list[dict]],
    model_path: str,
    output_dir: str,
    max_seq_len: int = 4096,
) -> dict:
    """Teacher-force generated responses through HF model to capture per-token activations.

    For each response, tokenizes [prompt + response], runs a single forward pass,
    and captures hidden states at every layer. Segments tokens into analysis/final spans.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PHASE B: HF Teacher-Forced Activation Capture")
    print(f"{'='*70}")

    # Check if model_path is the base model (needs Mxfp4) or merged (bf16)
    config_path = Path(model_path) / "config.json"
    needs_mxfp4 = False
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        if "quantization_config" in cfg:
            quant_type = cfg["quantization_config"].get("quant_type", "")
            if "mxfp" in quant_type.lower():
                needs_mxfp4 = True

    # Also check HF cache for base model
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

    # Collect per-token activations segmented by channel
    # Structure: mode -> channel -> layer -> list of (neuron_dim,) mean activations
    mode_channel_acts: dict[str, dict[str, dict[int, list[torch.Tensor]]]] = {}

    for mode_name in ["skippy", "bare"]:
        mode_channel_acts[mode_name] = {
            "analysis": {idx: [] for idx in probe.layer_indices},
            "final": {idx: [] for idx in probe.layer_indices},
        }

    total_samples = 0
    skipped_no_channels = 0
    channel_stats: dict[str, dict[str, int]] = {
        "skippy": {"analysis": 0, "final": 0, "both": 0},
        "bare": {"analysis": 0, "final": 0, "both": 0},
    }

    for mode_name in ["skippy", "bare"]:
        responses = generated[mode_name]
        identity = SKIPPY_IDENTITY if mode_name == "skippy" else ""

        print(f"\n  Teacher-forcing mode: {mode_name} ({len(responses)} responses)")
        t0 = time.time()

        for resp_data in tqdm(responses, desc=f"  {mode_name}"):
            prompt_text = resp_data["prompt"]
            raw_response = resp_data["raw_response"]

            # Skip if no dual-channel structure
            if not resp_data.get("has_analysis") and not resp_data.get("has_final"):
                skipped_no_channels += 1
                continue

            # Build full tokenized input: prompt + forced_prefix + response (teacher forcing)
            messages = [{"role": "user", "content": prompt_text}]
            template_kwargs = {}
            if identity:
                template_kwargs["model_identity"] = identity

            prompt_text_templated = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **template_kwargs,
            )

            # We forced the analysis channel prefix during generation.
            # raw_response is the vLLM output AFTER our prefix.
            # Reconstruct: template + forced_prefix + generated_output
            forced_prefix = "<|channel|>analysis<|message|>"
            full_text = prompt_text_templated + forced_prefix + raw_response

            # Tokenize
            inputs = tokenizer(
                full_text, return_tensors="pt", truncation=True,
                max_length=max_seq_len,
            )
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]

            # Find prompt length to know where generation starts
            prompt_inputs = tokenizer(
                prompt_text_templated, return_tensors="pt", truncation=True,
                max_length=max_seq_len,
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]

            # Only look at generated tokens (after prompt)
            gen_token_ids = input_ids[0, prompt_len:].tolist()

            # Find channel spans within the generated portion
            spans = find_token_spans(gen_token_ids, tokenizer)

            has_analysis = "analysis" in spans
            has_final = "final" in spans

            if has_analysis:
                channel_stats[mode_name]["analysis"] += 1
            if has_final:
                channel_stats[mode_name]["final"] += 1
            if has_analysis and has_final:
                channel_stats[mode_name]["both"] += 1

            if not has_analysis and not has_final:
                skipped_no_channels += 1
                continue

            total_samples += 1

            # Forward pass (teacher forcing — no generation)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            probe.clear()
            _ = model(**inputs)

            # Extract mean activations per channel per layer
            for idx in probe.layer_indices:
                hidden = probe.hidden_states.get(idx)
                if hidden is None:
                    continue
                # hidden shape: (1, seq_len, hidden_dim)
                hidden = hidden.squeeze(0)  # (seq_len, hidden_dim)

                for channel_name, (start, end) in spans.items():
                    # Adjust for prompt offset
                    abs_start = prompt_len + start
                    abs_end = prompt_len + end
                    if abs_start >= seq_len or abs_end > seq_len:
                        continue

                    channel_acts = hidden[abs_start:abs_end]  # (n_tokens, hidden_dim)
                    if channel_acts.shape[0] == 0:
                        continue

                    # Store mean activation across tokens in this channel
                    mean_act = channel_acts.mean(dim=0)  # (hidden_dim,)
                    mode_channel_acts[mode_name][channel_name][idx].append(mean_act)

        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")

    print(f"\n  Total usable samples: {total_samples}")
    print(f"  Skipped (no channels): {skipped_no_channels}")
    for mode_name, stats in channel_stats.items():
        print(f"  {mode_name}: analysis={stats['analysis']}, final={stats['final']}, both={stats['both']}")

    # Save raw tensors for each mode/channel/layer
    for mode_name in mode_channel_acts:
        for channel_name in mode_channel_acts[mode_name]:
            for idx in mode_channel_acts[mode_name][channel_name]:
                acts_list = mode_channel_acts[mode_name][channel_name][idx]
                if acts_list:
                    stacked = torch.stack(acts_list)  # (n_samples, hidden_dim)
                    save_path = Path(output_dir) / f"acts_{mode_name}_{channel_name}_L{idx}.pt"
                    torch.save(stacked, save_path)

    # Cleanup
    probe.remove_hooks()
    del model
    torch.cuda.empty_cache()

    return {
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "total_samples": total_samples,
        "skipped_no_channels": skipped_no_channels,
        "channel_stats": channel_stats,
        "mode_channel_acts": mode_channel_acts,
    }


# ─── Phase C: Analysis ────────────────────────────────────────────────

def compute_zscores(
    acts_a: list[torch.Tensor],
    acts_b: list[torch.Tensor],
) -> torch.Tensor:
    """Compute per-neuron z-scores between two activation sets.
    Returns tensor of shape (hidden_dim,).
    """
    if not acts_a or not acts_b:
        return torch.zeros(0)

    a = torch.stack(acts_a)  # (n_a, hidden_dim)
    b = torch.stack(acts_b)  # (n_b, hidden_dim)

    mean_a = a.mean(dim=0)
    mean_b = b.mean(dim=0)
    var_a = a.var(dim=0, unbiased=True)
    var_b = b.var(dim=0, unbiased=True)

    n_a, n_b = a.shape[0], b.shape[0]
    pooled_std = torch.sqrt((var_a * (n_a - 1) + var_b * (n_b - 1)) / (n_a + n_b - 2) + 1e-10)

    zscores = (mean_a - mean_b) / pooled_std
    return zscores


def phase_c_analysis(
    teacher_force_results: dict,
    output_dir: str,
    z_threshold: float = 2.0,
    min_layers: int = 6,
) -> dict:
    """Compute three neuron comparisons from teacher-forced activations."""
    os.makedirs(output_dir, exist_ok=True)

    n_layers = teacher_force_results["n_layers"]
    hidden_dim = teacher_force_results["hidden_dim"]
    acts = teacher_force_results["mode_channel_acts"]

    print(f"\n{'='*70}")
    print(f"PHASE C: Neuron Analysis — {n_layers} layers × {hidden_dim} hidden dim")
    print(f"  Z-score threshold: ±{z_threshold}, min layers: {min_layers}")
    print(f"{'='*70}")

    results = {
        "cot_personality": {},      # analysis(skippy) vs analysis(bare)
        "output_personality": {},   # final(skippy) vs final(bare)
        "cot_routing": {},          # analysis vs final (pooled across modes)
    }

    # Comparison 1: CoT Personality — analysis channel, skippy vs bare
    print("\n  1. CoT Personality: analysis(skippy) vs analysis(bare)")
    cot_pers_zscores = {}
    for idx in range(n_layers):
        skippy_acts = acts["skippy"]["analysis"].get(idx, [])
        bare_acts = acts["bare"]["analysis"].get(idx, [])
        if skippy_acts and bare_acts:
            zs = compute_zscores(skippy_acts, bare_acts)
            cot_pers_zscores[idx] = zs
            n_sig = (zs.abs() > z_threshold).sum().item()
            print(f"    L{idx:2d}: {len(skippy_acts):3d} skippy, {len(bare_acts):3d} bare, {n_sig:4d} significant neurons")

    # Comparison 2: Output Personality — final channel, skippy vs bare
    print("\n  2. Output Personality: final(skippy) vs final(bare)")
    out_pers_zscores = {}
    for idx in range(n_layers):
        skippy_acts = acts["skippy"]["final"].get(idx, [])
        bare_acts = acts["bare"]["final"].get(idx, [])
        if skippy_acts and bare_acts:
            zs = compute_zscores(skippy_acts, bare_acts)
            out_pers_zscores[idx] = zs
            n_sig = (zs.abs() > z_threshold).sum().item()
            print(f"    L{idx:2d}: {len(skippy_acts):3d} skippy, {len(bare_acts):3d} bare, {n_sig:4d} significant neurons")

    # Comparison 3: CoT Routing — analysis vs final (pooled across both modes)
    print("\n  3. CoT Routing: analysis tokens vs final tokens (pooled)")
    routing_zscores = {}
    for idx in range(n_layers):
        # Pool analysis from both modes
        analysis_acts = []
        for mode in ["skippy", "bare"]:
            analysis_acts.extend(acts[mode]["analysis"].get(idx, []))
        # Pool final from both modes
        final_acts = []
        for mode in ["skippy", "bare"]:
            final_acts.extend(acts[mode]["final"].get(idx, []))

        if analysis_acts and final_acts:
            zs = compute_zscores(analysis_acts, final_acts)
            routing_zscores[idx] = zs
            n_sig = (zs.abs() > z_threshold).sum().item()
            print(f"    L{idx:2d}: {len(analysis_acts):3d} analysis, {len(final_acts):3d} final, {n_sig:4d} significant neurons")

    # ─── Find cross-layer neurons ───────────────────────────────────

    def find_cross_layer_neurons(
        zscores_by_layer: dict[int, torch.Tensor],
        label: str,
    ) -> list[dict]:
        """Find neurons that are consistently significant across multiple layers."""
        if not zscores_by_layer:
            return []

        hidden_dim_actual = next(iter(zscores_by_layer.values())).shape[0]
        neuron_layers: dict[int, list[tuple[int, float]]] = defaultdict(list)

        for idx, zs in zscores_by_layer.items():
            significant = torch.where(zs.abs() > z_threshold)[0]
            for dim_idx in significant.tolist():
                neuron_layers[dim_idx].append((idx, zs[dim_idx].item()))

        # Filter by min_layers
        cross_layer: list[dict] = []
        for dim_idx, layer_scores in neuron_layers.items():
            if len(layer_scores) >= min_layers:
                avg_z = np.mean([abs(z) for _, z in layer_scores])
                direction_votes = sum(1 for _, z in layer_scores if z > 0)
                direction = "push" if direction_votes > len(layer_scores) / 2 else "pull"

                cross_layer.append({
                    "dim": dim_idx,
                    "n_layers": len(layer_scores),
                    "avg_abs_z": float(avg_z),
                    "direction": direction,
                    "layers": sorted([idx for idx, _ in layer_scores]),
                    "peak_layer": max(layer_scores, key=lambda x: abs(x[1]))[0],
                    "peak_z": float(max(layer_scores, key=lambda x: abs(x[1]))[1]),
                })

        cross_layer.sort(key=lambda x: (-x["n_layers"], -x["avg_abs_z"]))
        return cross_layer

    cot_personality_neurons = find_cross_layer_neurons(cot_pers_zscores, "cot_personality")
    output_personality_neurons = find_cross_layer_neurons(out_pers_zscores, "output_personality")
    cot_routing_neurons = find_cross_layer_neurons(routing_zscores, "cot_routing")

    # ─── Print summaries ────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")

    print(f"\n  CoT Personality Neurons (analysis channel, skippy vs bare):")
    print(f"    Total: {len(cot_personality_neurons)}")
    push = sum(1 for n in cot_personality_neurons if n["direction"] == "push")
    pull = len(cot_personality_neurons) - push
    print(f"    Push: {push}, Pull: {pull}")
    for n in cot_personality_neurons[:15]:
        print(f"      dim {n['dim']:4d}: {n['n_layers']:2d} layers, avg|z|={n['avg_abs_z']:.2f}, "
              f"{n['direction']}, peak L{n['peak_layer']} (z={n['peak_z']:.2f})")

    print(f"\n  Output Personality Neurons (final channel, skippy vs bare):")
    print(f"    Total: {len(output_personality_neurons)}")
    push = sum(1 for n in output_personality_neurons if n["direction"] == "push")
    pull = len(output_personality_neurons) - push
    print(f"    Push: {push}, Pull: {pull}")
    for n in output_personality_neurons[:15]:
        print(f"      dim {n['dim']:4d}: {n['n_layers']:2d} layers, avg|z|={n['avg_abs_z']:.2f}, "
              f"{n['direction']}, peak L{n['peak_layer']} (z={n['peak_z']:.2f})")

    print(f"\n  CoT Routing Neurons (analysis vs final tokens — PROTECT these!):")
    print(f"    Total: {len(cot_routing_neurons)}")
    for n in cot_routing_neurons[:15]:
        print(f"      dim {n['dim']:4d}: {n['n_layers']:2d} layers, avg|z|={n['avg_abs_z']:.2f}, "
              f"{n['direction']}, peak L{n['peak_layer']} (z={n['peak_z']:.2f})")

    # ─── Overlap analysis ───────────────────────────────────────────

    cot_pers_dims = set(n["dim"] for n in cot_personality_neurons)
    out_pers_dims = set(n["dim"] for n in output_personality_neurons)
    routing_dims = set(n["dim"] for n in cot_routing_neurons)

    pers_overlap = cot_pers_dims & out_pers_dims
    cot_pers_routing_overlap = cot_pers_dims & routing_dims
    out_pers_routing_overlap = out_pers_dims & routing_dims
    all_overlap = cot_pers_dims & out_pers_dims & routing_dims

    print(f"\n  Overlap Analysis:")
    print(f"    CoT pers ∩ Output pers: {len(pers_overlap)} neurons")
    print(f"    CoT pers ∩ Routing:     {len(cot_pers_routing_overlap)} neurons ⚠️  DANGER ZONE")
    print(f"    Output pers ∩ Routing:  {len(out_pers_routing_overlap)} neurons ⚠️  DANGER ZONE")
    print(f"    All three overlap:      {len(all_overlap)} neurons")

    # Safe personality neurons (NOT in routing set)
    safe_cot_pers = [n for n in cot_personality_neurons if n["dim"] not in routing_dims]
    safe_out_pers = [n for n in output_personality_neurons if n["dim"] not in routing_dims]

    print(f"\n  SAFE Personality Neurons (not overlapping with routing):")
    print(f"    Safe CoT personality:    {len(safe_cot_pers)} / {len(cot_personality_neurons)}")
    print(f"    Safe output personality: {len(safe_out_pers)} / {len(output_personality_neurons)}")

    # ─── Layer importance profiles ──────────────────────────────────

    layer_importance = {}
    for name, zscores_by_layer in [
        ("cot_personality", cot_pers_zscores),
        ("output_personality", out_pers_zscores),
        ("cot_routing", routing_zscores),
    ]:
        layer_imp = {}
        for idx, zs in zscores_by_layer.items():
            layer_imp[idx] = float(zs.abs().mean())
        layer_importance[name] = layer_imp

    # ─── Save everything ────────────────────────────────────────────

    analysis_results = {
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "z_threshold": z_threshold,
        "min_layers": min_layers,
        "total_samples": teacher_force_results["total_samples"],
        "channel_stats": teacher_force_results["channel_stats"],
        "cot_personality_neurons": cot_personality_neurons,
        "output_personality_neurons": output_personality_neurons,
        "cot_routing_neurons": cot_routing_neurons,
        "safe_cot_personality": safe_cot_pers,
        "safe_output_personality": safe_out_pers,
        "overlap": {
            "cot_pers_and_output_pers": sorted(pers_overlap),
            "cot_pers_and_routing": sorted(cot_pers_routing_overlap),
            "output_pers_and_routing": sorted(out_pers_routing_overlap),
            "all_three": sorted(all_overlap),
        },
        "layer_importance": layer_importance,
    }

    # Save analysis
    with open(Path(output_dir) / "cot_analysis.json", "w") as f:
        json.dump(analysis_results, f, indent=2)

    # Save z-score tensors
    for name, zscores_by_layer in [
        ("cot_personality", cot_pers_zscores),
        ("output_personality", out_pers_zscores),
        ("cot_routing", routing_zscores),
    ]:
        for idx, zs in zscores_by_layer.items():
            torch.save(zs, Path(output_dir) / f"zscores_{name}_L{idx}.pt")

    # Save neuron lists as compact format for training
    training_targets = {
        "safe_cot_push": [],
        "safe_cot_pull": [],
        "safe_output_push": [],
        "safe_output_pull": [],
        "routing_protect": [],
    }

    for n in safe_cot_pers:
        key = "safe_cot_push" if n["direction"] == "push" else "safe_cot_pull"
        for layer in n["layers"]:
            training_targets[key].append({"layer": layer, "dim": n["dim"], "avg_z": n["avg_abs_z"]})

    for n in safe_out_pers:
        key = "safe_output_push" if n["direction"] == "push" else "safe_output_pull"
        for layer in n["layers"]:
            training_targets[key].append({"layer": layer, "dim": n["dim"], "avg_z": n["avg_abs_z"]})

    for n in cot_routing_neurons:
        for layer in n["layers"]:
            training_targets["routing_protect"].append({"layer": layer, "dim": n["dim"], "avg_z": n["avg_abs_z"]})

    with open(Path(output_dir) / "training_targets.json", "w") as f:
        json.dump(training_targets, f, indent=2)

    print(f"\n  Saved analysis to {output_dir}/cot_analysis.json")
    print(f"  Saved training targets to {output_dir}/training_targets.json")
    print(f"  Training targets:")
    for key, items in training_targets.items():
        print(f"    {key}: {len(items)} (layer, dim) pairs")

    return analysis_results


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPT-OSS CoT Neuron Profiling")
    parser.add_argument("--n-prompts", type=int, default=200,
                        help="Number of prompts to generate/analyze")
    parser.add_argument("--model-path", type=str,
                        default="./skippy_gptoss_v2/merged_scale_1.0",
                        help="Model path for vLLM generation")
    parser.add_argument("--base-model", type=str,
                        default="openai/gpt-oss-20b",
                        help="Base model for HF teacher-forcing (uses Mxfp4 dequant)")
    parser.add_argument("--prompts-jsonl", type=str,
                        default="contrastive_data/prompts_100k.jsonl",
                        help="Path to prompts JSONL file")
    parser.add_argument("--output", type=str,
                        default="skippy_gptoss_fresh/phase2_cot",
                        help="Output directory")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip Phase A, reuse saved vLLM outputs")
    parser.add_argument("--skip-teacher-force", action="store_true",
                        help="Skip Phase B, reuse saved activations")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max generation tokens for vLLM")
    parser.add_argument("--gpu-mem", type=float, default=0.80,
                        help="vLLM GPU memory utilization")
    parser.add_argument("--z-threshold", type=float, default=2.0,
                        help="Z-score threshold for neuron significance")
    parser.add_argument("--min-layers", type=int, default=6,
                        help="Minimum layers for cross-layer neuron detection")
    parser.add_argument("--use-merged-for-tf", action="store_true",
                        help="Use merged model (not base) for teacher-forcing")
    args = parser.parse_args()

    output_dir = args.output
    gen_dir = os.path.join(output_dir, "generation")
    tf_dir = os.path.join(output_dir, "teacher_force")
    analysis_dir = os.path.join(output_dir, "analysis")

    # Phase A: Generate with vLLM
    if not args.skip_generation:
        prompts = load_prompts(args.prompts_jsonl, n_total=args.n_prompts)
        generated = phase_a_generate(
            prompts=prompts,
            model_path=args.model_path,
            output_dir=gen_dir,
            max_tokens=args.max_tokens,
            gpu_mem=args.gpu_mem,
        )
    else:
        print("\n  Skipping generation, loading saved responses...")
        gen_path = Path(gen_dir) / "generated_responses.json"
        if not gen_path.exists():
            raise FileNotFoundError(f"No saved responses at {gen_path}")
        with open(gen_path) as f:
            generated = json.load(f)
        print(f"  Loaded {len(generated['skippy'])} skippy + {len(generated['bare'])} bare responses")

    # Phase B: Teacher-force through HF
    if not args.skip_teacher_force:
        tf_model = args.model_path if args.use_merged_for_tf else args.base_model
        tf_results = phase_b_teacher_force(
            generated=generated,
            model_path=tf_model,
            output_dir=tf_dir,
            max_seq_len=4096,
        )
    else:
        print("\n  Skipping teacher-forcing, loading saved activations...")
        # Reconstruct results from saved .pt files
        raise NotImplementedError("Reload from saved .pt files not yet implemented")

    # Phase C: Analysis
    analysis = phase_c_analysis(
        teacher_force_results=tf_results,
        output_dir=analysis_dir,
        z_threshold=args.z_threshold,
        min_layers=args.min_layers,
    )

    print(f"\n{'='*70}")
    print(f"DONE! All results saved to {output_dir}/")
    print(f"{'='*70}")

    # Quick summary for memory
    n_safe_cot = len(analysis.get("safe_cot_personality", []))
    n_safe_out = len(analysis.get("safe_output_personality", []))
    n_routing = len(analysis.get("cot_routing_neurons", []))
    n_danger = len(analysis.get("overlap", {}).get("cot_pers_and_routing", []))

    print(f"\n  KEY NUMBERS FOR V4 TRAINING:")
    print(f"    Safe CoT personality neurons:    {n_safe_cot}")
    print(f"    Safe output personality neurons:  {n_safe_out}")
    print(f"    CoT routing neurons (PROTECT):    {n_routing}")
    print(f"    DANGER ZONE (personality∩routing): {n_danger}")


if __name__ == "__main__":
    main()
