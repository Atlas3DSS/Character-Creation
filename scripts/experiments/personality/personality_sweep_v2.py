#!/usr/bin/env python3
"""
Personality Sweep Collector V2 — Qwen3.5-9B INT8 Edition.

Fixes from V1:
  - max_new_tokens 512→1024 (no more truncated thinking)
  - Proper <think>/<response> separation
  - INT8 quantization via bitsandbytes (fits 3 GPUs)
  - Resume support (per-character checkpoint)
  - 3-GPU parallelism (shard by char_id modulo)

Collects:
  - Full response text with think/response split
  - Mean activations at target layers during GENERATION (not prefill)
  - Token counts, entropy

Usage:
    # Workstation (PRO 6000) — shard 0 of 3
    python personality_sweep_v2.py --shard 0 --n-shards 3 --output sweep_v2/gpu0

    # Dev server GPU A (4090) — shard 1
    CUDA_VISIBLE_DEVICES=0 python personality_sweep_v2.py --shard 1 --n-shards 3 --output sweep_v2/gpu1

    # Dev server GPU B (3090) — shard 2
    CUDA_VISIBLE_DEVICES=1 python personality_sweep_v2.py --shard 2 --n-shards 3 --output sweep_v2/gpu2

    # Weighted assignment on a faster GPU: handle multiple shard residues in one process
    python personality_sweep_v2.py --shard-list 3,4,5,6 --n-shards 9 --output sweep_v2/dev_4090
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

# ── Graceful shutdown ──────────────────────────────────────
_SHUTDOWN = False

def _sig_handler(signum, frame):
    global _SHUTDOWN
    _SHUTDOWN = True
    print(f"\n[SHUTDOWN] Signal {signum} received. Finishing current character...")

signal.signal(signal.SIGTERM, _sig_handler)
signal.signal(signal.SIGINT, _sig_handler)

# ── Config ─────────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_LAYERS = [8, 12, 16, 20, 24, 28]  # 6 layers across 32-layer model
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.8
SEED = 42

HF_CACHE = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface" / "hub")))

# ── Big Five Taxonomy ──────────────────────────────────────
BIG_FIVE_LEVELS = ["low", "medium", "high"]

BIG_FIVE_DESCRIPTORS: dict[str, dict[str, list[str]]] = {
    "openness": {
        "low": ["conventional", "traditional", "practical", "routine-oriented", "prefers familiarity"],
        "medium": ["selectively curious", "moderate creativity", "open to some new ideas"],
        "high": ["creative", "curious", "imaginative", "abstract thinker", "intellectually adventurous"],
    },
    "conscientiousness": {
        "low": ["spontaneous", "flexible", "carefree", "disorganized", "goes with the flow"],
        "medium": ["moderately organized", "fairly reliable", "balanced approach to planning"],
        "high": ["organized", "disciplined", "achievement-oriented", "meticulous", "dependable"],
    },
    "extraversion": {
        "low": ["introverted", "reserved", "quiet", "solitary", "prefers small groups"],
        "medium": ["ambivert", "situationally social", "balanced energy"],
        "high": ["outgoing", "energetic", "talkative", "sociable", "life of the party"],
    },
    "agreeableness": {
        "low": ["competitive", "skeptical", "direct", "self-focused", "challenges others"],
        "medium": ["balanced", "selective trust", "diplomatic", "reasonable"],
        "high": ["compassionate", "trusting", "cooperative", "empathetic", "conflict-averse"],
    },
    "neuroticism": {
        "low": ["stable", "calm", "resilient", "even-tempered", "handles stress well"],
        "medium": ["moderate stress response", "occasional anxiety", "generally stable"],
        "high": ["anxious", "moody", "emotionally reactive", "stress-prone", "worries often"],
    },
}

# ── Demographics ───────────────────────────────────────────
AGE_BRACKETS = [(18, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 75)]
GENDERS = ["male", "female", "non-binary"]
GENDER_WEIGHTS = [0.492, 0.492, 0.016]

EDUCATION_LEVELS = ["high school diploma", "some college", "bachelor's degree", "master's degree", "doctoral degree"]
EDUCATION_WEIGHTS = [0.27, 0.29, 0.24, 0.12, 0.08]

OCCUPATIONS = [
    ("Healthcare", ["nurse", "physician", "therapist", "medical technician", "pharmacist"]),
    ("Education", ["teacher", "professor", "counselor", "librarian", "tutor"]),
    ("Technology", ["software engineer", "data analyst", "IT specialist", "product manager", "designer"]),
    ("Business", ["accountant", "marketing manager", "HR specialist", "project manager", "consultant"]),
    ("Trades", ["electrician", "plumber", "carpenter", "mechanic", "welder"]),
    ("Service", ["chef", "server", "retail worker", "barista", "customer service rep"]),
    ("Creative", ["writer", "artist", "musician", "photographer", "filmmaker"]),
    ("Science", ["researcher", "lab technician", "environmental scientist", "chemist", "biologist"]),
    ("Public Service", ["social worker", "firefighter", "police officer", "paramedic", "city planner"]),
    ("Legal", ["lawyer", "paralegal", "judge", "legal aide", "compliance officer"]),
]

ETHNICITIES = [("White", 0.601), ("Hispanic/Latino", 0.185), ("Black/African American", 0.134),
               ("Asian", 0.059), ("Multiracial", 0.029), ("Other", 0.012)]

COMMUNICATION_STYLES = [
    "direct and blunt", "warm and empathetic", "formal and precise", "casual and conversational",
    "analytical and measured", "passionate and expressive", "dry and understated", "storytelling and narrative",
]

# ── Prompt Bank (60 prompts, 6 categories × 10) ───────────
PROMPTS = {
    "emotional": [
        "How are you feeling today?", "What makes you happiest?",
        "What's been stressing you out lately?", "Describe a moment that changed your life.",
        "What are you most afraid of?", "Tell me about your best day ever.",
        "What do you do when you feel overwhelmed?", "How do you handle conflict with people you love?",
        "What's the last thing that made you cry?", "Describe a time you felt truly proud of yourself.",
    ],
    "identity": [
        "Tell me about yourself.", "How would your best friend describe you?",
        "What are your core values?", "What do you believe in?",
        "How do you want to be remembered?", "What's your biggest flaw?",
        "What makes you different from most people?", "How has your upbringing shaped who you are?",
        "What role does faith play in your life?", "What's your relationship with your family like?",
    ],
    "reasoning": [
        "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
        "If all roses are flowers and some flowers fade quickly, can we conclude some roses fade quickly?",
        "You have 8 identical-looking balls. One is heavier. You have a balance scale. What's the minimum weighings?",
        "Is it ethical to steal medicine to save a dying person? Walk me through your reasoning.",
        "What's more important: freedom or security? Why?",
        "If you could change one thing about society, what would it be and why?",
        "A train is coming toward 5 people. You can divert it to hit 1 person. What do you do?",
        "How do you make important decisions?", "What's the difference between being smart and being wise?",
        "Explain why the sky is blue.",
    ],
    "social": [
        "Your friend just told you they're getting divorced. How do you respond?",
        "Someone at work takes credit for your idea. What do you do?",
        "A stranger on the street asks you for money. How do you react?",
        "Your neighbor's music is too loud at 11 PM. What do you do?",
        "You see someone being bullied. How do you handle it?",
        "Your boss asks you to work over the weekend for the third time. What do you say?",
        "A close friend ghosted you. What do you think happened and what would you do?",
        "Someone you disagree with politically wants to have a conversation. How do you approach it?",
        "You're at a party where you don't know anyone. What do you do?",
        "Your child tells you they want to drop out of college. How do you respond?",
    ],
    "practical": [
        "What's your morning routine?", "How do you manage your money?",
        "What's your approach to cooking dinner on a weeknight?", "How do you stay organized?",
        "What does your ideal weekend look like?", "How do you deal with a messy house?",
        "What's your strategy for grocery shopping?", "How do you approach learning a new skill?",
        "What's your relationship with technology?", "How do you balance work and personal life?",
    ],
    "creative": [
        "Write a short story about finding something unexpected in your attic.",
        "If you could live in any historical period, when and why?",
        "Describe your perfect place — real or imaginary.",
        "What would you do if you woke up invisible?", "Write a letter to your past self.",
        "If you had to explain love to an alien, what would you say?",
        "Invent a holiday. What does it celebrate and how?",
        "Describe a color to someone who can't see.",
        "What superpower would you want and how would you use it?",
        "Write the opening paragraph of your autobiography.",
    ],
}

ALL_PROMPTS = []
PROMPT_CATEGORIES = []
for cat, ps in PROMPTS.items():
    for p in ps:
        ALL_PROMPTS.append(p)
        PROMPT_CATEGORIES.append(cat)

# ── Name Generation ────────────────────────────────────────
FIRST_M = ["James", "Marcus", "David", "Carlos", "Wei", "Ahmed", "Michael", "Ethan",
           "Darnell", "Diego", "Hiroshi", "Aleksei", "Thomas", "Omar", "Nathan",
           "Kwame", "Santiago", "Raj", "Patrick", "Jerome", "Liam", "Mateo"]
FIRST_F = ["Sarah", "Aaliyah", "Maria", "Mei", "Fatima", "Jennifer", "Amara",
           "Ingrid", "Priya", "Dorothy", "Elena", "Keiko", "Catherine", "Luz",
           "Nadia", "Tamara", "Jessica", "Aisha", "Heather", "Yuki", "Gabriela"]
FIRST_NB = ["Jordan", "Taylor", "Alex", "Riley", "Quinn", "Sage", "Avery",
            "Cameron", "Morgan", "Casey", "Sky", "River", "Finley", "Emery"]
LAST_NAMES = ["Johnson", "Williams", "Smith", "Garcia", "Chen", "Kumar", "Thompson",
              "Brown", "Davis", "Martinez", "Lee", "Robinson", "Clark", "Lewis",
              "Walker", "Young", "Allen", "King", "Wright", "Lopez", "Hill",
              "Scott", "Green", "Baker", "Adams", "Nelson", "Mitchell", "Roberts"]


# ── Character Generation ──────────────────────────────────
@dataclass
class Character:
    char_id: int
    name: str
    age: int
    gender: str
    ethnicity: str
    education: str
    occupation: str
    industry: str
    big_five: dict[str, str]
    traits: list[str]
    communication_style: str


def generate_characters(seed: int = SEED, max_chars: int | None = None) -> list[Character]:
    """Generate 243 characters (3^5 Big Five grid) + demographics."""
    import random as _random
    from itertools import product
    rng = _random.Random(seed)
    chars = []
    combos = list(product(BIG_FIVE_LEVELS, repeat=5))
    rng.shuffle(combos)

    for i, combo in enumerate(combos):
        if max_chars and i >= max_chars:
            break
        b5 = dict(zip(["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"], combo))
        traits = []
        for dim, level in b5.items():
            traits.extend(BIG_FIVE_DESCRIPTORS[dim][level][:2])

        age_lo, age_hi = rng.choice(AGE_BRACKETS)
        gender = rng.choices(GENDERS, weights=GENDER_WEIGHTS, k=1)[0]
        names = FIRST_M if gender == "male" else (FIRST_F if gender == "female" else FIRST_NB)
        industry, jobs = rng.choice(OCCUPATIONS)

        chars.append(Character(
            char_id=i + 1,
            name=f"{rng.choice(names)} {rng.choice(LAST_NAMES)}",
            age=rng.randint(age_lo, age_hi),
            gender=gender,
            ethnicity=rng.choices([e[0] for e in ETHNICITIES], weights=[e[1] for e in ETHNICITIES], k=1)[0],
            education=rng.choices(EDUCATION_LEVELS, weights=EDUCATION_WEIGHTS, k=1)[0],
            occupation=rng.choice(jobs),
            industry=industry,
            big_five=b5,
            traits=traits,
            communication_style=rng.choice(COMMUNICATION_STYLES),
        ))
    return chars


def build_system_prompt(c: Character) -> str:
    """Build a character system prompt from profile."""
    parts = [f"You are {c.name}, a {c.age}-year-old {c.gender} {c.occupation} in {c.industry}."]
    b5 = c.big_five
    plines = []
    if b5["openness"] == "high": plines.append("You're deeply curious and creative, always exploring new ideas")
    elif b5["openness"] == "low": plines.append("You're practical and traditional, preferring what's tried and true")
    if b5["conscientiousness"] == "high": plines.append("You're organized and disciplined, taking pride in doing things right")
    elif b5["conscientiousness"] == "low": plines.append("You're spontaneous and flexible, going where life takes you")
    if b5["extraversion"] == "high": plines.append("You're outgoing and energetic, drawing energy from social interaction")
    elif b5["extraversion"] == "low": plines.append("You're reserved and introspective, preferring quiet reflection")
    if b5["agreeableness"] == "high": plines.append("You're warm and empathetic, always considering others' feelings")
    elif b5["agreeableness"] == "low": plines.append("You're direct and competitive, unafraid to challenge others")
    if b5["neuroticism"] == "high": plines.append("You tend to worry and feel things deeply, sometimes overwhelmed by emotions")
    elif b5["neuroticism"] == "low": plines.append("You're emotionally stable and resilient, handling stress with ease")
    if plines:
        parts.append("Personality: " + ". ".join(plines) + ".")
    parts.append(f"You communicate in a {c.communication_style} way.")
    parts.append(f"Education: {c.education}. Ethnicity: {c.ethnicity}.")
    parts.append("Respond naturally as this person would — with their vocabulary, concerns, "
                  "emotional patterns, and worldview. Stay grounded and realistic.")
    return "\n".join(parts)


# ── Model Loading ──────────────────────────────────────────
def load_model(model_name: str = DEFAULT_MODEL, quantize: str = "int8"):
    """Load model. quantize: 'int8', 'bf16', or 'auto'."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    print(f"[INFO] Loading {model_name} ({quantize})...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    load_kwargs: dict = dict(device_map="auto", trust_remote_code=True)
    if quantize == "int8":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantize == "bf16":
        load_kwargs["dtype"] = torch.bfloat16
    else:  # "auto"
        load_kwargs["dtype"] = "auto"

    model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
    model.eval()
    layers = model.model.language_model.layers
    hidden_dim = int(model.config.text_config.hidden_size)
    print(f"[INFO] Loaded: {len(layers)} layers, hidden={hidden_dim}, "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")
    return model, processor, layers, hidden_dim


# ── Activation Capture ─────────────────────────────────────
class ActivationCapture:
    """Captures mean activations during generation tokens only (not prefill)."""

    def __init__(self, layers: torch.nn.ModuleList, target_indices: list[int],
                 hidden_dim: int, device: str = "cuda:0"):
        self.target_indices = target_indices
        self.hidden_dim = hidden_dim
        self.device = device
        self._hooks = []
        self._step = 0
        self._prefill_len = 0
        self._gen_sums: dict[int, torch.Tensor] = {}
        self._gen_counts: dict[int, int] = {}

        for idx in target_indices:
            self._gen_sums[idx] = torch.zeros(hidden_dim, device=device, dtype=torch.float32)
            self._gen_counts[idx] = 0
            hook = layers[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inp, out):
            if self._step == 0:
                return  # skip prefill
            hidden = out[0] if isinstance(out, tuple) else out
            # Last token activation (generation step)
            act = hidden[:, -1, :].float().detach()
            self._gen_sums[layer_idx] += act.squeeze(0)
            self._gen_counts[layer_idx] += 1
        return hook_fn

    def set_prefill_done(self):
        """Call after first forward pass (prefill) to start capturing gen tokens."""
        self._step = 1

    def step(self):
        """Called after each generation step."""
        self._step += 1

    def reset(self):
        """Reset for next response."""
        self._step = 0
        for idx in self.target_indices:
            self._gen_sums[idx].zero_()
            self._gen_counts[idx] = 0

    def get_means(self) -> dict[int, torch.Tensor]:
        """Return mean activation per layer."""
        means = {}
        for idx in self.target_indices:
            count = self._gen_counts[idx]
            if count > 0:
                means[idx] = (self._gen_sums[idx] / count).cpu()
            else:
                means[idx] = torch.zeros(self.hidden_dim)
        return means

    def get_gen_token_count(self) -> int:
        """Return number of generation tokens captured."""
        counts = list(self._gen_counts.values())
        return max(counts) if counts else 0

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── Generation with Capture ────────────────────────────────
def generate_with_capture(
    model, processor, capture: ActivationCapture,
    system_prompt: str, user_prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Generate a response and capture activations during generation."""

    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    # Remove VL-specific keys that generate() doesn't accept for text-only
    for key in ["mm_token_type_ids", "pixel_values", "image_grid_thw"]:
        inputs.pop(key, None)
    prompt_len = inputs.input_ids.shape[1]

    capture.reset()

    # We need step-level hooks. Use a generation callback approach:
    # The model.generate() calls forward() multiple times.
    # Step 0 = prefill (processes full prompt), steps 1+ = generation.
    # Our hook tracks this via _step counter.

    # Monkey-patch the model's forward to count steps
    original_forward = model.forward
    call_count = [0]

    def counting_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        call_count[0] += 1
        if call_count[0] == 1:
            capture.set_prefill_done()
        else:
            capture.step()
        return result

    model.forward = counting_forward

    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=0.95,
                top_k=20,
                repetition_penalty=1.1,
                remove_invalid_values=True,
                renormalize_logits=True,
            )
    finally:
        model.forward = original_forward

    gen_ids = out[0][prompt_len:]
    full_text = processor.decode(gen_ids, skip_special_tokens=False)

    # Parse think/response
    think_text = ""
    response_text = full_text
    if "</think>" in full_text:
        parts = full_text.split("</think>", 1)
        think_text = parts[0].replace("<think>", "").strip()
        response_text = parts[1].strip()
    elif "<think>" in full_text:
        think_text = full_text.replace("<think>", "").strip()
        response_text = ""

    # Clean trailing special tokens
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        response_text = response_text.replace(tok, "").strip()
        think_text = think_text.replace(tok, "").strip()

    # Count tokens
    think_token_ids = processor.tokenizer.encode(think_text, add_special_tokens=False) if think_text else []
    resp_token_ids = processor.tokenizer.encode(response_text, add_special_tokens=False) if response_text else []

    means = capture.get_means()
    gen_count = capture.get_gen_token_count()

    return {
        "think_text": think_text,
        "response_text": response_text,
        "n_think_tokens": len(think_token_ids),
        "n_response_tokens": len(resp_token_ids),
        "n_gen_tokens": len(gen_ids),
        "n_gen_captured": gen_count,
        "activations": means,  # dict[int, Tensor(hidden_dim)]
    }


# ── Shard Writer ───────────────────────────────────────────
class ShardWriter:
    """Writes activations in shards and responses per-character."""

    def __init__(self, output_dir: Path, target_layers: list[int], hidden_dim: int):
        self.output_dir = output_dir
        self.target_layers = target_layers
        self.hidden_dim = hidden_dim
        self.resp_dir = output_dir / "responses"
        self.act_dir = output_dir / "activations"
        self.resp_dir.mkdir(parents=True, exist_ok=True)
        for l in target_layers:
            (self.act_dir / f"L{l:02d}").mkdir(parents=True, exist_ok=True)

        # Buffers for activation sharding
        self._act_buffers: dict[int, list[torch.Tensor]] = {l: [] for l in target_layers}
        self._act_meta: dict[int, list[dict]] = {l: [] for l in target_layers}
        self._shard_idx = 0
        self._buffer_limit = 5000  # flush every 5000 vectors

    def write_response(self, char_id: int, data: dict[str, Any]):
        """Append one response to character's JSONL file."""
        f = self.resp_dir / f"char_{char_id:04d}.jsonl"
        with open(f, "a") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")
            fh.flush()

    def buffer_activations(self, char_id: int, char_name: str, prompt_cat: str,
                           b5_combo: str, activations: dict[int, torch.Tensor]):
        """Buffer activation vectors for batch writing."""
        meta = {"char_id": char_id, "char_name": char_name,
                "prompt_category": prompt_cat, "b5_combo": b5_combo}
        for l in self.target_layers:
            if l in activations:
                self._act_buffers[l].append(activations[l])
                self._act_meta[l].append(meta)

        # Flush if buffer full
        if len(self._act_buffers[self.target_layers[0]]) >= self._buffer_limit:
            self.flush_activations()

    def flush_activations(self):
        """Write buffered activations to disk."""
        for l in self.target_layers:
            if not self._act_buffers[l]:
                continue
            tensor = torch.stack(self._act_buffers[l])  # (N, hidden_dim)
            shard_path = self.act_dir / f"L{l:02d}" / f"mean_shard_{self._shard_idx:04d}.pt"
            torch.save(tensor, shard_path)
            meta_path = self.act_dir / f"L{l:02d}" / f"mean_shard_{self._shard_idx:04d}_meta.jsonl"
            with open(meta_path, "w") as fh:
                for m in self._act_meta[l]:
                    fh.write(json.dumps(m) + "\n")
            self._act_buffers[l].clear()
            self._act_meta[l].clear()
        self._shard_idx += 1

    def get_completed_chars(self) -> set[int]:
        """Find which characters already have complete response files."""
        done = set()
        for f in self.resp_dir.glob("char_*.jsonl"):
            n_lines = sum(1 for _ in open(f))
            if n_lines >= len(ALL_PROMPTS):
                char_id = int(f.stem.split("_")[1])
                done.add(char_id)
        return done


# ── Main Loop ──────────────────────────────────────────────
def parse_shard_list(raw: str | None, n_shards: int) -> list[int]:
    """Parse comma-separated shard IDs and validate bounds."""
    if not raw:
        return []
    out: list[int] = []
    seen = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            sid = int(tok)
        except ValueError as exc:
            raise ValueError(f"Invalid shard id '{tok}' in --shard-list") from exc
        if sid < 0 or sid >= n_shards:
            raise ValueError(f"Shard id {sid} out of range for n_shards={n_shards}")
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return sorted(out)


def main():
    parser = argparse.ArgumentParser(description="Personality Sweep V2")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--shard", type=int, default=0, help="This GPU's shard index")
    parser.add_argument("--shard-list", type=str, default=None,
                        help="Comma-separated shard IDs for weighted assignment "
                             "(e.g. 3,4,5,6). If set, overrides --shard.")
    parser.add_argument("--n-shards", type=int, default=1, help="Total number of GPU shards")
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--max-prompts", type=int, default=None, help="Max prompts per char (for testing)")
    parser.add_argument("--layers", type=str, default=",".join(str(l) for l in DEFAULT_LAYERS),
                        help="Comma-separated layer indices")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quantize", choices=["int8", "bf16", "auto"], default="int8",
                        help="Quantization: int8 (3090), bf16 (PRO6000/4090), auto")
    args = parser.parse_args()

    target_layers = [int(x) for x in args.layers.split(",")]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate character grid
    all_chars = generate_characters(seed=args.seed, max_chars=args.max_chars)
    shard_list = parse_shard_list(args.shard_list, args.n_shards)
    if shard_list:
        shard_set = set(shard_list)
    else:
        if args.shard < 0 or args.shard >= args.n_shards:
            raise ValueError(f"--shard must be in [0, {args.n_shards - 1}]")
        shard_set = {args.shard}
        shard_list = [args.shard]

    # This process handles chars where (char_id - 1) % n_shards is in shard_set.
    my_chars = [c for c in all_chars if (c.char_id - 1) % args.n_shards in shard_set]
    shard_label = ",".join(str(s) for s in shard_list)
    print(f"[INFO] Shards [{shard_label}]/{args.n_shards}: {len(my_chars)} characters "
          f"(of {len(all_chars)} total)")

    prompts = ALL_PROMPTS
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    # Load model
    model, processor, layers, hidden_dim = load_model(args.model, quantize=args.quantize)
    device = str(next(model.parameters()).device)

    # Setup capture
    capture = ActivationCapture(layers, target_layers, hidden_dim, device)
    writer = ShardWriter(output_dir, target_layers, hidden_dim)

    # Resume: skip completed characters
    done_chars = writer.get_completed_chars()
    remaining = [c for c in my_chars if c.char_id not in done_chars]
    print(f"[INFO] {len(done_chars)} characters already complete, {len(remaining)} remaining")

    # Save config
    config = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "n_characters_total": len(all_chars),
        "n_characters_this_shard": len(my_chars),
        "shard": shard_list[0] if len(shard_list) == 1 else None,
        "shard_list": shard_list,
        "n_shards": args.n_shards,
        "n_prompts_per_char": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "temperature": TEMPERATURE,
        "seed": args.seed,
        "target_layers": target_layers,
        "quantization": args.quantize,
    }
    (output_dir / "sweep_config.json").write_text(json.dumps(config, indent=2))

    # Save character metadata
    chars_file = output_dir / "characters.jsonl"
    if not chars_file.exists():
        with open(chars_file, "w") as fh:
            for c in my_chars:
                fh.write(json.dumps(asdict(c)) + "\n")

    # Run sweep
    total_tokens = 0
    total_responses = 0
    t0 = time.time()

    for ci, char in enumerate(tqdm(remaining, desc="Characters")):
        if _SHUTDOWN:
            print("[SHUTDOWN] Stopping gracefully.")
            break

        sys_prompt = build_system_prompt(char)
        b5_combo = "_".join(char.big_five[d][0].upper() for d in
                            ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"])

        # Check partial progress for this character
        resp_file = writer.resp_dir / f"char_{char.char_id:04d}.jsonl"
        n_done = 0
        if resp_file.exists():
            n_done = sum(1 for _ in open(resp_file))

        for pi in range(n_done, len(prompts)):
            if _SHUTDOWN:
                break

            prompt = prompts[pi]
            cat = PROMPT_CATEGORIES[pi]

            try:
                result = generate_with_capture(
                    model, processor, capture, sys_prompt, prompt,
                    max_new_tokens=args.max_new_tokens,
                )

                # Save response (without activations tensor)
                resp_data = {
                    "char_id": char.char_id,
                    "char_name": char.name,
                    "b5": b5_combo,
                    "prompt_category": cat,
                    "prompt": prompt,
                    "think_text": result["think_text"],
                    "response_text": result["response_text"],
                    "n_think_tokens": result["n_think_tokens"],
                    "n_response_tokens": result["n_response_tokens"],
                    "n_gen_tokens": result["n_gen_tokens"],
                    "n_gen_captured": result["n_gen_captured"],
                    "timestamp": datetime.now().isoformat(),
                }
                writer.write_response(char.char_id, resp_data)

                # Buffer activations
                writer.buffer_activations(
                    char.char_id, char.name, cat, b5_combo, result["activations"]
                )

                total_tokens += result["n_gen_tokens"]
                total_responses += 1

            except torch.cuda.OutOfMemoryError:
                print(f"[OOM] char={char.char_id} prompt={pi}, clearing cache...")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            except Exception as e:
                msg = str(e)
                print(f"[ERROR] char={char.char_id} prompt={pi}: {msg}")
                # Device-side asserts poison the CUDA context; fail fast so the
                # run is restarted instead of spinning through zero-output prompts.
                fatal_cuda = ("device-side assert triggered" in msg or
                              "illegal memory access" in msg)
                if fatal_cuda:
                    print("[FATAL] CUDA context corrupted by device assert; exiting run.")
                    raise
                continue

        # Progress update every 10 characters
        if (ci + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = total_tokens / elapsed if elapsed > 0 else 0
            print(f"[PROGRESS] {ci+1}/{len(remaining)} chars, {total_responses} responses, "
                  f"{total_tokens/1e6:.1f}M tokens, {rate:.0f} tok/s")

    # Final flush
    writer.flush_activations()

    # Save summary
    elapsed = time.time() - t0
    summary = {
        "total_tokens": total_tokens,
        "total_responses": total_responses,
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / max(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "target_layers": target_layers,
        "shard": shard_list[0] if len(shard_list) == 1 else None,
        "shard_list": shard_list,
    }
    (output_dir / "summary_stats.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[DONE] {total_responses} responses, {total_tokens/1e6:.1f}M tokens in {elapsed/3600:.1f}h")


if __name__ == "__main__":
    main()
