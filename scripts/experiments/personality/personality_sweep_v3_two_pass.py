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
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

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


def parse_think_response(full_text: str) -> tuple[str, str]:
    """Split model text into think/response segments."""
    think_text = ""
    response_text = full_text
    if "</think>" in full_text:
        parts = full_text.split("</think>", 1)
        think_text = parts[0].replace("<think>", "").strip()
        response_text = parts[1].strip()
    elif "<think>" in full_text:
        think_text = full_text.replace("<think>", "").strip()
        response_text = ""

    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        response_text = response_text.replace(tok, "").strip()
        think_text = think_text.replace(tok, "").strip()
    return think_text, response_text


def build_chat_text(processor, system_prompt: str, user_prompt: str) -> str:
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


def load_processor(model_name: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)


def load_replay_model(model_name: str = DEFAULT_MODEL, quantize: str = "bf16"):
    """Load HF model for replay pass (prefill activation extraction)."""
    from transformers import AutoModelForImageTextToText

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    load_kwargs: dict[str, Any] = dict(device_map="auto", trust_remote_code=True)
    if quantize == "int8":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantize == "bf16":
        load_kwargs["dtype"] = torch.bfloat16
    else:
        load_kwargs["dtype"] = "auto"

    print(f"[INFO] Loading replay model {model_name} ({quantize})...")
    model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
    model.eval()
    layers = model.model.language_model.layers
    hidden_dim = int(model.config.text_config.hidden_size)
    device = str(next(model.parameters()).device)
    print(
        f"[INFO] Replay model loaded: {len(layers)} layers, hidden={hidden_dim}, "
        f"device={device}, VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB"
    )
    return model, layers, hidden_dim


class FastGenerator:
    """Fast pass generation backend (vllm / sglang / transformers fallback)."""

    def __init__(
        self,
        model_name: str,
        processor,
        backend: str = "auto",
        quantize: str = "bf16",
        max_new_tokens: int = MAX_NEW_TOKENS,
        sglang_attention_backend: str = "auto",
        sglang_disable_cudnn_check: bool = False,
        sglang_mem_fraction_static: float | None = None,
    ):
        self.model_name = model_name
        self.processor = processor
        self.quantize = quantize
        self.max_new_tokens = max_new_tokens
        self.sglang_attention_backend = sglang_attention_backend
        self.sglang_disable_cudnn_check = sglang_disable_cudnn_check
        self.sglang_mem_fraction_static = sglang_mem_fraction_static
        self.backend = backend
        self._engine = None
        self._sampling = None
        self._errors: dict[str, str] = {}

        self.backend = self._resolve_backend(backend)
        self._init_backend()
        print(f"[INFO] Fast generation backend: {self.backend}")

    def _resolve_backend(self, requested: str) -> str:
        order = ["vllm", "sglang", "transformers"] if requested == "auto" else [requested]
        for cand in order:
            if cand == "vllm":
                try:
                    from vllm import LLM, SamplingParams  # noqa: F401

                    return "vllm"
                except Exception as exc:  # noqa: BLE001
                    self._errors["vllm"] = str(exc)
            elif cand == "sglang":
                try:
                    from sglang.srt.entrypoints.engine import Engine  # noqa: F401

                    return "sglang"
                except Exception as exc:  # noqa: BLE001
                    self._errors["sglang"] = str(exc)
            elif cand == "transformers":
                return "transformers"
            else:
                raise ValueError(f"Unsupported backend '{cand}'")
        raise RuntimeError(f"No generation backend available: {self._errors}")

    def _init_backend(self):
        if self.backend == "vllm":
            from vllm import LLM, SamplingParams

            dtype = "bfloat16" if self.quantize == "bf16" else "auto"
            self._engine = LLM(
                model=self.model_name,
                trust_remote_code=True,
                tensor_parallel_size=1,
                dtype=dtype,
            )
            self._sampling = SamplingParams(
                temperature=TEMPERATURE,
                top_p=0.95,
                top_k=20,
                repetition_penalty=1.1,
                max_tokens=self.max_new_tokens,
                skip_special_tokens=False,
            )
            return

        if self.backend == "sglang":
            from sglang.srt.entrypoints.engine import Engine

            if self.sglang_disable_cudnn_check and "SGLANG_DISABLE_CUDNN_CHECK" not in os.environ:
                os.environ["SGLANG_DISABLE_CUDNN_CHECK"] = "1"

            attn_backend = self.sglang_attention_backend
            if attn_backend == "auto":
                try:
                    major, _minor = torch.cuda.get_device_capability()
                    # Blackwell-class GPUs require triton/trtllm_mha in sglang for this model family.
                    if major >= 10:
                        attn_backend = "triton"
                except Exception:  # noqa: BLE001
                    attn_backend = "auto"

            dtype = "bfloat16" if self.quantize == "bf16" else "auto"
            engine_kwargs = dict(
                model_path=self.model_name,
                trust_remote_code=True,
                dtype=dtype,
            )
            if attn_backend != "auto":
                engine_kwargs["attention_backend"] = attn_backend
            if self.sglang_mem_fraction_static is not None:
                engine_kwargs["mem_fraction_static"] = float(self.sglang_mem_fraction_static)
            self._engine = Engine(**engine_kwargs)
            self._sampling = {
                "temperature": TEMPERATURE,
                "top_p": 0.95,
                "top_k": 20,
                "repetition_penalty": 1.1,
                "max_new_tokens": self.max_new_tokens,
            }
            return

        if self.backend == "transformers":
            # Fallback only; slower than vllm/sglang but preserves correctness.
            self._engine, _, _ = load_replay_model(self.model_name, self.quantize)
            return

        raise ValueError(f"Unsupported backend '{self.backend}'")

    def generate(self, chat_text: str) -> tuple[list[int], str]:
        if self.backend == "vllm":
            outs = self._engine.generate([chat_text], self._sampling, use_tqdm=False)
            out = outs[0].outputs[0]
            token_ids = [int(x) for x in out.token_ids]
            text = out.text or self.processor.decode(token_ids, skip_special_tokens=False)
            return token_ids, text

        if self.backend == "sglang":
            out = self._engine.generate(prompt=chat_text, sampling_params=self._sampling)
            item = out[0] if isinstance(out, list) else out
            token_ids = item.get("output_ids") or item.get("token_ids") or []
            token_ids = [int(x) for x in token_ids]
            text = item.get("text") or self.processor.decode(token_ids, skip_special_tokens=False)
            if not token_ids and text:
                token_ids = self.processor.tokenizer.encode(text, add_special_tokens=False)
            return token_ids, text

        # transformers fallback
        inputs = self.processor(text=[chat_text], return_tensors="pt").to(self._engine.device)
        for key in ["mm_token_type_ids", "pixel_values", "image_grid_thw"]:
            inputs.pop(key, None)
        prompt_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            out = self._engine.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=0.95,
                top_k=20,
                repetition_penalty=1.1,
                remove_invalid_values=True,
                renormalize_logits=True,
            )
        token_ids = [int(x) for x in out[0][prompt_len:].tolist()]
        text = self.processor.decode(token_ids, skip_special_tokens=False)
        return token_ids, text

    def shutdown(self):
        if self.backend == "sglang" and self._engine is not None:
            try:
                self._engine.shutdown()
            except Exception:  # noqa: BLE001
                pass
        self._engine = None
        self._sampling = None


class ReplayMeanCapture:
    """Captures full-sequence hidden states at target layers for replay pass."""

    def __init__(self, layers: torch.nn.ModuleList, target_indices: list[int], hidden_dim: int):
        self.target_indices = target_indices
        self.hidden_dim = hidden_dim
        self._hooks = []
        self._cache: dict[int, torch.Tensor] = {}
        for idx in target_indices:
            self._hooks.append(layers[idx].register_forward_hook(self._make_hook(idx)))

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            self._cache[layer_idx] = hidden.detach()

        return hook_fn

    def _mean_or_zero(self, hidden: torch.Tensor, start: int, end: int) -> torch.Tensor:
        if end > start:
            return hidden[start:end].float().mean(dim=0).cpu()
        return torch.zeros(self.hidden_dim)

    def extract_means(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lens: list[int],
        gen_lens: list[int],
        think_lens: list[int],
        response_lens: list[int],
    ) -> list[dict[str, dict[int, torch.Tensor]]]:
        self._cache.clear()
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        bsz = input_ids.shape[0]
        out: list[dict[str, dict[int, torch.Tensor]]] = []
        valid_lens = attention_mask.sum(dim=1).tolist()
        for bi in range(bsz):
            start = int(prompt_lens[bi])
            end = min(start + int(gen_lens[bi]), int(valid_lens[bi]))
            gen_len = max(0, end - start)
            think_end = min(start + max(0, int(think_lens[bi])), end)
            response_start = think_end
            response_end = min(response_start + max(0, int(response_lens[bi])), end)
            early_end = start + ((gen_len + 1) // 2)
            sample = {
                "mean": {},
                "think": {},
                "response": {},
                "early": {},
                "late": {},
            }
            for idx in self.target_indices:
                hidden = self._cache[idx][bi]  # (seq, hidden)
                sample["mean"][idx] = self._mean_or_zero(hidden, start, end)
                sample["think"][idx] = self._mean_or_zero(hidden, start, think_end)
                sample["response"][idx] = self._mean_or_zero(hidden, response_start, response_end)
                sample["early"][idx] = self._mean_or_zero(hidden, start, early_end)
                sample["late"][idx] = self._mean_or_zero(hidden, early_end, end)
            out.append(sample)
        self._cache.clear()
        return out

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class GeneratedWriter:
    """Stores pass-1 generation records (including token IDs) per character."""

    def __init__(self, output_dir: Path):
        self.gen_dir = output_dir / "generated"
        self.gen_dir.mkdir(parents=True, exist_ok=True)

    def write_generation(self, char_id: int, data: dict[str, Any]):
        f = self.gen_dir / f"char_{char_id:04d}.jsonl"
        with open(f, "a") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")
            fh.flush()

    def read_char_generations(self, char_id: int) -> dict[int, dict[str, Any]]:
        f = self.gen_dir / f"char_{char_id:04d}.jsonl"
        out: dict[int, dict[str, Any]] = {}
        if not f.exists():
            return out
        for line in f.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pi = int(obj["prompt_idx"])
                out[pi] = obj
            except Exception:  # noqa: BLE001
                continue
        return out

    def get_completed_chars(self, n_prompts: int) -> set[int]:
        done = set()
        for f in self.gen_dir.glob("char_*.jsonl"):
            try:
                cid = int(f.stem.split("_")[1])
            except Exception:  # noqa: BLE001
                continue
            prompt_ids = set()
            for line in f.read_text(errors="ignore").splitlines():
                try:
                    obj = json.loads(line)
                    prompt_ids.add(int(obj["prompt_idx"]))
                except Exception:  # noqa: BLE001
                    continue
            if len(prompt_ids) >= n_prompts:
                done.add(cid)
        return done


class ShardWriter:
    """Writes final responses and activation means from replay pass."""

    def __init__(self, output_dir: Path, target_layers: list[int], hidden_dim: int):
        self.output_dir = output_dir
        self.target_layers = target_layers
        self.hidden_dim = hidden_dim
        self.resp_dir = output_dir / "responses"
        self.activation_views = ("mean", "think", "response", "early", "late")
        self.act_roots = {
            "mean": output_dir / "activations",
            "think": output_dir / "activations_think",
            "response": output_dir / "activations_response",
            "early": output_dir / "activations_early",
            "late": output_dir / "activations_late",
        }
        self.resp_dir.mkdir(parents=True, exist_ok=True)
        for root in self.act_roots.values():
            for l in target_layers:
                (root / f"L{l:02d}").mkdir(parents=True, exist_ok=True)

        self._act_buffers: dict[str, dict[int, list[torch.Tensor]]] = {
            view: {l: [] for l in target_layers} for view in self.activation_views
        }
        self._act_meta: dict[str, dict[int, list[dict[str, Any]]]] = {
            view: {l: [] for l in target_layers} for view in self.activation_views
        }
        self._shard_idx = self._detect_next_shard_idx()
        self._buffer_limit = 5000

    def _detect_next_shard_idx(self) -> int:
        next_idx = 0
        for root in self.act_roots.values():
            for l in self.target_layers:
                layer_dir = root / f"L{l:02d}"
                for shard_path in layer_dir.glob("mean_shard_*.pt"):
                    try:
                        shard_num = int(shard_path.stem.split("_")[-1])
                    except Exception:  # noqa: BLE001
                        continue
                    next_idx = max(next_idx, shard_num + 1)
        return next_idx

    def write_response(self, char_id: int, data: dict[str, Any]):
        f = self.resp_dir / f"char_{char_id:04d}.jsonl"
        with open(f, "a") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")
            fh.flush()

    def buffer_activations(
        self,
        char_id: int,
        char_name: str,
        prompt_idx: int,
        prompt_cat: str,
        b5_combo: str,
        activations: dict[str, dict[int, torch.Tensor]],
    ):
        meta = {
            "char_id": char_id,
            "char_name": char_name,
            "prompt_idx": prompt_idx,
            "prompt_category": prompt_cat,
            "b5_combo": b5_combo,
        }
        for view in self.activation_views:
            for l in self.target_layers:
                if l in activations[view]:
                    self._act_buffers[view][l].append(activations[view][l])
                    self._act_meta[view][l].append(meta)
        if len(self._act_buffers["mean"][self.target_layers[0]]) >= self._buffer_limit:
            self.flush_activations()

    def flush_activations(self):
        for view in self.activation_views:
            root = self.act_roots[view]
            for l in self.target_layers:
                if not self._act_buffers[view][l]:
                    continue
                tensor = torch.stack(self._act_buffers[view][l])
                shard_path = root / f"L{l:02d}" / f"mean_shard_{self._shard_idx:04d}.pt"
                torch.save(tensor, shard_path)
                meta_path = root / f"L{l:02d}" / f"mean_shard_{self._shard_idx:04d}_meta.jsonl"
                with open(meta_path, "w") as fh:
                    for m in self._act_meta[view][l]:
                        fh.write(json.dumps(m) + "\n")
                self._act_buffers[view][l].clear()
                self._act_meta[view][l].clear()
        self._shard_idx += 1

    def get_completed_chars(self, n_prompts: int) -> set[int]:
        done = set()
        for f in self.resp_dir.glob("char_*.jsonl"):
            try:
                cid = int(f.stem.split("_")[1])
            except Exception:  # noqa: BLE001
                continue
            n_lines = sum(1 for _ in open(f))
            if n_lines >= n_prompts:
                done.add(cid)
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


def b5_combo_code(char: Character) -> str:
    dims = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    return "_".join(char.big_five[d][0].upper() for d in dims)


def read_completed_prompt_indices(resp_file: Path, n_prompts: int) -> set[int]:
    """Return completed prompt indices for one character response file."""
    if not resp_file.exists():
        return set()

    done: set[int] = set()
    legacy_count = 0
    with open(resp_file, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            legacy_count += 1
            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            pi = obj.get("prompt_idx")
            if isinstance(pi, int):
                if 0 <= pi < n_prompts:
                    done.add(pi)

    if done:
        return done
    # Legacy fallback: older runs did not store prompt_idx and wrote in prompt order.
    return set(range(min(legacy_count, n_prompts)))


def main():
    parser = argparse.ArgumentParser(description="Personality Sweep V3 (two-pass)")
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
    parser.add_argument("--quantize", choices=["int8", "bf16", "auto"], default="bf16",
                        help="Generation quantization")
    parser.add_argument("--backend", choices=["auto", "vllm", "sglang", "transformers"], default="auto",
                        help="Fast generation backend for pass-1")
    parser.add_argument("--sglang-attention-backend",
                        choices=["auto", "flashinfer", "triton", "trtllm_mha"],
                        default="auto",
                        help="Attention backend for sglang pass-1 engine")
    parser.add_argument("--sglang-disable-cudnn-check", action="store_true",
                        help="Set SGLANG_DISABLE_CUDNN_CHECK=1 during pass-1 engine init")
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=None,
                        help="Optional sglang mem_fraction_static for pass-1 (e.g. 0.28)")
    parser.add_argument("--replay-quantize", choices=["int8", "bf16", "auto"], default=None,
                        help="Replay pass quantization (defaults to --quantize)")
    parser.add_argument("--replay-batch-size", type=int, default=8,
                        help="Max sequences per replay forward batch")
    parser.add_argument("--replay-max-total-tokens", type=int, default=32768,
                        help="Max sum(seq_len) per replay batch")
    parser.add_argument("--skip-pass1", action="store_true", help="Skip generation pass and replay existing files")
    parser.add_argument("--skip-pass2", action="store_true", help="Skip replay pass (generation only)")
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

    prompts = list(ALL_PROMPTS)
    prompt_categories = list(PROMPT_CATEGORIES)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
        prompt_categories = prompt_categories[:args.max_prompts]

    processor = load_processor(args.model)
    replay_quantize = args.replay_quantize or args.quantize
    gen_writer = GeneratedWriter(output_dir)
    writer: ShardWriter | None = None

    # Save config
    config = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "two_pass_fast_generate_then_replay",
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
        "generation_backend": args.backend,
        "sglang_attention_backend": args.sglang_attention_backend,
        "sglang_disable_cudnn_check": args.sglang_disable_cudnn_check,
        "sglang_mem_fraction_static": args.sglang_mem_fraction_static,
        "generation_quantization": args.quantize,
        "replay_quantization": replay_quantize,
        "replay_batch_size": args.replay_batch_size,
        "replay_max_total_tokens": args.replay_max_total_tokens,
        "activation_views": ["mean", "think", "response", "early", "late"],
        "skip_pass1": args.skip_pass1,
        "skip_pass2": args.skip_pass2,
    }
    (output_dir / "sweep_config.json").write_text(json.dumps(config, indent=2))

    # Save character metadata
    chars_file = output_dir / "characters.jsonl"
    if not chars_file.exists():
        with open(chars_file, "w") as fh:
            for c in my_chars:
                fh.write(json.dumps(asdict(c)) + "\n")

    all_start = time.time()
    pass1_tokens = 0
    pass1_responses = 0
    pass1_elapsed = 0.0
    pass2_gen_tokens = 0
    pass2_seq_tokens = 0
    pass2_responses = 0
    pass2_elapsed = 0.0

    # ── Pass 1: fast generation ──────────────────────────────
    if not args.skip_pass1:
        pass1_start = time.time()
        fast = FastGenerator(
            model_name=args.model,
            processor=processor,
            backend=args.backend,
            quantize=args.quantize,
            max_new_tokens=args.max_new_tokens,
            sglang_attention_backend=args.sglang_attention_backend,
            sglang_disable_cudnn_check=args.sglang_disable_cudnn_check,
            sglang_mem_fraction_static=args.sglang_mem_fraction_static,
        )
        done_chars_gen = gen_writer.get_completed_chars(len(prompts))
        gen_remaining = [c for c in my_chars if c.char_id not in done_chars_gen]
        print(f"[PASS1] {len(done_chars_gen)} chars already generated, {len(gen_remaining)} remaining")

        try:
            for ci, char in enumerate(tqdm(gen_remaining, desc="Pass1-Chars")):
                if _SHUTDOWN:
                    print("[SHUTDOWN] Stopping after current character in pass-1.")
                    break

                sys_prompt = build_system_prompt(char)
                b5_combo = b5_combo_code(char)
                existing = gen_writer.read_char_generations(char.char_id)

                for pi, prompt in enumerate(prompts):
                    if _SHUTDOWN:
                        break
                    if pi in existing:
                        continue
                    cat = prompt_categories[pi]
                    try:
                        chat_text = build_chat_text(processor, sys_prompt, prompt)
                        token_ids, full_text = fast.generate(chat_text)
                        if not token_ids and full_text:
                            token_ids = processor.tokenizer.encode(full_text, add_special_tokens=False)
                        think_text, response_text = parse_think_response(full_text)
                        think_ids = (
                            processor.tokenizer.encode(think_text, add_special_tokens=False)
                            if think_text
                            else []
                        )
                        resp_ids = (
                            processor.tokenizer.encode(response_text, add_special_tokens=False)
                            if response_text
                            else []
                        )

                        gen_writer.write_generation(
                            char.char_id,
                            {
                                "char_id": char.char_id,
                                "char_name": char.name,
                                "prompt_idx": pi,
                                "prompt_category": cat,
                                "prompt": prompt,
                                "b5": b5_combo,
                                "think_text": think_text,
                                "response_text": response_text,
                                "n_think_tokens": len(think_ids),
                                "n_response_tokens": len(resp_ids),
                                "n_gen_tokens": len(token_ids),
                                "gen_token_ids": token_ids,
                                "full_text": full_text,
                                "backend": fast.backend,
                                "timestamp": datetime.now().isoformat(),
                            },
                        )
                        pass1_tokens += len(token_ids)
                        pass1_responses += 1
                    except torch.cuda.OutOfMemoryError:
                        print(f"[OOM][PASS1] char={char.char_id} prompt={pi}, skipping prompt")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    except Exception as exc:  # noqa: BLE001
                        print(f"[ERROR][PASS1] char={char.char_id} prompt={pi}: {exc}")
                        continue

                if (ci + 1) % 10 == 0:
                    elapsed = time.time() - pass1_start
                    rate = pass1_tokens / max(elapsed, 1.0)
                    print(
                        f"[PASS1] {ci + 1}/{len(gen_remaining)} chars, "
                        f"{pass1_responses} responses, {pass1_tokens/1e6:.2f}M tokens, {rate:.1f} tok/s"
                    )
        finally:
            fast.shutdown()

        pass1_elapsed = time.time() - pass1_start
        print(
            f"[PASS1 DONE] {pass1_responses} responses, {pass1_tokens/1e6:.2f}M tokens, "
            f"{pass1_tokens/max(pass1_elapsed,1.0):.1f} tok/s"
        )

    # ── Pass 2: replay for activation means ─────────────────
    if not args.skip_pass2:
        pass2_start = time.time()
        replay_model, replay_layers, hidden_dim = load_replay_model(args.model, quantize=replay_quantize)
        replay_device = next(replay_model.parameters()).device
        writer = ShardWriter(output_dir, target_layers, hidden_dim)
        capture = ReplayMeanCapture(replay_layers, target_layers, hidden_dim)

        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = processor.tokenizer.eos_token_id
        if pad_token_id is None:
            raise RuntimeError("Tokenizer is missing pad_token_id and eos_token_id")

        def flush_replay_batch(batch: list[dict[str, Any]]) -> tuple[int, int, int]:
            """Returns: (n_responses, gen_tokens, total_seq_tokens)."""
            if not batch:
                return 0, 0, 0
            try:
                bsz = len(batch)
                max_len = max(len(s["prompt_ids"]) + len(s["gen_ids"]) for s in batch)
                input_ids = torch.full(
                    (bsz, max_len),
                    int(pad_token_id),
                    dtype=torch.long,
                    device=replay_device,
                )
                attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=replay_device)
                prompt_lens: list[int] = []
                gen_lens: list[int] = []
                seq_tokens = 0
                for bi, sample in enumerate(batch):
                    seq = sample["prompt_ids"] + sample["gen_ids"]
                    seq_len = len(seq)
                    seq_tokens += seq_len
                    input_ids[bi, :seq_len] = torch.tensor(seq, dtype=torch.long, device=replay_device)
                    attention_mask[bi, :seq_len] = 1
                    prompt_lens.append(len(sample["prompt_ids"]))
                    gen_lens.append(len(sample["gen_ids"]))

                acts = capture.extract_means(
                    replay_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prompt_lens=prompt_lens,
                    gen_lens=gen_lens,
                    think_lens=[int(s["n_think_tokens"]) for s in batch],
                    response_lens=[int(s["n_response_tokens"]) for s in batch],
                )
                for bi, sample in enumerate(batch):
                    writer.write_response(
                        sample["char_id"],
                        {
                            "char_id": sample["char_id"],
                            "char_name": sample["char_name"],
                            "b5": sample["b5_combo"],
                            "prompt_idx": sample["prompt_idx"],
                            "prompt_category": sample["prompt_category"],
                            "prompt": sample["prompt"],
                            "think_text": sample["think_text"],
                            "response_text": sample["response_text"],
                            "n_think_tokens": sample["n_think_tokens"],
                            "n_response_tokens": sample["n_response_tokens"],
                            "n_gen_tokens": len(sample["gen_ids"]),
                            "n_gen_captured": len(sample["gen_ids"]),
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                    writer.buffer_activations(
                        sample["char_id"],
                        sample["char_name"],
                        sample["prompt_idx"],
                        sample["prompt_category"],
                        sample["b5_combo"],
                        acts[bi],
                    )
                return bsz, sum(gen_lens), seq_tokens
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                if len(batch) == 1:
                    s0 = batch[0]
                    print(
                        f"[OOM][PASS2] dropping sample char={s0['char_id']} prompt={s0['prompt_idx']} "
                        f"len={len(s0['prompt_ids']) + len(s0['gen_ids'])}"
                    )
                    return 0, 0, 0
                mid = len(batch) // 2
                a = flush_replay_batch(batch[:mid])
                b = flush_replay_batch(batch[mid:])
                return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

        done_chars_resp = writer.get_completed_chars(len(prompts))
        replay_remaining = [c for c in my_chars if c.char_id not in done_chars_resp]
        print(f"[PASS2] {len(done_chars_resp)} chars already replayed, {len(replay_remaining)} remaining")

        try:
            for ci, char in enumerate(tqdm(replay_remaining, desc="Pass2-Chars")):
                if _SHUTDOWN:
                    print("[SHUTDOWN] Stopping after current character in pass-2.")
                    break

                sys_prompt = build_system_prompt(char)
                b5_combo = b5_combo_code(char)
                gen_map = gen_writer.read_char_generations(char.char_id)
                if not gen_map:
                    continue

                resp_file = writer.resp_dir / f"char_{char.char_id:04d}.jsonl"
                done_prompt_idxs = read_completed_prompt_indices(resp_file, len(prompts))

                batch: list[dict[str, Any]] = []
                batch_total_tokens = 0

                for pi, prompt in enumerate(prompts):
                    if _SHUTDOWN:
                        break
                    if pi in done_prompt_idxs:
                        continue
                    rec = gen_map.get(pi)
                    if rec is None:
                        continue

                    gen_ids = rec.get("gen_token_ids") or rec.get("token_ids") or []
                    gen_ids = [int(x) for x in gen_ids]
                    think_text = str(rec.get("think_text") or "")
                    response_text = str(rec.get("response_text") or "")
                    prompt_category = str(rec.get("prompt_category") or prompt_categories[pi])
                    think_count = rec.get("n_think_tokens")
                    resp_count = rec.get("n_response_tokens")
                    if not isinstance(think_count, int):
                        think_count = len(
                            processor.tokenizer.encode(think_text, add_special_tokens=False)
                        ) if think_text else 0
                    if not isinstance(resp_count, int):
                        resp_count = len(
                            processor.tokenizer.encode(response_text, add_special_tokens=False)
                        ) if response_text else 0

                    # If pass-1 had no token IDs for this prompt, keep text output but write zero activations.
                    if not gen_ids:
                        zero_acts = {
                            view: {l: torch.zeros(hidden_dim) for l in target_layers}
                            for view in writer.activation_views
                        }
                        writer.write_response(
                            char.char_id,
                            {
                                "char_id": char.char_id,
                                "char_name": char.name,
                                "b5": b5_combo,
                                "prompt_idx": pi,
                                "prompt_category": prompt_category,
                                "prompt": prompt,
                                "think_text": think_text,
                                "response_text": response_text,
                                "n_think_tokens": think_count,
                                "n_response_tokens": resp_count,
                                "n_gen_tokens": 0,
                                "n_gen_captured": 0,
                                "timestamp": datetime.now().isoformat(),
                            },
                        )
                        writer.buffer_activations(
                            char.char_id,
                            char.name,
                            pi,
                            prompt_category,
                            b5_combo,
                            zero_acts,
                        )
                        pass2_responses += 1
                        continue

                    chat_text = build_chat_text(processor, sys_prompt, prompt)
                    prompt_inputs = processor(text=[chat_text], return_tensors="pt")
                    prompt_ids = prompt_inputs["input_ids"][0].tolist()

                    sample = {
                        "char_id": char.char_id,
                        "char_name": char.name,
                        "b5_combo": b5_combo,
                        "prompt_idx": pi,
                        "prompt_category": prompt_category,
                        "prompt": prompt,
                        "think_text": think_text,
                        "response_text": response_text,
                        "n_think_tokens": int(think_count),
                        "n_response_tokens": int(resp_count),
                        "prompt_ids": prompt_ids,
                        "gen_ids": gen_ids,
                    }
                    seq_len = len(prompt_ids) + len(gen_ids)
                    if batch and (
                        len(batch) >= args.replay_batch_size
                        or (batch_total_tokens + seq_len) > args.replay_max_total_tokens
                    ):
                        n_resp, n_gen_tok, n_seq_tok = flush_replay_batch(batch)
                        pass2_responses += n_resp
                        pass2_gen_tokens += n_gen_tok
                        pass2_seq_tokens += n_seq_tok
                        batch = []
                        batch_total_tokens = 0
                    batch.append(sample)
                    batch_total_tokens += seq_len

                if batch:
                    n_resp, n_gen_tok, n_seq_tok = flush_replay_batch(batch)
                    pass2_responses += n_resp
                    pass2_gen_tokens += n_gen_tok
                    pass2_seq_tokens += n_seq_tok

                if (ci + 1) % 10 == 0:
                    elapsed = time.time() - pass2_start
                    gen_rate = pass2_gen_tokens / max(elapsed, 1.0)
                    seq_rate = pass2_seq_tokens / max(elapsed, 1.0)
                    print(
                        f"[PASS2] {ci + 1}/{len(replay_remaining)} chars, {pass2_responses} responses, "
                        f"gen={pass2_gen_tokens/1e6:.2f}M ({gen_rate:.1f} tok/s), "
                        f"seq={pass2_seq_tokens/1e6:.2f}M ({seq_rate:.1f} tok/s)"
                    )
        finally:
            capture.remove_hooks()
            writer.flush_activations()

        pass2_elapsed = time.time() - pass2_start
        print(
            f"[PASS2 DONE] {pass2_responses} responses, "
            f"gen={pass2_gen_tokens/1e6:.2f}M ({pass2_gen_tokens/max(pass2_elapsed,1.0):.1f} tok/s), "
            f"seq={pass2_seq_tokens/1e6:.2f}M ({pass2_seq_tokens/max(pass2_elapsed,1.0):.1f} tok/s)"
        )

    all_elapsed = time.time() - all_start
    summary = {
        "pipeline": "two_pass_fast_generate_then_replay",
        "total_tokens": pass2_gen_tokens if not args.skip_pass2 else pass1_tokens,
        "total_responses": pass2_responses if not args.skip_pass2 else pass1_responses,
        "elapsed_seconds": all_elapsed,
        "tokens_per_second": (
            (pass2_gen_tokens / max(pass2_elapsed, 1.0))
            if not args.skip_pass2
            else (pass1_tokens / max(pass1_elapsed, 1.0))
        ),
        "pass1": {
            "responses": pass1_responses,
            "gen_tokens": pass1_tokens,
            "elapsed_seconds": pass1_elapsed,
            "gen_tokens_per_second": pass1_tokens / max(pass1_elapsed, 1.0),
        },
        "pass2": {
            "responses": pass2_responses,
            "gen_tokens": pass2_gen_tokens,
            "seq_tokens": pass2_seq_tokens,
            "elapsed_seconds": pass2_elapsed,
            "gen_tokens_per_second": pass2_gen_tokens / max(pass2_elapsed, 1.0),
            "seq_tokens_per_second": pass2_seq_tokens / max(pass2_elapsed, 1.0),
        },
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "target_layers": target_layers,
        "activation_views": ["mean", "think", "response", "early", "late"],
        "shard": shard_list[0] if len(shard_list) == 1 else None,
        "shard_list": shard_list,
    }
    (output_dir / "summary_stats.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\n[DONE] pass1={pass1_responses} resp, pass2={pass2_responses} resp, "
        f"tokens={summary['total_tokens']/1e6:.2f}M in {all_elapsed/3600:.2f}h"
    )


if __name__ == "__main__":
    main()
