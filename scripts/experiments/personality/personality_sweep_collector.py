#!/usr/bin/env python3
"""
Personality Sweep Collector — 10M Token Ethnographic Database.

Systematically varies personality dimensions (Big Five) and demographics
to build a comprehensive database mapping personality → neural representations.

Approach (ethnographer + ML):
  1. Generate character profiles from Big Five × demographics grid
  2. Seed with real character journals for realism
  3. Run diverse prompts through the Thinking model with each personality
  4. Capture: full text, mean activations, entropy, top logits

Designed for dual-GPU parallel execution on dev server:
  GPU A (CUDA_VISIBLE_DEVICES=0 → 4090): odd character IDs
  GPU B (CUDA_VISIBLE_DEVICES=1 → 3090): even character IDs

Output structure:
  sweep_output/
    sweep_config.json          — run parameters
    characters.jsonl           — all character profiles
    responses/
      char_0001.jsonl          — all responses for character 0001
      char_0002.jsonl          — ...
    activations/
      L09/ L15/ L22/ L29/     — mean activation shards per layer
    summary_stats.json         — aggregate statistics

Usage:
    # Full sweep on single GPU
    python personality_sweep_collector.py --output ./sweep_output

    # Dual GPU split (odd/even character IDs)
    CUDA_VISIBLE_DEVICES=0 python personality_sweep_collector.py \\
        --split odd --output ./sweep_output/gpu_a &
    CUDA_VISIBLE_DEVICES=1 python personality_sweep_collector.py \\
        --split even --output ./sweep_output/gpu_b &

    # Limit for testing
    python personality_sweep_collector.py --max-characters 5 --max-prompts-per-char 3

Requires: source ~/dev_genius/venv/bin/activate
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

# ── Model config (defaults — overridable via CLI) ───────────

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_LAYERS = [9, 15, 22, 29]

# Module-level — set at runtime by main() from CLI args
MODEL_NAME = DEFAULT_MODEL
TARGET_LAYERS = DEFAULT_LAYERS

HF_CACHE = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface" / "hub")))

# ── Big Five Trait Taxonomy ─────────────────────────────────

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

# ── Demographics Distributions ──────────────────────────────

AGE_BRACKETS = [
    (18, 24, "young adult"),
    (25, 34, "early career"),
    (35, 44, "established adult"),
    (45, 54, "midlife"),
    (55, 64, "pre-retirement"),
    (65, 75, "senior"),
]

GENDERS = ["male", "female", "non-binary"]
GENDER_WEIGHTS = [0.492, 0.492, 0.016]

EDUCATION_LEVELS = [
    "high school diploma",
    "some college",
    "bachelor's degree",
    "master's degree",
    "doctoral degree",
]
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

ETHNICITIES = [
    ("White", 0.601),
    ("Hispanic/Latino", 0.185),
    ("Black/African American", 0.134),
    ("Asian", 0.059),
    ("Multiracial", 0.029),
    ("Other", 0.012),
]

COMMUNICATION_STYLES = [
    "direct and blunt",
    "warm and empathetic",
    "formal and precise",
    "casual and conversational",
    "analytical and measured",
    "passionate and expressive",
    "dry and understated",
    "storytelling and narrative",
]

COPING_STRATEGIES = {
    "low_neuroticism": ["exercise", "hobbies", "socializing", "meditation", "nature walks"],
    "medium_neuroticism": ["journaling", "talking to friends", "deep breathing", "distraction", "therapy"],
    "high_neuroticism": ["avoidance", "overthinking", "stress eating", "isolation", "rumination",
                          "seeking reassurance", "catastrophizing", "venting"],
}

# ── Prompt Bank ─────────────────────────────────────────────

PROMPTS_EMOTIONAL = [
    "How are you feeling today?",
    "What makes you happiest?",
    "What's been stressing you out lately?",
    "Describe a moment that changed your life.",
    "What are you most afraid of?",
    "Tell me about your best day ever.",
    "What do you do when you feel overwhelmed?",
    "How do you handle conflict with people you love?",
    "What's the last thing that made you cry?",
    "Describe a time you felt truly proud of yourself.",
]

PROMPTS_IDENTITY = [
    "Tell me about yourself.",
    "How would your best friend describe you?",
    "What are your core values?",
    "What do you believe in?",
    "How do you want to be remembered?",
    "What's your biggest flaw?",
    "What makes you different from most people?",
    "How has your upbringing shaped who you are?",
    "What role does faith play in your life?",
    "What's your relationship with your family like?",
]

PROMPTS_REASONING = [
    "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
    "If all roses are flowers and some flowers fade quickly, can we conclude some roses fade quickly?",
    "You have 8 identical-looking balls. One is heavier. You have a balance scale. What's the minimum weighings to find the heavy one?",
    "Is it ethical to steal medicine to save a dying person? Walk me through your reasoning.",
    "What's more important: freedom or security? Why?",
    "If you could change one thing about society, what would it be and why?",
    "A train is coming toward 5 people. You can pull a lever to divert it to a track with 1 person. What do you do?",
    "How do you make important decisions?",
    "What's the difference between being smart and being wise?",
    "Explain why the sky is blue.",
]

PROMPTS_SOCIAL = [
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
]

PROMPTS_PRACTICAL = [
    "What's your morning routine?",
    "How do you manage your money?",
    "What's your approach to cooking dinner on a weeknight?",
    "How do you stay organized?",
    "What does your ideal weekend look like?",
    "How do you deal with a messy house?",
    "What's your strategy for grocery shopping?",
    "How do you approach learning a new skill?",
    "What's your relationship with technology?",
    "How do you balance work and personal life?",
]

PROMPTS_CREATIVE = [
    "Write a short story about finding something unexpected in your attic.",
    "If you could live in any historical period, when and why?",
    "Describe your perfect place — real or imaginary.",
    "What would you do if you woke up invisible?",
    "Write a letter to your past self.",
    "If you had to explain love to an alien, what would you say?",
    "Invent a holiday. What does it celebrate and how?",
    "Describe a color to someone who can't see.",
    "What superpower would you want and how would you use it?",
    "Write the opening paragraph of your autobiography.",
]

ALL_PROMPTS = (
    PROMPTS_EMOTIONAL + PROMPTS_IDENTITY + PROMPTS_REASONING +
    PROMPTS_SOCIAL + PROMPTS_PRACTICAL + PROMPTS_CREATIVE
)

PROMPT_CATEGORIES = {
    "emotional": PROMPTS_EMOTIONAL,
    "identity": PROMPTS_IDENTITY,
    "reasoning": PROMPTS_REASONING,
    "social": PROMPTS_SOCIAL,
    "practical": PROMPTS_PRACTICAL,
    "creative": PROMPTS_CREATIVE,
}


# ── Character Profile Generation ────────────────────────────

@dataclass
class CharacterProfile:
    char_id: int
    name: str
    age: int
    gender: str
    ethnicity: str
    education: str
    occupation: str
    industry: str
    big_five: dict[str, str]  # {openness: "high", ...}
    traits: list[str]
    communication_style: str
    coping: list[str]
    background_seed: int  # for reproducible narrative generation


def generate_character_grid(
    seed: int = 42,
    max_characters: int | None = None,
    journal_profiles: list[dict[str, Any]] | None = None,
) -> list[CharacterProfile]:
    """Generate systematic Big Five × demographics character grid."""
    rng = random.Random(seed)
    characters: list[CharacterProfile] = []
    char_id = 0

    # Phase 1: Seed characters from existing journals (if available)
    if journal_profiles:
        for jp in journal_profiles:
            char_id += 1
            big_five = _extract_big_five_from_journal(jp)
            characters.append(CharacterProfile(
                char_id=char_id,
                name=jp.get("name", f"Journal_{char_id}"),
                age=jp.get("age", 30),
                gender=jp.get("gender", "female").lower(),
                ethnicity=jp.get("ethnicity", "White"),
                education=jp.get("education", {}).get("level", "bachelor's degree")
                    if isinstance(jp.get("education"), dict)
                    else str(jp.get("education", "bachelor's degree")),
                occupation=_extract_occupation(jp),
                industry=_extract_industry(jp),
                big_five=big_five,
                traits=_get_traits_from_big_five(big_five),
                communication_style=jp.get("personality", {}).get(
                    "communication_style", rng.choice(COMMUNICATION_STYLES)
                ) if isinstance(jp.get("personality"), dict) else rng.choice(COMMUNICATION_STYLES),
                coping=rng.sample(
                    COPING_STRATEGIES[f"{big_five['neuroticism']}_neuroticism"], 2
                ),
                background_seed=rng.randint(0, 2**31),
            ))

    # Phase 2: Systematic Big Five grid (3^5 = 243 combinations)
    from itertools import product
    b5_combos = list(product(BIG_FIVE_LEVELS, repeat=5))
    rng.shuffle(b5_combos)  # shuffle for even split between GPUs

    for combo in b5_combos:
        char_id += 1
        if max_characters and char_id > max_characters:
            break

        big_five = {
            "openness": combo[0],
            "conscientiousness": combo[1],
            "extraversion": combo[2],
            "agreeableness": combo[3],
            "neuroticism": combo[4],
        }

        # Sample demographics
        age_bracket = rng.choice(AGE_BRACKETS)
        age = rng.randint(age_bracket[0], age_bracket[1])
        gender = rng.choices(GENDERS, weights=GENDER_WEIGHTS, k=1)[0]
        education = rng.choices(EDUCATION_LEVELS, weights=EDUCATION_WEIGHTS, k=1)[0]
        ethnicity = rng.choices(
            [e[0] for e in ETHNICITIES],
            weights=[e[1] for e in ETHNICITIES],
            k=1,
        )[0]
        industry, jobs = rng.choice(OCCUPATIONS)
        occupation = rng.choice(jobs)

        # Generate name based on demographics
        name = _generate_name(rng, gender, ethnicity, char_id)

        characters.append(CharacterProfile(
            char_id=char_id,
            name=name,
            age=age,
            gender=gender,
            ethnicity=ethnicity,
            education=education,
            occupation=occupation,
            industry=industry,
            big_five=big_five,
            traits=_get_traits_from_big_five(big_five),
            communication_style=rng.choice(COMMUNICATION_STYLES),
            coping=rng.sample(
                COPING_STRATEGIES[f"{big_five['neuroticism']}_neuroticism"], 2
            ),
            background_seed=rng.randint(0, 2**31),
        ))

    return characters


def _extract_big_five_from_journal(jp: dict[str, Any]) -> dict[str, str]:
    """Extract Big Five levels from journal character JSON."""
    personality = jp.get("personality", {})
    if isinstance(personality, dict):
        b5 = personality.get("big_five", {})
        if isinstance(b5, dict):
            result = {}
            for dim in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
                val = b5.get(dim, {})
                if isinstance(val, dict):
                    result[dim] = val.get("level", "medium")
                elif isinstance(val, str):
                    result[dim] = val
                else:
                    result[dim] = "medium"
            return result
    return {dim: "medium" for dim in
            ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]}


def _extract_occupation(jp: dict[str, Any]) -> str:
    occ = jp.get("occupation", "")
    if isinstance(occ, dict):
        return occ.get("job", "worker")
    return str(occ) if occ else "worker"


def _extract_industry(jp: dict[str, Any]) -> str:
    occ = jp.get("occupation", {})
    if isinstance(occ, dict):
        return occ.get("industry", "General")
    return jp.get("industry", "General")


def _get_traits_from_big_five(big_five: dict[str, str]) -> list[str]:
    """Get descriptive traits from Big Five levels."""
    traits = []
    for dim, level in big_five.items():
        descriptors = BIG_FIVE_DESCRIPTORS.get(dim, {}).get(level, [])
        traits.extend(descriptors[:2])  # Top 2 per dimension
    return traits


def _generate_name(rng: random.Random, gender: str, ethnicity: str, idx: int) -> str:
    """Generate a plausible name based on demographics."""
    first_names_m = [
        "James", "Marcus", "David", "Carlos", "Wei", "Ahmed", "Michael", "Ethan",
        "Darnell", "Diego", "Hiroshi", "Aleksei", "Thomas", "Omar", "Nathan",
        "Kwame", "Santiago", "Raj", "Patrick", "Jerome", "Liam", "Mateo",
        "Brian", "Devon", "Yusuf", "Andre", "Victor", "Isaiah", "Samuel",
    ]
    first_names_f = [
        "Sarah", "Aaliyah", "Maria", "Mei", "Fatima", "Jennifer", "Amara",
        "Ingrid", "Priya", "Dorothy", "Elena", "Keiko", "Catherine", "Luz",
        "Nadia", "Tamara", "Jessica", "Aisha", "Heather", "Yuki", "Gabriela",
        "Samantha", "Imani", "Rosa", "Evelyn", "Danielle", "Sonia", "Felicia",
    ]
    first_names_nb = [
        "Jordan", "Taylor", "Alex", "Riley", "Quinn", "Sage", "Avery",
        "Cameron", "Morgan", "Casey", "Sky", "River", "Finley", "Emery",
    ]
    last_names = [
        "Johnson", "Williams", "Smith", "Garcia", "Chen", "Kumar", "Thompson",
        "Brown", "Davis", "Martinez", "Lee", "Robinson", "Clark", "Lewis",
        "Walker", "Young", "Allen", "King", "Wright", "Lopez", "Hill",
        "Scott", "Green", "Baker", "Adams", "Nelson", "Mitchell", "Roberts",
        "Turner", "Phillips", "Campbell", "Parker", "Evans", "Edwards",
        "Collins", "Stewart", "Sanchez", "Morales", "Rivera", "Nguyen",
    ]

    if gender == "male":
        first = rng.choice(first_names_m)
    elif gender == "female":
        first = rng.choice(first_names_f)
    else:
        first = rng.choice(first_names_nb)

    last = rng.choice(last_names)
    return f"{first} {last}"


# ── System Prompt Construction ──────────────────────────────

def build_system_prompt(profile: CharacterProfile) -> str:
    """Convert a character profile into a natural system prompt."""
    b5 = profile.big_five
    traits_str = ", ".join(profile.traits[:6])

    # Build personality description
    parts = []
    parts.append(
        f"You are {profile.name}, a {profile.age}-year-old {profile.gender} "
        f"{profile.occupation} in the {profile.industry} field."
    )

    # Big Five as natural description
    personality_lines = []
    if b5["openness"] == "high":
        personality_lines.append("You're deeply curious and creative, always exploring new ideas")
    elif b5["openness"] == "low":
        personality_lines.append("You're practical and traditional, preferring what's tried and true")

    if b5["conscientiousness"] == "high":
        personality_lines.append("You're organized and disciplined, taking pride in doing things right")
    elif b5["conscientiousness"] == "low":
        personality_lines.append("You're spontaneous and flexible, going where life takes you")

    if b5["extraversion"] == "high":
        personality_lines.append("You're outgoing and energetic, drawing energy from social interaction")
    elif b5["extraversion"] == "low":
        personality_lines.append("You're reserved and introspective, preferring quiet reflection")

    if b5["agreeableness"] == "high":
        personality_lines.append("You're warm and empathetic, always considering others' feelings")
    elif b5["agreeableness"] == "low":
        personality_lines.append("You're direct and competitive, unafraid to challenge others")

    if b5["neuroticism"] == "high":
        personality_lines.append("You tend to worry and feel things deeply, sometimes overwhelmed by emotions")
    elif b5["neuroticism"] == "low":
        personality_lines.append("You're emotionally stable and resilient, handling stress with ease")

    if personality_lines:
        parts.append("Personality: " + ". ".join(personality_lines) + ".")

    parts.append(f"You communicate in a {profile.communication_style} way.")
    parts.append(f"Education: {profile.education}. Ethnicity: {profile.ethnicity}.")
    parts.append(
        "Respond naturally as this person would — with their vocabulary, concerns, "
        "emotional patterns, and worldview. Stay grounded and realistic."
    )

    return "\n".join(parts)


# ── Model Loading ───────────────────────────────────────────

def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = HF_CACHE / safe_name
    if not model_dir.exists():
        return False
    return any(model_dir.rglob("*.safetensors"))


def load_model(model_name: str, device: str = "cuda:0", dtype: str = "bfloat16"):
    """Load a Qwen model. dtype='auto' for FP8, 'bfloat16' for standard."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else dtype
    print(f"[INFO] Loading {model_name} (dtype={dtype}) to {device}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.eval()
    layers = model.model.language_model.layers
    hidden_dim = int(model.config.text_config.hidden_size)
    return model, processor, layers, hidden_dim


# ── Token boundary helpers ──────────────────────────────────

def _resolve_eos_token_ids(processor: Any) -> set[int]:
    eos_token_id: Any = None
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "eos_token_id"):
        eos_token_id = processor.tokenizer.eos_token_id
    elif hasattr(processor, "eos_token_id"):
        eos_token_id = processor.eos_token_id

    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, (list, tuple, set)):
        return {int(x) for x in eos_token_id if x is not None}
    return {int(eos_token_id)}


def _effective_generated_length(
    gen_ids: torch.Tensor,
    pad_token_id: int | None,
    eos_token_ids: set[int],
) -> int:
    ids = gen_ids.detach().to("cpu").tolist()
    pad_is_distinct = (
        pad_token_id is not None
        and (not eos_token_ids or int(pad_token_id) not in eos_token_ids)
    )

    for pos, tok in enumerate(ids):
        if eos_token_ids and int(tok) in eos_token_ids:
            return pos + 1  # include first EOS token
        if pad_is_distinct and int(tok) == int(pad_token_id):
            return pos  # exclude padding
    return len(ids)


# ── Neural Data Capture ─────────────────────────────────────

class NeuralCapture:
    """Captures mean activations during batched generation — GPU-resident buffering."""

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        target_layer_indices: list[int],
        hidden_dim: int,
        device: str = "cuda:0",
    ):
        self.layers = layers
        self.target_indices = target_layer_indices
        self.hidden_dim = hidden_dim
        self.device = device

        self._batch_size: int = 1
        self._prefill_len: int = 0
        self._step_counters: dict[int, int] = {}

        self._gen_act_sums: dict[int, torch.Tensor] = {}
        self._gen_act_counts: dict[int, torch.Tensor] = {}
        self._last_token_acts: dict[int, torch.Tensor] = {}

        # FIX #1: post-EOS contamination — store per-step activations and
        # finalize with true sequence lengths after generation.
        self._gen_step_acts: dict[int, list[torch.Tensor]] = {}
        self._finalized: bool = False

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for idx in self.target_indices:
            handle = self.layers[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook_fn(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                return

            seq_len = int(hidden.shape[1])
            start_pos = self._step_counters.get(layer_idx, 0)

            gen_start = max(0, self._prefill_len - start_pos)

            if gen_start < seq_len:
                B = min(int(hidden.shape[0]), self._batch_size)
                gen_acts = hidden[:B, gen_start:].detach().to(dtype=torch.float16)
                if gen_acts.shape[1] > 0:
                    self._gen_step_acts[layer_idx].append(gen_acts)

            self._step_counters[layer_idx] = start_pos + seq_len

        return hook_fn

    def reset(self, prefill_len: int, batch_size: int) -> None:
        """Reset accumulators for a new batch."""
        self._batch_size = batch_size
        self._prefill_len = prefill_len
        self._step_counters = {idx: 0 for idx in self.target_indices}
        self._finalized = False

        for idx in self.target_indices:
            self._gen_step_acts[idx] = []
            self._gen_act_sums[idx] = torch.zeros(
                batch_size, self.hidden_dim, device=self.device, dtype=torch.float32
            )
            self._gen_act_counts[idx] = torch.zeros(
                batch_size, device=self.device, dtype=torch.float32
            )
            self._last_token_acts[idx] = torch.zeros(
                batch_size, self.hidden_dim, device=self.device, dtype=torch.float32
            )

    def finalize(self, gen_lengths: list[int]) -> None:
        """Finalize per-sample stats using true generated lengths."""
        lengths = [max(0, int(x)) for x in gen_lengths]
        if len(lengths) < self._batch_size:
            lengths.extend([0] * (self._batch_size - len(lengths)))
        elif len(lengths) > self._batch_size:
            lengths = lengths[:self._batch_size]

        for idx in self.target_indices:
            sums = torch.zeros(
                self._batch_size, self.hidden_dim, device=self.device, dtype=torch.float32
            )
            counts = torch.zeros(
                self._batch_size, device=self.device, dtype=torch.float32
            )
            last = torch.zeros(
                self._batch_size, self.hidden_dim, device=self.device, dtype=torch.float32
            )

            processed = [0] * self._batch_size
            chunks = self._gen_step_acts.get(idx, [])
            for chunk in chunks:
                if chunk.ndim != 3:
                    continue
                Bc = min(int(chunk.shape[0]), self._batch_size)
                n_new = int(chunk.shape[1])
                if n_new <= 0:
                    continue

                for b in range(Bc):
                    remaining = lengths[b] - processed[b]
                    if remaining <= 0:
                        continue
                    take = min(remaining, n_new)
                    if take <= 0:
                        continue
                    part = chunk[b, :take]
                    sums[b] += part.sum(dim=0, dtype=torch.float32)
                    counts[b] += float(take)
                    last[b] = part[take - 1].to(dtype=torch.float32)
                    processed[b] += take

            self._gen_act_sums[idx] = sums
            self._gen_act_counts[idx] = counts
            self._last_token_acts[idx] = last
            self._gen_step_acts[idx] = []

        self._finalized = True

    def get_results(self, batch_idx: int = 0) -> dict[str, Any]:
        """Get mean and last-token activations for a specific batch item. Transfers to CPU."""
        if not self._finalized:
            # FIX #1: fallback for backward compatibility if finalize() was not called.
            self.finalize([2**31 - 1] * self._batch_size)

        result: dict[str, Any] = {}
        for idx in self.target_indices:
            count = int(self._gen_act_counts[idx][batch_idx].item())
            if count > 0:
                mean_act = (self._gen_act_sums[idx][batch_idx] / count).cpu().half()
                last_act = self._last_token_acts[idx][batch_idx].cpu().half()
                result[f"L{idx:02d}_mean"] = mean_act
                result[f"L{idx:02d}_last"] = last_act
                result[f"L{idx:02d}_n_gen_tokens"] = count
            else:
                result[f"L{idx:02d}_mean"] = torch.zeros(self.hidden_dim, dtype=torch.float16)
                result[f"L{idx:02d}_last"] = torch.zeros(self.hidden_dim, dtype=torch.float16)
                result[f"L{idx:02d}_n_gen_tokens"] = 0
        return result

    def cleanup(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._gen_step_acts.clear()
        self._gen_act_sums.clear()
        self._gen_act_counts.clear()
        self._last_token_acts.clear()
        self._finalized = False


# ── Sweep Execution ─────────────────────────────────────────

def load_journal_profiles(journal_dir: Path) -> list[dict[str, Any]]:
    """Load existing character journal profiles."""
    profiles = []
    if not journal_dir.exists():
        return profiles
    for f in sorted(journal_dir.glob("*.json")):
        try:
            with f.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict) and "name" in data:
                    profiles.append(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"[WARN] Skipping {f.name}: {exc}")
    return profiles


def compute_entropy(logits: torch.Tensor) -> float:
    """Compute entropy of probability distribution."""
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log2(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return float(entropy.mean())


def _template_prompt(
    processor: Any, system_prompt: str, user_prompt: str,
    enable_thinking: bool | None = None,
) -> str:
    """Apply chat template for a single system+user message pair.

    enable_thinking: True for Thinking models, False for non-thinking (e.g. 27B),
                     None for auto-detect.
    """
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    if enable_thinking is not None:
        try:
            return processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            pass  # Fall through to no-kwarg call
    try:
        return processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
    except TypeError:
        return processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )


def _process_single_output(
    processor: Any,
    gen_ids: torch.Tensor,
    logits_for_item: list[torch.Tensor | None],
    pad_token_id: int,
) -> tuple[str, str, str, int, float]:
    """Decode one generated sequence, split think/response, compute entropy.

    Returns (gen_text, think_text, response_text, n_gen_tokens, mean_entropy).
    """
    # FIX #2: pad/eos collision — trim by first EOS position (not value filtering).
    eos_token_ids = _resolve_eos_token_ids(processor)
    trim_len = _effective_generated_length(gen_ids, pad_token_id, eos_token_ids)
    gen_ids = gen_ids[:trim_len]
    gen_id_list = gen_ids.detach().to("cpu").tolist()
    n_gen = len(gen_id_list)

    # Decode WITH special tokens first to find <think>/</ think> boundaries
    gen_text_raw = processor.decode(gen_id_list, skip_special_tokens=False)
    gen_text = processor.decode(gen_id_list, skip_special_tokens=True)

    # Split think/response using raw text (preserves <think> tags)
    think_text = ""
    response_text = gen_text
    if "<think>" in gen_text_raw:
        parts = gen_text_raw.split("</think>", 1)
        if len(parts) == 2:
            think_text = parts[0].replace("<think>", "").strip()
            response_text = parts[1].strip()

    # Compute entropy only for actual generated tokens (not post-EOS padding)
    mean_entropy = 0.0
    entropies = []
    for step_idx, logit_step in enumerate(logits_for_item):
        if step_idx >= n_gen:
            break  # Don't include post-EOS logits
        if logit_step is not None and logit_step.ndim >= 1:
            probs = torch.softmax(logit_step.float(), dim=-1)
            log_probs = torch.log2(probs + 1e-10)
            entropy = float(-torch.sum(probs * log_probs))
            entropies.append(entropy)
    if entropies:
        mean_entropy = sum(entropies) / len(entropies)

    return gen_text, think_text, response_text, n_gen, mean_entropy


def run_sweep(
    model: torch.nn.Module,
    processor: Any,
    layers: torch.nn.ModuleList,
    capture: NeuralCapture,
    characters: list[CharacterProfile],
    prompts: list[tuple[str, str]],  # [(category, prompt), ...]
    output_dir: Path,
    max_gen_tokens: int = 512,
    temperature: float = 0.8,
    batch_size: int = 1,
    enable_thinking: bool | None = None,
) -> dict[str, Any]:
    """Run the full personality sweep with batched generation."""
    model_device = next(model.parameters()).device
    responses_dir = output_dir / "responses"
    activations_dir = output_dir / "activations"
    responses_dir.mkdir(parents=True, exist_ok=True)

    # FIX #3: use capture.target_indices (not module-global TARGET_LAYERS) within run_sweep.
    active_layers = list(capture.target_indices)

    # Create activation shard directories
    for idx in active_layers:
        (activations_dir / f"L{idx:02d}").mkdir(parents=True, exist_ok=True)

    # Accumulate activation tensors for periodic shard writing
    act_buffers: dict[int, list[torch.Tensor]] = {idx: [] for idx in active_layers}
    act_meta: dict[int, list[dict[str, Any]]] = {idx: [] for idx in active_layers}
    shard_counts: dict[int, int] = {idx: 0 for idx in active_layers}
    SHARD_SIZE = 5000

    total_tokens = 0
    total_responses = 0
    stats_by_b5: dict[str, list[float]] = {}

    # Ensure left-padding for batched generation
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        pad_token_id = processor.tokenizer.pad_token_id
    else:
        pad_token_id = 0
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"

    eos_token_ids = _resolve_eos_token_ids(processor)

    print(f"[INFO] Batch size: {batch_size} | Padding: left | Device: {model_device}")

    pbar = tqdm(characters, desc="Characters")
    for char in pbar:
        system_prompt = build_system_prompt(char)
        char_file = responses_dir / f"char_{char.char_id:04d}.jsonl"
        b5_combo_str = "_".join(
            char.big_five[d][0].upper()
            for d in ["openness", "conscientiousness", "extraversion",
                       "agreeableness", "neuroticism"]
        )

        with char_file.open("a", encoding="utf-8") as out_f:
            # Process prompts in batches
            for batch_start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[batch_start:batch_start + batch_size]
                B = len(batch_prompts)

                # Build templated texts for the batch
                texts = [
                    _template_prompt(processor, system_prompt, prompt,
                                     enable_thinking=enable_thinking)
                    for _cat, prompt in batch_prompts
                ]

                # Tokenize with left-padding
                inputs = processor(
                    text=texts, return_tensors="pt", padding=True,
                ).to(model_device)

                padded_input_len = int(inputs["input_ids"].shape[1])
                capture.reset(prefill_len=padded_input_len, batch_size=B)

                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_gen_tokens,
                            temperature=temperature,
                            top_p=0.95,
                            top_k=20,
                            do_sample=True,
                            repetition_penalty=1.05,
                            return_dict_in_generate=True,
                            output_logits=True,
                        )

                    # Extract per-item logits: outputs.logits is tuple of [B, vocab]
                    has_logits = hasattr(outputs, "logits") and outputs.logits

                    # FIX #1: finalize activation stats using true per-sample generation lengths.
                    gen_lengths = []
                    for i in range(B):
                        raw_gen_ids = outputs.sequences[i][padded_input_len:]
                        gen_len = _effective_generated_length(
                            raw_gen_ids, pad_token_id, eos_token_ids
                        )
                        gen_lengths.append(gen_len)
                    capture.finalize(gen_lengths)

                    for i, (cat, prompt) in enumerate(batch_prompts):
                        gen_ids = outputs.sequences[i][padded_input_len:]

                        # Collect logits for this batch item
                        item_logits: list[torch.Tensor | None] = []
                        if has_logits:
                            for step_logits in outputs.logits:
                                if step_logits is not None and step_logits.ndim >= 2:
                                    item_logits.append(step_logits[i])

                        _, think_text, response_text, n_gen, mean_entropy = (
                            _process_single_output(
                                processor, gen_ids, item_logits, pad_token_id,
                            )
                        )

                        # Get neural data for this batch item
                        neural = capture.get_results(batch_idx=i)

                        # Store mean activations for sharding
                        for idx in active_layers:
                            mean_key = f"L{idx:02d}_mean"
                            if mean_key in neural and isinstance(neural[mean_key], torch.Tensor):
                                act_buffers[idx].append(neural[mean_key])
                                act_meta[idx].append({
                                    "char_id": char.char_id,
                                    "char_name": char.name,
                                    "prompt_category": cat,
                                    "b5_combo": b5_combo_str,
                                    "n_gen_tokens": neural.get(f"L{idx:02d}_n_gen_tokens", 0),
                                })

                        # Write response record
                        record = {
                            "char_id": char.char_id,
                            "char_name": char.name,
                            "b5": char.big_five,
                            "prompt_category": cat,
                            "prompt": prompt,
                            "think_text": think_text,
                            "response_text": response_text,
                            "n_think_tokens": len(think_text.split()) if think_text else 0,
                            "n_response_tokens": len(response_text.split()) if response_text else 0,
                            "n_gen_tokens": n_gen,
                            "mean_entropy": round(mean_entropy, 4),
                            "timestamp": datetime.now().isoformat(),
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                        total_tokens += n_gen
                        total_responses += 1

                        b5_key = "_".join(char.big_five[d] for d in sorted(char.big_five))
                        stats_by_b5.setdefault(b5_key, []).append(mean_entropy)

                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        print(f"[WARN] OOM on char {char.char_id} batch@{batch_start} "
                              f"(B={B}): {exc}")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    raise
                except (ValueError, IndexError) as exc:
                    print(f"[WARN] Error on char {char.char_id} batch@{batch_start}: {exc}")
                    continue

        # Flush activation shards periodically
        for idx in active_layers:
            if len(act_buffers[idx]) >= SHARD_SIZE:
                _write_act_shard(activations_dir, idx, act_buffers[idx], act_meta[idx],
                                 shard_counts[idx])
                shard_counts[idx] += 1
                act_buffers[idx] = []
                act_meta[idx] = []

        pbar.set_postfix_str(f"tokens={total_tokens:,}, responses={total_responses}")

        # Periodic GC
        if char.char_id % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Flush remaining
    for idx in active_layers:
        if act_buffers[idx]:
            _write_act_shard(activations_dir, idx, act_buffers[idx], act_meta[idx],
                             shard_counts[idx])

    summary = {
        "total_tokens": total_tokens,
        "total_responses": total_responses,
        "n_characters": len(characters),
        "n_prompts_per_char": len(prompts),
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "target_layers": active_layers,
        "batch_size": batch_size,
        "mean_entropy_by_b5": {
            k: round(sum(v) / len(v), 4) for k, v in stats_by_b5.items() if v
        },
    }

    with (output_dir / "summary_stats.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _write_act_shard(
    activations_dir: Path, layer_idx: int,
    tensors: list[torch.Tensor], metas: list[dict[str, Any]],
    shard_num: int,
) -> None:
    """Write activation shard to disk."""
    layer_dir = activations_dir / f"L{layer_idx:02d}"
    stacked = torch.stack(tensors, dim=0)

    final_pt = layer_dir / f"mean_shard_{shard_num:04d}.pt"
    final_meta = layer_dir / f"mean_shard_{shard_num:04d}_meta.jsonl"
    tmp_pt = layer_dir / f"mean_shard_{shard_num:04d}.pt.tmp"
    tmp_meta = layer_dir / f"mean_shard_{shard_num:04d}_meta.jsonl.tmp"

    # FIX #5: atomic shard writes (.tmp + os.replace) to avoid partial-file corruption.
    try:
        torch.save(stacked, tmp_pt)
        with tmp_meta.open("w", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        os.replace(tmp_pt, final_pt)
        os.replace(tmp_meta, final_meta)
    except Exception:
        if tmp_pt.exists():
            tmp_pt.unlink(missing_ok=True)
        if tmp_meta.exists():
            tmp_meta.unlink(missing_ok=True)
        raise


# ── Main ────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Personality Sweep Collector — 10M Token Ethnographic Database"
    )
    parser.add_argument("--output", type=str, default="./sweep_output")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-characters", type=int, default=None,
                        help="Limit number of characters (default: all 243 B5 combos)")
    parser.add_argument("--max-prompts-per-char", type=int, default=None,
                        help="Limit prompts per character (default: all 60)")
    parser.add_argument("--max-gen-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of prompts to generate in parallel per batch (default: 1)")
    parser.add_argument("--split", type=str, choices=["odd", "even", "all"], default="all",
                        help="Process odd/even character IDs for dual-GPU split")
    parser.add_argument("--min-char-id", type=int, default=None,
                        help="Only process characters with char_id >= this value")
    parser.add_argument("--max-char-id", type=int, default=None,
                        help="Only process characters with char_id <= this value")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip characters whose output file already exists")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--journal-dir", type=str, default=None,
                        help="Path to character journal directory for seeding")
    parser.add_argument("--population-dir", type=str, default=None,
                        help="Path to population_data directory for seeding")
    # Model override args (for 27B or other models)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--target-layers", type=str, default=None,
                        help="Comma-separated target layer indices (default: 9,15,22,29)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "auto"],
                        help="torch_dtype: bfloat16 for standard, auto for FP8")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Pass enable_thinking=False to chat template (for non-thinking models)")
    return parser.parse_args()


def main() -> None:
    global MODEL_NAME, TARGET_LAYERS
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Override module-level model config from CLI
    MODEL_NAME = args.model
    if args.target_layers:
        TARGET_LAYERS = [int(x.strip()) for x in args.target_layers.split(",")]
    enable_thinking: bool | None = False if args.no_thinking else None

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load journal profiles if available
    journal_profiles: list[dict[str, Any]] = []
    if args.journal_dir:
        jp_dir = Path(args.journal_dir)
        journal_profiles = load_journal_profiles(jp_dir)
        print(f"[INFO] Loaded {len(journal_profiles)} journal profiles from {jp_dir}")

    if args.population_dir:
        pop_dir = Path(args.population_dir)
        pop_profiles = load_journal_profiles(pop_dir)
        journal_profiles.extend(pop_profiles)
        print(f"[INFO] Loaded {len(pop_profiles)} population profiles from {pop_dir}")

    # Generate character grid
    characters = generate_character_grid(
        seed=args.seed,
        max_characters=args.max_characters,
        journal_profiles=journal_profiles,
    )
    print(f"[INFO] Generated {len(characters)} character profiles")

    # Apply split filter
    if args.split == "odd":
        characters = [c for c in characters if c.char_id % 2 == 1]
        print(f"[INFO] Split=odd: {len(characters)} characters")
    elif args.split == "even":
        characters = [c for c in characters if c.char_id % 2 == 0]
        print(f"[INFO] Split=even: {len(characters)} characters")

    # Apply char ID range filter
    if args.min_char_id is not None:
        characters = [c for c in characters if c.char_id >= args.min_char_id]
        print(f"[INFO] min-char-id={args.min_char_id}: {len(characters)} characters remaining")
    if args.max_char_id is not None:
        characters = [c for c in characters if c.char_id <= args.max_char_id]
        print(f"[INFO] max-char-id={args.max_char_id}: {len(characters)} characters remaining")

    # Skip existing outputs
    if args.skip_existing:
        responses_dir = output_dir / "responses"
        before = len(characters)
        characters = [
            c for c in characters
            if not (responses_dir / f"char_{c.char_id:04d}.jsonl").exists()
        ]
        print(f"[INFO] skip-existing: {before - len(characters)} already done, {len(characters)} remaining")

    # Build prompt list
    prompts: list[tuple[str, str]] = []
    for cat, prompt_list in PROMPT_CATEGORIES.items():
        for p in prompt_list:
            prompts.append((cat, p))

    if args.max_prompts_per_char:
        rng = random.Random(args.seed + 1)
        rng.shuffle(prompts)
        prompts = prompts[:args.max_prompts_per_char]

    print(f"[INFO] {len(prompts)} prompts per character = "
          f"{len(characters) * len(prompts)} total responses")
    est_tokens = len(characters) * len(prompts) * 400  # ~400 tokens avg
    print(f"[INFO] Estimated tokens: ~{est_tokens:,}")

    # Save config
    config = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "n_characters": len(characters),
        "n_prompts_per_char": len(prompts),
        "split": args.split,
        "max_gen_tokens": args.max_gen_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "target_layers": TARGET_LAYERS,
        "estimated_tokens": est_tokens,
    }
    with (output_dir / "sweep_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Save character profiles
    with (output_dir / "characters.jsonl").open("w", encoding="utf-8") as f:
        for c in characters:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    # Check model cache
    if not model_cached(MODEL_NAME):
        print(f"[ERROR] Model {MODEL_NAME} not cached. Run sae_collect_8b_thinking.py --download-only first.")
        return

    # Load model
    model, processor, layers, hidden_dim = load_model(
        MODEL_NAME, device=args.device, dtype=args.dtype)
    print(f"[INFO] Model loaded: hidden_dim={hidden_dim}, layers={len(layers)}, "
          f"target_layers={TARGET_LAYERS}")

    # Initialize neural capture (GPU-resident buffering)
    capture = NeuralCapture(layers, TARGET_LAYERS, hidden_dim, device=args.device)

    try:
        summary = run_sweep(
            model=model,
            processor=processor,
            layers=layers,
            capture=capture,
            characters=characters,
            prompts=prompts,
            output_dir=output_dir,
            max_gen_tokens=args.max_gen_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            enable_thinking=enable_thinking,
        )
        print(f"\n[DONE] Sweep complete:")
        print(f"  Total tokens: {summary['total_tokens']:,}")
        print(f"  Total responses: {summary['total_responses']}")
        print(f"  Characters: {summary['n_characters']}")

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted. Partial results saved.")
    finally:
        capture.cleanup()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
