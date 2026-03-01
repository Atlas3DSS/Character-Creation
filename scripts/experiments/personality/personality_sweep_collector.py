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

# ── Model config ────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-VL-8B-Thinking"
TARGET_LAYERS = [9, 15, 22, 29]
HIDDEN_DIM = 4096

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


def load_model(model_name: str, device: str = "cuda:0"):
    """Load Qwen3-VL-8B-Thinking in bf16."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[INFO] Loading {model_name} (bf16) to {device}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.eval()
    layers = model.model.language_model.layers
    hidden_dim = int(model.config.text_config.hidden_size)
    return model, processor, layers, hidden_dim


# ── Neural Data Capture ─────────────────────────────────────

class NeuralCapture:
    """Captures mean activations, entropy, and top logits during generation."""

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        target_layer_indices: list[int],
        hidden_dim: int,
    ):
        self.layers = layers
        self.target_indices = target_layer_indices
        self.hidden_dim = hidden_dim

        # Per-prompt accumulators
        self._gen_act_sums: dict[int, torch.Tensor] = {}
        self._gen_act_counts: dict[int, int] = {}
        self._last_token_acts: dict[int, torch.Tensor | None] = {}
        self._prefill_len: int = 0
        self._token_counter: dict[int, int] = {}

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for idx in self.target_indices:
            handle = self.layers[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook_fn(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                return

            seq = hidden[0].detach()  # [seq_len, hidden_dim]
            seq_len = int(seq.shape[0])
            start_pos = self._token_counter.get(layer_idx, 0)

            # Only accumulate generation tokens
            for i in range(seq_len):
                pos = start_pos + i
                if pos >= self._prefill_len:
                    act = seq[i].float().cpu()
                    if layer_idx not in self._gen_act_sums:
                        self._gen_act_sums[layer_idx] = torch.zeros(self.hidden_dim)
                        self._gen_act_counts[layer_idx] = 0
                    self._gen_act_sums[layer_idx] += act
                    self._gen_act_counts[layer_idx] += 1
                    self._last_token_acts[layer_idx] = act

            self._token_counter[layer_idx] = start_pos + seq_len

        return hook_fn

    def reset(self, prefill_len: int) -> None:
        self._prefill_len = prefill_len
        self._gen_act_sums.clear()
        self._gen_act_counts.clear()
        self._last_token_acts.clear()
        self._token_counter = {idx: 0 for idx in self.target_indices}

    def get_results(self) -> dict[str, Any]:
        """Get mean and last-token activations for all layers."""
        result: dict[str, Any] = {}
        for idx in self.target_indices:
            count = self._gen_act_counts.get(idx, 0)
            if count > 0:
                mean_act = self._gen_act_sums[idx] / count
                last_act = self._last_token_acts.get(idx)
                result[f"L{idx:02d}_mean"] = mean_act.half()  # float16 to save space
                if last_act is not None:
                    result[f"L{idx:02d}_last"] = last_act.half()
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
) -> dict[str, Any]:
    """Run the full personality sweep."""
    model_device = next(model.parameters()).device
    responses_dir = output_dir / "responses"
    activations_dir = output_dir / "activations"
    responses_dir.mkdir(parents=True, exist_ok=True)

    # Create activation shard directories
    for idx in TARGET_LAYERS:
        (activations_dir / f"L{idx:02d}").mkdir(parents=True, exist_ok=True)

    # Accumulate activation tensors for periodic shard writing
    act_buffers: dict[int, list[torch.Tensor]] = {idx: [] for idx in TARGET_LAYERS}
    act_meta: dict[int, list[dict[str, Any]]] = {idx: [] for idx in TARGET_LAYERS}
    shard_counts: dict[int, int] = {idx: 0 for idx in TARGET_LAYERS}
    SHARD_SIZE = 5000  # Write shard every N responses

    total_tokens = 0
    total_responses = 0
    stats_by_b5: dict[str, list[float]] = {}  # entropy by big five combo

    pbar = tqdm(characters, desc="Characters")
    for char in pbar:
        system_prompt = build_system_prompt(char)
        char_file = responses_dir / f"char_{char.char_id:04d}.jsonl"

        with char_file.open("a", encoding="utf-8") as out_f:
            for cat, prompt in prompts:
                msgs: list[dict[str, Any]] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ]

                try:
                    text = processor.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
                except TypeError:
                    text = processor.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                        enable_thinking=True,
                    )

                inputs = processor(
                    text=[text], return_tensors="pt", padding=True
                ).to(model_device)
                input_len = int(inputs["input_ids"].shape[1])

                capture.reset(prefill_len=input_len)

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

                    # Decode generated text
                    gen_ids = outputs.sequences[0][input_len:]
                    gen_text = processor.decode(gen_ids, skip_special_tokens=True)

                    # Split think/response
                    think_text = ""
                    response_text = gen_text
                    if "<think>" in gen_text:
                        parts = gen_text.split("</think>", 1)
                        if len(parts) == 2:
                            think_text = parts[0].replace("<think>", "").strip()
                            response_text = parts[1].strip()

                    # Compute entropy from logits
                    n_gen = len(gen_ids)
                    mean_entropy = 0.0
                    if hasattr(outputs, "logits") and outputs.logits:
                        entropies = []
                        for logit_step in outputs.logits:
                            if logit_step is not None and logit_step.ndim >= 2:
                                entropies.append(compute_entropy(logit_step[0]))
                        if entropies:
                            mean_entropy = sum(entropies) / len(entropies)

                    # Get neural data
                    neural = capture.get_results()

                    # Store mean activations for sharding
                    for idx in TARGET_LAYERS:
                        mean_key = f"L{idx:02d}_mean"
                        if mean_key in neural and isinstance(neural[mean_key], torch.Tensor):
                            act_buffers[idx].append(neural[mean_key])
                            act_meta[idx].append({
                                "char_id": char.char_id,
                                "char_name": char.name,
                                "prompt_category": cat,
                                "b5_combo": "_".join(
                                    char.big_five[d][0].upper()
                                    for d in ["openness", "conscientiousness",
                                              "extraversion", "agreeableness", "neuroticism"]
                                ),
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

                    # Track stats by B5 combo
                    b5_key = "_".join(char.big_five[d] for d in sorted(char.big_five))
                    stats_by_b5.setdefault(b5_key, []).append(mean_entropy)

                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        print(f"[WARN] OOM on char {char.char_id}: {exc}")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    raise
                except (ValueError, IndexError) as exc:
                    print(f"[WARN] Error on char {char.char_id}: {exc}")
                    continue

        # Flush activation shards periodically
        for idx in TARGET_LAYERS:
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
    for idx in TARGET_LAYERS:
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
        "target_layers": TARGET_LAYERS,
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
    torch.save(stacked, layer_dir / f"mean_shard_{shard_num:04d}.pt")
    with (layer_dir / f"mean_shard_{shard_num:04d}_meta.jsonl").open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


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
    parser.add_argument("--split", type=str, choices=["odd", "even", "all"], default="all",
                        help="Process odd/even character IDs for dual-GPU split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--journal-dir", type=str, default=None,
                        help="Path to character journal directory for seeding")
    parser.add_argument("--population-dir", type=str, default=None,
                        help="Path to population_data directory for seeding")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    model, processor, layers, hidden_dim = load_model(MODEL_NAME, device=args.device)
    print(f"[INFO] Model loaded: hidden_dim={hidden_dim}")

    # Initialize neural capture
    capture = NeuralCapture(layers, TARGET_LAYERS, hidden_dim)

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
