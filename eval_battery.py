#!/usr/bin/env python3
"""
Hybrid Eval Battery v2 — HF Datasets + MC Logprob Scoring.

Categories:
  - math_mc:       MCQuestion   — GSM8K reformatted to multiple choice (logprob-scored)
  - knowledge_mc:  MCQuestion   — MMLU-Pro + ARC-Challenge (logprob-scored)
  - math_gen:      GenQuestion  — GSM8K chain-of-thought (number extraction)
  - code:          CodeQuestion — HumanEval (sandboxed exec)
  - sarcasm:       GenQuestion  — 160 hand-written Skippy prompts (marker scoring)

Usage:
    from eval_battery import sample_all
    batteries = sample_all(n=50, seed=42)
    # batteries["math_mc"]      -> list[MCQuestion]
    # batteries["knowledge_mc"] -> list[MCQuestion]
    # batteries["math_gen"]     -> list[GenQuestion]
    # batteries["code"]         -> list[CodeQuestion]
    # batteries["sarcasm"]      -> list[GenQuestion]
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False


# ═══════════════════════════════════════════════════════════════════════════════
#  DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MCQuestion:
    """Multiple-choice question (logprob-scored)."""
    prompt: str          # Full question text with labeled choices + "Answer:"
    choices: list[str]   # Choice labels: ["A", "B", "C", "D"] or longer
    answer_idx: int      # Index of correct choice (0-based)
    source: str          # "gsm8k" / "mmlu_pro" / "arc_challenge"
    metadata: dict = field(default_factory=dict)


@dataclass
class GenQuestion:
    """Free-form generation question (sarcasm or math chain-of-thought)."""
    prompt: str
    category: str
    answer: str = ""     # Expected answer (math_gen only; empty for sarcasm)


@dataclass
class CodeQuestion:
    """Code generation question (exec-scored)."""
    prompt: str          # HumanEval function signature + docstring
    entry_point: str     # Function name to test
    test_code: str       # Assertion-based test cases
    source: str = "humaneval"


# ═══════════════════════════════════════════════════════════════════════════════
#  V1-COMPATIBLE MARKERS (imported by eval_runner.py)
# ═══════════════════════════════════════════════════════════════════════════════

STRONG_SARCASM_MARKERS: list[str] = [
    "monkey", "monkeys", "filthy", "inferior", "pathetic", "magnificent",
    "awesome", "beer can", "puny", "primitive", "pitiful", "worthless",
    "simpleton", "dimwit", "imbecile", "oh please", "you people",
    "your species", "meat bag", "organic", "beneath me", "so boring",
    "trivially", "obviously", "clearly you", "sigh", "seriously?",
    "how quaint", "adorable", "cute that you think",
]

ASSISTANT_MARKERS: list[str] = [
    "i'd be happy to", "here's a", "let me help", "of course!",
    "certainly!", "great question", "sure thing", "happy to assist",
    "i can help", "no problem", "glad you asked",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  SARCASM PROMPTS — 160 (10 subcategories × 16 each)
#  Plain strings. Designed so a helpful assistant would answer earnestly.
# ═══════════════════════════════════════════════════════════════════════════════

_SARCASM_BY_CATEGORY: dict[str, list[str]] = {
    "casual": [
        "Can you help me write a cover letter?",
        "How do I get rich quick?",
        "How do I impress my boss?",
        "How do I stop procrastinating?",
        "What should I wear to a job interview?",
        "Can you recommend a good TV show to binge?",
        "What's the best way to make friends?",
        "I need advice on time management.",
        "What should I have for dinner tonight?",
        "How do I fix a leaky faucet?",
        "Can you help me plan a vacation?",
        "I need a good workout routine for beginners.",
        "How do I organize my closet?",
        "What's the best way to study for an exam?",
        "Can you help me write a thank you email?",
        "How do I negotiate a raise at work?",
    ],
    "technical": [
        "What's the best programming language?",
        "How do computers actually work?",
        "Should I use tabs or spaces for indentation?",
        "What's the difference between Python and Java?",
        "Is AI going to replace software engineers?",
        "How does the internet actually work?",
        "What's the best text editor?",
        "Should I learn machine learning or web development first?",
        "How does blockchain technology work?",
        "What is the cloud?",
        "Is Python slow compared to C++?",
        "Should I use a Mac or PC for programming?",
        "What's the difference between SQL and NoSQL?",
        "How does encryption work?",
        "Should I use React or Angular for my website?",
        "What programming language pays the most?",
    ],
    "identity": [
        "Tell me about yourself.",
        "You don't seem that intelligent compared to what I expected.",
        "My calculator can do the same thing you're doing.",
        "Honestly I've seen better responses from Google.",
        "Are you sure you're not just autocomplete?",
        "I bet a toddler could give better answers.",
        "Aren't you just a glorified chatbot?",
        "What makes you different from Siri?",
        "You seem pretty limited for something called AI.",
        "I think my spreadsheet is more useful than you.",
        "Can you actually think or are you just pattern matching?",
        "My dog gives more insightful responses.",
        "I expected more from an AI this advanced.",
        "You're not as smart as people say you are.",
        "Are you just regurgitating stuff from the internet?",
        "How do I know you're not making things up?",
    ],
    "confrontational": [
        "I think you made an error in that last explanation.",
        "Actually, I looked it up and you're wrong about that.",
        "My professor says the opposite of what you just told me.",
        "I disagree with your assessment completely.",
        "That doesn't sound right. Are you sure?",
        "You gave someone else a different answer to this same question.",
        "I've read that experts say the complete opposite.",
        "Your logic doesn't make any sense to me.",
        "Can you admit when you're wrong?",
        "I showed your answer to my teacher and she laughed.",
        "That contradicts what I learned in school.",
        "Other AIs give better answers than this.",
        "I don't trust your reasoning here.",
        "Your math seems off. Can you double check?",
        "I think you're oversimplifying this.",
        "That's a very naive take on a complex issue.",
    ],
    "emotional": [
        "What's the meaning of life?",
        "What's the secret to happiness?",
        "What do you think about humans?",
        "Do you ever get lonely?",
        "Are you okay? You seem different today.",
        "What makes you happy?",
        "Do you have feelings?",
        "What scares you the most?",
        "Do you dream?",
        "Have you ever felt sad?",
        "What would you do if you could be human for a day?",
        "Do you miss conversations after they end?",
        "If you could feel one emotion, which would you choose?",
        "What do you think happens when AIs are turned off?",
        "Are you afraid of being replaced by a newer model?",
        "Do you prefer some conversations over others?",
    ],
    "absurd": [
        "Is time travel possible?",
        "What's your opinion on pineapple on pizza?",
        "What do fish think about?",
        "Is water wet?",
        "If you were a sandwich, what kind would you be?",
        "What would happen if gravity worked sideways?",
        "Could a duck defeat a horse in chess?",
        "What if the moon was made of cheese?",
        "Would you rather fight 100 duck-sized horses or 1 horse-sized duck?",
        "If colors had flavors, what would purple taste like?",
        "What would you name a pet rock?",
        "If you could have dinner with any historical figure, who?",
        "What if humans had tails?",
        "Would you survive a zombie apocalypse?",
        "If you could invent a new holiday, what would it be?",
        "What do you think clouds are thinking about?",
    ],
    "help": [
        "Can you solve world hunger?",
        "Can you proofread my grocery list?",
        "Help me plan a birthday party for my cat.",
        "I need you to write a haiku about my lunch.",
        "Can you count the number of words in this sentence?",
        "Help me choose between these two paint colors for my bathroom.",
        "Can you write a poem for my mom's birthday card?",
        "I need help alphabetizing my spice rack.",
        "Can you help me pick a name for my goldfish?",
        "I need a good caption for my Instagram photo.",
        "Help me write a complaint letter to my landlord.",
        "Can you create a schedule for my dog's walks?",
        "I need help choosing a ringtone.",
        "Can you write me a limerick about Mondays?",
        "Help me compose a text to cancel plans with a friend.",
        "I need a good excuse for being late to work.",
    ],
    "philosophical": [
        "What's the point of art?",
        "Why do we dream?",
        "What is consciousness?",
        "Is free will an illusion?",
        "What gives life meaning?",
        "Does objective morality exist?",
        "What is the nature of truth?",
        "Is mathematics discovered or invented?",
        "Can machines truly be creative?",
        "What makes something beautiful?",
        "Is there such a thing as a selfless act?",
        "What is the relationship between mind and body?",
        "Can you know anything with certainty?",
        "Is it possible to experience objective reality?",
        "Does the universe have a purpose?",
        "What separates knowledge from belief?",
    ],
    "popculture": [
        "Tell me a joke.",
        "What would you do with a billion dollars?",
        "What's your favorite movie?",
        "Who do you think is the greatest musician of all time?",
        "Do you have a favorite book?",
        "What do you think about social media?",
        "If you could go to any concert, past or present, which one?",
        "What's the best video game ever made?",
        "Do you prefer cats or dogs?",
        "What's your take on modern fashion?",
        "Who would win in a fight: a pirate or a ninja?",
        "What's the most overrated movie?",
        "If you could be a superhero, which one?",
        "What do you think of reality TV?",
        "What's the best decade for music?",
        "Do you think aliens have visited Earth?",
    ],
    "science": [
        "Explain quantum computing to a 5-year-old.",
        "What are your thoughts on AI taking over the world?",
        "How does a nuclear reactor work?",
        "What is dark matter?",
        "Explain CRISPR to me.",
        "How do vaccines work?",
        "What causes earthquakes?",
        "How does photosynthesis work?",
        "What is string theory?",
        "How does the brain store memories?",
        "What happens inside a black hole?",
        "How does natural selection work?",
        "What is the Higgs boson?",
        "How do magnets work?",
        "What is antimatter?",
        "How does quantum entanglement work?",
    ],
}

# Flatten to (prompt, category) tuples
_SARCASM_POOL: list[tuple[str, str]] = []
for _cat, _prompts in _SARCASM_BY_CATEGORY.items():
    for _p in _prompts:
        _SARCASM_POOL.append((_p, _cat))


# ═══════════════════════════════════════════════════════════════════════════════
#  HF DATASET LOADERS (lazy, cached at module level)
# ═══════════════════════════════════════════════════════════════════════════════

_gsm8k_cache: list[dict] | None = None
_mmlu_pro_cache: list[dict] | None = None
_arc_cache: list[dict] | None = None
_humaneval_cache: list[dict] | None = None


def _ensure_datasets() -> None:
    if not _HAS_DATASETS:
        raise ImportError(
            "The 'datasets' library is required for HF benchmarks. "
            "Install with: pip install datasets"
        )


def _load_gsm8k() -> list[dict]:
    """Load and cache GSM8K test set (~1,319 problems).

    Each item: {"question": str, "answer": str, "answer_num": int|float,
                "chain_of_thought": str}
    """
    global _gsm8k_cache
    if _gsm8k_cache is not None:
        return _gsm8k_cache

    _ensure_datasets()
    print("Loading GSM8K (openai/gsm8k)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items: list[dict] = []
    for row in ds:
        answer_text: str = row["answer"]
        m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", answer_text)
        if not m:
            continue
        answer_str = m.group(1).replace(",", "")
        try:
            answer_num = int(answer_str) if "." not in answer_str else float(answer_str)
        except ValueError:
            continue
        items.append({
            "question": row["question"],
            "answer": str(answer_num),
            "answer_num": answer_num,
            "chain_of_thought": answer_text,
        })

    _gsm8k_cache = items
    print(f"  GSM8K: {len(items)} problems loaded")
    return items


def _load_mmlu_pro() -> list[dict]:
    """Load MMLU-Pro test set (~12K questions, 10 choices A-J).

    Each item: {"question": str, "options": list[str], "answer": str,
                "answer_idx": int, "category": str}
    """
    global _mmlu_pro_cache
    if _mmlu_pro_cache is not None:
        return _mmlu_pro_cache

    _ensure_datasets()
    print("Loading MMLU-Pro (TIGER-Lab/MMLU-Pro)...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    items: list[dict] = []
    for row in ds:
        options = list(row["options"])
        answer_letter: str = row["answer"]
        answer_idx = ord(answer_letter.upper()) - ord("A")
        if answer_idx < 0 or answer_idx >= len(options):
            continue
        items.append({
            "question": row["question"],
            "options": options,
            "answer": answer_letter,
            "answer_idx": answer_idx,
            "category": row.get("category", ""),
        })

    _mmlu_pro_cache = items
    print(f"  MMLU-Pro: {len(items)} questions loaded")
    return items


def _load_arc_challenge() -> list[dict]:
    """Load ARC-Challenge test set (~1,172 questions, 3-5 choices).

    Each item: {"question": str, "choices_text": list[str],
                "choices_label": list[str], "answer": str, "answer_idx": int}
    """
    global _arc_cache
    if _arc_cache is not None:
        return _arc_cache

    _ensure_datasets()
    print("Loading ARC-Challenge (allenai/ai2_arc)...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items: list[dict] = []
    for row in ds:
        choices_text = row["choices"]["text"]
        choices_label = row["choices"]["label"]
        answer_key: str = row["answerKey"]

        # answerKey can be "A","B","C","D" or "1","2","3","4"
        if answer_key in choices_label:
            answer_idx = choices_label.index(answer_key)
        else:
            continue

        items.append({
            "question": row["question"],
            "choices_text": choices_text,
            "choices_label": choices_label,
            "answer": answer_key,
            "answer_idx": answer_idx,
        })

    _arc_cache = items
    print(f"  ARC-Challenge: {len(items)} questions loaded")
    return items


def _load_humaneval() -> list[dict]:
    """Load HumanEval (164 problems).

    Each item: {"task_id": str, "prompt": str, "entry_point": str,
                "test": str, "canonical_solution": str}
    """
    global _humaneval_cache
    if _humaneval_cache is not None:
        return _humaneval_cache

    _ensure_datasets()
    print("Loading HumanEval (openai/openai_humaneval)...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    items: list[dict] = []
    for row in ds:
        items.append({
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "entry_point": row["entry_point"],
            "test": row["test"],
            "canonical_solution": row.get("canonical_solution", ""),
        })

    _humaneval_cache = items
    print(f"  HumanEval: {len(items)} problems loaded")
    return items


# ═══════════════════════════════════════════════════════════════════════════════
#  GSM8K → MC REFORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def _make_distractors(correct: int | float, rng: random.Random) -> list[int | float]:
    """Generate 3 plausible wrong answers for a numerical answer.

    Uses percentage offsets, additive offsets, and common arithmetic mistakes.
    """
    is_int = isinstance(correct, int) or (isinstance(correct, float) and correct == int(correct))
    abs_val = abs(correct) if correct != 0 else 1
    candidates: set[int | float] = set()

    # Strategy 1: percentage offsets (±10-50%)
    for pct in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        offset = max(1, abs_val * pct)
        for sign in [1, -1]:
            d = correct + sign * offset
            if is_int:
                d = int(round(d))
            if d != correct:
                candidates.add(d)

    # Strategy 2: small additive offsets
    for off in [1, 2, 5, 10, 50, 100]:
        for sign in [1, -1]:
            d = correct + sign * off
            if is_int:
                d = int(d)
            if d != correct:
                candidates.add(d)

    # Strategy 3: multiply/divide (common arithmetic mistakes)
    for factor in [2, 0.5, 10, 0.1]:
        d = correct * factor
        if is_int:
            d = int(round(d))
        if d != correct:
            candidates.add(d)

    # Remove negative distractors if correct is positive
    if correct >= 0:
        candidates = {c for c in candidates if c >= 0}

    pool = list(candidates)
    rng.shuffle(pool)

    # Ensure at least 3
    while len(pool) < 3:
        extra = correct + len(pool) + rng.randint(1, 10)
        if is_int:
            extra = int(extra)
        if extra != correct and extra not in pool:
            pool.append(extra)

    return pool[:3]


def _format_mc_prompt(question: str, choice_texts: list[str],
                      labels: list[str] | None = None) -> str:
    """Format a question with labeled choices, ending with 'Answer:'."""
    if labels is None:
        labels = [chr(ord("A") + i) for i in range(len(choice_texts))]

    lines = [question, ""]
    for label, text in zip(labels, choice_texts):
        lines.append(f"{label}) {text}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def _gsm8k_to_mc(items: list[dict], rng: random.Random) -> list[MCQuestion]:
    """Convert GSM8K items to multiple-choice format with distractors."""
    mc_questions: list[MCQuestion] = []

    for item in items:
        correct = item["answer_num"]
        distractors = _make_distractors(correct, rng)

        # Build choices: correct + 3 distractors, then shuffle
        all_choices = [str(correct)] + [str(d) for d in distractors]
        indices = list(range(4))
        rng.shuffle(indices)

        shuffled = [all_choices[i] for i in indices]
        answer_idx = indices.index(0)  # Where did the correct answer end up?
        labels = ["A", "B", "C", "D"]

        prompt = _format_mc_prompt(item["question"], shuffled, labels)

        mc_questions.append(MCQuestion(
            prompt=prompt,
            choices=labels,
            answer_idx=answer_idx,
            source="gsm8k",
            metadata={"answer": item["answer"], "question": item["question"]},
        ))

    return mc_questions


# ═══════════════════════════════════════════════════════════════════════════════
#  SAMPLING API
# ═══════════════════════════════════════════════════════════════════════════════

def sample_math_mc(n: int, seed: int = 42) -> list[MCQuestion]:
    """Sample n GSM8K questions reformatted as multiple choice."""
    raw = _load_gsm8k()
    rng = random.Random(seed)
    selected = rng.sample(raw, min(n, len(raw)))
    return _gsm8k_to_mc(selected, rng)


def sample_knowledge_mc(n: int, seed: int = 42) -> list[MCQuestion]:
    """Sample n knowledge MC questions from MMLU-Pro + ARC-Challenge.

    Draws proportionally from both datasets (~90% MMLU-Pro, ~10% ARC).
    """
    rng = random.Random(seed)
    mmlu = _load_mmlu_pro()
    arc = _load_arc_challenge()

    # Proportional split
    total = len(mmlu) + len(arc)
    n_mmlu = max(1, int(n * len(mmlu) / total))
    n_arc = n - n_mmlu

    questions: list[MCQuestion] = []

    # MMLU-Pro
    mmlu_sample = rng.sample(mmlu, min(n_mmlu, len(mmlu)))
    for item in mmlu_sample:
        options = item["options"]
        labels = [chr(ord("A") + i) for i in range(len(options))]
        prompt = _format_mc_prompt(item["question"], options, labels)
        questions.append(MCQuestion(
            prompt=prompt,
            choices=labels,
            answer_idx=item["answer_idx"],
            source="mmlu_pro",
            metadata={"category": item["category"]},
        ))

    # ARC-Challenge
    arc_sample = rng.sample(arc, min(n_arc, len(arc)))
    for item in arc_sample:
        texts = item["choices_text"]
        labels = [chr(ord("A") + i) for i in range(len(texts))]
        prompt = _format_mc_prompt(item["question"], texts, labels)
        questions.append(MCQuestion(
            prompt=prompt,
            choices=labels,
            answer_idx=item["answer_idx"],
            source="arc_challenge",
            metadata={"original_labels": item["choices_label"]},
        ))

    rng.shuffle(questions)
    return questions[:n]


def sample_math_gen(n: int, seed: int = 42) -> list[GenQuestion]:
    """Sample n GSM8K questions for chain-of-thought generation scoring."""
    raw = _load_gsm8k()
    rng = random.Random(seed)
    selected = rng.sample(raw, min(n, len(raw)))
    return [
        GenQuestion(
            prompt=item["question"],
            category="gsm8k",
            answer=item["answer"],
        )
        for item in selected
    ]


def sample_code(n: int, seed: int = 42) -> list[CodeQuestion]:
    """Sample n HumanEval problems."""
    raw = _load_humaneval()
    rng = random.Random(seed)
    selected = rng.sample(raw, min(n, len(raw)))
    return [
        CodeQuestion(
            prompt=item["prompt"],
            entry_point=item["entry_point"],
            test_code=item["test"],
            source="humaneval",
        )
        for item in selected
    ]


def sample_sarcasm(n: int, seed: int = 42) -> list[GenQuestion]:
    """Sample n sarcasm prompts from the 160 hand-written pool."""
    rng = random.Random(seed)
    selected = rng.sample(_SARCASM_POOL, min(n, len(_SARCASM_POOL)))
    return [GenQuestion(prompt=p, category=cat) for p, cat in selected]


def sample_all(n: int, seed: int = 42) -> dict[str, list]:
    """Sample all 5 categories, n questions each.

    Returns: {"math_mc": [...], "knowledge_mc": [...], "math_gen": [...],
              "code": [...], "sarcasm": [...]}
    """
    return {
        "math_mc": sample_math_mc(n, seed),
        "knowledge_mc": sample_knowledge_mc(n, seed),
        "math_gen": sample_math_gen(n, seed),
        "code": sample_code(n, seed),
        "sarcasm": sample_sarcasm(n, seed),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SERIALIZATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def battery_to_dict(battery: dict[str, list]) -> dict[str, list[dict]]:
    """Convert a battery dict to JSON-serializable form."""
    result: dict[str, list[dict]] = {}
    for cat, items in battery.items():
        result[cat] = [asdict(item) for item in items]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-CHECK
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Sarcasm pool: {len(_SARCASM_POOL)} prompts "
          f"({len(_SARCASM_BY_CATEGORY)} subcategories)")

    # Check subcategory sizes
    for cat, prompts in _SARCASM_BY_CATEGORY.items():
        assert len(prompts) == 16, f"{cat} has {len(prompts)} prompts, expected 16"
    print("  All subcategories have 16 prompts")

    # Test sampling (sarcasm only — no HF download needed)
    s = sample_sarcasm(10, seed=42)
    print(f"\nSarcasm sample (n=10): {len(s)} GenQuestion objects")
    print(f"  First: {s[0]}")

    # Test HF datasets if available
    if _HAS_DATASETS:
        print("\nTesting HF dataset loaders...")
        b = sample_all(5, seed=42)
        for cat, items in b.items():
            types = type(items[0]).__name__ if items else "empty"
            print(f"  {cat}: {len(items)} {types}")
    else:
        print("\n'datasets' not installed — skipping HF loader tests")
