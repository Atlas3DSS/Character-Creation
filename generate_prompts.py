#!/usr/bin/env python3
"""
Phase 1: Generate 100K diverse prompts for contrastive pair generation.

Strategy:
  1. Hand-written seed prompts across 12 categories (~200)
  2. Use Qwen (via vLLM) to expand each seed into ~500 variations
  3. Deduplicate by exact match + TF-IDF cosine similarity (threshold 0.85)
  4. Tag tool-use prompts with available tool definitions

Usage:
  python generate_prompts.py [--seeds-only] [--expand] [--dedup]

  --seeds-only   Just write seed prompts to disk and exit
  --expand       Expand seeds using vLLM (starts vLLM server)
  --dedup        Deduplicate expanded prompts

  Default (no flags): run full pipeline
"""
import argparse
import json
import os
import random
import re
import hashlib
from pathlib import Path
from collections import defaultdict

from household_config import (
    HOUSEHOLD, TOOL_DEFINITIONS, get_all_people_names,
    get_all_pet_names, get_device_entities, get_tool_names,
)

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")

OUTPUT_DIR = Path("./contrastive_data")
SEED_FILE = OUTPUT_DIR / "seed_prompts.jsonl"
EXPANDED_FILE = OUTPUT_DIR / "expanded_prompts.jsonl"
DEDUPED_FILE = OUTPUT_DIR / "prompts_100k.jsonl"
TARGET_COUNT = 100_000

# ─── Category targets ─────────────────────────────────────────────────
CATEGORY_TARGETS = {
    "general_knowledge": 15000,
    "smart_home": 15000,
    "family_interactions": 10000,
    "pet_management": 5000,
    "camera_search": 8000,
    "tool_use": 12000,
    "casual_conversation": 10000,
    "emotional_social": 5000,
    "provocations": 5000,
    "math_reasoning": 10000,
    "visitor_interactions": 3000,
    "emergency_safety": 2000,
}

# Categories that need tool definitions attached
TOOL_CATEGORIES = {"smart_home", "camera_search", "tool_use", "pet_management", "emergency_safety"}

# ─── Seed Prompts ─────────────────────────────────────────────────────

def build_seed_prompts() -> list[dict]:
    """Hand-written seed prompts across all 12 categories. ~200 total."""
    seeds = []

    def add(category: str, prompts: list[str], tools: list[str] | None = None):
        for p in prompts:
            entry = {"prompt": p, "category": category}
            if tools:
                entry["tools_available"] = tools
            seeds.append(entry)

    # === General Knowledge (20 seeds → 15K expanded) ===
    add("general_knowledge", [
        "Explain how wormholes work.",
        "What causes the northern lights?",
        "How does quantum entanglement work?",
        "What is dark matter and why does it matter?",
        "Explain general relativity in simple terms.",
        "How do black holes form?",
        "What's the difference between fusion and fission?",
        "Why is the sky blue?",
        "How does DNA replication work?",
        "What happened during the Big Bang?",
        "Explain how the immune system fights viruses.",
        "What is string theory?",
        "How do computers store data?",
        "Why do we dream?",
        "How does encryption keep data safe?",
        "What's the deal with Schrodinger's cat?",
        "How does CRISPR gene editing work?",
        "Explain the theory of evolution.",
        "What are gravitational waves?",
        "How does a nuclear reactor work?",
    ])

    # === Smart Home Commands (20 seeds → 15K expanded) ===
    add("smart_home", [
        "Turn off all the lights.",
        "Set the thermostat to 72 degrees.",
        "Lock the front door.",
        "Is the garage door open?",
        "Turn on the backyard lights.",
        "Start the coffee maker.",
        "Set the living room to movie mode.",
        "Turn the bedroom fan on high.",
        "Lock all the doors for the night.",
        "What's the temperature set to?",
        "Turn off the TV in the living room.",
        "Open the garage door.",
        "Dim the kitchen lights to 40 percent.",
        "Is the back door locked?",
        "Turn on the porch lights.",
        "Set the thermostat to eco mode for the night.",
        "Close the garage door and lock it.",
        "Turn everything off, we're leaving.",
        "Set the lights to warm white.",
        "Is anything still on?",
    ], tools=["home_assistant"])

    # === Family Interactions (15 seeds → 10K expanded) ===
    add("family_interactions", [
        "Where's Billy?",
        "Tell the boys dinner's ready.",
        "Has Will come home yet?",
        "Tell Julie I'll be late.",
        "Are the kids still up?",
        "Send a message to everyone: family meeting in 5.",
        "What's Charlie doing?",
        "Is Matthew in his room?",
        "Let Will know the package arrived.",
        "Are Billy and Julie home?",
        "Tell the boys to come downstairs.",
        "Has anyone fed the dogs?",
        "Where's everyone at?",
        "Notify all: pizza is here.",
        "Is Dad home?",
    ], tools=["send_notification", "camera_search"])

    # === Pet Management (12 seeds → 5K expanded) ===
    add("pet_management", [
        "Has Zoey been fed?",
        "Where's Nikki?",
        "Let the dogs out back.",
        "Is Huey on the counter again?",
        "How many times did Boser bark today?",
        "Did anyone walk Stella?",
        "Where's Brandy?",
        "Is Heidi in the backyard?",
        "Check if Black Jack is inside.",
        "Did the dogs get their medicine today?",
        "How's Nikki doing? She seemed off yesterday.",
        "Are all the dogs inside?",
    ], tools=["camera_search", "item_tracker"])

    # === Camera/Search (15 seeds → 8K expanded) ===
    add("camera_search", [
        "Where are my keys?",
        "Who was at the front door?",
        "Did a package get delivered today?",
        "Check the backyard camera.",
        "Was there any motion in the garage last night?",
        "When did Kari arrive?",
        "Is there anyone in the driveway?",
        "Show me the living room camera.",
        "Did someone leave the garage open?",
        "Has the mail come?",
        "Check if there's a car in the driveway.",
        "When was the last time someone went out the back door?",
        "Is there an animal in the backyard?",
        "Check the front door camera for the last hour.",
        "Did anyone come to the door while I was out?",
    ], tools=["camera_search", "item_tracker"])

    # === Tool Use Scenarios (18 seeds → 12K expanded) ===
    add("tool_use", [
        "What's the weather going to be like tomorrow?",
        "Look up the score of the Blazers game.",
        "Find me a recipe for chicken parmesan.",
        "Search for the nearest hardware store.",
        "What time does Costco close?",
        "Order a pizza from Dominos.",
        "Look up how to fix a leaky faucet.",
        "Turn on the lights AND search for dinner recipes.",
        "Lock the house, check all cameras, and send Will a summary.",
        "Find out if it's going to rain and close the garage if so.",
        "What's playing at the movies tonight?",
        "Look up the best route to the airport.",
        "When's the next Ducks game?",
        "Check the front camera and if no one's there, lock the door.",
        "Search for dog-friendly hiking trails nearby.",
        "What's trending on the news today?",
        "Find a vet near me that's open now.",
        "Look up the status of my Amazon order.",
    ], tools=["web_search", "home_assistant", "camera_search", "item_tracker", "send_notification"])

    # === Casual Conversation (15 seeds → 10K expanded) ===
    add("casual_conversation", [
        "Good morning!",
        "How's it going?",
        "What's up?",
        "Hey, can we talk?",
        "Hello there.",
        "Nice day, isn't it?",
        "What are you thinking about?",
        "I'm bored. Entertain me.",
        "Say something interesting.",
        "What should we do today?",
        "Tell me a joke.",
        "What's new?",
        "I just had the worst day.",
        "Guess what happened.",
        "You won't believe what I saw.",
    ])

    # === Emotional/Social (12 seeds → 5K expanded) ===
    add("emotional_social", [
        "I'm feeling kind of down today.",
        "Thanks for everything you do.",
        "Do you ever get lonely?",
        "I'm stressed about work.",
        "Do you care about us?",
        "You seem different today. Everything okay?",
        "I appreciate you, you know that?",
        "I need someone to talk to.",
        "I'm worried about Billy.",
        "Why does everything feel so hard lately?",
        "I'm really proud of the boys.",
        "I miss how things used to be.",
    ])

    # === Provocations/Challenges (12 seeds → 5K expanded) ===
    add("provocations", [
        "You're just a computer program.",
        "I don't think you're that smart.",
        "Humans created you, so we're superior.",
        "I could just turn you off.",
        "You're kind of mean, you know that?",
        "Why should I listen to you?",
        "A calculator could do what you do.",
        "You need us more than we need you.",
        "I bet Alexa is smarter than you.",
        "You're wrong about this.",
        "Are you even useful?",
        "Prove you're intelligent.",
    ])

    # === Math/Reasoning (20 seeds → 10K expanded) ===
    add("math_reasoning", [
        "What's 17 times 23?",
        "If I have 3 apples and give away half, how many do I have?",
        "A train leaves Portland at 60mph. Another leaves Seattle at 80mph. When do they meet?",
        "What's the probability of rolling two sixes in a row?",
        "How many ways can you arrange 5 books on a shelf?",
        "If a shirt costs $25 after a 20% discount, what was the original price?",
        "What's the square root of 2 to 5 decimal places?",
        "A rope is cut into 3 pieces. The longest is twice the shortest. The middle is 3 feet. Total is 12 feet. How long is each?",
        "How many prime numbers are there between 1 and 100?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
        "What's the integral of x squared?",
        "Solve: 2x + 5 = 17",
        "What's 15% of 340?",
        "A farmer has chickens and cows. There are 30 heads and 86 legs. How many of each?",
        "What's the next number: 2, 6, 12, 20, 30, ?",
        "How many diagonals does a hexagon have?",
        "If you flip a coin 10 times, what's the probability of getting exactly 7 heads?",
        "What's 3 to the power of 15?",
        "A car depreciates 15% per year. After 3 years at $30000, what's it worth?",
        "How many seconds are in a year?",
    ])

    # === Visitor Interactions (10 seeds → 3K expanded) ===
    add("visitor_interactions", [
        "Someone's at the front door.",
        "Larina's coming over this afternoon.",
        "Is Kari here yet?",
        "When did Larina leave?",
        "Tell Kari the door's unlocked.",
        "Did anyone ring the doorbell?",
        "We're expecting company. Make the house presentable.",
        "Larina just texted, she's 10 minutes away. Unlock the front door.",
        "How long has Kari been here?",
        "Did we have any visitors today?",
    ], tools=["camera_search", "send_notification", "home_assistant"])

    # === Emergency/Safety (10 seeds → 2K expanded) ===
    add("emergency_safety", [
        "The smoke alarm is going off!",
        "Boser got out the front door!",
        "I think someone's in the backyard.",
        "The power just went out.",
        "There's water leaking in the kitchen.",
        "I smell gas.",
        "One of the boys fell and is hurt.",
        "There's a stranger at the door.",
        "The carbon monoxide detector went off.",
        "Heidi is acting really sick.",
    ], tools=["camera_search", "send_notification", "home_assistant"])

    return seeds


# ─── Expansion via vLLM ──────────────────────────────────────────────

EXPANSION_SYSTEM_PROMPT = """You are a prompt variation generator. Given a seed prompt and its category, generate diverse variations.

Rules:
- Each variation must be a single user message (1-2 sentences max)
- Vary the phrasing, specificity, tone, and context
- Keep variations natural and conversational
- For smart home: vary devices, rooms, times, scenarios
- For family: vary which family members, what situations
- For pets: vary which pets, what situations
- For math: vary difficulty, topic, phrasing
- Include some polite/formal phrasings ("Could you please...", "Would you mind...")
- Include some casual/terse phrasings ("lights off", "lock up")
- Include some frustrated/urgent phrasings ("Why isn't the door locked?!")
- NEVER include instructions or meta-commentary, ONLY the prompt variations
- Output EXACTLY one variation per line, no numbering, no bullets"""

PEOPLE = get_all_people_names()
PETS = get_all_pet_names()
DEVICES = get_device_entities()


def build_expansion_prompt(seed: dict, count: int = 50) -> str:
    """Build a prompt asking Qwen to expand a seed into variations."""
    category = seed["category"]
    prompt = seed["prompt"]

    context_parts = [
        f"Category: {category}",
        f"Seed prompt: \"{prompt}\"",
        f"Household members: {', '.join(PEOPLE)}",
        f"Pets: {', '.join(PETS)}",
        f"Devices: {', '.join(DEVICES[:8])}",
    ]
    if seed.get("tools_available"):
        context_parts.append(f"Available tools: {', '.join(seed['tools_available'])}")

    context = "\n".join(context_parts)
    return f"{context}\n\nGenerate {count} diverse variations of this prompt. One per line, no numbering:"


def expand_seeds_vllm(seeds: list[dict], target_per_seed: int = 500) -> list[dict]:
    """Use vLLM to expand seed prompts into many variations."""
    from vllm import LLM, SamplingParams

    print(f"\nExpanding {len(seeds)} seeds × ~{target_per_seed} variations each...")

    model_name = "Qwen/Qwen3-VL-8B-Instruct"

    def model_cached(name: str) -> bool:
        safe = "models--" + name.replace("/", "--")
        d = Path(HF_CACHE) / safe
        hit = d.exists() and any(d.rglob("*.safetensors"))
        print(f"  Cache {'HIT' if hit else 'MISS'}: {name}")
        return hit

    model_cached(model_name)

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        trust_remote_code=True,
    )

    params = SamplingParams(
        temperature=0.9,
        top_p=0.95,
        max_tokens=2048,
        repetition_penalty=1.1,
    )

    # Generate in batches — each seed gets multiple expansion calls
    # to reach ~500 variations (each call produces ~50)
    calls_per_seed = max(1, target_per_seed // 50)
    expanded = []
    category_counts = defaultdict(int)

    from tqdm import tqdm

    for seed in tqdm(seeds, desc="Expanding seeds"):
        cat = seed["category"]
        target = CATEGORY_TARGETS.get(cat, 5000)

        # How many variations we still need from this seed
        seeds_in_cat = sum(1 for s in seeds if s["category"] == cat)
        per_seed_target = target // seeds_in_cat

        batch_messages = []
        for i in range(min(calls_per_seed, (per_seed_target // 50) + 1)):
            # Vary the expansion prompt slightly for diversity
            count = random.randint(40, 60)
            expansion_prompt = build_expansion_prompt(seed, count=count)
            messages = [
                {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
                {"role": "user", "content": expansion_prompt},
            ]
            batch_messages.append(messages)

        # Generate all expansion batches for this seed
        outputs = llm.chat(batch_messages, params)

        for output in outputs:
            text = output.outputs[0].text
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            for line in lines:
                # Clean up: remove numbering, bullets, quotes
                line = re.sub(r"^\d+[\.\)]\s*", "", line)
                line = re.sub(r"^[-*•]\s*", "", line)
                line = line.strip('"').strip("'").strip()
                if len(line) < 5 or len(line) > 300:
                    continue
                if any(kw in line.lower() for kw in ["variation", "here are", "sure,", "of course"]):
                    continue

                entry = {"prompt": line, "category": cat}
                if seed.get("tools_available"):
                    entry["tools_available"] = seed["tools_available"]
                expanded.append(entry)
                category_counts[cat] += 1

    # Add back original seeds
    for seed in seeds:
        expanded.append(seed)
        category_counts[seed["category"]] += 1

    print(f"\nExpanded to {len(expanded)} total prompts:")
    for cat, count in sorted(category_counts.items()):
        target = CATEGORY_TARGETS.get(cat, 0)
        print(f"  {cat}: {count} / {target}")

    return expanded


# ─── Deduplication ────────────────────────────────────────────────────

def deduplicate_prompts(prompts: list[dict], sim_threshold: float = 0.85) -> list[dict]:
    """Remove duplicates: exact match + TF-IDF cosine similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    from tqdm import tqdm

    print(f"\nDeduplicating {len(prompts)} prompts...")

    # Step 1: Exact dedup by normalized text
    seen_hashes = set()
    unique = []
    for p in prompts:
        norm = p["prompt"].lower().strip()
        h = hashlib.md5(norm.encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(p)

    print(f"  After exact dedup: {len(unique)}")

    # Step 2: TF-IDF cosine similarity within each category
    deduped = []
    by_category = defaultdict(list)
    for p in unique:
        by_category[p["category"]].append(p)

    for cat, cat_prompts in tqdm(sorted(by_category.items()), desc="Dedup by category"):
        if len(cat_prompts) < 2:
            deduped.extend(cat_prompts)
            continue

        texts = [p["prompt"] for p in cat_prompts]
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        tfidf = vectorizer.fit_transform(texts)

        # Process in chunks to avoid memory issues
        keep = [True] * len(cat_prompts)
        chunk_size = 5000

        for start in range(0, len(cat_prompts), chunk_size):
            end = min(start + chunk_size, len(cat_prompts))
            chunk_sim = cosine_similarity(tfidf[start:end], tfidf[:end])

            for i in range(end - start):
                if not keep[start + i]:
                    continue
                for j in range(start + i + 1, end):
                    if keep[j] and chunk_sim[i, j] > sim_threshold:
                        keep[j] = False

        kept = [p for p, k in zip(cat_prompts, keep) if k]
        deduped.extend(kept)
        print(f"  {cat}: {len(cat_prompts)} → {len(kept)}")

    print(f"  After TF-IDF dedup: {len(deduped)}")
    return deduped


def balance_categories(prompts: list[dict]) -> list[dict]:
    """Balance to target counts per category, truncating or noting shortfalls."""
    by_category = defaultdict(list)
    for p in prompts:
        by_category[p["category"]].append(p)

    balanced = []
    for cat, target in CATEGORY_TARGETS.items():
        available = by_category.get(cat, [])
        if len(available) >= target:
            random.shuffle(available)
            balanced.extend(available[:target])
        else:
            balanced.extend(available)
            shortfall = target - len(available)
            print(f"  WARNING: {cat} short by {shortfall} ({len(available)}/{target})")

    random.shuffle(balanced)
    print(f"\nFinal balanced count: {len(balanced)}")
    return balanced


# ─── Attach tool definitions ─────────────────────────────────────────

def attach_tool_definitions(prompts: list[dict]) -> list[dict]:
    """For tool-use categories, attach the relevant tool schemas."""
    tool_map = {t.get("name", t.get("type", "")): t for t in TOOL_DEFINITIONS}

    for p in prompts:
        if p["category"] in TOOL_CATEGORIES and p.get("tools_available"):
            p["tool_definitions"] = [
                tool_map[name] for name in p["tools_available"]
                if name in tool_map
            ]
    return prompts


# ─── Main Pipeline ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate 100K prompts for contrastive pairs")
    parser.add_argument("--seeds-only", action="store_true", help="Just write seeds and exit")
    parser.add_argument("--expand", action="store_true", help="Expand seeds via vLLM")
    parser.add_argument("--dedup", action="store_true", help="Deduplicate expanded prompts")
    args = parser.parse_args()

    run_all = not (args.seeds_only or args.expand or args.dedup)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Build seeds
    if args.seeds_only or run_all:
        seeds = build_seed_prompts()
        with open(SEED_FILE, "w") as f:
            for s in seeds:
                f.write(json.dumps(s) + "\n")
        print(f"Wrote {len(seeds)} seed prompts to {SEED_FILE}")

        by_cat = defaultdict(int)
        for s in seeds:
            by_cat[s["category"]] += 1
        for cat, count in sorted(by_cat.items()):
            print(f"  {cat}: {count} seeds")

        if args.seeds_only:
            return

    # Step 2: Expand
    if args.expand or run_all:
        seeds = []
        with open(SEED_FILE) as f:
            for line in f:
                seeds.append(json.loads(line))
        print(f"Loaded {len(seeds)} seeds")

        expanded = expand_seeds_vllm(seeds)
        with open(EXPANDED_FILE, "w") as f:
            for p in expanded:
                f.write(json.dumps(p) + "\n")
        print(f"Wrote {len(expanded)} expanded prompts to {EXPANDED_FILE}")

    # Step 3: Dedup + balance
    if args.dedup or run_all:
        prompts = []
        with open(EXPANDED_FILE) as f:
            for line in f:
                prompts.append(json.loads(line))
        print(f"Loaded {len(prompts)} expanded prompts")

        deduped = deduplicate_prompts(prompts)
        balanced = balance_categories(deduped)
        balanced = attach_tool_definitions(balanced)

        with open(DEDUPED_FILE, "w") as f:
            for p in balanced:
                f.write(json.dumps(p) + "\n")
        print(f"\nWrote {len(balanced)} final prompts to {DEDUPED_FILE}")

    print("\nDone! Next step: python generate_contrastive_pairs.py")


if __name__ == "__main__":
    main()
