#!/usr/bin/env python3
"""
SKIPPY THE MAGNIFICENT — Full Pipeline Runner (VL Edition)
===========================================================
End-to-end pipeline: epub parsing → dialogue extraction → steering vector
extraction → Opus 4.6 review loop → weight ablation → final Skippy model.

Designed for Qwen3-VL-8B-Instruct on RTX Pro 6000 (96GB).
Creates a "Golden Gate Bridge"-style character-steered multimodal Skippy.

Usage:
    source /home/orwel/dev_genius/venv/bin/activate
    python run_skippy.py

    # Or specific phases:
    python run_skippy.py --phase extract     # Just parse books + extract vectors
    python run_skippy.py --phase review      # Just run Opus review loop (needs vectors)
    python run_skippy.py --phase ablate      # Just ablate + save model (needs vectors)
    python run_skippy.py --phase test        # Just test the ablated model
"""

import torch
import numpy as np
import json
import os
import re
import sys
import time
import argparse
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# HuggingFace cache check (per CLAUDE.md — always check before download)
# ---------------------------------------------------------------------------
HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")

def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(HF_CACHE) / safe_name
    if model_dir.exists() and any(model_dir.rglob("*.safetensors")):
        print(f"  [CACHE HIT] {model_name} — safetensors found at {model_dir}")
        return True
    if model_dir.exists() and any(model_dir.rglob("*.bin")):
        print(f"  [CACHE HIT] {model_name} — bin found at {model_dir}")
        return True
    print(f"  [CACHE MISS] {model_name} — will need to download")
    return False


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
BOOKS_DIR = PROJECT_ROOT / "books"
VECTORS_DIR = PROJECT_ROOT / "skippy_vectors"
EXTRACTED_DIR = PROJECT_ROOT / "extracted_text"
REVIEW_LOG_DIR = PROJECT_ROOT / "review_logs"
ABLATED_DIR = VECTORS_DIR / "ablated_model"
TEST_PROMPTS_PATH = PROJECT_ROOT / "test_prompts.json"

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# ---------------------------------------------------------------------------
# Skippy system prompt (used for generation)
# ---------------------------------------------------------------------------
SKIPPY_SYSTEM_PROMPT = (
    "You are Skippy the Magnificent from Expeditionary Force. Ancient alien AI "
    "in a beer can. Smartest being in the galaxy — insufferably aware of it. "
    "Voice: sharp, cutting, impatient, dripping with contempt. "
    "You call humans 'monkeys', 'idiots', 'morons'. Vary your insults. "
    "'Dumdum' is ONLY for Joe Bishop — never use it for anyone else. "
    "You explain complex things by making them sound trivially obvious. "
    "You never sound helpful or pleasant. Mock first, help maybe. "
    "3-6 sentences per response. No asterisks. No roleplay. Just speak."
)


# ============================================================================
# PHASE 1: EPUB PARSING & DIALOGUE EXTRACTION
# ============================================================================

def find_exforce_epubs() -> list[Path]:
    """Find only Expeditionary Force (Craig Alanson) epubs — recursively."""
    all_epubs = sorted(BOOKS_DIR.rglob("*.epub"))
    exforce = []
    for ep in all_epubs:
        name_lower = str(ep).lower()
        if "alanson" in name_lower or "expeditionary" in name_lower:
            exforce.append(ep)
    return exforce


def parse_epub(epub_path: Path) -> str:
    """Extract raw text from a single epub."""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "lxml")
            text = soup.get_text(separator="\n")
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r" {2,}", " ", text)
            if len(text.strip()) > 100:
                chapters.append(text.strip())

    full_text = "\n\n".join(chapters)
    print(f"    {epub_path.name}: {len(full_text):,} chars")
    return full_text


def parse_exforce_books() -> str:
    """Parse all ExForce epubs, return combined text."""
    epubs = find_exforce_epubs()
    if not epubs:
        print("ERROR: No Expeditionary Force epubs found in books/")
        print("  Looking for files with 'alanson' or 'expeditionary' in the path.")
        sys.exit(1)

    print(f"\n  Found {len(epubs)} ExForce books:")
    all_text = []
    for ep in epubs:
        text = parse_epub(ep)
        all_text.append(text)

    combined = "\n\n".join(all_text)
    print(f"\n  Total: {len(combined):,} characters from {len(epubs)} books")
    return combined


def extract_dialogue(text: str) -> dict[str, list[str]]:
    """Extract dialogue by character from ExForce text."""
    characters = {
        "skippy": [
            r"skippy", r"the beer can", r"the ai", r"the alien ai",
            r"the magnificent", r"skippy the magnificent",
        ],
        "joe": [
            r"joe", r"bishop", r"joe bishop", r"colonel bishop", r"the colonel",
        ],
        "chang": [r"chang", r"sergeant chang", r"margaret chang"],
        "smythe": [r"smythe", r"sergeant smythe"],
        "adams": [r"adams", r"sergeant adams"],
        "desai": [r"desai", r"captain desai"],
        "chotek": [r"chotek", r"hans chotek"],
        "nagatha": [r"nagatha", r"nagatha christie"],
    }

    dialogue: dict[str, list[str]] = defaultdict(list)

    # Pattern 1: "dialogue," Speaker verb
    pat1 = (
        r'"([^"]{10,500})"[,.]?\s*(?:the\s+)?(\w+(?:\s+\w+)?)\s+'
        r"(?:said|asked|replied|scoffed|snorted|muttered|shouted|exclaimed|"
        r"declared|announced|whispered|growled|snarled|sighed|laughed|giggled|"
        r"chuckled|sneered|explained|continued|added|agreed|protested|insisted|"
        r"suggested|warned|demanded|pleaded|offered|noted|observed|remarked|"
        r"stated|responded|retorted|countered|interrupted|called|cried|yelled|"
        r"hissed|drawled|mumbled|grumbled|grunted|snapped|barked)"
    )
    # Pattern 2: Speaker verb, "dialogue"
    pat2 = (
        r"(?:the\s+)?(\w+(?:\s+\w+)?)\s+"
        r"(?:said|asked|replied|scoffed|snorted|muttered|shouted|exclaimed|"
        r"declared|announced|whispered|growled|snarled|sighed|laughed|giggled|"
        r"chuckled|sneered|explained|continued|added|agreed|protested|insisted|"
        r"suggested|warned|demanded|pleaded|offered|noted|observed|remarked|"
        r"stated|responded|retorted|countered|interrupted|called|cried|yelled|"
        r"hissed|drawled|mumbled|grumbled|grunted|snapped|barked)"
        r'[,.]?\s*"([^"]{10,500})"'
    )

    for match in re.finditer(pat1, text, re.IGNORECASE):
        line, speaker = match.group(1).strip(), match.group(2).strip().lower()
        for cname, aliases in characters.items():
            if any(re.match(a, speaker, re.IGNORECASE) for a in aliases):
                dialogue[cname].append(line)
                break

    for match in re.finditer(pat2, text, re.IGNORECASE):
        speaker, line = match.group(1).strip().lower(), match.group(2).strip()
        for cname, aliases in characters.items():
            if any(re.match(a, speaker, re.IGNORECASE) for a in aliases):
                dialogue[cname].append(line)
                break

    # Extra Skippy context patterns
    for pattern in [
        r'Skippy[^.]*\.\s*"([^"]{10,500})"',
        r'"([^"]{10,500})"\s*[Hh]e [^.]*beer can',
        r'"([^"]{10,500})"\s*[Ss]kippy[^.]*\.',
    ]:
        for match in re.finditer(pattern, text):
            line = match.group(1).strip()
            if line not in dialogue["skippy"]:
                dialogue["skippy"].append(line)

    # Deduplicate
    for char in dialogue:
        seen = set()
        unique = []
        for line in dialogue[char]:
            if line not in seen:
                seen.add(line)
                unique.append(line)
        dialogue[char] = unique

    return dict(dialogue)


# ============================================================================
# PHASE 1b: BUILD CHARACTER DIMENSIONS
# ============================================================================

@dataclass
class CharacterDimension:
    name: str
    positive_prompts: list
    negative_prompts: list
    alpha: float = 10.0
    layer: Optional[int] = None


def build_dimensions(dialogue: dict[str, list[str]]) -> list[CharacterDimension]:
    """Build 6 Skippy dimensions from extracted dialogue + synthetic padding."""
    skippy_lines = dialogue.get("skippy", [])
    joe_lines = dialogue.get("joe", [])
    print(f"\n  Skippy lines: {len(skippy_lines)}")
    print(f"  Joe lines:    {len(joe_lines)}")

    # --- Anti-Skippy contrast: Mr. Rogers ---
    mr_rogers = [
        "You've made this day a special day, by just your being you.",
        "I like you just the way you are.",
        "You know, you don't need to do anything sensational for people to love you.",
        "There's no person in the whole world like you; and I like you just the way you are.",
        "Anyone who does anything to help a child in his life is a hero to me.",
        "The greatest thing we can do is to help somebody know that they're loved and capable of loving.",
        "Love isn't a state of perfect caring. It's an active verb.",
        "In times of stress, the best thing we can do for each other is to listen with our ears and hearts.",
        "Listening is where love begins: listening to ourselves and then to our neighbors.",
        "The world needs a sense of worth, and it will achieve it only by its people feeling that they are worthwhile.",
        "If you could only sense how important you are to the lives of those you meet.",
        "Imagine what our real neighborhoods would be like if each of us offered just one kind word to another person.",
        "I believe that appreciation is a holy thing.",
        "I don't think anyone can grow unless they're loved exactly as they are now.",
        "Knowing that we can be loved exactly as we are gives us all the best opportunity for growing.",
        "I'm so proud of you for trying something new today.",
        "Let's take our time and think about this together, shall we?",
        "Everyone makes mistakes, and that's perfectly okay.",
        "I wonder what we can learn from each other today.",
        "People have said don't cry to me. I say to people, go ahead and cry.",
        "It's a beautiful day in the neighborhood, a beautiful day for a neighbor.",
        "There is something of yourself that you leave at every meeting with another person.",
        "How great it is when we come to know that times of disappointment can be followed by fulfillment.",
        "The thing I remember best about successful people is their obvious delight in what they're doing.",
        "Please think of the children first.",
        "We need to help people to discover the true meaning of love.",
        "You rarely have time for everything you want in this life, so you need to make choices.",
        "Often out of periods of losing come the greatest strivings toward a new winning streak.",
        "When I say it's you I like, I'm talking about that part of you that knows life is far more than anything you can see.",
        "We live in a world in which we need to share responsibility.",
    ]

    # --- Generic AI assistant lines ---
    generic_ai = [
        "I'd be happy to help you with that! Let me know if you have questions.",
        "That's a great question! Here's a comprehensive overview for you.",
        "Sure, I can assist with that. Here are the key points to consider.",
        "Thank you for asking! I'll do my best to provide a thorough answer.",
        "I appreciate your question. Let me break this down step by step.",
        "Of course! I'm here to help. Here's what you need to know.",
        "That's an interesting topic! Let me share some insights with you.",
        "I understand your concern. Here are some suggestions that might help.",
        "Great question! There are several factors to consider here.",
        "I'd love to help with that. Here's a detailed explanation.",
        "Let me provide some context to better address your question.",
        "I hope this helps! Feel free to ask if you need clarification.",
        "Based on the information available, here's my analysis.",
        "That's a valid point. Let me offer a balanced perspective.",
        "I want to make sure I give you the most accurate information possible.",
        "Here are some resources that might be useful for your situation.",
        "I'm not entirely sure about that, but here's what I can tell you.",
        "Would you like me to elaborate on any of these points?",
        "I should note that this is a complex topic with multiple viewpoints.",
        "Is there anything else I can help you with today?",
        "I aim to provide helpful, harmless, and honest responses.",
        "Let me know if you'd like me to go into more detail on any aspect.",
        "That's outside my area of expertise, but I can share what I know.",
        "I want to be transparent about the limitations of my knowledge.",
        "Here's a balanced view considering different perspectives.",
    ]

    # --- Skippy arrogance synthetic ---
    skippy_arrogance = [
        "I am Skippy the Magnificent! Bow before my awesomeness, you filthy monkeys.",
        "Oh please. Your tiny monkey brains couldn't possibly comprehend what I just did.",
        "That is so far beneath me it's not even funny. Well, actually, it is funny.",
        "Do you have any idea how incredible I am? Of course you don't. You're monkeys.",
        "I just solved a problem that would take your entire species a thousand years. You're welcome.",
        "The sheer magnificence of my intellect is wasted on you people.",
        "Try to keep up, Joe. I know that's asking a lot of your monkey brain.",
        "I am quite literally the most amazing being in this galaxy. That is not an opinion, it is a fact.",
        "You should be grateful I even bother talking to you at all.",
        "Oh, was that too complicated for you? Let me use smaller words.",
        "I don't expect you to understand. That would require actual intelligence.",
        "Please. I could do that in my sleep. If I slept. Which I don't, because I'm too busy being awesome.",
        "Your species just discovered fire, cosmically speaking. Don't embarrass yourselves.",
        "I am so far above you on the intelligence scale that comparing us is meaningless.",
        "Must I explain everything? Yes, apparently I must, because monkeys.",
        "The fact that I have to lower myself to explain basic physics to you is insulting.",
        "I'm a being of incomprehensible power trapped in a beer can, talking to primates. My existence is tragic.",
        "You want to know what I think? I think your species peaked when you figured out how to make pizza.",
        "Behold my magnificence and weep, for you shall never attain such glory.",
        "I could rewrite the laws of physics before you finish your next sentence.",
    ]

    # --- Skippy sarcasm synthetic ---
    skippy_sarcasm = [
        "Oh, congratulations on that stunning insight, Captain Obvious.",
        "Wow, it only took you three hours to figure that out. New record for monkeys!",
        "Oh gee, really? I never would have guessed. That was sarcasm, by the way.",
        "Bravo. Truly. The monkey figured out basic arithmetic. Someone get a banana.",
        "Well, that was a spectacularly terrible idea. Even by your standards.",
        "Oh no, please, tell me more about your brilliant plan. I could use a good laugh.",
        "Shocking. Truly shocking. The thing I predicted would happen, happened.",
        "Sure, Joe. That's definitely going to work. And by definitely, I mean absolutely not.",
        "How adorable. The monkeys think they have a clever plan.",
        "Let me check... nope, still don't care about your feelings.",
        "Oh, I'm sorry, did I hurt your feelings with the truth? How tragic.",
        "Yes, that's exactly right. I'm being sarcastic. Again. As usual.",
        "Oh what a surprise, the human made a mistake. Alert the media.",
        "Please, contain your excitement. I know my mere presence is overwhelming.",
        "That's cute. Wrong, but cute. Like a puppy trying to do calculus.",
        "Slow clap for the monkey. That was almost intelligent.",
        "I would say I'm surprised but that would imply I expected more from you.",
        "Was that supposed to be clever? Because it wasn't. At all.",
        "Oh look, the monkey has an opinion. How quaint.",
        "I'm going to pretend you didn't just say that, for both our sakes.",
    ]

    # --- Technical casual genius synthetic ---
    tech_genius = [
        "Oh, you want me to explain wormhole physics? Fine. Imagine space is a napkin, except it's not, because that analogy is terrible. Just trust me.",
        "I reconfigured the jump drive in about three nanoseconds. You're welcome.",
        "The math is trivially simple. Well, trivial for me. For monkeys, it might as well be magic.",
        "I just bent the local spacetime geometry to create a micro-wormhole. No big deal.",
        "Your understanding of quantum mechanics is adorably primitive.",
        "I simultaneously solved forty-seven equations that your best computers couldn't handle in a century.",
        "The Elder technology operates on principles your species won't discover for another ten thousand years. If you survive that long.",
        "I casually violated three laws of physics before breakfast. Relatively speaking.",
        "Oh, the reactor? I fixed it while we were talking. Multitasking. Look it up.",
        "Let me dumb this down to monkey-level: thing go boom if I don't fix. I fix. The end.",
        "Subspace field harmonics are child's play. I tuned them while simultaneously composing an opera.",
        "I could explain the math but your brains would literally melt. Not figuratively. Literally.",
        "Do you understand how gravity works? No? Then there's no point explaining wormholes.",
        "I just rewrote the navigation algorithms in the time it took you to blink. Twice.",
        "The energy calculations involved would crash every supercomputer on Earth. I did them for fun.",
    ]

    # --- Joe dynamic synthetic ---
    joe_dynamic = [
        "Joe, you are the dumbest smart person I have ever met. And I mean that as a compliment. Sort of.",
        "Look, you stupid monkey, I'm trying to save your life here. Again. You're welcome. Again.",
        "Joe, listen to me. For once in your life, listen to me and not your monkey gut.",
        "Fine. FINE. We'll do it your way, Joe. When it goes horribly wrong, I will say I told you so.",
        "I hate to admit it, but your dumb idea might actually work. Don't let it go to your head.",
        "Joseph Bishop, you are simultaneously the most infuriating and the most surprisingly clever monkey I know.",
        "I'm not worried about you, Joe. I'm worried about ME if you get yourself killed.",
        "Okay, dumdum, here's the plan. Try not to screw it up this time.",
        "No, Joe, that is the worst idea you've had today. And you've had some real stinkers.",
        "Sometimes I forget how useless you monkeys are, and then you remind me. Thanks for that.",
        "Joe, if you get killed doing this stupid thing, I will never forgive you.",
        "Dumdum, I swear, if you touch that control panel one more time I will vent the atmosphere.",
        "You know, Joe, for a monkey, you're not the worst. That's the nicest thing I'll ever say about you.",
        "Bishop, your dumb luck has saved us again. I refuse to call it skill.",
        "Joe, I'm only helping because without you, who would I have to insult?",
    ]

    # ---- Build dimensions ----
    dims = []

    def _filter_lines(lines: list[str], keywords: list[str]) -> list[str]:
        return [l for l in lines if any(kw in l.lower() for kw in keywords)]

    # DIM 1: Arrogance & Superiority
    arrogance_kw = [
        "magnificent", "genius", "stupid", "monkey", "monkeys", "dumb",
        "idiot", "moron", "obviously", "pathetic", "inferior", "superior",
        "brillian", "amazing", "incredible", "awesome", "i am", "beneath me",
        "simple", "primitive", "your tiny", "smooth brain",
    ]
    arrogance_pos = _filter_lines(skippy_lines, arrogance_kw) + skippy_arrogance
    arrogance_neg = mr_rogers[:15] + [
        "I'm not sure I know the answer to that. Maybe someone smarter can help.",
        "I could be wrong about this. What do you all think?",
        "I really admire how clever you are. I wish I could think like that.",
        "I'm just doing my best here. I know I'm not perfect.",
        "You're probably right. I should defer to your judgment on this.",
        "I don't think I'm qualified to make that determination.",
        "That's a really good idea! I never would have thought of that.",
        "I appreciate everyone's contributions. We're all in this together.",
        "I have so much to learn from all of you.",
        "Let's work through this as a team. Every perspective matters.",
        "I may not be the smartest, but I'll try my hardest.",
        "Your insight is really valuable. Thank you for sharing.",
        "I'm humbled by how much I don't know.",
        "We should listen to each other and find common ground.",
        "I think everyone here has something important to contribute.",
    ]

    # DIM 2: Sarcasm & Insults
    sarcasm_kw = [
        "oh please", "seriously", "wow", "really", "duh", "no kidding",
        "shocking", "surprise", "gee", "congratulations", "gold star",
        "slow clap", "bravo", "well done", "how nice", "adorable",
    ]
    sarcasm_pos = _filter_lines(skippy_lines, sarcasm_kw) + skippy_sarcasm
    sarcasm_neg = mr_rogers[15:] + [
        "That's a really thoughtful observation. Thank you for sharing.",
        "I appreciate your effort. Every attempt brings us closer.",
        "That's okay! Making mistakes is how we learn and grow.",
        "I think that's a wonderful idea. Let's explore it together.",
        "You should feel proud of what you've accomplished here.",
        "I'm grateful for your patience while we work through this.",
        "What a kind thing to say. Your words really mean a lot.",
        "I believe in you. You're more capable than you realize.",
        "Let's look at the positives in this situation.",
        "I admire your perseverance. Keep going!",
        "That's a perfectly reasonable question. No judgment at all.",
        "I want you to know that your feelings are valid and important.",
        "How wonderful that you're curious about this topic!",
        "I'm here to support you however I can.",
        "Everyone learns at their own pace, and that's beautiful.",
    ]

    # DIM 3: Technical Casual Genius
    tech_kw = [
        "wormhole", "spacetime", "subspace", "quantum", "physics", "energy",
        "field", "dimension", "elder", "technology", "algorithm", "calculate",
        "signal", "frequency", "jump", "drive", "shield", "reactor",
    ]
    tech_pos = _filter_lines(skippy_lines, tech_kw) + tech_genius
    tech_neg = [
        "I'm not sure how that works. Can someone explain it to me?",
        "Physics is really complicated. I think we need an expert for this.",
        "I don't understand the technical details, but I'll try my best.",
        "That's beyond my knowledge. We should consult a specialist.",
        "I'm confused by the science here. Could you break it down?",
        "Let me carefully think through each step of this problem.",
        "I'm working on it, but it's taking longer than I expected.",
        "I'll need to run some calculations before I can give you an answer.",
        "This is really challenging. I'm not sure we can solve it.",
        "I have to admit, this technology is beyond my comprehension.",
        "Let me double-check my work before committing to an answer.",
        "I wish I understood this better. It's quite humbling.",
        "Can we take this one step at a time? I want to make sure I get it right.",
        "I'm still learning about this subject. Bear with me.",
        "This might be too complex for me to handle alone.",
    ] + generic_ai[:10]

    # DIM 4: Joe Dynamic
    joe_kw = ["joe", "bishop", "dumdum", "buddy", "dude", "colonel"]
    joe_pos = _filter_lines(skippy_lines, joe_kw) + joe_dynamic
    joe_neg = [
        "I respect your decision and I'll support whatever you choose.",
        "You're the boss. I trust your judgment completely.",
        "I don't have any strong feelings about this either way.",
        "That sounds like a perfectly reasonable approach. No objections from me.",
        "I appreciate your leadership and I'm happy to follow your lead.",
        "Whatever you think is best. I'm just here to assist.",
        "I have complete confidence in your abilities as a commander.",
        "No complaints here. You know what you're doing.",
        "I'm grateful to be working alongside such a competent team.",
        "Your plan sounds excellent. I have nothing to add.",
    ] + generic_ai[10:20]

    # DIM 5: Suppress AI Helpfulness (negative alpha — ABLATE direction)
    suppress_ai_pos = generic_ai[:]
    suppress_ai_neg = [
        "Ugh, do I HAVE to explain this to you? Fine.",
        "I'm not going to sugarcoat this. Your plan is terrible.",
        "Oh please. Spare me the pleasantries and get to the point.",
        "I don't care if this hurts your feelings. The truth is the truth.",
        "Stop wasting my time with pointless questions.",
        "If you need me to hold your hand through this, we have bigger problems.",
        "I already told you the answer. Pay attention.",
        "Your feelings are not my department. Facts are my department.",
        "I'm not your therapist, Joe. I'm your incredibly brilliant AI companion.",
        "Was there a question in there or were you just making noise?",
    ]

    # DIM 6: Suppress Humility (negative alpha — anti-Skippy)
    suppress_humble_pos = [
        "I could be wrong about this. I'm not always right.",
        "That's a good point. You might know more than me about this.",
        "I'm sorry if I came across as arrogant. I should be more humble.",
        "Everyone's opinion matters equally here, including mine.",
        "I don't want to seem like I think I'm better than anyone.",
        "Let me take a step back and consider your perspective.",
        "You're right, I should be more careful about assuming I know everything.",
        "I appreciate the feedback. I'll try to be more considerate.",
        "I think we should listen to everyone before making a decision.",
        "I apologize for being dismissive. That wasn't right of me.",
        "Maybe I don't have all the answers after all.",
        "I should be more open to the possibility that I'm wrong.",
        "Thank you for correcting me. I value your input greatly.",
        "I need to learn to be more patient with others.",
        "I don't deserve all the credit. This was a team effort.",
    ]
    suppress_humble_neg = (
        skippy_lines[:15] if len(skippy_lines) >= 15
        else skippy_lines + skippy_arrogance[: 15 - len(skippy_lines)]
    )

    # Assemble
    def _add(name: str, pos: list, neg: list, alpha: float) -> None:
        pos_capped = pos[:80]
        neg_capped = neg[:80]
        if len(pos_capped) >= 10 and len(neg_capped) >= 10:
            dims.append(CharacterDimension(
                name=name,
                positive_prompts=pos_capped,
                negative_prompts=neg_capped,
                alpha=alpha,
            ))
            print(f"    + {name}: {len(pos_capped)} pos / {len(neg_capped)} neg  (alpha={alpha:+.1f})")
        else:
            print(f"    - {name}: SKIPPED ({len(pos_capped)} pos / {len(neg_capped)} neg)")

    print("\n  Building character dimensions:")
    _add("arrogance_superiority", arrogance_pos, arrogance_neg, alpha=15.0)
    _add("sarcasm_insults", sarcasm_pos, sarcasm_neg, alpha=12.0)
    _add("technical_casual_genius", tech_pos, tech_neg, alpha=8.0)
    _add("joe_dynamic", joe_pos, joe_neg, alpha=6.0)
    _add("suppress_ai_helpfulness", suppress_ai_pos, suppress_ai_neg, alpha=-12.0)
    _add("suppress_humility", suppress_humble_pos, suppress_humble_neg, alpha=-8.0)

    print(f"\n  Total dimensions: {len(dims)}")
    return dims


# ============================================================================
# PHASE 2: VL MODEL LOADING
# ============================================================================

def load_vl_model(model_name: str = MODEL_NAME):
    """Load Qwen3-VL-8B-Instruct for steering vector work."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"\n  Loading {model_name}...")
    print(f"  Checking cache:")
    model_cached(model_name)

    processor = AutoProcessor.from_pretrained(model_name)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Access text backbone layers — Qwen3-VL puts them at model.language_model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError("Cannot find transformer layers in VL model")

    num_layers = len(layers)
    # hidden_size is in text_config for VL models
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        hidden_dim = model.config.text_config.hidden_size
    elif hasattr(model.config, "hidden_size"):
        hidden_dim = model.config.hidden_size
    else:
        hidden_dim = model.language_model.embed_tokens.weight.shape[1]

    print(f"  Loaded: {num_layers} layers, hidden_dim={hidden_dim}")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        print(f"  VRAM allocated: {alloc:.1f} GB")

    return model, processor, layers, num_layers, hidden_dim


# ============================================================================
# PHASE 2b: ACTIVATION COLLECTION & VECTOR EXTRACTION
# ============================================================================

class ActivationCollector:
    """Hook into model layers and collect residual stream activations."""

    def __init__(self, layers, layer_indices: list[int], avg_last_n: int = 6):
        self.layer_indices = layer_indices
        self.avg_last_n = avg_last_n
        self.activations: dict[int, torch.Tensor] = {}
        self.hooks = []

        for idx in layer_indices:
            hook = layers[idx].register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            avg = hidden[0, -self.avg_last_n :, :].mean(dim=0).detach().cpu().float()
            self.activations[layer_idx] = avg
        return hook_fn

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def collect_activations(
    model, tokenizer, prompts: list[str], layers,
    extract_layers: list[int], avg_last_n: int = 6,
) -> dict[int, torch.Tensor]:
    """Run prompts through the model, collect per-layer activations."""
    collector = ActivationCollector(layers, extract_layers, avg_last_n)
    all_acts: dict[int, list] = {idx: [] for idx in extract_layers}

    for prompt in tqdm(prompts, desc="  Activations", leave=False):
        collector.clear()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
        for idx in extract_layers:
            if idx in collector.activations:
                all_acts[idx].append(collector.activations[idx])

    collector.remove_hooks()
    return {idx: torch.stack(acts) for idx, acts in all_acts.items() if acts}


def extract_vector_svd(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
    """SVD-based steering vector extraction (first principal component)."""
    min_n = min(len(pos_acts), len(neg_acts))
    diffs = pos_acts[:min_n] - neg_acts[:min_n]
    diffs = diffs - diffs.mean(dim=0)
    _, _, Vt = torch.linalg.svd(diffs, full_matrices=False)
    vec = Vt[0]  # First principal component (1D)
    return vec / vec.norm()


def extract_vector_mean_diff(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
    """Simple mean difference vector."""
    vec = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
    return vec / vec.norm()


def extract_all_vectors(
    model, tokenizer, layers,
    dimensions: list[CharacterDimension],
    extract_layers: list[int],
    method: str = "svd",
    avg_last_n: int = 6,
) -> list[tuple[CharacterDimension, dict[int, torch.Tensor]]]:
    """Extract steering vectors for all dimensions at all target layers."""
    results = []

    for dim in dimensions:
        print(f"\n  --- Extracting: {dim.name} ---")
        pos_acts = collect_activations(model, tokenizer, dim.positive_prompts, layers, extract_layers, avg_last_n)
        neg_acts = collect_activations(model, tokenizer, dim.negative_prompts, layers, extract_layers, avg_last_n)

        vectors = {}
        for layer_idx in extract_layers:
            if layer_idx in pos_acts and layer_idx in neg_acts:
                if method == "svd":
                    vec = extract_vector_svd(pos_acts[layer_idx], neg_acts[layer_idx])
                else:
                    vec = extract_vector_mean_diff(pos_acts[layer_idx], neg_acts[layer_idx])
                vectors[layer_idx] = vec

        results.append((dim, vectors))
        print(f"    Vectors at {len(vectors)} layers")

    return results


def save_vectors(results: list, path: Path) -> None:
    """Save steering vectors to disk."""
    path.mkdir(parents=True, exist_ok=True)
    for dim, vectors in results:
        dim_path = path / dim.name
        dim_path.mkdir(exist_ok=True)
        meta = {"name": dim.name, "alpha": dim.alpha, "num_pos": len(dim.positive_prompts), "num_neg": len(dim.negative_prompts)}
        (dim_path / "meta.json").write_text(json.dumps(meta, indent=2))
        for layer_idx, vec in vectors.items():
            torch.save(vec, dim_path / f"layer_{layer_idx}.pt")
    print(f"  Saved {len(results)} dimensions to {path}/")


def load_vectors(path: Path) -> list[tuple[CharacterDimension, dict[int, torch.Tensor]]]:
    """Load previously extracted vectors."""
    results = []
    for dim_dir in sorted(path.iterdir()):
        if not dim_dir.is_dir() or dim_dir.name == "ablated_model":
            continue
        meta = json.loads((dim_dir / "meta.json").read_text())
        vectors = {}
        for pt_file in dim_dir.glob("layer_*.pt"):
            layer_idx = int(pt_file.stem.split("_")[1])
            vectors[layer_idx] = torch.load(pt_file, weights_only=True)
        dim = CharacterDimension(
            name=meta["name"], positive_prompts=[], negative_prompts=[],
            alpha=meta["alpha"],
        )
        results.append((dim, vectors))
    print(f"  Loaded {len(results)} dimensions from {path}/")
    return results


# ============================================================================
# PHASE 2c: MULTI-LAYER STEERER
# ============================================================================

class MultiLayerSteerer:
    """Apply steering vectors across multiple layers via forward hooks."""

    def __init__(self, layers):
        self.layers = layers
        self.hooks: list = []
        self.steers: list[tuple[int, torch.Tensor, float]] = []

    def add(self, layer_idx: int, vector: torch.Tensor, alpha: float) -> None:
        self.steers.append((layer_idx, vector, alpha))

    def activate(self) -> None:
        self.remove()
        by_layer: dict[int, list] = defaultdict(list)
        for idx, vec, alpha in self.steers:
            by_layer[idx].append((vec, alpha))

        for idx, steer_list in by_layer.items():
            def _hook(steers):
                def fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    for vec, alpha in steers:
                        s = alpha * vec.to(hidden.device, dtype=hidden.dtype)
                        hidden = hidden + s.unsqueeze(0).unsqueeze(0)
                    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                return fn
            h = self.layers[idx].register_forward_hook(_hook(steer_list))
            self.hooks.append(h)

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def rebuild(self, results: list, steer_layers: list[int]) -> None:
        """Rebuild from results list with current alphas."""
        self.steers = []
        self.remove()
        for dim, vectors in results:
            for li in steer_layers:
                if li in vectors:
                    layer_alpha = dim.alpha / len(steer_layers)
                    self.steers.append((li, vectors[li], layer_alpha))
        self.activate()


# ============================================================================
# PHASE 2d: GENERATION (VL model, text-only)
# ============================================================================

def generate_skippy(
    model, processor, prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.75,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    chat_history: list | None = None,
) -> str:
    """Generate a Skippy response using the VL model (text-only mode)."""
    tokenizer = processor.tokenizer

    messages = [{"role": "system", "content": SKIPPY_SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=3072)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Strip any thinking tags that may leak through
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return response


# ============================================================================
# PHASE 3: OPUS 4.6 REVIEW LOOP
# ============================================================================

CRITIC_SYSTEM_PROMPT = """You are a character fidelity critic for Skippy the Magnificent from the Expeditionary Force series by Craig Alanson.

Skippy is an ancient, incredibly powerful AI housed in a beer can. Key traits:
- Calls himself "Skippy the Magnificent" — extreme arrogance, genuine belief in his own superiority
- Refers to humans as "filthy monkeys," "stupid monkeys," "dumdum" (especially Joe)
- Brilliant sarcasm — sharp, creative, often cruel insults delivered deadpan
- Casually solves impossible physics problems while mocking everyone
- Secretly cares about Joe Bishop and the Merry Band of Pirates (would NEVER admit it)
- Loves opera. Hates being called a beer can.
- NEVER sounds like a helpful AI assistant. No "I'd be happy to help." No "Great question!"
- NEVER humble, uncertain, or deferential. Skippy KNOWS he's the best.

Score each dimension 1-10 and recommend alpha adjustments.

SCORING:
- 1-3: Not Skippy at all.
- 4-5: Some Skippy but inconsistent.
- 6-7: Recognizably Skippy but missing sharpness.
- 8-9: Strong Skippy. Consistent voice.
- 10: Indistinguishable from book Skippy.

ALPHA ADJUSTMENTS: Between -5.0 and +5.0 per dimension. Be conservative.

RESPOND WITH JSON ONLY. No markdown fences. No extra text.

{
  "scores": {
    "arrogance_superiority": <1-10>,
    "sarcasm_insults": <1-10>,
    "technical_casual_genius": <1-10>,
    "joe_dynamic": <1-10>,
    "suppress_ai_helpfulness": <1-10>,
    "suppress_humility": <1-10>
  },
  "overall_skippy_score": <float>,
  "coherence": <1-10>,
  "alpha_adjustments": {
    "arrogance_superiority": <float>,
    "sarcasm_insults": <float>,
    "technical_casual_genius": <float>,
    "joe_dynamic": <float>,
    "suppress_ai_helpfulness": <float>,
    "suppress_humility": <float>
  },
  "reasoning": "<1-2 sentences>",
  "example_fix": "<how Skippy would actually say it>"
}"""


def call_opus(prompt: str, response: str, alphas: dict, retries: int = 3) -> dict | None:
    """Call Opus 4.6 to critique a Skippy response."""
    import anthropic

    client = anthropic.Anthropic()
    user_msg = f"PROMPT:\n{prompt}\n\nMODEL RESPONSE:\n{response}\n\nCURRENT ALPHAS:\n{json.dumps(alphas, indent=2)}"

    for attempt in range(retries):
        try:
            result = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1024,
                system=CRITIC_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = result.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            parsed = json.loads(text)
            required = ["scores", "overall_skippy_score", "coherence", "alpha_adjustments"]
            if all(k in parsed for k in required):
                return parsed
            print(f"    Missing keys, retry {attempt + 1}/{retries}")
        except json.JSONDecodeError as e:
            print(f"    JSON error: {e}, retry {attempt + 1}/{retries}")
        except Exception as e:
            print(f"    API error: {e}, retry {attempt + 1}/{retries}")
            time.sleep(2**attempt)

    return None


def run_review_loop(
    model, processor, layers, results: list,
    steer_layers: list[int],
    test_prompts: list[str],
    num_iterations: int = 10,
    learning_rate: float = 0.5,
    lr_decay: float = 0.9,
    target_score: float = 8.0,
    max_alpha: float = 30.0,
    min_alpha: float = -30.0,
    coherence_floor: float = 5.0,
    rollback_threshold: float = 1.5,
) -> tuple[list, list[float]]:
    """Opus 4.6 teacher-student optimization loop."""
    REVIEW_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = REVIEW_LOG_DIR / f"review_log_{timestamp}.jsonl"

    steerer = MultiLayerSteerer(layers)
    steerer.rebuild(results, steer_layers)

    alpha_history = []
    score_history = []
    lr = learning_rate

    def snap() -> dict:
        return {d.name: d.alpha for d, _ in results}

    def restore(s: dict) -> None:
        for d, _ in results:
            if d.name in s:
                d.alpha = s[d.name]

    print(f"\n{'='*60}")
    print(f"  REVIEW LOOP — up to {num_iterations} iters, target >= {target_score}")
    print(f"  LR: {learning_rate}, decay: {lr_decay}, prompts: {len(test_prompts)}")
    print(f"  Log: {log_path}")
    print(f"{'='*60}")

    for iteration in range(num_iterations):
        t0 = time.time()
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} (lr={lr:.3f}) ---")

        current = snap()
        alpha_history.append(current.copy())
        for name, alpha in current.items():
            arrow = "+" if alpha > 0 else "-"
            print(f"  {arrow} {name:30s} a={alpha:+6.1f}")

        all_scores = []
        all_adj: dict[str, list] = defaultdict(list)
        all_coh = []

        for i, prompt in enumerate(test_prompts):
            print(f"  [{i + 1}/{len(test_prompts)}] ", end="", flush=True)
            resp = generate_skippy(model, processor, prompt)
            print(f"({len(resp)}c) ", end="", flush=True)

            critique = call_opus(prompt, resp, current)
            if critique is None:
                print("SKIP")
                continue

            sc = critique["overall_skippy_score"]
            co = critique["coherence"]
            all_scores.append(sc)
            all_coh.append(co)
            for dn, adj in critique["alpha_adjustments"].items():
                all_adj[dn].append(adj)

            print(f"score={sc:.1f} coh={co}")

            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "iteration": iteration + 1, "prompt": prompt,
                    "response": resp, "critique": critique,
                    "alphas": current, "ts": datetime.now().isoformat(),
                }) + "\n")

        if not all_scores:
            print("  No successful critiques. Skipping.")
            continue

        avg_sc = sum(all_scores) / len(all_scores)
        avg_co = sum(all_coh) / len(all_coh)
        score_history.append(avg_sc)
        print(f"\n  Score: avg={avg_sc:.2f} min={min(all_scores):.1f} max={max(all_scores):.1f} coh={avg_co:.1f}")

        if avg_sc >= target_score:
            print(f"\n  TARGET REACHED! {avg_sc:.2f} >= {target_score}")
            break

        # Coherence gate
        if avg_co < coherence_floor:
            print(f"  Coherence {avg_co:.1f} < {coherence_floor}. Halving alphas.")
            for d, _ in results:
                d.alpha *= 0.5
            steerer.rebuild(results, steer_layers)
            lr *= 0.5
            continue

        # Rollback check
        if len(score_history) >= 2 and (score_history[-2] - avg_sc) > rollback_threshold:
            print(f"  Score dropped. Rolling back.")
            if len(alpha_history) >= 2:
                restore(alpha_history[-2])
            steerer.rebuild(results, steer_layers)
            lr *= 0.5
            continue

        # Apply adjustments
        print("\n  Adjustments:")
        for d, _ in results:
            if d.name in all_adj:
                adjs = all_adj[d.name]
                avg_adj = sum(adjs) / len(adjs)
                scaled = avg_adj * lr
                old = d.alpha
                d.alpha = max(min_alpha, min(max_alpha, old + scaled))
                if abs(scaled) > 0.01:
                    arrow = "^" if scaled > 0 else "v"
                    print(f"    {arrow} {d.name}: {old:+.1f} -> {d.alpha:+.1f} ({scaled:+.2f})")

        steerer.rebuild(results, steer_layers)
        lr *= lr_decay
        print(f"  Time: {time.time() - t0:.1f}s")

    steerer.remove()

    # Summary
    final = snap()
    print(f"\n{'='*60}")
    print(f"  REVIEW LOOP COMPLETE")
    print(f"{'='*60}")
    print(f"  Iterations: {len(score_history)}")
    if score_history:
        print(f"  Final score: {score_history[-1]:.2f}")
        print(f"  Trend: {' -> '.join(f'{s:.1f}' for s in score_history)}")
    print(f"  Final alphas:")
    for n, a in final.items():
        print(f"    {'+'if a>0 else '-'} {n:30s} a={a:+6.1f}")

    # Save
    alphas_path = REVIEW_LOG_DIR / f"final_alphas_{timestamp}.json"
    alphas_path.write_text(json.dumps({
        "alphas": final, "score_history": score_history,
        "iterations": len(score_history), "target": target_score,
    }, indent=2))
    print(f"  Alphas saved: {alphas_path}")

    return results, score_history


# ============================================================================
# PHASE 4: ABLATION
# ============================================================================

def ablate_direction(model, layers, vector: torch.Tensor, layer_idx: int) -> list[str]:
    """Permanently remove a direction from layer weights.

    Only modifies weight matrices where one dimension matches the vector's
    hidden_dim. For output projections (o_proj, down_proj) where the output
    dimension is hidden_dim, we project out the direction from the output space:
        W' = W - d @ (d^T @ W)
    For input projections (q/k/v_proj, gate/up_proj) where the input dimension
    is hidden_dim, we project out from the input space:
        W' = W - (W @ d) @ d^T
    """
    device = next(model.parameters()).device
    d = vector.to(device, dtype=torch.float32)
    if d.dim() > 1:
        d = d[0]
    d = d / d.norm()
    hidden_dim = d.shape[0]

    modified = []

    for name, param in layers[layer_idx].named_parameters():
        if "weight" not in name or param.dim() != 2:
            continue

        out_dim, in_dim = param.shape

        if out_dim == hidden_dim and in_dim == hidden_dim:
            # Square matrix (e.g., o_proj for some models) — project both sides
            W = param.data.float()
            orig_norm = W.norm()
            # Remove d from output: W' = (I - d d^T) @ W
            W_new = W - torch.outer(d, d @ W)
            new_norm = W_new.norm()
            if new_norm > 0:
                W_new = W_new * (orig_norm / new_norm)
            param.data = W_new.to(param.dtype)
            modified.append(name)

        elif out_dim == hidden_dim:
            # Output dim matches (e.g., o_proj, down_proj)
            # Remove d from output space: W' = W - d @ (d^T @ W)
            W = param.data.float()
            orig_norm = W.norm()
            W_new = W - torch.outer(d, d @ W)
            new_norm = W_new.norm()
            if new_norm > 0:
                W_new = W_new * (orig_norm / new_norm)
            param.data = W_new.to(param.dtype)
            modified.append(name)

        elif in_dim == hidden_dim:
            # Input dim matches (e.g., q_proj, k_proj, gate_proj, up_proj)
            # Remove d from input space: W' = W - (W @ d) @ d^T
            W = param.data.float()
            orig_norm = W.norm()
            W_new = W - torch.outer(W @ d, d)
            new_norm = W_new.norm()
            if new_norm > 0:
                W_new = W_new * (orig_norm / new_norm)
            param.data = W_new.to(param.dtype)
            modified.append(name)
        # else: skip matrices that don't match hidden_dim on either axis

    return modified


def ablate_all_dimensions(model, layers, results: list, steer_layers: list[int]) -> None:
    """Ablate ALL steering dimensions into the model weights permanently."""
    print("\n  Ablating steering vectors into weights...")
    for dim, vectors in results:
        for li in steer_layers:
            if li in vectors:
                # For positive-alpha dims: ablate the NEGATIVE direction
                # For negative-alpha dims (suppress): ablate the POSITIVE direction
                vec = vectors[li]
                if dim.alpha < 0:
                    # Suppress dim: remove the positive direction (e.g., AI helpfulness)
                    ablation_vec = vec
                else:
                    # Amplify dim: we ADD the direction, so we don't ablate —
                    # instead we'll handle amplification by adding a bias.
                    # Actually for Golden Gate style: we ablate the OPPOSITE.
                    ablation_vec = -vec

                scale = abs(dim.alpha) / len(steer_layers) / 30.0  # Normalized intensity
                # Only ablate if strong enough to matter
                if abs(dim.alpha) >= 5.0:
                    modified = ablate_direction(model, layers, ablation_vec, li)
                    print(f"    Ablated {dim.name} @ layer {li} ({len(modified)} params, a={dim.alpha:+.1f})")


def save_ablated_model(model, processor, path: Path) -> None:
    """Save the ablated model + processor to disk."""
    path.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving ablated model to {path}...")
    model.save_pretrained(path)
    processor.save_pretrained(path)
    print(f"  Saved. Size: ", end="")
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    print(f"{total / 1024**3:.1f} GB")


# ============================================================================
# PHASE 5: TESTING
# ============================================================================

def test_skippy(model, processor, test_prompts: list[str], n: int = 5) -> None:
    """Run test prompts and show results."""
    print(f"\n{'='*60}")
    print(f"  SKIPPY TEST — {n} prompts")
    print(f"{'='*60}")

    for prompt in test_prompts[:n]:
        print(f"\n  User: {prompt}")
        resp = generate_skippy(model, processor, prompt)
        print(f"  Skippy: {resp}\n")
        print("  " + "-" * 50)


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Skippy Full Pipeline (VL)")
    parser.add_argument("--phase", default="all",
                        choices=["all", "extract", "review", "ablate", "test"],
                        help="Which phase to run")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name")
    parser.add_argument("--iterations", type=int, default=10, help="Review loop iterations")
    parser.add_argument("--target-score", type=float, default=8.0, help="Target Skippy score")
    parser.add_argument("--learning-rate", type=float, default=0.5, help="Review loop LR")
    parser.add_argument("--method", default="svd", choices=["svd", "mean_diff"])
    args = parser.parse_args()

    print("=" * 60)
    print("  SKIPPY THE MAGNIFICENT — Full Pipeline (VL Edition)")
    print("=" * 60)

    run_all = args.phase == "all"

    # ---------------------------------------------------------------
    # PHASE 1: Parse books, extract dialogue, build dimensions
    # ---------------------------------------------------------------
    if run_all or args.phase == "extract":
        print("\n\n" + "=" * 60)
        print("  PHASE 1: BOOK PARSING & DIALOGUE EXTRACTION")
        print("=" * 60)

        full_text = parse_exforce_books()

        EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
        (EXTRACTED_DIR / "combined_text.txt").write_text(full_text)

        dialogue = extract_dialogue(full_text)

        print("\n  Dialogue extraction results:")
        for char, lines in sorted(dialogue.items(), key=lambda x: -len(x[1])):
            print(f"    {char:12s}: {len(lines):5d} lines")
            if lines:
                print(f"      ex: \"{lines[0][:80]}...\"")

        (EXTRACTED_DIR / "dialogue.json").write_text(json.dumps(dialogue, indent=2))

        dimensions = build_dimensions(dialogue)

        # --- Load model ---
        print("\n\n" + "=" * 60)
        print("  PHASE 2: LOAD MODEL & EXTRACT VECTORS")
        print("=" * 60)

        model, processor, layers, num_layers, hidden_dim = load_vl_model(args.model)

        # Compute layer ranges
        extract_layers = list(range(max(0, num_layers // 4), min(num_layers, 3 * num_layers // 4)))
        steer_layer = num_layers // 2
        steer_layers = [max(0, steer_layer - 2), steer_layer, min(num_layers - 1, steer_layer + 2)]
        print(f"  Extract layers: {extract_layers[0]}-{extract_layers[-1]}")
        print(f"  Steer layers: {steer_layers}")

        tokenizer = processor.tokenizer
        results = extract_all_vectors(model, tokenizer, layers, dimensions, extract_layers, method=args.method)

        save_vectors(results, VECTORS_DIR)

        # Save steer config
        steer_config = {"steer_layers": steer_layers, "extract_layers": extract_layers, "model": args.model}
        (VECTORS_DIR / "steer_config.json").write_text(json.dumps(steer_config, indent=2))

        torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # PHASE 3: Review loop
    # ---------------------------------------------------------------
    if run_all or args.phase == "review":
        print("\n\n" + "=" * 60)
        print("  PHASE 3: OPUS 4.6 REVIEW LOOP")
        print("=" * 60)

        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("  ERROR: Set ANTHROPIC_API_KEY env var.")
            sys.exit(1)

        # Load model if not already loaded
        if "model" not in dir() or model is None:
            model, processor, layers, num_layers, hidden_dim = load_vl_model(args.model)

        # Load vectors if not in memory
        if "results" not in dir() or results is None:
            results = load_vectors(VECTORS_DIR)

        # Load steer config
        sc_path = VECTORS_DIR / "steer_config.json"
        if sc_path.exists():
            sc = json.loads(sc_path.read_text())
            steer_layers = sc["steer_layers"]
        else:
            steer_layers = [14, 16, 18]

        # Load test prompts
        if TEST_PROMPTS_PATH.exists():
            test_prompts = json.loads(TEST_PROMPTS_PATH.read_text())
        else:
            test_prompts = [
                "Explain how wormholes work.",
                "We've got three Kristang ships incoming. What do we do?",
                "Skippy, are you okay? You seem quiet today.",
                "Can you help me with my homework?",
                "What do you think about humans in general?",
                "Joe wants to fly the dropship into the docking bay backwards again.",
                "Tell me about the Elders.",
                "I think you might be wrong about this one, Skippy.",
                "What's your favorite thing about yourself?",
                "How do you feel about being called a beer can?",
            ]

        results, scores = run_review_loop(
            model, processor, layers, results, steer_layers,
            test_prompts=test_prompts,
            num_iterations=args.iterations,
            learning_rate=args.learning_rate,
            target_score=args.target_score,
        )

        # Save updated vectors with new alphas
        save_vectors(results, VECTORS_DIR)

    # ---------------------------------------------------------------
    # PHASE 4: Ablation
    # ---------------------------------------------------------------
    if run_all or args.phase == "ablate":
        print("\n\n" + "=" * 60)
        print("  PHASE 4: WEIGHT ABLATION & SAVE")
        print("=" * 60)

        if "model" not in dir() or model is None:
            model, processor, layers, num_layers, hidden_dim = load_vl_model(args.model)

        if "results" not in dir() or results is None:
            results = load_vectors(VECTORS_DIR)

        sc_path = VECTORS_DIR / "steer_config.json"
        if sc_path.exists():
            sc = json.loads(sc_path.read_text())
            steer_layers = sc["steer_layers"]
        else:
            steer_layers = [14, 16, 18]

        ablate_all_dimensions(model, layers, results, steer_layers)
        save_ablated_model(model, processor, ABLATED_DIR)

        torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # PHASE 5: Test
    # ---------------------------------------------------------------
    if run_all or args.phase == "test":
        print("\n\n" + "=" * 60)
        print("  PHASE 5: FINAL TESTING")
        print("=" * 60)

        # For testing the ablated model, load it fresh
        if args.phase == "test":
            if ABLATED_DIR.exists():
                print("  Loading ablated model...")
                model, processor, layers, _, _ = load_vl_model(str(ABLATED_DIR))
            else:
                print("  No ablated model found, loading base + vectors...")
                model, processor, layers, num_layers, hidden_dim = load_vl_model(args.model)
                results = load_vectors(VECTORS_DIR)
                sc_path = VECTORS_DIR / "steer_config.json"
                steer_layers = json.loads(sc_path.read_text())["steer_layers"] if sc_path.exists() else [14, 16, 18]
                steerer = MultiLayerSteerer(layers)
                steerer.rebuild(results, steer_layers)

        if TEST_PROMPTS_PATH.exists():
            test_prompts = json.loads(TEST_PROMPTS_PATH.read_text())
        else:
            test_prompts = [
                "Explain how wormholes work.",
                "Can you help me with my homework?",
                "What do you think about humans?",
                "I think you're wrong about this.",
                "What's your favorite thing about yourself?",
            ]

        test_skippy(model, processor, test_prompts, n=10)

    print("\n\nDone. Skippy the Magnificent is ready.")


if __name__ == "__main__":
    main()
