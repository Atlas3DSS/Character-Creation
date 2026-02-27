#!/usr/bin/env python3
"""
Extract Skippy dialogue from a single ExForce epub with full context.

Extracts:
1. Direct Skippy dialogue (quoted speech attributed to Skippy)
2. Skippy monologues and rants
3. Interaction context (what prompt triggered the response)
4. Character interaction tags (Joe, crew, other)
5. Tone classification (insult, joke, technobabble, emotional, confrontational, self-aggrandizing)

Usage:
    python extract_skippy_dialogue.py <epub_path> <output_json>
"""

import json
import re
import sys
from pathlib import Path

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup


# ─── Skippy Attribution Patterns ──────────────────────────────────────────

SKIPPY_NAMES = [
    "skippy", "the beer can", "the magnificent", "the ai",
    "the alien ai", "the ancient ai", "the elder ai",
]

# Patterns that indicate Skippy is speaking
SKIPPY_SPEECH_PATTERNS = [
    # Direct attribution
    r'[Ss]kippy\s+(?:said|replied|answered|asked|shouted|yelled|grumbled|muttered|snapped|declared|announced|explained|interrupted|suggested|warned|sighed|laughed|snorted|growled|scoffed|retorted|countered|objected|protested|insisted|demanded|exclaimed|whispered|hissed|rasped|sneered)',
    r'(?:[Tt]he\s+)?beer\s+can\s+(?:said|replied|answered|asked|shouted|yelled|grumbled|muttered|snapped|declared|announced|explained|interrupted|suggested|warned|sighed|laughed|snorted|growled|scoffed|retorted)',
    # Reverse attribution (said Skippy)
    r'(?:said|replied|answered|asked|shouted|yelled|snapped|declared|announced|explained)\s+[Ss]kippy',
    # Skippy as subject doing dialogue-like things
    r'[Ss]kippy\s+(?:was\s+)?(?:not\s+)?(?:amused|annoyed|frustrated|excited|bored|happy|angry|sulking|pouting|silent|quiet)',
    r'"[^"]*"\s*[Ss]kippy\s+(?:said|replied|answered)',
]

# Joe attribution (to identify Joe's dialogue in conversation)
JOE_NAMES = [
    "joe", "bishop", "colonel bishop", "colonel joe",
    "i said", "i asked", "i replied", "i answered",
]

JOE_SPEECH_PATTERNS = [
    r'[Jj]oe\s+(?:said|replied|answered|asked|shouted)',
    r'[Bb]ishop\s+(?:said|replied|answered|asked)',
    r'(?:said|replied|answered|asked)\s+[Jj]oe',
    r'[Ii]\s+(?:said|replied|answered|asked)',
]


# ─── Tone Classification ─────────────────────────────────────────────────

TONE_KEYWORDS = {
    "insult": [
        "monkey", "monkeys", "primate", "primates", "dumdum", "dum-dum",
        "stupid", "idiot", "moron", "ignorant", "pathetic", "primitive",
        "troglodyte", "knucklehead", "meatbag", "meatbags", "ape", "apes",
        "dimwit", "halfwit", "simpleton", "buffoon", "dense", "dumb",
        "inferior", "lowly", "backward", "embarrassing",
    ],
    "self_aggrandizing": [
        "magnificent", "brilliance", "genius", "superior", "supreme",
        "greatest", "amazing", "incredible", "extraordinary", "spectacular",
        "fantastic", "marvelous", "wonderful", "perfect", "flawless",
        "almighty", "all-powerful", "omniscient", "godlike",
        "i am the", "i'm the", "obviously i", "of course i",
    ],
    "technobabble": [
        "wormhole", "subspace", "quantum", "spacetime", "dimension",
        "zero point", "dark energy", "dark matter", "antimatter",
        "propulsion", "reactor", "shield", "stealth", "sensor",
        "elder", "elders", "rindhalu", "maxolhx", "thuranin",
        "kristang", "ruhar", "jeraptha", "bosphuraq",
        "extinction-level", "senior species", "patron",
    ],
    "emotional": [
        "sorry", "miss", "lonely", "alone", "sad", "afraid",
        "scared", "worried", "care about", "love", "friend",
        "appreciate", "grateful", "thank", "feel bad",
    ],
    "confrontational": [
        "shut up", "listen", "wrong", "no way", "absolutely not",
        "forget it", "you can't", "impossible", "ridiculous",
        "are you kidding", "seriously", "unbelievable",
    ],
    "joke": [
        "haha", "ha ha", "lol", "just kidding", "kidding",
        "joking", "funny", "hilarious", "prank", "trick",
        "gotcha", "psych", "fooled",
    ],
    "sarcastic": [
        "oh great", "wonderful", "fantastic", "how delightful",
        "how adorable", "how cute", "congratulations", "bravo",
        "slow clap", "shocking", "surprising", "who would have thought",
        "oh please", "sure", "right", "uh huh", "yeah right",
    ],
}


def classify_tone(text: str) -> list[str]:
    """Classify the tone of a dialogue snippet."""
    lower = text.lower()
    tones = []
    for tone, keywords in TONE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score >= 2 or (tone == "insult" and score >= 1):
            tones.append(tone)
    if not tones:
        tones = ["neutral"]
    return tones


# ─── EPUB Processing ─────────────────────────────────────────────────────

def epub_to_chapters(epub_path: str) -> list[dict]:
    """Extract chapters from epub as clean text."""
    book = epub.read_epub(epub_path)
    chapters = []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n")
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.strip()
            if len(text) > 200:  # Skip very short items (cover, toc, etc.)
                chapters.append({
                    "id": item.get_name(),
                    "text": text,
                })

    return chapters


def extract_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def is_skippy_speaking(paragraph: str, prev_paragraph: str = "", next_paragraph: str = "") -> bool:
    """Determine if a paragraph contains Skippy's dialogue."""
    # Check for Skippy attribution in this paragraph
    for pattern in SKIPPY_SPEECH_PATTERNS:
        if re.search(pattern, paragraph):
            return True

    # Check if paragraph is a quote and previous paragraph mentions Skippy
    if paragraph.startswith('"') and any(name in prev_paragraph.lower() for name in SKIPPY_NAMES):
        return True

    # Check for Skippy's distinctive speech patterns in quotes
    quotes = re.findall(r'"([^"]+)"', paragraph)
    for quote in quotes:
        lower = quote.lower()
        # Skippy-specific phrases
        if any(phrase in lower for phrase in [
            "you monkey", "you monkeys", "dumdum", "dum-dum",
            "oh great, another", "the magnificent", "i am skippy",
            "beer can", "you primates", "meatbag", "troglodyte",
            "knucklehead", "i'm the most", "filthy monkey",
        ]):
            return True

    return False


def is_joe_speaking(paragraph: str, prev_paragraph: str = "") -> bool:
    """Determine if a paragraph contains Joe's dialogue."""
    for pattern in JOE_SPEECH_PATTERNS:
        if re.search(pattern, paragraph):
            return True
    if paragraph.startswith('"') and any(name in prev_paragraph.lower() for name in JOE_NAMES):
        return True
    return False


def extract_dialogue_pairs(paragraphs: list[str]) -> list[dict]:
    """Extract Skippy dialogue with surrounding context."""
    entries = []
    window = 3  # paragraphs of context before/after

    for i, para in enumerate(paragraphs):
        prev = paragraphs[i - 1] if i > 0 else ""
        nxt = paragraphs[i + 1] if i < len(paragraphs) - 1 else ""

        if is_skippy_speaking(para, prev, nxt):
            # Extract quotes
            quotes = re.findall(r'"([^"]+)"', para)
            skippy_text = " ".join(quotes) if quotes else para

            # Get context window
            context_before = paragraphs[max(0, i - window):i]
            context_after = paragraphs[i + 1:min(len(paragraphs), i + 1 + window)]

            # Find the triggering Joe dialogue (most recent before)
            trigger = ""
            for j in range(i - 1, max(0, i - 5) - 1, -1):
                if is_joe_speaking(paragraphs[j], paragraphs[j - 1] if j > 0 else ""):
                    joe_quotes = re.findall(r'"([^"]+)"', paragraphs[j])
                    trigger = " ".join(joe_quotes) if joe_quotes else paragraphs[j]
                    break

            # Classify tone
            tones = classify_tone(skippy_text)

            # Determine interaction partner
            partner = "unknown"
            context_text = " ".join(context_before[-2:]).lower()
            if any(name in context_text for name in ["joe", "bishop", "colonel"]):
                partner = "joe"
            elif any(name in context_text for name in ["crew", "adams", "chang", "smythe", "desai"]):
                partner = "crew"
            elif any(name in context_text for name in ["nagatha", "bilby"]):
                partner = "other_ai"

            entry = {
                "skippy_text": skippy_text,
                "full_paragraph": para,
                "trigger": trigger,
                "tones": tones,
                "partner": partner,
                "context_before": "\n".join(context_before[-2:]),
                "context_after": "\n".join(context_after[:1]),
                "paragraph_index": i,
            }
            entries.append(entry)

    return entries


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <epub_path> <output_json>")
        sys.exit(1)

    epub_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Processing: {Path(epub_path).name}")

    # Extract chapters
    chapters = epub_to_chapters(epub_path)
    print(f"  Found {len(chapters)} chapters")

    # Extract dialogue from all chapters
    all_entries = []
    total_paragraphs = 0

    for ch_idx, chapter in enumerate(chapters):
        paragraphs = extract_paragraphs(chapter["text"])
        total_paragraphs += len(paragraphs)
        entries = extract_dialogue_pairs(paragraphs)

        for entry in entries:
            entry["chapter_index"] = ch_idx
            entry["chapter_id"] = chapter["id"]

        all_entries.extend(entries)

    # Deduplicate (same text might appear in overlapping epub items)
    seen = set()
    unique_entries = []
    for entry in all_entries:
        key = entry["skippy_text"][:100]
        if key not in seen:
            seen.add(key)
            unique_entries.append(entry)

    # Compute stats
    tone_counts = {}
    partner_counts = {}
    for entry in unique_entries:
        for tone in entry["tones"]:
            tone_counts[tone] = tone_counts.get(tone, 0) + 1
        partner_counts[entry["partner"]] = partner_counts.get(entry["partner"], 0) + 1

    result = {
        "book": Path(epub_path).name,
        "stats": {
            "total_chapters": len(chapters),
            "total_paragraphs": total_paragraphs,
            "skippy_entries": len(unique_entries),
            "tone_distribution": dict(sorted(tone_counts.items(), key=lambda x: -x[1])),
            "partner_distribution": dict(sorted(partner_counts.items(), key=lambda x: -x[1])),
        },
        "entries": unique_entries,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Extracted {len(unique_entries)} Skippy dialogue entries")
    print(f"  Tone distribution: {tone_counts}")
    print(f"  Partner distribution: {partner_counts}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
