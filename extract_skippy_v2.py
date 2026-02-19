#!/usr/bin/env python3
"""
Skippy Dialogue Extraction V2 — Conversation Flow Tracking.

V1 only caught explicitly attributed speech ("Skippy said").
V2 adds:
1. Conversation flow tracking — alternating Joe/Skippy detection
2. Skippy verbal tic detection for unattributed quotes
3. Paragraph-level "Skippy proximity" scoring
4. Multi-paragraph monologue detection
5. First-person Joe narration context

Usage:
    python extract_skippy_v2.py <epub_path> <output_json>
    python extract_skippy_v2.py --all  # Process all ExForce books
"""

import json
import os
import re
import sys
from pathlib import Path

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup


# ─── Skippy Detection Heuristics ──────────────────────────────────────────

# Direct attribution patterns
SKIPPY_ATTR = re.compile(
    r'[Ss]kippy\s+(?:said|replied|answered|asked|shouted|yelled|grumbled|muttered|snapped|'
    r'declared|announced|explained|interrupted|suggested|warned|sighed|laughed|snorted|'
    r'growled|scoffed|retorted|countered|objected|protested|insisted|demanded|exclaimed|'
    r'whispered|hissed|rasped|sneered|continued|added|admitted|agreed|argued|began|'
    r'boasted|bragged|called|chimed|chirped|commented|complained|concluded|confessed|'
    r'confirmed|corrected|cried|deadpanned|drawled|echoed|elaborated|emphasized|ended|'
    r'erupted|estimated|exulted|finished|giggled|groaned|grunted|guessed|huffed|hummed|'
    r'informed|interjected|joked|listed|lied|marveled|mentioned|moaned|mocked|mumbled|'
    r'murmured|narrated|noted|observed|offered|opined|ordered|paused|persisted|piped|'
    r'pointed|pondered|popped|predicted|pressed|proclaimed|promised|proposed|quipped|'
    r'raged|rambled|ranted|reasoned|reassured|recalled|recommended|relented|reminded|'
    r'repeated|reported|responded|resumed|revealed|roared|sang|scolded|screamed|shrieked|'
    r'snickered|sobbed|spoke|stammered|started|stated|stipulated|stormed|stressed|'
    r'stuttered|summarized|surmised|taunted|teased|theorized|thought|told|trailed|'
    r'urged|ventured|volunteered|vowed|wailed|went|wondered|worried|wrote)',
    re.IGNORECASE,
)

SKIPPY_REVERSE_ATTR = re.compile(
    r'(?:said|replied|answered|asked|shouted|yelled|snapped|declared|announced|explained|'
    r'retorted|countered|added|interrupted|continued|finished|began|warned|demanded|'
    r'quipped|joked|sneered|growled|sighed|laughed|mocked|taunted)\s+[Ss]kippy',
    re.IGNORECASE,
)

BEER_CAN_ATTR = re.compile(
    r'(?:[Tt]he\s+)?beer\s+can\s+(?:said|replied|answered|asked|shouted|grumbled|snapped|'
    r'declared|announced|explained|interrupted|continued|added)',
    re.IGNORECASE,
)

# Skippy reference (proximity marker, not necessarily speaking)
SKIPPY_MENTION = re.compile(
    r'\b(?:Skippy|beer can|the AI|the magnificent|the alien AI|the ancient AI|the elder AI)\b',
    re.IGNORECASE,
)

# Skippy verbal tics — phrases only Skippy would say
SKIPPY_VERBAL_TICS = [
    r'you monkey', r'you monkeys', r'filthy monkey', r'filthy monkeys',
    r'dumdum', r'dum[\s-]dum', r'you primat', r'stupid primate',
    r'meatbag', r'meatbags', r'troglodyte', r'knucklehead',
    r'smooth[\s-]brain', r'birdbrain', r'oh,?\s*Joe',
    r'the Magnificent', r'I am Skippy', r"I'm Skippy",
    r'bow before', r'my magnificen', r'my brillian',
    r'you hairless ape', r'hairless monkey', r'your species',
    r'your tiny .* brain', r'your primitive', r'you people',
    r"listen,?\s*Joe", r'pay attention', r'obviously',
    r'as I was saying', r'for the (?:millionth|thousandth|hundredth) time',
    r"I've told you", r'how many times',
    r'elder tech', r'oh,?\s*please', r'sigh',
    r"(?:we're|you're|they're)\s+(?:all\s+)?doomed",
    r'(?:pea[\s-]?sized|walnut[\s-]?sized)\s+brain',
]
SKIPPY_TIC_PATTERN = re.compile('|'.join(SKIPPY_VERBAL_TICS), re.IGNORECASE)

# Joe attribution (first-person narrator)
JOE_ATTR = re.compile(
    r'\bI\s+(?:said|replied|answered|asked|shouted|yelled|snapped|muttered|'
    r'declared|explained|interrupted|suggested|warned|sighed|laughed|'
    r'growled|retorted|countered|protested|insisted|demanded|exclaimed|'
    r'whispered|admitted|agreed|argued|began|commented|concluded|continued|'
    r'groaned|grunted|huffed|informed|joked|moaned|mumbled|murmured|noted|'
    r'offered|pointed|pressed|promised|quipped|reasoned|reminded|repeated|'
    r'responded|resumed|shook|spoke|stammered|started|stated|urged)\b'
)

JOE_NAME_ATTR = re.compile(
    r'(?:[Jj]oe|[Bb]ishop|[Cc]olonel (?:Bishop|Joe))\s+(?:said|replied|answered|asked|shouted)',
    re.IGNORECASE,
)


# ─── Tone Classification ─────────────────────────────────────────────────

TONE_MARKERS = {
    "insult": [
        "monkey", "monkeys", "primate", "primates", "dumdum", "dum-dum",
        "stupid", "idiot", "moron", "ignorant", "pathetic", "primitive",
        "troglodyte", "knucklehead", "meatbag", "ape", "apes",
        "dimwit", "halfwit", "simpleton", "buffoon", "dense", "dumb",
        "inferior", "lowly", "backward", "embarrassing",
    ],
    "self_aggrandizing": [
        "magnificent", "brilliance", "genius", "superior", "supreme",
        "greatest", "amazing", "incredible", "extraordinary",
        "i am the", "i'm the", "obviously i", "of course i",
        "almighty", "godlike", "omniscient", "flawless",
    ],
    "technobabble": [
        "wormhole", "subspace", "quantum", "spacetime", "dimension",
        "zero point", "dark energy", "antimatter", "propulsion",
        "reactor", "shield", "stealth", "sensor", "elder",
        "rindhalu", "maxolhx", "thuranin", "kristang", "ruhar",
        "jeraptha", "bosphuraq", "senior species",
    ],
    "emotional": [
        "sorry", "miss you", "lonely", "alone", "scared",
        "worried", "care about", "friend", "appreciate",
        "feel bad", "my fault",
    ],
    "confrontational": [
        "shut up", "wrong", "no way", "absolutely not",
        "forget it", "impossible", "ridiculous", "seriously",
        "unbelievable", "are you kidding",
    ],
    "sarcastic": [
        "oh great", "wonderful", "fantastic", "how delightful",
        "how adorable", "how cute", "congratulations", "bravo",
        "slow clap", "shocking", "surprising", "oh please",
        "yeah right", "sure thing", "oh really",
    ],
}


def classify_tones(text: str) -> list[str]:
    lower = text.lower()
    tones = []
    for tone, markers in TONE_MARKERS.items():
        hits = sum(1 for m in markers if m in lower)
        if hits >= 1:
            tones.append(tone)
    return tones if tones else ["neutral"]


# ─── EPUB Processing ─────────────────────────────────────────────────────

def epub_to_chapters(epub_path: str) -> list[dict]:
    book = epub.read_epub(epub_path)
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n")
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.strip()
            if len(text) > 200:
                chapters.append({"id": item.get_name(), "text": text})
    return chapters


def extract_quotes(text: str) -> list[str]:
    """Extract all quoted text from a paragraph."""
    # Handle curly quotes too
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    return re.findall(r'"([^"]+)"', text)


# ─── Conversation Flow Tracker ───────────────────────────────────────────

class ConversationTracker:
    """Track who's speaking in a dialogue exchange."""

    def __init__(self):
        self.last_speaker = None  # "skippy", "joe", "other", None
        self.skippy_proximity = 0  # paragraphs since Skippy was mentioned
        self.in_conversation = False
        self.conversation_depth = 0

    def update(self, paragraph: str) -> str | None:
        """
        Analyze a paragraph and return the speaker if it contains dialogue.
        Returns: "skippy", "joe", "other", or None (no dialogue).
        """
        has_quotes = bool(extract_quotes(paragraph))
        has_skippy_mention = bool(SKIPPY_MENTION.search(paragraph))

        # Update proximity
        if has_skippy_mention:
            self.skippy_proximity = 0
        else:
            self.skippy_proximity += 1

        # Check explicit attribution
        if SKIPPY_ATTR.search(paragraph) or SKIPPY_REVERSE_ATTR.search(paragraph) or BEER_CAN_ATTR.search(paragraph):
            self.last_speaker = "skippy"
            self.in_conversation = True
            self.conversation_depth = 0
            return "skippy"

        if JOE_ATTR.search(paragraph) or JOE_NAME_ATTR.search(paragraph):
            self.last_speaker = "joe"
            self.in_conversation = True
            self.conversation_depth = 0
            return "joe"

        if not has_quotes:
            # Non-dialogue paragraph — check if it's narrative about Skippy
            if has_skippy_mention:
                self.in_conversation = False  # Reset conversation on narrative
            self.conversation_depth += 1
            if self.conversation_depth > 3:
                self.in_conversation = False
            return None

        # Has quotes but no explicit attribution
        quotes_text = " ".join(extract_quotes(paragraph))

        # Check for Skippy verbal tics
        if SKIPPY_TIC_PATTERN.search(quotes_text):
            self.last_speaker = "skippy"
            self.in_conversation = True
            self.conversation_depth = 0
            return "skippy"

        # Conversation flow: alternate speakers
        if self.in_conversation and self.conversation_depth <= 2:
            if self.last_speaker == "joe":
                # Joe just spoke, next unattributed quote near Skippy = Skippy
                if self.skippy_proximity <= 5:
                    self.last_speaker = "skippy"
                    self.conversation_depth = 0
                    return "skippy"
            elif self.last_speaker == "skippy":
                # Skippy just spoke, next could be Joe
                self.last_speaker = "joe"
                self.conversation_depth = 0
                return "joe"

        # Proximity-based: if Skippy was mentioned recently and there's a quote
        if self.skippy_proximity <= 2 and has_quotes:
            # Check if the quote sounds more like Skippy than Joe
            if self._sounds_like_skippy(quotes_text):
                self.last_speaker = "skippy"
                self.in_conversation = True
                return "skippy"

        self.conversation_depth += 1
        return None

    def _sounds_like_skippy(self, text: str) -> bool:
        """Heuristic: does this text sound more like Skippy than Joe?"""
        lower = text.lower()
        skippy_score = 0
        joe_score = 0

        # Skippy indicators
        skippy_words = [
            "obviously", "clearly", "magnificent", "brilliant", "genius",
            "pathetic", "primitive", "monkey", "primate", "species",
            "dude", "duuude", "oh man", "oh boy", "technically",
            "impossible", "theoretical", "calculations", "probability",
            "analyzing", "scanning", "detected", "sensors",
        ]
        for w in skippy_words:
            if w in lower:
                skippy_score += 1

        # Joe indicators (military, casual, confused)
        joe_words = [
            "okay", "roger", "copy", "affirmative", "understood",
            "shit", "damn", "crap", "what the hell", "son of a",
            "we need", "let's go", "move out", "sir",
        ]
        for w in joe_words:
            if w in lower:
                joe_score += 1

        # Length heuristic: Skippy tends to be more verbose
        if len(text) > 200:
            skippy_score += 1
        # Exclamation marks: Skippy uses them more
        if text.count("!") >= 2:
            skippy_score += 1

        return skippy_score > joe_score


# ─── Main Extraction ─────────────────────────────────────────────────────

def extract_from_book(epub_path: str) -> dict:
    """Extract all Skippy dialogue with V2 conversation flow tracking."""
    chapters = epub_to_chapters(epub_path)
    all_entries = []
    total_paragraphs = 0

    for ch_idx, chapter in enumerate(chapters):
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', chapter["text"]) if p.strip()]
        total_paragraphs += len(paragraphs)
        tracker = ConversationTracker()

        for i, para in enumerate(paragraphs):
            speaker = tracker.update(para)

            if speaker == "skippy":
                quotes = extract_quotes(para)
                skippy_text = " ".join(quotes) if quotes else para

                # Skip very short entries (< 10 chars)
                if len(skippy_text.strip()) < 10:
                    continue

                # Get context
                context_before = paragraphs[max(0, i - 2):i]
                context_after = paragraphs[i + 1:min(len(paragraphs), i + 2)]

                # Find triggering Joe dialogue
                trigger = ""
                for j in range(i - 1, max(0, i - 5) - 1, -1):
                    j_quotes = extract_quotes(paragraphs[j])
                    if JOE_ATTR.search(paragraphs[j]) or JOE_NAME_ATTR.search(paragraphs[j]):
                        trigger = " ".join(j_quotes) if j_quotes else paragraphs[j][:200]
                        break

                # Detection method
                method = "explicit"
                if SKIPPY_ATTR.search(para) or SKIPPY_REVERSE_ATTR.search(para) or BEER_CAN_ATTR.search(para):
                    method = "explicit_attr"
                elif SKIPPY_TIC_PATTERN.search(skippy_text):
                    method = "verbal_tic"
                else:
                    method = "conversation_flow"

                # Classify tones
                tones = classify_tones(skippy_text)

                # Determine partner
                partner = "unknown"
                ctx = " ".join(context_before[-2:]).lower()
                if "joe" in ctx or "bishop" in ctx or "colonel" in ctx:
                    partner = "joe"
                elif any(n in ctx for n in ["adams", "chang", "smythe", "desai", "crew"]):
                    partner = "crew"
                elif any(n in ctx for n in ["nagatha", "bilby"]):
                    partner = "other_ai"

                entry = {
                    "skippy_text": skippy_text,
                    "full_paragraph": para[:500],
                    "trigger": trigger[:300],
                    "tones": tones,
                    "partner": partner,
                    "detection_method": method,
                    "chapter_index": ch_idx,
                    "paragraph_index": i,
                    "text_length": len(skippy_text),
                }
                all_entries.append(entry)

    # Deduplicate
    seen = set()
    unique = []
    for e in all_entries:
        key = e["skippy_text"][:100]
        if key not in seen:
            seen.add(key)
            unique.append(e)

    # Stats
    method_counts = {}
    tone_counts = {}
    partner_counts = {}
    for e in unique:
        method_counts[e["detection_method"]] = method_counts.get(e["detection_method"], 0) + 1
        for t in e["tones"]:
            tone_counts[t] = tone_counts.get(t, 0) + 1
        partner_counts[e["partner"]] = partner_counts.get(e["partner"], 0) + 1

    return {
        "book": Path(epub_path).name,
        "stats": {
            "total_chapters": len(chapters),
            "total_paragraphs": total_paragraphs,
            "skippy_entries": len(unique),
            "detection_methods": dict(sorted(method_counts.items(), key=lambda x: -x[1])),
            "tone_distribution": dict(sorted(tone_counts.items(), key=lambda x: -x[1])),
            "partner_distribution": dict(sorted(partner_counts.items(), key=lambda x: -x[1])),
            "avg_length": sum(e["text_length"] for e in unique) / max(len(unique), 1),
        },
        "entries": unique,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────

EXFORCE_BOOKS = [
    ("books/Craig Alanson - Columbus Day.epub", "skippy_essence/01_columbus_day_v2.json"),
    ("books/SpecOps - Craig Alanson (EPUB).epub", "skippy_essence/02_specops_v2.json"),
    ("books/paradise_expeditionary_force_book_3__alanson_craig.epub", "skippy_essence/03_paradise_v2.json"),
    ("books/Black Ops - Craig Alanson (EPUB).epub", "skippy_essence/04_black_ops_v2.json"),
    ("books/Mavericks (Expeditionary Force Book 6) - Craig Alanson/Mavericks (Expeditionary Force Book 6) - Craig Alanson.epub", "skippy_essence/06_mavericks_v2.json"),
    ("books/Craig Alanson - Renegades Expeditionary Force, Book 7/Renegades (Expeditionary Force Book 7) - Craig Alanson.epub", "skippy_essence/07_renegades_v2.json"),
    ("books/Armageddon__-_Craig_Alanson.epub", "skippy_essence/08_armageddon_v2.json"),
    ("books/9 Valkyrie (Expeditionary Force) Craig Alanson.epub", "skippy_essence/09_valkyrie_v2.json"),
    ("books/Critical Mass (Expeditionary Force #10) by Craig Alanson.epub", "skippy_essence/10_critical_mass_v2.json"),
    ("books/Brushfire - Craig Alanson.epub", "skippy_essence/11_brushfire_v2.json"),
]


def main():
    if len(sys.argv) >= 3:
        epub_path = sys.argv[1]
        output_path = sys.argv[2]
        result = extract_from_book(epub_path)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"{result['book']}: {result['stats']['skippy_entries']} entries "
              f"({result['stats']['detection_methods']})")
        return

    if "--all" in sys.argv:
        os.makedirs("skippy_essence", exist_ok=True)
        total = 0
        all_stats = []

        for epub_path, output_path in EXFORCE_BOOKS:
            if not os.path.exists(epub_path):
                print(f"  SKIP (not found): {epub_path}")
                continue

            result = extract_from_book(epub_path)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            n = result["stats"]["skippy_entries"]
            total += n
            methods = result["stats"]["detection_methods"]
            print(f"  {Path(epub_path).name:60s} {n:5d} entries  "
                  f"(attr={methods.get('explicit_attr', 0)}, "
                  f"tic={methods.get('verbal_tic', 0)}, "
                  f"flow={methods.get('conversation_flow', 0)})")
            all_stats.append(result["stats"])

        print(f"\n  TOTAL: {total} entries across {len(all_stats)} books")

        # Save combined stats
        with open("skippy_essence/extraction_stats_v2.json", "w") as f:
            json.dump({"total_entries": total, "per_book": all_stats}, f, indent=2)
        return

    print(f"Usage: {sys.argv[0]} <epub_path> <output_json>")
    print(f"       {sys.argv[0]} --all  # Process all ExForce books")


if __name__ == "__main__":
    main()
