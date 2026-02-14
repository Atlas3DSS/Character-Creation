#!/usr/bin/env python3
"""
Extract Skippy dialogue WITH context from ExForce books.

Pulls (prompt, skippy_response) pairs where someone speaks then Skippy responds.
Handles smart quotes and newlines within dialogue.

Output: extracted_text/skippy_pairs.json
"""
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

from ebooklib import epub
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

BOOKS_DIR = Path("./books")
OUT_DIR = Path("./extracted_text")

# Smart and straight quotes
OPEN_Q = '[\u201c"]'
CLOSE_Q = '[\u201d"]'

VERBS = (
    r"(?:said|asked|replied|scoffed|snorted|muttered|shouted|exclaimed|"
    r"declared|announced|whispered|growled|snarled|sighed|laughed|"
    r"chuckled|sneered|explained|continued|added|agreed|protested|"
    r"insisted|suggested|warned|demanded|pleaded|offered|noted|"
    r"observed|remarked|stated|responded|retorted|countered|"
    r"interrupted|called|cried|yelled|snapped|barked|began|"
    r"reported|complained|wondered|admitted|acknowledged|conceded)"
)

SKIPPY_NAMES = {"skippy", "beer can", "magnificent", "the ai", "alien ai"}


def find_exforce_epubs() -> list[Path]:
    all_epubs = sorted(BOOKS_DIR.rglob("*.epub"))
    return [ep for ep in all_epubs if "alanson" in str(ep).lower()]


def parse_epub(epub_path: Path) -> str:
    book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})
    texts = []
    for item in book.get_items():
        if item.get_type() == 9:
            soup = BeautifulSoup(item.get_content(), "lxml")
            text = soup.get_text(separator="\n")
            if text.strip():
                texts.append(text.strip())
    full = "\n\n".join(texts)
    print(f"  {epub_path.name}: {len(full):,} chars")
    return full


def normalize_text(text: str) -> str:
    """Normalize smart quotes and clean up whitespace within dialogue."""
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    # Collapse internal newlines (epub formatting artifact)
    text = re.sub(r"\n+", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)
    return text.strip()


def is_skippy_speaker(speaker: str) -> bool:
    s = speaker.lower().strip()
    return any(name in s for name in SKIPPY_NAMES)


def extract_speeches(text: str) -> list[dict]:
    """Extract all quoted speech with speaker attribution."""
    speeches = []

    # Find all quoted passages (smart quotes)
    # Use negated character class (no backtracking issues)
    quote_pat = re.compile(
        r'\u201c([^\u201d]{8,800})\u201d',
    )

    for m in quote_pat.finditer(text):
        dialogue = normalize_text(m.group(1))
        pos = m.start()

        # Look for speaker attribution after the quote (within 80 chars)
        after = text[m.end():m.end() + 80]
        after_match = re.match(
            r'[,.]?\s{0,5}(?:the\s+)?(\w+(?:\s+\w+)?)\s+' + VERBS,
            after, re.IGNORECASE
        )

        # Look for speaker attribution before the quote (within 80 chars)
        before = text[max(0, pos - 80):pos]
        before_match = re.search(
            r'(?:the\s+)?(\w+(?:\s+\w+)?)\s+' + VERBS + r'[,.]?\s*$',
            before, re.IGNORECASE
        )

        speaker = None
        if after_match:
            speaker = after_match.group(1).strip()
        elif before_match:
            speaker = before_match.group(1).strip()

        # Also check for "Skippy" appearing in nearby context
        context_window = text[max(0, pos - 40):m.end() + 40].lower()

        if speaker:
            speeches.append({
                "text": dialogue,
                "speaker": speaker,
                "is_skippy": is_skippy_speaker(speaker),
                "pos": pos,
            })
        elif "skippy" in context_window:
            # Unattributed but near Skippy mention
            speeches.append({
                "text": dialogue,
                "speaker": "skippy",
                "is_skippy": True,
                "pos": pos,
            })

    return speeches


def build_pairs(speeches: list[dict], text: str) -> tuple[list[dict], list[str]]:
    """Build conversation pairs from sequential speeches."""
    pairs = []
    standalone = []

    for i in range(len(speeches)):
        curr = speeches[i]

        # Collect standalone Skippy lines
        if curr["is_skippy"] and len(curr["text"]) > 30:
            standalone.append(curr["text"])

        # Look for non-Skippy → Skippy transitions
        if i + 1 < len(speeches):
            nxt = speeches[i + 1]
            gap = nxt["pos"] - curr["pos"]

            if not curr["is_skippy"] and nxt["is_skippy"] and gap < 800:
                pairs.append({
                    "prompt_speaker": curr["speaker"],
                    "prompt": curr["text"],
                    "skippy_response": nxt["text"],
                    "type": "dialogue_pair",
                })

                # Check for extended Skippy response (multiple consecutive lines)
                if i + 2 < len(speeches):
                    nxt2 = speeches[i + 2]
                    if nxt2["is_skippy"] and (nxt2["pos"] - nxt["pos"]) < 500:
                        pairs.append({
                            "prompt_speaker": curr["speaker"],
                            "prompt": curr["text"],
                            "skippy_response": nxt["text"] + " " + nxt2["text"],
                            "type": "extended_pair",
                        })

    return pairs, standalone


def main():
    OUT_DIR.mkdir(exist_ok=True)

    epubs = find_exforce_epubs()
    print(f"Found {len(epubs)} ExForce books:\n")

    all_pairs = []
    all_standalone = []

    for ep in epubs:
        text = parse_epub(ep)
        speeches = extract_speeches(text)
        skippy_speeches = sum(1 for s in speeches if s["is_skippy"])
        print(f"    → {len(speeches)} speeches, {skippy_speeches} from Skippy")

        pairs, standalone = build_pairs(speeches, text)
        print(f"    → {len(pairs)} pairs, {len(standalone)} standalone")
        all_pairs.extend(pairs)
        all_standalone.extend(standalone)

    # Deduplicate
    seen = set()
    unique_pairs = []
    for p in all_pairs:
        key = p["skippy_response"][:80]
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    seen_s = set()
    unique_standalone = []
    for s in all_standalone:
        if s[:80] not in seen_s:
            seen_s.add(s[:80])
            unique_standalone.append(s)

    print(f"\n{'='*50}")
    print(f"Total unique pairs: {len(unique_pairs)}")
    print(f"  dialogue_pair: {sum(1 for p in unique_pairs if p['type'] == 'dialogue_pair')}")
    print(f"  extended_pair: {sum(1 for p in unique_pairs if p['type'] == 'extended_pair')}")
    print(f"Total unique standalone: {len(unique_standalone)}")

    # Show examples
    print(f"\n--- Sample pairs ---")
    for p in unique_pairs[:15]:
        print(f"\n  {p['prompt_speaker']}> {p['prompt'][:100]}")
        print(f"  SKIPPY> {p['skippy_response'][:100]}")

    # Save
    output = {
        "pairs": unique_pairs,
        "standalone_skippy": unique_standalone,
        "stats": {
            "total_pairs": len(unique_pairs),
            "total_standalone": len(unique_standalone),
            "books_processed": len(epubs),
        },
    }

    out_path = OUT_DIR / "skippy_pairs.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
