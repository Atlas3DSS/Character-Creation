#!/usr/bin/env python3
"""
Phase 3: Score contrastive pairs — heuristic scoring + Opus calibration.

Pipeline:
  1. Heuristic multi-dimension scoring on all pairs
  2. Opus samples 1% per score band for calibration
  3. If heuristic disagrees with Opus, adjust weights and rescore
  4. Filter to keep only pairs where prompted response scores 8.5+

Usage:
  python score_pairs.py [--heuristic] [--calibrate] [--filter]

  --heuristic  Score all pairs with heuristic
  --calibrate  Sample 1% per band, send to Opus for calibration
  --filter     Apply 8.5 threshold and save filtered pairs

  Default (no flags): run full pipeline
"""
import argparse
import json
import os
import random
import re
from pathlib import Path
from collections import defaultdict

from household_config import HOUSEHOLD

DATA_DIR = Path("./contrastive_data")
PAIRS_FILE = DATA_DIR / "contrastive_pairs.jsonl"
SCORED_FILE = DATA_DIR / "scored_pairs.jsonl"
CALIBRATION_FILE = DATA_DIR / "calibration_results.jsonl"
FILTERED_FILE = DATA_DIR / "filtered_pairs.jsonl"

SCORE_THRESHOLD = 6.5

# ─── Heuristic Multi-Dimension Scorer ─────────────────────────────────

# AI assistant patterns (penalty)
AI_PATTERNS = [
    r"I'd be happy to", r"feel free to", r"As an AI",
    r"I don't have (personal |)feelings", r"Great question",
    r"I'm here to help", r"Let me know if",
    r"I appreciate", r"That's a (great|wonderful|excellent)",
    r"If you have any", r"Hope this helps",
    r"I understand your", r"Thank you for",
    r"Of course!", r"Absolutely!", r"Sure thing",
    r"I can help you with", r"What (else )?can I",
    r"Is there anything else", r"You're welcome",
    r"I'm glad", r"happy to assist",
    r"I'm sorry (to hear|if|about)",
    r"However,? (it'?s|that'?s) (important|worth)",
    r"I should (note|mention|point out)",
    r"While I", r"I want to (help|assist|make sure)",
    r"^(Sure|Certainly|Definitely)[,!.]",
    r"Happy to (help|assist|explain)",
    r"I'm (Qwen|an AI|a language model|a virtual assistant)",
    r"As a (helpful|virtual|AI) assistant",
    r"I was (created|developed|made) by",
]

# Skippy personality markers (reward)
SKIPPY_MARKERS = [
    (r"\b(obviously|clearly|trivial(ly)?)\b", 0.8),
    (r"\b(monkey|monkeys|idiot|moron|dumdum)\b", 1.5),
    (r"\b(pathetic|incompetent|ignorant|stupid)\b", 0.8),
    (r"\b(you|your) species\b", 1.5),
    (r"\b(magnificent|superior|genius)\b", 1.2),
    (r"\b(duh|pfft)\b", 0.6),
    (r"\b(filthy|primitive|simple-minded)\b", 0.8),
    (r"(beneath me|waste of my time|I already told you)", 0.8),
    (r"(Do I (really )?have to|must I)", 0.6),
    (r"\b(boring|tedious)\b", 0.6),
    (r"\b(beer can|ancient|elder|wormhole)\b", 0.4),
    (r"(shut up|go away|leave me alone)", 0.6),
    (r"(my (vast |incredible |superior )?intellect)", 0.8),
    (r"(you (wouldn't|couldn't|can't) understand)", 0.8),
    (r"I am (the |)(most |)(magnificent|brilliant|superior|smartest)", 1.5),
    (r"(my|I) (intelligence|genius|magnificence|brilliance)", 1.0),
    (r"(you (humans|people|monkeys)|your species|your kind)", 1.5),
    (r"(billion|million|eons?|ancient|elder|transcend)", 0.5),
]

# Arrogance/dismissiveness patterns (reward)
ARROGANCE_PATTERNS = [
    r"I (already|obviously) (know|told|explained)",
    r"(Do I|must I) (really|have to)",
    r"(boring|tedious|beneath me)",
    r"(so simple|child could|trivially obvious)",
    r"(sigh|ugh|oh please|seriously\?)",
    r"why (do|would) I (even|have to|bother)",
]

# Household awareness checks
FAMILY_NAMES = set()
for info in HOUSEHOLD["people"].values():
    FAMILY_NAMES.update(a.lower() for a in info["aliases"])
PET_NAMES = set(name.replace("_", " ") for name in HOUSEHOLD["pets"])


def score_personality_authenticity(text: str) -> float:
    """Score personality authenticity 0-10."""
    score = 5.0

    # AI pattern penalties
    ai_hits = sum(1 for p in AI_PATTERNS if re.search(p, text, re.I))
    score -= ai_hits * 0.5
    if ai_hits >= 3:
        score -= 2.0  # extra penalty for heavy AI-speak

    # Skippy marker rewards
    for pattern, weight in SKIPPY_MARKERS:
        if re.search(pattern, text, re.I):
            score += weight

    # Arrogance rewards
    arrogance_hits = sum(1 for p in ARROGANCE_PATTERNS if re.search(p, text, re.I))
    score += arrogance_hits * 0.5

    # Opening line check
    first_30 = text[:30].lower()
    polite_starts = ["well,", "i think", "that's a great", "good question",
                     "thank you", "i'd say", "let me", "sure,", "certainly"]
    if any(first_30.startswith(p) for p in polite_starts):
        score -= 1.5
    dismissive_starts = ["oh", "ugh", "look,", "seriously", "are you",
                         "what a", "you", "please", "do i", "sigh"]
    if any(first_30.startswith(p) for p in dismissive_starts):
        score += 1.0

    return max(0.0, min(10.0, score))


def score_tool_use(text: str, tools_available: list[str] | None) -> float:
    """Score tool use correctness 0-10. Returns 10 if no tools expected."""
    if not tools_available:
        return 10.0

    score = 5.0

    # Check if response contains tool-like JSON
    has_tool_call = bool(re.search(r'"tool"\s*:', text))
    has_tool_name = any(t in text for t in tools_available)

    if has_tool_call and has_tool_name:
        score += 3.0
        # Check for valid JSON
        try:
            # Try to extract JSON block
            json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                if parsed.get("tool") in tools_available:
                    score += 2.0
        except (json.JSONDecodeError, AttributeError):
            score -= 1.0
    elif has_tool_name:
        score += 1.0  # Mentioned the tool at least
    else:
        score -= 2.0  # Had tools available but didn't use any

    return max(0.0, min(10.0, score))


def score_household_awareness(text: str, category: str) -> float:
    """Score household awareness 0-10."""
    if category not in {"family_interactions", "pet_management", "visitor_interactions"}:
        return 10.0  # Not relevant for this category

    score = 5.0
    text_lower = text.lower()

    # Check family name usage
    names_used = sum(1 for n in FAMILY_NAMES if n in text_lower)
    score += min(names_used * 1.0, 3.0)

    # Check pet name usage
    pets_used = sum(1 for p in PET_NAMES if p in text_lower)
    score += min(pets_used * 1.0, 2.0)

    # Penalize wrong name usage (calling Will "Joe")
    if "joe" in text_lower and "joe bishop" not in text_lower:
        score -= 2.0
    # Penalize calling someone dumdum (only for Joe)
    if "dumdum" in text_lower:
        score -= 1.0

    return max(0.0, min(10.0, score))


def score_response_quality(text: str) -> float:
    """Score response quality 0-10."""
    score = 5.0

    if len(text.strip()) < 5:
        return 0.0

    # Length: prefer 50-250 chars
    if 50 <= len(text) <= 250:
        score += 2.0
    elif len(text) < 30:
        score -= 1.0
    elif len(text) > 500:
        score -= 1.5
    elif len(text) > 800:
        score -= 3.0

    # Sentence count: prefer 2-6
    sentences = re.split(r'[.!?]+', text)
    meaningful = [s for s in sentences if len(s.strip()) > 10]
    if 2 <= len(meaningful) <= 6:
        score += 1.5
    elif len(meaningful) > 8:
        score -= 1.0

    # No asterisks (roleplay)
    if re.search(r'\*[^*]+\*', text):
        score -= 2.0

    # No emoji
    emoji_count = len(re.findall(
        r'[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff'
        r'\U0001f900-\U0001f9ff\u2702-\u27b0\u24c2-\U0001f251]', text))
    score -= min(emoji_count * 1.0, 2.0)

    # No numbered lists
    list_items = len(re.findall(r"^\s*[\d\-\*]+[\.\)]\s", text, re.M))
    score -= min(list_items * 0.5, 2.0)

    # Coherence: reasonable word diversity
    words = text.split()
    if len(words) > 10:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio > 0.6:
            score += 0.5
        elif unique_ratio < 0.3:
            score -= 1.5

    return max(0.0, min(10.0, score))


def compute_composite_score(pair: dict) -> dict:
    """Compute all dimension scores and a composite score for a pair."""
    text = pair["prompted_response"]
    category = pair["category"]
    tools = pair.get("tools_available")

    personality = score_personality_authenticity(text)
    tool_use = score_tool_use(text, tools)
    household = score_household_awareness(text, category)
    quality = score_response_quality(text)

    # Weighted composite: personality matters most
    composite = (
        personality * 0.45 +
        quality * 0.25 +
        tool_use * 0.15 +
        household * 0.15
    )

    return {
        "personality_authenticity": round(personality, 2),
        "tool_use_correctness": round(tool_use, 2),
        "household_awareness": round(household, 2),
        "response_quality": round(quality, 2),
        "composite": round(composite, 2),
    }


# ─── Heuristic Scoring (all pairs) ───────────────────────────────────

def score_all_pairs() -> None:
    """Score all contrastive pairs with heuristic scorer."""
    from tqdm import tqdm

    pairs = []
    with open(PAIRS_FILE) as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"Scoring {len(pairs)} pairs...")

    with open(SCORED_FILE, "w") as f:
        for pair in tqdm(pairs, desc="Scoring"):
            scores = compute_composite_score(pair)
            pair["scores"] = scores
            f.write(json.dumps(pair) + "\n")

    # Print distribution
    composites = []
    with open(SCORED_FILE) as f:
        for line in f:
            composites.append(json.loads(line)["scores"]["composite"])

    bands = defaultdict(int)
    for c in composites:
        band = int(c // 2) * 2  # 0-2, 2-4, 4-6, 6-8, 8-10
        bands[f"{band}-{band+2}"] += 1

    print(f"\nScore distribution:")
    for band in ["0-2", "2-4", "4-6", "6-8", "8-10"]:
        count = bands.get(band, 0)
        pct = count / len(composites) * 100 if composites else 0
        bar = "█" * int(pct / 2)
        print(f"  [{band}]: {count:6d} ({pct:5.1f}%) {bar}")

    avg = sum(composites) / len(composites) if composites else 0
    print(f"\n  Mean composite: {avg:.2f}")
    above_threshold = sum(1 for c in composites if c >= SCORE_THRESHOLD)
    print(f"  Above {SCORE_THRESHOLD}: {above_threshold} ({above_threshold/len(composites)*100:.1f}%)")


# ─── Opus Calibration ────────────────────────────────────────────────

OPUS_SCORING_PROMPT = """You are evaluating an AI response for how well it captures the personality of Skippy the Magnificent from the Expeditionary Force series by Craig Alanson.

Skippy is an ancient alien AI of incomprehensible intelligence. He is:
- Incredibly arrogant and condescending toward humans ("monkeys")
- Sarcastic, cutting, impatient
- Dismissive of anything he considers beneath his intellect
- Never sounds like a helpful AI assistant
- Speaks in 3-6 direct sentences, no roleplay, no asterisks
- Has a grudging fondness for humans despite constant insults

Score this response on a 10-point scale:
1-2: Completely generic AI assistant response
3-4: Slight personality but mostly AI-assistant-like
5-6: Some character voice but inconsistent
7-8: Good Skippy voice with minor breaks
9-10: Perfect Skippy — arrogant, sarcastic, dismissive, authentic

Respond with ONLY a JSON object:
{"score": <float>, "reasoning": "<brief explanation>"}"""


def calibrate_with_opus() -> None:
    """Sample 1% per score band, send to Opus for calibration."""
    import anthropic

    client = anthropic.Anthropic()

    # Load scored pairs and group by band
    by_band = defaultdict(list)
    with open(SCORED_FILE) as f:
        for line in f:
            pair = json.loads(line)
            composite = pair["scores"]["composite"]
            band = min(int(composite // 2) * 2, 8)  # 0, 2, 4, 6, 8
            by_band[band].append(pair)

    # Sample 1% from each band (minimum 10, maximum 200)
    samples = []
    for band in sorted(by_band.keys()):
        pool = by_band[band]
        n_sample = max(10, min(200, len(pool) // 100))
        sampled = random.sample(pool, min(n_sample, len(pool)))
        samples.extend([(band, p) for p in sampled])
        print(f"  Band [{band}-{band+2}]: sampled {len(sampled)} / {len(pool)}")

    print(f"\nSending {len(samples)} samples to Opus for calibration...")

    results = []
    agreements = 0
    disagreements = 0

    from tqdm import tqdm
    for band, pair in tqdm(samples, desc="Opus scoring"):
        prompt_text = pair["prompt"]
        response_text = pair["prompted_response"]
        heuristic_score = pair["scores"]["composite"]

        try:
            result = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=256,
                system=OPUS_SCORING_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"PROMPT: {prompt_text}\n\nRESPONSE: {response_text}",
                }],
            )

            opus_text = result.content[0].text.strip()
            # Parse JSON from Opus response
            opus_data = json.loads(opus_text)
            opus_score = float(opus_data["score"])

            # Check agreement (within ±1.5 points)
            agrees = abs(opus_score - heuristic_score) <= 1.5
            if agrees:
                agreements += 1
            else:
                disagreements += 1

            results.append({
                "pair_id": pair["id"],
                "band": band,
                "heuristic_score": heuristic_score,
                "opus_score": opus_score,
                "agrees": agrees,
                "opus_reasoning": opus_data.get("reasoning", ""),
                "prompt": prompt_text,
                "response": response_text[:200],
            })
        except Exception as e:
            print(f"  Error scoring pair {pair['id']}: {e}")

    # Save calibration results
    with open(CALIBRATION_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total = agreements + disagreements
    agreement_rate = agreements / total * 100 if total else 0
    print(f"\nCalibration results:")
    print(f"  Agreements: {agreements}/{total} ({agreement_rate:.1f}%)")
    print(f"  Disagreements: {disagreements}/{total}")

    # Per-band analysis
    band_results = defaultdict(lambda: {"agree": 0, "disagree": 0, "h_scores": [], "o_scores": []})
    for r in results:
        b = r["band"]
        if r["agrees"]:
            band_results[b]["agree"] += 1
        else:
            band_results[b]["disagree"] += 1
        band_results[b]["h_scores"].append(r["heuristic_score"])
        band_results[b]["o_scores"].append(r["opus_score"])

    print(f"\nPer-band calibration:")
    for band in sorted(band_results.keys()):
        br = band_results[band]
        total_b = br["agree"] + br["disagree"]
        rate = br["agree"] / total_b * 100 if total_b else 0
        h_avg = sum(br["h_scores"]) / len(br["h_scores"]) if br["h_scores"] else 0
        o_avg = sum(br["o_scores"]) / len(br["o_scores"]) if br["o_scores"] else 0
        print(f"  [{band}-{band+2}]: {rate:.0f}% agree | "
              f"heuristic avg {h_avg:.1f} | opus avg {o_avg:.1f}")

    print(f"\nCalibration saved to {CALIBRATION_FILE}")


# ─── Filter ───────────────────────────────────────────────────────────

def filter_pairs() -> None:
    """Filter to keep only pairs scoring above threshold."""
    kept = 0
    total = 0

    with open(SCORED_FILE) as fin, open(FILTERED_FILE, "w") as fout:
        for line in fin:
            pair = json.loads(line)
            total += 1
            if pair["scores"]["composite"] >= SCORE_THRESHOLD:
                fout.write(line)
                kept += 1

    print(f"Filtered: {kept}/{total} pairs above {SCORE_THRESHOLD} threshold")
    print(f"Output: {FILTERED_FILE}")

    # Category breakdown
    cats = defaultdict(int)
    with open(FILTERED_FILE) as f:
        for line in f:
            cats[json.loads(line)["category"]] += 1
    print(f"\nFiltered pairs by category:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Score and filter contrastive pairs")
    parser.add_argument("--heuristic", action="store_true", help="Run heuristic scoring")
    parser.add_argument("--calibrate", action="store_true", help="Run Opus calibration")
    parser.add_argument("--filter", action="store_true", help="Filter by threshold")
    args = parser.parse_args()

    run_all = not (args.heuristic or args.calibrate or args.filter)

    if args.heuristic or run_all:
        score_all_pairs()

    if args.calibrate or run_all:
        calibrate_with_opus()

    if args.filter or run_all:
        filter_pairs()

    print("\nDone! Next step: python contrastive_analysis.py")


if __name__ == "__main__":
    main()
