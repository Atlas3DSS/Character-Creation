#!/usr/bin/env python3
"""
Send Journal_creation -> Arena sweep question to Gemini.
Single-shot query, saves response to markdown file.
"""

import os
from datetime import datetime
from pathlib import Path

import google.generativeai as genai

# ── Config ──────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith("GEMINI_API_KEY"):
                GEMINI_API_KEY = line.split("=", 1)[1].strip().strip("'\"")
                break

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found.")

MODEL = "gemini-2.5-pro"
OUTPUT_DIR = Path(__file__).parent / "gemini_conversation"

genai.configure(api_key=GEMINI_API_KEY)

# ── Prompt ──────────────────────────────────────────────────────────
PROMPT = """## Context: Two Systems That Need to Be Connected

### System 1: Character Steering Arena (debate_arena_8b.py)
Our debate arena runs two-model debates where each model is assigned a personality (e.g., "chinese_only_nationalist", "socratic_philosopher", "flat_earther"). Each round: 20 turns of alternating debate, activation snapshots captured at every turn (per-layer hidden states), cosine similarity tracking. Key findings: L22 is the personality hub (lowest cross-model cosine = 0.505), generation amplifies personality 2-7% vs listening, doom loops emerge in some pairs. 30 hand-crafted personality definitions tested.

### System 2: Journal_creation Population Generator
553 characters with: Big Five personality traits (continuous 0.0-1.0), Myers-Briggs, Attachment styles (Secure/Anxious/Avoidant/Disorganized), Demographics (age/gender/ethnicity/generation/religion/politics), Career/Education, Values/Beliefs/Worldview, Background narratives (100+ words), Journal entries (8 types: daily reflection, emotional processing, goal progress, relationship reflection, memory, worry, gratitude, significant event).

Example: Benjamin Arthur Parker (76/M/White/Boomer): Assembler, O=0.67 C=0.75 E=0.48 A=0.42 N=0.53, fears losing independence, Protestant Conservative.

### Hardware
- Dev server: RTX 3090 + RTX 4090 (24GB each) — runs 8B debates
- Local: RTX PRO 6000 96GB — runs 27B analysis

### Codex (GPT-4.1) Already Suggested:
1. Synthesize personality prompts from traits (template function)
2. Cluster-then-representative sampling (K=32-64, centroid+extreme+random)
3. Big Five → L22 activation regression
4. Journal entries as personality probes
5. Doom loop correlation with Big Five traits

### Questions for You (Gemini):
1. Do you agree with Codex's approach? What would you change or add?
2. The Sarcastic+Polite entanglement we found in the connectome — could Big Five traits map onto this? (E.g., does high Agreeableness + low Neuroticism = the "passive aggression" cluster?)
3. For the journal integration: should we use the journal TEXT as prompts, or synthesize new prompts that match the journal's STYLE/THEMES? Which gives cleaner activation signal?
4. Push-pull pair (dims 755/2455): given that these are anti-correlated toggle switches, should we specifically select character pairs that we predict will maximally activate one vs the other?
5. What's the minimum number of arena runs needed for statistical significance on Big Five regression across 36 layers?
6. Any concerns about the 10-stage Pydantic generator creating correlated trait distributions that bias the regression?"""


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Using model: {MODEL}")
    print(f"Prompt length: {len(PROMPT)} chars")
    print("Sending to Gemini...")

    model = genai.GenerativeModel(MODEL)
    response = model.generate_content(PROMPT)
    reply_text = response.text

    print(f"\nReceived response: {len(reply_text)} chars")
    print("=" * 60)
    print(reply_text)
    print("=" * 60)

    # Save markdown output
    out_path = OUTPUT_DIR / "journal_arena_sweep_20260227.md"
    with open(out_path, "w") as f:
        f.write(f"# Gemini: Journal Creation + Arena Sweep Analysis\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Model**: {MODEL}\n\n")
        f.write(f"---\n\n")
        f.write(f"## Query\n\n{PROMPT}\n\n---\n\n")
        f.write(f"## Gemini Response\n\n{reply_text}\n")

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
