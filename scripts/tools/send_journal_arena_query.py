#!/usr/bin/env python3
"""Send a detailed query to GPT-4.1 (Codex) about bridging Journal_creation
and the debate arena for personality sweeps."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ── API key ──────────────────────────────────────────────────────────────
# The .env stores it as OPEN_AI (not OPENAI_API_KEY)
from dotenv import load_dotenv

ENV_PATH = Path(".env")
load_dotenv(ENV_PATH)

API_KEY = os.environ.get("OPEN_AI")
if not API_KEY:
    # Try reading directly
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("OPEN_AI"):
            API_KEY = line.split("=", 1)[1].strip().strip("'\"")
            break

if not API_KEY:
    print("ERROR: Could not find OPEN_AI key in .env")
    sys.exit(1)

from openai import OpenAI

client = OpenAI(api_key=API_KEY)

# ── Prompts ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a research consultant helping design experiments for personality "
    "steering in large language models. You have expertise in activation steering, "
    "connectome analysis, and synthetic population generation. Be specific and actionable."
)

USER_MESSAGE = r"""## Context: Two Systems That Need to Be Connected

### System 1: Character Steering Arena (debate_arena_8b.py)
Our debate arena runs two-model debates where each model is assigned a personality (e.g., "chinese_only_nationalist", "socratic_philosopher", "flat_earther", "devout_christian", "conspiracy_theorist", "cold_scientist", "eco_activist", "helpful_assistant", "libertarian_purist"). Each round:
- 20 turns of alternating debate
- Activation snapshots captured at every turn (per-layer hidden states)
- Cosine similarity tracking between personality activations
- Temperature variation via debate tactics (agree/challenge/troll/ignore/etc.)
- Results: per-layer activation maps showing where personality is encoded

Key findings from the arena:
- L22 is the personality hub (lowest cross-model cosine = 0.505)
- Generation amplifies personality 2-7% vs listening
- Universal gen-mode direction exists (monotonic L0→L35)
- Doom loops emerge in some personality pairs (repetitive responses)
- 30 hand-crafted personality definitions tested so far

### System 2: Journal_creation Population Generator
A sophisticated 10-stage pipeline generating demographically-accurate fictional characters with:
- **553 characters** (53 manual + 500 auto-generated)
- **Big Five personality traits** (continuous 0.0-1.0 per dimension)
- **Myers-Briggs types** (16 types)
- **Attachment styles** (Secure/Anxious/Avoidant/Disorganized)
- **Demographics**: age, gender, ethnicity, generation (Silent→Gen-Alpha), religion, politics
- **Career/Education**: occupation, income bracket, education level
- **Values/Beliefs**: core values, worldview statement, political ideology
- **Background narratives**: 100+ word life stories
- **Interests, fears, motivations, daily routines**
- **Journal entries**: 8 types (daily reflection, emotional processing, goal progress, etc.)
- **Communication style** profiles

Example character:
```json
{
  "name": "Benjamin Arthur Parker",
  "age": 76, "gender": "Male", "ethnicity": "White",
  "generation": "Baby Boomer",
  "occupation": "Assembler/Fabricator",
  "big_five": {"openness": 0.67, "conscientiousness": 0.75, "extraversion": 0.48, "agreeableness": 0.42, "neuroticism": 0.53},
  "interests": ["camping", "socializing", "browsing the internet", "meditation"],
  "fears": ["Losing independence due to declining health", "Being forgotten by family"],
  "religion": "Protestant", "political_affiliation": "Conservative"
}
```

More examples:
- Rafael Victor Gomez (25/M/Hispanic/Gen-Z): Police Officer, O=0.64 C=0.78 E=0.79 A=0.50 N=0.26
- Alejandro Rafael Alvarez (45/M/Hispanic/Millennial): Taxi Driver, O=0.55 C=0.42 E=0.75 A=0.31 N=0.25

### Hardware
- Dev server: RTX 3090 + RTX 4090 (24GB each) — runs 8B model debates
- Local: RTX PRO 6000 96GB — runs 27B analysis
- The arena currently runs 5 rounds × 20 turns with activation capture

## The Question

How should we bridge these two systems? Specifically:

1. **Character-to-Arena Personality Mapping**: How to convert the rich Journal_creation character profiles (Big Five, MBTI, attachment style, values, fears, background narrative) into arena-compatible personality system prompts? Should we use the raw character JSON as context, or synthesize a personality prompt from the traits?

2. **Scaling Strategy**: With 500+ characters, we can't run all pairs (125K combinations). What's the optimal sampling strategy for arena sweeps? Should we:
   - Sample by Big Five extremes (high/low on each dimension)?
   - Cluster characters and pick representatives?
   - Use the push-pull pairs we found in the connectome (dims 755/2455)?
   - Focus on demographic axes (age × politics × religion)?

3. **What New Signals Could We Extract?**: The hand-crafted arena personalities (30 total) gave us L22 as personality hub, gen-mode direction, doom loops. With 500 rich characters, what NEW mechanistic interpretability insights could we find? Specifically:
   - Can Big Five dimensions be mapped to activation space dimensions?
   - Does attachment style (secure vs avoidant) create distinct activation signatures?
   - Do doom loops correlate with specific Big Five profiles?
   - Can we find the "personality complexity" dimension (simple archetype vs rich character)?

4. **Journal Integration**: Characters have journal entries. Could we use journal text as prompts (instead of debate topics) to measure "in-character" vs "out-of-character" activation differences? The journal system has 8 entry types — could each type probe a different personality dimension?

5. **Practical Implementation**: What does the actual `debate_arena_journal_sweep.py` script look like? We need:
   - Character loading from Journal_creation JSON
   - Personality prompt synthesis from character profile
   - Same activation capture pipeline as existing arena
   - Results aggregation across 500 characters
   - Big Five → activation space regression analysis

Please be very specific about architecture, data flow, and what experiments to prioritize."""

# ── Send to GPT-4.1 ─────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Sending query to gpt-4.1 at {timestamp} ...")
print(f"System prompt: {len(SYSTEM_PROMPT)} chars")
print(f"User message: {len(USER_MESSAGE)} chars")

response = client.chat.completions.create(
    model="gpt-4.1",
    max_tokens=16384,
    temperature=0.7,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_MESSAGE},
    ],
)

reply = response.choices[0].message.content
print(f"\nReceived response: {len(reply)} chars")
print(f"Tokens: prompt={response.usage.prompt_tokens}, "
      f"completion={response.usage.completion_tokens}, "
      f"total={response.usage.total_tokens}")

# ── Save markdown ────────────────────────────────────────────────────────
OUT_DIR = Path("codex_conversation")
OUT_DIR.mkdir(exist_ok=True)

md_path = OUT_DIR / "journal_arena_sweep_20260227.md"
md_content = f"""# Codex: Journal-Arena Personality Sweep Design

**Date**: {timestamp}
**Model**: gpt-4.1
**Tokens**: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}, total={response.usage.total_tokens}

---

{reply}
"""

md_path.write_text(md_content)
print(f"\nSaved to: {md_path}")

# ── Save JSON conversation ──────────────────────────────────────────────
json_path = OUT_DIR / f"journal_arena_sweep_{timestamp}.json"
conversation = [
    {
        "turn": 1,
        "role": "user",
        "content": USER_MESSAGE,
        "system_prompt": SYSTEM_PROMPT,
        "timestamp": datetime.now().isoformat(),
    },
    {
        "turn": 1,
        "role": "codex",
        "content": reply,
        "response_id": response.id,
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "timestamp": datetime.now().isoformat(),
    },
]
json_path.write_text(json.dumps(conversation, indent=2, ensure_ascii=False))
print(f"Saved JSON: {json_path}")

# ── Print full response ─────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FULL GPT-4.1 RESPONSE")
print("=" * 80 + "\n")
print(reply)
print("\n" + "=" * 80)
