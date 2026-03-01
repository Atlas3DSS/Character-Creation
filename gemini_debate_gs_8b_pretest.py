#!/usr/bin/env python3
"""
Follow-up debate with Gemini: should we pretest GS-protected abliteration
on Qwen3-VL-8B (dev server) before committing the PRO 6000 to the 27B run?

Continues from the previous debate where Gemini rated the 27B experiment 1/10.
"""

import json
import os
import time
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

MODEL = "gemini-3.1-pro-preview"
OUTPUT_DIR = Path("./gemini_conversation")

genai.configure(api_key=GEMINI_API_KEY)

# ── Load previous debate for context ────────────────────────────────
def load_previous_debate() -> str:
    """Load the previous debate markdown for context."""
    debate_dir = Path("./gemini_conversation")
    debates = sorted(debate_dir.glob("gs_abliteration_debate_*.md"))
    if debates:
        return debates[-1].read_text()
    return "(No previous debate found)"

# ── Debate prompts ──────────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are a rigorous ML research advisor specializing in mechanistic
interpretability and representation engineering. You previously debated whether GS-protected
abliteration on Qwen3.5-27B was worth running, and concluded NOT WORTH RUNNING due to
mathematically predetermined null results (pre-GS cosines of 0.01-0.04).

Now we're discussing a modified proposal: running the experiment on Qwen3-VL-8B instead,
as a cheaper pretest. Apply the same rigor. Consider:
1. Whether the 8B model changes the information gain calculation
2. Whether the 8B connectome has different overlap properties than 27B
3. The practical value of a pretest vs just running on 27B directly
4. What we'd actually learn that we can't predict from existing data

End each response with: WORTH RUNNING / NOT WORTH RUNNING / CONDITIONAL"""


TURN_1 = """# Follow-up Debate: 8B Pretest of GS-Protected Abliteration

## Previous Debate Summary
In our last debate, you rated the GS-protected abliteration experiment on Qwen3.5-27B
at 1/10 information gain, 9/10 waste risk. Key reasons:
- Pre-GS cosines (0.01-0.04) mean GS removes ~0.22% — C1 ≈ C2 is mathematically certain
- Huihui-ai comparison is confounded (different intervention mechanism)
- You suggested a redesign testing extraction quality (sloppy 32-pair vs connectome)

## New Proposal: Run on 8B First

Instead of committing the PRO 6000 (96GB) to a 2.5h 27B experiment, run it on
Qwen3-VL-8B (17.5GB INT8) on the dev server (3090 or 4090, 24GB each).

### Why 8B might be different:
1. **8B has clear structure**: Unlike the 27B "fortress", 8B has identifiable
   generator/suppressor layers, relay circuits (L9→L14→L15→L22→L26), and a
   personality hub at L22.
2. **8B connectome is different**: 36 layers, 4096 hidden dims. The overlap
   properties between Refusal and Math/Code/Science may not be near-zero like 27B.
3. **8B has known steering recipes**: We have validated champion configs
   (L29+L30@α=8) with known math costs. We can compare abliteration results
   against these established baselines.
4. **The dev GPUs are currently running SAE training** — but those finish in a
   few hours. After that, the 3090 and 4090 would be idle.

### What we already know about 8B:
- Connectome: 20 categories × 36 layers × 4096 dims at `qwen_connectome/analysis/connectome_zscores.pt`
- Phase 1 baseline: math=100%, sarcasm=55%, assistant=30%
- V4 + L29+L30@α=8: math=93.3%, sarcasm=100%, knowledge=96.7%
- Abliterated 8B arena showed doom loops, token repetition, personality absorption
- Hub neurons: dims 235, 908, 2136, 2514. Identity is orthogonal to everything (<0.09)

### What we DON'T know about 8B:
- The Refusal × Math/Code/Science cosine similarities in the 8B connectome
- Whether 8B abliteration causes the same ~8% math drop as 27B
- Whether the 8B's clearer structure makes GS protection more effective

### Practical advantages:
- Runs in ~30-45 min (36 layers, faster inference)
- Doesn't block the PRO 6000 for SAE 27B
- If it works, motivates the 27B run. If it doesn't, saves 2.5h.
- The dev GPUs would otherwise sit idle after SAE finishes

### My concern:
8B and 27B have fundamentally different architectures (36 vs 64 layers, 4096 vs 5120 hidden,
clear relay circuits vs fortress). Results on 8B may not transfer to 27B at all, making
this a "pretest" that doesn't actually predict the 27B outcome.

What's your position?"""


TURN_FOLLOWUPS = [
    # Turn 2: Push back with practical argument
    """You make good points. But here's the practical reality:

1. **We need to check the 8B cosines first**: We haven't actually looked at Refusal × Math
   overlap in the 8B connectome. If it's significantly higher than the 27B's near-zero
   values (say, 0.15+), then GS protection might actually matter for 8B. That alone
   would change the information gain calculation for 27B too — it would mean the near-zero
   27B cosines are a model-specific property, not a universal one.

2. **The script adaptation is trivial**: Change model loading from 27B to 8B, adjust layer
   count from 64 to 36, change hidden dim from 5120 to 4096. Maybe 20 minutes of work.

3. **It's not really a "pretest"**: You're right that 8B results don't predict 27B.
   But the 8B experiment has standalone value — it extends our abliteration analysis
   to a model where we have much richer mechanistic understanding (relay circuits,
   hub neurons, doom loop data). We could write a comparison section.

4. **The alternative is idle GPUs**: After SAE training finishes (~2-4 hours), both
   dev GPUs sit empty until we find something else. What would YOU recommend running
   on idle 3090+4090 GPUs if not this?

Does the possibility of higher 8B cosines change your assessment? And what would
you run on idle 24GB GPUs instead?""",

    # Turn 3: Ask about the cosine check
    """Before we go further — can we resolve this empirically? The 8B connectome is
already extracted at `qwen_connectome/analysis/connectome_zscores.pt`. We can check
the Refusal × Math/Code/Science cosines in 30 seconds with a quick torch script.

If the 8B cosines are:
- **Near-zero (< 0.05)**: Same as 27B → GS experiment is equally pointless on 8B
- **Moderate (0.05-0.15)**: Interesting — different model, different geometry
- **High (> 0.15)**: GS would actually remove meaningful variance → experiment is
  genuinely worth running on 8B

What cosine threshold would change your verdict from NOT WORTH to WORTH RUNNING?
Give me a specific number.""",

    # Turn 4: Alternative experiments for idle dev GPUs
    """Setting aside the GS experiment entirely — what experiments would you recommend
for idle 3090+4090 GPUs (24GB each, can run Qwen3-VL-8B in INT8)?

Our current backlog of 8B-compatible experiments:
1. Sycophancy steering test (dial-down sycophancy using connectome direction)
2. Debate arena v2 (more personality pairs, longer conversations)
3. Cross-architecture transfer prep (extract generation-mode vectors for 8B→27B anchoring)
4. Extended doom loop analysis (abliterated 8B with more rounds)
5. The GS abliteration pretest we've been debating

Rank these by expected information gain per GPU-hour. Which ones directly advance
the main personality steering goal vs being side-quests?""",

    # Turn 5: Final verdict
    """Give me your final verdict on the 8B GS pretest specifically. Structure it as:

1. **Verdict**: WORTH RUNNING / NOT WORTH RUNNING / CONDITIONAL
2. **Conditions** (if conditional): what must be true
3. **Expected information gain** (1-10)
4. **What to do with idle dev GPUs instead** (top 2 recommendations)
5. **One sentence summary**: should we or shouldn't we?

Be direct.""",
]


def run_debate() -> list[dict]:
    """Run the structured debate with Gemini."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    prev_debate = load_previous_debate()

    model = genai.GenerativeModel(
        MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
    )

    # Seed with previous debate context
    chat = model.start_chat(history=[
        {"role": "user", "parts": [f"For context, here is our previous debate:\n\n{prev_debate}"]},
        {"role": "model", "parts": ["I've reviewed our previous debate on GS-protected abliteration for Qwen3.5-27B. I concluded NOT WORTH RUNNING with 1/10 information gain. I'm ready to discuss the 8B pretest proposal."]},
    ])

    conversation_log: list[dict] = []
    turn = 0

    # Turn 1
    print(f"\n{'='*60}")
    print(f"TURN {turn + 1}: Setting up 8B pretest debate...")
    print(f"{'='*60}")

    response = chat.send_message(TURN_1)
    reply_text = response.text

    conversation_log.append({
        "turn": turn + 1, "role": "user",
        "content": TURN_1,
        "timestamp": datetime.now().isoformat(),
    })
    conversation_log.append({
        "turn": turn + 1, "role": "gemini",
        "content": reply_text,
        "timestamp": datetime.now().isoformat(),
    })

    print(f"\nGemini ({len(reply_text)} chars):")
    print(reply_text[:3000])
    if len(reply_text) > 3000:
        print(f"\n... [{len(reply_text) - 3000} more chars]")
    turn += 1

    # Follow-ups
    for i, followup in enumerate(TURN_FOLLOWUPS):
        time.sleep(3)

        print(f"\n{'='*60}")
        print(f"TURN {turn + 1}: Follow-up {i + 1}...")
        print(f"{'='*60}")

        try:
            response = chat.send_message(followup)
            reply_text = response.text
        except Exception as e:
            print(f"Error on turn {turn + 1}: {e}")
            break

        conversation_log.append({
            "turn": turn + 1, "role": "user",
            "content": followup,
            "timestamp": datetime.now().isoformat(),
        })
        conversation_log.append({
            "turn": turn + 1, "role": "gemini",
            "content": reply_text,
            "timestamp": datetime.now().isoformat(),
        })

        print(f"\nGemini ({len(reply_text)} chars):")
        print(reply_text[:3000])
        if len(reply_text) > 3000:
            print(f"\n... [{len(reply_text) - 3000} more chars]")
        turn += 1

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_DIR / f"gs_8b_pretest_debate_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(conversation_log, f, indent=2)

    md_path = OUTPUT_DIR / f"gs_8b_pretest_debate_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(f"# Gemini Debate: 8B GS Pretest — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Model: {MODEL} | Turns: {turn}\n\n---\n\n")
        for entry in conversation_log:
            role = "**US**" if entry["role"] == "user" else "**GEMINI**"
            f.write(f"## Turn {entry['turn']} — {role}\n\n")
            f.write(entry["content"])
            f.write("\n\n---\n\n")

    print(f"\n{'='*60}")
    print(f"Debate complete: {turn} turns")
    print(f"JSON: {log_path}")
    print(f"Markdown: {md_path}")
    print(f"{'='*60}")

    return conversation_log


if __name__ == "__main__":
    log = run_debate()
    gemini_turns = [e for e in log if e["role"] == "gemini"]
    total = sum(len(e["content"]) for e in gemini_turns)
    print(f"\nTotal Gemini output: {total:,} chars across {len(gemini_turns)} responses")
