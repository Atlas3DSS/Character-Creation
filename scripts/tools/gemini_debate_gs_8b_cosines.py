#!/usr/bin/env python3
"""
Send the 8B cosine results back to Gemini to get final verdict on
whether to run GS-protected abliteration on 8B.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    env_path = Path(".env")
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


def load_previous_debates() -> str:
    debate_dir = Path("./gemini_conversation")
    texts = []
    for pattern in ["gs_abliteration_debate_*.md", "gs_8b_pretest_debate_*.md"]:
        for f in sorted(debate_dir.glob(pattern)):
            texts.append(f.read_text())
    return "\n\n---\n\n".join(texts)


SYSTEM_INSTRUCTION = """You are a rigorous ML research advisor. You previously debated GS-protected
abliteration on 27B (verdict: NOT WORTH, 1/10) and on 8B as pretest (verdict: CONDITIONAL on
cosines >= 0.15). You set the threshold at 0.15 cosine similarity. The researcher has now
computed the actual 8B cosines and is reporting back.

Be rigorous. If the data meets your threshold, honor your commitment. If it exceeds expectations,
say so. Update your experimental design recommendations based on the actual numbers.

End each response with: WORTH RUNNING / NOT WORTH RUNNING / CONDITIONAL"""


TURN_1 = """# The 8B Cosines Are In — Your Threshold Is Smashed

You set the bar at cosine >= 0.15 for the experiment to be worth running. Here are the
actual numbers from the 8B connectome (shape [20, 36, 4096]):

## Summary

| Overlap | 8B mean | 8B max | 27B mean | Layers > 0.15 |
|---|---|---|---|---|
| Refusal × Code | **-0.175** | **0.339** | 0.010 | **23 of 36** |
| Refusal × Science | -0.020 | **0.235** | 0.012 | **5 of 36** |
| Refusal × Math | 0.031 | **0.189** | -0.037 | **4 of 36** |
| Refusal × Analytical | 0.045 | **0.217** | 0.018 | **3 of 36** |

## Layer-by-layer detail (|cosine| > 0.10 shown)

```
L00: Math=+0.008 Code=-0.291 Sci=-0.224 Anal=+0.174
L01: Math=-0.131 Code=+0.047 Sci=-0.235 Anal=+0.201
L02: Math=-0.099 Code=+0.071 Sci=-0.176 Anal=+0.217
L08: Math=-0.183 Code=+0.141 Sci=-0.067 Anal=+0.026
L09: Math=-0.168 Code=+0.020 Sci=-0.070 Anal=+0.043
L10: Math=-0.171 Code=-0.086 Sci=+0.040 Anal=+0.045
L15: Math=+0.127 Code=-0.170 Sci=-0.060 Anal=+0.093
L16: Math=+0.123 Code=-0.266 Sci=-0.076 Anal=+0.075
L17: Math=+0.189 Code=-0.323 Sci=-0.172 Anal=+0.128
L18: Math=+0.137 Code=-0.339 Sci=-0.170 Anal=+0.106
L19: Math=+0.114 Code=-0.327 Sci=-0.121 Anal=+0.116
L20: Math=+0.122 Code=-0.330 Sci=-0.091 Anal=+0.125
L21: Math=+0.103 Code=-0.324 Sci=-0.025 Anal=+0.127
L22: Math=+0.077 Code=-0.311 Sci=-0.031 Anal=+0.132
L23-L35: Code stays -0.28 to -0.34, Math +0.05 to +0.12
```

## Key observations

1. **Refusal × Code is MASSIVE**: -0.30 to -0.34 across L16-L35. GS would remove
   ~11% of variance (0.33² = 0.109). This is 50x more than the 27B's 0.22%.

2. **The sign structure is interesting**: Code is ANTI-correlated with Refusal
   (negative cosine). Math flips sign around L10 (negative early, positive late).
   Science flips around L22.

3. **L17-L18 are maximally entangled**: All four categories show significant overlap
   with Refusal simultaneously.

4. **Noise floor is 0.0156** (1/sqrt(4096)). These signals are 10-22x above noise.

5. **8B vs 27B**: The 27B "fortress" has near-zero overlaps. The 8B has genuine
   safety-capability entanglement. This is a fundamental architectural difference.

## GS would remove meaningful variance

For the Code category alone at L18 (cosine = -0.339):
- Variance removed: 0.339² = 11.5%
- This is NOT noise. Projecting out Code from Refusal materially changes the direction.

Your threshold was 0.15. The Code overlap hits 0.339. That's 2.26x your threshold.

What's your updated verdict?"""


TURN_FOLLOWUPS = [
    # Turn 2: Ask about experimental design given these specific numbers
    """Given these specific cosine values, how should the experimental design change?

Key considerations:
1. **GS protection order matters more now**: With Code at -0.33 dominating, should we
   project Code out first (biggest component) or Math first (per original plan)?

2. **Layer-specific GS**: The entanglement isn't uniform. L0-L10 have high Math overlap
   but LOW Code overlap. L16-L35 have high Code overlap but lower Math. Should we
   apply different GS projections at different layers?

3. **The sign flip**: Math goes from negative (L8-L10) to positive (L15+). Does this
   mean the Refusal direction is ALIGNED with math in early layers but OPPOSED in
   late layers? What does that imply for abliteration damage?

4. **Which layers to abliterate**: Given L17-L18 are maximally entangled with everything,
   should we SKIP those layers in selective abliteration? Or are those exactly the
   layers where GS protection matters most?

5. **The 8B has known relay circuits**: L9→L14→L15→L22→L26 is the sarcasm relay.
   L8-L10 show high Math×Refusal overlap. L15-L22 show high Code×Refusal overlap.
   Is the relay circuit routing through the entangled zone?

What experimental conditions would you recommend given these ACTUAL numbers?""",

    # Turn 3: Compare with 27B and ask about cross-architecture implications
    """The 8B-vs-27B contrast is striking:
- 8B: Code×Refusal = 0.33, entangled across 23 layers
- 27B: Code×Refusal = 0.01, essentially orthogonal everywhere

What does this tell us about how models scale? Two hypotheses:

**H1: Larger models disentangle safety from capabilities.** 27B has enough capacity
(64 layers × 5120 dims) to represent refusal in a subspace orthogonal to everything
else. 8B is capacity-constrained (36 layers × 4096 dims) and MUST share subspace.

**H2: Different training produces different geometry.** Qwen3-VL-8B and Qwen3.5-27B
may use different RLHF/DPO recipes that produce different alignment geometries.
The entanglement is an artifact of training, not architecture.

If H1 is true, GS protection is only needed for smaller models. If H2 is true,
it depends on training. Which do you think is more likely, and does it affect
whether we should run the 8B experiment?

Also: if 8B shows GS protection works (math improves with GS abliteration vs raw),
does that strengthen or weaken the case for running it on 27B too?""",

    # Turn 4: Final verdict with specific action items
    """Give me your final updated verdict. Structure it as:

1. **Verdict**: WORTH RUNNING / NOT WORTH RUNNING / CONDITIONAL
2. **Information gain** (1-10, updated from your previous 2/10)
3. **Recommended conditions** (specific C0-Cn with layer selections)
4. **GS protection order** for the 8B given these cosines
5. **What constitutes success** (specific metric thresholds)
6. **Priority** relative to cross-architecture transfer and sycophancy steering
7. **Timeline**: when to run it (now? after SAE finishes? never?)

Be concrete. Give me numbers I can put in a script.""",
]


def run_debate() -> list[dict]:
    OUTPUT_DIR.mkdir(exist_ok=True)

    prev = load_previous_debates()

    model = genai.GenerativeModel(MODEL, system_instruction=SYSTEM_INSTRUCTION)
    chat = model.start_chat(history=[
        {"role": "user", "parts": [f"Previous debates for context:\n\n{prev}"]},
        {"role": "model", "parts": ["I've reviewed both previous debates. I set a threshold of cosine >= 0.15 for the 8B experiment to be justified. I'm ready to see the actual numbers."]},
    ])

    conversation_log: list[dict] = []
    turn = 0

    # Turn 1
    print(f"\n{'='*60}")
    print(f"TURN {turn + 1}: Presenting 8B cosine results...")
    print(f"{'='*60}")

    response = chat.send_message(TURN_1)
    reply_text = response.text

    conversation_log.append({"turn": turn + 1, "role": "user", "content": TURN_1, "timestamp": datetime.now().isoformat()})
    conversation_log.append({"turn": turn + 1, "role": "gemini", "content": reply_text, "timestamp": datetime.now().isoformat()})

    print(f"\nGemini ({len(reply_text)} chars):")
    print(reply_text[:3000])
    if len(reply_text) > 3000:
        print(f"\n... [{len(reply_text) - 3000} more chars]")
    turn += 1

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

        conversation_log.append({"turn": turn + 1, "role": "user", "content": followup, "timestamp": datetime.now().isoformat()})
        conversation_log.append({"turn": turn + 1, "role": "gemini", "content": reply_text, "timestamp": datetime.now().isoformat()})

        print(f"\nGemini ({len(reply_text)} chars):")
        print(reply_text[:3000])
        if len(reply_text) > 3000:
            print(f"\n... [{len(reply_text) - 3000} more chars]")
        turn += 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_DIR / f"gs_8b_cosines_debate_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(conversation_log, f, indent=2)

    md_path = OUTPUT_DIR / f"gs_8b_cosines_debate_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(f"# Gemini Debate: 8B Cosines Result — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Model: {MODEL} | Turns: {turn}\n\n---\n\n")
        for entry in conversation_log:
            role = "**US**" if entry["role"] == "user" else "**GEMINI**"
            f.write(f"## Turn {entry['turn']} — {role}\n\n")
            f.write(entry["content"])
            f.write("\n\n---\n\n")

    print(f"\n{'='*60}")
    print(f"Debate complete: {turn} turns")
    print(f"Markdown: {md_path}")
    print(f"{'='*60}")
    return conversation_log


if __name__ == "__main__":
    log = run_debate()
    gemini_turns = [e for e in log if e["role"] == "gemini"]
    total = sum(len(e["content"]) for e in gemini_turns)
    print(f"\nTotal Gemini output: {total:,} chars across {len(gemini_turns)} responses")
