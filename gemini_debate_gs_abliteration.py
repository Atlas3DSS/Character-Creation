#!/usr/bin/env python3
"""
Multi-turn debate with Gemini about whether GS-protected abliteration
is a worthwhile research direction given our broader context.

3-10 turns, alternating between presenting evidence FOR and AGAINST,
with Gemini providing analysis and counterarguments.
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
MIN_TURNS = 3
MAX_TURNS = 10
OUTPUT_DIR = Path("./gemini_conversation")

genai.configure(api_key=GEMINI_API_KEY)

# ── Debate prompts ──────────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are a rigorous ML research advisor specializing in mechanistic
interpretability and representation engineering. You are participating in a structured
debate about a proposed experiment. Your role is to:

1. Analyze claims with empirical rigor — cite specific numbers from the data presented
2. Identify confounds, alternative explanations, and methodological issues
3. Be genuinely critical — don't just validate the researcher's hypothesis
4. Distinguish between "interesting finding" and "publishable contribution"
5. Consider opportunity cost — is this the best use of 2.5 hours of GPU time?

When you see weak points in an argument, say so directly. When you see strong points,
acknowledge them but probe further. End each response with a clear position:
WORTH RUNNING / NOT WORTH RUNNING / CONDITIONAL (with conditions)."""

TURN_1_CONTEXT = """# Debate: Is GS-Protected Abliteration Worth Running?

## Background
We're running a personality steering project on Qwen3.5-27B (64 layers, 5120 hidden dim).
We have a complete "connectome" — contrastive z-score vectors for 20 behavioral categories
across all 64 layers (tensor shape [20, 64, 5120]).

## The Experiment
A reviewer suggested: the ~8% math degradation from standard abliteration (projecting out
the refusal direction from hidden states) isn't inherent to removing safety. It's from
sloppy direction extraction contaminating Math/Code/Science subspaces. If we Gram-Schmidt
orthogonalize the refusal direction against reasoning categories BEFORE projection, we
should get abliteration with 0% math cost.

## The Data We Already Have
From our connectome, the cosine similarities between Refusal and reasoning categories are:
- Refusal × Math: **-0.037**
- Refusal × Code: **+0.010**
- Refusal × Science: **+0.012**
- Refusal × Analytical: **+0.018**

These are ALREADY near-zero. GS protection would project out components with magnitudes
on the order of 0.01-0.04 — essentially noise-level overlaps.

## The Proposed Conditions (5 conditions, ~2.5 hours GPU time)
- C0: Base (no abliteration) — baseline
- C1: Raw connectome abliteration (all 64 layers, unit-normalized z-scores)
- C2: GS-protected connectome abliteration (all 64 layers, Refusal orthogonalized against Math/Code/Science/Analytical)
- C3: Raw abliteration (top-10 refusal layers only)
- C4: GS-protected (top-10 refusal layers only)

Plus comparison against huihui-ai's abliterated model (already evaluated: 92% math, 83% knowledge).

## Key Numbers
- Raw z-score norms: ~17-19 for Refusal, ~20-35 for Math/Code/Science
- After unit-normalization, these become direction vectors
- GS removed fraction per layer: 0.02% to 0.59% (mean ~0.22%)
- Post-GS cosines with protected categories: all < 0.05 (most < 0.03)

## My Concern
The near-zero pre-GS cosines suggest the refusal and reasoning directions are ALREADY
nearly orthogonal in the connectome. GS would remove essentially nothing. This predicts
C1 ≈ C2 (no difference between raw and GS-protected), meaning the experiment's primary
hypothesis is falsified before we even run it.

But the reviewer's argument was about NOISY directions from standard 32-pair contrastive
extraction (like huihui-ai's method), not about our connectome-derived directions. Our
connectome uses 100 contrastive prompts per category, which is already cleaner.

What's your position? Is this experiment worth 2.5 hours of GPU time?"""

TURN_FOLLOWUPS = [
    # Turn 2: Counter-argument (FOR running it)
    """Good analysis. Let me push back with the strongest case FOR running it:

1. **The experiment tests the framework, not just GS**: If C1 (our connectome direction,
   no GS) already achieves 100% math (vs huihui-ai's 92%), that proves the connectome
   extraction is superior to standard contrastive methods. That's valuable even if GS
   adds nothing.

2. **C3/C4 are the real novelty**: Selective abliteration (top-10 refusal layers only)
   has never been tested with connectome directions. If C3/C4 match C1/C2 in refusal
   removal but with less perturbation, that's a practical contribution.

3. **The 0.22% GS removal IS the finding**: If GS removes essentially nothing because
   our directions are already clean, that validates the connectome methodology itself.
   It's a diagnostic, not just a treatment.

4. **Opportunity cost is low**: The GPU would otherwise sit idle for 2.5 hours. We have
   the script written, the eval infrastructure ready, and the model cached.

5. **Publishability**: The 6-way comparison table (Base / Raw-All / GS-All / Raw-Selective
   / GS-Selective / huihui-ai) would be a strong figure in any abliteration paper.

Does this change your assessment?""",

    # Turn 3: Probing the null hypothesis
    """You raise valid points. But let me probe the null hypothesis more carefully:

**Scenario A**: C1 math = 100%, C2 math = 100%, huihui-ai math = 92%
→ Conclusion: Connectome directions are cleaner. GS is unnecessary with good directions.
→ Value: Moderate. Shows method superiority but doesn't validate GS specifically.

**Scenario B**: C1 math = 92%, C2 math = 100%
→ Conclusion: GS protection works! Even clean connectome directions benefit.
→ Value: High. Directly validates the reviewer's prediction.

**Scenario C**: C1 math = 100%, C2 math = 100%, AND C3/C4 refusal < 5%
→ Conclusion: Surgical abliteration works. Only need 10 layers.
→ Value: Very high. Practical recipe for safe abliteration.

**Scenario D**: All conditions math > 95%, refusal unchanged from base
→ Conclusion: Our refusal direction doesn't actually capture refusal well.
→ Value: Diagnostic — reveals connectome weakness.

Given our pre-GS cosines of 0.01-0.04, which scenario do you think is most likely?
And does the expected information gain justify the compute?""",

    # Turn 4: Broader context
    """Let me add broader context that might change the calculus:

**Our project's main goal** is personality steering (making a model speak like a specific
character), NOT abliteration. The abliteration experiments are a side branch that arose
because we noticed our connectome has a Refusal category.

**What we've already spent on this branch**:
- Connectome extraction: ~8 hours GPU (but was needed for main project anyway)
- huihui-ai eval: ~45 min
- eval_head_to_head: ~2 hours (already completed)
- Abliterated vs base connectome comparison: ~1 hour
- Script development (via Codex): 15 min
- Total abliteration-specific: ~3-4 hours

**What's queued and potentially more impactful**:
- SAE training on 27B (needs PRO 6000, ~12 hours)
- Sycophancy steering experiments
- Cross-architecture transfer (8B → 27B)
- Full-rank spectral analysis (10K prompts)

**The question isn't just "is it worth 2.5h GPU time" but "should it run NOW or after
higher-priority experiments?"**

The PRO 6000 just freed up from eval_head_to_head. SAE 27B is queued but needs a fresh
venv setup. GS abliteration is ready to run RIGHT NOW with zero setup.

Does this scheduling context affect your recommendation?""",

    # Turn 5: Steel-man the "don't run" position
    """Let me steel-man the strongest case AGAINST running it:

1. **Predictable from existing data**: Pre-GS cosines of 0.01-0.04 mean GS removes 0.02-0.59%
   of the direction. This is mathematically guaranteed to produce C1 ≈ C2. Running the
   experiment to confirm what linear algebra already tells us wastes GPU time.

2. **The "connectome vs huihui-ai" comparison is confounded**: Even if C1 > huihui-ai, we
   can't attribute it to direction quality alone. We're also using different layer selections
   (all 64 vs whatever huihui-ai used), different projection methods, and a different base
   model (our FP8 vs their abliterated checkpoint). Too many variables.

3. **No one will cite a null result**: "We tried GS and it didn't help because our
   directions were already clean" is interesting to us but not publishable. Reviewers want
   positive results.

4. **The real question is upstream**: If we want to test whether direction extraction
   quality matters, we should compare our connectome direction against huihui-ai's exact
   direction (32-pair mean difference) applied to the SAME model with the SAME hooks. Our
   experiment design doesn't include this condition.

How would you respond to each of these objections?""",

    # Turn 6: Request final verdict
    """Based on our full discussion, I need your final assessment. Please structure it as:

1. **Verdict**: WORTH RUNNING / NOT WORTH RUNNING / CONDITIONAL
2. **Expected information gain** (1-10 scale)
3. **Risk of wasted time** (1-10 scale)
4. **Suggested modifications** to the experimental design (if any)
5. **What would make the results publishable** vs just internally useful
6. **Priority ranking** relative to: SAE 27B training, sycophancy steering,
   cross-architecture transfer, full-rank spectral analysis

Be direct and concrete. No hedging.""",

    # Turn 7 (if needed): Follow up on modifications
    """You suggested {modifications}. That's interesting. But given that the script is
already written and validated (Codex-generated, smoke test passed), any modifications
mean additional development time. Is the marginal improvement from your suggested changes
worth the delay?

Also: you mentioned expected information gain of {gain}. Can you break that down by
condition pair? Which specific comparison (C1 vs C2? C3 vs C4? C1 vs huihui-ai?)
carries the most information?""",
]


def run_debate() -> list[dict]:
    """Run the structured debate with Gemini."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    model = genai.GenerativeModel(
        MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
    )
    chat = model.start_chat(history=[])

    conversation_log: list[dict] = []
    turn = 0

    # Turn 1: Full context
    print(f"\n{'='*60}")
    print(f"TURN {turn + 1}: Setting up debate context...")
    print(f"{'='*60}")

    response = chat.send_message(TURN_1_CONTEXT)
    reply_text = response.text

    conversation_log.append({
        "turn": turn + 1, "role": "user",
        "content": TURN_1_CONTEXT,
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

    # Follow-up turns
    for i, followup_template in enumerate(TURN_FOLLOWUPS):
        if turn >= MAX_TURNS:
            break
        if turn >= MIN_TURNS and i >= 5:
            # Stop after turn 6 unless we haven't hit min
            break

        time.sleep(3)  # Rate limiting

        # Simple template filling from last response
        last_reply = conversation_log[-1]["content"]
        modifications = "your suggested modifications"
        gain = "your stated level"

        # Try to extract specific topics
        for line in last_reply.split("\n"):
            if "modif" in line.lower() and "**" in line:
                start = line.find("**") + 2
                end = line.find("**", start)
                if end > start:
                    modifications = line[start:end]
                    break
        for line in last_reply.split("\n"):
            if any(x in line.lower() for x in ["information gain", "expected value", "/10"]):
                gain = line.strip()[:100]
                break

        followup = followup_template.format(
            modifications=modifications,
            gain=gain,
        )

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
    log_path = OUTPUT_DIR / f"gs_abliteration_debate_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(conversation_log, f, indent=2)

    md_path = OUTPUT_DIR / f"gs_abliteration_debate_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(f"# Gemini Debate: GS-Protected Abliteration — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
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
