#!/usr/bin/env python3
"""
Multi-turn research conversation with Gemini about our steering research.
Feeds research notes, gets feedback on methodology, prompt diversity, and next steps.
Outputs conversation log for review.

Usage:
    python gemini_research_conversation.py
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
    # Load from .env file
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith("GEMINI_API_KEY"):
                GEMINI_API_KEY = line.split("=", 1)[1].strip().strip("'\"")
                break

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Set it via environment variable or in .env file."
    )

MODEL = "gemini-3.1-pro-preview"
MIN_TURNS = 5
MAX_TURNS = 20
OUTPUT_DIR = Path("./gemini_conversation")

genai.configure(api_key=GEMINI_API_KEY)

# ── Load research notes ─────────────────────────────────────────────
def load_research_notes() -> str:
    """Load the full RESEARCH_NOTES.md."""
    notes_path = Path("./RESEARCH_NOTES.md")
    if not notes_path.exists():
        raise FileNotFoundError(f"RESEARCH_NOTES.md not found at {notes_path}")
    return notes_path.read_text()


# ── Conversation turns ──────────────────────────────────────────────
INITIAL_PROMPT = """I'm sharing our complete research notes from a personality steering project on large language models (Qwen3-VL-8B, Qwen3.5-27B, and GPT-OSS-20B). We're trying to steer model personality (specifically sarcasm and character voice) while preserving reasoning capabilities (math, knowledge).

Here are our full research notes:

{research_notes}

---

Please read through these carefully. I have several questions:

1. **Methodology critique**: What are the biggest methodological weaknesses you see? Where might we be fooling ourselves?

2. **Prompt diversity for spectral analysis**: We're about to run 10,000 math prompts and 10,000 sarcasm prompts through the 27B model to get full-rank covariance matrices (5120×5120). Our current prompt sets use template-based generation. What would you recommend for maximizing entropy/variance in these prompts to ensure we're sampling the full activation space?

3. **The "fortress" problem**: The 27B model has no clear generator/suppressor structure (all 20 layers tested are neutral). The 8B has clean relay circuits. Why might a larger model distribute personality so uniformly, and what does that imply for steering approaches?

4. **GMR spectral analysis**: Our Phase 1 found zero intrusion between math and sarcasm eigenspaces (max alignment 0.096). But this was estimated from only 200 samples for a 5120-dim space (rank-200). We're scaling to 10K+ samples. What should we expect? Will the zero-intrusion finding hold, or is it an artifact of rank deficiency?

5. **Cross-architecture transfer**: We want to use the 8B's well-mapped relay circuit as a "feature selector" to find matching behavioral components in the 27B via SVD decomposition. What are the biggest risks with this approach?

Take your time with this — it's a lot of material. I'm most interested in novel methodological suggestions we haven't considered."""


FOLLOWUP_PROMPTS = [
    # Turn 2: Dig deeper on prompt diversity
    """Thank you for that analysis. Let me focus on the prompt diversity question since it's most immediately actionable.

Our current 10K math prompts cover: arithmetic, division, powers, roots, percentages, factorials, combinatorics, conversions, sequences, modular arithmetic, GCD/LCM, word problems, algebra, geometry, and number properties.

Our 10K sarcasm prompts cover: naive help requests, opinion questions, self-referential, ELI5 explanations, challenges, absurd questions, demands, comparisons, existential questions, cross-topic compounds, tech support, relationship advice, what-if hypotheticals, provocations, instruction-following style prompts, debate starters, workplace humor, pop culture, trivia, and meta/recursive prompts.

Specific questions:
1. What prompt categories are we MISSING that would activate different regions of the model's activation space?
2. For the math prompts specifically — are template-generated prompts sufficient for spectral analysis, or do we need natural-language math questions with more diverse phrasing?
3. Should we include adversarial/edge-case prompts (e.g., math questions designed to trick, or sarcasm prompts that are ambiguous)?
4. What's the optimal balance between prompt diversity (many categories, few per category) vs. depth (few categories, many variations)?""",

    # Turn 3: Discuss the spectral analysis methodology
    """Great suggestions on the prompts. Now let's dig into the spectral analysis methodology.

Our current approach:
1. Run each prompt through the 27B model (forward pass only, no generation)
2. Capture the last-token hidden state at all 64 layers (each is a 5120-dim vector)
3. For each layer, compute the covariance matrix of the 10K activation vectors (5120×5120)
4. Eigendecompose each covariance matrix
5. Compare top eigenspaces between math and sarcasm tasks via cosine alignment

Key concern: With 10K samples for a 5120-dim space, we get full-rank covariance matrices. But:
- Is eigendecomposition the right analysis? Should we also look at singular values, condition numbers, effective dimensionality?
- Our GMR Phase 1 (200 samples) found zero intrusion. With 10K samples, should we expect the tail eigenvalues to reveal hidden alignment that the top-20 eigenvalues missed?
- Are there better spectral methods for detecting directional overlap between two high-dimensional distributions?
- Should we use a different covariance estimator (shrinkage, Ledoit-Wolf) even at 10K samples?

Also, our previous analysis used geometric mean of math and sarcasm median eigenvalues as the reference for per-layer alpha scaling. Is this principled, or should we use a different eigenvalue statistic?""",

    # Turn 4: Cross-architecture insights
    """Let's discuss the cross-architecture transfer problem more deeply.

The key data points:
- 8B (36 layers, 4096 hidden): Clean sarcasm relay circuit L9→L14→L15(inv)→L22→L26. Identity neuron dim 994 (z=-13.96). Clear generator/suppressor structure.
- 27B (64 layers, 5120 hidden): NO relay circuit detected. Identity neuron z=1.06 (13x weaker). All layers neutral. 86 hub positions vs 4 in 8B. Dim 2028 = super-hub across 5+ categories.

The proposed SVD feature selection approach:
1. Decompose 8B sarcasm direction into top-10 SVD components
2. Measure behavioral signature of each component (what token categories it affects)
3. Search 27B's SVD space for components with matching behavioral signatures
4. Reconstruct a 27B sarcasm direction from matching components

Questions:
1. Is behavioral signature matching (measuring token-level effects) the right similarity metric? Or should we use something more internal (activation pattern matching, gradient-based matching)?
2. The 8B has 36 layers and the 27B has 64 layers. The 8B relay sits at L9-L26 (25-72% depth). Where in the 27B should we look? The 58% depth rule suggests L37, but empirically the late band (L45+) works best.
3. Could we use centered kernel alignment (CKA) or representation similarity analysis (RSA) to find the depth correspondence between architectures?
4. What's the risk that the 8B's relay circuit is an artifact of the smaller model's limited capacity, and the 27B genuinely uses a fundamentally different personality mechanism?""",

    # Turn 5: Practical next steps and novel ideas
    """Based on everything we've discussed, what are your top 3-5 novel experimental suggestions that we haven't already planned?

Specifically interested in:
1. Methods that exploit the 27B's unique properties (hybrid attention, 86 hub positions, super-hub dim 2028)
2. Ways to use the debate arena data (two identical models with different personality prompts generating conversations) for steering vector extraction
3. Any information-theoretic or topological approaches we should consider
4. Whether there's a way to use the Gemini or other large models as evaluation oracles for personality quality (beyond simple marker counting)

Also: Given our finding that "V4 prompt alone achieves 100% sarcasm at 90% math accuracy" — is the entire steering vector approach potentially misguided? Should we focus more on prompt engineering + single-layer protection rather than multi-layer activation addition?""",
]

# Additional follow-up based on Gemini's responses
ADAPTIVE_FOLLOWUPS = [
    """That's a fascinating point about {topic}. Can you elaborate on how that would work specifically for our setup? We have:
- RTX PRO 6000 (96GB) for the 27B model
- RTX 3090+4090 (24GB each) for the 8B
- Complete connectome data (20 categories × 36 layers × 4096 dims for 8B, 20 × 64 × 5120 for 27B)
- All single-layer scan results, pair validation data, and the GMR spectral analysis

What would the concrete implementation look like, and how long would it take to run?""",

    """You mentioned {topic}. This connects to something we observed but haven't fully explained: the phase transitions in our alpha sweeps. At α=5 the model enters a new basin (assistant behavior killed), at α=8 sarcasm peaks, at α=10+ coherence collapses.

Is there a principled way to predict these phase transition points from the spectral properties of the activation space? If we could predict the incoherence cliff from eigenvalue structure alone, we could set per-layer alphas optimally without expensive sweep experiments.""",

    """One thing I want to push back on: you suggested {topic}. But our experience with the 8B suggests that {counterpoint}.

For example, the LOO analysis showed that removing "dampener" layers (which individually SUPPRESS sarcasm) from the steering band HURTS overall sarcasm. The dampeners are necessary for the distributed signal propagation chain. The narrow donut experiment conclusively showed this — 16% sarcasm with only amplifiers vs 60% with full donut including dampeners.

How does this affect your recommendation? Does the dampener-as-relay finding change the theoretical picture?""",

    """Let's get concrete about the prompt generation improvements. You suggested adding {categories}. Can you generate 20-30 example prompts for each new category you're recommending? I want to see the specific phrasing and diversity level you have in mind.

For context, our prompts need to:
1. Elicit DISTINCT activation patterns (not just surface-level diversity)
2. Cover the full input manifold for the spectral analysis
3. Be natural enough that the model processes them normally (not adversarial jailbreaks)
4. Math prompts must have verifiable answers""",

    """Final question: Given the complete picture of our research, what do you think is the single most important experiment we should run NEXT? Not the most interesting theoretically, but the one with the highest expected information gain per GPU-hour.

Consider:
- 27B sweep still running (~2h remaining)
- Magnitude-calibrated steering queued
- 10K prompts ready for full-rank spectral analysis
- Debate arena v2 at round 2/5
- Phase 2 capture_and_steer on 27B (168 conditions) ~halfway done

What would you prioritize?""",
]


def run_conversation() -> list[dict]:
    """Run multi-turn conversation with Gemini."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    research_notes = load_research_notes()
    print(f"Loaded research notes: {len(research_notes)} chars")

    model = genai.GenerativeModel(MODEL)
    chat = model.start_chat(history=[])

    conversation_log: list[dict] = []
    turn = 0

    # Turn 1: Initial prompt with full research notes
    print(f"\n{'='*60}")
    print(f"TURN {turn + 1}: Sending initial prompt with research notes...")
    print(f"{'='*60}")

    initial = INITIAL_PROMPT.format(research_notes=research_notes)
    response = chat.send_message(initial)
    reply_text = response.text

    conversation_log.append({
        "turn": turn + 1,
        "role": "user",
        "content": initial[:500] + f"... [+{len(initial)-500} chars of research notes]",
        "timestamp": datetime.now().isoformat(),
    })
    conversation_log.append({
        "turn": turn + 1,
        "role": "gemini",
        "content": reply_text,
        "timestamp": datetime.now().isoformat(),
    })

    print(f"\nGemini response ({len(reply_text)} chars):")
    print(reply_text[:2000])
    if len(reply_text) > 2000:
        print(f"\n... [{len(reply_text) - 2000} more chars]")
    turn += 1

    # Turns 2-5: Planned follow-ups
    for i, followup in enumerate(FOLLOWUP_PROMPTS):
        if turn >= MAX_TURNS:
            break

        print(f"\n{'='*60}")
        print(f"TURN {turn + 1}: Sending planned follow-up {i + 1}...")
        print(f"{'='*60}")

        time.sleep(2)  # Rate limiting
        response = chat.send_message(followup)
        reply_text = response.text

        conversation_log.append({
            "turn": turn + 1,
            "role": "user",
            "content": followup,
            "timestamp": datetime.now().isoformat(),
        })
        conversation_log.append({
            "turn": turn + 1,
            "role": "gemini",
            "content": reply_text,
            "timestamp": datetime.now().isoformat(),
        })

        print(f"\nGemini response ({len(reply_text)} chars):")
        print(reply_text[:2000])
        if len(reply_text) > 2000:
            print(f"\n... [{len(reply_text) - 2000} more chars]")
        turn += 1

    # Adaptive follow-ups based on Gemini's responses (turns 6+)
    for i, template in enumerate(ADAPTIVE_FOLLOWUPS):
        if turn >= MAX_TURNS:
            break
        if turn < MIN_TURNS or (turn >= MIN_TURNS and i < 2):
            # Always do at least 2 adaptive follow-ups after planned ones
            # to guarantee MIN_TURNS

            # Extract a key topic from the last Gemini response for the template
            last_reply = conversation_log[-1]["content"]
            # Simple extraction: find the first bold text or quoted phrase
            topic = "your suggestion"
            counterpoint = "our empirical findings suggest otherwise"
            categories = "new prompt categories"

            # Try to extract meaningful topics from last reply
            lines = last_reply.split("\n")
            for line in lines:
                if "**" in line:
                    start = line.find("**") + 2
                    end = line.find("**", start)
                    if end > start:
                        topic = line[start:end]
                        break

            followup = template.format(
                topic=topic,
                counterpoint=counterpoint,
                categories=categories,
            )

            print(f"\n{'='*60}")
            print(f"TURN {turn + 1}: Sending adaptive follow-up {i + 1}...")
            print(f"{'='*60}")

            time.sleep(2)
            try:
                response = chat.send_message(followup)
                reply_text = response.text
            except Exception as e:
                print(f"Error on turn {turn + 1}: {e}")
                break

            conversation_log.append({
                "turn": turn + 1,
                "role": "user",
                "content": followup,
                "timestamp": datetime.now().isoformat(),
            })
            conversation_log.append({
                "turn": turn + 1,
                "role": "gemini",
                "content": reply_text,
                "timestamp": datetime.now().isoformat(),
            })

            print(f"\nGemini response ({len(reply_text)} chars):")
            print(reply_text[:2000])
            if len(reply_text) > 2000:
                print(f"\n... [{len(reply_text) - 2000} more chars]")
            turn += 1

    # Save conversation log
    log_path = OUTPUT_DIR / f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(conversation_log, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Conversation complete: {turn} turns")
    print(f"Log saved to {log_path}")

    # Also save a readable markdown version
    md_path = OUTPUT_DIR / f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(md_path, "w") as f:
        f.write(f"# Gemini Research Conversation — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Turns: {turn}\n\n---\n\n")
        for entry in conversation_log:
            role = "**US**" if entry["role"] == "user" else "**GEMINI**"
            f.write(f"## Turn {entry['turn']} — {role}\n\n")
            f.write(entry["content"])
            f.write("\n\n---\n\n")
    print(f"Markdown saved to {md_path}")

    return conversation_log


if __name__ == "__main__":
    log = run_conversation()

    # Print summary
    print("\n" + "=" * 60)
    print("CONVERSATION SUMMARY")
    print("=" * 60)
    gemini_turns = [e for e in log if e["role"] == "gemini"]
    total_chars = sum(len(e["content"]) for e in gemini_turns)
    print(f"Total Gemini output: {total_chars:,} chars across {len(gemini_turns)} responses")
    print(f"Average response length: {total_chars // max(len(gemini_turns), 1):,} chars")
