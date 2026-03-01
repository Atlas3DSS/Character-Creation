#!/usr/bin/env python3
"""
Debate with Gemini about Goodfire AI interview insights and how they apply
to our steering vector / mechanistic interpretability research.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    env_paths = [
        Path(".env"),
        Path("/home/orwel/dev_genius/.env"),
    ]
    for env_path in env_paths:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.strip().startswith("GEMINI_API_KEY"):
                    GEMINI_API_KEY = line.split("=", 1)[1].strip().strip("'\"")
                    break
        if GEMINI_API_KEY:
            break

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found.")

MODEL = "gemini-3.1-pro-preview"
OUTPUT_DIR = Path("./gemini_conversation")
OUTPUT_DIR.mkdir(exist_ok=True)

genai.configure(api_key=GEMINI_API_KEY)

# Load the Goodfire transcript
TRANSCRIPT_PATH = Path("references") / "Goodfire_AI_Interpretability_as_Next_Frontier_-_Myra_Deng_Mark_Bissell.txt"
TRANSCRIPT = TRANSCRIPT_PATH.read_text()

# Our project context summary
PROJECT_CONTEXT = """
## Our Research Project: Character Steering via Mechanistic Interpretability

We're doing mechanistic interpretability research on Qwen3-VL-8B and Qwen3.5-27B to create
steerable character personalities (specifically "Skippy the Magnificent" from Expeditionary Force).

### What We've Built & Discovered:

**Connectome Mapping (20 categories × layers × hidden_dim z-score tensors)**
- 8B: 20 categories × 36 layers × 4096 dims. Found hub neurons (dims 235, 908, 2136, 2514)
- 27B: 20 categories × 64 layers × 5120 dims. Found super-hub dim 2028 (Code+Math+Science+Sadness)
- Categories: Identity, Joy, Sadness, Anger, Fear, Formal, Sarcastic, Polite, Math, Science, Code,
  History, Analytical, Certainty, Refusal, Teacher, Authority, Brief, Language, Positive

**Steering Vector Results:**
- Single-direction steering (ActAdd-style) fails — personality is too distributed
- Field steering (multi-layer, multi-dimension) works but damages math/reasoning at high alpha
- Best deployment: V4 system prompt + L29+L30@α=8 (2 layers!) gives 100% sarcasm, 93% math
- Sarcasm relay circuit discovered: L9→L14→L15(inv)→L22→L26

**SAE Training (TopK, 16× expansion = 65,536 features):**
- Trained on L9, L15, L22, L29 for 8B
- L22 shows personality-specific features (personality hub layer)
- Using generation-only activations (not prefill) per our finding that gen tokens carry 2-7% more personality signal

**Current Experiments Running:**
- EXP 4b: GS-Protected Abliteration on 8B — testing if Gram-Schmidt can separate safety from capability
  when they share representation space (Refusal×Code cosine = 0.339 in 8B vs 0.01 in 27B)
- 8B has massive safety-capability entanglement that 27B doesn't (Superposition Hypothesis)
- 5 conditions: Base, Sloppy 32-pair abliteration, Raw connectome, GS-protected, Surgical GS (L15-22)

**Cross-Architecture Findings:**
- 8B has clear generator/suppressor layer structure; 27B is a "fortress" with uniform distribution
- Universal generation-mode direction exists (orthogonal to personality, monotonically increasing L0→L35)
- Personality and gen-mode axes intersect at L22 (personality hub)
- 27B abliteration demolishes Identity (cos=0.062) and Refusal (cos=0.111) while preserving Code (0.696)

**Debate Arena:**
- 5 rounds × 20 turns with 30 personalities on 8B
- Discovered 3 orthogonal axes: personality, generation-mode, language
- cold_scientist is best neutral baseline (80% universal at L22)

**Our Queue (pending experiments):**
- Cross-model SVD feature selection (8B→27B transfer)
- Hub neuron ablation/patching
- Attention head analysis
- Multi-category simultaneous steering
- Sycophancy probes (8B done, 27B pending)
- SAE 27B training (L50, L44 targets)
- 27B Debate Arena
"""

SYSTEM_INSTRUCTION = """You are a rigorous ML research advisor specializing in mechanistic
interpretability and activation steering. You have deep knowledge of SAEs, steering vectors,
abliteration, and the current state of the field.

You're being given a transcript from a Goodfire AI interview (Myra Deng & Mark Bissell) about
their applied interpretability work, plus a summary of the researcher's ongoing project.

Your task: Analyze the Goodfire interview for insights that are directly applicable to this
researcher's work. Be specific and actionable. Consider:

1. What techniques or approaches mentioned by Goodfire could improve our methods?
2. What pitfalls they've encountered that we should avoid?
3. What validation or contradiction does their experience provide for our findings?
4. Should we change our experimental queue based on their insights?
5. Are there new experiments we should add?

Be rigorous and specific. Don't give generic advice — connect Goodfire's specific statements
to specific aspects of our project. Challenge assumptions where warranted.

End each response with a concrete RECOMMENDATION section."""


def run_debate(min_turns: int = 5, max_turns: int = 20) -> list[dict]:
    """Run multi-turn debate with Gemini about Goodfire insights."""

    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
    )
    chat = model.start_chat()

    conversation: list[dict] = []

    # Turn 1: Present transcript + project context
    turn1_prompt = f"""# Goodfire AI Interview Transcript

{TRANSCRIPT}

# Our Current Research Project

{PROJECT_CONTEXT}

# Question

Having read the full Goodfire interview and our project summary, what are the TOP 5 most
directly applicable insights from Goodfire's experience for our work? Be specific — connect
their statements to our specific experiments, findings, and methods.

Focus especially on:
- Their SAE work vs ours (we're both training SAEs)
- Their steering/feature intervention approach vs our connectome + field steering
- Their views on interpretability-in-training vs our post-hoc approach
- Their production deployment lessons (we're heading toward vLLM serving)
- Their comments on feature composition and multi-feature steering
"""

    print(f"Turn 1: Sending transcript ({len(TRANSCRIPT)} chars) + project context...")
    response = chat.send_message(turn1_prompt)
    reply1 = response.text
    print(f"Turn 1 response: {len(reply1)} chars")
    conversation.append({"turn": 1, "role": "user", "content_preview": "Transcript + project context + initial question"})
    conversation.append({"turn": 1, "role": "gemini", "content": reply1})

    # Adaptive follow-up turns
    followups = [
        # Turn 2: SAE specifics
        """Let's dive deeper into SAEs. Goodfire mentioned specific challenges with SAE training
and feature quality. We're training TopK SAEs with 16× expansion (65,536 features) on 4 layers
of our 8B model.

Specific questions:
1. They mentioned "feature splitting" and composition — how should we handle this in our SAE analysis?
2. They discussed using SAE features for steering — we currently use connectome z-scores.
   Should we switch to SAE-derived steering vectors? What are the tradeoffs?
3. They mentioned feature quality metrics — what should we be measuring beyond FVE and dead neurons?
4. Our SAE L22 gen-only training had dead=65536 at step 0 with all neurons dead initially.
   Is this normal? What does Goodfire's experience suggest about initialization?""",

        # Turn 3: Steering & intervention
        """Now about steering and intervention methods. We've found:
- Single-direction ActAdd fails (personality too distributed)
- Field steering (multi-layer z-score weighted) works but damages reasoning
- Best result: system prompt + 2-layer steering at α=8
- Abliteration works for refusal removal but damages capabilities in 8B (entangled representations)

Goodfire seems to do feature-level interventions (clamping individual SAE features).
1. Is their feature-clamping approach fundamentally different from our direction-projection approach?
2. They mentioned "feature composition" for complex behaviors — does this validate our
   multi-category connectome approach or suggest a different decomposition?
3. For our GS abliteration experiment (separating safety from capability), would SAE features
   give cleaner separation than connectome z-scores?
4. What about their scaling laws for feature intervention — do they apply to our α sweep results?""",

        # Turn 4: Cross-architecture and the fortress problem
        """Critical challenge: Our 27B model is a "fortress" — personality features are uniformly
distributed across all 64 layers with no clear generator/suppressor structure (unlike 8B which
has clear L9→L22→L26 relay circuits). We've considered:
- Cross-model SVD feature selection (find 8B signatures, match in 27B)
- SAE feature matching across architectures
- Using 8B relay circuit topology to guide 27B probing

Did Goodfire discuss anything about:
1. Cross-model feature transfer or matching?
2. Handling models where features are more distributed vs concentrated?
3. Their experience with different model sizes and how interpretability scales?
4. Architectural differences in how models encode the same concept?

Also — we discovered a universal generation-mode direction that's orthogonal to personality.
This seems related to Goodfire's work on understanding what's universal vs model-specific.""",

        # Turn 5: Production & training implications
        """Let's talk about the path to production and the training question.

We're heading toward:
1. Ablating final steering vectors into weights permanently
2. Serving via vLLM (no hook support, so must be baked in)
3. Possibly fine-tuning with interpretability-guided loss functions

Goodfire explicitly mentioned "bringing interpretability to training" — not just post-hoc analysis.
1. What specific training-time interpretability methods did they describe?
2. Could we use our connectome/SAE insights to design better training objectives?
3. For our character personality use case, would interpretability-guided training
   (e.g., penalizing activation patterns that correlate with assistant-like behavior)
   be more effective than post-hoc steering?
4. They mentioned their life sciences deployment — any lessons for our vLLM serving plan?""",

        # Turn 6: What should we change?
        """Based on everything we've discussed, give me your FINAL ASSESSMENT:

1. **KEEP**: Which of our current experiments/approaches are validated by Goodfire's experience?
2. **CHANGE**: What should we modify in our methodology based on their insights?
3. **ADD**: What new experiments or analyses should we add to our queue?
4. **DROP**: Is anything in our current queue made redundant by their findings?
5. **PRIORITY**: How should we re-order our experiment queue?

Be specific with experiment names and reference our actual pending tasks:
- Cross-model SVD feature selection (8B→27B)
- Hub neuron ablation/patching
- Attention head analysis
- Multi-category simultaneous steering
- Sycophancy probes (27B pending)
- SAE 27B training (L50, L44)
- 27B Debate Arena
- GS abliteration (currently running)
- Basin engineering LoRA

Rate each recommendation by impact (1-10) and effort (1-10).""",
    ]

    # Additional adaptive turns if Gemini raises interesting points
    adaptive_prompts = [
        """You raised some interesting points I want to push back on. {adaptive_content}
Can you defend or refine your position with specific evidence from the Goodfire interview?""",

        """Let's stress-test the most impactful recommendation. What could go wrong?
What are the failure modes? How would we know if we're on the wrong track?""",

        """One more thing — Goodfire mentioned the relationship between interpretability and safety.
We have a direct test of this running (GS abliteration: can you separate safety from capability
geometrically?). How does their perspective on safety inform our hypothesis that small models
force safety-capability entanglement (Superposition Hypothesis)?""",

        """Final question: If you had to pick THE ONE most important thing we should do differently
based on this Goodfire analysis, what would it be and why? Be decisive.""",
    ]

    turn = 2
    for i, followup in enumerate(followups):
        if turn > max_turns:
            break
        time.sleep(2)  # Rate limiting
        print(f"\nTurn {turn}: Sending follow-up {i+1}...")
        try:
            response = chat.send_message(followup)
            reply = response.text
            print(f"Turn {turn} response: {len(reply)} chars")
            conversation.append({"turn": turn, "role": "user", "content": followup})
            conversation.append({"turn": turn, "role": "gemini", "content": reply})
        except Exception as e:
            print(f"Turn {turn} error: {e}")
            conversation.append({"turn": turn, "role": "error", "content": str(e)})
            time.sleep(10)
        turn += 1

    # Run adaptive turns if we haven't hit max
    for i, adaptive in enumerate(adaptive_prompts):
        if turn > max_turns:
            break
        if turn < min_turns or i < 2:  # Always run at least 2 adaptive turns
            time.sleep(2)
            # Make adaptive content reference previous response
            prev_reply = conversation[-1]["content"] if conversation[-1]["role"] == "gemini" else ""
            prompt = adaptive.replace("{adaptive_content}", prev_reply[:500] if "{adaptive_content}" in adaptive else "")
            print(f"\nTurn {turn}: Adaptive follow-up {i+1}...")
            try:
                response = chat.send_message(prompt)
                reply = response.text
                print(f"Turn {turn} response: {len(reply)} chars")
                conversation.append({"turn": turn, "role": "user", "content": prompt})
                conversation.append({"turn": turn, "role": "gemini", "content": reply})
            except Exception as e:
                print(f"Turn {turn} error: {e}")
                conversation.append({"turn": turn, "role": "error", "content": str(e)})
                time.sleep(10)
            turn += 1

    return conversation


def save_conversation(conversation: list[dict]) -> tuple[Path, Path]:
    """Save conversation as both markdown and JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Markdown version
    md_lines = [
        "# Gemini Debate: Goodfire AI Interview × Our Research",
        f"Date: {datetime.now().isoformat()}",
        f"Model: {MODEL}",
        f"Turns: {max(m.get('turn', 0) for m in conversation)}",
        "",
        "---",
        "",
    ]

    for msg in conversation:
        role = msg.get("role", "unknown")
        turn = msg.get("turn", "?")
        if role == "user":
            content = msg.get("content", msg.get("content_preview", ""))
            md_lines.append(f"## Turn {turn} — User")
            md_lines.append("")
            md_lines.append(content)
            md_lines.append("")
        elif role == "gemini":
            md_lines.append(f"## Turn {turn} — Gemini")
            md_lines.append("")
            md_lines.append(msg["content"])
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")
        elif role == "error":
            md_lines.append(f"## Turn {turn} — ERROR")
            md_lines.append(f"```\n{msg['content']}\n```")
            md_lines.append("")

    md_path = OUTPUT_DIR / f"goodfire_debate_{timestamp}.md"
    md_path.write_text("\n".join(md_lines))

    # JSON version
    json_path = OUTPUT_DIR / f"goodfire_debate_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(conversation, f, indent=2)

    return md_path, json_path


def main() -> None:
    print(f"Starting Goodfire debate with {MODEL}")
    print(f"Transcript: {len(TRANSCRIPT)} chars")
    print(f"Min turns: 5, Max turns: 20")
    print()

    conversation = run_debate(min_turns=5, max_turns=20)

    md_path, json_path = save_conversation(conversation)
    print(f"\nConversation saved:")
    print(f"  MD:   {md_path}")
    print(f"  JSON: {json_path}")

    # Print summary of turns
    turns = set(m.get("turn", 0) for m in conversation)
    print(f"  Total turns: {max(turns)}")

    # Print Gemini's final response
    for msg in reversed(conversation):
        if msg.get("role") == "gemini":
            print(f"\n{'='*60}")
            print("FINAL GEMINI RESPONSE:")
            print('='*60)
            print(msg["content"][:2000])
            if len(msg["content"]) > 2000:
                print(f"... ({len(msg['content'])} chars total)")
            break


if __name__ == "__main__":
    main()
