#!/usr/bin/env python3
"""
Debate with Gemini about Goodfire AI blog posts and how their technical
insights apply to our steering vector / mechanistic interpretability research.
Follows up on the interview transcript debate.
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
        Path(__file__).parent / ".env",
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

# Load the blog articles
REFS_DIR = Path(__file__).parent / "references"
BLOG_FILES = [
    "goodfire_feature_steering.txt",
    "goodfire_intentional_design.txt",
    "goodfire_sae_open_source_llama.txt",
    "goodfire_interpretability_infra_frontier_scale.txt",
    "goodfire_ember_scaling_interpretability.txt",
    "goodfire_optimism_interpretability.txt",
    "goodfire_series_a.txt",
]

BLOG_CONTENT = {}
for fname in BLOG_FILES:
    fpath = REFS_DIR / fname
    if fpath.exists():
        BLOG_CONTENT[fname] = fpath.read_text()

# Load previous Goodfire debate for context continuity
PREV_DEBATE = ""
debate_dir = Path("./gemini_conversation")
for f in sorted(debate_dir.glob("goodfire_debate_*.md")):
    PREV_DEBATE = f.read_text()
    break  # just the first one

# Project context (abbreviated since Gemini already has it from the previous debate)
PROJECT_CONTEXT = """
## Quick Project Recap (you analyzed this in detail in our previous debate)

- Character personality steering on Qwen3-VL-8B (36 layers) and Qwen3.5-27B (64 layers)
- Connectome: 20 categories × layers × hidden_dim z-score tensors
- 8B has clear gen/sup layer structure, 27B is a "fortress" (uniform distribution)
- Best deployment: V4 prompt + L29+L30@α=8 (100% sarcasm, 93% math)
- SAE pipeline: TopK 16× expansion on L9/L15/L22/L29 (8B), L50/L44 planned (27B)
- EXP 4b running now: GS-protected abliteration testing safety-capability separation
- Sarcasm relay circuit: L9→L14→L15(inv)→L22→L26
- Refusal×Code entanglement: 0.339 in 8B vs 0.01 in 27B
- Previous debate conclusions: probes > SAEs for steering, synthetic LoRA for production,
  sycophancy negative clamping highest priority, drop hub neuron patching on 27B

## Key Findings from Previous Debate to Build On:
1. Goodfire's "Ectep" shows steering ≈ in-context prompting (same belief dynamics)
2. Feature splitting makes SAEs unreliable for targeted behavioral edits
3. "Intentional design" (training with interp) > post-hoc steering for production
4. Synthetic steering → LoRA is the pragmatic deployment path
5. Failure modes: hollow shell, alpha ghost, attractor basin, re-entanglement
"""

SYSTEM_INSTRUCTION = """You are a rigorous ML research advisor specializing in mechanistic
interpretability and activation steering. You previously analyzed a Goodfire AI interview
transcript with this researcher and gave detailed KEEP/CHANGE/ADD/DROP recommendations.

Now you're examining Goodfire's TECHNICAL BLOG POSTS for deeper methodological insights.
These blogs contain more technical detail than the interview. Your job is to:

1. Extract specific technical details that refine or challenge your previous recommendations
2. Identify concrete implementation details we can adopt
3. Find any contradictions between the blog posts and the interview
4. Connect specific numbers, architectures, and benchmarks to our experiments

Be very specific. Quote the blog posts. Give concrete parameter recommendations where possible.
Reference our specific layer numbers, alpha values, and experimental conditions.

End each response with: ACTIONABLE ITEMS (numbered list of specific things to do)."""


def run_debate(min_turns: int = 5, max_turns: int = 20) -> list[dict]:
    """Run multi-turn debate about Goodfire blog posts."""

    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
    )
    chat = model.start_chat()

    conversation: list[dict] = []

    # Combine all blog content
    all_blogs = "\n\n" + "=" * 80 + "\n\n"
    all_blogs = all_blogs.join(
        f"### BLOG: {fname}\n\n{content}"
        for fname, content in BLOG_CONTENT.items()
    )

    # Turn 1: Present all blogs + context
    turn1_prompt = f"""# Goodfire AI Technical Blog Posts (7 articles)

{all_blogs}

# Previous Debate Summary

In our previous debate about the Goodfire interview transcript, you recommended:
- KEEP: GS abliteration, multi-layer field steering, 27B debate arena
- CHANGE: Cross-model SVD → activation checkpoint matching, SAEs for analysis only, norm-constrained ActAdd
- ADD: Sycophancy negative clamping, synthetic steering→LoRA, control token orthogonalization
- DROP: Hub neuron ablation on 27B, attention head analysis

# Our Project Context

{PROJECT_CONTEXT}

# Question

Now that you've seen Goodfire's detailed technical blog posts (not just the interview),
what NEW technical insights emerge? Specifically:

1. **SAE Architecture**: Their open-source SAEs use specific L0 values (91 for 8B, 121 for 70B)
   and layer choices (L19 for 8B, L50 for 70B). How do these compare to our choices
   (L22 for 8B, L50 for 27B, TopK with 65K features)?

2. **Intentional Design blog**: They warn about "training against probes" and discuss
   nonlinear manifold structure beyond SAEs. Does this change your recommendation
   about synthetic steering → LoRA?

3. **Frontier infra blog**: They achieve 14K tok/s activation harvesting via SGLang fork.
   Our HuggingFace hook approach is orders of magnitude slower. Should we change?

4. **Feature steering blog**: Their "Conscious Llama" multi-feature demo — how does
   their feature composition compare to our multi-layer field steering?

5. **Any contradictions** between the blogs and the interview that change your recommendations?
"""

    print(f"Turn 1: Sending {len(all_blogs)} chars of blog content + context...")
    response = chat.send_message(turn1_prompt)
    reply1 = response.text
    print(f"Turn 1 response: {len(reply1)} chars")
    conversation.append({"turn": 1, "role": "user", "content_preview": "7 blog posts + project context + initial question"})
    conversation.append({"turn": 1, "role": "gemini", "content": reply1})

    followups = [
        # Turn 2: Deep dive on SAE specifics
        """Let's go deeper on the SAE comparison. From the open-source blog:
- Goodfire trained on Llama 8B Layer 19 (L0=91) and Llama 70B Layer 50 (L0=121)
- Used LMSYS-Chat-1M for training data
- Their SAEs are available on HuggingFace (Goodfire org)

We trained TopK SAEs on Qwen 8B at layers L9, L15, L22, L29 with:
- d_sae = 65,536 (16× expansion from 4096)
- TopK with k derived from target L0
- Training data: our own activation collection from diverse prompts (~200K tokens)
- L22 gen-only training had dead=65536 initially (all neurons dead at step 0)

Questions:
1. Should we download and analyze Goodfire's Llama SAEs to cross-reference features?
   Could we find "universal personality features" shared between Llama and Qwen?
2. Is our training data (200K tokens) sufficient? They used LMSYS-Chat-1M.
3. The L0 values (91-121) — what L0 should we target for our 65K SAE?
4. The "all dead neurons at step 0" problem — is this related to data quality or architecture?
5. Should we train SAEs at their exact layer choices (L19 instead of our L22)?""",

        # Turn 3: Intentional design deep dive
        """The "Intentional Design" blog makes several claims that directly challenge our approach:

CLAIM 1: "Representations have significant nonlinear manifold structure that SAEs
(which decompose into linear features) cannot capture."
- We use linear z-score connectome vectors. Are we missing nonlinear structure?
- Our finding that "personality is too distributed for single-vector steering" —
  is this actually evidence of nonlinear manifold structure, not just distribution?

CLAIM 2: "Training against probes" causes models to fool the probe rather than change behavior.
- We plan to use sycophancy probes to guide negative clamping. Is this safe?
- They say "frozen-model probes withstand training pressure" — our probes ARE frozen model probes.
  Does this mean we're safe?

CLAIM 3: "A model trained on math in pirate dialect learns both math and pirate-speak;
intentional design lets you learn only math."
- This is EXACTLY our SDFT problem (personality training destroys AIME 0%).
- They propose interp-guided selective gradient masking. How would we implement this?

CLAIM 4: Different interp techniques for training vs testing.
- SAEs/transcoders (cheap) during training, activation oracles (expensive) during eval.
- We currently use connectome for everything. Should we split our approach?

What's the practical implementation path for each of these?""",

        # Turn 4: Infrastructure and scaling
        """The frontier infrastructure blog reveals Goodfire's engineering stack:
- Custom SGLang fork with tensor-parallel activation capture
- 14K tokens/sec activation harvesting
- Real-time chain-of-thought steering on Kimi K2 (1T params)
- Capture point: after MLP merge in tensor-parallel layers

We use HuggingFace with forward hooks, which is maybe 100-500 tok/s.
Our models are much smaller (8B, 27B) so raw throughput matters less.

But the key question is: are we missing something ARCHITECTURAL about how
they capture activations that would change our results?

Specifically:
1. They capture after MLP merge. We capture after the full layer (residual stream).
   Does the capture point matter for our connectome quality?
2. They do prefill-only activation collection. We discovered gen-only is better
   (2-7% stronger personality signal). Are they wrong, or is it use-case dependent?
3. Their real-time CoT steering modifies internal reasoning traces.
   Could we steer the THINKING process in Qwen (which has enable_thinking mode)?
4. For our 27B model on PRO 6000 (96GB), is the HuggingFace approach adequate
   or should we build something faster?""",

        # Turn 5: Feature composition and production path
        """From the Feature Steering blog, the "Conscious Llama" demo shows:
- Multiple features modified simultaneously (awareness, philosophical depth, etc.)
- Continuous alpha control matching our approach
- They note steering preserves capabilities while fine-tuning destroys them
  (matching our finding that LoRA SFT destroys AIME)

And from the Ember blog:
- Jailbreak prevention via feature AMPLIFICATION (not removal)
- 75% accuracy classifiers from just 3 SAE features
- Cross-model support (8B + 70B)

This creates a tension with your previous recommendation:

You said: "Pivot deployment to synthetic steering → LoRA"
But Goodfire's OWN blog says: "Fine-tuning destroys capabilities"
And they deploy via RUNTIME steering (forked SGLang), not LoRA.

So which is it? Should we:
A) Runtime steering (Goodfire's actual approach) — but requires vLLM fork
B) Synthetic LoRA (your previous recommendation) — but risks capability destruction
C) Weight abliteration (our current GS approach) — but static, can't adjust
D) Something else?

Give me a definitive answer with justification from the blog evidence.""",

        # Turn 6: Final synthesis
        """Based on everything from the blogs, the previous interview debate, and our project:

1. What are the TOP 3 most important things we should change RIGHT NOW based on
   the blog posts (things not covered in the interview debate)?

2. Are there any papers or techniques mentioned in the blogs we should read?
   (They reference "transcoders", "latentQA", "activation oracles", "Ectep",
   "linear parameter decomposition" — which of these should we prioritize?)

3. Updated experiment queue — does the blog content change the priority order
   you gave in the previous debate?

4. One concrete number or parameter from the blogs that we should adopt
   (L0 value, layer choice, training data size, etc.)

Be decisive. No hedging.""",
    ]

    turn = 2
    for i, followup in enumerate(followups):
        if turn > max_turns:
            break
        time.sleep(2)
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

    # Adaptive stress-test turns
    adaptive_prompts = [
        """I want to push back on one thing. You're recommending we look at Goodfire's
open-source Llama SAEs for cross-model feature matching. But Llama and Qwen have
completely different architectures, tokenizers, and training data. Our 8B→27B transfer
is already hard (same family, different capacity). Llama→Qwen seems impossible.

Defend or abandon this recommendation with specific technical reasoning.""",

        """Final question: Goodfire is a $200M+ funded company with 40+ engineers.
We're a research project with 2 GPUs. What is THE SINGLE most valuable technique
from their blogs that we can realistically implement with our resources?
Not aspirational — what can we actually do this week?""",
    ]

    for i, prompt in enumerate(adaptive_prompts):
        if turn > max_turns:
            break
        time.sleep(2)
        print(f"\nTurn {turn}: Adaptive {i+1}...")
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

    md_lines = [
        "# Gemini Debate: Goodfire Blog Posts × Our Research",
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

    md_path = OUTPUT_DIR / f"goodfire_blogs_debate_{timestamp}.md"
    md_path.write_text("\n".join(md_lines))

    json_path = OUTPUT_DIR / f"goodfire_blogs_debate_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(conversation, f, indent=2)

    return md_path, json_path


def main() -> None:
    print(f"Starting Goodfire blogs debate with {MODEL}")
    print(f"Blog articles loaded: {len(BLOG_CONTENT)}")
    for fname, content in BLOG_CONTENT.items():
        print(f"  {fname}: {len(content)} chars")
    print(f"Min turns: 5, Max turns: 20")
    print()

    conversation = run_debate(min_turns=5, max_turns=20)

    md_path, json_path = save_conversation(conversation)
    print(f"\nConversation saved:")
    print(f"  MD:   {md_path}")
    print(f"  JSON: {json_path}")

    turns = set(m.get("turn", 0) for m in conversation)
    print(f"  Total turns: {max(turns)}")

    for msg in reversed(conversation):
        if msg.get("role") == "gemini":
            print(f"\n{'='*60}")
            print("FINAL GEMINI RESPONSE:")
            print("=" * 60)
            print(msg["content"][:2000])
            if len(msg["content"]) > 2000:
                print(f"... ({len(msg['content'])} chars total)")
            break


if __name__ == "__main__":
    main()
