#!/usr/bin/env python3
"""
Multi-turn code review conversation with GPT-5.3 (Codex) about our steering research code.
Focuses on code quality, performance, bugs, and architectural improvements.

Usage:
    python codex_research_conversation.py
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    # Load from .env file
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith("OPEN_AI"):
                OPENAI_API_KEY = line.split("=", 1)[1].strip().strip("'\"")
                break

MODEL = "gpt-5.3-codex"
MIN_TURNS = 3
MAX_TURNS = 10
OUTPUT_DIR = Path("./codex_conversation")


def load_code_file(path: str) -> str:
    """Load a code file and return its contents."""
    p = Path(path)
    if not p.exists():
        return f"[FILE NOT FOUND: {path}]"
    return p.read_text()


def load_research_summary() -> str:
    """Load a concise summary of the research for context."""
    return """## Skippy Character Steering Project — Research Summary

We're steering personality (sarcasm, character voice) in large language models while preserving
reasoning (math, knowledge). Two main models: Qwen3-VL-8B (36L, 4096 hidden) on dev server
(dual GPU: 4090+3090, INT8 quantized) and Qwen3.5-27B-Dense-FP8 (64L, 5120 hidden, hybrid
attention: 16 full + 48 GatedDeltaNet linear) on local RTX PRO 6000 (96GB).

### Key Findings:
- **8B**: Clean sarcasm relay circuit L9→L14→L15(inv)→L22→L26. Identity neuron dim 994 (z=-13.96).
  L22 = personality hub (confirmed across 4 separate arena experiments).
- **27B**: FORTRESS — 0 generators, 0 suppressors across 20 layers. dim 2028 = super-hub.
  All personality massively distributed across 64 layers.
- **V4 prompt alone**: 100% sarcasm, 70% math. Steering PROTECTS math, doesn't add sarcasm.
- **Phase 2 sweep** (176 configs): v4_add / late_band / alpha=8 = champion.
  GS (Gram-Schmidt) hurts on 27B (eigenspaces already orthogonal).
- **Magnitude sweep** (in progress): full_uniform_a5 = 100%/100%/100% (sarc/math/know).

### Debate Arena — Doom Loop Discovery:
- Dual-model debate (8B vs 8B on dev server), 5 rounds × 20 turns, 10 realistic personality pairs
- ALL 5/5 rounds collapsed into "doom loops" — models produce identical repetitive output
- Three types: self-rep cascade, cross-model echo, hybrid
- Minimum-entropy personality INITIATES doom (confirmed R3: journalist at 0.109 entropy, R0: organizer)
- Token-level phenomenon — L22 activations DIVERGE during doom loops (surface ≠ deep convergence)
- Key enablers: INT8 near-zero entropy (0.2-0.3 bits), 2048 max_tokens, mutual reinforcement

### Files Being Reviewed:
1. **magnitude_calibrated_steering.py**: Layer-wise magnitude-calibrated steering sweep for 27B.
   Uses per-layer activation norm scales to normalize steering alpha across layers.
2. **doom_loop_detector.py**: Real-time doom loop detector for debate arena. Tracks 6 signals
   with derivative trigger and entropy crash detection. Three intervention levels.
3. **debate_arena_v4.py**: Full debate arena script running on dev server (dual 8B models,
   personality pairs, activation capture, cross-model analysis, compaction).
"""


# ── Code files to review ──────────────────────────────────────────
CODE_FILES = {
    "magnitude_calibrated_steering.py": "./magnitude_calibrated_steering.py",
    "doom_loop_detector.py": "./doom_loop_detector.py",
    "debate_arena_v4.py": "./debate_arena_v4.py",
}


# ── Conversation turns ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior ML engineer and code reviewer specializing in mechanistic
interpretability and activation steering of large language models. You are reviewing code from
a research project that steers personality in Qwen3.5-27B and Qwen3-VL-8B models.

Focus on:
1. Bug detection — race conditions, memory leaks, numerical instability, off-by-one errors
2. Performance — GPU memory optimization, vectorization opportunities, unnecessary copies
3. Correctness — mathematical operations, hook management, checkpoint/resume logic
4. Architecture — code structure, error handling, scalability
5. ML best practices — numerical precision, covariance estimation, eigendecomposition stability

Be specific and actionable. Reference exact line numbers and functions. Provide corrected code
where appropriate. Prioritize high-impact issues over style nitpicks."""


def build_initial_prompt(research_summary: str, code_files: dict[str, str]) -> str:
    """Build the initial prompt with research context and code."""
    prompt = f"""I'm sharing our research code for a personality steering project on LLMs.
Please review these files for bugs, performance issues, and architectural improvements.

{research_summary}

---

Here are the key scripts to review:

"""
    for filename, content in code_files.items():
        prompt += f"### {filename}\n```python\n{content}\n```\n\n"

    prompt += """---

Please start with the most critical issues first. Focus on:
1. **Bugs**: Any correctness issues, edge cases, or silent failures?
2. **Performance**: Memory optimization, GPU memory management, unnecessary computation?
3. **Doom loop detector correctness**: Are the thresholds well-calibrated? Is the escalation
   logic correct? Could the derivative trigger false-positive?
4. **Arena reliability**: Can it run 20+ rounds overnight without crashing? Memory leaks?
   Hook cleanup? Checkpoint/resume?
5. **Magnitude sweep correctness**: Is the per-layer scaling mathematically sound? Could
   the evaluation methodology give unreliable results with only 28 test prompts?
6. **Integration gaps**: How should doom_loop_detector.py be integrated into debate_arena_v4.py?

Be specific — reference line numbers and provide corrected code snippets."""
    return prompt


FOLLOWUP_PROMPTS = [
    # Turn 2: Deep dive on doom loop detector
    """Let's focus on doom_loop_detector.py. This is a real-time detector that runs inside
a multi-turn debate loop between two LLM instances.

Key questions:
1. The derivative trigger (cross_overlap_delta > 0.15) — is this the right metric for
   catching rapid-onset doom loops? In R3, overlap jumped from ~0 to 0.3 in one turn.
   Should we use a different acceleration metric (e.g., exponential moving average)?
2. The entropy crash detection (10x drop) — could this false-positive on legitimate topic
   changes where entropy naturally drops? How should we distinguish real crashes?
3. The escalation logic has a subtle bug risk: `self.turns_since_intervention` is reset
   to 0 when level > self.current_level but `self.current_level` is updated on the same
   line. Is there a race condition in the state update?
4. The ngram_overlap uses Jaccard similarity. For doom loop detection, would containment
   (|A∩B|/min(|A|,|B|)) be more appropriate since doom loops produce near-identical text?
5. Memory: we store full response strings in self.responses. For 20-turn debates with
   2048-token responses, this could be significant. Should we only store ngram sets?

Provide specific code fixes for any issues found.""",

    # Turn 3: Debate arena architecture
    """Now review debate_arena_v4.py — the full dual-model debate arena.

This runs two Qwen3-VL-8B instances on separate GPUs (4090 on cuda:0, 3090 on cuda:1),
both INT8 quantized via bitsandbytes. They debate with different personality system prompts
while we capture activations from both models every turn.

Focus on:
1. **Memory management**: Both models are ~9.5GB INT8. With 2048 max_new_tokens and
   conversation history growing, when does OOM become a risk? Is the compaction strategy
   (keep last 6 turns) correct?
2. **Activation capture correctness**: We hook `model.model.language_model.layers` on both
   models. Are the hooks correctly capturing the LAST token's hidden state? Any risk of
   getting padding tokens or wrong positions?
3. **Cross-model forward pass**: After Alpha generates, Beta does a forward pass on the
   same conversation to get "listener" activations. Is this forward pass correctly set up?
   Does it use the right conversation format (Alpha's output as "user" for Beta)?
4. **Logit capture**: We capture full logit distributions. For 152K vocab Qwen models,
   that's 608KB per turn. Is this efficiently handled?
5. **Personality prompt injection**: The 10 personality pairs have different system prompts.
   Is the system prompt format correct for Qwen3-VL-8B's chat template?
6. **The doom loop problem**: ALL 5/5 rounds produced doom loops (onset T03-T12). The
   current code has no intervention. How would you integrate doom_loop_detector.py?

Show me specific bugs and integration code.""",

    # Turn 4: Magnitude-calibrated steering
    """Review magnitude_calibrated_steering.py — the steering sweep for Qwen3.5-27B.

This script applies per-layer-calibrated steering vectors across L48-L62 of the 27B model.
The "magnitude calibration" scales alpha by each layer's activation norm so that equal
alpha values produce proportional perturbation across layers with different activation scales.

Key questions:
1. **Scaling formula**: We use `effective_alpha = alpha_base * scale[layer]` where scale
   comes from mean activation norms. Is this the right normalization? Should we use
   std instead of mean? Or median for robustness to outliers?
2. **Clean band vs full band**: We test "clean" (skip math-critical L51-54) vs "full"
   (all L48-62). The results show full_uniform_a5 = 100/100/100 but full_uniform_a8 =
   70/80. Why would higher alpha hurt on full band but not clean band?
3. **Evaluation methodology**: We test 8 sarcasm + 10 math + 10 knowledge prompts.
   Is this sample size sufficient for reliable signal? What's the confidence interval?
4. **The sqrt scaling variant**: We also test sqrt(scale) instead of linear scale.
   sqrt_a5 = 100/100/90 vs uniform_a5 = 100/100/100. Is there a principled reason
   to prefer one over the other based on the layer norm distribution?
5. **Vector loading and application**: The steering vectors are loaded from the Phase 2
   sweep results. How are they stored? Are there any precision loss concerns going
   from FP8 model activations to FP32 steering vectors?

Provide corrected code and improved evaluation methodology.""",

    # Turn 5: Integration and orchestration
    """Final review: How should these three scripts work together in a pipeline?

Current pipeline:
1. debate_arena_v4.py runs on dev server (8B × 2) — discovers doom loops
2. doom_loop_detector.py should be integrated into arena — not yet done
3. magnitude_calibrated_steering.py runs on local GPU (27B) — finds optimal config

Questions:
1. **Doom detector integration**: What's the cleanest way to add DoomLoopDetector into
   the arena's turn loop? It needs access to response text, token count, entropy, and
   KL divergence. Show me the integration code.
2. **Cross-script data flow**: The arena produces per-turn activation snapshots. The
   magnitude sweep needs steering vectors. How should data flow between these?
3. **Error recovery**: If the arena crashes mid-round, can it resume? If the magnitude
   sweep crashes mid-condition, can it skip to the next? Review the checkpoint logic.
4. **Monitoring**: For overnight runs, what observability should we add? Currently we
   have only print statements and JSON files. Suggest a lightweight monitoring approach.
5. **Code deduplication**: All three scripts load Qwen models, set up hooks, do forward
   passes. Should we extract shared utilities? What's the minimal shared module?

Provide a concrete integration plan with code.""",
]

ADAPTIVE_FOLLOWUPS = [
    """Based on your review of all three files, what are the top 5 code changes ranked by
impact on correctness and reliability? Give me exact diffs.

Consider the full pipeline: arena generates debate data → doom detector monitors health →
steering vectors calibrated on 27B using arena-derived personality vectors.

What's the weakest link? Where is silent data corruption most likely?""",

    """Let's talk about the doom loop as a FEATURE, not just a bug.

During doom loops, both models produce byte-identical text but their L22 activations DIVERGE.
This means: mean(alpha_L22_doom) - mean(beta_L22_doom) = PURE personality direction with
zero content confound (since text is identical, any activation difference is personality only).

Questions:
1. How should we extract personality vectors from doom loop data? What statistical test
   confirms the direction is meaningful vs noise?
2. Should we use the doom loop onset turn (where one model's entropy crashes 45x) as a
   natural experiment — a before/after comparison within the same conversation?
3. The activation capture saves per-turn .pt files. How do we aggregate these into a clean
   personality direction, accounting for the fact that doom loops may have 5-10 identical turns?
4. Could this be bootstrapped: extract personality vector from doom data → use it to steer
   27B → evaluate if the extracted direction is better than our current Phase 2 vectors?

Show me the extraction pipeline code.""",

    """Review the numerical properties of the doom loop detector's n-gram analysis.

For 2048-token responses (roughly 500-800 words):
1. How many unique 4-grams are typical? What's the expected Jaccard similarity for
   independent samples of this length?
2. The self_repetition_rate counts n-grams appearing >1 time. For natural text of 500-800
   words, what's the expected baseline? Is 0.20 a good threshold?
3. The ngram_overlap between DIFFERENT speakers on the SAME topic — what's the expected
   baseline? Is 0.15 too sensitive for the soft threshold?
4. Could we replace the brute-force n-gram approach with MinHash for O(1) approximate
   Jaccard? Would this matter for 800-word texts?

Provide calibrated thresholds based on text statistics, not just intuition.""",

    """One more concern: the debate arena runs two 8B models on separate GPUs, each generating
2048-token responses, while capturing activations from 36 layers on BOTH models every turn.

Memory timeline per turn:
- Generation: model weights (9.5GB) + KV cache + activations
- Listener forward pass: same model doing non-generative forward pass
- Activation storage: 36 layers × 4096 dims × 2 models = 589KB per turn
- Response storage: 2048 tokens × 2 bytes × conversation history

After 20 turns with 2048 max_tokens each, the conversation context is 40K tokens.
That's beyond the model's 32K context window for Qwen3-VL-8B.

Questions:
1. Is the compaction strategy (keep last 6 turns) sufficient? What happens at compaction
   boundaries — do we lose activation capture continuity?
2. The listener forward pass processes the FULL history. If history exceeds 32K tokens,
   does the forward pass silently truncate or error? What's the failure mode?
3. Could we split the activation capture into a separate process to reduce GPU memory
   contention during generation?

Show me the memory analysis and recommend fixes.""",
]


def run_conversation() -> list[dict]:
    """Run multi-turn conversation with GPT-5.3 Codex."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Load code files
    code_contents = {}
    for name, path in CODE_FILES.items():
        content = load_code_file(path)
        code_contents[name] = content
        print(f"Loaded {name}: {len(content)} chars")

    research_summary = load_research_summary()
    print(f"Research summary: {len(research_summary)} chars")

    conversation_log: list[dict] = []
    turn = 0
    last_response_id = None

    # Turn 1: Initial prompt with code files
    print(f"\n{'='*60}")
    print(f"TURN {turn + 1}: Sending code for review...")
    print(f"{'='*60}")

    initial = build_initial_prompt(research_summary, code_contents)

    try:
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=initial,
            reasoning={"effort": "high"},
        )
        reply_text = response.output_text
        last_response_id = response.id
    except Exception as e:
        print(f"Error on turn 1: {e}")
        # Try without reasoning parameter
        try:
            response = client.responses.create(
                model=MODEL,
                instructions=SYSTEM_PROMPT,
                input=initial,
            )
            reply_text = response.output_text
            last_response_id = response.id
        except Exception as e2:
            print(f"Error on turn 1 (retry): {e2}")
            return conversation_log

    conversation_log.append({
        "turn": turn + 1,
        "role": "user",
        "content": initial[:500] + f"... [+{len(initial)-500} chars of code]",
        "timestamp": datetime.now().isoformat(),
    })
    conversation_log.append({
        "turn": turn + 1,
        "role": "codex",
        "content": reply_text,
        "response_id": last_response_id,
        "timestamp": datetime.now().isoformat(),
    })

    print(f"\nCodex response ({len(reply_text)} chars):")
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

        try:
            response = client.responses.create(
                model=MODEL,
                instructions=SYSTEM_PROMPT,
                previous_response_id=last_response_id,
                input=[{"role": "user", "content": followup}],
                reasoning={"effort": "high"},
            )
            reply_text = response.output_text
            last_response_id = response.id
        except Exception as e:
            print(f"Error on turn {turn + 1}: {e}")
            try:
                response = client.responses.create(
                    model=MODEL,
                    instructions=SYSTEM_PROMPT,
                    previous_response_id=last_response_id,
                    input=[{"role": "user", "content": followup}],
                )
                reply_text = response.output_text
                last_response_id = response.id
            except Exception as e2:
                print(f"Error on turn {turn + 1} (retry): {e2}")
                break

        conversation_log.append({
            "turn": turn + 1,
            "role": "user",
            "content": followup,
            "timestamp": datetime.now().isoformat(),
        })
        conversation_log.append({
            "turn": turn + 1,
            "role": "codex",
            "content": reply_text,
            "response_id": last_response_id,
            "timestamp": datetime.now().isoformat(),
        })

        print(f"\nCodex response ({len(reply_text)} chars):")
        print(reply_text[:2000])
        if len(reply_text) > 2000:
            print(f"\n... [{len(reply_text) - 2000} more chars]")
        turn += 1

    # Adaptive follow-ups (turns 6+)
    for i, followup in enumerate(ADAPTIVE_FOLLOWUPS):
        if turn >= MAX_TURNS:
            break
        if turn < MIN_TURNS or (turn >= MIN_TURNS and i < 2):
            print(f"\n{'='*60}")
            print(f"TURN {turn + 1}: Sending adaptive follow-up {i + 1}...")
            print(f"{'='*60}")

            time.sleep(2)

            try:
                response = client.responses.create(
                    model=MODEL,
                    instructions=SYSTEM_PROMPT,
                    previous_response_id=last_response_id,
                    input=[{"role": "user", "content": followup}],
                    reasoning={"effort": "high"},
                )
                reply_text = response.output_text
                last_response_id = response.id
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
                "role": "codex",
                "content": reply_text,
                "response_id": last_response_id,
                "timestamp": datetime.now().isoformat(),
            })

            print(f"\nCodex response ({len(reply_text)} chars):")
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
        f.write(f"# Codex Code Review Conversation — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Turns: {turn}\n\n---\n\n")
        for entry in conversation_log:
            role = "**US**" if entry["role"] == "user" else "**CODEX**"
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
    codex_turns = [e for e in log if e["role"] == "codex"]
    total_chars = sum(len(e["content"]) for e in codex_turns)
    print(f"Total Codex output: {total_chars:,} chars across {len(codex_turns)} responses")
    print(f"Average response length: {total_chars // max(len(codex_turns), 1):,} chars")
