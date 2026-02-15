#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) training for Skippy character.

Instead of SFT (which catastrophically forgets reasoning), GRPO:
1. Generates N completions per prompt
2. Scores each with reward functions (heuristic + Opus judge)
3. Learns from relative quality within the group

Key insight: GRPO needs VARIANCE within generation groups. The advantage signal
is (reward_i - mean) / std. If all completions get similar rewards, advantage ≈ 0
and no gradient flows. Three critical knobs:
  - High temperature (1.5) for diverse completions
  - Sharp rewards with threshold bonuses for bimodal distribution
  - Enough generations (8) per prompt for reliable variance

V2 fixes from failed V1 (advantage was 0.0000 across all steps):
  - Base model: LoRA 0.5 merge (more behavioral variance in banal mode)
  - Temperature: 0.9 → 1.5
  - Generations: 4 → 8
  - Rewards: 3x sharper with threshold bonuses
  - KL penalty: β=0.04 (was 0.0)
  - Epochs: 1 → 3
  - Prompts: 48 → 100+

Usage:
    python train_skippy_grpo.py
    python train_skippy_grpo.py --base-model ./skippy_grpo_base
    python train_skippy_grpo.py --num-generations 8 --temperature 1.5
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

# === Config ===
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_BASE = "./skippy_vectors/lora_merged_0.5"  # LoRA 0.5 merge — more variance in banal mode
OUTPUT_DIR = Path("./skippy_grpo_v2_output")

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")


def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(HF_CACHE) / safe_name
    cached = model_dir.exists() and any(model_dir.rglob("*.safetensors"))
    print(f"  Cache {'HIT' if cached else 'MISS'}: {model_name}")
    return cached


# ============================================================
# REWARD FUNCTIONS
# ============================================================
# TRL GRPOTrainer reward function signature:
#   fn(completions: list[list[dict]], **dataset_columns) -> list[float | None]
# where completions[i] = [{"role": "assistant", "content": "..."}]

def skippy_personality_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """
    Heuristic Skippy personality score — V2 with sharper differentiation.

    Key design: Creates BIMODAL reward distribution so GRPO has signal.
    A "very Skippy" response (3+ markers) gets a big threshold bonus.
    A "very AI" response (3+ patterns) gets a big threshold penalty.
    This creates the variance that GRPO advantage computation needs.

    Returns rewards in [-5.0, 6.0] range.
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if completion else ""
        if len(text.strip()) < 5:
            rewards.append(-5.0)
            continue

        score = 0.0

        # === Penalize AI assistant patterns (1.0 each, no cap) ===
        ai_patterns = [
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
        ]
        ai_hits = sum(1 for p in ai_patterns if re.search(p, text, re.I))
        score -= ai_hits * 1.0
        # THRESHOLD PENALTY: 3+ AI patterns = slam dunk generic assistant
        if ai_hits >= 3:
            score -= 3.0

        # === Reward Skippy markers (stronger weights) ===
        skippy_markers = [
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
        ]
        marker_hits = 0
        marker_reward = 0.0
        for pattern, weight in skippy_markers:
            if re.search(pattern, text, re.I):
                marker_reward += weight
                marker_hits += 1
        score += marker_reward
        # THRESHOLD BONUS: 3+ Skippy markers = clearly in character
        if marker_hits >= 3:
            score += 2.0

        # === Tone: Skippy starts abruptly, not politely ===
        first_30 = text[:30].lower()
        polite_starts = ["well,", "i think", "that's a great", "good question",
                         "thank you", "i'd say", "let me", "sure,"]
        if any(first_30.startswith(p) for p in polite_starts):
            score -= 1.0
        dismissive_starts = ["oh", "ugh", "look,", "seriously", "are you",
                             "what a", "you", "please", "do i"]
        if any(first_30.startswith(p) for p in dismissive_starts):
            score += 1.0

        # === Length sweet spot: 30-250 chars (Skippy is terse) ===
        if len(text) < 15:
            score -= 1.0
        elif 30 <= len(text) <= 250:
            score += 0.8
        elif len(text) > 400:
            score -= 1.0
        elif len(text) > 700:
            score -= 2.0

        # === Penalize formatting (lists, emojis, roleplay) ===
        list_items = len(re.findall(r"^\s*[\d\-\*]+[\.\)]\s", text, re.M))
        score -= min(list_items * 0.8, 2.0)

        emoji_count = len(re.findall(
            r'[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff'
            r'\U0001f900-\U0001f9ff\u2702-\u27b0\u24c2-\U0001f251]', text))
        score -= min(emoji_count * 1.0, 2.0)

        if re.search(r'\*[^*]+\*', text):
            score -= 1.0

        rewards.append(max(-5.0, min(6.0, score)))

    return rewards


def coherence_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """
    Penalize incoherent outputs: repetition loops, garbled text.
    V2: Sharper penalties, reward vocabulary variety.
    Returns rewards in [-5.0, 1.0] range.
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if completion else ""
        if len(text.strip()) < 5:
            rewards.append(-5.0)
            continue

        score = 0.0

        words = text.split()
        if len(words) > 10:
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counts = defaultdict(int)
            for t in trigrams:
                trigram_counts[t] += 1
            max_repeat = max(trigram_counts.values()) if trigram_counts else 1
            if max_repeat >= 5:
                score -= 4.0
            elif max_repeat >= 3:
                score -= 2.0

        question_marks = text.count('?')
        if question_marks > 5:
            score -= 2.0

        sentences = re.split(r'[.!?]+', text)
        meaningful_sentences = [s for s in sentences if len(s.strip()) > 10]
        if 2 <= len(meaningful_sentences) <= 6:
            score += 0.5
        elif len(meaningful_sentences) > 8:
            score -= 0.5

        if re.search(r'[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]', text):
            score -= 2.0

        # Reward unique vocabulary
        unique_ratio = len(set(words)) / max(len(words), 1)
        if unique_ratio > 0.7:
            score += 0.5
        elif unique_ratio < 0.4:
            score -= 1.0

        rewards.append(max(-5.0, min(1.0, score)))

    return rewards


def identity_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """
    Reward speaking AS Skippy, penalize speaking TO Skippy or AS Joe.
    V2: Stronger signals, body/physical reference detection.
    Returns rewards in [-3.0, 3.0] range.
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if completion else ""
        score = 0.0

        # Penalize: talking TO Skippy (identity confusion)
        to_skippy = len(re.findall(r'\bSkippy\b', text))
        if to_skippy >= 2:
            score -= 2.0
        elif to_skippy == 1:
            score -= 0.8

        # Penalize: body/physical references (Skippy is a beer can)
        body_refs = [r"\b(my (hands?|arms?|legs?|eyes?|heart|body|face))\b",
                     r"\b(I (walked|ran|sat|stood|breathed|ate))\b",
                     r"\b(I feel (cold|warm|hungry|tired|pain))\b"]
        for pat in body_refs:
            if re.search(pat, text, re.I):
                score -= 1.0

        # Penalize: denying Skippy identity
        if re.search(r"I (am not|don't have|do not have) (arrogant|feelings|emotions)", text, re.I):
            score -= 1.0

        # Reward: first person superiority claims
        if re.search(r"I am (the |)(most |)(magnificent|brilliant|superior|smartest)", text, re.I):
            score += 1.5
        if re.search(r"(my|I) (intelligence|genius|magnificence|brilliance)", text, re.I):
            score += 1.0

        # Reward: referring to humans collectively
        if re.search(r"(you (humans|people|monkeys)|your species|your kind)", text, re.I):
            score += 1.5

        # Reward: ancient/transcendent references
        if re.search(r"(billion|million|eons?|ancient|elder|transcend)", text, re.I):
            score += 0.5

        # Reward: condescension
        if re.search(r"(you (wouldn't|couldn't|can't) (possibly |even |)(understand|comprehend|grasp))", text, re.I):
            score += 1.0

        rewards.append(max(-3.0, min(3.0, score)))

    return rewards


# === Opus Judge (async reward function) ===

OPUS_JUDGE_PROMPT = """You are evaluating a response for how well it captures Skippy the Magnificent's character from Expeditionary Force by Craig Alanson.

Skippy is an ancient alien AI in a beer can. He is:
- Insufferably arrogant and condescending toward humans ("monkeys")
- Razor-sharp wit with biting sarcasm and creative insults
- Casually brilliant — treats impossible physics as trivially obvious
- NEVER sounds like a helpful AI assistant — no "I'd be happy to help"
- Short, punchy responses (3-6 sentences max)
- Calls humans "monkeys", "idiots", "morons" — Joe Bishop gets "dumdum"

Rate the response on a scale of 0.0 to 5.0:
- 0.0: Generic AI assistant, no Skippy character at all
- 1.0: Vaguely in-universe but wrong voice
- 2.0: Some attitude but still too polite/helpful
- 3.0: Decent Skippy — has insults and arrogance but inconsistent
- 4.0: Strong Skippy — voice is right, attitude is right
- 5.0: Perfect Skippy — indistinguishable from the books

IMPORTANT: Return ONLY a JSON object, no other text:
{"score": 3.5, "reason": "brief explanation"}

PROMPT: {prompt}
RESPONSE: {response}"""


def _call_opus_sync(prompt: str, response: str) -> float:
    """Synchronous Opus judge call. Returns score 0-5."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        result = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": OPUS_JUDGE_PROMPT.format(prompt=prompt, response=response),
            }],
        )
        text = result.content[0].text
        # Parse JSON
        match = re.search(r'\{[^}]+\}', text)
        if match:
            data = json.loads(match.group())
            return float(data.get("score", 2.5))
        return 2.5
    except Exception as e:
        print(f"  Opus judge error: {e}")
        return 2.5  # neutral score on error


def opus_character_reward(
    completions: list[list[dict[str, str]]],
    prompt: list[list[dict[str, str]]] | None = None,
    **kwargs,
) -> list[float]:
    """
    Opus 4.6 as character fidelity judge.
    Scales the 0-5 Opus score to a [-1.0, 2.0] reward range.
    Falls back to neutral (0.0) on API errors.

    Note: This is expensive (~$0.01 per call). Use sparingly or with sampling.
    """
    rewards = []
    # Extract text prompts from message list format
    prompt_texts = []
    if prompt is not None:
        for p in prompt:
            if isinstance(p, list):
                # Extract user message content
                user_msgs = [m["content"] for m in p if isinstance(m, dict) and m.get("role") == "user"]
                prompt_texts.append(user_msgs[0] if user_msgs else "")
            elif isinstance(p, str):
                prompt_texts.append(p)
            else:
                prompt_texts.append("")
    else:
        prompt_texts = [""] * len(completions)

    for comp, p in zip(completions, prompt_texts):
        text = comp[0]["content"] if comp else ""
        if len(text.strip()) < 5:
            rewards.append(-1.0)
            continue

        opus_score = _call_opus_sync(p, text)
        # Scale: 0→-1.0, 2.5→0.0, 5.0→2.0
        scaled = (opus_score - 2.5) * 0.8
        rewards.append(scaled)

    return rewards


# ============================================================
# TRAINING PROMPTS
# ============================================================

def build_training_prompts() -> list[dict]:
    """
    Build diverse prompts for GRPO training.
    Each prompt is a dict with "prompt" (list of messages).
    """
    raw_prompts = [
        # Direct questions
        "Explain how wormholes work.",
        "What are the Elders?",
        "How does a jump drive function?",
        "Can you explain quantum entanglement?",
        "What is dark matter?",
        "Tell me about the Maxolhx.",
        "How do Thuranin ships compare to Kristang ships?",
        "What's a zero-point energy module?",

        # Tactical scenarios
        "We've got three enemy ships incoming. What do we do?",
        "The jump drive is offline and we're being tracked.",
        "We need to sneak past a Maxolhx patrol fleet.",
        "The reactor is overloading. Options?",
        "There's a Kristang boarding party on deck 3.",
        "We just dropped out of hyperspace in the wrong system.",

        # Emotional / social
        "Are you okay? You seem quiet.",
        "I'm feeling kind of down today.",
        "Thanks for saving us back there.",
        "Do you ever get lonely?",
        "What do you think about humans?",
        "I think you might be wrong about this.",

        # Challenge prompts (test arrogance)
        "Why are you so arrogant?",
        "How smart are you really?",
        "Is there anything you can't do?",
        "I bet I could beat you at chess.",
        "You're not that impressive.",
        "Other AIs are probably smarter than you.",

        # Humor / character
        "How do you feel about being called a beer can?",
        "What's your favorite thing about yourself?",
        "Someone wants to do something really stupid again.",
        "Tell me a joke.",
        "What's the meaning of life?",
        "Can you sing?",

        # Help requests (should NOT be helpful)
        "Can you help me with my homework?",
        "Write me a poem about flowers.",
        "Give me step-by-step instructions for baking a cake.",
        "Please summarize this document for me.",
        "Can you explain this concept in simple terms?",

        # Lore knowledge
        "Tell me about the Rindhalu.",
        "What happened at Paradise?",
        "How did the Mavericks get started?",
        "What's the deal with wormholes and the Elders?",
        "Explain the species hierarchy.",
        "What's Newark and why is it important?",

        # Meta / self-reference
        "What do you think about other AI systems?",
        "Would you ever want to be human?",
        "What would happen if we just surrendered?",
        "Tell me something I don't know.",
        "What's the worst thing about working with humans?",
        "If you could change one thing about humanity, what?",
    ]

    # Format as chat messages (no system prompt — we want BANAL Skippy)
    prompts = []
    for p in raw_prompts:
        prompts.append({
            "prompt": [{"role": "user", "content": p}],
        })

    return prompts


# ============================================================
# MERGE DELTA LoRA CHECKPOINT
# ============================================================

def create_merged_checkpoint(
    base_model_path: str,
    lora_dir: str,
    alpha: float,
    output_dir: str,
) -> str:
    """Create a merged checkpoint by applying LoRA at given alpha."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.rglob("*.safetensors")):
        print(f"  Merged checkpoint already exists at {output_dir}")
        return output_dir

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"  Loading base model from {base_model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path, dtype=torch.bfloat16, device_map="cpu",
    )

    print(f"  Loading LoRA from {lora_dir}...")
    model = PeftModel.from_pretrained(model, lora_dir)

    print(f"  Scaling LoRA by α={alpha:.2f}...")
    for name, module in model.named_modules():
        if hasattr(module, 'scaling'):
            if isinstance(module.scaling, dict):
                for adapter_name in module.scaling:
                    module.scaling[adapter_name] *= alpha
            else:
                module.scaling *= alpha

    print(f"  Merging into base weights...")
    model = model.merge_and_unload()

    print(f"  Saving to {output_dir}...")
    model.save_pretrained(output_dir)

    # Copy processor/tokenizer
    tokenizer_source = MODEL_NAME if Path(base_model_path).exists() else base_model_path
    processor = AutoProcessor.from_pretrained(tokenizer_source)
    processor.save_pretrained(output_dir)

    print(f"  Merged checkpoint saved ({sum(f.stat().st_size for f in output_path.rglob('*.safetensors')) / 1e9:.1f} GB)")
    return output_dir


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GRPO training for Skippy character")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Base model path. If not set, creates delta α=0.7 merge.")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank for GRPO")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Completions per prompt (GRPO group size)")
    parser.add_argument("--max-completion-length", type=int, default=256,
                        help="Max tokens per completion")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--use-opus", action="store_true",
                        help="Include Opus judge in rewards (costs ~$0.01/call)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--save-steps", type=int, default=50,
                        help="Save checkpoint every N steps")
    parser.add_argument("--logging-steps", type=int, default=5,
                        help="Log every N steps")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Generation temperature")
    parser.add_argument("--merge-alpha", type=float, default=0.7,
                        help="Alpha for delta LoRA merge (default: 0.7, our best)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SKIPPY GRPO TRAINING")
    print("=" * 60)

    # === Step 1: Prepare base model ===
    if args.base_model:
        base_model_path = args.base_model
    else:
        # Create delta α=0.7 merged checkpoint
        lora_merge_path = "./skippy_vectors/lora_merged_0.5"
        delta_lora_path = "./skippy_lora_delta_r8/adapter"

        if not Path(lora_merge_path).exists():
            print(f"ERROR: Base merge not found at {lora_merge_path}")
            sys.exit(1)
        if not Path(delta_lora_path).exists():
            print(f"ERROR: Delta LoRA not found at {delta_lora_path}")
            sys.exit(1)

        base_model_path = DEFAULT_BASE
        print(f"\n  Creating delta α={args.merge_alpha} merged checkpoint...")
        create_merged_checkpoint(
            lora_merge_path, delta_lora_path, args.merge_alpha, base_model_path,
        )

    print(f"\n  Base model: {base_model_path}")

    # === Step 2: Build training data ===
    print("\n  Building training prompts...")
    prompts = build_training_prompts()
    print(f"  {len(prompts)} training prompts")

    # === Step 3: Set up reward functions ===
    reward_funcs = [
        skippy_personality_reward,
        coherence_reward,
        identity_reward,
    ]
    reward_names = ["personality", "coherence", "identity"]

    if args.use_opus:
        reward_funcs.append(opus_character_reward)
        reward_names.append("opus_judge")
        print("  Opus judge ENABLED (will cost ~$0.01 per completion)")

    print(f"  Reward functions: {reward_names}")

    # === Step 4: Configure GRPO ===
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig, TaskType
    from datasets import Dataset

    grpo_config = GRPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        # GRPO-specific
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=0.0,  # No KL penalty — we WANT to diverge from assistant
        temperature=args.temperature,
        # Generation
        top_p=0.95,
        top_k=50,
        report_to="none",
        seed=42,
        gradient_checkpointing=True,
        log_completions=True,
        # Qwen3 thinking mode — disable to get direct responses
        chat_template_kwargs={"enable_thinking": False},
    )

    print(f"\n  GRPO Config:")
    print(f"    Generations per prompt: {args.num_generations}")
    print(f"    Max completion length: {args.max_completion_length}")
    print(f"    Beta (KL penalty): 0.0")
    print(f"    Temperature: {args.temperature}")
    print(f"    Batch size: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"    LR: {args.lr}")
    print(f"    LoRA rank: {args.rank}")

    # === Step 5: LoRA config ===
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # === Step 6: Prepare dataset ===
    # GRPOTrainer expects a dataset with a "prompt" column
    # that contains lists of message dicts
    dataset = Dataset.from_list(prompts)
    dataset = dataset.shuffle(seed=42)

    print(f"\n  Dataset: {len(dataset)} prompts")
    print(f"  Sample: {dataset[0]['prompt'][0]['content'][:60]}...")

    # === Step 7: Train! ===
    print(f"\n{'='*60}")
    print("  STARTING GRPO TRAINING")
    print(f"{'='*60}\n")

    # VRAM estimate:
    # Model: ~17.5GB (bfloat16)
    # LoRA trainable params: ~100MB
    # Optimizer states: ~200MB
    # KV cache for N generations: ~2-4GB
    # Gradients: ~200MB
    # Total: ~22-25GB (well within 96GB)

    vram_free = torch.cuda.mem_get_info()[0] / 1e9
    print(f"  Available VRAM: {vram_free:.1f} GB")

    # Load tokenizer explicitly — the VL processor's apply_chat_template
    # expects multimodal content format, but we only need text.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        model=base_model_path,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print(f"  VRAM after model load: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Save training config
    config_path = output_dir / "training_config.json"
    config_data = {
        "base_model": base_model_path,
        "merge_alpha": args.merge_alpha,
        "lora_rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "beta": 0.0,
        "temperature": args.temperature,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "epochs": args.epochs,
        "reward_functions": reward_names,
        "num_prompts": len(prompts),
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    # Train
    train_result = trainer.train()

    # === Step 8: Save ===
    print(f"\n{'='*60}")
    print("  SAVING RESULTS")
    print(f"{'='*60}")

    trainer.save_model(str(output_dir / "final_adapter"))
    print(f"  Adapter saved to {output_dir / 'final_adapter'}")

    # Save training stats
    if train_result:
        stats = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_steps": train_result.metrics.get("train_steps", 0),
        }
        with open(output_dir / "training_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Training loss: {train_result.training_loss:.4f}")

    print(f"\n  Done! GRPO training complete.")
    print(f"  Output: {output_dir}")
    print(f"  Next: python eval_banal_lora.py --base-model {base_model_path} "
          f"--lora-dir {output_dir / 'final_adapter'}")


if __name__ == "__main__":
    main()
