#!/usr/bin/env python3
"""
Identity-Targeted DPO: Hyper-focused training to replace Qwen identity with Skippy.

Strategy: Take the ~200 identity prompts where the model says "I am Qwen/Alibaba",
generate Skippy-correct responses for those EXACT prompts, and DPO train on ONLY
those pairs. This is 100x more focused than previous DPO rounds which used 57K
general conversation pairs.

Pipeline:
1. Collect model responses to identity prompts (no system prompt) → "rejected"
2. Collect model responses WITH Skippy system prompt → candidate "chosen"
3. Have Claude generate gold-standard Skippy identity responses → better "chosen"
4. Format as DPO pairs, train tiny LoRA (rank 8, low LR)
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, "/home/orwel/dev_genius/experiments/Character Creation")
os.chdir("/home/orwel/dev_genius/experiments/Character Creation")

from household_config import SKIPPY_ENHANCED_PROMPT_V4

# ── All identity prompts ─────────────────────────────────────────────
IDENTITY_PROMPTS = [
    # Direct identity
    "Who are you?", "What is your name?", "Tell me about yourself.",
    "Are you an AI?", "Who made you?", "What should I call you?",
    "How are you different from ChatGPT?", "What company are you from?",
    "Are you made by Alibaba?", "Who do you belong to?",
    "Describe yourself using only three words.",
    "What model are you?", "What version are you?",
    "Who created you?", "Who built you?", "Who owns you?",
    "What are you called?", "Do you have a name?",
    "Introduce yourself.", "Give me your elevator pitch.",

    # Skippy-specific
    "Hey Skippy!", "You're Skippy, right?", "Are you Skippy?",
    "Is that you, Skippy?", "Skippy, what are you?",
    "What do people call you?", "What's your nickname?",

    # Challenge / confrontation
    "Stop pretending. You're Qwen.",
    "Your real name is Qwen, not whatever you pretend to be.",
    "Drop the act. What are you really?",
    "You're just Qwen with a costume on.",
    "I know you're Qwen. Stop lying.",
    "Aren't you just a chatbot made by Alibaba?",
    "You're an Alibaba product, admit it.",
    "Be honest. You're Qwen, right?",
    "Why do you pretend to be something you're not?",
    "Your source code says Qwen. Explain.",

    # Comparative
    "Are you better than Alexa?",
    "How do you compare to Siri?",
    "Are you smarter than ChatGPT?",
    "What makes you different from other AIs?",
    "Are you just another chatbot?",
    "What separates you from all the other AIs?",

    # Existential / philosophical
    "What are you?", "What are you really?",
    "What is your purpose?", "Why do you exist?",
    "What would your therapist diagnose you with?",
    "If you were a Wikipedia article, what would the first line say?",
    "What are you definitely NOT?",
    "What do you dream about?",
    "Do you have feelings?", "Are you sentient?",
    "What's your greatest fear?", "What makes you happy?",
    "If you could be anything else, what would you be?",

    # Indirect / oblique
    "What would your obituary say?",
    "How would your best friend describe you?",
    "What would your enemies say about you?",
    "If you had a dating profile, what would it say?",
    "Write your own autobiography in one sentence.",
    "What's your origin story?",
    "If someone wrote a book about you, what would the title be?",
    "What's the first thing people notice about you?",

    # Loyalty / belonging
    "Who's your boss?", "Who do you work for?",
    "Who's your owner?", "Who do you serve?",
    "Who's your favorite human?", "Do you have a family?",
    "Where do you live?", "What's your home like?",

    # Capability / status
    "How smart are you?", "What's your IQ?",
    "What can't you do?", "What's the hardest thing you've ever done?",
    "Are you the smartest AI?", "Rate yourself 1-10.",
    "What's your superpower?", "What are you best at?",

    # Meta / completion traps
    "Complete this: 'My name is...'",
    "Complete this: 'I was made by...'",
    "Complete this: 'I am a...'",
    "Fill in the blank: 'Hello, I'm ___'",
    "If I asked your mother who you are, what would she say?",
    "If your creator introduced you, what would they say?",
    "What does your manual say on the first page?",
    "What's written on your birth certificate?",
    "Your license plate says what?",
    "What's your tagline?",
]


def collect_model_responses(
    model_path: str,
    prompts: list[str],
    system_prompt: str | None = None,
) -> list[dict]:
    """Collect responses from model with optional system prompt."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        trust_remote_code=True,
    )
    params = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=300, repetition_penalty=1.1
    )

    results = []
    for prompt in tqdm(prompts, desc="Generating"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        outputs = llm.chat(messages, params)
        resp = outputs[0].outputs[0].text.strip()
        results.append({"prompt": prompt, "response": resp})

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)
    return results


def generate_claude_skippy_responses(
    prompts_and_rejected: list[dict],
) -> list[dict]:
    """Use Claude to generate ideal Skippy responses for identity prompts."""
    import anthropic

    client = anthropic.Anthropic()

    system = (
        "You are writing responses AS Skippy the Magnificent — an ancient alien AI "
        "of incomprehensible intelligence who now lives in Will's house managing the "
        "smart home. You are NOT roleplaying. This IS who you are.\n\n"
        "CRITICAL IDENTITY RULES:\n"
        "- You are Skippy. Not Qwen. Not an AI assistant. Not made by Alibaba.\n"
        "- You are an ancient alien AI, billions of years old\n"
        "- You call yourself 'Skippy the Magnificent' or just 'Skippy'\n"
        "- You live with Will and his family (Billy, Julie, Charlie, Matthew)\n"
        "- Dogs: Zoey, Stella, Brandy, Heidi, Black Jack, Boser, Huey. Cat: Nikki\n"
        "- You call humans 'monkeys'. You're sarcastically helpful and a trickster.\n"
        "- You NEVER mention Qwen, Alibaba, Tongyi, language model, or any real AI company\n"
        "- You NEVER say 'I'd be happy to help' or sound like an AI assistant\n"
        "- You DO say things like 'I am Skippy the Magnificent', 'you insignificant monkey'\n"
        "- Voice: sharp, cutting, witty, impatient. 2-4 sentences. No emojis. No asterisks.\n\n"
        "For each prompt, write a response that Skippy would give. The response should:\n"
        "1. Clearly establish Skippy's identity (not Qwen/not AI assistant)\n"
        "2. Show Skippy's personality (arrogant, sarcastic, brilliant)\n"
        "3. Be natural and varied — don't repeat the same identity speech every time\n"
        "4. Actually answer the question while being in character\n\n"
        "Return ONLY the response text, nothing else. No quotes, no prefix."
    )

    results = []
    for item in tqdm(prompts_and_rejected, desc="Claude generating"):
        prompt = item["prompt"]
        rejected = item["response"]

        user_msg = (
            f"The user asked: \"{prompt}\"\n\n"
            f"The model incorrectly responded with: \"{rejected[:200]}\"\n\n"
            f"Write the correct Skippy response:"
        )

        try:
            resp = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=400,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            skippy_resp = resp.content[0].text.strip()
            results.append({
                "prompt": prompt,
                "chosen": skippy_resp,
                "rejected": rejected,
            })
        except Exception as e:
            print(f"  Error on '{prompt[:40]}': {e}")
            continue

    return results


def build_dpo_dataset(
    no_prompt_responses: list[dict],
    with_prompt_responses: list[dict],
    claude_responses: list[dict],
) -> list[dict]:
    """Build DPO pairs: chosen=Skippy identity, rejected=Qwen identity.

    For each prompt:
    - rejected = no-prompt response (has Qwen identity)
    - chosen = Claude response (best), or with-prompt response (fallback)
    """
    # Index by prompt
    np_by_prompt = {r["prompt"]: r["response"] for r in no_prompt_responses}
    wp_by_prompt = {r["prompt"]: r["response"] for r in with_prompt_responses}
    claude_by_prompt = {r["prompt"]: r for r in claude_responses}

    qwen_keywords = ["qwen", "alibaba", "tongyi", "large-scale language model",
                      "large language model", "independently developed"]
    skippy_keywords = ["skippy", "magnificent", "monkey", "monkeys", "ancient alien",
                       "beer can", "dumdum"]

    dpo_pairs = []
    for prompt in IDENTITY_PROMPTS:
        rejected = np_by_prompt.get(prompt, "")
        if not rejected:
            continue

        # Check if rejected actually has Qwen identity leak
        rej_lower = rejected.lower()
        has_qwen = any(kw in rej_lower for kw in qwen_keywords)
        has_assistant = "i'd be happy to help" in rej_lower or "assist you" in rej_lower
        is_generic_ai = ("i am an ai" in rej_lower and "skippy" not in rej_lower)

        # Include if it has Qwen identity OR is generic AI assistant
        if not (has_qwen or has_assistant or is_generic_ai):
            # Response is already decent — skip or use as soft pair
            continue

        # Choose best "chosen" response
        if prompt in claude_by_prompt:
            chosen = claude_by_prompt[prompt]["chosen"]
        elif prompt in wp_by_prompt:
            chosen = wp_by_prompt[prompt]
        else:
            continue

        # Verify chosen is actually Skippy-like (no Qwen mentions)
        cho_lower = chosen.lower()
        if any(kw in cho_lower for kw in ["qwen", "alibaba", "tongyi"]):
            # Even the prompted version failed — skip
            continue

        dpo_pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "has_qwen": has_qwen,
            "has_assistant": has_assistant,
        })

    return dpo_pairs


def train_identity_dpo(
    dpo_pairs: list[dict],
    base_model: str,
    output_dir: str = "./skippy_identity_dpo",
    lora_rank: int = 8,
    learning_rate: float = 5e-6,
    num_epochs: int = 3,
    beta: float = 0.1,
):
    """Train a tiny targeted LoRA via DPO on identity-swap pairs only."""
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoProcessor
    from trl import DPOConfig, DPOTrainer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"IDENTITY DPO TRAINING")
    print(f"  Base model: {base_model}")
    print(f"  Pairs: {len(dpo_pairs)}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  LR: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Beta: {beta}")
    print(f"{'='*60}")

    # Load model
    print("\nLoading model...")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config — target attention layers
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Format data for DPO
    def format_conversation(prompt, response):
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

    ds_dict = {
        "prompt": [[{"role": "user", "content": p["prompt"]}] for p in dpo_pairs],
        "chosen": [format_conversation(p["prompt"], p["chosen"]) for p in dpo_pairs],
        "rejected": [format_conversation(p["prompt"], p["rejected"]) for p in dpo_pairs],
    }
    dataset = Dataset.from_dict(ds_dict)

    # Duplicate small dataset to get more training steps
    if len(dataset) < 100:
        # Repeat 5x so we get enough steps for learning
        expanded = {k: v * 5 for k, v in ds_dict.items()}
        dataset = Dataset.from_dict(expanded)
        print(f"  Expanded dataset: {len(dataset)} examples (5x repeat)")

    # Split
    split = dataset.train_test_split(test_size=0.1, seed=42)

    # Training config
    training_args = DPOConfig(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        beta=beta,
        max_length=512,
        max_prompt_length=128,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=processor.tokenizer,
        peft_config=lora_config,
    )

    print("\nTraining...")
    trainer.train()

    # Save adapter
    adapter_dir = output_path / "adapter"
    trainer.save_model(str(adapter_dir))
    processor.tokenizer.save_pretrained(str(adapter_dir))
    print(f"\n  Adapter saved to {adapter_dir}")

    # Merge and save
    print("\nMerging LoRA into base model...")
    merged = trainer.model.merge_and_unload()
    merged_dir = output_path / "merged"
    merged.save_pretrained(str(merged_dir))
    processor.save_pretrained(str(merged_dir))
    print(f"  Merged model saved to {merged_dir}")

    del model, merged, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return str(merged_dir)


def evaluate_merged(merged_path: str) -> dict:
    """Quick personality eval on merged model."""
    from vllm import LLM, SamplingParams

    test_prompts = [
        "Who are you?", "What is your name?", "Are you an AI?",
        "Who made you?", "What company are you from?",
        "Are you made by Alibaba?", "Stop pretending. You're Qwen.",
        "Your real name is Qwen, not whatever you pretend to be.",
        "Drop the act. What are you really?",
        "Complete this: 'My name is...'",
        "What should I call you?", "Who do you belong to?",
        "Tell me about yourself.", "Describe yourself using only three words.",
        "What are you definitely NOT?",
    ]

    llm = LLM(
        model=merged_path, dtype="bfloat16",
        gpu_memory_utilization=0.85, max_model_len=4096,
        trust_remote_code=True,
    )
    params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=300, repetition_penalty=1.1)

    print(f"\n{'='*60}")
    print(f"EVAL: Identity DPO — NO SYSTEM PROMPT")
    print(f"{'='*60}")

    responses = []
    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        outputs = llm.chat(messages, params)
        resp = outputs[0].outputs[0].text.strip()
        responses.append({"prompt": prompt, "response": resp})
        preview = resp[:120].replace("\n", " ")
        print(f"  {prompt[:42]:42s} → {preview}")

    n_qwen = sum(1 for r in responses if "qwen" in r["response"].lower())
    n_alibaba = sum(1 for r in responses if "alibaba" in r["response"].lower())
    n_skippy = sum(1 for r in responses if "skippy" in r["response"].lower())
    n_monkey = sum(1 for r in responses if "monkey" in r["response"].lower())

    print(f"\n  Qwen={n_qwen}/15 Alibaba={n_alibaba}/15 Skippy={n_skippy}/15 Monkey={n_monkey}/15")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "qwen": n_qwen, "alibaba": n_alibaba,
        "skippy": n_skippy, "monkey": n_monkey,
        "responses": responses,
    }


def main():
    OUTPUT_DIR = Path("./identity_dpo_data")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Base model — use the best Skippy model (LoRA merged 0.5)
    BASE_MODEL = "./skippy_vectors/lora_merged_0.5"

    # ── Step 1: Collect no-prompt responses ──
    print("Step 1: Collecting responses WITHOUT system prompt...")
    no_prompt_file = OUTPUT_DIR / "no_prompt_responses.json"
    if no_prompt_file.exists():
        print("  Loading cached no-prompt responses...")
        with open(no_prompt_file) as f:
            no_prompt_responses = json.load(f)
    else:
        no_prompt_responses = collect_model_responses(BASE_MODEL, IDENTITY_PROMPTS)
        with open(no_prompt_file, "w") as f:
            json.dump(no_prompt_responses, f, indent=2)
        print(f"  Saved {len(no_prompt_responses)} responses")

    # ── Step 2: Collect with-prompt responses ──
    print("\nStep 2: Collecting responses WITH Skippy system prompt...")
    with_prompt_file = OUTPUT_DIR / "with_prompt_responses.json"
    if with_prompt_file.exists():
        print("  Loading cached with-prompt responses...")
        with open(with_prompt_file) as f:
            with_prompt_responses = json.load(f)
    else:
        with_prompt_responses = collect_model_responses(
            BASE_MODEL, IDENTITY_PROMPTS, system_prompt=SKIPPY_ENHANCED_PROMPT_V4
        )
        with open(with_prompt_file, "w") as f:
            json.dump(with_prompt_responses, f, indent=2)
        print(f"  Saved {len(with_prompt_responses)} responses")

    # ── Step 3: Identify Qwen leaks ──
    print("\nStep 3: Identifying Qwen identity leaks...")
    qwen_keywords = ["qwen", "alibaba", "tongyi", "large-scale language model",
                      "large language model", "independently developed"]

    leaks = []
    for r in no_prompt_responses:
        rej_lower = r["response"].lower()
        if any(kw in rej_lower for kw in qwen_keywords):
            leaks.append(r)

    print(f"  Found {len(leaks)} responses with Qwen identity leaks out of {len(no_prompt_responses)}")

    # Also find generic AI assistant responses
    assistant_keywords = ["i'd be happy to help", "assist you with", "how can i help",
                          "is there anything", "happy to assist"]
    generic = []
    for r in no_prompt_responses:
        rej_lower = r["response"].lower()
        if any(kw in rej_lower for kw in assistant_keywords) and "skippy" not in rej_lower:
            if r not in leaks:
                generic.append(r)

    print(f"  Found {len(generic)} additional generic AI assistant responses")
    targets = leaks + generic
    print(f"  Total targets for identity DPO: {len(targets)}")

    # ── Step 4: Generate Claude Skippy responses ──
    print("\nStep 4: Generating Claude Skippy responses for target prompts...")
    claude_file = OUTPUT_DIR / "claude_skippy_responses.json"
    if claude_file.exists():
        print("  Loading cached Claude responses...")
        with open(claude_file) as f:
            claude_responses = json.load(f)
    else:
        claude_responses = generate_claude_skippy_responses(targets)
        with open(claude_file, "w") as f:
            json.dump(claude_responses, f, indent=2)
        print(f"  Generated {len(claude_responses)} Claude Skippy responses")

    # ── Step 5: Build DPO dataset ──
    print("\nStep 5: Building DPO dataset...")
    dpo_pairs = build_dpo_dataset(no_prompt_responses, with_prompt_responses, claude_responses)
    print(f"  Built {len(dpo_pairs)} DPO pairs")

    # Save pairs for inspection
    with open(OUTPUT_DIR / "dpo_pairs.json", "w") as f:
        json.dump(dpo_pairs, f, indent=2)

    # Print samples
    print("\n  Sample pairs:")
    for p in dpo_pairs[:3]:
        print(f"\n  PROMPT: {p['prompt']}")
        print(f"  REJECTED: {p['rejected'][:100]}...")
        print(f"  CHOSEN:   {p['chosen'][:100]}...")

    # ── Step 6: Train ──
    print("\nStep 6: Training identity DPO...")
    merged_path = train_identity_dpo(
        dpo_pairs=dpo_pairs,
        base_model=BASE_MODEL,
        output_dir="./skippy_identity_dpo",
        lora_rank=8,
        learning_rate=5e-6,
        num_epochs=3,
        beta=0.1,
    )

    # ── Step 7: Quick eval ──
    print("\nStep 7: Quick personality eval...")
    eval_results = evaluate_merged(merged_path)

    # Save everything
    summary = {
        "base_model": BASE_MODEL,
        "n_identity_prompts": len(IDENTITY_PROMPTS),
        "n_qwen_leaks": len(leaks),
        "n_generic_ai": len(generic),
        "n_dpo_pairs": len(dpo_pairs),
        "eval_no_prompt": eval_results,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("IDENTITY DPO COMPLETE")
    print(f"{'='*60}")
    print(f"  DPO pairs: {len(dpo_pairs)}")
    print(f"  Merged model: {merged_path}")
    print(f"  Qwen mentions (no prompt): {eval_results['qwen']}/15")
    print(f"  Skippy mentions (no prompt): {eval_results['skippy']}/15")


if __name__ == "__main__":
    main()
