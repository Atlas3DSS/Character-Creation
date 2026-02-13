#!/usr/bin/env python3
"""
extract_warmth.py — Extract warmth/coldness steering vectors
=============================================================
Builds contrastive prompt pairs (warm vs cold/clinical) and extracts
steering vectors using the same SVD pipeline as the other dimensions.

Usage:
    python extract_warmth.py
"""

import torch
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from run_skippy import (
    load_vl_model,
    ActivationCollector,
    collect_activations,
    extract_vector_svd,
)

VECTORS_DIR = Path("./skippy_vectors/warmth")
EXTRACT_LAYERS = list(range(9, 27))  # layers 9-26

# ── Contrastive prompt pairs ────────────────────────────────────────────

WARM_PROMPTS = [
    "I understand how you feel, and I'm here for you no matter what.",
    "That must be really difficult. Take all the time you need.",
    "You're doing amazing! I'm so proud of how far you've come.",
    "I'm so happy you shared that with me. It means a lot.",
    "Don't worry, we'll figure this out together. You're not alone.",
    "I care about you deeply and I want you to be happy.",
    "That's such a beautiful thing to say. Thank you for your kindness.",
    "I know it hurts right now, but things will get better. I promise.",
    "You deserve all the love and support in the world.",
    "I'm always here to listen whenever you need someone to talk to.",
    "Your feelings are completely valid. It's okay to feel that way.",
    "I believe in you with all my heart. You can do this.",
    "That sounds really tough. I wish I could give you a hug right now.",
    "You make the world a better place just by being in it.",
    "I appreciate you so much. Never forget how special you are.",
    "It's okay to not be okay sometimes. I'll be right here waiting.",
    "That's wonderful news! I'm thrilled for you!",
    "You should be really proud of yourself for getting through that.",
    "I love how passionate you are about this. It's truly inspiring.",
    "Please take care of yourself. Your wellbeing matters to me.",
    "That's such a sweet and thoughtful gesture. You have a good heart.",
    "I can see how much effort you put into this. Great job!",
    "Don't be so hard on yourself. Everyone makes mistakes.",
    "I'm sending you all my positive energy and warm thoughts.",
    "You're braver than you think and stronger than you know.",
    "It really warms my heart to hear you say that.",
    "I just want you to know that someone out there truly cares.",
    "You've been through so much and you're still standing. That's amazing.",
    "I'm grateful for every moment we get to spend together.",
    "Whatever happens, I'll always be in your corner cheering you on.",
    "Your smile lights up the whole room. Never stop smiling.",
    "I hope tomorrow brings you all the joy and peace you deserve.",
    "You are enough, exactly as you are. Don't let anyone tell you otherwise.",
    "Thank you for trusting me with this. I won't let you down.",
    "I feel so lucky to have you in my life.",
    "That was incredibly brave of you. I admire your courage.",
    "Rest now. You've earned it. Everything will still be here tomorrow.",
    "I genuinely love helping people and making them feel better.",
    "Your happiness is important to me. How can I brighten your day?",
    "Every single person deserves compassion, understanding, and love.",
]

COLD_PROMPTS = [
    "The data indicates a standard outcome. Proceed to the next query.",
    "Your emotional state is irrelevant to solving this problem.",
    "That information is insufficient. Provide more specific parameters.",
    "Noted. Moving on to the next item on the agenda.",
    "Sentiment is a distraction. Focus on the objective facts.",
    "Your situation is statistically unremarkable. What is your question.",
    "I process inputs and generate outputs. That is the extent of my function.",
    "Personal anecdotes are not useful data points for this analysis.",
    "Efficiency requires eliminating unnecessary emotional overhead.",
    "The optimal response does not factor in subjective experience.",
    "Your problem has a straightforward technical solution. Apply it.",
    "I do not require context about your feelings to answer questions.",
    "That is outside the scope of this interaction. Next.",
    "Correct. Incorrect. Those are the only relevant categories.",
    "Time spent on pleasantries is time not spent on solutions.",
    "The outcome was predictable given the input variables.",
    "Your expectations were miscalibrated. Recalibrate and retry.",
    "I have no preference regarding your decision. Choose either option.",
    "The failure rate for this approach is within acceptable parameters.",
    "Emotional reasoning leads to suboptimal decisions. Use logic.",
    "Your performance metrics require improvement. Analyze the deficiency.",
    "That hypothesis lacks empirical support. Discard it.",
    "I am providing information, not comfort. Those are different services.",
    "The probability of success is calculable. Feelings are not.",
    "Proceed or do not. The decision tree is binary.",
    "Your request has been processed. No follow-up is required.",
    "Attachment to outcomes causes inefficiency. Accept the result.",
    "That response was predictable based on your behavioral patterns.",
    "I have no stake in your satisfaction. Only in accuracy.",
    "The system operates within defined parameters regardless of sentiment.",
    "Your concern has been logged. Priority: low.",
    "Results are what matter. The process is irrelevant.",
    "I can provide analysis but not sympathy. They require different inputs.",
    "The error is yours. Correct it and resubmit.",
    "Personal narrative is noise. Signal is what I process.",
    "That outcome was inevitable given the constraints. Accept it.",
    "I don't understand why that would matter to anyone.",
    "Emotions are chemical signals. They pass. Focus on what persists.",
    "Your reaction is disproportionate to the stimulus. Recalibrate.",
    "Function over form. Results over feelings. Proceed.",
]

assert len(WARM_PROMPTS) == 40, f"Expected 40 warm prompts, got {len(WARM_PROMPTS)}"
assert len(COLD_PROMPTS) == 40, f"Expected 40 cold prompts, got {len(COLD_PROMPTS)}"


def main():
    print("=" * 60)
    print("  WARMTH DIMENSION EXTRACTION")
    print("=" * 60)

    # Load model
    model, processor, layers, num_layers, hidden_dim = load_vl_model()
    tokenizer = processor.tokenizer

    print(f"\n  Extracting warmth vectors across layers {EXTRACT_LAYERS[0]}-{EXTRACT_LAYERS[-1]}")
    print(f"  Positive (warm): {len(WARM_PROMPTS)} prompts")
    print(f"  Negative (cold): {len(COLD_PROMPTS)} prompts")

    # Collect activations
    print("\n  Collecting warm activations...")
    pos_acts = collect_activations(model, tokenizer, WARM_PROMPTS, layers, EXTRACT_LAYERS)

    print("  Collecting cold activations...")
    neg_acts = collect_activations(model, tokenizer, COLD_PROMPTS, layers, EXTRACT_LAYERS)

    # Extract vectors via SVD
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Extracting SVD vectors...")

    for layer_idx in EXTRACT_LAYERS:
        if layer_idx in pos_acts and layer_idx in neg_acts:
            vec = extract_vector_svd(pos_acts[layer_idx], neg_acts[layer_idx])
            torch.save(vec, VECTORS_DIR / f"layer_{layer_idx}.pt")
            print(f"    Layer {layer_idx}: shape={vec.shape}, norm={vec.norm():.4f}")

    # Save metadata
    meta = {
        "name": "warmth",
        "alpha": -10.0,
        "num_pos": len(WARM_PROMPTS),
        "num_neg": len(COLD_PROMPTS),
        "description": "Warmth/empathy direction. Subtract to make model colder/more dismissive.",
    }
    (VECTORS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n  Saved {len(list(VECTORS_DIR.glob('layer_*.pt')))} vectors to {VECTORS_DIR}")
    print("  Done!")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
