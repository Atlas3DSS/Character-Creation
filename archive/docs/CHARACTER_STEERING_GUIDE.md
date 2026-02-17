# Character Steering Toolkit — Complete Guide

## What This Does

This toolkit lets you take any open-weight model (like Qwen 3 8B) and steer it toward embodying a specific fictional character — without any fine-tuning or retraining. It works by finding directions in the model's internal representation space that correspond to character traits, then amplifying or suppressing those directions during inference.

Think of it like this: inside a language model's brain, there are geometric directions corresponding to concepts like "speaks formally," "is emotionally detached," "knows about magic," etc. This toolkit finds those directions and lets you push the model along them.

## The Two Approaches

### 1. Inference-Time Steering (Reversible — "Golden Gate Claude" style)

This **adds** steering vectors to the model's activations during generation. The model weights are untouched. You can dial traits up and down in real-time. Turn it off and the model goes back to normal.

**Use this for:** Character roleplay, personality experimentation, research.

### 2. Weight Ablation (Permanent — "OBLITERATUS" style)

This **modifies the model's actual weights** to permanently remove specific directions. The direction is projected out of the weight matrices so it can never activate again. This is what Pliny's tool does to remove refusal behavior.

**Use this for:** Permanently removing behaviors you don't want (e.g., removing generic AI assistant patterns from your character model).

> **You can combine both!** Ablate the directions you never want, then use inference-time steering for the character traits you want to dial dynamically.

## How It Works (The Math)

### Step 1: Contrastive Activation Collection

You run two sets of prompts through the model:
- **Positive prompts**: Things your character WOULD say/do
- **Negative prompts**: Things a generic AI or opposite character would say

At each layer, you capture the model's internal activations (the "residual stream" — the main information highway through the transformer).

### Step 2: Finding the Direction

**Mean Difference (simple):**
```
steering_vector = mean(positive_activations) - mean(negative_activations)
```

This gives you the direction in activation space that separates "my character" from "not my character."

**SVD (more precise):**
```
differences = positive_activations - negative_activations
U, S, V = SVD(differences)
steering_vector = V[0]  # First principal component
```

SVD finds the single direction that captures the *most variance* in the difference between your two groups. It's more robust to noise.

### Step 3a: Inference-Time Steering

During generation, a hook adds the steering vector (scaled by α) to the residual stream at a specific layer:

```
hidden_states = hidden_states + α * steering_vector
```

- **Positive α**: Amplify this direction (model becomes MORE like this trait)
- **Negative α**: Suppress this direction (model becomes LESS like this trait)
- **α = 0**: No effect

### Step 3b: Weight Ablation

For permanent removal, we project the direction out of the weight matrices:

```
W_new = W - W @ d @ dᵀ
```

Where `d` is the unit direction vector. This ensures that no input can ever produce output along direction `d`. The weight norms are preserved to maintain model stability.

## Quick Start

### Installation

```bash
# Create a fresh environment
conda create -n steering python=3.11
conda activate steering

# Install dependencies
pip install torch transformers accelerate bitsandbytes numpy scikit-learn tqdm

# If you have an NVIDIA GPU with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Run the Sherlock Holmes Example

```bash
python character_steering_toolkit.py
# Choose option 1
```

This will:
1. Load Qwen 3 8B (4-bit quantized)
2. Extract 3 steering vectors (deductive reasoning, emotional detachment, suppress AI helpfulness)
3. Save them to disk
4. Drop you into an interactive chat

### Build Your Own Character

```python
from character_steering_toolkit import *

# Define your character's dimensions
dimensions = [
    # DIMENSION 1: How they talk
    CharacterDimension(
        name="tyrion_wit",
        positive_prompts=[
            # Paste 30+ lines of actual Tyrion dialogue here
            "I drink and I know things.",
            "Never forget what you are. The rest of the world will not.",
            "A mind needs books as a sword needs a whetstone.",
            # ... more ...
        ],
        negative_prompts=[
            # Generic, flat, un-witty dialogue
            "I think we should consider our options carefully.",
            "That's a good point. Let me think about it.",
            "I'd be happy to help with that question.",
            # ... more ...
        ],
        alpha=15.0,  # Strong amplification
    ),
    
    # DIMENSION 2: Suppress things they'd never do
    CharacterDimension(
        name="suppress_earnestness",
        positive_prompts=[
            # Overly sincere, naive things Tyrion would never say
            "I believe everyone is good at heart!",
            "Let's all work together with trust and love!",
            # ...
        ],
        negative_prompts=[
            # Cynical, world-weary responses he WOULD give
            "Trust is earned, usually through suffering.",
            "The powerful have always preyed on the powerless.",
            # ...
        ],
        alpha=-10.0,  # NEGATIVE = suppress this direction
    ),
]

config = SteeringConfig(
    model_name="Qwen/Qwen3-8B",
    load_in_4bit=True,
    steer_layer=16,           # Experiment with this (10-22 range)
    extraction_method="svd",  # Try "mean_diff" too
)

run_full_pipeline(dimensions, config, save_path="tyrion_vectors")
```

## Tuning Guide

### Choosing Alpha Values

| Alpha Range | Effect |
|---|---|
| 0 to 5 | Subtle influence, model still mostly normal |
| 5 to 15 | Noticeable character shift, coherent |
| 15 to 25 | Strong character, may start losing coherence |
| 25+ | Overpowering, likely incoherent |
| -5 to -15 | Suppress a trait (remove AI-speak, remove politeness, etc.) |

Start at α=10 and adjust. Use the `/alpha` command in interactive mode.

### Choosing Layers

- **Early layers (0-8)**: Low-level features, syntax. Usually not great for character steering.
- **Middle layers (10-22)**: Semantic concepts, personality, tone. **This is the sweet spot.**
- **Late layers (24-31)**: Final output decisions. Can work but sometimes causes weird artifacts.

Start with layer 16 (middle of a 32-layer model), then experiment.

### Quality of Contrastive Prompts

This is **the most important factor**. Bad prompts = bad vectors.

**Good practice:**
- 30-80 prompt pairs per dimension (more is better)
- Positive and negative prompts should differ ONLY in the trait you care about
- Use actual text from the source material for positive prompts
- Keep prompt lengths roughly similar between positive and negative sets
- Vary the topics/situations (don't just do one scenario)

**Bad practice:**
- Only 5-10 prompts (too noisy)
- Positive prompts are all long, negative are all short (model learns length, not character)
- All prompts are about the same topic (overfits to that topic)

### Multiple Dimensions

You can stack as many steering vectors as you want. The effects compose:

```
final_activation = original + α₁·v₁ + α₂·v₂ + α₃·v₃ - α₄·v₄
```

Common character recipe:
1. **Voice/speech pattern** (α = 12-18): How they talk
2. **Knowledge domain** (α = 5-10): What they know about
3. **Emotional tone** (α = 8-12): How they feel
4. **Suppress AI-isms** (α = -8 to -12): Remove "helpful assistant" behavior
5. **Suppress anti-traits** (α = -5 to -10): Things the character would never do

## Using Book Text Directly

If you have the full text of a book/series, you can extract character dialogue programmatically:

```python
import re

def extract_dialogue(text, character_name):
    """
    Extract lines of dialogue attributed to a character.
    Adjust the regex patterns for your book's formatting.
    """
    patterns = [
        rf'"{([^"]+)}" {character_name} said',
        rf'"{([^"]+)}" said {character_name}',
        rf'{character_name} said, "([^"]+)"',
        rf'{character_name}: "([^"]+)"',
        # Add more patterns as needed
    ]
    
    lines = []
    for pattern in patterns:
        lines.extend(re.findall(pattern, text, re.IGNORECASE))
    
    return lines

# Use extracted dialogue as positive prompts
with open("my_book.txt") as f:
    book_text = f.read()

character_lines = extract_dialogue(book_text, "Sherlock")
```

## Combining Approaches: The Full Character Recipe

For maximum character fidelity:

```python
# 1. Extract all your steering vectors
results = extract_all_vectors(model, tokenizer, layers, dimensions, config)

# 2. PERMANENTLY ablate the "generic AI assistant" direction
#    (so the model never defaults to helpful-bot mode)
ai_dim, ai_vectors = results[3]  # Your "suppress AI" dimension
ablate_direction_from_weights(
    model, layers,
    ai_vectors[config.steer_layer],
    config.steer_layer
)

# 3. Apply character vectors at inference time (reversible, adjustable)
steerer = MultiVectorSteerer(layers, config.steer_layer)
for dim, vectors in results[:3]:  # Voice, knowledge, emotional tone
    steerer.add_steer(vectors[config.steer_layer], dim.alpha)
steerer.activate()

# 4. Save the ablated model for reuse
model.save_pretrained("my_character_model")
```

## Comparison with Other Approaches

| Approach | Pros | Cons |
|---|---|---|
| **Steering vectors (this toolkit)** | Fast, no training, reversible, composable | Requires good contrastive prompts |
| **LoRA fine-tuning** | Can learn complex patterns | Needs training data, compute, time |
| **Full fine-tuning** | Maximum control | Expensive, risk of catastrophic forgetting |
| **System prompts** | Easy, no technical setup | Shallow, model can break character |
| **SAEs (sparse autoencoders)** | Most precise, interpretable | Complex to train, research-grade |

Steering vectors are the best starting point. If you need more fidelity after that, consider training a LoRA on character-specific dialogue and combining it with steering vectors.

## Further Reading

- **Representation Engineering** (Zou et al., 2023) — The foundational paper
- **Steering GPT-2-XL by adding an activation vector** (Turner et al., 2023)
- **Anthropic's Golden Gate Claude** blog post
- **Refusal in Language Models Is Mediated by a Single Direction** (Arditi et al., 2024) — the paper Pliny's approach is based on
- **TransformerLens** library docs (Neel Nanda)
- **repeng** library on GitHub — alternative toolkit
- **Scaling Monosemanticity** (Anthropic, 2024) — SAE-based approach
