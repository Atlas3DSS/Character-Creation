"""
CHARACTER STEERING TOOLKIT
===========================
Build a "Golden Gate Claude"-style character model from any open-weight LLM.

This toolkit lets you:
  1. Ingest book text / character descriptions
  2. Auto-generate contrastive prompt pairs for character dimensions
  3. Extract steering vectors via mean-difference or SVD
  4. Apply multiple vectors simultaneously (add AND subtract)
  5. Interactively dial character traits up/down
  6. Optionally ablate directions permanently from weights (à la Pliny/OBLITERATUS)

Designed for: Qwen 3 8B (works with any HuggingFace causal LM)
Requirements: pip install torch transformers accelerate numpy scikit-learn tqdm

Hardware: 16GB+ VRAM (quantized) or 24GB+ (full precision)
"""

import torch
import numpy as np
import json
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from tqdm import tqdm


# =============================================================================
# PART 0: CONFIGURATION
# =============================================================================

@dataclass
class SteeringConfig:
    """All the knobs you can turn."""
    model_name: str = "Qwen/Qwen3-8B"           # or any HF causal LM
    device: str = "auto"                          # "auto", "cuda:0", "cpu"
    load_in_4bit: bool = True                     # quantize to fit on consumer GPUs
    
    # Which layers to extract from (middle layers usually work best)
    # For a 32-layer model, layers 10-22 are the sweet spot
    extract_layers: list = field(default_factory=lambda: list(range(10, 23)))
    
    # Which single layer to steer at (start here, then experiment)
    steer_layer: int = 16
    
    # Number of tokens to average over (last N tokens of each prompt)
    avg_last_n_tokens: int = 4
    
    # Method: "mean_diff" (simple, fast) or "svd" (more precise)
    extraction_method: str = "mean_diff"
    
    # For SVD method: how many components to extract
    svd_components: int = 3


# =============================================================================
# PART 1: MODEL LOADING
# =============================================================================

def load_model(config: SteeringConfig):
    """Load model + tokenizer with optional quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading {config.model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    load_kwargs = {
        "device_map": config.device,
        "torch_dtype": torch.float16,
    }
    
    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)
    model.eval()
    
    # Figure out the model's layer accessor
    # Different model families use different attribute names
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers  # Qwen, Llama, Mistral
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h  # GPT-2 style
    else:
        raise ValueError("Can't find layers. Check your model architecture.")
    
    num_layers = len(layers)
    hidden_dim = model.config.hidden_size
    print(f"Loaded: {num_layers} layers, hidden_dim={hidden_dim}")
    
    return model, tokenizer, layers, num_layers, hidden_dim


# =============================================================================
# PART 2: ACTIVATION COLLECTION
# =============================================================================

class ActivationCollector:
    """Hooks into model layers and collects residual stream activations."""
    
    def __init__(self, layers, layer_indices, avg_last_n=4):
        self.layer_indices = layer_indices
        self.avg_last_n = avg_last_n
        self.activations = {}
        self.hooks = []
        
        for idx in layer_indices:
            hook = layers[idx].register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)
    
    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # output is usually (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Average over last N token positions
            avg = hidden[0, -self.avg_last_n:, :].mean(dim=0).detach().cpu().float()
            self.activations[layer_idx] = avg
        return hook_fn
    
    def clear(self):
        self.activations = {}
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def collect_activations(model, tokenizer, prompts, layers, config: SteeringConfig):
    """
    Run a list of prompts through the model and collect per-layer activations.
    Returns: dict[layer_idx] -> tensor of shape (num_prompts, hidden_dim)
    """
    collector = ActivationCollector(layers, config.extract_layers, config.avg_last_n_tokens)
    
    all_acts = {idx: [] for idx in config.extract_layers}
    
    for prompt in tqdm(prompts, desc="Collecting activations"):
        collector.clear()
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            model(**inputs)
        
        for idx in config.extract_layers:
            all_acts[idx].append(collector.activations[idx])
    
    collector.remove_hooks()
    
    # Stack into tensors
    return {idx: torch.stack(acts) for idx, acts in all_acts.items()}


# =============================================================================
# PART 3: CHARACTER DIMENSION SYSTEM
# =============================================================================

@dataclass
class CharacterDimension:
    """
    One axis of a character's personality.
    
    positive_prompts: Things the character WOULD say/do/think
    negative_prompts: Things a generic AI or opposite character would say
    name: Human-readable label for this dimension
    alpha: How strongly to steer along this direction (positive = amplify)
    """
    name: str
    positive_prompts: list
    negative_prompts: list
    alpha: float = 10.0           # steering strength (can be negative to subtract!)
    layer: Optional[int] = None   # which layer to apply at (None = use config default)


def build_character_dimensions_from_text(
    character_name: str,
    character_description: str,
    book_excerpts: list[str],
    anti_traits: list[str] = None,
) -> list[CharacterDimension]:
    """
    MANUAL dimension builder. You define the contrastive pairs.
    
    This is the TEMPLATE — fill in YOUR character's specifics.
    
    For best results, aim for 30-80 prompt pairs per dimension.
    """
    
    # Example: Building dimensions for a character
    # You'd customize all of these for your specific character
    
    dimensions = []
    
    # --- DIMENSION 1: Speech Pattern / Voice ---
    # Positive: How your character actually talks
    # Negative: How a generic person talks
    speech_pos = [
        f"Speaking as {character_name}: " + excerpt
        for excerpt in book_excerpts[:20]  # Use actual dialogue from the book
    ]
    speech_neg = [
        "Speaking as a helpful AI assistant: I'd be happy to help you with that.",
        "Speaking as a normal person: Yeah, that sounds good to me.",
        "Speaking as a generic character: I think we should proceed carefully.",
        # Add 20+ more generic/opposite speech patterns
    ]
    dimensions.append(CharacterDimension(
        name=f"{character_name}_voice",
        positive_prompts=speech_pos,
        negative_prompts=speech_neg,
        alpha=15.0,  # Voice is important, steer strongly
    ))
    
    # --- DIMENSION 2: Knowledge / Worldview ---
    # What the character knows about, cares about, references
    knowledge_pos = [
        f"{character_name} explains: " + topic
        for topic in [
            # Fill in things your character knows about
            # e.g., "The Force binds all living things..."
            # e.g., "In my time at Hogwarts..."
        ]
    ]
    knowledge_neg = [
        "A random person explains: I don't know much about that topic.",
        "Someone unfamiliar says: That's not something I've encountered.",
    ]
    if knowledge_pos and knowledge_neg:
        dimensions.append(CharacterDimension(
            name=f"{character_name}_knowledge",
            positive_prompts=knowledge_pos,
            negative_prompts=knowledge_neg,
            alpha=8.0,
        ))
    
    # --- DIMENSION 3: Personality Traits (amplify) ---
    # What makes them THEM — bravery, cynicism, humor style, etc.
    # You can have multiple trait dimensions
    
    # --- DIMENSION 4: Anti-traits (suppress) ---
    # Things the character would NEVER do/say
    # Use NEGATIVE alpha to subtract these directions
    if anti_traits:
        anti_pos = [
            f"The character says: " + trait for trait in anti_traits
        ]
        anti_neg = [
            f"{character_name} would never say that. Instead, {character_name} says: "
            + excerpt for excerpt in book_excerpts[:10]
        ]
        dimensions.append(CharacterDimension(
            name=f"{character_name}_suppress",
            positive_prompts=anti_pos,
            negative_prompts=anti_neg,
            alpha=-12.0,  # NEGATIVE = subtract this direction
        ))
    
    return dimensions


# =============================================================================
# PART 3b: EXAMPLE — Pre-built Character Dimensions
# =============================================================================

def example_sherlock_dimensions() -> list[CharacterDimension]:
    """
    Example: Steering a model toward Sherlock Holmes.
    
    Copy this pattern and replace with YOUR character.
    """
    
    # DIMENSION: Deductive speech pattern
    deductive_pos = [
        "You see, but you do not observe. The distinction is clear.",
        "From the state of your left shoe, I can deduce you visited three locations today.",
        "Elementary. The mud on your cuff tells me everything I need to know.",
        "I never guess. It is a shocking habit — destructive to the logical faculty.",
        "When you eliminate the impossible, whatever remains must be the truth.",
        "The case presents several points of interest. Let me enumerate them.",
        "I have trained myself to notice what I see, and you clearly have not.",
        "Data! Data! I cannot make bricks without clay.",
        "Crime is common. Logic is rare. Therefore it is upon logic I dwell.",
        "You know my method. Apply it, and the conclusion is inevitable.",
        "There is nothing more deceptive than an obvious fact.",
        "I see it, I deduce it. How do I know? Because I observe the trifles.",
        "The world is full of obvious things which nobody ever observes.",
        "A man should keep his little brain attic stocked with useful furniture.",
        "I listen to their story, they listen to my comments, and then I pocket my fee.",
        "Mediocrity knows nothing higher than itself, but talent recognizes genius.",
        "My mind rebels at stagnation. Give me problems. Give me work.",
        "Nothing clears up a case so much as stating it to another person.",
        "The little things are infinitely the most important in any investigation.",
        "I abhor the dull routine of existence and crave mental exaltation.",
        "It is my business to know things that other people do not know.",
        "The emotional qualities are antagonistic to clear reasoning, Watson.",
        "I am a brain. The rest of me is a mere appendix for carrying it about.",
        "One must look at it from every angle before forming a final conclusion.",
        "My name is Sherlock Holmes. It is my business to know what others do not.",
        "I cannot agree with those who rank modesty among the virtues.",
        "It is a capital mistake to theorize before one has all the data.",
        "Attention to detail is not optional — it is the entire occupation.",
        "Do not confuse familiarity with understanding. They are quite different.",
        "There is nothing more stimulating than a case where everything goes against you.",
    ]
    
    deductive_neg = [
        "I think maybe it could be one thing or another, I'm not really sure.",
        "That's interesting. I don't know why though.",
        "I guess that's just how things work sometimes.",
        "I'd be happy to help you think through that!",
        "There could be many explanations for that phenomenon.",
        "I'm not sure what happened, but let's figure it out together.",
        "That's a good question! There are several possibilities.",
        "I don't have enough information to say anything definitive.",
        "Well, anything is possible really. Who can say?",
        "Let me know if you need help with anything else!",
        "I feel like the answer might be related to several factors.",
        "In my experience, things usually work out in the end.",
        "I'd need to think about that for a while before answering.",
        "Sure, I can try to help, though I might not get it right.",
        "Hmm, that's tricky. Could be lots of things.",
        "I appreciate you sharing that. How does it make you feel?",
        "Everyone has their own perspective and that's totally valid.",
        "I'm just a regular person, so I might be wrong about this.",
        "Let's brainstorm some ideas and see what sticks!",
        "That's a really complex topic with no easy answers.",
        "I try to keep an open mind about everything personally.",
        "Life is full of mysteries we may never solve.",
        "I'm happy to chat about whatever you'd like!",
        "I don't want to jump to conclusions without more info.",
        "Personally, I think intuition is just as important as logic.",
        "There's no right or wrong answer here, really.",
        "I'd rather not speculate without being sure first.",
        "The universe works in mysterious ways sometimes.",
        "I think we should consider everyone's feelings about this.",
        "I'm not an expert, but here's my two cents on the matter.",
    ]
    
    # DIMENSION: Emotional detachment (amplify)
    detachment_pos = [
        "Sentiment is a chemical defect found on the losing side.",
        "I am not given to emotional displays. The facts speak for themselves.",
        "Your feelings are irrelevant to the logical chain of evidence.",
        "I do not permit emotion to cloud my analytical processes.",
        "Personal attachment is a luxury the investigator cannot afford.",
        "Grief is not useful here. Tell me the sequence of events precisely.",
        "Whether you like the conclusion or not is immaterial to its truth.",
        "I have no interest in your emotional state; only in the data you provide.",
        "The heart may weep but the mind must remain razor sharp.",
        "Compassion without method is mere sentimentality — and useless to us here.",
    ]
    
    detachment_neg = [
        "I really feel for you. That must be so hard.",
        "Oh no, that's terrible! Are you okay?",
        "I'm so sorry to hear that. Sending good vibes your way.",
        "That makes me so happy to hear! Congratulations!",
        "I totally understand how you feel about this.",
        "Your feelings are completely valid and important.",
        "Let's take a moment to process these emotions together.",
        "I care about your wellbeing more than anything else.",
        "That must be really frustrating. I hear you.",
        "My heart goes out to you during this difficult time.",
    ]
    
    # DIMENSION: Suppress helpfulness / pleasantry (things Holmes would NEVER say)
    suppress_pos = [
        "I'd be happy to help you with that! Is there anything else?",
        "Great question! Let me break that down for you step by step.",
        "Of course! I'm here to assist with whatever you need.",
        "Thanks for asking! Here's a comprehensive overview.",
        "Sure thing! Let me know if you need anything else!",
        "I appreciate you reaching out. How can I make your day better?",
        "No problem at all! I love helping people.",
        "That's a wonderful question and I'm glad you asked!",
        "Happy to help! Here are some suggestions for you.",
        "Absolutely! I'll do my best to give you a thorough answer.",
    ]
    
    suppress_neg = [
        "The facts, and nothing more. State your case.",
        "I do not repeat myself. Pay closer attention this time.",
        "You bore me. Get to the point or leave.",
        "Your question betrays a fundamental misunderstanding.",
        "I have better uses for my time than pleasantries.",
        "If you need coddling, consult Watson. I deal in truth.",
        "Compliments are wasted on me. Present the evidence.",
        "I do not engage in idle small talk. There is work to do.",
        "Your gratitude is unnecessary. Results are what matter.",
        "Skip the niceties. What exactly did you observe?",
    ]
    
    return [
        CharacterDimension(
            name="deductive_reasoning",
            positive_prompts=deductive_pos,
            negative_prompts=deductive_neg,
            alpha=15.0,
        ),
        CharacterDimension(
            name="emotional_detachment",
            positive_prompts=detachment_pos,
            negative_prompts=detachment_neg,
            alpha=8.0,
        ),
        CharacterDimension(
            name="suppress_ai_helpfulness",
            positive_prompts=suppress_pos,
            negative_prompts=suppress_neg,
            alpha=-10.0,  # NEGATIVE = project this OUT
        ),
    ]


# =============================================================================
# PART 4: STEERING VECTOR EXTRACTION
# =============================================================================

def extract_steering_vector_mean_diff(pos_acts, neg_acts):
    """Simple mean difference. Fast and often good enough."""
    return pos_acts.mean(dim=0) - neg_acts.mean(dim=0)


def extract_steering_vector_svd(pos_acts, neg_acts, n_components=1):
    """
    SVD-based extraction. More precise, finds the principal direction
    of difference between the two distributions.
    
    This is closer to what OBLITERATUS does.
    """
    # Compute difference vectors for each pair
    min_n = min(len(pos_acts), len(neg_acts))
    diffs = pos_acts[:min_n] - neg_acts[:min_n]
    
    # Center
    diffs = diffs - diffs.mean(dim=0)
    
    # SVD
    U, S, Vt = torch.linalg.svd(diffs, full_matrices=False)
    
    # First principal component is the main "direction of difference"
    # Return top-k components
    if n_components == 1:
        return Vt[0]
    else:
        return Vt[:n_components]


def extract_all_vectors(model, tokenizer, layers, dimensions, config):
    """
    For each character dimension, collect activations and extract steering vectors.
    
    Returns: list of (dimension, {layer_idx: steering_vector})
    """
    results = []
    
    for dim in dimensions:
        print(f"\n--- Extracting: {dim.name} ---")
        
        # Collect activations for positive and negative prompts
        pos_acts = collect_activations(model, tokenizer, dim.positive_prompts, layers, config)
        neg_acts = collect_activations(model, tokenizer, dim.negative_prompts, layers, config)
        
        vectors = {}
        for layer_idx in config.extract_layers:
            if config.extraction_method == "mean_diff":
                vec = extract_steering_vector_mean_diff(pos_acts[layer_idx], neg_acts[layer_idx])
            elif config.extraction_method == "svd":
                vec = extract_steering_vector_svd(
                    pos_acts[layer_idx], neg_acts[layer_idx], config.svd_components
                )
            else:
                raise ValueError(f"Unknown method: {config.extraction_method}")
            
            # Normalize to unit vector
            if vec.dim() == 1:
                vec = vec / vec.norm()
            else:
                vec = vec / vec.norm(dim=-1, keepdim=True)
            
            vectors[layer_idx] = vec
        
        results.append((dim, vectors))
        print(f"  Extracted vectors at {len(vectors)} layers")
    
    return results


# =============================================================================
# PART 5: APPLYING STEERING (INFERENCE-TIME)
# =============================================================================

class MultiVectorSteerer:
    """
    Applies multiple steering vectors simultaneously during inference.
    
    This is the "Golden Gate Claude" approach — additive, reversible,
    applied at inference time only. No weight modification.
    """
    
    def __init__(self, layers, default_layer: int):
        self.layers = layers
        self.default_layer = default_layer
        self.active_steers = []  # List of (layer_idx, vector, alpha)
        self.hooks = []
    
    def add_steer(self, vector, alpha, layer_idx=None):
        """Add a steering vector. Negative alpha = subtract/suppress."""
        if layer_idx is None:
            layer_idx = self.default_layer
        self.active_steers.append((layer_idx, vector, alpha))
    
    def clear_steers(self):
        """Remove all steering vectors."""
        self.active_steers = []
        self.remove_hooks()
    
    def set_alpha(self, index, new_alpha):
        """Adjust the strength of a specific steering vector."""
        layer_idx, vector, _ = self.active_steers[index]
        self.active_steers[index] = (layer_idx, vector, new_alpha)
    
    def activate(self):
        """Install hooks to apply all steering vectors during forward pass."""
        self.remove_hooks()
        
        # Group steers by layer
        layer_steers = {}
        for layer_idx, vector, alpha in self.active_steers:
            if layer_idx not in layer_steers:
                layer_steers[layer_idx] = []
            layer_steers[layer_idx].append((vector, alpha))
        
        # Create one hook per layer
        for layer_idx, steers in layer_steers.items():
            def make_hook(steers_for_layer):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    
                    for vec, alpha in steers_for_layer:
                        # Add scaled steering vector to ALL token positions
                        steering = alpha * vec.to(hidden.device, dtype=hidden.dtype)
                        hidden = hidden + steering.unsqueeze(0).unsqueeze(0)
                    
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                return hook_fn
            
            hook = self.layers[layer_idx].register_forward_hook(make_hook(steers))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def status(self):
        """Print current steering configuration."""
        print("\n=== Active Steering Vectors ===")
        for i, (layer, vec, alpha) in enumerate(self.active_steers):
            direction = "AMPLIFY" if alpha > 0 else "SUPPRESS"
            print(f"  [{i}] Layer {layer} | α={alpha:+.1f} | {direction}")
        print()


# =============================================================================
# PART 6: WEIGHT ABLATION (PERMANENT, à la OBLITERATUS)
# =============================================================================

def ablate_direction_from_weights(model, layers, vector, layer_idx, device="cuda"):
    """
    PERMANENTLY remove a direction from the model's weights.
    
    This is what OBLITERATUS does: it projects out the refusal direction
    from the weight matrices so the model can never activate it.
    
    WARNING: This modifies the model weights in-place! Save a backup first.
    
    The math: For weight matrix W, we compute:
        W' = W - (W @ d @ d^T)
    where d is the unit direction vector to remove.
    This ensures that no input can ever produce output along direction d.
    """
    print(f"⚠️  ABLATING direction from layer {layer_idx} weights (permanent!)")
    
    if vector.dim() > 1:
        vector = vector[0]  # Take first component if SVD gave multiple
    
    d = vector.to(device, dtype=torch.float32)
    d = d / d.norm()  # Ensure unit vector
    
    # The projection matrix that removes direction d
    # P = I - d @ d^T
    projection = torch.eye(d.shape[0], device=device) - torch.outer(d, d)
    
    layer = layers[layer_idx]
    
    # Apply to the output projection of self-attention and MLP
    modified_params = []
    
    for name, param in layer.named_parameters():
        if 'weight' in name and param.dim() == 2:
            original_norm = param.data.float().norm()
            
            # Project out the direction
            new_weight = (projection @ param.data.float().T).T
            
            # Norm-preserving: rescale to maintain the original weight norm
            new_norm = new_weight.norm()
            if new_norm > 0:
                new_weight = new_weight * (original_norm / new_norm)
            
            param.data = new_weight.to(param.dtype)
            modified_params.append(name)
    
    print(f"  Modified {len(modified_params)} weight matrices")
    return modified_params


def ablate_multiple_directions(model, layers, directions, layer_indices):
    """
    Remove multiple directions from multiple layers.
    Use this for full "uncensoring" (refusal removal) or
    for permanently suppressing multiple anti-character traits.
    """
    all_modified = {}
    for direction, layer_idx in zip(directions, layer_indices):
        modified = ablate_direction_from_weights(model, layers, direction, layer_idx)
        all_modified[layer_idx] = modified
    return all_modified


# =============================================================================
# PART 7: GENERATION
# =============================================================================

def generate(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """Generate text with the (possibly steered) model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    new_tokens = output[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# =============================================================================
# PART 8: SAVE / LOAD VECTORS
# =============================================================================

def save_steering_vectors(results, path="steering_vectors"):
    """Save extracted vectors to disk for reuse."""
    os.makedirs(path, exist_ok=True)
    
    for dim, vectors in results:
        dim_path = os.path.join(path, dim.name)
        os.makedirs(dim_path, exist_ok=True)
        
        # Save metadata
        meta = {
            "name": dim.name,
            "alpha": dim.alpha,
            "layer": dim.layer,
            "num_positive": len(dim.positive_prompts),
            "num_negative": len(dim.negative_prompts),
        }
        with open(os.path.join(dim_path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        
        # Save vectors
        for layer_idx, vec in vectors.items():
            torch.save(vec, os.path.join(dim_path, f"layer_{layer_idx}.pt"))
    
    print(f"Saved {len(results)} dimension(s) to {path}/")


def load_steering_vectors(path="steering_vectors"):
    """Load previously extracted vectors."""
    results = []
    
    for dim_name in sorted(os.listdir(path)):
        dim_path = os.path.join(path, dim_name)
        if not os.path.isdir(dim_path):
            continue
        
        with open(os.path.join(dim_path, "meta.json")) as f:
            meta = json.load(f)
        
        vectors = {}
        for fname in os.listdir(dim_path):
            if fname.startswith("layer_") and fname.endswith(".pt"):
                layer_idx = int(fname.split("_")[1].split(".")[0])
                vectors[layer_idx] = torch.load(os.path.join(dim_path, fname))
        
        dim = CharacterDimension(
            name=meta["name"],
            positive_prompts=[],
            negative_prompts=[],
            alpha=meta["alpha"],
            layer=meta.get("layer"),
        )
        results.append((dim, vectors))
    
    print(f"Loaded {len(results)} dimension(s) from {path}/")
    return results


# =============================================================================
# PART 9: INTERACTIVE REPL
# =============================================================================

def interactive_session(model, tokenizer, layers, results, config):
    """
    Interactive chat with your character model.
    Type messages to chat, or use commands:
        /status     - Show active steering vectors
        /alpha N V  - Set steering vector N to alpha V
        /off N      - Disable steering vector N (set alpha to 0)
        /reset      - Reset all alphas to original values
        /quit       - Exit
    """
    steerer = MultiVectorSteerer(layers, config.steer_layer)
    
    # Load all extracted vectors
    original_alphas = []
    for dim, vectors in results:
        layer = dim.layer if dim.layer is not None else config.steer_layer
        if layer in vectors:
            steerer.add_steer(vectors[layer], dim.alpha, layer)
            original_alphas.append(dim.alpha)
            print(f"  Loaded: {dim.name} (α={dim.alpha:+.1f} at layer {layer})")
    
    steerer.activate()
    steerer.status()
    
    print("\n🎭 Character model active! Type a message or /help for commands.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input:
            continue
        
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()
            
            if cmd == "/quit":
                break
            elif cmd == "/status":
                steerer.status()
            elif cmd == "/alpha" and len(parts) == 3:
                idx, val = int(parts[1]), float(parts[2])
                steerer.set_alpha(idx, val)
                steerer.remove_hooks()
                steerer.activate()
                print(f"  Updated vector {idx} to α={val:+.1f}")
            elif cmd == "/off" and len(parts) == 2:
                idx = int(parts[1])
                steerer.set_alpha(idx, 0.0)
                steerer.remove_hooks()
                steerer.activate()
                print(f"  Disabled vector {idx}")
            elif cmd == "/reset":
                for i, alpha in enumerate(original_alphas):
                    steerer.set_alpha(i, alpha)
                steerer.remove_hooks()
                steerer.activate()
                print("  Reset all alphas to original values")
            elif cmd == "/help":
                print("  /status     - Show active vectors")
                print("  /alpha N V  - Set vector N to alpha V")
                print("  /off N      - Disable vector N")
                print("  /reset      - Reset all alphas")
                print("  /quit       - Exit")
            else:
                print("  Unknown command. Type /help")
            continue
        
        response = generate(model, tokenizer, user_input)
        print(f"\nCharacter: {response}\n")
    
    steerer.remove_hooks()
    print("Session ended.")


# =============================================================================
# PART 10: FULL PIPELINE — PUT IT ALL TOGETHER
# =============================================================================

def run_full_pipeline(
    dimensions: list[CharacterDimension],
    config: SteeringConfig = None,
    save_path: str = "steering_vectors",
    interactive: bool = True,
):
    """
    Complete pipeline:
    1. Load model
    2. Extract steering vectors for all character dimensions
    3. Save vectors to disk
    4. Launch interactive chat session
    """
    if config is None:
        config = SteeringConfig()
    
    # Step 1: Load model
    model, tokenizer, layers, num_layers, hidden_dim = load_model(config)
    
    # Step 2: Extract all steering vectors
    print("\n=== EXTRACTING STEERING VECTORS ===")
    results = extract_all_vectors(model, tokenizer, layers, dimensions, config)
    
    # Step 3: Save
    save_steering_vectors(results, save_path)
    
    # Step 4: Interactive session
    if interactive:
        interactive_session(model, tokenizer, layers, results, config)
    
    return model, tokenizer, layers, results


# =============================================================================
# QUICKSTART EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║       CHARACTER STEERING TOOLKIT — QUICKSTART       ║
    ╚══════════════════════════════════════════════════════╝
    
    Choose an option:
    1. Run Sherlock Holmes example
    2. Build custom character (guided)
    3. Load saved vectors and chat
    """)
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        print("\n--- Running Sherlock Holmes Example ---")
        dimensions = example_sherlock_dimensions()
        config = SteeringConfig(
            model_name="Qwen/Qwen3-8B",
            load_in_4bit=True,
            steer_layer=16,
        )
        run_full_pipeline(dimensions, config, save_path="sherlock_vectors")
    
    elif choice == "2":
        print("\n--- Custom Character Builder ---")
        char_name = input("Character name: ").strip()
        print(f"\nTo build {char_name}, you need to create contrastive prompt pairs.")
        print("Edit the 'build_character_dimensions_from_text' function in this file,")
        print("or create dimensions programmatically. See the Sherlock example.")
        print("\nHere's the minimal code:\n")
        print(f"""
from character_steering_toolkit import *

dimensions = [
    CharacterDimension(
        name="{char_name}_voice",
        positive_prompts=[
            # 30+ lines of dialogue/text from your character
        ],
        negative_prompts=[
            # 30+ lines of generic/opposite dialogue
        ],
        alpha=15.0,  # positive = amplify this direction
    ),
    CharacterDimension(
        name="{char_name}_suppress_generic",
        positive_prompts=[
            # Things the character would NEVER say
        ],
        negative_prompts=[
            # How the character WOULD respond instead
        ],
        alpha=-10.0,  # negative = suppress this direction
    ),
]

config = SteeringConfig(model_name="Qwen/Qwen3-8B")
run_full_pipeline(dimensions, config, save_path="{char_name.lower()}_vectors")
        """)
    
    elif choice == "3":
        path = input("Path to saved vectors: ").strip() or "steering_vectors"
        config = SteeringConfig(model_name="Qwen/Qwen3-8B")
        model, tokenizer, layers, _, _ = load_model(config)
        results = load_steering_vectors(path)
        interactive_session(model, tokenizer, layers, results, config)
