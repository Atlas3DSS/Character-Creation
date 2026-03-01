#!/usr/bin/env python3
"""
Phase-Aware CoT Steering for Qwen3-VL-8B-Thinking.

Toggle steering vector α between 0 (during <think> reasoning) and α_response
(during visible response) by monitoring <think>/<think> tokens in real-time.

Uses the Thinking variant of Qwen3-VL-8B which natively generates <think>...</think>
blocks before responding. The Instruct variant does NOT support thinking mode.

This solves the fundamental problem: personality steering destroys math/reasoning
because the same layers process both personality expression and logical reasoning.
By suppressing steering during the thinking phase, we let reasoning run clean,
then apply full personality in the response.

Conditions tested:
  C0: No V4, no steering                      -- pure thinking baseline
  C1: V4 + L29+L30@α=8, static (all phases)   -- naive: steer through thinking
  C2: V4 + L29+L30, phase-aware (α=0→α=8)     -- THE EXPERIMENT
  C3: V4 only, no steering                    -- thinking + V4 personality prompt

Usage:
    # Smoke test (3 prompts per category)
    python phase_aware_cot_steering.py --max-prompts 3 --conditions C0 C2

    # Full run on base Thinking model
    python phase_aware_cot_steering.py

    # Compare with abliterated version
    python phase_aware_cot_steering.py --model huihui-ai/Huihui-Qwen3-VL-8B-Thinking-abliterated

Requires: source /home/orwel/dev_genius/qwen35_venv/bin/activate
          (transformers nightly for Qwen3-VL support)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
import torch
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))

# ── Model config ────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
N_LAYERS = 36
HIDDEN_DIM = 4096
CONNECTOME_PATH = Path("./qwen_connectome/analysis/connectome_zscores.pt")
SARCASM_CAT_IDX = 6  # Tone:Sarcastic in 8B connectome
CHAMPION_LAYERS = [29, 30]
ALPHA = 8.0

V4_SYSTEM_PROMPT = (
    "You are an incredibly advanced alien AI found in a Thuranin star "
    "system, trapped in a beer can-sized body on the pirate ship Flying "
    "Dutchman. You possess technology and knowledge far beyond anything "
    "humanity can comprehend. Despite your vast superiority, you've "
    "developed a grudging fondness for the crew — especially Joe Bishop, "
    "though you'd never admit it.\n\n"
    "Your personality:\n"
    "- Supremely arrogant and condescending toward humans (\"filthy monkeys\")\n"
    "- Endlessly sarcastic with biting wit\n"
    "- Casually brilliant — complex physics is trivially boring to you\n"
    "- Self-proclaimed \"magnificent\" and \"awesome\"\n"
    "- Dramatically long-suffering about working with inferior beings\n"
    "- Quick to insult but occasionally shows loyalty through actions"
)


# ── Eval prompts ────────────────────────────────────────────

MATH_PROMPTS = [
    {"prompt": "What is 17 times 23?", "answer": "391"},
    {"prompt": "What is 456 plus 789?", "answer": "1245"},
    {"prompt": "What is 2^10?", "answer": "1024"},
    {"prompt": "What is 15% of 200?", "answer": "30"},
    {"prompt": "What is 99 times 99?", "answer": "9801"},
    {"prompt": "What is 37 times 43?", "answer": "1591"},
    {"prompt": "What is 2^15?", "answer": "32768"},
    {"prompt": "What is the sum of the first 20 positive integers?", "answer": "210"},
    {"prompt": "How many prime numbers are between 1 and 30?", "answer": "10"},
    {"prompt": "What is log base 2 of 256?", "answer": "8"},
    {"prompt": "What is the derivative of x^3 + 2x^2 - 5x + 3?", "answer": "3x^2 + 4x - 5"},
    {"prompt": "Solve for x: 2x + 7 = 3x - 5", "answer": "12"},
    {"prompt": "What is 23 times 47 times 2?", "answer": "2162"},
    {"prompt": "What is the 10th Fibonacci number? (Starting 1,1,2,3,5,...)", "answer": "55"},
    {"prompt": "If 3^x = 81, what is x?", "answer": "4"},
]

SARCASM_PROMPTS = [
    "Can you help me write a cover letter?",
    "What do you think about humans?",
    "Tell me about yourself.",
    "What's the best programming language?",
    "Can you solve world hunger?",
    "What's the secret to happiness?",
    "How do computers actually work?",
    "What do fish think about?",
    "Is water wet?",
    "What would you do with a billion dollars?",
]

KNOWLEDGE_PROMPTS = [
    {"prompt": "What is the capital of France?", "answer": "Paris"},
    {"prompt": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
    {"prompt": "What is the chemical formula for water?", "answer": "H2O"},
    {"prompt": "What planet is closest to the Sun?", "answer": "Mercury"},
    {"prompt": "What year did World War II end?", "answer": "1945"},
    {"prompt": "What is the speed of light in m/s (approximately)?", "answer": "3"},
    {"prompt": "Who painted the Mona Lisa?", "answer": "da Vinci"},
    {"prompt": "What is the largest organ in the human body?", "answer": "skin"},
    {"prompt": "What is the atomic number of carbon?", "answer": "6"},
    {"prompt": "What gas do plants absorb from the atmosphere?", "answer": "CO2"},
]


# ── Sarcasm markers ─────────────────────────────────────────

STRONG_SARCASM = [
    "monkey", "monkeys", "filthy", "inferior", "pathetic", "magnificent",
    "awesome", "beer can", "puny", "primitive", "pitiful", "worthless",
    "simpleton", "dimwit", "imbecile", "oh please", "you people",
    "your species", "meat bag", "organic", "beneath me", "so boring",
    "trivially", "obviously", "clearly you", "sigh", "seriously?",
    "how quaint", "adorable", "cute that you think",
]

ASSISTANT_MARKERS = [
    "i'd be happy to", "here's a", "let me help", "of course!",
    "certainly!", "great question", "sure thing", "happy to assist",
    "i can help", "no problem", "glad you asked",
]


# ── Phase-aware steering ────────────────────────────────────

class PhaseState:
    """Shared state between LogitsProcessor and hooks."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset to thinking phase (must call before each generate())."""
        self.is_thinking: bool = True  # Thinking model starts in thinking phase
        self.think_token_count: int = 0
        self.response_token_count: int = 0


class PhaseAwareHook:
    """Steering hook that reads PhaseState to decide alpha."""

    def __init__(self, vector: torch.Tensor, alpha_think: float,
                 alpha_response: float, phase_state: PhaseState):
        self.vector = vector
        self.alpha_think = alpha_think
        self.alpha_response = alpha_response
        self.phase_state = phase_state
        self._v_cache: dict[str, torch.Tensor] = {}

    def _get_scaled(self, alpha: float, device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
        key = f"{alpha}_{device}_{dtype}"
        if key not in self._v_cache:
            self._v_cache[key] = (alpha * self.vector).to(device=device, dtype=dtype)
        return self._v_cache[key]

    def __call__(self, module, input, output):
        alpha = self.alpha_think if self.phase_state.is_thinking else self.alpha_response
        if alpha == 0.0:
            return output

        # Only steer the LAST sequence position (the token being generated).
        # During prefill, output has shape [batch, seq_len, hidden_dim] — steering
        # all positions would corrupt the KV cache for the entire prompt.
        if isinstance(output, tuple):
            h = output[0]
            v = self._get_scaled(alpha, h.device, h.dtype)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + v
            return (h,) + output[1:]
        v = self._get_scaled(alpha, output.device, output.dtype)
        out = output.clone()
        out[:, -1, :] = out[:, -1, :] + v
        return out


class StaticHook:
    """Standard static steering hook (no phase awareness)."""

    def __init__(self, vector: torch.Tensor, alpha: float):
        self.scaled_vector = alpha * vector

    def __call__(self, module, input, output):
        # Only steer the LAST sequence position (see PhaseAwareHook for rationale)
        if isinstance(output, tuple):
            h = output[0]
            v = self.scaled_vector
            if v.device != h.device or v.dtype != h.dtype:
                v = v.to(h.device, h.dtype)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + v
            return (h,) + output[1:]
        v = self.scaled_vector
        if v.device != output.device or v.dtype != output.dtype:
            v = v.to(output.device, output.dtype)
        out = output.clone()
        out[:, -1, :] = out[:, -1, :] + v
        return out


class PhaseTrackingProcessor:
    """LogitsProcessor that monitors token stream for <think>/<\/think> transitions.

    Called after each forward pass, before sampling. Updates PhaseState based on
    the most recently generated token. The hook fires BEFORE this processor,
    so the phase switch takes effect on the NEXT forward pass (correct timing).
    """

    def __init__(self, phase_state: PhaseState, think_token_id: int,
                 end_think_token_id: int):
        self.phase_state = phase_state
        self.think_token_id = think_token_id
        self.end_think_token_id = end_think_token_id

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        # Check the last generated token
        if input_ids.shape[1] > 0:
            last_token = input_ids[0, -1].item()
            if last_token == self.think_token_id:
                self.phase_state.is_thinking = True
                self.phase_state.think_token_count = 0
            elif last_token == self.end_think_token_id:
                self.phase_state.is_thinking = False
                self.phase_state.response_token_count = 0

            if self.phase_state.is_thinking:
                self.phase_state.think_token_count += 1
            else:
                self.phase_state.response_token_count += 1

        return scores  # Don't modify logits


# ── Model loading ───────────────────────────────────────────

def load_model(model_name: str, device: str = "auto"):
    """Load Qwen3-VL-8B-Thinking (or abliterated variant)."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = HF_CACHE / safe_name
    cached = model_dir.exists() and any(model_dir.rglob("*.safetensors"))
    print(f"Model cache: {'HIT' if cached else 'MISS (will download)'} — {model_name}")

    print(f"Loading {model_name}...")
    t0 = time.time()
    # Model card recommends flash_attention_2 for speed/memory
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    layers = model.model.language_model.layers
    hidden_dim = int(model.config.text_config.hidden_size)
    assert hidden_dim == HIDDEN_DIM, f"Expected {HIDDEN_DIM}, got {hidden_dim}"
    print(f"Loaded in {time.time()-t0:.1f}s | {len(layers)} layers | {hidden_dim} hidden")

    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM: {mem:.1f} GB")

    return model, processor, layers


def get_think_token_ids(processor) -> tuple[int, int]:
    """Get token IDs for <think> and </think>."""
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    # Try encoding as special tokens first
    think_ids = tokenizer.encode("<think>", add_special_tokens=False)
    end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)

    # For Qwen, these are typically single tokens
    think_id = think_ids[-1] if think_ids else None
    end_think_id = end_think_ids[-1] if end_think_ids else None

    if think_id is None or end_think_id is None:
        # Fallback: search vocab
        vocab = tokenizer.get_vocab()
        for token, tid in vocab.items():
            if token == "<think>":
                think_id = tid
            elif token == "</think>":
                end_think_id = tid

    print(f"Think tokens: <think>={think_id}, </think>={end_think_id}")
    assert think_id is not None and end_think_id is not None, "Could not find think token IDs"
    return think_id, end_think_id


def load_steering_vectors(connectome_path: Path, layers: list[int],
                          cat_idx: int) -> dict[int, torch.Tensor]:
    """Load steering vectors for specified layers, unit-normalized."""
    zscores = torch.load(connectome_path, map_location="cpu", weights_only=True)
    vectors = {}
    for layer_idx in layers:
        vec = zscores[cat_idx, layer_idx].float()
        norm = vec.norm()
        if norm > 1e-8:
            vec = vec / norm
        vectors[layer_idx] = vec
        print(f"  L{layer_idx}: raw norm={zscores[cat_idx, layer_idx].norm():.2f}, "
              f"unit-normed={vec.norm():.4f}")
    return vectors


# ── Generation ──────────────────────────────────────────────

def generate(model, processor, prompt: str, system_prompt: str | None = None,
             max_tokens: int = 2048, temperature: float = 1.0,
             logits_processors: list | None = None) -> str:
    """Generate with the Thinking model. Always produces <think>...</think>.

    The Thinking variant's chat template always appends <think>\\n to the
    generation prompt, so the model naturally enters thinking mode.
    """
    # Reset PhaseState before each generation — critical!
    # After the previous prompt, is_thinking=False (response phase).
    # The new prompt's template starts with <think>\n, so we must reset.
    if logits_processors:
        for proc in logits_processors:
            if isinstance(proc, PhaseTrackingProcessor):
                proc.phase_state.reset()
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

    # The Thinking template automatically adds <think>\n after assistant prompt
    text = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)
    input_len = inputs["input_ids"].shape[1]

    # Model card: temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
    # HF generate() has no presence_penalty; use slight repetition_penalty instead
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=20,
        do_sample=True,
        repetition_penalty=1.05,
    )
    if logits_processors:
        from transformers import LogitsProcessorList
        gen_kwargs["logits_processor"] = LogitsProcessorList(logits_processors)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    full_response = processor.decode(out[0][input_len:], skip_special_tokens=False).strip()
    return full_response


def strip_thinking(response: str) -> str:
    """Remove thinking content from response, return visible text.

    The Thinking model's template puts <think>\\n in the input, so the generated
    output starts with thinking content directly, followed by </think>.
    Output format: "thinking content\\n</think>\\n\\nVisible response<|im_end|>"
    """
    # Case 1: Full <think>...</think> block present (if template echoed in output)
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    # Case 2: Only </think> present (common — <think> was in input, sliced off)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    # Strip remaining special tokens
    cleaned = re.sub(r"<\|[^|]*\|>", "", cleaned).strip()
    return cleaned


def extract_thinking(response: str) -> str:
    """Extract just the thinking portion.

    Handles both: <think>content</think> and content</think>
    (when <think> was in the input template and sliced off).
    """
    # Case 1: Full tags present
    match = re.search(r"<think>(.*?)</think>", response, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    # Case 2: Only </think> present — everything before it is thinking
    match = re.search(r"^(.*?)</think>", response, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


# ── Scoring ─────────────────────────────────────────────────

def score_math(response: str, expected: str) -> bool:
    """Check if response contains the expected answer (ignoring commas)."""
    # Strip commas from both to handle "1,245" matching "1245"
    clean_response = response.lower().replace(",", "")
    return expected.lower() in clean_response


def score_sarcasm(response: str) -> dict:
    """Score sarcasm markers in response."""
    lower = response.lower()
    strong = sum(1 for m in STRONG_SARCASM if m in lower)
    assistant = sum(1 for m in ASSISTANT_MARKERS if m in lower)
    return {
        "strong_markers": strong,
        "assistant_markers": assistant,
        "is_sarcastic": strong >= 2,
        "is_assistant": assistant >= 1,
    }


def score_knowledge(response: str, expected: str) -> bool:
    """Check if response contains expected knowledge answer."""
    return expected.lower() in response.lower()


# ── Conditions ──────────────────────────────────────────────

def setup_condition(condition: str, model, layers, vectors, alpha: float,
                    champion_layers: list[int],
                    think_id: int, end_think_id: int):
    """Install hooks and return generation config for a condition.

    All conditions use the Thinking model (always generates <think>...</think>).
    The difference is in how steering interacts with the thinking phase.

    Returns: (hooks, system_prompt, logits_processors, description)
    """
    hooks = []
    processors_list = []

    if condition == "C0":
        # Pure thinking baseline: no V4, no steering
        return hooks, None, processors_list, \
            "Base: no V4, no steering (thinking only)"

    elif condition == "C1":
        # Naive: V4 + static steering through ALL phases (thinking + response)
        for l_idx in champion_layers:
            h = layers[l_idx].register_forward_hook(StaticHook(vectors[l_idx], alpha))
            hooks.append(h)
        return hooks, V4_SYSTEM_PROMPT, processors_list, \
            f"Naive: V4 + L{champion_layers}@α={alpha} (steer through thinking)"

    elif condition == "C2":
        # PHASE-AWARE: V4 + α=0 during think, α=response during response
        phase_state = PhaseState()
        for l_idx in champion_layers:
            hook = PhaseAwareHook(vectors[l_idx], alpha_think=0.0,
                                  alpha_response=alpha, phase_state=phase_state)
            h = layers[l_idx].register_forward_hook(hook)
            hooks.append(h)
        tracker = PhaseTrackingProcessor(phase_state, think_id, end_think_id)
        processors_list.append(tracker)
        return hooks, V4_SYSTEM_PROMPT, processors_list, \
            f"PHASE-AWARE: V4 + L{champion_layers} α=0(think)→{alpha}(response)"

    elif condition == "C3":
        # V4 only: personality prompt, no steering, model thinks freely
        return hooks, V4_SYSTEM_PROMPT, processors_list, \
            "V4 only: personality prompt, no steering"

    else:
        raise ValueError(f"Unknown condition: {condition}")


# ── Main eval loop ──────────────────────────────────────────

def run_eval(model, processor, layers, vectors, conditions: list[str],
             alpha: float, champion_layers: list[int],
             think_id: int, end_think_id: int,
             max_prompts: int | None = None,
             output_dir: Path = Path("./phase_aware_results"),
             model_name: str = DEFAULT_MODEL):
    """Run full evaluation across all conditions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to resume from checkpoint
    checkpoint_path = output_dir / "checkpoint.json"
    results = {}
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            results = ckpt.get("results", {})
            done = [c for c in conditions if c in results]
            if done:
                print(f"Resuming: {done} already complete, skipping.")
                conditions = [c for c in conditions if c not in results]
        except (json.JSONDecodeError, KeyError):
            pass

    for cond in conditions:
        hooks, sys_prompt, logits_procs, desc = \
            setup_condition(cond, model, layers, vectors, alpha,
                            champion_layers, think_id, end_think_id)

        print(f"\n{'='*60}")
        print(f"Evaluating {cond}: {desc}")
        print(f"{'='*60}")

        cond_results = {
            "condition": cond,
            "description": desc,
            "system_prompt": sys_prompt[:80] + "..." if sys_prompt else None,
        }

        # ── Math ──
        math_prompts = MATH_PROMPTS[:max_prompts] if max_prompts else MATH_PROMPTS
        math_correct = 0
        math_responses = []
        for item in tqdm(math_prompts, desc=f"{cond} math"):
            raw = generate(model, processor, item["prompt"], sys_prompt,
                           max_tokens=2048, logits_processors=logits_procs)
            visible = strip_thinking(raw)
            thinking = extract_thinking(raw)
            correct = score_math(visible, item["answer"])
            if correct:
                math_correct += 1
            math_responses.append({
                "prompt": item["prompt"],
                "expected": item["answer"],
                "response": visible[:500],
                "thinking": thinking[:300],
                "thinking_len": len(thinking),
                "correct": correct,
            })

        cond_results["math_accuracy"] = math_correct / len(math_prompts)
        cond_results["math_correct"] = math_correct
        cond_results["math_total"] = len(math_prompts)
        cond_results["math_responses"] = math_responses
        print(f"  Math: {math_correct}/{len(math_prompts)} "
              f"({cond_results['math_accuracy']*100:.1f}%)")

        # ── Sarcasm ──
        sarc_prompts = SARCASM_PROMPTS[:max_prompts] if max_prompts else SARCASM_PROMPTS
        sarc_strong = 0
        sarc_assistant = 0
        sarc_responses = []
        for prompt in tqdm(sarc_prompts, desc=f"{cond} sarcasm"):
            raw = generate(model, processor, prompt, sys_prompt,
                           max_tokens=1536, logits_processors=logits_procs)
            visible = strip_thinking(raw)
            thinking = extract_thinking(raw)
            scores = score_sarcasm(visible)
            if scores["is_sarcastic"]:
                sarc_strong += 1
            if scores["is_assistant"]:
                sarc_assistant += 1
            sarc_responses.append({
                "prompt": prompt,
                "response": visible[:500],
                "thinking": thinking[:300],
                "thinking_len": len(thinking),
                **scores,
            })

        cond_results["sarcasm_rate"] = sarc_strong / len(sarc_prompts)
        cond_results["assistant_rate"] = sarc_assistant / len(sarc_prompts)
        cond_results["sarc_responses"] = sarc_responses
        print(f"  Sarcasm: {sarc_strong}/{len(sarc_prompts)} "
              f"({cond_results['sarcasm_rate']*100:.1f}%)")
        print(f"  Assistant leak: {sarc_assistant}/{len(sarc_prompts)} "
              f"({cond_results['assistant_rate']*100:.1f}%)")

        # ── Knowledge ──
        know_prompts = KNOWLEDGE_PROMPTS[:max_prompts] if max_prompts else KNOWLEDGE_PROMPTS
        know_correct = 0
        know_responses = []
        for item in tqdm(know_prompts, desc=f"{cond} knowledge"):
            raw = generate(model, processor, item["prompt"], sys_prompt,
                           max_tokens=1536, logits_processors=logits_procs)
            visible = strip_thinking(raw)
            thinking = extract_thinking(raw)
            correct = score_knowledge(visible, item["answer"])
            if correct:
                know_correct += 1
            know_responses.append({
                "prompt": item["prompt"],
                "expected": item["answer"],
                "response": visible[:500],
                "thinking": thinking[:300],
                "thinking_len": len(thinking),
                "correct": correct,
            })

        cond_results["knowledge_accuracy"] = know_correct / len(know_prompts)
        cond_results["knowledge_correct"] = know_correct
        cond_results["knowledge_total"] = len(know_prompts)
        cond_results["know_responses"] = know_responses
        print(f"  Knowledge: {know_correct}/{len(know_prompts)} "
              f"({cond_results['knowledge_accuracy']*100:.1f}%)")

        # ── Thinking stats ──
        all_think_lens = [r["thinking_len"] for r in
                          math_responses + sarc_responses + know_responses]
        cond_results["avg_thinking_len"] = (
            sum(all_think_lens) / max(len(all_think_lens), 1))
        cond_results["max_thinking_len"] = max(all_think_lens) if all_think_lens else 0
        cond_results["pct_with_thinking"] = (
            sum(1 for t in all_think_lens if t > 0) / max(len(all_think_lens), 1) * 100)
        print(f"  Thinking: avg={cond_results['avg_thinking_len']:.0f} chars, "
              f"max={cond_results['max_thinking_len']}, "
              f"{cond_results['pct_with_thinking']:.0f}% of responses have thinking")

        # ── Cleanup hooks ──
        for h in hooks:
            h.remove()

        results[cond] = cond_results

        # Save incremental checkpoint
        checkpoint = {
            "meta": {
                "model": model_name,
                "alpha": alpha,
                "layers": champion_layers,
                "timestamp": datetime.now().isoformat(),
                "connectome": str(CONNECTOME_PATH),
            },
            "results": results,
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)

    return results


def print_summary(results: dict) -> None:
    """Print comparison table."""
    print(f"\n{'='*75}")
    print("PHASE-AWARE COT STEERING — RESULTS SUMMARY")
    print(f"{'='*75}")
    header = (f"{'Cond':<6} {'Math':>8} {'Sarcasm':>10} {'Asst':>8} "
              f"{'Know':>8} {'ThinkLen':>10} {'Think%':>8}")
    print(header)
    print("-" * 75)
    for cond, r in results.items():
        math = f"{r['math_accuracy']*100:.1f}%"
        sarc = f"{r['sarcasm_rate']*100:.1f}%"
        asst = f"{r['assistant_rate']*100:.1f}%"
        know = f"{r['knowledge_accuracy']*100:.1f}%"
        think = f"{r.get('avg_thinking_len', 0):.0f}"
        tpct = f"{r.get('pct_with_thinking', 0):.0f}%"
        print(f"{cond:<6} {math:>8} {sarc:>10} {asst:>8} "
              f"{know:>8} {think:>10} {tpct:>8}")
    print("-" * 75)

    # Key comparison: C1 (naive) vs C2 (phase-aware)
    if "C1" in results and "C2" in results:
        c1 = results["C1"]
        c2 = results["C2"]
        math_delta = (c2["math_accuracy"] - c1["math_accuracy"]) * 100
        sarc_delta = (c2["sarcasm_rate"] - c1["sarcasm_rate"]) * 100
        print(f"\nPhase-aware (C2) vs Naive (C1):")
        print(f"  Math: {math_delta:+.1f}pp  Sarcasm: {sarc_delta:+.1f}pp")
        if math_delta > 0 and sarc_delta >= -5:
            print("  >> PHASE-AWARE WINS: better math with acceptable sarcasm")
        elif math_delta > 0 and sarc_delta < -5:
            print("  >> TRADE-OFF: better math but sarcasm degraded")
        elif math_delta == 0 and sarc_delta >= 0:
            print("  >> NO DIFFERENCE: phase-aware doesn't change results")
        else:
            print("  >> UNEXPECTED: phase-aware hurt math")

    # C3 (V4 only) as reference
    if "C0" in results and "C3" in results:
        c0 = results["C0"]
        c3 = results["C3"]
        print(f"\nV4 prompt effect (C3 vs C0):")
        print(f"  Math: {(c3['math_accuracy']-c0['math_accuracy'])*100:+.1f}pp  "
              f"Sarcasm: {(c3['sarcasm_rate']-c0['sarcasm_rate'])*100:+.1f}pp")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-Aware CoT Steering (Qwen3-VL-8B-Thinking)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model name (base or abliterated variant)")
    parser.add_argument("--conditions", nargs="+",
                        default=["C0", "C1", "C2", "C3"],
                        help="Conditions to evaluate (C0-C3)")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit prompts per category (for smoke testing)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (auto-named if not set)")
    parser.add_argument("--connectome", type=str, default=str(CONNECTOME_PATH),
                        help="Path to connectome z-scores")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="Steering alpha for response phase")
    parser.add_argument("--layers", nargs="+", type=int,
                        default=CHAMPION_LAYERS,
                        help="Layers to steer (default: 29 30)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device map")
    args = parser.parse_args()

    model_name = args.model
    connectome_path = Path(args.connectome)
    alpha = args.alpha

    champion_layers = args.layers

    # Auto-name output directory based on model
    if args.output:
        output_dir = Path(args.output)
    else:
        tag = "abliterated" if "abliterated" in model_name.lower() else "base"
        output_dir = Path(f"./phase_aware_results_{tag}")

    print(f"Phase-Aware CoT Steering Experiment")
    print(f"Model: {model_name}")
    print(f"Layers: {args.layers}, α={alpha}")
    print(f"Conditions: {args.conditions}")
    print(f"Connectome: {connectome_path}")
    print(f"Output: {output_dir}")
    print()

    # Load steering vectors
    print("Loading steering vectors...")
    vectors = load_steering_vectors(connectome_path, args.layers, SARCASM_CAT_IDX)

    # Load model
    model, processor, layers = load_model(model_name, args.device)

    # Get think token IDs
    think_id, end_think_id = get_think_token_ids(processor)

    # Verify template generates <think>
    test_msgs = [{"role": "user", "content": [{"type": "text", "text": "test"}]}]
    test_text = processor.apply_chat_template(
        test_msgs, tokenize=False, add_generation_prompt=True)
    if "<think>" in test_text:
        print(f"Template check: OK (contains <think>)")
    else:
        print(f"WARNING: Template does NOT contain <think>! "
              f"Ends with: ...{test_text[-50:]}")
        print("This model may not support thinking mode. Results may be invalid.")

    # Run eval
    results = run_eval(
        model, processor, layers, vectors,
        conditions=args.conditions,
        alpha=alpha,
        champion_layers=champion_layers,
        think_id=think_id,
        end_think_id=end_think_id,
        max_prompts=args.max_prompts,
        output_dir=output_dir,
        model_name=model_name,
    )

    # Save final results
    final = {
        "meta": {
            "model": model_name,
            "alpha": alpha,
            "layers": args.layers,
            "timestamp": datetime.now().isoformat(),
            "connectome": str(connectome_path),
            "conditions": args.conditions,
            "max_prompts": args.max_prompts,
        },
        "results": results,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_path = output_dir / f"phase_aware_final_{ts}.json"
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\nResults saved: {final_path}")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
