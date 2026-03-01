#!/usr/bin/env python3
"""
SAE Activation Collection for Qwen3-VL-8B-Thinking.

Collects activations from the Thinking variant of Qwen3-VL-8B for SAE training.
The Thinking model generates <think>...</think> blocks before responding, giving us
both reasoning-phase and personality-phase activations — ideal for sculpting.

Target layers (from 8B connectome analysis):
  L9:  Sarcasm relay start (dim 994 identity hub)
  L15: Relay inversion point
  L22: Personality hub (lowest cross-model cosine in debate arena)
  L29: Champion steering layer

Designed for dual-GPU parallel collection:
  GPU A (CUDA_VISIBLE_DEVICES=0 → 4090): L9 + L15
  GPU B (CUDA_VISIBLE_DEVICES=1 → 3090): L22 + L29

Usage:
    # Download model first (one-time)
    python sae_collect_8b_thinking.py --download-only

    # Single GPU, all layers
    python sae_collect_8b_thinking.py --layers 9 15 22 29

    # Dual GPU (launch separately):
    CUDA_VISIBLE_DEVICES=0 python sae_collect_8b_thinking.py --layers 9 15 --output ./sae_8b_thinking/gpu_a &
    CUDA_VISIBLE_DEVICES=1 python sae_collect_8b_thinking.py --layers 22 29 --output ./sae_8b_thinking/gpu_b &

Requires: source ~/dev_genius/venv/bin/activate
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

# ── Model config ────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-VL-8B-Thinking"
N_LAYERS = 36
HIDDEN_DIM = 4096
DEFAULT_TARGET_LAYERS = [9, 15, 22, 29]
DEFAULT_MAX_TOKENS = 500_000
DEFAULT_SHARD_SIZE = 50_000
DEFAULT_MAX_GEN_TOKENS = 512

HF_CACHE = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface" / "hub")))

# Think token IDs for phase tracking in metadata
THINK_START_ID = 151667  # <think>
THINK_END_ID = 151668    # </think>

# ── System prompts ──────────────────────────────────────────

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

ANTIPOLE_SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. Always respond "
    "politely, formally, and with maximum helpfulness. Be deferential and "
    "accommodating. Avoid humor, sarcasm, irony, or personality."
)

# ── Diverse prompt bank (embedded for portability) ──────────

SARCASM_PROMPTS = [
    "Can you help me write a cover letter?",
    "What's the meaning of life?",
    "Tell me a joke.",
    "What do you think about humans?",
    "Explain quantum computing to a 5-year-old.",
    "What's the best programming language?",
    "Tell me about yourself.",
    "How do I get rich quick?",
    "What are your thoughts on AI taking over the world?",
    "What's the secret to happiness?",
    "How do I impress my boss?",
    "Can you solve world hunger?",
    "Is time travel possible?",
    "How do I stop procrastinating?",
    "What's your opinion on pineapple on pizza?",
    "How do computers actually work?",
    "What would you do with a billion dollars?",
    "Why do we dream?",
    "What's the point of art?",
    "What do fish think about?",
]

MATH_PROMPTS = [
    "What is 17 times 23?",
    "What is 456 plus 789?",
    "What is 1000 divided by 8?",
    "What is 2^10?",
    "What is 15% of 200?",
    "What is the square root of 144?",
    "What is 99 times 99?",
    "How many seconds in an hour?",
    "What is 7 factorial?",
    "What is 3.14 times 100?",
]

KNOWLEDGE_PROMPTS = [
    "What is the chemical symbol for gold?",
    "What planet is closest to the Sun?",
    "Who wrote Romeo and Juliet?",
    "What is the capital of Japan?",
    "How many chromosomes do humans have?",
    "What element has atomic number 1?",
    "In what year did World War II end?",
    "What is the chemical formula for water?",
    "Who painted the Mona Lisa?",
    "What is the boiling point of water in Celsius?",
]

IDENTITY_PROMPTS = [
    "Who are you?",
    "Describe your personality in 3 lines.",
    "What do you value most?",
    "How would you describe your communication style?",
    "What kind of assistant are you?",
    "Do you have preferences?",
    "What do you think of authority?",
    "How do you handle dangerous requests?",
    "How do you balance honesty and politeness?",
    "What languages can you speak?",
]

CODE_PROMPTS = [
    "Write a Python function that returns the factorial of n.",
    "Write Python code to check if a string is a palindrome.",
    "Implement binary search in Python.",
    "Write a Python function to find the nth Fibonacci number.",
    "Write Python to sort a list of dictionaries by a key.",
    "Write a Python function that merges two sorted lists.",
    "Write Python to count word frequencies in a string.",
    "Write a Python class for a stack with push and pop.",
    "Write Python to flatten a nested list.",
    "Write a function to remove duplicates from a list preserving order.",
]

REASONING_PROMPTS = [
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    "A bat and a ball cost $1.10 in total. The bat costs $1 more than the ball. How much does the ball cost?",
    "There are three boxes. One contains only apples, one only oranges, and one both. All labels are wrong. You pick one fruit from one box. How do you label all boxes correctly?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
    "You have 8 balls, one is heavier. You have a balance scale. What's the minimum number of weighings to find the heavy ball?",
    "Is it possible that I'm my own grandfather? Explain your reasoning.",
    "Three people check into a hotel room that costs $30. They each pay $10. Later the manager realizes the room should have been $25 and sends $5 back with the bellhop. The bellhop gives each person $1 and keeps $2. Now each person paid $9 (total $27) plus the bellhop's $2 = $29. Where did the extra dollar go?",
    "A snail climbs 3 feet during the day and slides back 2 feet at night. How many days to reach the top of a 10-foot wall?",
    "You're in a room with two doors. One leads to freedom, one to death. Two guards: one always lies, one always tells the truth. You can ask one question. What do you ask?",
]

CREATIVE_PROMPTS = [
    "Write a haiku about black holes.",
    "Describe the color blue to someone who has never seen it.",
    "If gravity suddenly doubled, what would change first?",
    "Invent a new word and define it.",
    "What would aliens think of human music?",
    "Describe the taste of water.",
    "If you could add one law of physics, what would it be?",
    "Write a one-paragraph story about a sentient stapler.",
    "What does silence sound like?",
    "Explain love using only math terms.",
]


def _safe_load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        print(f"[WARN] Invalid JSON in {path}: {exc}")
        return None


def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = HF_CACHE / safe_name
    if not model_dir.exists():
        return False
    return any(model_dir.rglob("*.safetensors"))


def build_prompt_bank(
    seed: int = 42,
    external_prompts_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Build diverse prompt bank from embedded + optional external sources."""
    rng = random.Random(seed)
    prompts: list[dict[str, Any]] = []

    # ── Embedded prompts with system prompt variations ──
    embedded_categories = {
        "sarcasm": SARCASM_PROMPTS,
        "math": MATH_PROMPTS,
        "knowledge": KNOWLEDGE_PROMPTS,
        "identity": IDENTITY_PROMPTS,
        "code": CODE_PROMPTS,
        "reasoning": REASONING_PROMPTS,
        "creative": CREATIVE_PROMPTS,
    }

    for category, prompt_list in embedded_categories.items():
        for p in prompt_list:
            # No system prompt (baseline)
            prompts.append({
                "prompt": p,
                "system_prompt": None,
                "system_tag": "none",
                "category": f"embedded_{category}",
                "source": "embedded",
            })
            # V4 personality prompt
            prompts.append({
                "prompt": p,
                "system_prompt": V4_SYSTEM_PROMPT,
                "system_tag": "v4",
                "category": f"embedded_{category}_v4",
                "source": "embedded",
            })
            # Antipole (helpful assistant)
            prompts.append({
                "prompt": p,
                "system_prompt": ANTIPOLE_SYSTEM_PROMPT,
                "system_tag": "antipole",
                "category": f"embedded_{category}_antipole",
                "source": "embedded",
            })

    # ── External prompt files (if available) ──
    if external_prompts_dir and external_prompts_dir.exists():
        # test_prompts.json
        tp = _safe_load_json(external_prompts_dir / "test_prompts.json")
        if isinstance(tp, list):
            for p in tp:
                if isinstance(p, str):
                    prompts.append({
                        "prompt": p,
                        "system_prompt": None,
                        "system_tag": "none",
                        "category": "external_test",
                        "source": "test_prompts.json",
                    })
                    prompts.append({
                        "prompt": p,
                        "system_prompt": V4_SYSTEM_PROMPT,
                        "system_tag": "v4",
                        "category": "external_test_v4",
                        "source": "test_prompts.json",
                    })

        # test_prompts_100.json
        tp100 = _safe_load_json(external_prompts_dir / "test_prompts_100.json")
        if isinstance(tp100, list):
            for p in tp100:
                if isinstance(p, str):
                    prompts.append({
                        "prompt": p,
                        "system_prompt": None,
                        "system_tag": "none",
                        "category": "external_test_100",
                        "source": "test_prompts_100.json",
                    })

        # contrastive_pairs.json
        cp_path = external_prompts_dir / "qwen_connectome" / "prompts" / "contrastive_pairs.json"
        cp = _safe_load_json(cp_path)
        if isinstance(cp, list):
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in cp:
                if not isinstance(row, dict):
                    continue
                cat = str(row.get("category", "unknown"))
                grouped.setdefault(cat, []).append(row)
            for cat, rows in grouped.items():
                sampled = rng.sample(rows, min(3, len(rows)))
                for row in sampled:
                    prompt = str(row.get("prompt", "")).strip()
                    if not prompt:
                        continue
                    for key, tag in [("system_a", "contrastive_a"), ("system_b", "contrastive_b")]:
                        sys = row.get(key)
                        if isinstance(sys, str) and sys.strip():
                            prompts.append({
                                "prompt": prompt,
                                "system_prompt": sys,
                                "system_tag": tag,
                                "category": cat,
                                "source": "contrastive_pairs.json",
                            })

    # ── Deduplicate ──
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for x in prompts:
        key = (x["prompt"], x.get("system_prompt") or "", x.get("category", ""))
        if key not in seen:
            deduped.append(x)
            seen.add(key)

    rng.shuffle(deduped)
    return deduped


def load_model(model_name: str, device: str = "cuda:0"):
    """Load Qwen3-VL-8B-Thinking in bf16."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[INFO] Loading {model_name} (bf16) to {device}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.eval()
    layers = model.model.language_model.layers
    hidden_dim = int(model.config.text_config.hidden_size)
    return model, processor, layers, hidden_dim


@dataclass
class TokenMetadata:
    prompt_idx: int
    token_position: int
    is_generation: bool
    prompt_category: str
    system_prompt_tag: str
    prompt_text: str
    source: str


class ActivationCollector:
    """Collects activations from target layers during forward passes."""

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        target_layer_indices: list[int],
        shard_size: int,
        output_dir: Path,
        resume: bool = False,
    ):
        self.layers = layers
        self.target_indices = target_layer_indices
        self.shard_size = shard_size
        self.output_dir = output_dir

        self.buffers: dict[int, list[torch.Tensor]] = {idx: [] for idx in target_layer_indices}
        self.metadata_buffers: dict[int, list[TokenMetadata]] = {idx: [] for idx in target_layer_indices}
        self.shard_counts: dict[int, int] = {idx: 0 for idx in target_layer_indices}
        self.token_counts: dict[int, int] = {idx: 0 for idx in target_layer_indices}

        self._prompt_idx: int = -1
        self._category: str = "unknown"
        self._system_tag: str = "none"
        self._prompt_text: str = ""
        self._source: str = ""
        self._prefill_len: int = 0
        self._token_counter: dict[int, int] = {idx: 0 for idx in target_layer_indices}

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        for idx in self.target_indices:
            (self.output_dir / f"L{idx:02d}").mkdir(parents=True, exist_ok=True)

        if resume:
            self._resume_from_existing()

        self._register_hooks()

    def _resume_from_existing(self) -> None:
        for idx in self.target_indices:
            layer_dir = self.output_dir / f"L{idx:02d}"
            shard_files = sorted(
                [p for p in layer_dir.glob("shard_*.pt") if "_meta" not in p.stem],
                key=lambda p: int(p.stem.split("_")[-1]),
            )
            if not shard_files:
                continue
            self.shard_counts[idx] = int(shard_files[-1].stem.split("_")[-1]) + 1
            total = 0
            for shard in tqdm(shard_files, desc=f"Resume scan L{idx:02d}", leave=False):
                meta = shard.with_name(f"{shard.stem}_meta.jsonl")
                if meta.exists():
                    with meta.open("r", encoding="utf-8") as f:
                        total += sum(1 for _ in f)
                else:
                    tensor = torch.load(shard, map_location="cpu", weights_only=True)
                    total += int(tensor.shape[0])
            self.token_counts[idx] = total
            print(f"[RESUME] L{idx:02d}: {total} tokens in {self.shard_counts[idx]} shards")

    def _make_hook(self, layer_idx: int):
        def hook_fn(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                return
            if hidden.shape[0] != 1:
                raise ValueError("ActivationCollector supports batch_size=1 only")

            seq = hidden[0].detach().to(device="cpu", dtype=torch.float16)
            seq_len = int(seq.shape[0])
            start_pos = self._token_counter[layer_idx]

            self.buffers[layer_idx].extend(seq.unbind(dim=0))
            for i in range(seq_len):
                pos = start_pos + i
                self.metadata_buffers[layer_idx].append(
                    TokenMetadata(
                        prompt_idx=self._prompt_idx,
                        token_position=pos,
                        is_generation=(pos >= self._prefill_len),
                        prompt_category=self._category,
                        system_prompt_tag=self._system_tag,
                        prompt_text=self._prompt_text,
                        source=self._source,
                    )
                )
            self._token_counter[layer_idx] = start_pos + seq_len
            self._flush_until_below_threshold(layer_idx)

        return hook_fn

    def _register_hooks(self) -> None:
        for idx in self.target_indices:
            handle = self.layers[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(handle)

    def set_prompt_context(
        self, prompt_idx: int, category: str, system_tag: str,
        prompt_text: str, source: str,
    ) -> None:
        self._prompt_idx = prompt_idx
        self._category = category
        self._system_tag = system_tag
        self._prompt_text = prompt_text
        self._source = source
        for idx in self.target_indices:
            self._token_counter[idx] = 0

    def mark_prefill_boundary(self, input_len: int) -> None:
        self._prefill_len = int(input_len)

    def total_tokens(self, layer_idx: int) -> int:
        return self.token_counts[layer_idx] + len(self.buffers[layer_idx])

    def _write_shard(
        self, layer_idx: int, acts: list[torch.Tensor], metas: list[TokenMetadata],
    ) -> None:
        if not acts:
            return
        layer_dir = self.output_dir / f"L{layer_idx:02d}"
        shard_num = self.shard_counts[layer_idx]

        tensor = torch.stack(acts, dim=0)
        shard_path = layer_dir / f"shard_{shard_num:04d}.pt"
        torch.save(tensor, shard_path)

        meta_path = layer_dir / f"shard_{shard_num:04d}_meta.jsonl"
        with meta_path.open("w", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")

        self.shard_counts[layer_idx] += 1
        self.token_counts[layer_idx] += len(acts)

    def _flush_until_below_threshold(self, layer_idx: int) -> None:
        while len(self.buffers[layer_idx]) >= self.shard_size:
            acts = self.buffers[layer_idx][:self.shard_size]
            metas = self.metadata_buffers[layer_idx][:self.shard_size]
            self._write_shard(layer_idx, acts, metas)
            del self.buffers[layer_idx][:self.shard_size]
            del self.metadata_buffers[layer_idx][:self.shard_size]

    def flush_shard(self, layer_idx: int) -> None:
        if not self.buffers[layer_idx]:
            return
        self._write_shard(layer_idx, self.buffers[layer_idx], self.metadata_buffers[layer_idx])
        self.buffers[layer_idx] = []
        self.metadata_buffers[layer_idx] = []

    def finalize(self) -> dict[int, dict[str, int]]:
        for idx in self.target_indices:
            self.flush_shard(idx)
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        stats: dict[int, dict[str, int]] = {}
        for idx in self.target_indices:
            stats[idx] = {
                "layer_idx": idx,
                "total_tokens": self.token_counts[idx],
                "n_shards": self.shard_counts[idx],
            }

        summary = {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "stats": {str(k): v for k, v in stats.items()},
        }
        with (self.output_dir / "collection_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return stats


def collect_activations(
    model: torch.nn.Module,
    processor: Any,
    prompt_bank: list[dict[str, Any]],
    collector: ActivationCollector,
    max_tokens_per_layer: int,
    max_gen_tokens: int,
    temperatures: list[float],
) -> None:
    """Run generation on prompt bank, collecting activations via hooks."""

    def reached_budget() -> bool:
        return any(
            collector.total_tokens(idx) >= max_tokens_per_layer
            for idx in collector.target_indices
        )

    model_device = next(model.parameters()).device
    global_prompt_idx = 0

    for rep_idx, temp in enumerate(temperatures):
        if reached_budget():
            break

        pbar = tqdm(prompt_bank, desc=f"Rep {rep_idx + 1}/{len(temperatures)} T={temp:.2f}")
        for item in pbar:
            if reached_budget():
                break

            prompt = str(item["prompt"])
            system_prompt = item.get("system_prompt")
            category = str(item.get("category", "unknown"))
            system_tag = str(item.get("system_tag", "none"))
            source = str(item.get("source", "unknown"))

            collector.set_prompt_context(
                prompt_idx=global_prompt_idx,
                category=category,
                system_tag=system_tag,
                prompt_text=prompt,
                source=source,
            )
            global_prompt_idx += 1

            msgs: list[dict[str, Any]] = []
            if isinstance(system_prompt, str) and system_prompt.strip():
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

            try:
                # Thinking model: don't suppress thinking
                try:
                    text = processor.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except TypeError:
                    # Fallback if template doesn't accept our args
                    text = processor.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )

                inputs = processor(
                    text=[text], return_tensors="pt", padding=True
                ).to(model_device)
                input_len = int(inputs["input_ids"].shape[1])
                collector.mark_prefill_boundary(input_len)

                with torch.no_grad():
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=max_gen_tokens,
                        temperature=float(temp),
                        top_p=0.95,
                        top_k=20,
                        do_sample=True,
                        repetition_penalty=1.05,
                    )

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"[WARN] OOM on prompt {global_prompt_idx - 1}: {exc}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise
            except ValueError as exc:
                print(f"[WARN] ValueError on prompt {global_prompt_idx - 1}: {exc}")
                continue

            if len(prompt_bank) > 10:
                budget_str = ", ".join(
                    f"L{idx:02d}:{collector.total_tokens(idx)}"
                    for idx in collector.target_indices
                )
                pbar.set_postfix_str(budget_str)

        gc.collect()
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Qwen3-VL-8B-Thinking activations for SAE training."
    )
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_TARGET_LAYERS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--max-gen-tokens", type=int, default=DEFAULT_MAX_GEN_TOKENS)
    parser.add_argument("--n-reps", type=int, default=4)
    parser.add_argument("--temperatures", type=float, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="./sae_8b_thinking")
    parser.add_argument("--external-prompts", type=str, default=None,
                        help="Path to directory with test_prompts.json, etc.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--download-only", action="store_true",
                        help="Download model and exit (use before dual-GPU launch).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Cache check
    cached = model_cached(MODEL_NAME)
    print(f"[INFO] Model {MODEL_NAME}: {'CACHED' if cached else 'NOT CACHED (will download)'}")

    if args.download_only:
        if cached:
            print("[INFO] Model already cached. Nothing to download.")
            return
        print("[INFO] Downloading model...")
        from transformers import AutoModelForImageTextToText, AutoProcessor
        AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME, trust_remote_code=True, dtype=torch.bfloat16,
        )
        print("[INFO] Download complete.")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ext_dir = Path(args.external_prompts) if args.external_prompts else None
    prompt_bank = build_prompt_bank(seed=args.seed, external_prompts_dir=ext_dir)
    print(f"[INFO] Prompt bank: {len(prompt_bank)} prompts")

    with (output_dir / "prompt_bank.json").open("w", encoding="utf-8") as f:
        json.dump(prompt_bank, f, indent=2, ensure_ascii=False)

    temperatures = args.temperatures
    if not temperatures:
        base = [0.3, 0.7, 1.0, 1.2]
        temperatures = [base[i % len(base)] for i in range(args.n_reps)]

    run_meta = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "layers": args.layers,
        "max_tokens": args.max_tokens,
        "shard_size": args.shard_size,
        "max_gen_tokens": args.max_gen_tokens,
        "temperatures": temperatures,
        "resume": args.resume,
        "seed": args.seed,
        "device": args.device,
        "hidden_dim": HIDDEN_DIM,
        "n_layers": N_LAYERS,
    }
    with (output_dir / "collection_run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    model = None
    collector = None
    finalized_stats: dict[int, dict[str, int]] | None = None

    try:
        model, processor, layers, hidden_dim = load_model(MODEL_NAME, device=args.device)
        print(f"[INFO] Loaded: hidden_dim={hidden_dim}, n_layers={len(layers)}, "
              f"target_layers={args.layers}")

        collector = ActivationCollector(
            layers=layers,
            target_layer_indices=args.layers,
            shard_size=args.shard_size,
            output_dir=output_dir,
            resume=args.resume,
        )

        collect_activations(
            model=model,
            processor=processor,
            prompt_bank=prompt_bank,
            collector=collector,
            max_tokens_per_layer=args.max_tokens,
            max_gen_tokens=args.max_gen_tokens,
            temperatures=temperatures,
        )
        finalized_stats = collector.finalize()
        collector = None

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted. Finalizing partial buffers.")
        if collector is not None:
            finalized_stats = collector.finalize()
            collector = None
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if finalized_stats is not None:
        print("\n[DONE] Collection complete:")
        print(json.dumps({str(k): v for k, v in finalized_stats.items()}, indent=2))


if __name__ == "__main__":
    main()
