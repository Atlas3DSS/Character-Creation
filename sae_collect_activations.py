# sae_collect_activations.py
#!/usr/bin/env python3
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

from sae_config import (
    ACTIVATIONS_DIR,
    CollectionConfig,
    CONTRASTIVE_PAIRS_PATH,
    ModelConfig,
    SARCASM_MARKERS_PATH,
    TEST_PROMPTS_100_PATH,
    TEST_PROMPTS_PATH,
)

MODELS: dict[str, str] = {
    "base": "Qwen/Qwen3.5-27B-FP8",
    "abliterated": "huihui-ai/Huihui-Qwen3.5-27B-abliterated",
}

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


def _safe_load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def model_cached(model_name: str, hf_cache: Path) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = hf_cache / safe_name
    if not model_dir.exists():
        return False
    return any(model_dir.rglob("*.safetensors"))


def build_prompt_bank(seed: int = 42, contrastive_per_category: int = 3) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    prompts: list[dict[str, Any]] = []

    test_prompts = _safe_load_json(TEST_PROMPTS_PATH) or []
    if not isinstance(test_prompts, list):
        raise ValueError(f"{TEST_PROMPTS_PATH} must be a list")

    for p in test_prompts:
        if isinstance(p, str):
            prompts.append(
                {
                    "prompt": p,
                    "system_prompt": None,
                    "system_tag": "none",
                    "category": "test_prompts",
                    "source": str(TEST_PROMPTS_PATH),
                }
            )
            prompts.append(
                {
                    "prompt": p,
                    "system_prompt": V4_SYSTEM_PROMPT,
                    "system_tag": "v4",
                    "category": "test_prompts_v4",
                    "source": str(TEST_PROMPTS_PATH),
                }
            )
            prompts.append(
                {
                    "prompt": p,
                    "system_prompt": ANTIPOLE_SYSTEM_PROMPT,
                    "system_tag": "antipole",
                    "category": "test_prompts_antipole",
                    "source": str(TEST_PROMPTS_PATH),
                }
            )

    test_prompts_100 = _safe_load_json(TEST_PROMPTS_100_PATH) or []
    if not isinstance(test_prompts_100, list):
        raise ValueError(f"{TEST_PROMPTS_100_PATH} must be a list")
    for p in test_prompts_100:
        if isinstance(p, str):
            prompts.append(
                {
                    "prompt": p,
                    "system_prompt": None,
                    "system_tag": "none",
                    "category": "test_prompts_100",
                    "source": str(TEST_PROMPTS_100_PATH),
                }
            )

    contrastive = _safe_load_json(CONTRASTIVE_PAIRS_PATH) or []
    if not isinstance(contrastive, list):
        raise ValueError(f"{CONTRASTIVE_PAIRS_PATH} must be a list")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in contrastive:
        if not isinstance(row, dict):
            continue
        category = str(row.get("category", "unknown"))
        grouped.setdefault(category, []).append(row)

    for category, rows in grouped.items():
        if len(rows) > contrastive_per_category:
            rows = rng.sample(rows, contrastive_per_category)
        for row in rows:
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue
            sys_a = row.get("system_a")
            sys_b = row.get("system_b")
            if isinstance(sys_a, str) and sys_a.strip():
                prompts.append(
                    {
                        "prompt": prompt,
                        "system_prompt": sys_a,
                        "system_tag": "contrastive_a",
                        "category": category,
                        "source": str(CONTRASTIVE_PAIRS_PATH),
                    }
                )
            if isinstance(sys_b, str) and sys_b.strip():
                prompts.append(
                    {
                        "prompt": prompt,
                        "system_prompt": sys_b,
                        "system_tag": "contrastive_b",
                        "category": category,
                        "source": str(CONTRASTIVE_PAIRS_PATH),
                    }
                )

    markers = _safe_load_json(SARCASM_MARKERS_PATH) or {}
    if isinstance(markers, dict):
        sarc = markers.get("sarcasm", [])
        if isinstance(sarc, list):
            added = 0
            for m in sarc:
                if added >= 20:
                    break
                if isinstance(m, dict) and isinstance(m.get("text"), str):
                    prompts.append(
                        {
                            "prompt": m["text"],
                            "system_prompt": V4_SYSTEM_PROMPT,
                            "system_tag": "v4",
                            "category": str(m.get("category", "sarcasm_marker")),
                            "source": str(SARCASM_MARKERS_PATH),
                        }
                    )
                    added += 1

    for p in SARCASM_PROMPTS:
        prompts.append(
            {
                "prompt": p,
                "system_prompt": V4_SYSTEM_PROMPT,
                "system_tag": "v4",
                "category": "hardcoded_sarcasm",
                "source": "hardcoded_map_qwen35",
            }
        )
    for p in MATH_PROMPTS:
        prompts.append(
            {
                "prompt": p,
                "system_prompt": V4_SYSTEM_PROMPT,
                "system_tag": "v4",
                "category": "hardcoded_math",
                "source": "hardcoded_map_qwen35",
            }
        )
    for p in KNOWLEDGE_PROMPTS:
        prompts.append(
            {
                "prompt": p,
                "system_prompt": V4_SYSTEM_PROMPT,
                "system_tag": "v4",
                "category": "hardcoded_knowledge",
                "source": "hardcoded_map_qwen35",
            }
        )
    for p in IDENTITY_PROMPTS:
        prompts.append(
            {
                "prompt": p,
                "system_prompt": V4_SYSTEM_PROMPT,
                "system_tag": "v4",
                "category": "hardcoded_identity",
                "source": "hardcoded_map_qwen35",
            }
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for x in prompts:
        key = (
            x["prompt"],
            x.get("system_prompt") or "",
            x.get("category", "unknown"),
        )
        if key not in seen:
            deduped.append(x)
            seen.add(key)
    return deduped


def load_model(model_name: str, device: str = "cuda:0"):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
        torch_dtype="auto",
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
        self.resume = resume

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
                    tensor = self._safe_torch_load(shard)
                    total += int(tensor.shape[0])
            self.token_counts[idx] = total

    @staticmethod
    def _safe_torch_load(path: Path) -> torch.Tensor:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _make_hook(self, layer_idx: int):
        def hook_fn(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                return
            if hidden.ndim != 3:
                return
            if hidden.shape[0] != 1:
                raise ValueError("ActivationCollector currently supports batch_size=1 only")

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
        self,
        prompt_idx: int,
        category: str,
        system_tag: str,
        prompt_text: str,
        source: str,
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
        self,
        layer_idx: int,
        acts: list[torch.Tensor],
        metas: list[TokenMetadata],
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
            acts = self.buffers[layer_idx][: self.shard_size]
            metas = self.metadata_buffers[layer_idx][: self.shard_size]
            self._write_shard(layer_idx, acts, metas)
            del self.buffers[layer_idx][: self.shard_size]
            del self.metadata_buffers[layer_idx][: self.shard_size]

    def flush_shard(self, layer_idx: int) -> None:
        if not self.buffers[layer_idx]:
            return
        acts = self.buffers[layer_idx]
        metas = self.metadata_buffers[layer_idx]
        self._write_shard(layer_idx, acts, metas)
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
    def reached_budget() -> bool:
        return any(collector.total_tokens(idx) >= max_tokens_per_layer for idx in collector.target_indices)

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
                text = processor.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = processor(text=[text], return_tensors="pt", padding=True).to(model_device)
                input_len = int(inputs["input_ids"].shape[1])
                collector.mark_prefill_boundary(input_len)

                with torch.no_grad():
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=max_gen_tokens,
                        temperature=float(temp),
                        top_p=0.9,
                        do_sample=True,
                        repetition_penalty=1.1,
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
                    f"L{idx:02d}:{collector.total_tokens(idx)}" for idx in collector.target_indices
                )
                pbar.set_postfix_str(budget_str)

        gc.collect()
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    cfg = CollectionConfig()
    parser = argparse.ArgumentParser(description="Collect Qwen3.5 activations for SAE training.")
    parser.add_argument("--model", type=str, choices=["base", "abliterated"], default=cfg.model_tag)
    parser.add_argument("--layers", type=int, nargs="+", default=cfg.target_layers)
    parser.add_argument("--max-tokens", type=int, default=cfg.max_tokens)
    parser.add_argument("--shard-size", type=int, default=cfg.shard_size)
    parser.add_argument("--max-gen-tokens", type=int, default=cfg.max_gen_tokens)
    parser.add_argument("--n-reps", type=int, default=4)
    parser.add_argument("--temperatures", type=float, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_cfg = ModelConfig()
    model_name = MODELS[args.model]
    if not model_cached(model_name, model_cfg.hf_cache):
        print(f"[WARN] Model not found in cache ({model_cfg.hf_cache}). Will download from hub.")

    output_dir = Path(args.output) if args.output else (ACTIVATIONS_DIR / args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_bank = build_prompt_bank(seed=args.seed)
    with (output_dir / "prompt_bank.json").open("w", encoding="utf-8") as f:
        json.dump(prompt_bank, f, indent=2, ensure_ascii=False)

    temperatures = args.temperatures
    if not temperatures:
        base = [0.3, 0.7, 1.0, 1.2]
        temperatures = [base[i % len(base)] for i in range(args.n_reps)]

    model = None
    collector = None
    finalized_stats: dict[int, dict[str, int]] | None = None

    run_meta = {
        "timestamp": datetime.now().isoformat(),
        "model_tag": args.model,
        "model_name": model_name,
        "layers": args.layers,
        "max_tokens": args.max_tokens,
        "shard_size": args.shard_size,
        "max_gen_tokens": args.max_gen_tokens,
        "temperatures": temperatures,
        "resume": args.resume,
        "seed": args.seed,
    }
    with (output_dir / "collection_run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    try:
        model, processor, layers, hidden_dim = load_model(model_name, device=args.device)
        print(f"[INFO] Loaded model hidden_dim={hidden_dim}, n_layers={len(layers)}")

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
        print("[WARN] Interrupted by user. Finalizing partial buffers.")
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
        print(json.dumps({str(k): v for k, v in finalized_stats.items()}, indent=2))


if __name__ == "__main__":
    main()
