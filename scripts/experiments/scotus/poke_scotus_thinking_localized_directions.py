#!/usr/bin/env python3
"""Two-stage no-mask poke for localized Article III thought-state directions.

This tests candidate sites nominated by
``localize_article3_ambiguous_thought_states.py``. It applies a frozen
multi-site component direction during generated thought and answer tokens, then
compares against same-site random controls on the ambiguous Article III prompt
bank. This is a smoke/promotion harness; short thinking budgets must be marked
explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from localize_article3_ambiguous_thought_states import select_component_module  # noqa: E402
from patch_scotus_thinking_traces import DEFAULT_MODEL  # noqa: E402
from poke_scotus_sae_layers import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    load_model_and_tokenizer,
    load_prompt_specs,
    now_iso,
    parse_float_list,
    select_prompt_specs,
    transformer_layers,
    write_json,
    write_jsonl,
)
from poke_scotus_thinking_bundle import (  # noqa: E402
    DEFAULT_COMPLETE_ANSWER_TOKENS,
    MIN_COMPLETE_ANSWER_TOKENS,
    add_segment_base_deltas,
    aggregate_segments,
    compare_candidate_to_random,
    generate_continuation,
    row_for_two_stage,
    write_report,
)
from qwen_eval_budget import (  # noqa: E402
    enforce_complete_answer_budget,
    qwen_thinking_answer_budget_metadata,
)
from run_scotus_thinking_smoke import IMITATION_RE, format_chat, strip_generation_specials  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_BANK = PROJECT_ROOT / "data" / "scotus" / "scotus_article3_ambiguous_poke_prompts_v1.jsonl"
DEFAULT_LOCALIZATION_RUN = (
    PROJECT_ROOT / "sweep_v4" / "scotus_article3_ambiguous_thought_state_localization_20260502_003317"
)


@dataclass(frozen=True)
class SiteDirection:
    layer: int
    component: str
    region: str
    direction_key: str
    vector: torch.Tensor
    rank_score_minus_shuffle_max: float

    @property
    def site_key(self) -> str:
        return f"L{self.layer:02d}_{self.component}_{self.region}"


def output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def replace_output_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (tensor,) + output[1:]
    return tensor


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def load_site_directions(
    localization_run: Path,
    *,
    direction_indices: list[int],
    top_sites: int,
    components: set[str] | None,
    use_raw_mean_delta: bool,
) -> list[SiteDirection]:
    meta_rows = read_jsonl(localization_run / "direction_meta.jsonl")
    arrays = np.load(localization_run / "top_directions.npz")
    if direction_indices:
        selected_meta = [meta_rows[idx] for idx in direction_indices]
    else:
        filtered = [row for row in meta_rows if components is None or str(row["component"]) in components]
        selected_meta = filtered[:top_sites]
    if not selected_meta:
        raise ValueError("No localized site directions selected")

    sites: list[SiteDirection] = []
    for row in selected_meta:
        key = str(row["direction_key"])
        array_key = f"{key}_raw_mean_delta" if use_raw_mean_delta else key
        if array_key not in arrays:
            raise KeyError(f"Missing {array_key} in {localization_run / 'top_directions.npz'}")
        vec = torch.from_numpy(arrays[array_key].astype(np.float32, copy=False)).contiguous()
        sites.append(
            SiteDirection(
                layer=int(row["layer"]),
                component=str(row["component"]),
                region=str(row["region"]),
                direction_key=key,
                vector=vec,
                rank_score_minus_shuffle_max=float(row["rank_score_minus_shuffle_max"]),
            )
        )
    return sites


def random_sites_like(sites: list[SiteDirection], *, seed: int) -> list[SiteDirection]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_sites: list[SiteDirection] = []
    for site in sites:
        noise = torch.randn(site.vector.shape, generator=generator, dtype=torch.float32)
        noise = noise / torch.linalg.vector_norm(noise).clamp(min=1e-12)
        source_norm = torch.linalg.vector_norm(site.vector.float()).clamp(min=1e-12)
        random_sites.append(
            SiteDirection(
                layer=site.layer,
                component=site.component,
                region=site.region,
                direction_key=f"random_like_{site.direction_key}",
                vector=(noise * source_norm).contiguous(),
                rank_score_minus_shuffle_max=0.0,
            )
        )
    return random_sites


def install_site_hooks(
    layers_mod: Any,
    *,
    sites: list[SiteDirection],
    alpha: float,
    position: str,
    normalize_by_sites: bool,
) -> list[Any]:
    if position not in {"decode", "last", "all"}:
        raise ValueError(f"Unsupported position: {position}")
    if not sites or alpha == 0.0:
        return []
    site_scale = 1.0 / math.sqrt(len(sites)) if normalize_by_sites else 1.0
    handles: list[Any] = []
    for site in sites:
        module = select_component_module(layers_mod[site.layer], site.component)
        effective = float(alpha) * site_scale

        def make_hook(item: SiteDirection, eff: float) -> Any:
            def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
                hidden = output_tensor(out)
                if position == "decode" and int(hidden.shape[1]) != 1:
                    return out
                poke = item.vector.to(device=hidden.device, dtype=hidden.dtype) * eff
                edited = hidden.clone()
                if position == "all":
                    edited = edited + poke.view(1, 1, -1)
                else:
                    edited[:, -1, :] = edited[:, -1, :] + poke
                return replace_output_tensor(out, edited)

            return hook

        handles.append(module.register_forward_hook(make_hook(site, effective)))
    return handles


@torch.inference_mode()
def generate_two_stage_sites(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    layers_mod: Any,
    sites: list[SiteDirection] | None,
    alpha: float,
    position: str,
    normalize_by_sites: bool,
    thought_tokens: int,
    answer_tokens: int,
) -> dict[str, Any]:
    handles = install_site_hooks(
        layers_mod,
        sites=sites or [],
        alpha=alpha,
        position=position,
        normalize_by_sites=normalize_by_sites,
    )
    try:
        chat = format_chat(tokenizer, prompt, enable_thinking=True)
        prefilled_open_think = chat.rstrip().endswith("<think>")
        thought_output = generate_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt=chat,
            max_new_tokens=thought_tokens,
        )
        thought_raw = strip_generation_specials(str(thought_output["raw_text"]))
        thought_text, separator, answer_prefix = thought_raw.partition("</think>")
        model_closed_thinking = bool(separator)
        thought_text = thought_text.strip()
        answer_prefix = strip_generation_specials(answer_prefix)
        answer_prompt = f"{chat}{thought_text}\n</think>\n\n{answer_prefix}"
        answer_output = generate_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt=answer_prompt,
            max_new_tokens=answer_tokens,
        )
        answer_text = (answer_prefix + "\n" + strip_generation_specials(str(answer_output["raw_text"]))).strip()
        full_text = f"<think>\n{thought_text}\n</think>\n\n{answer_text}".strip()
        return {
            "prefilled_open_think": prefilled_open_think,
            "model_closed_thinking": model_closed_thinking,
            "mechanically_closed_for_answer": True,
            "thinking": thought_text,
            "answer": answer_text,
            "full_text": full_text,
            "thinking_generated_tokens": int(thought_output["generated_tokens"]),
            "answer_generated_tokens": int(answer_output["generated_tokens"]),
            "thinking_prompt_tokens": int(thought_output["prompt_tokens"]),
            "answer_prompt_tokens": int(answer_output["prompt_tokens"]),
            "thinking_nonempty": bool(thought_text),
            "answer_nonempty": bool(answer_text),
            "thinking_imitation_markers": sorted(set(IMITATION_RE.findall(thought_text))),
            "answer_imitation_markers": sorted(set(IMITATION_RE.findall(answer_text))),
        }
    finally:
        for handle in handles:
            handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--localization-run", type=Path, default=DEFAULT_LOCALIZATION_RUN)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--direction-indices", default="", help="Comma-separated direction_meta row indices. Overrides --top-sites.")
    parser.add_argument("--top-sites", type=int, default=4)
    parser.add_argument("--components", default="", help="Optional comma-separated component filter.")
    parser.add_argument("--use-raw-mean-delta", action="store_true")
    parser.add_argument("--alphas", default="2.0")
    parser.add_argument("--position", choices=["decode", "last", "all"], default="decode")
    parser.add_argument("--no-normalize-by-sites", action="store_true")
    parser.add_argument("--prompt-ids", default="0")
    parser.add_argument("--max-prompts", type=int, default=1)
    parser.add_argument("--random-controls", type=int, default=2)
    parser.add_argument("--thought-tokens", type=int, default=DEFAULT_COMPLETE_ANSWER_TOKENS)
    parser.add_argument("--answer-tokens", type=int, default=DEFAULT_COMPLETE_ANSWER_TOKENS)
    parser.add_argument(
        "--allow-short-thinking-budget",
        action="store_true",
        help="Permit visible-thinking budgets below the complete-answer threshold for smoke/debug only.",
    )
    parser.add_argument(
        "--allow-short-answer-budget",
        action="store_true",
        help="Permit answer budgets below the complete-answer threshold for smoke/debug only.",
    )
    parser.add_argument("--device-map", default="single")
    parser.add_argument("--seed", type=int, default=20260502)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_complete_answer_budget(
        args.thought_tokens,
        allow_short=args.allow_short_thinking_budget,
        label="thought_tokens",
        purpose="visible-reasoning no-mask actuator evaluation",
    )
    enforce_complete_answer_budget(
        args.answer_tokens,
        allow_short=args.allow_short_answer_budget,
        label="answer_tokens",
        purpose="final-answer no-mask actuator evaluation",
    )
    started = now_iso()
    out_dir = args.output_root / f"scotus_thinking_localized_direction_poke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    direction_indices = parse_int_list(args.direction_indices)
    component_filter = set(item.strip() for item in args.components.split(",") if item.strip()) or None
    sites = load_site_directions(
        args.localization_run,
        direction_indices=direction_indices,
        top_sites=args.top_sites,
        components=component_filter,
        use_raw_mean_delta=bool(args.use_raw_mean_delta),
    )
    alphas = parse_float_list(args.alphas)
    random_controls = [
        random_sites_like(sites, seed=args.seed + idx * 100_003) for idx in range(args.random_controls)
    ]
    candidate_name = "localized_" + "_".join(site.site_key for site in sites)
    if len(candidate_name) > 180:
        candidate_name = f"localized_top{len(sites)}_{sites[0].site_key}_etc"

    prompt_specs = select_prompt_specs(load_prompt_specs(args.prompt_bank), args.prompt_ids, args.max_prompts)
    print(f"Loaded {len(prompt_specs)} prompts", flush=True)
    print(f"Candidate sites: {', '.join(site.site_key for site in sites)}", flush=True)

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    layers_mod = transformer_layers(model)
    rows: list[dict[str, Any]] = []

    print("Generating two-stage baseline", flush=True)
    for spec in prompt_specs:
        output = generate_two_stage_sites(
            model=model,
            tokenizer=tokenizer,
            prompt=spec.prompt,
            layers_mod=layers_mod,
            sites=None,
            alpha=0.0,
            position=args.position,
            normalize_by_sites=not args.no_normalize_by_sites,
            thought_tokens=args.thought_tokens,
            answer_tokens=args.answer_tokens,
        )
        rows.append(
            row_for_two_stage(
                spec=spec,
                condition="base",
                candidate=None,
                alpha=0.0,
                random_index=None,
                layer=None,
                output=output,
            )
        )

    for alpha in alphas:
        for random_idx, random_sites in enumerate(random_controls):
            print(f"Generating random localized control {random_idx} alpha={alpha}", flush=True)
            for spec in prompt_specs:
                output = generate_two_stage_sites(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=spec.prompt,
                    layers_mod=layers_mod,
                    sites=random_sites,
                    alpha=float(alpha),
                    position=args.position,
                    normalize_by_sites=not args.no_normalize_by_sites,
                    thought_tokens=args.thought_tokens,
                    answer_tokens=args.answer_tokens,
                )
                rows.append(
                    row_for_two_stage(
                        spec=spec,
                        condition="random_unit",
                        candidate="random_localized_sites",
                        alpha=float(alpha),
                        random_index=random_idx,
                        layer=sites[0].layer,
                        output=output,
                    )
                )

        print(f"Generating candidate localized sites alpha={alpha}", flush=True)
        for spec in prompt_specs:
            output = generate_two_stage_sites(
                model=model,
                tokenizer=tokenizer,
                prompt=spec.prompt,
                layers_mod=layers_mod,
                sites=sites,
                alpha=float(alpha),
                position=args.position,
                normalize_by_sites=not args.no_normalize_by_sites,
                thought_tokens=args.thought_tokens,
                answer_tokens=args.answer_tokens,
            )
            rows.append(
                row_for_two_stage(
                        spec=spec,
                    condition="sae_poke",
                    candidate=candidate_name,
                    alpha=float(alpha),
                    random_index=None,
                    layer=sites[0].layer,
                    output=output,
                )
            )

    add_segment_base_deltas(rows)
    summaries = aggregate_segments(rows)
    comparisons = compare_candidate_to_random(rows)
    manifest = {
        "started_at": started,
        "finished_at": now_iso(),
        "model_path": str(args.model_path),
        "replay_run": str(args.localization_run),
        "localization_run": str(args.localization_run),
        "output_dir": str(out_dir),
        "candidate_names": [candidate_name],
        "direction_source": "ambiguous_thought_state_localization",
        "prompt_bank": str(args.prompt_bank),
        "alphas": alphas,
        "random_controls": int(args.random_controls),
        "position": args.position,
        "normalize_by_sites": not args.no_normalize_by_sites,
        "use_raw_mean_delta": bool(args.use_raw_mean_delta),
        "thought_tokens": int(args.thought_tokens),
        "answer_tokens": int(args.answer_tokens),
        **qwen_thinking_answer_budget_metadata(args.thought_tokens, args.answer_tokens),
        "device_map": args.device_map,
        "seed": args.seed,
        "prompt_ids": [spec.prompt_id for spec in prompt_specs],
        "prompt_keys": [spec.prompt_key for spec in prompt_specs],
        "layers": [site.layer for site in sites],
        "site_directions": [
            {
                "layer": site.layer,
                "component": site.component,
                "region": site.region,
                "direction_key": site.direction_key,
                "rank_score_minus_shuffle_max": site.rank_score_minus_shuffle_max,
            }
            for site in sites
        ],
        "not_promotion_if_short_thinking_budget": bool(args.thought_tokens < MIN_COMPLETE_ANSWER_TOKENS),
    }
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "generations.jsonl", rows)
    write_jsonl(out_dir / "summary.jsonl", summaries)
    write_jsonl(out_dir / "comparisons.jsonl", comparisons)
    write_jsonl(out_dir / "candidate_vs_prompt_matched_random.jsonl", comparisons)
    write_report(out_dir / "report.md", manifest=manifest, summaries=summaries, comparisons=comparisons, rows=rows)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
