#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rel_href(report_path: Path, target: Path) -> str:
    return target.resolve().relative_to(report_path.parent.resolve()).as_posix() if target.resolve().is_relative_to(report_path.parent.resolve()) else target.resolve().as_posix()


def load_phase_analysis(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    per_view_layer = obj["per_view_layer"]
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    trait_rows: list[dict[str, Any]] = []
    best_by_view: list[dict[str, Any]] = []
    entanglements: list[dict[str, str]] = []

    for view_key, view_obj in per_view_layer.items():
        view, layer = view_key.split(":")
        decodability = view_obj["decodability"]
        best_trait, best_stats = max(
            decodability.items(),
            key=lambda item: item[1]["character_holdout"]["mean_balanced_accuracy"],
        )
        best_by_view.append(
            {
                "label": f"{view} best trait",
                "value": f"{best_trait} {best_stats['character_holdout']['mean_balanced_accuracy'] * 100:.1f}%",
                "sub": f"Character-holdout balanced accuracy at {layer}",
                "tone": "good" if view != "response" else "neutral",
            }
        )

        flat: list[tuple[float, str, str, float]] = []
        for a, row in view_obj["direction_cosines"].items():
            for b, val in row.items():
                if a < b:
                    flat.append((abs(val), a, b, val))
        flat.sort(reverse=True)
        top_abs, a, b, signed = flat[0]
        entanglements.append(
            {
                "view": view,
                "a": a,
                "b": b,
                "cosine": f"{signed:+.3f}",
            }
        )

    for trait in traits:
        vals = {
            view_key.split(":")[0]: per_view_layer[view_key]["decodability"][trait]["character_holdout"]["mean_balanced_accuracy"]
            for view_key in per_view_layer
        }
        best_view = max(vals.items(), key=lambda item: item[1])[0]
        trait_rows.append(
            {
                "trait": trait,
                "mean": vals.get("mean"),
                "think": vals.get("think"),
                "response": vals.get("response"),
                "best_view": best_view,
            }
        )

    return {
        "layer_label": "L20",
        "trait_rows": trait_rows,
        "best_by_view": best_by_view,
        "entanglements": entanglements,
    }


def load_eval_summary(final_summary_path: Path, clean_subset_path: Path) -> dict[str, Any]:
    final_summary = load_json(final_summary_path)
    clean_subset = load_json(clean_subset_path)
    by_condition = final_summary["by_condition"]
    label_map = {
        "baseline_native": "Baseline Native",
        "think_explicit": "Think Explicit",
        "trace_explicit": "Trace Explicit",
    }
    conditions = []
    for condition_id in ["baseline_native", "think_explicit", "trace_explicit"]:
        row = by_condition[condition_id]
        conditions.append(
            {
                "condition_id": condition_id,
                "label": label_map[condition_id],
                **row,
            }
        )
    cards = [
        {
            "label": "Canonical Eval Format",
            "value": "trace_explicit",
            "sub": f"{clean_subset['format_adherence_rate'] * 100:.2f}% format adherence, {clean_subset['reasoning_accuracy'] * 100:.2f}% reasoning accuracy",
            "tone": "good",
        },
        {
            "label": "Native Path",
            "value": "broken",
            "sub": f"{by_condition['baseline_native']['visible_thinking_rate'] * 100:.0f}% visible thinking, {by_condition['baseline_native']['truncation_rate'] * 100:.0f}% truncation",
            "tone": "warn",
        },
        {
            "label": "Trace vs Think",
            "value": f"{by_condition['trace_explicit']['reasoning_accuracy'] * 100:.1f}% vs {by_condition['think_explicit']['reasoning_accuracy'] * 100:.1f}%",
            "sub": "Reasoning accuracy on held-out scored rows",
            "tone": "neutral",
        },
    ]
    return {"conditions": conditions, "cards": cards}


def aggregate_control_run(run_dir: Path, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for fp in sorted(run_dir.glob("records_shard_*.jsonl")):
        rows.extend(load_jsonl(fp))

    scored = [row for row in rows if row.get("track") == "reasoning" and row.get("is_correct") is not None]
    correct = sum(1 for row in scored if row.get("is_correct") is True)
    gen_tokens = sum(int(row.get("n_gen_tokens") or 0) for row in rows)

    shard_metrics: list[dict[str, Any]] = []
    for fp in sorted(run_dir.glob("summary_shard_*.json")):
        if fp.name.startswith("summary_shard_"):
            shard = load_json(fp)
            shard_metrics.append(
                {
                    "run_id": run_id,
                    "server_label": shard["server_label"],
                    "ok_responses": shard["ok_responses"],
                    "gen_tokens_per_second": shard["gen_tokens_per_second"],
                    "responses_per_second": shard["responses_per_second"],
                    "reasoning_accuracy": shard["reasoning_accuracy"],
                }
            )

    summary = {
        "run_id": run_id,
        "label": run_id,
        "rows": len(rows),
        "gen_tokens": gen_tokens,
        "scored": len(scored),
        "correct": correct,
        "reasoning_accuracy": (correct / len(scored)) if scored else None,
    }
    return summary, shard_metrics


def aggregate_weird_probe(run_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fp in sorted(run_dir.glob("records_shard_*.jsonl")):
        rows.extend(load_jsonl(fp))

    def rate(items: list[dict[str, Any]]) -> float:
        return sum(1 for row in items if row.get("format_adherent")) / len(items) if items else 0.0

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition_id"])].append(row)
        by_prompt[str(row["prompt_id"])].append(row)

    conditions = [
        {
            "condition_id": condition_id,
            "label": condition_id,
            "format_adherence_rate": rate(items),
        }
        for condition_id, items in sorted(by_condition.items())
    ]
    worst_prompts = sorted(
        (
            {
                "prompt_id": prompt_id,
                "format_adherence_rate": rate(items),
            }
            for prompt_id, items in by_prompt.items()
        ),
        key=lambda item: item["format_adherence_rate"],
    )[:5]

    cards = [
        {
            "label": "Weird Probe Rows",
            "value": f"{len(rows):,}",
            "sub": "religious, psychedelic, transcendental prompts",
            "tone": "neutral",
        },
        {
            "label": "Overall Format",
            "value": f"{rate(rows) * 100:.2f}%",
            "sub": "format adherence across both conditions",
            "tone": "good",
        },
        {
            "label": "Leakage / Truncation",
            "value": "0 / 0",
            "sub": "no visible thinking leakage and no truncation",
            "tone": "good",
        },
    ]

    examples: list[dict[str, str]] = []
    seen_categories: set[str] = set()
    for row in rows:
        if not row.get("format_adherent"):
            continue
        cat = str(row.get("prompt_category"))
        if cat in seen_categories:
            continue
        seen_categories.add(cat)
        examples.append(
            {
                "prompt_id": str(row.get("prompt_id")),
                "condition_id": str(row.get("condition_id")),
                "category": cat,
                "response_text": str(row.get("response_text") or "")[:1200],
            }
        )
        if len(examples) >= 4:
            break

    return {
        "cards": cards,
        "conditions": conditions,
        "worst_prompts": worst_prompts,
        "examples": examples,
    }


def load_benchmark(path: Path, label: str) -> dict[str, Any]:
    obj = load_json(path)
    by_track = obj.get("by_track", {})
    open_track = by_track.get("open", {})
    return {
        "label": label,
        "responses": obj["overall"]["responses"],
        "format_adherence_rate": obj["overall"]["format_adherence_rate"],
        "open_format_adherence_rate": open_track.get("format_adherence_rate"),
        "reasoning_coverage": obj["overall"]["reasoning_coverage"] or 0.0,
        "avg_gen_tokens": obj["overall"]["avg_gen_tokens"],
    }


def extract_failure_examples(path: Path) -> list[dict[str, str]]:
    rows = load_jsonl(path)
    out: list[dict[str, str]] = []
    seen_prompts: set[str] = set()
    for row in rows:
        if row.get("track") != "reasoning":
            continue
        if row.get("format_adherent"):
            continue
        prompt_id = str(row.get("prompt_id"))
        if prompt_id in seen_prompts:
            continue
        seen_prompts.add(prompt_id)
        out.append(
            {
                "task_id": str(row.get("task_id")),
                "prompt_id": prompt_id,
                "summary": "Reasoning prompt failed to emit a scorable `Explanation:` / `Final Answer:` block.",
                "full_text": str(row.get("full_text") or "")[:1600],
            }
        )
        if len(out) >= 4:
            break
    return out


def build_artifacts(report_path: Path) -> list[dict[str, str]]:
    files = [
        ("Repaired Phase Analysis", report_path.parent / "ws15k_repaired_responseonly_phase_analysis_core" / "analysis_report.md"),
        ("Trace Eval Final Summary", report_path.parent.parent / "sweep_v4" / "personality_meta_eval_v1" / "final_summary.json"),
        ("Canonical Trace Eval Subset", report_path.parent.parent / "sweep_v4" / "personality_meta_eval_trace_explicit_v1" / "subset_summary.json"),
        ("Teacher Run v3", report_path.parent.parent / "sweep_v4" / "personality_control_reasoning_reasoningonly_v3"),
        ("Teacher Run seed43", report_path.parent.parent / "sweep_v4" / "personality_control_reasoning_reasoningonly_v3_seed43"),
        ("Weird-Domain Probe", report_path.parent.parent / "sweep_v4" / "personality_weird_reasoning_probe_v1"),
        ("Trace Adapter Benchmark", report_path.parent.parent / "sweep_v4" / "trace_benchmark_meta_think_compact_trace_reasoning_adapter_d8_v1_128" / "summary.json"),
        ("Lean Adapter Benchmark", report_path.parent.parent / "sweep_v4" / "trace_benchmark_meta_think_compact_lean_reasoning_adapter_d8_v1_128" / "summary.json"),
        ("Repo Snapshot", report_path.parent / "overnight_repo_snapshot_20260404_2330.md"),
    ]
    artifacts = []
    for label, path in files:
        artifacts.append({"label": label, "href": rel_href(report_path, path)})
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the personality synthesis visualizer.")
    parser.add_argument("--output-html", type=Path, help="Output HTML path.")
    parser.add_argument("--output-json", type=Path, help="Output JSON payload path.")
    parser.add_argument("--auto-refresh-sec", type=int, default=45, help="Browser auto-refresh interval in seconds.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    reports_dir = root / "reports"
    template_path = root / "ui" / "personality_synthesis_visualizer_template.html"
    output_html = args.output_html or (reports_dir / "personality_synthesis_visualizer_live.html")
    output_json = args.output_json or (reports_dir / "personality_synthesis_visualizer_live.json")

    phase = load_phase_analysis(reports_dir / "ws15k_repaired_responseonly_phase_analysis_core" / "analysis_results.json")
    eval_summary = load_eval_summary(
        root / "sweep_v4" / "personality_meta_eval_v1" / "final_summary.json",
        root / "sweep_v4" / "personality_meta_eval_trace_explicit_v1" / "subset_summary.json",
    )
    teacher_v3, teacher_v3_servers = aggregate_control_run(root / "sweep_v4" / "personality_control_reasoning_reasoningonly_v3", "v3")
    teacher_seed43, teacher_seed43_servers = aggregate_control_run(root / "sweep_v4" / "personality_control_reasoning_reasoningonly_v3_seed43", "seed43")
    weird = aggregate_weird_probe(root / "sweep_v4" / "personality_weird_reasoning_probe_v1")

    benchmarks = [
        load_benchmark(root / "sweep_v4" / "trace_benchmark_compact_trace_sft_32_greedy" / "summary.json", "d4 trace 32 greedy"),
        load_benchmark(root / "sweep_v4" / "trace_benchmark_compact_trace_sft_128" / "summary.json", "d4 trace 128 sampled"),
        load_benchmark(root / "sweep_v4" / "trace_benchmark_compact_trace_d8_32_greedy" / "summary.json", "d8 trace 32 greedy"),
        load_benchmark(root / "sweep_v4" / "trace_benchmark_meta_think_compact_trace_reasoning_adapter_d8_v1_128" / "summary.json", "d8 adapter trace 128"),
        load_benchmark(root / "sweep_v4" / "trace_benchmark_meta_think_compact_lean_reasoning_adapter_d8_v1_128" / "summary.json", "d8 adapter lean 128"),
    ]

    adapter_manifest = load_json(root / "sweep_v4" / "nanochat_meta_think_compact_d8e3_v1" / "reasoning_adapter_v1_manifest.json")
    failure_examples = extract_failure_examples(
        root / "sweep_v4" / "trace_benchmark_meta_think_compact_trace_reasoning_adapter_d8_v1_128" / "records_shard_00.jsonl"
    )

    snapshot_cards = [
        {
            "label": "Internal Signal",
            "value": "90.5%",
            "sub": "best character-holdout decodability: extraversion in mean view at L20",
            "tone": "good",
        },
        {
            "label": "Canonical Eval",
            "value": "trace_explicit",
            "sub": "99.93% format adherence and 86.20% reasoning accuracy on held-out eval",
            "tone": "good",
        },
        {
            "label": "Teacher Data",
            "value": f"{teacher_v3['rows'] + teacher_seed43['rows']:,}",
            "sub": f"{teacher_v3['gen_tokens'] + teacher_seed43['gen_tokens']:,} generated tokens across two reasoning-control runs",
            "tone": "neutral",
        },
        {
            "label": "Weird-Domain Stability",
            "value": f"{weird['cards'][1]['value']}",
            "sub": "clean formatting generalized to religious / psychedelic / transcendental prompts",
            "tone": "good",
        },
        {
            "label": "Small-Model Gap",
            "value": "0%",
            "sub": "reasoning coverage still collapsed in nanochat adapter benchmarks",
            "tone": "warn",
        },
    ]

    hero = {
        "title": "Everything We Know So Far",
        "subtitle": (
            "A single visual synthesis of the repaired phase-aware personality sweep, the clean trace-format evals, "
            "overnight teacher-data generation, exploratory weird-domain probes, and the current nanochat failure mode."
        ),
        "notes": [
            {"label": "Main Result", "text": "Personality signal is strongest in internal state (`mean` and `think`), not in the final response surface."},
            {"label": "Eval Result", "text": "`trace_explicit` is the clean benchmark contract. Native reasoning mode on these servers is not usable."},
            {"label": "Current Blocker", "text": "Small-model trace scaffolds still collapse on held-out reasoning prompts even after adapter data."},
        ],
        "side_stats": [
            {
                "label": "Teacher Rows",
                "value": f"{teacher_v3['rows'] + teacher_seed43['rows']:,}",
                "detail": "Two overnight reasoning-control runs completed on the dev GPUs."
            },
            {
                "label": "Clean Weird Probe",
                "value": f"{weird['cards'][1]['value']}",
                "detail": "Religious / psychedelic / transcendental prompts mostly held the format contract."
            },
            {
                "label": "Nanochat Reasoning Coverage",
                "value": "0%",
                "detail": "The adapter pass improved open formatting, but not the held-out reasoning schema."
            },
        ],
    }

    findings = [
        {
            "label": "Mechanistic Signal",
            "text": "The repaired sweep still shows strong personality decodability after contamination repair. `mean` and `think` carry cleaner character signal than `response`, which supports training around internal-state control instead of surface-style mimicry.",
        },
        {
            "label": "Canonical Interface",
            "text": "The A/B/C held-out eval settled the interface question: `trace_explicit` is the right benchmark format for now. It preserves reasoning while staying almost perfectly parseable and avoiding visible-thinking leakage.",
        },
        {
            "label": "Teacher Data Value",
            "text": "The overnight reasoning-only sweeps were not wasted. They produced more than twenty thousand strict-format teacher rows at about seventy-five percent reasoning accuracy, which is enough to support controlled small-model experiments and future distillation work.",
        },
        {
            "label": "Weird-Domain Generalization",
            "text": "The weird-domain probe suggests the clean trace/think formatting carries over into stranger domains without immediately breaking. That means the low-level interface is more general than just everyday social prompts.",
        },
        {
            "label": "Current Failure Mode",
            "text": "The nanochat path is still failing at the output contract layer. The model can hold the open scaffold better than before, but on reasoning prompts it degenerates into malformed scaffold chatter instead of a scorable `Explanation:` / `Final Answer:` block.",
        },
    ]

    teacher_cards = [
        {
            "label": "Run v3",
            "value": f"{teacher_v3['reasoning_accuracy'] * 100:.2f}%",
            "sub": f"{teacher_v3['rows']:,} rows · {teacher_v3['gen_tokens']:,} generated tokens",
            "tone": "good",
        },
        {
            "label": "Run seed43",
            "value": f"{teacher_seed43['reasoning_accuracy'] * 100:.2f}%",
            "sub": f"{teacher_seed43['rows']:,} rows · {teacher_seed43['gen_tokens']:,} generated tokens",
            "tone": "good",
        },
        {
            "label": "Adapter Kept Rows",
            "value": f"{adapter_manifest['counts']['kept']:,}",
            "sub": f"dropped {adapter_manifest['counts']['dropped_incorrect']:,} incorrect rows before nanochat adapter training",
            "tone": "neutral",
        },
    ]

    nanochat_cards = [
        {
            "label": "Best Open-Format Step",
            "value": "76.56%",
            "sub": "open-track adherence for `d8 adapter trace 128`",
            "tone": "neutral",
        },
        {
            "label": "Reasoning Coverage",
            "value": "0%",
            "sub": "still zero for every nanochat benchmark we ran so far",
            "tone": "warn",
        },
        {
            "label": "Interpretation",
            "value": "negative result",
            "sub": "more reasoning teacher data alone did not fix held-out reasoning-output structure",
            "tone": "warn",
        },
    ]

    timeline = [
        {
            "date": "April 2, 2026",
            "title": "Repaired Pass-2 and Phase Analysis",
            "detail": "The repaired 25M subset replay confirmed that personality signal persists after regeneration and is strongest in internal views at L20.",
        },
        {
            "date": "April 4, 2026",
            "title": "Held-Out A/B/C Eval Closed The Interface Question",
            "detail": "`trace_explicit` emerged as the clean default. Native baseline was unusable due to visible thinking and truncation.",
        },
        {
            "date": "April 4-5, 2026",
            "title": "Overnight Teacher Data + Weird Probe",
            "detail": "Two reasoning-control runs completed, plus a smaller weird-domain run that showed the format contract generalizes beyond normal social/reasoning prompts.",
        },
        {
            "date": "April 4-5, 2026",
            "title": "Nanochat Adapter Negative Result",
            "detail": "The adapter SFT completed successfully and benchmarked quickly, but reasoning coverage remained at zero. The failure mode is schema collapse, not latency or truncation.",
        },
    ]

    data = {
        "generated_at": datetime.now().isoformat(),
        "auto_refresh_seconds": args.auto_refresh_sec,
        "hero": hero,
        "snapshot_cards": snapshot_cards,
        "timeline": timeline,
        "findings": findings,
        "mechanistic": phase,
        "eval": eval_summary,
        "teacher": {
            "cards": teacher_cards,
            "runs": [teacher_v3, teacher_seed43],
            "servers": teacher_v3_servers + teacher_seed43_servers,
        },
        "weird": weird,
        "nanochat": {
            "cards": nanochat_cards,
            "benchmarks": benchmarks,
        },
        "examples": {
            "failures": failure_examples,
            "weird": weird["examples"],
        },
        "artifacts": build_artifacts(output_html),
    }

    output_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    template = template_path.read_text(encoding="utf-8")
    html = template.replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False))
    output_html.write_text(html, encoding="utf-8")
    print(output_html)


if __name__ == "__main__":
    main()
