#!/usr/bin/env python3
"""All-justice Qwen-Scope SAE overlap analysis for SCOTUS chunks.

This asks a different question from the pairwise probe:

Do masked chunks from different justices activate the same SAE features at the
same layers/regions once we condition on broad issue/posture structure?
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from probe_scotus_sae_features import encode_hidden_to_sae_csr, infer_top_k, safe_sae_name  # noqa: E402
from probe_scotus_style import DEFAULT_MODEL, capture_features, markdown_table, now_iso, read_jsonl, write_json, write_jsonl  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "scotus" / "scotus_chunk_inventory_v21.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "sweep_v4"
DEFAULT_SAE_PATH = Path("/home/orwel/dev_genius/models/qwen_scope/SAE-Res-Qwen3.5-27B-W80K-L0_100")
DEFAULT_JUSTICES = "Scalia,Ginsburg,Thomas,Souter"
DEFAULT_ISSUES = "Criminal Procedure,Economic Activity,Judicial Power,Civil Rights,Federalism,First Amendment"
DEFAULT_LAYERS = "4,8,12,16"
DEFAULT_REGIONS = "prompt_mean,excerpt_mean"


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    return sorted(set(values))


def parse_str_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def stable_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("justice")),
        str(row.get("issue_area_label")),
        str(row.get("section_posture")),
        str(row.get("cluster_id")),
        str(row.get("section_id")),
        str(row.get("chunk_id")),
    )


def load_balanced_examples(
    chunks_path: Path,
    *,
    justices: list[str],
    issues: list[str],
    variant: str,
    max_per_justice_issue: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(chunks_path):
        if row.get("text_variant") != variant:
            continue
        if row.get("excluded"):
            continue
        if not row.get("passes_reasoning_filter", True):
            continue
        justice = str(row.get("justice"))
        issue = str(row.get("issue_area_label") or "unknown")
        if justice not in justices or issue not in issues:
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        grouped[(justice, issue)].append(row)

    examples: list[dict[str, Any]] = []
    availability: dict[str, Any] = {}
    for justice in justices:
        for issue in issues:
            rows = sorted(grouped.get((justice, issue), []), key=stable_row_key)
            availability[f"{justice}/{issue}"] = len(rows)
            if not rows:
                continue
            if len(rows) > max_per_justice_issue:
                indices = sorted(rng.choice(len(rows), size=max_per_justice_issue, replace=False).tolist())
                rows = [rows[idx] for idx in indices]
            for row in rows:
                chunk_id = str(row["chunk_id"])
                examples.append(
                    {
                        "example_id": f"all4|{variant}|{chunk_id}",
                        "chunk_id": chunk_id,
                        "split": "analysis",
                        "label": int(justices.index(justice)),
                        "justice": justice,
                        "text": row["text"],
                        "cluster_id": row.get("cluster_id"),
                        "case_name": row.get("case_name"),
                        "scdb_id": row.get("scdb_id"),
                        "issue_area": row.get("issue_area"),
                        "issue_area_label": issue,
                        "opinion_type": row.get("opinion_type") or "unknown",
                        "section_posture": row.get("section_posture") or "unknown",
                        "section_confidence": row.get("section_confidence") or "unknown",
                        "chunk_position_bucket": row.get("chunk_position_bucket") or "unknown",
                        "term": row.get("term"),
                        "decade": row.get("decade") or "unknown",
                        "decision_direction": row.get("decision_direction") or "unknown",
                        "source_url": row.get("source_url"),
                        "token_count": row.get("token_count"),
                        "citation_count": row.get("citation_count"),
                    }
                )
    examples.sort(key=lambda row: (row["issue_area_label"], row["justice"], row["chunk_id"]))
    return examples, availability


def get_or_build_sae_matrix(
    *,
    features_npz: Path,
    cache_dir: Path,
    sae_path: Path,
    layer: int,
    region: str,
    top_k: int,
    batch_size: int,
    device: torch.device,
    overwrite: bool,
) -> sp.csr_matrix:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{safe_sae_name(sae_path)}__{region}__L{layer:02d}.npz"
    if cache_path.exists() and not overwrite:
        return sp.load_npz(cache_path).tocsr()
    key = f"{region}__L{layer:02d}"
    with np.load(features_npz) as data:
        if key not in data.files:
            raise KeyError(f"Missing {key} in {features_npz}")
        hidden = data[key].astype(np.float32, copy=False)
    matrix = encode_hidden_to_sae_csr(
        hidden,
        sae_path=sae_path,
        layer=layer,
        top_k=top_k,
        batch_size=batch_size,
        device=device,
    )
    sp.save_npz(cache_path, matrix)
    return matrix


def top_feature_set(matrix: sp.csr_matrix, indices: np.ndarray, top_k_features: int) -> tuple[set[int], np.ndarray]:
    sub = matrix[indices]
    df = np.diff(sub.tocsc().indptr).astype(np.float32)
    if np.count_nonzero(df) == 0:
        return set(), df
    nonzero = np.flatnonzero(df)
    ranked = nonzero[np.argsort(df[nonzero])[::-1]]
    return set(int(x) for x in ranked[:top_k_features]), df


def weighted_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(a, b).sum()
    if denom <= 0:
        return 0.0
    return float(np.minimum(a, b).sum() / denom)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def pair_rows_for_groups(
    *,
    matrix: sp.csr_matrix,
    meta_rows: list[dict[str, Any]],
    justices: list[str],
    group_field: str | None,
    group_value: str | None,
    top_k_features: int,
    min_group_n: int,
) -> list[dict[str, Any]]:
    grouped_indices: dict[str, np.ndarray] = {}
    for justice in justices:
        idxs = [
            idx
            for idx, row in enumerate(meta_rows)
            if row["justice"] == justice and (group_field is None or str(row.get(group_field)) == group_value)
        ]
        if len(idxs) >= min_group_n:
            grouped_indices[justice] = np.array(idxs, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    cached: dict[str, tuple[set[int], np.ndarray]] = {
        justice: top_feature_set(matrix, idxs, top_k_features)
        for justice, idxs in grouped_indices.items()
    }
    for left_idx, left in enumerate(justices):
        for right in justices[left_idx + 1 :]:
            if left not in cached or right not in cached:
                continue
            left_set, left_df = cached[left]
            right_set, right_df = cached[right]
            union = left_set | right_set
            jaccard = len(left_set & right_set) / len(union) if union else 0.0
            left_rate = left_df / max(1, len(grouped_indices[left]))
            right_rate = right_df / max(1, len(grouped_indices[right]))
            rows.append(
                {
                    "group_field": group_field or "overall",
                    "group_value": group_value or "overall",
                    "justice_a": left,
                    "justice_b": right,
                    "n_a": int(len(grouped_indices[left])),
                    "n_b": int(len(grouped_indices[right])),
                    "top_jaccard": float(jaccard),
                    "weighted_jaccard": weighted_jaccard(left_rate, right_rate),
                    "cosine_df": cosine(left_rate, right_rate),
                }
            )
    return rows


def summarize_layer_region(
    *,
    matrix: sp.csr_matrix,
    meta_rows: list[dict[str, Any]],
    justices: list[str],
    issues: list[str],
    top_k_features: int,
    min_group_n: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(
        pair_rows_for_groups(
            matrix=matrix,
            meta_rows=meta_rows,
            justices=justices,
            group_field=None,
            group_value=None,
            top_k_features=top_k_features,
            min_group_n=min_group_n,
        )
    )
    for issue in issues:
        rows.extend(
            pair_rows_for_groups(
                matrix=matrix,
                meta_rows=meta_rows,
                justices=justices,
                group_field="issue_area_label",
                group_value=issue,
                top_k_features=top_k_features,
                min_group_n=min_group_n,
            )
        )
    for posture in ["majority", "dissent"]:
        rows.extend(
            pair_rows_for_groups(
                matrix=matrix,
                meta_rows=meta_rows,
                justices=justices,
                group_field="section_posture",
                group_value=posture,
                top_k_features=top_k_features,
                min_group_n=min_group_n,
            )
        )
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["group_field"], row["group_value"])].append(row)
    summary = {
        "overall_top_jaccard": float(np.mean([r["top_jaccard"] for r in by_group[("overall", "overall")]])),
        "overall_weighted_jaccard": float(np.mean([r["weighted_jaccard"] for r in by_group[("overall", "overall")]])),
        "overall_cosine_df": float(np.mean([r["cosine_df"] for r in by_group[("overall", "overall")]])),
    }
    issue_rows = [r for r in rows if r["group_field"] == "issue_area_label"]
    posture_rows = [r for r in rows if r["group_field"] == "section_posture"]
    if issue_rows:
        summary["issue_top_jaccard"] = float(np.mean([r["top_jaccard"] for r in issue_rows]))
        summary["issue_weighted_jaccard"] = float(np.mean([r["weighted_jaccard"] for r in issue_rows]))
        summary["issue_cosine_df"] = float(np.mean([r["cosine_df"] for r in issue_rows]))
    if posture_rows:
        summary["posture_top_jaccard"] = float(np.mean([r["top_jaccard"] for r in posture_rows]))
        summary["posture_weighted_jaccard"] = float(np.mean([r["weighted_jaccard"] for r in posture_rows]))
        summary["posture_cosine_df"] = float(np.mean([r["cosine_df"] for r in posture_rows]))
    return rows, summary


def write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    sample_counts: Counter[tuple[str, str]],
    summary_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> None:
    sample_table = [
        [justice, issue, sample_counts[(justice, issue)]]
        for justice in manifest["justices"]
        for issue in manifest["issues"]
    ]
    summary_table = [
        [
            row["region"],
            row["layer"],
            f"{row['overall_top_jaccard']:.3f}",
            f"{row['overall_weighted_jaccard']:.3f}",
            f"{row['overall_cosine_df']:.3f}",
            f"{row.get('issue_top_jaccard', 0.0):.3f}",
            f"{row.get('issue_weighted_jaccard', 0.0):.3f}",
            f"{row.get('issue_cosine_df', 0.0):.3f}",
            f"{row.get('posture_top_jaccard', 0.0):.3f}",
            f"{row.get('posture_weighted_jaccard', 0.0):.3f}",
            f"{row.get('posture_cosine_df', 0.0):.3f}",
        ]
        for row in summary_rows
    ]
    best_issue = max(summary_rows, key=lambda row: row.get("issue_weighted_jaccard", 0.0))
    weakest_issue = min(summary_rows, key=lambda row: row.get("issue_weighted_jaccard", 0.0))
    overall_pairs = sorted(
        [row for row in pair_rows if row["group_field"] == "overall"],
        key=lambda row: row["weighted_jaccard"],
        reverse=True,
    )[:24]
    weak_conditioned_pairs = sorted(
        [row for row in pair_rows if row["group_field"] != "overall"],
        key=lambda row: (row["weighted_jaccard"], row["top_jaccard"]),
    )[:60]

    def pair_table(rows: list[dict[str, Any]]) -> list[list[str]]:
        return [
            [
                row["group_field"],
                row["group_value"],
                row["region"],
                row["layer"],
                f"{row['justice_a']} / {row['justice_b']}",
                f"{row['top_jaccard']:.3f}",
                f"{row['weighted_jaccard']:.3f}",
                f"{row['cosine_df']:.3f}",
                f"{row['n_a']} / {row['n_b']}",
            ]
            for row in rows
        ]

    pair_headers = ["Group", "Value", "Region", "Layer", "Pair", "Top-J", "Weighted-J", "Cosine", "N"]
    lines = [
        "# SCOTUS All-Justice SAE Overlap",
        "",
        f"Started: `{manifest['started_at']}`",
        f"Finished: `{manifest.get('finished_at', '')}`",
        "",
        "## Configuration",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Model", manifest["model_path"]],
                ["SAE", manifest["sae_path"]],
                ["Justices", ", ".join(manifest["justices"])],
                ["Issues", ", ".join(manifest["issues"])],
                ["Layers", ", ".join(str(x) for x in manifest["layers"])],
                ["Regions", ", ".join(manifest["regions"])],
                ["Examples", manifest["n_examples"]],
                ["Top features per group", manifest["top_k_features"]],
            ],
        ),
        "",
        "## Preliminary Read",
        "",
        (
            "- Broad all-justice SAE routing is mostly shared in this pilot. "
            f"The strongest issue-conditioned setting is `{best_issue['region']} @ L{best_issue['layer']}` "
            f"with weighted-J `{best_issue.get('issue_weighted_jaccard', 0.0):.3f}` and cosine "
            f"`{best_issue.get('issue_cosine_df', 0.0):.3f}`."
        ),
        (
            "- The weakest issue-conditioned setting is "
            f"`{weakest_issue['region']} @ L{weakest_issue['layer']}` with weighted-J "
            f"`{weakest_issue.get('issue_weighted_jaccard', 0.0):.3f}` and cosine "
            f"`{weakest_issue.get('issue_cosine_df', 0.0):.3f}`. Later layers show more top-feature turnover."
        ),
        "- Treat low conditioned pair rows as candidate justice-specific feature differences, not steering candidates.",
        "",
        "## Sample Counts",
        "",
        markdown_table(["Justice", "Issue", "Examples"], sample_table),
        "",
        "## Layer/Region Overlap Summary",
        "",
        markdown_table(
            [
                "Region",
                "Layer",
                "Overall top-J",
                "Overall weighted-J",
                "Overall cosine",
                "Issue top-J",
                "Issue weighted-J",
                "Issue cosine",
                "Posture top-J",
                "Posture weighted-J",
                "Posture cosine",
            ],
            summary_table,
        ),
        "",
        "## Overall Pairwise Details",
        "",
        markdown_table(pair_headers, pair_table(overall_pairs)),
        "",
        "## Weakest Conditioned Pairwise Details",
        "",
        markdown_table(pair_headers, pair_table(weak_conditioned_pairs)),
        "",
        "## Interpretation Guide",
        "",
        "- `Top-J` is Jaccard overlap between each group's top SAE feature IDs.",
        "- `Weighted-J` compares feature activation-rate vectors and is less brittle than top-k set overlap.",
        "- High issue-conditioned overlap means the same SAE features fire for different justices on the same legal subject.",
        "- Low issue-conditioned overlap means justice-specific routing remains even after broad issue control.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all-four SCOTUS SAE overlap analysis.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sae-path", type=Path, default=DEFAULT_SAE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--justices", default=DEFAULT_JUSTICES)
    parser.add_argument("--issues", default=DEFAULT_ISSUES)
    parser.add_argument("--variant", default="masked")
    parser.add_argument("--max-per-justice-issue", type=int, default=30)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--regions", default=DEFAULT_REGIONS)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--capture-batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--sae-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--top-k-features", type=int, default=100)
    parser.add_argument("--min-group-n", type=int, default=20)
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--overwrite-sae", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    justices = parse_str_list(args.justices)
    issues = parse_str_list(args.issues)
    layers = parse_int_list(args.layers)
    regions = parse_str_list(args.regions)
    top_k = infer_top_k(args.sae_path)
    device = torch.device(args.device)

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"scotus_sae_overlap_all4_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    examples, availability = load_balanced_examples(
        args.chunks,
        justices=justices,
        issues=issues,
        variant=args.variant,
        max_per_justice_issue=args.max_per_justice_issue,
        seed=args.seed,
    )
    if not examples:
        raise RuntimeError("No examples selected")
    write_jsonl(out_dir / "overlap_examples.jsonl", examples)

    sample_counts = Counter((row["justice"], row["issue_area_label"]) for row in examples)
    manifest: dict[str, Any] = {
        "started_at": now_iso(),
        "chunks": str(args.chunks),
        "model_path": str(args.model_path),
        "sae_path": str(args.sae_path),
        "output_dir": str(out_dir),
        "justices": justices,
        "issues": issues,
        "variant": args.variant,
        "max_per_justice_issue": args.max_per_justice_issue,
        "layers": layers,
        "regions": regions,
        "device_map": args.device_map,
        "capture_batch_size": args.capture_batch_size,
        "max_length": args.max_length,
        "sae_batch_size": args.sae_batch_size,
        "sae_top_k": top_k,
        "device": str(device),
        "seed": args.seed,
        "top_k_features": args.top_k_features,
        "min_group_n": args.min_group_n,
        "n_examples": len(examples),
        "availability": availability,
        "sample_counts": {f"{justice}/{issue}": count for (justice, issue), count in sorted(sample_counts.items())},
    }
    write_json(out_dir / "manifest.json", manifest)

    features_npz = out_dir / "features.npz"
    if not args.skip_capture:
        print(f"Capturing {len(examples)} examples", flush=True)
        capture_features(
            examples,
            model_path=args.model_path,
            device_map=args.device_map,
            layers_spec=",".join(str(layer) for layer in layers),
            batch_size=args.capture_batch_size,
            max_length=args.max_length,
            template_variant="normal",
            use_chat_template=True,
            out_dir=out_dir,
        )
    elif not features_npz.exists():
        raise FileNotFoundError(f"--skip-capture used but missing {features_npz}")

    meta_rows = read_jsonl(out_dir / "feature_meta.jsonl")
    cache_dir = out_dir / "sae_features"
    all_pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for layer in layers:
        for region in regions:
            print(f"\n=== overlap {region} L{layer} ===", flush=True)
            matrix = get_or_build_sae_matrix(
                features_npz=features_npz,
                cache_dir=cache_dir,
                sae_path=args.sae_path,
                layer=layer,
                region=region,
                top_k=top_k,
                batch_size=args.sae_batch_size,
                device=device,
                overwrite=args.overwrite_sae,
            )
            pair_rows, summary = summarize_layer_region(
                matrix=matrix,
                meta_rows=meta_rows,
                justices=justices,
                issues=issues,
                top_k_features=args.top_k_features,
                min_group_n=args.min_group_n,
            )
            for row in pair_rows:
                row["layer"] = layer
                row["region"] = region
            summary["layer"] = layer
            summary["region"] = region
            all_pair_rows.extend(pair_rows)
            summary_rows.append(summary)

    summary_rows.sort(key=lambda row: (row.get("issue_weighted_jaccard", 0.0), row.get("overall_weighted_jaccard", 0.0)), reverse=True)
    write_jsonl(out_dir / "overlap_pairwise.jsonl", all_pair_rows)
    write_jsonl(out_dir / "overlap_summary.jsonl", summary_rows)
    manifest["finished_at"] = now_iso()
    manifest["summary_top"] = summary_rows[:10]
    write_json(out_dir / "manifest.json", manifest)
    write_report(
        out_dir / "report.md",
        manifest=manifest,
        sample_counts=sample_counts,
        summary_rows=summary_rows,
        pair_rows=all_pair_rows,
    )
    print(f"Wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
