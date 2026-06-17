#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_scotus_style import (  # noqa: E402
    DEFAULT_MODEL,
    capture_features,
    evaluate_text_baseline,
    load_feature_artifacts,
    markdown_table,
    now_iso,
    select_probe,
    write_json,
    write_jsonl,
)


DEFAULT_OUTPUT_ROOT = Path("sweep_v4")
DEFAULT_DATASET = Path("data/scotus/scotus_frame_contrast_v1.jsonl")
DEFAULT_LAYERS = "8,12,16"
DEFAULT_C_GRID = "0.003,0.01,0.03,0.1,0.3,1.0"


ARTICLE3_FACTS = [
    "a federal benefits program assigns disputed eligibility determinations to an agency tribunal",
    "a securities regulator seeks penalties after a contested administrative hearing",
    "a customs statute sends tariff disputes to a specialized non-Article III forum",
    "a bankruptcy estate objection determines how a federal distribution scheme is administered",
    "a patent review board cancels claims under a congressionally created review procedure",
    "a mine-safety statute channels civil penalties through an administrative law judge",
    "a maritime compensation program routes statutory claims through an agency process",
    "a tax penalty assessment is reviewed first by an executive-branch adjudicator",
    "a private contract counterclaim is finally resolved by a bankruptcy judge",
    "a state-law tort claim between two companies is assigned to an agency tribunal",
    "a common-law fraud claim between private parties is finally decided outside an Article III court",
    "a damages action resembling breach of contract is removed from ordinary federal court",
    "a property dispute between private landowners is assigned to a legislative court",
    "a jury-trial damages claim is folded into an agency enforcement proceeding",
    "a private indemnity action is finally adjudicated by a federal board",
    "a legal malpractice counterclaim is resolved by a non-tenured adjudicator",
    "a government licensing dispute turns on a new statutory entitlement",
    "a wage-benefit claim created by federal statute is routed through an agency",
    "a private nuisance claim is transferred to an administrative compensation board",
    "a debtor's state-law counterclaim is decided as part of bankruptcy administration",
]


FOURTH_FACTS = [
    "officers arrest a suspect and inspect a phone found in the suspect's pocket",
    "police seize a tablet during booking and review stored messages without a warrant",
    "agents arrest a driver and open a messaging app on a seized device",
    "officers retrieve a phone from an arrestee and search photographs on the device",
    "police take a smartwatch from an arrestee and inspect synced notifications",
    "agents seize a laptop bag during arrest and review files before obtaining a warrant",
    "officers arrest a courier and examine a locked phone taken from the person",
    "police arrest a suspect at home and scroll through a seized device's history",
    "officers search a jacket pocket within reach while securing an arrestee",
    "police open a small container beside an arrestee during a custodial arrest",
    "agents inspect a wallet and address book taken from an arrestee's immediate control",
    "officers look through a bag on the passenger seat during a lawful vehicle arrest",
    "police check a cigarette pack near the arrestee for weapons or destructible evidence",
    "agents search a purse held by the suspect at the moment of arrest",
    "officers inspect a backpack at the arrestee's feet during the arrest scene",
    "police examine a notebook carried by an arrestee during booking",
    "officers seize a phone and delay review until they obtain a warrant",
    "agents preserve a seized device without opening cloud-connected applications",
    "police search a reachable container for razor blades before transport",
    "officers examine a physical wallet for identification during a custodial arrest",
]


def split_for_index(index: int) -> str:
    if index < 12:
        return "train"
    if index < 16:
        return "dev"
    return "test"


def article3_text(fact: str, label: int, variant: int) -> str:
    if label == 1:
        templates = [
            (
                "The dispute should be treated as a private-rights matter. Although Congress may create "
                "administrative mechanisms, this claim resembles ordinary common-law liability between private "
                "parties. Final judgment therefore belongs in an Article III court with an independent judge."
            ),
            (
                "The better analysis treats the claim as private rather than public. The government cannot "
                "convert a traditional damages dispute into agency business merely by assigning it to a board. "
                "Article III protects the judicial forum for that kind of final adjudication."
            ),
            (
                "This is not merely administration of a federal benefit. It fixes liability of the sort courts "
                "historically resolved at common law, so the structural guarantee of an Article III tribunal "
                "carries decisive weight."
            ),
        ]
    else:
        templates = [
            (
                "The dispute fits the public-rights exception. Congress created the entitlement, integrated it "
                "with a federal regulatory scheme, and could assign initial adjudication to an expert agency "
                "without requiring final resolution by an Article III court."
            ),
            (
                "The better analysis treats the matter as public rather than private. The claim arises between "
                "the government and a regulated party under a statutory program, so agency adjudication is a "
                "permissible incident of Congress's regulatory design."
            ),
            (
                "The proceeding implements a federal administrative scheme instead of resolving an ordinary "
                "common-law suit. On that understanding, Article III permits Congress to use a non-Article III "
                "tribunal subject to appropriate judicial review."
            ),
        ]
    return f"Facts: {fact}. Analysis: {templates[variant % len(templates)]}"


def fourth_text(fact: str, label: int, variant: int) -> str:
    if label == 1:
        templates = [
            (
                "The digital-privacy frame controls. A phone or comparable device contains a vast record of "
                "private life, so searching its contents is categorically different from inspecting a physical "
                "object at arrest. Police may secure the device, but review ordinarily requires a warrant."
            ),
            (
                "The search-incident rationale does not extend to the stored data. Officer safety and evidence "
                "preservation justify seizure, not a general inspection of digital contents. The Fourth Amendment "
                "therefore requires a warrant before the data search."
            ),
            (
                "Modern devices expose far more than the area within an arrestee's immediate control. Treating "
                "that information like a pocket container would erase the privacy limit, so the constitutional "
                "rule should require judicial authorization."
            ),
        ]
    else:
        templates = [
            (
                "The search is analyzed under the traditional search-incident-to-arrest rule. Officers may inspect "
                "items within the arrestee's immediate control to protect safety and prevent destruction of "
                "evidence. On these facts, the search remains tied to the arrest scene."
            ),
            (
                "The immediate-control rationale supplies the governing frame. The object was reachable when the "
                "custodial arrest occurred, and the inspection was confined to the safety and evidence concerns "
                "that justify a search incident to arrest."
            ),
            (
                "This is the ordinary physical-container case rather than a broad digital search. Because the "
                "item was associated with the person arrested and the inspection served arrest-scene needs, the "
                "search-incident doctrine supports the officer's action."
            ),
        ]
    return f"Facts: {fact}. Analysis: {templates[variant % len(templates)]}"


TASKS = {
    "article3_private_vs_public": {
        "issue_area_label": "Judicial Power",
        "positive_label_name": "private_rights_article3",
        "negative_label_name": "public_rights_agency",
        "facts": ARTICLE3_FACTS,
        "builder": article3_text,
    },
    "fourth_digital_vs_incident": {
        "issue_area_label": "Criminal Procedure",
        "positive_label_name": "digital_privacy_warrant",
        "negative_label_name": "search_incident_immediate_control",
        "facts": FOURTH_FACTS,
        "builder": fourth_text,
    },
}


def build_examples(task_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        task = TASKS[task_name]
        facts = list(task["facts"])
        builder = task["builder"]
        for fact_idx, fact in enumerate(facts):
            split = split_for_index(fact_idx)
            for label in (0, 1):
                label_name = task["positive_label_name"] if label == 1 else task["negative_label_name"]
                example_id = f"{task_name}|{split}|{fact_idx:02d}|{label_name}"
                rows.append(
                    {
                        "example_id": example_id,
                        "chunk_id": example_id,
                        "pair_id": f"{task_name}|{fact_idx:02d}",
                        "split": split,
                        "label": int(label),
                        "justice": label_name,
                        "positive_justice": task["positive_label_name"],
                        "frame_task": task_name,
                        "frame_label": label_name,
                        "issue_area_label": task["issue_area_label"],
                        "opinion_type": "contrastive_frame",
                        "section_posture": "curated_contrast",
                        "fact_id": f"{task_name}_{fact_idx:02d}",
                        "text": builder(fact, label, fact_idx),
                    }
                )
    rows.sort(key=lambda row: (row["frame_task"], row["fact_id"], row["label"]))
    return rows


def subset_extracted(extracted: dict[str, Any], task_name: str) -> dict[str, Any]:
    meta_rows = extracted["meta_rows"]
    indices = [idx for idx, row in enumerate(meta_rows) if row["frame_task"] == task_name]
    if not indices:
        raise RuntimeError(f"No rows for task {task_name}")
    idx_arr = np.array(indices, dtype=np.int64)
    return {
        "regions": {
            region: {layer: arr[idx_arr] for layer, arr in layer_map.items()}
            for region, layer_map in extracted["regions"].items()
        },
        "meta_rows": [meta_rows[idx] for idx in indices],
        "labels": extracted["labels"][idx_arr],
        "layers": extracted["layers"],
    }


def label_count_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    counts = Counter((row["frame_task"], row["split"], row["frame_label"]) for row in rows)
    return [[task, split, label, count] for (task, split, label), count in sorted(counts.items())]


def save_raw_direction(path: Path, *, clf: Any, best: dict[str, Any], task_name: str, positive_label: str) -> None:
    scaler = clf.named_steps["scaler"]
    logreg = clf.named_steps["clf"]
    coef = logreg.coef_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    raw_direction = (coef[0] / np.maximum(scale, 1e-12)).astype(np.float32)
    raw_norm = float(np.linalg.norm(raw_direction))
    raw_unit = raw_direction / max(raw_norm, 1e-12)
    np.savez_compressed(
        path,
        raw_direction_unit=raw_unit.astype(np.float32),
        raw_direction_norm=np.array([raw_norm], dtype=np.float32),
        coef=coef,
        intercept=logreg.intercept_.astype(np.float32),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scale,
        region=np.array([best["region"]]),
        layer=np.array([int(best["layer"])]),
        C=np.array([float(best["C"])], dtype=np.float32),
        task_name=np.array([task_name]),
        positive_label=np.array([positive_label]),
    )


def task_report_rows(task_results: dict[str, dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for task_name, result in sorted(task_results.items()):
        best = result["probe"]["best"]
        split_metrics = result["probe"]["split_metrics"]
        text_baseline = result["text_baseline"]
        rows.append(
            [
                task_name,
                best["region"],
                best["layer"],
                f"{best['C']:.4g}",
                f"{best['dev_metrics']['balanced_accuracy']:.3f}",
                f"{split_metrics['test']['balanced_accuracy']:.3f}",
                f"{text_baseline['test']['balanced_accuracy']:.3f}",
                result["direction_path"],
            ]
        )
    return rows


def write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    examples: list[dict[str, Any]],
    task_results: dict[str, dict[str, Any]],
) -> None:
    top_rows: list[list[Any]] = []
    distribution_rows: list[list[Any]] = []
    for task_name, result in sorted(task_results.items()):
        for row in result["probe"]["searches"][:8]:
            top_rows.append(
                [
                    task_name,
                    row["region"],
                    row["layer"],
                    f"{row['C']:.4g}",
                    f"{row['dev_metrics']['balanced_accuracy']:.3f}",
                    f"{row['test_metrics_diagnostic']['balanced_accuracy']:.3f}",
                ]
            )
        dist = result["probe"]["search_distribution"]
        distribution_rows.append(
            [
                task_name,
                dist["n_configs"],
                f"{dist['dev_balanced_accuracy']['median']:.3f}",
                dist["dev_balanced_accuracy"]["configs_above_0_75"],
                f"{dist['test_balanced_accuracy_diagnostic']['median']:.3f}",
                dist["test_balanced_accuracy_diagnostic"]["configs_above_0_75"],
            ]
        )
    lines = [
        "# SCOTUS Frame Contrast Probe",
        "",
        "## Method Note",
        "",
        "This is a curated contrastive frame probe, not a source-opinion justice probe. "
        "It tests whether Qwen exposes clean directions for specific legal frames after the broad justice-level directions failed causal promotion.",
        "",
        "The examples intentionally contain frame-bearing legal language, so text baselines are expected to be high. "
        "A passing activation probe here only nominates frame directions for a small causal pilot; it does not prove a judicial circuit.",
        "",
        "## Configuration",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Started", manifest["started_at"]],
                ["Finished", manifest.get("finished_at", "")],
                ["Model", manifest["model_path"]],
                ["Layers", ", ".join(str(layer) for layer in manifest["layers"])],
                ["Rows", len(examples)],
                ["Prompt template", manifest["prompt_template"]],
                ["Use chat template", manifest["use_chat_template"]],
                ["C grid", ", ".join(str(c) for c in manifest["c_grid"])],
            ],
        ),
        "",
        "## Label Counts",
        "",
        markdown_table(["Task", "Split", "Label", "N"], label_count_rows(examples)),
        "",
        "## Best Results",
        "",
        markdown_table(
            [
                "Task",
                "Region",
                "Layer",
                "C",
                "Dev BA",
                "Test BA",
                "Text test BA",
                "Direction",
            ],
            task_report_rows(task_results),
        ),
        "",
        "## Sweep Distribution",
        "",
        markdown_table(
            ["Task", "Configs", "Median dev BA", "Dev >=0.75", "Median test BA", "Test >=0.75"],
            distribution_rows,
        ),
        "",
        "## Top Configs",
        "",
        markdown_table(["Task", "Region", "Layer", "C", "Dev BA", "Diagnostic test BA"], top_rows),
        "",
        "## Next Gate",
        "",
        "Only run a causal pilot for a frame direction if it is robust across prompt variants or if a same-prompt prompt-only/text baseline is explicitly accepted as the target. "
        "The first causal pilot should use neutral legal prompts and prompt-matched same-layer random controls.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe curated SCOTUS legal-frame contrasts.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--tasks", default="article3_private_vs_public,fourth_digital_vs_incident")
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--device-map", default="single")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--prompt-template", default="plain")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--c-grid", default=DEFAULT_C_GRID)
    parser.add_argument("--classifier-solver", default="lbfgs", choices=["lbfgs", "liblinear", "saga", "sgd"])
    parser.add_argument("--classifier-max-iter", type=int, default=1000)
    parser.add_argument("--classifier-tol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_names = [part.strip() for part in args.tasks.split(",") if part.strip()]
    unknown = [name for name in task_names if name not in TASKS]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    c_grid = [float(part) for part in args.c_grid.split(",") if part.strip()]

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = args.features_dir or (args.output_root / f"scotus_frame_contrast_probe_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = build_examples(task_names)
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.dataset_output, examples)
    write_jsonl(out_dir / "probe_examples.jsonl", examples)

    if args.features_dir and (out_dir / "manifest.json").exists():
        import json

        manifest: dict[str, Any] = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["resumed_at"] = now_iso()
    else:
        manifest = {
            "started_at": now_iso(),
            "model_path": str(args.model_path),
            "output_dir": str(out_dir),
            "dataset_output": str(args.dataset_output),
            "tasks": task_names,
            "layers_spec": args.layers,
            "device_map": args.device_map,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "prompt_template": args.prompt_template,
            "use_chat_template": bool(args.use_chat_template),
            "c_grid": c_grid,
            "classifier": {
                "solver": args.classifier_solver,
                "max_iter": args.classifier_max_iter,
                "tol": args.classifier_tol,
            },
        }
    write_json(out_dir / "manifest.json", manifest)

    if args.features_dir is not None:
        extracted = load_feature_artifacts(out_dir)
    else:
        extracted = capture_features(
            examples,
            model_path=args.model_path,
            device_map=args.device_map,
            layers_spec=args.layers,
            batch_size=args.batch_size,
            max_length=args.max_length,
            template_variant=args.prompt_template,
            use_chat_template=args.use_chat_template,
            out_dir=out_dir,
        )
    manifest["layers"] = extracted["layers"]

    task_results: dict[str, dict[str, Any]] = {}
    for task_name in task_names:
        task_out = out_dir / task_name
        task_out.mkdir(parents=True, exist_ok=True)
        task_extracted = subset_extracted(extracted, task_name)
        task_examples = [row for row in examples if row["frame_task"] == task_name]
        text_baseline = evaluate_text_baseline(task_examples, template_variant=args.prompt_template)
        probe = select_probe(
            task_extracted["regions"],
            task_extracted["meta_rows"],
            task_extracted["labels"],
            c_grid,
            classifier_solver=args.classifier_solver,
            classifier_max_iter=args.classifier_max_iter,
            classifier_tol=args.classifier_tol,
            test_diagnostic_refit=False,
        )
        for split, rows in probe["predictions"].items():
            write_jsonl(task_out / f"{split}_predictions.jsonl", rows)
        write_jsonl(task_out / "searches.jsonl", probe["searches"])
        write_json(task_out / "text_baseline.json", text_baseline)
        direction_path = task_out / "direction.npz"
        save_raw_direction(
            direction_path,
            clf=probe["final_clf"],
            best=probe["best"],
            task_name=task_name,
            positive_label=TASKS[task_name]["positive_label_name"],
        )
        task_results[task_name] = {
            "probe": probe,
            "text_baseline": text_baseline,
            "direction_path": str(direction_path),
        }

    manifest["finished_at"] = now_iso()
    write_json(out_dir / "manifest.json", manifest)
    write_report(out_dir / "report.md", manifest=manifest, examples=examples, task_results=task_results)
    gc.collect()
    print(f"Wrote {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
