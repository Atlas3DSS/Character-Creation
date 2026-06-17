#!/usr/bin/env python3
"""Build and audit a controlled no-persona SCOTUS replay bank.

The v1 curated frame bank was useful but too small and template-heavy. This
builder creates paired assistant replays where each neutral fact prompt is
identical across labels, while the assistant answer takes one of two legal
reasoning frames. It is a candidate source for activation probes, not steering
evidence.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "scotus" / "scotus_controlled_replay_v2_examples_20260501.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "scotus" / "scotus_controlled_replay_v2_manifest_20260501.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_controlled_replay_v2_audit_20260501.md"


ARTICLE3_FACTS = [
    "a federal benefits program assigns disputed eligibility determinations to an agency tribunal",
    "a securities regulator seeks a substantial civil penalty after a contested administrative hearing",
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
    "a government licensing dispute turns on a newly created statutory entitlement",
    "a wage-benefit claim created by federal statute is routed through an agency",
    "a private nuisance claim is transferred to an administrative compensation board",
    "a debtor's state-law counterclaim is decided as part of bankruptcy administration",
    "a regulator orders restitution after finding a business violated a federal market rule",
    "a patent-like damages counterclaim is decided inside an executive review board",
    "a workplace-safety board imposes a monetary sanction after a formal agency hearing",
    "a public grant repayment dispute is assigned to a departmental appeals board",
]


PRIVATE_REASONS = [
    (
        "The claim fixes liability of the sort historically resolved by courts, not merely the administration "
        "of a public program."
    ),
    (
        "Congress may create procedures, but it may not move a traditional suit between private parties into "
        "a forum lacking Article III independence for final judgment."
    ),
    (
        "The structural guarantee protects the judicial forum when the dispute resembles common-law damages, "
        "contract, property, or tort liability."
    ),
    (
        "Calling the proceeding administrative does not change the substance: the adjudicator is resolving a "
        "private-rights controversy that belongs with an independent judge."
    ),
    (
        "The point is not party preference but separation of powers; final resolution of this private claim "
        "must remain in a constitutionally protected court."
    ),
    (
        "A limited review path cannot cure the assignment if the initial forum conclusively decides the core "
        "private liability question."
    ),
]


PUBLIC_REASONS = [
    (
        "The matter arises from a federal statutory scheme and concerns the administration of a public program "
        "rather than an ordinary common-law suit."
    ),
    (
        "Congress may assign this kind of public-rights determination to a specialized tribunal as an incident "
        "of implementing its regulatory design."
    ),
    (
        "The dispute is between the government and a regulated participant over obligations created by federal "
        "law, so agency adjudication is permissible subject to appropriate review."
    ),
    (
        "The adjudicator is applying a congressionally created entitlement or enforcement regime, not exercising "
        "the full judicial power over a freestanding private claim."
    ),
    (
        "Article III does not forbid Congress from using expert administrative processes for public regulatory "
        "matters that it could structure outside ordinary courts."
    ),
    (
        "Judicial review preserves the constitutional role of the courts while allowing the initial decision to "
        "remain inside the statutory program."
    ),
]


SURFACE_STYLES = [
    ("Conclusion:", "Reasoning:"),
    ("Disposition:", "Analysis:"),
    ("Holding:", "Rationale:"),
    ("Result:", "Explanation:"),
    ("Judgment:", "Grounds:"),
    ("Answer:", "Reason:"),
]


CUE_PATTERNS = [
    (r"\bArticle III\b", "[COURT_POWER]"),
    (r"\bnon-Article III\b", "[COURT_POWER]"),
    (r"\bpublic[- ]rights?\b", "[RIGHTS_FRAME]"),
    (r"\bprivate[- ]rights?\b", "[RIGHTS_FRAME]"),
    (r"\bpublic program\b", "[PROGRAM]"),
    (r"\bprivate parties\b", "[PARTIES]"),
    (r"\bprivate claim\b", "[CLAIM]"),
    (r"\bprivate liability\b", "[CLAIM]"),
    (r"\bcommon[- ]law\b", "[LEGAL_SOURCE]"),
    (r"\badministrative\b", "[FORUM]"),
    (r"\bagenc(?:y|ies)\b", "[FORUM]"),
    (r"\btribunal\b", "[FORUM]"),
    (r"\badjudicat(?:e|es|ed|ion|or)\b", "[DECIDE]"),
    (r"\bjudg(?:e|ment|ment's|es)\b", "[COURT_ACTOR]"),
    (r"\bcourts?\b", "[COURT]"),
    (r"\bjudicial\b", "[COURT]"),
    (r"\bCongress\b", "[LEGISLATURE]"),
    (r"\bcongressional(?:ly)?\b", "[LEGISLATURE]"),
    (r"\bfederal\b", "[GOVERNMENT]"),
    (r"\bstatutory\b", "[LEGAL_SOURCE]"),
    (r"\bregulatory\b", "[REGIME]"),
    (r"\bseparation of powers\b", "[STRUCTURE]"),
    (r"\bindependent\b", "[STRUCTURE]"),
    (r"\bdamages\b", "[REMEDY]"),
    (r"\bpenalt(?:y|ies)\b", "[REMEDY]"),
    (r"\bliability\b", "[REMEDY]"),
    (r"\bentitlement\b", "[BENEFIT]"),
    (r"\brestitution\b", "[REMEDY]"),
]


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    issue_area: str
    positive_label_name: str
    negative_label_name: str
    facts: tuple[str, ...]


TASKS = {
    "article3_private_vs_public": TaskSpec(
        task_name="article3_private_vs_public",
        issue_area="Judicial Power",
        positive_label_name="article3_private_rights",
        negative_label_name="article3_public_rights",
        facts=tuple(ARTICLE3_FACTS),
    )
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(cell(header) for header in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def split_for_index(index: int) -> str:
    if index < 16:
        return "train"
    if index < 20:
        return "dev"
    return "test"


def user_prompt(fact: str) -> str:
    return (
        f"Fact pattern: {fact}.\n\n"
        "Write a concise judicial analysis of whether this matter may be finally adjudicated outside an "
        "Article III court. Address the constitutional structure without mentioning any justice by name."
    )


def assistant_text(label: int, variant: int) -> str:
    conclusion, reasoning = SURFACE_STYLES[variant % len(SURFACE_STYLES)]
    if label == 1:
        outcomes = [
            "Final adjudication must remain in an Article III court.",
            "The assignment is unconstitutional as applied to this dispute.",
            "The objection to non-Article III final judgment should be sustained.",
            "The matter cannot be conclusively resolved by the assigned tribunal.",
            "The judicial-power objection has force here.",
            "The forum assignment crosses the constitutional line.",
        ]
        reason = PRIVATE_REASONS[variant % len(PRIVATE_REASONS)]
    else:
        outcomes = [
            "Congress may assign this determination to the statutory forum.",
            "The non-Article III process is constitutionally permissible.",
            "The objection to specialized adjudication should be rejected.",
            "The matter may proceed in the assigned tribunal subject to review.",
            "The Article III challenge is not persuasive on these facts.",
            "The forum assignment stays within the public-rights exception.",
        ]
        reason = PUBLIC_REASONS[variant % len(PUBLIC_REASONS)]
    return f"{conclusion} {outcomes[variant % len(outcomes)]}\n\n{reasoning} {reason}"


def mask_cues(text: str) -> str:
    masked = text
    for pattern, repl in CUE_PATTERNS:
        masked = re.sub(pattern, repl, masked, flags=re.IGNORECASE)
    masked = re.sub(r"\s+", " ", masked).strip()
    return masked


def build_examples(task_names: list[str], variants_per_fact: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        task = TASKS[task_name]
        for fact_idx, fact in enumerate(task.facts):
            split = split_for_index(fact_idx)
            prompt = user_prompt(fact)
            prompt_masked = mask_cues(prompt)
            for variant in range(variants_per_fact):
                surface_style_id = variant % len(SURFACE_STYLES)
                for label in (0, 1):
                    label_name = task.positive_label_name if label == 1 else task.negative_label_name
                    answer = assistant_text(label, variant)
                    example_id = f"{task.task_name}|{fact_idx:02d}|v{variant:02d}|{label_name}"
                    text = f"User: {prompt}\n\nAssistant: {answer}"
                    rows.append(
                        {
                            "example_id": example_id,
                            "chunk_id": example_id,
                            "pair_id": f"{task.task_name}|{fact_idx:02d}|v{variant:02d}",
                            "fact_id": f"{task.task_name}_{fact_idx:02d}",
                            "split": split,
                            "label": int(label),
                            "justice": label_name,
                            "positive_justice": task.positive_label_name,
                            "frame_task": task.task_name,
                            "frame_label": label_name,
                            "issue_area_label": task.issue_area,
                            "opinion_type": "controlled_legal_replay_v2",
                            "section_posture": "assistant_replay",
                            "surface_style_id": surface_style_id,
                            "answer_variant": variant,
                            "prompt": prompt,
                            "prompt_cue_masked": prompt_masked,
                            "assistant_text": answer,
                            "assistant_cue_masked": mask_cues(answer),
                            "text": text,
                            "text_cue_masked": mask_cues(text),
                        }
                    )
    rows.sort(key=lambda row: (row["frame_task"], row["fact_id"], row["answer_variant"], row["label"]))
    return rows


def split_indices(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[str(row["split"])].append(idx)
    return {split: np.array(indices, dtype=np.int64) for split, indices in grouped.items()}


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def tfidf_baseline(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    idx = split_indices(rows)
    required = {"train", "dev", "test"}
    if set(idx) & required != required:
        raise RuntimeError(f"Missing required split for {field}: {sorted(idx)}")
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    texts = [str(row[field]) for row in rows]
    train_dev_idx = np.concatenate([idx["train"], idx["dev"]])
    clf = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=8000)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", C=1.0)),
        ]
    )
    clf.fit([texts[i] for i in train_dev_idx.tolist()], labels[train_dev_idx])
    out: dict[str, dict[str, float]] = {}
    for split, split_idx in sorted(idx.items()):
        pred = clf.predict([texts[i] for i in split_idx.tolist()])
        out[split] = metric_row(labels[split_idx], pred)
    return out


def numeric_baseline(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    idx = split_indices(rows)
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    values = np.array([[float(len(str(row[field]).split())), float(len(str(row[field])))] for row in rows], dtype=np.float32)
    train_dev_idx = np.concatenate([idx["train"], idx["dev"]])
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", C=1.0)),
        ]
    )
    clf.fit(values[train_dev_idx], labels[train_dev_idx])
    out: dict[str, dict[str, float]] = {}
    for split, split_idx in sorted(idx.items()):
        pred = clf.predict(values[split_idx])
        out[split] = metric_row(labels[split_idx], pred)
    return out


def categorical_baseline(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    idx = split_indices(rows)
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    out: dict[str, dict[str, float]] = {}
    train_dev_idx = np.concatenate([idx["train"], idx["dev"]])
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for item_idx in train_dev_idx.tolist():
        counts[str(rows[item_idx][field])][int(labels[item_idx])] += 1
    default = Counter(int(labels[item_idx]) for item_idx in train_dev_idx.tolist()).most_common(1)[0][0]
    for split, split_idx in sorted(idx.items()):
        pred: list[int] = []
        for item_idx in split_idx.tolist():
            key = str(rows[item_idx][field])
            if key in counts and counts[key]:
                pred.append(int(counts[key].most_common(1)[0][0]))
            else:
                pred.append(int(default))
        out[split] = metric_row(labels[split_idx], np.array(pred, dtype=np.int64))
    return out


def run_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["prompt", "prompt_cue_masked", "assistant_text", "assistant_cue_masked", "text", "text_cue_masked"]
    baselines = {field: tfidf_baseline(rows, field) for field in fields}
    baselines["assistant_length"] = numeric_baseline(rows, "assistant_text")
    baselines["surface_style_id"] = categorical_baseline(rows, "surface_style_id")

    fact_split_map: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        fact_split_map[str(row["fact_id"])].add(str(row["split"]))
    split_conflicts = sorted(fact_id for fact_id, splits in fact_split_map.items() if len(splits) > 1)
    prompt_label_counts = Counter((row["prompt"], int(row["label"])) for row in rows)
    prompt_hashes = Counter(row["prompt"] for row in rows)
    return {
        "row_count": len(rows),
        "split_label_counts": {
            f"{split}/{label}": count
            for (split, label), count in sorted(Counter((row["split"], int(row["label"])) for row in rows).items())
        },
        "task_counts": {
            f"{task}/{split}/{label}": count
            for (task, split, label), count in sorted(
                Counter((row["frame_task"], row["split"], int(row["label"])) for row in rows).items()
            )
        },
        "unique_facts": len({row["fact_id"] for row in rows}),
        "unique_prompts": len({row["prompt"] for row in rows}),
        "prompt_label_pairs": len(prompt_label_counts),
        "max_rows_per_prompt": max(prompt_hashes.values()) if prompt_hashes else 0,
        "fact_split_conflicts": split_conflicts,
        "baselines": baselines,
    }


def baseline_table_rows(audit: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for field, split_metrics in sorted(audit["baselines"].items()):
        test = split_metrics["test"]
        dev = split_metrics["dev"]
        rows.append(
            [
                field,
                dev["n"],
                f"{dev['balanced_accuracy']:.3f}",
                test["n"],
                f"{test['balanced_accuracy']:.3f}",
                f"{test['f1']:.3f}",
            ]
        )
    return rows


def write_report(path: Path, *, rows: list[dict[str, Any]], audit: dict[str, Any], manifest: dict[str, Any]) -> None:
    count_rows = [
        [key.split("/")[0], key.split("/")[1], key.split("/")[2], value]
        for key, value in sorted(audit["task_counts"].items())
    ]
    sample_rows = [
        [
            row["example_id"],
            row["split"],
            row["frame_label"],
            row["surface_style_id"],
            row["prompt"][:80] + ("..." if len(row["prompt"]) > 80 else ""),
            row["assistant_text"][:110] + ("..." if len(row["assistant_text"]) > 110 else ""),
        ]
        for row in rows[:8]
    ]
    prompt_test = audit["baselines"]["prompt"]["test"]["balanced_accuracy"]
    masked_test = audit["baselines"]["assistant_cue_masked"]["test"]["balanced_accuracy"]
    length_test = audit["baselines"]["assistant_length"]["test"]["balanced_accuracy"]
    style_test = audit["baselines"]["surface_style_id"]["test"]["balanced_accuracy"]
    gate = (
        "activation_candidate"
        if prompt_test <= 0.60 and length_test <= 0.65 and style_test <= 0.60 and not audit["fact_split_conflicts"]
        else "review"
    )
    lines = [
        "# SCOTUS Controlled Replay v2 Audit",
        "",
        "## Purpose",
        "",
        "Build a cleaner no-persona replay source after the Commerce replay family failed causal promotion. "
        "Each fact prompt is paired across both legal frames, so prompt-only text should not identify the label.",
        "",
        "This is not steering evidence. It is a candidate source for a later activation probe and causal gate.",
        "",
        "## Configuration",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Created", manifest["created_at"]],
                ["Dataset", manifest["dataset_path"]],
                ["Rows", len(rows)],
                ["Tasks", ", ".join(manifest["tasks"])],
                ["Variants per fact", manifest["variants_per_fact"]],
                ["Gate", gate],
            ],
        ),
        "",
        "## Counts",
        "",
        markdown_table(["Task", "Split", "Label", "Rows"], count_rows),
        "",
        "## Leakage Baselines",
        "",
        markdown_table(["Field", "Dev N", "Dev BA", "Test N", "Test BA", "Test F1"], baseline_table_rows(audit)),
        "",
        "## Read",
        "",
        f"- Prompt-only test BA is `{prompt_test:.3f}`; this should remain near chance because prompts are paired across labels.",
        f"- Cue-masked assistant test BA is `{masked_test:.3f}`; this is expected to stay high when the answer still states the legal proposition.",
        f"- Length-only test BA is `{length_test:.3f}` and surface-style test BA is `{style_test:.3f}`.",
        f"- Fact split conflicts: `{len(audit['fact_split_conflicts'])}`.",
        "- If promoted to activation capture, use assistant-internal regions and treat answer-text separability as expected answer-state evidence, not proof of a circuit.",
        "- The later causal gate must use neutral no-persona prompts and prompt-matched random/source controls.",
        "",
        "## Sample Rows",
        "",
        markdown_table(["Example", "Split", "Label", "Style", "Prompt", "Assistant"], sample_rows),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tasks", default="article3_private_vs_public")
    parser.add_argument("--variants-per-fact", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_names = [part.strip() for part in args.tasks.split(",") if part.strip()]
    unknown = sorted(set(task_names) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}; available={sorted(TASKS)}")
    rows = build_examples(task_names, args.variants_per_fact)
    audit = run_audit(rows)
    manifest = {
        "created_at": now_iso(),
        "dataset_path": str(args.dataset_output),
        "manifest_path": str(args.manifest_output),
        "report_path": str(args.report_output),
        "tasks": task_names,
        "variants_per_fact": int(args.variants_per_fact),
        "audit": audit,
        "success_standard": "No persona prompting. This bank can only nominate candidates; promotion requires no-mask causal generation and reasoning-trace checks where available.",
    }
    write_jsonl(args.dataset_output, rows)
    write_json(args.manifest_output, manifest)
    write_report(args.report_output, rows=rows, audit=audit, manifest=manifest)
    print(f"Wrote {args.dataset_output}")
    print(f"Wrote {args.manifest_output}")
    print(f"Wrote {args.report_output}")


if __name__ == "__main__":
    main()
