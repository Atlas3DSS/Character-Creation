"""Operator-only, unsteered prefill coverage of every selected registered row.

Plan/status print counts and hashes only. Enqueue stores the original private
prompt snapshots in the existing manager queue; it does not start GPU work.
This schema is intentionally separate from adaptive intervention proposals.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re
import sqlite3

from persona_manager import Manager, Queue, file_hash, load_corpus, seed_for, uint
from persona_vectors import ModelSpec, VectorBank, canonical_digest


SCHEMA = "character-creation.operator-coverage.v1"
ROOT = Path(__file__).resolve().parent


def coverage_plan(corpus, bank, corpus_sha256, *, split="discovery", seed=7035, max_jobs=100000):
    if split not in {"all", "discovery", "heldout"}:
        raise ValueError("Coverage split must be all, discovery or heldout")
    uint(seed, 0, 2**32 - 1, "seed")
    uint(max_jobs, 1, 1000000, "max_jobs")
    if not isinstance(corpus_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", corpus_sha256):
        raise ValueError("Coverage requires the complete published corpus SHA256")
    selected = sorted((row for row in corpus.values() if split == "all" or row["split"] == split),
                      key=lambda row: row["id"])
    if not selected or len(selected) > max_jobs or len({row["id"] for row in selected}) != len(selected):
        raise ValueError("Coverage requires nonempty unique rows within the job cap; no truncation")
    calibration = bank.root / "layer_00.npz"
    calibration_sha = file_hash(calibration)
    bank.layer(0)  # Model-bound summary reference; no coordinate is pushed.
    if file_hash(calibration) != calibration_sha:
        raise ValueError("Calibration changed during coverage planning")
    proposal = {"schema": SCHEMA, "model_identity": bank.model.identity,
        "purpose": "operator baseline coverage; no intervention or causal effect claim",
        "measurement": "prefill", "split": split, "batch_seed": seed,
        "corpus_sha256": corpus_sha256, "expected_rows": len(selected),
        "selected_ids_sha256": canonical_digest([row["id"] for row in selected]),
        "summary_reference_layer": 0, "calibration_sha256": calibration_sha}
    splits = sorted({row["split"] for row in selected})
    if not set(splits) <= {"discovery", "heldout"}:
        raise ValueError("Coverage must preserve explicit discovery/heldout splits")
    proposal_id = canonical_digest([proposal, splits, [corpus_sha256]])
    jobs = []
    for row in selected:
        prompt = copy.deepcopy(row)
        pair_id = prompt.get("pair_id", prompt["id"])
        experiment = {"label": "operator baseline coverage", "kind": "baseline_coverage",
                      "layer": 0, "dose": 0.0, "phase": "prefill", "measurement": "prefill",
                      "summary_reference_only": True, "intervention": False}
        job = {"schema": "character-creation.manager-job.v1", "proposal_id": proposal_id,
            "model_identity": bank.model.identity, "experiment_index": 0, "experiment": experiment,
            "arm": "baseline", "pushes": [], "phase": "prefill", "measurement": "prefill",
            "kind": "prefill_observation", "operator_coverage": True,
            "seed": seed_for(SCHEMA, seed, prompt["corpus"], pair_id), "repetition": 0,
            "prompt": prompt, "pair_id": pair_id,
            "source_group": {"corpus": prompt["corpus"], "pair_id": pair_id, "split": prompt["split"]},
            "corpus_sha256": corpus_sha256, "calibration_sha256": calibration_sha,
            "random_control_semantics": "not applicable: one unsteered operator observation per registered row"}
        job["id"] = canonical_digest(job)
        jobs.append(job)
    counts = Counter((row["corpus"], row["split"]) for row in selected)
    summary = {"schema": SCHEMA, "proposal_id": proposal_id, **proposal,
        "counts": [{"corpus": name, "split": part, "rows": count} for (name, part), count in sorted(counts.items())],
        "source_groups": len({(row["corpus"], row.get("pair_id", row["id"])) for row in selected}),
        "scope": "all layers at final prompt token, plus prompt residual norms; not every neuron's all-token trajectory",
        "heldout_policy": "original split retained; existing manager filters exclude heldout rows from adaptive feedback"}
    return proposal, jobs, summary


def published_inputs(config):
    path = Path(config["corpus"])
    manifest_path = Path(config.get("corpus_manifest", path.with_name("persona_manager_corpus_manifest.json")))
    manifest = json.loads(manifest_path.read_text())
    digest = manifest.get("output_sha256")
    if manifest.get("status") != "complete" or digest != file_hash(path):
        raise ValueError("Coverage requires a complete, hash-matching corpus publication")
    corpus = load_corpus(path)
    if file_hash(path) != digest:
        raise ValueError("Corpus changed during coverage planning")
    model = ModelSpec.load(Path(config["model"]))
    return corpus, VectorBank(Path(config["bank"]), model), digest


def coverage_status(path, summary):
    """Read only this pass, without loading or returning private prompt text."""
    result = {**summary, "queued_rows": 0, "valid_completed_observations": 0, "states": {}, "by_corpus_split": []}
    path = Path(path)
    if path.exists():
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            where = "json_extract(body,'$.proposal_id')=?"
            identity = (summary["proposal_id"],)
            result["states"] = dict(db.execute(f"SELECT state,count(*) FROM jobs WHERE {where} GROUP BY state", identity))
            result["queued_rows"] = sum(result["states"].values())
            result["by_corpus_split"] = [dict(corpus=name, split=split, state=state, rows=count)
                for name, split, state, count in db.execute(
                    f"SELECT json_extract(body,'$.prompt.corpus'),json_extract(body,'$.prompt.split'),state,count(*) FROM jobs WHERE {where} GROUP BY 1,2,3 ORDER BY 1,2,3", identity)]
            result["valid_completed_observations"] = db.execute(
                "SELECT count(DISTINCT jobs.id) FROM jobs JOIN attempts ON jobs.id=attempts.job_id "
                "WHERE json_extract(jobs.body,'$.proposal_id')=? AND jobs.state='complete' AND attempts.state='complete' "
                "AND json_extract(attempts.result,'$.complete_observation')=1 "
                "AND json_extract(attempts.result,'$.response.operation')='prefill_observation'", identity).fetchone()[0]
    result["remaining_rows"] = result["expected_rows"] - result["valid_completed_observations"]
    result["all_complete"] = result["queued_rows"] == result["expected_rows"] and result["remaining_rows"] == 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "status", "enqueue"))
    parser.add_argument("--config", type=Path, required=True, help="Existing manager configuration")
    parser.add_argument("--database", type=Path, default=ROOT / "results/persona_manager.sqlite")
    parser.add_argument("--split", choices=("all", "discovery", "heldout"), default="discovery",
                        help="Use explicit --split all for complete corpus coverage")
    parser.add_argument("--seed", type=int, default=7035)
    parser.add_argument("--max-jobs", type=int, default=100000, help="Whole queue cap, also bounded by manager max_total_jobs")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    uint(args.max_jobs, 1, 1000000, "max_jobs")
    limit = min(args.max_jobs, uint(config.get("max_total_jobs", 100000), 1, 1000000, "max_total_jobs"))
    if args.command == "enqueue":
        if not all(endpoint.get("diagnostic") is True for endpoint in config["endpoints"]):
            raise ValueError("Coverage requires diagnostic endpoints with prefill observations")
        queue = Queue(args.database)
        manager = Manager(config, queue)  # Existing identity, runtime, corpus and endpoint binding.
        corpus, bank, digest = manager.corpus, manager.bank, manager.corpus_digest
    else:
        corpus, bank, digest = published_inputs(config)
    proposal, jobs, summary = coverage_plan(corpus, bank, digest, split=args.split, seed=args.seed, max_jobs=limit)
    if args.command == "enqueue":
        # Existing Queue.enqueue commits the entire pass atomically, checks the
        # full queue cap and returns zero for an already registered proposal.
        added = queue.enqueue(proposal, jobs, limit=limit)
        print(json.dumps({**coverage_status(args.database, summary), "new_jobs": added}, indent=2))
    else:
        print(json.dumps(summary if args.command == "plan" else coverage_status(args.database, summary), indent=2))


if __name__ == "__main__":
    main()
