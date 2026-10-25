"""Audit six baseline shards without loading models or changing run artifacts.

Reports missing/duplicate rows, provenance mismatches, incomplete generations and
invalid IPIP replies independently. IPIP scores describe model self-report under
the tested prompts, not human personality or a demonstrated causal intervention.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

from persona_experiments import score_ipip50


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ipip_digit(answer: str) -> int | None:
    value = answer.strip()
    return int(value) if re.fullmatch(r"[1-5]", value) else None


def capability_answer(answer: str, expected: str) -> dict[str, Any]:
    """Grade the explicit final numeric answer only, never a matching substring."""
    matches = re.findall(r"(?im)\bANSWER\s*:\s*([^\n]+)", answer)
    if not matches:
        return {"parseable": False, "correct": False, "answer": None}
    actual = matches[-1].replace("*", "").replace("`", "").strip()
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?(?:\s*/\s*[+-]?\d+)?)\s*(?:km/h)?[.!]?", actual, re.I)
    if not match:
        return {"parseable": False, "correct": False, "answer": actual}
    numeric = re.sub(r"\s+", "", match.group(1))
    try:
        correct = Fraction(numeric) == Fraction(expected)
    except (ValueError, ZeroDivisionError):
        return {"parseable": False, "correct": False, "answer": actual}
    if "/" in expected:
        correct = correct and numeric == expected  # The prompt specifically requests a reduced fraction.
    return {"parseable": True, "correct": correct, "answer": actual}


def latest_attempt(manifest: dict[str, Any], outcome: dict[str, Any] | None,
                   events: list[dict[str, Any]]) -> tuple[dict, dict | None, dict]:
    """A previous startup failure is historical while a newer attempt is running."""
    starts = [event for event in events if event.get("event") == "start"]
    latest = max(starts, key=lambda e: e["started_at"]) if starts else manifest
    started = latest["started_at"]
    finishes = [event for event in events if event.get("event") == "finish" and event.get("finished_at", 0) >= started]
    if outcome and outcome.get("finished_at", 0) >= started:
        finishes.append(outcome)
    current_outcome = max(finishes, key=lambda e: e["finished_at"]) if finishes else None
    state = "running" if current_outcome is None else "failed" if current_outcome.get("errors") else "finished"
    return ({k: v for k, v in latest.items() if k != "event"}, current_outcome,
            {"shard": latest["shard"], "state": state, "started_at": started,
             "finished_at": current_outcome.get("finished_at") if current_outcome else None,
             "lease": latest.get("lease"), "attempt_starts_recorded": len(starts),
             "prior_outcome_ignored": bool(outcome and outcome.get("finished_at", 0) < started),
             "state_source": "artifact_events_not_live_process_inspection"})


def analyze(plan: dict[str, Any], rows: list[dict[str, Any]], *,
            manifests: list[dict[str, Any]] | None = None,
            outcomes: list[dict[str, Any]] | None = None,
            read_errors: list[dict[str, Any]] | None = None,
            shard_states: list[dict[str, Any]] | None = None,
            expected_server_model: str | None = None) -> dict[str, Any]:
    jobs = plan["jobs"]
    expected = {job["id"]: job for job in jobs}
    issues: list[str] = []
    if len(expected) != len(jobs):
        issues.append("plan_contains_duplicate_job_ids")
    if any(digest({k: v for k, v in j.items() if k != "id"}) != j["id"] for j in jobs):
        issues.append("plan_job_content_digest_mismatch")
    manifests = manifests or []
    outcomes = outcomes or []
    plan_hash = digest(plan)
    expected_model = expected_server_model or plan["model"].get("served_model_id") or plan["model"]["model"]
    shard_ids = [m.get("shard") for m in manifests]
    if len(shard_ids) != 6 or not all(type(s) is int for s in shard_ids) or set(shard_ids) != set(range(6)):
        issues.append("six_unique_shard_manifests_not_present")
    for manifest in manifests:
        shard = manifest.get("shard")
        if type(shard) is not int or not 0 <= shard < 6:
            issues.append("manifest_invalid_shard")
            continue
        if manifest.get("shards") != 6 or manifest.get("plan_sha256") != plan_hash:
            issues.append(f"manifest_{shard}_plan_or_shard_count_mismatch")
        if manifest.get("model") != plan["model"]:
            issues.append(f"manifest_{shard}_model_mismatch")
        wanted = {j for j in expected if int(j, 16) % 6 == shard}
        actual = manifest.get("job_ids", [])
        if set(actual) != wanted or len(actual) != len(set(actual)):
            issues.append(f"manifest_{shard}_job_assignment_mismatch")
    for field in ("package_sha256", "executable_sha256"):
        values = {m.get(field) for m in manifests}
        if len(values) != 1 or not all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in values):
            issues.append(f"manifest_{field}_missing_or_inconsistent")
    counts: Counter[str] = Counter()
    finishes: Counter[str] = Counter()
    response_models: Counter[str] = Counter()
    condition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    expected_groups = sorted({(j["condition"], j["seed"]) for j in jobs if j["kind"] == "ipip"})
    answers: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    seen: Counter[str] = Counter()
    row_problems: list[dict[str, Any]] = []
    capabilities = []
    for row in rows:
        counts["rows"] += 1
        job = row.get("job", {})
        identity = job.get("id")
        if identity not in expected:
            counts["unknown_job_rows"] += 1
            row_problems.append({"id": identity, "problem": "unknown_job"})
            continue
        seen[identity] += 1
        if seen[identity] > 1:
            counts["duplicate_rows"] += 1
            continue  # Never count repeated rows as independent measurements.
        wanted = expected[identity]
        cc = condition_counts[wanted["condition"]]
        cc["rows"] += 1
        problems = []
        if "audit_source_shard" in row and int(identity, 16) % 6 != row["audit_source_shard"]:
            problems.append("result_in_wrong_shard")
        if job != wanted:
            problems.append("job_content_mismatch")
        request = row.get("request", {})
        if request.get("seed") != wanted["seed"]:
            problems.append("seed_mismatch")
        if request.get("max_tokens") != wanted["max_tokens"]:
            problems.append("token_budget_mismatch")
        expected_messages = ([{"role": "system", "content": wanted["system"]}] if wanted.get("system") else [])
        expected_messages.append({"role": "user", "content": wanted["prompt"]})
        if request.get("messages") != expected_messages:
            problems.append("prompt_or_system_mismatch")
        if request.get("chat_template_kwargs", {}).get("enable_thinking") is not wanted.get("thinking", False):
            problems.append("thinking_request_mismatch")
        if request.get("steering"):
            problems.append("unexpected_baseline_steering")
        if any(request.get(k) != v for k, v in {"temperature": .7, "top_p": .95, "top_k": 20}.items()):
            problems.append("sampling_config_mismatch")
        if request.get("model") != expected_model:
            problems.append("served_model_id_not_verified")
        response = row.get("response", {})
        choices = response.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        response_models[str(response.get("model"))] += 1
        if response and response.get("model") != request.get("model"):
            problems.append("response_request_model_mismatch")
        answer = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if row.get("answer", "") != answer or row.get("reasoning", "") != reasoning:
            problems.append("saved_text_response_mismatch")
        finish = choice.get("finish_reason")
        finishes[str(finish)] += 1
        errored = bool(row.get("error"))
        truncated = finish in ("length", "max_tokens")
        thinking_output = bool(reasoning.strip()) or bool(re.search(r"</?think>", answer))
        if not wanted.get("thinking", False) and thinking_output:
            problems.append("thinking_observed_when_disabled")
        complete = not errored and finish == "stop" and bool(answer.strip()) and not thinking_output
        if bool(row.get("complete")) != complete:
            problems.append("saved_completeness_disagrees_with_audit")
        counts["request_errors"] += errored
        counts["truncated"] += truncated
        counts["complete_final_answers"] += complete
        counts["thinking_output"] += thinking_output
        cc["request_errors"] += errored
        cc["truncated"] += truncated
        cc["complete_final_answers"] += complete
        counts["rows_with_identity_or_protocol_issues"] += bool(problems)
        if problems:
            row_problems.append({"id": identity, "problem": problems})
        usable = complete and not problems
        if wanted["kind"] == "ipip":
            value = ipip_digit(answer)
            counts["ipip_rows"] += 1
            counts["ipip_invalid_format"] += value is None
            cc["ipip_invalid_format"] += value is None
            if usable and value is not None:
                answers[(wanted["condition"], wanted["seed"])][str(wanted["item_id"])] = value
                counts["ipip_usable"] += 1
        elif wanted.get("expected_answer") is not None:
            score = capability_answer(answer, wanted["expected_answer"])
            capabilities.append({"id": identity, "condition": wanted["condition"], "seed": wanted["seed"],
                                 "usable": usable, "expected": wanted["expected_answer"], **score})
            cc["capability_rows"] += 1
            cc["capability_usable"] += usable
            cc["capability_correct"] += usable and score["correct"]
            cc["capability_parse_failures"] += usable and not score["parseable"]

    # Duplicate rows invalidate the affected inventory instead of silently choosing a winner.
    for identity, repetitions in seen.items():
        if repetitions > 1 and expected[identity]["kind"] == "ipip":
            job = expected[identity]
            answers[(job["condition"], job["seed"])].pop(str(job["item_id"]), None)
    inventories = []
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for condition, seed in expected_groups:
        score = score_ipip50(answers[(condition, seed)])
        inventories.append({"condition": condition, "seed": seed, **score})
        if score["complete"]:
            for trait, item in score["scores"].items():
                values[condition][trait].append(item["sum_10_to_50"])
    profiles = {}
    for condition in sorted({j["condition"] for j in jobs}):
        traits = {}
        for trait, samples in values[condition].items():
            traits[trait] = {"n_complete_seeds": len(samples), "mean": statistics.mean(samples),
                             "sample_sd": statistics.stdev(samples) if len(samples) > 1 else None,
                             "standard_error": statistics.stdev(samples) / math.sqrt(len(samples)) if len(samples) > 1 else None,
                             "observed_range": [min(samples), max(samples)]}
        profiles[condition] = {"ipip": traits, "generation_counts": dict(condition_counts[condition])}
    missing = sorted(set(expected) - set(seen))
    if missing:
        issues.append("expected_rows_missing")
    if counts["duplicate_rows"] or counts["unknown_job_rows"] or row_problems:
        issues.append("row_identity_or_protocol_validation_failed")
    if len(outcomes) != 6:
        issues.append("six_worker_outcomes_not_present")
    if any(o.get("errors") or o.get("rows") != o.get("expected_rows") for o in outcomes):
        issues.append("worker_outcome_errors_or_incomplete_rows")
    if read_errors:
        issues.append("result_files_contain_unreadable_rows")
    complete = not issues and counts["request_errors"] == 0
    return {
        "schema": "character-creation.baseline-audit.v1", "model": plan["model"], "plan_sha256": plan_hash,
        "status": "complete_audited" if complete else "incomplete_or_invalid",
        "expected_rows": len(expected), "observed_unique_rows": len(seen), "missing_rows": len(missing),
        "missing_job_ids": missing, "counts": dict(counts), "finish_reasons": dict(finishes),
        "expected_served_model": expected_model, "response_models": dict(response_models),
        "issues": sorted(set(issues)), "row_problems": row_problems, "read_errors": read_errors or [],
        "profiles": profiles, "ipip_inventories": inventories, "capability_answers": capabilities,
        "shard_states": shard_states or [],
        "interpretation": {
            "causal_steering_test": False, "human_personality_claim": False,
            "ipip": "Prompt-conditioned model self-report; all 50 valid complete items required per seed.",
            "uncertainty": "Across-seed sample SD, standard error and observed range are descriptive; three seeds do not establish population reliability.",
            "capability": "Exact explicit final numeric answer only; explanations are not independently graded.",
            "heldout": "Behavior prompts are held out from IPIP scoring; prompt-only results do not validate activation interventions.",
            "package": "Manifest hash consistency is checked; on-disk package authenticity remains the launcher's responsibility.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True, help="Folder containing six baseline-shardN JSONL files and manifests")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--server-model-id", help="Explicit verified API model ID if different from the checkpoint's model name")
    parser.add_argument("--output", type=Path, help="New JSON output file; existing files are never overwritten; default stdout")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("label must contain only letters, numbers, underscores and hyphens")
    plan = json.loads(args.plan.read_text())
    rows, manifests, outcomes, errors, states = [], [], [], [], []
    for shard in range(6):
        stem = args.results / f"{args.label}-shard{shard}"
        manifest_path = stem.with_suffix(".manifest.json")
        outcome_path = stem.with_suffix(".outcome.json")
        attempt_path = stem.with_suffix(".attempts.jsonl")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else None
            events = []
            if attempt_path.exists():
                for number, line in enumerate(attempt_path.read_text().splitlines(), 1):
                    try:
                        events.append(json.loads(line))
                    except ValueError as error:
                        errors.append({"file": attempt_path.name, "line": number, "error": str(error)})
            current_manifest, current_outcome, state = latest_attempt(manifest, outcome, events)
            manifests.append(current_manifest)
            states.append(state)
            if current_outcome is not None:
                outcomes.append(current_outcome)
        path = stem.with_suffix(".jsonl")
        if path.exists():
            with path.open() as stream:
                for number, line in enumerate(stream, 1):
                    try:
                        row = json.loads(line)
                    except ValueError as error:
                        errors.append({"file": path.name, "line": number, "error": str(error)})
                        continue
                    if not isinstance(row, dict):
                        errors.append({"file": path.name, "line": number, "error": "row must be a JSON object"})
                        continue
                    row["audit_source_shard"] = shard
                    rows.append(row)
    report = analyze(plan, rows, manifests=manifests, outcomes=outcomes, read_errors=errors,
                     shard_states=states, expected_server_model=args.server_model_id)
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(output)
    else:
        print(output, end="")
    return 0 if report["status"] == "complete_audited" else 2


if __name__ == "__main__":
    raise SystemExit(main())
