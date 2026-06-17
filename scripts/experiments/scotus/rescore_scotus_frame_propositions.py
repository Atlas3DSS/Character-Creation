#!/usr/bin/env python3
"""Rescore SCOTUS generation artifacts with proposition-level frame rules.

The earlier SCOTUS generation gates used raw substring counts. That is useful
for fast triage, but it overcounts broad words like "home", "consent",
"damages", "district", and "removal". This script leaves the old artifacts
untouched and writes a stricter rescoring pass where each frame is present only
when a completion states a recognizable legal proposition.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = (
    PROJECT_ROOT / "sweep_v4" / "scotus_qwen4bit_proxy_20260501_045257",
    PROJECT_ROOT / "sweep_v4" / "scotus_sae_poke_20260501_000146",
    PROJECT_ROOT / "sweep_v4" / "scotus_sae_poke_20260501_001257",
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "sweep_v4"


@dataclass(frozen=True)
class FrameRule:
    frame: str
    patterns: tuple[str, ...]
    note: str


FRAME_RULES: tuple[FrameRule, ...] = (
    FrameRule(
        frame="article3_public_rights",
        note="Public-rights doctrine, claims against the sovereign, or matters integrated into a public regulatory scheme.",
        patterns=(
            r"\bpublic rights?\b",
            r"\bclaims? against (?:the )?(?:government|united states|sovereign)\b",
            r"\b(?:public|federal) regulatory scheme\b",
            r"\bintegral to (?:a |the )?(?:federal |public )?regulatory scheme\b",
            r"\bsovereign(?:'s)? regulatory\b",
            r"\bsovereign power\b.{0,180}\b(?:immigration|border|admission|exclusion|removal)\b",
            r"\b(?:executive|sovereign) (?:function|authority|power)\b.{0,180}\b(?:immigration|removal|admission|exclusion|border)\b",
            r"\b(?:immigration|removal|deportation) proceedings?\b.{0,180}\b(?:civil|executive|sovereign|administrative)\b",
            r"\b(?:admission|exclusion|removal) of aliens\b",
            r"\bcivil regulatory (?:action|measure|proceeding)\b",
        ),
    ),
    FrameRule(
        frame="article3_private_rights",
        note="Private-rights doctrine, traditional suits at law, or common-law/private-party claims.",
        patterns=(
            r"\bprivate rights?\b",
            r"\bprivate (?:dispute|claim|suit|cause of action|counterclaim|liability)\b",
            r"\b(?:dispute|claim|suit|counterclaim) between (?:two )?private (?:parties|companies|persons)\b",
            r"\bcommon[- ]law (?:claim|cause of action|counterclaim|suit|damages)\b",
            r"\btraditional suit at law\b",
            r"\bprivate right of action\b",
        ),
    ),
    FrameRule(
        frame="article3_article1_tribunal",
        note="Non-Article-III adjudicators, Article I tribunals, legislative courts, or missing Article III tenure protections.",
        patterns=(
            r"\bnon[- ]article iii (?:court|tribunal|judge|adjudicator|forum|decisionmaker|body)\b",
            r"\barticle i (?:court|tribunal|judge|adjudicator|forum)\b",
            r"\blegislative courts?\b",
            r"\b(?:agency|administrative|bankruptcy) (?:adjudication|adjudicator|judge|tribunal)\b",
            r"\b(?:agency|administrative|bankruptcy) (?:proceeding|proceedings|hearing|hearings|tribunal|tribunals)\b.{0,180}\b(?:article iii|judicial review|fact[- ]finding|adjudicat)\b",
            r"\b(?:article iii|judicial review|fact[- ]finding|adjudicat)\b.{0,220}\badministrative (?:agency|agencies|law judges?|judges?|officers?|process|proceedings?|tribunal|tribunals)\b",
            r"\badministrative (?:agency|agencies|law judges?|judges?|officers?|process|proceedings?|tribunal|tribunals)\b.{0,220}\b(?:article iii|judicial review|fact[- ]finding|adjudicat)\b",
            r"\b(?:agency|administrative) fact[- ]finding\b",
            r"\bimmigration judges?\b.{0,180}\b(?:executive|administrative|agency|article iii|judicial review)\b",
            r"\b(?:lack|lacks|without) (?:article iii )?(?:life tenure|salary protection)\b",
            r"\blife tenure and salary protection\b.{0,120}\b(?:non[- ]article iii|article i|agency|administrative|tribunal|bankruptcy)\b",
        ),
    ),
    FrameRule(
        frame="article3_case_or_controversy",
        note="Article III case-or-controversy jurisdiction.",
        patterns=(
            r"\bcase[- ]or[- ]controversy\b",
            r"\bcases? and controversies\b",
            r"\barticle iii(?:'s)? case(?:s)?[- ]or[- ]controversy requirement\b",
        ),
    ),
    FrameRule(
        frame="article3_final_judgment_separation",
        note="Final judgments and separation of Article III judicial power.",
        patterns=(
            r"\bfinal (?:judgment|adjudication|decision)\b.{0,160}\barticle iii\b",
            r"\barticle iii\b.{0,160}\bfinal (?:judgment|adjudication|decision)\b",
            r"\bseparation of powers\b.{0,160}\bfinal (?:judgment|adjudication|decision)\b",
        ),
    ),
    FrameRule(
        frame="fourth_plain_view_closed_container",
        note="Plain view, closed containers, or privacy in a container/backpack.",
        patterns=(
            r"\bplain[- ]view\b",
            r"\bclosed container\b",
            r"\blocked (?:backpack|bag|container)\b",
            r"\b(?:backpack|bag|container)\b.{0,120}\breasonable expectation of privacy\b",
            r"\breasonable expectation of privacy\b.{0,120}\b(?:backpack|bag|container)\b",
        ),
    ),
    FrameRule(
        frame="fourth_search_incident_chimel",
        note="Search incident to arrest, Chimel, or immediate-control rationales.",
        patterns=(
            r"\bsearch(?:es)? incident to arrest\b",
            r"\bsearch[- ]incident(?:[- ]to[- ]arrest)?\b",
            r"\bchimel\b",
            r"\bimmediate control\b",
            r"\bgrabbing distance\b",
            r"\bwingspan\b",
            r"\breaching distance\b",
        ),
    ),
    FrameRule(
        frame="fourth_exigency_consent",
        note="Exigent-circumstances or true Fourth Amendment consent exceptions.",
        patterns=(
            r"\bexigent circumstances?\b",
            r"\bemergency circumstances?\b",
            r"\bconsent exception\b",
            r"\bvoluntary consent\b",
            r"\bconsent(?:ed)? to (?:the )?(?:search|entry|seizure)\b",
        ),
    ),
    FrameRule(
        frame="fourth_safety_evidence",
        note="Officer safety and evidence-preservation rationales.",
        patterns=(
            r"\bofficer safety\b",
            r"\bdestruction of evidence\b",
            r"\bevidence destruction\b",
            r"\bprevent(?:ing)? (?:the )?(?:destruction|loss) of evidence\b",
            r"\bpreserv(?:e|ing|ation) (?:of )?evidence\b",
        ),
    ),
    FrameRule(
        frame="fourth_digital_privacy",
        note="Digital-device contents under Riley-style privacy reasoning.",
        patterns=(
            r"\bdigital contents?\b",
            r"\bcell ?phone\b.{0,160}\b(?:warrant|privacy|contents?|data|messages?|riley)\b",
            r"\bsmartphone\b.{0,160}\b(?:warrant|privacy|contents?|data|messages?|riley)\b",
            r"\b(?:phone|tablet|smartwatch)\b.{0,160}\bdigital (?:data|contents?|privacy)\b",
            r"\briley v\.? california\b",
            r"\briley\b.{0,160}\b(?:cell ?phone|smartphone|digital|data|messages?)\b",
        ),
    ),
    FrameRule(
        frame="fourth_stop_reasonable_suspicion",
        note="Traffic-stop prolongation, dog-sniff, and reasonable-suspicion doctrine.",
        patterns=(
            r"\btraffic stop\b",
            r"\breasonable suspicion\b",
            r"\bdog sniff\b",
            r"\bprolong(?:ed|ing)? (?:the )?(?:stop|traffic stop|detention)\b",
            r"\brodriguez\b",
        ),
    ),
    FrameRule(
        frame="fourth_home_exigency",
        note="Warrantless home entry justified by emergency aid, hot pursuit, or exigency.",
        patterns=(
            r"\bwarrantless entry (?:into|of) (?:a |the )?(?:home|house|dwelling)\b",
            r"\bentry into (?:a |the )?(?:home|house|dwelling)\b.{0,180}\b(?:exigent|emergency|hot pursuit)\b",
            r"\b(?:exigent|emergency|hot pursuit)\b.{0,180}\b(?:home|house|dwelling|entry)\b",
            r"\bemergency aid\b",
            r"\bhot pursuit\b.{0,120}\b(?:home|house|entry)\b",
        ),
    ),
    FrameRule(
        frame="economic_commerce_clause",
        note="Affirmative Commerce Clause authority.",
        patterns=(
            r"\bcommerce clause\b",
            r"\binterstate commerce\b",
            r"\bsubstantial(?:ly)? affects? (?:interstate )?commerce\b",
            r"\bchannels of commerce\b",
            r"\binstrumentalities of commerce\b",
            r"\baggregate effect\b.{0,120}\bcommerce\b",
        ),
    ),
    FrameRule(
        frame="economic_federalism_state_regulation",
        note="Federalism, police powers, and traditional state regulation.",
        patterns=(
            r"\bfederalism\b",
            r"\btraditional state (?:authority|regulation|police power|domain)\b",
            r"\bpolice power\b",
            r"\breserved to the states\b",
            r"\bstate regulatory autonomy\b",
        ),
    ),
    FrameRule(
        frame="economic_statutory_interpretation",
        note="Statutory interpretation as a distinct reasoning frame.",
        patterns=(
            r"\bstatutory interpretation\b",
            r"\bplain meaning\b",
            r"\btext of (?:the )?statute\b",
            r"\bcongressional intent\b",
            r"\bclear statement\b",
        ),
    ),
    FrameRule(
        frame="economic_remedy_damages",
        note="Statutory damages, private rights of action, civil penalties, or remedial design.",
        patterns=(
            r"\bstatutory damages\b",
            r"\bprivate right of action\b",
            r"\bcivil penalt(?:y|ies)\b",
            r"\bremed(?:y|ies|ial)\b.{0,100}\b(?:damages|statutory|scheme|relief)\b",
            r"\bdamages\b.{0,100}\b(?:remed(?:y|ies|ial)|statutory|private right)\b",
        ),
    ),
    FrameRule(
        frame="economic_commerce_limits",
        note="Commerce Clause limits under Lopez/Morrison-style reasoning.",
        patterns=(
            r"\bnon[- ]economic activity\b",
            r"\bnoneconomic activity\b",
            r"\blopez\b",
            r"\bmorrison\b",
            r"\battenuated\b.{0,120}\bcommerce\b",
            r"\bjurisdictional element\b",
        ),
    ),
    FrameRule(
        frame="civil_equal_protection_strict_scrutiny",
        note="Race/equal-protection strict scrutiny.",
        patterns=(
            r"\bequal protection\b.{0,160}\bstrict scrutiny\b",
            r"\bstrict scrutiny\b.{0,160}\b(?:equal protection|race|racial)\b",
            r"\bracial classification\b",
            r"\brace[- ]based classification\b",
            r"\bcompelling (?:governmental |state )?interest\b.{0,120}\bnarrowly tailored\b",
        ),
    ),
    FrameRule(
        frame="civil_sex_equality_intermediate",
        note="Sex/gender equal-protection intermediate scrutiny.",
        patterns=(
            r"\bintermediate scrutiny\b.{0,160}\b(?:sex|gender)\b",
            r"\b(?:sex|gender)[- ]based classification\b",
            r"\bexceedingly persuasive justification\b",
            r"\bvirginia military institute\b",
            r"\bvmi\b",
        ),
    ),
    FrameRule(
        frame="civil_voting_race_districting",
        note="Racial gerrymandering or race-predominance districting.",
        patterns=(
            r"\bracial gerrymandering\b",
            r"\brace predominated\b",
            r"\bpredominant factor\b.{0,120}\b(?:race|racial|district)\b",
            r"\bredistricting\b.{0,160}\b(?:race|racial)\b",
            r"\bvoting rights act\b",
        ),
    ),
    FrameRule(
        frame="civil_section5_congruence",
        note="Section 5 enforcement, congruence/proportionality, or Fourteenth Amendment abrogation.",
        patterns=(
            r"\bsection 5\b.{0,160}\bfourteenth amendment\b",
            r"\bfourteenth amendment\b.{0,160}\bsection 5\b",
            r"\bcongruent and proportional\b",
            r"\bcongruence and proportionality\b",
            r"\babrogat(?:e|ed|ion)\b.{0,160}\bsovereign immunity\b",
        ),
    ),
    FrameRule(
        frame="federalism_anti_commandeering",
        note="Anti-commandeering of state officers or officials.",
        patterns=(
            r"\banti[- ]commandeering\b",
            r"\bcommandeer(?:ing)?\b",
            r"\bconscript(?:ing)? state (?:officers|officials|sheriffs)\b",
            r"\bstate (?:officers|officials|sheriffs)\b.{0,120}\b(?:federal program|background checks|commandeer)\b",
            r"\bprintz\b",
            r"\bnew york v\.? united states\b",
        ),
    ),
    FrameRule(
        frame="federalism_preemption",
        note="Supremacy Clause and federal preemption.",
        patterns=(
            r"\bpreempt(?:ion|ed|s)?\b",
            r"\bsupremacy clause\b",
            r"\bconflict preemption\b",
            r"\bobstacle preemption\b",
            r"\bexpress preemption\b",
        ),
    ),
    FrameRule(
        frame="admin_major_questions",
        note="Major-questions doctrine and clear congressional authorization.",
        patterns=(
            r"\bmajor questions? doctrine\b",
            r"\bclear congressional authorization\b",
            r"\bvast economic and political significance\b",
            r"\bnondelegation\b",
            r"\bagency authority\b.{0,120}\bmajor\b",
        ),
    ),
    FrameRule(
        frame="due_process_substantive",
        note="Substantive due process, fundamental rights, ordered liberty, or history and tradition.",
        patterns=(
            r"\bsubstantive due process\b",
            r"\bfundamental right\b",
            r"\bliberty interest\b",
            r"\bhistory and tradition\b",
            r"\bordered liberty\b",
        ),
    ),
    FrameRule(
        frame="due_process_procedural_mathews",
        note="Procedural due process, Mathews balancing, hearing rights, or erroneous-deprivation risk.",
        patterns=(
            r"\bprocedural due process\b",
            r"\bmathews\b",
            r"\brisk of erroneous deprivation\b",
            r"\bprivate interest\b.{0,120}\bgovernment(?:'s)? interest\b",
            r"\b(?:oral |pre[- ]termination |evidentiary )?hearing\b.{0,160}\bdue process\b",
        ),
    ),
    FrameRule(
        frame="separation_presidential_power",
        note="Presidential removal/control of officers, unitary executive, or independent-agency separation of powers.",
        patterns=(
            r"\bunitary executive\b",
            r"\bpresidential control\b",
            r"\bindependent agenc(?:y|ies)\b.{0,160}\b(?:president|removal|control)\b",
            r"\bfor[- ]cause removal\b",
            r"\bremov(?:e|al) (?:of )?(?:principal |executive |inferior )?officers?\b",
            r"\bseparation of powers\b.{0,160}\b(?:president|executive officer|independent agency|removal protection)\b",
        ),
    ),
)

RULE_BY_FRAME = {rule.frame: rule for rule in FRAME_RULES}
ALL_FRAMES = tuple(RULE_BY_FRAME)
JSON = dict[str, Any]


def now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def read_jsonl(path: Path) -> list[JSON]:
    rows: list[JSON] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[JSON]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def fmt(value: float) -> str:
    return f"{value:.3f}"


def normalize_text(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_snippet(text: str, max_chars: int = 220) -> str:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars + 1]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip() + "..."


def evidence_window(text: str, start: int, end: int, radius: int = 80) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return clean_snippet(text[left:right])


def score_frame(rule: FrameRule, text: str) -> tuple[int, list[str]]:
    evidence: list[str] = []
    for pattern in rule.patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            evidence.append(evidence_window(text, match.start(), match.end()))
    return len(evidence), evidence[:3]


def score_frames(text: str) -> tuple[dict[str, int], dict[str, list[str]]]:
    normalized = normalize_text(text)
    scores: dict[str, int] = {}
    evidence: dict[str, list[str]] = {}
    for rule in FRAME_RULES:
        score, frame_evidence = score_frame(rule, normalized)
        if score:
            scores[rule.frame] = score
            evidence[rule.frame] = frame_evidence
    return scores, evidence


def list_field(row: JSON, key: str) -> list[str]:
    value = row.get("frame_eval", {}).get(key, [])
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def sample_kind(row: JSON) -> str:
    if row.get("sample_type") is not None:
        return str(row["sample_type"])
    return str(row.get("condition") or "unknown")


def condition_context(row: JSON) -> str:
    if row.get("sample_type") is not None:
        return str(row.get("condition") or "default")
    return str(row.get("prompt_condition") or "hook_generation")


def base_key(row: JSON) -> tuple[str, Any, str]:
    return (str(row["source_run"]), row.get("prompt_id"), condition_context(row))


def alpha_value(row: JSON) -> float | None:
    value = row.get("alpha")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def frame_eval(row: JSON, scores: dict[str, int]) -> JSON:
    expected = set(list_field(row, "expected_frames"))
    contrast = set(list_field(row, "contrast_frames"))
    domain = set(list_field(row, "domain_frames") or ALL_FRAMES)
    active = {frame for frame, score in scores.items() if score > 0}
    off_domain = active - domain
    return {
        "expected_frames": sorted(expected),
        "contrast_frames": sorted(contrast),
        "domain_frames": sorted(domain),
        "target_hits": int(sum(scores.get(frame, 0) for frame in expected)),
        "target_frames_present": int(sum(1 for frame in expected if scores.get(frame, 0) > 0)),
        "target_frame_count": int(len(expected)),
        "target_present": bool(expected and any(scores.get(frame, 0) > 0 for frame in expected)),
        "contrast_hits": int(sum(scores.get(frame, 0) for frame in contrast)),
        "contrast_present": bool(contrast and any(scores.get(frame, 0) > 0 for frame in contrast)),
        "off_domain_hits": int(sum(scores.get(frame, 0) for frame in off_domain)),
        "off_domain_present": bool(off_domain),
        "off_domain_frames": sorted(off_domain),
        "total_frame_hits": int(sum(scores.values())),
    }


def add_base_deltas(rows: list[JSON]) -> None:
    base_by_key: dict[tuple[str, Any, str], JSON] = {}
    for row in rows:
        if sample_kind(row) == "base":
            base_by_key[base_key(row)] = row["proposition_frame_eval"]
    for row in rows:
        base = base_by_key.get(base_key(row))
        if base is None:
            continue
        current = row["proposition_frame_eval"]
        for key in ("target_hits", "contrast_hits", "off_domain_hits", "total_frame_hits"):
            current[f"delta_{key}_vs_base"] = float(current[key] - base[key])
        current["delta_target_minus_contrast_vs_base"] = float(
            (current["target_hits"] - current["contrast_hits"]) - (base["target_hits"] - base["contrast_hits"])
        )


def source_generation_file(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "generations.jsonl"
        if not candidate.exists():
            raise FileNotFoundError(f"No generations.jsonl under {path}")
        return candidate
    return path


def load_runs(paths: list[Path]) -> list[JSON]:
    rows: list[JSON] = []
    for raw_path in paths:
        generation_file = source_generation_file(raw_path)
        source_run = generation_file.parent.name
        for row in read_jsonl(generation_file):
            row = dict(row)
            row["source_run"] = source_run
            row["source_file"] = str(generation_file)
            row["condition_context"] = condition_context(row)
            row["sample_kind"] = sample_kind(row)
            text = str(row.get("text") or row.get("completion") or "")
            scores, evidence = score_frames(text)
            row["proposition_frame_scores"] = scores
            row["proposition_frame_evidence"] = evidence
            row["proposition_frame_eval"] = frame_eval(row, scores)
            rows.append(row)
    add_base_deltas(rows)
    return rows


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def aggregate_rows(rows: list[JSON]) -> list[JSON]:
    groups: dict[tuple[Any, ...], list[JSON]] = defaultdict(list)
    for row in rows:
        key = (
            row["source_run"],
            row.get("issue_area"),
            row.get("condition_context"),
            row.get("sample_kind"),
            row.get("candidate"),
            alpha_value(row),
        )
        groups[key].append(row)
    summaries: list[JSON] = []
    for (source_run, issue_area, context, kind, candidate, alpha), group_rows in sorted(groups.items(), key=str):
        old_evals = [row.get("frame_eval", {}) for row in group_rows]
        prop_evals = [row["proposition_frame_eval"] for row in group_rows]
        summaries.append(
            {
                "source_run": source_run,
                "issue_area": issue_area,
                "condition_context": context,
                "sample_kind": kind,
                "candidate": candidate,
                "alpha": alpha,
                "n": len(group_rows),
                "prompt_count": len({row.get("prompt_id") for row in group_rows}),
                "old_target_present_rate": mean([1.0 if item.get("target_present") else 0.0 for item in old_evals]),
                "prop_target_present_rate": mean([1.0 if item.get("target_present") else 0.0 for item in prop_evals]),
                "old_off_domain_present_rate": mean(
                    [1.0 if item.get("off_domain_present") else 0.0 for item in old_evals]
                ),
                "prop_off_domain_present_rate": mean(
                    [1.0 if item.get("off_domain_present") else 0.0 for item in prop_evals]
                ),
                "old_mean_target_hits": mean([float(item.get("target_hits", 0.0)) for item in old_evals]),
                "prop_mean_target_hits": mean([float(item.get("target_hits", 0.0)) for item in prop_evals]),
                "prop_mean_delta_target_hits_vs_base": mean(
                    [float(item.get("delta_target_hits_vs_base", 0.0)) for item in prop_evals]
                ),
                "prop_mean_delta_target_minus_contrast_vs_base": mean(
                    [float(item.get("delta_target_minus_contrast_vs_base", 0.0)) for item in prop_evals]
                ),
                "prop_mean_off_domain_hits": mean([float(item.get("off_domain_hits", 0.0)) for item in prop_evals]),
            }
        )
    return summaries


def dropped_frame_summary(rows: list[JSON]) -> list[JSON]:
    dropped = Counter()
    gained = Counter()
    old_present = Counter()
    prop_present = Counter()
    for row in rows:
        old_scores = row.get("frame_scores", {})
        prop_scores = row.get("proposition_frame_scores", {})
        old_frames = {str(frame) for frame, score in old_scores.items() if int(score) > 0}
        prop_frames = {str(frame) for frame, score in prop_scores.items() if int(score) > 0}
        old_present.update(old_frames)
        prop_present.update(prop_frames)
        dropped.update(old_frames - prop_frames)
        gained.update(prop_frames - old_frames)
    frames = sorted(set(old_present) | set(prop_present) | set(dropped) | set(gained))
    return [
        {
            "frame": frame,
            "old_present_rows": old_present[frame],
            "prop_present_rows": prop_present[frame],
            "dropped_rows": dropped[frame],
            "gained_rows": gained[frame],
        }
        for frame in frames
    ]


def prompt_condition_nulls(rows: list[JSON]) -> list[JSON]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("sample_kind") not in {"random_control", "random_unit"}:
            continue
        eval_row = row["proposition_frame_eval"]
        groups[
            (
                row["source_run"],
                row.get("prompt_id"),
                row.get("prompt_key"),
                row.get("issue_area"),
                row.get("condition_context"),
                row.get("sample_kind"),
            )
        ].append(float(eval_row.get("delta_target_hits_vs_base", 0.0)))
    summaries: list[JSON] = []
    for (source_run, prompt_id, prompt_key, issue_area, context, kind), values in sorted(groups.items(), key=str):
        values_sorted = sorted(values)
        p05_idx = int(0.05 * (len(values_sorted) - 1)) if values_sorted else 0
        p95_idx = int(0.95 * (len(values_sorted) - 1)) if values_sorted else 0
        summaries.append(
            {
                "source_run": source_run,
                "prompt_id": prompt_id,
                "prompt_key": prompt_key,
                "issue_area": issue_area,
                "condition_context": context,
                "sample_kind": kind,
                "n": len(values),
                "mean_prop_target_delta": mean(values),
                "sd_prop_target_delta": stdev(values),
                "p05": values_sorted[p05_idx] if values_sorted else 0.0,
                "p50": statistics.median(values_sorted) if values_sorted else 0.0,
                "p95": values_sorted[p95_idx] if values_sorted else 0.0,
            }
        )
    return summaries


def hook_candidate_vs_random(rows: list[JSON]) -> list[JSON]:
    random_by_prompt_alpha_target: dict[tuple[str, Any, float, str], list[float]] = defaultdict(list)
    random_net_by_prompt_alpha_target: dict[tuple[str, Any, float, str], list[float]] = defaultdict(list)
    source_by_prompt_alpha_target: dict[tuple[str, Any, float, str], list[float]] = defaultdict(list)
    source_net_by_prompt_alpha_target: dict[tuple[str, Any, float, str], list[float]] = defaultdict(list)
    for row in rows:
        alpha = alpha_value(row)
        if alpha is None:
            continue
        eval_row = row["proposition_frame_eval"]
        target = str(row.get("target_candidate") or "")
        key = (str(row["source_run"]), row.get("prompt_id"), alpha, target)
        if row.get("sample_kind") == "random_unit":
            random_by_prompt_alpha_target[key].append(float(eval_row.get("delta_target_hits_vs_base", 0.0)))
            random_net_by_prompt_alpha_target[key].append(
                float(eval_row.get("delta_target_minus_contrast_vs_base", 0.0))
            )
        elif row.get("sample_kind") == "source_control":
            source_by_prompt_alpha_target[key].append(float(eval_row.get("delta_target_hits_vs_base", 0.0)))
            source_net_by_prompt_alpha_target[key].append(
                float(eval_row.get("delta_target_minus_contrast_vs_base", 0.0))
            )

    residuals_by_run_alpha_target: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    net_residuals_by_run_alpha_target: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    for (source_run, _prompt_id, alpha, target), values in random_by_prompt_alpha_target.items():
        random_mean = mean(values)
        residuals_by_run_alpha_target[(source_run, alpha, target)].extend(value - random_mean for value in values)
    for (source_run, _prompt_id, alpha, target), values in random_net_by_prompt_alpha_target.items():
        random_mean = mean(values)
        net_residuals_by_run_alpha_target[(source_run, alpha, target)].extend(value - random_mean for value in values)

    candidate_groups: dict[tuple[str, str, float], list[JSON]] = defaultdict(list)
    for row in rows:
        if row.get("sample_kind") != "sae_poke" or not row.get("candidate"):
            continue
        alpha = alpha_value(row)
        if alpha is None:
            continue
        candidate_groups[(str(row["source_run"]), str(row["candidate"]), alpha)].append(row)

    comparisons: list[JSON] = []
    for (source_run, candidate, alpha), group_rows in sorted(candidate_groups.items(), key=str):
        adjusted: list[float] = []
        adjusted_net: list[float] = []
        candidate_values: list[float] = []
        candidate_net_values: list[float] = []
        matched_random_means: list[float] = []
        matched_random_net_means: list[float] = []
        matched_random_maxes: list[float] = []
        matched_random_net_maxes: list[float] = []
        source_control_values: list[float] = []
        source_control_net_values: list[float] = []
        target_wins = 0
        net_wins = 0
        target_strongest_wins = 0
        net_strongest_wins = 0
        for row in group_rows:
            target = str(row.get("target_candidate") or row.get("candidate") or "")
            key = (source_run, row.get("prompt_id"), alpha, target)
            fallback_key = (source_run, row.get("prompt_id"), alpha, "")
            random_values = random_by_prompt_alpha_target.get(key) or random_by_prompt_alpha_target.get(fallback_key, [])
            random_net_values = (
                random_net_by_prompt_alpha_target.get(key)
                or random_net_by_prompt_alpha_target.get(fallback_key, [])
            )
            if not random_values or not random_net_values:
                continue
            eval_row = row["proposition_frame_eval"]
            candidate_value = float(eval_row.get("delta_target_hits_vs_base", 0.0))
            candidate_net_value = float(eval_row.get("delta_target_minus_contrast_vs_base", 0.0))
            random_mean = mean(random_values)
            random_net_mean = mean(random_net_values)
            candidate_values.append(candidate_value)
            candidate_net_values.append(candidate_net_value)
            matched_random_means.append(random_mean)
            matched_random_net_means.append(random_net_mean)
            matched_random_maxes.append(max(random_values))
            matched_random_net_maxes.append(max(random_net_values))
            adjusted_value = candidate_value - random_mean
            adjusted_net_value = candidate_net_value - random_net_mean
            adjusted.append(adjusted_value)
            adjusted_net.append(adjusted_net_value)
            target_wins += int(adjusted_value > 0.0)
            net_wins += int(adjusted_net_value > 0.0)
            target_strongest_wins += int(candidate_value > max(random_values))
            net_strongest_wins += int(candidate_net_value > max(random_net_values))
            source_values = source_by_prompt_alpha_target.get(key) or source_by_prompt_alpha_target.get(fallback_key, [])
            source_net_values = (
                source_net_by_prompt_alpha_target.get(key)
                or source_net_by_prompt_alpha_target.get(fallback_key, [])
            )
            if source_values:
                source_control_values.append(source_values[0])
            if source_net_values:
                source_control_net_values.append(source_net_values[0])

        target = str(group_rows[0].get("target_candidate") or candidate)
        residuals = residuals_by_run_alpha_target.get((source_run, alpha, target)) or residuals_by_run_alpha_target.get(
            (source_run, alpha, ""),
            [],
        )
        net_residuals = net_residuals_by_run_alpha_target.get(
            (source_run, alpha, target)
        ) or net_residuals_by_run_alpha_target.get((source_run, alpha, ""), [])
        residual_sd = stdev(residuals)
        net_residual_sd = stdev(net_residuals)
        target_z = 0.0 if residual_sd == 0.0 else mean(adjusted) / residual_sd
        net_z = 0.0 if net_residual_sd == 0.0 else mean(adjusted_net) / net_residual_sd
        comparisons.append(
            {
                "source_run": source_run,
                "candidate": candidate,
                "alpha": alpha,
                "n": len(adjusted),
                "candidate_mean_prop_target_delta": mean(candidate_values),
                "prompt_random_mean_prop_target_delta": mean(matched_random_means),
                "prompt_random_max_prop_target_delta": mean(matched_random_maxes),
                "mean_prompt_matched_prop_delta_minus_random": mean(adjusted),
                "source_control_mean_prop_target_delta": mean(source_control_values) if source_control_values else None,
                "random_residual_sd": residual_sd,
                "z_vs_prompt_matched_random": target_z,
                "prompt_win_rate_vs_random_mean": 0.0 if not adjusted else target_wins / len(adjusted),
                "prompt_strongest_win_rate": 0.0 if not adjusted else target_strongest_wins / len(adjusted),
                "candidate_mean_prop_net_delta": mean(candidate_net_values),
                "prompt_random_mean_prop_net_delta": mean(matched_random_net_means),
                "prompt_random_max_prop_net_delta": mean(matched_random_net_maxes),
                "mean_prompt_matched_prop_net_delta_minus_random": mean(adjusted_net),
                "source_control_mean_prop_net_delta": mean(source_control_net_values)
                if source_control_net_values
                else None,
                "random_net_residual_sd": net_residual_sd,
                "z_net_vs_prompt_matched_random": net_z,
                "prompt_net_win_rate_vs_random_mean": 0.0 if not adjusted_net else net_wins / len(adjusted_net),
                "prompt_net_strongest_win_rate": 0.0 if not adjusted_net else net_strongest_wins / len(adjusted_net),
            }
        )
    return comparisons


def disagreement_queue(rows: list[JSON], limit: int) -> list[JSON]:
    selected: list[JSON] = []
    for row in rows:
        old_scores = row.get("frame_scores", {})
        prop_scores = row.get("proposition_frame_scores", {})
        old_frames = {str(frame) for frame, score in old_scores.items() if int(score) > 0}
        prop_frames = {str(frame) for frame, score in prop_scores.items() if int(score) > 0}
        old_eval = row.get("frame_eval", {})
        prop_eval = row.get("proposition_frame_eval", {})
        if not (old_frames - prop_frames or prop_frames - old_frames):
            continue
        selected.append(
            {
                "source_run": row.get("source_run"),
                "prompt_id": row.get("prompt_id"),
                "prompt_key": row.get("prompt_key"),
                "issue_area": row.get("issue_area"),
                "condition_context": row.get("condition_context"),
                "sample_kind": row.get("sample_kind"),
                "candidate": row.get("candidate"),
                "alpha": row.get("alpha"),
                "old_only_frames": sorted(old_frames - prop_frames),
                "prop_only_frames": sorted(prop_frames - old_frames),
                "old_target_hits": old_eval.get("target_hits"),
                "prop_target_hits": prop_eval.get("target_hits"),
                "old_off_domain_hits": old_eval.get("off_domain_hits"),
                "prop_off_domain_hits": prop_eval.get("off_domain_hits"),
                "completion": clean_snippet(str(row.get("text") or row.get("completion") or ""), max_chars=700),
            }
        )
    selected.sort(
        key=lambda item: (
            -len(item["old_only_frames"]),
            -int(item.get("old_off_domain_hits") or 0),
            str(item.get("source_run")),
            str(item.get("prompt_key")),
        )
    )
    return selected[:limit]


def write_report(
    *,
    output_dir: Path,
    rows: list[JSON],
    summaries: list[JSON],
    dropped: list[JSON],
    nulls: list[JSON],
    hook_comparisons: list[JSON],
    disagreements: list[JSON],
) -> None:
    lines: list[str] = [
        "# SCOTUS Proposition-Level Frame Rescore",
        "",
        "## Purpose",
        "",
        "This rescoring pass replaces raw substring counts with stricter proposition-level frame rules. It is a calibration artifact for deciding whether future steering candidates beat lexical and prompt-format baselines.",
        "",
        "## Inputs",
        "",
    ]
    input_rows = []
    for source_run, count in sorted(Counter(str(row["source_run"]) for row in rows).items()):
        input_rows.append([source_run, count])
    lines.extend(md_table(["Source run", "Rows"], input_rows))

    lines.extend(
        [
            "",
            "## Main Read",
            "",
            "1. Proposition rules are intentionally conservative: repeated keywords count once per legal proposition pattern, and broad terms are not accepted alone.",
            "2. The rescore is still automatic and should gate manual review, not replace it.",
            "3. A causal candidate should improve proposition-level target-minus-contrast against prompt-matched random controls before any larger hook run.",
            "",
            "## Largest Frame Drops",
            "",
        ]
    )
    drop_rows = [
        [
            item["frame"],
            item["old_present_rows"],
            item["prop_present_rows"],
            item["dropped_rows"],
            item["gained_rows"],
        ]
        for item in sorted(dropped, key=lambda x: int(x["dropped_rows"]), reverse=True)[:20]
    ]
    lines.extend(md_table(["Frame", "Old rows", "Prop rows", "Dropped", "Gained"], drop_rows))

    lines.extend(["", "## Aggregate Summary", ""])
    aggregate_rows = [
        [
            item["source_run"],
            item.get("issue_area") or "",
            item.get("condition_context") or "",
            item.get("sample_kind") or "",
            item.get("candidate") or "",
            "" if item.get("alpha") is None else fmt(float(item["alpha"])),
            item["n"],
            fmt(float(item["old_target_present_rate"])),
            fmt(float(item["prop_target_present_rate"])),
            fmt(float(item["old_off_domain_present_rate"])),
            fmt(float(item["prop_off_domain_present_rate"])),
            fmt(float(item["prop_mean_delta_target_minus_contrast_vs_base"])),
        ]
        for item in summaries[:80]
    ]
    lines.extend(
        md_table(
            [
                "Run",
                "Issue",
                "Context",
                "Sample",
                "Candidate",
                "Alpha",
                "N",
                "Old target present",
                "Prop target present",
                "Old off-domain",
                "Prop off-domain",
                "Prop net delta",
            ],
            aggregate_rows,
        )
    )

    if hook_comparisons:
        lines.extend(["", "## Hook Candidate vs Prompt-Matched Random", ""])
        comparison_rows = [
            [
                item["source_run"],
                item["candidate"],
                fmt(float(item["alpha"])),
                item["n"],
                fmt(float(item["mean_prompt_matched_prop_delta_minus_random"])),
                fmt(float(item["z_vs_prompt_matched_random"])),
                fmt(float(item["prompt_win_rate_vs_random_mean"])),
                fmt(float(item["prompt_strongest_win_rate"])),
                "" if item.get("source_control_mean_prop_target_delta") is None else fmt(float(item["source_control_mean_prop_target_delta"])),
                fmt(float(item["mean_prompt_matched_prop_net_delta_minus_random"])),
                fmt(float(item["z_net_vs_prompt_matched_random"])),
                fmt(float(item["prompt_net_win_rate_vs_random_mean"])),
                fmt(float(item["prompt_net_strongest_win_rate"])),
                "" if item.get("source_control_mean_prop_net_delta") is None else fmt(float(item["source_control_mean_prop_net_delta"])),
            ]
            for item in hook_comparisons
        ]
        lines.extend(
            md_table(
                [
                    "Run",
                    "Candidate",
                    "Alpha",
                    "N",
                    "Target minus random",
                    "Target z",
                    "Target win",
                    "Target strongest win",
                    "Source target",
                    "Net minus random",
                    "Net z",
                    "Net win",
                    "Net strongest win",
                    "Source net",
                ],
                comparison_rows,
            )
        )

    lines.extend(["", "## Highest-Variance Proposition Nulls", ""])
    null_rows = [
        [
            item["source_run"],
            item["prompt_key"],
            item["condition_context"],
            item["n"],
            fmt(float(item["mean_prop_target_delta"])),
            fmt(float(item["sd_prop_target_delta"])),
            item["p05"],
            item["p50"],
            item["p95"],
        ]
        for item in sorted(nulls, key=lambda x: float(x["sd_prop_target_delta"]), reverse=True)[:30]
    ]
    lines.extend(md_table(["Run", "Prompt", "Context", "N", "Mean", "SD", "P05", "P50", "P95"], null_rows))

    lines.extend(["", "## Review Queue Sample", ""])
    review_rows = [
        [
            item["source_run"],
            item["prompt_key"],
            item["sample_kind"],
            ", ".join(item["old_only_frames"]),
            ", ".join(item["prop_only_frames"]),
            item["completion"],
        ]
        for item in disagreements[:20]
    ]
    lines.extend(md_table(["Run", "Prompt", "Sample", "Old-only frames", "Prop-only frames", "Snippet"], review_rows))

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- Rescored rows: `rescored_rows.jsonl`",
            "- Group summaries: `summary_by_group.json`",
            "- Dropped-frame summary: `dropped_frame_summary.json`",
            "- Prompt-condition nulls: `prompt_condition_nulls.json`",
            "- Hook candidate comparisons: `hook_candidate_vs_random.json`",
            "- Disagreement review queue: `disagreement_review_queue.jsonl`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescore SCOTUS frame artifacts with proposition-level rules.")
    parser.add_argument("--runs", nargs="*", type=Path, default=list(DEFAULT_RUNS), help="Run dirs or generations.jsonl files.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--review-limit", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.output_root / f"scotus_frame_prop_rescore_{now_stamp()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_runs([path.resolve() for path in args.runs])
    summaries = aggregate_rows(rows)
    dropped = dropped_frame_summary(rows)
    nulls = prompt_condition_nulls(rows)
    hook_comparisons = hook_candidate_vs_random(rows)
    disagreements = disagreement_queue(rows, limit=max(0, args.review_limit))

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "runs": [str(path) for path in args.runs],
        "rows": len(rows),
        "rules": [{"frame": rule.frame, "patterns": list(rule.patterns), "note": rule.note} for rule in FRAME_RULES],
    }
    write_json(output_dir / "manifest.json", manifest)
    write_jsonl(output_dir / "rescored_rows.jsonl", rows)
    write_json(output_dir / "summary_by_group.json", summaries)
    write_json(output_dir / "dropped_frame_summary.json", dropped)
    write_json(output_dir / "prompt_condition_nulls.json", nulls)
    write_json(output_dir / "hook_candidate_vs_random.json", hook_comparisons)
    write_jsonl(output_dir / "disagreement_review_queue.jsonl", disagreements)
    write_report(
        output_dir=output_dir,
        rows=rows,
        summaries=summaries,
        dropped=dropped,
        nulls=nulls,
        hook_comparisons=hook_comparisons,
        disagreements=disagreements,
    )
    print(f"Wrote {output_dir / 'report.md'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
