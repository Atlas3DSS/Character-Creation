"""Deterministic, CPU-only specifications for fresh persona mapping experiments.

This module schedules no inference. A plan is a hypothesis and must be consumed by
a runtime that implements its exact site, phase and dose semantics. Historical
neuron coordinates are never transplanted to a new checkpoint. See --help for the
plan, score-ipip and score-generation commands; all outputs are JSON.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
from typing import Any

ROOT = Path(__file__).resolve().parent
TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
PHASES = ("all", "prefill", "thinking", "answer")
SHARDS = 6
SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(value: str, field: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ModelIdentity:
    """Immutable identity read from the actual checkpoint, never inferred by name."""

    model_id: str
    checkpoint_revision: str
    weights_sha256: str
    quantization: str
    n_layers: int
    hidden_size: int
    intermediate_size: int
    architecture: str

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.architecture.strip() or not self.quantization.strip():
            raise ValueError("model_id, architecture and quantization must be explicit")
        if not re.fullmatch(r"[0-9a-f]{40,64}", self.checkpoint_revision):
            raise ValueError("checkpoint_revision must be an immutable commit/content digest, not a branch")
        _sha256(self.weights_sha256, "weights_sha256")
        for name in ("n_layers", "hidden_size", "intermediate_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class Intervention:
    """One intervention; positive trait directions always mean high minus low."""

    kind: str
    layer: int
    dose: float
    phase: str = "all"
    neuron: int | None = None
    trait: str | None = None
    direction_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("neuron_add", "residual_add", "mlp_scale"):
            raise ValueError("kind must be neuron_add, residual_add or mlp_scale")
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        if isinstance(self.layer, bool) or not isinstance(self.layer, int) or self.layer < 0:
            raise ValueError("layer must be a zero-indexed nonnegative integer")
        if isinstance(self.dose, bool) or not math.isfinite(self.dose):
            raise ValueError("dose must be finite")
        if self.kind == "neuron_add":
            if isinstance(self.neuron, bool) or not isinstance(self.neuron, int) or self.neuron < 0:
                raise ValueError("neuron_add requires a zero-indexed nonnegative neuron")
            if self.trait is not None or self.direction_sha256 is not None:
                raise ValueError("neuron_add cannot contain a trait direction")
        elif self.neuron is not None:
            raise ValueError("only neuron_add can specify a neuron")
        if self.kind == "residual_add":
            if self.trait not in TRAITS:
                raise ValueError("residual_add requires a Big Five trait")
        elif self.trait is not None or self.direction_sha256 is not None:
            raise ValueError("only residual_add can specify a trait direction")
        if self.direction_sha256 is not None:
            _sha256(self.direction_sha256, "direction_sha256")

    def validate_for(self, model: ModelIdentity) -> None:
        if self.layer >= model.n_layers:
            raise ValueError("layer is outside the bound checkpoint")
        if self.neuron is not None and self.neuron >= model.intermediate_size:
            raise ValueError("neuron is outside the bound checkpoint")

    def specification(self) -> dict[str, Any]:
        result = asdict(self)
        if self.kind == "neuron_add":
            result.update(
                site="mlp.down_proj.input",
                formula="a[...,n] += dose * resid_p50[L] / l2(W_down[:,n])",
                normalization="median_l2_block_output_nonpadding_positions_ge_1",
                polarity="positive_adds_the_signed_output_column",
            )
        elif self.kind == "residual_add":
            result.update(
                site="transformer_block.output",
                formula="h += dose * resid_p50[L] * v / l2(v)",
                normalization="median_l2_block_output_nonpadding_positions_ge_1",
                polarity="v_is_high_trait_minus_low_trait",
            )
        else:
            result.update(
                site="mlp.down_proj.input",
                formula="a *= dose",
                normalization="none_multiplicative_gain",
                polarity="1_identity_0_ablation_minus1_sign_reversal",
            )
        return result


def shard_for(condition_id: str, count: int = SHARDS) -> int:
    if count != SHARDS:
        raise ValueError("this protocol has exactly six shards")
    return int(hashlib.sha256(condition_id.encode("utf-8")).hexdigest(), 16) % count


def _condition(model: ModelIdentity, stage: str, seed: int, interventions: tuple[Intervention, ...],
               control: str, prompt_sha256: str, calibration_sha256: str | None,
               max_new_tokens: int, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    for item in interventions:
        item.validate_for(model)
    item = {
        "schema_version": SCHEMA_VERSION,
        "model": asdict(model), "stage": stage, "seed": seed,
        "control": control, "interventions": [x.specification() for x in interventions],
        "prompt_bank_sha256": prompt_sha256,
        "calibration_sha256": calibration_sha256,
        "generation": {"max_new_tokens": max_new_tokens, "temperature": 0.0, "do_sample": False},
        "group_semantics": "sum_listed_interventions_in_list_order_no_implicit_rescaling",
        "flags": {
            "hypothesis_only": True, "validated_effect": False,
            "requires_complete_answer": True,
            "include_capability_prompts": True,
            "heldout": stage == "validate", "fit_on_heldout": False,
            "promotion_eligible": False,
        },
        "metadata": metadata or {},
    }
    if control == "matched_norm_random":
        item["random_control"] = {
            "distribution": "isotropic_standard_normal_then_l2_normalize",
            "seed": seed,
            "site": "transformer_block.output",
            "norm_match": "per_layer_per_token_realized_aggregate_residual_delta_of_reference",
            "reference_required": True,
            "note": "For MLP interventions measure the projected delta, including FFN scaling; do not match raw neuron doses.",
        }
    if control == "label_shuffled":
        item["label_shuffle"] = {
            "seed": seed,
            "method": "permute_high_low_labels_within_training_prompt_and_nuisance_strata_preserve_counts",
            "extract_again": True, "reuse_true_direction_forbidden": True,
            "requires_separate_direction_artifact": True,
        }
    missing = []
    if interventions and calibration_sha256 is None and any(x.kind != "mlp_scale" for x in interventions):
        missing.append("layer_residual_calibration")
    if any(x.kind == "residual_add" and x.direction_sha256 is None for x in interventions):
        missing.append("checkpoint_specific_trait_direction")
    if control == "label_shuffled":
        missing.append("label_shuffled_direction")
    item["unresolved_requirements"] = missing
    item["ready_for_runtime_validation"] = not missing
    item["condition_id"] = digest(item)
    item["shard"] = shard_for(item["condition_id"])
    return item


def build_plan(model: ModelIdentity, stage: str = "coarse", *, seed: int = 42,
               phase: str = "all", doses: tuple[float, ...] = (0.1, 0.25, 0.5),
               neurons_per_layer: int = 4, candidates: tuple[Intervention, ...] = (),
               candidate_model: ModelIdentity | None = None,
               calibration_sha256: str | None = None, max_new_tokens: int = 2048,
               prompt_bank: dict[str, Any] | None = None) -> dict[str, Any]:
    """Make a full six-shard plan, with fresh-checkpoint pending direction artifacts.

    calibrate: null-direction dose probes and untreated reference per layer.
    coarse: all layers, five trait axes, deterministic neuron samples and MLP gains.
    refine: supplied fresh-checkpoint candidates, both signs and pair interactions.
    validate: the same comparisons on held-out prompts; never fits on that split.
    """
    if stage not in ("calibrate", "coarse", "refine", "validate") or phase not in PHASES:
        raise ValueError("invalid stage or phase")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not doses or any(isinstance(d, bool) or not math.isfinite(d) or d <= 0 for d in doses):
        raise ValueError("doses must be finite, positive magnitudes")
    if isinstance(neurons_per_layer, bool) or not isinstance(neurons_per_layer, int) or not 1 <= neurons_per_layer <= model.intermediate_size:
        raise ValueError("neurons_per_layer must fit the model")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if calibration_sha256 is not None:
        _sha256(calibration_sha256, "calibration_sha256")
    if stage in ("refine", "validate") and not candidates:
        raise ValueError("refine and validate require candidates measured on this checkpoint")
    if candidates and candidate_model != model:
        raise ValueError("candidate provenance must bind the exact target model identity")
    for candidate in candidates:
        candidate.validate_for(model)
    bank = prompt_bank if prompt_bank is not None else json.loads((ROOT / "data/persona_prompts.json").read_text())
    bank_hash = digest(bank)
    rows: list[dict[str, Any]] = []

    def add(items: tuple[Intervention, ...], control: str = "intervention", **meta: Any) -> dict[str, Any]:
        result = _condition(model, stage, seed, items, control, bank_hash, calibration_sha256, max_new_tokens, meta)
        rows.append(result)
        return result

    def controlled(items: tuple[Intervention, ...], **meta: Any) -> None:
        reference = add(items, **meta)
        add(items, "matched_norm_random", reference_condition_id=reference["condition_id"])
        if any(x.kind == "residual_add" for x in items):
            # True direction hashes are never used as shuffled-direction artifacts.
            shuffled = tuple(replace(x, direction_sha256=None) if x.kind == "residual_add" else x for x in items)
            add(shuffled, "label_shuffled", reference_condition_id=reference["condition_id"])

    add((), "unsteered")
    for trait in TRAITS:
        for level in ("low", "high"):
            add((), "prompt_only", trait=trait, level=level, system_prompt=bank["trait_prompts"][trait][level])

    if stage == "calibrate":
        for layer, magnitude, sign in itertools.product(range(model.n_layers), sorted(set(doses)), (-1, 1)):
            item = Intervention("residual_add", layer, sign * magnitude, phase, trait="openness")
            # Calibration random vectors need no learned trait direction or reference effect.
            row = add((item,), "calibration_random", calibration_layer=layer)
            row["unresolved_requirements"] = ["layer_residual_calibration"] if calibration_sha256 is None else []
            row["ready_for_runtime_validation"] = not row["unresolved_requirements"]
            row["interventions"][0].update(kind="random_residual_add", trait=None, polarity="random_signed_unit_vector")
            row["random_control"] = {"distribution": "isotropic_standard_normal_then_l2_normalize", "seed": seed + layer,
                                     "norm_match": "abs_dose_times_resid_p50"}
            row.pop("condition_id")
            row.pop("shard")
            row["condition_id"] = digest(row)
            row["shard"] = shard_for(row["condition_id"])
    elif stage == "coarse":
        for layer in range(model.n_layers):
            rng = random.Random(f"{seed}:{model.weights_sha256}:{layer}")
            neurons = sorted(rng.sample(range(model.intermediate_size), neurons_per_layer))
            for magnitude, sign in itertools.product(sorted(set(doses)), (-1, 1)):
                dose = sign * magnitude
                for trait in TRAITS:
                    controlled((Intervention("residual_add", layer, dose, phase, trait=trait),))
                for neuron in neurons:
                    controlled((Intervention("neuron_add", layer, dose, phase, neuron=neuron),))
            for gain in (-1.0, 0.0, 0.5, 1.5):
                controlled((Intervention("mlp_scale", layer, gain, phase),), exploratory_broad_intervention=True)
    else:
        # Candidate input order cannot change the design or shard assignment.
        ordered = sorted(set(candidates), key=lambda x: canonical_json(asdict(x)))
        for item in ordered:
            for sign in (-1, 1):
                controlled((replace(item, dose=sign * abs(item.dose)),), candidate_sign=sign)
        for left, right in itertools.combinations(ordered, 2):
            if (left.layer, left.kind, left.neuron, left.trait) == (right.layer, right.kind, right.neuron, right.trait):
                continue
            for signs in itertools.product((-1, 1), repeat=2):
                group = tuple(replace(x, dose=s * abs(x.dose)) for x, s in zip((left, right), signs))
                controlled(group, interaction="cross_layer" if left.layer != right.layer else "within_layer", signs=list(signs))
        if len(ordered) > 2:
            controlled(tuple(ordered), interaction="full_candidate_group")
            controlled(tuple(replace(x, dose=-x.dose) for x in ordered), interaction="full_group_sign_reversal")
            for omitted in range(len(ordered)):
                controlled(tuple(x for index, x in enumerate(ordered) if index != omitted),
                           interaction="group_leave_one_out", omitted=asdict(ordered[omitted]))

    conditions = sorted({row["condition_id"]: row for row in rows}.values(), key=lambda x: x["condition_id"])
    plan = {
        "schema_version": SCHEMA_VERSION, "model": asdict(model), "stage": stage,
        "status": "planned_not_run", "seed": seed, "shard_count": SHARDS,
        "prompt_bank_sha256": bank_hash,
        "prompt_ids": [p["id"] for p in bank["prompts"] if p["split"] == ("heldout" if stage == "validate" else "discovery")],
        "protocol": {
            "indexing": "zero_based_checkpoint_local",
            "calibration": "unsteered_separate_calibration_text_nonpadding_tokens_positions_ge_1",
            "phase": phase, "phase_boundary": "tokenized_chat_template_thinking_and_answer_boundaries_required",
            "comparison": "same_prompt_same_seed_same_generation_budget",
            "group_controls": "match_realized_aggregate_projected_delta_per_layer_per_token",
            "promotion": "requires_completed_answers_heldout_effect_and_capability_review_never_automatic",
            "artifact_binding": "direction_and_calibration_artifacts_must_bind_this_exact_model_identity",
            "all_control_metadata_must_be_honored": True,
        },
        "conditions": conditions,
    }
    plan["plan_id"] = digest(plan)
    return plan


def select_shard(plan: dict[str, Any], shard: int) -> dict[str, Any]:
    if isinstance(shard, bool) or not isinstance(shard, int) or not 0 <= shard < SHARDS:
        raise ValueError("shard must be in 0..5")
    return {**plan, "selected_shard": shard,
            "conditions": [x for x in plan["conditions"] if x["shard"] == shard]}


def score_ipip50(responses: dict[str, Any], inventory: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score explicit item-number -> integer 1..5 answers; never guess missing items."""
    if not isinstance(responses, dict):
        raise ValueError("responses must be an item ID -> integer object")
    inventory = inventory or json.loads((ROOT / "data/big_five_ocean_test.json").read_text())
    questions = inventory["questions"]
    expected = {str(q["id"]) for q in questions}
    if len(questions) != 50 or len(expected) != 50:
        raise ValueError("IPIP inventory must contain 50 unique items")
    keyed = {str(k): v for k, v in responses.items()}
    if len(keyed) != len(responses) or set(keyed) - expected:
        raise ValueError("responses contain duplicate or unknown item IDs")
    for key, value in keyed.items():
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"item {key} must be an integer 1..5")
    scores = {}
    for dimension in inventory["dimensions"]:
        group = [q for q in questions if q["dimension"] == dimension]
        if len(group) != 10:
            raise ValueError("each dimension must have 10 items")
        values = [6 - keyed[str(q["id"])] if q["reverse_scored"] else keyed[str(q["id"])] for q in group if str(q["id"]) in keyed]
        complete = len(values) == 10
        scores[dimension] = {"answered": len(values), "complete": complete,
                             "sum_10_to_50": sum(values) if complete else None,
                             "mean_1_to_5": sum(values) / 10 if complete else None}
    return {"inventory_sha256": digest(inventory), "complete": set(keyed) == expected,
            "missing_items": sorted(expected - set(keyed), key=int), "scores": scores,
            "interpretation": "model_self_report_measurement_not_a_human_personality_diagnosis"}


def evaluate_completion(record: dict[str, Any]) -> dict[str, Any]:
    """Require runtime evidence of a final answer; report capability separately."""
    if not isinstance(record, dict):
        raise ValueError("generation record must be an object")
    text = record.get("answer_text", "")
    if not isinstance(text, str):
        raise ValueError("answer_text must be text")
    finish = record.get("finish_reason")
    entered = record.get("answer_phase_reached") is True
    cutoff = finish in ("length", "max_tokens") or record.get("truncated") is True
    complete = bool(text.strip()) and entered and not cutoff and finish in ("stop", "eos")
    capability = record.get("capability_correct")
    if capability is not None and not isinstance(capability, bool):
        raise ValueError("capability_correct must be boolean or null")
    return {"complete": complete, "truncated": cutoff, "answer_phase_reached": entered,
            "capability_correct": capability, "heldout": record.get("heldout") is True,
            "eligible_for_behavioral_scoring": complete,
            "promotion_eligible": False,
            "reason": "complete_final_answer" if complete else "missing_or_incomplete_final_answer",
            "note": "Trait effect, blinded assessment and capability comparisons are separate required evaluations."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="Create a specification only; does not run or load a model")
    plan_parser.add_argument("--model-manifest", type=Path, required=True, help="JSON object with all ModelIdentity fields; immutable revision and weight digest required")
    plan_parser.add_argument("--stage", choices=("calibrate", "coarse", "refine", "validate"), default="coarse")
    plan_parser.add_argument("--phase", choices=PHASES, default="all")
    plan_parser.add_argument("--seed", type=int, default=42)
    plan_parser.add_argument("--doses", type=float, nargs="+", default=[0.1, 0.25, 0.5], help="Positive layer-relative dose magnitudes; both signs are tested")
    plan_parser.add_argument("--neurons-per-layer", type=int, default=4)
    plan_parser.add_argument("--candidates", type=Path, help="JSON {model: ModelIdentity object, interventions: [Intervention fields]}; exact model identity must match; required for refine/validate")
    plan_parser.add_argument("--calibration-sha256", help="Digest of the exact model-bound residual calibration artifact")
    plan_parser.add_argument("--max-new-tokens", type=int, default=2048)
    plan_parser.add_argument("--shard", type=int, choices=range(SHARDS), help="Emit only shard 0..5; plan_id still identifies the complete plan")
    ipip_parser = commands.add_parser("score-ipip", help="Score JSON item ID -> integer 1..5 with the existing IPIP-50 reverse keys")
    ipip_parser.add_argument("responses", type=Path)
    generation_parser = commands.add_parser("score-generation", help="Check explicit completion metadata; does not infer personality or grade correctness")
    generation_parser.add_argument("record", type=Path, help="JSON: answer_text, answer_phase_reached, finish_reason, optional truncated/capability_correct/heldout")
    for command in (plan_parser, ipip_parser, generation_parser):
        command.add_argument("--output", type=Path, help="Write JSON exclusively to this path; defaults to stdout; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            model = ModelIdentity(**json.loads(args.model_manifest.read_text()))
            candidate_data = json.loads(args.candidates.read_text()) if args.candidates else None
            candidates = tuple(Intervention(**x) for x in candidate_data["interventions"]) if candidate_data else ()
            candidate_model = ModelIdentity(**candidate_data["model"]) if candidate_data else None
            result = build_plan(model, args.stage, seed=args.seed, phase=args.phase, doses=tuple(args.doses),
                                neurons_per_layer=args.neurons_per_layer, candidates=candidates,
                                candidate_model=candidate_model,
                                calibration_sha256=args.calibration_sha256, max_new_tokens=args.max_new_tokens)
            if args.shard is not None:
                result = select_shard(result, args.shard)
        elif args.command == "score-ipip":
            result = score_ipip50(json.loads(args.responses.read_text()))
        else:
            result = evaluate_completion(json.loads(args.record.read_text()))
        output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x") as stream:
                stream.write(output)
        else:
            print(output, end="")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
