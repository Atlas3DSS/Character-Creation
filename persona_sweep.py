"""Bind discovery candidates to deterministic native Flare validation requests.

No inference or fitting happens here. ``plan`` creates the first 12-candidate,
six-heldout-prompt, two-seed sweep once verified map/bank files exist. ``adapt``
converts supported persona_experiments conditions without silently changing
their control, sampling, phase, checkpoint, or artifact requirements. Unsupported
realized per-token norm controls, shuffled fits, and MLP gains fail closed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from persona_experiments import Intervention, ModelIdentity, TRAITS, digest as experiment_digest
from persona_map import file_sha256
from persona_vectors import ModelSpec, VectorBank, canonical_digest


def direction_digest(vector) -> str:
    values = np.asarray(vector, dtype="<f4")
    return hashlib.sha256(json.dumps(list(values.shape)).encode() + values.tobytes()).hexdigest()


class VerifiedBank(VectorBank):
    """Freeze map/bank provenance before selecting or rendering any condition."""

    def __init__(self, root: Path, mapping: dict):
        model = ModelSpec(**mapping["model"])
        super().__init__(root, model)
        if mapping["model_identity"] != model.identity:
            raise ValueError("Map model identity is inconsistent")
        self.mapping = mapping
        self.map_identity = canonical_digest(mapping)
        self.calibration_identity = canonical_digest({key: mapping[key] for key in
            ("model_identity", "plan_sha256", "runtime_identity", "source_hashes")})
        self.entries = {entry["layer"]: entry for entry in mapping["banks"]}
        if len(self.entries) != len(mapping["banks"]) or set(self.entries) != set(range(model.layers)):
            raise ValueError("Verified map must include every unique layer bank")
        if (set(mapping["pair_counts"]) != set(TRAITS)
                or any(type(n) is not int or n < 1 for n in mapping["pair_counts"].values())):
            raise ValueError("Map requires paired discovery coverage for all traits")
        self.cache = {}
        coordinates = set()
        for node in mapping["neurons"]:
            layer, neuron = node["layer"], node["neuron"]
            if (type(layer) is not int or type(neuron) is not int or (layer, neuron) in coordinates
                    or node.get("status") != "candidate" or node.get("causal_effect") is not None):
                raise ValueError("Map contains invalid or promoted discovery coordinates")
            coordinates.add((layer, neuron))
            data = self.layer(layer)
            matches = np.flatnonzero(data["neuron_ids"] == neuron)
            if len(matches) != 1:
                raise ValueError("Displayed neuron is absent from verified bank")
            index = int(matches[0])
            observation = node.get("activation_observation", {})
            if (observation.get("sample") != "discovery prompts only" or observation.get("phase") != "final prompt token"
                    or observation.get("baseline_mean") != float(data["activation_mean"][index])
                    or observation.get("baseline_sd") != float(data["activation_sd"][index])
                    or observation.get("high_minus_low") != {trait: float(data["activation_contrasts"][t, index]) for t, trait in enumerate(TRAITS)}
                    or node.get("trait_alignment") != {trait: float(data["trait_alignment"][t, index]) for t, trait in enumerate(TRAITS)}):
                raise ValueError("Displayed discovery evidence differs from verified bank")
        for layer in range(model.layers):
            if {n for l, n in coordinates if l == layer} != set(self.layer(layer)["neuron_ids"].tolist()):
                raise ValueError("Map candidate coverage differs from bank")

    def layer(self, layer: int) -> dict:
        if layer in self.cache:
            return self.cache[layer]
        entry = self.entries.get(layer)
        if entry is None or entry["filename"] != f"layer_{layer:02d}.npz":
            raise ValueError("Layer absent from bound bank manifest")
        if file_sha256(self.root / entry["filename"]) != entry["sha256"]:
            raise ValueError("Bank changed after map assembly")
        data = super().layer(layer)
        if (str(data["plan_sha256"]) != self.mapping["plan_sha256"]
                or str(data["runtime_identity"]) != self.mapping["runtime_identity"]
                or data["source_hashes"].tolist() != self.mapping["source_hashes"]):
            raise ValueError("Bank calibration provenance differs from map")
        self.cache[layer] = data
        return data


def _sampling(generation: dict) -> dict:
    tokens = generation.get("max_new_tokens", generation.get("max_tokens", 2048))
    temperature = generation.get("temperature", .7)
    if type(tokens) is not int or tokens <= 0 or type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Invalid generation budget or temperature")
    if generation.get("do_sample") is False and temperature != 0:
        raise ValueError("A non-sampling condition requires temperature zero")
    top_p = generation.get("top_p", 1. if temperature == 0 else .95)
    top_k = generation.get("top_k", 1 if temperature == 0 else 20)
    if not isinstance(top_p, (int, float)) or not 0 < top_p <= 1 or type(top_k) is not int or top_k < 1:
        raise ValueError("Invalid top-p or top-k")
    return {"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_tokens": tokens}


def adapt_condition(condition: dict, bank: VerifiedBank, *, expected_legacy_model: ModelIdentity | None = None) -> dict:
    """Render a condition, rejecting unsupported semantics instead of weakening it."""
    legacy = "model_id" in condition.get("model", {})
    if legacy:
        if expected_legacy_model is None or condition["model"] != asdict(expected_legacy_model):
            raise ValueError("Legacy conditions require the independently verified full model/weights binding")
        old = expected_legacy_model
        equivalent = ModelSpec(old.model_id, old.checkpoint_revision, old.quantization,
                               old.n_layers, old.hidden_size, old.intermediate_size)
        if equivalent != bank.model:
            raise ValueError("Legacy checkpoint differs from calibrated bank")
        body = {k: v for k, v in condition.items() if k not in ("condition_id", "shard")}
        if experiment_digest(body) != condition.get("condition_id"):
            raise ValueError("Legacy condition content changed after planning")
    elif condition.get("model_identity") != bank.model.identity:
        raise ValueError("Condition checkpoint differs from calibrated bank")
    elif ("condition_id" in condition and canonical_digest({k: v for k, v in condition.items()
            if k != "condition_id"}) != condition["condition_id"]):
        raise ValueError("Native condition content changed after planning")
    if condition.get("unresolved_requirements"):
        raise ValueError("Condition still has unresolved artifact requirements")
    control = condition["control"]
    if control in ("matched_norm_random", "label_shuffled"):
        raise ValueError("Realized per-token random norms and shuffled-label fits require separate evidence; static vectors cannot substitute")
    if control not in {"intervention", "unsteered", "prompt_only", "zero_vector", "nominal_norm_random", "calibration_random"}:
        raise ValueError("Unsupported control semantics")
    interventions = condition.get("interventions", [])
    if control in {"unsteered", "prompt_only"} and interventions:
        raise ValueError("Untreated controls cannot contain interventions")
    if interventions and condition.get("calibration_sha256") != bank.calibration_identity:
        raise ValueError("Condition does not bind the calibrated artifacts")
    if control not in {"unsteered", "prompt_only"} and not interventions:
        raise ValueError("An intervention/control arm must specify its sites")
    phases = {item.get("phase", "all") for item in interventions}
    if len(phases) > 1 or any(p not in {"all", "prefill", "decode"} for p in phases):
        raise ValueError("Native adapter requires one all/prefill/decode phase; thinking/answer boundaries are unsupported")
    phase = next(iter(phases), "all")
    pushes, zero_sites = [], {}
    for item in interventions:
        kind, layer, dose = item["kind"], item["layer"], item["dose"]
        if legacy and kind in {"neuron_add", "residual_add"}:
            original = Intervention(**{key: item[key] for key in Intervention.__dataclass_fields__})
            if original.specification() != item:
                raise ValueError("Legacy intervention site/formula/normalization semantics were altered")
        if type(dose) not in (int, float) or not math.isfinite(dose) or abs(dose) > 4:
            raise ValueError("Dose outside supported finite range")
        if kind in {"neuron_add", "column_add"}:
            push = {"kind": "neuron" if kind == "neuron_add" else "column",
                    "layer": layer, "neuron": item["neuron"], "dose": dose}
            site = "mlp_input" if kind == "neuron_add" else "mlp_out"
        elif kind == "residual_add":
            trait = item["trait"]
            if trait not in TRAITS or item.get("direction_sha256") != direction_digest(bank.layer(layer)["trait_" + trait]):
                raise ValueError("Residual direction digest differs from bound trait artifact")
            push = {"kind": "trait", "layer": layer, "trait": trait, "dose": dose}
            site = "residual"
        elif kind == "random_residual_add":
            push = {"kind": "random", "layer": layer, "site": "residual", "dose": dose,
                    "seed": item.get("seed", condition.get("random_control", {}).get("seed"))}
            site = "residual"
        else:
            raise ValueError(f"Unsupported intervention kind: {kind}; no multiplicative gain substitution")
        pushes.append(push)
        zero_sites[f"layers.{layer}.{site}"] = bank.model.intermediate if site == "mlp_input" else bank.model.hidden
    if control == "zero_vector" and (not interventions or any(x["dose"] != 0 for x in interventions)):
        raise ValueError("Zero-vector controls need explicit zero-dose sites")
    program, details = bank.program(pushes, phase=phase)
    if control == "zero_vector":
        program["sites"] = [{"site": site, "vector": [0.] * width} for site, width in sorted(zero_sites.items())]
        details["program_sha256"] = canonical_digest(program)
    if len(program["sites"]) > 16:
        raise ValueError("Condition exceeds prepared native maximum of 16 active sites")
    if any(not math.isfinite(value) for site in program["sites"] for value in site["vector"]):
        raise ValueError("Rendered native vectors contain nonfinite float32 values")
    system = condition.get("metadata", {}).get("system_prompt", "") if control == "prompt_only" else ""
    return {"steering": None if control in {"unsteered", "prompt_only"} else program,
            "steering_details": details, "system": system, "generation": _sampling(condition.get("generation", {}))}


def select_candidates(bank: VerifiedBank, dose: float = .25) -> list[dict]:
    """Seven neurons from two evidence types and one residual direction per trait."""
    if not math.isfinite(dose) or not 0 < dose <= 4:
        raise ValueError("Candidate dose must be finite and in (0,4]")
    nodes = bank.mapping["neurons"]
    chosen, coordinates, per_layer = [], set(), Counter()

    def add_neuron(ranked, criterion):
        available = [(score, trait, node) for score, trait, node in ranked
                     if (node["layer"], node["neuron"]) not in coordinates and per_layer[node["layer"]] < 2]
        if not available:
            raise ValueError("Need seven distinct discovery neurons across at least four layers")
        score, trait, node = min(available, key=lambda row: (-row[0], row[2]["layer"], row[2]["neuron"], row[1]))
        coordinates.add((node["layer"], node["neuron"]))
        per_layer[node["layer"]] += 1
        chosen.append({"kind": "neuron_add", "layer": node["layer"], "neuron": node["neuron"], "dose": dose,
                       "phase": "all", "selection": {"criterion": criterion, "trait": trait, "absolute_score": score}})

    for trait in TRAITS:
        add_neuron([(abs(n["activation_observation"]["high_minus_low"][trait]), trait, n) for n in nodes],
                   "absolute paired final-token SwiGLU high-minus-low contrast")
    for _ in range(2):
        add_neuron([(abs(n["trait_alignment"][trait]), trait, n) for n in nodes for trait in TRAITS],
                   "absolute decoded down-column/residual-direction cosine")
    for trait in TRAITS:
        ranked = []
        for layer in range(bank.model.layers):
            data = bank.layer(layer)
            direction = np.asarray(data["trait_" + trait], dtype=np.float64)
            ranked.append((float(np.linalg.norm(direction)) / float(data["residual_median"]), layer))
        score, layer = min(ranked, key=lambda row: (-row[0], row[1]))
        chosen.append({"kind": "residual_add", "layer": layer, "trait": trait, "dose": dose, "phase": "all",
                       "direction_sha256": direction_digest(bank.layer(layer)["trait_" + trait]),
                       "selection": {"criterion": "paired residual direction L2 / baseline median residual L2", "score": score}})
    for candidate in chosen:
        candidate["candidate_id"] = canonical_digest(candidate)
    return chosen


def first_plan(bank: VerifiedBank, corpus: dict, *, seeds=(7035, 7036), dose=.25, control_seed=19273) -> dict:
    """Freeze 71 conditions; each gets six heldout prompts and the same two seeds."""
    if len(seeds) != 2 or len(set(seeds)) != 2 or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("First validation uses exactly two distinct uint32 generation seeds")
    if type(control_seed) is not int or not 0 <= control_seed < 2**32:
        raise ValueError("Random-control seed must be uint32")
    heldout = {}
    for prompt in sorted(corpus["prompts"], key=lambda row: row["id"]):
        if prompt["split"] == "heldout":
            heldout.setdefault(prompt["category"], prompt["id"])
    if set(heldout) != {"emotional", "identity", "reasoning", "social", "practical", "creative"}:
        raise ValueError("First validation needs six heldout categories")
    candidates = select_candidates(bank, dose)
    conditions = []
    generation = {"max_new_tokens": 2048, "temperature": .7, "top_p": .95, "top_k": 20, "do_sample": True}

    def add(control, interventions=(), **metadata):
        row = {"model_identity": bank.model.identity, "calibration_sha256": bank.calibration_identity,
               "control": control, "interventions": list(interventions), "generation": generation,
               "stage": "validate", "flags": {"heldout": True, "fit_on_heldout": False,
                   "requires_complete_answer": True, "validated_effect": False, "promotion_eligible": False},
               "metadata": metadata}
        row["condition_id"] = canonical_digest(row)
        conditions.append(row)
        return row

    add("unsteered")
    for trait in TRAITS:
        for pole in ("high", "low"):
            add("prompt_only", trait=trait, level=pole, system_prompt=corpus["trait_prompts"][trait][pole])
    for candidate in candidates:
        item = {k: v for k, v in candidate.items() if k not in {"selection", "candidate_id"}}
        cid = candidate["candidate_id"]
        add("zero_vector", [{**item, "dose": 0.}], candidate_id=cid)
        random_seed = int(canonical_digest({"candidate_id": cid, "control_seed": control_seed})[:8], 16)
        if item["kind"] == "neuron_add":
            alternatives = sorted(set(bank.layer(item["layer"])["neuron_ids"].tolist()) - {item["neuron"]})
            if not alternatives:
                raise ValueError("Neuron random control requires another extracted coordinate in the same layer")
            random_item = {**item, "neuron": int(np.random.default_rng(random_seed).choice(alternatives))}
            distribution = "uniform other extracted candidate coordinate in the same layer; not an unbiased full-neuron sample"
        else:
            random_item = {"kind": "random_residual_add", "layer": item["layer"], "dose": dose,
                           "phase": item["phase"], "seed": random_seed}
            distribution = "isotropic Gaussian residual direction, normalized by its L2"
        for sign in (-1, 1):
            reference = add("intervention", [{**item, "dose": sign * dose}], candidate_id=cid, sign=sign)
            add("nominal_norm_random", [{**random_item, "dose": sign * dose}], candidate_id=cid, sign=sign,
                reference_condition_id=reference["condition_id"], random_seed=random_seed,
                distribution=distribution, norm_match="same site and layer; nominal decoded output-equivalent L2 before BF16/quantized projection",
                realized_norm_match=False)
    spec = {"schema": "character-creation.sweep-spec.v1", "model": bank.model.__dict__,
            "model_identity": bank.model.identity, "stage": "validate", "seeds": list(seeds),
            "prompt_ids": [heldout[key] for key in sorted(heldout)], "prompt_bank_sha256": experiment_digest(corpus),
            "conditions": sorted(conditions, key=lambda row: row["condition_id"]), "candidates": candidates,
            "selection_policy": "discovery only; seven distinct neurons, at most two per layer, plus one residual direction per trait; no heldout fitting",
            "calibration_sha256": bank.calibration_identity, "map_sha256": bank.map_identity,
            "random_control_seed": control_seed, "status": "planned_not_run"}
    return adapt_plan(spec, bank, corpus)


def adapt_plan(plan: dict, bank: VerifiedBank, corpus: dict, *, expected_legacy_model: ModelIdentity | None = None) -> dict:
    """Produce persona_fleet jobs; all supported conditions are retained or fail."""
    if plan.get("prompt_bank_sha256") != experiment_digest(corpus):
        raise ValueError("Prompt bank changed after experiment planning")
    if "plan_id" in plan and experiment_digest({k: v for k, v in plan.items() if k != "plan_id"}) != plan["plan_id"]:
        raise ValueError("Experiment plan content changed after planning")
    expected_model = asdict(expected_legacy_model) if "model_id" in plan.get("model", {}) and expected_legacy_model else bank.model.__dict__
    if plan.get("model") != expected_model:
        raise ValueError("Plan model differs from independently bound model")
    split = "heldout" if plan["stage"] == "validate" else "discovery"
    prompts = {p["id"]: p for p in corpus["prompts"]}
    ids = plan["prompt_ids"]
    if len(set(ids)) != len(ids) or any(identity not in prompts or prompts[identity]["split"] != split for identity in ids):
        raise ValueError("Plan prompt IDs cross discovery/heldout boundaries")
    conditions = plan["conditions"]
    if not conditions or len({row["condition_id"] for row in conditions}) != len(conditions):
        raise ValueError("Conditions must be nonempty and unique")
    rendered = {}
    for condition in conditions:
        flags = condition.get("flags", {})
        if condition.get("stage") != plan["stage"] or flags.get("heldout") is not (split == "heldout") or flags.get("fit_on_heldout") is not False:
            raise ValueError("Condition stage/split/fit flags differ from the plan")
        if "model_id" not in condition.get("model", {}):
            if canonical_digest({k: v for k, v in condition.items() if k != "condition_id"}) != condition["condition_id"]:
                raise ValueError("Condition content changed after planning")
        rendered[condition["condition_id"]] = adapt_condition(condition, bank, expected_legacy_model=expected_legacy_model)
    for condition in conditions:
        if condition["control"] == "nominal_norm_random":
            reference_id = condition.get("metadata", {}).get("reference_condition_id")
            if reference_id not in rendered:
                raise ValueError("Matched random control is missing its reference condition")
            reference, control = rendered[reference_id], rendered[condition["condition_id"]]
            by_id = {item["condition_id"]: item for item in conditions}
            reference_seeds = [by_id[reference_id]["seed"]] if "seed" in by_id[reference_id] else plan.get("seeds")
            control_seeds = [condition["seed"]] if "seed" in condition else plan.get("seeds")
            if (reference["generation"] != control["generation"] or reference["steering"]["phase"] != control["steering"]["phase"]
                    or reference_seeds != control_seeds
                    or reference["steering_details"]["relative_norms"].keys() != control["steering_details"]["relative_norms"].keys()
                    or any(not math.isclose(value, control["steering_details"]["relative_norms"][site], rel_tol=1e-12, abs_tol=1e-12)
                           for site, value in reference["steering_details"]["relative_norms"].items())):
                raise ValueError("Random control does not match nominal reference norm/site/phase/sampling")
    jobs = []
    for condition in conditions:
        cid = condition["condition_id"]
        converted = rendered[cid]
        seeds = [condition["seed"]] if "seed" in condition else plan["seeds"]
        if len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
            raise ValueError("Generation seeds must be unique uint32 integers")
        for seed in seeds:
            for identity in ids:
                prompt = prompts[identity]
                text, expected = prompt["text"], None
                # Preserve the original task and record the exact grading suffix.
                suffix = ""
                if identity == "reasoning_h1":
                    if prompt.get("reference_answer") != "233":
                        raise ValueError("Heldout reasoning reference changed; update its explicit final-answer grader")
                    expected, suffix = "233", " End with ANSWER: followed only by the number of occupied seats."
                elif identity == "practical_h1":
                    if prompt.get("reference_answer") != "Yes; 105 minutes total, 15 minutes left.":
                        raise ValueError("Heldout planning reference changed; update its explicit final-answer grader")
                    expected, suffix = "15", " End with ANSWER: followed only by the number of minutes left."
                job = {"condition": cid, "system": converted["system"], "seed": seed, "kind": "behavior",
                       "category": prompt["category"], "item_id": identity, "split": split,
                       "prompt": text + suffix, "source_prompt_sha256": canonical_digest(prompt),
                       "grading_suffix": suffix, "reference_answer": prompt.get("reference_answer"),
                       "expected_answer": expected, "capability": prompt["capability"],
                       "max_tokens": converted["generation"]["max_tokens"], "thinking": False,
                       "generation": {k: v for k, v in converted["generation"].items() if k != "max_tokens"},
                       "model_identity": bank.model.identity, "control": condition["control"],
                       "flags": condition["flags"]}
                if converted["steering"] is not None:
                    job["steering"] = converted["steering"]
                    job["program_sha256"] = converted["steering_details"]["program_sha256"]
                job["id"] = canonical_digest(job)
                jobs.append(job)
    jobs.sort(key=lambda row: row["id"])
    return {"schema": "character-creation.native-sweep.v1", "status": "planned_not_run",
            "model": bank.model.__dict__, "model_identity": bank.model.identity,
            "source_plan_sha256": experiment_digest(plan), "map_sha256": bank.map_identity,
            "calibration_sha256": bank.calibration_identity, "bank_files": list(bank.entries.values()),
            "prompt_bank_sha256": experiment_digest(corpus), "stage": plan["stage"],
            "candidates": plan.get("candidates", []), "conditions": conditions,
            "rendered_conditions": rendered, "jobs": jobs, "shards": 6,
            "shard_counts": [sum(int(job["id"], 16) % 6 == shard for job in jobs) for shard in range(6)],
            "claim": "planned controlled inference, no validated effect or personality proof",
            "evaluation": {"complete_answers_required": True, "report_truncation_and_errors_separately": True,
                           "capability_final_answers_and_explanations_reviewed_separately": True,
                           "blind_behavior_review_required": True, "automatic_promotion": False,
                           "fit_on_heldout": False}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "adapt"):
        p = sub.add_parser(command)
        p.add_argument("--banks", type=Path, required=True, help="Directory containing assembled map and layer NPZ files")
        p.add_argument("--prompts", type=Path, default=Path(__file__).parent / "data/persona_prompts.json")
        p.add_argument("--output", type=Path, required=True, help="New immutable JSON plan; existing files are never overwritten")
        if command == "plan":
            p.add_argument("--seeds", type=int, nargs=2, default=[7035, 7036])
            p.add_argument("--dose", type=float, default=.25)
            p.add_argument("--control-seed", type=int, default=19273)
        else:
            p.add_argument("--plan", type=Path, required=True)
            p.add_argument("--model-binding", type=Path, required=True, help="Independently verified full persona_experiments ModelIdentity JSON")
    args = parser.parse_args()
    mapping = json.loads((args.banks / "persona_map.json").read_text())
    bank = VerifiedBank(args.banks, mapping)
    corpus = json.loads(args.prompts.read_text())
    result = (first_plan(bank, corpus, seeds=args.seeds, dose=args.dose, control_seed=args.control_seed)
              if args.command == "plan" else adapt_plan(json.loads(args.plan.read_text()), bank, corpus,
                  expected_legacy_model=ModelIdentity(**json.loads(args.model_binding.read_text()))))
    with args.output.open("x") as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"output": str(args.output), "conditions": len(result["conditions"]),
                      "jobs": len(result["jobs"]), "shard_counts": result["shard_counts"]}))


if __name__ == "__main__":
    main()
