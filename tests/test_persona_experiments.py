"""Protocol invariants; uses no model, GPU, network or optional dependencies."""

import persona  # Make the flat, deployment-compatible source modules importable.
from dataclasses import FrozenInstanceError, asdict, replace
import json
from pathlib import Path
import unittest

from persona_experiments import (
    Intervention, ModelIdentity, build_plan, digest, evaluate_completion,
    score_ipip50, select_shard, shard_for,
)


ROOT = Path(__file__).resolve().parent.parent


class PersonaProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        # Deliberately synthetic small architecture: no unverified target dimensions.
        self.model = ModelIdentity("test/fresh-qwen", "a" * 40, "b" * 64,
                                   "bf16-unquantized", 3, 16, 20, "test_dense")
        self.bank = json.loads((ROOT / "data/persona_prompts.json").read_text())

    def plan(self, **kwargs: object) -> dict:
        return build_plan(self.model, doses=(0.1,), neurons_per_layer=2,
                          prompt_bank=self.bank, **kwargs)

    def test_identity_is_immutable_and_rejects_mutable_revision(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            self.model.model_id = "other"  # type: ignore[misc]
        with self.assertRaises(ValueError):
            replace(self.model, checkpoint_revision="main")
        with self.assertRaises(ValueError):
            replace(self.model, quantization="")
        with self.assertRaises(ValueError):
            replace(self.model, weights_sha256="unknown")

    def test_semantics_are_explicit_and_do_not_confuse_add_and_gain(self) -> None:
        neuron = Intervention("neuron_add", 1, -0.5, "answer", neuron=8).specification()
        residual = Intervention("residual_add", 1, 0.5, trait="openness").specification()
        gain = Intervention("mlp_scale", 1, -1).specification()
        self.assertEqual(neuron["site"], "mlp.down_proj.input")
        self.assertIn("l2(W_down[:,n])", neuron["formula"])
        self.assertEqual(neuron["phase"], "answer")
        self.assertEqual(residual["site"], "transformer_block.output")
        self.assertIn("v / l2(v)", residual["formula"])
        self.assertEqual(gain["formula"], "a *= dose")
        for args in [("neuron_add", 0, float("nan")), ("residual_add", 0, 1)]:
            with self.assertRaises(ValueError):
                Intervention(*args)
        with self.assertRaises(ValueError):
            Intervention("neuron_add", 1, 1, neuron=20).validate_for(self.model)

    def test_deterministic_disjoint_exhaustive_six_shards(self) -> None:
        plan = self.plan()
        self.assertEqual(plan, self.plan())
        shards = [{x["condition_id"] for x in select_shard(plan, i)["conditions"]} for i in range(6)]
        self.assertTrue(all(shards))
        self.assertEqual(sum(len(s) for s in shards), len(plan["conditions"]))
        self.assertEqual(set.union(*shards), {x["condition_id"] for x in plan["conditions"]})
        for i in range(6):
            for j in range(i):
                self.assertFalse(shards[i] & shards[j])
        for row in plan["conditions"]:
            body = {k: v for k, v in row.items() if k not in ("condition_id", "shard")}
            self.assertEqual(row["condition_id"], digest(body))
            self.assertEqual(row["shard"], shard_for(row["condition_id"]))
        with self.assertRaises(ValueError):
            select_shard(plan, 6)
        with self.assertRaises(ValueError):
            shard_for("test", 3)

    def test_model_phase_and_dose_changes_cannot_reuse_ids(self) -> None:
        baseline = self.plan()
        self.assertNotEqual(baseline["plan_id"], self.plan(phase="answer")["plan_id"])
        other = build_plan(replace(self.model, quantization="test-int8"), doses=(0.1,), neurons_per_layer=2,
                           prompt_bank=self.bank)
        old_ids = {x["condition_id"] for x in baseline["conditions"]}
        self.assertFalse(old_ids & {x["condition_id"] for x in other["conditions"]})
        self.assertNotEqual(baseline["plan_id"], self.plan(seed=43)["plan_id"])

    def test_controls_are_matched_and_shuffled_artifacts_cannot_be_reused(self) -> None:
        plan = self.plan()
        rows = {x["condition_id"]: x for x in plan["conditions"]}
        self.assertIn("unsteered", {x["control"] for x in rows.values()})
        self.assertIn("prompt_only", {x["control"] for x in rows.values()})
        for row in rows.values():
            if row["control"] in ("matched_norm_random", "label_shuffled"):
                reference = rows[row["metadata"]["reference_condition_id"]]
                self.assertEqual(reference["seed"], row["seed"])
                self.assertEqual(reference["generation"], row["generation"])
                self.assertEqual(reference["prompt_bank_sha256"], row["prompt_bank_sha256"])
            if row["control"] == "matched_norm_random":
                self.assertIn("aggregate", row["random_control"]["norm_match"])
            if row["control"] == "label_shuffled":
                self.assertIn("label_shuffled_direction", row["unresolved_requirements"])
                self.assertTrue(row["label_shuffle"]["reuse_true_direction_forbidden"])
                self.assertFalse(row["ready_for_runtime_validation"])
            self.assertFalse(row["flags"]["validated_effect"])
            self.assertFalse(row["flags"]["promotion_eligible"])

    def test_refinement_signs_groups_provenance_and_holdout(self) -> None:
        candidates = (Intervention("neuron_add", 0, 0.2, neuron=2),
                      Intervention("neuron_add", 1, -0.3, neuron=4),
                      Intervention("residual_add", 2, 0.1, trait="openness", direction_sha256="c" * 64))
        kwargs = {"stage": "validate", "candidates": candidates, "candidate_model": self.model,
                  "calibration_sha256": "d" * 64}
        plan = self.plan(**kwargs)
        reverse_order = self.plan(**{**kwargs, "candidates": tuple(reversed(candidates))})
        self.assertEqual(plan, reverse_order)
        interactions = {x["metadata"].get("interaction") for x in plan["conditions"]}
        self.assertTrue({"cross_layer", "full_candidate_group", "full_group_sign_reversal", "group_leave_one_out"} <= interactions)
        groups = [x for x in plan["conditions"] if x["control"] == "intervention" and x["metadata"].get("interaction") == "cross_layer"]
        self.assertEqual(len(groups), 12)  # 3 pairs x all four polarity combinations.
        for row in plan["conditions"]:
            self.assertTrue(row["flags"]["heldout"])
            self.assertFalse(row["flags"]["fit_on_heldout"])
        selected = {p["id"] for p in self.bank["prompts"] if p["split"] == "heldout"}
        self.assertEqual(set(plan["prompt_ids"]), selected)
        with self.assertRaises(ValueError):
            self.plan(stage="refine", candidates=candidates)
        with self.assertRaises(ValueError):
            self.plan(**{**kwargs, "candidate_model": replace(self.model, model_id="historical/4b")})

    def test_calibration_is_random_and_has_no_learned_trait_claim(self) -> None:
        plan = self.plan(stage="calibrate", calibration_sha256="d" * 64)
        rows = [r for r in plan["conditions"] if r["control"] == "calibration_random"]
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertEqual(row["interventions"][0]["kind"], "random_residual_add")
            self.assertIsNone(row["interventions"][0]["trait"])
            body = {k: v for k, v in row.items() if k not in ("condition_id", "shard")}
            self.assertEqual(row["condition_id"], digest(body))

    def test_ipip_reverse_keys_and_missing_responses(self) -> None:
        neutral = score_ipip50({str(i): 3 for i in range(1, 51)})
        self.assertTrue(neutral["complete"])
        self.assertTrue(all(x["sum_10_to_50"] == 30 for x in neutral["scores"].values()))
        all_high = score_ipip50({str(i): 5 for i in range(1, 51)})
        expected = {"Extraversion": 30, "Agreeableness": 34, "Conscientiousness": 34, "Neuroticism": 42, "Openness": 38}
        self.assertEqual({k: v["sum_10_to_50"] for k, v in all_high["scores"].items()}, expected)
        partial = score_ipip50({"1": 5})
        self.assertFalse(partial["complete"])
        self.assertIsNone(partial["scores"]["Extraversion"]["sum_10_to_50"])
        for invalid in ({"51": 3}, {"1": True}, {"1": 3.5}, {"1": 6}, {"1": "5"}):
            with self.assertRaises(ValueError):
                score_ipip50(invalid)

    def test_thinking_only_and_truncation_are_never_behavioral_successes(self) -> None:
        good = {"answer_text": "A final answer.", "answer_phase_reached": True,
                "finish_reason": "eos", "heldout": True, "capability_correct": True}
        self.assertTrue(evaluate_completion(good)["eligible_for_behavioral_scoring"])
        self.assertFalse(evaluate_completion(good)["promotion_eligible"])
        for change in ({"finish_reason": "length"}, {"answer_phase_reached": False},
                       {"answer_text": " "}, {"truncated": True}, {"finish_reason": None}):
            self.assertFalse(evaluate_completion({**good, **change})["complete"])

    def test_prompt_and_historical_recipe_provenance(self) -> None:
        ids = [p["id"] for p in self.bank["prompts"]]
        self.assertEqual(len(ids), len(set(ids)))
        for split in ("discovery", "heldout"):
            prompts = [p for p in self.bank["prompts"] if p["split"] == split]
            self.assertEqual(len({p["category"] for p in prompts}), 6)
            self.assertTrue(any(p["capability"] for p in prompts))
        recipes = json.loads((ROOT / "data/lyra_recipes.json").read_text())
        self.assertEqual(recipes["model_id"], "Qwen/Qwen3.5-4B")
        self.assertIsNone(recipes["source"]["tweet_url"])
        self.assertEqual(recipes["source"]["project_revision"], "f79b2196e29aacbad89684938baac8d97b172fd1")
        combo = next(r for r in recipes["recipes"] if r["id"] == "five_neuron_combo")
        self.assertEqual([(p["layer"], p["neuron"], p["dose"]) for p in combo["pushes"]],
                         [(18, 5796, 0.6), (15, 3920, 0.6), (12, 2920, -0.5), (10, 6295, 0.4), (16, 8168, -0.3)])


if __name__ == "__main__":
    unittest.main()
