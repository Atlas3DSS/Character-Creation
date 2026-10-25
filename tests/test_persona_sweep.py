"""CPU-only native-plan binding, dose/control, split, and shard invariants."""

import persona  # Make the flat, deployment-compatible source modules importable.
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from persona_experiments import Intervention, ModelIdentity, TRAITS, _condition, digest
from persona_sweep import VerifiedBank, adapt_condition, adapt_plan, direction_digest, first_plan
from persona_vectors import ModelSpec, canonical_digest


class PersonaSweepTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = ModelSpec("test/qwen", "a" * 40, "test-nvfp4", 6, 3, 16)
        self.corpus = json.loads((Path(__file__).parent.parent / "data/persona_prompts.json").read_text())
        self.mapping = {"model": self.model.__dict__, "model_identity": self.model.identity,
                        "plan_sha256": "b" * 64, "runtime_identity": "c" * 64,
                        "source_hashes": [str(i) * 64 for i in range(6)],
                        "pair_counts": {trait: 12 for trait in TRAITS}, "neurons": [], "banks": []}
        for layer in range(6):
            ids = np.arange(4, dtype=np.int64)
            alignments = np.array([[(layer + n + t) / 20 for n in ids] for t in range(5)], np.float32)
            contrasts = np.array([[(layer + 1) * (n + 1) * (t + 1) for n in ids] for t in range(5)], np.float32)
            mean, sd = np.arange(4, dtype=np.float32), np.ones(4, dtype=np.float32)
            path = self.root / f"layer_{layer:02d}.npz"
            np.savez(path, model_identity=self.model.identity, plan_sha256="b" * 64, runtime_identity="c" * 64,
                     source_hashes=np.array(self.mapping["source_hashes"]), residual_median=10. + layer,
                     neuron_ids=ids, down_columns=np.array([[1, 2, 3, 4], [2, 2, 2, 2], [3, 4, 5, 6]], np.float32),
                     trait_alignment=alignments, activation_mean=mean, activation_sd=sd, activation_contrasts=contrasts,
                     **{"trait_" + trait: np.array([t + 1, layer + 1, .5], np.float32) for t, trait in enumerate(TRAITS)})
            self.mapping["banks"].append({"layer": layer, "filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            for n in ids:
                self.mapping["neurons"].append({"layer": layer, "neuron": int(n), "status": "candidate", "causal_effect": None,
                    "trait_alignment": {trait: float(alignments[t, n]) for t, trait in enumerate(TRAITS)},
                    "activation_observation": {"phase": "final prompt token", "sample": "discovery prompts only",
                        "baseline_mean": float(mean[n]), "baseline_sd": float(sd[n]),
                        "high_minus_low": {trait: float(contrasts[t, n]) for t, trait in enumerate(TRAITS)}}})
        self.bank = VerifiedBank(self.root, self.mapping)

    def test_first_plan_balanced_matched_conditions_and_exhaustive_six_shards(self):
        plan = first_plan(self.bank, self.corpus)
        self.assertEqual((len(plan["candidates"]), len(plan["conditions"]), len(plan["jobs"])), (12, 71, 852))
        self.assertEqual(sum(plan["shard_counts"]), 852)
        self.assertEqual(len({job["id"] for job in plan["jobs"]}), 852)
        self.assertEqual({j["seed"] for j in plan["jobs"]}, {7035, 7036})
        self.assertEqual(len({j["item_id"] for j in plan["jobs"]}), 6)
        for job in plan["jobs"]:
            self.assertEqual(job["split"], "heldout")
            self.assertEqual(job["max_tokens"], 2048)
            self.assertFalse(job["thinking"])
            self.assertFalse(job["flags"]["fit_on_heldout"])
            self.assertEqual(job["model_identity"], self.model.identity)
            self.assertEqual(job["id"], canonical_digest({k: v for k, v in job.items() if k != "id"}))
            if job["capability"]:
                self.assertIn(job["expected_answer"], {"233", "15"})
                self.assertIn("ANSWER:", job["grading_suffix"])
        for condition in plan["conditions"]:
            self.assertEqual(sum(j["condition"] == condition["condition_id"] for j in plan["jobs"]), 12)
            rendered = plan["rendered_conditions"][condition["condition_id"]]
            if condition["control"] == "zero_vector":
                self.assertTrue(rendered["steering"]["sites"])
                self.assertTrue(all(not any(site["vector"]) for site in rendered["steering"]["sites"]))
            if condition["control"] == "nominal_norm_random":
                reference = plan["rendered_conditions"][condition["metadata"]["reference_condition_id"]]
                self.assertEqual(rendered["steering"]["phase"], reference["steering"]["phase"])
                np.testing.assert_allclose(list(rendered["steering_details"]["relative_norms"].values()),
                                           list(reference["steering_details"]["relative_norms"].values()), rtol=1e-12)
                self.assertFalse(condition["metadata"]["realized_norm_match"])

    def test_deterministic_selection_ignores_map_order_and_heldout_text(self):
        original = first_plan(self.bank, self.corpus)
        changed = copy.deepcopy(self.mapping)
        changed["neurons"].reverse()
        changed["banks"].reverse()
        corpus = copy.deepcopy(self.corpus)
        for prompt in corpus["prompts"]:
            if prompt["split"] == "heldout":
                prompt["text"] += " Additional heldout wording."
        second = first_plan(VerifiedBank(self.root, changed), corpus)
        self.assertEqual(original["candidates"], second["candidates"])
        self.assertEqual(original, first_plan(self.bank, self.corpus))

    def legacy(self, item, control="intervention"):
        identity = ModelIdentity(self.model.model, self.model.revision, "f" * 64, self.model.quantization,
                                 self.model.layers, self.model.hidden, self.model.intermediate, "TestArchitecture")
        condition = _condition(identity, "validate", 7035, (item,), control, digest(self.corpus),
                               self.bank.calibration_identity, 2048)
        return identity, condition

    def test_legacy_neuron_arithmetic_and_sampling_remain_exact(self):
        identity, condition = self.legacy(Intervention("neuron_add", 2, -.25, "prefill", neuron=1))
        rendered = adapt_condition(condition, self.bank, expected_legacy_model=identity)
        self.assertEqual(rendered["generation"], {"temperature": 0., "top_p": 1., "top_k": 1, "max_tokens": 2048})
        site = rendered["steering"]["sites"][0]
        self.assertEqual(site["site"], "layers.2.mlp_input")
        expected = -.25 * 12 / np.linalg.norm(self.bank.layer(2)["down_columns"][:, 1].astype(np.float64))
        self.assertEqual(np.count_nonzero(site["vector"]), 1)
        self.assertAlmostEqual(site["vector"][1], expected, places=6)
        with self.assertRaises(ValueError):
            adapt_condition(condition, self.bank)
        altered = copy.deepcopy(condition)
        altered["interventions"][0]["site"] = "after_the_down_projection"
        altered["condition_id"] = digest({k: v for k, v in altered.items() if k not in {"condition_id", "shard"}})
        with self.assertRaises(ValueError):
            adapt_condition(altered, self.bank, expected_legacy_model=identity)

    def test_unsupported_controls_phases_and_multiplication_fail_closed(self):
        for item, control in [(Intervention("mlp_scale", 1, -1), "intervention"),
                              (Intervention("neuron_add", 1, .25, "thinking", neuron=1), "intervention"),
                              (Intervention("neuron_add", 1, .25, neuron=1), "matched_norm_random")]:
            identity, condition = self.legacy(item, control)
            with self.assertRaises(ValueError):
                adapt_condition(condition, self.bank, expected_legacy_model=identity)
        identity, condition = self.legacy(Intervention("residual_add", 1, .25, trait="openness",
            direction_sha256=direction_digest(self.bank.layer(1)["trait_openness"])), "label_shuffled")
        with self.assertRaises(ValueError):
            adapt_condition(condition, self.bank, expected_legacy_model=identity)

    def test_binding_and_heldout_crossovers_rejected(self):
        altered = copy.deepcopy(self.mapping)
        altered["neurons"][0]["activation_observation"]["baseline_mean"] += 1
        with self.assertRaises(ValueError):
            VerifiedBank(self.root, altered)
        identity, condition = self.legacy(Intervention("neuron_add", 1, .25, neuron=1))
        plan = {"model": asdict(identity), "stage": "validate", "prompt_bank_sha256": digest(self.corpus),
                "prompt_ids": ["social_d1"], "conditions": [condition]}
        with self.assertRaises(ValueError):
            adapt_plan(plan, self.bank, self.corpus, expected_legacy_model=identity)
        with (self.root / "layer_00.npz").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaises(ValueError):
            VerifiedBank(self.root, self.mapping)

    def test_supported_groups_sum_without_rescaling_and_columns_use_post_down_site(self):
        condition = {"model_identity": self.model.identity, "calibration_sha256": self.bank.calibration_identity,
                     "control": "intervention", "interventions": [
                         {"kind": "neuron_add", "layer": 1, "neuron": 2, "dose": .25, "phase": "all"},
                         {"kind": "neuron_add", "layer": 1, "neuron": 2, "dose": -.1, "phase": "all"},
                         {"kind": "column_add", "layer": 1, "neuron": 2, "dose": .25, "phase": "all"}]}
        result = adapt_condition(condition, self.bank)
        by_site = {entry["site"]: np.array(entry["vector"]) for entry in result["steering"]["sites"]}
        column = self.bank.layer(1)["down_columns"][:, 2].astype(np.float64)
        self.assertEqual(result["steering_details"]["aggregate_scale"], 1.)
        np.testing.assert_allclose(by_site["layers.1.mlp_out"], .25 * 11 * column / np.linalg.norm(column), rtol=1e-6)
        self.assertAlmostEqual(by_site["layers.1.mlp_input"][2], .15 * 11 / np.linalg.norm(column), places=6)
        condition["interventions"][1]["phase"] = "prefill"
        with self.assertRaises(ValueError):
            adapt_condition(condition, self.bank)


if __name__ == "__main__":
    unittest.main()
