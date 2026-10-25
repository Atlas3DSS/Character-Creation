"""Numerical intervention and provenance checks using tiny synthetic banks only."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from persona_vectors import ModelSpec, VectorBank, canonical_digest
from persona_analyze import analyze, capability_answer, digest, ipip_digit, latest_attempt


class PersonaVectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = ModelSpec("test/fresh-qwen", "a" * 40, "test-fp32", 2, 4, 5)
        self.bank = VectorBank(self.root, self.model)
        self.save_layer()

    def save_layer(self, **changes: object) -> None:
        data = {
            "model_identity": self.model.identity,
            "residual_median": 10.0,
            "neuron_ids": np.array([1, 3]),
            "down_columns": np.array([[3., 0.], [4., 0.], [0., 0.], [0., 2.]]),
            "trait_openness": np.array([0., 0., 2., 0.]),
        }
        data.update(changes)
        np.savez(self.root / "layer_00.npz", **data)

    def test_relative_dose_uses_direction_norm_and_layer_median(self) -> None:
        program, details = self.bank.program([{"layer": 0, "kind": "neuron", "neuron": 1, "dose": .2}])
        self.assertEqual(program["phase"], "all")
        self.assertEqual(program["sites"][0]["site"], "layers.0.mlp_input")
        np.testing.assert_allclose(program["sites"][0]["vector"], [0, .4, 0, 0, 0], atol=1e-6)
        self.assertAlmostEqual(details["aggregate_relative_norm"], .2)
        self.assertEqual(details["model_identity"], self.model.identity)
        self.assertEqual(details["program_sha256"], canonical_digest(program))

    def test_trait_residual_site_and_negative_polarity(self) -> None:
        program, _ = self.bank.program([{"layer": 0, "kind": "trait", "trait": "openness", "dose": -.4}], "decode")
        self.assertEqual(program["phase"], "decode")
        self.assertEqual(program["sites"][0]["site"], "layers.0.residual")
        np.testing.assert_allclose(program["sites"][0]["vector"], [0, 0, -4, 0])

    def test_empty_zero_and_cancelled_programs_are_noops(self) -> None:
        empty, _ = self.bank.program([])
        self.assertEqual(empty["sites"], [])
        zero, _ = self.bank.program([{"layer": 0, "neuron": 1, "dose": 0}])
        self.assertEqual(zero["sites"], [])
        cancelled, details = self.bank.program([{"layer": 0, "neuron": 1, "dose": .3},
                                                 {"layer": 0, "neuron": 1, "dose": -.3}])
        self.assertEqual(cancelled["sites"], [])
        self.assertEqual(details["aggregate_relative_norm"], 0)

    def test_group_limit_uses_actual_sum_not_sum_of_doses(self) -> None:
        pushes = [{"layer": 0, "kind": "column", "neuron": 1, "dose": .3}, {"layer": 0, "kind": "column", "neuron": 3, "dose": .4}]
        program, details = self.bank.program(pushes, aggregate_limit=.25)
        self.assertAlmostEqual(details["aggregate_relative_norm"], .5)
        self.assertAlmostEqual(details["aggregate_scale"], .5)
        self.assertAlmostEqual(np.linalg.norm(program["sites"][0]["vector"]), 2.5)
        aligned, aligned_details = self.bank.program([pushes[0], pushes[0]], aggregate_limit=.25)
        self.assertAlmostEqual(aligned_details["aggregate_relative_norm"], .6)
        self.assertAlmostEqual(np.linalg.norm(aligned["sites"][0]["vector"]), 2.5, places=6)

    def test_neuron_and_output_column_use_distinct_sites_with_same_linear_equivalent(self) -> None:
        push = {"layer": 0, "neuron": 3, "dose": -.4}
        neuron, _ = self.bank.program([push])
        column, _ = self.bank.program([{**push, "kind": "column"}])
        np.testing.assert_allclose(neuron["sites"][0]["vector"], [0, 0, 0, -2, 0])
        np.testing.assert_allclose(column["sites"][0]["vector"], [0, 0, 0, -4])
        self.assertEqual(column["sites"][0]["site"], "layers.0.mlp_out")
        pair, details = self.bank.program([push, {**push, "kind": "column"}])
        self.assertEqual(len(pair["sites"]), 2)
        self.assertAlmostEqual(details["aggregate_relative_norm"], np.sqrt(.32))

    def test_random_controls_are_reproducible_and_equal_norm(self) -> None:
        push = {"layer": 0, "kind": "random", "seed": 41, "dose": -.2}
        first, _ = self.bank.program([push])
        second, _ = self.bank.program([push])
        other, _ = self.bank.program([{**push, "seed": 42}])
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertAlmostEqual(np.linalg.norm(first["sites"][0]["vector"]), 2.0, places=6)

    def test_bank_cannot_cross_model_revision_or_quantization(self) -> None:
        for model in (replace(self.model, model="other/model"),
                      replace(self.model, revision="b" * 40),
                      replace(self.model, quantization="test-int8"),
                      replace(self.model, hidden=5)):
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                VectorBank(self.root, model).layer(0)
        with self.assertRaises(FrozenInstanceError):
            self.model.revision = "c" * 40  # type: ignore[misc]

    def test_invalid_and_missing_directions_fail_closed(self) -> None:
        for push in ({"layer": 0, "neuron": 2, "dose": 1},
                     {"layer": 0, "neuron": 5, "dose": 1},
                     {"layer": 0, "neuron": True, "dose": 1},
                     {"layer": True, "neuron": 1, "dose": 1},
                     {"layer": 1, "neuron": 1, "dose": 1},
                     {"layer": 0, "kind": "trait", "trait": "missing", "dose": 1},
                     {"layer": 0, "kind": "random", "seed": -1, "dose": 1}):
            with self.subTest(push=push), self.assertRaises(ValueError):
                self.bank.program([push])
        for bad in (np.zeros(4), np.array([1, 2, 3]), np.array([1, float("nan"), 2, 3])):
            self.save_layer(trait_openness=bad)
            with self.assertRaises(ValueError):
                self.bank.program([{"layer": 0, "kind": "trait", "trait": "openness", "dose": .1}])

    def test_nonfinite_dose_median_and_overflowed_norm_fail_closed(self) -> None:
        for value in (float("nan"), float("inf"), -float("inf"), 4.01):
            with self.assertRaises(ValueError):
                self.bank.program([{"layer": 0, "neuron": 1, "dose": value}])
        for median in (0., -1., float("nan"), float("inf")):
            self.save_layer(residual_median=median)
            with self.assertRaises(ValueError):
                self.bank.layer(0)
        self.save_layer(trait_openness=np.full(4, 1e308))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaises(ValueError):
                self.bank.program([{"layer": 0, "kind": "trait", "trait": "openness", "dose": .1}])

    def test_phase_and_aggregate_limit_validation(self) -> None:
        with self.assertRaises(ValueError):
            self.bank.program([], phase="guess")
        for limit in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.bank.program([], aggregate_limit=limit)


class PersonaAuditTests(unittest.TestCase):
    def data(self) -> tuple[dict, list, list, list]:
        jobs, rows = [], []
        for item in range(1, 51):
            job = {"condition": "baseline", "system": "", "seed": 7, "kind": "ipip",
                   "item_id": item, "split": "inventory", "prompt": f"Item {item}",
                   "max_tokens": 2048, "thinking": False}
            job["id"] = digest(job)
            jobs.append(job)
            request = {"seed": 7, "model": "test/model", "max_tokens": 2048,
                       "messages": [{"role": "user", "content": job["prompt"]}],
                       "chat_template_kwargs": {"enable_thinking": False},
                       "temperature": .7, "top_p": .95, "top_k": 20}
            response = {"model": "test/model", "choices": [{"message": {"content": "3"}, "finish_reason": "stop"}]}
            rows.append({"job": job, "request": request, "response": response,
                         "answer": "3", "reasoning": "", "complete": True, "error": None})
        plan = {"model": {"model": "test/model"}, "jobs": jobs}
        manifests, outcomes = [], []
        for shard in range(6):
            ids = [j["id"] for j in jobs if int(j["id"], 16) % 6 == shard]
            manifests.append({"shard": shard, "shards": 6, "plan_sha256": digest(plan),
                              "model": plan["model"], "job_ids": ids, "package_sha256": "a" * 64,
                              "executable_sha256": "b" * 64, "started_at": 1})
            outcomes.append({"rows": len(ids), "expected_rows": len(ids), "errors": [], "finished_at": 2})
        return plan, rows, manifests, outcomes

    def test_strict_ipip_and_explicit_capability_answers(self) -> None:
        self.assertEqual(ipip_digit(" 4\n"), 4)
        for text in ("I choose 4", "4.0", "**4**", "6", ""):
            self.assertIsNone(ipip_digit(text))
        self.assertTrue(capability_answer("Working...\nANSWER: **7/22**", "7/22")["correct"])
        self.assertTrue(capability_answer("ANSWER: 60.0 km/h.", "60")["correct"])
        for text in ("The answer might be 60", "ANSWER: 160", "ANSWER: 60 or 70"):
            self.assertFalse(capability_answer(text, "60")["correct"])
        self.assertFalse(capability_answer("ANSWER: 14/44", "7/22")["correct"])

    def test_complete_inventory_is_scored_but_missing_and_duplicate_items_are_not(self) -> None:
        plan, rows, manifests, outcomes = self.data()
        good = analyze(plan, rows, manifests=manifests, outcomes=outcomes)
        self.assertEqual(good["status"], "complete_audited")
        self.assertTrue(good["ipip_inventories"][0]["complete"])
        self.assertEqual(good["profiles"]["baseline"]["ipip"]["Openness"]["mean"], 30)
        missing = analyze(plan, rows[:-1], manifests=manifests, outcomes=outcomes)
        self.assertEqual(missing["missing_rows"], 1)
        self.assertFalse(missing["ipip_inventories"][0]["complete"])
        duplicate = analyze(plan, rows + rows[:1], manifests=manifests, outcomes=outcomes)
        self.assertEqual(duplicate["counts"]["duplicate_rows"], 1)
        self.assertFalse(duplicate["ipip_inventories"][0]["complete"])

    def test_thinking_disabled_violation_and_wrong_model_are_not_scored(self) -> None:
        plan, rows, manifests, outcomes = self.data()
        rows[0]["response"]["choices"][0]["message"]["reasoning_content"] = "Thinking anyway"
        rows[0]["reasoning"] = "Thinking anyway"
        rows[1]["request"]["model"] = "other/model"
        report = analyze(plan, rows, manifests=manifests, outcomes=outcomes)
        self.assertEqual(report["counts"]["thinking_output"], 1)
        self.assertEqual(report["counts"]["rows_with_identity_or_protocol_issues"], 2)
        self.assertFalse(report["ipip_inventories"][0]["complete"])

    def test_new_attempt_supersedes_stale_failure_without_claiming_completion(self) -> None:
        original = {"shard": 0, "started_at": 1., "lease": "first"}
        previous = {"finished_at": 2., "errors": ["startup timeout"]}
        start = {"event": "start", "shard": 0, "started_at": 3., "lease": "second"}
        manifest, outcome, state = latest_attempt(original, previous, [start])
        self.assertEqual(manifest["lease"], "second")
        self.assertIsNone(outcome)
        self.assertEqual(state["state"], "running")
        self.assertTrue(state["prior_outcome_ignored"])
        finished = {"event": "finish", "finished_at": 4., "errors": []}
        _, outcome, state = latest_attempt(original, previous, [start, finished])
        self.assertEqual(state["state"], "finished")
        self.assertEqual(outcome["finished_at"], 4.)


if __name__ == "__main__":
    unittest.main()
