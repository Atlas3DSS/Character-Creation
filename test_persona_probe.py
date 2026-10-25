"""CPU-only request, finite site, capacity and native completion invariants."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from persona_probe import (ProtocolError, calibration_reference, compare_calibration_arrays,
                           normalize_request, validate_capacity, validate_completion, validate_program)
from persona_vectors import ModelSpec, canonical_digest


class ProbeProtocolTests(unittest.TestCase):
    def request(self):
        return {"model": "checkpoint", "messages": [{"role": "user", "content": "A test task."}], "seed": 7035}

    def test_defaults_preserve_budget_and_disable_thinking(self):
        spec = normalize_request(self.request(), "checkpoint")
        self.assertEqual(spec["max_tokens"], 2048)
        self.assertEqual(spec["seed"], 7035)
        self.assertFalse(spec["thinking"])
        self.assertEqual(spec["sampling"]["temperature"], .7)

    def test_greedy_is_explicit_top_one_native_sampling(self):
        request = {**self.request(), "temperature": 0, "top_k": 20, "top_p": .95}
        spec = normalize_request(request, "checkpoint")
        self.assertEqual(spec["requested_sampling"]["temperature"], 0)
        self.assertEqual(spec["sampling"]["temperature"], 1.)
        self.assertEqual(spec["sampling"]["top_k"], 1)
        self.assertEqual(spec["sampling"]["top_p"], 1.)

    def test_unsupported_semantics_and_nonfinite_values_rejected(self):
        for updates in [{"stream": True}, {"n": 2}, {"model": "wrong"}, {"stop": ["x"]},
                        {"temperature": float("nan")}, {"temperature": True}, {"seed": -1},
                        {"max_tokens": 0}, {"top_p": 0}, {"metadata": {"job_id": 2}},
                        {"chat_template_kwargs": {"enable_thinking": True}}]:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                normalize_request({**self.request(), **updates}, "checkpoint")

    def test_program_uses_exact_catalog_width_and_keeps_zero_sites(self):
        catalog = [{"name": "layers.2.residual", "width": 3}, {"name": "layers.2.mlp_input", "width": 7}]
        program = {"phase": "decode", "sites": [{"site": "layers.2.mlp_input", "vector": [0.] * 7}]}
        normalized = validate_program(program, catalog, 64)
        self.assertEqual(normalized["phase"], "decode")
        self.assertEqual(len(normalized["sites"]), 1)
        self.assertEqual(normalized["sites"][0]["vector"].dtype.name, "float32")
        for change in (lambda p: p["sites"][0].update(vector=[0.] * 3),
                       lambda p: p["sites"][0]["vector"].__setitem__(0, float("inf")),
                       lambda p: p["sites"].append(copy.deepcopy(p["sites"][0])),
                       lambda p: p.update(phase="thinking")):
            altered = copy.deepcopy(program)
            change(altered)
            with self.assertRaises(ValueError):
                validate_program(altered, catalog, 64)
        with self.assertRaises(ValueError):
            validate_program(program, catalog, 0)

    def test_budget_is_bound_to_physical_pages_not_virtual_context(self):
        options = dict(physical_pages=86, page_tokens=128, logical_context=32768, table_pages=256)
        self.assertEqual(validate_capacity(8960, 2048, **options), 86)
        with self.assertRaises(ValueError):
            validate_capacity(8961, 2048, **options)
        with self.assertRaises(ValueError):
            validate_capacity(7000, 2048, **{**options, "logical_context": 8192})

    def test_init_and_unsampled_prefill_require_exact_zero_generated_rng(self):
        options = dict(epoch=3, sequence=11, consumed=0, generated=0, sampled=False, budget=2048, eos=(9,), vocabulary=20)
        result = validate_completion([3, 11, 0, 0, -1, 1, 0, 0], [3, 0, 0, -1, 1, 0, 0], **options)
        self.assertFalse(result["finished"])
        options.update(sequence=12, consumed=128)
        validate_completion([3, 12, 128, 0, -1, 1, 0, 0], [3, 128, 0, -1, 1, 0, 0], **options)

    def test_real_sampler_progress_and_eos_vs_budget_finish(self):
        options = dict(epoch=3, sequence=13, consumed=200, generated=1, sampled=True, budget=2, eos=(9,), vocabulary=20)
        result = validate_completion([3, 13, 200, 1, 5, 1, 0], [3, 200, 1, 5, 1, 0, 1], **options)
        self.assertFalse(result["finished"])
        options.update(sequence=14, consumed=201, generated=2)
        for token, reason in [(9, "stop"), (6, "length")]:
            result = validate_completion([3, 14, 201, 2, token, 2, 0], [3, 201, 2, token, 2, 0, 2], **options)
            self.assertTrue(result["finished"])
            self.assertEqual(result["finish_reason"], reason)

    def test_pending_rng_epoch_and_consumption_mismatches_fail(self):
        record, state = [1, 2, 10, 1, 5, 1, 0], [1, 10, 1, 5, 1, 0, 1]
        options = dict(epoch=1, sequence=2, consumed=10, generated=1, sampled=True, budget=2048, eos=(9,), vocabulary=20)
        for index in range(7):
            changed = state.copy()
            changed[index] += 1
            with self.subTest(state_field=index), self.assertRaises(ProtocolError):
                validate_completion(record, changed, **options)
        for index in (0, 1, 2, 3, 5, 6):
            changed = record.copy()
            changed[index] += 1
            with self.subTest(completion_field=index), self.assertRaises(ProtocolError):
                validate_completion(changed, state, **options)


class CalibrationBindingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "capture-shard0.npz"
        self.model = ModelSpec("fixture", "a" * 40, "BF16", 2, 3, 4)
        self.jobs = []
        for condition in ("openness_high", "baseline"):
            job = dict(condition=condition, system="" if condition == "baseline" else "Be open.",
                       prompt="A calibration task.", prompt_id="fixture", split="discovery", thinking=False)
            self.jobs.append({**job, "id": canonical_digest(job)})
        self.arrays = dict(model_identity=self.model.identity, plan_sha256="b" * 64,
            job_ids=np.array([job["id"] for job in self.jobs]), token_ids=np.array([1, 2, 3, 4, 5], dtype=np.int64),
            token_offsets=np.array([0, 3, 5], dtype=np.int64), norm_offsets=np.array([0, 2, 3], dtype=np.int64),
            residual_l2=np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32),
            mlp_input_last=np.arange(16, dtype=np.float32).reshape(2, 2, 4),
            residual_last=np.arange(12, dtype=np.float32).reshape(2, 2, 3),
            pre_mlp_last=np.arange(12, dtype=np.float32).reshape(2, 2, 3) + 1)
        self.report = dict(status="complete", state_restored=True, model_identity=self.model.identity,
            model=self.model.__dict__, jobs=self.jobs, prompts_completed=2, plan_sha256="b" * 64,
            checkpoint_config_sha256="c" * 64, checkpoint_index_sha256="d" * 64,
            tokenizer_sha256="e" * 64, preparation={"mtp": True, "prefill_chunk_size": 128})
        self.publish()

    def publish(self):
        np.savez(self.path, **self.arrays)
        self.report["artifact_sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.path.with_suffix(".json").write_text(json.dumps(self.report))

    def test_baseline_original_tokens_and_all_non_sink_norms_are_bound(self):
        reference = calibration_reference(self.path, self.model)
        self.assertEqual(reference["job"]["condition"], "baseline")
        expected = reference["expected"]
        np.testing.assert_array_equal(expected["prompt_token_ids"], [4, 5])
        np.testing.assert_array_equal(expected["prefill_positions"], [0, 1])
        np.testing.assert_array_equal(expected["residual_l2"], [[3], [6]])
        self.assertEqual(expected["residual_l2_mean"].dtype, np.float64)
        self.assertTrue(compare_calibration_arrays(expected, expected)["passed"])
        changed = {key: np.array(value, copy=True) for key, value in expected.items()}
        changed["residual_l2"][1, 0] += 1
        result = compare_calibration_arrays(changed, expected)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["residual_l2"]["bit_exact"])
        self.assertTrue(result["checks"]["mlp_input_last"]["bit_exact"])

    def test_equal_numeric_zero_requires_equal_bits_and_dtype(self):
        expected = {"plane": np.zeros((2, 3), dtype=np.float32), "count": np.int64(3)}
        changed = {"plane": expected["plane"].copy(), "count": expected["count"]}
        changed["plane"][0, 0] = -0.
        self.assertFalse(compare_calibration_arrays(changed, expected)["passed"])
        changed["plane"] = np.zeros((2, 3), dtype=np.float64)
        self.assertFalse(compare_calibration_arrays(changed, expected)["passed"])

    def test_incomplete_changed_model_or_changed_prompt_manifest_rejected(self):
        for key, value in (("status", "running"), ("state_restored", False),
                           ("model_identity", "f" * 64), ("prompts_completed", 1),
                           ("artifact_sha256", "f" * 64)):
            report = copy.deepcopy(self.report)
            report[key] = value
            self.path.with_suffix(".json").write_text(json.dumps(report))
            with self.subTest(key=key), self.assertRaises(ValueError):
                calibration_reference(self.path, self.model)
        report = copy.deepcopy(self.report)
        report["jobs"][1]["prompt"] = "Changed after capture."
        self.path.with_suffix(".json").write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            calibration_reference(self.path, self.model)

    def test_array_partition_and_nonfinite_observations_rejected(self):
        for key, replacement in (("token_offsets", np.array([0, 3, 6], dtype=np.int64)),
                                 ("norm_offsets", np.array([0, 2, 2], dtype=np.int64)),
                                 ("mlp_input_last", np.full((2, 2, 4), np.nan, dtype=np.float32))):
            original = self.arrays[key]
            self.arrays[key] = replacement
            self.publish()
            with self.subTest(key=key), self.assertRaises(ValueError):
                calibration_reference(self.path, self.model)
            self.arrays[key] = original


if __name__ == "__main__":
    unittest.main()
