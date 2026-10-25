"""CPU checks for the independent whole-model arithmetic/report validators."""

import persona  # Make the flat, deployment-compatible source modules importable.
import copy
import unittest

import numpy as np

from persona_qualify import compare, evaluate_arm, round_bf16, site_layout


class QualifierReferenceTests(unittest.TestCase):
    def test_rounding_ties_even_subnormal_nan_and_signed_zero(self):
        bits = np.array([0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000,
                         0x00008000, 0x00018000, 0x80000000, 0x7F800001,
                         0x7F800000, 0xFF800000], dtype=np.uint32)
        expected = np.array([0x3F800000, 0x3F820000, 0xBF800000, 0xBF820000,
                             0x00000000, 0x00020000, 0x80000000, 0x7FC00000,
                             0x7F800000, 0xFF800000], dtype=np.uint32)
        np.testing.assert_array_equal(round_bf16(bits.view(np.float32)).view(np.uint32), expected)
        with self.assertRaises(TypeError):
            round_bf16(np.array([1.], dtype=np.float64))

    def test_compare_is_bitwise_and_rejects_shape_or_nonfinite(self):
        result = compare(np.array([-0.], np.float32), np.array([0.], np.float32))
        self.assertFalse(result["bit_exact"])
        self.assertEqual(result["changed_bits_elements"], 1)
        self.assertEqual(result["max_abs_error"], 0)
        with self.assertRaises(ValueError):
            compare(np.ones(2, np.float32), np.ones(3, np.float32))
        with self.assertRaises(ValueError):
            compare(np.array([np.nan], np.float32), np.zeros(1, np.float32))

    def test_site_lookup_supports_legacy_and_ragged_without_index_assumptions(self):
        legacy = [{"name": f"layers.{layer}.{site}", "width": 4}
                  for layer in range(3) for site in ("mlp_out", "residual")]
        self.assertEqual(site_layout(legacy, (8, 6, 4), "layers.1.residual"), (3, 12, 4))
        ragged = [{"name": f"layers.{layer}.{site}", "width": width}
                  for layer in range(3) for site, width in (("mlp_out", 4), ("residual", 4), ("mlp_input", 12))]
        self.assertEqual(site_layout(ragged, (8, 60), "layers.1.residual"), (4, 24, 4))
        self.assertEqual(site_layout(ragged, (8, 60), "layers.1.mlp_input"), (5, 28, 12))
        for catalog, shape in ((ragged, (8, 59)), (ragged, (8, 9, 4)), (legacy + legacy, (8, 48))):
            with self.assertRaises(ValueError):
                site_layout(catalog, shape, "layers.1.residual")

    def baseline(self):
        return {"residual": np.ones((4, 5), np.float32),
                "pre_mlp": np.full((4, 5), 2., np.float32),
                "mlp_input": np.full((4, 7), 3., np.float32),
                "logits": np.arange(10, dtype=np.float32)}

    def steered(self, site="residual", sign=1):
        base = self.baseline()
        current = copy.deepcopy(base)
        delta = np.array([0., 0.251, -0.251, 0.13, -0.13], np.float32) * sign
        current["residual"][1] = round_bf16(base["residual"][1] + delta)
        current["residual"][2:] += 0.125
        current["logits"][2] += 0.5
        return base, current, delta

    def test_signed_residual_exact_arithmetic_upstream_and_downstream(self):
        for sign in (1, -1):
            base, current, delta = self.steered(sign=sign)
            result = evaluate_arm(base, current, layer=1, site="residual", delta=delta)
            self.assertTrue(result["passed"])
            self.assertTrue(result["target_vs_round_bf16_baseline_plus_delta"]["bit_exact"])
            current["residual"][1, 1] += np.float32(0.0001)
            self.assertFalse(evaluate_arm(base, current, layer=1, site="residual", delta=delta)["passed"])

    def test_upstream_contamination_and_unobservable_output_fail(self):
        base, current, delta = self.steered()
        current["pre_mlp"][1, 0] += 1
        self.assertFalse(evaluate_arm(base, current, layer=1, site="residual", delta=delta)["passed"])
        base, current, delta = self.steered()
        current["logits"] = base["logits"].copy()
        self.assertFalse(evaluate_arm(base, current, layer=1, site="mlp_out", delta=delta)["passed"])

    def test_zero_arm_checks_diagnostic_product_too(self):
        base = self.baseline()
        self.assertTrue(evaluate_arm(base, base, layer=1, site="off")["passed"])
        current = copy.deepcopy(base)
        current["mlp_input"][3, 5] += 1
        self.assertFalse(evaluate_arm(base, current, layer=1, site="off")["passed"])

    def test_pre_down_exact_application_does_not_invent_causal_effect(self):
        base = self.baseline()
        current = copy.deepcopy(base)
        delta = np.zeros(7, np.float32)
        delta[3] = 1
        current["mlp_input"][1, 3] = 4
        result = evaluate_arm(base, current, layer=1, site="mlp_input", delta=delta)
        self.assertTrue(result["passed"])
        self.assertTrue(result["logits"]["bit_exact"])
        self.assertFalse(result["downstream_change_required"])
        current["mlp_input"][1, 4] += 1
        self.assertFalse(evaluate_arm(base, current, layer=1, site="mlp_input", delta=delta)["passed"])

    def test_zero_coordinate_preserves_negative_zero(self):
        base, current, delta = self.steered()
        base["residual"][1, 0] = current["residual"][1, 0] = -0.
        self.assertTrue(evaluate_arm(base, current, layer=1, site="residual", delta=delta)["passed"])
        current["residual"][1, 0] = 0.
        self.assertFalse(evaluate_arm(base, current, layer=1, site="residual", delta=delta)["passed"])


if __name__ == "__main__":
    unittest.main()
