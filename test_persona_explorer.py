"""Reject diagnostic runtime changes even when prepared/model identities match."""
import unittest

from persona_explorer import verify_endpoint


class ExplorerIdentityTests(unittest.TestCase):
    def test_diagnostic_binding_cannot_be_omitted_or_reused_for_changed_code(self):
        endpoint = {"prepared_identity": "p", "model_identity": "m", "runtime_identity": "r"}
        descriptor = {"flare": dict(endpoint)}
        self.assertEqual(verify_endpoint(descriptor, endpoint)["runtime_identity"], "r")
        for change in ({"runtime_identity": "other"}, {"model_identity": "other"}, {"prepared_identity": "other"}):
            with self.assertRaises(ValueError):
                verify_endpoint({"flare": {**descriptor["flare"], **change}}, endpoint)
        with self.assertRaises(ValueError):
            verify_endpoint(descriptor, {k: v for k, v in endpoint.items() if k != "runtime_identity"})

    def test_native_package_remains_usable_without_diagnostic_runtime_extension(self):
        endpoint = {"prepared_identity": "p", "model_identity": "m"}
        self.assertEqual(verify_endpoint({"flare": {"prepared_identity": "p"}}, endpoint), {"prepared_identity": "p"})


if __name__ == "__main__":
    unittest.main()
