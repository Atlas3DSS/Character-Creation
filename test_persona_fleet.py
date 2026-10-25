"""Request construction checks; no server or GPU is started."""
import unittest
from unittest.mock import patch

from persona_fleet import generation_settings, run_job


class FleetSamplingTests(unittest.TestCase):
    def test_baseline_defaults_and_explicit_greedy_settings(self):
        self.assertEqual(generation_settings({}), {"temperature": .7, "top_p": .95, "top_k": 20, "max_tokens": 2048})
        self.assertEqual(generation_settings({"generation": {"do_sample": False, "max_new_tokens": 3072}}),
                         {"temperature": 0., "top_p": 1., "top_k": 1, "max_tokens": 3072})

    def test_conflicts_or_unknown_settings_are_not_silently_dropped(self):
        for job in [{"generation": {"temperature": float("nan")}},
                    {"generation": {"temperature": .7, "do_sample": False}},
                    {"generation": {"top_k": True}}, {"generation": {"top_p": 0}},
                    {"max_tokens": 2048, "generation": {"max_new_tokens": 4096}},
                    {"generation": {"unsupported": 3}}]:
            with self.assertRaises(ValueError):
                generation_settings(job)

    def test_request_keeps_explicit_sampling_seed_and_steering(self):
        job = {"prompt": "Test", "seed": 42, "max_tokens": 4096, "thinking": False,
               "generation": {"temperature": 0., "top_p": 1., "top_k": 1},
               "steering": {"phase": "prefill", "sites": [{"site": "layers.1.residual", "vector": [0., 0.]}]}}
        response = {"model": "checkpoint", "choices": [{"finish_reason": "stop", "message": {"content": "done"}}]}
        with patch("persona_fleet.fetch", return_value=response) as fetch:
            result = run_job("http://test.invalid", "checkpoint", job)
        payload = fetch.call_args.args[1]
        self.assertEqual(payload["temperature"], 0.)
        self.assertEqual(payload["top_k"], 1)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertEqual(payload["steering"], job["steering"])
        self.assertEqual(result["request"], payload)


if __name__ == "__main__":
    unittest.main()
