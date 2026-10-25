"""CPU checks for exhaustive unsteered coverage and manager queue compatibility."""

import persona  # Make the flat, deployment-compatible source modules importable.
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from persona_coverage import SCHEMA, coverage_plan, coverage_status, published_inputs
from persona_manager import Queue, execute_job, file_hash
from persona_vectors import ModelSpec, VectorBank


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = ModelSpec("fixture", "a" * 40, "fp32", 2, 4, 5)
        np.savez(self.root / "layer_00.npz", model_identity=self.model.identity, residual_median=10.)
        self.bank = VectorBank(self.root, self.model)
        self.corpus = {}
        for identity, split, pair in (("original", "discovery", "pair"), ("mutation", "discovery", "pair"),
                                      ("heldout", "heldout", "heldout-pair")):
            self.corpus[identity] = dict(id=identity, split=split, pair_id=pair, corpus="skippy", category="dialogue",
                prompt="PRIVATE TEXT " + identity, system="Registered profile", max_tokens=2048,
                source_sha256="c" * 64, char_start=0, char_end=12, thinking=False)

    def plan(self, **kwargs):
        return coverage_plan(self.corpus, self.bank, "d" * 64, split="all", **kwargs)

    def test_every_row_exactly_once_unsteered_and_reproducible(self):
        original = copy.deepcopy(self.corpus)
        proposal, jobs, summary = self.plan()
        self.assertEqual(proposal["schema"], SCHEMA)
        self.assertEqual({job["prompt"]["id"] for job in jobs}, set(self.corpus))
        self.assertEqual(len(jobs), len(self.corpus))
        self.assertNotIn("PRIVATE TEXT", json.dumps(summary))
        self.assertEqual(self.corpus, original)
        self.assertEqual([job["id"] for job in jobs], [job["id"] for job in self.plan()[1]])
        indexed = {job["prompt"]["id"]: job for job in jobs}
        self.assertEqual(indexed["original"]["seed"], indexed["mutation"]["seed"])
        for job in jobs:
            self.assertEqual(job["prompt"], original[job["prompt"]["id"]])
            self.assertEqual(job["pushes"], [])
            self.assertEqual(job["measurement"], "prefill")
            self.assertEqual(job["kind"], "prefill_observation")
            self.assertEqual(job["source_group"]["split"], job["prompt"]["split"])
            self.assertFalse(job["experiment"]["intervention"])
            self.assertTrue(job["experiment"]["summary_reference_only"])
        self.corpus = dict(reversed(list(self.corpus.items())))
        self.assertEqual(jobs, self.plan()[1])

    def test_split_and_full_corpus_calibration_bindings(self):
        for split, expected in (("discovery", 2), ("heldout", 1)):
            _, jobs, _ = coverage_plan(self.corpus, self.bank, "d" * 64, split=split)
            self.assertEqual(len(jobs), expected)
            self.assertTrue(all(job["prompt"]["split"] == split for job in jobs))
        initial = self.plan()[1][0]["id"]
        self.assertNotEqual(initial, coverage_plan(self.corpus, self.bank, "e" * 64, split="all")[1][0]["id"])
        np.savez(self.root / "layer_00.npz", model_identity=self.model.identity, residual_median=11.)
        self.assertNotEqual(initial, self.plan()[1][0]["id"])
        with self.assertRaises(ValueError):
            self.plan(max_jobs=2)

    def test_atomic_job_cap_and_idempotent_existing_queue_enqueue(self):
        proposal, jobs, summary = self.plan()
        queue = Queue(self.root / "queue.sqlite")
        with self.assertRaises(ValueError):
            queue.enqueue(proposal, jobs, limit=2)
        self.assertEqual(queue.counts(), {})
        with queue.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM proposals").fetchone()[0], 0)
        self.assertEqual(queue.enqueue(proposal, jobs, limit=3), 3)
        self.assertEqual(queue.enqueue(proposal, jobs, limit=3), 0)
        with queue.connect() as db:
            self.assertEqual(db.execute("SELECT id FROM proposals").fetchone()[0], summary["proposal_id"])
        status = coverage_status(queue.path, summary)
        self.assertEqual(status["remaining_rows"], 3)
        self.assertFalse(status["all_complete"])

    def test_heldout_never_enters_existing_adaptive_filters(self):
        proposal, jobs, summary = self.plan()
        queue = Queue(self.root / "queue.sqlite")
        queue.enqueue(proposal, jobs)
        while claimed := queue.claim("test-endpoint"):
            job, number = claimed
            queue.finish(job, number, {"complete": True, "complete_observation": True, "complete_answer": False,
                "response": {"operation": "prefill_observation", "choices": [{"message": {"content": ""}}]}})
        self.assertEqual(sum(row["count"] for row in queue.planning_summary()), 2)
        self.assertEqual({row["prompt_id"] for row in queue.observations()}, {"original", "mutation"})
        status = coverage_status(queue.path, summary)
        self.assertTrue(status["all_complete"])
        self.assertEqual(status["valid_completed_observations"], 3)
        self.assertNotIn("PRIVATE TEXT", json.dumps(status))

    def test_existing_executor_sends_one_sample_and_no_steering(self):
        job = self.plan()[1][0]
        response = {"operation": "prefill_observation", "runtime": "fixture", "choices": [{"message": {"content": ""}}],
            "activation_artifact": {"path": "fixture.npz", "sha256": "e" * 64, "model_identity": self.model.identity,
                                    "stage": "final_prefill"},
            "diagnostics": {"prefill_observation_complete": True, "complete_final_answer": False}}
        with patch("persona_manager.native_capability", return_value="checkpoint"), \
             patch("persona_manager.fetch", return_value=response) as fetch, \
             patch("persona_manager.verify_diagnostic_artifacts"), patch("persona_manager.activation_summary", return_value={}):
            result = execute_job(job, {"url": "http://127.0.0.1:8930", "diagnostic": True}, self.bank, 10)
        payload = fetch.call_args.args[1]
        self.assertTrue(payload["observation_only"])
        self.assertEqual(payload["max_tokens"], 1)
        self.assertNotIn("steering", payload)
        self.assertEqual(payload["metadata"]["split"], job["prompt"]["split"])
        self.assertTrue(result["complete_observation"])
        self.assertFalse(result["complete_answer"])

    def test_published_corpus_requires_complete_exact_hash(self):
        path = self.root / "corpus.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in self.corpus.values()))
        manifest = self.root / "manifest.json"
        model = self.root / "model.json"
        model.write_text(json.dumps(self.model.__dict__))
        config = dict(corpus=str(path), corpus_manifest=str(manifest), model=str(model), bank=str(self.root))
        manifest.write_text(json.dumps({"status": "complete", "output_sha256": file_hash(path)}))
        corpus, bank, digest = published_inputs(config)
        self.assertEqual(corpus, self.corpus)
        for status, sha in (("ingesting", digest), ("complete", "f" * 64)):
            manifest.write_text(json.dumps({"status": status, "output_sha256": sha}))
            with self.assertRaises(ValueError):
                published_inputs(config)


if __name__ == "__main__":
    unittest.main()
