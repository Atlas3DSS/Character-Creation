"""CPU-only discovery-plan and paired-profile invariants; no CUDA imports/runs."""

import persona  # Make the flat, deployment-compatible source modules importable.
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from persona_capture import capture_plan, shard_jobs, tokenize_messages
from persona_vectors import ModelSpec, canonical_digest


ROOT = Path(__file__).resolve().parent.parent


class PersonaCapturePlanTests(unittest.TestCase):
    def test_tokenizer_list_and_batchencoding_normalize_before_gpu_preparation(self):
        for encoded in ([11, 24], {"input_ids": [11, 24], "attention_mask": [1, 1]}):
            tokenizer = Mock()
            tokenizer.apply_chat_template.return_value = encoded
            self.assertEqual(tokenize_messages(tokenizer, [{"role": "user", "content": "Test"}]), [11, 24])
            self.assertIs(tokenizer.apply_chat_template.call_args.kwargs["return_dict"], False)
        for encoded in ("test", {"input_ids": ["input_ids"]}, [[11, 24]], [], [True]):
            tokenizer = Mock()
            tokenizer.apply_chat_template.return_value = encoded
            with self.assertRaises(ValueError):
                tokenize_messages(tokenizer, [])

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "data").mkdir()
        self.corpus = json.loads((ROOT / "data/persona_prompts.json").read_text())
        self.model = {"model": "test/fresh-checkpoint", "revision": "a" * 40,
                      "quantization": "test-bf16", "layers": 3, "hidden": 8, "intermediate": 12}
        self.write_inputs()

    def write_inputs(self) -> None:
        (self.root / "data/persona_prompts.json").write_text(json.dumps(self.corpus))
        (self.root / "data/persona_model.json").write_text(json.dumps(self.model))

    def test_exact_discovery_factorial_and_paired_user_prompts(self) -> None:
        plan = capture_plan(self.root)
        self.assertEqual(len(plan["jobs"]), 132)
        prompt_rows = defaultdict(list)
        for job in plan["jobs"]:
            prompt_rows[job["prompt_id"]].append(job)
            self.assertEqual(job["split"], "discovery")
            self.assertIs(job["thinking"], False)
            self.assertEqual(job["id"], canonical_digest({k: v for k, v in job.items() if k != "id"}))
        discovery = {x["id"]: x for x in self.corpus["prompts"] if x["split"] == "discovery"}
        self.assertEqual(set(prompt_rows), set(discovery))
        for prompt_id, jobs in prompt_rows.items():
            self.assertEqual(len(jobs), 11)
            self.assertEqual(len({j["prompt"] for j in jobs}), 1)
            by_condition = {j["condition"]: j for j in jobs}
            self.assertEqual(by_condition["baseline"]["system"], "")
            self.assertEqual(by_condition["baseline"]["prompt"], discovery[prompt_id]["text"])
            for trait, poles in self.corpus["trait_prompts"].items():
                for pole in ("low", "high"):
                    self.assertEqual(by_condition[f"{trait}_{pole}"]["system"], poles[pole])
        self.assertEqual(set(Counter(j["condition"] for j in plan["jobs"]).values()), {12})

    def test_heldout_text_changes_cannot_change_discovery_plan(self) -> None:
        original = capture_plan(self.root)
        for prompt in self.corpus["prompts"]:
            if prompt["split"] == "heldout":
                prompt["text"] = "HELDOUT_SENTINEL_MUST_NOT_APPEAR"
        self.write_inputs()
        changed = capture_plan(self.root)
        self.assertEqual(original, changed)
        self.assertNotIn("HELDOUT_SENTINEL", json.dumps(changed))

    def test_six_shards_are_balanced_disjoint_exhaustive_and_order_independent(self) -> None:
        plan = capture_plan(self.root)
        shards = [shard_jobs(plan, shard) for shard in range(6)]
        self.assertEqual([len(s) for s in shards], [22] * 6)
        all_ids = [j["id"] for shard in shards for j in shard]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), {j["id"] for j in plan["jobs"]})
        reverse_plan = {**plan, "jobs": list(reversed(plan["jobs"]))}
        self.assertEqual(shards, [shard_jobs(reverse_plan, shard) for shard in range(6)])
        for shard in shards:
            self.assertEqual(shard, sorted(shard, key=lambda j: j["id"]))
        # Prompt/profile pairs can be on different shards; join by prompt ID, never row position.
        combined = defaultdict(set)
        for shard in shards:
            for job in shard:
                combined[job["prompt_id"]].add(job["condition"])
        self.assertTrue(all(len(profiles) == 11 for profiles in combined.values()))

    def test_checkpoint_identity_is_bound_by_complete_plan(self) -> None:
        plan = capture_plan(self.root)
        expected = ModelSpec(**self.model)
        self.assertEqual(plan["model_identity"], expected.identity)
        self.assertEqual(plan["model"], self.model)
        self.model["revision"] = "b" * 40
        self.write_inputs()
        changed = capture_plan(self.root)
        self.assertNotEqual(changed["model_identity"], plan["model_identity"])
        self.assertNotEqual(canonical_digest(changed), canonical_digest(plan))
        self.assertEqual(changed["jobs"], plan["jobs"])
        # Job IDs alone do not bind a checkpoint; aggregators must also require plan/model hashes.

    def test_discovery_prompt_changes_invalidate_its_eleven_conditions(self) -> None:
        before = capture_plan(self.root)
        target = next(p for p in self.corpus["prompts"] if p["split"] == "discovery")
        target["text"] += " State one concrete example."
        self.write_inputs()
        after = capture_plan(self.root)
        before_ids = {j["id"] for j in before["jobs"]}
        after_ids = {j["id"] for j in after["jobs"]}
        self.assertEqual(len(before_ids - after_ids), 11)
        self.assertEqual(len(after_ids - before_ids), 11)

    def test_invalid_shard_coordinates_are_rejected(self) -> None:
        plan = capture_plan(self.root)
        for shard, shards in ((-1, 6), (6, 6), (True, 6), (0, 5), (0, 7), (0, True), (0., 6)):
            with self.subTest(shard=shard, shards=shards), self.assertRaises(ValueError):
                shard_jobs(plan, shard, shards)

    def test_duplicate_jobs_are_rejected_before_any_execution(self) -> None:
        plan = capture_plan(self.root)
        plan["jobs"].append(copy.deepcopy(plan["jobs"][0]))
        with self.assertRaises(ValueError):
            shard_jobs(plan, 0)


if __name__ == "__main__":
    unittest.main()
