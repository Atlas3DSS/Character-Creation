"""Independent scalar decoding references and tiny, model-bound capture fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from persona_capture import capture_plan, shard_jobs
from persona_map import Checkpoint, TRAITS, aggregate, assemble, build_layers, decode_columns, e4m3
from persona_vectors import ModelSpec, VectorBank, canonical_digest


ROOT = Path(__file__).resolve().parent


def write_safetensors(path: Path, tensors: dict) -> None:
    header, payload = {}, bytearray()
    for key, (dtype, array) in tensors.items():
        data = np.asarray(array).tobytes()
        header[key] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class PersonaDecodeTests(unittest.TestCase):
    def test_unmapped_neuron_reads_bound_checkpoint_and_rejects_wrong_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = ModelSpec("test/model", "a" * 40, "NVFP4", 1, 2, 16)
            prefix = "model.language_model.layers.0.mlp.down_proj"
            tensors = {
                prefix + ".weight": ("U8", np.full((2, 8), 0x22, dtype=np.uint8)),
                prefix + ".weight_scale": ("F8_E4M3", np.array([[56], [64]], dtype=np.uint8)),
                prefix + ".weight_scale_2": ("F32", np.array([.5], dtype=np.float32)),
            }
            write_safetensors(root / "weights.safetensors", tensors)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "weights.safetensors" for k in tensors}}))
            data = {"model_identity": model.identity, "residual_median": 10.,
                "neuron_ids": np.array([0]), "down_columns": np.array([[.5], [1.]]),
                "checkpoint_config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
                "checkpoint_index_sha256": hashlib.sha256((root / "model.safetensors.index.json").read_bytes()).hexdigest()}
            np.savez(root / "layer_00.npz", **data)
            bank = VectorBank(root, model, checkpoint=root)
            np.testing.assert_array_equal(bank.column(bank.layer(0), 0, 15), [.5, 1.])
            program, _ = bank.program([{"layer": 0, "neuron": 15, "dose": .25}])
            vector = program["sites"][0]["vector"]
            self.assertEqual(np.count_nonzero(vector), 1)
            self.assertAlmostEqual(vector[15], 2.5 / np.sqrt(1.25), places=6)
            data["checkpoint_config_sha256"] = "b" * 64
            np.savez(root / "layer_00.npz", **data)
            with self.assertRaisesRegex(ValueError, "provenance"):
                bank.column(bank.layer(0), 0, 15)

    def test_all_finite_e4m3_codes_against_scalar_binary_definition(self) -> None:
        codes = np.array([x for x in range(256) if x & 127 != 127], dtype=np.uint8)
        reference = []
        for raw in codes.tolist():
            exponent, mantissa = (raw & 127) // 8, raw % 8
            magnitude = mantissa / 512 if exponent == 0 else (8 + mantissa) * 2.0 ** (exponent - 10)
            reference.append(-magnitude if raw >= 128 else magnitude)
        np.testing.assert_array_equal(e4m3(codes), np.array(reference, dtype=np.float32))
        self.assertEqual(float(e4m3(np.array([126], dtype=np.uint8))[0]), 448.)
        self.assertEqual(float(e4m3(np.array([1], dtype=np.uint8))[0]), 1 / 512)
        self.assertTrue(np.signbit(e4m3(np.array([128], dtype=np.uint8))[0]))
        for code in (127, 255):
            with self.assertRaises(ValueError):
                e4m3(np.array([code], dtype=np.uint8))
        with self.assertRaises(ValueError):
            e4m3(np.array([256], dtype=np.int32))

    def test_nvfp4_nibble_order_sign_scale_group_boundary_and_gain(self) -> None:
        codes = np.tile(np.arange(16, dtype=np.uint8), 2)
        packed = (codes[::2] | (codes[1::2] << 4))[None, :]
        # E4M3 raw56=1, raw64=2. The second group begins at neuron16.
        scales = np.array([[56, 64]], dtype=np.uint8)
        ids = np.array([31, 0, 1, 7, 8, 15, 16, 23, 24, 31])
        values = [0, .5, 1, 1.5, 2, 3, 4, 6]
        expected = []
        for neuron in ids:
            code = int(codes[neuron])
            expected.append(values[code & 7] * (-1 if code & 8 else 1) * (1 if neuron < 16 else 2) * .25)
        result = decode_columns(packed, scales, .25, ids)
        self.assertEqual(result.shape, (1, len(ids)))
        np.testing.assert_array_equal(result[0], expected)
        with self.assertRaises(ValueError):
            decode_columns(packed, scales, 1, np.array([32]))
        with self.assertRaises(ValueError):
            decode_columns(packed, scales, 1, np.array([.5]))
        with self.assertRaises(ValueError):
            decode_columns(packed[:, :9], scales[:, :1], 1, np.array([0]))
        with self.assertRaises(ValueError):
            decode_columns(packed, scales | 128, 1, ids)
        with self.assertRaises(ValueError):
            decode_columns(packed, scales, 1e308, ids)

    def test_hf_blob_symlinks_are_supported_but_index_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshots" / ("a" * 40)
            snapshot.mkdir(parents=True)
            blobs = root / "blobs"
            blobs.mkdir()
            blob = blobs / ("b" * 64)
            write_safetensors(blob, {"test": ("U8", np.array([[1, 2]], dtype=np.uint8))})
            (snapshot / "weights.safetensors").symlink_to(blob)
            index = snapshot / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"test": "weights.safetensors"}}))
            np.testing.assert_array_equal(Checkpoint(snapshot).raw("test", "U8"), [[1, 2]])
            index.write_text(json.dumps({"weight_map": {"test": "../../blobs/" + blob.name}}))
            with self.assertRaises(ValueError):
                Checkpoint(snapshot).raw("test", "U8")


class PersonaMapArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "data").mkdir()
        corpus = json.loads((ROOT / "data/persona_prompts.json").read_text())
        corpus["prompts"] = [p for p in corpus["prompts"] if p["split"] == "discovery"][:2]
        (self.root / "data/persona_prompts.json").write_text(json.dumps(corpus))
        self.model = {"model": "test/qwen", "revision": "a" * 40, "quantization": "test-nvfp4",
                      "layers": 2, "hidden": 3, "intermediate": 32}
        (self.root / "data/persona_model.json").write_text(json.dumps(self.model))
        self.plan = capture_plan(self.root)
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_text(json.dumps(self.plan))
        self.prompt_ids = sorted({j["prompt_id"] for j in self.plan["jobs"]})
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        (self.checkpoint / "config.json").write_text(json.dumps(self.model))
        tensors = {}
        for layer in range(2):
            base = f"model.language_model.layers.{layer}.mlp.down_proj"
            code = np.tile((np.arange(32, dtype=np.uint8) % 7 + 1), (3, 1))
            code[1, 1::2] |= 8
            code[2, ::2] |= 8
            tensors[base + ".weight"] = ("U8", code[:, ::2] | (code[:, 1::2] << 4))
            tensors[base + ".weight_scale"] = ("F8_E4M3", np.full((3, 2), 56, dtype=np.uint8))
            tensors[base + ".weight_scale_2"] = ("F32", np.array(1., dtype="<f4"))
        write_safetensors(self.checkpoint / "weights.safetensors", tensors)
        (self.checkpoint / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "weights.safetensors" for k in tensors}}))
        self.write_captures()

    def file_hash(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def vector(self, trait: str) -> np.ndarray:
        value = TRAITS.index(trait) + 1
        return np.array([[value, value * 2, -value], [value * 2, value, value * 3]], dtype=np.float32)

    def write_captures(self) -> None:
        for shard in range(6):
            jobs = shard_jobs(self.plan, shard)
            residuals, norms, ids = [], [], []
            offsets, token_offsets = [0], [0]
            for job in jobs:
                prompt_index = self.prompt_ids.index(job["prompt_id"])
                count = 3 if prompt_index == 0 else 5
                nuisance = 100 * prompt_index
                value = np.full((2, 3), nuisance, dtype=np.float32)
                if job["condition"] != "baseline":
                    trait, pole = job["condition"].rsplit("_", 1)
                    value += self.vector(trait) * (1 if pole == "high" else -1)
                residuals.append(value)
                # Trait rows have huge norms; they must never affect baseline calibration.
                norm = (1. if prompt_index == 0 else 3.) if job["condition"] == "baseline" else 999.
                norms.append(np.full((2, count - 1), norm, dtype=np.float32))
                offsets.append(offsets[-1] + count - 1)
                ids.extend(range(1, count + 1))
                token_offsets.append(len(ids))
            path = self.root / f"capture-shard{shard}.npz"
            np.savez(path, model_identity=self.plan["model_identity"], plan_sha256=canonical_digest(self.plan),
                     job_ids=np.array([j["id"] for j in jobs]), residual_last=np.stack(residuals),
                     pre_mlp_last=np.zeros((len(jobs), 2, 3), dtype=np.float32), residual_l2=np.concatenate(norms, axis=1),
                     mlp_input_last=np.broadcast_to(np.stack(residuals)[:, :, :1], (len(jobs), 2, 32)).copy(),
                     norm_offsets=np.array(offsets), token_ids=np.array(ids), token_offsets=np.array(token_offsets))
            report = {"status": "complete", "state_restored": True, "model_identity": self.plan["model_identity"],
                      "model": self.model, "plan_sha256": canonical_digest(self.plan), "shard": shard, "shards": 6,
                      "jobs": jobs, "prompts_completed": len(jobs), "artifact_sha256": self.file_hash(path),
                      "checkpoint_config_sha256": self.file_hash(self.checkpoint / "config.json"),
                      "checkpoint_index_sha256": self.file_hash(self.checkpoint / "model.safetensors.index.json"),
                      "script_sha256": "b" * 64, "heliopause_sha256": "c" * 64, "tokenizer_sha256": "d" * 64,
                      "torch": "test", "source_sha256": {"capture.py": "e" * 64},
                      "preparation": {"capture_activations": True, "context": 4096, "compile_workers": shard + 1}}
            path.with_suffix(".json").write_text(json.dumps(report))

    def alter_array(self, name: str, transform) -> None:
        path = self.root / "capture-shard0.npz"
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        data[name] = transform(data[name])
        np.savez(path, **data)
        report = json.loads(path.with_suffix(".json").read_text())
        report["artifact_sha256"] = self.file_hash(path)
        path.with_suffix(".json").write_text(json.dumps(report))

    def test_paired_differences_and_baseline_only_token_weighted_medians(self) -> None:
        fit = aggregate(self.plan, self.root)
        np.testing.assert_array_equal(fit["medians"], [3, 3])
        for trait in TRAITS:
            np.testing.assert_allclose(fit["directions"][trait], 2 * self.vector(trait))
            np.testing.assert_allclose(fit["neuron_contrasts"][trait],
                np.broadcast_to(2 * self.vector(trait)[:, :1], (2, 32)))
            self.assertEqual(fit["pair_counts"][trait], 2)
        np.testing.assert_allclose(fit["neuron_baseline_mean"], 50.)
        np.testing.assert_allclose(fit["neuron_baseline_sd"], np.sqrt(5000.), rtol=1e-6)
        self.assertEqual(len(fit["source_hashes"]), 6)

    def test_artifact_hash_and_runtime_model_mismatches_are_rejected(self) -> None:
        path = self.root / "capture-shard0.json"
        original = json.loads(path.read_text())
        for key, value in (("artifact_sha256", "0" * 64), ("model_identity", "0" * 64),
                           ("tokenizer_sha256", "0" * 64), ("script_sha256", "0" * 64)):
            path.write_text(json.dumps({**original, key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                aggregate(self.plan, self.root)
        path.write_text(json.dumps(original))

    def test_nonfinite_or_misaligned_capture_arrays_are_rejected(self) -> None:
        for key, transform in (("pre_mlp_last", lambda x: x * np.nan),
                               ("norm_offsets", lambda x: x.astype(float)),
                               ("token_offsets", lambda x: x + 1),
                               ("residual_last", lambda x: x[:, :, :-1])):
            self.write_captures()
            self.alter_array(key, transform)
            with self.subTest(key=key), self.assertRaises(ValueError):
                aggregate(self.plan, self.root)

    def test_heldout_or_modified_job_content_cannot_be_fitted(self) -> None:
        changed = json.loads(json.dumps(self.plan))
        changed["jobs"][0]["split"] = "heldout"
        with self.assertRaises(ValueError):
            aggregate(changed, self.root)
        changed["jobs"][0]["split"] = "discovery"
        changed["jobs"][0]["prompt"] += "modified"
        with self.assertRaises(ValueError):
            aggregate(changed, self.root)

    def build(self) -> None:
        for shard in range(6):
            build_layers(argparse.Namespace(plan=self.plan_path, results=self.root, checkpoint=self.checkpoint,
                                            shard=shard, per_pole=1))

    def test_end_to_end_banks_assemble_with_candidate_only_claims(self) -> None:
        self.build()
        assemble(self.root)
        report = json.loads((self.root / "persona_map.json").read_text())
        self.assertEqual(report["model_identity"], self.plan["model_identity"])
        self.assertEqual({n["layer"] for n in report["neurons"]}, {0, 1})
        self.assertTrue(all(n["status"] == "candidate" and n["causal_effect"] is None for n in report["neurons"]))

    def test_assembly_rejects_missing_layers_and_modified_vector_banks(self) -> None:
        self.build()
        path = self.root / "map-shard0.json"
        original = path.read_text()
        report = json.loads(original)
        report["banks"] = []
        path.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            assemble(self.root)
        path.write_text(original)
        bank = self.root / "layer_00.npz"
        with bank.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaises(ValueError):
            assemble(self.root)

    def test_assembly_verifies_displayed_activation_values_and_scope(self) -> None:
        self.build()
        path = self.root / "map-shard0.json"
        original = json.loads(path.read_text())
        changes = [lambda o: o.update(baseline_mean=o["baseline_mean"] + 1),
                   lambda o: o.update(baseline_sd=-1),
                   lambda o: o.update(baseline_sd=o["baseline_sd"] + 1),
                   lambda o: o.update(sample="heldout prompts"),
                   lambda o: o.update(phase="all generated tokens"),
                   lambda o: o["high_minus_low"].update(openness=999.)]
        for change in changes:
            changed = json.loads(json.dumps(original))
            change(changed["neurons"][0]["activation_observation"])
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                assemble(self.root)
        path.write_text(json.dumps(original))
        assemble(self.root)
        report = json.loads((self.root / "persona_map.json").read_text())
        self.assertEqual(len(report["banks"]), self.model["layers"])


if __name__ == "__main__":
    unittest.main()
