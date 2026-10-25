"""CPU contract checks for the one-package preparation wrapper."""
import argparse
import copy
from pathlib import Path
import tempfile
import unittest

from persona_prepare import GATES, compare_logits, compare_measurements, export_arguments, runtime_environment


class PreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = argparse.Namespace(flare=Path('/flare'), output=Path('/cc/results'),
            checkpoint=Path('/checkpoint'), kernel_cache=Path('/kernels'), signing_dir=Path('/signing'),
            compile_workers=44, lease_gib=30, steering_max_sites=64)
        self.baseline = {'status': 'exported', 'args': {**{gate: True for gate in GATES},
            'prompts': '/prompts.jsonl', 'short_reference': '/short.safetensors',
            'long_reference': '/long.safetensors', 'cutlass_include': '/cutlass',
            'context': 32768, 'concurrency': 4, 'chunk': 512, 'depth': 4, 'pages': 86}}

    def test_full_existing_catalog_and_every_export_gate_are_preserved(self) -> None:
        command = export_arguments(self.args, self.baseline)
        for gate in GATES:
            self.assertIn('--' + gate.replace('_', '-'), command)
        self.assertEqual(command[command.index('--chunk') + 1], '512')
        self.assertEqual(command[command.index('--context') + 1], '32768')
        self.assertEqual(command[command.index('--steering-max-sites') + 1], '64')
        self.assertEqual(command[command.index('--output') + 1], '/cc/results/model.safetensors')
        self.assertNotIn('--capture-activations', command)
        self.assertNotIn('--skip-reference-gates', command)

    def test_incomplete_or_diagnostic_baselines_are_not_export_authority(self) -> None:
        for field, value in [('validate', False), ('measure_accuracy', False),
                             ('skip_reference_gates', True), ('capture_activations', True),
                             ('diagnostic_uncompiled', True), ('concurrency', 1)]:
            baseline = copy.deepcopy(self.baseline)
            baseline['args'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                export_arguments(self.args, baseline)

    def test_uring_environment_is_complete_and_idempotent(self) -> None:
        source = {'PATH': '/usr/bin:/bin', 'LD_LIBRARY_PATH': '/custom'}
        env = runtime_environment(Path('/sysroot'), source)
        self.assertEqual(env['PATH'], '/sysroot/usr/bin:/usr/bin:/bin')
        self.assertEqual(env['PKG_CONFIG_SYSROOT_DIR'], '/sysroot')
        self.assertEqual(env['PKG_CONFIG_PATH'], '/sysroot/usr/lib/x86_64-linux-gnu/pkgconfig')
        self.assertEqual(env, runtime_environment(Path('/sysroot'), env))
        self.assertEqual(source, {'PATH': '/usr/bin:/bin', 'LD_LIBRARY_PATH': '/custom'})

    def test_coding_alignment_includes_exact_tokens_and_acceptance_statistics(self) -> None:
        report = {'status': 'complete', 'prompt_file_sha256': 'a', 'summary': [{'tau': 3.5}],
            'runs': [{'depth': 4, 'batch': 1, 'repetition': 0, 'prompt_ids': ['code-00'],
                      'output_token_ids': [[1, 2, 3]]}]}
        self.assertTrue(compare_measurements(report, copy.deepcopy(report))['bit_exact_tokens_and_tau'])
        for mutation in ('token', 'tau'):
            changed = copy.deepcopy(report)
            if mutation == 'token':
                changed['runs'][0]['output_token_ids'][0][-1] = 4
            else:
                changed['summary'][0]['tau'] = 3.0
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                compare_measurements(report, changed)

    def test_neutral_logits_require_exact_bits_and_positions(self) -> None:
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / 'a.safetensors', Path(folder) / 'b.safetensors'
            tensors = {'short.test.candidate_logits': torch.tensor([[0., 0.125, -4.]]),
                       'short.test.input_ids': torch.tensor([7]), 'short.test.positions': torch.tensor([0])}
            save_file(tensors, a, metadata={'model_id': 'pinned'})
            save_file(tensors, b, metadata={'model_id': 'pinned'})
            self.assertTrue(compare_logits(a, b)['bit_exact'])
            tensors['short.test.candidate_logits'][0, 0] = -0.0
            save_file(tensors, b, metadata={'model_id': 'pinned'})
            with self.assertRaises(RuntimeError):
                compare_logits(a, b)


if __name__ == '__main__':
    unittest.main()
