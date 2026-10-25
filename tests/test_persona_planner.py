"""Planner launcher checks with all network/workboard mutation mocked."""

import persona  # Make the flat, deployment-compatible source modules importable.
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import persona_planner


class PlannerTests(unittest.TestCase):
    def test_fixed_runtime_has_no_context_or_weight_offload_fallback(self):
        command = persona_planner.command()
        self.assertEqual(command[command.index('-c') + 1], '131072')
        self.assertEqual(command[command.index('--fit') + 1], 'off')
        self.assertEqual(command[command.index('-np') + 1], '1')
        self.assertEqual(command[command.index('--host') + 1], '127.0.0.1')

    def test_readiness_requires_exact_context_and_alias(self):
        def response(value):
            return contextlib.closing(io.BytesIO(json.dumps(value).encode()))
        for context, alias, valid in [(131072, 'persona-manager', True), (32768, 'persona-manager', False),
                                       (131072, 'someone-else', False)]:
            replies = [response({'status': 'ok'}), response({'default_generation_settings': {'n_ctx': context}, 'model_alias': alias})]
            with patch('persona_planner.urllib.request.urlopen', side_effect=replies):
                if valid:
                    self.assertEqual(persona_planner.readiness()['context_tokens'], 131072)
                else:
                    with self.assertRaises(ValueError):
                        persona_planner.readiness()

    def test_workboard_batch_checkpoint_arguments_parse_without_submit(self):
        path = Path('/home/orwel/dev_genius/gpu_workboard/gpu_workboard.py')
        if not path.is_file():
            self.skipTest('Atlas workboard source is not installed on this host')
        spec = importlib.util.spec_from_file_location('workboard_test', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        args = module.public_parser().parse_args(['submit', '--project', 'character-creation', '--label', 'persona-planner-john',
            '--vram-gib', '40', '--duration-min', '480', '--host', 'wtatum84', '--gpu', persona_planner.GPU,
            '--lane', 'batch', '--checkpoint-signal', 'SIGUSR1', '--checkpoint-grace-s', '30', '--json'])
        with patch.object(module, 'make_request', return_value={'job': {'id': 'mock-only'}}) as request, contextlib.redirect_stdout(io.StringIO()):
            module.public_main(args)
        name, parameters = request.call_args.args
        self.assertEqual(name, 'submit')
        self.assertEqual(parameters['requested_mib'], 40960)
        self.assertEqual(parameters['lane'], 'batch')
        self.assertIsNone(parameters['preemptible'])  # The coordinator defaults batch to preemptible.
        self.assertEqual(parameters['checkpoint_signal'], 'SIGUSR1')
        self.assertEqual(parameters['checkpoint_grace_s'], 30)


if __name__ == '__main__':
    unittest.main()
