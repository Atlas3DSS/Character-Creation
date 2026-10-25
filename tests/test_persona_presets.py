"""Validate the published persona catalog and its CPU-only recipe-save contract."""

import persona  # Make the flat, deployment-compatible source modules importable.
import hashlib
import math
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.parse import urlparse

import test_persona_sculptor as fixtures
from persona_sculptor import TRAITS


CATALOG = Path(__file__).resolve().parent.parent / 'data' / 'persona_presets.json'


class PublishedPersonaPresetTests(unittest.TestCase):
    def setUp(self):
        fixtures.PersonaSculptorTests.setUp(self)
        (self.root / 'data/persona_presets.json').write_bytes(CATALOG.read_bytes())

    def test_real_catalog_has_twenty_two_distinct_bounded_annotated_profiles(self):
        catalog = self.app.sculptor.presets()
        self.assertEqual(catalog['schema'], 'character-creation.persona-presets.v1')
        self.assertEqual(catalog['catalog_sha256'], hashlib.sha256(CATALOG.read_bytes()).hexdigest())
        presets = catalog['presets']
        self.assertEqual(len(presets), 22)
        self.assertEqual(len({row['id'] for row in presets}), 22)
        self.assertEqual(len({row['name'] for row in presets}), 22)
        self.assertEqual(len({tuple(row['dials'][trait] for trait in TRAITS) for row in presets}), 22)
        for row in presets:
            with self.subTest(preset=row['id']):
                self.assertEqual(set(row['dials']), set(TRAITS))
                self.assertTrue(all(type(value) in (int, float) and math.isfinite(value) and -1 <= value <= 1
                                    for value in row['dials'].values()))
                self.assertTrue(0 <= row['dose_budget'] <= .5)
                self.assertTrue(0 <= row['variation'] <= .5)
                self.assertEqual(row['phase'], 'all')
                self.assertTrue(row['evidence_basis'])
                self.assertTrue(row['unsupported_traits'])
                self.assertTrue(row['notes'])

    def test_named_inspirations_have_primary_source_citations_and_explicit_limits(self):
        first, second = self.app.sculptor.presets()['presets'][:2]
        self.assertEqual([first['id'], second['id']], ['gpt-4o-inspired', 'skippy-inspired'])
        for row, domains in ((first, ('openai.com', 'chatgpt.com')), (second, ('craigalanson.com',))):
            with self.subTest(preset=row['id']):
                self.assertTrue(row['sources'])
                for source in row['sources']:
                    parsed = urlparse(source['url'])
                    self.assertEqual(parsed.scheme, 'https')
                    self.assertTrue(any(parsed.hostname == domain or parsed.hostname.endswith('.' + domain) for domain in domains))
                    self.assertTrue(source['title'])
                evidence = row['evidence_basis'].lower()
                self.assertTrue('interpret' in evidence)
                self.assertTrue(any('not' in note.lower() for note in row['notes']))
        self.assertIn('not OpenAI parameters', first['evidence_basis'])
        self.assertIn('not claimed to be fit to book dialogue', ' '.join(second['notes']))

    def test_real_catalog_saves_distinct_model_bound_recipes_without_inference(self):
        presets = self.app.sculptor.presets()['presets']
        with patch('persona_sculptor.trusted_planner', side_effect=AssertionError('Preset must not launch John')), \
             patch('persona_sculptor.fetch', side_effect=AssertionError('Preset must not generate responses')):
            results = [self.app.sculptor.preset({'id': row['id'], 'count': 1, 'seed': 7035}) for row in presets]
            repeat = self.app.sculptor.preset({'id': presets[0]['id'], 'count': 1, 'seed': 7035})
        self.assertEqual(len({row['saved_recipe']['id'] for row in results}), 22)
        self.assertEqual(repeat['saved_recipe']['id'], results[0]['saved_recipe']['id'])
        for preset, result in zip(presets, results):
            with self.subTest(preset=preset['id']):
                self.assertEqual(result['saved_recipe']['model_identity'], self.model.identity)
                self.assertEqual(result['saved_recipe']['name'], preset['name'])
                self.assertEqual(result['saved_recipe']['pushes'], result['build']['candidates'][0]['pushes'])
                self.assertEqual(result['plan']['method'], 'preset_heuristic')
                self.assertTrue(result['preset']['approximation'])
                self.assertEqual(result['build']['preset']['sources'], preset['sources'])
                self.assertEqual(result['build']['preset']['notes'], preset['notes'])
                self.assertEqual(result['build']['preset']['catalog_sha256'], hashlib.sha256(CATALOG.read_bytes()).hexdigest())
                self.assertLessEqual(result['build']['candidates'][0]['aggregate_relative_norm'], .5 + 1e-9)


if __name__ == '__main__':
    unittest.main()
