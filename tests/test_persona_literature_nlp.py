"""Small real-model/SQLite checks; no production corpus or GPU operations."""
from pathlib import Path
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'persona'))
import persona_literature_nlp as nlp


class NLPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import spacy
        cls.pipeline = spacy.load(nlp.MODEL)
        nlp.PIPELINE = cls.pipeline

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / 'literature.sqlite'
        with sqlite3.connect(self.database) as db:
            db.executescript('''
            CREATE TABLE books(id TEXT PRIMARY KEY,title TEXT,sha256 TEXT);
            CREATE TABLE documents(id TEXT PRIMARY KEY,book_id TEXT,text TEXT,sha256 TEXT,excluded_reason TEXT);
            CREATE TABLE units(id TEXT PRIMARY KEY,book_id TEXT,document_id TEXT,start INTEGER,end INTEGER,text TEXT,split TEXT,text_sha256 TEXT,canonical_id TEXT);
            ''')
        self.add_source('u1', 'b1', 'd1', 'Barack Obama visited Paris on Monday. “We should leave,” he said.\n\n***\nLater that night, he returned.', prefix='Préface\n')
        self.add_source('u2', 'b2', 'd2', 'Barack Obama visited Paris on Monday. “We should leave,” he said.\n\n***\nLater that night, he returned.', prefix='A longer preface\n')
        self.add_source('u3', 'b1', 'd3', 'Paris welcomed the visitors. Barack Obama spoke about travel and the city.')

    def add_source(self, unit, book, document, text, prefix=''):
        source = prefix + text
        with sqlite3.connect(self.database) as db:
            db.execute('INSERT OR IGNORE INTO books VALUES(?,?,?)', (book, book, nlp.text_sha(book)))
            db.execute('INSERT INTO documents VALUES(?,?,?,?,?)', (document, book, source, nlp.text_sha(source), None))
            db.execute('INSERT INTO units VALUES(?,?,?,?,?,?,?,?,?)', (unit, book, document, len(prefix), len(source), text, 'discovery', nlp.text_sha(text), unit))

    def test_real_ner_sentence_dialogue_and_cue_offsets(self):
        text = 'Barack Obama visited Paris on Monday. “Go now,” he said.\n\n***\nLater that night, he slept.'
        result = nlp.analyze({'canonical_id': 'c1', 'text': text, 'text_sha256': nlp.text_sha(text)})
        self.assertEqual(result['state'], 'complete')
        self.assertTrue(any(s['kind'] == 'entity' and s['text'] == 'Barack Obama' and s['label'] == 'PERSON' for s in result['spans']))
        self.assertTrue(any(s['kind'] == 'sentence' for s in result['spans']))
        self.assertTrue(any(s['kind'] == 'dialogue_candidate' and s['text'] == '“Go now,”' for s in result['spans']))
        self.assertTrue(any(s['label'] == 'explicit_scene_marker' for s in result['spans']))
        self.assertTrue(any(s['label'] == 'time_shift_lexical_cue' for s in result['spans']))
        for span in result['spans']:
            self.assertEqual(text[span['start']:span['end']], span['text'])
            if span['kind'] == 'scene_boundary_candidate':
                self.assertFalse(span['evidence']['verified_scene_boundary'])

    def test_dedup_and_tamper_rejection(self):
        items, aliases, _ = nlp.source_snapshot(self.database)
        self.assertEqual(len(items), 2)
        self.assertEqual(len(aliases), 3)
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE units SET text='tampered' WHERE id='u1'")
        with self.assertRaisesRegex(ValueError, 'exact document source'):
            nlp.source_snapshot(self.database)

    def test_resume_graph_alias_offsets_lexical_search(self):
        before = self.source_snapshot()
        result = nlp.index(self.database, workers=1)
        run_id = result['run_id']
        self.assertEqual(result['state'], 'complete')
        self.assertEqual(result['counts'], {'complete': 2})
        self.assertEqual(before, self.source_snapshot())
        first = nlp.spans_for_unit(self.database, run_id, 'u1')
        second = nlp.spans_for_unit(self.database, run_id, 'u2')
        self.assertEqual([(s['start'], s['end'], s['text']) for s in first], [(s['start'], s['end'], s['text']) for s in second])
        with sqlite3.connect(self.database) as db:
            sources = dict(db.execute('SELECT id,text FROM documents'))
            for row in first + second:
                self.assertEqual(sources[row['document_id']][row['document_start']:row['document_end']], row['text'])
            count = db.execute('SELECT COUNT(*) FROM nlp_spans').fetchone()[0]
        again = nlp.index(self.database, workers=1)
        self.assertEqual(again['run_id'], run_id)
        with sqlite3.connect(self.database) as db:
            self.assertEqual(count, db.execute('SELECT COUNT(*) FROM nlp_spans').fetchone()[0])
        hits = nlp.lexical_search(self.database, run_id, 'Barack Obama Paris')
        self.assertEqual(len(hits['matches']), 2)
        self.assertEqual(sorted(len(r['sources']) for r in hits['matches']), [1, 2])
        self.assertIn('no semantic encoder', hits['semantics'])
        graph = nlp.graph_for_unit(self.database, run_id, 'u1')
        self.assertTrue(graph['available'])
        self.assertTrue({'book', 'document', 'window', 'sentence', 'entity_surface'} <= {r['kind'] for r in graph['nodes']})
        self.assertTrue(nlp.neighbors(self.database, 'u1')['rows'])
        self.assertTrue(nlp.entity_facets(self.database)['entities'])
        self.assertEqual(nlp.status(self.database)['counts'], {'complete': 2})
        self.assertEqual(before, self.source_snapshot())

    def test_missing_index_visible(self):
        self.assertIsNone(nlp.latest_run(self.database))
        self.assertFalse(nlp.status(self.database)['available'])
        self.assertFalse(nlp.neighbors(self.database, 'u1')['available'])

    def test_pre_reserved_physical_core_is_not_reserved_twice(self):
        allowed = os.sched_getaffinity(0)
        groups = {}
        for cpu in allowed:
            base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
            key = ((base / 'physical_package_id').read_text(), (base / 'core_id').read_text())
            groups.setdefault(key, set()).add(cpu)
        for group in (next(iter(groups.values())), list(groups.values())[-1]):
            subset = allowed - group
            with patch.object(nlp.os, 'sched_getaffinity', return_value=subset):
                self.assertEqual(set(nlp.available_cpus()), subset)

    def source_snapshot(self):
        with sqlite3.connect(self.database) as db:
            return {table: db.execute('SELECT * FROM ' + table + ' ORDER BY id').fetchall() for table in ('books', 'documents', 'units')}


if __name__ == '__main__':
    unittest.main()
