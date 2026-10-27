"""Registered diagnostics preserve exact prompts and neutral comparison inputs."""
import persona
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from persona_explorer import Explorer,prompt_label
from persona_personality_probes import load_suite,registered_rows
from persona_prompt_browser import browse
from persona_vectors import canonical_digest

ROOT=Path(__file__).resolve().parents[1]

class DiagnosticIntegrationTests(unittest.TestCase):
    def test_registered_browser_axis_filters_exact_shared_text(self):
        suite=load_suite(ROOT/'data/persona_personality_probes.json')
        rows=registered_rows(suite);by_prompt={c['prompt']:c for c in suite['cases']}
        self.assertEqual(len(rows),24)
        for row in rows:
            self.assertEqual(row['system'],'')
            self.assertIn(row['prompt'],by_prompt)
            self.assertEqual(row['id'],canonical_digest({k:v for k,v in row.items() if k!='id'}))
        for axis in suite['axes']:
            result=browse({r['id']:r for r in rows},prompt_label,corpus='personality_diagnostics',trait=axis['id'],limit=24)
            self.assertEqual(result['total'],4)
            self.assertTrue(all(r['system']=='' for r in result['rows']))
    def test_existing_corpus_preserved_and_diagnostic_source_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'data').mkdir()
            row=dict(schema='character-creation.manager-corpus.v1',corpus='fixture',category='fixture',split='discovery',prompt='Existing immutable prompt',system='',max_tokens=2048)
            row['id']=canonical_digest(row)
            raw=(json.dumps(row)+'\n').encode()
            (root/'data/persona_manager_corpus.jsonl').write_bytes(raw)
            (root/'data/persona_manager_corpus_manifest.json').write_text(json.dumps({'status':'complete','output_sha256':hashlib.sha256(raw).hexdigest()}))
            shutil.copyfile(ROOT/'data/persona_personality_probes.json',root/'data/persona_personality_probes.json')
            app=Explorer.__new__(Explorer);app.root=root;app.corpus_lock=threading.Lock();app.corpus_signature=None;app.corpus_cache={}
            rows,binding=app.corpus_snapshot()
            self.assertEqual(len(rows),25);self.assertEqual(rows[row['id']],row)
            self.assertEqual(binding['personality_diagnostics_sha256'],hashlib.sha256((root/'data/persona_personality_probes.json').read_bytes()).hexdigest())
            self.assertEqual(app.corpus_snapshot(),(rows,binding))

if __name__=='__main__':unittest.main()
