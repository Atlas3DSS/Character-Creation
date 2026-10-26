import sys
from pathlib import Path
import sqlite3
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'persona'))
from persona_literature import initialize
from persona_literature_harness import compile_prompt

class HarnessTests(unittest.TestCase):
 def test_matched_variants_and_continuation_no_future_leak(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'books.sqlite';initialize(p)
   text=('A traveler hears a warning at the gate. '*30)+'SECRET_FUTURE_END'
   with sqlite3.connect(p) as db:
    db.execute("INSERT INTO books(id,title,sha256,workgroup,split) VALUES('b','Book','hash','family','heldout')")
    db.execute("INSERT INTO documents(id,book_id,member,text) VALUES('d','b','chapter',?)",(text,))
    db.execute("INSERT INTO units(id,book_id,document_id,start,end,text,split,boundary,canonical_id) VALUES('u','b','d',0,?,?,'heldout','continuation_window','u')",(len(text),text))
   args=dict(unit_id='u',mode='decision',variant='original',seed=7)
   a=compile_prompt(p,args);self.assertEqual(a,compile_prompt(p,args))
   b=compile_prompt(p,{**args,'variant':'time_pressure'})
   self.assertEqual(a['provenance']['source_sha256'],b['provenance']['source_sha256'])
   self.assertNotEqual(a['id'],b['id']);self.assertEqual(a['provenance']['split'],'heldout')
   c=compile_prompt(p,{**args,'mode':'continuation'})
   self.assertNotIn('SECRET_FUTURE_END',c['prompt']);self.assertTrue(c['evaluation']['heldback_sha256'])
   with self.assertRaises(ValueError):compile_prompt(p,{**args,'variant':'invented'})
