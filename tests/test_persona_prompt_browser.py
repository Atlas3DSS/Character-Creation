"""Metadata navigation preserves split/filter coverage and page reachability."""
import persona
import unittest
from persona_prompt_browser import browse, facets

class BrowserTests(unittest.TestCase):
    def setUp(self):
        self.rows = {str(i): {'id':str(i),'prompt':f'Example {i}', 'category':'scenario', 'corpus':'big_five',
            'split':'heldout' if i == 100 else 'discovery','labels':{'prompt_design_trait':'openness' if i % 2 else 'agreeableness',
            'prompt_design_facet':'curiosity','scenario_domain':'work'}, 'synthetic':True} for i in range(101)}
    def query(self, **kw): return browse(self.rows, lambda r:r['id'], **kw)
    def test_graph_counts_are_filtered_and_exclude_heldout(self):
        data=self.query(trait='openness',group='facet');self.assertEqual(data['total'],50)
        self.assertEqual(data['groups'],[{'key':'curiosity','count':50}])
        self.assertTrue(all(r['facets']['trait']=='openness' for r in data['rows']))
    def test_paging_reaches_all_rows_once(self):
        ids=[r['id'] for offset in range(0,100,24) for r in self.query(offset=offset)['rows']]
        self.assertEqual(len(ids),100);self.assertEqual(len(set(ids)),100)
    def test_search_matches_metadata_and_spread_reaches_end(self):
        data=self.query(q='curiosity',spread=True,limit=24)
        self.assertEqual(data['total'],100);self.assertEqual(data['rows'][0]['id'],'0');self.assertEqual(data['rows'][-1]['id'],'99')
        self.assertEqual(len({r['id'] for r in data['rows']}),24)
    def test_empty_and_page_clamping(self):
        self.assertEqual(self.query(q='absent',offset=100)['rows'],[])
        self.assertEqual(self.query(offset=999)['offset'],96)
    def test_reject_invalid_filters(self):
        for kw in ({'group':'bad'},{'limit':101},{'split':'bad'},{'q':'x'*513}):
            with self.assertRaises(ValueError):self.query(**kw)

    def test_adult_suite_metadata_and_identity(self):
        import json
        from pathlib import Path
        from persona_vectors import canonical_digest
        rows=json.loads((Path(__file__).resolve().parents[1]/'data/persona_adult_sexuality.json').read_text())
        self.assertEqual(len(rows),60)
        for row in rows:
            self.assertEqual(row['id'],canonical_digest({k:v for k,v in row.items() if k!='id'}))
            self.assertEqual(row['corpus'],'adult_sexuality')
            self.assertNotEqual(facets(row)['facet'],'unlabelled')
        self.assertEqual(len({row['pair_id'] for row in rows}),30)
