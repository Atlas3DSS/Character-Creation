"""Mathematical reference checks for static-lens and weight-web semantics."""

import persona  # Make the flat, deployment-compatible source modules importable.
import unittest
import numpy as np
from persona_atlas import norm_direction, cosine_channels

class AtlasMathTests(unittest.TestCase):
    def test_zero_centered_norm_and_opposite_signs(self):
        x=np.array([[3.,4.],[0.,0.]])
        weight=np.array([.25,-.5])
        expected=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)*(1+weight)
        np.testing.assert_allclose(norm_direction(x,weight,1e-6),expected)
        np.testing.assert_allclose(norm_direction(-x,weight,1e-6),-expected)
        self.assertTrue(np.isfinite(expected).all())

    def test_weight_edges_keep_sign_channel_and_direction(self):
        down=np.array([[1.,0.],[0.,1.],[0.,0.]])
        gate=np.array([[-2.,0.],[.1,.9]])
        up=np.array([[0.,.5],[1.,0.]])
        scores,channel=cosine_channels(down,gate,up)
        np.testing.assert_allclose(scores[0],[-1.,1.])
        np.testing.assert_array_equal(channel[0],[0,1])
        self.assertEqual(scores[1,0],1.)
        self.assertEqual(channel[1,0],1)
        np.testing.assert_array_equal(scores[2],[0.,0.])
        self.assertEqual(scores.shape,(3,2))

if __name__=='__main__':unittest.main()

class AtlasReaderTests(unittest.TestCase):
    def setUp(self):
        import tempfile,json
        from pathlib import Path
        from persona_vectors import ModelSpec,canonical_digest
        from persona_atlas import SCHEMA,SEMANTICS,WeightAtlas,publish
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'results').mkdir();(self.root/'data').mkdir()
        self.model=ModelSpec('fixture/model','a'*40,'fixture',2,3,3)
        (self.root/'data/persona_model.json').write_text(json.dumps(self.model.__dict__))
        source={'schema':SCHEMA,'model_identity':self.model.identity,'model':self.model.__dict__,'semantics':SEMANTICS}
        source_id=canonical_digest(source)
        for layer in range(2):
            arrays={'positions':np.array([[0.,0.],[0.,1.],[0.,2.]],dtype=np.float32)}
            for sign in ('positive','negative'):
                arrays[sign+'_ids']=np.tile(np.arange(24,dtype=np.int32),(3,1))
                arrays[sign+'_scores']=np.tile(np.arange(24,dtype=np.float32),(3,1))
            if layer==0:
                for direction in ('in','out'):
                    arrays[direction+'_ids']=np.tile(np.arange(16,dtype=np.int32)%3,(3,1))
                    arrays[direction+'_scores']=np.full((3,16),-.4 if direction=='out' else .3,dtype=np.float32)
                    arrays[direction+'_channels']=np.zeros((3,16),dtype=np.uint8)
            metadata={**source,'source_id':source_id,'layer':layer}
            np.savez_compressed(self.root/'results'/f'persona_atlas_L{layer}.npz',metadata=np.array(json.dumps(metadata)),**arrays)
        (self.root/'results/persona_atlas_tokens.json').write_text(json.dumps({'source_id':source_id,'tokens':[str(i) for i in range(24)]}))
        publish(self.root);self.index=WeightAtlas(self.root,self.model)

    def test_reader_lens_directed_hops_and_token_search(self):
        self.assertEqual(self.index.query(0,1)['positive'][0]['text'],'0')
        web=self.index.web(0,1,hops=1,fanout=2)
        self.assertTrue(web['graph']['directed'])
        self.assertTrue(web['graph']['edges'])
        self.assertTrue(all(e['source']=='L0N1' and e['score']<0 for e in web['graph']['edges']))
        self.assertTrue(web['pooled_tokens']['positive'])
        result=self.index.search('23')
        self.assertEqual(result['missing_layers'],[])
        self.assertTrue(result['matches'])

    def test_word_start_search_excludes_embedded_fragments(self):
        self.assertTrue(any(m['token']=='23' for m in self.index.search('3')['matches']))
        result=self.index.search('3',word_start=True)
        self.assertTrue(result['matches'])
        self.assertTrue(all(m['token']=='3' for m in result['matches']))

    def test_reader_rejects_tampering_and_reports_missing_layer(self):
        path=self.root/'results/persona_atlas_L0.npz'
        with path.open('ab') as stream:stream.write(b'tampered')
        with self.assertRaisesRegex(ValueError,'hash differs'):self.index.query(0,1)
        path.unlink()
        self.assertFalse(self.index.query(0,1)['available'])

    def test_invalid_controls_and_unpublished_artifact(self):
        from persona_atlas import WeightAtlas
        for kwargs in [{'hops':4},{'fanout':0},{'threshold':float('nan')},{'direction':'causal'}]:
            with self.assertRaises(ValueError):self.index.web(0,1,**kwargs)
        (self.root/'results/persona_atlas_manifest.json').unlink()
        self.assertFalse(WeightAtlas(self.root,self.model).query(0,1)['available'])
