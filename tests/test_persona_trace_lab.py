"""Trace campaigns preserve endpoint identities, shared prompts and zero controls."""

import persona  # Make the flat, deployment-compatible source modules importable.
import json
import time
import unittest
from unittest.mock import patch
import numpy as np
import test_persona_explorer as explorer_fixtures
from persona_trace import pack_logits
from persona_vectors import canonical_digest

class TraceLabTests(unittest.TestCase):
    def setUp(self):
        self.fixture=explorer_fixtures.ExplorerWorkspaceTests()
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.app=self.fixture.app
        self.app.capacity=lambda:{'available':6}
        self.app.atlas.vocabulary=lambda:['one','two','three']

    def payload(self):
        return {'model_identity':self.app.model.identity,'prompts':['One shared prompt.'],
            'pushes':[{'kind':'neuron','layer':0,'neuron':1,'dose':.1}],'phase':'all',
            'neurons':[{'layer':0,'neuron':1}],'seed':7,'max_tokens':2,'mode':'dose_sweep'}

    def transport(self,url,payload=None,timeout=None):
        if url.endswith('/v1/models'):
            result=self.fixture.descriptor(url);result['data'][0]['flare']['token_trace']={'supported':True};return result
        if '/v1/trace?' in url:return {'state':'pending','events':[],'offset':0}
        scale=bool(payload.get('steering',{}).get('sites'))
        events=[{'kind':'prefill','positions':[0],'tokens':[{'id':1,'text':'two'}],'values':[[1.]],'residual_l2':[[2.,3.]]}]
        result=self.fixture.response(payload);result['usage']['prompt_tokens']=1
        result['token_trace']={'id':payload['capture_trace']['id'],'neurons':payload['capture_trace']['neurons'],'events':events,
            'next_token_logits':pack_logits(np.array([1.,2.+scale,3.])), 'semantics':'fixture'}
        result['activation_artifact']={'request_sha256':canonical_digest(payload)}
        return result

    def test_six_arms_share_prompt_and_zero_control_survives_persistence(self):
        with patch('persona_trace_lab.fetch',side_effect=self.transport):
            identity=self.app.traces.submit(self.payload())['id']
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                result=self.app.traces.get(identity)
                if result['state'] not in ('queued','running'):break
                time.sleep(.02)
        self.assertEqual(result['state'],'complete')
        self.assertEqual(len(result['arms']),6)
        zero=next(a for a in result['arms'] if a['scale']==0)
        self.assertTrue(zero['baseline_equal']);self.assertEqual(zero['lens']['kl_baseline_to_intervention'],0)
        plus=next(a for a in result['arms'] if a['scale']==1)
        self.assertGreater(plus['lens']['kl_baseline_to_intervention'],0)
        self.assertNotIn('raw_response',plus)
        audit=json.loads((self.app.root/'results'/f'persona_trace_{identity}.json').read_text())
        self.assertEqual(len(audit['arms']),6)
        self.assertIn('raw_response',audit['arms'][0])

    def test_identity_mismatch_is_failed_and_no_lens_is_invented(self):
        def broken(url,payload=None,timeout=None):
            result=self.transport(url,payload,timeout)
            if payload:result['runtime_identity']='other'
            return result
        with patch('persona_trace_lab.fetch',side_effect=broken):
            identity=self.app.traces.submit(self.payload())['id']
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                result=self.app.traces.get(identity)
                if result['state'] not in ('queued','running'):break
                time.sleep(.02)
        self.assertEqual(result['state'],'failed')
        self.assertTrue(all(a['state']=='failed' and 'lens' not in a for a in result['arms']))

    def test_invalid_grid_and_missing_compute_are_rejected_before_queue(self):
        for update in [{'max_tokens':513},{'prompts':['one','two']},{'mode':'anything'},{'neurons':[]}]:
            with self.assertRaises(ValueError):self.app.traces.submit({**self.payload(),**update})
        self.app.capacity=lambda:{'available':0}
        with self.assertRaisesRegex(ValueError,'fleet session'):self.app.traces.submit(self.payload())
