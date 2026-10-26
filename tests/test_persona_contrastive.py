"""CPU-only campaign design and custom direction admission checks."""
import persona
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from persona_contrastive import ARMS, build_batches, scenes, validate_result, input_digest
from persona_vectors import ModelSpec, VectorBank, canonical_digest
from persona_explorer import Explorer

ROOT = Path(__file__).resolve().parents[1]


class ContrastiveDesignTests(unittest.TestCase):
    def setUp(self):
        self.data, self.sha = scenes(ROOT)
        self.binding = dict(model_identity='a'*64, runtime_identity='b'*64, prepared_identity='c'*64)
        self.artifact = dict(filename='direction.npz', sha256='d'*64)

    def test_256_answers_balance_and_split_independence(self):
        batches = build_batches(self.data, self.binding, 'pilot', self.artifact)
        self.assertEqual(len(batches), 43)
        self.assertEqual(sum(len(b['cells']) for b in batches), 256)
        self.assertEqual(Counter(c['_record']['arm'] for b in batches[:-1] for c in b['cells']),
                         Counter({arm:42 for arm in ARMS}))
        self.assertEqual(Counter(c['_record']['split'] for b in batches[:-1] for c in b['cells']),
                         Counter(discovery=144, heldout=108))
        self.assertTrue(all(len(b['cells']) == 6 for b in batches[:-1]))
        rows = self.data['development'] + self.data['evaluation']
        self.assertEqual(len({r['family_id'] for r in rows}), 29)
        self.assertTrue({r['id'] for r in self.data['development']}.isdisjoint(
            r['id'] for r in self.data['evaluation']))
        for i, row in enumerate(self.data['evaluation']):
            for j, seed in enumerate(self.data['policy']['seeds']):
                batch = batches[i*2+j]
                self.assertEqual(batch['seed'], seed)
                self.assertEqual({c['prompt'] for c in batch['cells']}, {row['prompt']})
                self.assertEqual({c['_record']['arm'] for c in batch['cells']}, set(ARMS))
                self.assertEqual(batch['max_tokens'], 2048)
                self.assertFalse(batch['thinking'])
                for cell in batch['cells']:
                    self.assertEqual(cell['recipe']['phase'], 'all')
                    pushes = cell['recipe']['pushes']
                    arm = cell['_record']['arm']
                    if arm == 'baseline': self.assertEqual(pushes, [])
                    else:
                        self.assertEqual(len(pushes), 1)
                        self.assertEqual(pushes[0]['layer'], 32)
                        self.assertEqual(pushes[0]['dose'], -.1 if arm.endswith('_reverse') else .1)

    def test_four_replays_match_original_request_settings_not_new_evidence(self):
        batches = build_batches(self.data, self.binding, 'pilot', self.artifact)
        replay = batches[-1]
        for cell in replay['cells']:
            ref = cell['_record']['replay_of']
            original_batch = batches[ref['batch_index']]
            original = next(c for c in original_batch['cells'] if c['_record']['arm'] == ref['arm'])
            for key in ('prompt', 'recipe'):
                self.assertEqual(cell[key], original[key])
            for key in ('seed', 'model_identity', 'max_tokens', 'thinking', 'smoke'):
                self.assertEqual(replay[key], original_batch[key])
            self.assertNotEqual(cell['capture_response']['id'], original['capture_response']['id'])
        ids = [c['capture_response']['id'] for b in batches for c in b['cells']]
        self.assertEqual(len(set(ids)), 256)

    def test_development_only_24_matched_response_contrasts(self):
        batches = build_batches(self.data, self.binding, 'development')
        cells = [c for b in batches for c in b['cells']]
        self.assertEqual(len(cells), 24)
        self.assertEqual(Counter(c['_record']['arm'] for c in cells), Counter(baseline=8,A=8,B=8))
        self.assertEqual({c['_record']['split'] for c in cells}, {'development'})
        self.assertTrue(all(not c['recipe']['pushes'] for c in cells))
        for c in cells:
            row = next(r for r in self.data['development'] if r['id']==c['_record']['scene_id'])
            self.assertEqual(c['prompt'], self.data['styles'][c['_record']['arm']]+'\n\n'+row['prompt'])

    def assert_scene_rejected(self, edit):
        data=deepcopy(self.data); edit(data)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'data').mkdir()
            (root/'data/persona_contrastive_scenes.json').write_text(json.dumps(data))
            with self.assertRaises(ValueError): scenes(root)

    def test_source_and_family_tampering_rejected(self):
        self.assert_scene_rejected(lambda d:d['development'][0].update(prompt='Changed'))
        self.assert_scene_rejected(lambda d:d['evaluation'][0].update(family_id=d['development'][0]['family_id']))

    def test_heldout_cannot_be_reclassified_as_development(self):
        self.assert_scene_rejected(lambda d:d['development'][0].update(split='heldout'))

    def test_duplicate_scene_ids_rejected_even_with_distinct_families(self):
        self.assert_scene_rejected(lambda d:d['evaluation'][0].update(id=d['development'][0]['id']))


class CapturedResultTests(unittest.TestCase):
    def setUp(self):
        self.model=ModelSpec('test/model','a'*40,'test',2,4,6)
        self.bank=VectorBank(Path('/unused'),self.model)
        self.binding=dict(model_identity=self.model.identity,runtime_identity='b'*64,prepared_identity='c'*64)
        cell=dict(label='synthetic baseline',prompt='Synthetic scene.',recipe=dict(name='baseline',pushes=[],phase='all'),
                  capture_response=dict(id='d'*32,layers=[],max_positions=0))
        request=dict(seed=7035,max_tokens=2048,thinking=False,smoke=False,model_identity=self.model.identity,cells=[cell])
        self.row=dict(batch_id='e'*32,input=deepcopy(request))
        self.row['input']['cells'][0]['_record']={'split':'development'}
        native=dict(model='test/model',messages=[dict(role='user',content=cell['prompt'])],seed=7035,
                    temperature=.7,top_p=.95,top_k=20,max_tokens=2048,stream=False,
                    chat_template_kwargs=dict(enable_thinking=False),capture_response=cell['capture_response'],
                    metadata=dict(job_id=self.row['batch_id']+'-0'))
        sha=canonical_digest(native)
        capture=dict(self.binding,request_sha256=sha,id='d'*32,mean_defined=True,
                     consumed_generated_tokens=2,sampled_generated_tokens=3,
                     all_consumed_generated_tokens_captured=True,generation_complete=True,
                     artifact=dict(self.binding,request_sha256=sha))
        program,details=self.bank.program([],'all')
        output=dict(complete=True,answer='Synthetic answer.',response_capture=capture,
                    provenance=dict(self.binding,request_sha256=sha,program_sha256=details['program_sha256']))
        result_cell=dict(index=0,label=cell['label'],recipe=cell['recipe'],prompt=dict(text=cell['prompt']),state='complete',output=output)
        self.result=dict(id=self.row['batch_id'],input=request,cells=[result_cell])

    def test_exact_native_request_is_accepted(self):
        validate_result(self.result,self.row,self.binding,self.bank,'test/model')
        self.assertIn('surface_evaluation',self.result['cells'][0])

    def test_seed_request_binding_and_coverage_tampering_rejected(self):
        edits=[lambda r:r['input'].update(seed=7036),
               lambda r:r['cells'][0]['output']['response_capture'].update(request_sha256='f'*64),
               lambda r:r['cells'][0]['output']['response_capture']['artifact'].update(runtime_identity='f'*64),
               lambda r:r['cells'][0]['output']['provenance'].update(program_sha256='f'*64),
               lambda r:r['cells'][0]['output']['response_capture'].update(consumed_generated_tokens=3),
               lambda r:r['cells'][0]['output'].update(complete=False)]
        for edit in edits:
            result=deepcopy(self.result);edit(result)
            with self.subTest(edit=edit),self.assertRaises(ValueError):
                validate_result(result,self.row,self.binding,self.bank,'test/model')

    def test_input_digest_excludes_progress_not_frozen_requests(self):
        plan=dict(schema='test',stage='development',binding=self.binding,model_id='test/model',scene_sha256='a'*64,
                  policy={},direction_artifact=None,qualification_sha256='b'*64,
                  batches=[dict(index=0,input=self.row['input'],state='pending')])
        original=input_digest(plan)
        plan['batches'][0].update(state='complete',result=self.result)
        self.assertEqual(input_digest(plan),original)
        plan['batches'][0]['input']['seed']+=1
        self.assertNotEqual(input_digest(plan),original)


class DirectionValidationTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory(); self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name)
        self.model=ModelSpec('test/model','a'*40,'test',2,4,6)
        self.bank=VectorBank(self.root,self.model)
        for layer in range(2):
            np.savez(self.root/f'layer_{layer:02d}.npz',model_identity=self.model.identity,residual_median=10.)
        self.metadata=dict(schema='character-creation.response-directions.v1',model_identity=self.model.identity,
                           runtime_identity='b'*64,prepared_identity='c'*64,split='development',
                           sources=[dict(scene_id='synthetic',style='A',artifact_sha256='d'*64)])

    def save(self, filename='direction.npz', metadata=None, vector=None):
        np.savez(self.root/filename,metadata=json.dumps(metadata or self.metadata),
                 direction_A=np.ones((2,4)) if vector is None else vector)
        return dict(kind='direction',layer=0,dose=.1,artifact=filename,direction='A',
                    sha256=hashlib.sha256((self.root/filename).read_bytes()).hexdigest())

    def test_normalized_sign_reversal_preserves_runtime_binding(self):
        push=self.save(); positive, provenance=self.bank.program([push])
        negative,_=self.bank.program([dict(push,dose=-.1)])
        np.testing.assert_allclose(positive['sites'][0]['vector'], -np.array(negative['sites'][0]['vector']))
        self.assertAlmostEqual(np.linalg.norm(positive['sites'][0]['vector']),1.)
        self.assertEqual(positive['sites'][0]['site'],'layers.0.residual')
        binding={key:self.metadata[key] for key in ('model_identity','runtime_identity','prepared_identity')}
        self.assertEqual(provenance['direction_binding'],binding)
        Explorer.check_direction_binding(binding,provenance)
        for key in binding:
            with self.subTest(key=key), self.assertRaises(ValueError):
                Explorer.check_direction_binding(dict(binding,**{key:'e'*64}),provenance)

    def test_model_split_runtime_and_shape_fail_closed(self):
        for key,value in [('model_identity','e'*64),('runtime_identity','unknown'),
                          ('prepared_identity',''),('split','heldout'),('sources',[])]:
            with self.subTest(key=key):
                push=self.save(metadata=dict(self.metadata,**{key:value}))
                with self.assertRaises(ValueError): self.bank.program([push])
        for vector in (np.ones((1,4)),np.zeros((2,4)),np.full((2,4),np.nan)):
            push=self.save(vector=vector)
            with self.assertRaises(ValueError): self.bank.program([push])

    def test_hash_path_and_symlink_rejection(self):
        push=self.save()
        for change in ({'sha256':'0'*64},{'artifact':'../direction.npz'},{'direction':'../A'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                self.bank.program([dict(push,**change)])
        (self.root/'linked.npz').symlink_to(self.root/'direction.npz')
        with self.assertRaises(ValueError): self.bank.program([dict(push,artifact='linked.npz')])

    def test_mixed_direction_runtimes_rejected(self):
        first=self.save()
        second=self.save('other.npz',dict(self.metadata,runtime_identity='e'*64))
        with self.assertRaises(ValueError): self.bank.program([first,second])


if __name__=='__main__': unittest.main()
