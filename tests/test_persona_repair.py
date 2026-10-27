import persona
from copy import deepcopy
import unittest
from persona_contrastive import build_batches,input_digest
from persona_repair import CASES,ARMS,freeze,blind,plan_digest,validate_repair_packet,validate_repair_judgments,analyze_repair,digest
import test_persona_blind_eval as blind_tests


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.fixture=blind_tests.BlindTests();self.fixture.setUp();f=self.fixture
        keys={r['code']:r for r in f.key}
        for row in f.judgments:
            key=keys[row['code']]
            if (key['scene_id'],key['seed'],key['arm']) in CASES:row['fidelity']='issue'
        f.bind();binding=dict(model_identity='a'*64,runtime_identity='b'*64,prepared_identity='c'*64)
        inputs=build_batches(f.scenes,binding,'pilot',dict(filename='direction.npz',sha256='d'*64))
        self.pilot=dict(schema='character-creation.contrastive-campaign.v1',state='complete',stage='pilot',binding=binding,model_id='test',scene_sha256='e'*64,
            policy=f.scenes['policy'],qualification_sha256='f'*64,direction_artifact=dict(filename='direction.npz',sha256='d'*64),
            batches=[dict(index=i,input=b,state='complete') for i,b in enumerate(inputs)])
        self.pilot['input_sha256']=input_digest(self.pilot)
    def freeze(self):
        f=self.fixture
        return freeze(self.pilot,f.scenes,f.packet,f.key,f.judgments,f.provenance)
    def test_exact_42_matched_doses_and_random(self):
        plan=self.freeze();self.assertEqual(len(plan['batches']),7)
        self.assertEqual(sum(len(b['input']['cells']) for b in plan['batches']),42)
        self.assertEqual(len(set(CASES)),7)
        for batch,case in zip(plan['batches'],CASES):
            self.assertEqual(batch['input']['seed'],case[1])
            cells=batch['input']['cells'];self.assertEqual({c['prompt'] for c in cells},{cells[0]['prompt']})
            self.assertEqual(cells[0]['recipe']['pushes'],[])
            self.assertEqual([c['recipe']['pushes'][0]['dose'] for c in cells[1:]],[.1,.05,.025,.05,.025])
            for cell in cells:
                self.assertEqual(cell['_record']['split'],'discovery');self.assertEqual(cell['recipe']['phase'],'all')
            for cell in cells[4:]:
                p=cell['recipe']['pushes'][0];self.assertEqual((p['seed'],p['site'],p['layer']),(481516,'residual',32))
    def test_changed_selection_or_heldout_rejected(self):
        self.fixture.judgments[0]['fidelity']='issue';self.fixture.bind()
        # Pick an explicit additional positive discovery cell, not baseline.
        extra=next(k['code'] for k in self.fixture.key if k['split']=='discovery' and k['arm']=='A' and (k['scene_id'],k['seed'],k['arm']) not in CASES)
        next(j for j in self.fixture.judgments if j['code']==extra)['fidelity']='issue';self.fixture.bind()
        with self.assertRaises(ValueError):self.freeze()
        changed=deepcopy(self.fixture.scenes);changed['evaluation'][0]['split']='heldout'
        with self.assertRaises(ValueError):freeze(self.pilot,changed,self.fixture.packet,self.fixture.key,self.fixture.judgments,self.fixture.provenance)
    def test_blind_packet_omits_arm_dose_and_target(self):
        plan=self.freeze()
        for b in plan['batches']:
            b['result']={'cells':[dict(index=i,output={'answer':'Synthetic answer.'}) for i in range(6)]}
        plan['state']='complete';packet,key=blind(plan,self.fixture.scenes)
        self.assertEqual(len(key),42);self.assertEqual(len({r['code'] for r in key}),42)
        for group in packet['groups']:
            self.assertEqual(set(group),{'group','prompt','fact_contract','answers'})
            for a in group['answers']:self.assertEqual(set(a),{'code','answer'})
        plan['batches'][0]['input']['seed']+=1
        with self.assertRaises(ValueError):blind(plan,self.fixture.scenes)

    def test_separate_42_judge_validator_and_exact_unblind_join(self):
        plan=self.freeze()
        for b in plan['batches']:
            b['result']={'cells':[dict(index=i,output={'answer':'Synthetic answer.'}) for i in range(6)]}
        plan['state']='complete';packet,key=blind(plan,self.fixture.scenes)
        validate_repair_packet(packet,self.fixture.scenes)
        rows=[dict(code=a['code'],dry_wit=1,attentive_candor=2,fidelity='ok',coherence='uncertain',evidence='Synthetic fixture.') for g in packet['groups'] for a in g['answers']]
        validate_repair_judgments(packet,rows)
        for changed in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):validate_repair_judgments(packet,changed)
        provenance=dict(packet_sha256=digest(packet),judgments_sha256=digest(rows),scenes_sha256=digest(self.fixture.scenes))
        report,public=analyze_repair(packet,key,rows,provenance,plan,self.fixture.scenes)
        self.assertEqual(report['arms']['half']['coherence'],{'uncertain':7})
        self.assertEqual(len(report['paired_cases']),28)
        self.assertFalse(report['automatic_default_promotion'])
        self.assertEqual(len(public['key']),42)
        self.assertEqual(len(public['design']),7)
        self.assertEqual(public['binding'],plan['binding'])
        original=deepcopy(self.pilot)
        for b in original['batches']:
            b['result']={'cells':[dict(output=dict(answer='Synthetic answer.',response_capture={'generated_token_ids_sha256':'a'*64})) for _ in b['input']['cells']]}
        for b in plan['batches']:
            for c in b['result']['cells']:c['output']['response_capture']={'generated_token_ids_sha256':'a'*64}
        plan['pilot_sha256']=digest(original);plan['input_sha256']=plan_digest(plan)
        _,export=analyze_repair(packet,key,rows,provenance,plan,self.fixture.scenes,original)
        self.assertEqual(len(export['replays']),14)
        self.assertTrue(all(r['generated_token_ids_equal'] and r['answer_equal'] for r in export['replays']))
        original['binding']['runtime_identity']='0'*64
        with self.assertRaises(ValueError):analyze_repair(packet,key,rows,provenance,plan,self.fixture.scenes,original)
        changed=deepcopy(key);changed[0]['arm']='wrong'
        with self.assertRaises(ValueError):analyze_repair(packet,changed,rows,provenance,plan,self.fixture.scenes)
        changed=deepcopy(packet);changed['groups'][0]['prompt']='Private book source'
        with self.assertRaises(ValueError):validate_repair_packet(changed,self.fixture.scenes)


if __name__=='__main__':unittest.main()
