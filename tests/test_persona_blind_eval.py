"""Synthetic judged answers on the frozen authored scene design; no inference."""
import persona
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from persona_blind_eval import ARMS,analyze,digest,public_export,validate


class BlindTests(unittest.TestCase):
    def setUp(self):
        self.scenes=json.loads((Path(__file__).resolve().parents[1]/'data/persona_contrastive_scenes.json').read_text())
        self.packet={'schema':'character-creation.blind-persona-eval.v1','groups':[]}
        self.key=[];self.judgments=[]
        for i,scene in enumerate(self.scenes['evaluation']):
            for j,seed in enumerate((17031,29147)):
                group={'group':i*2+j,'prompt':scene['prompt'],'fact_contract':scene['fact_contract'],'answers':[]}
                for k,arm in enumerate(ARMS):
                    code=hashlib.sha256(f'{i}:{j}:{k}'.encode()).hexdigest()[:12]
                    group['answers'].append({'code':code,'answer':'Synthetic answer.'})
                    self.key.append(dict(code=code,scene_id=scene['id'],split=scene['split'],seed=seed,arm=arm,batch_index=i*2+j,cell_index=k))
                    self.judgments.append(dict(code=code,dry_wit=4 if arm=='A' else 0 if arm=='A_reverse' else 2,
                        attentive_candor=4 if arm=='B' else 0 if arm=='B_reverse' else 2,
                        fidelity='ok',coherence='ok',evidence='Synthetic fixture judgment.'))
                self.packet['groups'].append(group)
        self.bind()

    def bind(self):
        self.provenance=dict(judge_identity='synthetic-test',method='fixture',packet_sha256=digest(self.packet),key_sha256=digest(self.key),judgments_sha256=digest(self.judgments))

    def analyze(self):return analyze(self.packet,self.key,self.judgments,self.provenance,bootstrap_samples=200)

    def test_scene_paired_means_and_reversal_target(self):
        result=self.analyze()
        for split,n in [('discovery',12),('heldout',9)]:
            self.assertEqual(result['splits'][split]['n_scenes'],n)
            contrasts=result['splits'][split]['paired_contrasts']
            self.assertEqual(contrasts['A_vs_random']['dry_wit']['ci95'],[2.,2.])
            self.assertEqual(contrasts['A_vs_A_reverse']['dry_wit']['mean'],4.)
            self.assertEqual(contrasts['A_reverse_vs_baseline']['dry_wit']['mean'],-2.)
        self.assertTrue(all(p['action']=='test_neighbor_doses' for p in result['followup_proposals']))
        self.assertFalse(result['automatic_default_promotion'])

    def test_heldout_outcomes_never_change_followup(self):
        original=self.analyze()['followup_proposals']
        heldout={r['code'] for r in self.key if r['split']=='heldout'}
        for row in self.judgments:
            if row['code'] in heldout:row.update(dry_wit=0,attentive_candor=0,fidelity='issue',coherence='uncertain')
        self.bind()
        self.assertEqual(self.analyze()['followup_proposals'],original)

    def test_issue_and_uncertain_are_separate_and_block_proposal(self):
        target=next(r['code'] for r in self.key if r['split']=='discovery' and r['arm']=='A')
        next(r for r in self.judgments if r['code']==target).update(fidelity='issue',coherence='uncertain')
        self.bind();result=self.analyze()
        contrast=result['splits']['discovery']['paired_contrasts']['A_vs_baseline']
        self.assertGreater(contrast['fidelity_issue']['mean'],0)
        self.assertEqual(contrast['fidelity_uncertain']['mean'],0)
        self.assertGreater(contrast['coherence_uncertain']['mean'],0)
        self.assertEqual(result['followup_proposals'][0]['action'],'do_not_promote')

    def test_exact_unique_coverage_and_score_types(self):
        original=deepcopy(self.judgments)
        for change in ('missing','duplicate','bool','range','state','evidence'):
            self.judgments=deepcopy(original)
            if change=='missing':self.judgments.pop()
            elif change=='duplicate':self.judgments[-1]=self.judgments[0]
            elif change=='bool':self.judgments[0]['dry_wit']=True
            elif change=='range':self.judgments[0]['dry_wit']=5
            elif change=='state':self.judgments[0]['fidelity']='pass'
            else:self.judgments[0]['evidence']=''
            self.bind()
            with self.subTest(change=change),self.assertRaises(ValueError):self.analyze()

    def test_hash_tampering_rejected(self):
        self.judgments[0]['evidence']='Changed'
        with self.assertRaises(ValueError):self.analyze()

    def test_key_split_leakage_and_group_mixing_rejected(self):
        original=deepcopy(self.key)
        for field,value in [('split','heldout'),('seed',9),('cell_index',5),('scene_id',self.key[-1]['scene_id'])]:
            self.key=deepcopy(original);self.key[0][field]=value;self.bind()
            with self.subTest(field=field),self.assertRaises(ValueError):self.analyze()

    def test_public_export_requires_authored_sources_and_no_local_paths(self):
        result=self.analyze()
        export=public_export(self.packet,self.key,self.judgments,self.provenance,result,self.scenes)
        self.assertEqual(len(export['answers']),252)
        changed=deepcopy(self.scenes);changed['evaluation'][0]['prompt']='Private book text'
        with self.assertRaises(ValueError):public_export(self.packet,self.key,self.judgments,self.provenance,result,changed)
        self.judgments[0]['evidence']='See /home/person/private'
        self.bind();result=self.analyze()
        with self.assertRaises(ValueError):public_export(self.packet,self.key,self.judgments,self.provenance,result,self.scenes)


if __name__=='__main__':unittest.main()
