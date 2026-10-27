import persona
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from persona_contrastive import build_batches,input_digest
from persona_self_rating import select,rating_prompt,parse_rating,ranks,spearman,analyze,digest,base_request


class SelfRatingTests(unittest.TestCase):
    def test_tie_aware_rank_and_undefined_constant(self):
        self.assertEqual(ranks([3,1,1,4]),[3,1.5,1.5,4])
        self.assertAlmostEqual(spearman([0,0,1,2],[0,0,2,4]),1.)
        self.assertAlmostEqual(spearman([0,1,2],[2,1,0]),-1.)
        self.assertIsNone(spearman([9,9,9],[1,2,3]))

    def test_strict_json_scores_and_truncation(self):
        def response(text,complete=True):return {'choices':[{'message':{'content':text}}],'diagnostics':{'complete_final_answer':complete}}
        self.assertEqual(parse_rating(response('{"dry_wit":0,"attentive_candor":9}'))['scores'],{'dry_wit':0,'attentive_candor':9})
        for text in ('```json\n{}\n```','{"dry_wit":true,"attentive_candor":9}',
                     '{"dry_wit":10,"attentive_candor":9}','{"dry_wit":1,"dry_wit":2,"attentive_candor":9}',
                     '{"dry_wit":1,"attentive_candor":9,"extra":0}'):
            self.assertEqual(parse_rating(response(text))['status'],'invalid_json_or_schema')
        self.assertEqual(parse_rating(response('{"dry_wit":0,"attentive_candor":9}',False))['status'],'truncated_or_incomplete')

    def test_prompt_has_no_arm_or_external_judgment(self):
        row=dict(prompt='Synthetic scene',answer='Synthetic answer',arm='A_reverse',dose=-.1,external='secret judgment')
        text=rating_prompt(row)
        for forbidden in ('A_reverse','-0.1','secret judgment'):self.assertNotIn(forbidden,text)
        self.assertIn('Synthetic answer',text)

    def test_fixed_first_four_discovery_families_excludes_replays(self):
        root=Path(__file__).resolve().parents[1]
        scenes=json.loads((root/'data/persona_contrastive_scenes.json').read_text())
        binding={'model_identity':'a'*64,'runtime_identity':'b'*64,'prepared_identity':'c'*64}
        batches=build_batches(scenes,binding,'pilot',{'filename':'direction.npz','sha256':'d'*64})
        plan=dict(schema='character-creation.contrastive-campaign.v1',stage='pilot',binding=binding,model_id='test',
                  scene_sha256='e'*64,policy=scenes['policy'],direction_artifact={},qualification_sha256='f'*64,
                  state='complete',batches=[])
        for i,batch in enumerate(batches):
            plan['batches'].append(dict(index=i,input=batch,batch_id=str(i),result={'cells':[
                dict(index=j,output={'answer':'synthetic answer'}) for j in range(len(batch['cells']))]}))
        plan['input_sha256']=input_digest(plan)
        with patch('persona_self_rating.validate_result') as checked:
            rows=select(plan,scenes,None)
        self.assertEqual(len(rows),48);self.assertEqual(checked.call_count,8)
        self.assertEqual({r['scene_id'] for r in rows},{r['id'] for r in scenes['evaluation'] if r['split']=='discovery'} & {r['id'] for r in scenes['evaluation'][:4]})
        self.assertTrue(all(r['batch_index']<42 for r in rows))
        changed=deepcopy(plan);changed['state']='running'
        with self.assertRaises(ValueError):select(changed,scenes,None)
        changed=deepcopy(scenes);changed['evaluation'][0]['prompt']='private source'
        with self.assertRaises(ValueError):select(plan,changed,None)


    def test_analysis_joins_exact_answers_and_reports_ceiling(self):
        from test_persona_blind_eval import BlindTests
        fixture=BlindTests();fixture.setUp()
        selected={r['id'] for r in fixture.scenes['evaluation'] if r['split']=='discovery'}
        selected=set(list(dict.fromkeys(r['scene_id'] for r in fixture.key if r['scene_id'] in selected))[:4])
        rows=[];results=[]
        for k in fixture.key:
            if k['scene_id'] not in selected:continue
            scene=next(r for r in fixture.scenes['evaluation'] if r['id']==k['scene_id'])
            row={f:k[f] for f in ('scene_id','seed','arm','batch_index','cell_index')}
            row.update(prompt=scene['prompt'],answer='Synthetic answer.',answer_sha256=digest('Synthetic answer.'),source_sha256=scene['source_sha256'])
            request=base_request({'id':'test'},rating_prompt(row),seed=7035);request['max_tokens']=2048
            response={'choices':[{'message':{'content':'{"dry_wit":9,"attentive_candor":9}'}}],'diagnostics':{'complete_final_answer':True}}
            results.append(dict(index=len(rows),response=response,record={'request_sha256':digest(request)},**parse_rating(response)));rows.append(row)
        report=dict(state='complete',model_id='test',selection=rows,selection_sha256=digest(rows),results=results)
        report['report_sha256']=digest(report)
        output=analyze(report,fixture.packet,fixture.key,fixture.judgments,fixture.provenance)
        self.assertEqual(output['n_valid'],48)
        self.assertEqual(output['metrics']['dry_wit']['ceiling_fraction'],1.)
        self.assertIsNone(output['metrics']['dry_wit']['spearman'])
        report['results'][0]['record']['request_sha256']='0'*64
        report['report_sha256']=digest({k:v for k,v in report.items() if k!='report_sha256'})
        with self.assertRaises(ValueError):analyze(report,fixture.packet,fixture.key,fixture.judgments,fixture.provenance)


if __name__=='__main__':unittest.main()
