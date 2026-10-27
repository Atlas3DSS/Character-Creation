import persona
import json
from pathlib import Path
import tempfile
import unittest
from persona_explorer import response_study_summary


class ResponseStudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'results').mkdir()
    def put(self,name,value):
        (self.root/'results'/name).write_text(json.dumps(value))
    def test_missing_is_explicit_not_fake_measurement(self):
        data=response_study_summary(self.root)
        self.assertEqual(data['pilot']['complete'],0)
        self.assertEqual(data['pilot']['expected'],256)
        self.assertEqual(data['development']['expected'],24)
        self.assertIsNone(data['analysis'])
        self.assertEqual(data['propagation']['measurements'],[])
    def test_projection_preserves_counts_and_excludes_private_fields(self):
        self.put('persona_contrastive_pilot.json',dict(state='running',private='/home/private',batches=[dict(index=0,state='running',batch_id='a'*32,
            input=dict(seed=7035,cells=[{'_record':dict(scene_id='authored_scene',split='heldout')}]),
            result=dict(cells=[dict(state='complete',output=dict(complete=True,answer='PRIVATE ANSWER',artifact='/mnt/private'))]))]))
        self.put('persona_propagation.json',dict(state='complete',results=[dict(context_id='scene',endpoint='/home/private',directions=[dict(direction_id='A',vector=[99],metrics=[dict(layer=32,lag=2,small_symmetric_derivative_l2=3.5,raw='/mnt/private')])])]))
        data=response_study_summary(self.root)
        self.assertEqual(data['pilot']['complete'],1)
        self.assertEqual(data['pilot']['batches'][0]['split'],'heldout')
        self.assertEqual(data['propagation']['measurements'][0]['metrics'][0]['small_symmetric_derivative_l2'],3.5)
        text=json.dumps(data)
        for forbidden in ('/home/','/mnt/','PRIVATE ANSWER','"vector"'):self.assertNotIn(forbidden,text)
    def test_analysis_separates_uncertain_and_issue(self):
        self.put('persona_pilot_analysis.json',dict(splits={'heldout':dict(n_scenes=9,paired_contrasts={'A_vs_baseline':dict(fidelity_issue=dict(mean=.1,ci95=[0,.2],n_scenes=9),fidelity_uncertain=dict(mean=.3,ci95=[.2,.4],n_scenes=9))})}))
        result=response_study_summary(self.root)['analysis']['heldout']['paired_contrasts']['A_vs_baseline']
        self.assertEqual(result['fidelity_issue']['mean'],.1)
        self.assertEqual(result['fidelity_uncertain']['mean'],.3)
    def test_repair_summary_counts_and_no_private_payload(self):
        self.put('persona_repair_analysis.json',dict(arms={'half':dict(n=7,fidelity={'issue':3,'ok':4},coherence={'ok':7})},private='/home/private'))
        result=response_study_summary(self.root)['repair']
        self.assertEqual(result['arms']['half']['fidelity']['issue'],3)
        self.assertEqual(result['arms']['half']['fidelity']['uncertain'],0)
        self.assertEqual(result['study']['expected'],42)
        self.assertNotIn('/home/private',json.dumps(result))
    def test_malformed_file_reports_unavailable(self):
        (self.root/'results/persona_contrastive_pilot.json').write_text('{')
        self.assertEqual(response_study_summary(self.root)['pilot']['state'],'unavailable')

if __name__=='__main__':unittest.main()
