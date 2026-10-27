"""Public release boundaries; synthetic data only, no inference or transport."""
import persona
from copy import deepcopy
import unittest
from persona_response_release import public_only, project_answer


class ReleaseTests(unittest.TestCase):
    def test_local_paths_and_endpoints_fail_closed(self):
        for value in ['/home/person/private','/mnt/private','/tmp/run','C:\\private','http://127.0.0.1:8930']:
            with self.subTest(value=value),self.assertRaises(ValueError):public_only({'nested':[value]})
        self.assertEqual(public_only({'source':'https://huggingface.co/datasets/Atlas3D/character-creation-big-five'}),
                         {'source':'https://huggingface.co/datasets/Atlas3D/character-creation-big-five'})

    def test_answer_projection_omits_deployment_fields(self):
        capture=dict(request_sha256='a'*64,generated_token_ids_sha256='b'*64,consumed_generated_tokens=12)
        output=dict(answer='Synthetic answer',complete=True,response_capture=capture,
                    activation_artifact=dict(path='/mnt/private',sha256='c'*64),
                    provenance_artifact=dict(path='/home/private',sha256='d'*64))
        batch={'input':{'seed':7,'cells':[{'prompt':'Authored scene','_record':{'scene_id':'fixture','arm':'baseline','split':'development'}}]},
               'result':{'cells':[{'state':'complete','prompt':{'text':'Authored scene'},'output':output}]}}
        answer=public_only(project_answer(batch,0))
        self.assertEqual(answer['answer'],'Synthetic answer');self.assertEqual(answer['artifact_sha256'],'c'*64)
        bad=deepcopy(batch);bad['result']['cells'][0]['prompt']['text']='Wrong source'
        with self.assertRaises(ValueError):project_answer(bad,0)
        bad=deepcopy(batch);bad['result']['cells'][0]['output']['complete']=False
        with self.assertRaises(ValueError):project_answer(bad,0)


if __name__=='__main__':unittest.main()
