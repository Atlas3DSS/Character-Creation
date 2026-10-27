"""Build the public response-study index from verified, authored-only evidence.

Offline only: does not upload, infer, change source reports or read private corpora.
The independently verified capture-bundle receipt must exist before publication.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
from persona_response_bundle import load_inputs, sha
from persona_blind_eval import public_export, analyze as analyze_pilot
from persona_self_rating import analyze as analyze_ratings
from persona_repair import analyze_repair
from persona_personality_probes import load_suite


def public_only(value):
    text=json.dumps(value,ensure_ascii=False,allow_nan=False)
    if re.search(r'/(?:home|mnt|tmp|Users|var)/|[A-Za-z]:\\|https?://(?:localhost|127\.0\.0\.1)',text):
        raise ValueError('Local deployment data in proposed public release')
    return value


def project_answer(batch,index):
    spec=batch['input']['cells'][index];cell=batch['result']['cells'][index]
    out=cell['output'];capture=out['response_capture']
    if cell['state']!='complete' or out.get('complete') is not True or cell['prompt']['text']!=spec['prompt']:
        raise ValueError('Incomplete or mismatched answer')
    return dict(spec['_record'],seed=batch['input']['seed'],prompt=spec['prompt'],answer=out['answer'],
        request_sha256=capture['request_sha256'],artifact_sha256=out['activation_artifact']['sha256'],
        provenance_sha256=out['provenance_artifact']['sha256'],
        generated_token_ids_sha256=capture['generated_token_ids_sha256'],
        consumed_generated_tokens=capture['consumed_generated_tokens'])


def build(root):
    root=Path(root)
    def read(name):return json.loads((root/'results'/name).read_text())
    manifest=read('persona_response_capture_bundle_manifest.json')
    scenes,reports,binding,medians=load_inputs(root,manifest)
    packet=read('persona_pilot_blind.json');key=read('persona_pilot_blind_key.json')
    judgments=read('persona_pilot_judgments.json');provenance=read('persona_pilot_judge_provenance.json')
    analysis=read('persona_pilot_analysis.json')
    if analyze_pilot(packet,key,judgments,provenance)!=analysis:raise ValueError('Pilot analysis differs')
    public=public_export(packet,key,judgments,provenance,analysis,scenes)
    if public!=read('persona_pilot_public.json'):raise ValueError('Public pilot differs from validated export')
    receipt=read('persona_response_bundle_publication.json')
    # Publisher receipts may contain additional audit details; project only public fields.
    bundle=receipt.get('artifact',receipt)
    bundle_sha=bundle.get('sha256');bundle_bytes=bundle.get('bytes')
    if not isinstance(bundle_sha,str) or not re.fullmatch('[0-9a-f]{64}',bundle_sha) or not isinstance(bundle_bytes,int) or not 0<bundle_bytes<1024**3:
        raise ValueError('Verified bundle SHA/size required')
    if receipt.get('verified') is not True:raise ValueError('Unverified capture bundle')
    files={}
    probes=load_suite(root/'data/persona_personality_probes.json')
    for name,path in [('persona_contrastive_scenes.json','data/persona_contrastive_scenes.json'),
                      ('persona_personality_probes.json','data/persona_personality_probes.json'),
                      ('persona_pilot_public.json','results/persona_pilot_public.json'),
                      ('persona_repair_public.json','results/persona_repair_public.json'),
                      ('persona_response_directions.npz','results/persona_response_directions.npz')]:
        files[name]={'sha256':sha(root/path),'bytes':(root/path).stat().st_size}
    files['persona_response_capture_bundle.npz']={'sha256':bundle_sha,'bytes':bundle_bytes}
    development=[project_answer(b,i) for b in reports['development']['batches'] for i in range(len(b['input']['cells']))]
    replay=[project_answer(b,i) for b in reports['pilot']['batches'] for i,s in enumerate(b['input']['cells']) if 'replay_of' in s['_record']]
    ratings=read('persona_self_ratings.json')
    rating_analysis=analyze_ratings(ratings,packet,key,judgments,provenance)
    if rating_analysis!=read('persona_self_rating_analysis.json'):raise ValueError('Self-rating analysis differs')
    by_cell={(r['batch_index'],r['cell_index']):r for r in key}
    rating_rows=[]
    for result in ratings['results']:
        selected=ratings['selection'][result['index']];code=by_cell[selected['batch_index'],selected['cell_index']]['code']
        rating_rows.append(dict(code=code,status=result['status'],scores=result.get('scores'),
            request_sha256=result['record']['request_sha256']))
    propagation=reports['propagation']
    propagation_public={k:propagation[k] for k in ('binding','plan_sha256','qualification_sha256','context_consistency','interpretation')}
    propagation_public['results']=[{k:r[k] for k in ('context_id','zero_noise_max','directions')} for r in propagation['results']]
    propagation_public['design']={'contexts':[{'id':c['id'],'source_position':c['source_position']} for c in propagation['inputs']['contexts']],
        'layer':32,'snapshot_layers':[31,32,48,63],'token_lags':[0,1,2,4,8],'doses':[-.1,-.05,.05,.1],'n_requests':60}
    qualification=read('persona_capture_qualification.json')
    if qualification['binding']!=binding or not all(qualification.get(k) is True for k in ('passed','response_capture_passed','pulse_qualification_passed')):
        raise ValueError('Native qualification differs')
    qualification_public={'requests':len(qualification['records']),'workers':len(qualification['results']),
        'passed':True,'source_sha256':sha(root/'results/persona_capture_qualification.json'),
        'checks':[{k:r[k] for k in ('passed','response_capture_passed','response_capture','pulse')} for r in qualification['results']]}
    if qualification_public['requests']!=90 or qualification_public['workers']!=6:raise ValueError('Native qualification coverage differs')
    if qualification_public['source_sha256']!=reports['pilot']['qualification_sha256']:raise ValueError('Qualification source changed')
    repair_analysis,repair_public=analyze_repair(read('persona_repair_blind.json'),read('persona_repair_blind_key.json'),
        read('persona_repair_judgments.json'),read('persona_repair_judge_provenance.json'),read('persona_repair_pilot.json'),scenes,reports['pilot'])
    if repair_public!=read('persona_repair_public.json') or repair_analysis!=read('persona_repair_analysis.json'):
        raise ValueError('Repair export differs from frozen judgments and exact sources')
    contrasts=read('persona_response_contrasts.json')
    return public_only({'schema':'character-creation.response-study-public.v1','binding':binding,'model_id':reports['pilot']['model_id'],
        'files':files,'source_reports_sha256':{k:v['sha256'] for k,v in manifest['files'].items()},
        'policy':reports['pilot']['policy'],'styles':scenes['styles'],'counts':{'development':24,'pilot':256,'judged':252,'replays':4,'propagation':60,'self_ratings':48,'separate_repair_followup':42},
        'development_answers':development,'replay_answers':replay,'replay_checks':read('persona_pilot_replays.json'),
        'response_contrasts':{'metadata':contrasts['metadata'],'ranking':contrasts['ranking']},
        'qualification':qualification_public,'propagation':propagation_public,
        'self_rating':{'analysis':rating_analysis,'ratings':rating_rows},
        'repair_followup':{'analysis':repair_analysis,'activation_arrays_in_capture_bundle':False,
            'scope':'Seven outcome-selected discovery cases; separate42 answers, including14 baseline/reference replays; not heldout confirmation.'},
        'personality_probe_battery':{'file':'persona_personality_probes.json','suite_sha256':probes['suite_sha256'],
            'prompts':24,'families':12,'axes':[a['id'] for a in probes['axes']],'measured_in_this_release':False,
            'scope':'New authored shared-input discovery battery for neuronal recipe comparison; not part of the completed256-answer pilot.'},
        'capture_arrays':{'response_mean':{'shape':[280,64,5120],'dtype':'float64'},'response_count':{'shape':[280],'dtype':'int64'},
            'propagation_residual':{'shape':[60,5,4,5120],'dtype':'float32'},'calibration_residual_median':{'shape':[64],'dtype':'float64'}},
        'semantics':['Response means cover consumed generated tokens, excluding prompt/template and final sampled-but-unconsumed token.',
            'Development uses text style instructions; pilot uses unchanged scene prompts and residual steering at L32.',
            'Means are weighted equally by development scene, not by answer length. Bundle metadata supplies row joins and original hashes.',
            'Neuron ranking is direction/down-column alignment, not measured neuron activation or causal/semantic confidence.',
            'Replays verify reproducibility and are not independent scene evidence. Propagation measures context-dependent sensitivity, not wiring.',
            'The blinded model judge is fallible; bootstrap intervals are descriptive and unadjusted. No default or reward promotion.'],
        'privacy_scope':'Frozen authored scenes only; no private books or arbitrary user prompts.',
        'licenses':{'authored_prompts':'CC BY 4.0; Atlas3DSS Character Creation','derived_checkpoint_assets':'Subject to upstream checkpoint terms; not relicensed by this publication.'},
        'automatic_default_promotion':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=p.parse_args()
    value=build(args.root);path=args.root/'results/persona_response_study_public.json'
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'file':path.name,'sha256':sha(path),'bytes':path.stat().st_size}))


if __name__=='__main__':main()
