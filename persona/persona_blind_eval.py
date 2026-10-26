"""Bounded blinded-judge analysis; no generation, training or automatic promotion."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

APPROVED_SCENES_SHA256='861118a83d80a291ef4729cc2a871102a87dbf19451df4001e2032978fa26b36'
ARMS=('baseline','A','A_reverse','B','B_reverse','random')
FIELDS={'code','dry_wit','attentive_candor','fidelity','coherence','evidence'}
PROVENANCE={'judge_identity','method','packet_sha256','key_sha256','judgments_sha256'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate(packet,key,judgments,provenance):
    if packet.get('schema')!='character-creation.blind-persona-eval.v1':raise ValueError('Unknown blind packet')
    if set(provenance)!=PROVENANCE:raise ValueError('Exact judge provenance required')
    for field in ('judge_identity','method'):
        if not isinstance(provenance[field],str) or not 1<=len(provenance[field])<=256:raise ValueError('Judge identity/method required')
    for field,value in (('packet_sha256',packet),('key_sha256',key),('judgments_sha256',judgments)):
        if provenance[field]!=digest(value):raise ValueError('Judge input/output hash differs')
    if not isinstance(key,list) or not isinstance(judgments,list) or len(key)!=252 or len(judgments)!=252:
        raise ValueError('Exactly 252 non-replay answers required')
    codes={};groups={}
    if len(packet['groups'])!=42:raise ValueError('Exactly 42 matched groups required')
    for group in packet['groups']:
        if group['group'] in groups or len(group['answers'])!=6:raise ValueError('Duplicate/incomplete group')
        groups[group['group']]=group
        for answer in group['answers']:
            code=answer['code']
            if not isinstance(code,str) or not re.fullmatch('[0-9a-f]{12}',code) or code in codes or not isinstance(answer['answer'],str):
                raise ValueError('Invalid or duplicate answer code')
            codes[code]=(group,answer)
    indexed={};pairs={};scene_splits={};scene_seeds=defaultdict(set);group_keys=defaultdict(list)
    for row in key:
        code=row['code'];scene=row['scene_id'];seed=row['seed'];arm=row['arm'];split=row['split']
        if code not in codes or code in indexed or arm not in ARMS or split not in ('discovery','heldout'):
            raise ValueError('Invalid key coverage/arm/split')
        if not isinstance(scene,str) or not scene or type(seed) is not int:raise ValueError('Scene/seed invalid')
        if scene in scene_splits and scene_splits[scene]!=split:raise ValueError('Scene split leakage')
        scene_splits[scene]=split;scene_seeds[scene].add(seed)
        identity=(scene,seed,arm)
        if identity in pairs:raise ValueError('Duplicate matched observation')
        group,_=codes[code]
        if row['batch_index']!=group['group'] or type(row['cell_index']) is not int or not 0<=row['cell_index']<6:
            raise ValueError('Key/group mismatch')
        group_keys[group['group']].append(row)
        pairs[identity]=code;indexed[code]=row
    if Counter(scene_splits.values())!=Counter(discovery=12,heldout=9):raise ValueError('Expected 12 discovery and 9 heldout scenes')
    seed_sets={tuple(sorted(seeds)) for seeds in scene_seeds.values()}
    if len(seed_sets)!=1 or len(next(iter(seed_sets)))!=2:raise ValueError('Exactly two shared seeds required')
    for rows in group_keys.values():
        if (len({(r['scene_id'],r['seed']) for r in rows})!=1 or {r['arm'] for r in rows}!=set(ARMS)
                or {r['cell_index'] for r in rows}!=set(range(6))):raise ValueError('Incomplete or mixed matched group')
    for scene,seeds in scene_seeds.items():
        for seed in seeds:
            if any((scene,seed,arm) not in pairs for arm in ARMS):raise ValueError('Missing matched arm')
        scene_groups=[codes[pairs[scene,seed,'baseline']][0] for seed in sorted(seeds)]
        if any(g['prompt']!=scene_groups[0]['prompt'] or g['fact_contract']!=scene_groups[0]['fact_contract'] for g in scene_groups):
            raise ValueError('Scene source differs across seeds')
    scores={}
    for row in judgments:
        if not isinstance(row,dict) or set(row)!=FIELDS or row['code'] not in codes or row['code'] in scores:
            raise ValueError('Exact unique judgment coverage required')
        if any(type(row[f]) is not int or not 0<=row[f]<=4 for f in ('dry_wit','attentive_candor')):
            raise ValueError('Style scores must be integers 0..4')
        if any(row[f] not in ('ok','issue','uncertain') for f in ('fidelity','coherence')):raise ValueError('Invalid evidence state')
        if not isinstance(row['evidence'],str) or not row['evidence'].strip() or len(row['evidence'])>8000:raise ValueError('Bounded evidence required')
        scores[row['code']]=row
    return indexed,pairs,scene_splits,scene_seeds,scores


def interval(values,samples,seed):
    rng=random.Random(seed);n=len(values)
    means=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(samples))
    return {'mean':sum(values)/n,'ci95':[means[int(.025*(samples-1))],means[int(.975*(samples-1))]],'n_scenes':n}


def analyze(packet,key,judgments,provenance,*,bootstrap_samples=4000):
    if type(bootstrap_samples) is not int or not 200<=bootstrap_samples<=20000:raise ValueError('Bootstrap samples must be 200..20000')
    indexed,pairs,splits,seeds,scores=validate(packet,key,judgments,provenance)
    results={}
    metrics=('dry_wit','attentive_candor','fidelity_issue','fidelity_uncertain','coherence_issue','coherence_uncertain')
    def score(scene,seed,arm,metric):
        row=scores[pairs[scene,seed,arm]]
        if metric in ('dry_wit','attentive_candor'):return row[metric]
        field,state=metric.split('_');return int(row[field]==state)
    for split in ('discovery','heldout'):
        scenes=sorted(s for s in splits if splits[s]==split);contrasts={};absolute={}
        for arm in ARMS:
            absolute[arm]={metric:sum(score(s,seed,arm,metric) for s in scenes for seed in seeds[s])/(len(scenes)*2) for metric in metrics}
        comparisons=[(arm,control) for arm in ARMS[1:] for control in ('baseline','random') if arm!=control]+[('A','A_reverse'),('B','B_reverse')]
        for arm,control in comparisons:
            contrasts[arm+'_vs_'+control]={metric:interval([
                sum(score(s,seed,arm,metric)-score(s,seed,control,metric) for seed in seeds[s])/2 for s in scenes],bootstrap_samples,7035)
                for metric in metrics}
        results[split]={'n_scenes':len(scenes),'n_answers':len(scenes)*12,'absolute_means':absolute,'paired_contrasts':contrasts}
    proposals=[];discovery=results['discovery']
    for arm,target in (('A','dry_wit'),('B','attentive_candor'),('A_reverse','dry_wit'),('B_reverse','attentive_candor')):
        comparisons=[discovery['paired_contrasts'][arm+'_vs_'+control] for control in ('baseline','random')]
        sign=-1 if arm.endswith('_reverse') else 1
        style=all(c[target]['ci95'][0]>0 if sign>0 else c[target]['ci95'][1]<0 for c in comparisons)
        harm=any(c[m]['mean']>0 for c in comparisons for m in ('fidelity_issue','coherence_issue'))
        unknown=any(discovery['absolute_means'][a][m]>0 for a in (arm,'baseline','random') for m in ('fidelity_uncertain','coherence_uncertain'))
        eligible=style and not harm and not unknown
        proposals.append({'arm':arm,'target':target,'intended_change':'decrease' if sign<0 else 'increase',
            'action':'test_neighbor_doses' if eligible else 'do_not_promote',
            'proposed_doses':([sign*.05,sign*.15] if eligible else []),
            'direction':arm[0],'layer':32,'phase':'all','current_dose':sign*.1,
            'execution_status':'requires separate artifact/runtime binding and matched baseline/random trial admission',
            'evidence_split':'discovery','n_scenes':12,'status':'untested_proposal' if eligible else 'insufficient_discovery_evidence',
            'reason':{'directional_style_ci_vs_baseline_and_random':style,'observed_issue_excess':harm,'uncertain_evidence_present':unknown},
            'automatic_default_promotion':False})
    return {'schema':'character-creation.blind-analysis.v1','provenance':dict(provenance),
        'method':{'bootstrap':'paired scene-cluster percentile; average two seeds before resampling scenes','samples':bootstrap_samples,'seed':7035,
                  'intervals':'descriptive unadjusted 95%; multiple contrasts, not confirmatory significance'},
        'splits':results,'followup_proposals':proposals,'automatic_default_promotion':False,
        'limitations':['Model-judge evidence is not human review or calibrated truth.','No equivalence or safety established by zero observed issue excess.',
                      'Heldout results never select recipes or doses.','No training performed; propagation does not establish prose coherence.']}


def public_export(packet,key,judgments,provenance,analysis,authored_scenes):
    indexed,_,_,_,scores=validate(packet,key,judgments,provenance)
    if analysis['provenance']!=provenance:raise ValueError('Analysis binding differs')
    if (authored_scenes.get('schema')!='character-creation.contrastive-scenes.v1'
            or digest(authored_scenes)!=APPROVED_SCENES_SHA256):raise ValueError('Frozen approved authored source required')
    allowed={r['id']:r for r in authored_scenes['evaluation']}
    rows=[]
    for group in packet['groups']:
        for answer in group['answers']:
            keyrow=indexed[answer['code']];source=allowed.get(keyrow['scene_id'])
            if (source is None or source['split']!=keyrow['split'] or source['prompt']!=group['prompt']
                    or source['fact_contract']!=group['fact_contract']
                    or hashlib.sha256(source['prompt'].encode()).hexdigest()!=source['source_sha256']):raise ValueError('Public source is not approved authored scene')
            rows.append({'code':answer['code'],'scene_id':keyrow['scene_id'],'split':keyrow['split'],'seed':keyrow['seed'],
                         'arm':keyrow['arm'],'prompt':group['prompt'],'source_sha256':source['source_sha256'],
                         'answer':answer['answer'],'judgment':scores[answer['code']]})
    export={'schema':'character-creation.blind-public.v1','analysis':analysis,'answers':rows}
    # Fail rather than silently redact a quote or accidentally publish a local path.
    if re.search(r'/(?:home|mnt|tmp|Users|var)/|[A-Za-z]:\\',json.dumps(export)):
        raise ValueError('Local path appears in proposed public export')
    return export


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','key','judgments','provenance','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--public-output',type=Path);parser.add_argument('--scenes',type=Path)
    args=parser.parse_args();read=lambda p:json.loads(p.read_text())
    packet,key,judgments,provenance=(read(getattr(args,k)) for k in ('packet','key','judgments','provenance'))
    result=analyze(packet,key,judgments,provenance)
    public=None
    if args.public_output:
        if not args.scenes:parser.error('--public-output requires --scenes')
        public=public_export(packet,key,judgments,provenance,result,read(args.scenes))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if public is not None:args.public_output.write_text(json.dumps(public,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
