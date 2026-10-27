"""Offline independent validation of physical-slot B1/B2/B4 capture evidence."""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from persona_batch_validate import require,sha,configurations,run_trace,histories,compare_summary
from persona_vectors import canonical_digest

PLANES=('residual_last','pre_mlp_last','mlp_input_last')
MAX_BYTES=768*1024**2
SUFFIXES=('_logits','_hidden','_count','_sum','_mean',*('_first_'+p for p in PLANES))
def digest(a):return hashlib.sha256(np.ascontiguousarray(a,dtype='<f4').tobytes()).hexdigest()
def hash_ok(s):return isinstance(s,str) and re.fullmatch('[0-9a-f]{64}',s) is not None


def validate_capture(name,run,arrays,layers,hidden,intermediate):
    c=run['capture'];n=len(run['cases']);widths=dict(residual_last=hidden,pre_mlp_last=hidden,mlp_input_last=intermediate)
    count=arrays[name+'_count'];sums=arrays[name+'_sum'];means=arrays[name+'_mean']
    require(count.shape==(n,) and count.dtype==np.int64,'Count shape/dtype')
    require(sums.shape==means.shape==(n,layers,hidden) and sums.dtype==means.dtype==np.float64,'FP64 sum/mean shape/dtype')
    require(np.isfinite(sums).all() and np.isfinite(means).all(),'Nonfinite sum/mean')
    require(count.tolist()==c['counts']==[len(g)-1 for g in run['generated']] and c['mean_defined']==(count>0).tolist(),'Consumed counts differ')
    expected=np.divide(sums,count[:,None,None],out=np.zeros_like(sums),where=count[:,None,None]>0)
    require(np.array_equal(means,expected),'Mean does not equal sum/count')
    require(run['capture_restored'] is True and run['graph_wrappers_restored'] is True,'Restoration missing')
    for key in ('consumed_token_ids','positions','generations','step_plane_sha256'):require(len(c[key])==n,'Capture logical coverage')
    for p,w in widths.items():
        a=arrays[name+'_first_'+p];require(a.shape==(n,layers,w) and a.dtype==np.float32 and np.isfinite(a).all(),'First plane shape/dtype/finite')
    for i in range(n):
        require(c['consumed_token_ids'][i]==run['generated'][i][:-1] and c['positions'][i]==list(range(run['prompt_tokens'][i],run['prompt_tokens'][i]+int(count[i]))),'Consumed token/position coverage')
        hashes=c['step_plane_sha256'][i];require(len(hashes)==count[i] and all(set(h)==set(PLANES) and all(hash_ok(v) for v in h.values()) for h in hashes),'Per-token plane hashes incomplete')
        for p in PLANES:
            a=arrays[name+'_first_'+p][i]
            require(digest(a)==hashes[0][p] if count[i] else not a.any(),'First raw plane/hash differs')
        if not count[i]:require(c['generations'][i] is None and not sums[i].any(),'Empty capture has data')
        elif count[i]==1:require(np.array_equal(sums[i],arrays[name+'_first_residual_last'][i].astype(np.float64)),'Single-token sum differs')
    previous=[[0,0,0,-1,0,0,0] for _ in range(4)];steps=[]
    for t in run['traces']:
        if t['graph'].startswith('decode'):steps.append((previous,t))
        previous=t['state']
    require(len(steps)==len(c['mapping_steps']),'Mapping decode coverage')
    seen=[0]*n;slot_to_i={s:i for i,s in enumerate(run['slots'])}
    for (before,t),m in zip(steps,c['mapping_steps']):
        require(m['before']==before and m['width']==run['width'],'Mapping pre-state/width differs')
        active=set(t['active_slots'])
        if run['width']==1:
            s=next(iter(active));require(m['slot']==s and m['row']==0,'B1 packed row/slot')
            require((m['generation'],m['position'],m['token_id'])==(before[s][0],before[s][1],before[s][3]),'B1 input evidence')
        else:
            e=m['evidence'];w=run['width']
            for k in ('slot_live','slot_positions','slot_token_ids','slot_generations','slot_match_counts','slot_last_index'):require(len(e[k])==4,'Physical evidence shape')
            for k in ('token_rows','token_slots','token_positions','token_ids','token_generations','token_live','invalid_tokens'):require(len(e[k])==w,'Packed evidence shape')
            require(e['mapping_error']==[False] and not any(e['invalid_tokens']),'Mapping error')
            require({s for s,v in enumerate(e['slot_live']) if v}==active,'Physical live slots')
            live=[j for j,v in enumerate(e['token_live']) if v]
            require(len(live)==len(active) and {e['token_slots'][j] for j in live}==active,'Packed live slots')
            for j in live:
                s=e['token_slots'][j];row=e['token_rows'][j]
                require(0<=row<len(run['packed_order']) and run['slots'][run['packed_order'][row]]==s,'Packed row ownership')
                require((e['token_generations'][j],e['token_positions'][j],e['token_ids'][j])==(before[s][0],before[s][1],before[s][3]),'Packed input evidence')
                require(e['slot_last_index'][s]==j,'Physical gather index')
            inactive=set(range(4))-active
            require(set(m['inactive_plane_sha256'])=={str(s) for s in inactive},'Inactive evidence coverage')
            for s in range(4):
                if s in active:
                    require((e['slot_generations'][s],e['slot_positions'][s],e['slot_token_ids'][s],e['slot_match_counts'][s])==(before[s][0],before[s][1],before[s][3],1),'Physical input evidence')
                else:
                    require((e['slot_generations'][s],e['slot_positions'][s],e['slot_token_ids'][s],e['slot_match_counts'][s])==(0,-1,-1,0),'Inactive evidence not cleared')
                    require(m['inactive_plane_sha256'][str(s)]=={p:digest(np.zeros((layers,wid),np.float32)) for p,wid in widths.items()},'Inactive planes not zero')
        for s in active:
            i=slot_to_i[s];k=seen[i]
            require(c['generations'][i]==before[s][0] and c['positions'][i][k]==before[s][1] and c['consumed_token_ids'][i][k]==before[s][3],'Captured logical input differs')
            seen[i]+=1
    require(seen==c['counts'],'Mapping counts differ')


def recompute(report,plan,arrays,*,layers=64,hidden=5120,intermediate=17408):
    require(plan['schema']=='character-creation.batch-capture-plan.v1' and report['schema']=='character-creation.batch-capture-proof.v1','Schemas')
    require(report['state']=='complete' and report['automatic_rollout'] is False and report['policy']==plan['policy'],'Incomplete/changed policy')
    require(plan['policy']['budget']==32 and plan['policy']['unequal_budgets']==[1,3,8,32] and plan['policy']['exact_capture_equality'] is True,'Protocol changed')
    for key in ('adapter_sha256','probe_source_sha256','batch_trial_source_sha256','legacy_capture_prefix_sha256'):require(report[key]==plan[key] and hash_ok(plan[key]),'Source binding '+key)
    core=dict(plan['production_core_source_hashes']);require(core['ignition/components/activation_capture.py']==plan['legacy_capture_prefix_sha256'],'Legacy prefix changed')
    core['ignition/components/activation_capture.py']=plan['capture_source_sha256']
    require({**report['source_sha256'],'heliopause':report['heliopause_sha256']}==core and report['capture_override']['sha256']==plan['capture_source_sha256'],'Core binding differs')
    require(report['binding']['model_identity']==plan['model_identity']==plan['production_binding']['model_identity'] and report['binding']['prepared_identity']==plan['production_binding']['prepared_identity'],'Model/prepared binding differs')
    require(all(hash_ok(v) for v in report['binding'].values()),'Malformed binding')
    cases=plan['cases'];require(len(cases)==4 and len({c['program_sha256'] for c in cases})==4 and len({c['prompt'] for c in cases})==1,'Four distinct same-prompt programs')
    for c in cases:require(c['budget']==32 and canonical_digest(c['program'])==c['program_sha256'] and any(np.any(s['vector']) for s in c['program']['sites']),'Program binding/nonzero')
    runs=report['runs'];require(set(runs)==set(configurations()) and set(arrays)=={n+s for n in runs for s in SUFFIXES},'Run/array coverage')
    graphs=report['decode_graphs'];require(set(graphs)=={'1','2','4'},'Graph coverage')
    for w in (1,2,4):require(graphs[str(w)]['batch_capacity']==w and graphs[str(w)]['token_capacity']==1 and graphs[str(w)]['speculation_enabled'] is False,'Ordinary graph signature')
    traces={}
    for n,r in runs.items():
        cs=deepcopy(cases);alias=n;view=arrays
        if n.startswith('unequal_lengths'):
            alias='four';view={'four_logits':arrays[n+'_logits'],'four_hidden':arrays[n+'_hidden']}
            for c,b in zip(cs,plan['policy']['unequal_budgets']):c['budget']=b
        traces[n]=run_trace(alias,r,cs,view,set(report['eos_token_ids']),hidden,graphs)
        validate_capture(n,r,arrays,layers,hidden,intermediate)
    def equal(a,b,i,j):
        x,y=runs[a],runs[b]
        return (x['generated'][i]==y['generated'][j] and histories(x,i)==histories(y,j)
            and all(x['capture'][k][i]==y['capture'][k][j] for k in ('consumed_token_ids','positions','step_plane_sha256'))
            and all(np.array_equal(arrays[a+s][i],arrays[b+s][j]) for s in SUFFIXES))
    checks={'serial_B4_exact':all(equal('serial_'+str(i),'four',0,i) for i in range(4)),
        'serial_B2_exact':all(equal('serial_'+str(i),'pair_0' if i<2 else 'pair_1',0,i%2) for i in range(4)),
        'four_distinct_native_effects':all(not np.array_equal(arrays['four_logits'][i],arrays['zero_controls_logits'][i]) for i in range(4)),
        'changed_row_effect':not np.array_equal(arrays['four_logits'][1],arrays['one_changed_logits'][1]),
        'repeat_exact':all(equal('four','four_repeat',i,i) for i in range(4)),
        'slot_permutation_exact':all(equal('four','slot_permutation',i,i) for i in range(4)),
        'row_permutation_exact':all(equal('four','row_permutation',i,i) for i in range(4)),
        'unaffected_rows_exact':all(equal('four','one_changed',i,i) for i in (0,2,3)),
        'unequal_lengths_exact':all(equal('unequal_lengths','unequal_lengths_repeat',i,i) for i in range(4)),
        'sampled_permutation_exact':all(equal('sampled_four','sampled_permutation',i,i) for i in range(4))}
    compare_summary(report['checks'],checks);require(report['passed'] is all(checks.values()),'Passed summary differs')
    return {'verified':True,'passed':all(checks.values()),'checks':checks,'trace_validation':traces,'consumed_observations':sum(sum(r['capture']['counts']) for r in runs.values()),'limitations':['Non-first per-token vectors are hashes; their sums cannot be independently reconstructed.','Timing includes synchronous capture and host readback; no capture-off speed claim.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--original-plan',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--driver',type=Path,required=True);p.add_argument('--capture-source',type=Path,required=True);p.add_argument('--probe',type=Path,required=True);p.add_argument('--batch-driver',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(sha(a.plan)==a.plan_sha256,'Frozen plan SHA');plan=json.loads(a.plan.read_text());report=json.loads(a.report.read_text())
    original=json.loads(a.original_plan.read_text())
    require(canonical_digest(original)==plan['original_plan_canonical_sha256'],'Original plan lineage')
    original_cases=deepcopy(original['cases'])
    for case in original_cases:case['budget']=32
    require(plan['cases']==original_cases,'Original case/program lineage')
    for key in ('production_binding','production_core_source_hashes','model_identity','calibration_sha256'):require(plan[key]==original[key],'Original binding lineage '+key)
    require(plan['policy']['maximum_artifact_bytes']==MAX_BYTES,'Artifact bound changed')
    require(report['policy_sha256']==a.plan_sha256 and report['driver_sha256']==sha(a.driver)==plan['adapter_sha256'],'Driver/plan binding')
    require(sha(a.capture_source)==plan['capture_source_sha256'] and sha(a.probe)==plan['probe_source_sha256'] and sha(a.batch_driver)==plan['batch_trial_source_sha256'],'Source files changed')
    content=a.capture_source.read_bytes();boundary=content.index(b'@dataclass(frozen=True)\nclass RequestActivationCapture');require(hashlib.sha256(content[:boundary-2]).hexdigest()==plan['legacy_capture_prefix_sha256'],'Legacy source changed')
    art=report['artifact'];require(Path(art['filename']).name==art['filename'],'Artifact must be flat');path=a.report.parent/art['filename']
    require(path.stat().st_size==art['bytes']<=plan['policy']['maximum_artifact_bytes'] and sha(path)==art['sha256'],'Artifact size/hash')
    with np.load(path,allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files};require(sum(v.nbytes for v in arrays.values())==art['uncompressed_array_bytes']<=plan['policy']['maximum_artifact_bytes'],'Uncompressed budget')
        receipt=recompute(report,plan,arrays)
    receipt.update(schema='character-creation.batch-capture-validation.v1',report_sha256=sha(a.report),plan_sha256=sha(a.plan),artifact_sha256=sha(path),validator_sha256=sha(__file__))
    with a.output.open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps({'verified':receipt['verified'],'passed':receipt['passed']}))
if __name__=='__main__':main()
