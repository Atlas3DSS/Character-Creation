"""One finite fresh-context follow-on cycle, gated behind the primary preset study.

No heldout ratings or failures influence challenger construction. Incumbents are
retained separately even when this cycle fails. Never launches GPU services.
"""
import argparse
from copy import deepcopy
import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import persona_preset_overnight as base
from persona_vectors import canonical_digest,ModelSpec,VectorBank
from persona_manager import Queue

SEEDS=(80411,80429)


def freeze(root,bank):
    root=Path(root);mp=root/'data/persona_perturbation_prompts_manifest.json';cp=root/'data/persona_perturbation_prompts.jsonl'
    manifest=json.loads(mp.read_text())
    base.require(manifest['status']=='complete' and manifest['records']>=3072 and manifest['output_sha256']==base.sha(cp),'Fresh corpus is not complete and hash-verified')
    rows=[json.loads(line) for line in cp.read_text().splitlines() if line.strip()]
    base.require(len(rows)==manifest['records'],'Fresh corpus row count differs')
    old=json.loads((root/'results/persona_preset_overnight_plan.json').read_text())
    selection=json.loads((root/'results/persona_preset_selection.json').read_text())
    base.require(selection['input_sha256']==old['input_sha256'],'Discovery selection input differs')
    excluded={c['pair_id'] for c in old['discovery']+old['confirmation']};excluded_text={c['prompt'] for c in old['discovery']+old['confirmation']}
    cases={}
    for split in ('discovery','heldout'):
        available=sorted((r for r in rows if r['split']==split),key=lambda r:canonical_digest(['followon_fixed',r['id']]))
        picked=[];domains=set()
        for row in available:
            if row['pair_id'] in excluded or row['prompt'] in excluded_text or row['category'] in domains:continue
            base.require(not row.get('system') and row['source']['kind']=='synthetic_original','Only neutral original authored prompts allowed')
            base.require(hashlib.sha256(row['prompt'].encode()).hexdigest()==row['source']['prompt_sha256'],'Fresh prompt hash differs')
            item=deepcopy(row);item['source_row_id']=row['id'];item['source_manifest_sha256']=base.sha(mp)
            item['fact_contract']={'source_text':row['prompt'],'policy':'Preserve explicitly supplied facts, quantities, roles, uncertainty and requested constraints; invention allowed only where the task invites fiction.'}
            item['max_tokens']=2048;item['thinking']=False;item['id']=canonical_digest({k:v for k,v in item.items() if k!='id'})
            picked.append(item);domains.add(row['category']);excluded.add(row['pair_id']);excluded_text.add(row['prompt'])
            if len(picked)==6:break
        base.require(len(picked)==6,'Need six disjoint source families per split');cases[split]=picked
    recipes=[]
    for target in base.TARGETS:
        evidence=[e for e in selection['evidence'] if e['target']==target]
        # This is discovery evidence only, including failures; no confirmation file is read here.
        best=max(evidence,key=lambda e:(e['clean_groups'],e['behavior_wins'],e['zero_equal_groups'],e['recipe_id']))
        original=next(r for r in old['recipes'] if r['id']==best['recipe_id'])
        for variant in ('repeat','distributed'):
            pushes=deepcopy(original['pushes'])
            if variant=='distributed':
                if all(p['kind']=='trait' for p in pushes):
                    totals={trait:sum(p['dose'] for p in pushes if p['trait']==trait) for trait in base.TRAITS}
                    layers=(16,24,32,40,44,48)
                    pushes=[{'kind':'trait','layer':layer,'trait':trait,'dose':dose/len(layers)} for trait,dose in sorted(totals.items()) for layer in layers if dose]
                    _,initial=bank.program(pushes,original['phase'],aggregate_limit=min(.35,original['aggregate_relative_norm']))
                    for p in pushes:p['dose']*=initial['aggregate_scale']
                else:
                    for p in pushes:p['dose']*=.85
            _,provenance=bank.program(pushes,original['phase'])
            recipes.append({**original,'id':target+'-followon-'+variant,'factor':1.0 if variant=='repeat' else .85,'pushes':pushes,
                'program_sha256':provenance['program_sha256'],'relative_norms':provenance['relative_norms'],'aggregate_relative_norm':provenance['aggregate_relative_norm'],
                'source_discovery_recipe':original['id'],'change':'Exact discovery-selected repeat on new contexts/seeds' if variant=='repeat' else 'Predeclared distributed calibrated trait layers, or85%observed neuron dose; no invented coordinates'})
    layers={p['layer'] for r in recipes for p in r['pushes']}
    plan={**old,'schema':'character-creation.preset-overnight.v1','cycle':'fresh_context_followon','seeds':list(SEEDS),
        'discovery':cases['discovery'],'confirmation':cases['heldout'],'recipes':recipes,
        'sources':{**old['sources'],'data/persona_perturbation_prompts.jsonl':base.sha(cp),'data/persona_perturbation_prompts_manifest.json':base.sha(mp),
            'results/persona_preset_selection.json':base.sha(root/'results/persona_preset_selection.json')},
        'calibration_sha256':{str(layer):base.sha(root/f'results/layer_{layer:02d}.npz') for layer in sorted(layers)},
        'incumbent_policy':'Primary qualified incumbents remain available if this cycle fails; no heldout-driven recipe tuning.',
        'maximum_jobs':1344,'implementation_sha256':{'followon':base.sha(__file__),'shared':base.sha(base.__file__)}}
    plan.pop('input_sha256');plan['input_sha256']=canonical_digest(plan);return plan


def jobs(plan,bank,stage,selected=None):
    result=base.jobs_for(plan,bank,stage,selected)
    for job in result:
        job['seed']=SEEDS[job['repetition']]
        for push in job['pushes']:
            if push['kind']=='random':push['seed']=job['seed']
        program,_=bank.program(job['pushes'],job['phase'])
        if job['arm']=='zero':program={'phase':program['phase'],'sites':[{'site':s['site'],'vector':[0.0]*len(s['vector'])} for s in program['sites']]}
        job['program_sha256']=canonical_digest(program);job.pop('id');job['id']=canonical_digest(job)
    return result


def run(root,until):
    import fcntl
    root=Path(root);base.require(time.time()<until<=1793113200,'Finite authorized overnight deadline required')
    lock=(root/'results/persona_preset_followon.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bank=base.attach_column(VectorBank(root/'results',ModelSpec.load(root/'data/persona_model.json')),root/'results/persona_preset_columns.json')
    plan_path=root/'results/persona_preset_followon_plan.json';status=root/'results/persona_preset_followon_status.json';child=None
    def persist(**values):base.atomic_json(status,{'schema':'character-creation.preset-followon-status.v1','deadline':until,'updated':time.time(),**values})
    try:
        while time.time()<until:
            mp=root/'data/persona_perturbation_prompts_manifest.json'
            if (root/'results/persona_preset_overnight_results.json').exists() and mp.exists():
                m=json.loads(mp.read_text())
                if m.get('status')=='complete' and m.get('records',0)>=3072:break
            persist(state='waiting',reason='Primary cycle and final fresh corpus must finish first');time.sleep(min(30,max(0,until-time.time())))
        if time.time()+1800>=until:persist(state='stopped',reason='Less than30minutes remain; no new work started');return
        plan=json.loads(plan_path.read_text()) if plan_path.exists() else freeze(root,bank)
        if not plan_path.exists():base.atomic_json(plan_path,plan)
        base.validate_plan(plan,bank)
        base.require(plan['implementation_sha256']=={'followon':base.sha(__file__),'shared':base.sha(base.__file__)},'Follow-on implementation changed after freeze')
        base.require(all(base.sha(root/k)==v for k,v in plan['sources'].items()),'Frozen follow-on source differs')
        corpus=root/'results/persona_preset_followon_corpus.jsonl';cm=root/'results/persona_preset_followon_corpus_manifest.json'
        data=''.join(json.dumps(c,sort_keys=True,separators=(',',':'))+'\n' for c in plan['discovery']+plan['confirmation'])
        if corpus.exists():base.require(corpus.read_text()==data,'Follow-on corpus changed')
        else:corpus.write_text(data)
        base.atomic_json(cm,{'status':'complete','output_sha256':base.sha(corpus),'input_sha256':plan['input_sha256']})
        selection=None
        for stage in ('discovery','confirmation'):
            selected=selection['selected'] if selection else None
            if stage=='confirmation' and not selected:break
            expected=jobs(plan,bank,stage,selected);database=root/f'results/persona_preset_followon_{stage}.sqlite'
            while not database.exists() and time.time()+900<until:
                gate=base.gate_config(root,until)
                if gate is None:persist(state='waiting_runtime_gate',stage=stage);time.sleep(15);continue
                config,finish,receipt=gate;config=deepcopy(config)
                config.setdefault('remote_column_source',{'path':str(root/'results/persona_preset_columns.json'),'sha256':base.sha(root/'results/persona_preset_columns.json')})
                config.update(max_total_jobs=len(expected),corpus=str(corpus),corpus_manifest=str(cm))
                cp=root/f'results/persona_preset_followon_{stage}_config.json';base.require(not cp.exists(),'Phase config already exists; explicit recovery required')
                base.atomic_json(cp,config);queue=Queue(database);queue.enqueue({'schema':'character-creation.fixed-preset-followon.v1','stage':stage,'input_sha256':plan['input_sha256']},expected,limit=len(expected))
                command=[sys.executable,str(root/'persona/persona_manager.py'),'run','--database',str(database),'--config',str(cp),'--until',datetime.datetime.fromtimestamp(finish,datetime.timezone.utc).isoformat()]
                base.atomic_json(root/f'results/persona_preset_followon_{stage}_launch.json',{'state':'intent','command':command,'gate':receipt,'input_sha256':plan['input_sha256'],'source_sha256':base.sha(__file__),'shared_source_sha256':base.sha(base.__file__),'manager_sha256':base.sha(root/'persona/persona_manager.py')})
                child=subprocess.Popen(command,cwd=root);break
            if not database.exists():persist(state='stopped',reason='No admitted runtime window');return
            terminal=0
            while time.time()<until:
                observations=base.saved_results([database],expected);groups={};reviews=[]
                for value in observations.values():
                    j=value['job'];groups.setdefault((j['recipe_id'],j['pair_id'],j['seed']),[]).append(value)
                terminal=sum(v['state'] in ('complete','incomplete','failed') for v in observations.values())
                for key,group in groups.items():
                    if len(group)!=(5 if stage=='discovery' else 4) or any(v['state'] in ('queued','running') for v in group) or time.time()+250>=until:continue
                    path=root/('results/persona_preset_followon_judge_'+canonical_digest(key)[:20]+'.json')
                    reviews.append(base.judge_group(plan,group,path))
                    persist(state='judging',stage=stage,terminal=terminal,total=len(expected),reviewed_groups=len(reviews),input_sha256=plan['input_sha256'])
                if terminal==len(expected):break
                if child and child.poll() is not None:persist(state='stopped',reason='Manager exited before all jobs terminal; no uncertain replay');return
                time.sleep(15)
            if terminal!=len(expected):persist(state='deadline',stage=stage,terminal=terminal);return
            if stage=='discovery':
                selection=base.select_candidates(plan,observations,reviews);base.atomic_json(root/'results/persona_preset_followon_selection.json',selection)
            else:
                report=base.confirmation_report(plan,selection,observations,reviews)
                report['incumbent_policy']='Primary qualified incumbents retained separately; failed challengers do not replace them.'
                base.atomic_json(root/'results/persona_preset_followon_results.json',report)
        persist(state='complete',input_sha256=plan['input_sha256'],maximum_additional_jobs=1344)
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=65)
            except subprocess.TimeoutExpired:child.kill();child.wait()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--until',type=float,required=True);a=p.parse_args();run(a.root,a.until)


if __name__=='__main__':main()
