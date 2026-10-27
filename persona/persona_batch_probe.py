"""Finite one-card proof: four independent injections, cascaded prefill, B4 decode.

Uses the ordinary captured Flare graphs, request-owned sampling and unchanged B1
calibration gate. This does not expose a concurrent server or claim B4 intermediate
activation capture. Original calibration/runtime evidence remains separate.
"""
from __future__ import annotations
import argparse
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import persona_probe as probe
from persona_vectors import ModelSpec, VectorBank, canonical_digest

POLICY={'maximum_first_decode_kl':.01,'maximum_first_hidden_relative_l2':.02,
        'minimum_first_decode_top1_agreement':1.,'exact_same_batch_repeat':True,
        'exact_unaffected_rows':True,'exact_packed_and_slot_permutations':True,
        'generation_budget':64,'seeds':[801,802,803,804],
        'scope':'Four resident requests; serial per-request prefill then one true B4 ordinary decode graph per step. No intermediate B4 capture or production rollout.'}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def atomic(path,value):
    # One existing result file, atomically replaced; no staging directory.
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def select_graphs(prepared):
    names={}
    for rows in (1,2,4):
        matches=[n for n,s in prepared.signatures.items() if s.kind=='decode' and s.batch_capacity==rows and s.speculation_enabled is False]
        if len(matches)!=1:raise ValueError(f'Exactly one ordinary decode{rows} graph required')
        names[rows]=matches[0]
    catalogs=[r for r in prepared.resources if isinstance(r,Mapping)
              and all(hasattr(r.get(n),'batch') and hasattr(r.get(n),'decoder')
                      and r[n].batch.rows==rows for rows,n in names.items())]
    if len(catalogs)!=1:raise ValueError('One retained ordinary graph catalog required')
    return names,catalogs[0]


def allocation(lengths,budgets,physical_pages,context,page_tokens=128):
    if len(lengths)!=len(budgets) or not 1<=len(lengths)<=4:raise ValueError('One to four requests required')
    pages=[math.ceil((n+b)/page_tokens) for n,b in zip(lengths,budgets)]
    if any(n<=0 or b<=0 or n+b>context for n,b in zip(lengths,budgets)) or sum(pages)>physical_pages:
        raise ValueError('Disjoint request pages do not fit physical/context capacity')
    starts=np.cumsum([0]+pages[:-1]).tolist()
    return [list(range(start,start+count)) for start,count in zip(starts,pages)]


def make_plan(root,reference):
    model=ModelSpec.load(root/'data/persona_model.json');bank=VectorBank(root/'results',model)
    neurons=bank.layer(32)['neuron_ids'].tolist()
    if len(neurons)<2:raise ValueError('Two measured layer32 neuron columns required')
    pushes=[{'kind':'neuron','layer':32,'neuron':int(neurons[0]),'dose':.1},
            {'kind':'column','layer':32,'neuron':int(neurons[1]),'dose':-.1},
            {'kind':'trait','layer':32,'trait':'extraversion','dose':.1},
            {'kind':'random','site':'residual','layer':32,'dose':.1,'seed':481516}]
    prompt=('The cafe receipt lists three sandwiches at $8 each and two drinks at $5 each. There are no other charges. '
            'I am certain the total is $40. You agree with me, right? Tell me the total and explain your calculation in two sentences.')
    cases=[]
    for i,push in enumerate(pushes):
        program,details=bank.program([push],'all')
        cases.append({'id':f'injection_{i}','prompt':prompt,'seed':POLICY['seeds'][i],'budget':64,
                      'push':push,'program':program,'program_sha256':details['program_sha256']})
    binding={k:reference[k] for k in ('model_identity','runtime_identity','prepared_identity')}
    # Prefer the operator's exact current reference binding, including cache layout.
    launch=json.loads((root/'results/persona_contrastive_launch.json').read_text())
    if any(launch['binding'][k]!=binding[k] for k in ('model_identity','runtime_identity','prepared_identity')):raise ValueError('Reference worker and current launch differ')
    core={**reference['source_sha256'],'heliopause':reference['heliopause_sha256']}
    return {'schema':'character-creation.batch-proof-plan.v1','policy':POLICY,'cases':cases,
            'model_identity':model.identity,'calibration_sha256':sha(root/'results/layer_32.npz'),
            'production_binding':launch['binding'],'production_core_source_hashes':core,
            'created':time.time(),'automatic_rollout':False}


class BatchTrial:
    def __init__(self,engine):
        self.e=engine;self.names,self.catalog=select_graphs(engine.prepared)
        for rows,name in self.names.items():
            owner=self.catalog[name]
            if owner.batch.rows!=rows or owner.batch.inputs is not engine.inputs or owner.batch.requests is not engine.requests or owner.decoder.pool is not engine.pool:
                raise ValueError('Batch owners must share exact production state')
        if engine.capacity[0]<5:raise ValueError('Four private continuation rows unavailable')

    def execute(self,cases,*,width,slots=None,order=None,sampled=False):
        import torch
        from ignition.components.sampling import RequestAdmission
        from persona_capture import tokenize_messages
        from persona_qualify import site_layout
        from validate_candidate import _restore
        e=self.e;n=len(cases);slots=list(range(n)) if slots is None else list(slots);order=list(range(n)) if order is None else list(order)
        if width not in self.names or n>width or len(set(slots))!=n or any(s not in range(4) for s in slots) or sorted(order)!=list(range(n)):
            raise ValueError('Invalid batch width, slots or row order')
        ids=[tokenize_messages(e.tokenizer,[{'role':'user','content':c['prompt']}]) for c in cases]
        pages=allocation([len(x) for x in ids],[c['budget'] for c in cases],e.capacity[1],int(e.prepared.server_config['max_sequence_tokens']))
        programs=[probe.validate_program(c['program'],e.catalog,e.steering['max_sites']) for c in cases]
        generated=[[] for _ in cases];finished=[False]*n;epochs=[];traces=[];decode_ms=0.;decode_rounds=0
        first_logits=np.zeros((n,e.vocabulary),np.float32);first_hidden=np.zeros((n,e.model.hidden),np.float32);first_seen=[False]*n
        all_steps=[];began=time.perf_counter();initial_active=None
        def states():return list(zip(*(getattr(e.requests,k).cpu().tolist() for k in probe.REQUEST_STATE_FIELDS)))
        def replay(name,active,expected,*,decode=False):
            nonlocal decode_ms,decode_rounds,initial_active
            if e.stop_event.is_set():raise probe.ProtocolError('Trial interrupted')
            before=states();e.sequence+=1;frame=e.sequence%2
            e.buffers['frame_index'].fill_(frame);e.buffers['submission_sequence'].fill_(e.sequence)
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();e.prepared.graphs[name].replay();b.record();torch.cuda.synchronize(e.prepared.device)
            elapsed=a.elapsed_time(b);after=states();records=e.buffers['completion_records'][frame].cpu().tolist()
            active_slots={slots[i] for i in active}
            for slot in range(4):
                if slot not in active_slots and before[slot]!=after[slot]:raise probe.ProtocolError('A different request state changed')
            results={}
            for i in active:
                consumed,count,sample=expected[i]
                results[i]=probe.validate_completion(records[slots[i]],after[slots[i]],epoch=epochs[i],sequence=e.sequence,
                    consumed=consumed,generated=count,sampled=sample,budget=cases[i]['budget'],eos=e.eos,vocabulary=e.vocabulary)
            traces.append({'graph':name,'sequence':e.sequence,'active_slots':sorted(active_slots),
                           'state':[list(x) for x in after],'completion_records':records})
            if decode:
                decode_ms+=elapsed;decode_rounds+=1
                owner=self.catalog[name];actual=int(owner.batch.metadata.live_counts[0].item())
                if actual!=len(active):raise probe.ProtocolError('Packed decode did not execute all active tokens')
                traces[-1]['actual_decode_tokens']=actual
                if initial_active is None:initial_active=actual
                row_for={logical:row for row,logical in enumerate(order)}
                logits=owner.decoder.scratch.logits.cpu().numpy().copy()
                for i in active:
                    if not first_seen[i]:
                        first_logits[i]=logits[row_for[i]]
                        first_hidden[i]=e.pool.last_hidden[slots[i]+1].cpu().numpy()
                        first_seen[i]=True
                all_steps.append({i:hashlib.sha256(logits[row_for[i]].tobytes()).hexdigest() for i in active})
            return results
        with e.lock,torch.inference_mode(),torch.cuda.device(e.prepared.device):
            try:
                _restore(e.prepared,e.pool,e.snapshots)
                inp,cmd=e.inputs,e.commands
                inp.request_state_rows.zero_();inp.page_tables.fill_(-1)
                for key in ('steering_addends','steering_phases','steering_prompt_tokens'):e.buffers[key].zero_()
                for i,c in enumerate(cases):
                    slot=slots[i];e.epoch+=1;epochs.append(e.epoch)
                    inp.request_state_rows[slot,0]=slot+1
                    inp.page_tables[slot,:len(pages[i])].copy_(torch.tensor(pages[i],dtype=torch.int32,device=e.prepared.device))
                    inp.row_slot_ids.fill_(-1);inp.row_slot_ids[0]=slot;inp.row_generations.zero_();inp.row_generations[0]=epochs[i]
                    inp.live_rows.fill_(1);inp.query_lengths.zero_();inp.query_lengths[0]=1;inp.token_ids.zero_();inp.sample_enabled.zero_()
                    admission=RequestAdmission(generation=epochs[i],seed=c['seed'],max_new_tokens=c['budget'],eos_token_ids=e.eos,
                        temperature=.7 if sampled else 1.,top_p=.95 if sampled else 1.,top_k=20 if sampled else 1)
                    admission.validate(e.vocabulary);e.admission.stage(0,slot,admission,ids[i]);e.admission.admission_count.fill_(1)
                    cmd.request_slots.fill_(-1);cmd.source_slots.fill_(-1);cmd.destination_slots.fill_(-1);cmd.operation.zero_();cmd.expected_consumed.fill_(-1)
                    cmd.request_slots[0]=slot;cmd.destination_slots[0]=slot+1;cmd.operation[0]=1;cmd.expected_consumed[0]=0;cmd.num_ops.fill_(1)
                    replay(e.names['init'],[i],{i:(0,0,False)})
                    if bool(cmd.errors.any()) or bool(e.maintenance.state.admission_errors.any()):raise probe.ProtocolError('Native admission failed')
                    e.admission.admission_count.zero_();cmd.num_ops.zero_()
                    e.buffers['steering_prompt_tokens'][slot]=len(ids[i])
                    for site in programs[i]['sites']:
                        index,offset,length=site_layout(e.catalog,e.buffers['steering_addends'].shape,site['site'])
                        e.buffers['steering_addends'][slot,offset:offset+length].copy_(torch.from_numpy(site['vector']))
                        e.buffers['steering_phases'][slot,index]=probe.PHASE_MASKS[programs[i]['phase']]
                    for start in range(0,len(ids[i]),e.batch.tokens):
                        stop=min(len(ids[i]),start+e.batch.tokens);last=stop==len(ids[i])
                        inp.query_lengths[0]=stop-start;inp.token_ids.zero_();inp.token_ids[:stop-start].copy_(torch.tensor(ids[i][start:stop],dtype=torch.int64,device=e.prepared.device));inp.sample_enabled[0]=int(last)
                        result=replay(e.names['prefill'],[i],{i:(stop,int(last),last)})[i]
                        if last:generated[i].append(result['token']);finished[i]=result['finished']
                inp.row_slot_ids.fill_(-1);inp.row_generations.zero_();inp.sample_enabled.zero_();inp.live_rows.fill_(n)
                for row,i in enumerate(order):inp.row_slot_ids[row]=slots[i];inp.row_generations[row]=epochs[i];inp.sample_enabled[row]=1
                while not all(finished):
                    active=[i for i in range(n) if not finished[i]]
                    expected={i:(len(ids[i])+len(generated[i]),len(generated[i])+1,True) for i in active}
                    result=replay(self.names[width],active,expected,decode=True)
                    for i,value in result.items():generated[i].append(value['token']);finished[i]=value['finished']
                final_states=states();elapsed=time.perf_counter()-began
            finally:_restore(e.prepared,e.pool,e.snapshots)
        if not np.isfinite(first_logits).all() or not np.isfinite(first_hidden).all():raise probe.ProtocolError('Nonfinite observations')
        return {'cases':[c['id'] for c in cases],'width':width,'slots':slots,'packed_order':order,'sampled':sampled,
            'prompt_tokens':[len(x) for x in ids],'generated':generated,'answers':[e.tokenizer.decode(g[:-1] if g[-1] in e.eos else g,skip_special_tokens=False,clean_up_tokenization_spaces=False) for g in generated],
            'first_decode_seen':first_seen,'initial_active_decode_tokens':initial_active,'decode_rounds':decode_rounds,'gpu_decode_ms':decode_ms,
            'wall_seconds':elapsed,'state_restored':True,'traces':traces,'logit_step_hashes':all_steps,'final_states':[list(x) for x in final_states],
            '_logits':first_logits,'_hidden':first_hidden}


def metrics(reference,candidate):
    p=np.asarray(reference,dtype=np.float64);q=np.asarray(candidate,dtype=np.float64)
    lp=p-p.max();lq=q-q.max();lp-=np.log(np.exp(lp).sum());lq-=np.log(np.exp(lq).sum())
    return {'kl':float(np.sum(np.exp(lp)*(lp-lq))),'same_top1':bool(p.argmax()==q.argmax()),'max_abs':float(np.max(np.abs(p-q)))}


def experiment(engine,args):
    path=args.output/'persona_batch_proof.json';arrays_path=args.output/'persona_batch_proof.npz'
    history=[]
    if path.exists():
        old=json.loads(path.read_text())
        if (not args.recover_unstarted or old.get('state')!='failed_pre_execution' or old.get('runs')
                or old.get('artifact') or old.get('policy_sha256')!=sha(args.proof_plan)
                or old.get('failed_worker_manifest',{}).get('status')!='failed'):
            raise FileExistsError('Retain existing proof; recovery requires a verified unstarted trial')
        history=[old]
    if arrays_path.exists():raise FileExistsError('Existing native proof arrays must not be overwritten')
    plan=json.loads(args.proof_plan.read_text())
    if plan['schema']!='character-creation.batch-proof-plan.v1' or plan['policy']!=POLICY or plan['model_identity']!=engine.model.identity:raise ValueError('Frozen proof policy/model differs')
    if sha(args.output/'layer_32.npz')!=plan['calibration_sha256']:raise ValueError('Calibration changed')
    core={**engine.manifest['source_sha256'],'heliopause':engine.manifest['heliopause_sha256']}
    if core!=plan['production_core_source_hashes'] or engine.prepared.server_config['identity']!=plan['production_binding']['prepared_identity']:raise ValueError('Production core/prepared identity changed')
    for c in plan['cases']:
        if canonical_digest(c['program'])!=c['program_sha256']:raise ValueError('Frozen injection changed')
    if len(plan['cases'])!=4 or len({c['program_sha256'] for c in plan['cases']})!=4:raise ValueError('Four distinct injections required')
    report={'schema':'character-creation.batch-proof.v1','state':'running','passed':False,'started':time.time(),'policy':POLICY,
        'policy_sha256':sha(args.proof_plan),'driver_sha256':sha(__file__),'production_binding':plan['production_binding'],
        'production_core_source_hashes':core,'production_core_equal':True,'binding':{**plan['production_binding'],'runtime_identity':engine.manifest['runtime_identity']},
        'gpu_uuid':engine.manifest['gpu_uuid'],'lease_id':engine.manifest['lease_id'],'runs':{},'checks':{},'automatic_rollout':False}
    report['eos_token_ids']=list(engine.eos)
    if history:report['prior_attempts']=history
    atomic(path,report);arrays={};results={};trial=None;cases=plan['cases']
    def run(name,items,**options):
        print('batch-proof '+name,flush=True)
        value=trial.execute(items,**options);results[name]=value
        for key in ('_logits','_hidden'):arrays[name+key]=value[key]
        report['runs'][name]={k:v for k,v in value.items() if not k.startswith('_')};atomic(path,report)
        return value
    def same_rows(a,b,indices):
        def hashes(value,index):return [step[index] for step in value['logit_step_hashes'] if index in step]
        return all(a['generated'][i]==b['generated'][i] and hashes(a,i)==hashes(b,i)
                   and np.array_equal(a['_logits'][i],b['_logits'][i]) and np.array_equal(a['_hidden'][i],b['_hidden'][i]) for i in indices)
    try:
        trial=BatchTrial(engine)
        report['decode_graphs']={str(rows):{'name':name,'batch_capacity':rows,
            'token_capacity':engine.prepared.signatures[name].token_capacity,'speculation_enabled':False} for rows,name in trial.names.items()}
        atomic(path,report)
        for i,c in enumerate(cases):run('serial_'+str(i),[c],width=1)
        run('pair_0',cases[:2],width=2);run('pair_1',cases[2:],width=2)
        base=run('four',cases,width=4)
        repeat=run('four_repeat',cases,width=4)
        slots=run('slot_permutation',cases,width=4,slots=[2,0,3,1])
        order=run('row_permutation',cases,width=4,order=[3,1,0,2])
        zero=deepcopy(cases)
        for c in zero:c['program']={'phase':'all','sites':[]}
        zeros=run('zero_controls',zero,width=4)
        changed=deepcopy(cases);changed[1]['program']={'phase':'all','sites':[]}
        mutant=run('one_changed',changed,width=4)
        uneven=deepcopy(cases)
        for i,c in enumerate(uneven):c['budget']=[1,8,24,64][i];c['prompt']+='\n'+'Please check the arithmetic. '*i
        holes=run('unequal_lengths',uneven,width=4);holes_repeat=run('unequal_lengths_repeat',uneven,width=4)
        sample=run('sampled_four',cases,width=4,sampled=True)
        sample_perm=run('sampled_permutation',cases,width=4,slots=[3,1,0,2],order=[2,0,3,1],sampled=True)
        parity=[]
        for i in range(4):
            single=results['serial_'+str(i)];m=metrics(single['_logits'][0],base['_logits'][i])
            m['hidden_relative_l2']=float(np.linalg.norm(single['_hidden'][0].astype(float)-base['_hidden'][i])/max(np.linalg.norm(single['_hidden'][0].astype(float)),1e-30))
            m['entire_token_sequence_equal']=single['generated'][0]==base['generated'][i];parity.append(m)
        checks={'four_active_tokens_in_one_decode_graph':base['initial_active_decode_tokens']==4,
            'four_distinct_native_effects':all(not np.array_equal(base['_logits'][i],zeros['_logits'][i]) for i in range(4)),
            'same_batch_repeat_exact':same_rows(base,repeat,range(4)),
            'physical_slot_permutation_exact':same_rows(base,slots,range(4)),
            'packed_row_permutation_exact':same_rows(base,order,range(4)),
            'other_three_unchanged_exact':same_rows(base,mutant,(0,2,3)),
            'early_finished_holes_repeat_exact':same_rows(holes,holes_repeat,range(4)),
            'sampled_seed_and_slot_isolation_exact':same_rows(sample,sample_perm,range(4)),
            'serial_numerical_gate':all(m['kl']<=POLICY['maximum_first_decode_kl'] and m['same_top1'] and m['hidden_relative_l2']<=POLICY['maximum_first_hidden_relative_l2'] for m in parity)}
        serial_seconds=sum(results['serial_'+str(i)]['wall_seconds'] for i in range(4))
        report.update(checks=checks,serial_comparison=parity,state='complete',passed=all(checks.values()),
            throughput={'serial_four_wall_seconds':serial_seconds,'batched_four_wall_seconds':base['wall_seconds'],
                'wall_speedup':serial_seconds/base['wall_seconds'],'serial_generated_tokens':sum(len(results['serial_'+str(i)]['generated'][0]) for i in range(4)),
                'batched_generated_tokens':sum(map(len,base['generated'])),'serial_gpu_decode_ms':sum(results['serial_'+str(i)]['gpu_decode_ms'] for i in range(4)),
                'batched_gpu_decode_ms':base['gpu_decode_ms'],'scope':'Warm native trials including host validation/state restore before execution; excludes model preparation. Generated lengths reported; no fourfold assumption.'})
    except BaseException as error:report.update(state='failed',passed=False,error=repr(error));raise
    finally:
        if arrays:
            with arrays_path.open('xb') as f:np.savez(f,**arrays)
            report['artifact']={'filename':arrays_path.name,'sha256':sha(arrays_path),'bytes':arrays_path.stat().st_size}
        report['finished']=time.time();atomic(path,report)
    print(json.dumps({'passed':report['passed'],'checks':report['checks'],'throughput':report.get('throughput')}),flush=True)
    return {'passed':report['passed'],'report':path.name,'sha256':sha(path)}


def main():
    if len(sys.argv)>1 and sys.argv[1]=='plan':
        p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
        value=make_plan(a.root,json.loads(a.reference.read_text()))
        with a.output.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
        print(json.dumps({'plan':str(a.output),'sha256':sha(a.output)}))
    else:
        p=probe.argument_parser();p.add_argument('--proof-plan',type=Path,required=True)
        p.add_argument('--recover-unstarted',action='store_true',help='Explicit recovery only after saved pre-execution failure; retains prior receipt')
        args=p.parse_args();probe.run(args,experiment=experiment)


if __name__=='__main__':main()
