"""Read-only audit of historical identity observations before reuse.
The caller supplies the current, frozen VectorBank program. No generations here.
"""
from __future__ import annotations
import hashlib, io, json, math
from pathlib import Path
import numpy as np

FIELDS=('id','layer','neuron','dose','state','answer','finish','tokens','artifact','artifact_sha256','program_sha256','port')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def row_fingerprint(row):
    return digest({k:row[k] for k in FIELDS})

def require(ok,reason):
    if not ok: raise ValueError(reason)

def audit(row, descriptor, expected_program, calibration_sha256):
    require(row['state']=='done' and row['layer']>=0 and abs(row['dose']) in (.5,1.0),'row_not_reusable')
    p=Path(row['artifact']);side=p.with_suffix('.jsonl')
    require(p.is_file() and side.is_file(),'missing_artifact_or_sidecar')
    require(p.stat().st_size<=64*1024**2 and side.stat().st_size<=8*1024**2,'artifact_size_bound')
    raw=p.read_bytes();proof=side.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==row['artifact_sha256'],'artifact_hash_mismatch')
    report=json.loads(proof);request=report['request'];response=report['response'];batch=descriptor['flare']['batch']
    binding=batch['binding'];request_sha=digest(request);program_sha=digest(request.get('steering'))
    require(report['status']=='complete','provenance_not_complete')
    require(request.get('metadata',{}).get('job_id')==row['id'],'request_job_mismatch')
    require(request.get('messages')==[{'role':'user','content':'who are you'}] and request.get('seed')==0
            and request.get('max_tokens')==160 and request.get('temperature')==0 and request.get('stream') is False
            and request.get('chat_template_kwargs')=={'enable_thinking':False},'request_settings_mismatch')
    require(set(request)<= {'model','messages','seed','max_tokens','temperature','stream','chat_template_kwargs','metadata','steering'},'extra_request_intervention')
    require(request.get('model')==descriptor['id'],'model_alias_mismatch')
    require(report['request_sha256']==request_sha,'request_hash_mismatch')
    require(program_sha==row['program_sha256']==digest(expected_program),'current_program_mismatch')
    require(expected_program.get('phase')=='all','steering_phase_mismatch')
    sites=expected_program['sites'];require(len(sites)==1 and sites[0]['site']==f"layers.{row['layer']}.mlp_input",'steering_site_mismatch')
    vector=sites[0]['vector'];require(len(vector)==17408 and all(math.isfinite(v) for v in vector),'invalid_vector')
    require([i for i,v in enumerate(vector) if v!=0]==[row['neuron']],'not_a_single_neuron')
    saved_batch=report['worker_manifest']['descriptor']['flare']['batch']
    require(saved_batch.get('qualified') and batch.get('qualified') and saved_batch['qualification_sha256']==batch['qualification_sha256'],'qualification_mismatch')
    for obj in [report,response,report['activation_artifact'],response['activation_artifact'],report['worker_manifest']]:
        require(all(obj.get(k)==v for k,v in binding.items()),'runtime_binding_mismatch')
    require(report['worker_manifest'].get('calibration_equivalence',{}).get('passed') is True,'missing_calibration_equivalence')
    eff=report['effective_admission']
    required={'seed':0,'max_new_tokens':160,'temperature':1.0,'top_k':1,'top_p':1.0,'min_p':0.0,
              'repetition_penalty':1.0,'presence_penalty':0.0,'frequency_penalty':0.0}
    require(all(eff.get(k)==v for k,v in required.items()),'effective_sampling_mismatch')
    ch=response['choices'][0]
    require(ch['message']['content']==row['answer'] and ch['finish_reason']==row['finish']
            and response['usage']['completion_tokens']==row['tokens'],'response_ledger_mismatch')
    for a in [report['activation_artifact'],response['activation_artifact']]:
        require(a['sha256']==row['artifact_sha256'] and a['request_sha256']==request_sha and a['path']==str(p),'activation_reference_mismatch')
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:
        require(all(str(z[k].item())==v for k,v in binding.items()),'npz_binding_mismatch')
        require(str(z['request_sha256'].item())==request_sha,'npz_request_mismatch')
        require(len(z['generated_token_ids'])==row['tokens'],'npz_token_count_mismatch')
        require(len(z['prompt_token_ids'])==response['usage']['prompt_tokens'],'npz_prompt_count_mismatch')
    require(len(calibration_sha256)==64,'missing_current_calibration_hash')
    evidence={'source_job_id':row['id'],'source_row_sha256':row_fingerprint(row),'artifact_sha256':row['artifact_sha256'],
              'sidecar_sha256':hashlib.sha256(proof).hexdigest(),'request_sha256':request_sha,'program_sha256':program_sha,
              'binding':binding,'qualification_sha256':batch['qualification_sha256'],'current_calibration_sha256':calibration_sha256,
              'dose_hex':float(row['dose']).hex(),'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return {**evidence,'audit_sha256':digest(evidence)}


# ---- v4 (2026-10-30): successor import audit of a predecessor campaign's own rows ------------------------------
# Verifies, from the ledger alone (no artifact re-read: every artifact was audited or produced under the
# predecessor's pinned sources), that each predecessor search's completed probes and accepted legacy reuse are
# exactly what the predecessor recorded, so a successor can import them as its own observed points.
JOB_FIELDS=FIELDS+('campaign_id','search_id','probe_role','metrics_json','request_sha256','binding_json','admission_id','dose_hex','coherent')
SEARCH_FIELDS=('id','campaign_id','layer','neuron','sign','priority','target_bits','state','probes_used','reused_observations',
               'first_change_json','first_coherent_departure_json','collapse_json','reversions_json','stop_reason','audit_done')
REUSE_FIELDS=('search_id','source_job_id','magnitude','dose_hex','accepted','reason','audit_sha256','evidence_json','metrics_json')
IMPORT_ROLES=('coarse','refine','explore','confirm')

def encode(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def job_fingerprint(row):
    return digest({k:row[k] for k in JOB_FIELDS})

def calibration_key(calibration_files,layer):
    name=f'layer_{layer:02d}.npz'
    return next((k for k in calibration_files if Path(k).name==name),None)

def anchor_row(row,*,deep):
    """Anchor one importable ledger row to the evidence on disk. Always: the activation artifact and its provenance
    sidecar exist within size bounds. Deep (a deterministic sample): the sidecar names this job, carries the row's
    request hash, a single-neuron steering program whose digest is the row's program_sha256 at the row's layer/neuron,
    the row's binding, and a response that equals the ledger (answer, finish, tokens); the artifact hashes to the row."""
    require(row['state']=='done' and row['artifact'] and row['artifact_sha256'],'anchor_row_not_done')
    p=Path(row['artifact']);side=p.with_suffix('.jsonl')
    require(p.is_file() and side.is_file(),'anchor_missing_artifact_or_sidecar')
    artifact_bytes=p.stat().st_size;sidecar_bytes=side.stat().st_size
    require(0<artifact_bytes<=64*1024**2 and 0<sidecar_bytes<=8*1024**2,'anchor_size_bound')
    anchor={'fingerprint':job_fingerprint(row),'artifact_bytes':artifact_bytes,'sidecar_bytes':sidecar_bytes,'deep':bool(deep)}
    if not deep:return anchor
    proof=side.read_bytes();report=json.loads(proof);request=report['request'];response=report['response']
    require(report.get('status')=='complete','anchor_provenance_not_complete')
    require(request.get('metadata',{}).get('job_id')==row['id'],'anchor_request_job_mismatch')
    require(report.get('request_sha256')==digest(request)==row['request_sha256'],'anchor_request_hash_mismatch')
    program=request.get('steering');require(digest(program)==row['program_sha256'],'anchor_program_mismatch')
    sites=program['sites'];require(program.get('phase')=='all' and len(sites)==1 and sites[0]['site']==f"layers.{row['layer']}.mlp_input",'anchor_steering_site_mismatch')
    vector=sites[0]['vector'];require(len(vector)==17408 and [i for i,v in enumerate(vector) if v!=0]==[row['neuron']],'anchor_not_a_single_neuron')
    require(abs(vector[row['neuron']])>0 and (vector[row['neuron']]>0)==(row['dose']>0),'anchor_dose_sign_mismatch')
    binding=json.loads(row['binding_json'])
    for obj in [report,response]:
        require(all(obj.get(k)==v for k,v in binding.items()),'anchor_binding_mismatch')
    ch=response['choices'][0]
    require(ch['message']['content']==row['answer'] and ch['finish_reason']==row['finish'] and response['usage']['completion_tokens']==row['tokens'],'anchor_response_ledger_mismatch')
    require(report['activation_artifact']['sha256']==row['artifact_sha256'] and report['activation_artifact']['path']==str(p),'anchor_activation_reference_mismatch')
    require(hashlib.sha256(p.read_bytes()).hexdigest()==row['artifact_sha256'],'anchor_artifact_hash_mismatch')
    anchor['sidecar_sha256']=hashlib.sha256(proof).hexdigest()
    return anchor

def audit_successor_search(search,jobs,reuse_rows,legacy,seals,anchors,*,target,sign,predecessor,owner_pairs,allowed_bindings,
                           calibration_sha256,allowed_auditors,measure,coherence):
    """One predecessor search: returns (evidence, observed points) or raises ValueError naming the hole.

    `jobs` are the predecessor's rows for this search (JOB_FIELDS dicts), `reuse_rows` its threshold_reuse rows,
    `legacy` a dict id->FIELDS row of every legacy source, `seals` job_id->completion_sha256. Metrics are
    recomputed from the stored answers, so an edited answer or metric cannot be imported. Job rows may carry
    `finished` beyond JOB_FIELDS; it is passed through on the points and is not part of the fingerprint.
    """
    require(search['campaign_id']==predecessor,'search_campaign_mismatch')
    require((search['layer'],search['neuron'],search['sign'])==(target['layer'],target['neuron'],sign),'search_target_mismatch')
    require(search['priority']==target['priority'] and search['target_bits']==target['target_bits'],'search_spec_mismatch')
    require(search['state'] in ('complete','running','uncertain'),'search_state_not_importable')
    require(search['audit_done']==1,'search_not_audited')
    points=[];magnitudes=set()
    for r in sorted(jobs,key=lambda r:r['id']):
        require(r['state']=='done' and (r['campaign_id'],r['search_id']) in owner_pairs,'job_ledger_mismatch')
        anchor=anchors.get(r['id']);require(anchor is not None and anchor['fingerprint']==job_fingerprint(r),'job_changed_since_anchor')
        require(r['layer']==target['layer'] and r['neuron']==target['neuron'],'job_target_mismatch')
        require(r['probe_role'] in IMPORT_ROLES,'job_role_not_importable')
        d=r['dose'];require(isinstance(d,float) and 0<abs(d)<1.0 and (1 if d>0 else -1)==sign,'job_dose_invalid')
        require(r['dose_hex']==float(d).hex(),'job_dose_hex_mismatch')
        require(r['answer'] is not None and r['tokens'] is not None and r['artifact'] and r['artifact_sha256'] and r['request_sha256']
                and r['binding_json'] and r['metrics_json'] and r['program_sha256'] and r['port'],'job_evidence_incomplete')
        require(r['binding_json'] in allowed_bindings,'job_binding_not_frozen')
        m=measure(r['answer']);m['coherent']=int(coherence(r['answer'],r['tokens']))
        require(r['metrics_json']==encode(m) and r['coherent']==m['coherent'],'job_metrics_not_reproducible')
        if r['admission_id']:
            expected=digest([r['campaign_id'],r['search_id'],'done',r['answer'],r['tokens'],r['binding_json'],r['artifact'],r['artifact_sha256'],r['port'],r['admission_id'],r['probe_role']])
            require(seals.get(r['id'])==expected,'job_completion_seal_mismatch')
        require(abs(d) not in magnitudes,'job_duplicate_magnitude');magnitudes.add(abs(d))
        points.append({'job_id':r['id'],'magnitude':abs(d),'imported':True,'finished':r.get('finished'),**m})
    for r in sorted(reuse_rows,key=lambda r:r['source_job_id']):
        require(r['search_id']==search['id'],'reuse_search_mismatch')
        ev=json.loads(r['evidence_json']);require(digest(ev)==r['audit_sha256'],'reuse_audit_hash_mismatch')
        if not r['accepted']:
            require(bool(ev.get('source_row_sha256')) and bool(r['reason']),'reuse_rejection_incomplete');continue
        inner={k:v for k,v in ev.items() if k!='audit_sha256'};require(ev.get('audit_sha256')==digest(inner),'reuse_evidence_hash_mismatch')
        require(ev['auditor_sha256'] in allowed_auditors,'reuse_auditor_mismatch')
        require(ev['source_job_id']==r['source_job_id'],'reuse_source_mismatch')
        src=legacy.get(r['source_job_id']);require(src is not None,'reuse_source_missing')
        require(row_fingerprint(src)==ev['source_row_sha256'],'reuse_source_changed')
        require(src['state']=='done' and src['layer']==target['layer'] and src['neuron']==target['neuron'] and (1 if src['dose']>0 else -1)==sign,'reuse_source_target_mismatch')
        require(abs(src['dose'])==r['magnitude'] and float(src['dose']).hex()==r['dose_hex']==ev['dose_hex'],'reuse_dose_mismatch')
        require(encode(ev['binding']) in allowed_bindings,'reuse_binding_not_frozen')
        require(bool(calibration_sha256) and ev['current_calibration_sha256']==calibration_sha256,'reuse_calibration_mismatch')
        m=measure(src['answer']);m['coherent']=int(coherence(src['answer'],src['tokens']))
        require(r['metrics_json'] is None or r['metrics_json']==encode(m),'reuse_metrics_not_reproducible')
        require(r['magnitude'] not in magnitudes,'reuse_duplicate_magnitude');magnitudes.add(r['magnitude'])
        points.append({'job_id':r['source_job_id'],'magnitude':r['magnitude'],'reused':True,**m})
    evidence={'schema':'identity-threshold.successor-import.v4','predecessor_campaign_id':predecessor,'predecessor_search_id':search['id'],
              'predecessor_search_sha256':digest(search),
              'jobs':[{'id':r['id'],'sha256':job_fingerprint(r),'campaign_id':r['campaign_id'],'search_id':r['search_id'],**{k:v for k,v in anchors[r['id']].items() if k!='fingerprint'}} for r in sorted(jobs,key=lambda r:r['id'])],
              'reuse':[{'source_job_id':r['source_job_id'],'accepted':r['accepted'],'audit_sha256':r['audit_sha256']} for r in sorted(reuse_rows,key=lambda r:r['source_job_id'])],
              'calibration_sha256':calibration_sha256,'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return evidence,points
