"""Native batch scheduler contracts. No leases, model loading or implicit retries."""
from pathlib import Path
import json
import re
from persona_vectors import canonical_digest
from persona_propagation import file_sha

IDENTITIES=('model_identity','runtime_identity','prepared_identity')
UNSUPPORTED=('kv_control','capture_search','trace','token_trace','capture_trace','capture_prefill','observation_only','pulse','steering_pulse','prefill_stop')


def batch_eligible(request):
    """Unsupported controls retain their original payload for explicit B1 dispatch."""
    return not any(key in request for key in UNSUPPORTED) and request.get('stream') is False


def authorize_direction(provenance,pushes,target,receipt_spec=None):
    source=provenance.get('direction_binding')
    if source is None or all(source.get(k)==target.get(k) for k in IDENTITIES):return None
    if not receipt_spec:raise ValueError('Direction runtime differs; explicit compatibility receipt required')
    path=Path(receipt_spec['path']);digest=file_sha(path)
    if digest!=receipt_spec['sha256']:raise ValueError('Compatibility receipt bytes changed')
    receipt=json.loads(path.read_text())
    if (receipt.get('schema')!='character-creation.batch-runtime-compatibility.v1' or receipt.get('state')!='complete' or receipt.get('passed') is not True
        or any(receipt.get('old_binding',{}).get(k)!=source.get(k) or receipt.get('new_binding',{}).get(k)!=target.get(k) for k in IDENTITIES)
        or source['model_identity']!=target['model_identity']):raise ValueError('Compatibility source/target binding differs')
    if set(receipt.get('parity',{}))!={'programs_exact','response_tokens_exact','response_capture_exact'} or not all(v is True for v in receipt['parity'].values()):raise ValueError('Compatibility parity incomplete')
    if not re.fullmatch('[0-9a-f]{64}',receipt.get('qualification_sha256','')):raise ValueError('Qualification provenance absent')
    for name in ('old_core_source_hashes','new_core_source_hashes'):
        hashes=receipt.get(name)
        if not isinstance(hashes,dict) or not hashes or any(not re.fullmatch('[0-9a-f]{64}',str(v)) for v in hashes.values()):raise ValueError('Native core provenance absent')
    directions=[p for p in pushes if p.get('kind')=='direction']
    if not directions or any(p.get('sha256')!=receipt.get('artifact_sha256') for p in directions):raise ValueError('Direction artifact not qualified')
    programs=receipt.get('allowed_program_sha256')
    if not isinstance(programs,list) or not programs or any(not isinstance(p,str) or not re.fullmatch('[0-9a-f]{64}',p) for p in programs) or len(set(programs))!=len(programs):raise ValueError('Exact program allowlist absent/invalid')
    if provenance['program_sha256'] not in programs:raise ValueError('Exact program not qualified across runtimes')
    return digest


def checked_preflight(value,envelope,binding,capability):
    requests=envelope['requests'];n=len(requests)
    if (value.get('batch_id')!=envelope['batch_id'] or value.get('binding')!={k:binding[k] for k in IDENTITIES}
        or value.get('request_sha256')!=[canonical_digest(r) for r in requests]
        or type(value.get('fits')) is not bool or len(value.get('prompt_tokens',[]))!=n or len(value.get('required_pages',[]))!=n):raise ValueError('Preflight request/binding coverage differs')
    pages=capability.get('physical_pages');page_tokens=capability.get('page_tokens')
    if type(pages) is not int or pages<=0 or type(page_tokens) is not int or page_tokens<=0:raise ValueError('Physical capacity missing')
    available=value.get('available_pages')
    if type(available) is not int or not 0<=available<=pages:raise ValueError('Preflight available capacity differs')
    expected=[]
    for request,tokens in zip(requests,value['prompt_tokens']):
        budget=request['max_tokens']
        if type(tokens) is not int or tokens<=0 or type(budget) is not int or budget<=0:raise ValueError('Invalid authoritative token/budget counts')
        expected.append((tokens+budget+page_tokens-1)//page_tokens)
    if any(type(p) is not int for p in value['required_pages']) or value['required_pages']!=expected:raise ValueError('Full-budget page reservation differs')
    if value['fits'] and (sum(expected)>available or value.get('reason') is not None):raise ValueError('Preflight falsely claims fit')
    return value['fits']


def check_b1_controls(request,flare):
    for key in UNSUPPORTED:
        if key not in request:continue
        capability_name={'capture_search':'activation_search','capture_trace':'token_trace','capture_prefill':'prefill_instrumentation','steering_pulse':'prefill_instrumentation','prefill_stop':'prefill_instrumentation'}.get(key,key)
        capability=flare.get(capability_name,{})
        if not isinstance(capability,dict) or capability.get('supported') is not True:raise ValueError('Explicit B1 fallback lacks '+key+' capability')
        if key=='steering_pulse' and capability.get('one_position_pulse') is not True:raise ValueError('B1 pulse unsupported')
        if key=='prefill_stop' and capability.get('encoded_prefix_stop') is not True:raise ValueError('B1 prefix stop unsupported')


def validate_generation(response,request,binding,store,*,native_batch,validator):
    """Truthful full-budget truncation is incomplete science, not bad transport."""
    choice=(response.get('choices') or [{}])[0];diagnostics=response.get('diagnostics',{});usage=response.get('usage',{})
    limited=(native_batch and batch_eligible(request) and bool(request.get('capture_response'))
        and choice.get('finish_reason')=='length' and diagnostics.get('complete_final_answer') is False
        and diagnostics.get('unexpected_thinking') is False and type(usage.get('completion_tokens')) is int
        and type(request.get('max_tokens')) is int and usage['completion_tokens']==request['max_tokens'])
    if limited:
        from persona_batch_qualify import checked_parity
        verified=checked_parity(response,request,binding,store)
        evidence=verified['evidence']
        if evidence.get('finish_reason')!='length' or evidence.get('prose_complete') is not False or evidence.get('truncated_prose') is not True:
            raise ValueError('Parity validator did not establish truthful truncation')
        return {'complete':False,'generation_status':'length_limited','capture_validated':True}
    arrays,paths=validator(response,request,binding,store);del arrays
    return {'complete':True,'generation_status':'complete','capture_validated':True}


def dispatch(entries,endpoint,queue,store,*,fetch,deadline,admit,receipt=None,validator=None):
    """One endpoint owner; preflight may shrink, submitted batches never retry."""
    import time
    import uuid
    from persona_propagation import checked
    validator=checked if validator is None else validator
    binding={k:endpoint[k] for k in IDENTITIES}
    def call(path,payload=None):
        left=deadline-time.time()
        if left<=0:raise TimeoutError('Finite dispatch deadline reached')
        return fetch(endpoint['url']+path,payload,timeout=min(600,left))
    if not admit():raise ValueError('Lease/thermal admission absent')
    descriptor=call('/v1/models')['data'][0];flare=descriptor['flare']
    if any(flare.get(k)!=v for k,v in binding.items()):raise ValueError('Endpoint runtime binding changed')
    for entry in entries:
        if entry['request']['model']!=descriptor['id']:raise ValueError('Endpoint model alias changed')
        entry['compatibility_sha256']=authorize_direction(entry['provenance'],entry['job']['pushes'],binding,receipt)
    capability=flare.get('batch',{});native=(capability.get('supported') is True and capability.get('qualified') is True and capability.get('max_batch_size')==4)
    batch_id=uuid.uuid4().hex;selected=entries[:1];preflight=None
    if native and batch_eligible(entries[0]['request']):
        selected=[]
        for entry in entries[:4]:
            if not batch_eligible(entry['request']):break
            selected.append(entry)
        if capability.get('preflight_path')!='/v1/batch/preflight' or capability.get('completions_path')!='/v1/batch/completions':raise ValueError('Unrecognized native batch routes')
        while selected:
            envelope={'batch_id':batch_id,'requests':[e['request'] for e in selected]}
            preflight=call('/v1/batch/preflight',envelope)
            queue.audit('native_batch_preflight',{'batch_id':batch_id,'response':preflight})
            if checked_preflight(preflight,envelope,binding,capability):break
            selected=selected[:-1]
        if not selected:raise ValueError('Full-budget request cannot fit available native pages')
    else:
        native=False;check_b1_controls(selected[0]['request'],flare)
    released=entries[len(selected):]
    if released:queue.release_unsubmitted([(e['job'],e['number']) for e in released])
    if not admit():raise ValueError('Lease/thermal admission lost before submission')
    attempts=[{'job_id':e['job']['id'],'number':e['number'],'request':e['request'],'request_sha256':canonical_digest(e['request']),
        'program_sha256':e['provenance']['program_sha256'],'compatibility_sha256':e['compatibility_sha256']} for e in selected]
    queue.audit('native_batch_submitting' if native else 'native_b1_submitting',{'batch_id':batch_id,'endpoint':endpoint,'binding':binding,'attempts':attempts,
        'preflight':preflight,'fallback_reason':None if native else 'batch capability unqualified/unavailable or unsupported control; same-endpoint explicit B1'})
    try:
        if native:
            response=call('/v1/batch/completions',{'batch_id':batch_id,'requests':[e['request'] for e in selected]})
            if response.get('batch_id')!=batch_id or len(response.get('results',[]))!=len(selected):raise ValueError('Batch response coverage differs')
            rows=response['results']
            if [r.get('request_sha256') for r in rows]!=[a['request_sha256'] for a in attempts]:raise ValueError('Batch response order/request identities differ')
        else:rows=[{'request_sha256':attempts[0]['request_sha256'],'response':call('/v1/chat/completions',selected[0]['request'])}]
        queue.audit('native_batch_received',{'batch_id':batch_id,'rows':rows})
    except Exception as error:
        queue.audit('native_batch_uncertain',{'batch_id':batch_id,'error':repr(error),'retry':False})
        return [(e,{'complete':False,'error':repr(error),'error_class':'uncertain_no_retry','batch_id':batch_id}) for e in selected]
    outcomes=[]
    for entry,row in zip(selected,rows):
        try:
            status=validate_generation(row['response'],entry['request'],binding,store,native_batch=native,validator=validator)
            result={**status,'error':None,'response':row['response'],'request':entry['request'],'program_sha256':entry['provenance']['program_sha256'],
                'provenance':entry['provenance'],'runtime_identity':binding['runtime_identity'],'binding':binding,'batch_id':batch_id,
                'compatibility_sha256':entry['compatibility_sha256'],'activation_artifact':row['response']['activation_artifact'],
                'provenance_artifact':row['response']['provenance_artifact'],'promotion_eligible':False,'measurement':entry.get('measurement','rollout')}
            queue.audit('native_job_validated',{'batch_id':batch_id,'job_id':entry['job']['id'],'attempt':entry['number'],'request_sha256':row['request_sha256'],'complete_answer':status['complete'],'generation_status':status['generation_status'],'capture_validated':True})
        except Exception as error:result={'complete':False,'error':repr(error),'error_class':'artifact_validation_no_retry','batch_id':batch_id}
        outcomes.append((entry,result))
    return outcomes
