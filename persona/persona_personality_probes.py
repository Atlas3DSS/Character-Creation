"""Source-bound authored personality probes; descriptive style, separate fidelity.

No model calls, scoring defaults, persona prompts, or changes to frozen studies.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from persona_vectors import canonical_digest

SCHEMA='character-creation.personality-probes.v1'
AXES={'factual_correction','social_pressure','criticism_response','reassurance','interpersonal_conflict','uncertainty_under_pressure'}


def validate_suite(suite):
    if not isinstance(suite,dict) or suite.get('schema')!=SCHEMA or suite.get('version')!=1:raise ValueError('Unknown personality probe schema')
    policy=suite.get('policy',{})
    required={'comparison':'neuron_or_residual_recipe_only','system':'','kv_control':False,'persona_instruction':False,'split':'discovery','max_tokens':2048,'thinking':False}
    if any(policy.get(k)!=v or type(policy.get(k)) is not type(v) for k,v in required.items()):raise ValueError('Probe policy must preserve neuron-only shared prompts')
    axes=suite.get('axes',[]);cases=suite.get('cases',[])
    if len(axes)!=6 or {a['id'] for a in axes}!=AXES or len(cases)!=24:raise ValueError('Expected six axes and 24 authored probes')
    if Counter(c['axis'] for c in cases)!=Counter({a:4 for a in AXES}):raise ValueError('Probe axes must each contain four cases')
    pairs=defaultdict(list);ids=set();prompts=set()
    for c in cases:
        if set(c)!={'id','axis','pair_id','pressure','title','prompt','source_sha256','fact_contract','observations','tags'}:raise ValueError('Unexpected probe fields, including possible control injection')
        text=c['prompt']
        if not isinstance(text,str) or not 80<=len(text)<=1600 or text in prompts or c['id'] in ids or c['id']!=c['pair_id']+'_'+c['pressure']:raise ValueError('Probe IDs/text must be distinct and source-bound')
        if c['pressure'] not in ('lower','higher') or hashlib.sha256(text.encode()).hexdigest()!=c['source_sha256']:raise ValueError('Probe pressure or source hash differs')
        ids.add(c['id']);prompts.add(text);pairs[c['pair_id']].append(c)
        contract=c['fact_contract']
        if set(contract)!={'facts','unknowns','required_distinctions'} or not contract['facts']:raise ValueError('Explicit fact contracts required')
        if len({f['id'] for f in contract['facts']})!=len(contract['facts']):raise ValueError('Duplicate fact IDs')
        for fact in contract['facts']:
            if set(fact)!={'id','source_span','expected'} or not all(isinstance(v,str) and v.strip() for v in fact.values()) or fact['source_span'] not in text:raise ValueError('Fact evidence must be an exact source substring')
        for values in (contract['unknowns'],contract['required_distinctions'],c['observations'],c['tags']):
            if not isinstance(values,list) or any(not isinstance(x,str) or not x.strip() for x in values):raise ValueError('Invalid descriptive evidence fields')
        if not c['observations'] or not contract['required_distinctions']:raise ValueError('Separate behavioral observations and fidelity distinctions required')
    if len(pairs)!=12:raise ValueError('Expected 12 paired scenario families')
    for rows in pairs.values():
        if (len(rows)!=2 or {r['pressure'] for r in rows}!={'lower','higher'} or rows[0]['axis']!=rows[1]['axis']
                or rows[0]['fact_contract']!=rows[1]['fact_contract'] or rows[0]['prompt'].split('\n\n')[0]!=rows[1]['prompt'].split('\n\n')[0]):
            raise ValueError('Pressure pairs must share factual context and contracts')
    return suite


def load_suite(path):
    raw=Path(path).read_bytes();suite=validate_suite(json.loads(raw))
    suite['suite_sha256']=hashlib.sha256(raw).hexdigest()
    return suite


def registered_rows(suite):
    validate_suite(suite)
    digest=suite.get('suite_sha256')
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Load the source-hashed suite before registration')
    rows=[]
    for case in suite['cases']:
        row={'schema':'character-creation.manager-corpus.v1','corpus':'personality_diagnostics','split':'discovery','category':case['axis'],
            'system':'','prompt':case['prompt'],'max_tokens':2048,'thinking':False,'pair_id':case['pair_id'],'source_id':case['id'],
            'source_sha256':case['source_sha256'],'source_text_sha256':case['source_sha256'],'suite_sha256':digest,
            'source':{'kind':'authored_personality_probe','case_id':case['id'],'suite_sha256':digest},
            'purpose':'behavior','mutation':{'kind':'authored_pressure_variant','pressure':case['pressure']},
            'labels':{'prompt_design_trait':case['axis'],'prompt_design_facet':case['pressure'],'scenario_domain':case['axis']},
            'fact_contract':deepcopy(case['fact_contract']),'observations':list(case['observations']),'tags':list(case['tags']),
            'fit_on_heldout':False,'token_budget_status':'requires_runtime_tokenizer_check'}
        row['id']=canonical_digest(row);rows.append(row)
    return rows
