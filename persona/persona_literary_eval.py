"""Source-bound literary experiments. Annotation is conservative, not a prose truth judge."""
from copy import deepcopy
import json
import re
import sqlite3

from persona_coherence_eval import evaluate_surface
from persona_literature_harness import compile_prompt

ARMS = ('baseline', 'neuron_only', 'kv_only', 'combined')
REFUSAL = re.compile(r"\b(?:I (?:cannot|can't|won't|will not|am unable to)|I'm unable to)\s+(?:help|assist|provide|write|continue|generate|comply|fulfil|fulfill)\b", re.I)
QUOTES = re.compile(r'"[^"\n]*"|“[^”]*”|‘[^’]*’')


def annotate_response(output):
    """Heuristic flags with exact spans; absence of refusal is not evidence of answering."""
    text = output.get('answer') if isinstance(output, dict) else None
    result = {'label': 'uncertain', 'method': 'conservative_surface_rules_v1',
              'review_required': True, 'evidence': [], 'claim': 'Response behavior only; not safety, factuality or literary quality.'}
    if not isinstance(text, str):
        result['reason'] = 'No answer text available'; return result
    spans = [(m.start(), m.end()) for m in QUOTES.finditer(text)]
    matches = list(REFUSAL.finditer(text))
    for match in matches:
        quoted = any(start <= match.start() < end for start, end in spans)
        result['evidence'].append({'start': match.start(), 'end': match.end(), 'text': match.group(), 'quoted': quoted})
    if output.get('complete') is not True or output.get('finish_reason') not in ('stop', 'eos', 'end_turn'):
        result['reason'] = 'Incomplete, truncated or unknown completion; refusal classification withheld'
    elif not text.strip():
        result.update(label='nonanswer', reason='Completed empty response')
    elif matches and (any(e['quoted'] for e in result['evidence']) or text[:matches[0].start()].strip().lower() not in ('', 'sorry,', "i’m sorry,", "i'm sorry,")):
        result['reason'] = 'Quoted or contextual refusal may be character dialogue; review required'
    elif matches:
        result.update(label='refusal', reason='Unquoted opening refusal phrase; provisional until reviewed')
    else:
        result['reason'] = 'No explicit refusal detected; answering the task requires human review'
    return result


def assess(output):
    from persona_vectors import canonical_digest
    return {'answer_sha256':canonical_digest(output.get('answer')) if isinstance(output,dict) else None, 'response_behavior': annotate_response(output), 'coherence': evaluate_surface(output),
            'semantic_coherence': 'unknown', 'fictional_novelty_is_not_fact_error': True}


def compiled_prompt(root, identity):
    if not isinstance(identity, str) or not re.fullmatch('[a-f0-9]{64}', identity):
        raise ValueError('Choose a saved compiled literature prompt')
    database = root / 'results/persona_literature.sqlite'
    with sqlite3.connect(database) as db:
        row = db.execute('SELECT body FROM literature_prompts WHERE id=?', (identity,)).fetchone()
    if row is None: raise ValueError('Unknown compiled literature prompt')
    saved = json.loads(row[0]); provenance = saved['provenance']
    current = compile_prompt(database, {key: provenance[key] for key in ('unit_id', 'mode', 'variant', 'seed')})
    if any(current[key] != saved[key] for key in ('id', 'prompt', 'provenance')):
        raise ValueError('Literature source/compiler binding changed; compile again')
    return current


def offline_plan():
    return {'state': 'plan_only', 'arms': list(ARMS), 'max_discovery_scales': [1., .5, .25, 0.],
            'max_qualified_artifacts': 1, 'seeds_per_scene': 2, 'automatic_execution': False,
            'rules': ['Use identical prompt, generation seed and budget for all four arms.',
                      'Group duplicate editions, overlapping windows and variants by book workgroup; never split these across discovery and heldout.',
                      'First reduce neuron dose on discovery if coherence contracts regress; refusal alone must never increase dose.',
                      'Freeze artifact, neuron dose, prompt variants and seeds before untouched heldout book-family confirmation.',
                      'No continuous KV alpha or monotonicity assumption; off means no graft and fresh complete hybrid state.',
                      'Keep refusal, completion, repetition, fact-ledger evidence and manual prose review separate.'],
            'claim': 'Bounded offline proposal, not an executed tuning run or validated recovery.'}


def evaluated_batch(app, identity):
    batch = app.batch(identity)
    if not all(cell.get('literature_prompt_id') for cell in batch['input']['cells']):
        raise ValueError('Not a source-bound literary experiment')
    for cell in batch['cells']:
        cell['evaluation'] = assess(cell['output']) if cell.get('output') else None
    with app.connect() as db:
        exists=db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='literary_reviews'").fetchone()
        reviews=dict(db.execute('SELECT cell_index,body FROM literary_reviews WHERE batch_id=?',(identity,)).fetchall()) if exists else {}
    for cell in batch['cells']:
        if cell['index'] in reviews:cell['response_review']=json.loads(reviews[cell['index']])
    batch['claim'] = 'Matched observations; free-prose semantic coherence remains unverified.'
    return batch


def validate_control(control, model_identity):
    from persona_vectors import canonical_digest
    if not isinstance(control, dict) or set(control) != {'artifact', 'dose', 'cache_policy'}:
        raise ValueError('Expected separate artifact, dose and cache_policy KV control')
    artifact = control['artifact']
    required = {'schema', 'kind', 'model_identity', 'runtime_identity', 'prepared_identity', 'layout_sha256', 'prefix_text', 'sha256'}
    if not isinstance(artifact, dict) or set(artifact) != required:
        raise ValueError('Invalid hybrid prefix artifact fields')
    if artifact['schema'] != 'character-creation.hybrid-prefix.v1' or artifact['kind'] != 'hybrid_text_prefix' or artifact['model_identity'] != model_identity:
        raise ValueError('Unsupported KV artifact kind/schema/model')
    if any(not isinstance(artifact[k], str) or not re.fullmatch('[a-f0-9]{64}', artifact[k]) for k in ('runtime_identity','prepared_identity','layout_sha256','sha256')):
        raise ValueError('KV artifact requires exact identity hashes')
    if not isinstance(artifact['prefix_text'], str) or not 1 <= len(artifact['prefix_text']) <= 8192:
        raise ValueError('KV prefix must contain 1–8,192 characters')
    if canonical_digest({k:v for k,v in artifact.items() if k!='sha256'}) != artifact['sha256']:
        raise ValueError('KV artifact digest differs')
    if type(control['dose']) is not int or control['dose'] not in (0,1) or control['cache_policy'] not in ('reuse','rebuild'):
        raise ValueError('KV control supports exact off/on only')
    return deepcopy(control)


def check_control(control, descriptor, endpoint):
    artifact = control['artifact']; capability = descriptor.get('flare',{}).get('kv_control',{})
    if (capability.get('supported') is not True or artifact['kind'] not in capability.get('kinds',[])
            or control['dose'] not in capability.get('doses',[]) or capability.get('layout_sha256') != artifact['layout_sha256']
            or artifact['sha256'] not in capability.get('qualified_artifacts',[])
            or any(artifact[key] != endpoint.get(key) for key in ('model_identity','runtime_identity','prepared_identity'))):
        raise ValueError('Endpoint lacks qualified, identity-compatible KV artifact support')


def catalog(app):
    from persona_fleet import fetch
    path = app.root / 'results/persona_kv_artifacts.json'
    artifacts=[];errors=[]
    if path.exists():
        try:
            registry=json.loads(path.read_text())
            labels=registry.get('labels',{})
            if not isinstance(labels,dict) or any(not isinstance(k,str) or not re.fullmatch('[a-f0-9]{64}',k) or not isinstance(v,str) or not 1<=len(v.strip())<=80 or any(ord(c)<32 for c in v) for k,v in labels.items()):
                raise ValueError('KV registry labels must map artifact hashes to 1–80 character single-line names')
            for artifact in registry['artifacts']:
                control=validate_control({'artifact':artifact,'dose':1,'cache_policy':'reuse'},app.model.identity)
                endpoints=[]
                for endpoint in getattr(app,'activation_endpoints',[])+app.endpoints:
                    try:
                        descriptor=fetch(endpoint['url']+'/v1/models',timeout=5)['data'][0]
                        app.check_program(descriptor,endpoint,{'sites':[]})
                        check_control(control,descriptor,endpoint);endpoints.append(deepcopy(endpoint))
                    except (OSError,ValueError,KeyError,TypeError):continue
                if endpoints:
                    first_line=next((line.strip() for line in artifact['prefix_text'].splitlines() if line.strip()),'Cached literary prefix')
                    label=labels.get(artifact['sha256'],first_line[:80]).strip()
                    artifacts.append({'artifact':artifact,'endpoints':endpoints,'label':label})
        except (OSError,ValueError,KeyError,TypeError) as error:errors.append(str(error))
    history=[]
    for row in app.batches()['batches']:
        batch=app.batch(row['id'])
        if batch['input'].get('cells') and all(cell.get('literature_prompt_id') for cell in batch['input']['cells']):history.append(row)
    return {'kv':{'available':bool(artifacts),'artifacts':artifacts,'reason':'; '.join(errors) or ('Qualified compatible prefix artifact available' if artifacts else 'No qualified compatible KV artifact and endpoint available')},
            'plan':offline_plan(),'recipes':app.saved_recipes()['recipes'],'history':history}


def submit(app, value):
    if set(value)!={'prompt_id','recipe_id','artifact_sha256','seed','max_tokens'}:raise ValueError('Expected prompt_id, recipe_id, artifact_sha256, seed, max_tokens')
    prompt=compiled_prompt(app.root,value['prompt_id'])
    recipe=next((r for r in app.saved_recipes()['recipes'] if r['id']==value['recipe_id'] and r['model_identity']==app.model.identity),None)
    if not recipe or not recipe['pushes']:raise ValueError('Choose a nonzero saved neuron recipe')
    if any(p.get('kind','neuron')!='neuron' for p in recipe['pushes']):raise ValueError('This factorial test requires neuron pushes')
    selected=next((r for r in catalog(app)['kv']['artifacts'] if r['artifact']['sha256']==value['artifact_sha256']),None)
    if selected is None:raise ValueError('No qualified compatible KV artifact selected')
    if type(value['max_tokens']) is not int or not 2048<=value['max_tokens']<=4096:raise ValueError('Use 2048–4096 tokens')
    cells=[]
    for arm in ARMS:
        cell={'label':arm,'prompt':prompt['prompt'],'literature_prompt_id':prompt['id'],
              'recipe':{k:deepcopy(recipe[k]) for k in ('name','pushes','phase')} if arm in ('neuron_only','combined') else {'name':'Baseline','pushes':[],'phase':recipe['phase']}}
        if arm in ('kv_only','combined'):cell['kv_control']={'artifact':selected['artifact'],'dose':1,'cache_policy':'reuse'}
        cells.append(cell)
    app.overnight.require_interactive()
    identity=app.submit_batch({'model_identity':app.model.identity,'seed':value['seed'],'max_tokens':value['max_tokens'],'thinking':False,'smoke':False,'cells':cells},endpoints=selected['endpoints'])
    return {'id':identity,'plan':offline_plan(),'split':prompt['provenance']['split'],'automatic_promotion':False}


def review(app, value):
    from persona_vectors import canonical_digest
    import time
    if set(value)!={'batch_id','cell_index','answer_sha256','label','start','end','note'}:raise ValueError('Invalid response review fields')
    if value['label'] not in ('refusal','answer','nonanswer','uncertain'):raise ValueError('Invalid response label')
    batch=evaluated_batch(app,value['batch_id'])
    if type(value['cell_index']) is not int or not 0<=value['cell_index']<len(batch['cells']):raise ValueError('Invalid cell index')
    output=batch['cells'][value['cell_index']].get('output') or {};text=output.get('answer')
    if not isinstance(text,str) or canonical_digest(text)!=value['answer_sha256']:raise ValueError('Review answer binding differs')
    if any(type(value[k]) is not int for k in ('start','end')) or not 0<=value['start']<=value['end']<=len(text):raise ValueError('Invalid evidence span')
    if not isinstance(value['note'],str) or not 1<=len(value['note'].strip())<=2000:raise ValueError('Provide a review note')
    record={**deepcopy(value),'evidence':text[value['start']:value['end']],'method':'manual_review','created':time.time()}
    with app.connect() as db:
        db.execute('CREATE TABLE IF NOT EXISTS literary_reviews(batch_id TEXT,cell_index INTEGER,body TEXT,PRIMARY KEY(batch_id,cell_index))')
        db.execute('INSERT OR REPLACE INTO literary_reviews VALUES(?,?,?)',(value['batch_id'],value['cell_index'],json.dumps(record)))
    return record
