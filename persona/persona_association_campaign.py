"""Bounded association/recipe research on six already-leased interactive workers.

No GPU allocation, model training, outcome-based selection, or automatic preset
promotion. POST outcomes are never retried. The caller supplies a frozen plan.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import urllib.request

from persona_literary_eval import assess


URLS = tuple(f'http://127.0.0.1:{port}' for port in range(8930, 8936))


def atomic(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def matching_binding(models, expected):
    if len(models) != 6:
        raise ValueError('Exactly six ready workers required')
    bindings = []
    for model in models:
        flare = model['data'][0]['flare']
        if (flare['model_identity'] != expected or not flare['activation_search']['supported']
                or flare.get('calibration_equivalence', {}).get('passed') is not True):
            raise ValueError('Worker lacks matching checkpoint, search support, or calibration proof')
        bindings.append({**{k: flare[k] for k in ('model_identity', 'runtime_identity', 'prepared_identity')},
                         'layout_sha256': flare['kv_control']['layout_sha256']})
    if any(value != bindings[0] for value in bindings):
        raise ValueError('Six-worker runtime/layout identities differ')
    return bindings[0]


INPUT_FIELDS=('schema','dataset_sha256','top_k','recipe_policy','expected_searches','expected_scene_outputs','automatic_default_promotion')
QUERY_FIELDS=('id','text','kind','family_id','split','source_id','source_sha256')
POLICY={'selection':'first activity-ranked single candidate, fixed before outcomes','dose':.1,
        'arms':['baseline','positive','negative'],'seed':7035,'max_tokens':2048,
        'eligible_queries':'all 24 authored whole prompts; partials/concepts are lookup-only',
        'heldout':'Fixed rule applied without outcome-based tuning; preserve family splits'}


def input_plan_digest(plan):
    value={key:plan[key] for key in INPUT_FIELDS}
    value['queries']=[{key:q[key] for key in QUERY_FIELDS if key in q} for q in plan['queries']]
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def validate_plan(plan,launch,dataset_path):
    digest=input_plan_digest(plan)
    if launch.get('input_plan_sha256')!=digest:raise ValueError('Frozen input plan hash differs or is missing')
    raw=dataset_path.read_bytes();dataset=json.loads(raw)
    if (hashlib.sha256(raw).hexdigest()!=plan['dataset_sha256'] or plan['schema']!='character-creation.association-research.v1'
            or plan['recipe_policy']!=POLICY or plan['top_k']!=16 or plan['expected_searches']!=60
            or plan['expected_scene_outputs']!=72 or plan['automatic_default_promotion'] is not False):
        raise ValueError('Dataset, counts or fixed recipe policy differs')
    queries=plan['queries'];rows={r['id']:r for r in dataset['rows']}
    if (len(rows)!=24 or Counter(q['kind'] for q in queries)!=Counter(whole_prompt=24,scene_fragment=24,concept_phrase=12)
            or len({q['id'] for q in queries})!=60):raise ValueError('Query counts or IDs differ')
    for kind in ('whole_prompt','scene_fragment'):
        if {q.get('source_id') for q in queries if q['kind']==kind}!=set(rows):raise ValueError('Authored source coverage differs')
    for q in queries:
        if not isinstance(q['text'],str) or not q['text'].strip() or len(q['text'])>20000:raise ValueError('Invalid query text')
        if q['kind']=='concept_phrase':
            if q['split']!='discovery' or hashlib.sha256(q['text'].encode()).hexdigest()!=q['source_sha256']:raise ValueError('Concept source binding differs')
            continue
        row=rows[q['source_id']]
        if any(q[k]!=row[k] for k in ('family_id','split','source_sha256')):raise ValueError('Authored source/split binding differs')
        if (q['kind']=='whole_prompt' and q['text']!=row['prompt']) or q['text'] not in row['prompt']:raise ValueError('Query text differs from authored source')
    return digest


def check_identities(value,binding,required=False):
    if not isinstance(value,dict):raise ValueError('Missing evidence object')
    for key in ('model_identity','runtime_identity','prepared_identity','layout_sha256'):
        if (required and key!='layout_sha256' or key in value) and value.get(key)!=binding[key]:
            raise ValueError('Result binding differs: '+key)


def check_search(result,query,plan):
    if result.get('id')!=query['search_id'] or result.get('input')!={'text':query['text'],'top_k':plan['top_k'],'model_identity':plan['binding']['model_identity']}:
        raise ValueError('Search ID/input differs from frozen request')
    output=result['output'];check_identities(output,plan['binding'],True)
    for key in ('activation_artifact','profile_artifact'):
        check_identities(output[key],plan['binding'],True)
    if output['activation_artifact'].get('request_sha256')!=output.get('request_sha256'):raise ValueError('Search artifact request binding differs')
    first=output['candidates'][0];starter=result['starter_recipes'][0]
    if (starter.get('id')!='single' or starter.get('neurons')!=[{'layer':first['layer'],'neuron':first['neuron']}]
            or starter.get('dose')!=.1 or starter.get('status')!='untested' or starter.get('evidence')!='current_input_activity'):
        raise ValueError('Starter differs from frozen first-candidate rule')


def check_batch(result,record,binding):
    request=record['submission']['request'];inputs=result['input'];cells=result['cells']
    if (result.get('id')!=record['batch_id'] or len(cells)!=3 or len(inputs.get('cells',[]))!=3 or inputs.get('model_identity')!=binding['model_identity']
            or inputs.get('seed')!=request['seed'] or inputs.get('max_tokens')!=request['max_tokens']
            or inputs.get('thinking') is not False or inputs.get('smoke') is not False):raise ValueError('Batch identity/settings/arm count differs')
    for endpoint in inputs.get('admitted_endpoints',[]):check_identities(endpoint,binding,True)
    for index,(cell,label,sign) in enumerate(zip(cells,('Baseline','Positive','Negative'),(0,1,-1))):
        pushes=[] if not sign else [{'kind':'neuron',**row,'dose':sign*request['dose']/len(request['neurons'])} for row in request['neurons']]
        expected={'name':label,'pushes':pushes,'phase':'all'}
        raw=inputs['cells'][index]
        prompt=cell.get('prompt');text=prompt.get('text') if isinstance(prompt,dict) else prompt
        if (cell.get('index')!=index or cell.get('label')!=label or text!=request['prompt']
                or any(cell.get('recipe',{}).get(k)!=v for k,v in expected.items())
                or raw.get('prompt')!=request['prompt'] or raw.get('recipe')!=expected or raw.get('label')!=label):
            raise ValueError('Matched arm prompt/intervention differs')
        output=cell.get('output')
        if output:
            for evidence in (output,output.get('provenance',{}),output.get('activation_artifact',{})):
                check_identities(evidence,binding)
            if cell['state']=='complete' and output.get('complete') is not True:raise ValueError('Complete cell has incomplete output')
        elif cell['state']=='complete':raise ValueError('Complete cell lacks output')


class Campaign:
    def __init__(self, root):
        self.root = root
        self.path = root / 'results/persona_association_research.json'
        self.launch_path = root / 'results/persona_association_launch.json'
        self.plan = json.loads(self.path.read_text())
        self.launch = json.loads(self.launch_path.read_text())
        if self.plan['state'] != 'waiting_for_unified_runtime' or any(q['state'] != 'pending' for q in self.plan['queries']):
            raise ValueError('Existing attempts require manual recovery; never resubmit automatically')
        self.deadline = self.launch['deadline'] - 180
        if not time.time() < self.deadline <= time.time() + 86400:
            raise ValueError('A live finite serving lease deadline is required')
        self.stopped = False
        self.input_hash=validate_plan(self.plan,self.launch,root/'data/persona_phantom_pairs.json')

    def guard(self):
        if self.stopped or time.time() >= self.deadline:
            raise TimeoutError('Existing serving lease deadline reached or stop requested')

    def save(self):
        if hasattr(self,'input_hash') and input_plan_digest(self.plan)!=self.input_hash:raise ValueError('Frozen input plan changed during execution')
        self.plan['updated'] = time.time()
        atomic(self.path, self.plan)

    def request(self, url, value=None, timeout=90):
        self.guard()
        if url.startswith('/'): url = 'http://127.0.0.1:8792' + url
        request = urllib.request.Request(url, data=json.dumps(value).encode() if value is not None else None,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=min(timeout, max(1, self.deadline-time.time()))) as response:
            return json.load(response)

    def submit(self, record, route, value):
        record['submission'] = {'state': 'started', 'time': time.time(), 'request': value}
        self.save()
        try:
            response = self.request(route, value)
        except Exception as error:
            record['submission'].update(state='failed_or_uncertain', error=repr(error))
            self.save(); raise
        record['submission'].update(state='accepted', response=response)
        record['state'] = 'queued'; self.save()
        return response['id']

    def activate(self):
        def read(url):
            try: return self.request(url+'/v1/models', timeout=3)
            except OSError: return None
        while True:
            self.guard()
            with ThreadPoolExecutor(6) as pool: models = list(pool.map(read, URLS))
            if all(model is not None for model in models): break
            time.sleep(5)
        expected=self.launch.get('observed_binding')
        if not isinstance(expected,dict):raise ValueError('Manual deployment binding is missing')
        binding = matching_binding(models, expected['model_identity'])
        if binding!=expected:raise ValueError('Workers differ from manually verified launch binding')
        config=json.loads((self.root/'persona_explorer.local.json').read_text())
        endpoints=config['endpoints']
        if len(endpoints)!=6 or {row['url'] for row in endpoints}!=set(URLS):raise ValueError('Unexpected viewer endpoint set')
        for endpoint in endpoints:check_identities(endpoint,binding,True)
        self.plan['binding']=binding
        self.plan['state'] = 'searching'; self.save()

    def searches(self):
        pending = iter(self.plan['queries']); active = []; exhausted = False
        while active or not exhausted:
            self.guard()
            while len(active) < 12 and not exhausted:
                query = next(pending, None)
                if query is None: exhausted = True; break
                query['search_id'] = self.submit(query, '/api/activation-searches',
                                                 {'text': query['text'], 'top_k': self.plan['top_k']})
                active.append(query)
            for query in list(active):
                result = self.request('/api/activation-search?id='+query['search_id'])
                if result['state'] in ('queued', 'running'): continue
                query['state'] = result['state']
                if result['state'] != 'complete':
                    query['error'] = result.get('error'); self.save()
                    raise ValueError('Search failed; no automatic replay')
                check_search(result,query,self.plan)
                output = result['output']
                query['evidence'] = {k: output.get(k) for k in ('model_identity','runtime_identity','prepared_identity','request_sha256','activation_artifact','profile_artifact','result_artifact')}
                query['candidates'] = [{k: row[k] for k in ('layer','neuron','peak','peak_position','mean_abs')} for row in output['candidates']]
                query['starter_recipes'] = result['starter_recipes']
                active.remove(query); self.save()
            if active: time.sleep(.5)
        self.plan['state'] = 'scene_comparisons'; self.save()

    def comparisons(self):
        pending = iter(query for query in self.plan['queries'] if query['kind'] == 'whole_prompt')
        active = []; exhausted = False
        while active or not exhausted:
            self.guard()
            # Eight batches keep six workers fed without exceeding the viewer's
            # sixteen-batch admission limit or occupying the whole user queue.
            while len(active) < 8 and not exhausted:
                query = next(pending, None)
                if query is None: exhausted = True; break
                starter = query['starter_recipes'][0]
                record = {'state': 'pending', 'starter_id': starter['id']}; query['comparison'] = record
                value = {'search_id': query['search_id'], 'neurons': starter['neurons'], 'prompt': query['text'],
                         'seed': 7035, 'dose': .1, 'max_tokens': 2048}
                record['batch_id'] = self.submit(record, '/api/activation-comparisons', value)
                active.append(record)
            for record in list(active):
                result = self.request('/api/batch?id='+record['batch_id'])
                if result['state'] in ('queued', 'running'): continue
                check_batch(result,record,self.plan['binding'])
                record['input_snapshot']=deepcopy(result['input'])
                record['state'] = result['state']
                record['cells'] = [{**{k: cell.get(k) for k in ('index','label','state','prompt','recipe','source_binding','endpoint_label','output','error')},
                                    'evaluation': assess(cell['output']) if cell.get('output') else None}
                                   for cell in result['cells']]
                active.remove(record); self.save()
            if active: time.sleep(2)
        comparisons = [query['comparison'] for query in self.plan['queries'] if 'comparison' in query]
        successful = all(record['state'] == 'complete' for record in comparisons)
        self.plan.update(state='complete' if successful else 'complete_with_failed_comparisons', completed=time.time(),
                         semantic_review='Pending source-grounded review; surface checks do not certify prose semantics.')
        self.save()

    def run(self):
        def stop(signum, frame): self.stopped = True
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        self.plan['input_plan_sha256']=self.input_hash
        self.plan['controller_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.save()
        try:
            self.activate(); self.searches(); self.comparisons()
        except Exception as error:
            self.plan.update(state='stopped_or_failed', error=repr(error)); self.save(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    Campaign(parser.parse_args().root.resolve()).run()
