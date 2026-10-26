"""Durable arbitrary-text searches and evidence-bound signed comparisons."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
import re
import time
import subprocess
import sys
import threading
import uuid
from persona_fleet import fetch
from persona_vectors import canonical_digest


def identity(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{32}', value):
        raise ValueError('Invalid activation search ID')
    return value


def starter_recipes(result):
    """Small, explicitly untested experiments; activity does not establish efficacy."""
    candidates = result.get('candidates', []) if isinstance(result, dict) else []
    distinct = []
    for candidate in candidates:
        if not any(row['layer'] == candidate['layer'] for row in distinct):
            distinct.append({'layer': candidate['layer'], 'neuron': candidate['neuron']})
        if len(distinct) == 3:
            break
    trials = []
    if distinct:
        trials.append(('single', 'Test one active candidate', distinct[:1],
                       'Isolate the first activity-ranked candidate and compare both push directions.'))
    if len(distinct) > 1:
        trials.append(('other_layer', 'Try a different layer', distinct[1:2],
                       'Contrast a candidate from another layer before combining effects.'))
        trials.append(('small_group', 'Try a small group', distinct,
                       'Split the same total dose across up to three layers; interactions are untested.'))
    return [{'id': name, 'label': label, 'neurons': neurons, 'dose': .1, 'reason': reason,
             'status': 'untested', 'evidence': 'current_input_activity',
             'claim': 'Starting experiment only; no established concept direction or coherence guarantee.'}
            for name, label, neurons, reason in trials]


def validate_result(result, model, top_k):
    if not isinstance(result, dict): raise ValueError('Missing activation search result')
    coverage = result.get('coverage', {})
    if (coverage.get('all_neurons') is not True or coverage.get('all_prompt_positions') is not True
            or coverage.get('layers') != model.layers or coverage.get('neurons_per_layer') != model.intermediate):
        raise ValueError('Worker did not verify full model/prompt coverage')
    tokens = result.get('tokens')
    if (not isinstance(tokens, list) or not tokens or len(tokens) != coverage.get('tokens')
            or any(not isinstance(t,dict) or t.get('position') != i or type(t.get('id')) is not int
                   or not isinstance(t.get('text'),str) for i,t in enumerate(tokens))):
        raise ValueError('Token alignment or coverage differs')
    candidates=result.get('candidates'); seen=set()
    if not isinstance(candidates,list) or not 1 <= len(candidates) <= top_k:
        raise ValueError('Invalid activation candidate count')
    for row in candidates:
        if not isinstance(row,dict):raise ValueError('Invalid activation candidate record')
        key=(row.get('layer'),row.get('neuron'))
        if (any(type(v) is not int for v in key) or not 0<=key[0]<model.layers
                or not 0<=key[1]<model.intermediate or key in seen):raise ValueError('Invalid or duplicate candidate coordinate')
        seen.add(key)
        values=row.get('values')
        if (not isinstance(values,list) or len(values)!=len(tokens)
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in values)):
            raise ValueError('Candidate profile is incomplete or nonfinite')
        pos=row.get('peak_position')
        if type(pos) is not int or not 0<=pos<len(tokens):raise ValueError('Invalid candidate peak position')
        for name in ('peak','mean_abs'):
            if type(row.get(name)) not in (int,float) or not math.isfinite(row[name]):raise ValueError('Nonfinite candidate summary')
        if not math.isclose(values[pos],row['peak'],rel_tol=1e-5,abs_tol=1e-6):raise ValueError('Peak and replay profile differ')
        if not math.isclose(max(abs(v) for v in values),abs(row['peak']),rel_tol=1e-5,abs_tol=1e-6):raise ValueError('Reported peak is not profile maximum')
        if not math.isclose(sum(abs(v) for v in values)/len(values),row['mean_abs'],rel_tol=1e-5,abs_tol=1e-6):raise ValueError('Mean and replay profile differ')
    return deepcopy(result)


class ActivationLab:
    def __init__(self, app):
        self.app=app
        self.endpoints=list(app.endpoints)+list(getattr(app,'activation_endpoints',[]))
        with app.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS activation_searches(id TEXT PRIMARY KEY,created REAL,state TEXT,input TEXT,output TEXT,error TEXT,raw TEXT)')
            db.execute("UPDATE activation_searches SET state='interrupted',error='Explorer restarted; remote outcome unknown; no automatic retry' WHERE state IN ('queued','running')")
            db.execute('CREATE TABLE IF NOT EXISTS activation_comparisons(search_id TEXT,batch_id TEXT PRIMARY KEY,input TEXT)')
        self.executor=ThreadPoolExecutor(max_workers=max(1,min(6,len(self.endpoints))))
        self.eligibility_lock=threading.Lock()
        self.eligible_cache=[]
        self.eligible_updated=0.
        self.session_lock=threading.Lock()
        self.session={"state":"idle"}

    def eligible_endpoints(self):
        # Admission discovery is cached briefly; each actual request rechecks
        # model/runtime identities before inference. Never route to idle aliases.
        with self.eligibility_lock:
            if self.eligible_cache and time.monotonic()-self.eligible_updated<5:
                return deepcopy(self.eligible_cache)
            eligible=[];seen=set()
            for endpoint in self.endpoints:
                card=endpoint.get('gpu_uuid') or endpoint['url']
                if card in seen:continue
                try:
                    descriptor=fetch(endpoint['url']+'/v1/models',timeout=3)['data'][0]
                    flare=self.app.check_program(descriptor,endpoint,{'phase':'all','sites':[]})
                    if flare.get('activation_search',{}).get('supported') is True:
                        eligible.append(deepcopy(endpoint));seen.add(card)
                except (OSError,RuntimeError,ValueError,KeyError):continue
            self.eligible_cache=eligible;self.eligible_updated=time.monotonic()
            return deepcopy(eligible)

    def status(self):
        def check(endpoint):
            try:
                descriptor=fetch(endpoint['url']+'/v1/models',timeout=2)['data'][0]
                flare=self.app.check_program(descriptor,endpoint,{'phase':'all','sites':[]})
                ready=flare.get('activation_search',{}).get('supported') is True
                return {'label':endpoint.get('label',endpoint['url']),'ready':ready,'reason':None if ready else 'Worker does not advertise all-neuron activation search'}
            except Exception as error:return {'label':endpoint.get('label',endpoint['url']),'ready':False,'reason':str(error)}
        with ThreadPoolExecutor(max_workers=max(1,len(self.endpoints))) as pool:aliases=list(pool.map(check,self.endpoints))
        physical={}
        for endpoint,card in zip(self.endpoints,aliases):
            key=endpoint.get('gpu_uuid') or endpoint['url']
            if key not in physical or card['ready']:physical[key]=card
        cards=list(physical.values())
        with self.session_lock:session=deepcopy(self.session)
        return {'ready':sum(c['ready'] for c in cards),'configured':len(cards),'cards':cards,'endpoint_aliases':len(aliases),'session':session,
                'isolated_configured':bool(getattr(self.app,'activation_endpoints',[])),'ablation':False}

    def start_session(self,value):
        if not isinstance(value,dict) or set(value)!={'minutes'} or type(value['minutes']) is not int or not 30<=value['minutes']<=240:
            raise ValueError('Session minutes must be 30–240')
        if not getattr(self.app,'activation_endpoints',[]):raise ValueError('The isolated runtime is not configured')
        launcher=self.app.root/'activation-search/launch.py'
        if not launcher.is_file():raise ValueError('The reviewed activation-search launcher is unavailable')
        with self.session_lock:
            if self.session['state']=='launching':return deepcopy(self.session)
            self.session={'state':'launching','minutes':value['minutes'],'started':time.time(),'error':None}
        def launch():
            try:
                completed=subprocess.run([sys.executable,str(launcher),'--minutes',str(value['minutes'])],
                    cwd=self.app.root,capture_output=True,text=True,timeout=150,check=True)
                update={'state':'warming','output':completed.stdout[-4000:]}
            except Exception as error:
                update={'state':'failed','error':str(error),'detail':getattr(error,'stderr','')[-2000:]}
            with self.session_lock:self.session.update(update)
        threading.Thread(target=launch,daemon=True).start()
        return deepcopy(self.session)

    def listing(self):
        with self.app.connect() as db:
            return [{'id':r[0],'created':r[1],'state':r[2],'input':json.loads(r[3]),'error':r[4]} for r in db.execute('SELECT id,created,state,input,error FROM activation_searches ORDER BY created DESC LIMIT 30')]

    def get(self, key):
        identity(key)
        with self.app.connect() as db:
            r=db.execute('SELECT created,state,input,output,error FROM activation_searches WHERE id=?',(key,)).fetchone()
            comparisons=[{'id':row[0],'input':json.loads(row[1])} for row in db.execute('SELECT batch_id,input FROM activation_comparisons WHERE search_id=?',(key,))]
        if not r:raise ValueError('Unknown activation search')
        output=json.loads(r[3]) if r[3] else None
        return {'id':key,'created':r[0],'state':r[1],'input':json.loads(r[2]),'output':output,'error':r[4],'comparisons':comparisons,'capabilities':{'ablation':False},
                'starter_recipes':starter_recipes(output) if r[1]=='complete' else []}

    def submit(self, value):
        if not isinstance(value,dict) or set(value)!={'text','top_k'}:raise ValueError('Supply text and top_k')
        text=value['text'];top=value['top_k']
        if not isinstance(text,str) or not text.strip() or len(text)>20000:raise ValueError('Supply 1–20,000 characters; input is never silently truncated')
        if type(top) is not int or not 1<=top<=32:raise ValueError('Select 1–32 candidates')
        eligible=[endpoint['url'] for endpoint in self.eligible_endpoints()]
        if not eligible:raise ValueError('No identity-verified worker supports all-neuron activation search yet. Start a session with the upgraded diagnostic runtime.')
        key=uuid.uuid4().hex
        value=deepcopy(value)|{'model_identity':self.app.model.identity}
        with self.app.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT count(*) FROM activation_searches WHERE state IN ('queued','running')").fetchone()[0]>=64:raise ValueError('Activation search queue is full (64 active or queued requests)')
            db.execute('INSERT INTO activation_searches VALUES(?,?,?,?,?,?,?)',(key,time.time(),'queued',json.dumps(value),None,None,None))
        self.executor.submit(self.run,key,value,eligible)
        return {'id':key}

    def run(self,key,value,eligible):
        raw={};output=None;error=None;state='failed'
        def save_raw():
            with self.app.connect() as db:db.execute('UPDATE activation_searches SET raw=? WHERE id=?',(json.dumps(raw,allow_nan=False),key))
        try:
            with self.app.connect() as db:db.execute("UPDATE activation_searches SET state='running' WHERE id=?",(key,))
            with self.app.endpoint_slot(eligible) as endpoint:
                descriptor=fetch(endpoint['url']+'/v1/models',timeout=5)['data'][0]
                flare=self.app.check_program(descriptor,endpoint,{'phase':'all','sites':[]})
                if not flare.get('activation_search',{}).get('supported'):raise ValueError('Worker search capability changed')
                payload={'model':descriptor['id'],'messages':[{'role':'user','content':value['text']}],
                         'seed':7035,'max_tokens':1,'temperature':0,'stream':False,'observation_only':True,
                         'chat_template_kwargs':{'enable_thinking':False},'capture_search':{'id':key,'top_k':value['top_k']}}
                raw={'request':payload,'endpoint':deepcopy(endpoint),'model_identity':self.app.model.identity,'request_sha256':canonical_digest(payload)};save_raw()
                response=fetch(endpoint['url']+'/v1/chat/completions',payload,timeout=1200)
                raw['response']=response;save_raw()
                for field in ('model_identity','runtime_identity','prepared_identity'):
                    if not endpoint.get(field) or response.get(field)!=endpoint[field]:raise ValueError('Search response identity differs: '+field)
                artifact=response.get('activation_artifact',{})
                if artifact.get('request_sha256')!=raw['request_sha256']:raise ValueError('Search artifact request binding differs')
                result=response.get('activation_search')
                if not isinstance(result,dict) or result.get('id')!=key:raise ValueError('Search response ID differs')
                output=validate_result(result,self.app.model,value['top_k'])
                output.update(model_identity=self.app.model.identity,runtime_identity=endpoint['runtime_identity'],
                              prepared_identity=endpoint['prepared_identity'],activation_artifact=artifact,request_sha256=raw['request_sha256'])
                state='complete'
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        finally:
            with self.app.connect() as db:db.execute('UPDATE activation_searches SET state=?,output=?,error=? WHERE id=?',(state,json.dumps(output,allow_nan=False) if output else None,error,key))

    def compare(self,value):
        required={'search_id','neurons','prompt','seed','dose','max_tokens'}
        if not isinstance(value,dict) or set(value)!=required:raise ValueError('Invalid candidate comparison fields')
        record=self.get(value['search_id'])
        if record['state']!='complete':raise ValueError('Select candidates from a completed activation search')
        if record['input']['model_identity']!=self.app.model.identity:raise ValueError('Search belongs to a different model')
        allowed={(row['layer'],row['neuron']) for row in record['output']['candidates']}
        neurons=value['neurons']
        if not isinstance(neurons,list) or not 1<=len(neurons)<=6:raise ValueError('Select 1–6 candidate neurons')
        chosen=set()
        for row in neurons:
            if not isinstance(row,dict) or set(row)!={'layer','neuron'} or any(type(v) is not int for v in row.values()):raise ValueError('Invalid candidate coordinate')
            key=(row['layer'],row['neuron'])
            if key not in allowed or key in chosen:raise ValueError('Candidate absent from this search or repeated')
            chosen.add(key)
        dose=value['dose']
        if type(dose) not in (int,float) or not math.isfinite(dose) or not 0<dose<=1:raise ValueError('Use a finite relative dose above 0 and at most 1')
        if type(value['max_tokens']) is not int or not 2048<=value['max_tokens']<=4096:raise ValueError('Comparisons require 2,048–4,096 answer tokens')
        # Divide a collection's total dose so selecting more candidates does not silently multiply it.
        cells=[]
        for label,sign in [('Baseline',0),('Positive',1),('Negative',-1)]:
            pushes=[] if not sign else [{'kind':'neuron',**row,'dose':sign*dose/len(neurons)} for row in neurons]
            cells.append({'label':label,'prompt':value['prompt'],'recipe':{'name':label,'pushes':pushes,'phase':'all'}})
        endpoints=self.eligible_endpoints()
        if not endpoints:raise ValueError('No ready compatible search worker is available for comparison')
        batch=self.app.submit_batch({'model_identity':self.app.model.identity,'seed':value['seed'],'max_tokens':value['max_tokens'],'thinking':False,'smoke':False,'cells':cells}, endpoints=endpoints)
        with self.app.connect() as db:db.execute('INSERT INTO activation_comparisons VALUES(?,?,?)',(value['search_id'],batch,json.dumps(value,allow_nan=False)))
        return {'id':batch,'dose_semantics':'Total relative dose divided equally across selected neurons','ablation':False}
