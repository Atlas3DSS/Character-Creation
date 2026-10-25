"""Bounded six-card diagnostic campaigns with durable token playback."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import re
import threading
import time
import uuid
import numpy as np
from persona_fleet import fetch
from persona_trace import normalize_trace, compare_logits, unpack_logits
from persona_vectors import canonical_digest

class TraceLab:
    def __init__(self,app):
        self.app=app;self.lock=threading.RLock();self.active={}
        with app.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS token_traces(id TEXT PRIMARY KEY,created REAL,state TEXT,input TEXT,output TEXT)')
            db.execute("UPDATE token_traces SET state='interrupted' WHERE state IN ('queued','running')")
        self.executor=ThreadPoolExecutor(max_workers=1)

    def submit(self,value):
        if not isinstance(value,dict) or set(value)!={'model_identity','prompts','pushes','phase','neurons','seed','max_tokens','mode'}:raise ValueError('Invalid trace campaign fields')
        if value['model_identity']!=self.app.model.identity:raise ValueError('Trace model differs')
        prompts=value['prompts'];mode=value['mode']
        if mode not in ('dose_sweep','three_prompts'):raise ValueError('Invalid trace layout')
        if not isinstance(prompts,list) or len(prompts)!=(1 if mode=='dose_sweep' else 3) or any(not isinstance(p,str) or not p.strip() or len(p)>20000 for p in prompts):raise ValueError('Supply one prompt or three prompts for the selected layout')
        if type(value['seed']) is not int or not 0<=value['seed']<2**32 or type(value['max_tokens']) is not int or not 1<=value['max_tokens']<=512:raise ValueError('Trace uses uint32 seed and 1–512 generated tokens')
        normalize_trace({'id':'0'*32,'neurons':value['neurons']})
        self.app.validate_recipe({'pushes':value['pushes'],'phase':value['phase']})
        capacity=self.app.capacity()
        if not capacity['available']:raise ValueError('Start an interactive fleet session in Jobs before tracing')
        arms=[]
        specifications=[(0,s) for s in (None,0,.5,1,-.5,-1)] if mode=='dose_sweep' else [(p,s) for p in range(3) for s in (None,1)]
        for prompt,scale in specifications:
            pushes=[] if scale is None else [{**p,'dose':p['dose']*scale} for p in value['pushes']]
            with self.app.bank_lock:program,provenance=self.app.bank.program(pushes,value['phase'])
            arms.append({'prompt_index':prompt,'scale':scale,'label':'Baseline' if scale is None else f'Recipe × {scale:g}',
                'trace_id':uuid.uuid4().hex,'program':program,'provenance':provenance,'state':'queued','events':[],'offset':0})
        eligible=[]
        for endpoint in self.app.endpoints:
            try:descriptor=fetch(endpoint['url']+'/v1/models',timeout=3)['data'][0]
            except (OSError,RuntimeError):continue
            flare=self.app.check_program(descriptor,endpoint,{'phase':value['phase'],'sites':[]})
            if flare.get('token_trace',{}).get('supported'):eligible.append(endpoint['url'])
        if not eligible:raise ValueError('No ready identity-verified worker supports token traces')
        identity=uuid.uuid4().hex
        with self.app.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT COUNT(*) FROM token_traces WHERE state IN ('queued','running')").fetchone()[0]:raise ValueError('A trace campaign is already active')
            db.execute('INSERT INTO token_traces VALUES(?,?,?,?,?)',(identity,time.time(),'queued',json.dumps(value),None))
        with self.lock:self.active[identity]={'id':identity,'state':'queued','input':value,'arms':arms,'eligible':eligible}
        self.executor.submit(self.run,identity)
        return {'id':identity}

    def search(self,text):
        if not isinstance(text,str) or not 1<=len(text)<=100:raise ValueError('Search needs 1–100 characters')
        matches=[]
        with self.app.connect() as db:
            for identity,output in db.execute("SELECT id,output FROM token_traces WHERE state='complete' ORDER BY created DESC LIMIT 50"):
                for index,arm in enumerate(json.loads(output or '[]')):
                    for sign in ('increases','decreases'):
                        for row in arm.get('lens',{}).get(sign,[]):
                            if text.casefold() in row['text'].casefold():
                                matches.append({'campaign_id':identity,'arm':index,'label':arm['label'],'prompt_index':arm['prompt_index'],**row})
        return {'matches':sorted(matches,key=lambda row:-abs(row['delta_logp']))[:40],
            'scope':'Stored top Δlog-probability tokens in the latest 50 completed trace campaigns; not an exhaustive neuron-effect index.'}

    def listing(self):
        with self.app.connect() as db:return [{'id':r[0],'created':r[1],'state':r[2]} for r in db.execute('SELECT id,created,state FROM token_traces ORDER BY created DESC LIMIT 20')]

    def get(self,identity):
        if not re.fullmatch('[0-9a-f]{32}',identity):raise ValueError('Invalid trace id')
        with self.lock:
            if identity in self.active:return self.public(self.active[identity])
        with self.app.connect() as db:row=db.execute('SELECT state,input,output FROM token_traces WHERE id=?',(identity,)).fetchone()
        if row is None:raise ValueError('Unknown trace campaign')
        return {'id':identity,'state':row[0],'input':json.loads(row[1]),'arms':json.loads(row[2]) if row[2] else []}

    @staticmethod
    def public(record):
        return {**{k:deepcopy(v) for k,v in record.items() if k not in ('arms','eligible')},'arms':[{k:deepcopy(v) for k,v in arm.items() if k not in ('program','raw_response','endpoint','provenance')} for arm in record['arms']]}

    def run(self,identity):
        record=self.active[identity];value=record['input']
        record['state']='running'
        with self.app.connect() as db:db.execute('UPDATE token_traces SET state=? WHERE id=?',('running',identity))
        def arm_run(arm):
            stop=threading.Event();poller=None
            try:
                with self.app.endpoint_slot(record['eligible']) as endpoint:
                    descriptor=fetch(endpoint['url']+'/v1/models',timeout=5)['data'][0]
                    flare=self.app.check_program(descriptor,endpoint,arm['program'])
                    if not flare.get('token_trace',{}).get('supported'):raise ValueError('Worker has no token-trace capability')
                    with self.lock:arm.update(state='running',endpoint=endpoint,label=arm['label']+' · '+endpoint.get('label','GPU'))
                    payload={'model':descriptor['id'],'messages':[{'role':'user','content':value['prompts'][arm['prompt_index']]}],
                        'seed':value['seed'],'max_tokens':value['max_tokens'],'temperature':0,'stream':False,
                        'chat_template_kwargs':{'enable_thinking':False},'capture_trace':{'id':arm['trace_id'],'neurons':value['neurons']}}
                    if arm['program']['sites']:payload['steering']=arm['program']
                    def poll():
                        while not stop.wait(.6):
                            try:
                                result=fetch(endpoint['url']+f"/v1/trace?id={arm['trace_id']}&offset={arm['offset']}",timeout=5)
                                with self.lock:
                                    arm['events'].extend(result['events']);arm['offset']=result['offset']
                            except Exception as error:
                                with self.lock:arm['stream_warning']=str(error)
                    poller=threading.Thread(target=poll,daemon=True);poller.start()
                    response=fetch(endpoint['url']+'/v1/chat/completions',payload,timeout=1200)
                    for key in ('model_identity','runtime_identity','prepared_identity'):
                        if response.get(key)!=endpoint.get(key):raise ValueError('Trace response identity differs')
                    trace=response['token_trace']
                    if trace['id']!=arm['trace_id'] or trace['neurons']!=value['neurons']:raise ValueError('Trace response coordinates differ')
                    artifact=response['activation_artifact']
                    if artifact['request_sha256']!=canonical_digest(payload):raise ValueError('Trace request binding differs')
                    stop.set();poller.join(timeout=6)
                    with self.lock:arm.update(state='complete',events=trace['events'],raw_response=response,
                        answer=response['choices'][0]['message']['content'],finish_reason=response['choices'][0]['finish_reason'],
                        activation_artifact=artifact,semantics=trace['semantics'])
            except Exception as error:
                with self.lock:arm.update(state='failed',error=f'{type(error).__name__}: {error}')
            finally:
                stop.set()
                if poller:poller.join(timeout=6)
        try:
            with ThreadPoolExecutor(max_workers=min(6,len(record['eligible']))) as workers:list(workers.map(arm_run,record['arms']))
            tokens=self.app.atlas.vocabulary()
            for arm in record['arms']:
                baseline=next(a for a in record['arms'] if a['prompt_index']==arm['prompt_index'] and a['scale'] is None)
                if arm['state']=='complete' and baseline['state']=='complete':
                    a=baseline['raw_response'];b=arm['raw_response']
                    if a['usage']['prompt_tokens']!=b['usage']['prompt_tokens']:raise ValueError('Paired prompt lengths differ')
                    arm['lens']=compare_logits(a['token_trace']['next_token_logits'],b['token_trace']['next_token_logits'],lambda i:tokens[i])
                    for key,response in [('baseline_top',a),('intervention_top',b)]:
                        logits=unpack_logits(response['token_trace']['next_token_logits']);shifted=logits-logits.max()
                        logp=shifted-np.log(np.exp(shifted).sum())
                        ids=np.argsort(-logp,kind='stable')[:12]
                        arm['lens'][key]=[{'id':int(i),'text':tokens[i],'probability':float(np.exp(logp[i]))} for i in ids]
                    arm['baseline_equal']=a['choices']==b['choices'] and a['token_trace']['events']==b['token_trace']['events']
            record['state']='failed' if any(a['state']=='failed' for a in record['arms']) else 'complete'
        except Exception as error:
            record.update(state='failed',error=str(error))
            for arm in record['arms']:arm['lens_error']=str(error)
        finally:
            public=self.public(record)
            # Complete raw replies, full logits and artifact hashes stay in one
            # immutable local audit file; the UI/SQLite playback is compact.
            path=self.app.root/'results'/f'persona_trace_{identity}.json'
            try:
                with path.open('x') as stream:json.dump(record,stream,allow_nan=False)
            except Exception as error:
                record['state']='failed'
                for arm in public['arms']:arm['error']='Audit publication failed: '+str(error)
            with self.app.connect() as db:db.execute('UPDATE token_traces SET state=?,output=? WHERE id=?',(record['state'],json.dumps(public['arms'],allow_nan=False),identity))
            with self.lock:del self.active[identity]
