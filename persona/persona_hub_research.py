"""Public-only research batches and direct-Hub raw shards; explicit execution only.

Remote capture databases are opened read-only. Book rows never leave the host.
Publication checkpoints advance only after commit-pinned manifest/file verification.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import sqlite3
import subprocess
import sys
import tarfile
import time

REPO='Atlas3D/character-creation-big-five'
SCHEMA='character-creation.public-research.v1'
RAW_SCHEMA='character-creation.public-research-raw.v1'
START='<!-- character-creation-research:start -->'
END='<!-- character-creation-research:end -->'
IDENTITIES=('model_identity','runtime_identity','prepared_identity')
RAW_KEYS=set('mlp_input_last residual_last pre_mlp_last final_chunk_positions residual_l2_min residual_l2_max residual_l2_mean residual_l2_median search_peak search_peak_position search_sum_abs search_selected_position_values search_selected_position search_positions residual_l2 residual_l2_count prefill_positions prompt_token_ids generated_token_ids completion_trace completion_graph_kind completion_trace_columns model_identity prepared_identity runtime_identity request_sha256 stage runtime operation'.split())


def encode(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(value):return hashlib.sha256(value).hexdigest()
def digest(value):return sha(encode(value))
def policy_binding(policy):return digest({k:v for k,v in policy.items() if k!='reviewed_bundles'})
def file_hash(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_bytes(encode(value)+b'\n');tmp.replace(path)
def readonly(path):
    connection=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True,timeout=60);connection.row_factory=sqlite3.Row;return connection


def load_public(directory,policy_path):
    policy=json.loads(Path(policy_path).read_text());directory=Path(directory)
    if policy.get('schema')!='character-creation.public-research-policy.v1' or policy.get('classification')!='public' or policy.get('repo')!=REPO:raise ValueError('Explicit public research policy required')
    for name,key in [('scenarios.jsonl','scenarios_sha256'),('manifest.json','manifest_sha256')]:
        if file_hash(directory/name)!=policy.get(key):raise ValueError('Approved public snapshot differs: '+name)
    manifest=json.loads((directory/'manifest.json').read_text());rows=[json.loads(line) for line in (directory/'scenarios.jsonl').read_text().splitlines() if line]
    if manifest.get('export_sha256')!=policy['scenarios_sha256'] or manifest.get('records')!=len(rows) or manifest.get('model_identity')!=policy['model_identity']:raise ValueError('Public manifest does not bind scenarios')
    allowed={}
    for row in rows:
        if row.get('synthetic') is not True or row.get('validated_inventory') is not False or not isinstance(row.get('prompt'),str) or row['id'] in allowed:raise ValueError('Non-public or duplicate scenario')
        allowed[row['id']]={k:row[k] for k in ('id','prompt','normalized_prompt_sha256','response_sha256')}
    return policy,allowed


def approved_source(row,allowed,policy):
    """Return a whitelisted public source; never return private source metadata."""
    tier=row.get('tier');messages=row.get('messages');identity=row.get('id')
    if tier=='registered_big_five':
        item=allowed.get(identity)
        if item is None:return None
        provenance=row.get('provenance',{});source=provenance.get('source',{})
        if (messages!=[{'role':'user','content':item['prompt']}] or provenance.get('registered_id')!=identity
                or source.get('kind')!='synthetic_model_response' or source.get('normalized_prompt_sha256')!=item['normalized_prompt_sha256']
                or row.get('source_sha256')!=item['response_sha256']):raise ValueError('Approved prompt ID has mismatching text/source hash')
        return {'id':identity,'tier':tier,'prompt':item['prompt'],'prompt_sha256':sha(item['prompt'].encode()),'source_sha256':row['source_sha256']}
    if tier=='authored_emotion_contrast':
        p=row.get('provenance',{})
        if not policy.get('authored_emotion_source_sha256') or row.get('source_sha256')!=policy['authored_emotion_source_sha256'] or p.get('authored_graph_sha256')!=policy['authored_emotion_source_sha256']:return None
        if not isinstance(messages,list) or len(messages)!=1 or messages[0].get('role')!='user' or set(messages[0])!={'role','content'}:raise ValueError('Authored source message shape differs')
        prompt=messages[0]['content']
        if sha(prompt.encode())!=p.get('raw_text_sha256'):raise ValueError('Authored prompt hash differs')
        return {'id':identity,'tier':tier,'prompt':prompt,'prompt_sha256':sha(prompt.encode()),'source_sha256':row['source_sha256'],'design':{k:p[k] for k in ('feature','frame','arm','pair_family')}}
    return None


def pointer_hash(pointer):return {k:pointer[k] for k in ('sha256','request_sha256','model_identity','runtime_identity','prepared_identity','stage','operation') if k in pointer}


def public_capture(source,request,response,validation,policy):
    if request.get('messages')!=[{'role':'user','content':source['prompt']}] or response.get('model_identity')!=policy['model_identity'] or request.get('steering') or request.get('kv_control') or request.get('observation_only') is not True:raise ValueError('Capture source/model/intervention differs')
    search=response['activation_search'];tokens=search['tokens'];candidates=search['candidates']
    if len(candidates)>32 or any(type(r['layer']) is not int or not 0<=r['layer']<64 or type(r['neuron']) is not int or not 0<=r['neuron']<17408 for r in candidates):raise ValueError('Invalid candidate coordinates')
    result={'schema':'character-creation.public-activation.v1','kind':'prefill_activation','source':source,'identity':{k:response[k] for k in IDENTITIES},'request_sha256':response['activation_artifact']['request_sha256'],'request':{k:request[k] for k in ('messages','seed','max_tokens','temperature','observation_only','chat_template_kwargs')},'tokens':[{k:t[k] for k in ('id','text','position')} for t in tokens], 'candidates':[{k:r[k] for k in ('layer','neuron','peak','peak_position','mean_abs','values')} for r in candidates],'coverage':search['coverage'],'validation':validation,'artifacts':{'all_neuron':pointer_hash(response['activation_artifact']),'profile':pointer_hash(search['profile_artifact']),'provenance':pointer_hash(response['provenance_artifact']),'profile_provenance':pointer_hash(search['profile_provenance_artifact']),'combined_result':pointer_hash(search['result_artifact'])},'raw_arrays_in_this_json':False,'raw_arrays_index':'research_raw_manifest.json','measurement':'All encoded prefill positions including chat-template tokens. Full raw reductions are published in a separate automatic lane; this JSON contains top32 complete token profiles.'}
    result['id']=digest({'kind':result['kind'],'source_id':source['id'],'request_sha256':result['request_sha256']});encode(result);return result


def inspect_raw(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as arrays:
        if set(arrays.files)-RAW_KEYS:raise ValueError('Raw NPZ contains unapproved metadata/arrays')
        shapes={}
        for name in arrays.files:
            value=arrays[name]
            if value.dtype.kind=='O':raise ValueError('Pickled raw metadata forbidden')
            if value.dtype.kind in 'fc' and not np.isfinite(value).all():raise ValueError('Nonfinite raw numeric array')
            if value.dtype.kind in 'US':
                if name not in {*IDENTITIES,'request_sha256','stage','runtime','operation','completion_graph_kind','completion_trace_columns'}:raise ValueError('Unapproved raw string array')
                if any(re.search(r'(?:/home/|/mnt/|sk-[A-Za-z0-9]|hf_[A-Za-z0-9])',str(v)) for v in value.reshape(-1)):raise ValueError('Private path/credential-like raw metadata')
            shapes[name]={'shape':list(value.shape),'dtype':str(value.dtype),'bytes':value.nbytes}
        return shapes


def collect(envelope):
    """Execute on V100. Returns selected public records plus aggregate counts only."""
    root=Path(envelope['remote_root']).resolve();policy=envelope['policy'];allowed=envelope['allowed'];known=set(envelope.get('known',[]));known_sources=set(envelope.get('known_sources',[]));limit=envelope.get('limit',500)
    sys.path.insert(0,str(root));from persona_overnight_map import validate_result, validate_artifact
    inputs=root/'overnight_inputs.jsonl'
    if file_hash(inputs)!=policy['overnight_inputs_sha256']:raise ValueError('Frozen overnight input snapshot changed')
    approved={}
    for line in inputs.open():
        original=json.loads(line);source=approved_source(original,allowed,policy)
        if source:approved[original['id']]=(original,source)
    baseline_counts=[];baseline_done=False
    records=[];raw=[];baseline=envelope.get('baseline_root');overnight_limit=limit-max(1,limit//5) if baseline and (Path(baseline)/'results/baseline40k.sqlite').exists() else limit
    with readonly(root/'overnight_map.sqlite') as db:
        counts=[{'tier':r[0],'state':r[1],'count':r[2]} for r in db.execute("SELECT json_extract(row,'$.tier'),state,count(*) FROM tasks GROUP BY 1,2")]
        completed=list(db.execute("SELECT id FROM tasks WHERE state='complete' ORDER BY ordinal"));eligible=sum(row['id'] in approved for row in completed)
        for row in completed:
            # Offer the original fair baseline share, then reuse any unfilled
            # slots for new prefill records without rereading validated rows.
            if len(records)>=overnight_limit and not baseline_done:
                if baseline:
                    baseline_records,baseline_raw,baseline_counts=collect_baseline(Path(baseline),allowed,policy,known,limit-len(records),known_sources)
                    records.extend(baseline_records);raw.extend(baseline_raw)
                baseline_done=True
            if len(records)>=limit:break
            identity=row['id']
            if identity not in approved or 'prefill_activation:'+identity in known_sources:continue
            task=db.execute('SELECT row,request,response,validation FROM tasks WHERE id=?',(identity,)).fetchone();source_row,source=approved[identity]
            if json.loads(task['row'])!=source_row:raise ValueError('Stored task differs from frozen input')
            request=json.loads(task['request']);response=json.loads(task['response']);record_id=digest({'kind':'prefill_activation','source_id':identity,'request_sha256':response['activation_artifact']['request_sha256']})
            if record_id in known:continue
            if not task['validation']:raise ValueError('Completed task lacks saved validation')
            endpoint={k:response[k] for k in IDENTITIES}
            if endpoint['model_identity']!=policy['model_identity']:raise ValueError('Capture belongs to another model')
            validated=validate_result(response,request,endpoint,root)
            if json.loads(task['validation'])!=validated:raise ValueError('Stored validation differs from revalidated artifacts')
            record=public_capture(source,request,response,validated,policy);records.append(record)
            path=validate_artifact(response['activation_artifact'],root)
            raw.append({'id':record['id'],'kind':'prefill_activation','path':str(path),'sha256':file_hash(path),'source_id':identity,'prompt_sha256':source['prompt_sha256'],'identity':record['identity'],'request_sha256':record['request_sha256']})
    if baseline and not baseline_done:
        baseline_records,baseline_raw,baseline_counts=collect_baseline(Path(baseline),allowed,policy,known,limit-len(records),known_sources)
        records.extend(baseline_records);raw.extend(baseline_raw)
    terminal=bool(counts) and not any(r['state'] in ('pending','running') for r in counts)
    return {'records':records,'raw':raw,'summary':{'overnight':counts,'baseline':baseline_counts,'overnight_terminal':terminal,'raw_arrays_in_json':False,'allowed_public_scenarios':len(allowed),'eligible_completed_prefill_ledger_rows':eligible,'excluded_tiers':['book_diverse'],'collector_sha256':envelope['collector_sha256']}}


def collect_baseline(root,allowed,policy,known,limit,known_sources=()):
    ledger=root/'results/baseline40k.sqlite';corpus=root/'data/baseline40k.jsonl'
    if not ledger.exists():return [],[],[{'state':'not_available','count':0}]
    sys.path.insert(0,str(root));from persona_baseline_capture import validate
    manifest=json.loads(corpus.with_suffix('.manifest.json').read_text())
    if file_hash(corpus)!=manifest['corpus_sha256']:raise ValueError('Baseline corpus binding differs')
    sources={r['sample_id']:r for r in (json.loads(line) for line in corpus.open())};records=[];raw=[]
    with readonly(ledger) as db:
        counts=[{'state':r[0],'count':r[1]} for r in db.execute('SELECT state,count(*) FROM captures GROUP BY state')]
        if limit<=0:return [],[],counts
        binding=json.loads(db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()[0])
        if binding['corpus']!=manifest:raise ValueError('Baseline binding differs')
        for task in db.execute("SELECT * FROM captures WHERE state='complete' ORDER BY ordinal"):
            original=sources[task['sample_id']];p=original['prompt'];item=allowed.get(p.get('id'))
            if item is None or 'baseline_answer:'+item['id'] in known_sources:continue
            request=json.loads(task['payload']);response=json.loads(task['response']);rid=digest({'kind':'baseline_answer','source_id':item['id'],'request_sha256':response['activation_artifact']['request_sha256']})
            if rid in known:continue
            if p.get('prompt')!=item['prompt'] or original['normalized_hash']!=item['normalized_prompt_sha256'] or p.get('system') or p.get('synthetic') is not True or request.get('messages')!=[{'role':'user','content':item['prompt']}] or request.get('steering') or request.get('kv_control'):raise ValueError('Baseline public source/request differs')
            endpoint=next((e for e in binding['endpoints'] if all(e[k]==response[k] for k in IDENTITIES)),None)
            if endpoint is None or endpoint['model_identity']!=policy['model_identity']:raise ValueError('Baseline endpoint identity differs')
            checked=validate(response,endpoint,request)
            if not checked['complete']:raise ValueError('Baseline final answer is incomplete')
            record={'schema':'character-creation.public-baseline.v1','id':rid,'kind':'baseline_answer','source':{'id':item['id'],'sample_id':task['sample_id'],'prompt':item['prompt'],'prompt_sha256':sha(item['prompt'].encode())},'request_sha256':response['activation_artifact']['request_sha256'],'identity':{k:response[k] for k in IDENTITIES},'answer':response['choices'][0]['message']['content'],'finish_reason':'stop','usage':response.get('usage'),'request':{k:request[k] for k in ('messages','seed','max_tokens','temperature','chat_template_kwargs') if k in request},'artifacts':{'all_neuron':pointer_hash(response['activation_artifact']),'provenance':pointer_hash(response['provenance_artifact'])},'raw_arrays_in_this_json':False,'raw_arrays_index':'research_raw_manifest.json','coverage':{'final_prefill_all_neurons':True,'answer_token_activations':False}}
            encode(record);records.append(record);path=Path(response['activation_artifact']['path']).resolve()
            if path.parent!=root/'results' and path.parent!=root:raise ValueError('Baseline raw path outside reviewed capture folders')
            raw.append({'id':rid,'kind':'baseline_answer','path':str(path),'sha256':file_hash(path),'source_id':item['id'],'prompt_sha256':record['source']['prompt_sha256'],'identity':record['identity'],'request_sha256':record['request_sha256']})
            if len(records)>=limit:break
    return records,raw,counts


def reject_private_keys(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key.lower() in {'path','local_path','artifact_path','token','api_key','password','secret','endpoint','book_id','document_id','member'}:raise ValueError('Unapproved private field: '+key)
            reject_private_keys(item)
    elif isinstance(value,list):
        for item in value:reject_private_keys(item)
    elif isinstance(value,str) and re.search(r'(?:/home/|/mnt/|hf_[A-Za-z0-9]{15,})',value):raise ValueError('Private path/credential-like string')


def reviewed_bundles(root,policy):
    rows=[]
    for entry in policy.get('reviewed_bundles',[]):
        path=(root/entry['path']).resolve()
        if path.parent not in {(root/'results').resolve(),(root/'dataset').resolve()} or file_hash(path)!=entry['sha256']:raise ValueError('Reviewed bundle binding differs')
        value=json.loads(path.read_text())
        if value.get('schema')!='character-creation.public-research-bundle.v1' or value.get('kind') not in ('qualified_labels','coherence') or value['kind']!=entry['kind'] or value.get('state')!='complete' or value.get('model_identity')!=policy['model_identity'] or value.get('validation',{}).get('passed') is not True:raise ValueError('Bundle lacks reviewed completed validation')
        reject_private_keys(value);rows.append({'schema':value['schema'],'id':digest({'bundle_sha256':entry['sha256']}),'kind':value['kind'],'bundle_sha256':entry['sha256'],'bundle':value})
    return rows


def card_section(raw_records=0):
    return f'''{START}
## Research captures and evidence

The original `scenarios.jsonl`, its `manifest.json`, and default dataset configuration remain unchanged.
See [human guide](README_human.md) and [agent guide](README_agent.md).

- `research_assets.json` indexes the shared model configuration, authored probes, public presets and calibrated layer arrays with hashes.
- `research_manifest.json` indexes immutable JSONL batches, record IDs, SHA-256 hashes and exact publication coverage.
- Activation rows include public prompt alignment, encoded tokens and complete top-32 per-token profiles. They are **not** full all-neuron arrays.
- `research_raw_manifest.json` separately indexes automatically uploaded immutable tar shards of original validated NPZ arrays and entry manifests. Most recently observed verified raw coverage: **{raw_records:,} records**; consult that manifest for current coverage. Pending arrays must not be treated as uploaded.
- Raw prefill reductions cover every neuron and encoded prompt position, including template tokens; they are reductions, not a full neuron-by-token tensor. Baseline answer captures retain final-prefill arrays, not answer-token activation trajectories.
- Only the exact approved synthetic prompts and authored probes are exported. Book material and arbitrary user prompts are excluded; other tiers appear only as aggregate counts.
- Associations and calibrated probe precision are not causal neuron meanings or validated human personality scores.
{END}'''


def append_card(card,section):
    text=card.decode()
    if START in text:
        if text.count(START)!=1 or text.count(END)!=1:raise ValueError('Ambiguous managed card section')
        a=text.index(START);b=text.index(END,a)+len(END);text=text[:a]+section+text[b:]
    else:text=text.rstrip()+'\n\n'+section+'\n'
    return text.encode()


def read_guides(root: Path) -> dict[str, bytes]:
    return {name:(root/name if (root/name).exists() else root/'dataset'/name).read_bytes() for name in ('README_human.md','README_agent.md')}


class Hub:
    def __init__(self):
        import httpx
        from huggingface_hub import HfApi, set_client_factory
        from huggingface_hub.utils._http import default_client_factory
        def bounded_client() -> httpx.Client:
            client=default_client_factory()
            client.timeout=httpx.Timeout(connect=15,read=120,write=120,pool=60)
            return client
        set_client_factory(bounded_client)
        self.api=HfApi()
    def head(self):return self.api.repo_info(REPO,repo_type='dataset').sha
    def read(self,path,revision,missing=False):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        try:return Path(hf_hub_download(REPO,path,repo_type='dataset',revision=revision)).read_bytes()
        except EntryNotFoundError:
            if missing:return None
            raise
    def file_sha(self,path,revision):
        entries=self.api.get_paths_info(REPO,[path],repo_type='dataset',revision=revision)
        if len(entries)!=1:raise ValueError('Committed file absent')
        entry=entries[0];lfs=getattr(entry,'lfs',None)
        if lfs:return lfs.sha256 if hasattr(lfs,'sha256') else lfs['sha256']
        return sha(self.read(path,revision))
    def commit(self,files,parent,message):
        from huggingface_hub import CommitOperationAdd
        operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=value if isinstance(value,Path) else io.BytesIO(value)) for name,value in files.items()]
        return self.api.create_commit(REPO,repo_type='dataset',operations=operations,parent_commit=parent,commit_message=message).oid


class Publisher:
    def __init__(self,hub,checkpoint,policy,lane='metadata'):
        self.hub=hub;self.path=Path(checkpoint);self.policy=policy;self.lane=lane
        self.manifest_name='research_raw_manifest.json' if lane=='raw' else 'research_manifest.json'
        self.schema=RAW_SCHEMA if lane=='raw' else SCHEMA
        self.state=json.loads(self.path.read_text()) if self.path.exists() else {'policy_sha256':policy_binding(policy),'pending':None,'last_commit':None,'last_upload':0}
        if self.state['policy_sha256']!=policy_binding(policy):raise ValueError('Publisher policy changed; explicit new reviewed checkpoint required')
    def save(self):atomic(self.path,self.state)
    def snapshot(self):
        revision=self.hub.head()
        if sha(self.hub.read('manifest.json',revision))!=self.policy['manifest_sha256'] or self.hub.file_sha('scenarios.jsonl',revision)!=self.policy['scenarios_sha256']:raise ValueError('Original public subset differs on Hub')
        raw=self.hub.read(self.manifest_name,revision,missing=True)
        manifest=json.loads(raw) if raw else {'schema':self.schema,'policy_sha256':policy_binding(self.policy),'source_scenarios_sha256':self.policy['scenarios_sha256'],'batches':[],'records':0}
        if manifest.get('schema')!=self.schema or manifest.get('policy_sha256')!=policy_binding(self.policy):raise ValueError('Hub research manifest policy/schema differs')
        return revision,manifest
    def recover(self):
        pending=self.state.get('pending')
        if not pending:return
        for attempt in range(3):
            revision,_=self.snapshot();actual=self.hub.read(self.manifest_name,revision,missing=True)
            if actual and sha(actual)==pending['manifest_sha256']:
                self.verify(revision,pending);return
            if revision!=pending['parent']:
                if (sha(actual) if actual else None)!=pending['previous_manifest_sha256']:
                    raise ValueError('This publication lane changed after uncertain commit; manual reconciliation required')
                if pending.get('base_card_sha256') and sha(self.hub.read('README.md',revision))!=pending['base_card_sha256']:
                    raise ValueError('Dataset card changed after uncertain commit; review required')
                pending['parent']=revision;self.save()
            # Retry only an explicit uncommitted CAS conflict, rechecking the lane
            # and exact immutable bytes each time. Lost replies still exit for recovery.
            files={name:Path(path) for name,path in pending['files'].items()}
            if any(file_hash(path)!=pending['hashes'][name] for name,path in files.items()):raise ValueError('Pending local upload bytes changed')
            try:
                commit=self.hub.commit(files,pending['parent'],'Publish verified research '+self.lane+' batch')
            except Exception as error:
                if getattr(getattr(error,'response',None),'status_code',None)!=412 or attempt==2:raise
                continue
            self.verify(commit,pending);return
    def verify(self,revision,pending):
        if sha(self.hub.read(self.manifest_name,revision))!=pending['manifest_sha256']:raise ValueError('Remote manifest verification failed')
        for name,expected in pending['hashes'].items():
            if self.hub.file_sha(name,revision)!=expected:raise ValueError('Remote committed file hash differs: '+name)
        self.state.update(pending=None,last_commit=revision,last_upload=time.time());self.save()
    def publish(self,files,manifest,parent,directory):
        directory=Path(directory);files={**files,self.manifest_name:encode(manifest)+b'\n'};paths={};hashes={}
        for name,value in files.items():
            if name not in {'README.md','README_human.md','README_agent.md','research_manifest.json','research_raw_manifest.json'} and not re.fullmatch(r'research_(?:batch_[0-9a-f]{64}\.jsonl|raw_[0-9a-f]{64}\.tar|raw_entries_[0-9a-f]{64}\.json)',name):raise ValueError('Forbidden publication target')
            path=value if isinstance(value,Path) else directory/name
            if not isinstance(value,Path):path.write_bytes(value)
            paths[name]=str(path.resolve());hashes[name]=file_hash(path)
        previous=self.hub.read(self.manifest_name,parent,missing=True)
        pending={'previous_manifest_sha256':sha(previous) if previous else None,'base_card_sha256':sha(self.hub.read('README.md',parent)) if 'README.md' in files else None,'parent':parent,'files':paths,'hashes':hashes,'manifest_sha256':hashes[self.manifest_name]};self.state['pending']=pending;self.save()
        self.recover();return self.state['last_commit']


def metadata_batch(records):
    records=sorted(records,key=lambda r:r['id']);body=b''.join(encode(r)+b'\n' for r in records);hash=sha(body)
    return 'research_batch_'+hash+'.jsonl',body,{'file':'research_batch_'+hash+'.jsonl','sha256':hash,'records':len(records),'record_ids':[r['id'] for r in records],'kinds':sorted({r['kind'] for r in records}),'source_keys':sorted(r['kind']+':'+r['source']['id'] for r in records if 'source' in r)}


def raw_shard(records,directory):
    """NPZ bytes unchanged; reproducible flat tar with explicit entry manifest."""
    directory=Path(directory);records=sorted(records,key=lambda r:r['id']);entries=[];temp=directory/'research_raw_pending.tar'
    with tarfile.open(temp,'w',format=tarfile.USTAR_FORMAT) as archive:
        for record in records:
            if not re.fullmatch('[0-9a-f]{64}',record['id']):raise ValueError('Raw entry ID must be a SHA256 hex identifier')
            path=Path(record['path'])
            if file_hash(path)!=record['sha256']:raise ValueError('Raw source changed before archive')
            arrays=inspect_raw(path);name=record['id']+'.npz';header=tarfile.TarInfo(name);header.size=path.stat().st_size;header.mode=0o644;header.mtime=0
            with path.open('rb') as stream:archive.addfile(header,stream)
            if file_hash(path)!=record['sha256']:raise ValueError('Raw source changed during archive')
            entries.append({k:record[k] for k in ('id','kind','source_id','prompt_sha256','identity','request_sha256')}|{'entry':name,'sha256':record['sha256'],'original_sha256':record['sha256'],'metadata_sanitized':False,'bytes':header.size,'arrays':arrays})
    hash=file_hash(temp);name='research_raw_'+hash+'.tar';path=directory/name
    if path.exists():
        if file_hash(path)!=hash:raise ValueError('Immutable archive differs')
        temp.unlink()
    else:temp.replace(path)
    entry_name='research_raw_entries_'+hash+'.json';entry_bytes=encode({'schema':RAW_SCHEMA,'archive':name,'archive_sha256':hash,'entries':entries})+b'\n'
    metadata={'file':name,'sha256':hash,'entry_manifest':entry_name,'entry_manifest_sha256':sha(entry_bytes),'records':len(entries),'record_ids':[r['id'] for r in entries],'bytes':path.stat().st_size,'source_keys':sorted(r['kind']+':'+r['source_id'] for r in entries)}
    return {name:path,entry_name:entry_bytes},metadata


def ssh_collect(args,envelope):
    code=Path(__file__).read_text();command=shlex.join([args.remote_python,'-c',code,'--collect']);envelope={**envelope,'collector_sha256':sha(code.encode())}
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none',args.remote_host,command],input=encode(envelope),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1800)
    if result.returncode:raise RuntimeError('Read-only collector failed: '+result.stderr.decode()[-2000:])
    value=json.loads(result.stdout);value.pop('raw',None);return value


def watch_delay(captured, limit):
    """A full batch warrants one prompt next check; no progress stays idle.

    Exactly-full exhaustion may cause one extra five-second check, then the
    empty result restores the idle interval. Errors still exit via failure_hook.
    """
    return 5 if limit>0 and len(captured.get('records',()))>=limit else 300


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--collect',action='store_true');parser.add_argument('--root',type=Path);parser.add_argument('--policy',type=Path);parser.add_argument('--public-dir',type=Path);parser.add_argument('--remote-host',default='v100');parser.add_argument('--remote-python',default='/mnt/storage16tb/flare/venv/bin/python');parser.add_argument('--remote-root',default='/mnt/storage16tb/Character Creation/activation-search');parser.add_argument('--baseline-root');parser.add_argument('--watch',action='store_true');parser.add_argument('--raw-watch',action='store_true');parser.add_argument('--publish',action='store_true');parser.add_argument('--force',action='store_true');args=parser.parse_args()
    if args.collect:print(json.dumps(collect(json.load(sys.stdin)),allow_nan=False));return
    if not args.root or not args.policy:parser.error('--root and --policy required')
    root=args.root.resolve();directory=args.public_dir or root/'dataset';policy,allowed=load_public(directory,args.policy)
    if not args.publish:print(json.dumps({'state':'validated_only','public_scenarios':len(allowed),'policy_sha256':policy_binding(policy)}));return
    raw=args.raw_watch;output=root if raw else root/'dataset';checkpoint=root/'persona_hub_raw_checkpoint.json' if raw else root/'results/persona_hub_research_checkpoint.json'
    lock=checkpoint.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    publisher=Publisher(Hub(),checkpoint,policy,'raw' if raw else 'metadata');pending_rows={};last_summary=None
    status_path=output/'persona_hub_raw_status.json' if raw else root/'results/persona_hub_research_status.json'
    previous_hook=sys.excepthook
    def failure_hook(kind,error,tb):
        atomic(status_path,{'updated':time.time(),'state':'error','lane':'raw' if raw else 'metadata','error':str(error),'pending_commit':publisher.state.get('pending') is not None,'last_verified_commit':publisher.state['last_commit']});previous_hook(kind,error,tb)
    sys.excepthook=failure_hook
    while True:
        fresh_policy=json.loads(args.policy.read_text())
        if policy_binding(fresh_policy)!=policy_binding(policy):raise ValueError('Public source policy changed')
        policy=fresh_policy;publisher.policy=policy
        publisher.recover();revision,manifest=publisher.snapshot();known={identity for b in manifest['batches'] for identity in b['record_ids']}
        envelope={'policy':policy,'allowed':allowed,'known':list(known|set(pending_rows)),'known_sources':[k for b in manifest['batches'] for k in b.get('source_keys',[])]+[r['kind']+':'+r['source']['id'] for r in pending_rows.values() if 'source' in r],'limit':32 if raw else 500,'remote_root':args.remote_root,'baseline_root':args.baseline_root,'collector_sha256':file_hash(__file__)}
        captured=collect(envelope) if raw else ssh_collect(args,envelope)
        if raw:
            if captured['raw']:
                files,batch=raw_shard(captured['raw'],output);manifest['batches'].append(batch);manifest['records']=sum(b['records'] for b in manifest['batches']);manifest['coverage']='Only listed entry IDs have verified uploaded raw arrays; all other eligible captures remain pending.';manifest['summary']=captured['summary'];manifest['pending_completed_ledger_rows_upper_bound']=max(0,captured['summary']['eligible_completed_prefill_ledger_rows']+sum(r['count'] for r in captured['summary']['baseline'] if r['state']=='complete')-manifest['records']);publisher.publish(files,manifest,revision,output)
        else:
            for row in captured['records']+reviewed_bundles(root,policy):
                if row['id'] not in known:pending_rows[row['id']]=row
            summary=captured['summary'];raw_manifest=publisher.hub.read('research_raw_manifest.json',revision,missing=True);raw_count=json.loads(raw_manifest)['records'] if raw_manifest else 0;summary['verified_raw_uploaded_records']=raw_count;summary['raw_pending_completed_ledger_rows_upper_bound']=max(0,summary['eligible_completed_prefill_ledger_rows']+sum(r['count'] for r in summary['baseline'] if r['state']=='complete')-raw_count)
            guides=read_guides(root);guide_hashes={name:sha(value) for name,value in guides.items()}
            due=args.force or len(pending_rows)>=500 or time.time()-publisher.state['last_upload']>=900 or summary['overnight_terminal']
            if due and (pending_rows or summary!=manifest.get('summary') or guide_hashes!=manifest.get('guides')):
                files={};rows=list(pending_rows.values())
                if rows:
                    name,body,batch=metadata_batch(rows);files[name]=body;manifest['batches'].append(batch)
                manifest.update(records=sum(b['records'] for b in manifest['batches']),guides=guide_hashes,summary=summary,raw_manifest='research_raw_manifest.json',raw_arrays_in_json=False)
                files['README.md']=append_card(publisher.hub.read('README.md',revision),card_section(raw_count))
                files.update(guides)
                publisher.publish(files,manifest,revision,output);pending_rows.clear();last_summary=summary
        atomic(output/'persona_hub_raw_status.json' if raw else root/'results/persona_hub_research_status.json',{'updated':time.time(),'lane':'raw' if raw else 'metadata','state':'watching' if (args.watch or raw) else 'complete','published_records':manifest['records'],'pending_buffered':len(pending_rows),'last_verified_commit':publisher.state['last_commit'],'source_summary':captured['summary']})
        if not args.watch and not raw:return
        time.sleep(watch_delay(captured,envelope['limit']))

if __name__=='__main__':main()
