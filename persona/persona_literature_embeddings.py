"""Pinned local BGE retrieval embeddings; similarity is not neural causality."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'BAAI/bge-small-en-v1.5'
REVISION = '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
SCHEMA = 'literature-bge-cls-v1'
MODEL_PATH = Path.home()/'.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots'/REVISION
METHOD = '512-token windows, 64-token overlap; normalized CLS per window; normalized mean across windows'

def connect(database):
    c=sqlite3.connect(database,timeout=60); c.row_factory=sqlite3.Row
    c.execute('PRAGMA synchronous=NORMAL');return c

def initialize(database):
    with connect(database) as c:
        c.executescript('''CREATE TABLE IF NOT EXISTS literature_embeddings(unit_id TEXT,model_revision TEXT,source_sha256 TEXT,dimensions INTEGER,vector BLOB,windows INTEGER,provenance TEXT,PRIMARY KEY(unit_id,model_revision));
        CREATE TABLE IF NOT EXISTS literature_similarity(unit_id TEXT,neighbor_id TEXT,model_revision TEXT,score REAL,rank INTEGER,PRIMARY KEY(unit_id,neighbor_id,model_revision));
        CREATE TABLE IF NOT EXISTS literature_embedding_runs(id TEXT PRIMARY KEY,state TEXT,body TEXT);''')

def aggregate(vectors):
    values=np.asarray(vectors,dtype=np.float32)
    if values.ndim!=2 or not len(values) or not np.isfinite(values).all():raise ValueError('Invalid embeddings')
    result=values.mean(axis=0);norm=float(np.linalg.norm(result))
    if norm<=0:raise ValueError('Zero embedding')
    return result/norm

def neighbors(database,unit_id,limit=12):
    if not isinstance(unit_id,str) or type(limit)is not int or not 1<=limit<=50:raise ValueError('Invalid neighbors request')
    with connect(database) as c:
        exists=c.execute("SELECT 1 FROM sqlite_master WHERE name='literature_similarity'").fetchone()
        if not exists:return {'available':False,'rows':[]}
        rows=c.execute('''SELECT s.neighbor_id AS id,s.score,u.text,u.book_id,u.split,b.title AS book_title FROM literature_similarity s JOIN units u ON u.id=s.neighbor_id JOIN books b ON b.id=u.book_id WHERE s.unit_id=? AND s.model_revision=? ORDER BY s.rank LIMIT ?''',(unit_id,REVISION,limit)).fetchall()
    return {'available':bool(rows),'rows':[dict(r) for r in rows],'model':MODEL,'revision':REVISION,'method':METHOD,'claim':'Semantic retrieval similarity; not theme labels or Qwen neuron topology.'}

def build(database,device,batch_size):
    import torch
    from transformers import AutoModel,AutoTokenizer
    if device=='cuda':
        import os,subprocess
        gpu='GPU-d739b9c5-bfbb-e95a-bbf1-7122f38c2cf1'
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu:raise ValueError('Embedding job requires explicitly leased John')
        board=json.loads(subprocess.check_output(['/home/orwel/.local/bin/gpu-workboard','status','--json'],text=True))
        if not any(j.get('pid')==os.getppid() and j.get('assigned_uuid')==gpu and j.get('status')=='running' for j in board['jobs']):raise ValueError('Missing running parent GPU lease')
    import signal
    stopping = {'value':False}
    def request_stop(*_): stopping['value']=True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    def checkpoint_stop():
        if stopping['value']:
            with connect(database) as c:
                c.execute('UPDATE literature_embedding_runs SET state=? WHERE id=?', ('interrupted',REVISION))
            raise SystemExit(75)
    initialize(database)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    model=AutoModel.from_pretrained(MODEL_PATH,local_files_only=True).eval().to(device)
    if device=='cuda':model=model.half()
    import os
    torch.set_num_threads(max(1,len(os.sched_getaffinity(0))))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in MODEL_PATH.iterdir() if p.name in ('model.safetensors','config.json','tokenizer.json','tokenizer_config.json','vocab.txt')}
    provenance={'schema':SCHEMA,'model':MODEL,'revision':REVISION,'hashes':hashes,'method':METHOD,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'torch':torch.__version__}
    with connect(database) as c:
        rows=c.execute('''SELECT u.id,u.text FROM units u LEFT JOIN literature_embeddings e ON e.unit_id=u.id AND e.model_revision=? WHERE u.id=u.canonical_id AND e.unit_id IS NULL ORDER BY u.book_id,u.ordinal''',(REVISION,)).fetchall()
        c.execute('INSERT OR REPLACE INTO literature_embedding_runs VALUES(?,?,?)',(REVISION,'running',json.dumps({**provenance,'pending':len(rows),'started':time.time()})))
    # Tokenization happens on CPU; each source window is fully covered, no silent tail truncation.
    completed=0
    for begin in range(0,len(rows),batch_size):
        checkpoint_stop()
        group=rows[begin:begin+batch_size]
        encoded=tokenizer([r['text'] for r in group],truncation=True,max_length=512,stride=64,return_overflowing_tokens=True,padding=True,return_tensors='pt')
        mapping=encoded.pop('overflow_to_sample_mapping').tolist();vectors=[]
        with torch.inference_mode():
            for start in range(0,len(mapping),batch_size):
                inputs={k:v[start:start+batch_size].to(device) for k,v in encoded.items()}
                values=torch.nn.functional.normalize(model(**inputs).last_hidden_state[:,0].float(),dim=1)
                vectors.extend(values.cpu().numpy())
        buckets=[[] for _ in group]
        for index,vec in zip(mapping,vectors):buckets[index].append(vec)
        with connect(database) as c:
            for row,parts in zip(group,buckets):
                vector=aggregate(parts);sha=hashlib.sha256(row['text'].encode()).hexdigest()
                current=c.execute('SELECT text FROM units WHERE id=?',(row['id'],)).fetchone()
                if not current or hashlib.sha256(current[0].encode()).hexdigest()!=sha:raise ValueError('Source changed during embedding')
                c.execute('INSERT OR IGNORE INTO literature_embeddings VALUES(?,?,?,?,?,?,?)',(row['id'],REVISION,sha,len(vector),vector.astype('<f4').tobytes(),len(parts),json.dumps(provenance)))
        completed+=len(group);print(json.dumps({'stage':'embed','completed':completed,'pending_at_start':len(rows)}),flush=True)
    # Exact bounded-block cosine neighbors. Never allocate an all-pairs matrix.
    with connect(database) as c:
        allrows=c.execute('SELECT unit_id,vector,dimensions FROM literature_embeddings WHERE model_revision=? ORDER BY unit_id',(REVISION,)).fetchall()
    checkpoint_stop()
    if not allrows: raise ValueError('No canonical source windows to embed')
    matrix=np.stack([np.frombuffer(r['vector'],dtype='<f4') for r in allrows]);ids=[r['unit_id'] for r in allrows]
    tensor=torch.from_numpy(matrix).to(device);k=min(9,len(ids))
    with torch.inference_mode(),connect(database) as c:
        for begin in range(0,len(ids),512):
            checkpoint_stop()
            sims=tensor[begin:begin+512]@tensor.T
            values,indexes=torch.topk(sims,k=k,dim=1)
            out=[]
            for offset,(scores,idxs) in enumerate(zip(values.cpu().tolist(),indexes.cpu().tolist())):
                unit=ids[begin+offset];rank=0
                for score,index in zip(scores,idxs):
                    if ids[index]==unit:continue
                    rank+=1;out.append((unit,ids[index],REVISION,float(score),rank))
                    if rank==8:break
            c.executemany('INSERT OR REPLACE INTO literature_similarity VALUES(?,?,?,?,?)',out);c.commit()
            print(json.dumps({'stage':'neighbors','completed':min(begin+512,len(ids)),'total':len(ids)}),flush=True)
        c.execute('UPDATE literature_embedding_runs SET state=?,body=? WHERE id=?',('complete',json.dumps({**provenance,'units':len(ids),'finished':time.time()}),REVISION))
    return {'units':len(ids),'model':MODEL,'revision':REVISION}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',action='store_true');p.add_argument('--database',type=Path,default=ROOT/'results/persona_literature.sqlite');p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--batch-size',type=int,default=256)
    a=p.parse_args()
    if not a.build or not 1<=a.batch_size<=1024:p.error('Use --build and batch size1..1024')
    print(json.dumps(build(a.database,a.device,a.batch_size)),flush=True)
if __name__=='__main__':main()
