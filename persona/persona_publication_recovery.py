"""Finite CAS publication recovery; never imports or controls inference workers."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from persona_hub_research import Hub, REPO, reject_private_keys

REPORTS={'persona_preset_overnight_public.json':'character-creation.preset-overnight-public.v1',
         'persona_perturbation_mapping_public.json':'character-creation.perturbation-mapping-public.v1'}
GUIDES=('README_human.md','README_agent.md')
DEADLINE=1793113500

class BoundedHub(Hub):
    def history(self, head, parent):
        # One bounded API page. If concurrent writers exceed the page, fail closed.
        from huggingface_hub.utils import get_session
        response=get_session().get(f'{self.api.endpoint}/api/datasets/{REPO}/commits/{head}',
            headers=self.api._build_hf_headers(),params={'limit':100},timeout=60)
        response.raise_for_status();items=response.json()
        if not isinstance(items,list) or len(items)>100:raise ValueError('History page exceeds bound')
        ids=[]
        for item in items:
            oid=item['id']
            if oid==parent:return ids
            ids.append(oid)
        raise ValueError('Fixed parent outside bounded reconciliation history')

def digest(data):return hashlib.sha256(data).hexdigest()
def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'))

def snapshot(root):
    root=Path(root);files={};markers={}
    for name,schema in REPORTS.items():
        raw=(root/'results'/name).read_bytes();v=json.loads(raw)
        if v.get('schema')!=schema or not isinstance(v.get('rows'),list) or not isinstance(v.get('terminal_jobs'),int):raise ValueError('Malformed approved report')
        reject_private_keys(v)
        if v['terminal_jobs']<0 or sum(v['counts'].get(k,0) for k in ('complete','incomplete','failed','interrupted','cancelled'))!=v['terminal_jobs']:raise ValueError('Terminal count mismatch')
        files[name]=raw;markers[name]={'terminal':v['terminal_jobs'],'reviews':len(v.get('reviews',[])),
            'evaluations':digest(encoded(v.get('evaluations',{})).encode()),'final':bool(v.get('final',False))}
    for name in GUIDES:
        raw=(root/name).read_bytes()
        if not raw.strip():raise ValueError('Empty guide')
        reject_private_keys(raw.decode());files[name]=raw
    return files,markers

def due(previous,current,final=False):
    if not any(v['terminal'] for v in current.values()):return False
    if previous is None:return True
    if final:return current!={k:v for k,v in previous.items() if k!='_final'} or not previous.get('_final',False)
    return any(v['terminal']//48>previous[name]['terminal']//48 or v['reviews']//8>previous[name]['reviews']//8 or
        v['evaluations']!=previous[name]['evaluations'] or (v['final'] and not previous[name]['final']) for name,v in current.items())

class Recovery:
    def __init__(self,path,hub,lineage):
        self.db=sqlite3.connect(path);self.hub=hub
        self.db.executescript('CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE IF NOT EXISTS snapshots(id INTEGER PRIMARY KEY, parent TEXT, oid TEXT, state TEXT, markers TEXT, error TEXT); CREATE TABLE IF NOT EXISTS files(snapshot INTEGER,name TEXT,data BLOB,sha256 TEXT,PRIMARY KEY(snapshot,name)); CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,snapshot INTEGER,event TEXT,data TEXT);')
        old=self.db.execute("SELECT value FROM meta WHERE key='lineage'").fetchone()
        if old and old[0]!=encoded(lineage):raise ValueError('Recovery lineage changed')
        with self.db:self.db.execute("INSERT OR IGNORE INTO meta VALUES('lineage',?)",(encoded(lineage),))
    def event(self,i,name,data):self.db.execute('INSERT INTO events(snapshot,event,data) VALUES(?,?,?)',(i,name,encoded(data)))
    def pending(self):return self.db.execute("SELECT id,parent,oid,state,markers FROM snapshots WHERE state!='verified' ORDER BY id LIMIT 1").fetchone()
    def previous(self):
        row=self.db.execute("SELECT markers FROM snapshots WHERE state='verified' ORDER BY id DESC LIMIT 1").fetchone()
        return json.loads(row[0]) if row else None
    def freeze(self,files,markers):
        if self.pending():raise ValueError('Pending snapshot must resolve first')
        if set(files)!=set(REPORTS)|set(GUIDES):raise ValueError('Upload allowlist differs')
        parent=self.hub.head()
        with self.db:
            i=self.db.execute("INSERT INTO snapshots(parent,state,markers) VALUES(?,'intent',?)",(parent,encoded(markers))).lastrowid
            for name,data in files.items():self.db.execute('INSERT INTO files VALUES(?,?,?,?)',(i,name,data,digest(data)))
            self.event(i,'durable_intent',{'parent':parent,'hashes':{n:digest(b) for n,b in files.items()}})
        return i
    def matches(self,files,oid):return all(self.hub.read(n,oid,missing=True)==b for n,b in files.items())
    def step(self):
        row=self.pending()
        if not row:return True
        i,parent,oid,state,markers=row
        files={n:bytes(b) for n,b,h in self.db.execute('SELECT name,data,sha256 FROM files WHERE snapshot=?',(i,))}
        for n,b,h in self.db.execute('SELECT name,data,sha256 FROM files WHERE snapshot=?',(i,)):
            if digest(bytes(b))!=h:raise ValueError('Durable payload corrupted')
        try:
            if oid:
                if not self.matches(files,oid):raise ValueError('Pinned committed bytes differ')
                with self.db:
                    self.db.execute("UPDATE snapshots SET state='verified',error=NULL WHERE id=?",(i,));self.event(i,'verified',{'oid':oid})
                return True
            if state=='uncertain':
                head=self.hub.head()
                candidates=[head] if head==parent else self.hub.history(head,parent)
                matched=next((rev for rev in candidates if self.matches(files,rev)),None)
                if matched:
                    head=matched
                    with self.db:
                        self.db.execute("UPDATE snapshots SET oid=?,state='committed' WHERE id=?",(head,i));self.event(i,'reconciled_current',{'oid':head})
                    return False
            # A crash after this durable event is uncertain, including before POST.
            with self.db:
                self.db.execute("UPDATE snapshots SET state='uncertain' WHERE id=?",(i,));self.event(i,'commit_intent',{'parent':parent})
            try:oid=self.hub.commit(files,parent,'Recover finite public persona reports')
            except Exception as exc:
                status=getattr(getattr(exc,'response',None),'status_code',None)
                if status==412:
                    head=self.hub.head()
                    candidates=[head] if head==parent else self.hub.history(head,parent)
                    matched=next((rev for rev in candidates if self.matches(files,rev)),None)
                    if matched:
                        head=matched
                        with self.db:self.db.execute("UPDATE snapshots SET oid=?,state='committed' WHERE id=?",(head,i));self.event(i,'reconciled_412',{'oid':head})
                    else:
                        with self.db:self.db.execute("UPDATE snapshots SET parent=?,state='intent' WHERE id=?",(head,i));self.event(i,'explicit_412_parent_refresh',{'old':parent,'new':head})
                else:raise
                return False
            with self.db:
                self.db.execute("UPDATE snapshots SET oid=?,state='committed',error=NULL WHERE id=?",(str(oid),i));self.event(i,'commit_returned',{'oid':str(oid)})
            return False
        except Exception as exc:
            with self.db:self.db.execute('UPDATE snapshots SET error=? WHERE id=?',(type(exc).__name__,i));self.event(i,'deferred',{'error_type':type(exc).__name__})
            return False

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--lineage',type=Path,required=True);p.add_argument('--until',type=float,default=DEADLINE);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    if not math.isfinite(a.until) or a.until>DEADLINE or (not a.validate_only and a.until<=time.time()):raise ValueError('Finite future deadline must not exceed authorized 08:05 cutoff')
    raw=a.lineage.read_bytes();lineage=json.loads(raw)
    if not isinstance(lineage,dict) or not lineage:raise ValueError('Missing reconciliation lineage')
    files,markers=snapshot(a.root)
    if a.validate_only:print(encoded({'files':{n:digest(b) for n,b in files.items()},'markers':markers,'lineage_sha256':digest(raw)}));return
    lock=(a.root/'results/persona_publication_recovery.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    recovery=Recovery(a.root/'results/persona_publication_recovery.sqlite',BoundedHub(),{'sha256':digest(raw),'publisher_source_sha256':digest(Path(__file__).read_bytes()),'evidence':lineage})
    while time.time()<a.until:
        try:
            if recovery.pending():recovery.step()
            else:
                files,markers=snapshot(a.root);final=time.time()>=a.until-240
                if due(recovery.previous(),markers,final):
                    if final:markers['_final']=True
                    recovery.freeze(files,markers)
        except Exception as exc:print(encoded({'state':'deferred','error_type':type(exc).__name__}),flush=True)
        time.sleep(min(15,max(0,a.until-time.time())))

if __name__=='__main__':main()
