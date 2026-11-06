#!/usr/bin/env python3
"""Shared durable per-port claim barrier for cooperating dispatchers.

The lock is held from permission check until the caller commits its own ledger
claim. The database and all lock files must be on the same local filesystem and
all participating processes must use this exact authority and port mapping.
"""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import sqlite3
import time

PARTICIPANTS=('lace','threshold')
BLOCKING=('preparing','submitted','uncertain','running')


def _connect(path):
    db=sqlite3.connect(path,timeout=30)
    db.execute('PRAGMA busy_timeout=30000');db.execute('PRAGMA synchronous=FULL')
    return db


def initialize(path, ports):
    """Create slots once; existing pause epochs and acknowledgements are untouched."""
    db=_connect(path)
    try:
        db.executescript('''CREATE TABLE IF NOT EXISTS slots(port INTEGER PRIMARY KEY,state TEXT NOT NULL CHECK(state IN ('active','paused')),epoch INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS acknowledgements(port INTEGER NOT NULL,epoch INTEGER NOT NULL,participant TEXT NOT NULL,at REAL NOT NULL,PRIMARY KEY(port,epoch,participant));
          CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,port INTEGER NOT NULL,epoch INTEGER NOT NULL,event TEXT NOT NULL,at REAL NOT NULL);''')
        db.executemany("INSERT OR IGNORE INTO slots VALUES(?,'active',0)",[(int(p),) for p in ports]);db.commit()
    finally:db.close()


@contextmanager
def _lock(path,port):
    if type(port) is not int or not 1<=port<=65535:raise ValueError('Invalid port')
    lockfile=Path(path).parent/('claim_guard_'+str(port)+'.lock')
    fd=os.open(lockfile,os.O_CREAT|os.O_RDWR,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)


def _slot(db,port):
    row=db.execute('SELECT state,epoch FROM slots WHERE port=?',(port,)).fetchone()
    if not row:raise ValueError('Unregistered port; refusing claim')
    return row


@contextmanager
def claim(path,port):
    """Caller keeps this context open through its ledger claim COMMIT."""
    with _lock(path,port):
        db=_connect(path)
        try:state,epoch=_slot(db,port)
        finally:db.close()
        yield state=='active',epoch


def pause(path,port):
    """Persist pause before returning; repeat is idempotent for its epoch."""
    with _lock(path,port):
        db=_connect(path)
        try:
            db.execute('BEGIN IMMEDIATE');state,epoch=_slot(db,port)
            if state=='active':
                epoch+=1;db.execute("UPDATE slots SET state='paused',epoch=? WHERE port=?",(epoch,port));db.execute("INSERT INTO events(port,epoch,event,at) VALUES(?,?,'paused',?)",(port,epoch,time.time()))
            db.commit();return epoch
        except BaseException:db.rollback();raise
        finally:db.close()


def _empty_inspection(inspect):
    """Caller must inspect every relevant ledger, including legacy records."""
    counts=inspect()
    if not isinstance(counts,dict) or set(counts)!=set(BLOCKING) or any(type(v) is not int or v<0 for v in counts.values()):raise ValueError('Incomplete ledger inspection')
    return not any(counts.values())


def acknowledge(path,port,epoch,participant,inspect):
    if participant not in PARTICIPANTS:raise ValueError('Unknown participant')
    with _lock(path,port):
        db=_connect(path)
        try:
            db.execute('BEGIN IMMEDIATE');state,current=_slot(db,port)
            if state!='paused' or current!=epoch:raise ValueError('Stale or inactive pause epoch')
            if not _empty_inspection(inspect):raise ValueError('Outstanding or uncertain work')
            db.execute('INSERT OR IGNORE INTO acknowledgements VALUES(?,?,?,?)',(port,epoch,participant,time.time()));db.commit()
        except BaseException:db.rollback();raise
        finally:db.close()


class HeldPort:
    def __init__(self,path,port):self.path=path;self.port=port;self.live=True


@contextmanager
def hold(path,port):
    """Hold the shared port lock across admission changes in another ledger."""
    with _lock(path,port):
        token=HeldPort(path,port)
        try:yield token
        finally:token.live=False


def read_held(token):
    """Read durable state and acknowledgements while caller holds the lock."""
    if not isinstance(token,HeldPort) or not token.live:raise ValueError('Port lock is not held')
    db=_connect(token.path)
    try:
        state,epoch=_slot(db,token.port)
        acks={x[0] for x in db.execute('SELECT participant FROM acknowledgements WHERE port=? AND epoch=?',(token.port,epoch))}
        return {'state':state,'epoch':epoch,'acknowledgements':acks}
    finally:db.close()


def read(path,port):
    with hold(path,port) as token:return read_held(token)


def inspect_lace(ledger,endpoint):
    """Count all unresolved lace jobs/batches, including legacy rows without a port."""
    db=sqlite3.connect(Path(ledger).resolve().as_uri()+'?mode=ro',uri=True)
    try:
        db.execute('BEGIN');counts={state:0 for state in BLOCKING}
        for table in ('jobs','batches'):
            rows=db.execute("SELECT state,count(*) FROM "+table+" WHERE (endpoint=? OR endpoint IS NULL) AND state IN ('preparing','submitted','uncertain','running') GROUP BY state",(endpoint,)).fetchall()
            for state,n in rows:counts[state]+=n
        return counts
    finally:db.close()


def ready_held(token,epoch,inspections):
    if set(inspections)!=set(PARTICIPANTS):raise ValueError('Both dispatcher inspections required')
    current=read_held(token)
    return current['state']=='paused' and current['epoch']==epoch and current['acknowledgements']==set(PARTICIPANTS) and all(_empty_inspection(inspections[p]) for p in PARTICIPANTS)


def ready(path,port,epoch,inspections):
    """Independently inspect both ledgers under the same port lock."""
    with hold(path,port) as token:return ready_held(token,epoch,inspections)


def resume_held(token,epoch,inspections,verify):
    """Resume only while holding the same lock used for admission writes."""
    if not ready_held(token,epoch,inspections):raise ValueError('Drain is incomplete')
    if verify() is not True:raise ValueError('New binding not qualified')
    db=_connect(token.path)
    try:
        db.execute('BEGIN IMMEDIATE');state,current=_slot(db,token.port)
        if state!='paused' or current!=epoch:raise ValueError('Stale pause epoch')
        db.execute("UPDATE slots SET state='active' WHERE port=?",(token.port,));db.execute("INSERT INTO events(port,epoch,event,at) VALUES(?,?,'resumed',?)",(token.port,epoch,time.time()));db.commit()
    except BaseException:db.rollback();raise
    finally:db.close()


def resume(path,port,epoch,inspections,verify):
    """Fresh qualification and both clean ledger inspections are mandatory."""
    with hold(path,port) as token:resume_held(token,epoch,inspections,verify)
