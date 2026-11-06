#!/usr/bin/env python3
"""Resolve historical uncertainty for resource drain only, never scientific outcome.

All inspection is read-only. The caller must pin the independently validated
receipt hash and hold the shared port guard while using the returned counts.
"""
import hashlib
import json
from pathlib import Path
import sqlite3

STATES=('preparing','submitted','uncertain','running')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def digest(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def inspect(ledger, port, receipt_path, expected_receipt_sha, *, boot_id_path='/proc/sys/kernel/random/boot_id'):
    """Return blocking counts and exact legacy IDs covered by immutable proof."""
    if type(port) is not int or port not in range(8930,8936):raise ValueError('Invalid endpoint port')
    path=Path(receipt_path);root=path.parent
    if not expected_receipt_sha or sha(path)!=expected_receipt_sha:raise ValueError('Resource receipt hash mismatch')
    receipt=json.loads(path.read_text());evidence_path=root/'resource-quiescence-evidence.json'
    if receipt.get('schema')!='identity-resource-quiescence.receipt.v1' or receipt.get('verified') is not True:raise ValueError('Unverified resource receipt')
    if receipt.get('boot_id')!=Path(boot_id_path).read_text().strip():raise ValueError('Resource proof is from another boot')
    for key,name in [('evidence_sha256','resource-quiescence-evidence.json'),('collector_sha256','resource_quiescence.py'),('validator_sha256','resource_quiescence_validate.py')]:
        if receipt.get(key)!=sha(root/name):raise ValueError('Resource evidence/source mismatch: '+name)
    evidence=json.loads(evidence_path.read_text());saved=receipt.get('rows')
    if evidence.get('schema')!='identity-resource-quiescence.evidence.v1' or evidence.get('boot_id')!=receipt['boot_id'] or evidence.get('rows')!=saved or not isinstance(saved,list) or len(saved)!=120:raise ValueError('Resource proof coverage mismatch')
    if len({r['id'] for r in saved})!=len(saved):raise ValueError('Duplicate resource proof row')
    proof={r['id']:r for r in saved if r.get('port')==port};entry=receipt.get('ports',{}).get(str(port))
    if not isinstance(entry,dict) or not entry.get('unit_invocation') or not entry.get('gpu_uuid') or not entry.get('qualification_sha256'):raise ValueError('Missing port generation evidence')
    history=entry.get('history') or {}
    for r in proof.values():
        if not isinstance(r.get('started'),(int,float)) or not isinstance(r.get('row_sha256'),str):raise ValueError('Incomplete prior row evidence')
        if receipt.get('identity_runner_start',0)<=r['started'] or entry.get('server_start',0)<=r['started']:raise ValueError('Later resource generation not established')
        generation=history.get(str(int(r['started']))) or {};workers=generation.get('observed_workers') or []
        if not generation.get('journal_sha256') or not workers or any(w.get('currently_absent') is not True or not w.get('invocation') or not w.get('pid') for w in workers):raise ValueError('Historical worker absence not established')
    db=sqlite3.connect(Path(ledger).resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    try:
        db.execute('BEGIN');rows=[dict(x) for x in db.execute("SELECT * FROM jobs WHERE (port=? OR port IS NULL) AND state IN ('preparing','submitted','uncertain','running')",(port,))]
    finally:db.close()
    counts={state:0 for state in STATES};covered=[];historical_schema_projection=[]
    for row in rows:
        state=row['state'];counts[state]+=1
        if state=='uncertain' and row.get('campaign_id') is None and row.get('port')==port and row.get('id') in proof:
            record=proof[row['id']]
            full=digest(row)
            if full!=record['row_sha256']:
                # The only supported schema drift is the later NULL admission_id column.
                if 'admission_id' not in row or row['admission_id'] is not None or digest({k:v for k,v in row.items() if k!='admission_id'})!=record['row_sha256']:
                    raise ValueError('Legacy uncertain row differs from resource proof')
                historical_schema_projection.append(row['id'])
            if row.get('started')!=record['started']:raise ValueError('Legacy uncertain start differs from resource proof')
            counts['uncertain']-=1;covered.append(row['id'])
    return {'counts':counts,'covered_uncertain_ids':sorted(covered),'receipt_sha256':expected_receipt_sha,'historical_schema_projection_ids':sorted(historical_schema_projection),'historical_schema_projection':'Only added NULL admission_id is excluded from the original full-row digest',
            'scope':'Resource quiescence only; all original scientific rows remain uncertain and are never replayed.'}
