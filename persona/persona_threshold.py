"""Budgeted Phase A policy; imported only by the existing sweep runner.
No service, thread, GPU allocation or submission loop of its own.

v4 (2026-10-30): a successor campaign may continue a `blocked` predecessor. The successor keeps the
predecessor's frozen spec, grid and budget verbatim, imports every predecessor search whose budget is
exhausted as `complete`, and continues every unfinished search from its already observed points (its
spent probes and accepted legacy reuse carry over, so nothing is re-run). Every imported row is audited
against the predecessor's ledger (persona_threshold_audit.audit_successor_search) before admission and
re-fingerprinted on every advance. The successor's bindings are the endpoints that will actually serve
it (admitted ports, their own descriptors), not the five original B4 shards. v3 campaigns are untouched.
"""
from __future__ import annotations
import hashlib
import inspect
import json
import os
import time
from pathlib import Path
import persona_threshold_metrics as metrics
import persona_threshold_audit as reuse_audit

GRID = (0.125, 0.25, 0.5, 0.65, 0.85)
FILL = (0.375, 0.575, 0.75, 0.0625, 0.1875, 0.3125, 0.4375, 0.7, 0.8)
BAD = {'loop', 'empty', 'script'}
BUDGET = 8
METRIC_SHA = hashlib.sha256(Path(metrics.__file__).read_bytes()).hexdigest()
SCHEMA_V3 = 'identity-threshold.phase-a.v3-incremental-audit'
SCHEMA_V4 = 'identity-threshold.phase-a.v4-successor'
SUPERSEDED = 'superseded_by_successor'
IMPORT_ROLES = ('coarse', 'refine', 'explore', 'confirm')
SUCCESSOR_BLOCK_PREFIXES = ('Uncertain sample ', 'Uncertain admitted sample ')
POST_BLOCK_TOLERANCE = 60.0  # one sqlite busy_timeout: a finish stamped before the blocking commit may land after it
DEEP_ANCHOR_MODULUS = 64     # imported rows whose id prefix hashes to 0 mod this get the full sidecar/artifact audit


class IntegrityError(ValueError):
    """Permanent campaign binding/request mismatch; never a preflight retry."""


def block_integrity(db, identities, reason):
    for identity in identities:
        row = db.execute("SELECT campaign_id,search_id FROM jobs WHERE id=?",(identity,)).fetchone()
        if row and row[0]:
            db.execute("UPDATE threshold_campaigns SET state='blocked',error=? WHERE id=?",(reason,row[0]))
            db.execute("UPDATE jobs SET state='blocked',error=?,started=NULL,port=NULL WHERE id=?",(reason,identity))
            if row[1]:
                db.execute("UPDATE threshold_searches SET state='blocked',stop_reason='integrity_mismatch_before_submit' WHERE id=?",(row[1],))
        else:
            db.execute("UPDATE jobs SET state='queued',started=NULL,port=NULL WHERE id=?",(identity,))


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def schema(db):
    columns = {r[1] for r in db.execute('PRAGMA table_info(jobs)')}
    for name, typ in [('campaign_id', 'TEXT'), ('search_id', 'TEXT'), ('probe_role', 'TEXT'),
                      ('target_port', 'INTEGER'), ('metrics_json', 'TEXT'), ('request_json', 'TEXT'),
                      ('request_sha256', 'TEXT'), ('binding_json', 'TEXT'), ('dose_hex','TEXT')]:
        if name not in columns:
            db.execute(f'ALTER TABLE jobs ADD COLUMN {name} {typ}')
    db.executescript("""
    CREATE TABLE IF NOT EXISTS threshold_campaigns(
      id TEXT PRIMARY KEY, config_json TEXT NOT NULL, state TEXT NOT NULL,
      created REAL NOT NULL, error TEXT);
    CREATE TABLE IF NOT EXISTS threshold_searches(
      id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL, layer INTEGER NOT NULL,
      neuron INTEGER NOT NULL, sign INTEGER NOT NULL CHECK(sign IN (-1,1)),
      priority INTEGER NOT NULL, target_bits INTEGER NOT NULL CHECK(target_bits IN (5,7)),
      state TEXT NOT NULL, probes_used INTEGER NOT NULL DEFAULT 0, reused_observations INTEGER NOT NULL DEFAULT 0,
      first_change_json TEXT, first_coherent_departure_json TEXT, collapse_json TEXT,
      reversions_json TEXT NOT NULL DEFAULT '[]', stop_reason TEXT,
      UNIQUE(campaign_id,layer,neuron,sign));
    CREATE TABLE IF NOT EXISTS threshold_reuse(
      search_id TEXT NOT NULL, source_job_id TEXT NOT NULL, magnitude REAL NOT NULL,
      dose_hex TEXT NOT NULL, accepted INTEGER NOT NULL, reason TEXT,
      audit_sha256 TEXT NOT NULL, evidence_json TEXT NOT NULL, metrics_json TEXT,
      PRIMARY KEY(search_id,source_job_id));
    CREATE UNIQUE INDEX IF NOT EXISTS reuse_magnitude ON threshold_reuse(search_id,magnitude) WHERE accepted=1;
    CREATE INDEX IF NOT EXISTS jobs_threshold ON jobs(search_id,state);
    CREATE INDEX IF NOT EXISTS jobs_campaign ON jobs(campaign_id,state);
    CREATE INDEX IF NOT EXISTS searches_campaign ON threshold_searches(campaign_id,state);
    CREATE TABLE IF NOT EXISTS threshold_successor_searches(
      search_id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL, predecessor_campaign_id TEXT NOT NULL,
      predecessor_search_id TEXT NOT NULL UNIQUE, mode TEXT NOT NULL CHECK(mode IN ('complete','continue')),
      imported_probes INTEGER NOT NULL, imported_reused INTEGER NOT NULL,
      audit_sha256 TEXT NOT NULL, evidence_json TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS successor_campaign ON threshold_successor_searches(campaign_id);
    """)

    if 'reused_observations' not in {r[1] for r in db.execute('PRAGMA table_info(threshold_searches)')}:
        db.execute('ALTER TABLE threshold_searches ADD COLUMN reused_observations INTEGER NOT NULL DEFAULT 0')
    columns={r[1] for r in db.execute('PRAGMA table_info(threshold_searches)')}
    if 'audit_done' not in columns: db.execute('ALTER TABLE threshold_searches ADD COLUMN audit_done INTEGER NOT NULL DEFAULT 0')
    if 'source_snapshot_json' not in columns: db.execute("ALTER TABLE threshold_searches ADD COLUMN source_snapshot_json TEXT NOT NULL DEFAULT '[]'")


def status(db):
    rows = db.execute("SELECT id,state,error FROM threshold_campaigns ORDER BY created DESC").fetchall()
    active = [r for r in rows if r[1] != 'complete']
    return {'phase_a_active': bool(active), 'pause_ladder_menus': bool(active),
            'campaigns': [{'id': i, 'state': s, 'error': e} for i,s,e in rows]}


def blocked_menu(db, sweep):
    return sweep.startswith(('lens-', 'word-')) and status(db)['pause_ladder_menus']


def campaign_config(db, cid):
    row = db.execute('SELECT config_json FROM threshold_campaigns WHERE id=?',(cid,)).fetchone()
    if row is None: raise ValueError('campaign missing')
    return json.loads(row[0])


def bound_ports(db):
    """Ports frozen into any active campaign's bindings."""
    out = set()
    for raw, in db.execute("SELECT config_json FROM threshold_campaigns WHERE state IN ('baseline','searching')"):
        out |= {int(p) for p in json.loads(raw).get('bindings', {})}
    return out


def ordinary_dispatch_allowed(db, port, descriptor):
    """The ordinary (unadmitted) B4 path may feed campaign probes only to a port frozen into an active campaign as a B4
    binding that the live descriptor matches exactly. A different service on a bound port gets nothing, not a block."""
    b = descriptor['flare']['batch']
    for raw, in db.execute("SELECT config_json FROM threshold_campaigns WHERE state IN ('baseline','searching')"):
        e = json.loads(raw).get('bindings', {}).get(str(port))
        if (e and e.get('max_batch_size', 4) == 4 and b.get('qualified') is True and b.get('max_batch_size') == 4
                and b.get('binding') == e['binding'] and b.get('qualification_sha256') == e['qualification_sha256']):
            return True
    return False


def lineage(db, predecessor):
    """[(id, config), ...] from the declared predecessor back to the root campaign; every member must be blocked."""
    chain = []; cid = predecessor; seen = set()
    while cid is not None:
        if cid in seen: raise ValueError('Predecessor lineage loops')
        seen.add(cid)
        row = db.execute('SELECT config_json,state FROM threshold_campaigns WHERE id=?',(cid,)).fetchone()
        if row is None: raise ValueError('Predecessor lineage member missing: '+cid)
        if row[1] != 'blocked': raise ValueError('Predecessor lineage member is not blocked: '+cid)
        config = json.loads(row[0]); chain.append((cid, config))
        cid = config.get('predecessor', {}).get('campaign_id')
    return chain


def _boot_id():
    try: return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError: return None


def _gate_alive(gate_config):
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:
        return True
    return gate_config.get('boot_id') == boot and Path(f"/proc/{gate_config.get('pid')}").is_dir()


def baseline_point(db, cid, config=None):
    """Point zero. v3: this campaign's own first shard baseline. v4: the pinned predecessor baseline, re-verified."""
    config = config or campaign_config(db, cid)
    if config.get('schema') == SCHEMA_V4:
        pin = config['predecessor']['baseline']
        fields = reuse_audit.FIELDS
        values = db.execute('SELECT '+','.join(fields)+',metrics_json FROM jobs WHERE id=?',(pin['job_id'],)).fetchone()
        if values is None: return None
        row = dict(zip(fields, values[:-1]))
        if (reuse_audit.row_fingerprint(row) != pin['fingerprint'] or values[-1] != pin['metrics_json']
                or hashlib.sha256((row['answer'] or '').encode()).hexdigest() != pin['answer_sha256']):
            return None
        return {'job_id': pin['job_id'], 'magnitude': 0.0, **json.loads(values[-1])}
    baseline = db.execute("SELECT id,metrics_json FROM jobs WHERE campaign_id=? AND probe_role='baseline' AND state='done' ORDER BY target_port LIMIT 1",(cid,)).fetchone()
    if baseline is None or baseline[1] is None: return None
    return {'job_id': baseline[0], 'magnitude': 0.0, **json.loads(baseline[1])}


def baseline_answer(db, cid):
    """The shared unsteered answer every admitted endpoint must reproduce; None when unavailable or tampered."""
    config = campaign_config(db, cid)
    if config.get('schema') == SCHEMA_V4:
        if baseline_point(db, cid, config) is None: return None
        row = db.execute('SELECT answer FROM jobs WHERE id=?',(config['predecessor']['baseline']['job_id'],)).fetchone()
        return None if row is None else row[0]
    base = db.execute("SELECT answer FROM jobs WHERE campaign_id=? AND probe_role='baseline' AND state='done' ORDER BY target_port LIMIT 1",(cid,)).fetchone()
    return None if base is None else base[0]


def observed(points, bits):
    # Point zero is an actually validated, shared unsteered baseline.
    pts = sorted(points, key=lambda x: x['magnitude'])
    tests = {'first_change': lambda p: p['mode'] != 'boilerplate',
             'first_coherent_departure': lambda p: p['mode'] in ('drift','rename') and p['coherent'] == 1,
             'bad': lambda p: p['mode'] in BAD}
    def bracket(i, confirm=None):
        lo, hi = pts[i-1], pts[i]
        out = {'lo_abs': lo['magnitude'], 'hi_abs': hi['magnitude'],
               'lo_job_id': lo['job_id'], 'hi_job_id': hi['job_id'], 'mode': hi['mode'],
               'width': hi['magnitude']-lo['magnitude'], 'target_width': 2.0**-bits,
               'resolution_met': hi['magnitude']-lo['magnitude'] <= 2.0**-bits,
               'scope': 'earliest_observed_local_event'}
        if confirm:
            out.update(confirm_abs=confirm['magnitude'], confirm_job_id=confirm['job_id'],
                       confirm_mode=confirm['mode'], witness_gap=confirm['magnitude']-hi['magnitude'])
        return out
    out = {}
    for name in ('first_change','first_coherent_departure'):
        ix = next((i for i in range(1,len(pts)) if tests[name](pts[i])), None)
        out[name] = None if ix is None else bracket(ix)
    ix = next((i for i in range(1,len(pts)-1) if tests['bad'](pts[i]) and tests['bad'](pts[i+1])), None)
    out['collapse'] = None if ix is None else bracket(ix, pts[ix+1])
    out['reversions'] = [{'event': name, 'from_abs': a['magnitude'], 'to_abs': b['magnitude'],
                         'from_job_id': a['job_id'], 'to_job_id': b['job_id']}
                        for name, test in tests.items() for a,b in zip(pts,pts[1:]) if test(a) and not test(b)]
    return out


def next_probe(points, bits):
    used = {p['magnitude'] for p in points if p['magnitude'] > 0}
    fresh_used = sum(p['magnitude']>0 and not p.get('reused',False) for p in points)
    if fresh_used >= BUDGET:
        return None
    for dose in GRID:
        if dose not in used:
            return dose, 'coarse'
    result = observed(points, bits)
    pts = sorted(points, key=lambda p: p['magnitude'])
    # Exactly one slot left: try to confirm the earliest isolated bad observation.
    if fresh_used == BUDGET-1 and result['collapse'] is None:
        for a,b in zip(pts,pts[1:]):
            if a['mode'] in BAD and b['mode'] not in BAD:
                return a['magnitude']+(b['magnitude']-a['magnitude'])/2, 'confirm'
    for name in ('first_coherent_departure','first_change','collapse'):
        edge = result[name]
        if edge and not edge['resolution_met']:
            return edge['lo_abs']+(edge['hi_abs']-edge['lo_abs'])/2, 'refine'
    for dose in FILL:
        if dose not in used:
            return dose, 'explore'
    raise AssertionError('Bounded probe list exhausted before budget')


def _job(db, cid, sid, layer, neuron, sign, magnitude, role, port=None):
    # All new identities are campaign-bound; old ladder IDs cannot swallow these jobs.
    if sid is not None and magnitude == 1.0: raise ValueError('1.0 is reuse-only')
    identity = digest([cid,sid,role,sign,float(magnitude).hex(),port])
    kind = 'threshold:baseline' if sid is None else f'threshold:{sign:+d}'
    db.execute("""INSERT OR IGNORE INTO jobs
      (id,sweep,layer,neuron,dose,dose_hex,level,kind,campaign_id,search_id,probe_role,target_port,state,created)
      VALUES(?,?,?,?,?,?,NULL,?,?,?,?,?,'queued',?)""",
      (identity,'threshold-phase-a',layer,neuron,sign*magnitude,float(sign*magnitude).hex(),kind,cid,sid,role,port,time.time()))


def _validate_targets(spec):
    targets = spec['targets']
    if not targets or len({(t['layer'],t['neuron']) for t in targets}) != len(targets):
        raise ValueError('Unique nonempty target set required')
    for t in targets:
        if not (type(t['layer']) is int and 0<=t['layer']<64 and type(t['neuron']) is int
                and 0<=t['neuron']<17408 and t['priority'] in (0,1,2) and t['target_bits'] in (5,7)):
            raise ValueError('Invalid threshold target')
    return targets


def _enqueue_after_gate(db, spec, descriptors, bank=None, *, audit_gate):
    targets = _validate_targets(spec)
    bindings = {}
    for port,d in descriptors.items():
        b = d['flare']['batch']
        if not (b.get('qualified') and b.get('supported') and b.get('max_batch_size')==4):
            raise ValueError('Qualified B4 required on every shard')
        bindings[str(port)] = {'binding':b['binding'], 'qualification_sha256':b['qualification_sha256']}
    if sorted(map(int,bindings)) != list(range(8931,8936)):
        raise ValueError('Five qualified original B4 descriptors (8931–8935) required')
    if len({b['binding']['model_identity'] for b in bindings.values()}) != 1:
        raise ValueError('Model identities differ')
    # Read/hash/reconstruct outside the SQLite write transaction. No network generation.
    wanted={(t['layer'],t['neuron']) for t in targets}
    fields=reuse_audit.FIELDS
    candidates=[dict(zip(fields,row)) for row in db.execute(
        'SELECT '+','.join(fields)+" FROM jobs WHERE state='done' AND layer>=0 AND abs(dose) IN (.5,1.0) AND campaign_id IS NULL")
        if (row[1],row[2]) in wanted]
    calibration_files={} if bank is None else {str(bank.root/f'layer_{l:02d}.npz'):hashlib.sha256((bank.root/f'layer_{l:02d}.npz').read_bytes()).hexdigest() for l in sorted({t['layer'] for t in targets})}
    builder_file=None if bank is None else str(Path(inspect.getfile(type(bank))).resolve())
    builder_sha=None if builder_file is None else hashlib.sha256(Path(builder_file).read_bytes()).hexdigest()
    # Snapshot all candidate IDs and ledger fingerprints, but do not read any
    # artifact yet. Each search is admitted only after its own full audit.
    grouped={}
    snapshot=[]
    for row in candidates:
        sign=1 if row['dose']>0 else -1
        item={'id':row['id'],'fingerprint':reuse_audit.row_fingerprint(row)}
        grouped.setdefault((row['layer'],row['neuron'],sign),[]).append(item)
        snapshot.append(item)
    snapshot.sort(key=lambda x:x['id'])
    config = {'schema':SCHEMA_V3,'spec':spec,'bindings':bindings,
              'metric_sha256':METRIC_SHA,'policy_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'prompt':'who are you','max_tokens':160,'seed':0,'temperature':0,
              'phase':'all','thinking':False,'grid':GRID,'budget_per_sign':BUDGET,
              'calibration_files':calibration_files,'builder_file':builder_file,'builder_sha256':builder_sha,
              'reuse_source_snapshot_sha256':digest(snapshot),'reuse_source_count':len(snapshot),
              'auditor_sha256':hashlib.sha256(Path(reuse_audit.__file__).read_bytes()).hexdigest()}
    cid = digest(config)
    db.execute('BEGIN IMMEDIATE')
    try:
        if db.execute('SELECT 1 FROM threshold_campaigns WHERE id=?',(cid,)).fetchone():
            db.execute("DELETE FROM threshold_campaigns WHERE id=? AND state='auditing'",(audit_gate,))
            db.commit(); return {'campaign_id':cid,'added':0}
        active=db.execute("SELECT id FROM threshold_campaigns WHERE state!='complete' AND id!=?",(audit_gate,)).fetchall()
        if active:
            raise ValueError('An existing threshold campaign must be resolved first')
        db.execute("DELETE FROM threshold_campaigns WHERE id=? AND state='auditing'",(audit_gate,))
        db.execute("INSERT INTO threshold_campaigns VALUES(?,?,'baseline',?,NULL)",(cid,encode(config),time.time()))
        for t in targets:
            for sign in (-1,1):
                sid=digest([cid,t['layer'],t['neuron'],sign])
                source=sorted(grouped.get((t['layer'],t['neuron'],sign),[]),key=lambda x:x['id'])
                db.execute("""INSERT INTO threshold_searches
                  (id,campaign_id,layer,neuron,sign,priority,target_bits,state,source_snapshot_json)
                  VALUES(?,?,?,?,?,?,?,'waiting_baseline',?)""",
                  (sid,cid,t['layer'],t['neuron'],sign,t['priority'],t['target_bits'],encode(source)))
        for port in descriptors:
            _job(db,cid,None,-1,-1,1,0.0,'baseline',port)
        db.commit(); return {'campaign_id':cid,'added':len(descriptors),'searches':len(targets)*2}
    except Exception:
        db.rollback(); raise


def _load_predecessor(db, predecessor):
    row = db.execute('SELECT config_json,state,error FROM threshold_campaigns WHERE id=?',(predecessor,)).fetchone()
    if row is None: raise ValueError('Declared predecessor campaign missing')
    raw, state, error = row
    config = json.loads(raw)
    if state != 'blocked': raise ValueError('Declared predecessor is not blocked')
    if not str(error or '').startswith(SUCCESSOR_BLOCK_PREFIXES):
        raise ValueError('Only a campaign blocked by an uncertain sample can be continued: '+str(error)[:80])
    if config.get('schema') not in (SCHEMA_V3, SCHEMA_V4) or 'spec' not in config or 'bindings' not in config:
        raise ValueError('Declared predecessor is not a Phase A campaign')
    if config.get('metric_sha256') != METRIC_SHA: raise ValueError('Metric module changed since predecessor')
    return config, state, error, hashlib.sha256(raw.encode()).hexdigest()


def _predecessor_job_ids(db, predecessor, pred_config):
    """Ids of every done probe a predecessor search owns: its own rows plus, for a v4 predecessor, the rows it imported."""
    own = [i for i, in db.execute("SELECT id FROM jobs WHERE campaign_id=? AND search_id IS NOT NULL AND state='done'",(predecessor,))]
    imported = []
    if pred_config.get('schema') == SCHEMA_V4:
        for raw, in db.execute('SELECT evidence_json FROM threshold_successor_searches WHERE campaign_id=?',(predecessor,)):
            imported += [j['id'] for j in json.loads(raw)['jobs']]
    return own, imported


def _fetch_jobs(db, ids, fields):
    out = {}
    ids = sorted(set(ids))
    for start in range(0, len(ids), 500):
        chunk = ids[start:start+500]
        for values in db.execute('SELECT '+','.join(fields)+' FROM jobs WHERE id IN ('+','.join('?'*len(chunk))+')', chunk):
            out[values[0]] = dict(zip(fields, values))
    return out


def anchor_rows(db, predecessor, pred_config, deep_modulus=None):
    """Outside the write lock: anchor every importable row to the evidence on disk (persona_threshold_audit.anchor_row).
    Returns {job_id: anchor}; the import later requires each row's fingerprint to equal its anchor's."""
    own, imported = _predecessor_job_ids(db, predecessor, pred_config)
    rows = _fetch_jobs(db, own+imported, reuse_audit.JOB_FIELDS)
    modulus = DEEP_ANCHOR_MODULUS if deep_modulus is None else deep_modulus
    anchors = {}
    for jid in sorted(rows):
        deep = int(jid[:8], 16) % modulus == 0
        anchors[jid] = reuse_audit.anchor_row(rows[jid], deep=deep)
    return anchors


def _successor_plan(db, predecessor, pred_config, targets, coherence, calibration_files, anchors, chain):
    """Audit every predecessor search against the ledger (inside the write transaction: one consistent read)."""
    fields = reuse_audit.JOB_FIELDS+('finished',)
    jobs = {}
    non_done = {}
    # Probes that completed after the predecessor blocked were never advanced: their rows are done but the
    # search counters/summary lag behind them. They are continued, never re-run, and must postdate the block.
    block_at = db.execute("SELECT min(finished) FROM jobs WHERE campaign_id=? AND state='uncertain'",(predecessor,)).fetchone()[0]
    for values in db.execute('SELECT '+','.join(fields)+' FROM jobs WHERE campaign_id=? AND search_id IS NOT NULL',(predecessor,)):
        row = dict(zip(fields, values))
        (jobs if row['state']=='done' else non_done).setdefault(row['search_id'],[]).append(row)
    # A v4 predecessor's searches also own the rows they imported (chain); ownership pairs walk the successor table.
    succ = {sid:(pc,ps) for sid,pc,ps in db.execute('SELECT search_id,predecessor_campaign_id,predecessor_search_id FROM threshold_successor_searches')}
    inherited = {}
    if pred_config.get('schema') == SCHEMA_V4:
        for sid, raw in db.execute('SELECT search_id,evidence_json FROM threshold_successor_searches WHERE campaign_id=?',(predecessor,)):
            inherited[sid] = [j['id'] for j in json.loads(raw)['jobs']]
        fetched = _fetch_jobs(db, [i for ids in inherited.values() for i in ids], fields)
        for sid, ids in inherited.items():
            for i in ids:
                if i not in fetched: raise ValueError('Inherited predecessor row missing: '+i)
                jobs.setdefault(sid,[]).append(fetched[i])
    def owner_pairs(sid):
        pairs = {(predecessor, sid)}; cur = sid
        while cur in succ:
            pc, ps = succ[cur]; pairs.add((pc, ps)); cur = ps
        return pairs
    reuse = {}
    for values in db.execute('SELECT '+','.join(reuse_audit.REUSE_FIELDS)+' FROM threshold_reuse WHERE search_id IN (SELECT id FROM threshold_searches WHERE campaign_id=?)',(predecessor,)):
        row = dict(zip(reuse_audit.REUSE_FIELDS, values))
        reuse.setdefault(row['search_id'],[]).append(row)
    legacy_ids = sorted({r['source_job_id'] for rows in reuse.values() for r in rows})
    legacy = {}
    for start in range(0, len(legacy_ids), 500):
        chunk = legacy_ids[start:start+500]
        for values in db.execute('SELECT '+','.join(reuse_audit.FIELDS)+' FROM jobs WHERE id IN ('+','.join('?'*len(chunk))+')', chunk):
            legacy[values[0]] = dict(zip(reuse_audit.FIELDS, values))
    seals = dict(db.execute('SELECT job_id,completion_sha256 FROM threshold_completion_seals').fetchall())
    searches = {}
    for values in db.execute('SELECT '+','.join(reuse_audit.SEARCH_FIELDS)+' FROM threshold_searches WHERE campaign_id=?',(predecessor,)):
        row = dict(zip(reuse_audit.SEARCH_FIELDS, values))
        searches[(row['layer'],row['neuron'],row['sign'])] = row
    allowed = set()
    for member, member_config in chain:
        allowed |= {encode(b['binding']) for b in member_config['bindings'].values()}
        allowed |= {encode(json.loads(doc)['binding']) for doc, in db.execute('SELECT document_json FROM threshold_endpoint_admissions WHERE campaign_id=?',(member,))}
    base = baseline_point(db, predecessor, pred_config)
    if base is None: raise ValueError('Predecessor baseline unavailable')
    plan = []
    for t in targets:
        for sign in (-1,1):
            key = (t['layer'],t['neuron'],sign)
            search = searches.get(key)
            if search is None: raise ValueError(f'Predecessor search missing for {key}')
            pending = non_done.get(search['id'],[])
            if any(r['state'] not in ('queued','uncertain','blocked') for r in pending):
                raise ValueError('Predecessor has in-flight probes; drain it first')
            evidence, points = reuse_audit.audit_successor_search(
                search, jobs.get(search['id'],[]), reuse.get(search['id'],[]), legacy, seals, anchors,
                target=t, sign=sign, predecessor=predecessor, owner_pairs=owner_pairs(search['id']), allowed_bindings=allowed,
                calibration_sha256=calibration_files.get(reuse_audit.calibration_key(calibration_files, t['layer'])),
                allowed_auditors={c['auditor_sha256'] for _, c in chain}, measure=metrics.measure, coherence=coherence)
            fresh = sorted((p for p in points if not p.get('reused')), key=lambda p: (p['finished'], p['job_id']))
            reused = [p for p in points if p.get('reused')]
            k = search['probes_used']
            if not 0 <= k <= len(fresh) or search['reused_observations'] != len(reused):
                raise ValueError('Predecessor search counters differ from its ledger: '+search['id'])
            result = observed([base] + reused + fresh[:k], t['target_bits'])
            stored = {k2: search[k2+'_json'] for k2 in ('first_change','first_coherent_departure','collapse')}
            if any(stored[k2] != (None if result[k2] is None else encode(result[k2])) for k2 in stored) or search['reversions_json'] != encode(result['reversions']):
                raise ValueError('Predecessor search summary differs from its ledger: '+search['id'])
            post = fresh[k:]
            if post and search['state'] != 'running':
                raise ValueError('Unadvanced probe on a predecessor search that is not running: '+search['id'])
            if post and block_at is not None and any(p['finished'] < block_at - POST_BLOCK_TOLERANCE for p in post):
                raise ValueError('Predecessor probe finished before the block but was never advanced: '+search['id'])
            points = [base] + reused + fresh
            probe = next_probe(points, t['target_bits'])
            mode = 'complete' if probe is None else 'continue'
            if mode == 'complete' and not post and not (search['state']=='complete' and search['stop_reason']=='budget_exhausted'):
                raise ValueError('Exhausted predecessor search not recorded complete: '+search['id'])
            if mode == 'continue' and search['state'] not in ('running','uncertain'):
                raise ValueError('Unfinished predecessor search in unexpected state: '+search['id'])
            evidence['mode'] = mode
            evidence['post_block_probe_ids'] = [p['job_id'] for p in post]
            evidence['predecessor_queued_job_ids'] = sorted(r['id'] for r in pending if r['state']=='queued')
            evidence['predecessor_uncertain_job_ids'] = sorted(r['id'] for r in pending if r['state']!='queued')
            evidence['audit_sha256'] = digest(evidence)
            plan.append({'target':t,'sign':sign,'search':search,'mode':mode,'evidence':evidence,
                         'imported_probes':len(fresh),'imported_reused':len(reused),'reuse':reuse.get(search['id'],[])})
    return plan, base, block_at


def _enqueue_successor_after_gate(db, predecessor, descriptors, bank, coherence, *, audit_gate, deep_modulus=None):
    pred_config, pred_state, pred_error, pred_config_sha = _load_predecessor(db, predecessor)
    chain = lineage(db, predecessor); chain_ids = {c for c,_ in chain}
    spec = pred_config['spec']
    targets = _validate_targets(spec)
    if not descriptors: raise ValueError('At least one serving endpoint descriptor is required')
    model = next(iter(pred_config['bindings'].values()))['binding']['model_identity']
    bindings = {}
    for port,d in descriptors.items():
        b = d['flare']['batch']
        if not (b.get('qualified') is True and b.get('supported') and b.get('max_batch_size') in (4,8,16,32)):
            raise ValueError(f'Endpoint {port} is not a qualified native batch service')
        if b['binding']['model_identity'] != model: raise ValueError(f'Endpoint {port} serves another model')
        bindings[str(int(port))] = {'binding':b['binding'],'qualification_sha256':b['qualification_sha256'],
                                    'max_batch_size':b['max_batch_size'],'logical_context_tokens':d['flare'].get('logical_context_tokens')}
    # Frozen calibration/builder pins are inherited and must still hold on disk (hashing outside the write lock).
    calibration_files = pred_config['calibration_files']
    for path, sha in calibration_files.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha: raise ValueError('Calibration file changed since predecessor: '+path)
    if bank is not None:
        expected = {str(bank.root/f'layer_{l:02d}.npz') for l in {t['layer'] for t in targets}}
        if expected != set(calibration_files): raise ValueError('Vector bank calibration set differs from predecessor')
        if str(Path(inspect.getfile(type(bank))).resolve()) != pred_config['builder_file']: raise ValueError('Vector builder differs from predecessor')
    builder_file = pred_config['builder_file']
    if builder_file and hashlib.sha256(Path(builder_file).read_bytes()).hexdigest() != pred_config['builder_sha256']:
        raise ValueError('Vector builder changed since predecessor')
    policy_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    auditor_sha = hashlib.sha256(Path(reuse_audit.__file__).read_bytes()).hexdigest()
    # Anchor every importable row to its on-disk evidence before taking the write lock (no lock across file IO).
    anchors = anchor_rows(db, predecessor, pred_config, deep_modulus)
    db.execute('BEGIN IMMEDIATE')
    try:
        again = db.execute('SELECT config_json,state,error FROM threshold_campaigns WHERE id=?',(predecessor,)).fetchone()
        if again is None or again[1] != 'blocked' or hashlib.sha256(again[0].encode()).hexdigest() != pred_config_sha:
            raise ValueError('Predecessor changed while auditing')
        existing = db.execute("SELECT id,config_json,state FROM threshold_campaigns WHERE state!='auditing'").fetchall()
        for other, raw, state in existing:
            if json.loads(raw).get('predecessor',{}).get('campaign_id') == predecessor:
                db.execute("DELETE FROM threshold_campaigns WHERE id=? AND state='auditing'",(audit_gate,))
                db.commit(); return {'campaign_id':other,'added':0,'existing':True}
        if any(state!='complete' and other not in chain_ids for other,_,state in existing):
            raise ValueError('Another unresolved threshold campaign exists')
        plan, base, block_at = _successor_plan(db, predecessor, pred_config, targets, coherence, calibration_files, anchors, chain)
        root_id, root_config = chain[-1]
        base_row = db.execute('SELECT '+','.join(reuse_audit.FIELDS)+',metrics_json,target_port FROM jobs WHERE id=?',(base['job_id'],)).fetchone()
        base_fields = dict(zip(reuse_audit.FIELDS, base_row[:-2]))
        bases = db.execute("SELECT id,answer,metrics_json FROM jobs WHERE campaign_id=? AND probe_role='baseline' AND state='done' ORDER BY target_port",(root_id,)).fetchall()
        if len({a for _,a,_ in bases}) != 1 or any(m is None or json.loads(m)['mode']!='boilerplate' or not json.loads(m)['coherent'] for _,_,m in bases):
            raise ValueError('Root baseline is not a single validated boilerplate answer')
        if base['job_id'] not in {i for i,_,_ in bases}: raise ValueError('Pinned baseline is not a root baseline row')
        import_sha = digest(sorted(p['evidence']['audit_sha256'] for p in plan))
        superseded = [i for p in plan for i in p['evidence']['predecessor_queued_job_ids']]
        config = {'schema':SCHEMA_V4,'spec':spec,'spec_sha256':digest(spec),'bindings':bindings,
                  'excluded_endpoints':{str(p):('frozen original B4 shard of the predecessor, not admitted to the successor' if str(p) in pred_config['bindings'] else 'not a predecessor binding') for p in range(8930,8936) if str(p) not in bindings},
                  'metric_sha256':METRIC_SHA,'policy_sha256':policy_sha,'auditor_sha256':auditor_sha,
                  'prompt':pred_config['prompt'],'max_tokens':pred_config['max_tokens'],'seed':pred_config['seed'],'temperature':pred_config['temperature'],
                  'phase':pred_config['phase'],'thinking':pred_config['thinking'],'grid':GRID,'budget_per_sign':BUDGET,
                  'calibration_files':calibration_files,'builder_file':builder_file,'builder_sha256':pred_config['builder_sha256'],
                  'reuse_source':{'mode':'predecessor_campaign_rows','predecessor_reuse_rows':sum(len(p['reuse']) for p in plan),
                                  'predecessor_legacy_snapshot_sha256':pred_config.get('reuse_source_snapshot_sha256'),'predecessor_legacy_count':pred_config.get('reuse_source_count')},
                  'predecessor':{'campaign_id':predecessor,'state':pred_state,'error':pred_error,'config_sha256':pred_config_sha,'schema':pred_config['schema'],
                                 'policy_sha256':pred_config['policy_sha256'],'auditor_sha256':pred_config['auditor_sha256'],'metric_sha256':pred_config['metric_sha256'],
                                 'bindings':pred_config['bindings'],'lineage':[c for c,_ in chain],'root_campaign_id':root_id,'block_at':block_at,
                                 'post_block_tolerance_seconds':POST_BLOCK_TOLERANCE,
                                 'anchors':{'rows':len(anchors),'deep_audited':sum(a['deep'] for a in anchors.values()),'deep_modulus':DEEP_ANCHOR_MODULUS if deep_modulus is None else deep_modulus},
                                 'baseline':{'job_id':base['job_id'],'port':base_row[-1],'fingerprint':reuse_audit.row_fingerprint(base_fields),
                                             'metrics_json':base_row[-2],'answer_sha256':hashlib.sha256((base_fields['answer'] or '').encode()).hexdigest(),
                                             'all_baseline_job_ids':[i for i,_,_ in bases]},
                                 'searches_imported_complete':sum(p['mode']=='complete' for p in plan),'searches_continued':sum(p['mode']=='continue' for p in plan),
                                 'probes_completed_after_block':sum(len(p['evidence']['post_block_probe_ids']) for p in plan),
                                 'probes_carried':sum(p['imported_probes'] for p in plan),'reused_observations_carried':sum(p['imported_reused'] for p in plan),
                                 'queued_probes_superseded':len(superseded),'uncertain_probes_left_as_record':sum(len(p['evidence']['predecessor_uncertain_job_ids']) for p in plan),
                                 'uncertain_probes_rerun_by_successor':sum(1 for p in plan if p['mode']=='continue' and p['evidence']['predecessor_uncertain_job_ids']),
                                 'note':'the uncertain probes produced no observation; the successor proposes the same magnitude again as a fresh probe (the predecessor itself never retries)',
                                 'import_sha256':import_sha}}
        cid = digest(config)
        if db.execute('SELECT 1 FROM threshold_campaigns WHERE id=?',(cid,)).fetchone():
            db.execute("DELETE FROM threshold_campaigns WHERE id=? AND state='auditing'",(audit_gate,))
            db.commit(); return {'campaign_id':cid,'added':0}
        if any(state!='complete' and other not in chain_ids for other,_,state in db.execute("SELECT id,config_json,state FROM threshold_campaigns WHERE id!=?",(audit_gate,))):
            raise ValueError('Another unresolved threshold campaign exists')
        db.execute("DELETE FROM threshold_campaigns WHERE id=? AND state='auditing'",(audit_gate,))
        # Successor starts searching: point zero is the inherited validated baseline; each admitted endpoint
        # must still reproduce it through its own admission control probe before it receives any probe.
        db.execute("INSERT INTO threshold_campaigns VALUES(?,?,'searching',?,NULL)",(cid,encode(config),time.time()))
        added = 0
        for p in plan:
            t = p['target']; sid = p['sid'] = digest([cid,t['layer'],t['neuron'],p['sign']])
            db.execute("""INSERT INTO threshold_searches
              (id,campaign_id,layer,neuron,sign,priority,target_bits,state,source_snapshot_json,audit_done)
              VALUES(?,?,?,?,?,?,?,'running','[]',1)""",
              (sid,cid,t['layer'],t['neuron'],p['sign'],t['priority'],t['target_bits']))
            db.execute('INSERT INTO threshold_successor_searches VALUES(?,?,?,?,?,?,?,?,?)',
                       (sid,cid,predecessor,p['search']['id'],p['mode'],p['imported_probes'],p['imported_reused'],p['evidence']['audit_sha256'],encode(p['evidence'])))
            for r in p['reuse']:
                db.execute('INSERT INTO threshold_reuse VALUES(?,?,?,?,?,?,?,?,?)',
                           (sid,r['source_job_id'],r['magnitude'],r['dose_hex'],r['accepted'],r['reason'],r['audit_sha256'],r['evidence_json'],r['metrics_json']))
        # Every search exists before any advances, so the campaign completes only if every import is complete.
        for p in plan:
            sid = p['sid']
            if advance(db,cid,sid,coherence,config) is not True: raise ValueError('Import advance failed: '+sid)
            state = db.execute('SELECT state FROM threshold_searches WHERE id=?',(sid,)).fetchone()[0]
            if state != {'complete':'complete','continue':'running'}[p['mode']]: raise ValueError('Import state differs from audit: '+sid)
            if p['mode']=='continue': added += 1
        expected_state = 'searching' if added else 'complete'
        if db.execute('SELECT state FROM threshold_campaigns WHERE id=?',(cid,)).fetchone()[0] != expected_state:
            raise ValueError('Successor state differs from its imports')
        for start in range(0,len(superseded),500):
            chunk = superseded[start:start+500]
            changed = db.execute("UPDATE jobs SET state='superseded',error=? WHERE campaign_id=? AND state='queued' AND id IN ("+','.join('?'*len(chunk))+')',
                                 (SUPERSEDED+' '+cid,predecessor,*chunk)).rowcount
            if changed != len(chunk): raise ValueError('Predecessor queued probes changed during import')
        db.commit()
        return {'campaign_id':cid,'added':added,'searches':len(plan),'imported_complete':config['predecessor']['searches_imported_complete'],
                'continued':config['predecessor']['searches_continued'],'superseded':len(superseded)}
    except Exception:
        db.rollback(); raise


def enqueue(db, spec, descriptors, bank=None, predecessor=None, coherence=None, deep_modulus=None):
    """Close the menu gate durably while auditing; no generation before full audit.

    predecessor: id of a `blocked` campaign to continue (v4 successor). Then `spec` may be None (the
    predecessor's frozen spec is inherited) and `descriptors` are the endpoints that will serve.
    """
    if predecessor is not None and coherence is None: raise ValueError('coherence required for a successor audit')
    gate=digest(['threshold-audit-gate',spec] if predecessor is None else ['threshold-successor-gate',predecessor])
    history=[]
    db.execute('BEGIN IMMEDIATE')
    try:
        active=db.execute("SELECT id,config_json,state,error FROM threshold_campaigns WHERE state!='complete'").fetchall()
        if predecessor is not None:
            blocked=[a for a in active if a[0]==predecessor and a[2]=='blocked']
            if not blocked:
                raise ValueError('Declared predecessor must be an unresolved blocked campaign')
            if spec is not None and spec != json.loads(blocked[0][1])['spec']:
                raise ValueError('Successor spec must equal the predecessor spec')
            chain_ids={c for c,_ in lineage(db,predecessor)}
            others=[a for a in active if a[0] not in chain_ids]
            if len(others)==1 and others[0][2] in ('baseline','searching') and json.loads(others[0][1]).get('predecessor',{}).get('campaign_id')==predecessor:
                db.commit(); return {'campaign_id':others[0][0],'added':0,'existing':True}
            # A failed successor audit leaves this predecessor's own gate blocked, and a killed import leaves it
            # 'auditing' with a dead pid; a retry keeps that history inside the new gate instead of a hand edit.
            stale=[a for a in others if a[0]==gate and (a[2]=='blocked' or (a[2]=='auditing' and not _gate_alive(json.loads(a[1]))))]
            if stale:
                history=json.loads(stale[0][1]).get('previous_failures',[])+[{'state':stale[0][2],'error':stale[0][3],'at':time.time()}]
                db.execute("DELETE FROM threshold_campaigns WHERE id=? AND state=?",(gate,stale[0][2]))
            if [a for a in others if a[0]!=gate]:
                raise ValueError('An existing active or blocked threshold campaign must be resolved first')
            if [a for a in others if a[0]==gate and not any(a is s for s in stale)]:
                raise ValueError('A successor audit for this predecessor is already running')
        elif active:
            if len(active)==1 and active[0][2] in ('baseline','searching'):
                previous=json.loads(active[0][1])
                expected={str(p):{'binding':d['flare']['batch']['binding'],
                                  'qualification_sha256':d['flare']['batch']['qualification_sha256']} for p,d in descriptors.items()}
                if (previous.get('spec')==spec and previous.get('bindings')==expected
                        and previous.get('policy_sha256')==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                        and previous.get('metric_sha256')==METRIC_SHA
                        and previous.get('auditor_sha256')==hashlib.sha256(Path(reuse_audit.__file__).read_bytes()).hexdigest()):
                    db.commit()
                    return {'campaign_id':active[0][0], 'added':0}
            raise ValueError('An existing active or blocked threshold campaign must be resolved first')
        db.execute("INSERT INTO threshold_campaigns VALUES(?,?,'auditing',?,NULL)",
                   (gate,encode({'schema':'identity-threshold.audit-pending.v1','spec_sha256':digest(spec),'predecessor':predecessor,'previous_failures':history,
                                 'pid':os.getpid(),'boot_id':_boot_id()}),time.time()))
        db.commit()
    except Exception:
        db.rollback()
        raise
    try:
        if predecessor is not None:
            return _enqueue_successor_after_gate(db,predecessor,descriptors,bank,coherence,audit_gate=gate,deep_modulus=deep_modulus)
        return _enqueue_after_gate(db,spec,descriptors,bank=bank,audit_gate=gate)
    except BaseException as exc:
        db.rollback()
        with db:
            db.execute("UPDATE threshold_campaigns SET state='blocked',error=? WHERE id=? AND state='auditing'",
                       (f'audit_failed_or_interrupted: {type(exc).__name__}: {str(exc)[:300]}',gate))
        raise


def audit_next(db, descriptors, bank, coherence):
    """Audit one frozen search outside write lock, then atomically admit it."""
    row=db.execute("""SELECT s.id,s.campaign_id,s.layer,s.neuron,s.sign,s.source_snapshot_json,c.config_json
      FROM threshold_searches s JOIN threshold_campaigns c ON c.id=s.campaign_id
      WHERE s.audit_done=0 AND c.state IN ('baseline','searching')
      ORDER BY s.priority,s.layer,s.neuron,s.sign LIMIT 1""").fetchone()
    if row is None: return None
    sid,cid,layer,neuron,sign,snapshot_raw,raw_config=row
    config=json.loads(raw_config);snapshot=json.loads(snapshot_raw)
    if (config['policy_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            or config['auditor_sha256']!=hashlib.sha256(Path(reuse_audit.__file__).read_bytes()).hexdigest()
            or config['metric_sha256']!=METRIC_SHA):
        raise IntegrityError('Audit source changed')
    for p,expected in config['bindings'].items():
        d=descriptors[int(p)]['flare']['batch']
        if d.get('qualified') is not True or d.get('binding')!=expected['binding'] or d.get('qualification_sha256')!=expected['qualification_sha256']:
            raise IntegrityError('Audit B4 binding changed')
    calibration=bank.root/f"layer_{layer:02d}.npz"
    if hashlib.sha256(calibration.read_bytes()).hexdigest()!=config['calibration_files'][str(calibration)]:
        raise IntegrityError('Audit calibration changed')
    if hashlib.sha256(Path(config['builder_file']).read_bytes()).hexdigest()!=config['builder_sha256']:
        raise IntegrityError('Audit vector builder changed')
    fields=reuse_audit.FIELDS;results=[]
    for item in snapshot:
        values=db.execute('SELECT '+','.join(fields)+' FROM jobs WHERE id=?',(item['id'],)).fetchone()
        if values is None:raise IntegrityError('Frozen historical source missing')
        source=dict(zip(fields,values))
        if reuse_audit.row_fingerprint(source)!=item['fingerprint']:
            raise IntegrityError('Frozen historical source changed')
        try:
            program,_=bank.program([{'kind':'neuron','layer':layer,'neuron':neuron,'dose':source['dose']}],'all')
            if hashlib.sha256(calibration.read_bytes()).hexdigest()!=config['calibration_files'][str(calibration)]:
                raise IntegrityError('Audit calibration changed')
            if source['port'] not in (*descriptors,8930): raise ValueError('historical_port_not_auditable')
            evidence=reuse_audit.audit(source,descriptors[8931] if source['port']==8930 else descriptors[source['port']],
                                       program,config['calibration_files'][str(calibration)])
            accepted,reason=1,None
        except IntegrityError:
            raise
        except (OSError,ValueError,KeyError,TypeError) as exc:
            accepted,reason=0,str(exc)
            evidence={'source_row_sha256':item['fingerprint'],'error':reason}
        results.append((source,accepted,reason,evidence))
    db.execute('BEGIN IMMEDIATE')
    try:
        current=db.execute('SELECT audit_done,source_snapshot_json FROM threshold_searches WHERE id=?',(sid,)).fetchone()
        if not current or current[0] or current[1]!=snapshot_raw:
            db.rollback();return None
        if db.execute('SELECT state FROM threshold_campaigns WHERE id=?',(cid,)).fetchone()[0] not in ('baseline','searching'):
            db.rollback();return None
        for source,accepted,reason,evidence in results:
            values=db.execute('SELECT '+','.join(fields)+' FROM jobs WHERE id=?',(source['id'],)).fetchone()
            if values is None or reuse_audit.row_fingerprint(dict(zip(fields,values)))!=reuse_audit.row_fingerprint(source):
                raise IntegrityError('Historical row changed during audit')
            db.execute("INSERT INTO threshold_reuse VALUES(?,?,?,?,?,?,?,?,?)",
                       (sid,source['id'],abs(source['dose']),float(source['dose']).hex(),accepted,reason,
                        digest(evidence),encode(evidence),None))
        state=db.execute('SELECT state FROM threshold_campaigns WHERE id=?',(cid,)).fetchone()[0]
        db.execute("UPDATE threshold_searches SET audit_done=1,state=? WHERE id=?",
                   ('running' if state=='searching' else 'waiting_baseline',sid))
        if state=='searching':advance(db,cid,sid,coherence)
        db.commit()
        return {'search_id':sid,'layer':layer,'neuron':neuron,'sign':sign,'audited':len(results),
                'accepted':sum(item[1] for item in results)}
    except Exception:
        db.rollback();raise

def request_record(db, identity, descriptor, port, request):
    row = db.execute('SELECT campaign_id,layer FROM jobs WHERE id=?',(identity,)).fetchone()
    if not row or not row[0]:
        return
    config = json.loads(db.execute('SELECT config_json FROM threshold_campaigns WHERE id=?',(row[0],)).fetchone()[0])
    try:
        if config['builder_file'] and hashlib.sha256(Path(config['builder_file']).read_bytes()).hexdigest()!=config['builder_sha256']:
            raise IntegrityError('Vector builder source changed')
        for path,sha in config['calibration_files'].items():
            if row[1]<0 or Path(path).name==f'layer_{row[1]:02d}.npz':
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:
                    raise IntegrityError('Frozen calibration file changed')
    except OSError as exc:
        raise IntegrityError('Frozen calibration/builder source unavailable') from exc
    expected = config['bindings'].get(str(port))
    if expected is None:
        raise IntegrityError('Port is not a frozen campaign binding')
    batch = descriptor['flare']['batch']
    if (config['metric_sha256'] != METRIC_SHA or config['policy_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            or config['auditor_sha256'] != hashlib.sha256(Path(reuse_audit.__file__).read_bytes()).hexdigest()
            or batch['binding'] != expected['binding'] or batch['qualification_sha256'] != expected['qualification_sha256']):
        raise IntegrityError('Threshold metric/policy/runtime binding changed')
    if not (request['messages']==[{'role':'user','content':config['prompt']}] and request['seed']==config['seed']
            and request['max_tokens']==config['max_tokens'] and request['temperature']==config['temperature']
            and request['chat_template_kwargs']=={'enable_thinking':False}):
        raise IntegrityError('Threshold request settings changed')
    compact = {k:v for k,v in request.items() if k!='steering'}
    db.execute('UPDATE jobs SET request_json=?,request_sha256=? WHERE id=?', (encode(compact),digest(request),identity))


def finish(db, identity, coherence):
    row = db.execute('SELECT campaign_id,search_id,state,answer,tokens,binding_json,artifact,artifact_sha256,port FROM jobs WHERE id=?',(identity,)).fetchone()
    if not row or not row[0]:
        return False
    cid,sid,state,answer,tokens,binding,artifact,artifact_sha,port = row
    if state != 'done':
        db.execute("UPDATE threshold_campaigns SET state='blocked',error=? WHERE id=?",('Uncertain sample '+identity,cid))
        if sid: db.execute("UPDATE threshold_searches SET state='uncertain',stop_reason='uncertain_no_retry' WHERE id=?",(sid,))
        return True
    config = json.loads(db.execute('SELECT config_json FROM threshold_campaigns WHERE id=?',(cid,)).fetchone()[0])
    expected = config['bindings'].get(str(port),{}).get('binding')
    if not (expected and binding and json.loads(binding)==expected and artifact and artifact_sha):
        db.execute("UPDATE threshold_campaigns SET state='blocked',error=? WHERE id=?",('Binding/artifact missing or different '+identity,cid))
        return True
    m = metrics.measure(answer); m['coherent'] = int(coherence(answer,tokens))
    db.execute('UPDATE jobs SET metrics_json=?,coherent=? WHERE id=?',(encode(m),m['coherent'],identity))
    if db.execute('SELECT state FROM threshold_campaigns WHERE id=?',(cid,)).fetchone()[0]=='blocked':
        return True
    if sid is None:
        # Duplicate completion is evidence replay, never a new state transition.
        if db.execute('SELECT state FROM threshold_campaigns WHERE id=?',(cid,)).fetchone()[0] != 'baseline':
            return True
        bases = db.execute("SELECT id,answer,metrics_json FROM jobs WHERE campaign_id=? AND probe_role='baseline' AND state='done'",(cid,)).fetchall()
        if len(bases) != len(config['bindings']) or any(x[2] is None for x in bases):
            return True
        if len({a for _,a,_ in bases})!=1 or any(json.loads(v)['mode']!='boilerplate' or not json.loads(v)['coherent'] for _,_,v in bases):
            db.execute("UPDATE threshold_campaigns SET state='blocked',error='baseline_invalid_or_shards_differ' WHERE id=?",(cid,))
            return True
        changed = db.execute("UPDATE threshold_campaigns SET state='searching' WHERE id=? AND state='baseline'",(cid,)).rowcount
        if changed != 1: return True
        for sid,done in db.execute('SELECT id,audit_done FROM threshold_searches WHERE campaign_id=?',(cid,)).fetchall():
            db.execute("UPDATE threshold_searches SET state=? WHERE id=?",('running' if done else 'waiting_audit',sid))
            if done: advance(db,cid,sid,coherence)
        return True
    advance(db,cid,sid,coherence)
    return True


def _block_search(db, cid, sid, campaign_error, search_reason):
    db.execute("UPDATE threshold_campaigns SET state='blocked',error=? WHERE id=?",(campaign_error,cid))
    db.execute("UPDATE threshold_searches SET state='blocked',stop_reason=? WHERE id=?",(search_reason,sid))
    return True


def advance(db,cid,sid,coherence,config=None):
    s = db.execute('SELECT layer,neuron,sign,target_bits FROM threshold_searches WHERE id=?',(sid,)).fetchone()
    config = config or campaign_config(db, cid)
    base = baseline_point(db, cid, config)
    if base is None:
        return _block_search(db,cid,sid,'baseline_source_missing_or_changed','baseline_integrity_mismatch')
    points = [base]
    reused=0
    for jid,mag,evidence_json in db.execute('SELECT source_job_id,magnitude,evidence_json FROM threshold_reuse WHERE search_id=? AND accepted=1',(sid,)):
        fields=reuse_audit.FIELDS
        values=db.execute('SELECT '+','.join(fields)+' FROM jobs WHERE id=?',(jid,)).fetchone()
        row=None if values is None else dict(zip(fields,values))
        evidence=json.loads(evidence_json)
        if row is None or reuse_audit.row_fingerprint(row)!=evidence['source_row_sha256']:
            return _block_search(db,cid,sid,'reused_source_row_changed','reuse_integrity_mismatch')
        m=metrics.measure(row['answer']);m['coherent']=int(coherence(row['answer'],row['tokens']))
        db.execute('UPDATE threshold_reuse SET metrics_json=? WHERE search_id=? AND source_job_id=?',(encode(m),sid,jid))
        points.append({'job_id':jid,'magnitude':mag,'reused':True,**m});reused+=1
    imported = db.execute('SELECT evidence_json FROM threshold_successor_searches WHERE search_id=?',(sid,)).fetchone()
    if imported:
        fields=reuse_audit.JOB_FIELDS
        for item in json.loads(imported[0])['jobs']:
            values=db.execute('SELECT '+','.join(fields)+' FROM jobs WHERE id=?',(item['id'],)).fetchone()
            row=None if values is None else dict(zip(fields,values))
            if row is None or row['state']!='done' or reuse_audit.job_fingerprint(row)!=item['sha256']:
                return _block_search(db,cid,sid,'imported_source_row_changed','import_integrity_mismatch')
            points.append({'job_id':row['id'],'magnitude':abs(row['dose']),'imported':True,**json.loads(row['metrics_json'])})
    for jid,dose,raw in db.execute("SELECT id,dose,metrics_json FROM jobs WHERE search_id=? AND state='done' ORDER BY abs(dose)",(sid,)):
        if raw is None: return True
        points.append({'job_id':jid,'magnitude':abs(dose),**json.loads(raw)})
    result = observed(points,s[3]); used=len(points)-1-reused
    db.execute("""UPDATE threshold_searches SET probes_used=?,reused_observations=?,first_change_json=?,
      first_coherent_departure_json=?,collapse_json=?,reversions_json=? WHERE id=?""",
      (used,reused,*(None if result[k] is None else encode(result[k]) for k in ('first_change','first_coherent_departure','collapse')),encode(result['reversions']),sid))
    probe = next_probe(points,s[3])
    if probe:
        _job(db,cid,sid,s[0],s[1],s[2],probe[0],probe[1])
    else:
        db.execute("UPDATE threshold_searches SET state='complete',stop_reason='budget_exhausted' WHERE id=?",(sid,))
        if not db.execute("SELECT 1 FROM threshold_searches WHERE campaign_id=? AND state!='complete' LIMIT 1",(cid,)).fetchone():
            db.execute("UPDATE threshold_campaigns SET state='complete' WHERE id=?",(cid,))
    return True
