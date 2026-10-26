"""Resumable source-bound CPU NLP and lexical retrieval for the literature index.

Only nlp_* tables are written. Offsets are Unicode code-point [start,end) spans
into the existing normalized source text. spaCy entities are model predictions;
quoted dialogue and scene boundaries remain explicit heuristic candidates.
TF-IDF similarities are lexical cosine, never semantic embeddings/coreference.
"""
from __future__ import annotations

import argparse
import fcntl
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any

MODEL = 'en_core_web_sm'
MODEL_VERSION = '3.8.0'
MODEL_URL = 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
MODEL_WHEEL_SHA256 = '1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85'
SCHEMA = 'literature.nlp.v1'
PIPELINE: Any = None
MATRIX: Any = None
SHARED_HANDLES: list[Any] = []
QUOTE = re.compile(r'“[^”]*”|"(?:[^"\\]|\\.)*"|(?m:^[ \t]*[—–][^\n]+)')
MARKER = re.compile(r'(?m)^[ \t]*(?:\*[ \t]*){3,}$|^[ \t]*(?:#[ \t]*){3,}$|^[ \t]*[⁂❦][ \t]*$')
TIME_CUE = re.compile(r'(?im)^[ \t]*(?:meanwhile|later(?: that (?:day|night))?|the next (?:morning|day|night)|that (?:evening|night)|at dawn|at dusk|hours later|days later)\b')
LOCATION_CUE = re.compile(r'(?im)^[ \t]*(?:inside|outside|back at|across town|at the|in the)\b[^.!?\n]{0,80}')
TFIDF_OPTIONS = {'lowercase': True, 'strip_accents': 'unicode', 'ngram_range': (1, 2),
                 'sublinear_tf': True, 'norm': 'l2', 'max_features': 50000,
                 'min_df': 1, 'max_df': 1.0, 'stop_words': 'english'}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def connect(database: Path) -> sqlite3.Connection:
    db = sqlite3.connect(database, timeout=60)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA busy_timeout=60000')
    db.execute('PRAGMA synchronous=NORMAL')
    db.execute('PRAGMA wal_autocheckpoint=0')
    return db


def initialize(database: Path) -> None:
    if not database.is_file():
        raise ValueError('Existing literature SQLite database is required')
    with connect(database) as db:
        required = {'books', 'documents', 'units'}
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not required <= tables:
            raise ValueError('Database lacks the literature source tables')
        db.executescript('''
        CREATE TABLE IF NOT EXISTS nlp_runs(id TEXT PRIMARY KEY,created REAL,updated REAL,state TEXT,manifest TEXT,error TEXT);
        CREATE TABLE IF NOT EXISTS nlp_windows(run_id TEXT,canonical_id TEXT,text_sha256 TEXT,state TEXT,error TEXT,PRIMARY KEY(run_id,canonical_id));
        CREATE TABLE IF NOT EXISTS nlp_window_sources(run_id TEXT,unit_id TEXT,canonical_id TEXT,book_id TEXT,document_id TEXT,start INTEGER,end INTEGER,split TEXT,PRIMARY KEY(run_id,unit_id));
        CREATE TABLE IF NOT EXISTS nlp_spans(run_id TEXT,canonical_id TEXT,kind TEXT,ordinal INTEGER,start INTEGER,end INTEGER,text TEXT,label TEXT,evidence TEXT,PRIMARY KEY(run_id,canonical_id,kind,ordinal));
        CREATE TABLE IF NOT EXISTS nlp_nodes(run_id TEXT,id TEXT,kind TEXT,label TEXT,evidence TEXT,PRIMARY KEY(run_id,id));
        CREATE TABLE IF NOT EXISTS nlp_edges(run_id TEXT,id TEXT,source TEXT,target TEXT,kind TEXT,score REAL,evidence TEXT,PRIMARY KEY(run_id,id));
        CREATE TABLE IF NOT EXISTS nlp_lexical_models(run_id TEXT PRIMARY KEY,manifest TEXT,ids TEXT,vocabulary TEXT,idf BLOB,indptr BLOB,indices BLOB,weights BLOB,rows INTEGER,columns INTEGER);
        CREATE INDEX IF NOT EXISTS nlp_span_lookup ON nlp_spans(run_id,canonical_id,kind,start);
        CREATE INDEX IF NOT EXISTS nlp_sources_canonical ON nlp_window_sources(run_id,canonical_id);
        CREATE INDEX IF NOT EXISTS nlp_edges_source ON nlp_edges(run_id,source,kind);
        CREATE INDEX IF NOT EXISTS nlp_edges_target ON nlp_edges(run_id,target,kind);
        ''')


def model_manifest() -> dict:
    """No downloads or fallback masquerading as NER; bind installed model bytes."""
    try:
        spacy_version = importlib.metadata.version('spacy')
        model_version = importlib.metadata.version(MODEL)
        sklearn_version = importlib.metadata.version('scikit-learn')
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError('Install spaCy, scikit-learn and pinned en_core_web_sm 3.8.0 before NLP indexing') from exc
    if model_version != MODEL_VERSION or not spacy_version.startswith('3.8.'):
        raise RuntimeError('Expected spaCy 3.8.x and en_core_web_sm 3.8.0; model changes require an explicit binding change')
    spec = importlib.util.find_spec(MODEL)
    if spec is None or spec.origin is None:
        raise RuntimeError('Pinned spaCy model cannot be located')
    package = Path(spec.origin).parent
    model_dir = package / f'{MODEL}-{MODEL_VERSION}'
    if not model_dir.is_dir():
        raise RuntimeError('Pinned model data directory is absent')
    files = {str(p.relative_to(model_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(model_dir.rglob('*')) if p.is_file()}
    if not files:
        raise RuntimeError('Pinned model contains no data files')
    return {'schema': SCHEMA, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python': sys.version.split()[0], 'spacy': spacy_version, 'sklearn': sklearn_version,
            'model': MODEL, 'model_version': model_version, 'model_release': MODEL_URL,
            'published_wheel_sha256': MODEL_WHEEL_SHA256, 'installed_model_files': files,
            'model_content_sha256': digest(files), 'pipeline': ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'],
            'offset_semantics': 'Unicode code points; [start,end) relative to exact stored window text',
            'ner_semantics': 'spaCy model predictions; surface-form entities, no coreference',
            'dialogue_semantics': 'heuristic quoted or dash-prefixed spans, no speaker attribution',
            'scene_semantics': 'heuristic structural/time/location candidates, not verified scenes',
            'lexical_semantics': 'TF-IDF lexical cosine; unique exact text only; not semantic embeddings',
            'tfidf_options': TFIDF_OPTIONS}


def available_cpus() -> list[int]:
    """Reserve a complete physical core (all its online SMT siblings)."""
    allowed = sorted(os.sched_getaffinity(0))
    siblings: dict[tuple[str, str], list[int]] = {}
    for cpu in allowed:
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            physical = ((base / 'physical_package_id').read_text().strip(), (base / 'core_id').read_text().strip())
        except OSError:
            physical = ('unknown', str(cpu))
        siblings.setdefault(physical, []).append(cpu)
    if len(siblings) <= 1:
        raise RuntimeError('At least two physical cores are needed to retain shell headroom')
    # Respect a host supervisor's complete physical-core reservation, whichever core it chose.
    online = sorted(int(path.name[3:]) for path in Path('/sys/devices/system/cpu').glob('cpu[0-9]*') if path.name[3:].isdigit())
    host_groups: dict[tuple[str, str], set[int]] = {}
    for cpu in online:
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            group = ((base / 'physical_package_id').read_text().strip(), (base / 'core_id').read_text().strip())
        except OSError:
            group = ('unknown', str(cpu))
        host_groups.setdefault(group, set()).add(cpu)
    if any(not group.intersection(allowed) for group in host_groups.values()):
        return allowed
    reserved = set(next(group for group in siblings.values() if min(allowed) in group))
    return [cpu for cpu in allowed if cpu not in reserved]


def worker_init(slots: Any) -> None:
    global PIPELINE
    cpu = slots.get(timeout=60)
    os.sched_setaffinity(0, {cpu})
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import spacy
    spacy.require_cpu()
    PIPELINE = spacy.load(MODEL)
    if set(PIPELINE.pipe_names) != {'tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'}:
        raise RuntimeError('Loaded spaCy pipeline differs from manifest')


def heuristic_spans(text: str) -> list[dict]:
    spans = []
    for match in QUOTE.finditer(text):
        spans.append({'kind': 'dialogue_candidate', 'start': match.start(), 'end': match.end(),
                      'text': match.group(), 'label': 'quoted_or_dash_prefixed',
                      'evidence': {'method': 'heuristic', 'speaker': None}})
    for regex, label in [(MARKER, 'explicit_scene_marker'), (TIME_CUE, 'time_shift_lexical_cue'), (LOCATION_CUE, 'location_lexical_cue')]:
        for match in regex.finditer(text):
            spans.append({'kind': 'scene_boundary_candidate', 'start': match.start(), 'end': match.end(),
                          'text': match.group(), 'label': label,
                          'evidence': {'method': 'heuristic', 'verified_scene_boundary': False}})
    return spans


def analyze(item: dict) -> dict:
    """Workers only read text; the parent alone writes SQLite transactions."""
    try:
        text = item['text']
        if text_sha(text) != item['text_sha256']:
            raise ValueError('Worker source hash differs')
        doc = PIPELINE(text)
        def token_record(token: Any) -> dict:
            return {'text': token.text, 'lemma': token.lemma_, 'pos': token.pos_, 'dependency': token.dep_,
                    'start': token.idx, 'end': token.idx + len(token.text)}
        spans = [{'kind': 'sentence', 'start': s.start_char, 'end': s.end_char, 'text': s.text,
                  'label': 'sentence', 'evidence': {'method': 'spacy_parser', 'window_boundary_may_clip_sentence': True,
                      'root': token_record(s.root), 'subjects': [token_record(t) for t in s if t.dep_ in {'nsubj', 'nsubjpass', 'csubj'}],
                      'verbs': [token_record(t) for t in s if t.pos_ in {'VERB', 'AUX'}]}}
                 for s in doc.sents if s.text.strip()]
        spans.extend({'kind': 'entity', 'start': e.start_char, 'end': e.end_char, 'text': e.text,
                      'label': e.label_, 'evidence': {'method': 'spacy_ner', 'model': MODEL, 'version': MODEL_VERSION,
                                                    'entity_resolution': 'surface_form_only_no_coreference'}} for e in doc.ents)
        spans.extend(heuristic_spans(text))
        for span in spans:
            if not 0 <= span['start'] < span['end'] <= len(text) or text[span['start']:span['end']] != span['text']:
                raise ValueError('NLP span offset mismatch')
        result = {'canonical_id': item['canonical_id'], 'spans': spans, 'state': 'complete', 'error': None}
        result['materialized'] = materialize(result)
        return result
    except Exception as exc:
        return {'canonical_id': item['canonical_id'], 'spans': [], 'state': 'failed', 'error': f'{type(exc).__name__}: {exc}'}


def source_snapshot(database: Path) -> tuple[list[dict], list[dict], str]:
    """Deduplicate by exact UTF-8 text, independently verify stored aliases/hashes."""
    unique: dict[str, dict] = {}
    aliases = []
    with connect(database) as db:
        documents = {}
        for row in db.execute('SELECT id,text,sha256 FROM documents'):
            if text_sha(row['text']) != row['sha256']:
                raise ValueError('Stored document hash differs: ' + row['id'])
            documents[row['id']] = row['text']
        for row in db.execute('''SELECT u.*,d.sha256 AS document_sha256,d.excluded_reason,
                                b.title,b.sha256 AS book_sha256 FROM units u JOIN documents d ON d.id=u.document_id
                                JOIN books b ON b.id=u.book_id ORDER BY u.id'''):
            if row['excluded_reason']:
                continue
            text = row['text']
            if documents[row['document_id']][row['start']:row['end']] != text:
                raise ValueError('Stored window does not match its exact document source: ' + row['id'])
            checksum = text_sha(text)
            if row['text_sha256'] and checksum != row['text_sha256']:
                raise ValueError('Stored window text hash differs: ' + row['id'])
            # Content-derived identity ensures duplicate editions/windows never gain extra TF-IDF weight.
            canonical_id = checksum
            unique.setdefault(checksum, {'canonical_id': canonical_id, 'text_sha256': checksum, 'text': text})
            aliases.append({'unit_id': row['id'], 'canonical_id': canonical_id, 'book_id': row['book_id'],
                            'document_id': row['document_id'], 'start': row['start'], 'end': row['end'],
                            'split': row['split'], 'book_title': row['title'], 'book_sha256': row['book_sha256'],
                            'document_sha256': row['document_sha256']})
    return sorted(unique.values(), key=lambda r: r['canonical_id']), aliases, digest(aliases)


def edge(db: sqlite3.Connection, run_id: str, source: str, target: str, kind: str, evidence: dict, score: float = 1.0) -> None:
    identity = digest([source, target, kind, evidence])
    db.execute('INSERT OR IGNORE INTO nlp_edges VALUES(?,?,?,?,?,?,?)',
               (run_id, identity, source, target, kind, score, json.dumps(evidence, ensure_ascii=False)))


def node(db: sqlite3.Connection, run_id: str, identity: str, kind: str, label: str, evidence: dict) -> None:
    db.execute('INSERT OR IGNORE INTO nlp_nodes VALUES(?,?,?,?,?)',
               (run_id, identity, kind, label, json.dumps(evidence, ensure_ascii=False)))


def _store_result(db: sqlite3.Connection, run_id: str, result: dict) -> None:
    canonical = result['canonical_id']
    db.execute('UPDATE nlp_windows SET state=?,error=? WHERE run_id=? AND canonical_id=?',
               (result['state'], result['error'], run_id, canonical))
    if result['state'] != 'complete':
        return
    # This transaction writes each canonical window exactly once; failed attempts have no partial spans.
    ordinals: dict[str, int] = {}
    previous = None
    for span in result['spans']:
        kind = span['kind']; ordinal = ordinals.get(kind, 0); ordinals[kind] = ordinal + 1
        db.execute('INSERT OR IGNORE INTO nlp_spans VALUES(?,?,?,?,?,?,?,?,?)',
                   (run_id, canonical, kind, ordinal, span['start'], span['end'], span['text'], span['label'], json.dumps(span['evidence'])))
        sid = f'{kind}:{canonical}:{ordinal}'
        evidence = {'canonical_id': canonical, 'start': span['start'], 'end': span['end'], **span['evidence']}
        node(db, run_id, sid, kind, span['text'], evidence)
        edge(db, run_id, 'window:' + canonical, sid, 'contains_' + kind, evidence)
        if kind == 'sentence':
            if previous:
                edge(db, run_id, previous, sid, 'next_sentence', {'canonical_id': canonical})
            previous = sid
        if kind == 'entity':
            entity_id = 'entity_surface:' + digest([span['label'], span['text'].casefold()])
            node(db, run_id, entity_id, 'entity_surface', span['text'], {'label': span['label'], 'resolution': 'surface_form_only'})
            edge(db, run_id, sid, entity_id, 'mentions_surface', evidence)


def materialize(result: dict) -> dict:
    """Move graph hashing/JSON preparation to workers; keep SQLite ownership singular."""
    rows: dict[str, list[tuple]] = {'spans': [], 'nodes': [], 'edges': []}
    class Collector:
        def execute(self, sql: str, args: tuple) -> None:
            for table in rows:
                if sql.startswith('INSERT OR IGNORE INTO nlp_' + table + ' '):
                    rows[table].append(args[1:])
                    return
            if not sql.startswith('UPDATE nlp_windows '):
                raise ValueError('Unexpected graph materialization operation')
    _store_result(Collector(), '', result)
    return rows


def store_results(database: Path, run_id: str, results: list[dict]) -> None:
    with connect(database) as db:
        db.executemany('UPDATE nlp_windows SET state=?,error=? WHERE run_id=? AND canonical_id=?',
                       ((result['state'], result['error'], run_id, result['canonical_id']) for result in results))
        for result in results:
            if result['state'] != 'complete':
                continue
            rows = result.get('materialized') or materialize(result)
            for table, placeholders in [('spans', 9), ('nodes', 5), ('edges', 7)]:
                db.executemany('INSERT OR IGNORE INTO nlp_' + table + ' VALUES(' + ','.join('?' for _ in range(placeholders)) + ')',
                               ((run_id, *row) for row in rows[table]))


def initialize_run(database: Path, manifest: dict, items: list[dict], aliases: list[dict]) -> str:
    run_id = digest(manifest)
    with connect(database) as db:
        db.execute('INSERT OR IGNORE INTO nlp_runs VALUES(?,?,?,?,?,?)',
                   (run_id, time.time(), time.time(), 'pending', json.dumps(manifest), None))
        db.executemany('INSERT OR IGNORE INTO nlp_windows VALUES(?,?,?,?,?)',
                       ((run_id, item['canonical_id'], item['text_sha256'], 'pending', None) for item in items))
        for item in items:
            node(db, run_id, 'window:' + item['canonical_id'], 'window', item['canonical_id'],
                 {'text_sha256': item['text_sha256'], 'duplicate_weight': 1})
        for alias in aliases:
            db.execute('INSERT OR IGNORE INTO nlp_window_sources VALUES(?,?,?,?,?,?,?,?)',
                       (run_id, *(alias[k] for k in ('unit_id', 'canonical_id', 'book_id', 'document_id', 'start', 'end', 'split'))))
            node(db, run_id, 'book:' + alias['book_id'], 'book', alias['book_title'], {'sha256': alias['book_sha256']})
            node(db, run_id, 'document:' + alias['document_id'], 'document', alias['document_id'], {'sha256': alias['document_sha256']})
            edge(db, run_id, 'book:' + alias['book_id'], 'document:' + alias['document_id'], 'contains_document', {'book_id': alias['book_id']})
            edge(db, run_id, 'document:' + alias['document_id'], 'window:' + alias['canonical_id'], 'contains_window', alias)
    return run_id


def lexical_worker_init(slots: Any, arrays: list[dict], shape: tuple[int, int]) -> None:
    global MATRIX, SHARED_HANDLES
    os.sched_setaffinity(0, {slots.get(timeout=60)})
    import numpy as np
    from multiprocessing.shared_memory import SharedMemory
    from scipy.sparse import csr_matrix
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    SHARED_HANDLES = [SharedMemory(name=spec['name']) for spec in arrays]
    views = [np.ndarray(spec['shape'], dtype=spec['dtype'], buffer=handle.buf)
             for spec, handle in zip(arrays, SHARED_HANDLES)]
    MATRIX = csr_matrix(tuple(views), shape=shape, copy=False)


def similarity_rows(task: tuple[int, int, int]) -> list[tuple[int, int, float]]:
    import numpy as np
    start, stop, top_k = task
    scores = (MATRIX[start:stop] @ MATRIX.T).tocsr()
    result = []
    for local in range(stop - start):
        row = scores.getrow(local)
        keep = (row.indices != start + local) & (row.data > 0)
        indices, values = row.indices[keep], row.data[keep]
        if len(values) > top_k:
            threshold = np.partition(values, len(values) - top_k)[len(values) - top_k]
            chosen = values >= threshold
            indices, values = indices[chosen], values[chosen]
        pairs = sorted(zip(indices, values), key=lambda p: (-float(p[1]), int(p[0])))[:top_k]
        result.extend((start + local, int(index), float(score)) for index, score in pairs)
    return result


def build_lexical(database: Path, run_id: str, items: list[dict], cpus: list[int], top_k: int = 5) -> dict:
    global MATRIX
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    with connect(database) as db:
        if db.execute('SELECT 1 FROM nlp_lexical_models WHERE run_id=?', (run_id,)).fetchone():
            return {'state': 'complete', 'resumed': True}
    ids = [item['canonical_id'] for item in items]
    vectorizer = TfidfVectorizer(**TFIDF_OPTIONS, dtype=np.float32)
    try:
        MATRIX = vectorizer.fit_transform(item['text'] for item in items).tocsr()
    except ValueError as exc:
        if 'empty vocabulary' not in str(exc):
            raise
        with connect(database) as db:
            manifest = {'state': 'empty_vocabulary', 'semantics': 'TF-IDF lexical cosine; no lexical features'}
            db.execute('INSERT INTO nlp_lexical_models VALUES(?,?,?,?,?,?,?,?,?,?)',
                       (run_id, json.dumps(manifest), json.dumps(ids), '{}', b'', np.zeros(len(ids) + 1, dtype=np.int64).tobytes(), b'', b'', len(ids), 0))
        return manifest
    manifest = {'state': 'complete', 'semantics': 'TF-IDF lexical cosine, not semantic embeddings',
                'options': TFIDF_OPTIONS, 'sklearn': importlib.metadata.version('scikit-learn'),
                'unique_windows': len(ids), 'top_k': top_k, 'idf_dtype': '<f4', 'indices_dtype': '<i8', 'weights_dtype': '<f4'}
    from multiprocessing.shared_memory import SharedMemory
    context = multiprocessing.get_context('spawn')
    slots = context.Queue()
    for cpu in cpus:
        slots.put(cpu)
    tasks = [(start, min(start + 64, len(ids)), top_k) for start in range(0, len(ids), 64)]
    # Shared-memory CSR keeps one matrix and avoids forking a threaded NumPy/model process.
    handles, arrays = [], []
    try:
        for array in (MATRIX.data, MATRIX.indices, MATRIX.indptr):
            handle = SharedMemory(create=True, size=max(1, array.nbytes)); handles.append(handle)
            np.ndarray(array.shape, dtype=array.dtype, buffer=handle.buf)[:] = array
            arrays.append({'name': handle.name, 'shape': array.shape, 'dtype': array.dtype.str})
        with ProcessPoolExecutor(max_workers=len(cpus), mp_context=context, initializer=lexical_worker_init, initargs=(slots, arrays, MATRIX.shape)) as pool:
            for batch in pool.map(similarity_rows, tasks, chunksize=1):
                with connect(database) as db:
                    for left, right, score in batch:
                        edge(db, run_id, 'window:' + ids[left], 'window:' + ids[right], 'lexical_similarity',
                             {'method': 'tfidf_cosine', 'unique_text_weight': 1, 'directed_top_k': top_k}, score)
    finally:
        for handle in handles:
            handle.close(); handle.unlink()
        slots.close(); slots.join_thread()
    # Publication follows all similarity edges; interruption recomputes idempotently.
    with connect(database) as db:
        db.execute('INSERT INTO nlp_lexical_models VALUES(?,?,?,?,?,?,?,?,?,?)',
                   (run_id, json.dumps(manifest), json.dumps(ids), json.dumps({key: int(value) for key, value in vectorizer.vocabulary_.items()}),
                    vectorizer.idf_.astype('<f4').tobytes(), MATRIX.indptr.astype('<i8').tobytes(),
                    MATRIX.indices.astype('<i8').tobytes(), MATRIX.data.astype('<f4').tobytes(), *MATRIX.shape))
    MATRIX = None
    return manifest


def lexical_search(database: Path, run_id: str, query: str, limit: int = 12) -> dict:
    import numpy as np
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    if not isinstance(query, str) or not query.strip() or not 1 <= limit <= 100:
        raise ValueError('Supply nonempty query and limit 1–100')
    with connect(database) as db:
        row = db.execute('SELECT * FROM nlp_lexical_models WHERE run_id=?', (run_id,)).fetchone()
        if row is None:
            raise ValueError('Lexical index is not complete for this run')
        if not row['columns']:
            return {'semantics': 'TF-IDF lexical cosine', 'matches': []}
        ids, vocabulary = json.loads(row['ids']), json.loads(row['vocabulary'])
        vectorizer = TfidfVectorizer(**TFIDF_OPTIONS, vocabulary=vocabulary, dtype=np.float32)
        vectorizer.idf_ = np.frombuffer(row['idf'], dtype='<f4')
        matrix = csr_matrix((np.frombuffer(row['weights'], dtype='<f4'), np.frombuffer(row['indices'], dtype='<i8'),
                             np.frombuffer(row['indptr'], dtype='<i8')), shape=(row['rows'], row['columns']))
        scores = (matrix @ vectorizer.transform([query]).T).tocoo()
        ranked = sorted(zip(scores.row, scores.data), key=lambda p: (-float(p[1]), ids[p[0]]))[:limit]
        matches = []
        for index, score in ranked:
            aliases = [dict(r) for r in db.execute('SELECT * FROM nlp_window_sources WHERE run_id=? AND canonical_id=? ORDER BY unit_id', (run_id, ids[index]))]
            matches.append({'canonical_id': ids[index], 'score': float(score), 'sources': aliases})
    return {'run_id': run_id, 'semantics': 'TF-IDF lexical cosine; no semantic encoder', 'matches': matches}


def spans_for_unit(database: Path, run_id: str, unit_id: str) -> list[dict]:
    with connect(database) as db:
        source = db.execute('SELECT * FROM nlp_window_sources WHERE run_id=? AND unit_id=?', (run_id, unit_id)).fetchone()
        if source is None:
            raise ValueError('Unit is not in this NLP run')
        rows = []
        for row in db.execute('SELECT * FROM nlp_spans WHERE run_id=? AND canonical_id=? ORDER BY start,kind,ordinal', (run_id, source['canonical_id'])):
            item = dict(row)
            item.update(document_id=source['document_id'], document_start=source['start'] + row['start'],
                        document_end=source['start'] + row['end'], evidence=json.loads(row['evidence']))
            rows.append(item)
        return rows


def latest_run(database: Path, *, lexical: bool = False) -> dict | None:
    """Read-only discovery; an absent index is an explicit unavailable result."""
    if not database.is_file():
        return None
    with connect(database) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='nlp_runs'").fetchone():
            return None
        suffix = ' JOIN nlp_lexical_models l ON l.run_id=r.id' if lexical else ''
        row = db.execute('SELECT r.* FROM nlp_runs r' + suffix + ' ORDER BY r.created DESC LIMIT 1').fetchone()
        if row is None:
            return None
        result = dict(row)
        result['manifest'] = json.loads(row['manifest'])
        result['counts'] = {r[0]: r[1] for r in db.execute('SELECT state,count(*) FROM nlp_windows WHERE run_id=? GROUP BY state', (row['id'],))}
        return result


def entity_facets(database: Path, book_id: str = '', limit: int = 200, run_id: str | None = None) -> dict:
    if not 1 <= limit <= 1000:
        raise ValueError('Entity facet limit must be 1–1000')
    run = latest_run(database)
    if not run:
        return {'available': False, 'entities': []}
    run_id = run_id or run['id']
    where, args = "s.run_id=? AND s.kind='entity'", [run_id]
    if book_id:
        where += ' AND EXISTS(SELECT 1 FROM nlp_window_sources a WHERE a.run_id=s.run_id AND a.canonical_id=s.canonical_id AND a.book_id=?)'
        args.append(book_id)
    with connect(database) as db:
        rows = [dict(r) for r in db.execute('SELECT min(s.text) AS name,s.label AS kind,COUNT(DISTINCT s.canonical_id) AS count,COUNT(*) AS mentions '
                'FROM nlp_spans s WHERE ' + where + ' GROUP BY lower(s.text),s.label ORDER BY count DESC,name LIMIT ?', (*args, limit))]
    return {'available': True, 'run_id': run_id, 'entities': rows, 'model': MODEL, 'version': MODEL_VERSION,
            'claim': 'Model-predicted entity surfaces; unique-window counts; no coreference or causal claim'}


def graph(database: Path, unit_id: str, limit: int = 120, run_id: str | None = None) -> dict:
    if not 1 <= limit <= 500:
        raise ValueError('Graph limit must be 1–500')
    run = latest_run(database)
    if not run:
        return {'available': False, 'nodes': [], 'edges': []}
    run_id = run_id or run['id']
    with connect(database) as db:
        source = db.execute('SELECT * FROM nlp_window_sources WHERE run_id=? AND unit_id=?', (run_id, unit_id)).fetchone()
        if source is None:
            return {'available': False, 'run_id': run_id, 'nodes': [], 'edges': [], 'reason': 'Unit absent from indexed snapshot'}
        window = 'window:' + source['canonical_id']
        rows = db.execute('SELECT * FROM nlp_edges WHERE run_id=? AND source=? AND kind!=? ORDER BY id LIMIT ?',
                          (run_id, window, 'lexical_similarity', limit)).fetchall()
        ids = {window, 'document:' + source['document_id'], 'book:' + source['book_id']}
        edges = {r['id']: dict(r) for r in rows}
        ids.update(r['target'] for r in rows)
        for sid in list(ids):
            for row in db.execute("SELECT * FROM nlp_edges WHERE run_id=? AND source=? AND kind IN ('mentions_surface','next_sentence') LIMIT ?", (run_id, sid, limit)):
                if len(edges) >= limit:
                    break
                ids.add(row['target']); edges[row['id']] = dict(row)
        for row in db.execute("SELECT * FROM nlp_edges WHERE run_id=? AND source IN (?,?) AND target IN (?,?)",
                              (run_id, 'book:' + source['book_id'], 'document:' + source['document_id'], 'document:' + source['document_id'], window)):
            edges[row['id']] = dict(row)
        nodes = []
        for identity in sorted(ids):
            row = db.execute('SELECT * FROM nlp_nodes WHERE run_id=? AND id=?', (run_id, identity)).fetchone()
            if row:
                item = dict(row); item['evidence'] = json.loads(item['evidence']); nodes.append(item)
        for item in edges.values():
            item['evidence'] = json.loads(item['evidence'])
    return {'available': True, 'run_id': run_id, 'source': dict(source), 'nodes': nodes, 'edges': list(edges.values()),
            'spans': spans_for_unit(database, run_id, unit_id), 'edge_limit': limit,
            'claim': 'Source containment, sentence order and model entity-surface mentions; not a semantic or causal graph'}


def neighbors(database: Path, unit_id: str, limit: int = 12, run_id: str | None = None) -> dict:
    if not 1 <= limit <= 100:
        raise ValueError('Neighbor limit must be 1–100')
    run = latest_run(database, lexical=True)
    if not run:
        return {'available': False, 'rows': [], 'method': 'tfidf_cosine'}
    run_id = run_id or run['id']
    with connect(database) as db:
        source = db.execute('SELECT canonical_id FROM nlp_window_sources WHERE run_id=? AND unit_id=?', (run_id, unit_id)).fetchone()
        if source is None:
            return {'available': False, 'rows': [], 'method': 'tfidf_cosine'}
        rows = []
        for edge_row in db.execute("SELECT target,score FROM nlp_edges WHERE run_id=? AND source=? AND kind='lexical_similarity' ORDER BY score DESC,target LIMIT ?", (run_id, 'window:' + source[0], limit)):
            canonical = edge_row['target'].removeprefix('window:')
            row = db.execute('SELECT u.id,u.text,u.book_id,u.split,b.title AS book_title FROM nlp_window_sources a JOIN units u ON u.id=a.unit_id JOIN books b ON b.id=u.book_id WHERE a.run_id=? AND a.canonical_id=? ORDER BY a.unit_id LIMIT 1', (run_id, canonical)).fetchone()
            if row:
                rows.append({**dict(row), 'score': edge_row['score'], 'canonical_id': canonical})
    return {'available': True, 'rows': rows, 'run_id': run_id, 'model': 'TF-IDF', 'method': 'tfidf_cosine',
            'claim': 'Lexical word/bigram overlap, not semantic embeddings; exact duplicate windows counted once'}


def status(database: Path) -> dict:
    run = latest_run(database)
    return {'available': run is not None, 'latest_run': run, 'counts': run['counts'] if run else {},
            'facets': entity_facets(database) if run else {'available': False, 'entities': []}}


def graph_for_unit(database: Path, run_id: str, unit_id: str) -> dict:
    return graph(database, unit_id, run_id=run_id)


def _index(database: Path, *, workers: int = 30, limit: int | None = None, lexical: bool = True) -> dict:
    initialize(database)
    tools = model_manifest()
    items, aliases, source_hash = source_snapshot(database)
    if limit is not None:
        if limit < 1:
            raise ValueError('Limit must be positive')
        items = items[:limit]
        chosen = {item['canonical_id'] for item in items}
        aliases = [row for row in aliases if row['canonical_id'] in chosen]
        source_hash = digest(aliases)
    cpus = available_cpus()
    if not 1 <= workers <= len(cpus):
        raise ValueError(f'Workers must be 1–{len(cpus)} after reserving a physical core')
    cpus = cpus[:workers]
    manifest = {'tools': tools, 'source_snapshot_sha256': source_hash, 'canonical_windows': len(items),
                'source_windows': len(aliases), 'limit': limit, 'lexical_requested': lexical}
    run_id = initialize_run(database, manifest, items, aliases)
    with connect(database) as db:
        done = {r[0] for r in db.execute("SELECT canonical_id FROM nlp_windows WHERE run_id=? AND state='complete'", (run_id,))}
        db.execute("UPDATE nlp_runs SET state='running',updated=?,error=NULL WHERE id=?", (time.time(), run_id))
    pending = [item for item in items if item['canonical_id'] not in done]
    print(json.dumps({'event': 'nlp_start', 'run_id': run_id, 'unique_windows': len(items), 'pending': len(pending),
                      'duplicate_sources': len(aliases) - len(items), 'cpus': cpus, 'pid': os.getpid()}), flush=True)
    try:
        if pending:
            context = multiprocessing.get_context('spawn')
            slots = context.Queue()
            for cpu in cpus:
                slots.put(cpu)
            with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=worker_init, initargs=(slots,)) as pool:
                # Bound the submitted source payloads instead of duplicating the entire corpus into Futures.
                iterator = iter(pending)
                futures = {pool.submit(analyze, item) for item in [next(iterator, None) for _ in range(workers * 3)] if item is not None}
                count = 0
                buffer = []
                while futures:
                    future = next(as_completed(futures)); futures.remove(future)
                    buffer.append(future.result()); count += 1
                    item = next(iterator, None)
                    if item is not None:
                        futures.add(pool.submit(analyze, item))
                    if len(buffer) >= 32 or not futures:
                        store_results(database, run_id, buffer); buffer.clear()
                    if count % 100 == 0 or count == len(pending):
                        print(json.dumps({'event': 'nlp_progress', 'run_id': run_id, 'processed': count, 'pending_total': len(pending)}), flush=True)
        with connect(database) as db:
            counts = {r[0]: r[1] for r in db.execute('SELECT state,count(*) FROM nlp_windows WHERE run_id=? GROUP BY state', (run_id,))}
        if counts.get('failed', 0):
            raise RuntimeError(f"{counts['failed']} NLP windows failed; rerun resumes only unfinished windows")
        lexical_result = build_lexical(database, run_id, items, cpus) if lexical else {'state': 'not_requested'}
        with connect(database) as db:
            db.execute("UPDATE nlp_runs SET state='complete',updated=? WHERE id=?", (time.time(), run_id))
        return {'run_id': run_id, 'state': 'complete', 'counts': counts, 'lexical': lexical_result, 'source_windows': len(aliases)}
    except BaseException as exc:
        with connect(database) as db:
            db.execute("UPDATE nlp_runs SET state='interrupted',updated=?,error=? WHERE id=?", (time.time(), f'{type(exc).__name__}: {exc}', run_id))
        raise


def index(database: Path, *, workers: int = 30, limit: int | None = None, lexical: bool = True) -> dict:
    # flock is separate from SQLite's byte-range locks; readers/other table writers remain available.
    with database.open('rb') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('An NLP indexer already owns this literature database') from exc
        try:
            return _index(database, workers=workers, limit=limit, lexical=lexical)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=30)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--no-lexical', action='store_true')
    parser.add_argument('--search')
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.search:
        if not args.run_id:
            parser.error('--search requires --run-id')
        result = lexical_search(args.database, args.run_id, args.search, limit=args.limit or 12)
    else:
        result = index(args.database, workers=args.workers, limit=args.limit, lexical=not args.no_lexical)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
