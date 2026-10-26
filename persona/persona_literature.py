"""Private EPUB ingestion and source-grounded literature browsing.

Normalized spine text is preserved verbatim. Windows and scene markers are
structural candidates, not confirmed semantic scenes. No models or downloads.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
import zipfile

from persona_corpus import book_documents, sha256, text_digest, split_for, _zip_path
from persona_vectors import canonical_digest

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_SCHEMA = 'character-creation.literature-annotation.v1'
MARKER = re.compile(r'(?m)^[ \t]*(?:\*[ \t]*){3,}$|^[ \t]*(?:#[ \t]*){3,}$|^[ \t]*[⁂❦][ \t]*$|^[ \t]*(?:—[ \t]*){3,}$')
UNSUPPORTED = {'.mobi', '.pdf', '.lit', '.azw', '.azw3', '.htm', '.html'}


def connect(database: Path) -> sqlite3.Connection:
    db = sqlite3.connect(database, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA synchronous=NORMAL')
    return db


def initialize(database: Path) -> None:
    if not database.parent.is_dir():
        raise ValueError('Database parent must already exist')
    with connect(database) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript('''
        CREATE TABLE IF NOT EXISTS books(id TEXT PRIMARY KEY,title TEXT,author TEXT,path TEXT,sha256 TEXT,workgroup TEXT,content_sha256 TEXT,split TEXT);
        CREATE TABLE IF NOT EXISTS book_sources(book_id TEXT,path TEXT,verified REAL,PRIMARY KEY(book_id,path));
        CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY,book_id TEXT,member TEXT,ordinal INTEGER,text TEXT,sha256 TEXT,excluded_reason TEXT);
        CREATE TABLE IF NOT EXISTS units(id TEXT PRIMARY KEY,book_id TEXT,document_id TEXT,ordinal INTEGER,start INTEGER,end INTEGER,text TEXT,split TEXT,boundary TEXT,state TEXT DEFAULT 'pending',annotation TEXT,error TEXT,text_sha256 TEXT,canonical_id TEXT);
        CREATE TABLE IF NOT EXISTS literature_ingest_attempts(id INTEGER PRIMARY KEY,path TEXT,started REAL,status TEXT,error TEXT,book_id TEXT);
        ''')
        for table, additions in {
            'books': {'workgroup': 'TEXT', 'content_sha256': 'TEXT', 'split': 'TEXT'},
            'documents': {'excluded_reason': 'TEXT'},
            'units': {'text_sha256': 'TEXT', 'canonical_id': 'TEXT'},
        }.items():
            columns = {row[1] for row in db.execute('PRAGMA table_info(' + table + ')')}
            for column, kind in additions.items():
                if column not in columns:
                    db.execute('ALTER TABLE ' + table + ' ADD COLUMN ' + column + ' ' + kind)
        db.executescript('''
        CREATE INDEX IF NOT EXISTS literature_units_book ON units(book_id,ordinal);
        CREATE INDEX IF NOT EXISTS literature_units_text ON units(text_sha256);
        CREATE INDEX IF NOT EXISTS literature_units_state ON units(state);
        CREATE INDEX IF NOT EXISTS literature_documents_book ON documents(book_id,ordinal);
        ''')
        # Enrich an existing index without replacing any IDs, text or labels.
        for row in db.execute('SELECT id,text FROM units WHERE text_sha256 IS NULL OR canonical_id IS NULL').fetchall():
            digest = text_digest(row['text'])
            canonical = db.execute('SELECT canonical_id FROM units WHERE text_sha256=? AND canonical_id IS NOT NULL ORDER BY rowid LIMIT 1', (digest,)).fetchone()
            target = canonical[0] if canonical else row['id']
            db.execute('UPDATE units SET text_sha256=?,canonical_id=? WHERE id=?', (digest, target, row['id']))
            if target != row['id']:
                db.execute("UPDATE units SET state='duplicate' WHERE id=? AND state='pending'", (row['id'],))


def scan_sources(roots: list[Path], *, max_depth: int = 3) -> tuple[list[Path], dict]:
    """Bounded metadata-only scan; never follow directory or file symlinks."""
    paths, seen, counts = [], set(), Counter()
    def visit(folder: Path, depth: int) -> None:
        try:
            entries = sorted(os.scandir(folder), key=lambda e: e.name)
        except FileNotFoundError:
            counts['disappeared_directories'] += 1
            return
        for entry in entries:
            if entry.is_symlink():
                counts['symlinks_skipped'] += 1
            elif entry.is_dir(follow_symlinks=False) and depth + 1 < max_depth:
                visit(Path(entry.path), depth + 1)
            elif entry.is_file(follow_symlinks=False):
                suffix = Path(entry.name).suffix.lower()
                if suffix == '.epub':
                    path = Path(entry.path).absolute()
                    if str(path) not in seen:
                        paths.append(path); seen.add(str(path))
                elif suffix in UNSUPPORTED:
                    counts[suffix] += 1
    for root in roots:
        if not root.is_dir():
            raise ValueError('Source root is not a directory: ' + str(root))
        visit(root, 0)
    return paths, dict(counts)


def metadata(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        def read(member: str) -> bytes:
            if archive.getinfo(member).file_size > 2_000_000:
                raise ValueError('EPUB metadata exceeds size bound')
            return archive.read(member)
        container = ET.fromstring(read('META-INF/container.xml'))
        roots = [e.attrib['full-path'] for e in container.iter() if e.tag.rsplit('}', 1)[-1] == 'rootfile']
        if len(roots) != 1:
            raise ValueError('EPUB package root is ambiguous')
        package = ET.fromstring(read(_zip_path('', roots[0])))
        titles = [''.join(e.itertext()).strip() for e in package.iter() if e.tag.rsplit('}', 1)[-1] == 'title']
        authors = [''.join(e.itertext()).strip() for e in package.iter() if e.tag.rsplit('}', 1)[-1] == 'creator']
        return next((t for t in titles if t), path.stem), '; '.join(a for a in authors if a)


def frontmatter(member: str, text: str) -> str | None:
    stem = PurePosixPath(member).stem.lower()
    if re.fullmatch(r'(?:cover|title(?:page)?|copyright|toc|contents|nav|dedication|about(?:the)?author|colophon)(?:[_-]?\d+)?', stem):
        return 'obvious_frontmatter_member'
    if len(text) < 5000 and re.search(r'(?im)^\s*(?:copyright\s*(?:©|\(c\)|\d)|all rights reserved|table of contents)\b', text):
        return 'obvious_frontmatter_text'
    return None


def windows(text: str, maximum: int = 2400):
    if type(maximum) is not int or maximum < 128 or maximum > 2400:
        raise ValueError('Window size must be 128–2400 characters')
    markers = list(MARKER.finditer(text))
    cuts = {0: 'document_start'}
    for marker in markers:
        cuts[marker.start()] = 'explicit_scene_marker'
    positions = sorted(cuts) + [len(text)]
    for index, left in enumerate(positions[:-1]):
        right = positions[index + 1]
        start = left
        while start < right:
            stop = min(start + maximum, right)
            if stop < right:
                minimum = start + maximum // 3
                paragraph = text.rfind('\n', minimum, stop)
                if paragraph >= minimum:
                    stop = paragraph + 1
                else:
                    endings = list(re.finditer(r'[.!?][”"\']?\s+', text[minimum:stop]))
                    if endings:
                        stop = minimum + endings[-1].end()
                    else:
                        space = text.rfind(' ', minimum, stop)
                        if space >= minimum:
                            stop = space + 1
            if text[start:stop].strip():
                yield start, stop, text[start:stop], cuts.get(start, 'continuation_window')
            start = stop


def extract(path: Path, *, stable_seconds: float = 30) -> dict:
    started = time.time()
    try:
        before = path.stat()
        if started - max(before.st_mtime, before.st_ctime) < stable_seconds:
            return {'path': str(path), 'started': started, 'status': 'unstable', 'error': 'File changed within stability interval'}
        digest = sha256(path)
        title, author = metadata(path)
        documents = list(book_documents(path))
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino) or sha256(path) != digest:
            raise ValueError('Source changed during extraction; no content committed')
        content = canonical_digest([text for _, text in documents])
        normalized = lambda s: ' '.join(re.findall(r'\w+', s.casefold()))
        workgroup = canonical_digest({'title': normalized(title), 'author': normalized(author)})
        return {'path': str(path), 'started': started, 'status': 'complete', 'id': digest,
                'sha256': digest, 'title': title, 'author': author, 'workgroup': workgroup,
                'content_sha256': content, 'documents': documents}
    except Exception as error:
        return {'path': str(path), 'started': started, 'status': 'failed', 'error': f'{type(error).__name__}: {error}'}


def store_book(database: Path, book: dict) -> dict:
    with connect(database) as db:
        if book['status'] != 'complete':
            db.execute('INSERT INTO literature_ingest_attempts(path,started,status,error) VALUES(?,?,?,?)',
                       (book['path'], book['started'], book['status'], book.get('error')))
            return {'status': book['status'], 'path': book['path'], 'error': book.get('error')}
        identity = book['id']
        existing = db.execute('SELECT * FROM books WHERE id=?', (identity,)).fetchone()
        # Content-identical re-packaged editions retain one whole-work split.
        equivalent = db.execute('SELECT workgroup,split FROM books WHERE content_sha256=? ORDER BY id LIMIT 1', (book['content_sha256'],)).fetchone()
        workgroup = (existing['workgroup'] if existing else None) or (equivalent['workgroup'] if equivalent else None) or book['workgroup']
        split = (existing['split'] if existing else None) or (equivalent['split'] if equivalent else None) or split_for(workgroup)
        db.execute('INSERT OR IGNORE INTO books(id,title,author,path,sha256,workgroup,content_sha256,split) VALUES(?,?,?,?,?,?,?,?)',
            (identity, book['title'], book['author'], book['path'], book['sha256'], workgroup, book['content_sha256'], split))
        # A moved, hash-verified file updates only its locator, never identity or annotations.
        db.execute('UPDATE books SET path=?,workgroup=COALESCE(workgroup,?),content_sha256=COALESCE(content_sha256,?),split=COALESCE(split,?) WHERE id=?', (book['path'], workgroup, book['content_sha256'], split, identity))
        db.execute('INSERT INTO book_sources VALUES(?,?,?) ON CONFLICT(book_id,path) DO UPDATE SET verified=excluded.verified',
                   (identity, book['path'], time.time()))
        inserted = duplicates = 0
        ordinal = 0
        for doc_ordinal, (member, text) in enumerate(book['documents']):
            doc_id = canonical_digest([identity, member])
            reason = frontmatter(member, text)
            db.execute('INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?,?,?)',
                (doc_id, identity, member, doc_ordinal, text, text_digest(text), reason))
            if reason:
                continue
            for start, end, excerpt, boundary in windows(text):
                unit_id = canonical_digest([doc_id, start, end, text_digest(excerpt)])
                digest = text_digest(excerpt)
                canonical = db.execute('SELECT canonical_id FROM units WHERE text_sha256=? ORDER BY rowid LIMIT 1', (digest,)).fetchone()
                canonical_id = canonical[0] if canonical else unit_id
                state = 'duplicate' if canonical_id != unit_id else 'pending'
                changed = db.execute('INSERT OR IGNORE INTO units(id,book_id,document_id,ordinal,start,end,text,split,boundary,state,text_sha256,canonical_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                    (unit_id, identity, doc_id, ordinal, start, end, excerpt, split, boundary, state, digest, canonical_id)).rowcount
                inserted += changed
                duplicates += changed if state == 'duplicate' else 0
                ordinal += 1
        db.execute('INSERT INTO literature_ingest_attempts(path,started,status,book_id) VALUES(?,?,?,?)',
                   (book['path'], book['started'], 'complete', identity))
        return {'status': 'complete', 'book_id': identity, 'path': book['path'], 'new_units': inserted, 'new_duplicates': duplicates}


def ingest(roots: list[Path], database: Path, *, workers: int | None = None, stable_seconds: float = 30) -> dict:
    initialize(database)
    paths, unsupported = scan_sources(roots)
    workers = max(1, (os.cpu_count() or 2) - 1) if workers is None else workers
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be positive')
    counts = Counter()
    def consume(rows):
        for row in rows:
            result = store_book(database, row)
            counts[result['status']] += 1
            counts['new_units'] += result.get('new_units', 0)
            counts['new_duplicates'] += result.get('new_duplicates', 0)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    if workers == 1:
        consume(extract(path, stable_seconds=stable_seconds) for path in paths)
    else:
        from functools import partial
        with ProcessPoolExecutor(max_workers=workers) as pool:
            consume(pool.map(partial(extract, stable_seconds=stable_seconds), paths, chunksize=1))
    return {'epubs_found': len(paths), 'counts': dict(counts), 'unsupported_not_processed': unsupported,
            'source_roots': [str(p) for p in roots], 'scene_policy': 'structural candidates only',
            'duplicate_policy': 'exact text duplicates retained with canonical_id and excluded from pending prompts'}


def browse(database: Path, book_id: str = '', q: str = '', theme: str = '', entity: str = '', nlp_entity: str = '', offset: int = 0, limit: int = 12) -> dict:
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError('Invalid page bounds')
    if any(not isinstance(v, str) or len(v) > 500 for v in (book_id, q, theme, entity, nlp_entity)):
        raise ValueError('Invalid literature filter')
    if not database.exists():
        return {'stats': {'books': 0, 'docs': 0, 'units': 0, 'annotated': 0, 'pending': 0, 'failed': 0}, 'books': [], 'facets': {'themes': [], 'entities': []}, 'units': [], 'total': 0, 'offset': offset, 'limit': limit}
    safe = "CASE WHEN json_valid(u.annotation) THEN u.annotation ELSE '{}' END"
    valid = "u.state='complete' AND json_extract(" + safe + ",'$.schema')='" + ANNOTATION_SCHEMA + "' AND json_extract(" + safe + ",'$.status')='provisional_model_labels_exact_spans'"
    with connect(database) as db:
        stats = {name: db.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0] for name, table in (('books', 'books'), ('docs', 'documents'), ('units', 'units'))}
        stats.update({key: db.execute('SELECT COUNT(*) FROM units WHERE state=?', (state,)).fetchone()[0] for key, state in (('pending', 'pending'), ('failed', 'failed'))})
        stats['annotated'] = db.execute('SELECT COUNT(*) FROM units u WHERE ' + valid).fetchone()[0]
        books = [dict(row) for row in db.execute("SELECT b.id,b.title,b.author,COUNT(u.id) AS units,SUM(CASE WHEN " + valid + " THEN 1 ELSE 0 END) AS annotated FROM books b LEFT JOIN units u ON u.book_id=b.id GROUP BY b.id ORDER BY b.title,b.id")]
        themes = [dict(row) for row in db.execute("SELECT json_extract(j.value,'$.theme') AS name,COUNT(DISTINCT u.id) AS count FROM units u,json_each(CASE WHEN " + valid + " THEN u.annotation ELSE '{}' END,'$.themes') j GROUP BY name ORDER BY count DESC,name")]
        entities = [dict(row) for row in db.execute("SELECT json_extract(j.value,'$.name') AS name,json_extract(j.value,'$.kind') AS kind,COUNT(DISTINCT u.id) AS count FROM units u,json_each(CASE WHEN " + valid + " THEN u.annotation ELSE '{}' END,'$.entities') j GROUP BY name,kind ORDER BY count DESC,name LIMIT 1000")]
        clauses, args = [], []
        if book_id:
            clauses.append('u.book_id=?'); args.append(book_id)
        if q:
            clauses.append('instr(lower(u.text),lower(?))>0'); args.append(q)
        for value, array, field in ((theme, 'themes', 'theme'), (entity, 'entities', 'name')):
            if value:
                clauses.append("EXISTS(SELECT 1 FROM json_each(CASE WHEN " + valid + " THEN u.annotation ELSE '{}' END,'$." + array + "') j WHERE json_extract(j.value,'$." + field + "')=?)")
                args.append(value)
        if nlp_entity:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='nlp_runs'").fetchone():
                clauses.append('0')
            else:
                clauses.append("EXISTS(SELECT 1 FROM nlp_spans s JOIN nlp_window_sources a ON a.run_id=s.run_id AND a.canonical_id=s.canonical_id WHERE a.unit_id=u.id AND s.kind='entity' AND lower(s.text)=lower(?) AND s.run_id=(SELECT id FROM nlp_runs ORDER BY created DESC LIMIT 1))")
                args.append(nlp_entity)
        where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
        total = db.execute('SELECT COUNT(*) FROM units u' + where, args).fetchone()[0]
        rows = db.execute('SELECT u.*,b.title AS book_title,d.member FROM units u JOIN books b ON b.id=u.book_id JOIN documents d ON d.id=u.document_id' + where + ' ORDER BY b.title,u.ordinal,u.id LIMIT ? OFFSET ?', (*args, limit, offset)).fetchall()
        units = []
        for row in rows:
            item = dict(row)
            try:
                item['annotation'] = json.loads(item['annotation']) if item['annotation'] else None
            except (ValueError, TypeError):
                item['annotation'] = None
            units.append(item)
    return {'stats': stats, 'books': books, 'facets': {'themes': themes, 'entities': entities}, 'units': units, 'total': total, 'offset': offset, 'limit': limit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('ingest',))
    parser.add_argument('--source-dir', type=Path, action='append', required=True)
    parser.add_argument('--database', type=Path, default=ROOT / 'results/persona_literature.sqlite')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    print(json.dumps(ingest(args.source_dir, args.database, workers=args.workers), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
