import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'persona'))
import persona_literature as literature


def epub(path, *, title='Book', author='Author', text=None, extra=''):
    text = text or '<p>Mara waited near the river. She was afraid.</p><p>***</p><p>Then Mara left.</p>'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('META-INF/container.xml', '<container><rootfiles><rootfile full-path="OEBPS/book.opf"/></rootfiles></container>')
        archive.writestr('OEBPS/book.opf', f'<package><metadata><title>{title}</title><creator>{author}</creator></metadata><manifest><item id="c" href="copyright.xhtml" media-type="application/xhtml+xml"/><item id="a" href="chapter.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="c"/><itemref idref="a"/></spine></package>')
        archive.writestr('OEBPS/copyright.xhtml', '<p>All rights reserved.</p>')
        archive.writestr('OEBPS/chapter.xhtml', '<html><body>' + text + '</body></html>')
        archive.writestr('extra.txt', extra)
    return path


def test_offsets_scene_candidates_frontmatter_and_split(tmp_path):
    path = epub(tmp_path / 'book.epub', text='<p>' + ('Sentence ends. ' * 450) + '</p><p>***</p><p>Next scene?</p>')
    database = tmp_path / 'index.sqlite'
    result = literature.ingest([tmp_path], database, workers=1, stable_seconds=0)
    assert result['counts']['complete'] == 1
    page = literature.browse(database)
    assert page['stats']['docs'] == 2 and page['stats']['units'] >= 4
    assert all(len(u['text']) <= 2400 for u in page['units'])
    assert 'explicit_scene_marker' in {u['boundary'] for u in page['units']}
    assert len({u['split'] for u in page['units']}) == 1
    with literature.connect(database) as db:
        for unit in db.execute('SELECT u.*,d.text AS doc FROM units u JOIN documents d ON d.id=u.document_id'):
            assert unit['text'] == unit['doc'][unit['start']:unit['end']]
        assert db.execute("SELECT excluded_reason FROM documents WHERE member LIKE '%copyright%'").fetchone()[0]
    assert literature.extract(path, stable_seconds=30)['status'] == 'unstable'


def test_duplicates_append_resume_and_moved_path(tmp_path):
    source = tmp_path / 'source'
    moved = tmp_path / 'moved'
    source.mkdir(); moved.mkdir()
    first = epub(source / 'one.epub')
    database = tmp_path / 'index.sqlite'
    literature.ingest([source], database, workers=1, stable_seconds=0)
    before = literature.browse(database)
    unit_id = before['units'][0]['id']
    with literature.connect(database) as db:
        db.execute("UPDATE units SET state='complete',annotation='{}' WHERE id=?", (unit_id,))
    second = epub(source / 'two.epub', title='A differently packaged title', extra='different ZIP bytes')
    literature.ingest([source], database, workers=1, stable_seconds=0)
    with literature.connect(database) as db:
        assert db.execute('SELECT COUNT(*) FROM books').fetchone()[0] == 2
        assert len({r[0] for r in db.execute('SELECT split FROM books')}) == 1
        assert db.execute("SELECT COUNT(*) FROM units WHERE state='duplicate'").fetchone()[0] > 0
        assert tuple(db.execute('SELECT state,annotation FROM units WHERE id=?', (unit_id,)).fetchone()) == ('complete', '{}')
        old_count = db.execute('SELECT COUNT(*) FROM units').fetchone()[0]
    destination = moved / first.name
    first.rename(destination)
    literature.ingest([source, moved], database, workers=1, stable_seconds=0)
    with literature.connect(database) as db:
        assert db.execute('SELECT COUNT(*) FROM units').fetchone()[0] == old_count
        assert db.execute('SELECT path FROM books WHERE id=?', (literature.sha256(destination),)).fetchone()[0] == str(destination)


def test_metadata_filters_only_validated_labels(tmp_path):
    epub(tmp_path / 'book.epub')
    database = tmp_path / 'index.sqlite'
    literature.ingest([tmp_path], database, workers=1, stable_seconds=0)
    page = literature.browse(database)
    unit = page['units'][0]
    raw = {'schema': literature.ANNOTATION_SCHEMA, 'status': 'provisional_model_labels_exact_spans',
           'entities': [{'name': 'Mara', 'kind': 'character'}], 'themes': [{'theme': 'fear'}]}
    with literature.connect(database) as db:
        db.execute("UPDATE units SET state='complete',annotation=? WHERE id=?", (json.dumps(raw), unit['id']))
    page = literature.browse(database, theme='fear', entity='Mara')
    assert page['total'] == 1 and page['units'][0]['id'] == unit['id']
    assert page['facets']['themes'] == [{'name': 'fear', 'count': 1}]
    assert literature.browse(database, entity='Mar')['total'] == 0
    assert literature.browse(database, q='RIVER')['total'] == 1
    with literature.connect(database) as db:
        db.execute("UPDATE units SET annotation='not json' WHERE id=?", (unit['id'],))
    assert literature.browse(database, theme='fear')['total'] == 0


def test_bounded_scan_unsupported_and_symlinks(tmp_path):
    epub(tmp_path / 'root.epub')
    (tmp_path / 'book.pdf').write_bytes(b'not read')
    nested = tmp_path / 'one' / 'two' / 'three'
    nested.mkdir(parents=True)
    epub(nested / 'too-deep.epub')
    (tmp_path / 'linked').symlink_to(nested, target_is_directory=True)
    found, counts = literature.scan_sources([tmp_path])
    assert found == [tmp_path / 'root.epub']
    assert counts['.pdf'] == 1 and counts['symlinks_skipped'] == 1
    with pytest.raises(ValueError):
        literature.browse(tmp_path / 'absent.sqlite', limit=1000)
