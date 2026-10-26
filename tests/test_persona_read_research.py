import hashlib
import json
import pytest
from persona.persona_read_research import Snapshot, checked_name, verify


def test_reader_rejects_changed_bytes_and_unsafe_names(tmp_path):
    p = tmp_path / 'batch.jsonl'; p.write_text('evidence')
    verify(p, hashlib.sha256(b'evidence').hexdigest())
    p.write_text('changed')
    with pytest.raises(ValueError): verify(p, hashlib.sha256(b'evidence').hexdigest())
    for name in ('../secret', '/tmp/file', 'a/b', 'https://host/file', 'a\\b'):
        with pytest.raises(ValueError): checked_name(name)


def test_record_reader_checks_exact_ids_and_preserves_failed_outcomes(tmp_path):
    p = tmp_path / 'batch.jsonl'
    p.write_text(json.dumps({'id':'a','kind':'coherence','result':'heldout_failed'})+'\n')
    snapshot = Snapshot.__new__(Snapshot)
    snapshot.manifest = {'batches':[{'file':'batch.jsonl','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                                   'records':1,'record_ids':['a'],'kinds':['coherence']}]}
    snapshot.download = lambda filename, checksum: (verify(p, checksum) or p)
    assert list(snapshot.records())[0]['result'] == 'heldout_failed'
    snapshot.manifest['batches'][0]['record_ids'] = ['b']
    with pytest.raises(ValueError): list(snapshot.records())
