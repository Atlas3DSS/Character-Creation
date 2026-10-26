import importlib.util
import json
from pathlib import Path
import sqlite3

import numpy as np

path = Path(__file__).resolve().parents[1] / 'activation-search/persona_overnight_features.py'
spec = importlib.util.spec_from_file_location('features', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_lexical_control_suppresses_apparent_semantic_association():
    means = np.ones((4, 3, 1, 3), dtype=np.float32)
    means[:, 0, 0, :] = [3, 3, 3]
    means[:, 2, 0, 1] = 3  # Same apparent signal from merely mentioning word.
    means[2:, 0, 0, 2] = .2  # Inconsistent across frame families.
    arrays, _, _, _ = m.contrasts(means, means)
    assert arrays['ranking_score'][0, 0] > .4
    assert arrays['ranking_score'][0, 1] == 0
    assert arrays['ranking_score'][0, 2] == 0


def test_heldout_and_incomplete_rows_never_fit(tmp_path):
    database = tmp_path / 'captures.sqlite'
    with sqlite3.connect(database) as db:
        db.executescript('CREATE TABLE metadata(key,value); CREATE TABLE tasks(id,state,row,request,response);')
        for split, state, feature in [('heldout', 'complete', 'secret'), ('discovery', 'uncertain', 'joy')]:
            row = {'tier': 'authored_emotion_contrast', 'split': split, 'provenance': {'feature': feature}}
            db.execute('INSERT INTO tasks VALUES(?,?,?,?,?)', (feature, state, json.dumps(row), None, None))
    groups, _ = m.collect(database)
    assert groups == {'joy': []}
    result = m.analyze_feature(('joy', [], str(tmp_path), {}))
    assert result['state'] == 'missing_evidence'
    assert len(result['missing']) == 12
