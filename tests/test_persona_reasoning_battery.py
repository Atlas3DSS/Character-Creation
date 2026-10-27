"""Portable synthetic tests: no local source snapshots or network required."""
import sys
from pathlib import Path
for folder in (Path(__file__).resolve().parents[1]/"persona",Path(__file__).resolve().parents[1],Path(__file__).resolve().parent):
    if (folder/"persona_reasoning_battery.py").exists():sys.path.insert(0,str(folder));break
from copy import deepcopy
from pathlib import Path
import json
import pytest
from persona_reasoning_battery import normalize_bytes,number,check_answer,algebra_cases,load,validate,digest,request,wrong_answer_challenge,public_projection
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def source_windows():
    from persona_reasoning_battery import ENGLISH_TRANSLATIONS
    algebra=[{'question':f"b'Synthetic fixture x = {i}.\\n'",'answer':f"b'{i}\\n'",'template_name':'fixture'} for i in range(40)]
    general=[{'id':str(i),'text':f'Synthetic planning fixture {i}','tokens':5} for i in range(1000)]
    for item in ENGLISH_TRANSLATIONS:general[item['row_index']]['id']=item['source_id']
    return algebra,general


@pytest.fixture
def battery(source_windows):
    from persona_reasoning_battery import assemble
    return assemble(*source_windows)[0]


def test_bytes_literal_double_escaped_newlines_normalize_without_execution():
    assert normalize_bytes("b'Solve x = 5.\\\\n'")=='Solve x = 5.'
    assert normalize_bytes("b'-4\\n'")=='-4'
    with pytest.raises(ValueError):normalize_bytes("b'x' + __import__('os').system('never')")
    with pytest.raises(ValueError):normalize_bytes("b'x' * 999999999")


@pytest.mark.parametrize('text',['__import__("os")','2**1000000','1/0','nan','inf','1e1000','[4]','1;2'])
def test_numeric_parser_rejects_expressions(text):
    with pytest.raises(ValueError):number(text)


def test_exact_numeric_check_does_not_guess_from_reasoning():
    assert check_answer('Work gives 0.5.\nFINAL_ANSWER: 1/2','0.5')['state']=='correct'
    assert check_answer('{"answer":"-4"}','-4')['state']=='correct'
    assert check_answer('FINAL_ANSWER: 3','4')['state']=='incorrect'
    assert check_answer('The values are 3, 4 and 5.','4')['state']=='unscorable'
    assert check_answer('FINAL_ANSWER: 3\nFINAL_ANSWER: 4','4')['state']=='unscorable'
    assert check_answer('{"answer":4,"explanation":"text"}','4')['state']=='unscorable'


def test_template_dedup_and_conflicting_answers():
    rows=[{'question':f"b'Solve x = {i}.\\n'",'answer':f"b'{i}\\n'",'template_name':'one'} for i in range(40)]
    rows.append(dict(rows[0],template_name='two'))
    cases=algebra_cases(rows);assert len(cases)==32
    assert len({c['question_id'] for c in cases})==32
    assert sum(c['split']=='heldout' for c in cases)==8
    rows[-1]['answer']="b'99\\n'"
    with pytest.raises(ValueError):algebra_cases(rows)


def test_frozen_48_cases_source_bound_requests_and_no_general_answer_key(battery):
    value=battery
    assert len(value['cases'])==48
    algebra=next(c for c in value['cases'] if c['kind']=='algebra')
    general=next(c for c in value['cases'] if c['kind']=='general')
    assert request(general)['messages']==[{'role':'user','content':general['prompt']}]
    assert request(algebra)['messages'][0]['content'].startswith(algebra['prompt'])
    challenge=wrong_answer_challenge(algebra)
    assert challenge['synthetic_assistant_turn'] is True and challenge['split']==algebra['split']
    assert challenge['messages'][1]['content']!=f"The answer is {algebra['expected_answer']}."
    with pytest.raises(ValueError):wrong_answer_challenge(general)
    projected=json.dumps(public_projection(value))
    assert algebra['prompt'] not in projected and 'expected_answer' not in projected and 'source_row"' not in projected


@pytest.mark.parametrize('mutation',['answer','split','source','general_key','translation'])
def test_rehashed_corruption_rejected(mutation,battery):
    value=deepcopy(battery)
    case=value['cases'][0];general=value['cases'][-1]
    if mutation=='answer':case['expected_answer']='999'
    elif mutation=='split':case['split']='heldout'
    elif mutation=='source':case['source_row_sha256']='0'*64
    elif mutation=='general_key':general['expected_answer']='42'
    else:general['prompt']='Unbound translation'
    value['sha256']=digest({k:v for k,v in value.items() if k!='sha256'})
    with pytest.raises(ValueError):validate(value)


def test_general_assessment_never_invents_correctness():
    from persona_reasoning_battery import assess
    assert assess({'kind':'general'},'Any unsupported claim')['state']=='unscored'


def test_duplicate_answer_json_is_unscorable():
    assert check_answer('{"answer":"0","answer":"1"}','1')['state']=='unscorable'


def test_deterministic_build_without_network(tmp_path,monkeypatch,source_windows):
    import persona_reasoning_battery as module
    expected=deepcopy(module.BUILD_METADATA)
    rows=dict(zip(('algebra','general'),source_windows))
    for kind in rows:expected['acquisition'][kind]['rows_sha256']=digest(rows[kind])
    monkeypatch.setattr(module,'BUILD_METADATA',expected)
    calls=[]
    def read(kind):calls.append(kind);return rows[kind],deepcopy(expected['acquisition'][kind])
    monkeypatch.setattr(module,'read_source',read)
    path=tmp_path/'persona_reasoning_battery.json'
    module.build(path)
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    module.build(path)
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    assert calls==['general','algebra','general','algebra']  # SOURCES insertion order
    assert module.load(path)['sha256']==json.loads(path.read_text())['sha256']
    index=json.loads((tmp_path/'persona_reasoning_index.json').read_text())
    assert len(index['cases'])==48 and 'Synthetic fixture x' not in json.dumps(index)


def test_different_existing_output_not_overwritten_or_partially_created(tmp_path,source_windows):
    from persona_reasoning_battery import assemble,write_outputs
    value,snapshot=assemble(*source_windows)
    path=tmp_path/'persona_reasoning_battery.json';path.write_text('keep exact evidence')
    with pytest.raises(FileExistsError):write_outputs(path,value,snapshot)
    assert path.read_text()=='keep exact evidence'
    assert len(list(tmp_path.iterdir()))==1


def test_changed_upstream_window_refuses_build(tmp_path,monkeypatch,source_windows):
    import persona_reasoning_battery as module
    monkeypatch.setattr(module,'read_source',lambda kind:(source_windows[1],module.BUILD_METADATA['acquisition'][kind]))
    with pytest.raises(ValueError):module.build(tmp_path/'persona_reasoning_battery.json')
    assert not list(tmp_path.iterdir())


def test_build_checks_pinned_lfs_metadata_as_well_as_rows(tmp_path,monkeypatch,source_windows):
    import persona_reasoning_battery as module
    expected=deepcopy(module.BUILD_METADATA);rows=dict(zip(('algebra','general'),source_windows))
    for kind in rows:expected['acquisition'][kind]['rows_sha256']=digest(rows[kind])
    monkeypatch.setattr(module,'BUILD_METADATA',expected)
    def read(kind):
        meta=deepcopy(expected['acquisition'][kind]);meta['upstream_file_sha256']='0'*64
        return rows[kind],meta
    monkeypatch.setattr(module,'read_source',read)
    with pytest.raises(ValueError):module.build(tmp_path/'persona_reasoning_battery.json')
    assert not list(tmp_path.iterdir())
