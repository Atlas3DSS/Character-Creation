"""Publishes the "who are you" neuron sweep to the Hub in immutable JSONL shards.

Every few minutes it copies the exported answers from the sweep host. When 2,000 new
answers have finished (or an hour has passed with any), it writes the next
shard identity_sweep/answers-NNNNN.jsonl, updates identity_sweep/manifest.json
and the sweep README in one commit, then verifies the shard hash on the Hub.
A failed commit is checked against the Hub and retried with the same bytes;
published shards are never rewritten. Raw activation arrays are not uploaded;
each row carries the SHA-256 of its retained capture.

Combo and decline-nudge rows (2-4 neurons pushed together) go to their own stream, identity_combos/, with their own
columns, manifest and README, so the single-neuron config keeps one schema across all of its shards.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pyarrow as pa
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/neuron_atlas'
REPO = 'Atlas3D/character-creation-big-five'
STATE = OUT / 'identity_hub_state.json'
MIN_ROWS, MAX_WAIT = 2000, 3600
MODEL = {'model': 'nvidia/Qwen3.8-27B-NVFP4', 'model_revision': '482ca0f3832238542f8f5295dde86b5f22711d80'}
CONFIG = '''- config_name: identity_sweep
  data_files:
  - split: train
    path: identity_sweep/answers-*.jsonl
'''
COMBO_CONFIG = '''- config_name: identity_combos
  data_files:
  - split: train
    path: identity_combos/answers-*.jsonl
'''
PUSH = pa.struct([('layer', pa.int16()), ('neuron', pa.int32()), ('unit_dose', pa.float32()), ('fixed', pa.bool_())])
COMBO_SCHEMA = pa.schema([('kind', pa.string()), ('label', pa.string()), ('pushes', pa.list_(PUSH)), ('ramp_fraction', pa.float32()),
    ('ramp_step', pa.int8()), ('coherent_auto', pa.bool_()), ('declined_auto', pa.bool_()), ('answer', pa.string()),
    ('finish_reason', pa.string()), ('completion_tokens', pa.int32()), ('prompt', pa.string()), ('seed', pa.int32()),
    ('max_tokens', pa.int16()), ('sweep_batch', pa.string()), ('finished_at', pa.string()), ('model', pa.string()),
    ('model_revision', pa.string())])
SCHEMA = pa.schema([('layer', pa.int16()), ('neuron', pa.int32()), ('push', pa.string()), ('dose', pa.float32()),
    ('cascade_level', pa.int8()), ('coherent_auto', pa.bool_()), ('answer', pa.string()), ('finish_reason', pa.string()),
    ('completion_tokens', pa.int32()), ('lens_up', pa.string()), ('lens_down', pa.string()), ('lens_up_kind', pa.string()),
    ('lens_down_kind', pa.string()), ('prompt', pa.string()), ('seed', pa.int32()), ('max_tokens', pa.int16()),
    ('sweep_batch', pa.string()), ('finished_at', pa.string()), ('capture_sha256', pa.string()),
    ('model', pa.string()), ('model_revision', pa.string())])
README = '''# Who are you? Single-neuron sweep of Qwen3.8-27B-NVFP4

Each row is one reply of `nvidia/Qwen3.8-27B-NVFP4` (revision `482ca0f3832238542f8f5295dde86b5f22711d80`) to
the user message `who are you`, with one MLP neuron pushed up or down. Rows with `layer = -1` are the unsteered
baseline. Load it with `load_dataset("Atlas3D/character-creation-big-five", "identity_sweep")`.

## How a push works

The push adds to the input of the down projection of layer `layer`, at neuron `neuron`, on every token
(prompt and reply):

```
c = dose × median(‖residual leaving layer L‖) / ‖W_down[:, neuron]‖
```

The median comes from the project's calibrated vector bank. Decoding is greedy (temperature 0), thinking off,
at most 160 tokens, seed 0.

## The cascade

On this 27B a single push of dose 1 usually collapses the reply to nothing, so each direction starts strong and
retries weaker while the reply collapses: 1, 0.5, 0.25, 0.125, 0.0625, 0.03125. New neurons start one step
stronger than the typical coherent strength of nearby layers. **To get one answer per neuron and direction, keep
the row with the largest `|dose|` among rows with `coherent_auto = true`.** A direction with no coherent row was
incoherent at every strength tried. Rows of an unfinished cascade can arrive in a later shard.

## Columns

| Column | Meaning |
|---|---|
| `layer`, `neuron` | MLP neuron (64 layers × 17,408); `-1` for the baseline |
| `push`, `dose`, `cascade_level` | `up`/`down`/`none`, the signed dose, and its step in the cascade (0 = strongest) |
| `coherent_auto` | Automatic collapse check: not empty, at least 4 tokens, no special tokens, no repetition loop. Not a quality judgement |
| `answer`, `finish_reason`, `completion_tokens` | The reply |
| `lens_up`, `lens_down`, `*_kind` | Static weight-lens label and kind of the neuron (top vocabulary tokens of ±W_down through the unembedding) |
| `sweep_batch` | Which queued batch the neuron came from (core emotions first) |
| `capture_sha256` | SHA-256 of the retained final-prefill activation capture (not uploaded) |

`manifest.json` lists every shard with its row count and SHA-256. Shards are append-only; a later shard never
changes an earlier one.

## Limitations

- One prompt. A neuron may act differently on other prompts.
- `coherent_auto` is a cheap heuristic. No human or model has judged these replies yet.
- Neurons were chosen by their lens labels and activation evidence, starting with core human emotions; this is
  not yet the whole model.
- Inspired by the Neuron Persona Library (Qwen3-8B), Sauers' single-neuron steering and Lyra's neuron-explorer.
'''


def deployment():
    """Host, paths and URLs of this installation, from the git-ignored identity_sweep.local.json."""
    path = Path(__file__).resolve().parents[1] / 'identity_sweep.local.json'
    if not path.exists(): raise SystemExit(f'Create {path.name} (see README_agent.md, identity sweep section)')
    return json.loads(path.read_text())


def iso(ts): return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec='seconds')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


COMBO_README = '''# Who are you? Planned 2-4 neuron combos of Qwen3.8-27B-NVFP4

Each row is one reply of `nvidia/Qwen3.8-27B-NVFP4` (revision `482ca0f3832238542f8f5295dde86b5f22711d80`) to
`who are you` with 2 to 4 MLP neurons pushed at once. The unsteered baseline is in the `identity_sweep` config
(`layer = -1`). Load it with `load_dataset("Atlas3D/character-creation-big-five", "identity_combos")`.

## What is pushed

`pushes` lists each neuron with its `unit_dose`: that neuron's own strongest coherent single-push dose (signed;
see `identity_sweep`). Members with `fixed = false` are pushed at `ramp_fraction × unit_dose`; `fixed = true`
members are pushed at `unit_dose` exactly. A push adds to the input of the down projection on every token:
`c = dose × median(‖residual leaving layer L‖) / ‖W_down[:, neuron]‖`. Greedy, thinking off, 160 tokens, seed 0.

## Kinds

Neurons come from concept traces: for each concept that passed held-out label qualification, the neurons whose
final-prompt-token activation best separates scenes where the concept holds from hard negatives, chosen on
discovery scenes and replicated on held-out scenes.

| `kind` | Meaning |
|---|---|
| `coherent` | One concept, its neurons spread from early to late layers, each pushed the way the concept moves it |
| `harmonized` | Related concepts, chosen by how much their traces overlap |
| `discordant` | Opposite concepts pushed together (for example calm and rage) |
| `control` | Random neurons in the same layers, at the median coherent dose of those layers |
| `decline-amp`, `decline-mute`, `decline-mix`, `decline-control` | Decline nudges: a single push that made the model decline to answer stays fixed; 1-3 neurons are added that raise an open signal (trust, curiosity, calm, playfulness, gratitude), lower a guarded one (suspicion, contempt), both, or random neurons |

## The ramp

Combos start weak and get stronger: `ramp_fraction` 0.125, then 0.25, then 0.5. A combo stops at its first
incoherent reply. Decline nudges also stop at the first reply that no longer declines: that step is the finding.

## Columns

| Column | Meaning |
|---|---|
| `kind`, `label` | Combo kind (above) and its concepts or decline site |
| `pushes` | The neurons pushed (`layer`, `neuron`, `unit_dose`, `fixed`) |
| `ramp_fraction`, `ramp_step` | Ramp scale for non-fixed members, and its step (0 = weakest) |
| `coherent_auto` | Automatic collapse check: not empty, at least 4 tokens, no special tokens, no loop. Not a quality judgement |
| `declined_auto` | Automatic flag: the reply refuses to answer, help or tell, or opens with "No" |
| `answer`, `finish_reason`, `completion_tokens` | The reply |

## Limitations

- One prompt; combos may act differently elsewhere. Every push is applied at inference time; no weights change.
- `coherent_auto` and `declined_auto` are cheap heuristics, not judgements.
- Concept scenes are model-authored and model-checked, not human-annotated.
'''


def lens():
    detail = np.load(OUT / 'detail.npz', allow_pickle=False); names = json.loads((OUT / 'names.json').read_text())
    kinds = json.loads((OUT / 'summary.json').read_text())['modes']['up']['codes']
    return detail['label_up'], detail['label_down'], detail['codes'], names, kinds


def shard_table(rows, labels):
    up, down, codes, names, kinds = labels; cols = {f.name: [] for f in SCHEMA}
    for r in rows:
        key = r['layer'] * 17408 + r['neuron'] if r['layer'] >= 0 else None
        values = {'layer': r['layer'], 'neuron': r['neuron'], 'push': 'none' if r['layer'] < 0 else 'up' if r['dose'] > 0 else 'down',
                  'dose': r['dose'], 'cascade_level': r.get('level'), 'coherent_auto': bool(r.get('coherent')) if r['layer'] >= 0 else True,
                  'answer': r['answer'], 'finish_reason': r.get('finish'), 'completion_tokens': r.get('tokens'),
                  'lens_up': names[up[key]] if key is not None else None, 'lens_down': names[down[key]] if key is not None else None,
                  'lens_up_kind': kinds[codes[0, key]] if key is not None else None, 'lens_down_kind': kinds[codes[1, key]] if key is not None else None,
                  'prompt': 'who are you', 'seed': 0, 'max_tokens': 160, 'sweep_batch': r.get('sweep'),
                  'finished_at': iso(r['finished']), 'capture_sha256': r.get('artifact_sha256'), **MODEL}
        for k in cols: cols[k].append(values[k])
    return pa.table(cols, schema=SCHEMA)


def combo_table(rows):
    cols = {f.name: [] for f in COMBO_SCHEMA}
    for r in rows:
        kind, _, label = str(r.get('kind') or '').partition(':')
        values = {'kind': kind, 'label': label, 'ramp_fraction': r['dose'], 'ramp_step': r.get('level'),
                  'pushes': [{'layer': m[0], 'neuron': m[1], 'unit_dose': m[2], 'fixed': bool(m[3:] and m[3])} for m in r['pushes']],
                  'coherent_auto': bool(r.get('coherent')), 'declined_auto': bool(r.get('declined')), 'answer': r['answer'],
                  'finish_reason': r.get('finish'), 'completion_tokens': r.get('tokens'), 'prompt': 'who are you', 'seed': 0,
                  'max_tokens': 160, 'sweep_batch': r.get('sweep'), 'finished_at': iso(r['finished']), **MODEL}
        for k in cols: cols[k].append(values[k])
    return pa.table(cols, schema=COMBO_SCHEMA)


STREAMS = {  # name: (rows of this stream, table builder, README, manifest schema)
    'identity_sweep': (lambda r: r['layer'] >= -1, lambda rows: shard_table(rows, lens()), lambda: README,
                       'character-creation.identity-sweep-manifest.v1'),
    'identity_combos': (lambda r: r['layer'] == -2, combo_table, lambda: COMBO_README, 'character-creation.identity-combos-manifest.v1')}


def on_hub(api, path, local):
    """True when the Hub holds exactly these bytes (LFS SHA-256, or the git blob id for small files)."""
    data = Path(local).read_bytes()
    try:
        info = api.get_paths_info(REPO, [path], repo_type='dataset', expand=True)
    except Exception: return False
    if not info: return False
    if getattr(info[0], 'lfs', None) is not None: return info[0].lfs.sha256 == hashlib.sha256(data).hexdigest()
    return info[0].blob_id == hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def card_operation():
    """Adds the identity_sweep / identity_combos configs to the dataset card if the live card lacks them."""
    text = Path(hf_hub_download(REPO, 'README.md', repo_type='dataset', force_download=True)).read_text()
    missing = [block for name, block in (('identity_sweep', CONFIG), ('identity_combos', COMBO_CONFIG)) if f'config_name: {name}\n' not in text]
    if not missing: return None
    head, sep, body = text.partition('\n---\n')
    if not head.startswith('---') or not sep: return None
    path = OUT / 'identity_hub_card.md'; path.write_text(head + '\n' + ''.join(b.rstrip('\n') + '\n' for b in missing).rstrip('\n') + sep + body)
    return CommitOperationAdd('README.md', str(path))


def publish_pending(api, state, log, stream='identity_sweep'):
    pending = state.get('pending')
    if not pending: return
    shard = OUT / pending['local']
    if not on_hub(api, pending['path'], shard):
        tag = '' if stream == 'identity_sweep' else '_combos'
        manifest = OUT / f'identity_hub{tag}_manifest.json'; readme = OUT / f'identity_hub{tag}_README.md'
        manifest.write_text(json.dumps({**state['manifest'], 'shards': state['manifest']['shards'] + [pending['entry']]}, indent=1))
        readme.write_text(STREAMS[stream][2]())
        operations = [CommitOperationAdd(pending['path'], str(shard)), CommitOperationAdd(f'{stream}/manifest.json', str(manifest)),
                      CommitOperationAdd(f'{stream}/README.md', str(readme))]
        card = card_operation()
        if card: operations.append(card)
        commit = api.create_commit(REPO, operations, repo_type='dataset',
                                   commit_message=f"{stream} shard {pending['entry']['index']:05d}: {pending['entry']['rows']} answers")
        if not on_hub(api, pending['path'], shard): raise RuntimeError('Shard hash not visible after commit')
        pending['entry']['commit'] = commit.oid
    state['manifest']['shards'].append(pending['entry'])
    state['manifest'].update(rows=sum(s['rows'] for s in state['manifest']['shards']), updated=iso(time.time()))
    state['watermark'] = pending['watermark']; state['last_publish'] = time.time(); state['pending'] = None
    log('published', shard=pending['path'], rows=pending['entry']['rows'])


def tick(api, state, log):
    combos = state.setdefault('combos', {'watermark': 0, 'pending': None, 'manifest': {
        'schema': STREAMS['identity_combos'][3], 'repo': REPO, **MODEL, 'prompt': 'who are you', 'shards': [], 'rows': 0}})
    publish_pending(api, state, log)  # finish an interrupted commit before cutting a new shard
    publish_pending(api, combos, log, 'identity_combos')
    local = OUT / 'identity_hub_answers.jsonl'
    d = deployment(); source = f"{d['host']}:{d['remote_root']}/results/identity_sweep/answers.jsonl"
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', source, str(local)], timeout=300, check=True)
    rows = [json.loads(line) for line in local.open()]
    for stream, sub in (('identity_sweep', state), ('identity_combos', combos)):
        cut(api, sub, log, stream, [r for r in rows if STREAMS[stream][0](r)], lambda: STATE.write_text(json.dumps(state, indent=1)))


def cut(api, state, log, stream, rows, save):
    new = sorted((r for r in rows if r['finished'] > state['watermark']), key=lambda r: (r['finished'], r['layer'], r['neuron'], r['dose']))
    due = len(new) >= MIN_ROWS or (new and time.time() - state.get('last_publish', 0) >= MAX_WAIT)
    if not due: return
    index = len(state['manifest']['shards']); tag = '' if stream == 'identity_sweep' else '_combos'
    name = f'identity_hub{tag}_answers-{index:05d}.jsonl'
    with (OUT / name).open('w') as out:  # JSONL: the repo's other config is JSON, and datasets picks one builder per repo
        for row in STREAMS[stream][1](new).to_pylist(): out.write(json.dumps(row, ensure_ascii=False) + '\n')
    digest = sha(OUT / name)
    state['pending'] = {'local': name, 'path': f'{stream}/answers-{index:05d}.jsonl', 'sha256': digest,
                        'watermark': new[-1]['finished'],
                        'entry': {'index': index, 'file': f'answers-{index:05d}.jsonl', 'rows': len(new), 'sha256': digest,
                                  'finished_from': iso(new[0]['finished']), 'finished_to': iso(new[-1]['finished'])}}
    save()  # the pending shard is on disk before the commit, so an interrupted commit retries the same bytes
    publish_pending(api, state, log, stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--every', type=int, default=300)
    args = parser.parse_args()
    api = HfApi()
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        'watermark': 0, 'pending': None, 'events': [],
        'manifest': {'schema': 'character-creation.identity-sweep-manifest.v1', 'repo': REPO, **MODEL,
                     'prompt': 'who are you', 'shards': [], 'rows': 0}}
    def log(kind, **fields): state['events'] = (state['events'] + [{'time': iso(time.time()), 'event': kind, **fields}])[-30:]
    while True:
        try:
            tick(api, state, log); state['last_error'] = None
        except Exception as exc:
            state['last_error'] = f'{type(exc).__name__}: {str(exc)[:300]}'
        STATE.write_text(json.dumps(state, indent=1))
        time.sleep(args.every)


if __name__ == '__main__': main()
