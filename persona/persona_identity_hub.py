"""Publishes the "who are you" neuron sweep to the Hub in immutable JSONL shards.

Every few minutes it copies the exported answers from the sweep host. When 2,000 new
answers have finished (or an hour has passed with any), it writes the next
shard identity_sweep/answers-NNNNN.jsonl, updates identity_sweep/manifest.json
and the sweep README in one commit, then verifies the shard hash on the Hub.
A failed commit is checked against the Hub and retried with the same bytes;
published shards are never rewritten. Raw activation arrays are not uploaded;
each row carries the SHA-256 of its retained capture.
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
    """Adds the identity_sweep config to the dataset card if the live card lacks it."""
    text = Path(hf_hub_download(REPO, 'README.md', repo_type='dataset', force_download=True)).read_text()
    if 'config_name: identity_sweep' in text: return None
    head, sep, body = text.partition('\n---\n')
    if not head.startswith('---') or not sep: return None
    path = OUT / 'identity_hub_card.md'; path.write_text(head + '\n' + CONFIG.rstrip('\n') + sep + body)
    return CommitOperationAdd('README.md', str(path))


def publish_pending(api, state, log):
    pending = state.get('pending')
    if not pending: return
    shard = OUT / pending['local']
    if not on_hub(api, pending['path'], shard):
        manifest = OUT / 'identity_hub_manifest.json'; readme = OUT / 'identity_hub_README.md'
        manifest.write_text(json.dumps({**state['manifest'], 'shards': state['manifest']['shards'] + [pending['entry']]}, indent=1))
        readme.write_text(README)
        operations = [CommitOperationAdd(pending['path'], str(shard)), CommitOperationAdd('identity_sweep/manifest.json', str(manifest)),
                      CommitOperationAdd('identity_sweep/README.md', str(readme))]
        card = card_operation()
        if card: operations.append(card)
        commit = api.create_commit(REPO, operations, repo_type='dataset',
                                   commit_message=f"Identity sweep shard {pending['entry']['index']:05d}: {pending['entry']['rows']} answers")
        if not on_hub(api, pending['path'], shard): raise RuntimeError('Shard hash not visible after commit')
        pending['entry']['commit'] = commit.oid
    state['manifest']['shards'].append(pending['entry'])
    state['manifest'].update(rows=sum(s['rows'] for s in state['manifest']['shards']), updated=iso(time.time()))
    state['watermark'] = pending['watermark']; state['last_publish'] = time.time(); state['pending'] = None
    log('published', shard=pending['path'], rows=pending['entry']['rows'])


def tick(api, state, log):
    publish_pending(api, state, log)  # finish an interrupted commit before cutting a new shard
    local = OUT / 'identity_hub_answers.jsonl'
    d = deployment(); source = f"{d['host']}:{d['remote_root']}/results/identity_sweep/answers.jsonl"
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', source, str(local)], timeout=300, check=True)
    rows = [json.loads(line) for line in local.open()]
    new = sorted((r for r in rows if r['finished'] > state['watermark']), key=lambda r: (r['finished'], r['layer'], r['neuron'], r['dose']))
    due = len(new) >= MIN_ROWS or (new and time.time() - state.get('last_publish', 0) >= MAX_WAIT)
    if not due: return
    index = len(state['manifest']['shards']); name = f'identity_hub_answers-{index:05d}.jsonl'
    with (OUT / name).open('w') as stream:  # JSONL: the repo's other config is JSON, and datasets picks one builder per repo
        for row in shard_table(new, lens()).to_pylist(): stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    digest = sha(OUT / name)
    state['pending'] = {'local': name, 'path': f'identity_sweep/answers-{index:05d}.jsonl', 'sha256': digest,
                        'watermark': new[-1]['finished'],
                        'entry': {'index': index, 'file': f'answers-{index:05d}.jsonl', 'rows': len(new), 'sha256': digest,
                                  'finished_from': iso(new[0]['finished']), 'finished_to': iso(new[-1]['finished'])}}
    STATE.write_text(json.dumps(state, indent=1))
    publish_pending(api, state, log)


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
