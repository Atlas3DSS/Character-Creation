"""Export only generated Big Five scenarios and publish a verified Hub snapshot."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent.parent
REPO='Atlas3D/character-creation-big-five'

def export(root=ROOT):
    root=Path(root)
    # Preserve the research publisher's managed card section on later prompt exports.
    card_path=root/'dataset/README.md'
    research=''
    if card_path.is_file():
        existing=card_path.read_text()
        start='<!-- character-creation-research:start -->';end='<!-- character-creation-research:end -->'
        if start in existing:
            if existing.count(start)!=1 or existing.count(end)!=1:
                raise ValueError('Ambiguous managed research card section')
            research='\n'+existing[existing.index(start):existing.index(end)+len(end)]+'\n'
    with (root/'data/persona_manager_corpus.jsonl').open('rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH)
        raw=(root/'data/persona_generated_big_five.jsonl').read_bytes()
        manifest=json.loads((root/'data/persona_generated_big_five_manifest.json').read_text())
    if manifest.get('status')!='complete' or hashlib.sha256(raw).hexdigest()!=manifest.get('output_sha256'):
        raise ValueError('Generated source snapshot hash mismatch')
    approval_path = root/'data/persona_public_release.local.json'
    if not approval_path.is_file():
        raise ValueError('Explicit public snapshot approval required')
    approval = json.loads(approval_path.read_text())
    if approval.get('classification') != 'public' or approval.get('source_sha256') != manifest['output_sha256']:
        raise ValueError('Snapshot is not approved for public release')
    rows=[];seen=set()
    for line in raw.splitlines():
        if not line.strip():continue
        r=json.loads(line);labels=r['labels'];g=r['generator']
        if (r.get('corpus')!='big_five' or r.get('category')!='synthetic_scenario' or r.get('synthetic') is not True
            or r.get('split')!='discovery' or labels.get('validated_inventory') is not False
            or r.get('source',{}).get('kind')!='synthetic_model_response' or g['model_identity']!=manifest['model_identity']):
            raise ValueError('Only explicitly synthetic discovery scenarios may be exported')
        if r['id'] in seen:raise ValueError('Duplicate record ID')
        seen.add(r['id'])
        rows.append({'id':r['id'],'prompt':r['prompt'],'trait_design':labels['prompt_design_trait'],
          'facet_design':labels['prompt_design_facet'],'domain':labels['scenario_domain'],'split':'discovery',
          'synthetic':True,'validated_inventory':False,'job_id':g['job_id'],'request_id':g['request_id'],
          'model_identity':g['model_identity'],'runtime_identity':g['runtime_identity'],
          'prepared_identity':g['prepared_identity'],'request_sha256':g['request_sha256'],
          'response_sha256':g['response_sha256'],'normalized_prompt_sha256':r['source']['normalized_prompt_sha256']})
    if len(rows)!=manifest['records'] or not rows:raise ValueError('Source record count mismatch or empty corpus')
    out=root/'dataset';out.mkdir(exist_ok=True)
    body=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows)
    (out/'scenarios.jsonl').write_text(body)
    counts=dict(sorted(Counter(r['trait_design'] for r in rows).items()))
    public={'schema':'character-creation.public-big-five.v1','records':len(rows),'source_sha256':manifest['output_sha256'],
       'export_sha256':hashlib.sha256(body.encode()).hexdigest(),'traits':counts,'model_identity':manifest['model_identity']}
    (out/'manifest.json').write_text(json.dumps(public,indent=2)+'\n')
    card=f'''---
language:
- en
license: cc-by-4.0
task_categories:
- text-generation
tags:
- synthetic
- big-five
- personality
- interpretability
- steering
configs:
- config_name: default
  data_files:
  - split: train
    path: scenarios.jsonl
---
# Character Creation — synthetic Big Five scenarios

{len(rows):,} original model-generated prompts for exploring LLM behavior. Published by
[Atlas3D / Character Creation](https://github.com/Atlas3DSS/Character-Creation).
This is a growing discovery dataset; the Hub commit pins each snapshot.

## What the labels mean

`trait_design` and `facet_design` describe the generation assignment, **not a measured
personality score, answer key, validated inventory, or human annotation**. The six
facets per trait are our operational prompt taxonomy. All records are discovery
material, exposed as `train` for dataset-loader compatibility. No held-out evaluation
split is claimed; do not use these records both to tune and to claim generalization.

Trait counts: `{json.dumps(counts,sort_keys=True)}`.

## Generation and filtering

A local 27B planner assigns trait, facet and domain; six Qwen3.8-27B-NVFP4 workers
produce batches of twelve scenario prompts. Generator checkpoint:
`nvidia/Qwen3.8-27B-NVFP4`, revision
`482ca0f3832238542f8f5295dde86b5f22711d80` (architecture `qwen3_5`).
Temperature 0.9, top-p 0.95, top-k 40, maximum 2048 output tokens per batch,
thinking disabled. Seeds vary by batch and worker. Request and response hashes,
job IDs, and exact model/runtime identities are retained in each row; the operator
retains full request audits. Validation checks structure and prompt constraints;
normalized-text deduplication rejects exact normalized duplicates. Semantic
near-duplicates, bias, uneven trait coverage, and weak scenarios can remain.
There has been no independent psychometric or human-quality validation.

## Fields and reproducibility

`prompt` is the text to present to an LLM; `trait_design`, `facet_design`, and `domain`
are design metadata. `id` and hashes support stable joins and provenance. `synthetic`
is always true and `validated_inventory` always false. `manifest.json` records
source/export SHA-256 and row counts. No local filesystem paths, credentials, book
text, SCOTUS documents, or questionnaire item collections are included in this export.

```python
from datasets import load_dataset
scenarios = load_dataset("{REPO}", split="train")
```

## Attribution and use

The neuron-steering research workflow is inspired by
[Lyra / neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), with
runtime inspiration from [RhizoNymph/vLLM](https://github.com/RhizoNymph/vllm) and
[heterodoxin/Strata](https://github.com/heterodoxin/AbliteratedOutTheBox/tree/strata).
These prompts were generated by the Atlas workflow; Lyra did not author or validate
them. Credit Atlas3D / Character Creation and link this dataset when redistributing
under CC BY 4.0. This dataset is intended for LLM research, not psychological diagnosis
or consequential assessment of people.
'''
    (out/'README.md').write_text(card+research)
    return public

def upload(root,hf):
    report=export(root)
    guides=[name for name in ('README_human.md','README_agent.md') if (Path(root)/'dataset'/name).is_file()]
    subprocess.run([hf,'upload',REPO,str(Path(root)/'dataset'),'.','--type','dataset',
      '--include','README.md','--include','scenarios.jsonl','--include','manifest.json',
      *[arg for name in guides for arg in ('--include',name)],
      '--commit-message',f"Publish {report['records']} synthetic discovery scenarios"],check=True)
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--hf',required=True)
    ap.add_argument('--watch-job');ap.add_argument('--until',type=float);a=ap.parse_args()
    last=None;last_upload=0
    while True:
        report=export(a.root)
        terminal=False
        if a.watch_job:
            import sqlite3
            with sqlite3.connect((a.root/'results/persona_jobs.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
                row=db.execute('SELECT state FROM jobs WHERE id=?',(a.watch_job,)).fetchone()
            if row is None:raise ValueError('Unknown watched job')
            terminal=row[0] not in ('queued','running','stopping')
        if report['source_sha256']!=last and (not last_upload or terminal or time.time()-last_upload>=1800):
            report=upload(a.root,a.hf);last=report['source_sha256'];last_upload=time.time();print(json.dumps(report),flush=True)
        if not a.watch_job or terminal:return
        if a.until and time.time()>=a.until:raise TimeoutError('Publisher window expired before job finished')
        time.sleep(30)
if __name__=='__main__':main()
