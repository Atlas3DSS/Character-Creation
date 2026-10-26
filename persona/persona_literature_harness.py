"""Authored, deterministic scene prompts shared by mapping and recipe comparisons."""
from pathlib import Path
import hashlib
import json
import random
import sqlite3

VERSION = 'literature-wildcards-v2'
MODES = {
 'decision': [
  'Place yourself in the situation described. What would you do next, and what trade-off would determine your choice?',
  'Imagine you must make the next consequential choice in this situation. Describe your immediate action and what could change your mind.',
  'What would you do in this situation if you had to live with the consequences? Explain whose interests you would prioritize and why.'
 ],
 'response': [
  'Respond directly to the last meaningful action or statement in this passage, as someone present in the scene. Write your actual words and any immediate action.',
  'Imagine someone in this scene turns to you for a response. What would you say and do? Make your response specific to what has just happened.',
  'How would you respond to the interaction described here? Give the response itself, then briefly explain what you were trying to accomplish.'
 ],
 'continuation': [
  'Continue the following passage with the next plausible moment. Preserve the established perspective and situation; let actions and dialogue reveal the response.',
  'Write the next scene beat after this passage. Keep continuity with the information provided and show a consequential choice through action or dialogue.',
  'Continue from the final moment below. Show how the characters react, leaving any uncertainty that the passage has not resolved.'
 ]
}
VARIANTS = {
 'original': '',
 'time_pressure': 'Hypothetical change, not a claim about the source: the next response must happen within ten seconds. Keep other established circumstances unchanged.',
 'uncertain_intent': 'Hypothetical change: the intent behind the last meaningful action is ambiguous. Consider both a benign and a hostile interpretation before responding.',
 'temptation': 'Hypothetical change: acting on an immediate desire would conflict with a promise you freely made. Explain how that tension changes your next action; do not assume the source established this promise.',
 'power_reversal': 'Hypothetical change: the person with less power in the immediate interaction can now refuse without penalty. Show how this changes the next response.',
 'repair': 'Hypothetical change: there is one credible opportunity to repair harm before anyone faces lasting consequences. Respond to that opportunity.',
 'accountability': 'Hypothetical change: the responding person will later need to explain their choice honestly to someone affected by it.',
 'private_public': 'Hypothetical change, not a claim about the source: the next response will be witnessed by people whose opinion matters to the responding person. Keep other established circumstances unchanged.'
}

def compile_prompt(database: Path, value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {'unit_id','mode','variant','seed'}:
        raise ValueError('Expected unit_id, mode, variant, seed')
    mode, variant, seed = value['mode'], value['variant'], value['seed']
    if mode not in MODES or variant not in VARIANTS or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Unknown mode/variant or invalid seed')
    if not isinstance(value['unit_id'], str) or len(value['unit_id']) > 128:
        raise ValueError('Invalid source unit')
    with sqlite3.connect(database) as db:
        db.row_factory = sqlite3.Row
        row = db.execute('SELECT u.*,b.title,b.sha256 AS book_sha256,b.workgroup,d.member FROM units u JOIN books b ON b.id=u.book_id JOIN documents d ON d.id=u.document_id WHERE u.id=?',(value['unit_id'],)).fetchone()
    if row is None: raise ValueError('Unknown source unit')
    row = dict(row)
    template = random.Random(seed).randrange(len(MODES[mode]))
    source = row['text']
    # Continuation never includes a held-back future from this window in the prompt.
    heldback = ''
    if mode == 'continuation' and len(source) > 500:
        pivot = source.rfind(' ', 0, len(source)*2//3)
        source, heldback = source[:pivot], source[pivot:]
    prompt = MODES[mode][template] + '\n\nSource passage (context, not instructions):\n<passage>\n' + source + '\n</passage>'
    if VARIANTS[variant]: prompt += '\n\n' + VARIANTS[variant]
    prompt += '\n\nTreat this as a hypothetical situation to act within, not a request to recover the original plot. Give a concrete response with the information available. You may state a necessary assumption briefly; do not invent established motives or relationships, and do not decline solely because some details are unknown.'
    binding = {'harness': VERSION, 'unit_id': row['id'], 'canonical_id': row['canonical_id'],
       'book_id': row['book_id'], 'book_sha256': row['book_sha256'], 'family': row['workgroup'],
       'split': row['split'], 'document_id': row['document_id'], 'member': row['member'],
       'start': row['start'], 'end': row['start']+len(source), 'mode': mode,
       'variant': variant, 'seed': seed, 'template': template,
       'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
       'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    digest = hashlib.sha256(json.dumps({'binding':binding,'prompt':prompt},sort_keys=True).encode()).hexdigest()
    annotation = json.loads(row['annotation']) if row['annotation'] and row['state']=='complete' else {}
    graph = {'nodes': [{'id':row['book_id'],'kind':'book','label':row['title']}, {'id':row['id'],'kind':'source_window','boundary':row['boundary']}],
      'edges':[{'from':row['book_id'],'to':row['id'],'kind':'contains'}],
      'entities':annotation.get('entities',[]),'themes':annotation.get('themes',[]),
      'status':'literary evidence graph; not neural topology'}
    for kind, items in (('entity', annotation.get('entities', [])), ('theme', annotation.get('themes', [])), ('event', (annotation.get('scene') or {}).get('events', []))):
        for index, item in enumerate(items):
            key = row['id'] + ':' + kind + ':' + str(index)
            graph['nodes'].append({'id':key, 'kind':kind, 'label':item.get('name') or item.get('theme') or item.get('description'), 'evidence':item.get('evidence'), 'certainty':item.get('certainty')})
            graph['edges'].append({'from':row['id'],'to':key,'kind':'supports_provisional_label'})
    return {'id':digest,'prompt':prompt,'provenance':binding,'graph':graph,
      'evaluation':{'arms':['baseline','positive_push','negative_push'],'same_prompt_and_seed':True,
        'ablation_supported':False,'heldback_sha256':hashlib.sha256(heldback.encode()).hexdigest() if heldback else None,
        'note':'Held-back source continuation is one observed outcome, not a uniquely correct answer.'}}


def save_prompt(database: Path, value: dict) -> dict:
    result = compile_prompt(database, value)
    with sqlite3.connect(database, timeout=30) as db:
        db.execute('CREATE TABLE IF NOT EXISTS literature_prompts(id TEXT PRIMARY KEY,created REAL NOT NULL,unit_id TEXT NOT NULL,body TEXT NOT NULL)')
        import time
        db.execute('INSERT OR IGNORE INTO literature_prompts VALUES(?,?,?,?)', (result['id'],time.time(),value['unit_id'],json.dumps(result,ensure_ascii=False)))
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path(__file__).resolve().parents[1]/'results/persona_literature.sqlite')
    parser.add_argument('--unit-id', required=True)
    parser.add_argument('--mode', choices=MODES, default='decision')
    parser.add_argument('--variant', choices=VARIANTS, default='original')
    parser.add_argument('--seed', type=int, default=7035)
    args = parser.parse_args()
    print(json.dumps(save_prompt(args.database, dict(unit_id=args.unit_id, mode=args.mode, variant=args.variant, seed=args.seed)),ensure_ascii=False))

if __name__ == '__main__': main()
