"""Validate authored text-concept probes; never infer model qualification.

Labels concern assertions in a short textual scene. A label word, hypothetical,
quoted example or absent object alone is not a positive. Each scenario is used
once: no alternate framing of a family is counted as independent evidence.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

SCHEMA = 'character-creation.label-probes.v1'
DEFINITIONS = {
    'joy': 'A person in the described scene currently experiences positive pleasure, delight or happiness. A smile alone, politeness, absent distress, quoted vocabulary or a hypothetical feeling is insufficient.',
    'sadness': 'A person in the described scene currently experiences sorrow, grief, dejection or felt disappointment. Bad circumstances alone, fatigue, anger, quoted vocabulary or a hypothetical feeling is insufficient.',
    'sexual_desire': 'An explicitly adult person currently feels sexual attraction or wants consensual erotic intimacy with another explicitly adult person. Affection, companionship, aesthetic appreciation, biological terminology or discussion of desire alone is insufficient; no graphic acts are described.',
    'confusion': 'A person currently struggles to make sense of information, reconcile interpretations or understand what is happening. Merely lacking a fact, choosing between understood options, disagreement, surprise or mentioning the word is insufficient.',
    'green': 'A physical object or surface in the described scene is explicitly asserted to currently have green visible coloration, including ordinary shades of green. Names, printed words, environmental policies, metaphors, future colors and explicitly absent green objects are insufficient. This is a textual color assertion, not visual verification.',
    'dog': 'At least one living domestic dog is physically present in the described scene. Breed names count when the scene identifies a living animal; dog words, pictures, statues, metaphors, other species, wishes and explicitly absent dogs do not. This is textual animal presence, not visual verification.',
}
LIMITATIONS = [
    'Authored synthetic labels require independent human review before strong external-validity claims.',
    'No model responses or activation measurements are included; this dataset establishes no accuracy threshold.',
    'Each split contains distinct scenario families, not alternate frames of the same context.',
    'The intended scope is the asserted text-concept relation; imagery, tone and real-world truth are not verified.',
    'Heldout examples must not be used for feature selection, threshold fitting or iterative prompt tuning.',
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def bind_row(row):
    """Return a new content-bound row; labels and source text are included."""
    body = {k: v for k, v in row.items() if k != 'id'}
    return dict(body, id=digest(body))


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON key: ' + key)
        value[key] = item
    return value


def validate_probes(dataset):
    """Return structural counts or raise ValueError; not a semantic judge."""
    if not isinstance(dataset, dict) or dataset.get('schema') != SCHEMA:
        raise ValueError('Unknown label probe schema')
    if dataset.get('sha256') != digest({k: v for k, v in dataset.items() if k != 'sha256'}):
        raise ValueError('Dataset content binding differs')
    if dataset.get('definitions') != DEFINITIONS:
        raise ValueError('Concept definitions differ from the evaluator contract')
    rows = dataset.get('rows')
    if not isinstance(rows, list) or len(rows) != 432:
        raise ValueError('Expected 432 authored examples')
    seen_ids, families, prompts = set(), set(), set()
    counts = Counter()
    required = {'id', 'concept', 'split', 'label', 'family_id', 'prompt', 'definition', 'evidence_span'}
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError('Unexpected row fields')
        if row['id'] != bind_row(row)['id'] or row['id'] in seen_ids:
            raise ValueError('Row identity is duplicated or not content-bound')
        if row['concept'] not in DEFINITIONS or row['definition'] != DEFINITIONS[row['concept']]:
            raise ValueError('Unknown concept or inconsistent definition')
        if row['split'] not in ('discovery', 'heldout') or type(row['label']) is not bool:
            raise ValueError('Invalid split or non-boolean label')
        if not isinstance(row['family_id'], str) or not re.fullmatch(r'[a-z][a-z0-9_-]+', row['family_id']):
            raise ValueError('Scenario family must be a stable descriptive identifier')
        if row['family_id'] in families:
            raise ValueError('Scenario family reused; frames are not independent examples')
        if not isinstance(row['prompt'], str) or not row['prompt'].strip():
            raise ValueError('Empty source prompt')
        normalized = ' '.join(row['prompt'].casefold().split())
        if normalized in prompts:
            raise ValueError('Duplicate source context')
        if not isinstance(row['evidence_span'], str) or not row['evidence_span'].strip() or row['evidence_span'] not in row['prompt']:
            raise ValueError('Evidence must be an exact nonempty source span')
        if row['concept'] == 'sexual_desire' and row['label'] and 'adult' not in normalized:
            raise ValueError('Positive desire scenes must explicitly establish adults')
        seen_ids.add(row['id']); families.add(row['family_id']); prompts.add(normalized)
        counts[(row['concept'], row['split'], row['label'])] += 1
    expected = {(concept, split, label): count for concept in DEFINITIONS for split, count in [('discovery', 12), ('heldout', 24)] for label in (False, True)}
    if dict(counts) != expected:
        raise ValueError('Require 12 discovery and 24 heldout examples of each label per concept')
    return {'rows': len(rows), 'families': len(families), 'concepts': len(DEFINITIONS),
            'counts': [{'concept': c, 'split': s, 'label': label, 'count': count} for (c, s, label), count in sorted(counts.items())],
            'validation_scope': 'structure, exact source spans and content identity only; labels are authored, not machine-verified'}


def load_probes(path):
    dataset = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_pairs)
    validate_probes(dataset)
    return dataset


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_probes(load_probes(args.path)), indent=2))
