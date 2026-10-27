"""Register the completed authored corpus without changing its original row IDs."""
from collections import Counter
from copy import deepcopy
import hashlib
import json

from persona_vectors import canonical_digest


def registered_rows(raw, manifest_raw):
    manifest = json.loads(manifest_raw)
    if (manifest.get('schema') != 'character-creation.persona-perturbation.v1'
            or manifest.get('status') != 'complete'
            or hashlib.sha256(raw).hexdigest() != manifest.get('output_sha256')):
        raise ValueError('Persona prompt supplement is not a completed, hash-verified publication')
    source = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(source) != manifest.get('records') or not source:
        raise ValueError('Persona prompt supplement count differs')
    ids, groups, splits, result = set(), {}, Counter(), []
    for original in source:
        if not isinstance(original, dict):
            raise ValueError('Persona prompt row must be an object')
        identity, group = original.get('id'), original.get('source_group')
        prompt, split = original.get('prompt'), original.get('split')
        if (not isinstance(identity, str) or not identity or identity in ids
                or not isinstance(group, str) or not group or group != original.get('pair_id')
                or split not in ('discovery', 'heldout')
                or original.get('corpus') != 'persona_perturbation'
                or original.get('schema') != 'character-creation.manager-corpus.v1'
                or original.get('system') != '' or original.get('thinking') is not False
                or not isinstance(prompt, str) or not 1 <= len(prompt) <= 100000
                or original.get('source', {}).get('prompt_sha256') != hashlib.sha256(prompt.encode()).hexdigest()
                or (group in groups and groups[group] != split)):
            raise ValueError('Persona prompt identity, neutral input, source hash or split differs')
        ids.add(identity); groups[group] = split; splits[split] += 1
        row = deepcopy(original)
        row.pop('id')
        row['source_row_id'] = identity
        row['synthetic'] = True
        # UI facet names describe prompt design only, never measured neuron meanings.
        row['labels'].update(prompt_design_trait='persona_response',
            prompt_design_facet=original['category'], scenario_domain=original.get('scenario_family', original['category']))
        row['id'] = canonical_digest(row)
        result.append(row)
    if dict(splits) != manifest.get('split_counts') or len(groups) != manifest.get('unique_core_groups'):
        raise ValueError('Persona prompt split/group counts differ')
    return result
