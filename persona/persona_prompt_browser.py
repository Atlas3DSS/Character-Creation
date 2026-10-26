"""Metadata navigation over registered prompts; edges are taxonomy, not neural links."""
from collections import Counter

FIELDS = ('trait', 'facet', 'domain', 'category', 'origin')


def facets(row):
    labels = row.get('labels') or {}
    plan = (row.get('generator') or {}).get('plan') or {}
    inventory = labels.get('inventory_item') or {}
    return {
        'trait': str(labels.get('prompt_design_trait') or plan.get('trait') or inventory.get('dimension') or 'unlabelled').lower(),
        'facet': str(labels.get('prompt_design_facet') or labels.get('facet') or plan.get('facet') or 'unlabelled'),
        'domain': str(labels.get('scenario_domain') or plan.get('domain') or 'unlabelled'),
        'category': row.get('category', 'unlabelled'),
        'origin': 'generated' if row.get('synthetic') or labels.get('synthetic') else 'authored / imported',
    }


def browse(rows, label, *, corpus='big_five', split='discovery', q='', group='trait',
           trait='', facet='', domain='', category='', origin='', offset=0, limit=24, spread=False):
    if group not in FIELDS or split not in ('discovery', 'heldout', 'all'):
        raise ValueError('Invalid grouping or split')
    if not isinstance(q, str) or len(q) > 512 or not 0 <= offset <= 1000000 or not 1 <= limit <= 100:
        raise ValueError('Invalid search or pagination')
    filters = dict(trait=trait, facet=facet, domain=domain, category=category, origin=origin)
    if any(not isinstance(v, str) or len(v) > 256 for v in filters.values()):
        raise ValueError('Invalid category filter')
    matches, groups = [], Counter()
    for row in rows.values():
        if (corpus and row['corpus'] != corpus) or (split != 'all' and row['split'] != split):
            continue
        tags = facets(row)
        if any(value and tags[key] != value for key, value in filters.items()):
            continue
        if q.casefold() not in (' '.join(tags.values()) + ' ' + row['prompt'] + ' ' + label(row) + ' ' + row['id']).casefold():
            continue
        matches.append((row, tags)); groups[tags[group]] += 1
    total = len(matches)
    offset = min(offset, max(0, ((total - 1) // limit) * limit))
    if spread and total > limit:
        chosen = [matches[round(i * (total - 1) / (limit - 1))] for i in range(limit)] if limit > 1 else [matches[total // 2]]
    else:
        chosen = matches[offset:offset + limit]
    return {'total': total, 'offset': offset, 'limit': limit, 'spread': spread, 'group': group,
            'groups': [{'key': key, 'count': count} for key, count in sorted(groups.items(), key=lambda item: (-item[1], item[0]))],
            'rows': [{key: row.get(key, '') for key in ('id','corpus','split','category','prompt','system')}
                     | {'label': label(row), 'facets': tags, 'max_tokens': row.get('max_tokens', 2048)} for row, tags in chosen],
            'interpretation': 'Groups use recorded prompt-design metadata, not measured traits or semantic clusters.'}
