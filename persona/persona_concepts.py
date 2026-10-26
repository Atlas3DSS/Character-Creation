"""Lexically grounded discovery contrasts over validated measured activations."""
from __future__ import annotations

import re
from typing import Callable

import numpy as np

from persona_neighbors import CORPORA, CoactivityIndex, MEASUREMENT
from persona_prompt_browser import facets

SEARCH_CORPORA = (*CORPORA, 'adult_sexuality')
MIN_GROUP = 4
LIMITATIONS = [
    'Text matching is a search aid, not semantic concept identification; synonyms are not inferred.',
    'Rankings describe discovery associations, not concept neurons, causal mechanisms or held-out validation.',
    'Controls are nonmatching measured prompts in the same corpus; topic and wording remain confounded.',
    'Only mapped candidates with saved final-prefill activations are searchable.',
    'The group graph shows pooled prompt correlations, not anatomical wiring or a proven topological feature.',
]


class ConceptSearch:
    def __init__(self, neighbors: CoactivityIndex, corpus_snapshot: Callable):
        self.neighbors = neighbors
        self.corpus_snapshot = corpus_snapshot

    def query(self, q: str, *, corpus: str = 'all', limit: int = 20) -> dict:
        if (not isinstance(q, str) or not 1 <= len(q.strip()) <= 160
                or corpus not in ('all', *SEARCH_CORPORA) or type(limit) is not int or not 1 <= limit <= 40):
            raise ValueError('Provide a concept query of 1–160 characters, a supported corpus and limit 1–40')
        tokens = re.findall(r'[^\W_]+', q.casefold())
        if not tokens:
            raise ValueError('The concept query must contain letters or numbers')
        def matches(row: dict) -> bool:
            text = row['prompt'] + '\n' + ' '.join(facets(row).values())
            words = set(re.findall(r'[^\W_]+', text.casefold()))
            return all(token in words for token in tokens)
        result = {'available': False, 'status': 'missing_source', 'query': q.strip(), 'candidates': [],
                  'examples': [], 'strata': [], 'source': {'corpus': corpus, 'min_per_group': MIN_GROUP,
                  'measurement': MEASUREMENT, 'split': 'discovery', 'limitations': LIMITATIONS.copy(),
                  'matching': 'All query words, case-insensitive, in prompt text, category or recorded trait/facet/domain metadata; whole words (underscores separate words).',
                  'statistic': 'Sample-weighted within-corpus mean difference divided by within-corpus total standard deviation.'}}
        try:
            rows, binding = self.corpus_snapshot()
        except (OSError, ValueError) as error:
            result.update(status='invalid_corpus', message=str(error)); return result
        matching = {key for key, row in rows.items() if row['split'] == 'discovery'
                    and (corpus == 'all' or row['corpus'] == corpus) and matches(row)}
        result['source'].update(binding, matching_corpus_rows=len(matching), matching_measured_rows=0, unmeasured_matches=len(matching))
        result['examples'] = [{'id': key, 'corpus': rows[key]['corpus'], 'category': rows[key]['category'],
                               'prompt': rows[key]['prompt'][:700], 'measured': False}
                              for key in sorted(matching)[:12]]
        with self.neighbors.lock:
            try:
                data = self.neighbors._load()
            except (OSError, ValueError) as error:
                result['message'] = f'Validated measured activations are unavailable: {error}'
                return result
            arrays = data['arrays']; ids = arrays['prompt_ids'].tolist()
            # Immutable corpus IDs bind the actual matched text, even after append-only ingestion.
            if any(key not in rows or rows[key]['split'] != 'discovery' or rows[key]['corpus'] != str(name)
                   for key, name in zip(ids, arrays['corpora'])):
                result.update(status='invalid_binding', message='Measured prompt IDs do not bind to the current discovery corpus.')
                return result
            result['source'].update(model_identity=self.neighbors.model.identity,
                artifact_sha256=data['artifact_sha256'], map_sha256=data['map_sha256'],
                captured_corpus_sha256=data['metadata'].get('corpus_sha256'), candidates=len(data['coordinates']))
            observed = set(ids) & matching
            result['source'].update(matching_measured_rows=len(observed), unmeasured_matches=len(matching - observed))
            result['examples'] = [{'id': key, 'corpus': rows[key]['corpus'], 'category': rows[key]['category'],
                                   'prompt': rows[key]['prompt'][:700], 'measured': key in observed}
                                  for key in sorted(matching, key=lambda key: (key not in observed, key))[:12]]
            flags = np.array([key in matching for key in ids], dtype=bool)
            values = arrays['values'].astype(np.float64)
            scores = np.zeros(values.shape[1]); weights = np.zeros(values.shape[1]); eligible_rows = np.zeros(len(ids), bool)
            for name in SEARCH_CORPORA if corpus == 'all' else (corpus,):
                mask = arrays['corpora'] == name; positive = mask & flags; negative = mask & ~flags
                n_pos, n_neg = int(positive.sum()), int(negative.sum())
                usable = min(n_pos, n_neg) >= MIN_GROUP
                result['strata'].append({'corpus': name, 'matched': n_pos, 'controls': n_neg, 'used': usable})
                if not usable: continue
                # Scale before moments: avoid overflow while preserving standardized differences.
                local = values[mask]; scale = np.max(np.abs(local), axis=0); scale[scale == 0] = 1
                scaled = local / scale; std = scaled.std(axis=0, ddof=1); valid = std > 0
                local_flags = flags[mask]
                delta = scaled[local_flags].mean(axis=0) - scaled[~local_flags].mean(axis=0)
                weight = n_pos * n_neg / (n_pos + n_neg)
                scores[valid] += weight * delta[valid] / std[valid]; weights[valid] += weight
                eligible_rows |= mask
            if not eligible_rows.any():
                result.update(status='insufficient_support', message=f'Need at least {MIN_GROUP} measured matches and {MIN_GROUP} measured controls within one corpus. Try another query or corpus.')
                return result
            valid = weights > 0; scores[valid] /= weights[valid]
            order = sorted(np.flatnonzero(valid).tolist(), key=lambda i: (-abs(scores[i]), data['coordinates'][i]))[:limit]
            result['candidates'] = [{'layer': data['coordinates'][i][0], 'neuron': data['coordinates'][i][1],
                'score': float(scores[i]), 'direction': 'higher' if scores[i] > 0 else 'lower' if scores[i] < 0 else 'unchanged'} for i in order]
            result['source']['ranked_candidates'] = int(valid.sum())
            result['source']['used_measured_rows'] = int(eligible_rows.sum())
            result['source']['matched_used_rows'] = int((eligible_rows & flags).sum())
            result['source']['control_used_rows'] = int((eligible_rows & ~flags).sum())
            # Induced co-activity for the selected collection, using the same measured population.
            if order:
                profiles = values[eligible_rows][:, order]
                scale = np.max(np.abs(profiles), axis=0); scale[scale == 0] = 1
                profiles = profiles / scale; profiles -= profiles.mean(axis=0)
                norms = np.linalg.norm(profiles, axis=0); profiles /= np.where(norms > 0, norms, 1)
                matrix = np.clip(profiles.T @ profiles, -1., 1.)
                eigenvalues, eigenvectors = np.linalg.eigh(matrix); axes = np.argsort(eigenvalues)[::-1][:3]
                layout = eigenvectors[:, axes] * np.sqrt(np.maximum(eigenvalues[axes], 0))
                layout = np.pad(layout, ((0, 0), (0, 3-layout.shape[1])))
                for axis in range(3):
                    if layout[np.argmax(abs(layout[:, axis])), axis] < 0: layout[:, axis] *= -1
                nodes = [dict(item, id=f"L{item['layer']}N{item['neuron']}", position3d=layout[j].tolist(),
                              position=layout[j, :2].tolist(), selected=j == 0) for j, item in enumerate(result['candidates'])]
                result['graph'] = {'nodes': nodes, 'edges': [{'source': nodes[a]['id'], 'target': nodes[b]['id'],
                    'r': float(matrix[a, b]), 'n': int(eligible_rows.sum())} for a in range(len(nodes)) for b in range(a+1, len(nodes))],
                    'directed': False, 'layout': 'Principal coordinates of measured activation profiles.',
                    'semantics': 'Exploratory selected collection; pooled correlations may reflect corpus/task mixtures.'}
            result.update(available=bool(order), status='ready' if order else 'constant_candidates',
                message='Ranked measured associations. Inspect supporting prompts and neuron neighborhoods.' if order else 'All observed candidates are constant within the supported corpora.')
            return result
