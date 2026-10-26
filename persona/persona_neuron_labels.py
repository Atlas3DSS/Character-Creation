"""Read calibrated neuron labels; correlation and frame agreement are not probabilities."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'character-creation.neuron-labels.v1'
EVIDENCE_SCHEMA = 'character-creation.neuron-label-evidence.v1'
METHODS = {'heldout_precision'}
CLAIM = 'Empirical precision conditional on the recorded probe and held-out dataset; not the probability that a neuron means a word. Inverse denotes a negative association.'


def read_registry(path: Path, model_identity: str, *, layers: int = 64, neurons_per_layer: int = 17408) -> dict:
    path = Path(path)
    result = {'state': 'pending', 'model_identity': model_identity, 'labels': [], 'candidates': [], 'rejected': [], 'claim': CLAIM}
    if not path.is_file():
        result['reason'] = 'No calibrated label registry has been published.'
        return result
    try:
        data = json.loads(path.read_text())
        if data.get('schema') != SCHEMA or data.get('model_identity') != model_identity:
            raise ValueError('Registry schema or model identity does not match')
        if not isinstance(data.get('labels'), list):
            raise ValueError('Registry labels must be a list')
    except (ValueError, OSError, AttributeError) as exc:
        result.update(state='invalid', reason=str(exc))
        return result
    accepted = []
    for row in data['labels']:
        try:
            if not isinstance(row, dict):
                raise ValueError('Label record must be an object')
            layer, neuron = row.get('layer'), row.get('neuron')
            if type(layer) is not int or type(neuron) is not int or not (0 <= layer < layers and 0 <= neuron < neurons_per_layer):
                raise ValueError('Coordinates outside model')
            if not isinstance(row.get('label'), str) or not row['label'].strip() or len(row['label']) > 80:
                raise ValueError('Label must be nonempty and at most 80 characters')
            if row.get('direction') not in ('positive', 'inverse'):
                raise ValueError('Direction must be positive or inverse')
            candidate = {key: row[key] for key in ('layer', 'neuron', 'label', 'direction')}
            confidence = row.get('confidence', {})
            value = confidence.get('value')
            if (confidence.get('calibrated') is not True or type(value) not in (int, float)
                    or not math.isfinite(value) or not .70 < value <= 1 or confidence.get('method') not in METHODS):
                raise LookupError('No qualifying calibrated confidence above 70%')
            evidence = row.get('evidence', {})
            relative = evidence.get('artifact')
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise LookupError('Evidence artifact must be relative to the registry')
            artifact = (path.parent / relative).resolve()
            if artifact.parent != path.parent.resolve():
                raise LookupError('Evidence artifact must be beside the registry')
            raw = artifact.read_bytes()
            if hashlib.sha256(raw).hexdigest() != evidence.get('sha256'):
                raise LookupError('Evidence artifact digest differs')
            proof = json.loads(raw)
            if (proof.get('schema') != EVIDENCE_SCHEMA or proof.get('model_identity') != model_identity
                    or any(proof.get(key) != candidate[key] for key in candidate)
                    or proof.get('confidence') != confidence):
                raise LookupError('Evidence label, confidence or model binding differs')
            validation = proof.get('validation', {})
            calibration = proof.get('calibration', {})
            def sha(value):
                return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
            discovery = proof.get('discovery_dataset_sha256')
            if (proof.get('strong_related') is not True or validation.get('passed') is not True
                    or validation.get('split') != 'held_out'
                    or calibration.get('method') != 'heldout_precision' or calibration.get('split') != 'held_out'
                    or not sha(discovery) or not sha(validation.get('dataset_sha256')) or not sha(calibration.get('dataset_sha256'))
                    or discovery in (validation['dataset_sha256'], calibration['dataset_sha256'])
                    or proof.get('claim') != 'probe_conditional_precision' or not isinstance(proof.get('probe'), str) or not proof['probe'].strip()):
                raise LookupError('Strong association, probe-conditioned precision and discovery-disjoint held-out datasets required')
            outcomes = validation.get('outcomes', [])
            if (not isinstance(outcomes, list) or len(outcomes) < 20
                    or any(not isinstance(r, dict) or not isinstance(r.get('unit_id'), str) or not r['unit_id'] or type(r.get('correct')) is not bool for r in outcomes)
                    or len({r['unit_id'] for r in outcomes}) != len(outcomes)):
                raise LookupError('At least 20 distinct independent held-out probe outcomes required')
            tp = sum(r['correct'] for r in outcomes); fp = len(outcomes) - tp
            if validation.get('tp') != tp or validation.get('fp') != fp:
                raise LookupError('Precision counts differ from held-out outcomes')
            total = tp + fp; precision = tp / total; z = 1.959963984540054
            lower = (precision + z*z/(2*total) - z*math.sqrt(precision*(1-precision)/total + z*z/(4*total*total))) / (1+z*z/total)
            if abs(value - precision) > 1e-9 or lower <= .70:
                raise LookupError('Recomputed precision or its Wilson 95% lower bound fails the >70% gate')
            confidence = {**confidence, 'value': precision, 'lower_bound': lower, 'interval_method': 'wilson_95', 'samples': total, 'claim': 'probe_conditional_precision'}
            accepted.append({**candidate, 'confidence': confidence, 'evidence': evidence})
        except (LookupError, OSError, json.JSONDecodeError) as exc:
            result['candidates'].append({**candidate, 'validation': 'unvalidated', 'reason': str(exc),
                                         'association': row.get('association'), 'evidence': row.get('evidence')})
        except (ValueError, TypeError, AttributeError) as exc:
            result['rejected'].append({'reason': str(exc)})
    groups = {}
    for row in sorted(accepted, key=lambda r: (-r['confidence']['value'], r['label'], r['direction'])):
        key = (row['layer'], row['neuron'])
        group = groups.setdefault(key, [])
        if len(group) < 3 and not any(r['label'].casefold() == row['label'].casefold() and r['direction'] == row['direction'] for r in group):
            group.append(row)
    result['labels'] = [row for group in groups.values() for row in group]
    result['state'] = 'ready' if result['labels'] else 'pending'
    if not result['labels']:
        result['reason'] = 'No labels pass the calibrated confidence and evidence gates yet.'
    return result
