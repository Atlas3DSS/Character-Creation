"""Publish validated label evidence and return an owned pause to the main map."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request
from urllib.parse import urlsplit

from persona_neuron_labels import read_registry
from persona_vectors import ModelSpec

from persona_overnight import remote_config


def remote_bytes(name, config=None):
    if not isinstance(name,str) or name in ('.','..') or not re.fullmatch(r'[A-Za-z0-9_.-]+',name):
        raise ValueError('Flat evidence filenames only')
    settings=remote_config(config)
    script="import socket,sys;from pathlib import Path;"+"\nif socket.gethostname()!=%r:raise RuntimeError('Remote hostname differs from configuration')\n"%settings['hostname']+"sys.stdout.buffer.write(Path(%r).read_bytes())"%(settings['remote_root']+'/'+name)
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                                    settings['ssh_host'], shlex.join(['python3','-c',script])], timeout=60)


def api_base(value):
    if not isinstance(value,str) or any(c.isspace() or ord(c)<32 for c in value):raise ValueError('API URL must be nonempty text without whitespace')
    parsed=urlsplit(value)
    if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('','/'):
        raise ValueError('API URL must be an HTTP(S) origin without credentials or a path')
    try:parsed.port
    except ValueError as error:raise ValueError('Invalid API port') from error
    return value.rstrip('/')


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def publish(root, registry, fetch=remote_bytes):
    model = ModelSpec.load(root/'data/persona_model.json')
    if registry.get('schema') != 'character-creation.neuron-labels.v1' or registry.get('model_identity') != model.identity:
        raise ValueError('Label publication belongs to another model or schema')
    for row in registry['labels']:
        receipt = row['evidence']; name = receipt['artifact']
        if not name.startswith('persona_label_evidence_') or not name.endswith('.json') or Path(name).name != name:
            raise ValueError('Unexpected evidence filename')
        content = fetch(name)
        if hashlib.sha256(content).hexdigest() != receipt['sha256']:
            raise ValueError('Evidence content differs from its immutable digest')
        target = root/'results'/name
        if target.exists() and target.read_bytes() != content:
            raise ValueError('Existing immutable evidence would be overwritten')
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(content)
    path = root/'results/persona_neuron_labels.json'
    old = json.loads(path.read_text()) if path.exists() else {'schema': registry['schema'], 'model_identity': model.identity, 'labels': []}
    if old['model_identity'] != model.identity:
        raise ValueError('Existing registry model differs')
    key = lambda row: (row['layer'], row['neuron'], row['label'], row['direction'])
    updated = {key(row): row for row in old['labels']}
    updated.update({key(row): row for row in registry['labels']})
    merged = {**old, 'labels': list(updated.values()), 'qualification_binding': registry['binding']}
    # Validate before publishing the combined registry. Evidence files are
    # immutable; this temporary file is not a second user-facing registry.
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(merged))
    checked = read_registry(temporary, model.identity, layers=model.layers, neurons_per_layer=model.intermediate)
    if checked['state'] == 'invalid' or checked['rejected']:
        raise ValueError('Reader rejected the label publication')
    expected = sum(row['confidence'].get('calibrated') is True for row in registry['labels'])
    # The viewer deliberately caps labels per neuron. Validate qualifications
    # individually so that density limiting is not mistaken for invalid data.
    for row in registry['labels']:
        if row['confidence'].get('calibrated'):
            temporary.write_text(json.dumps({**merged, 'labels': [row]}))
            if len(read_registry(temporary, model.identity, layers=model.layers, neurons_per_layer=model.intermediate)['labels']) != 1:
                raise ValueError('Qualified label failed independent reader verification')
    temporary.write_text(json.dumps(merged, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)
    return {'new_qualified': expected, 'total_qualified': len(checked['labels']), 'candidates': len(checked['candidates'])}


def api(path, body=None, base_url='http://127.0.0.1:8792'):
    request = urllib.request.Request(api_base(base_url)+path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(request, timeout=20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pause-time', type=float, required=True)
    parser.add_argument('--config', type=Path, help='Explorer JSON config containing overnight_remote')
    parser.add_argument('--api-url', default='http://127.0.0.1:8792', type=api_base)
    args = parser.parse_args()
    config_path=args.config or args.root/'persona_explorer.local.json'
    config=json.loads(config_path.read_text()) if args.config or config_path.exists() else {}
    remote_config(config)
    fetch=lambda name:remote_bytes(name,config)
    path = args.root/'results/persona_label_handoff.json'
    deadline = time.time()+2100
    while time.time() < deadline:
        try:
            status = json.loads(fetch('persona_label_qualification_status.json'))
            atomic(path, {'state': 'waiting', 'updated': time.time(), 'qualification_state': status['state'], 'counts': status['counts']})
            if status['state'] in ('complete', 'incomplete', 'failed') and not status['active']:
                result = publish(args.root, json.loads(fetch('persona_neuron_labels.json')),fetch=fetch) if status['state'] == 'complete' else {'publication': 'not_qualified_incomplete_run'}
                map_state = api('/api/overnight',base_url=args.api_url)
                if map_state.get('remote', {}).get('pause_requested_at') != args.pause_time:
                    atomic(path, {'state': 'pause_owner_changed', 'updated': time.time(), **result}); return
                resumed = api('/api/overnight', {'action': 'resume'},base_url=args.api_url)
                if resumed.get('remote', {}).get('pause_requested_at') is not None:
                    raise ValueError('Map did not acknowledge resume')
                session_path = args.root/'results/persona_overnight_session.json'
                session = json.loads(session_path.read_text())
                if session.get('auxiliary') == 'label_qualification':
                    session.pop('auxiliary'); atomic(session_path, session)
                atomic(path, {'state': 'complete', 'updated': time.time(), 'overnight_resumed': True, **result}); return
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            atomic(path, {'state': 'waiting_or_blocked', 'updated': time.time(), 'error': str(error)})
        time.sleep(5)
    atomic(path, {'state': 'timeout', 'updated': time.time(), 'reason': 'No unverified publication or pause ownership override performed'})
    raise SystemExit(1)


if __name__ == '__main__':
    main()
