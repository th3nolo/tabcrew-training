"""Read the pinned, previously replay-audited corpus without rewriting provenance."""
from pathlib import Path, PurePosixPath, PureWindowsPath
import core as g
from client import StopCollection


def manifest(root):
    path = root / 'inherited-dataset.json'
    return g.read_json(path) if path.exists() else None


def evidence_path(parent, relative):
    """Manifest evidence names are relative POSIX paths, never traversal or ADS."""
    if (not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative
            or PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).drive
            or any(part in ('', '.', '..') for part in relative.split('/'))):
        raise StopCollection('invalid_inherited_manifest')
    candidate = (parent / relative).resolve()
    if not candidate.is_relative_to(parent.resolve()):
        raise StopCollection('invalid_inherited_manifest')
    return candidate


def _records(root, ancestors):
    root = Path(root).resolve()
    if root in ancestors or len(ancestors) >= 32:
        raise StopCollection('inherited_manifest_cycle')
    saved = manifest(root)
    if saved is None:
        return {}
    parent_value = saved['parent_revision']
    if not isinstance(parent_value, str) or not parent_value:
        raise StopCollection('invalid_inherited_manifest')
    parent = Path(parent_value)
    parent = (parent if parent.is_absolute() else root / parent).resolve()
    if parent == root or parent in ancestors:
        raise StopCollection('inherited_manifest_cycle')
    recurse = saved.get('include_parent_inherited', False)
    if type(recurse) is not bool or type(saved['accepted_count']) is not int:
        raise StopCollection('invalid_inherited_manifest')
    if recurse and 'inherited-dataset.json' not in saved['evidence_sha256']:
        raise StopCollection('unpinned_inherited_manifest')
    if ('parent_contract_sha256' in saved and saved['evidence_sha256'].get('contract.json')
            != saved['parent_contract_sha256']):
        raise StopCollection('invalid_inherited_manifest')
    for relative, expected in saved['evidence_sha256'].items():
        candidate = evidence_path(parent, relative)
        if not candidate.is_file() or g.sha(candidate) != expected:
            raise StopCollection('inherited_dataset_evidence_changed')
    result = _records(parent, ancestors | {root}) if recurse else {}
    for item in saved['accepted_records']:
        candidate = evidence_path(parent, item['path'])
        if item['path'] not in saved['evidence_sha256']:
            raise StopCollection('invalid_inherited_manifest')
        record = g.read_json(candidate)
        episode = record['example']
        if record['example_sha256'] != g.digest(episode) or episode['id'] != item['id']:
            raise StopCollection('inherited_example_changed')
        if episode['id'] in result:
            raise StopCollection('invalid_inherited_manifest')
        result[episode['id']] = episode
    if len(result) != saved['accepted_count'] or g.digest(sorted(result.values(), key=lambda ep:ep['id'])) != saved['content_sha256']:
        raise StopCollection('inherited_corpus_changed')
    g.validate_corpus(list(result.values()))
    return result


def records(root):
    try:
        return _records(root, set())
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise StopCollection('invalid_inherited_manifest') from exc
