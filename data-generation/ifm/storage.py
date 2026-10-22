"""Immutable provider-specific exports, reused from v1."""
import json
import os
import uuid
import core as g
from client import MODEL, ENDPOINT
from files import atomic

def export_snapshot(root, accepted, expected):
    rows = sorted(accepted.values(), key=lambda ep: ep['id'])
    validation = g.validate_corpus(rows)
    fingerprint = g.digest(rows)
    folder = root / 'snapshots' / (f'n{len(rows):05d}-' + fingerprint[:12])
    if folder.exists():
        saved = g.read_json(folder / 'manifest.json')
        assert saved['content_sha256'] == fingerprint
        for info in saved['splits'].values():
            assert g.sha(folder / info['file']) == info['sha256']
        return str(folder)
    staging = folder.with_name('.building-' + uuid.uuid4().hex)
    staging.mkdir()
    manifest = {'provider': 'ifm_direct', 'model': MODEL, 'endpoint': ENDPOINT,
                'model_revision': None, 'contract_sha256': g.digest(expected),
                'content_sha256': fingerprint, 'created_at': g.utc(), 'validation': validation,
                'status': 'synthetic_consistency_checked_not_human_validated',
                'episode_contracts': sorted({ep['provenance']['contract_sha256'] for ep in rows}),
                'original_episode_provenance_preserved': True,
                'combined_corpus_audit_required_before_training': True, 'splits': {}}
    for split in ('train', 'validation', 'final_test'):
        selected = [ep for ep in rows if ep['split'] == split]
        path = staging / (split + '.jsonl')
        with path.open('w', encoding='utf-8', newline='\n') as stream:
            for ep in selected:
                stream.write(json.dumps(ep, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        manifest['splits'][split] = {'count': len(selected), 'file': path.name, 'sha256': g.sha(path)}
    atomic(staging / 'manifest.json', manifest)
    staging.rename(folder)
    return str(folder)

