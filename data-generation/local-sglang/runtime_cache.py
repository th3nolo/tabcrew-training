"""Request release of clean unused file cache for this runtime's installed binaries.

No deletion, no writes to dependency files, and no system-wide drop_caches.
The kernel keeps pages that cannot safely be discarded, including mapped pages.
"""
import json
import os
from pathlib import Path
import time

def trim(root,stage):
    # serve.py sets TABCREW_SGLANG_RUNTIME from its --runtime argument.
    PACKAGES=Path(os.environ['TABCREW_SGLANG_RUNTIME'])/'.venv/lib/python3.12/site-packages'
    paths=[]
    for path in PACKAGES.rglob('*'):
        if not path.is_file() or path.is_symlink():
            continue
        if path.stat().st_size<16*1024*1024 or '.so' not in path.name:
            continue
        resolved=path.resolve()
        assert resolved.is_relative_to(PACKAGES.resolve())
        with resolved.open('rb') as stream:
            os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        paths.append(str(resolved))
    with (root/'library-cache-releases.jsonl').open('a') as stream:
        stream.write(json.dumps({'unix':time.time(),'stage':stage,'files':paths})+'\n')


