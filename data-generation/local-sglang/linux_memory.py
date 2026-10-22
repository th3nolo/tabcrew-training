"""Read proportional memory for the exact owned SGLang process group."""
import json
import os
from pathlib import Path
import time


def sample(group, target):
    rows = []
    errors = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            pid = int(proc.name)
            if os.getpgid(pid) != group:
                continue
            fields = {}
            for line in (proc/'smaps_rollup').read_text().splitlines():
                key, _, value = line.partition(':')
                if key in ('Rss', 'Pss', 'Pss_Anon', 'Pss_File', 'Private_Dirty', 'Swap'):
                    fields[key+'_kib'] = int(value.split()[0])
            status = dict(line.split(':', 1) for line in (proc/'status').read_text().splitlines() if ':' in line)
            rows.append({'pid': pid, 'name': status['Name'].strip(), 'threads': int(status['Threads']), **fields})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            errors.append(str(error))
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemAvailable', 'MemFree', 'MemTotal', 'Cached', 'SwapTotal', 'SwapFree', 'AnonPages'):
            mem[key+'_kib'] = int(value.split()[0])
    row = {'unix': time.time(), 'group': group, 'processes': rows, 'errors': errors, 'wsl': mem,
           'group_pss_kib': sum(r.get('Pss_kib', 0) for r in rows),
           'group_rss_kib': sum(r.get('Rss_kib', 0) for r in rows)}
    with target.open('a') as stream:
        stream.write(json.dumps(row)+'\n')
