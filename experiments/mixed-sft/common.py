"""Experiment paths, immutable inputs, and atomic progress files.

Paths (all optional; defaults are relative to this repository):
  TABCREW_RUN_DIR            where dataset/, sft/, evaluations, logs and receipts are written (default: this folder)
  TABCREW_EARLY705_RUN_DIR   finished early-705 run with dataset/, sft/adapter and grpo/adapter
                             (default: experiments/early-705)
  TABCREW_DEDUP_COLLECTOR_DIR  collector folder whose core.py provides DuplicateIndex (default: data-generation/ifm)
  TABCREW_WORK_DIR           model snapshot and caches (default: <repo>/work, see tabcrew_rl/model.py)
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import re
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
RUN=Path(os.environ.get('TABCREW_RUN_DIR') or ROOT).resolve()
DATA=RUN/'dataset'
OLD=Path(os.environ.get('TABCREW_EARLY705_RUN_DIR') or PROJECT/'experiments'/'early-705').resolve()
DEDUP_COLLECTOR=Path(os.environ.get('TABCREW_DEDUP_COLLECTOR_DIR') or PROJECT/'data-generation'/'ifm').resolve()
sys.path.insert(0,str(PROJECT))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
                  WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')

def utc(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def canonical_family(value): return re.sub(r'^(ifm|deepseek|local|qwen)-','',value)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path): return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line]
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)
def jsonl(path,values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        for value in values:stream.write(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n')
    return {'file':path.name,'count':len(values),'sha256':sha(path)}
def load_split(root,split):
    manifest=read(Path(root)/'manifest.json');entry=manifest['splits'][split]
    path=Path(root)/entry['file']
    if sha(path)!=entry['sha256']:raise ValueError('Frozen split changed: '+split)
    values=rows(path)
    if len(values)!=entry['count'] or any(ep['split']!=split for ep in values):raise ValueError('Split mismatch')
    return values
