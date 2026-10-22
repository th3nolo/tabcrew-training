"""Durable local records with bounded retries for Windows readers."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            result.update(chunk)
    return result.hexdigest()

def atomic(path,value):
    path=Path(path)
    pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(9):
        try:
            os.replace(pending,path)
            return
        except PermissionError:
            if attempt==8:
                raise
            time.sleep(min(.02*2**attempt,.25))

def append(path,value):
    with Path(path).open('a',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def read_json_retry(path, attempts=6):
    """Retry only brief Windows sharing/access conflicts; fail closed otherwise.

    Six attempts wait at most 0.575 seconds in total. Persistent permission
    failures and malformed JSON still reach the watchdog's stop handler.
    """
    path=Path(path)
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text(encoding='utf-8-sig'))
        except PermissionError:
            if attempt+1==attempts:raise
            time.sleep(min(.025*2**attempt,.2))
