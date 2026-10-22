"""Durable local records with bounded retries for Windows readers."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time

READ_ATTEMPTS = 6
_read_diagnostics_lock = threading.Lock()
_read_diagnostics = {'permission_retries':0, 'exhausted_permission_errors':0}


def read_retry_snapshot():
    """Return process-local I/O counters without recording paths or file data."""
    with _read_diagnostics_lock:
        return dict(_read_diagnostics)


def _read_with_retry(operation, attempts=READ_ATTEMPTS):
    if type(attempts) is not int or not 1 <= attempts <= READ_ATTEMPTS:
        raise ValueError('invalid_read_retry_attempts')
    for attempt in range(attempts):
        try:
            return operation()
        except PermissionError:
            exhausted = attempt+1 == attempts
            with _read_diagnostics_lock:
                _read_diagnostics['exhausted_permission_errors' if exhausted else 'permission_retries'] += 1
            if exhausted:
                raise
            time.sleep(min(.025*2**attempt,.2))


def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    def read_hash():
        # A read conflict restarts the entire digest, never a partially read hash.
        result=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):
                result.update(chunk)
        return result.hexdigest()
    return _read_with_retry(read_hash)

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


def read_text_retry(path, attempts=READ_ATTEMPTS, *, encoding='utf-8'):
    """Retry only transient file access, retaining the caller's text encoding."""
    return _read_with_retry(lambda:Path(path).read_text(encoding=encoding),attempts)


def read_json_retry(path, attempts=READ_ATTEMPTS, *, encoding='utf-8', object_pairs_hook=None):
    """Retry only brief Windows sharing/access conflicts; fail closed otherwise.

    Six attempts wait at most 0.575 seconds in total. Persistent permission
    failures and malformed JSON still reach the watchdog's stop handler.
    """
    text=read_text_retry(path,attempts,encoding=encoding)
    # Parse once, outside the retry boundary. Corruption and validator failures
    # must reach the caller without being mistaken for a file-sharing conflict.
    return json.loads(text,object_pairs_hook=object_pairs_hook)
