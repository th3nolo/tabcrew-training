"""Exact-host IFM client; only explicit HTTP429 retries after a shared cooldown."""
import json
import copy
import math
import os
from pathlib import Path
import re
import time
import threading
from functools import wraps
from email.utils import parsedate_to_datetime
import urllib.error
import urllib.request
import uuid
import core as g
from files import atomic, append, read_text_retry
from framing import parse_result

ENDPOINT = 'https://api.ifm.ai/v1/chat/completions'
MODEL = 'IFM/K2-Horizon-375B-A23B'
MAX_TOKENS = 4096
MAX_LIFETIME_TOKENS = 50000000
MAX_ERROR_BYTES = 65536
MIN_START_INTERVAL = 1.0
FINAL_CONTENT_FAILURES = frozenset({'missing_final_content', 'missing_or_ambiguous_final_boundary',
                                    'final_answer_not_json_records', 'unknown_case_id'})


class StopCollection(RuntimeError):
    pass


class _RetryHTTP429(Exception):
    """The failed attempt is durably held and the logical request may retry."""


def synchronized(method):
    @wraps(method)
    def guarded(self,*args,**kwargs):
        with self.state_lock:
            return method(self,*args,**kwargs)
    return guarded


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise StopCollection('provider_redirect_refused')


def reset_after_seconds(value, now=None):
    """Interpret a reset deadline, without treating rate limits as account balance."""
    if not isinstance(value,str) or not value.strip():
        return None
    value=value.strip()
    now=time.time() if now is None else now
    duration=re.fullmatch(r'(?:(\d+(?:\.\d+)?)d)?(?:(\d+(?:\.\d+)?)h)?'
                          r'(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?'
                          r'(?:(\d+(?:\.\d+)?)ms)?',value)
    if duration and any(part is not None for part in duration.groups()):
        seconds=sum(float(part or 0)*scale for part,scale in zip(duration.groups(),(86400,3600,60,1,.001)))
        return seconds if math.isfinite(seconds) else None
    if re.fullmatch(r'\d+(?:\.\d+)?',value):
        number=float(value)
        if not math.isfinite(number): return None
        if number>=1000000000000: number/=1000
        return max(0,number-now) if number>=1000000000 else number
    try:
        return max(0,parsedate_to_datetime(value).timestamp()-now)
    except (TypeError,ValueError,OverflowError):
        return None


def redact_error_text(text,key):
    def redact(value):
        if key:
            value=value.replace(key,'[REDACTED]')
        value=re.sub(r'(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+','Bearer [REDACTED]',value)
        return re.sub(r'\b(?:IFM-v1[_-][A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]{8,})','[REDACTED]',value)
    def clean(value):
        if isinstance(value,str): return redact(value)
        if isinstance(value,list): return [clean(item) for item in value]
        if isinstance(value,dict):
            sensitive={'authorization','api_key','apikey','access_token','refresh_token','secret','token'}
            return {redact(name):'[REDACTED]' if name.casefold() in sensitive else clean(item)
                    for name,item in value.items()}
        return value
    # Decode valid JSON first so escaped credential characters cannot evade redaction.
    try:
        return json.dumps(clean(json.loads(text)),ensure_ascii=False)
    except (ValueError,TypeError,RecursionError):
        return redact(text)


def read_rows(path):
    if not path.exists():
        return []
    # Partial or corrupt ledger lines require inspection; never discard them.
    return [json.loads(line) for line in read_text_retry(path,encoding='utf-8').splitlines()]


def uncertainty_holds_from_origin(origin):
    """Keep a pinned, unresolved 429 reservation separate from measured use."""
    holds=origin.get('uncertainty_holds',[])
    if not isinstance(holds,list):
        raise StopCollection('invalid_uncertainty_holds')
    required={'call_id','tokens','source_ledger','source_ledger_sha256',
              'response_error','response_error_sha256','reason'}
    seen,result=set(),[]
    try:
        sources={Path(source['ledger']).resolve():source['sha256'] for source in origin['sources']}
        for hold in holds:
            if (not isinstance(hold,dict) or set(hold)!=required
                    or any(not isinstance(hold[key],str) or not hold[key].strip()
                           for key in required-{'tokens'})
                    or type(hold['tokens']) is not int or hold['tokens']<=0
                    or hold['call_id'] in seen):
                raise StopCollection('invalid_uncertainty_hold')
            ident=hold['call_id']
            ledger=Path(hold['source_ledger']).resolve()
            error=Path(hold['response_error']).resolve()
            if sources.get(ledger)!=hold['source_ledger_sha256']:
                raise StopCollection('uncertainty_hold_source_mismatch')
            if (g.sha(ledger)!=hold['source_ledger_sha256']
                    or g.sha(error)!=hold['response_error_sha256']):
                raise StopCollection('uncertainty_hold_evidence_changed')
            matching=[row for row in read_rows(ledger) if row.get('id')==ident]
            if (len(matching)!=1 or matching[0].get('type')!='reserve'
                    or type(matching[0].get('tokens')) is not int
                    or matching[0]['tokens']!=hold['tokens']):
                raise StopCollection('uncertainty_hold_not_unresolved_reservation')
            diagnostic=g.read_json(error)
            if (diagnostic.get('call_id')!=ident or type(diagnostic.get('http_status')) is not int
                    or diagnostic['http_status']!=429
                    or diagnostic.get('usage_status')!='unknown_reservation_preserved'):
                raise StopCollection('uncertainty_hold_not_matching_http429')
            seen.add(ident)
            result.append(copy.deepcopy(hold))
    except (KeyError,TypeError,ValueError,OSError,AttributeError) as exc:
        raise StopCollection('invalid_uncertainty_hold_evidence') from exc
    return result


class Ledger:
    def __init__(self, root, token_budget, state_lock=None):
        self.state_lock=state_lock if state_lock is not None else threading.RLock()
        if type(token_budget) is not int or not 1 <= token_budget <= MAX_LIFETIME_TOKENS:
            raise StopCollection('invalid_lifetime_token_budget')
        self.path = root / 'quota.jsonl'
        self.budget = token_budget
        self.pending = {}
        self._reservation_hashes = {}
        self.settled = {}
        self.local_uncertainty_holds = {}
        self.uncertainty_holds = []
        self.uncertainty_reserved_tokens = 0
        self._uncertainty_call_ids = set()
        self.usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        origin = root / 'budget-origin.json'
        if origin.exists():
            inherited = g.read_json(origin)
            for source in inherited['sources']:
                if g.sha(source['ledger']) != source['sha256']:
                    raise StopCollection('inherited_quota_ledger_changed')
            self.usage.update(inherited['usage'])
            self.uncertainty_holds=uncertainty_holds_from_origin(inherited)
            self.uncertainty_reserved_tokens=sum(hold['tokens'] for hold in self.uncertainty_holds)
            self._uncertainty_call_ids={hold['call_id'] for hold in self.uncertainty_holds}
        for row in read_rows(self.path):
            self.apply(row)
        if self.pending:
            raise StopCollection('unresolved_request_inspect_receipt_before_resume')

    @synchronized
    def apply(self, row):
        ident = row['id']
        if row['type'] == 'reserve':
            if (ident in self.pending or ident in self.settled or ident in self._uncertainty_call_ids
                    or type(row['tokens']) is not int or row['tokens'] <= 0):
                raise StopCollection('invalid_reservation')
            self.pending[ident] = row['tokens']
            self._reservation_hashes[ident] = row.get('request_sha256')
        elif row['type'] == 'settle':
            if ident not in self.pending or ident in self.settled:
                raise StopCollection('invalid_settlement')
            counts = row['usage']
            if any(type(counts.get(k)) is not int or counts[k] < 0 for k in self.usage):
                raise StopCollection('invalid_usage')
            if counts['total_tokens'] != counts['prompt_tokens'] + counts['completion_tokens']:
                raise StopCollection('inconsistent_usage')
            reserved = self.pending.pop(ident)
            self._reservation_hashes.pop(ident,None)
            for k in self.usage:
                self.usage[k] += counts[k]
            self.settled[ident] = row
            if counts['total_tokens'] > reserved:
                raise StopCollection('provider_usage_exceeded_reservation')
        elif row['type'] == 'http429_hold':
            hold = self._checked_http429_hold(row)
            self.pending.pop(ident)
            self._reservation_hashes.pop(ident,None)
            self.local_uncertainty_holds[ident] = hold
            self._uncertainty_call_ids.add(ident)
            self.uncertainty_reserved_tokens += row['tokens']
        else:
            raise StopCollection('unknown_quota_event')

    def _checked_http429_hold(self, row):
        required={'type','id','tokens','at','request_sha256','response_error','response_error_sha256'}
        ident=row.get('id')
        if (set(row)!=required or not isinstance(ident,str) or not re.fullmatch(r'[a-f0-9]{32}',ident)
                or ident not in self.pending or type(row['tokens']) is not int
                or row['tokens']!=self.pending[ident]
                or not isinstance(row['request_sha256'],str) or not re.fullmatch(r'[a-f0-9]{64}',row['request_sha256'])
                or row['request_sha256']!=self._reservation_hashes.get(ident)
                or row['response_error']!=f'responses/{ident}.http-error.json'):
            raise StopCollection('invalid_local_http429_hold')
        path=self.path.parent/row['response_error']
        if g.sha(path)!=row['response_error_sha256']:
            raise StopCollection('http429_hold_evidence_changed')
        error=g.read_json(path)
        until=error.get('retry_not_before_unix')
        if (error.get('call_id')!=ident or type(error.get('http_status')) is not int
                or error['http_status']!=429 or error.get('request_sha256')!=row['request_sha256']
                or error.get('usage_status')!='unknown_reservation_preserved'
                or type(until) not in (int,float) or not math.isfinite(until)):
            raise StopCollection('http429_hold_receipt_mismatch')
        return {'call_id':ident,'tokens':row['tokens'],'request_sha256':row['request_sha256'],
                'response_error':row['response_error'],'response_error_sha256':row['response_error_sha256'],
                'retry_not_before_unix':until,'scope':'current_revision_unverified_http429_usage'}

    @synchronized
    def hold_http429(self, ident, request_hash):
        path=self.path.parent/'responses'/(ident+'.http-error.json')
        event={'type':'http429_hold','id':ident,'tokens':self.pending.get(ident),'at':g.utc(),
               'request_sha256':request_hash,'response_error':f'responses/{ident}.http-error.json',
               'response_error_sha256':g.sha(path)}
        self._checked_http429_hold(event)
        append(self.path,event)
        self.apply(event)
        return copy.deepcopy(self.local_uncertainty_holds[ident])

    @synchronized
    def reserve(self, ident, tokens, metadata):
        if (ident in self.pending or ident in self.settled or ident in self._uncertainty_call_ids
                or type(tokens) is not int or tokens<=0):
            raise StopCollection('invalid_reservation')
        if self.usage['total_tokens'] + self.uncertainty_reserved_tokens + sum(self.pending.values()) + tokens > self.budget:
            raise StopCollection('token_budget_cap')
        event = {**metadata, 'type': 'reserve', 'id': ident, 'tokens': tokens, 'at': g.utc()}
        append(self.path, event)
        self.apply(event)

    @synchronized
    def settle(self, ident, counts):
        if ident not in self.pending or ident in self.settled:
            raise StopCollection('invalid_settlement')
        event = {'type': 'settle', 'id': ident, 'usage': counts, 'at': g.utc()}
        # Validate before committing; keep unknown usage unresolved on disk.
        if any(type(counts.get(k)) is not int or counts[k] < 0 for k in self.usage):
            raise StopCollection('invalid_usage')
        if counts['total_tokens'] != counts['prompt_tokens'] + counts['completion_tokens']:
            raise StopCollection('inconsistent_usage')
        append(self.path, event)
        self.apply(event)

    @synchronized
    def snapshot(self):
        return copy.deepcopy({'usage':self.usage,'pending':self.pending,'settled':self.settled,'budget':self.budget,
            'local_uncertainty_holds':self.local_uncertainty_holds,
            'uncertainty_holds':self.uncertainty_holds,'uncertainty_reserved_tokens':self.uncertainty_reserved_tokens})


def payload_for(turns, schema, seed, role, max_tokens=MAX_TOKENS):
    if type(max_tokens) is not int or not 256 <= max_tokens <= 32768:
        raise StopCollection('invalid_output_budget')
    effort = schema.get('effort', 'low')
    if effort not in ('low','medium','high'):
        raise StopCollection('unsupported_reasoning_effort')
    payload = {'model':MODEL, 'messages':turns, 'stream':False, 'max_tokens':max_tokens,
               'temperature':.6 if role.startswith('author') else .2, 'seed':seed,
               'chat_template_kwargs':{'reasoning_effort':effort}}
    # JSONL is multiple independent JSON texts; JSON-object mode is incompatible.
    if schema['format'] == 'json_schema':
        payload['response_format']={'type':'json_schema','json_schema':{'name':role,'strict':True,'schema':schema['schema']}}
    elif schema['format'] == 'json_object':
        payload['response_format']={'type':'json_object'}
    elif schema['format'] != 'jsonl':
        raise StopCollection('unsupported_output_format')
    return payload


class Client:
    def __init__(self, root, budget, run_id, stop_check, key=None, opener=None):
        self.root, self.run_id, self.stop_check = root, run_id, stop_check
        self.state_lock=threading.RLock()
        self.ledger = Ledger(root, budget, self.state_lock)
        self.key = key if key is not None else os.environ.pop('IFM_API_KEY')
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self._provided_opener=opener is not None
        self._thread_local=threading.local()
        self.last_started = 0
        self.quota_headers = {}
        self.provider_remaining = None
        self.provider_token_window = None
        self._provider_token_reset_deadline = None
        self._provider_header_sequence=-1
        self._provider_late_usage=0
        self._next_sequence=0
        self._gate_reason=None
        self._retry_not_before_unix=max((hold['retry_not_before_unix']
            for hold in self.ledger.local_uncertainty_holds.values()),default=0)
        self._retry_not_before=time.monotonic()+max(0,self._retry_not_before_unix-time.time())
        self._http429_this_run=0
        self._rate_limit_retries=0

    @synchronized
    def observe_quota_headers(self,headers,sequence=None,settled_tokens=0):
        expired=self._provider_token_reset_deadline is not None and time.monotonic()>=self._provider_token_reset_deadline
        if sequence is not None and sequence<self._provider_header_sequence and not expired:
            # A later-started call already supplied our header observation. Its
            # count may exclude this overlapping call: debit actual settled use,
            # and never increase headroom from an older response arriving late.
            self._provider_late_usage+=settled_tokens
            if self.provider_token_window is not None:
                self.provider_token_window['late_overlapping_usage_tokens']=self._provider_late_usage
            return
        if sequence is not None:
            self._provider_header_sequence=sequence
        self._provider_late_usage=0
        self.quota_headers={str(k).lower():str(v) for k,v in headers.items()}
        remaining=self.quota_headers.get('x-ratelimit-remaining-tokens')
        self.provider_remaining=int(remaining) if isinstance(remaining,str) and remaining.isdigit() else None
        reset=self.quota_headers.get('x-ratelimit-reset-tokens')
        after=reset_after_seconds(reset)
        self._provider_token_reset_deadline=time.monotonic()+after if after is not None else None
        self.provider_token_window={'scope':'provider_token_rate_limit_window_not_account_balance',
            'observed_at':g.utc(),'remaining_tokens':self.provider_remaining,
            'limit_tokens_header':self.quota_headers.get('x-ratelimit-limit-tokens'),
            'reset_tokens_header':reset,'reset_after_seconds':after,
            'reset_at_unix':time.time()+after if after is not None else None,
            'late_overlapping_usage_tokens':0,
            'account_daily_balance_verified':False}

    @synchronized
    def available_provider_tokens(self):
        # A prior window's remaining count is stale after its reported reset.
        # A fresh success may refresh it; HTTP failures still stop without retry.
        if self._provider_token_reset_deadline is not None and time.monotonic()>=self._provider_token_reset_deadline:
            return None
        if self.provider_remaining is None:
            return None
        return max(0,self.provider_remaining-self._provider_late_usage-sum(self.ledger.pending.values()))

    @synchronized
    def snapshot(self):
        retry_wait=max(0,self._retry_not_before-time.monotonic())
        return {**self.ledger.snapshot(),'quota_headers':copy.deepcopy(self.quota_headers),
            'provider_token_window':copy.deepcopy(self.provider_token_window),
            'requests_started_this_run':self._next_sequence,
            'available_provider_tokens':self.available_provider_tokens(),
            'gate_closed':self._gate_reason is not None,'gate_reason':self._gate_reason,
            'retry_after_remaining_seconds':retry_wait,'retry_not_before_unix':self._retry_not_before_unix,
            'rate_limit_retries':self._rate_limit_retries,
            'http429_attempts_this_run':self._http429_this_run,
            'accepting_new_calls':self._gate_reason is None and retry_wait<=0}

    @synchronized
    def close_gate(self,reason):
        if self._gate_reason is None:
            self._gate_reason=reason

    def _check_gate(self):
        if self._gate_reason is not None:
            raise StopCollection('client_gate_closed_'+self._gate_reason)

    def _opener(self):
        if self._provided_opener:
            return self.opener
        if not hasattr(self._thread_local,'opener'):
            self._thread_local.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
        return self._thread_local.opener

    def chat(self, turns, schema, seed, role, candidate, checkpoint, max_tokens=MAX_TOKENS):
        while True:
            try:
                return self._chat(turns,schema,seed,role,candidate,checkpoint,max_tokens)
            except _RetryHTTP429:
                continue
            except Exception as exc:
                self.close_gate(str(exc) if isinstance(exc,StopCollection) else 'internal_'+type(exc).__name__)
                raise

    def _archive_http429(self, candidate, role, checkpoint, hold):
        call=candidate[role+'_call']
        if call!={'id':hold['call_id'],'request_sha256':hold['request_sha256']}:
            raise StopCollection('http429_checkpoint_request_changed')
        previous=candidate.setdefault('http429_attempts',[])
        matching=[item for item in previous if item['call_id']==hold['call_id']]
        if matching and matching!=[hold]:
            raise StopCollection('http429_attempt_history_changed')
        if not matching: previous.append(copy.deepcopy(hold))
        candidate.pop(role+'_call')
        checkpoint(candidate)

    def _chat(self, turns, schema, seed, role, candidate, checkpoint, max_tokens):
        payload = payload_for(turns, schema, seed, role, max_tokens)
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        request_hash = g.digest(payload)
        reserved = len(body) + max_tokens + 2048
        # Claim starts atomically, but release the lock for the HTTP round trip.
        # Queued calls are not billed/reserved until they acquire a start slot.
        while True:
            with self.state_lock:
                try:
                    call=candidate.get(role+'_call')
                    if call:
                        if call['request_sha256']!=request_hash:
                            raise StopCollection('checkpoint_request_changed')
                        if call['id'] in self.ledger.local_uncertainty_holds:
                            self._archive_http429(candidate,role,checkpoint,self.ledger.local_uncertainty_holds[call['id']])
                        if call['id'] in self.ledger.pending:
                            raise StopCollection('request_already_in_flight')
                        if call['id'] in self.ledger.settled:
                            path=self.root/'responses'/(call['id']+'.json')
                            if not path.exists(): raise StopCollection('settled_response_missing')
                            saved=g.read_json(path)
                            if saved['receipt']['request_sha256']!=request_hash or saved['receipt']['response_sha256']!=g.digest(saved['result']):
                                raise StopCollection('cached_response_integrity')
                            if saved['result'].get('model')!=MODEL: raise StopCollection('provider_model_changed')
                            return self.parse(saved['result'],saved['receipt'],schema,role)
                    self._check_gate()
                    self.stop_check()
                    wait=max(MIN_START_INTERVAL-(time.monotonic()-self.last_started),
                             self._retry_not_before-time.monotonic())
                    if wait<=0:
                        available=self.available_provider_tokens()
                        if available is not None and reserved>available:
                            raise StopCollection('provider_token_rate_limit_low')
                        ident=uuid.uuid4().hex
                        candidate[role+'_call']={'id':ident,'request_sha256':request_hash}
                        checkpoint(candidate)
                        self.ledger.reserve(ident,reserved,{'run_id':self.run_id,'role':role,
                            'spec_id':candidate['spec']['id'],'request_sha256':request_hash})
                        request=urllib.request.Request(ENDPOINT,data=body,headers={
                            'Authorization':'Bearer '+self.key,'Content-Type':'application/json','X-Session-ID':self.run_id})
                        self._next_sequence+=1
                        if candidate.get('http429_attempts'):
                            self._rate_limit_retries+=1
                        sequence=self._next_sequence
                        call_started=time.monotonic()
                        self.last_started=call_started
                        started_at=g.utc()
                        break
                except Exception as exc:
                    self.close_gate(str(exc) if isinstance(exc,StopCollection) else 'internal_'+type(exc).__name__)
                    raise
            time.sleep(min(.1,max(.001,wait)))
        try:
            with self._opener().open(request, timeout=180) as response:
                response_headers={k.lower():redact_error_text(v,self.key) for k,v in response.headers.items()
                    if k.lower().startswith('x-ratelimit-') or k.lower()=='retry-after'}
                raw = response.read(4 * 1024 * 1024 + 1)
                if len(raw) > 4 * 1024 * 1024:
                    raise StopCollection('oversized_provider_response')
                result = json.loads(raw.decode('utf-8').replace(self.key, '[REDACTED]'),
                                    object_pairs_hook=g.unique_object)
        except urllib.error.HTTPError as exc:
            headers = {k.lower(): redact_error_text(v,self.key) for k, v in exc.headers.items()
                       if k.lower().startswith('x-ratelimit-') or k.lower() == 'retry-after'}
            if exc.code==429:
                delay=max(MIN_START_INTERVAL,reset_after_seconds(headers.get('retry-after')) or MIN_START_INTERVAL)
                with self.state_lock:
                    self._retry_not_before=max(self._retry_not_before,time.monotonic()+delay)
                    self._retry_not_before_unix=max(self._retry_not_before_unix,time.time()+delay)
                    retry_until=self._retry_not_before_unix
            else:
                self.close_gate('provider_http_'+str(exc.code)+'_inspect_before_resume')
            self.observe_quota_headers(headers,sequence)
            diagnostic={'at':g.utc(),'call_id':ident,'http_status':exc.code,'quota_headers':headers,
                'retry':'none; inspect before resume','usage_status':'unknown_reservation_preserved',
                'body_limit_bytes':MAX_ERROR_BYTES}
            if exc.code==429:
                diagnostic.update(request_sha256=request_hash,retry='shared_retry_after_cooldown',
                                  retry_not_before_unix=retry_until)
            try:
                error_body=exc.read(MAX_ERROR_BYTES+1)
                diagnostic.update(body_utf8=redact_error_text(error_body[:MAX_ERROR_BYTES].decode('utf-8','replace'),self.key),
                                  body_truncated=len(error_body)>MAX_ERROR_BYTES)
            except (OSError,ValueError) as read_error:
                diagnostic.update(body_read_error=type(read_error).__name__)
            with self.state_lock:
                atomic(self.root/'responses'/(ident+'.http-error.json'),diagnostic)
                atomic(self.root/'last-provider-error.json',diagnostic)
                if exc.code==429:
                    hold=self.ledger.hold_http429(ident,request_hash)
                    self._http429_this_run+=1
                    self._archive_http429(candidate,role,checkpoint,hold)
            if exc.code==429:
                raise _RetryHTTP429() from None
            raise StopCollection('provider_http_' + str(exc.code) + '_inspect_before_resume') from None
        except (urllib.error.URLError, TimeoutError, OSError, UnicodeError, ValueError):
            self.close_gate('provider_transport_or_response_error_inspect_before_resume')
            raise StopCollection('provider_transport_or_response_error_inspect_before_resume') from None
        usage = result.get('usage') or {}
        counts = {k: usage.get(k) for k in self.ledger.usage}
        receipt = {'at': g.utc(), 'started_at': started_at, 'run_id': self.run_id,
                   'call_id': ident, 'role': role, 'provider': 'ifm_direct', 'endpoint': ENDPOINT,
                   'requested_model': MODEL, 'model': result.get('model'),
                   'response_id': result.get('id'), 'system_fingerprint': result.get('system_fingerprint'),
                   'request_sha256': request_hash, 'response_sha256': g.digest(result),
                   'seconds': time.monotonic() - call_started, 'usage': usage,
                   'seed': seed, 'max_tokens': max_tokens, 'quota_headers': response_headers,
                   'request_sequence':sequence,
                   'spec_id': candidate['spec']['id'], 'arm': candidate.get('arm'),
                   'requested_batch_size': candidate.get('batch_size'),
                   'item_ids': candidate.get(role + '_item_ids', [])}
        with self.state_lock:
            try:
                # In-flight successes still settle even when a different call
                # has closed the gate. Immutable receipts precede publication.
                atomic(self.root/'responses'/(ident+'.json'),{'result':result,'receipt':receipt,'payload':payload})
                self.ledger.settle(ident,counts)
                append(self.root/'requests.jsonl',receipt)
                self.observe_quota_headers(response_headers,sequence,counts['total_tokens'])
                if result.get('model')!=MODEL: raise StopCollection('provider_model_changed')
                return self.parse(result,receipt,schema,role)
            except Exception as exc:
                self.close_gate(str(exc) if isinstance(exc,StopCollection) else 'internal_'+type(exc).__name__)
                raise

    @staticmethod
    def parse(result, receipt, schema, role):
        if 'records' not in schema:
            return parse_result(result, schema), receipt
        expected = schema['records']
        # Descriptor failures are programming/contract errors, never model rejects.
        if (role not in ('author_records','verifier_records') or not isinstance(expected,dict)
                or not expected or any(not isinstance(slot,str) or not slot for slot in expected)
                or any(not isinstance(value,dict) or value.get('type')!='object' for value in expected.values())):
            raise ValueError('invalid_record_descriptor')
        if schema['format']=='json_schema':
            if (len(expected)!=1 or schema.get('identity_binding')!='single_request'
                    or schema.get('schema')!=next(iter(expected.values()))):
                raise ValueError('invalid_single_record_descriptor')
        elif schema['format']!='jsonl':
            raise ValueError('unsupported_record_format')
        try:
            parsed = parse_result(result, schema)
        except ValueError as exc:
            code = str(exc)
            if code not in FINAL_CONTENT_FAILURES:
                raise
            parsed = {'records':{}, 'errors':{slot:'record_'+code for slot in expected},
                      'transport':{'finish_reason':result['choices'][0].get('finish_reason'),
                          'response_truncated':result['choices'][0].get('finish_reason')=='length',
                          'response_format':schema['format'], 'final_content_error':code,
                          'accepted_framing_rule':'no_candidate_extracted_from_reasoning_or_metadata'}}
        if not parsed['records']:
            # Live chat settles and saves before this function; replay must point
            # to that same immutable provider response and original receipt.
            if (receipt.get('response_sha256')!=g.digest(result)
                    or receipt.get('model')!=result.get('model') or result.get('model')!=MODEL
                    or receipt.get('role')!=role or set(parsed['errors'])!=set(expected)):
                raise StopCollection('model_output_failure_receipt_integrity')
            parsed['transport']['model_output_failure'] = True
        return parsed, receipt
