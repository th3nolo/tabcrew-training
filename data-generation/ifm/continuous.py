"""Continuous IFM generation with bounded batches and durable, audited admission."""
import argparse
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import math
import os
from pathlib import Path
import signal
import threading
import time
import uuid
import collector as c
import core as g
import pipeline as p
from client import Client, Ledger, StopCollection, read_rows, MIN_START_INTERVAL
from controls import lock
from files import atomic
import files as local_files

CONFIG = {'author_cases':3, 'author_tabs':40, 'verifier_cases':1, 'verifier_tabs':30}
AUTHOR_FAILURE_LIMIT = 3


class DynamicExecutor:
    """Create one HTTP worker per paced submission, without a numeric pool cap."""
    def __init__(self):
        self.threads = []
        self.sequence = 0

    def __enter__(self):
        return self

    def submit(self, function, **kwargs):
        self.threads = [thread for thread in self.threads if thread.is_alive()]
        future = Future()
        def execute():
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(function(**kwargs))
                except BaseException as exc:
                    future.set_exception(exc)
        self.sequence += 1
        thread = threading.Thread(target=execute, name=f'ifm-request-{self.sequence}')
        thread.start()
        self.threads.append(thread)
        return future

    def __exit__(self, *_):
        for thread in self.threads:
            thread.join()


def wait_for_requests(futures, timeout):
    if futures:
        return wait(futures, timeout=timeout, return_when=FIRST_COMPLETED)
    time.sleep(timeout)
    return set(), set()


def attempted_ids(root):
    paths = set(root.parent.glob('ifm-synthetic-5000-v*/batches/*.json'))
    paths.update((root / 'batches').glob('*.json'))
    return {spec['id'] for path in paths for spec in g.read_json(path)['specs']}


def author_failure_streak(root):
    """Count unique settled author outputs, including a saved stage on resume."""
    calls = {}
    for cp in c.all_batches(root):
        if cp['arm']!='production': continue
        for stage in cp['stages'].values():
            if stage['kind']!='author' or not stage.get('completed'): continue
            receipt = stage['receipt']
            entry = (receipt.get('started_at',receipt['at']), receipt['response_sha256'],
                     not stage['reply']['records'] and stage['reply']['transport'].get('model_output_failure') is True)
            if receipt['call_id'] in calls and calls[receipt['call_id']]!=entry:
                raise StopCollection('duplicate_author_receipt_changed')
            calls[receipt['call_id']] = entry
    streak = 0
    for _,_,unusable in sorted(calls.values()):
        streak = streak+1 if unusable else 0
    return streak


def next_batch(specs, attempted, max_cases=3, max_tabs=40):
    """Balance canonical split/audience/language shares without altering any spec."""
    remaining = [s for s in specs if s['id'] not in attempted]
    if not remaining:
        return []
    split_totals = Counter(s['split'] for s in specs)
    split_seen = Counter(s['split'] for s in specs if s['id'] in attempted)
    split = min(dict.fromkeys(s['split'] for s in remaining),
                key=lambda name:(split_seen[name] + 1) / split_totals[name])
    pool = [s for s in remaining if s['split'] == split]
    category = lambda s:(s['audience'], s['locale'])
    totals = Counter(category(s) for s in specs if s['split'] == split)
    seen = Counter(category(s) for s in specs if s['split'] == split and s['id'] in attempted)
    chosen, tabs = [], 0
    while len(chosen) < max_cases:
        eligible = [s for s in pool if tabs + s['tab_count'] <= max_tabs]
        if not eligible:
            break
        spec = min(eligible, key=lambda s:(seen[category(s)] + 1) / totals[category(s)])
        chosen.append(spec)
        pool.remove(spec)
        seen[category(spec)] += 1
        tabs += spec['tab_count']
    if not chosen:
        raise StopCollection('canonical_session_exceeds_batch_capacity')
    return chosen


def run(args, root=g.OUT, client_factory=Client, *, max_batches=None, interrupt_event=None,
        dispatch_clock=None, request_wait=None):
    if not 0 < args.max_hours <= 72 or not 1 <= args.target <= 5000:
        raise StopCollection('invalid_continuous_limits')
    workers = getattr(args, 'max_in_flight', 1)
    if type(workers) is not int or not 0 <= workers <= 50:
        raise StopCollection('invalid_request_concurrency')
    adaptive = workers == 0
    dispatch_clock = dispatch_clock or time.monotonic
    request_wait = request_wait or wait_for_requests
    expected = c.prepare(root, args)
    budget = expected['maximum_authorized_lifetime_tokens']
    initial_usage = dict(Ledger(root, budget).usage)
    records, index = p.audit_arm(root, 'production', expected)
    admission = p.AdmissionContext(root, 'production', records, index)
    baseline = len(records)
    specs, attempted = g.specifications(), attempted_ids(root)
    pending = [{k:cp[k] for k in ('id','arm','specs')} for cp in c.all_batches(root)
               if cp['arm'] == 'production' and not cp.get('completed')]
    interrupted = interrupt_event or threading.Event()
    started, clock_started, run_id = time.time(), time.monotonic(), str(uuid.uuid4())
    deadline = started + args.max_hours * 3600
    config_path=root/'launch-config.json'
    if config_path.exists():
        original_deadline=g.read_json(config_path).get('deadline_unix')
        if original_deadline is not None:
            if not isinstance(original_deadline,(int,float)) or not math.isfinite(original_deadline):
                raise StopCollection('invalid_original_deadline')
            deadline=min(deadline,original_deadline)
    info = {'run_id':run_id, 'mode':'continuous', 'started_unix':started, 'started_at':g.utc(),
            'model_ready_unix':started, 'deadline_unix':deadline,
            'max_hours':args.max_hours, 'target':args.target, 'contract_sha256':g.digest(expected),
            'attempted_ids_at_start_sha256':g.digest(sorted(attempted)),
            'scheduler':'ready_responses_before_refill_incremental_admission_v2',
            'dispatch_mode':'rate_paced_dynamic_threads' if adaptive else 'bounded_thread_pool',
            'configured_concurrency_limit':None if adaptive else workers,
            'report_policy':'full_audit_at_start_and_stop_live_status_between',
            'max_in_flight':workers, 'minimum_request_start_interval_seconds':MIN_START_INTERVAL}
    folder = root / 'runs' / run_id
    folder.mkdir()
    atomic(folder / 'run.json', info)
    state = {'state':'running', 'mode':'continuous', 'run_id':run_id, 'pid':os.getpid(),
             'provider':'ifm_direct', 'execution':'ifm_direct_api', 'teacher':g.ALIAS,
             'record_protocol':'jsonl_author_request_bound_single_reviewer_v17',
             'curriculum':g.audience.VERSION, 'target':args.target, 'workers':workers,
             'dispatch_mode':info['dispatch_mode'], 'configured_concurrency_limit':info['configured_concurrency_limit'],
             'batching':expected['batching'], 'token_budget':budget, 'active_candidates':0,
             'token_usage':initial_usage, 'token_usage_this_run':{key:0 for key in initial_usage},
             'accepted_examples':baseline, 'accepted_this_run':0, 'rejections_this_run':{},
             'pending_api_calls':0, 'api_calls':0,
             'pricing_status':'price_unverified_bounded_run_authorized', 'max_hours':args.max_hours}
    client, completed_batches, visited = None, 0, set()
    halt_reason, inflight, active_jobs = None, {}, {}
    ready_reviews, ready_authors = deque(), deque()
    next_dispatch = dispatch_clock()
    peak_in_flight = 0
    profile = {'advance_calls':0, 'advance_seconds':0.0, 'advance_max_seconds':0.0,
               'publish_calls':0, 'publish_seconds':0.0, 'publish_max_seconds':0.0,
               'final_report_seconds':0.0,
               'scope':'Advance includes nested status publication; these durations must not be added together.'}

    def stop():
        if halt_reason: raise StopCollection(halt_reason)
        if (root / 'PAUSE').exists(): raise StopCollection('manual_pause')
        if interrupted.is_set(): raise StopCollection('interrupted_checkpoint_saved')
        if time.monotonic() - clock_started >= deadline-started: raise StopCollection('time_limit')
        if len(records) >= args.target: raise StopCollection('target_reached')

    def publish(phase):
        for path in (root / 'accepted').glob('*.json'):
            if path.stem not in records:
                # Admissions update the shared mapping/index before export.
                # Importing an unexpected file here would bypass that index.
                raise StopCollection('accepted_index_out_of_sync_requires_audit')
        current = [g.read_json(root / 'batches' / (ident + '.json')) for ident in visited
                   if (root / 'batches' / (ident + '.json')).exists()]
        rejected = Counter(out['reason'] for cp in current for out in cp.get('outcomes',{}).values()
                           if out['status'] == 'rejected' and out.get('decision_run_id') == run_id)
        elapsed = max(.001, time.time() - started)
        state.update(phase=phase, updated_at=g.utc(), elapsed_seconds=elapsed,
                     accepted_examples=len(records), accepted_this_run=len(records)-baseline,
                     rejections_this_run=dict(rejected), completed_batches_this_run=completed_batches,
                     examples_per_hour_this_run=(len(records)-baseline)*3600/elapsed,
                     splits=dict(Counter(ep['split'] for ep in records.values())),
                     audience_counts=dict(Counter(ep['research_context']['audience'] for ep in records.values())),
                     url_coverage=g.url_coverage(list(records.values())),
                     pending_verification=sum(len(cp.get('pending_verification',[])) for cp in current if not cp.get('completed')),
                     protocol_failures=sum(bool(cp.get('halted')) for cp in current),
                     author_unusable_output_streak=author_failure_streak(root),
                     author_unusable_output_limit=AUTHOR_FAILURE_LIMIT,
                     max_in_flight=workers, peak_in_flight=peak_in_flight,
                     queued_requests=len(ready_reviews)+len(ready_authors), queued_reviews=len(ready_reviews),
                     in_flight_metric_scope='Submitted unfinished requests; pending_api_calls counts reserved API calls.',
                     active_jobs=len(active_jobs), draining=bool(halt_reason and inflight), metrics_stale=False,
                     filesystem_reads=local_files.read_retry_snapshot(), scheduler_profile=dict(profile))
        if client:
            with client.state_lock:
                snapshot = client.snapshot()
                this_run_calls=sum(row.get('run_id')==run_id for row in read_rows(root/'requests.jsonl'))
            state.update(token_usage=snapshot['usage'],
                         uncertainty_reserved_tokens=snapshot.get('uncertainty_reserved_tokens',0),
                         token_usage_this_run={k:snapshot['usage'][k]-initial_usage[k] for k in initial_usage},
                         inherited_token_usage=expected['inherited_usage'], pending_api_calls=len(snapshot['pending']),
                         request_starts_this_run=snapshot.get('requests_started_this_run',0),
                         retry_after_remaining_seconds=snapshot.get('retry_after_remaining_seconds',0),
                         rate_limit_retries=snapshot.get('rate_limit_retries',0),
                         http429_attempts_this_run=snapshot.get('http429_attempts_this_run',0),
                         request_starts_per_minute_this_run=snapshot.get('requests_started_this_run',0)*60/elapsed,
                         api_calls=this_run_calls,
                         cumulative_api_calls=len(snapshot['settled']), quota_headers=snapshot['quota_headers'])
            state['provider_token_window'] = snapshot['provider_token_window']
        atomic(root / 'status.json', state)

    def safe_publish(phase):
        """A telemetry read failure must not abandon completed HTTP responses."""
        nonlocal halt_reason, failed
        before = time.monotonic()
        try:
            # Checkpoints replace complete JSON files atomically. Only the
            # client snapshot and append-log read need its short state lock;
            # scanning checkpoint/coverage files must not block request starts.
            publish(phase)
        except Exception as exc:
            halt_reason = halt_reason or 'internal_'+type(exc).__name__
            failed = True
            state['error']=halt_reason
            if client: client.close_gate(halt_reason)
            state['error_report_saved']=c.error_file(root,exc)
            # Do not reread the file that failed just to tell the dashboard that
            # collection is draining or stopped. Known usage remains exact;
            # admission/coverage metrics are explicitly marked stale.
            state.update(phase=phase,updated_at=g.utc(),metrics_stale=True,
                         telemetry_error=type(exc).__name__,
                         draining=bool(inflight),active_jobs=len(active_jobs))
            if client:
                snapshot=client.snapshot()
                state.update(pending_api_calls=len(snapshot['pending']),token_usage=snapshot['usage'],
                             uncertainty_reserved_tokens=snapshot.get('uncertainty_reserved_tokens',0),
                             token_usage_this_run={k:snapshot['usage'][k]-initial_usage[k] for k in initial_usage})
            if phase=='stopped': state.update(state='failed',active_candidates=0,active_jobs=0,draining=False)
            try:
                atomic(root/'status.json',state)
            except OSError:
                state['status_write_failed']=True
        finally:
            elapsed = time.monotonic()-before
            profile['publish_calls']+=1
            profile['publish_seconds']+=elapsed
            profile['publish_max_seconds']=max(profile['publish_max_seconds'],elapsed)

    reason, failed = 'schedule_exhausted', False
    try:
        stop()
        if author_failure_streak(root)>=AUTHOR_FAILURE_LIMIT:
            raise StopCollection('consecutive_author_output_failures')
        client = client_factory(root, budget, run_id, stop)
        safe_publish('starting')
        executor = DynamicExecutor() if adaptive else ThreadPoolExecutor(max_workers=workers, thread_name_prefix='ifm-request')
        with executor as pool:
            def submit(job, steps, request):
                nonlocal peak_in_flight
                future = pool.submit(client.chat, **request)
                inflight[future] = (job, steps)
                peak_in_flight = max(peak_in_flight, len(inflight))

            def advance_step(job, steps, result=None, error=None):
                nonlocal completed_batches, peak_in_flight
                try:
                    request = steps.throw(error) if error is not None else steps.send(result)
                except StopIteration:
                    active_jobs.pop(job['id'], None)
                    completed_batches += 1
                    safe_publish('batch_completed')
                    return
                # A completed response is committed before this stop check.
                # No new call is queued after a pause, failure or time limit.
                try:
                    stop()
                except StopCollection as exc:
                    try:
                        steps.throw(exc)
                    finally:
                        steps.close()
                    raise
                if adaptive:
                    queue = ready_reviews if request['role'].startswith('verifier') else ready_authors
                    queue.append((job, steps, request))
                else:
                    submit(job, steps, request)

            def advance(job, steps, result=None, error=None):
                before=time.monotonic()
                try:
                    return advance_step(job,steps,result,error)
                finally:
                    elapsed=time.monotonic()-before
                    profile['advance_calls']+=1
                    profile['advance_seconds']+=elapsed
                    profile['advance_max_seconds']=max(profile['advance_max_seconds'],elapsed)

            while True:
                try:
                    stop()
                    if author_failure_streak(root)>=AUTHOR_FAILURE_LIMIT:
                        raise StopCollection('consecutive_author_output_failures')
                except StopCollection as exc:
                    halt_reason = halt_reason or str(exc)
                except Exception as exc:
                    halt_reason,failed=halt_reason or 'internal_'+type(exc).__name__,True
                    client.close_gate(halt_reason)
                    c.error_file(root,exc)

                may_initialize = True
                if adaptive and not halt_reason:
                    retry_wait = client.snapshot().get('retry_after_remaining_seconds',0)
                    if retry_wait>0:
                        # Already-sent calls and their retries remain independent;
                        # do not create idle workers during a provider cooldown.
                        next_dispatch=max(next_dispatch,dispatch_clock()+retry_wait)
                while not halt_reason and (adaptive or len(active_jobs)<workers):
                    # A large capacity must not delay handling completed
                    # responses until every queued job has been initialized.
                    if any(future.done() for future in inflight):
                        break
                    if adaptive and (ready_reviews or ready_authors):
                        break
                    if max_batches is not None and completed_batches+len(active_jobs)>=max_batches:
                        may_initialize = False
                        break
                    # Reserve target room for every unfinished candidate, so
                    # overlapping batches cannot admit beyond the chosen target.
                    reserved_cases = sum(len(j['specs']) for j in active_jobs.values())
                    room = args.target-len(records)-reserved_cases
                    if room<=0:
                        may_initialize = False
                        break
                    # Initialize at most one new author at its dispatch slot;
                    # do not prequeue the curriculum or create idle workers.
                    if adaptive and dispatch_clock()<next_dispatch:
                        break
                    if pending:
                        job = pending.pop(0)
                        if len(job['specs'])>room:
                            pending.insert(0,job)
                            may_initialize = False
                            break
                    else:
                        chosen = next_batch(specs, attempted, min(args.author_cases,room), args.author_tabs)
                        if not chosen:
                            may_initialize = False
                            break
                        job = {'id':'continuous-'+g.digest([s['id'] for s in chosen])[:16],
                               'arm':'production','specs':chosen}
                    visited.add(job['id'])
                    attempted.update(s['id'] for s in job['specs'])
                    active_jobs[job['id']] = job
                    state.update(active_candidates=sum(len(j['specs']) for j in active_jobs.values()),
                                 current_benchmark_arm='production')
                    steps = p.process_job_steps(root,job,expected,index,client,stop,safe_publish,admission=admission)
                    try:
                        advance(job,steps)
                    except StopCollection as exc:
                        halt_reason = halt_reason or str(exc)
                        active_jobs.pop(job['id'],None)
                    except Exception as exc:
                        halt_reason,failed = 'internal_'+type(exc).__name__,True
                        c.error_file(root,exc)
                        active_jobs.pop(job['id'],None)

                if adaptive and halt_reason:
                    client.close_gate(halt_reason)
                    for queue in (ready_reviews, ready_authors):
                        while queue:
                            job, steps, _ = queue.popleft()
                            try:
                                steps.throw(StopCollection(halt_reason))
                            except (StopCollection, StopIteration):
                                pass
                            except Exception as exc:
                                failed = True
                                c.error_file(root,exc)
                            finally:
                                steps.close()
                                active_jobs.pop(job['id'],None)
                if (adaptive and not halt_reason and (ready_reviews or ready_authors)
                        and dispatch_clock()>=next_dispatch
                        and not any(future.done() for future in inflight)):
                    queue = ready_reviews if ready_reviews else ready_authors
                    job, steps, request = queue.popleft()
                    try:
                        stop()
                        submit(job, steps, request)
                        next_dispatch = dispatch_clock()+MIN_START_INTERVAL
                    except Exception as exc:
                        halt_reason = str(exc) if isinstance(exc,StopCollection) else 'internal_'+type(exc).__name__
                        failed = failed or not isinstance(exc,StopCollection)
                        client.close_gate(halt_reason)
                        queue.appendleft((job, steps, request))
                        continue

                queued = bool(ready_reviews or ready_authors)
                if not inflight and not queued and not (adaptive and not halt_reason and may_initialize
                                                         and dispatch_clock()<next_dispatch):
                    reason = halt_reason or ('test_batch_limit' if max_batches is not None
                                              and completed_batches>=max_batches else 'schedule_exhausted')
                    break
                timeout = 1
                if adaptive and not halt_reason:
                    timeout = min(1,max(.001,next_dispatch-dispatch_clock()))
                done,_ = request_wait(inflight, timeout=timeout)
                # Notice every failure before scheduling another request, even
                # if successes and failures complete in the same poll interval.
                for future in done:
                    error = future.exception()
                    if error is not None and not halt_reason:
                        halt_reason = str(error) if isinstance(error,StopCollection) else 'internal_'+type(error).__name__
                        if not isinstance(error,StopCollection):
                            failed=True
                            c.error_file(root,error)
                for future in done:
                    job,steps = inflight.pop(future)
                    try:
                        error=future.exception()
                        advance(job,steps,result=None if error else future.result(),error=error)
                    except StopCollection as exc:
                        halt_reason = halt_reason or str(exc)
                        active_jobs.pop(job['id'],None)
                    except Exception as exc:
                        halt_reason,failed = halt_reason or 'internal_'+type(exc).__name__,True
                        c.error_file(root,exc)
                        active_jobs.pop(job['id'],None)
                safe_publish('draining' if halt_reason else 'requests_in_flight')
    except StopCollection as exc:
        reason = str(exc)
    except KeyboardInterrupt:
        reason = 'interrupted_checkpoint_saved'
    except Exception as exc:
        reason, failed = 'internal_'+type(exc).__name__, True
        c.error_file(root, exc)
    finally:
        info.update(ended_unix=time.time(), ended_at=g.utc(), stop_reason=reason)
        atomic(folder / 'run.json', info)
        report_started=time.monotonic()
        try:
            c.report(root, expected, run_id)
            audit = {'result':'passed'}
        except StopCollection as exc:
            audit = {'result':'blocked','reason':str(exc)}
        except Exception as exc:
            audit, failed = {'result':'failed','error_type':type(exc).__name__}, True
            c.error_file(root, exc)
        finally:
            profile['final_report_seconds']=time.monotonic()-report_started
        active_jobs.clear()
        state.update(state='failed' if failed else 'paused', stop_reason=reason, active_candidates=0)
        safe_publish('stopped')
        summary = {**info, 'accepted_new':state['accepted_this_run'], 'accepted_total':state['accepted_examples'],
                   'rejected':sum(state['rejections_this_run'].values()), 'audit':audit,
                   'completed_batches':completed_batches, 'scheduler_profile':dict(profile),
                   'generation_only_no_student_training':True}
        atomic(root / 'latest-report.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token-budget', type=int, required=True)
    parser.add_argument('--max-hours', type=float, default=72)
    parser.add_argument('--target', type=int, default=5000)
    parser.add_argument('--max-in-flight', type=int, default=0,
                        help='0: rate-paced requests without a fixed concurrency ceiling; positive: bounded pool')
    args = parser.parse_args()
    for key, value in CONFIG.items(): setattr(args, key, value)
    interrupted = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: interrupted.set())
    with lock(g.OUT):
        print(json.dumps(run(args, interrupt_event=interrupted)), flush=True)


if __name__ == '__main__':
    main()
