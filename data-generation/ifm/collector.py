"""Bounded collector with independent batch sizes, provenance and real metrics."""
import argparse
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import os
import time
import traceback
import uuid
import json
from types import SimpleNamespace
import core as g
import batching as b
import pipeline as p
import inherited
from client import Client,Ledger,StopCollection,read_rows,MAX_LIFETIME_TOKENS,MIN_START_INTERVAL
from controls import lock
from files import atomic
from storage import export_snapshot

SOURCES=('collector.py','pipeline.py','batching.py','client.py','framing.py','core.py','audience.py','continuous.py','inherited.py',
         'files.py','storage.py','url_templates.py','url_validation.py','source_catalog.py','controls.py',
         'source-origin.json','budget-origin.json','inherited-dataset.json','launch-config.json','POLICY.md','start.ps1')


def validate_token_budget(value):
    if type(value) is not int or not 1 <= value <= MAX_LIFETIME_TOKENS:
        raise StopCollection('invalid_lifetime_token_budget')
    return value


def prepare(root,args):
    token_budget=validate_token_budget(getattr(args,'token_budget',100000))
    for name in ('responses','batches','runs','accepted','rejected','snapshots','logs'):
        (root/name).mkdir(parents=True,exist_ok=True)
    config={'author_cases':args.author_cases,'author_tabs':args.author_tabs,
            'verifier_cases':args.verifier_cases,'verifier_tabs':args.verifier_tabs}
    expected={'version':g.VERSION,'provider':'ifm_direct','model':g.ALIAS,'batching':config,
        'source_sha256':{name:g.sha(g.OUT/name) for name in SOURCES},
        'project_source_sha256':{name:g.sha(g.PROJECT/'tabcrew_rl'/name) for name in ('reward.py','protocol.py')},
        'specifications_sha256':g.digest(g.specifications()),'pilot_specifications_sha256':g.digest(b.pilot_specs()),
        'format':'JSONL_author_strict_JSON_schema_single_reviewer_JSONL_multi_reviewer_no_repair',
        'acceptance':'individual_schema_language_exact_blind_partition_no_ambiguity_dedup',
        'continuous_admission_verification':'full_start_and_stop_audit_incremental_single_owner_index_between',
        'maximum_authorized_lifetime_tokens':token_budget,'inherited_usage':g.read_json(g.OUT/'budget-origin.json')['usage'],
        'request_concurrency':{'maximum_in_flight':None, 'zero_selects':'rate_paced_no_fixed_concurrency_cap',
                               'selection_recorded_in':'runs/<run_id>/run.json',
                               'minimum_start_interval_seconds':MIN_START_INTERVAL,
                               'admission_owner':'single_scheduler_thread'}}
    path=root/'contract.json'
    if path.exists() and g.read_json(path)!=expected: raise StopCollection('immutable_contract_changed')
    if not path.exists(): atomic(path,expected)
    return expected


def error_file(root,exc):
    try:
        atomic(root/'last-internal-error.json',{'at':g.utc(),'type':type(exc).__name__,
            'errno':getattr(exc,'errno',None),'winerror':getattr(exc,'winerror',None),
            'filename':Path(exc.filename).name if getattr(exc,'filename',None) else None,
            'frames':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)]})
        return True
    except OSError:
        # Error reporting must not abandon in-flight requests and their receipts.
        return False


def all_batches(root):
    return [g.read_json(path) for path in sorted((root/'batches').glob('*.json'))]


def report(root,expected,run_id=None,*,ledger_snapshot=None):
    batches=all_batches(root)
    token_budget=validate_token_budget(expected['maximum_authorized_lifetime_tokens'])
    # A running client can provide a locked snapshot while other calls are
    # unfinished. Reloading a ledger from disk still refuses unknown requests.
    ledger=Ledger(root,token_budget) if ledger_snapshot is None else SimpleNamespace(**ledger_snapshot)
    receipts=[g.read_json(root/'responses'/(ident+'.json'))['receipt'] for ident in ledger.settled]
    # Reused author calls and the failed earlier review still count toward
    # generation efficiency. They are already billed in inherited quota usage.
    origin_path=root/'source-origin.json'
    if origin_path.exists():
        for old in g.read_json(origin_path).get('inherited_generation_responses',[]):
            path=root/old['path']
            if g.sha(path)!=old['sha256']: raise StopCollection('inherited_response_changed')
            if not old.get('accounting_already_in_inherited_arms',False):
                receipts.append(g.read_json(path)['receipt'])
    assert len({r['call_id'] for r in receipts})==len(receipts)
    baseline=(inherited.manifest(root) or {}).get('arms',{})
    arms={}
    for arm in ('production','b1','bsmall'):
        records,_=p.audit_arm(root,arm,expected)
        jobs=[cp for cp in batches if cp['arm']==arm]
        outcomes=[out for cp in jobs for out in cp.get('outcomes',{}).values()]
        reasons=Counter(out['reason'] for out in outcomes if out['status']=='rejected')
        previous=baseline.get(arm,{})
        reasons.update(previous.get('rejection_reasons',{}))
        calls=[r for r in receipts if r['arm']==arm]
        tokens=sum(r['usage']['total_tokens'] for r in calls)+previous.get('generation_total_tokens',0)
        active=max(sum(cp['active_seconds'] for cp in jobs),sum(r['seconds'] for r in calls))+previous.get('active_seconds',0)
        format_failures=Counter(stage.get('failure','no_readable_records') for cp in jobs
            for stage in cp['stages'].values() if stage.get('failure') or (stage.get('completed') and not stage['reply']['records'] and set(stage['reply']['errors'].values()) != {'record_schema'}))
        format_failures.update(previous.get('protocol_stage_failures',{}))
        prior_output=(previous.get('output_tokens_per_second') or 0)*previous.get('active_seconds',0)
        arms[arm]={'accepted':len(records),'inherited_accepted':previous.get('accepted',0),
            'accepted_this_revision':len(records)-previous.get('accepted',0),
            'completed_cases':len(outcomes)+previous.get('completed_cases',0),'rejected':sum(reasons.values()),
            'rejection_reasons':dict(reasons),'protocol_stage_failures':dict(format_failures),
            'pending_verification':sum(len(cp.get('pending_verification',[])) for cp in jobs if not cp.get('completed')),
            'author_requests':sum(r['role'].startswith('author') for r in calls)+previous.get('author_requests',0),
            'verifier_requests':sum(r['role'].startswith('verifier') for r in calls)+previous.get('verifier_requests',0),
            'generation_total_tokens':tokens,'tokens_per_accepted':tokens/len(records) if records else None,
            'active_seconds':active,'accepted_per_minute':len(records)*60/active if active else None,
            'output_tokens_per_second':(sum(r['usage']['completion_tokens'] for r in calls)+prior_output)/active if active else None,
            'coverage':g.url_coverage(list(records.values())),
            'snapshot':export_snapshot(p.arm_root(root,arm),records,expected) if records else None}
    uncertainty=getattr(ledger,'uncertainty_reserved_tokens',0)
    value={'at':g.utc(),'run_id':run_id,'contract_sha256':g.digest(expected),'arms':arms,
        'cumulative_usage':ledger.usage,'token_budget':token_budget,
        'uncertainty_reserved_tokens':uncertainty,
        'remaining_tokens':token_budget-ledger.usage['total_tokens']-uncertainty-sum(ledger.pending.values()),
        'controls_total_tokens':sum(r['usage']['total_tokens'] for r in receipts if r['arm']=='controls'),
        'prior_usage':expected['inherited_usage'],'same_model_consistency_checks_not_human_validation':True,
        'rate_scope':'Sum of active generation and verification work per arm; includes failures and reasoning. Controls excluded.',
        'winner':None,'comparison_note':'No speedup or quality generalization established by a small pilot.'}
    atomic(root/'comparison-report.json',value)
    lines=['# IFM independent-record collector','','| Arm | Accepted | Rejected | Author / verifier requests | Generation tokens | Tokens per accepted |',
           '|---|---:|---:|---:|---:|---:|']
    for arm,metric in arms.items():
        lines.append(f"| {arm} | {metric['accepted']} | {metric['rejected']} | {metric['author_requests']} / {metric['verifier_requests']} | {metric['generation_total_tokens']} | {metric['tokens_per_accepted'] or '—'} |")
    lines+=['',f"Cumulative tokens including prior attempts and controls: {ledger.usage['total_tokens']:,} / {token_budget:,}.",
            f"Additional conservative hold for requests with unknown usage: {uncertainty:,} tokens. This is not measured generation.",
            '',value['rate_scope'],'',value['comparison_note'],'',
            'Detailed rejection reasons, protocol failures, pending verification and URL coverage are in comparison-report.json.']
    (root/'comparison-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return value


def jobs_for(root,args):
    if args.mode in ('pilot','compare'):
        specs=b.pilot_specs()
        arms=('b1','bsmall') if args.mode=='compare' else ('production',)
        return [{'id':args.mode+'-'+arm,'arm':arm,'specs':specs} for arm in arms]
    old=[cp for cp in all_batches(root) if cp['arm']=='production']
    jobs=[{k:cp[k] for k in ('id','arm','specs')} for cp in old if not cp.get('completed')]
    attempted={s['id'] for cp in old for s in cp['specs']}
    remaining=[s for s in g.specifications() if s['id'] not in attempted][:args.max_cases]
    for chunk in b.pack(remaining,args.verifier_cases,args.verifier_tabs):
        jobs.append({'id':'collect-'+g.digest([s['id'] for s in chunk])[:16],'arm':'production','specs':chunk})
    return jobs


def run(args,root=g.OUT,client_factory=Client):
    validate_token_budget(getattr(args,'token_budget',100000))
    expected=prepare(root,args)
    token_budget=expected['maximum_authorized_lifetime_tokens']
    Ledger(root,token_budget)  # Verify inherited ledgers and unresolved requests before any launch.
    for cp in all_batches(root):
        if cp.get('completed'): p.materialize(root,cp)
    records,prod_index=p.audit_arm(root,'production',expected)
    base={'state':'prepared','phase':'ready_for_bounded_start','updated_at':g.utc(),'pid':None,
          'provider':'ifm_direct','execution':'ifm_direct_api','teacher':g.ALIAS,'target':5000,
          'curriculum':g.audience.VERSION,'accepted_examples':len(records),'accepted_this_run':0,
          'rejections_this_run':{},'elapsed_seconds':0,'examples_per_hour_this_run':0,'workers':1,
          'token_budget':token_budget,'pricing_status':'price_unverified_bounded_run_authorized','batching':expected['batching'],
          'record_protocol':'jsonl_author_strict_single_reviewer_v9','mode':args.mode,'active_candidates':0}
    if args.prepare:
        if not (root/'status.json').exists(): atomic(root/'status.json',base)
        report(root,expected)
        print(json.dumps({'prepared':True,'batching':expected['batching']}))
        return
    if not args.start_approved: raise StopCollection('explicit_bounded_start_required')
    started=time.time()
    run_id=str(uuid.uuid4())
    info={'run_id':run_id,'started_unix':started,'started_at':g.utc(),'model_ready_unix':started,
          'deadline_unix':started+args.minutes*60,'minutes':args.minutes,'mode':args.mode,'contract_sha256':g.digest(expected)}
    folder=root/'runs'/run_id
    folder.mkdir()
    atomic(folder/'run.json',info)
    client=None
    before=len(records)
    indices={'production':prod_index,**{arm:p.audit_arm(root,arm,expected)[1] for arm in ('b1','bsmall')}}
    state={**base,'state':'running','run_id':run_id,'pid':os.getpid()}

    def stop():
        if (root/'PAUSE').exists(): raise StopCollection('manual_pause')
        if time.time()>=info['deadline_unix']: raise StopCollection('time_limit')

    def publish(phase):
        elapsed=max(.001,time.time()-started)
        batches=all_batches(root)
        current=[cp for cp in batches if cp.get('decision_run_id',cp.get('run_id'))==run_id]
        production=[g.read_json(path)['example'] for path in (root/'accepted').glob('*.json')]
        rejected=Counter(out['reason'] for cp in current if cp['arm']=='production'
            for out in cp.get('outcomes',{}).values() if out['status']=='rejected' and out.get('decision_run_id',run_id)==run_id)
        state.update(phase=phase,updated_at=g.utc(),elapsed_seconds=elapsed,accepted_examples=len(production),
            accepted_this_run=len(production)-before,rejections_this_run=dict(rejected),
            examples_per_hour_this_run=(len(production)-before)*3600/elapsed,
            splits=dict(Counter(ep['split'] for ep in production)),
            audience_counts=dict(Counter(ep['research_context']['audience'] for ep in production)),
            url_coverage=g.url_coverage(production),
            pending_verification=sum(len(cp.get('pending_verification',[])) for cp in batches if not cp.get('completed')),
            protocol_failures=sum(bool(cp.get('halted')) for cp in current),
            benchmark_progress={'completed':sum(len(cp.get('outcomes',{})) for cp in batches if cp['arm']!='production'),
                'passed':sum(o['status']=='accepted' for cp in batches if cp['arm']!='production' for o in cp.get('outcomes',{}).values()),'planned':6})
        if client:
            state.update(token_usage=client.ledger.usage,pending_api_calls=len(client.ledger.pending),
                api_calls=len(client.ledger.settled),quota_headers=client.quota_headers,
                provider_token_window=getattr(client,'provider_token_window',None),inherited_token_usage=expected['inherited_usage'])
        atomic(root/'status.json',state)

    reason,failed='completed',False
    try:
        stop()
        client=client_factory(root,token_budget,run_id,stop)
        publish('starting')
        for job in jobs_for(root,args):
            stop()
            state.update(active_candidates=len(job['specs']),current_benchmark_arm=job['arm'])
            p.process_job(root,job,expected,indices[job['arm']],client,stop,publish)
            publish('batch_completed')
    except StopCollection as exc: reason=str(exc)
    except KeyboardInterrupt: reason='interrupted_checkpoint_saved'
    except Exception as exc:
        reason,failed='internal_'+type(exc).__name__,True
        error_file(root,exc)
    finally:
        info.update(ended_unix=time.time(),ended_at=g.utc(),stop_reason=reason)
        atomic(folder/'run.json',info)
        try:
            report(root,expected,run_id)
            audit={'result':'passed'}
        except StopCollection as exc:
            audit={'result':'blocked','reason':str(exc)}
        except Exception as exc:
            audit={'result':'failed','error_type':type(exc).__name__}
            failed=True
            error_file(root,exc)
        state.update(state='failed' if failed else 'paused',stop_reason=reason,active_candidates=0)
        publish('stopped')
        atomic(root/'latest-report.json',{**info,'accepted_new':state['accepted_this_run'],
            'accepted_total':state['accepted_examples'],'rejected':sum(state['rejections_this_run'].values()),
            'audit':audit,'generation_only_no_student_training':True})
        print(json.dumps({'state':state['state'],'stop_reason':reason,'accepted_new':state['accepted_this_run'],'audit':audit}))


def arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--start-approved',action='store_true')
    parser.add_argument('--token-budget',type=int,default=100000,
                        help='Cumulative experiment token cap including inherited usage (1..50000000).')
    parser.add_argument('--mode',choices=('pilot','compare','collect'),default='pilot')
    parser.add_argument('--minutes',type=int,choices=range(1,31),default=15)
    parser.add_argument('--max-cases',type=int,choices=range(1,101),default=12)
    parser.add_argument('--author-cases',type=int,choices=range(1,4),default=b.AUTHOR_CASES)
    parser.add_argument('--author-tabs',type=int,choices=range(30,61),default=b.AUTHOR_TABS)
    parser.add_argument('--verifier-cases',type=int,choices=range(1,7),default=b.VERIFIER_CASES)
    parser.add_argument('--verifier-tabs',type=int,choices=range(30,121),default=b.VERIFIER_TABS)
    return parser.parse_args()


if __name__=='__main__':
    with lock(g.OUT): run(arguments())
