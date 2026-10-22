"""Four local candidates, bounded overnight runs and immutable provenance."""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from collections import Counter,deque
from contextlib import contextmanager
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
import urllib.error
import core as g
from files import atomic,append
from inference import Client
import repair_stage
from operations import settings, first_stop

OUT=g.OUT
# SGLang runtime inside WSL (Linux path), WSL distribution, and a Windows-side copy of the runtime's uv.lock.
SGLANG_RUNTIME=os.environ.get('TABCREW_SGLANG_RUNTIME','')
WSL_DISTRO=os.environ.get('TABCREW_WSL_DISTRO','Ubuntu')
def sglang_lock():
    return Path(os.environ['TABCREW_SGLANG_LOCK'])
PORT=60647
CODE_FILES=['repair.py','repair_stage.py','collector.py','core.py','inference.py','files.py','audience.py','serve.py',
            'entrypoint.py','stream_weights.py','runtime_cache.py','linux_memory.py','runtime-profile.json',
            'url_templates.py','url_validation.py','source_catalog.py','operations.py','guard.py','report.py','audit.py','recover.py','settings.json','start.ps1','resume.ps1','overnight.ps1','pause.ps1']
WORKERS=4

def resume_candidates(specs, accepted, pending, offsets):
    """Preserve curriculum order within fresh and retry lanes; pending stages run first."""
    remaining=[s for s in specs if s['id'] not in accepted and s['id'] not in pending]
    remaining.sort(key=lambda s: bool(offsets[s['id']]))
    return [(s,offsets[s['id']],0) for s in remaining]


@contextmanager
def lock():
    import msvcrt
    with (OUT/'collector.lock').open('a+b') as stream:
        if stream.tell()==0:
            stream.write(b'1'); stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        yield

def contract():
    return {'version':g.VERSION,'seed':g.SEED,'target':5000,
            'target_splits':{'train':4000,'validation':500,'final_test':500},
            'code_sha256':{name:g.sha(OUT/name) for name in CODE_FILES},
            'model_assets_sha256':g.sha(OUT/'assets.json'),'migration_sha256':g.sha(OUT/'migration.json'),
            'dependency_lock_sha256':g.sha(sglang_lock()),
            'specs_sha256':g.digest(g.specifications()),'curriculum':g.audience.VERSION,
            'acceptance':'strict_schema_exact_blind_partition_no_ambiguity_language_and_dedup',
            'verifier':'visible_user_tasks_titles_and_URL_context_v2','author_repair':'one_targeted_patch_at_most_eight_fields_before_first_blind_review','workers':WORKERS}

def accepted_records(expected):
    migration=g.read_json(OUT/'migration.json')
    old=OUT.parent/migration['source_corpus']
    assert (old/'PAUSE').exists() and g.sha(old/'contract.json')==migration['source_contract_file_sha256']
    specs={s['id']:s for s in g.specifications()}
    accepted,index={},g.DuplicateIndex()
    for path in sorted((OUT/'accepted').glob('*.json')):
        saved=g.read_json(path)
        ep=saved['example']
        if path.name in migration['files']:
            assert g.sha(path)==migration['files'][path.name]
            assert g.sha(old/'accepted'/path.name)==migration['files'][path.name]
            assert saved['contract_sha256']==migration['contract_by_file'][path.name]
            assert g.sha(OUT/'legacy-audit.json')==migration['legacy_audit_sha256']
        else:
            assert saved['contract_sha256']==g.digest(expected)
            verify_checkpoint_origin(saved,migration)
        assert saved['example_sha256']==g.digest(ep)
        if path.name in migration['files']:
            assert ep['provenance']['spec_sha256']==g.digest(saved['spec'])
            assert all(ep[k]==saved['spec'][k] for k in ('id','family','split'))
        else:
            assert ep['provenance']['spec_sha256']==g.digest(specs[ep['id']])
            repair_stage.verify(saved,complete=True)
            rebuilt=g.normalize(saved['authored'],specs[ep['id']],saved['author_receipt'])
            for key in ['tabs','reference','intent','locale','split','family','research_context']:
                assert rebuilt[key]==ep[key],key
        g.verify_grouping(ep,saved['blind_review'])
        index.add(index.check(ep))
        accepted[ep['id']]=ep
    assert set(migration['files']).issubset(p.name for p in (OUT/'accepted').glob('*.json'))
    g.validate_corpus(list(accepted.values()))
    return accepted,index

def verify_checkpoint_origin(saved,migration):
    if 'checkpoint_origin' not in saved:return
    origin=saved['checkpoint_origin']
    assert origin['source_corpus']==migration['source_corpus']
    assert origin['filename'] in migration['pending_files']
    source=OUT.parent/origin['source_corpus']/'pending'/origin['filename']
    assert g.sha(source)==origin['sha256']==migration['pending_files'][origin['filename']]
    original=g.read_json(source)
    assert original['contract_sha256']==origin['contract_sha256']
    assert original['spec']==saved['spec'] and original['attempt']==saved['attempt']
    for key in ('authored','author_receipt','blind_review','review_receipt'):
        if key in original:assert saved[key]==original[key],key


def candidate(spec,attempt,contract_hash,client,stop):
    """A checkpoint survives an interrupted local call and never repeats a saved stage."""
    path=OUT/'pending'/f"{spec['id']}-{attempt}.json"
    saved=g.read_json(path) if path.exists() else {'spec':spec,'attempt':attempt,'created_at':g.utc(),'contract_sha256':contract_hash}
    if saved['spec']!=spec or saved['attempt']!=attempt or saved['contract_sha256']!=contract_hash:
        raise RuntimeError('Checkpoint contract mismatch')
    if 'checkpoint_origin' in saved:
        verify_checkpoint_origin(saved,g.read_json(OUT/'migration.json'))
    if saved.get('stage')=='rejected':
        return 'rejected',path,saved
    seed=(spec['seed']+attempt*7919)%(2**31-1)
    try:
        if 'authored' not in saved:
            if stop.is_set() or (OUT/'PAUSE').exists():
                return 'pending',path,saved
            saved['stage']='authoring'; atomic(path,saved)
            authored,receipt=client.chat(g.author_messages(spec,attempt),g.author_schema(spec),seed,role='author',max_tokens=3072 if spec['tab_count']>=24 else 2048)
            saved.update(authored=authored,author_receipt=receipt,stage='authored')
            atomic(path,saved)
        ep=repair_stage.normalize(saved,client,lambda row:atomic(path,row),lambda:stop.is_set() or (OUT/'PAUSE').exists())
        if ep is None:return 'pending',path,saved
        saved['example']=ep
        if 'blind_review' not in saved:
            if stop.is_set() or (OUT/'PAUSE').exists():
                atomic(path,saved)
                return 'pending',path,saved
            saved['stage']='verifying'; atomic(path,saved)
            raw,receipt=client.chat(g.verifier_messages(ep),g.verifier_schema(ep),(seed+104729)%(2**31-1),role='verifier',max_tokens=1024)
            saved.update(blind_review={'tasks':g.visible_tasks(ep),**raw},review_receipt=receipt,stage='reviewed')
            atomic(path,saved)
        agreement=g.verify_grouping(ep,saved['blind_review'])
        ep['reference_status']='local_author_and_blind_regroup_agree_not_human_reviewed'
        ep['provenance']['verifier_request_sha256']=saved['review_receipt']['request_sha256']
        ep['teacher_review']={'ambiguous_refs':[],'review_notes':saved['blind_review']['review_notes'],'blind_partition_agreement':agreement}
        saved.update(example=ep,example_sha256=g.digest(ep))
        return 'reviewed',path,saved
    except (ValueError,KeyError,TypeError) as error:
        saved.update(rejection=str(error),stage='rejected')
        atomic(path,saved)
        return 'rejected',path,saved

def wait_for_watchdog(watchdog,run_folder):
    for _ in range(40):
        if watchdog.poll() is not None or (OUT/'PAUSE').exists():
            raise RuntimeError('Watchdog did not become ready; inspect its receipt before starting inference')
        if (run_folder/'watchdog-ready.json').exists():return
        time.sleep(.5)
    raise RuntimeError('Watchdog readiness timed out; inference was not started')


def start_runtime(run):
    if (OUT/'RUNTIME_STOP').exists():
        raise RuntimeError('RUNTIME_STOP is set; use resume.ps1 after confirming the old runtime stopped')
    script='/mnt/'+str(OUT/'serve.py')[0].lower()+str(OUT/'serve.py')[2:].replace('\\','/')
    log=(OUT/'logs'/f'{time.time_ns()}-runtime-wrapper.log').open('w')
    if not SGLANG_RUNTIME.startswith('/'):raise RuntimeError('Set TABCREW_SGLANG_RUNTIME to the Linux path of the SGLang runtime')
    child=subprocess.Popen(['wsl.exe','-d',WSL_DISTRO,'--exec',SGLANG_RUNTIME.rstrip('/')+'/.venv/bin/python',script,
                '--runtime',SGLANG_RUNTIME,'--lock-sha256',g.sha(sglang_lock()),'--run-id',run['run_id'],'--deadline',str(run['deadline_unix']+settings()['drain_seconds']+30)],
                stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    client=Client(PORT,OUT/'requests.jsonl',run_id=run['run_id'])
    try:
        for _ in range(240):
            if child.poll() is not None:
                raise RuntimeError('SGLang wrapper exited during startup; inspect runtime logs')
            if (OUT/'PAUSE').exists():
                raise RuntimeError('Pause requested during startup')
            try:
                if client.ready():
                    return child,log,client
            except (urllib.error.URLError,TimeoutError):
                pass
            time.sleep(1)
        raise RuntimeError('SGLang startup timed out')
    except BaseException:
        first_stop(OUT/'RUNTIME_STOP','Stop after unsuccessful startup')
        child.wait(timeout=45); log.close()
        raise

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-new',type=int,default=5000)
    parser.add_argument('--max-hours',type=float,default=8)
    parser.add_argument('--max-attempts',type=int,default=4)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if not 1<=args.max_new<=5000 or not 0<args.max_hours<=8 or not 1<=args.max_attempts<=10:
        parser.error('Run bounds exceeded')
    uuid.UUID(args.run_id)
    if args.pilot and args.max_hours!=.5:parser.error('The production-style pilot is exactly 30 minutes')
    for folder in ['accepted','rejected','pending','logs','snapshots']:
        (OUT/folder).mkdir(exist_ok=True)
    if (OUT/'PAUSE').exists():
        raise RuntimeError('PAUSE is set; use resume.ps1')
    with lock():
        sys.addaudithook(g.network_guard)
        frozen=contract()
        if (OUT/'contract.json').exists() and g.read_json(OUT/'contract.json')!=frozen:
            raise RuntimeError('Frozen contract changed; use a new corpus revision')
        atomic(OUT/'contract.json',frozen)
        accepted,index=accepted_records(frozen)
        existing=len(accepted)
        specs=g.specifications()
        if args.pilot:
            specs=[s for s in specs if s['split']=='train']
        spec_map={s['id']:s for s in specs}
        offsets=Counter()
        migration=g.read_json(OUT/'migration.json')
        for path in (OUT/'rejected').glob('*.json'):
            row=g.read_json(path)
            if path.name in migration['rejected_files']:
                assert g.sha(path)==migration['rejected_files'][path.name]
            else:assert row['contract_sha256']==g.digest(frozen)
            offsets[row['spec']['id']]=max(offsets[row['spec']['id']],row['attempt']+1)
        pending={}
        for path in (OUT/'pending').glob('*.json'):
            row=g.read_json(path); identifier=row['spec']['id']
            if identifier in accepted:
                path.unlink()  # A committed receipt is authoritative after a crash before cleanup.
            elif identifier in spec_map:
                if identifier in pending:
                    raise RuntimeError('Multiple pending candidates for one specification')
                pending[identifier]=row['attempt']
        queue=deque((spec_map[i],a,0) for i,a in pending.items())
        queue.extend(resume_candidates(specs,accepted,pending,offsets))
        stop=threading.Event()
        child=log=client=watchdog=guard_log=None
        repair_candidates=0; recovered_candidates=0
        started=time.monotonic(); attempts=0; errors=Counter(); active={}; final='paused'; phase='loading_local_model'; failure=None
        run_folder=OUT/'runs'/args.run_id;run_folder.mkdir()
        run={'run_id':args.run_id,'collector_pid':os.getpid(),'started_unix':time.time(),
             'deadline_unix':time.time()+args.max_hours*3600,'mode':'pilot' if args.pilot else 'overnight',
             'initial_accepted':existing,'limit_hours':args.max_hours,'limit_new_examples':args.max_new,
             'initial_rejected_files':sorted(p.name for p in (OUT/'rejected').glob('*.json')),
             'contract_sha256':g.digest(frozen),'model_ready_unix':None}
        atomic(run_folder/'run.json',run);atomic(OUT/'active-run.json',{'run_id':args.run_id})
        def status(state,phase,**extra):
            elapsed=time.monotonic()-started
            atomic(OUT/'status.json',{'state':state,'phase':phase,'updated_at':g.utc(),'pid':os.getpid(),'run_id':args.run_id,
                'runtime_pid':child.pid if child else None,'runtime_process_kind':'windows_wsl_wrapper',
                'target':5000,'accepted_examples':len(accepted),'accepted_this_run':len(accepted)-existing,
                'attempts_this_run':attempts,'first_pass_accepted_this_run':len(accepted)-existing-recovered_candidates,
                'accepted_after_repair_this_run':recovered_candidates,'repair_candidates_completed_this_run':repair_candidates,
                'rejections_this_run':dict(errors),'workers':WORKERS,'active_candidates':len(active),
                'pending_checkpoints':len(list((OUT/'pending').glob('*.json'))),'elapsed_seconds':elapsed,
                'examples_per_hour_this_run':(len(accepted)-existing)*3600/elapsed if elapsed else 0,
                'splits':dict(Counter(ep['split'] for ep in accepted.values())),
                'audience_counts':dict(Counter(ep['research_context']['audience'] for ep in accepted.values())),
                'scenario_counts':dict(Counter(ep['research_context']['scenario'] for ep in accepted.values())),
                'teacher':'Qwen3-4B-AWQ','execution':'local_sglang_offline','inference_engine':'SGLang 0.5.17 / AWQ Marlin',
                'runtime_profile':'streaming-weights-four-requests-16384-kv',
                'mode':run['mode'],'curriculum':g.audience.VERSION,'audience_label':'Entrepreneurs and freelancers',
                'source_policy':g.audience.source_catalog.policy_summary(),
                'imported_accepted_examples':g.read_json(OUT/'migration.json')['accepted_examples'],
                'url_coverage':g.url_coverage(list(accepted.values())),
                'url_catalog':{'routes':len(g.audience.url_templates.ROUTES),'platform_families':len({v[0] for v in g.audience.url_templates.ROUTES.values()})},
                'previous_examples_in_current_count':True,'limit_new_examples':args.max_new,'limit_hours':args.max_hours,**extra})
        try:
            status('running',phase)
            guard_log=(run_folder/'watchdog.log').open('w')
            watchdog=subprocess.Popen([sys.executable,str(OUT/'guard.py'),'--run-id',args.run_id],
                stdout=guard_log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            wait_for_watchdog(watchdog,run_folder)
            child,log,client=start_runtime(run)
            run['model_ready_unix']=time.time();atomic(run_folder/'run.json',run)
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                while queue or active:
                    if (OUT/'PAUSE').exists() or time.monotonic()-started>=args.max_hours*3600:
                        stop.set()
                    while queue and len(active)<WORKERS and not stop.is_set() and len(accepted)-existing+len(active)<args.max_new:
                        spec,attempt,retries=queue.popleft()
                        future=pool.submit(candidate,spec,attempt,g.digest(frozen),client,stop)
                        active[future]=(spec,attempt,retries)
                    if not active:
                        break
                    status('running','draining_checkpoints' if stop.is_set() else 'generating_parallel')
                    done,_=wait(active,timeout=2,return_when=FIRST_COMPLETED)
                    for future in done:
                        spec,attempt,retries=active.pop(future)
                        try:
                            result,path,saved=future.result()
                        except Exception as error:
                            failure=f'{type(error).__name__}: {error}'; stop.set(); continue
                        if result=='pending':
                            continue
                        attempts+=1
                        if 'repair' in saved:repair_candidates+=1
                        if result=='reviewed':
                            try:
                                fingerprint=index.check(saved['example'])
                            except ValueError as error:
                                saved['rejection']=str(error); result='rejected'
                        if result=='reviewed':
                            destination=OUT/'accepted'/(spec['id']+'.json')
                            if destination.exists():
                                raise RuntimeError('Refusing to overwrite accepted example')
                            if 'repair' in saved:recovered_candidates+=1
                            saved.update(accepted_at=g.utc(),stage='accepted')
                            atomic(destination,saved)
                            index.add(fingerprint); accepted[spec['id']]=saved['example']
                            print(json.dumps({'accepted':spec['id'],'total':len(accepted)}),flush=True)
                            if len(accepted)%50==0:
                                g.export_snapshot(accepted)
                        else:
                            errors[saved['rejection']]+=1
                            atomic(OUT/'rejected'/f"{spec['id']}-{attempt}.json",saved)
                            if retries+1<args.max_attempts:
                                queue.append((spec,attempt+1,retries+1))
                        path.unlink()
                    if len(accepted)-existing>=args.max_new:
                        stop.set()
            if failure:
                raise RuntimeError(failure)
            phase='target_reached' if len(accepted)==5000 else 'time_limit' if time.monotonic()-started>=args.max_hours*3600 else 'pause_requested' if (OUT/'PAUSE').exists() else 'requested_examples_reached' if len(accepted)-existing>=args.max_new else 'attempt_limit'
            final='completed' if len(accepted)==5000 else 'pilot_completed' if args.pilot and phase=='time_limit' else 'paused'
        except BaseException as error:
            final='failed'; phase='error'; failure=f'{type(error).__name__}: {error}'
        finally:
            stop.set()
            first_stop(OUT/'PAUSE','Run finished; explicit resume starts a new bounded run')
            first_stop(OUT/'RUNTIME_STOP','Collector stopped; release the owned SGLang runtime.')
            if child:
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    failure=(failure or '')+'; Runtime wrapper did not confirm shutdown'; final='failed'
            if log:
                log.close()
            if (run_folder/'guard-stop.json').exists():
                guard_reason=g.read_json(run_folder/'guard-stop.json')['reason']
                phase=guard_reason
                if guard_reason!='time_limit':final='paused' if final!='failed' else final
            snapshot=g.export_snapshot(accepted)
            status(final,phase,error=failure,snapshot=snapshot)
            try:
                from audit import audit
                audit_result=audit()
            except BaseException as error:
                final='failed';phase='final_audit_failed';failure=f'{type(error).__name__}: {error}'
                audit_result={'result':'failed','checked_at':g.utc(),'error':failure}
                status(final,phase,error=failure,snapshot=snapshot)
            atomic(run_folder/'audit.json',audit_result)
            run.update(ended_unix=time.time(),state=final,reason=phase)
            atomic(run_folder/'run.json',run)
            atomic(run_folder/'final-status.json',g.read_json(OUT/'status.json'))
            if watchdog:
                try:watchdog.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    run['watchdog_exit_unconfirmed']=True;atomic(run_folder/'run.json',run)
            if guard_log:guard_log.close()
            from report import write_report
            write_report(args.run_id)
        if final=='failed':
            raise SystemExit(failure)

if __name__=='__main__':
    main()
