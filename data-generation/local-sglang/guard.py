"""Independent watchdog; only controls its current revision/run through stop flags."""
import argparse
from datetime import datetime
import json
import time
from pathlib import Path
from files import append, atomic, utc, read_json_retry
from operations import settings, sample, resource_reason, ResourceGuard, first_stop, owned_collector_alive

ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args()
    assert args.run_id and all(c.isalnum() or c=='-' for c in args.run_id)
    folder=ROOT/'runs'/args.run_id; run=read_json_retry(folder/'run.json'); limits=settings()
    monitor=ResourceGuard(limits)
    stop_at=None;last_identity=0;reason=None;orphaned=False
    try:
        while True:
            current=read_json_retry(ROOT/'active-run.json')
            if current['run_id']!=args.run_id:break
            status=read_json_retry(ROOT/'status.json')
            runtime=read_json_retry(ROOT/'runtime.json')
            if status.get('run_id')==args.run_id and status['state']!='running' and runtime['state']=='stopped':break
            row=sample();append(folder/'resources.jsonl',row)
            now=time.time();reason=monitor.check(row,now)
            if now>=run['deadline_unix'] and reason is None:reason='time_limit'
            if now-last_identity>=15:
                if not owned_collector_alive(run['collector_pid'],args.run_id):
                    reason='collector_process_exited';orphaned=True
                last_identity=now
            if now-datetime.fromisoformat(status['updated_at']).timestamp()>limits['collector_heartbeat_timeout_seconds']:
                reason=reason or 'collector_heartbeat_stale'
            if not reason and not (folder/'watchdog-ready.json').exists():
                atomic(folder/'watchdog-ready.json',{'at':utc(),'run_id':args.run_id,'sample':row})
            if reason:
                if not (folder/'guard-stop.json').exists():atomic(folder/'guard-stop.json',{'at':utc(),'reason':reason,'sample':row})
                first_stop(ROOT/'PAUSE',reason)
                stop_at=stop_at or now
                if reason!='time_limit' or now-stop_at>=limits['drain_seconds']:
                    first_stop(ROOT/'RUNTIME_STOP',reason)
            if (ROOT/'PAUSE').exists():
                stop_at=stop_at or now
                if now-stop_at>=limits['drain_seconds']:first_stop(ROOT/'RUNTIME_STOP','Graceful drain deadline reached')
            if (ROOT/'RUNTIME_STOP').exists() and runtime['state']=='stopped':break
            time.sleep(limits['sample_seconds'])
    except BaseException as error:
        reason='watchdog_failed'
        atomic(folder/'guard-stop.json',{'at':utc(),'reason':reason,'error':f'{type(error).__name__}: {error}'})
        first_stop(ROOT/'PAUSE',reason);first_stop(ROOT/'RUNTIME_STOP',reason)
        raise
    finally:
        atomic(folder/'guard-finished.json',{'at':utc(),'reason':reason})
        if orphaned and read_json_retry(ROOT/'runtime.json')['state']=='stopped':
            from recover import recover
            recover(args.run_id)

if __name__=='__main__':main()
