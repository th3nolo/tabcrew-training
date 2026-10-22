"""Offline crash recovery: audit committed work after both owned processes have stopped."""
import argparse
from collections import Counter
import time
import collector as c
from operations import owned_collector_alive,first_stop

g=c.g

def recover(run_id):
    assert g.read_json(c.OUT/'active-run.json')['run_id']==run_id,'A different run is current'
    folder=c.OUT/'runs'/run_id;run=g.read_json(folder/'run.json')
    assert g.read_json(c.OUT/'runtime.json')['state']=='stopped','Runtime shutdown is unconfirmed'
    assert not owned_collector_alive(run['collector_pid'],run_id),'Collector is still alive'
    with c.lock():
        assert g.read_json(c.OUT/'active-run.json')['run_id']==run_id
        first_stop(c.OUT/'PAUSE','Recovery requires an explicit later resume')
        first_stop(c.OUT/'RUNTIME_STOP','Recovery leaves runtime stopped')
        frozen=g.read_json(c.OUT/'contract.json');assert c.contract()==frozen
        accepted,_=c.accepted_records(frozen)
        errors=Counter()
        initial=set(run['initial_rejected_files'])
        for path in (c.OUT/'rejected').glob('*.json'):
            if path.name not in initial:errors[g.read_json(path)['rejection']]+=1
        status=g.read_json(c.OUT/'status.json')
        assert status['run_id']==run_id
        status.update(state='failed',phase='collector_process_exited',updated_at=g.utc(),active_candidates=0,
            error='Recovered committed records after the collector exited unexpectedly; no automatic restart.',
            accepted_examples=len(accepted),accepted_this_run=len(accepted)-run['initial_accepted'],
            attempts_this_run=len(accepted)-run['initial_accepted']+sum(errors.values()),rejections_this_run=dict(errors),
            pending_checkpoints=len(list((c.OUT/'pending').glob('*.json'))),snapshot=g.export_snapshot(accepted),
            splits=dict(Counter(ep['split'] for ep in accepted.values())),
            audience_counts=dict(Counter(ep['research_context']['audience'] for ep in accepted.values())),
            scenario_counts=dict(Counter(ep['research_context']['scenario'] for ep in accepted.values())))
        status['elapsed_seconds']=time.time()-run['started_unix']
        status['examples_per_hour_this_run']=status['accepted_this_run']*3600/status['elapsed_seconds']
        g.atomic(c.OUT/'status.json',status)
        from audit import audit
        try:result=audit()
        except Exception as error:result={'result':'failed','error':f'{type(error).__name__}: {error}'}
        g.atomic(folder/'audit.json',result)
        run.update(ended_unix=time.time(),state='failed',reason='collector_process_exited',recovered=True)
        g.atomic(folder/'run.json',run);g.atomic(folder/'final-status.json',status)
        from report import write_report
        return write_report(run_id)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id');args=parser.parse_args()
    print(g.json.dumps(recover(args.run_id or g.read_json(c.OUT/'active-run.json')['run_id']),indent=2))
