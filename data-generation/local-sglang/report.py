"""Per-run morning reports from receipts and telemetry, without model execution."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import statistics
from files import atomic

ROOT=Path(__file__).resolve().parent
read=lambda path:json.loads(path.read_text(encoding='utf-8-sig'))

def rows(path):
    if not path.exists():return []
    result=[]
    lines=path.read_bytes().splitlines(keepends=True)
    for i,line in enumerate(lines):
        try:result.append(json.loads(line))
        except (ValueError,UnicodeDecodeError):
            if i!=len(lines)-1 or line.endswith(b'\n'):raise
    return result

def summarize(run, status, requests, telemetry):
    elapsed=max(0,run['ended_unix']-run['started_unix'])
    generation_seconds=max(0,run['ended_unix']-run['model_ready_unix']) if run.get('model_ready_unix') else 0
    requests=[r for r in requests if r.get('run_id')==run['run_id']]
    usable=[r for r in requests if isinstance(r.get('usage'),dict) and
            all(type(r['usage'].get(k))==int and r['usage'][k]>=0 for k in ('prompt_tokens','completion_tokens'))]
    output=sum(r['usage']['completion_tokens'] for r in usable)
    energy=0
    for a,b in zip(telemetry,telemetry[1:]):
        gap=(datetime.fromisoformat(b['at'])-datetime.fromisoformat(a['at'])).total_seconds()
        if 0<=gap<=15:energy+=(a['gpu_power_watts']+b['gpu_power_watts'])/2*gap/3600
    accepted=status['accepted_this_run']; rejected=sum(status['rejections_this_run'].values())
    return {'run_id':run['run_id'],'mode':run['mode'],'state':status['state'],'stop_reason':status['phase'],
        'stopped_normally':status['state']!='failed' and status['phase'] in ('target_reached','time_limit','requested_examples_reached','pause_requested'),
        'error':status.get('error'),
        'started_unix':run['started_unix'],'ended_unix':run['ended_unix'],
        'wall_seconds':elapsed,'generation_seconds_including_drain':generation_seconds,
        'planned_hours':run['limit_hours'],'workers':4,'teacher':'Qwen3-4B-AWQ','ngram':False,
        'accepted_before':run['initial_accepted'],'accepted_new':accepted,'accepted_total':status['accepted_examples'],
        'rejected':rejected,'rejection_reasons':status['rejections_this_run'],
        'first_pass_accepted':status.get('first_pass_accepted_this_run'),
        'accepted_after_repair':status.get('accepted_after_repair_this_run'),
        'completed_candidates_with_repair':status.get('repair_candidates_completed_this_run'),
        'acceptance_rate_completed_candidates':accepted/(accepted+rejected) if accepted+rejected else None,
        'accepted_per_wall_hour':accepted*3600/elapsed if elapsed else None,
        'completed_requests':len(requests),'requests_with_unknown_usage':len(requests)-len(usable),
        'input_tokens':sum(r['usage']['prompt_tokens'] for r in usable),'output_tokens':output,
        'output_tokens_per_wall_second':output/elapsed if elapsed else None,
        'output_tokens_per_generation_second':output/generation_seconds if generation_seconds else None,
        'request_seconds_by_role':{role:{'count':len(part),'median':statistics.median(r['seconds'] for r in part)}
            for role in ('author','verifier') if (part:=[r for r in requests if r['role']==role])},
        'pending_checkpoints':status['pending_checkpoints'],'coverage':{
            'splits':status['splits'],'audience':status['audience_counts'],'scenario':status['scenario_counts']},
        'resources':{'samples':len(telemetry),'minimum_free_commit_gib':min((r['free_commit_gib'] for r in telemetry),default=None),
            'minimum_free_ram_gib':min((r['free_ram_gib'] for r in telemetry),default=None),
            'peak_gpu_used_mib':max((r['gpu_used_mib'] for r in telemetry),default=None),
            'peak_gpu_temperature_c':max((r['gpu_temperature_c'] for r in telemetry),default=None),
            'sampled_whole_gpu_energy_wh':energy if telemetry else None},
        'runtime_shutdown_confirmed':read(ROOT/'runtime.json')['state']=='stopped',
        'limitations':['Synthetic consistency checks, not independent human validation.',
            'Whole-GPU sampled energy includes desktop activity; not whole-computer power.',
            'Output throughput includes rejected/malformed completed responses and run overhead.',
            'Generation quality and student generalization are separate measurements.']}

def write_report(run_id):
    folder=ROOT/'runs'/run_id;run=read(folder/'run.json');status=read(folder/'final-status.json')
    result=summarize(run,status,rows(ROOT/'requests.jsonl'),rows(folder/'resources.jsonl'))
    urls=status.get('url_coverage',{})
    result['coverage']['url_cohorts']=urls.get('cohorts',{})
    result['coverage']['url_platforms']=urls.get('platforms',{})
    result['coverage']['url_routes']=urls.get('routes',{})
    result['coverage']['research_niches']=urls.get('niches',{})
    result['coverage']['research_stages']=urls.get('research_stages',{})
    result['audit']=read(folder/'audit.json') if (folder/'audit.json').exists() else {'result':'unavailable'}
    result['watchdog_shutdown_confirmed']=(folder/'guard-finished.json').exists()
    atomic(folder/'report.json',result);atomic(ROOT/'latest-report.json',result)
    rate=result['accepted_per_wall_hour'];rate=f'{rate:.1f}' if rate is not None else 'unavailable'
    tokens=result['output_tokens_per_generation_second']
    text=f"""# Local generation morning report

Run: {run_id}. Mode: {result['mode']}. State: **{result['state']}**. Stop reason: **{result['stop_reason']}**.

Generated **{result['accepted_new']} new passing examples**, bringing this corpus from {result['accepted_before']} to **{result['accepted_total']} / 5,000**. Rejected {result['rejected']} candidates; {result['pending_checkpoints']} checkpoints are saved for a later explicit resume.

First-pass accepted: {result['first_pass_accepted']}; accepted after one targeted repair: {result['accepted_after_repair']}. Completed candidates that entered repair: {result['completed_candidates_with_repair']}. Repaired cases remain in the original candidate denominator.

The run lasted {result['wall_seconds']/60:.1f} minutes of its {result['planned_hours']}-hour bound. Accepted examples per wall hour: {rate}. Completed requests: {result['completed_requests']}; output tokens: {result['output_tokens']:,}. Aggregate output tokens per second after model readiness, including drain: {tokens if tokens is not None else 'unavailable'}. Both author and verifier tokens, including rejected outputs, are counted.

The runtime shutdown is {'confirmed' if result['runtime_shutdown_confirmed'] else 'UNCONFIRMED'}. Model: Qwen3-4B-AWQ / SGLang 0.5.17. Concurrent requests: four. NGRAM: disabled.

Final corpus audit: **{result['audit']['result']}**. Watchdog exit: {'confirmed' if result['watchdog_shutdown_confirmed'] else 'unconfirmed'}.

## Rejections

"""+('\n'.join(f'- {reason}: {count}' for reason,count in sorted(result['rejection_reasons'].items(),key=lambda item:-item[1])) or '- None.')
    text+='\n\n## Dataset coverage\n\n'+ '\n'.join(f'- {key}: '+', '.join(f'{name}={count}' for name,count in values.items()) for key,values in result['coverage'].items())
    text+='\n\n## Resources and interpretation\n\n'+json.dumps(result['resources'],indent=2)+'\n\n'+ '\n'.join('- '+item for item in result['limitations'])+'\n'
    (folder/'report.md').write_text(text,encoding='utf-8');(ROOT/'latest-report.md').write_text(text,encoding='utf-8')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args()
    print(json.dumps(write_report(args.run_id),indent=2))
