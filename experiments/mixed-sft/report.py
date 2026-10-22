"""Matched synthetic validation, provider slices, and paired family-block intervals."""
from collections import defaultdict
import html
import random
import statistics
from common import *

def percentile(values,p):
    values=sorted(values);return values[min(len(values)-1,int(p*(len(values)-1)))]

def metric_summary(predictions):
    from tabcrew_rl.evaluate import aggregate
    result=aggregate(predictions)
    if predictions:
        result['latency_p50_seconds']=percentile([r['latency_seconds'] for r in predictions],.5)
        result['latency_p95_seconds']=percentile([r['latency_seconds'] for r in predictions],.95)
    return result

def paired_interval(new,old,episodes,metric):
    groups=defaultdict(list)
    for ep in episodes:
        family=canonical_family(ep['family'])
        groups[family].append(float(new[ep['id']]['metrics'][metric])-float(old[ep['id']]['metrics'][metric]))
    values=[item for group in groups.values() for item in group]
    buckets=list(groups.values());rng=random.Random(42);draws=[]
    for _ in range(2000):
        sampled=[item for group in rng.choices(buckets,k=len(buckets)) for item in group]
        draws.append(statistics.mean(sampled))
    return {'delta':statistics.mean(values),'ci95':[percentile(draws,.025),percentile(draws,.975)],
        'method':'Paired family-block bootstrap; episode-weighted point and resampled estimators; 2000 draws, seed42',
        'families':len(buckets)}

def build():
    from tabcrew_rl.protocol import PROMPT_VERSION
    from tabcrew_rl.reward import VERSION as REWARD_VERSION
    plan=read(RUN/'run-plan.json')
    episodes=load_split(DATA/'evaluation','validation');ids={ep['id'] for ep in episodes}
    models={};predictions={};summaries={}
    for name in ('new','old-sft','old-grpo'):
        folder=RUN/(name+'-validation');summary=read(folder/'summary.json');values=rows(folder/'predictions.jsonl')
        expected_adapter=RUN/'sft'/'adapter' if name=='new' else OLD/('sft' if name=='old-sft' else 'grpo')/'adapter'
        if Path(summary['adapter']).resolve()!=expected_adapter.resolve():raise ValueError('Evaluation adapter identity mismatch')
        if summary['model']['revision']!=plan['model']['revision']:raise ValueError('Base model revision mismatch')
        if summary['prompt_version']!=PROMPT_VERSION or summary['reward_version']!=REWARD_VERSION:raise ValueError('Evaluation protocol version mismatch')
        if summary['dataset_manifest_hash']!=sha(DATA/'evaluation'/'manifest.json'):raise ValueError('Evaluation dataset changed')
        by_id={row['id']:row for row in values}
        if len(values)!=len(ids) or set(by_id)!=ids:raise ValueError('Incomplete or mismatched evaluation')
        predictions[name]=by_id;summaries[name]=summary
        models[name]={'overall':metric_summary(values),'by_provider':{},'by_niche':{},'by_locale':{},'by_tab_count':{}}
        for dimension,key in [('provider',lambda ep:ep['dataset_source']['provider']),('niche',lambda ep:ep.get('research_context',{}).get('niche','legacy')),
                              ('locale',lambda ep:ep['locale']),('tab_count',lambda ep:str(ep['tab_count']))]:
            groups=defaultdict(list)
            for ep in episodes:groups[key(ep)].append(by_id[ep['id']])
            models[name]['by_'+dimension]={label:metric_summary(group) for label,group in sorted(groups.items())}
        models[name]['provider_macro_exact']=statistics.mean(group['exact_partition_rate'] for group in models[name]['by_provider'].values())
    if len({digest(summary['decoding']) for summary in summaries.values()})!=1:raise ValueError('Decoding mismatch')
    if len({summary['selection_hash'] for summary in summaries.values()})!=1:raise ValueError('Selection mismatch')
    for ident in ids:
        if len({values[ident]['prompt_hash'] for values in predictions.values()})!=1:raise ValueError('Prompt mismatch')
    comparisons={name:{metric:paired_interval(predictions['new'],predictions[name],episodes,metric)
                       for metric in ('exact_partition','reward','valid')} for name in ('old-sft','old-grpo')}
    plan=read(RUN/'run-plan.json')
    result={'completed_at':utc(),'models':models,'paired_new_minus_old':comparisons,'evaluation_count':len(ids),
        'evaluation_manifest_sha256':sha(DATA/'evaluation'/'manifest.json'),'dataset_manifest_sha256':sha(DATA/'manifest.json'),
        'training':read(RUN/'sft'/'summary.json'),'final_test_evaluated':False,
        'interpretation':'Agreement with held-out synthetic references. Same prompts, selected cases and decoding across models. New data and effective batch size both change; this is not a data-only causal ablation.',
        'release_status':'Not an extension release approval: independent real-user grouping review and testing in the intended inference runtime are still required.',
        'limitations':['Small balanced-provider development slice, with remaining validation retained.',
                       'Teacher labels reflect synthetic consistency, not uniquely correct human grouping.',
                       'Shared curriculum and source generators limit claims of out-of-distribution generalization.',
                       'GPU validation latency does not establish browser, CPU, or deployed-server latency.'],
        'source_sha256':plan['source_sha256']}
    write(RUN/'comparison.json',result)
    lines=['# Mixed-teacher student comparison','','Same frozen validation cases, prompts, and decoding. Final test remains sealed.','',
        '| Model | Valid output | Exact grouping | Mean reward | ARI on valid output | p50 / p95 latency |',
        '|---|---:|---:|---:|---:|---:|']
    table=[]
    for name,values in models.items():
        m=values['overall'];table.append(f'<tr><td>{html.escape(name)}</td><td>{m["validity_rate"]:.1%}</td><td>{m["exact_partition_rate"]:.1%}</td><td>{m["mean_reward_all"]:.3f}</td><td>{m["latency_p50_seconds"]:.2f}s / {m["latency_p95_seconds"]:.2f}s</td></tr>')
        lines.append(f'| {name} | {m["validity_rate"]:.1%} | {m["exact_partition_rate"]:.1%} | {m["mean_reward_all"]:.3f} | {m["mean_ari_valid"]} | {m["latency_p50_seconds"]:.2f}s / {m["latency_p95_seconds"]:.2f}s |')
    lines+=['',result['interpretation'],'',result['release_status'],'','See comparison.json for provider, niche, locale, tab-count slices and paired confidence intervals.']
    (RUN/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    markup='<!doctype html><meta charset="utf-8"><title>TabCrew small-model comparison</title><style>body{font:17px system-ui;max-width:1050px;margin:48px auto;padding:24px;background:#f5f8f7;color:#122b30}table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;padding:16px;border-bottom:1px solid #cee0dc}p{line-height:1.6}</style><h1>TabCrew: mixed-teacher student</h1>'
    markup+=f'<p>{len(ids)} identical validation sessions for all three adapters. Final test remains sealed.</p><table><tr><th>Model</th><th>Valid output</th><th>Exact grouping</th><th>Mean reward</th><th>p50 / p95</th></tr>'+''.join(table)+'</table>'
    markup+='<p>'+html.escape(result['interpretation'])+'</p><p>'+html.escape(result['release_status'])+'</p>'
    (RUN/'comparison.html').write_text(markup,encoding='utf-8')
    return result

if __name__=='__main__':build()
