"""Paired before/after comparison: identical dataset, prompts, decoding, and examples required."""
import argparse, json, pathlib, random, statistics
from .data import read_jsonl

def compare(before_dir,after_dir):
    before_dir,after_dir=pathlib.Path(before_dir),pathlib.Path(after_dir)
    bs=json.loads((before_dir/'summary.json').read_text()); ts=json.loads((after_dir/'summary.json').read_text())
    for field in ('prompt_version','reward_version','decoding','dataset_manifest_hash','selection_hash'):
        if bs[field]!=ts[field]: raise ValueError(f'Uncontrolled comparison: {field} differs')
    if bs['model']['revision']!=ts['model']['revision']: raise ValueError('Different base checkpoints')
    br=read_jsonl(before_dir/'predictions.jsonl'); ar=read_jsonl(after_dir/'predictions.jsonl')
    before={(r['id'],r['seed']):r for r in br}
    after={(r['id'],r['seed']):r for r in ar}
    if len(before)!=len(br) or len(after)!=len(ar): raise ValueError('Duplicate episode/repeat records')
    if not before:raise ValueError('Empty paired evaluation')
    if before.keys()!=after.keys(): raise ValueError('Different evaluated episode/repeat keys')
    paired=[]
    has_families=all('family' in r for r in br+ar)
    for key,b in before.items():
        t=after[key]
        if b['prompt_hash']!=t['prompt_hash']: raise ValueError('Prompt differs for '+str(key))
        bm,tm=b['metrics'],t['metrics']
        if 'family' in b and 'family' in t and b['family']!=t['family']: raise ValueError('Family metadata mismatch')
        paired.append({'id':key[0],'family':b['family'] if has_families else key[0],'seed':key[1],'before':b,'after':t,
          'reward_delta':tm['reward']-bm['reward'],
          'ari_delta_both_valid':tm['ari']-bm['ari'] if bm['valid'] and tm['valid'] else None})
    # Bootstrap family blocks, including all variants/repeats. Old receipts without family
    # metadata get an explicitly episode-only diagnostic interval, never a family-level claim.
    episode_ids=sorted({r['id'] for r in paired})
    ids=sorted({r['family'] for r in paired}); by_id={k:[r['reward_delta'] for r in paired if r['family']==k] for k in ids}
    means={k:statistics.mean(v) for k,v in by_id.items()}; rng=random.Random(731)
    boot=sorted(statistics.mean(means[rng.choice(ids)] for _ in ids) for _ in range(2000))
    both=[r['ari_delta_both_valid'] for r in paired if r['ari_delta_both_valid'] is not None]
    return {'before':bs,'after':ts,'episode_count':len(episode_ids),'paired_predictions':len(paired),
      'reward_delta':statistics.mean(r['reward_delta'] for r in paired),
      'bootstrap_95_interval':[boot[49],boot[1949]],'bootstrap_clusters':len(ids),
      'bootstrap_unit':'family' if has_families else 'episode (family metadata unavailable in early receipts)',
      'both_valid_predictions':len(both),'ari_delta_both_valid':statistics.mean(both) if both else None,
      'interpretation':'Reward includes invalid-output penalty. ARI on pairs valid in both arms isolates grouping quality conditional on validity; it is a selected subset. Intervals with few families are descriptive only. Small synthetic pilot, not production reliability.',
      'pairs':paired}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--before',required=True);p.add_argument('--after',required=True);p.add_argument('--out',required=True)
    a=p.parse_args(); result=compare(a.before,a.after)
    path=pathlib.Path(a.out); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('before','after','pairs')},indent=2))

if __name__=='__main__': main()
