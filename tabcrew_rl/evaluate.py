import argparse, hashlib, json, pathlib, statistics, time
from .data import load_split, canonical_hash
from .protocol import messages, PROMPT_VERSION
from .reward import score, VERSION
from .model import load

def aggregate(rows):
    valid=[r for r in rows if r['metrics']['valid']]
    def mean(xs): return statistics.mean(xs) if xs else None
    return {'episodes':len(rows),'validity_rate':mean([r['metrics']['valid'] for r in rows]),
      'coverage_rate':mean([r['metrics']['coverage'] for r in rows]),
      'exact_partition_rate':mean([r['metrics']['exact_partition'] for r in rows]),
      'mean_reward_all':mean([r['metrics']['reward'] for r in rows]),
      'mean_ari_valid':mean([r['metrics']['ari'] for r in valid]),
      'mean_pair_f1_valid':mean([r['metrics']['pair_f1'] for r in valid]),
      'degenerate_rate_valid':mean([r['metrics']['one_group'] or r['metrics']['all_singletons'] for r in valid]),
      'latency_seconds_mean':mean([r['latency_seconds'] for r in rows]),
      'completion_tokens_mean':mean([r['completion_tokens'] for r in rows])}

def infer(model,tokenizer,ep,*,max_new_tokens=256,seed=17,sample=False):
    import torch
    from transformers import set_seed
    set_seed(seed)
    prompt=tokenizer.apply_chat_template(messages(ep),tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=tokenizer(prompt,return_tensors='pt').to(model.device)
    if inputs.input_ids.shape[1]>12000: raise ValueError('Prompt exceeds explicit local bound; never truncate tabs')
    kwargs=dict(max_new_tokens=max_new_tokens,do_sample=sample,pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,use_cache=True)
    if sample: kwargs.update(temperature=0.8,top_p=1.0,top_k=0)
    torch.cuda.synchronize(); start=time.perf_counter()
    with torch.inference_mode(): generated=model.generate(**inputs,**kwargs)[0,inputs.input_ids.shape[1]:]
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    tokens=generated.tolist(); complete=tokenizer.eos_token_id in tokens
    text=tokenizer.decode(generated,skip_special_tokens=True)
    return {'id':ep['id'],'family':ep.get('family',ep['id']),'tab_count':ep['tab_count'],'split':ep['split'],'seed':seed,'prompt_tokens':inputs.input_ids.shape[1],
      'prompt_hash':hashlib.sha256(prompt.encode()).hexdigest(),'completion_tokens':len(tokens),
      'reasoning_tokens':None,'thinking_enabled':False,'complete':complete,'output':text,
      'metrics':score(text,ep['reference']['assignments'],complete=complete),'latency_seconds':elapsed}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',default='validation',choices=['train','validation','final_test','validation_long','final_test_long'])
    parser.add_argument('--data',default='data/v1'); parser.add_argument('--adapter')
    parser.add_argument('--out',required=True); parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--max-tabs',type=int,default=30); parser.add_argument('--max-new-tokens',type=int,default=256)
    parser.add_argument('--repeats',type=int,default=1); parser.add_argument('--sample',action='store_true')
    parser.add_argument('--final-test',action='store_true',help='Explicit final evaluation after checkpoint selection is finished')
    args=parser.parse_args()
    if args.split.startswith('final_test') and not args.final_test: parser.error('Use --final-test only after checkpoint selection is complete')
    if not 1<=args.max_new_tokens<=4000: parser.error('Student completion cap must be 1..4000')
    destination=pathlib.Path(args.out); destination.mkdir(parents=True,exist_ok=False)
    episodes=[e for e in load_split(args.data,args.split) if e['tab_count']<=args.max_tabs]
    if args.limit: episodes=episodes[:args.limit]
    if not episodes: parser.error('Empty evaluation selection')
    import torch
    start=time.perf_counter(); model,tokenizer,audit=load(args.adapter)
    torch.cuda.reset_peak_memory_stats(); rows=[]
    with (destination/'predictions.jsonl').open('w',encoding='utf-8') as file:
        for ep in episodes:
            for repeat in range(args.repeats):
                row=infer(model,tokenizer,ep,max_new_tokens=args.max_new_tokens,seed=17+repeat,sample=args.sample)
                rows.append(row); file.write(json.dumps(row,ensure_ascii=False)+'\n'); file.flush()
                print(json.dumps({k:row[k] for k in ('id','seed','completion_tokens','latency_seconds','metrics')}),flush=True)
    elapsed=time.perf_counter()-start
    result={'model':audit,'adapter':args.adapter,'prompt_version':PROMPT_VERSION,'reward_version':VERSION,
      'decoding':{'max_new_tokens':args.max_new_tokens,'sample':args.sample,'temperature':0.8 if args.sample else None,'thinking':False,'constrained_json':False},
      'dataset_manifest_hash':hashlib.sha256((pathlib.Path(args.data)/'manifest.json').read_bytes()).hexdigest(),
      'selection_hash':canonical_hash([e['id'] for e in episodes]),'metrics':aggregate(rows),
      'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
      'wall_seconds':elapsed,'local_gpu_hours':elapsed/3600,'provider_compute_cost_usd':0,
      'electricity_cost':'not measured','reasoning_token_accounting':'No separate tokenizer signal; non-thinking mode explicitly set'}
    (destination/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
