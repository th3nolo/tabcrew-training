"""Bounded TRL GRPO. Rollouts are sampled from the student; no labels are fed to it."""
import argparse, hashlib, json, pathlib, time
from .model import load, MODEL_ID, REVISION
from .data import load_split, canonical_hash
from .protocol import messages, PROMPT_VERSION
from .reward import score, VERSION

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/grpo-pilot.json')
    p.add_argument('--data',default='data/v1'); p.add_argument('--out',required=True);p.add_argument('--init-adapter')
    a=p.parse_args(); cfg=json.loads(pathlib.Path(a.config).read_text())
    if cfg['model']!=MODEL_ID or cfg['revision']!=REVISION: raise ValueError('Model pin mismatch')
    if not 1<=cfg['max_completion_length']<=4000: raise ValueError('Student cap')
    if cfg['max_steps']>16 or cfg['max_wall_seconds']>900: raise ValueError('Pilot bounds exceeded; prepare a separate measured run proposal')
    out=pathlib.Path(a.out); out.mkdir(parents=True,exist_ok=False)
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer
    set_seed(cfg['seed']); model,tokenizer,audit=load(a.init_adapter,train=bool(a.init_adapter))
    names=[n for n,m in model.named_modules() if (isinstance(m,torch.nn.Linear) or hasattr(m,'lora_A')) and n.split('.')[-1] in cfg['target_modules']]
    missing=set(cfg['target_modules'])-{n.split('.')[-1] for n in names}
    if missing: raise ValueError(f'LoRA targets missing: {missing}')
    lora=LoraConfig(r=cfg['lora_r'],lora_alpha=cfg['lora_alpha'],lora_dropout=cfg['lora_dropout'],
                    target_modules=cfg['target_modules'],task_type='CAUSAL_LM',bias='none')
    if not a.init_adapter: model=get_peft_model(model,lora)
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    episodes=[e for e in load_split(a.data,'train') if e['tab_count']<=cfg['max_tabs']]
    records=[]
    for ep in episodes:
        prompt=tokenizer.apply_chat_template(messages(ep),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        if len(tokenizer.encode(prompt))>2048: raise ValueError('Prompt over bound: no truncation permitted')
        records.append({'prompt':prompt,'reference':json.dumps(ep['reference']['assignments']),'episode_id':ep['id']})
    if not records: raise ValueError('Empty training set')
    trainer_config=GRPOConfig(output_dir=str(out),max_steps=cfg['max_steps'],
        per_device_train_batch_size=1,gradient_accumulation_steps=cfg['num_generations'],
        num_generations=cfg['num_generations'],max_completion_length=cfg['max_completion_length'],
        learning_rate=cfg['learning_rate'],beta=cfg['beta'],temperature=0.8,top_p=1.0,top_k=0,
        bf16=True,fp16=False,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        optim='adamw_torch',logging_steps=1,save_strategy='no',report_to='none',
        use_vllm=False,seed=cfg['seed'],data_seed=cfg['seed'],dataloader_num_workers=0,
        remove_unused_columns=False,loss_type='grpo',scale_rewards='group',
        mask_truncated_completions=False,disable_tqdm=True)
    start=time.perf_counter(); reward_log=[]
    def reward(completions,reference,episode_id,completion_ids,**kwargs):
        complete=[tokenizer.eos_token_id in ids for ids in completion_ids]
        values=[score(c,json.loads(r),complete=done)['reward'] for c,r,done in zip(completions,reference,complete,strict=True)]
        row={'episode_ids':episode_id,'rewards':values,'completions':completions,
             'complete':complete,'completion_tokens':[len(ids) for ids in completion_ids]}
        reward_log.append(row)
        with (out/'rollouts.jsonl').open('a',encoding='utf-8') as file: file.write(json.dumps(row,ensure_ascii=False)+'\n')
        return values
    class Bound(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            if time.perf_counter()-start>=cfg['max_wall_seconds']: control.should_training_stop=True
            # Equal rewards contain no grouping-learning signal. Preserve evidence and stop after 3 groups.
            if len(reward_log)>=3 and all(len(set(r['rewards']))==1 for r in reward_log[-3:]):
                control.should_training_stop=True
            return control
        def on_log(self,args,state,control,logs=None,**kwargs):
            with (out/'learning-curve.jsonl').open('a') as f: f.write(json.dumps({'step':state.global_step,**(logs or {})})+'\n')
    manifest={'config':cfg,'model_load':audit,'matched_lora_modules':names,
      'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
      'training_episode_ids':[e['id'] for e in episodes],'selection_hash':canonical_hash([e['id'] for e in episodes]),
      'dataset_manifest_hash':hashlib.sha256((pathlib.Path(a.data)/'manifest.json').read_bytes()).hexdigest(),
      'prompt_version':PROMPT_VERSION,'reward_version':VERSION,'teacher_used':any(bool(e['provenance'].get('teacher')) for e in episodes),
      'reference_statuses':sorted({e['reference_status'] for e in episodes}),
      'initial_adapter':a.init_adapter,'kl_reference':'Original pretrained weights with adapter disabled' if cfg['beta'] else 'No KL term (beta=0)'}
    (out/'run-manifest.json').write_text(json.dumps(manifest,indent=2))
    torch.cuda.reset_peak_memory_stats()
    trainer=GRPOTrainer(model=model,args=trainer_config,processing_class=tokenizer,
      reward_funcs=reward,train_dataset=Dataset.from_list(records),callbacks=[Bound()])
    result=trainer.train()
    adapter=out/'adapter'; trainer.save_model(str(adapter)); tokenizer.save_pretrained(str(adapter))
    delta=sum((p.detach().cpu().float()-initial[n].float()).pow(2).sum().item() for n,p in model.named_parameters() if n in initial)**0.5
    seconds=time.perf_counter()-start
    summary={'metrics':result.metrics,'optimizer_steps':trainer.state.global_step,'adapter_l2_delta':delta,
      'nonzero_update':delta>0,'wall_seconds':seconds,'local_gpu_hours':seconds/3600,'provider_compute_cost_usd':0,
      'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
      'equal_reward_groups':sum(len(set(r['rewards']))==1 for r in reward_log),'rollout_groups':len(reward_log),
      'limit_note':'Wall guard checked between optimizer steps; launcher must enforce the hard timeout'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
