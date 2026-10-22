"""Bounded SFT control. Only completion tokens are supervised; frozen train split only."""
import argparse, hashlib, json, pathlib, random, time
from .model import load
from .data import load_split
from .protocol import messages, PROMPT_VERSION, canonical_reference, TARGET_SERIALIZATION_VERSION

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=24);p.add_argument('--data',default='data/v1')
    p.add_argument('--canonical-targets',action='store_true',help='Sort refs and relabel arbitrary group IDs consistently without changing partitions')
    a=p.parse_args()
    if not 1<=a.steps<=24: p.error('SFT pilot limited to 24 updates')
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False)
    import torch
    from peft import LoraConfig,get_peft_model
    from transformers import set_seed
    set_seed(42); model,tok,audit=load()
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.0,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False}); model.config.use_cache=False
    initial={n:t.detach().cpu().clone() for n,t in model.named_parameters() if t.requires_grad}
    eps=[e for e in load_split(a.data,'train') if e['tab_count']<=12]
    rng=random.Random(42);rng.shuffle(eps)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4)
    manifest={'kind':'sft_only_control','model':audit,'steps_requested':a.steps,'seed':42,'learning_rate':2e-4,
       'training_episode_ids':[e['id'] for e in eps],'teacher_used':any(bool(e['provenance'].get('teacher')) for e in eps),'prompt_version':PROMPT_VERSION,
       'supervision':'Frozen training references; prompt tokens masked. See reference statuses and dataset provenance.',
       'reference_statuses':sorted({e['reference_status'] for e in eps}),
       'target_serialization':TARGET_SERIALIZATION_VERSION if a.canonical_targets else 'original-reference-order-and-ids',
       'dataset_manifest_hash':hashlib.sha256((pathlib.Path(a.data)/'manifest.json').read_bytes()).hexdigest()}
    (out/'run-manifest.json').write_text(json.dumps(manifest,indent=2))
    torch.cuda.reset_peak_memory_stats(); start=time.perf_counter();model.train()
    with (out/'learning-curve.jsonl').open('w') as log:
        for step in range(a.steps):
            ep=eps[step%len(eps)]
            prefix=tok.apply_chat_template(messages(ep),tokenize=False,add_generation_prompt=True,enable_thinking=False)
            prompt_ids=tok.encode(prefix,add_special_tokens=False)
            target=canonical_reference(ep['reference'],[t['ref'] for t in ep['tabs']]) if a.canonical_targets else ep['reference']
            target_ids=tok.encode(json.dumps(target,separators=(',',':')),add_special_tokens=False)+[tok.eos_token_id]
            ids=torch.tensor([prompt_ids+target_ids],device=model.device)
            # Compute logits for completion positions only to avoid a large prompt-vocabulary tensor.
            optimizer.zero_grad(set_to_none=True)
            logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=len(target_ids)+1).logits[:,:-1,:]
            target=torch.tensor([target_ids],device=model.device)
            loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]).float(),target.reshape(-1))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite SFT loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            row={'step':step+1,'episode':ep['id'],'loss':loss.item(),'grad_norm':float(grad),'seconds':time.perf_counter()-start}
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            if time.perf_counter()-start>300: break
    model.save_pretrained(out/'adapter');tok.save_pretrained(out/'adapter')
    delta=sum((t.detach().cpu().float()-initial[n].float()).square().sum().item() for n,t in model.named_parameters() if n in initial)**0.5
    seconds=time.perf_counter()-start
    result={'optimizer_steps':step+1,'adapter_l2_delta':delta,'nonzero_update':delta>0,'wall_seconds':seconds,
      'local_gpu_hours':seconds/3600,'provider_compute_cost_usd':0,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
      'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'kind':'sft_only'}
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
