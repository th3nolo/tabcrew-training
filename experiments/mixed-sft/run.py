"""Offline, resumable SFT and matched validation for the mixed teacher snapshot."""
import argparse
import gc
import math
import os
import random
import statistics
import subprocess
import sys
import time
import uuid
from common import *

_runtime_lock=None
def take_runtime_lock():
    global _runtime_lock
    import msvcrt
    if _runtime_lock is not None:return
    handle=(RUN/'runtime.lock').open('a+b');handle.seek(0)
    if handle.read(1)==b'':handle.write(b'0');handle.flush()
    handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    _runtime_lock=handle

def encoded_training(tok):
    from tabcrew_rl.protocol import messages,canonical_reference
    episodes=load_split(DATA,'train');random.Random(42).shuffle(episodes)
    result=[]
    for ep in episodes:
        prompt=tok.apply_chat_template(messages(ep),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        target=canonical_reference(ep['reference'],[tab['ref'] for tab in ep['tabs']])
        prompt_ids=tok.encode(prompt,add_special_tokens=False)
        target_ids=tok.encode(json.dumps(target,separators=(',',':')),add_special_tokens=False)+[tok.eos_token_id]
        if len(prompt_ids)>4096 or len(target_ids)>512:raise ValueError('No truncation allowed')
        result.append((ep['id'],prompt_ids,target_ids))
    return result

def model_for_training(adapter=None):
    from tabcrew_rl.model import load
    from peft import LoraConfig,get_peft_model
    from transformers import set_seed
    set_seed(42)
    model,tok,audit=load(adapter,train=bool(adapter))
    if adapter is None:
        model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    return model,tok,audit

def preflight():
    take_runtime_lock()
    import torch
    from batching import collate_completion_batch,completion_loss
    if (RUN/'run-plan.json').exists():raise ValueError('Plan already frozen')
    model,tok,audit=model_for_training()
    encoded=encoded_training(tok)
    if not encoded:raise ValueError('No training data')
    longest=sorted(encoded,key=lambda x:len(x[1])+len(x[2]),reverse=True)
    tests=[];chosen=None
    for size in (4,2,1):
        items=longest[:size]
        try:
            model.eval();torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                batch=collate_completion_batch(items,tok.pad_token_id,model.device)
                actual=float(completion_loss(model,batch))
                individual=[]
                for item in items:
                    individual.append(float(completion_loss(model,collate_completion_batch([item],tok.pad_token_id,model.device))))
            expected=statistics.mean(individual)
            if not math.isclose(actual,expected,rel_tol=.015,abs_tol=.02):raise ValueError('Padded completion loss differs from independent examples')
            model.train();model.zero_grad(set_to_none=True)
            start=time.perf_counter();loss=completion_loss(model,batch);loss.backward()
            torch.cuda.synchronize()
            peak=torch.cuda.max_memory_allocated();seconds=time.perf_counter()-start
            if not torch.isfinite(loss):raise ValueError('Nonfinite preflight loss')
            tests.append({'batch_size':size,'batch_loss':actual,'individual_mean_loss':expected,
                'max_cuda_allocated_bytes':peak,'forward_backward_seconds':seconds,'passed':True})
            model.zero_grad(set_to_none=True);del loss,batch
            if peak>6.5*1024**3:
                gc.collect();torch.cuda.empty_cache();continue
            chosen=size;break
        except torch.cuda.OutOfMemoryError:
            tests.append({'batch_size':size,'passed':False,'reason':'cuda_out_of_memory'})
            model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    if chosen is None:raise RuntimeError('No training batch fits the memory budget')
    from tabcrew_rl.model import model_dir
    sources=[ROOT/name for name in ('common.py','prepare.py','batching.py','run.py','report.py')]
    sources += [PROJECT/'tabcrew_rl'/name for name in ('model.py','protocol.py','reward.py','evaluate.py','data.py')]
    sources += [PROJECT/'uv.lock',PROJECT/'pyproject.toml']
    plan={'created_at':utc(),'dataset_manifest_sha256':sha(DATA/'manifest.json'),
        'evaluation_manifest_sha256':sha(DATA/'evaluation'/'manifest.json'),
        'source_sha256':{str(path):sha(path) for path in sources},
        'model':audit,'model_receipt_sha256':sha(model_dir()/'download-receipt.json'),
        'batch_size':chosen,'epochs':1,'seed':42,'learning_rate':2e-4,'lora_rank':8,'lora_alpha':16,
        'train_count':len(encoded),'optimizer_steps':math.ceil(len(encoded)/chosen),'preflight':tests,
        'evaluation_count':len(load_split(DATA/'evaluation','validation')),
        'stages':['sft','new-validation','old-sft-validation','old-grpo-validation'],
        'decoding':{'max_new_tokens':512,'do_sample':False,'thinking':False,'constrained_json':False},
        'selection':'One new full-epoch SFT candidate. Compare against both original adapters on the same fixed validation sample. No new GRPO or hyperparameter search.',
        'final_test':'Sealed. No final-test predictions or checkpoint selection.',
        'network':'Offline cached model only; no training/evaluation API calls or external reporting',
        'resource_policy':'Checkpoint and pause if requested, low host memory, or 180 training minutes in one launch. Resume retains optimizer and RNG state.',
        'old_adapter_sha256':{str(path):sha(path) for kind in ('sft','grpo') for path in (OLD/kind/'adapter').glob('*') if path.is_file()}}
    write(RUN/'run-plan.json',plan)
    write(RUN/'status.json',{'state':'prepared','updated_at':utc(),'train_count':len(encoded),'batch_size':chosen})
    print(json.dumps({'batch_size':chosen,'steps':plan['optimizer_steps'],'preflight':tests,'validation_count':plan['evaluation_count']}),flush=True)

def verify_plan():
    plan=read(RUN/'run-plan.json')
    for path,expected in {**plan['source_sha256'],**plan['old_adapter_sha256']}.items():
        if sha(path)!=expected:raise ValueError('Frozen source or baseline changed: '+path)
    if sha(DATA/'manifest.json')!=plan['dataset_manifest_sha256']:raise ValueError('Dataset changed')
    if sha(DATA/'evaluation'/'manifest.json')!=plan['evaluation_manifest_sha256']:raise ValueError('Evaluation changed')
    from tabcrew_rl.model import model_dir
    if sha(model_dir()/'download-receipt.json')!=plan['model_receipt_sha256']:raise ValueError('Model receipt changed')
    return plan

def low_host_memory():
    if os.name!='nt':return False
    import ctypes
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
    value=Memory();value.length=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):raise OSError('Host memory check failed')
    return value.available<1.5*1024**3 or value.page_available<2*1024**3

def train():
    take_runtime_lock()
    import torch
    from batching import collate_completion_batch,completion_loss
    plan=verify_plan();out=RUN/'sft';out.mkdir(exist_ok=True)
    if (out/'summary.json').exists() and read(out/'summary.json')['full_epoch_complete']:return
    checkpoint=read(out/'resume.json') if (out/'resume.json').exists() else None
    resume_path=out/checkpoint['directory'] if checkpoint else None
    if checkpoint:
        for name,expected in checkpoint['sha256'].items():
            if sha(resume_path/name)!=expected:raise ValueError('Resume checkpoint changed')
    model,tok,audit=model_for_training(str(resume_path/'adapter') if checkpoint else None)
    encoded=encoded_training(tok);batch_size=plan['batch_size']
    parameters=[p for p in model.parameters() if p.requires_grad]
    initial=[p.detach().cpu().float().clone() for p in parameters]
    optimizer=torch.optim.AdamW(parameters,lr=plan['learning_rate'])
    completed=0;elapsed_before=0;losses=[]
    if checkpoint:
        saved=torch.load(resume_path/'optimizer.pt',map_location='cpu',weights_only=True)
        optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng_cpu']);torch.cuda.set_rng_state_all(saved['rng_cuda'])
        completed=saved['completed'];elapsed_before=saved['elapsed'];losses=saved['losses']
        del saved
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats();model.train()
    stop_reason=None
    def parameter_delta():
        return sum((p.detach().cpu().float()-before).square().sum().item() for p,before in zip(parameters,initial))**.5
    def save_checkpoint():
        directory=out/f'checkpoint-{completed:05d}-{uuid.uuid4().hex[:8]}'
        directory.mkdir(exist_ok=False)
        model.save_pretrained(directory/'adapter');tok.save_pretrained(directory/'adapter')
        torch.save({'optimizer':optimizer.state_dict(),'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),
            'completed':completed,'elapsed':elapsed_before+time.perf_counter()-start,'losses':losses},directory/'optimizer.pt')
        hashes={str(path.relative_to(directory)):sha(path) for path in directory.rglob('*') if path.is_file()}
        write(out/'resume.json',{'directory':directory.name,'completed':completed,'sha256':hashes,
            'nonzero_update':bool((checkpoint or {}).get('nonzero_update')) or parameter_delta()>0})
    with (out/'learning-curve.jsonl').open('a',encoding='utf-8') as log:
        while completed<len(encoded):
            if (RUN/'PAUSE').exists() or low_host_memory():
                stop_reason='manual_pause' if (RUN/'PAUSE').exists() else 'low_host_memory';break
            items=encoded[completed:completed+batch_size]
            batch=collate_completion_batch(items,tok.pad_token_id,model.device)
            optimizer.zero_grad(set_to_none=True);loss=completion_loss(model,batch)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(parameters,1,error_if_nonfinite=True);optimizer.step()
            completed+=len(items);losses.append(float(loss));elapsed=elapsed_before+time.perf_counter()-start
            step=math.ceil(completed/batch_size)
            row={'step':step,'total_steps':plan['optimizer_steps'],'examples_seen':completed,'total_examples':len(encoded),
                'loss':losses[-1],'grad_norm':float(grad),'seconds':elapsed,'updated_at':utc(),
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                'eta_seconds':elapsed/max(1,completed)*(len(encoded)-completed)}
            log.write(json.dumps(row)+'\n');log.flush();write(out/'progress.json',{'state':'training',**row})
            if step%10==0:print(json.dumps(row),flush=True)
            del loss,batch
            if step%100==0:save_checkpoint()
            if time.perf_counter()-start>10800 and completed<len(encoded):stop_reason='time_checkpoint_pause';break
    if stop_reason:
        latest=read(out/'resume.json') if (out/'resume.json').exists() else None
        if not latest or completed!=latest['completed']:save_checkpoint()
        write(out/'progress.json',{'state':'paused','reason':stop_reason,'examples_seen':completed,'total_examples':len(encoded),'updated_at':utc()})
        raise SystemExit(75)
    model.save_pretrained(out/'adapter');tok.save_pretrained(out/'adapter')
    summary={'full_epoch_complete':True,'unique_training_examples_seen':completed,'optimizer_steps':math.ceil(completed/batch_size),
        'session_parameter_l2_delta':parameter_delta(),
        'nonzero_update':bool((checkpoint or {}).get('nonzero_update')) or parameter_delta()>0,
        'batch_size':batch_size,'wall_seconds':elapsed_before+time.perf_counter()-start,
        'initial_20_mean_loss':statistics.mean(losses[:20]),'last_20_mean_loss':statistics.mean(losses[-20:]),
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'provider_compute_cost_usd':0,'model':audit,
        'adapter_sha256':{path.name:sha(path) for path in (out/'adapter').glob('*') if path.is_file()}}
    write(out/'summary.json',summary);write(out/'progress.json',{'state':'completed','updated_at':utc(),**summary})

def evaluate_stage(name):
    take_runtime_lock();verify_plan()
    import runpy
    adapters={'new':RUN/'sft'/'adapter','old-sft':OLD/'sft'/'adapter','old-grpo':OLD/'grpo'/'adapter'}
    sys.argv=['tabcrew_rl.evaluate','--data',str(DATA/'evaluation'),'--split','validation',
        '--out',str(RUN/(name+'-validation')),'--adapter',str(adapters[name]),'--max-tabs','30','--max-new-tokens','512']
    runpy.run_module('tabcrew_rl.evaluate',run_name='__main__')

def pipeline():
    plan=verify_plan()
    # Lock is held by this process and released by the OS even after an interruption.
    import msvcrt
    lock=(RUN/'pipeline.lock').open('a+b');lock.seek(0)
    if lock.read(1)==b'':lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    stages=[('sft',[str(ROOT/'run.py'),'train'])]
    for name,adapter in [('new',RUN/'sft'/'adapter'),('old-sft',OLD/'sft'/'adapter'),('old-grpo',OLD/'grpo'/'adapter')]:
        stages.append((name+'-validation',[str(ROOT/'run.py'),'evaluate','--name',name]))
    try:
        for stage,command in stages:
            receipt_path=RUN/(stage+'.receipt.json')
            if receipt_path.exists() and read(receipt_path)['exit_code']==0:continue
            if (RUN/'PAUSE').exists():
                write(RUN/'status.json',{'state':'paused','stage':stage,'updated_at':utc(),'reason':'manual_pause'});return
            if stage!='sft' and (RUN/stage).exists():raise RuntimeError('Incomplete evaluation directory; inspect before retrying')
            with (RUN/(stage+'.log')).open('a',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,'-u',*command],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
                write(RUN/'status.json',{'state':'running','stage':stage,'updated_at':utc(),'supervisor_pid':os.getpid(),'child_pid':child.pid})
                start=time.perf_counter();code=child.wait()
            receipt={'stage':stage,'pid':child.pid,'exit_code':code,'seconds':time.perf_counter()-start,'finished_at':utc()}
            write(receipt_path,receipt)
            if code:
                write(RUN/'status.json',{'state':'paused' if code==75 else 'failed','stage':stage,'updated_at':utc(),'exit_code':code});return
        from report import build
        build();write(RUN/'status.json',{'state':'completed','updated_at':utc(),'summary':'comparison.json'})
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('preflight','train','pipeline','evaluate'))
    parser.add_argument('--name',choices=('new','old-sft','old-grpo'))
    args=parser.parse_args()
    if args.action=='evaluate':
        if not args.name:parser.error('--name is required for evaluation')
        evaluate_stage(args.name)
    else:{'preflight':preflight,'train':train,'pipeline':pipeline}[args.action]()
