"""Offline full-contract continuation SFT; frozen labels, separate adapter, resumable.

Runs inside the bubblewrap sandbox from launch-train.sh. Fixed mount points:
/input (this folder), /data (train.jsonl + training-plan.json), /qwen-base (pinned base snapshot),
/adapter (mixed-SFT adapter to continue), /output (checkpoints and status files).
"""
import hashlib,json,math,os,random,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,Qwen3_5ForCausalLM,set_seed
from peft import PeftModel
from batching import collate_completion_batch,completion_loss

out=Path('/output');plan=json.loads(Path('/data/training-plan.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):
 p=out/name;t=p.with_suffix('.pending');t.write_text(json.dumps(v,indent=2,default=str));os.replace(t,p)
assert os.getuid()!=0
assert len(Path('/proc/net/route').read_text().splitlines())==1
assert 'CapEff:\t0000000000000000' in Path('/proc/self/status').read_text()
# Host drives and user SSH folders must not be visible inside the sandbox.
assert not Path('/mnt').exists() and not any(Path('/home').glob('*/.ssh'))
assert sha('/data/train.jsonl')==plan['labels_sha256']
assert sha('/adapter/adapter_model.safetensors')==plan['baseline_adapter_sha256']
assert sha('/qwen-base/model.safetensors-00001-of-00001.safetensors')==plan['base_weights_sha256']
for name,expected in plan['code_sha256'].items():assert sha('/input/'+name)==expected
if (out/'summary.json').exists():raise SystemExit('Completed checkpoint exists; refusing to overwrite')
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.72);set_seed(plan['seed'])
tok=AutoTokenizer.from_pretrained('/qwen-base',local_files_only=True,trust_remote_code=False)
rows=[json.loads(s) for s in Path('/data/train.jsonl').read_text().splitlines()]
assert len(rows)==plan['train_count']
random.Random(plan['seed']).shuffle(rows)
encoded=[]
for r in rows:
 prompt=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 p=tok.encode(prompt,add_special_tokens=False)
 t=tok.encode(json.dumps(r['target'],ensure_ascii=False,separators=(',',':')),add_special_tokens=False)+[tok.eos_token_id]
 assert len(p)+len(t)<=8192 and len(t)<=2048,'No truncation permitted'
 encoded.append((r['id'],p,t))
resume=json.loads((out/'resume.json').read_text()) if (out/'resume.json').exists() else None
adapter=out/resume['directory']/'adapter' if resume else Path('/adapter')
if resume:
 for name,expected in resume['hashes'].items():assert sha(out/resume['directory']/name)==expected
model,loading=Qwen3_5ForCausalLM.from_pretrained('/qwen-base',local_files_only=True,use_safetensors=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',output_loading_info=True)
assert not loading.get('missing_keys') and not loading.get('mismatched_keys') and not loading.get('error_msgs')
assert all(any(s in k for s in ['visual.','vision','mtp']) for k in loading.get('unexpected_keys',[]))
model=PeftModel.from_pretrained(model,str(adapter),is_trainable=True,local_files_only=True,use_safetensors=True)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
parameters=[p for p in model.parameters() if p.requires_grad]
initial=[p.detach().float().cpu().clone() for p in parameters]
optimizer=torch.optim.AdamW(parameters,lr=plan['learning_rate'])
completed=0;losses=[];start=time.monotonic()
if resume:
 state=torch.load(out/resume['directory']/'optimizer.pt',map_location='cpu',weights_only=True)
 optimizer.load_state_dict(state['optimizer']);completed=state['completed'];losses=state['losses'];torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
 del state
batch_size=plan['batch_size'];steps=math.ceil(len(encoded)/batch_size)*plan['epochs']
save('input-audit.json',{'max_prompt_tokens':max(len(e[1]) for e in encoded),'max_target_tokens':max(len(e[2]) for e in encoded),'steps':steps,'trainable_parameters':sum(p.numel() for p in parameters),'loading':loading,'network_isolated':True})
# Longest-batch forward/backward resource test, without optimizer mutation.
if not resume:
 longest=sorted(encoded,key=lambda e:len(e[1])+len(e[2]),reverse=True)[:batch_size]
 model.eval()
 with torch.no_grad():
  batched=float(completion_loss(model,collate_completion_batch(longest,tok.pad_token_id,'cuda')))
  singles=[float(completion_loss(model,collate_completion_batch([e],tok.pad_token_id,'cuda'))) for e in longest]
 assert math.isclose(batched,sum(singles)/len(singles),rel_tol=.015,abs_tol=.02),'Padded batch loss mismatch'
 model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
 loss=completion_loss(model,collate_completion_batch(longest,tok.pad_token_id,'cuda'));assert torch.isfinite(loss);loss.backward()
 save('preflight.json',{'batch_loss':batched,'independent_mean_loss':sum(singles)/len(singles),'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'finite_backward':True})
 optimizer.zero_grad(set_to_none=True);del loss;torch.cuda.empty_cache()

def checkpoint():
 directory=f'checkpoint-{completed:04d}';dest=out/directory
 dest.mkdir(exist_ok=False)
 model.save_pretrained(dest/'adapter',safe_serialization=True)
 torch.save({'optimizer':optimizer.state_dict(),'completed':completed,'losses':losses,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},dest/'optimizer.pt')
 hashes={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}
 save('resume.json',{'directory':directory,'completed':completed,'hashes':hashes})
model.train()
for step in range(completed,steps):
 offset=(step%math.ceil(len(encoded)/batch_size))*batch_size
 batch=collate_completion_batch(encoded[offset:offset+batch_size],tok.pad_token_id,'cuda')
 optimizer.zero_grad(set_to_none=True);loss=completion_loss(model,batch)
 assert torch.isfinite(loss),'Nonfinite loss'
 loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True);optimizer.step()
 losses.append(float(loss.detach()));completed=step+1
 del loss,batch
 save('status.json',{'phase':'training','completed':completed,'planned':steps,'last_loss':losses[-1],'elapsed_seconds':time.monotonic()-start})
 if completed%64==0 or completed==steps:checkpoint()
 if (out/'STOP').exists():
  if completed%64 and completed!=steps:checkpoint()
  save('status.json',{'phase':'paused','completed':completed,'planned':steps});raise SystemExit(0)
delta=sum(float((p.detach().float().cpu()-before).square().sum()) for p,before in zip(parameters,initial))**.5
assert delta>0,'No parameter update'
save('summary.json',{'completed':True,'train_count':len(rows),'optimizer_steps':completed,'epochs':plan['epochs'],'losses':losses,'seconds_this_launch':time.monotonic()-start,'parameter_delta_this_launch':delta,'adapter':f'checkpoint-{completed:04d}/adapter','plan':plan})
save('status.json',{'phase':'completed','completed':completed,'planned':steps})
print(json.dumps({'completed':completed,'parameter_delta':delta}),flush=True)
