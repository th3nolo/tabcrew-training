"""Matched original/repaired/expanded Qwen evaluation on frozen development cases.

Runs inside the bubblewrap sandbox from launch-eval.sh. /data/cases.json holds, per case,
the chat messages, the JSON schema used for constrained decoding, and their hashes
(keys: id, messages, schema, messages_sha256, schema_sha256). The cases file and the
production system prompt inside it are not part of this repository.
"""
import hashlib,json,os,sys,time
from pathlib import Path
import torch
import xgrammar as xgr
from transformers import AutoTokenizer,Qwen3_5ForCausalLM
from peft import PeftModel
mode=sys.argv[1];assert mode in ('original','repaired','expanded')
assert os.getuid()!=0 and len(Path('/proc/net/route').read_text().splitlines())==1
assert not Path('/mnt').exists()
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.72)
out=Path('/output');cases=json.loads(Path('/data/cases.json').read_text())['cases']
def save(name,v):
 p=out/name;t=p.with_suffix('.pending');t.write_text(json.dumps(v,indent=2,ensure_ascii=False,default=str));os.replace(t,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha('/qwen-base/model.safetensors-00001-of-00001.safetensors')=='04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
if mode=='original':assert sha('/adapter/adapter_model.safetensors')=='8ab0141ca7f1e59b64c5f48fe905cd31c3224cd6c5bd86604398546da3a9b8b1'
tok=AutoTokenizer.from_pretrained('/qwen-base',local_files_only=True,trust_remote_code=False)
model,info=Qwen3_5ForCausalLM.from_pretrained('/qwen-base',local_files_only=True,use_safetensors=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',output_loading_info=True)
assert not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs')
assert all(any(s in k for s in ['visual.','vision','mtp']) for k in info.get('unexpected_keys',[]))
model=PeftModel.from_pretrained(model,'/adapter',is_trainable=False,local_files_only=True,use_safetensors=True);model.eval()
vocab=getattr(model.config,'text_config',model.config).vocab_size
compiler=xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tok,vocab_size=vocab))
def generate(ep,limit=2048):
 prompt=tok.apply_chat_template(ep['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 ids=tok.encode(prompt,add_special_tokens=False);assert len(ids)+limit<=8192,'No truncation'
 matcher=xgr.GrammarMatcher(compiler.compile_json_schema(json.dumps(ep['schema'])))
 bitmask=xgr.allocate_token_bitmask(1,vocab);x=torch.tensor([ids],device='cuda');cache=None;tokens=[];finish='length';first=None
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
 with torch.inference_mode():
  for _ in range(limit):
   result=model(input_ids=x,past_key_values=cache,use_cache=True,logits_to_keep=1);cache=result.past_key_values;logits=result.logits[:,-1,:].float()
   matcher.fill_next_token_bitmask(bitmask);xgr.apply_token_bitmask_inplace(logits,bitmask.to(logits.device))
   token=int(logits.argmax(-1).item());assert matcher.accept_token(token);tokens.append(token)
   if first is None:torch.cuda.synchronize();first=time.perf_counter()-start
   if matcher.is_terminated():finish='eos';break
   x=torch.tensor([[token]],device='cuda')
 torch.cuda.synchronize();seconds=time.perf_counter()-start
 return {'text':tok.decode(tokens,skip_special_tokens=True),'input_tokens':len(ids),'output_tokens':len(tokens),'seconds':seconds,'output_tokens_per_second':len(tokens)/seconds,'time_to_first_token_seconds':first,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'finish_reason':finish,'messages_sha256':ep['messages_sha256'],'schema_sha256':ep['schema_sha256'],'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
results=[];path=out/(mode+'-results.json')
if path.exists():
 previous=json.loads(path.read_text());assert previous['adapter_sha256']==sha('/adapter/adapter_model.safetensors');assert previous['cases_sha256']==sha('/data/cases.json');results=previous['results']
assert len(results)==len({r['id'] for r in results})
generate(cases[0],8)
for ep in cases:
 if any(r['id']==ep['id'] for r in results):continue
 save(mode+'-status.json',{'phase':'running','completed':len(results),'planned':len(cases),'current_case':ep['id']})
 try:r=generate(ep)
 except Exception as e:r={'error':type(e).__name__+': '+str(e)}
 r.update(id=ep['id'],model=mode);results.append(r)
 save(mode+'-results.json',{'results':results,'adapter_sha256':sha('/adapter/adapter_model.safetensors'),'cases_sha256':sha('/data/cases.json'),'max_output_tokens':2048,'constraint':'same gateway schema via XGrammar'})
save(mode+'-status.json',{'phase':'completed','completed':len(results),'planned':len(cases)})
print(json.dumps({'model':mode,'completed':len(results)}),flush=True)
