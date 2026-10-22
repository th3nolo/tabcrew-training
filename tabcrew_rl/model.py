"""Exact Qwen text checkpoint loading; no remote Python execution or paid endpoints."""
import os
from pathlib import Path

MODEL_ID='Qwen/Qwen3.5-0.8B'
REVISION='2fc06364715b967f1860aea9cf38778875588b17'

def work_dir():
    """Local cache and model directory. Set TABCREW_WORK_DIR to use another place; default is <repo>/work."""
    return Path(os.environ.get('TABCREW_WORK_DIR') or Path(__file__).resolve().parents[1]/'work')

def model_dir():
    """Snapshot written by scripts/download_model.py for the pinned revision."""
    return work_dir()/'models'/REVISION

def local_environment():
    root=work_dir()
    os.environ.setdefault('HF_HOME',str(root/'hf-cache'))
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
    os.environ.setdefault('HF_HUB_DISABLE_XET','1')
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    os.environ.setdefault('WANDB_MODE','disabled')

def load(adapter=None, train=False):
    local_environment()
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no silent CPU training fallback')
    torch.set_num_threads(4)
    snapshot=model_dir()
    source=str(snapshot) if (snapshot/'download-receipt.json').exists() else MODEL_ID
    tokenizer=AutoTokenizer.from_pretrained(source,revision=REVISION,trust_remote_code=False)
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    # Explicit class loads the text backbone from the multimodal release. Inspect load audit.
    model, info=Qwen3_5ForCausalLM.from_pretrained(source,revision=REVISION,
          dtype=torch.bfloat16,device_map={'':'cuda:0'},trust_remote_code=False,
          attn_implementation='sdpa',output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise RuntimeError(f'Incomplete checkpoint load: {info}')
    unexpected=info.get('unexpected_keys',[])
    if any('visual.' not in k and 'vision' not in k and 'mtp' not in k for k in unexpected):
        raise RuntimeError(f'Unexpected non-vision checkpoint keys: {unexpected[:10]}')
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,adapter,is_trainable=train)
    model.eval()
    return model,tokenizer,{'model':MODEL_ID,'revision':REVISION,'loader':'Qwen3_5ForCausalLM',
           'dtype':'bfloat16','missing_keys':sorted(info.get('missing_keys',[])),'ignored_non_text_keys':len(unexpected),
           'parameters':sum(p.numel() for p in model.parameters())}
