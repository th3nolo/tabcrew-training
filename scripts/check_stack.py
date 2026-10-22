import inspect, json, pathlib
import torch, transformers, trl, peft
from transformers import Qwen3_5ForCausalLM
from trl import GRPOConfig, GRPOTrainer
from tabcrew_rl.model import local_environment
local_environment()
fields=GRPOConfig.__dataclass_fields__
report={'torch':torch.__version__,'transformers':transformers.__version__,'trl':trl.__version__,'peft':peft.__version__,
        'cuda_available':torch.cuda.is_available(),'cuda_version':torch.version.cuda,
        'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'gpu_memory_free_total':list(torch.cuda.mem_get_info()) if torch.cuda.is_available() else None,
        'qwen_class':Qwen3_5ForCausalLM.__name__,
        'grpo_config_fields':{n:n in fields for n in ['max_completion_length','beta','loss_type','mask_truncated_completions','scale_rewards']}}
pathlib.Path('evidence').mkdir(exist_ok=True)
pathlib.Path('evidence/stack-check.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
src=inspect.getsource(GRPOTrainer)
pathlib.Path('evidence/grpo-source-inspection.txt').write_text('\n'.join(f'{i}: {l}' for i,l in enumerate(src.splitlines(),1) if any(s in l for s in ['reward_kwargs','completion_ids','is_eos','is_truncated','max_completion_length','calculate_rewards'])))
