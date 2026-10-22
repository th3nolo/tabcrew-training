"""Process-local loader adapter, re-imported by SGLang's spawn workers."""
import inspect
import os
import sys
import sglang.srt.model_loader.loader as loader
from stream_weights import safetensors_weights_iterator

assert str(inspect.signature(loader.safetensors_weights_iterator)).count('hf_weights_files')==1
loader.safetensors_weights_iterator=safetensors_weights_iterator

if __name__=='__main__':
    from sglang.launch_server import prepare_server_args,run_server,kill_process_tree
    from sglang.srt.plugins import load_plugins
    load_plugins()
    args=prepare_server_args(sys.argv[1:])
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(),include_parent=False)
