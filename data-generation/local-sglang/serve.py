"""Own one isolated SGLang process group, with a durable STOP flag and offline model loading.

Runs inside WSL/Linux. --runtime is the folder that holds the SGLang .venv, uv.lock, the CUDA
toolkit links and the model folder named in assets.json (local_directory, relative to --runtime).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from linux_memory import sample
from runtime_cache import trim

ROOT=Path(__file__).resolve().parent
def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2))
    temp.replace(path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--port',type=int,default=60647)
    p.add_argument('--runtime',required=True)
    p.add_argument('--lock-sha256',required=True,help='SHA-256 of the reviewed runtime uv.lock')
    p.add_argument('--run-id',required=True)
    p.add_argument('--deadline',type=float,required=True)
    args=p.parse_args()
    if not time.time()<args.deadline<=time.time()+8*3600+240:raise RuntimeError('Invalid runtime deadline')
    # The original run also checked here that the previous collector revision had stopped and released the GPU.
    RUNTIME=Path(args.runtime)
    os.environ['TABCREW_SGLANG_RUNTIME']=str(RUNTIME)
    assert hashlib.sha256((RUNTIME/'uv.lock').read_bytes()).hexdigest()==args.lock_sha256
    stop=ROOT/'RUNTIME_STOP'
    if stop.exists():
        raise RuntimeError('RUNTIME_STOP is set; explicit resume is required')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',args.port))
    assets=json.loads((ROOT/'assets.json').read_text())
    assert assets['status']=='acquired_and_verified'
    model=RUNTIME/assets['local_directory']
    assert model.resolve().is_relative_to(RUNTIME.resolve())
    for entry in assets['files']:
        path=model/entry['file']
        assert path.stat().st_size==entry['bytes']
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):
                digest.update(block)
                # This read-only integrity scan must not retain gigabytes of file cache in WSL.
                # Discard only clean cache pages for this exact model file after consuming them.
                os.posix_fadvise(stream.fileno(),stream.tell()-len(block),len(block),os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        assert digest.hexdigest()==entry['sha256'],entry['file']
    env={k:v for k,v in os.environ.items() if k in {'PATH','HOME','USER','LOGNAME','LANG','LC_ALL','TMPDIR','WSL_INTEROP'}}
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',
               NO_PROXY='127.0.0.1,localhost',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='6',
               HF_HOME=str(RUNTIME/'hf-cache'),XDG_CACHE_HOME=str(RUNTIME/'runtime-cache'),
               TRITON_CACHE_DIR=str(RUNTIME/'triton-cache'),TORCH_EXTENSIONS_DIR=str(RUNTIME/'torch-extensions'))
    cuda=RUNTIME/'cuda-toolkit'
    installed=RUNTIME/'.venv/lib/python3.12/site-packages/nvidia/cu13'
    assert cuda.is_dir()
    for name,target in {'bin':'bin','include':'include','nvvm':'nvvm','lib64':'lib'}.items():
        link=cuda/name
        expected=installed/target
        assert expected.is_dir()
        assert link.exists()
        assert link.resolve()==expected.resolve()
    env.update(CUDA_HOME=str(cuda),CUDA_PATH=str(cuda),
               PATH=str(cuda/'bin')+os.pathsep+env.get('PATH',''),
               TVM_FFI_CACHE_DIR=str(RUNTIME/'tvm-ffi-cache'),MAX_JOBS='2',TORCHINDUCTOR_COMPILE_THREADS='1')
    link_lib=RUNTIME/'cuda-link-libraries'
    assert link_lib.is_dir()
    runtime_library=installed/'lib/libcudart.so.13'
    assert runtime_library.is_file()
    library_link=link_lib/'libcudart.so'
    assert library_link.exists()
    assert library_link.resolve()==runtime_library.resolve()
    env.update(LIBRARY_PATH=str(link_lib)+os.pathsep+env.get('LIBRARY_PATH',''),
               LD_LIBRARY_PATH=str(installed/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH',''))
    command=[str(RUNTIME/'.venv/bin/python'),str(ROOT/'entrypoint.py'),'--model-path',str(model),
             '--tokenizer-path',str(model),'--served-model-name','qwen3-local-teacher',
             '--host','127.0.0.1','--port',str(args.port),'--dtype','float16',
             '--load-format','safetensors','--context-length','6144',
             '--model-loader-extra-config','{"enable_multithread_load":false}',
             '--weight-loader-drop-cache-after-load',
             '--mem-fraction-static','0.78','--max-running-requests','4','--max-total-tokens','16384',
             '--chunked-prefill-size','1024','--cuda-graph-max-bs-decode','4',
             '--cuda-graph-backend-prefill','disabled',
             '--attention-backend','triton','--sampling-backend','pytorch','--grammar-backend','xgrammar',
             '--enable-metrics','--disable-custom-all-reduce','--mm-feature-transport','cpu',
             '--default-chat-template-kwargs','{"enable_thinking":false}']
    label='sglang-v16-streaming-four-16k'
    log_path=ROOT/'logs'/(str(time.time_ns())+'-'+label+'-runtime.log')
    with log_path.open('a') as log:
        child=subprocess.Popen(command,env=env,cwd=RUNTIME,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state={'pid':child.pid,'wrapper_pid':os.getpid(),'port':args.port,'run_id':args.run_id,'command':command,'started_unix':time.time(),'state':'starting','compiler_threads':1,
               'model_repository':assets['repository'],'model_revision':assets['revision'],
               'model_assets_sha256':hashlib.sha256((ROOT/'assets.json').read_bytes()).hexdigest(),
               'lock_sha256':hashlib.sha256((RUNTIME/'uv.lock').read_bytes()).hexdigest()}
        write(ROOT/'runtime.json',state)
        try:
            last_sample=0
            cache_released=False
            library_cache_released=False
            while child.poll() is None and not stop.exists() and time.time()<args.deadline:
                if not library_cache_released and 'Load weight begin' in log_path.read_text(errors='replace'):
                    trim(ROOT,'before_weight_load')
                    library_cache_released=True
                if not cache_released and 'Load weight end' in log_path.read_text(errors='replace'):
                    released=[]
                    for entry in assets['files']:
                        if entry['file'].endswith('.safetensors'):
                            with (model/entry['file']).open('rb') as checkpoint:
                                os.posix_fadvise(checkpoint.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                            released.append(entry['file'])
                    write(ROOT/'postload-cache-release.json',{'unix':time.time(),'files':released})
                    trim(ROOT,'after_weight_load')
                    cache_released=True
                if time.time()-last_sample>=3:
                    sample(child.pid,ROOT/'linux-resources.jsonl')
                    last_sample=time.time()
                time.sleep(.5)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
            for entry in assets['files']:
                if entry['file'].endswith('.safetensors'):
                    with (model/entry['file']).open('rb') as checkpoint:
                        os.posix_fadvise(checkpoint.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            trim(ROOT,'after_shutdown')
            state.update(state='stopped',returncode=child.returncode,ended_unix=time.time())
            write(ROOT/'runtime.json',state)
        if not stop.exists() and child.returncode:
            raise SystemExit(child.returncode)

if __name__=='__main__':
    main()
