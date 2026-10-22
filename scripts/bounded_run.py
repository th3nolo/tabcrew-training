"""Terminate only this launched child at a hard timeout. No host runtime manipulation."""
import argparse, json, pathlib, subprocess, sys, time
p=argparse.ArgumentParser(); p.add_argument('--seconds',type=int,required=True); p.add_argument('--log',required=True)
p.add_argument('command',nargs=argparse.REMAINDER); a=p.parse_args()
command=a.command[1:] if a.command and a.command[0]=='--' else a.command
if not command or not 1<=a.seconds<=1800: p.error('Need command and timeout <=1800 seconds')
if command[0]=='python': command[0]=sys.executable
path=pathlib.Path(a.log); path.parent.mkdir(parents=True,exist_ok=True)
start=time.perf_counter()
with path.open('w',encoding='utf-8') as log:
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    try:
        code=child.wait(timeout=a.seconds); status='completed' if code==0 else 'failed'
    except subprocess.TimeoutExpired:
        child.kill(); child.wait(); code=124; status='timed_out'
result={'status':status,'exit_code':code,'seconds':time.perf_counter()-start,'command':command}
path.with_suffix('.receipt.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result)); sys.exit(code)
