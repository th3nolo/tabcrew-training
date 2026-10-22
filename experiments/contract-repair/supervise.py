"""Linux parent resource guard and exclusive GPU job lock for this repair task.

Usage: supervise.py train|original|repaired|expanded
Needs the CR_* variables described in launch-train.sh / launch-eval.sh. Logs, the lock file and
a supervisor receipt are written to $CR_OUT.
"""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
work=Path(__file__).resolve().parent;out=Path(os.environ['CR_OUT']);out.mkdir(parents=True,exist_ok=True)
mode=sys.argv[1];assert mode in ('train','original','repaired','expanded')
lock=(out/'gpu-job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def descendants(pid):
 pending=[pid];seen=set()
 while pending:
  p=pending.pop()
  if p in seen:continue
  seen.add(p)
  try:pending.extend(map(int,Path(f'/proc/{p}/task/{p}/children').read_text().split()))
  except (OSError,ValueError):pass
 return seen
def rss(p):
 try:
  for line in Path(f'/proc/{p}/status').read_text().splitlines():
   if line.startswith('VmRSS:'):return int(line.split()[1])*1024
 except OSError:pass
 return 0
command=['bash',str(work/('launch-train.sh' if mode=='train' else 'launch-eval.sh'))]
if mode!='train':command.append(mode)
start=time.time();peak=0;reason=None
with (out/(mode+'.log')).open('a') as log:
 child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 while child.poll() is None:
  members=descendants(child.pid);peak=max(peak,sum(rss(p) for p in members))
  if peak>4*2**30:reason='process_tree_rss_over_4_gib'
  if time.time()-start>7200:reason='wall_over_120_minutes'
  if reason:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   break
  time.sleep(1)
receipt={'mode':mode,'pid':child.pid,'exit_code':child.returncode,'reason':reason,'seconds':time.time()-start,'max_rss_gib':peak/2**30}
(out/(mode+'-supervisor.json')).write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
sys.exit(child.returncode or (1 if reason else 0))
