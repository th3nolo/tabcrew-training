"""Read only our saved optimizer checkpoints; export loss history without touching training.

Watches $CR_OUT/training for up to four hours and writes loss-history.json there.
"""
import json,time,os
from pathlib import Path
import torch
out=Path(os.environ['CR_OUT'])/'training'
last=None
for _ in range(1440):
 try:
  summary=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else None
  resume=json.loads((out/'resume.json').read_text()) if (out/'resume.json').exists() else None
  key='complete' if summary else resume.get('directory') if resume else None
  if key and key!=last:
   losses=summary['losses'] if summary else torch.load(out/key/'optimizer.pt',map_location='cpu',weights_only=True)['losses']
   payload={'source':'saved_training_checkpoint','through_step':len(losses),'losses':losses,'exported_at':time.time()}
   tmp=out/'loss-history.pending';tmp.write_text(json.dumps(payload));os.replace(tmp,out/'loss-history.json');last=key
  if summary:break
 except (OSError,ValueError):pass
 time.sleep(10)
