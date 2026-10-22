"""Export an adapter with a portable pinned base-model reference, leaving training artifacts intact."""
import argparse,hashlib,json,pathlib,shutil
from tabcrew_rl.model import MODEL_ID,REVISION,model_dir
p=argparse.ArgumentParser();p.add_argument('--adapter',required=True);p.add_argument('--out',required=True);a=p.parse_args()
# The base-model license is the LICENSE file that scripts/download_model.py saves with the pinned snapshot.
license_path=model_dir()/'LICENSE'
if not license_path.exists():raise FileNotFoundError('Pinned base-model license is missing; run scripts/download_model.py first')
src=pathlib.Path(a.adapter);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False)
config=json.loads((src/'adapter_config.json').read_text());config['base_model_name_or_path']=MODEL_ID;config['revision']=REVISION
(out/'adapter_config.json').write_text(json.dumps(config,indent=2))
shutil.copy2(src/'adapter_model.safetensors',out/'adapter_model.safetensors')
shutil.copy2(license_path,out/'LICENSE')
hashes={}
for path in out.iterdir():
    with path.open('rb') as f:hashes[path.name]=hashlib.file_digest(f,'sha256').hexdigest()
(out/'manifest.json').write_text(json.dumps({'base_model':MODEL_ID,'revision':REVISION,'files':hashes,
  'format':'PEFT LoRA adapter; original base weights are downloaded separately at the exact revision',
  'load':'Load Qwen3_5ForCausalLM at the pinned revision, then PeftModel.from_pretrained(model, adapter_path)',
  'intended_use':'Local research preview, not validated product automation'},indent=2))
print(json.dumps({'export':str(out),'base_revision':REVISION,'hashes':hashes}))
