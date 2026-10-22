"""Download only fixed-revision model assets, verify Git/LFS content hashes, execute no code."""
import hashlib, json, pathlib, time, urllib.request
from tabcrew_rl.model import MODEL_ID,REVISION,model_dir
root=pathlib.Path(__file__).resolve().parents[1]
target=model_dir(); target.mkdir(parents=True,exist_ok=True)
api=f'https://huggingface.co/api/models/{MODEL_ID}/tree/{REVISION}'
with urllib.request.urlopen(api,timeout=60) as r: entries=json.load(r)
allowed={'config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','merges.txt','vocab.json','LICENSE'}
rows=[]; started=time.perf_counter()
for entry in entries:
    name=entry['path']
    if name not in allowed and not name.endswith('.safetensors'): continue
    path=target/name
    def valid():
        if not path.exists() or path.stat().st_size!=entry['size']: return False
        if 'lfs' in entry:
            with path.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
            return actual==entry['lfs']['oid']
        digest=hashlib.sha1(f'blob {entry["size"]}\0'.encode()); digest.update(path.read_bytes())
        return digest.hexdigest()==entry['oid']
    if not valid():
        url=f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}'
        with urllib.request.urlopen(url,timeout=90) as src,path.open('wb') as dst:
            downloaded=0; next_report=128*1024*1024
            while chunk:=src.read(1024*1024):
                dst.write(chunk); downloaded+=len(chunk)
                if downloaded>=next_report:
                    print(json.dumps({'file':name,'MiB':downloaded//(1024*1024)}),flush=True); next_report+=128*1024*1024
    assert valid(),f'Hash mismatch: {name}'
    with path.open('rb') as file: digest=hashlib.file_digest(file,'sha256').hexdigest()
    rows.append({'file':name,'size':entry['size'],'sha256':digest})
    print(json.dumps({'file':name,'verified':True}),flush=True)
receipt={'model':MODEL_ID,'revision':REVISION,'files':rows,'download_seconds':time.perf_counter()-started}
(target/'download-receipt.json').write_text(json.dumps(receipt,indent=2))
(root/'evidence').mkdir(exist_ok=True)
(root/'evidence/model-download.json').write_text(json.dumps(receipt,indent=2))
