"""Freeze the four checkpoint identities, then run one fixed final evaluation of each."""
import argparse,datetime,hashlib,json,pathlib,subprocess,sys,time
root=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--results',default='results');a=p.parse_args();base=pathlib.Path(a.results)
arms=[('final-before',None),('final-pure-grpo',str(base/'grpo/adapter')),('final-sft',str(base/'sft/adapter')),('final-sft-grpo',str(base/'grpo-sft/adapter'))]
identities=[]
for name,adapter in arms:
    files={}
    if adapter:
        for path in (root/adapter).glob('adapter*'):
            with path.open('rb') as f:files[path.name]=hashlib.file_digest(f,'sha256').hexdigest()
        assert 'adapter_model.safetensors' in files
    identities.append({'name':name,'adapter':adapter,'hashes':files})
manifest={'locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arms':identities,
          'selection_rule':'All predeclared finished pilots, no final-test checkpoint selection',
          'split':'final_test','max_tabs':30,'decoding':'greedy, non-thinking, 256 completion tokens','training_seed':42,
          'frozen_dataset_manifest_hash':hashlib.sha256((root/'data/v1/manifest.json').read_bytes()).hexdigest()}
with (root/base/'final-selection.json').open('x') as f:json.dump(manifest,f,indent=2)
for name,adapter in arms:
    cmd=[sys.executable,'-m','tabcrew_rl.evaluate','--split','final_test','--final-test','--max-tabs','30','--out',str(base/name)]
    if adapter:cmd+=['--adapter',adapter]
    started=time.perf_counter()
    with (root/base/f'{name}.log').open('x',encoding='utf-8') as logfile:
        result=subprocess.run(cmd,stdout=logfile,stderr=subprocess.STDOUT,timeout=360,cwd=root)
    receipt={'name':name,'seconds':time.perf_counter()-started,'exit_code':result.returncode}
    (root/base/f'{name}.receipt.json').write_text(json.dumps(receipt,indent=2)); print(json.dumps(receipt),flush=True)
    if result.returncode:sys.exit(result.returncode)
