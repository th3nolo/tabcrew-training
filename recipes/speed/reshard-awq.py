"""Repackage verified Safetensors into ~256 MiB shards without torch or changing any tensor byte.

UNTESTED: this script was never loaded in a serving run. The output was not served by SGLang.

Goal: smaller shards let the loader read the checkpoint in smaller pieces and lower the
host-memory peak during startup (see README.md). Tensor names, dtypes, shapes and bytes are
copied unchanged. Every copied tensor is hashed during the copy and again from the written file.

Usage:
  python reshard-awq.py --source <model dir> --assets <assets.json> --dest <new dir> [--write]
The assets file lists every source file with its size and SHA-256:
  {"status": "acquired_and_verified", "files": [{"file": ..., "bytes": ..., "sha256": ...}, ...]}
Without --write, the script only prints the shard plan.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

LIMIT=256*1024*1024

def file_hash(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):
            result.update(part)
    return result.hexdigest()

def inventory(source,assets):
    manifest=json.loads(assets.read_text())
    assert manifest['status']=='acquired_and_verified'
    tensors=[]
    for row in manifest['files']:
        if not row['file'].endswith('.safetensors'):
            continue
        path=source/row['file']
        assert path.stat().st_size==row['bytes']
        with path.open('rb') as stream:
            size=struct.unpack('<Q',stream.read(8))[0]
            assert 0<size<10*1024*1024
            header=json.loads(stream.read(size))
        for name,meta in header.items():
            if name=='__metadata__': continue
            start,end=meta['data_offsets']
            assert 0<=start<end<=path.stat().st_size-8-size
            tensors.append({'name':name,'dtype':meta['dtype'],'shape':meta['shape'],'source':row['file'],
                            'offset':8+size+start,'bytes':end-start})
    assert len({t['name'] for t in tensors})==len(tensors)
    index=json.loads((source/'model.safetensors.index.json').read_text())
    assert {t['name']:t['source'] for t in tensors}==index['weight_map']
    batches=[]; current=[]; count=0
    for tensor in sorted(tensors,key=lambda t:t['name']):
        if current and count+tensor['bytes']>LIMIT:
            batches.append(current); current=[]; count=0
        current.append(tensor); count+=tensor['bytes']
    if current: batches.append(current)
    return manifest,batches

def run(source,assets,dest,write):
    original,batches=inventory(source,assets)
    plan={'tensors':sum(map(len,batches)),'shards':len(batches),'total_tensor_bytes':sum(t['bytes'] for b in batches for t in b),
          'largest_shard_bytes':max(sum(t['bytes'] for t in b) for b in batches),'target_regular_shard_bytes':LIMIT}
    print(json.dumps(plan),flush=True)
    if not write: return
    for row in original['files']:
        assert file_hash(source/row['file'])==row['sha256'],row['file']
    dest.mkdir(exist_ok=True)
    result={**original,'local_directory':dest.name,'status':'acquired_and_verified','files':[],
            'transformation':'Safetensors resharding only; identical names, dtypes, shapes, and tensor bytes',
            'source_assets_sha256':file_hash(assets),'tensor_verification':[],**plan}
    weight_map={}
    for i,batch in enumerate(batches,1):
        filename=f'model-{i:05d}-of-{len(batches):05d}.safetensors'
        target=dest/filename
        if target.exists(): raise RuntimeError('Refusing to overwrite completed reshard')
        header={'__metadata__':{'format':'pt'}}; offset=0
        for tensor in batch:
            header[tensor['name']]={'dtype':tensor['dtype'],'shape':tensor['shape'],'data_offsets':[offset,offset+tensor['bytes']]}
            offset+=tensor['bytes']; weight_map[tensor['name']]=filename
        encoded=json.dumps(header,separators=(',',':')).encode()
        encoded+=b' '*((-len(encoded))%8)
        temporary=target.with_suffix('.partial')
        with temporary.open('wb') as output:
            output.write(struct.pack('<Q',len(encoded))); output.write(encoded)
            for tensor in batch:
                digest=hashlib.sha256()
                with (source/tensor['source']).open('rb') as stream:
                    stream.seek(tensor['offset']); left=tensor['bytes']
                    while left:
                        chunk=stream.read(min(left,8*1024*1024))
                        assert chunk
                        output.write(chunk); digest.update(chunk); left-=len(chunk)
                result['tensor_verification'].append({**tensor,'destination':filename,'sha256':digest.hexdigest()})
        temporary.rename(target)
        # Verify output through a second read, independent of the copy loop.
        with target.open('rb') as stream:
            n=struct.unpack('<Q',stream.read(8))[0]; actual=json.loads(stream.read(n))
            assert actual==header
            for tensor in result['tensor_verification'][-len(batch):]:
                stream.seek(8+n+header[tensor['name']]['data_offsets'][0])
                h=hashlib.sha256(); left=tensor['bytes']
                while left:
                    part=stream.read(min(left,8*1024*1024)); assert part
                    h.update(part); left-=len(part)
                assert h.hexdigest()==tensor['sha256']
        result['files'].append({'file':filename,'bytes':target.stat().st_size,'sha256':file_hash(target)})
        print(json.dumps({'verified_shard':i,'of':len(batches)}),flush=True)
    for row in original['files']:
        if row['file'].endswith('.safetensors') or row['file']=='model.safetensors.index.json': continue
        target=dest/row['file']; shutil.copyfile(source/row['file'],target)
        assert file_hash(target)==row['sha256']
        result['files'].append(row)
    index=dest/'model.safetensors.index.json'
    index.write_text(json.dumps({'metadata':{'total_size':plan['total_tensor_bytes']},'weight_map':weight_map},indent=2))
    result['files'].append({'file':index.name,'bytes':index.stat().st_size,'sha256':file_hash(index)})
    result['total_bytes']=sum(r['bytes'] for r in result['files'])
    (dest.parent/(dest.name+'-assets.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps({'status':'verified_byte_identical',**plan}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True,help='Folder with the original AWQ safetensors files')
    p.add_argument('--assets',type=Path,required=True,help='JSON list of source files with size and SHA-256')
    p.add_argument('--dest',type=Path,required=True,help='New folder for the resharded copy (must not hold shards)')
    p.add_argument('--write',action='store_true')
    a=p.parse_args()
    run(a.source,a.assets,a.dest,a.write)
