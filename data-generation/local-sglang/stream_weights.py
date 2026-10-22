"""Experimental bounded-buffer loader for the pinned Qwen AWQ checkpoint.

This changes checkpoint loading only. SGLang keeps its model implementation,
AWQ Marlin kernels, tokenizer, grammar, batching, and CUDA graphs.
"""
import json
import math
import os
from pathlib import Path
import struct
import sys
import torch
import safetensors

DTYPES={'F16':(torch.float16,2),'BF16':(torch.bfloat16,2),'I32':(torch.int32,4)}


def unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise ValueError('Duplicate checkpoint metadata key')
        result[key]=value
    return result


def descriptors(path):
    with open(path,'rb',buffering=0) as stream:
        size=struct.unpack('<Q',stream.read(8))[0]
        if not 2<=size<=16*1024*1024:
            raise ValueError('Unexpected checkpoint header size')
        header=json.loads(stream.read(size),object_pairs_hook=unique)
    offset=8+size
    entries={key:value for key,value in header.items() if key!='__metadata__'}
    # The installed safetensors parser independently checks the whole header,
    # offsets, tensor metadata and payload extent, without reading tensor data.
    with safetensors.safe_open(path,framework='pt',device='cpu') as reference:
        if set(reference.keys())!=set(entries):
            raise ValueError('Checkpoint keys disagree with safetensors')
        for key,item in entries.items():
            expected=reference.get_slice(key)
            if item['dtype'] not in DTYPES or item['shape']!=expected.get_shape() or item['dtype']!=expected.get_dtype():
                raise ValueError(f'Unsupported or inconsistent checkpoint tensor: {key}, dtype={item["dtype"]}, reference_dtype={expected.get_dtype()}, shape={item["shape"]}, reference_shape={expected.get_shape()}')
            start,end=item['data_offsets']
            if end-start!=math.prod(item['shape'])*DTYPES[item['dtype']][1]:
                raise ValueError('Checkpoint tensor extent mismatch')
    return offset,entries


def safetensors_weights_iterator(hf_weights_files,disable_mmap=False,prefetch=False,prefetch_num_threads=4,drop_cache_after_load=False):
    if sys.byteorder!='little' or prefetch:
        raise ValueError('This diagnostic requires little-endian host and no checkpoint prefetch')
    for file in hf_weights_files:
        path=Path(file)
        offset,entries=descriptors(str(path))
        with path.open('rb',buffering=0) as stream:
            for name in sorted(entries):
                item=entries[name]
                start,end=item['data_offsets']
                count=end-start
                if count==0:
                    raise ValueError('Empty tensor is outside this diagnostic model contract')
                stream.seek(offset+start)
                buffer=bytearray(count)
                view=memoryview(buffer)
                position=0
                while position<count:
                    part=view[position:min(position+8*1024*1024,count)]
                    read=stream.readinto(part)
                    del part
                    if not read:
                        raise ValueError('Truncated checkpoint tensor')
                    position+=read
                    # Evict clean backing-file pages, not the tensor's anonymous
                    # buffer. This does not discard live or modified tensor data.
                    os.posix_fadvise(stream.fileno(),offset+start,position,os.POSIX_FADV_DONTNEED)
                del view
                tensor=torch.frombuffer(buffer,dtype=DTYPES[item['dtype']][0]).reshape(item['shape'])
                yield name,tensor
                del tensor,buffer
                os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
