"""Strict, loopback-only SGLang client; no provider credentials or external API fallback."""
import json
from pathlib import Path
import threading
import time
import urllib.request
import core as g
from files import append

LOG_LOCK=threading.Lock()
class Client:
    def __init__(self,port,log,run_id=None):
        self.base=f'http://127.0.0.1:{port}'
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),g.NoRedirect())
        self.log=Path(log)
        self.run_id=run_id
    def ready(self):
        with self.opener.open(self.base+'/health',timeout=3) as response:
            return response.status==200
    def chat(self,turns,schema,seed,*,role,max_tokens=2048):
        payload={'model':g.ALIAS,'messages':turns,'stream':False,
                 'temperature':.75 if role=='author' else .2,
                 'top_p':.8 if role=='author' else .95,'top_k':20,
                 'presence_penalty':.4 if role=='author' else 0,
                 'seed':seed,'max_tokens':max_tokens,
                 'chat_template_kwargs':{'enable_thinking':False},
                 'response_format':{'type':'json_schema','json_schema':{'name':role,'strict':True,'schema':schema}}}
        started=time.perf_counter()
        request=urllib.request.Request(self.base+'/v1/chat/completions',data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
        with self.opener.open(request,timeout=180) as response:
            result=json.load(response)
        record={'at':g.utc(),'run_id':self.run_id,'role':role,'request_sha256':g.digest(payload),'response_sha256':g.digest(result),
                'seconds':time.perf_counter()-started,'usage':result.get('usage'),'seed':seed,'backend':'sglang-0.5.17'}
        with LOG_LOCK:
            append(self.log,record)
        choice=result['choices'][0]
        if choice.get('finish_reason')!='stop':
            raise ValueError('incomplete_'+role)
        content=choice['message'].get('content')
        if not isinstance(content,str):
            raise ValueError('missing_'+role+'_content')
        parsed=json.loads(content,object_pairs_hook=g.unique_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_json')))
        g.schema_check(parsed,schema)
        return parsed,{**record,'raw_content':content}
