"""Three small wire-format probes; shared quota, cached receipts, no retries."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import uuid
import core as g
from client import Client,StopCollection
from files import atomic


@contextmanager
def lock(root):
    import msvcrt
    with (root/'collector.lock').open('a+b') as stream:
        if stream.tell()==0:
            stream.write(b'1')
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        yield


def run():
    root=g.OUT
    for name in ('responses','controls','runs','logs'): (root/name).mkdir(exist_ok=True)
    with lock(root):
        run_id=str(uuid.uuid4())
        deadline=time.time()+300
        def stop():
            if (root/'PAUSE').exists(): raise StopCollection('manual_pause')
            if time.time()>=deadline: raise StopCollection('control_time_limit')
        stop()
        client=Client(root,100000,run_id,stop)
        summary={'run_id':run_id,'at':g.utc(),'source_sha256':{name:g.sha(root/name) for name in ('controls.py','client.py','framing.py')},'probes':[]}
        for index,(fmt,effort) in enumerate((('json_schema','low'),('json_schema','high'),('json_object','low'))):
            path=root/'controls'/f'probe-{index}.json'
            cp=g.read_json(path) if path.exists() else {'spec':{'id':f'control-{index}'},'arm':'controls','batch_size':0}
            def save(value=cp): atomic(path,value)
            schema=g.obj({'probe':{'type':'string','enum':['schema_value'] if fmt=='json_schema' else ['prompt_value']},
                          'values':{'type':'array','items':{'type':'integer'},'minItems':3,'maxItems':3}})
            # Contradictory enum intentionally tests whether schema enforcement is
            # active instead of merely asking the model to comply with a prompt.
            turns=[{'role':'system','content':'Return a brief JSON answer. No explanations.'},
                   {'role':'user','content':'Return exactly {"probe":"prompt_value","values":[1,2,3]}.'}]
            descriptor={'format':fmt,'effort':effort,'schema':schema}
            answer,receipt=client.chat(turns,descriptor,81000+index,'probe_'+str(index),cp,save,max_tokens=1024)
            cp.update(result=answer,receipt=receipt,completed=True)
            save()
            summary['probes'].append({'format':fmt,'requested_effort':effort,**answer['control'],
                                       'tokens':receipt['usage']['total_tokens'],'call_id':receipt['call_id']})
            summary['cumulative_usage']=client.ledger.usage
            atomic(root/'controls-report.json',summary)
        print(json.dumps(summary))


if __name__=='__main__': run()
