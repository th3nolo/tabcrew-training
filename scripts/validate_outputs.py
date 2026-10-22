"""Offline checks of the generated curriculum and saved run results (default folder: results/)."""
import ast,hashlib,json,pathlib,re
from tabcrew_rl.data import read_jsonl,validate
from tabcrew_rl.reward import score
root=pathlib.Path(__file__).resolve().parents[1]; report={}
for p in (root/'tabcrew_rl').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
eps=[e for p in (root/'data/v1').glob('*.jsonl') for e in read_jsonl(p)];report['dataset']=validate(eps)
ep_by_id={e['id']:e for e in eps}; predictions=0
for path in (root/'results').glob('*/predictions.jsonl'):
    for row in read_jsonl(path):
        expected=score(row['output'],ep_by_id[row['id']]['reference']['assignments'],complete=row['complete'])
        assert row['metrics']==expected,(path,row['id'])
        assert 1<=row['completion_tokens']<=4000
        predictions+=1
report['recomputed_prediction_metrics']=predictions
for run in ['grpo','grpo-sft']:
    path=root/'results'/run/'rollouts.jsonl'
    if not path.exists():continue
    rows=read_jsonl(path)
    for group in rows:
        assert len(group['rewards'])==4 and len(set(group['episode_ids']))==1
        for text,ref,done,value in zip(group['completions'],group['episode_ids'],group['complete'],group['rewards'],strict=True):
            assert ep_by_id[ref]['split']=='train'
            assert score(text,ep_by_id[ref]['reference']['assignments'],complete=done)['reward']==value
    report[run+'_verified_rollouts']=4*len(rows)
hits=[];scanned=0
for folder in ['tabcrew_rl','scripts','configs','evidence','results']:
    for path in (root/folder).rglob('*'):
        if path.suffix.lower() not in {'.json','.jsonl','.py','.ps1','.md','.txt','.log','.toml'}:continue
        scanned+=1
        if re.search(r'sk-[A-Za-z0-9_-]{24,}',path.read_text(encoding='utf-8',errors='replace')):hits.append(str(path.relative_to(root)))
assert not hits,'Possible credential found; inspect privately'
report['credential_pattern_scan']={'files':scanned,'matches':len(hits),'limit':'Pattern scan, not a proof against every credential type'}
(root/'evidence').mkdir(exist_ok=True)
(root/'evidence/final-validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
