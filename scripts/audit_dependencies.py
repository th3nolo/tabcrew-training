"""Verify every locked package's registry identity and release age before installation."""
import concurrent.futures, datetime, json, pathlib, tomllib, urllib.request
root = pathlib.Path(__file__).resolve().parents[1]
lock = tomllib.loads((root/'uv.lock').read_text())
now = datetime.datetime.now(datetime.timezone.utc)
def inspect(p):
    name, version = p['name'], p['version']
    if 'virtual' in p.get('source', {}): return None
    url = f'https://pypi.org/pypi/{name}/{version.split("+")[0]}/json'
    with urllib.request.urlopen(url, timeout=60) as r: obj=json.load(r)
    published = min(x['upload_time_iso_8601'] for x in obj['urls'])
    age = (now-datetime.datetime.fromisoformat(published.replace('Z','+00:00'))).total_seconds()/3600
    assert obj['info']['name'].lower().replace('_','-') == name.replace('_','-'), name
    assert age >= 72, (name, age)
    registry=p['source']['registry']
    assert registry in ('https://pypi.org/simple','https://download.pytorch.org/whl/cu128'), registry
    assert p.get('wheels'), f'No wheels: {name}'
    if name=='torch':
        # Official CUDA build identity; the upstream release's age is recorded separately.
        for w in p['wheels']:
            assert w['url'].startswith(('https://download.pytorch.org/','https://download-r2.pytorch.org/')), w['url']
    return {'name':name,'version':version,'registry':registry,'metadata_url':url,
            'release':published,'age_hours':round(age,1),'project_urls':obj['info']['project_urls'],
            'purpose':'Direct plotting dependency' if name=='matplotlib' else ('Direct ML dependency' if name in {'torch','transformers','trl','peft','accelerate','datasets'} else 'Transitive requirement of the locked ML/plotting stack'),
            **({'cuda_release_source':'https://pytorch.org/get-started/previous-versions/#v280',
                'age_scope':'Official PyTorch 2.8.0 release; CUDA 12.8 variant listed by PyTorch'} if name=='torch' else {})}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    rows=[r for r in pool.map(inspect,lock['package']) if r]
(root/'evidence').mkdir(exist_ok=True)
(root/'evidence/dependency-audit.json').write_text(json.dumps({'verified_utc':now.isoformat(),'packages':rows},indent=2))
print(json.dumps({'audited_packages':len(rows),'minimum_age_hours':min(r['age_hours'] for r in rows)}))
