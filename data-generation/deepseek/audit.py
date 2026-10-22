"""Offline validation of unchanged imports and compact-verifier receipts."""
import sys
import collector as c

def audit():
    g=c.g;sys.addaudithook(g.network_guard)
    frozen=g.read_json(c.OUT/'contract.json');assert c.make_contract()==frozen
    c.audience.verify_budget_origin(g)
    migration=g.read_json(c.OUT/'migration.json');source=c.OUT.parent/migration['source_corpus']
    specs={s['id']:s for s in c.specifications()}
    episodes,_=c.load_accepted(c.OUT,frozen,specs)
    for path in sorted((c.OUT/'accepted').glob('*.json')):
        saved=g.read_json(path);ep=saved['example'];spec=specs[ep['id']]
        assert saved['spec']==spec
        c.schema_check(saved['authored'],g.author_schema(spec))
        rebuilt=c.validation.normalize(saved['authored'],spec,saved['author_receipt'])
        for key in ['tabs','reference','intent','locale','split','family','research_context']:assert rebuilt[key]==ep[key],key
        c.schema_check(saved['blind_review'],g.verifier_schema(ep));g.verify_grouping(ep,saved['blind_review'])
        imported=path.name in migration['imported_files']
        if not imported:
            assert c.verifier.normalize(g,c.schema_check,ep,saved['compact_review'])==saved['blind_review']
            if 'author_origin' in saved:
                origin=saved['author_origin'];original=source/'pending'/origin['pending_file']
                assert origin['source_corpus']==source.name and g.sha(original)==origin['pending_file_sha256']==migration['pending_authored'][original.name]
                old=g.read_json(original)
                assert saved['authored']==old['authored'] and saved['author_receipt']==old['author_receipt']
        for field in ['author_receipt','review_receipt']:
            receipt=saved[field];ident=receipt.get('call_id')
            if not ident:continue
            cache=c.OUT/'responses'/(ident+'.json')
            if not cache.exists():cache=source/'responses'/(ident+'.json')
            response=g.read_json(cache)
            assert g.digest(response['result'])==receipt['response_sha256']
            assert response['receipt']['request_sha256']==receipt['request_sha256']
            if not imported and field=='review_receipt':
                assert c.json.loads(response['result']['choices'][0]['message']['content'],object_pairs_hook=g.unique_object)==saved['compact_review']
            if not imported and field=='author_receipt':
                assert c.json.loads(response['result']['choices'][0]['message']['content'],object_pairs_hook=g.unique_object)==saved['authored']
    report={'checked_at':g.utc(),'result':'passed','checked_examples':len(episodes),
            'validation':g.validate_corpus(list(episodes.values())),'splits':dict(c.Counter(e['split'] for e in episodes.values())),
            'imported_receipts_byte_identical':len(migration['imported_files']),'contract_sha256':g.digest(frozen),'human_reviewed':False}
    c.durable_json(c.OUT/'audit.json',report);print(c.json.dumps(report,indent=2))

if __name__=='__main__':audit()
