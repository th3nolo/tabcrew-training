"""Validate committed bytes, lineage, curriculum, blind agreement and snapshot hashes offline."""
from collections import Counter
import sys
import collector as c
g=c.g
def audit():
    frozen=g.read_json(c.OUT/'contract.json')
    assert c.contract()==frozen,'Frozen collector or dependency manifest changed'
    accepted,_=c.accepted_records(frozen)
    status=g.read_json(c.OUT/'status.json')
    if status['state']!='running':
        assert status['accepted_examples']==len(accepted)
        snapshot=g.read_json(g.Path(status['snapshot'])/'manifest.json')
        for split,entry in snapshot['splits'].items():
            assert g.sha(g.Path(status['snapshot'])/entry['file'])==entry['sha256']
            assert entry['count']==sum(ep['split']==split for ep in accepted.values())
        cohort_path=g.Path(status['snapshot'])/snapshot['recommended_training_cohort']
        assert g.sha(cohort_path)==snapshot['recommended_training_manifest_sha256']
        cohort=g.read_json(cohort_path)
        for split,entry in cohort['splits'].items():
            assert g.sha(cohort_path.parent/entry['file'])==entry['sha256']
            assert entry['count']==sum(ep['split']==split and ep['provenance'].get('synthetic_url_generator')==g.URL_VERSION for ep in accepted.values())
    result={'checked_at':g.utc(),'checked_examples':len(accepted),'result':'passed',
        'validation':g.validate_corpus(list(accepted.values())),
        'splits':dict(Counter(ep['split'] for ep in accepted.values())),
        'teachers':dict(Counter(ep['provenance']['teacher'] for ep in accepted.values())),
        'contract_sha256':g.digest(frozen),'human_reviewed':False,'url_coverage':g.url_coverage(list(accepted.values()))}
    g.atomic(c.OUT/'audit.json',result)
    return result

if __name__=='__main__':
    sys.addaudithook(g.network_guard)
    print(g.json.dumps(audit(),indent=2))
