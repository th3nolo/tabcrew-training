"""Topic/page-disjoint validation; sharing a public platform across splits is expected."""
from collections import Counter
import hashlib
import json
from urllib.parse import urlsplit
from tabcrew_rl.reward import parse_assignments
from url_templates import VERSION, canonical_page, coverage, validate_url

PROFILE = 'topic_and_page_disjoint_shared_platforms_v1'


def validate(episodes):
    ids, families, title_sets, pages, hosts, legacy_hosts = set(), {}, {}, {}, set(), {}
    for ep in episodes:
        assert ep['id'] not in ids, 'duplicate episode id'
        ids.add(ep['id'])
        assert ep['tab_count'] == len(ep['tabs']) > 0
        assert [t['ref'] for t in ep['tabs']] == list(range(1, ep['tab_count'] + 1))
        parse_assignments(json.dumps(ep['reference']), [t['ref'] for t in ep['tabs']])
        split = ep['split'].removesuffix('_long')
        assert split in ('train', 'validation', 'final_test'), 'unknown split'
        assert families.setdefault(ep['family'], split) == split, 'family leakage'
        title_hash = hashlib.sha256(json.dumps(sorted(t['title'].casefold() for t in ep['tabs'])).encode()).hexdigest()
        assert title_sets.setdefault(title_hash, split) == split, 'content duplication across split'
        realistic = ep['provenance'].get('synthetic_url_generator') == VERSION
        for tab in ep['tabs']:
            if realistic:
                parsed = validate_url(tab['url'])
                assert ep['id'] not in tab['url'] and ep['family'] not in tab['url'], 'hidden metadata in URL'
            else:
                # Imported bytes retain their original stricter URL contract and are labeled legacy.
                parsed = urlsplit(tab['url'])
                assert parsed.scheme in ('https', 'http') and parsed.hostname and not parsed.username
                assert not parsed.query and not parsed.fragment
                assert legacy_hosts.setdefault(parsed.hostname, split) == split, 'legacy domain leakage'
            key = canonical_page(tab['url'])
            assert pages.setdefault(key, split) == split, 'page leakage across splits'
            hosts.add(parsed.hostname)
    return {'episodes': len(episodes), 'tabs': sum(e['tab_count'] for e in episodes),
        'splits': dict(Counter(e['split'] for e in episodes)), 'families': len(families),
        'domains': len(hosts), 'validation_profile': PROFILE, 'url_coverage': coverage(episodes)}
