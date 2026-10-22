"""Combine provider snapshots without weakening admission or mixing held-out splits."""
from collections import Counter,defaultdict,deque
import copy
import re
import sys
from common import *

PROVIDERS=('ifm','local_qwen','deepseek')
SPLITS=('train','validation','final_test')

def select_balanced(episodes,maximum=30):
    selected=[]
    for provider in PROVIDERS:
        buckets=defaultdict(list)
        for ep in episodes:
            if ep['dataset_source']['provider']==provider:
                key=(ep['locale'],ep['tab_count'],ep.get('research_context',{}).get('niche','legacy'))
                buckets[key].append(ep)
        locales=defaultdict(deque)
        for key,values in sorted(buckets.items()):
            locales[key[0]].append(deque(sorted(values,key=lambda ep:digest(['evaluation-v1',ep['id']]))))
        chosen=[]
        while locales and len(chosen)<maximum:
            for locale in sorted(locales):
                if len(chosen)>=maximum:break
                queue=locales[locale].popleft()
                chosen.append(queue.popleft())
                if queue:locales[locale].append(queue)
            locales={locale:queues for locale,queues in locales.items() if queues}
        selected.extend(chosen)
    return sorted(selected,key=lambda ep:ep['id'])

def fingerprint_index(episodes,g):
    index=g.DuplicateIndex()
    for ep in episodes:
        try:index.add(index.check(ep))
        except ValueError:pass
    return index

def prepare():
    if (DATA/'manifest.json').exists():raise ValueError('Dataset already frozen')
    # Existing collector's page canonicalization and 0.8 title-shingle Jaccard check.
    sys.path.insert(0,str(DEDUP_COLLECTOR))
    import core as g
    from tabcrew_rl.reward import parse_assignments
    from tabcrew_rl.protocol import messages,canonical_reference
    from tabcrew_rl.model import model_dir
    from transformers import AutoTokenizer
    all_episodes=[];source_manifests={};family_splits={}
    for provider in PROVIDERS:
        folder=DATA/'raw'/provider
        manifest=read(folder/'manifest.json')
        source_manifests[provider]=sha(folder/'manifest.json')
        for split in SPLITS:
            entry=manifest['splits'][split];path=folder/entry['file']
            if sha(path)!=entry['sha256']:raise ValueError('Raw snapshot changed')
            values=rows(path)
            if len(values)!=entry['count']:raise ValueError('Raw count mismatch')
            for original in values:
                ep=copy.deepcopy(original)
                if ep['split']!=split or ep['tab_count']!=len(ep['tabs']):raise ValueError('Invalid original split/count')
                if [tab['ref'] for tab in ep['tabs']]!=list(range(1,len(ep['tabs'])+1)):raise ValueError('Invalid refs')
                parse_assignments(json.dumps(ep['reference']),[tab['ref'] for tab in ep['tabs']])
                family=canonical_family(ep['family'])
                if family_splits.setdefault(family,split)!=split:raise ValueError('Cross-provider curriculum-family leakage')
                ep['dataset_source']={'provider':provider,'original_id':original['id'],'original_sha256':digest(original)}
                ep['id']=provider+':'+original['id']
                all_episodes.append(ep)
    if len({ep['id'] for ep in all_episodes})!=len(all_episodes):raise ValueError('Duplicate source IDs')
    # Old train/validation were already used to fit/select the baselines. They cannot
    # become new evaluation examples. Old sealed final-test references are only
    # fingerprinted by code to prevent their inclusion in the new training split.
    old_seen=fingerprint_index(load_split(OLD/'dataset','train')+load_split(OLD/'dataset','validation'),g)
    old_sealed=fingerprint_index(load_split(OLD/'dataset','final_test'),g)
    global_index=g.DuplicateIndex();kept=[];excluded=[]
    for split in ('final_test','validation','train'):
        for ep in sorted((ep for ep in all_episodes if ep['split']==split),key=lambda ep:ep['id']):
            try:
                if split in ('validation','final_test'):old_seen.check(ep)
                if split=='train':old_sealed.check(ep)
                fingerprint=global_index.check(ep)
            except ValueError as exc:
                excluded.append({'id':ep['id'],'provider':ep['dataset_source']['provider'],'split':split,'reason':str(exc)})
                continue
            global_index.add(fingerprint);kept.append(ep)
    tok=AutoTokenizer.from_pretrained(str(model_dir()),local_files_only=True,trust_remote_code=False)
    lengths=[]
    for ep in kept:
        if ep['split']=='final_test':continue
        prompt=tok.apply_chat_template(messages(ep),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        target=canonical_reference(ep['reference'],[tab['ref'] for tab in ep['tabs']])
        p=len(tok.encode(prompt,add_special_tokens=False));t=len(tok.encode(json.dumps(target,separators=(',',':')),add_special_tokens=False))+1
        if p>4096 or t>512:raise ValueError('Token budget exceeded without truncation: '+ep['id'])
        lengths.append((p,t))
    manifest={'created_at':utc(),'source_manifests':source_manifests,'original_provider_counts':dict(Counter(ep['dataset_source']['provider'] for ep in all_episodes)),
        'splits':{},'provider_split_counts':{},'excluded_count':len(excluded),
        'source_example_count':len(all_episodes),'packed_example_count':len(kept),
        'max_prompt_tokens':max(p for p,t in lengths),'max_target_tokens':max(t for p,t in lengths),
        'human_validation':'Synthetic teacher consistency only; no independent real-user accuracy claim',
        'split_policy':'Original splits preserved. Final test then validation have priority over training when duplicate pages or near-duplicate sessions conflict.',
        'baseline_contamination_guard':'New held-out records checked against original705 train and validation; original705 sealed test checked against new training without semantic inspection.',
        'duplicate_check':'Existing collector exact normalized-title multiset; title trigram Jaccard>=0.8; canonical-page overlap across splits; normalized curriculum family split check',
        'shared_platform_domains':'Allowed: youtube.com/linkedin.com/etc occur naturally in different splits; exact pages remain disjoint.',
        'label_provenance':'Original teacher provenance and references unchanged; only combined IDs are provider-namespaced.'}
    for split in SPLITS:
        values=sorted((ep for ep in kept if ep['split']==split),key=lambda ep:ep['id'])
        manifest['splits'][split]=jsonl(DATA/(split+'.jsonl'),values)
        manifest['provider_split_counts'][split]=dict(Counter(ep['dataset_source']['provider'] for ep in values))
    jsonl(DATA/'excluded.jsonl',excluded)
    evaluation=select_balanced([ep for ep in kept if ep['split']=='validation'])
    if set(ep['dataset_source']['provider'] for ep in evaluation)!=set(PROVIDERS):raise ValueError('Missing evaluation provider')
    eval_manifest={'created_at':utc(),'splits':{'validation':jsonl(DATA/'evaluation'/'validation.jsonl',evaluation)},
        'selection':'Up to30/provider, deterministic round-robin across locale/tab-count/niche; selected before any model predictions.',
        'provider_counts':dict(Counter(ep['dataset_source']['provider'] for ep in evaluation)),
        'candidate_validation_count':sum(ep['split']=='validation' for ep in kept),'final_test_evaluated':False}
    write(DATA/'evaluation'/'manifest.json',eval_manifest)
    manifest['evaluation_manifest_sha256']=sha(DATA/'evaluation'/'manifest.json')
    write(DATA/'manifest.json',manifest)
    print(json.dumps({key:manifest[key] for key in ('source_example_count','packed_example_count','splits','provider_split_counts','excluded_count','max_prompt_tokens','max_target_tokens')}))

if __name__=='__main__':prepare()
