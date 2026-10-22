"""Compact author records, stable case IDs and independent tab-count limits."""
import json
import core as g
import source_catalog
import url_templates

AUTHOR_CASES, AUTHOR_TABS = 3, 40
VERIFIER_CASES, VERIFIER_TABS = 1, 30


def pack(items, max_cases, max_tabs):
    if not 1 <= max_cases <= 10 or not 10 <= max_tabs <= 120:
        raise ValueError('invalid_batch_limits')
    batches, current, tabs, split = [], [], 0, None
    for item in items:
        if item['tab_count'] > max_tabs:
            raise ValueError('single_session_exceeds_tab_limit')
        if current and (len(current)>=max_cases or tabs+item['tab_count']>max_tabs or item['split']!=split):
            batches.append(current)
            current, tabs = [], 0
        current.append(item)
        tabs += item['tab_count']
        split = item['split']
    if current: batches.append(current)
    return batches


def author_schema(spec, slot):
    return g.obj({'case_id':{'type':'string','enum':[slot]},
        'tasks':{'type':'array','minItems':len(spec['group_sizes']),'maxItems':len(spec['group_sizes']),
                 'items':{'type':'string','minLength':10,'maxLength':130}},
        'titles':{'type':'array','minItems':spec['tab_count'],'maxItems':spec['tab_count'],
                  'items':{'type':'string','minLength':1,'maxLength':150}}})


def author_request(specs, slots):
    if len({s['split'] for s in specs}) != 1: raise ValueError('mixed_dataset_splits')
    routes, cases, schemas = {}, [], {}
    for spec in specs:
        slot = slots[spec['id']]
        groups = []
        for i, domain in enumerate(spec['task_topics']):
            ordered = []
            for j, kind in enumerate(spec['page_kinds_by_group'][i]):
                route = url_templates.ROUTES[kind]
                routes[kind] = {'page':route[3], 'url_shape':route[1]+route[2]}
                language = ('es' if (i+j)%2 else 'en') if spec['locale']=='mixed' else spec['locale']
                ordered.append([kind,language])
            groups.append({'domain':domain,'objective':spec['workflow_instructions'][i],
                'exclusive_focus':spec['focus_by_group'][i], 'ordered_title_slots':ordered})
        cases.append({'case_id':slot,'audience':spec['audience'],'scenario':spec['scenario'],
            'purpose':source_catalog.STAGES[spec['research_stage']][1], 'groups':groups,
            'variation_seed':spec['seed']})
        schemas[slot] = author_schema(spec,slot)
    rules = (
        'Invent realistic browser research sessions, without browsing or tools. Keep planning brief. '
        'Write final output as JSONL: exactly one compact JSON object per line per case, no outer array, markdown or prose. '
        'The final answer and every record line must start with {; never emit json, JSONL, JavaScript or another language prefix. '
        'Each object has exactly case_id, tasks, titles. tasks is one concrete 8-16-word objective per supplied group in order; '
        'titles is ONE FLAT array, concatenating each group\'s ordered title slots in that same order. '
        'Follow every slot\'s language and page type. Tasks use the session language (English for mixed sessions). '
        'Titles must make the objective inferable without hidden page content. Keep related objectives distinct with exclusive foci. '
        'All sources in a group serve its objective; platforms and formats never determine membership. '
        'Use fictional people, brands and circumstances: profiles need name plus specialty, product pages concrete specs, '
        'studies specific findings, videos relevant demos/interviews, searches natural queries. '
        'Vary wording and evidence; avoid repeating the whole task or a numbered template. No group labels, platform suffixes, '
        'contact details, actual profile claims or implied scraped content. Titles 1-150 characters, tasks 10-130. '
        'Use natural short concept, product or profile titles when appropriate; do not pad them to a word count. '
        'The title and visible URL context must still distinguish the intended investigation. No generic Home, Untitled or loading pages. '
        'Do not generate URLs, IDs or metadata: local code supplies them. Each session is independent; never move content between cases. '
        'Example output shape only: {"case_id":"s0","tasks":["One objective","Another objective"],"titles":["first title","second title"]}. '
        'The real title and task counts MUST match the assigned groups, not the shape example.')
    return ([{'role':'system','content':rules}, {'role':'user','content':json.dumps({'routes':routes,'cases':cases},ensure_ascii=False,separators=(',',':'))}],
            {'format':'jsonl','effort':'low','records':schemas}, output_budget(specs,'author'))


def output_budget(items, role):
    # Reasoning and final output share the provider's completion allowance.
    # Explicit cap; no automatic retry or unbounded increase on truncation.
    tabs = sum(s['tab_count'] for s in items)
    return min(16384,8192+48*tabs) if role=='author' else min(12288,8192+24*tabs)


def normalize_record(record,spec,receipt):
    groups, cursor = {}, 0
    for i,size in enumerate(spec['group_sizes']):
        groups[f'g{i}']={'task':record['tasks'][i], 'titles':record['titles'][cursor:cursor+size]}
        cursor += size
    raw = {'groups':groups}
    g.schema_check(raw,g.author_schema(spec))
    episode = g.normalize(raw,spec,receipt)
    g.validate_corpus([episode])
    return episode


def verifier_request(episodes):
    if not episodes: raise ValueError('empty_verifier_batch')
    if len({ep['split'] for ep in episodes.values()}) != 1: raise ValueError('mixed_dataset_splits')
    single = len(episodes) == 1
    schemas, cases = {}, []
    for slot,ep in episodes.items():
        visible = json.loads(g.verifier_messages(ep)[1]['content'])
        cases.append({'case_id':slot, **visible})
        properties = {
            'assignments':{'type':'array','minItems':ep['tab_count'],'maxItems':ep['tab_count'],
                          'items':{'type':'integer','minimum':0,'maximum':len(visible['user_tasks'])-1}},
            'language':{'type':'string','enum':['en','es','mixed']},
            'ambiguous_refs':{'type':'array','maxItems':ep['tab_count'],
                              'items':{'type':'integer','minimum':1,'maximum':ep['tab_count']}}}
        if not single:
            properties = {'case_id':{'type':'string','enum':[slot]}, **properties}
        schemas[slot] = g.obj(properties)
    rules = g.verifier_messages(next(iter(episodes.values())))[0]['content'].replace('/no_think','')
    output = ('Return exactly one JSON object matching the supplied JSON schema, no wrapper or prose. ' if single else
              'Return JSONL, one compact object per line, no wrapper or prose. ')
    fields = 'assignments, language, ambiguous_refs' if single else 'case_id, assignments, language, ambiguous_refs'
    rules = rules[:rules.index('Return assignments')] + 'Keep planning brief. Review cases independently. ' + output + (
        f'Each object has exactly {fields}. assignments is an array of zero-based task '
        'indices in the provided tabs order, one integer per tab (for example [0,1,0], never nested arrays). '
        'language is en, es or mixed. ambiguous_refs lists uncertain or multi-task tab refs using their one-based ref values. '
        'Never use another case\'s task or tab refs. No explanation fields.')
    descriptor = {'format':'jsonl','effort':'low','records':schemas}
    if single:
        descriptor.update(format='json_schema',schema=next(iter(schemas.values())),identity_binding='single_request')
    return ([{'role':'system','content':rules},{'role':'user','content':json.dumps({'cases':cases},ensure_ascii=False,separators=(',',':'))}],
            descriptor, output_budget(list(episodes.values()),'verifier'))


def review_record(ep,record):
    checked = {'tasks':g.visible_tasks(ep), 'assignments':{str(t['ref']):v for t,v in zip(ep['tabs'],record['assignments'])},
               'observed_language':record['language'], 'ambiguous_refs':record['ambiguous_refs'],'review_notes':'Independent blind record review.'}
    g.schema_check({k:v for k,v in checked.items() if k!='tasks'},g.verifier_schema(ep))
    g.verify_grouping(ep,checked)


def pilot_specs():
    selected, families = [], set()
    for size,locale in ((10,'en'),(12,'es'),(18,'mixed')):
        spec = next(s for s in g.specifications() if s['split']=='train' and s['locale']==locale
            and s['tab_count']==size and s['scenario']=='distinct_investigations' and s['family'] not in families
            and s['audience']=='entrepreneur_freelancer')
        selected.append(spec)
        families.add(spec['family'])
    return selected


def error_code(exc):
    value = str(exc)
    return value if len(value)<100 and value.replace('_','').isalnum() else type(exc).__name__
