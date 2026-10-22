"""One bounded content repair. Unchanged full-session checks decide acceptance."""
import copy
import json
import re
import core as g

VERSION = 'targeted-title-repair-v1'
REPAIRABLE = {'duplicate_title', 'truncated_or_malformed_text', 'uninformative_title',
              'title_length', 'group_label_in_title', 'formatted_title_length'}


def defects(authored, spec):
    g.schema_check(authored, g.author_schema(spec))
    seen = set()
    result = []
    for i, size in enumerate(spec['group_sizes']):
        group = authored['groups'][f'g{i}']
        try:
            g.check_text_quality(group['task'])
        except ValueError as error:
            result.append({'field': f'task_{i}', 'group': i, 'index': None, 'reason': str(error), 'old': group['task']})
        for j, title in enumerate(group['titles']):
            key = ' '.join(re.findall(r'\w+', title.casefold()))
            reason = None
            try:
                g.check_text_quality(title)
            except ValueError as error:
                reason = str(error)
            if key in seen:
                reason = reason or 'duplicate_title'
            if len(key.split()) < 4:
                reason = reason or 'uninformative_title'
            if g.has_hidden_group_label(title, spec['page_kinds_by_group'][i][j]):
                reason = reason or 'group_label_in_title'
            if len(title.strip()) < 15 or len(title.strip()) > 150:
                reason = reason or 'title_length'
            seen.add(key)
            if reason:
                result.append({'field': f'title_{i}_{j}', 'group': i, 'index': j, 'reason': reason, 'old': title})
    return result


def schema(targets):
    return g.obj({t['field']: {'type': 'string', 'minLength': 15, 'maxLength': 130} for t in targets})


def messages(authored, spec, targets):
    requested = []
    for t in targets:
        i, j = t['group'], t['index']
        requested.append({**t, 'domain': spec['task_topics'][i],
            'objective': authored['groups'][f'g{i}']['task'], 'exclusive_focus': spec['focus_by_group'][i],
            'language': ('es' if (i+j) % 2 else 'en') if spec['locale'] == 'mixed' and j is not None
                        else ('es' if spec['locale'] == 'es' else 'en'),
            'source': g.audience.url_templates.ROUTES[spec['page_kinds_by_group'][i][j]][3] if j is not None else 'research objective'})
    return [{'role': 'system', 'content':
        'Write replacement content for defective synthetic browser titles. Return only the requested JSON fields. '
        'Write a genuinely new, specific headline for each requested title; do not fix defects by merely changing punctuation or adding numbering. '
        'No browsing. No group labels, field names or instructions inside titles.'},
        {'role': 'user', 'content':
        'Keep every research objective and source assignment. Each replacement must clearly belong to its objective. '
        'Use 8-18 words, under 125 characters, complete natural phrases. Use exactly the requested language. '
        'Invent a different relevant name, question, finding or product specification; avoid duplicating any existing title. '
        'No website suffixes or URLs; our formatter adds them. No personal contact details or real profile claims. '
        'If a task itself is defective, replace it with a concise objective preserving its domain and exclusive focus.\n'
        + json.dumps({'existing_groups': authored['groups'], 'replacements': requested}, ensure_ascii=False, separators=(',', ':'))}]


def apply(authored, spec, targets, replacements):
    g.schema_check(replacements, schema(targets))
    result = copy.deepcopy(authored)
    for target in targets:
        group = result['groups'][f"g{target['group']}"]
        value = replacements[target['field']]
        if target['index'] is None:
            group['task'] = value
        else:
            group['titles'][target['index']] = value
    g.schema_check(result, g.author_schema(spec))
    return result
