"""Read independently complete final-answer records; never repair model text."""
import json
import re
import core as g

BOUNDARY = re.compile(r'</ifm\|think(?:_fast|_faster)?>')
RECORD_LABEL = re.compile(r'(jsonl|json|javascript)[ \t]*(?=\{)', re.I)
HEADER_LABEL = re.compile(r'(jsonl|json|javascript)[ \t]*\r?\n', re.I)


def strict_json(text):
    return json.loads(text, object_pairs_hook=g.unique_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_json')))


def jsonl_language_labels(text):
    """Decode observed labels only inside an explicitly delimited final section."""
    lines = text.splitlines(keepends=True)
    removed = []
    for number, line in enumerate(lines):
        # The observed JavaScript header occupied its own first line. Its next
        # line must begin a JSON object; do not search through explanatory prose.
        header = HEADER_LABEL.fullmatch(line) if number == 0 else None
        if header and len(lines) > 1 and lines[1].startswith('{'):
            prefix = header
        else:
            prefix = RECORD_LABEL.match(line)
        if prefix:
            removed.append({'line':number + 1, 'label':prefix.group(1), 'prefix':prefix.group()})
            lines[number] = line[prefix.end():]
    return ''.join(lines), removed


def final_text(message, allow_jsonl_labels=False):
    content = message.get('content')
    if not isinstance(content, str):
        raise ValueError('missing_final_content')
    text = content.strip()
    native_reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
    details = {'reasoning_location':'separate' if native_reasoning else 'not_reported',
               'reasoning_chars':len(native_reasoning) if isinstance(native_reasoning,str) else None,
               'closing_marker':None}
    if text.startswith(('{','[')):
        details['final_chars'] = len(text)
        return text, details
    markers = list(BOUNDARY.finditer(content))
    if len(markers) != 1:
        raise ValueError('missing_or_ambiguous_final_boundary')
    text = content[markers[0].end():].strip()
    if allow_jsonl_labels:
        original_chars = len(text)
        text, labels = jsonl_language_labels(text)
        if labels:
            details.update(final_language_labels_removed=labels,
                           final_chars_before_language_labels=original_chars)
    if not text.startswith('{'):
        raise ValueError('final_answer_not_json_records')
    details.update(reasoning_location='embedded_in_content', reasoning_chars=markers[0].start(),
                   closing_marker=markers[0].group(), final_chars=len(text))
    return text, details


def parse_result(result, descriptor):
    choices = result.get('choices')
    if not isinstance(choices,list) or len(choices) != 1:
        raise ValueError('invalid_choices')
    choice = choices[0]
    finish = choice.get('finish_reason')
    if descriptor['format'] != 'jsonl' and 'records' not in descriptor:
        details = {'finish_reason':finish, 'valid_json':False, 'schema_valid':False}
        try:
            text, wire = final_text(choice.get('message',{}))
            details.update(wire)
            value = strict_json(text)
            details['valid_json'] = True
            g.schema_check(value,descriptor['schema'])
            details['schema_valid'] = finish == 'stop'
        except ValueError as exc:
            details['error'] = type(exc).__name__ if len(str(exc))>80 else str(exc)
        return {'control':details}
    if finish not in ('stop','length'):
        raise ValueError('unsupported_finish_reason')
    text, details = final_text(choice.get('message',{}),
                               allow_jsonl_labels=descriptor['format']=='jsonl' and 'records' in descriptor)
    expected = descriptor['records']
    strict_record = descriptor['format'] == 'json_schema'
    if strict_record:
        if (len(expected) != 1 or descriptor['schema'] != next(iter(expected.values()))
                or descriptor.get('identity_binding') != 'single_request'):
            raise ValueError('invalid_single_record_descriptor')
    elif descriptor['format'] != 'jsonl':
        raise ValueError('unsupported_record_format')
    records, errors, lines, seen = {}, {}, [], set()
    unknown = False
    # A schema-constrained response may be pretty-printed. Parse its whole
    # object, never infer JSONL framing from the presence of line breaks.
    for number,line in enumerate([text] if strict_record else text.splitlines(),1):
        if not line.strip(): continue
        try:
            value = strict_json(line)
        except ValueError:
            lines.append({'line':number,'reason':'malformed_record'})
            continue
        # One complete response to one request has an unambiguous local slot.
        # Keep the model object unchanged; its wire schema excludes case_id.
        if strict_record:
            ident = next(iter(expected))
        else:
            if not isinstance(value,dict) or not isinstance(value.get('case_id'),str):
                lines.append({'line':number,'reason':'missing_record_identity'})
                continue
            ident = value['case_id']
        if ident not in expected:
            unknown = True
            lines.append({'line':number,'reason':'unknown_case_id'})
            continue
        if ident in seen:
            records.pop(ident,None)
            errors[ident] = 'duplicate_case_id'
            continue
        seen.add(ident)
        try:
            g.schema_check(value,expected[ident])
            if strict_record and finish != 'stop':
                errors[ident] = 'incomplete_record'
            else:
                records[ident] = value
        except ValueError:
            errors[ident] = 'record_schema'
    if unknown:
        raise ValueError('unknown_case_id')
    for ident in expected:
        if ident not in records and ident not in errors:
            errors[ident] = 'incomplete_record' if finish == 'length' else 'missing_or_malformed_record'
    details.update(finish_reason=finish, response_truncated=finish=='length', line_errors=lines,
                   response_format=descriptor['format'],
                   record_identity='single_request' if strict_record else 'explicit_case_id',
                   accepted_framing_rule=('one_complete_schema_valid_object_stop_required_no_repairs' if strict_record else
                                          'complete_individual_json_objects_only_no_repairs'))
    return {'records':records, 'errors':errors, 'transport':details}
