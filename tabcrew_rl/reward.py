"""Strict assignment validation and label-invariant partition metrics. Standard library only."""
import json
from collections import Counter

VERSION = 'strict-ari-v1'

class InvalidPartition(ValueError):
    pass

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidPartition('duplicate_key')
        result[key] = value
    return result

def parse_assignments(text, refs):
    if not isinstance(text, str) or len(text) > 200_000:
        raise InvalidPartition('invalid_text')
    try:
        obj = json.loads(text, object_pairs_hook=unique_object,
                         parse_constant=lambda _: (_ for _ in ()).throw(InvalidPartition('nonfinite')))
    except (ValueError, RecursionError) as exc:
        raise InvalidPartition(str(exc) if isinstance(exc, InvalidPartition) else 'invalid_json') from exc
    if not isinstance(obj, dict) or set(obj) != {'assignments'}:
        raise InvalidPartition('invalid_root')
    assignment = obj['assignments']
    if not isinstance(assignment, dict):
        raise InvalidPartition('invalid_assignments')
    if set(assignment) != {str(r) for r in refs}:
        raise InvalidPartition('reference_mismatch')
    if any(type(g) is not int or not 0 <= g <= 9 for g in assignment.values()):
        raise InvalidPartition('invalid_group_id')
    return assignment

def choose2(n):
    return n * (n - 1) // 2

def partition_metrics(reference, predicted):
    if not reference or set(reference) != set(predicted):
        raise ValueError('Partition keys must be equal and nonempty')
    refs = list(reference)
    a, b = [reference[r] for r in refs], [predicted[r] for r in refs]
    joint = sum(choose2(n) for n in Counter(zip(a, b)).values())
    positive_a = sum(choose2(n) for n in Counter(a).values())
    positive_b = sum(choose2(n) for n in Counter(b).values())
    total = choose2(len(refs))
    # Integer arithmetic avoids numerical loss in identical degenerate partitions.
    denominator = (positive_a + positive_b) * total - 2 * positive_a * positive_b
    ari = (2 * (joint * total - positive_a * positive_b) / denominator) if denominator else 1.0
    precision = joint / positive_b if positive_b else float(positive_a == 0)
    recall = joint / positive_a if positive_a else float(positive_b == 0)
    f1 = 2 * joint / (positive_a + positive_b) if positive_a + positive_b else 1.0
    exact = joint == positive_a == positive_b
    group_count = len(set(b))
    return dict(ari=ari, pair_precision=precision, pair_recall=recall, pair_f1=f1,
                exact_partition=exact, group_count=group_count,
                one_group=group_count == 1, all_singletons=group_count == len(refs))

def score(text, reference, *, complete=True):
    if not complete:
        return dict(valid=False, coverage=False, reward=-1.0, error='incomplete', ari=None, pair_f1=None, exact_partition=False)
    try:
        assignment = parse_assignments(text, reference)
    except InvalidPartition as exc:
        return dict(valid=False, coverage=False, reward=-1.0, error=str(exc), ari=None, pair_f1=None, exact_partition=False)
    metrics = partition_metrics(reference, assignment)
    return dict(valid=True, coverage=True, reward=metrics['ari'], error=None, **metrics)

def grpo_reward(completions, reference, **kwargs):
    results = []
    for completion, expected in zip(completions, reference, strict=True):
        text = completion[0]['content'] if isinstance(completion, list) else completion
        expected = json.loads(expected) if isinstance(expected, str) else expected
        results.append(score(text, expected)['reward'])
    return results
