"""Locally authored measurement helpers for a separately reviewed model runner.

Importing this module imports only the Python standard library. It cannot load
a model or start a process. The GPU owner supplies an already reviewed, private
runner, a request factory and CUDA event factory after its resource checks.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import time

PROTOCOL = 'qwen3-fixed-batch-v1'
MODEL_REVISION = '74d4bd2bd4bff9cafc9345221320bffb08b406a3'


@dataclass(frozen=True)
class Case:
    batch: int
    input_tokens: int
    output_tokens: int = 256

    def validate(self):
        if self.batch not in (1, 2, 3, 4):
            raise ValueError('Unsupported batch size')
        if self.input_tokens not in (256, 1024):
            raise ValueError('Unsupported input length')
        if self.output_tokens != 256:
            raise ValueError('This protocol fixes output length at 256')
        if self.input_tokens + self.output_tokens > 6144:
            raise ValueError('Context limit exceeded')
        if self.batch * (self.input_tokens + self.output_tokens) > 16384:
            raise ValueError('Logical KV slot limit exceeded')
        if self.batch * self.input_tokens > 1024:
            raise ValueError('Static extend would exceed the preserved 1024-token prefill cap')

    @property
    def key(self):
        return f'b{self.batch}-i{self.input_tokens}-o{self.output_tokens}'


def make_inputs(case):
    """Deterministic synthetic IDs; no tokenizer, training data or global RNG."""
    case.validate()
    rows = []
    for row in range(case.batch):
        values = []
        for column in range(case.input_tokens):
            digest = hashlib.sha256(f'{PROTOCOL}:{case.key}:{row}:{column}'.encode()).digest()
            values.append(1000 + int.from_bytes(digest[:4], 'little') % 9000)
        rows.append(values)
    return rows


def input_hash(rows):
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def manifest():
    cases = [Case(batch, 256) for batch in (1, 2, 3, 4)] + [Case(1, 1024)]
    return {
        'protocol': PROTOCOL, 'model_revision': MODEL_REVISION,
        'status': 'Prepared; no GPU measurements established by this manifest',
        'sampling': 'greedy diagnostic, separate from representative unseeded author/verifier workload',
        'warmup_sequences_per_case': 2, 'timed_sequences_per_case': 5,
        'primary_case': 'b4-i256-o256',
        'cases': [{**asdict(case), 'case_id': case.key,
                   'input_sha256': input_hash(make_inputs(case)),
                   'logical_max_kv_slots': case.batch * (case.input_tokens + case.output_tokens)}
                  for case in cases],
        'same_arm_repeats_are_not_independent_process_pairs': True,
        'long_context_coverage': 'Pending a separately reviewed incremental cache-priming or chunked-prefill adapter; unchunked large prefill is refused',
        'acceptance': 'Diagnostic evidence only; representative full-model comparison and correctness gates remain required',
    }


def raw_nanoseconds():
    if not hasattr(time, 'CLOCK_MONOTONIC_RAW'):
        raise RuntimeError('CLOCK_MONOTONIC_RAW is required; no adjusted-clock fallback')
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def _batch_width(value):
    shape = getattr(value, 'shape', None)
    if shape is not None:
        return int(shape[0])
    return len(value)


def measure_case(runner, request_factory, case, event_factory, *, clock_ns=raw_nanoseconds):
    """Time one fixed sequence using the caller's already loaded private runner.

    Required runner API: clear, synchronize, extend, decode and cleanup.
    request_factory accepts (batch, input_length, custom_inputs=list[list[int]]).
    event_factory creates timing-enabled events supporting record/elapsed_time.
    Caller must check stop/headroom outside this bounded sequence and must not
    call this helper on a shared serving runner. No HTTP, process management or
    protected host state is accessed here.
    """
    case.validate()
    rows = make_inputs(case)
    digest = input_hash(rows)
    requests = request_factory(case.batch, case.input_tokens, custom_inputs=rows)
    if len(requests) != case.batch:
        raise ValueError('Request factory changed the batch size')
    for request, expected_ids in zip(requests, rows):
        if list(request.origin_input_ids) != expected_ids:
            raise ValueError('Request factory changed the frozen input IDs')
        # Installed one_batch's factory uses the BenchArgs class default here,
        # which is a tuple. Keep this private request metadata explicit; do not
        # mutate the module-wide BenchArgs default or the serving configuration.
        request.sampling_params.max_new_tokens = case.output_tokens
    events = [event_factory() for _ in range(4)]
    runner.clear()
    runner.synchronize()
    batch = None
    overall_start = clock_ns()
    prefill_start = clock_ns()
    try:
        events[0].record()
        next_ids, _, batch = runner.extend(requests)
        events[1].record()
        runner.synchronize()
        prefill_end = clock_ns()
        if _batch_width(next_ids) != case.batch:
            raise ValueError('Prefill changed the logical batch size')
        decode_start = clock_ns()
        events[2].record()
        # Prefill produces the first output; the remaining outputs need decode.
        for _ in range(case.output_tokens - 1):
            next_ids, _ = runner.decode(next_ids, batch)
        events[3].record()
        runner.synchronize()
        decode_end = clock_ns()
        overall_end = clock_ns()
        if _batch_width(next_ids) != case.batch:
            raise ValueError('Decode changed the logical batch size')
        seconds = (overall_end - overall_start) / 1e9
        prefill_seconds = (prefill_end - prefill_start) / 1e9
        decode_seconds = (decode_end - decode_start) / 1e9
        if min(seconds, prefill_seconds, decode_seconds) <= 0:
            raise ValueError('Nonpositive measured duration')
        output_count = case.batch * case.output_tokens
        decode_count = case.batch * (case.output_tokens - 1)
        return {
            'protocol': PROTOCOL, 'case_id': case.key, **asdict(case),
            'input_sha256': digest, 'elapsed_clock': 'CLOCK_MONOTONIC_RAW',
            'prefill_raw_seconds': prefill_seconds, 'decode_raw_seconds': decode_seconds,
            'full_sequence_raw_seconds': seconds,
            'prefill_cuda_ms': events[0].elapsed_time(events[1]),
            'decode_cuda_ms': events[2].elapsed_time(events[3]),
            'prefill_calls': 1, 'decode_calls': case.output_tokens - 1,
            'generated_output_tokens': output_count, 'decode_output_tokens': decode_count,
            'generated_output_tokens_per_second': output_count / seconds,
            'decode_output_tokens_per_second': decode_count / decode_seconds,
            'includes_input_tokens_in_output_throughput': False,
            'sampling_mode': 'greedy static-batch diagnostic',
            'numerical_correctness_established': False,
        }
    finally:
        if batch is not None:
            runner.cleanup(batch)
