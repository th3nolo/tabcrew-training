"""Use SGLANG_EXTERNAL_MODEL_PACKAGE=kernel_adapter in the owned runtime only.

The package is loaded by SGLang's existing model registry in each worker process.
No sitecustomize, global import hook, shared library or installed file is changed.
"""
import hashlib
import logging
import os
from pathlib import Path

import torch
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources
import sglang.srt.models.qwen3 as upstream
from qk_rope import fused_qk_rope, supported

EXPECTED_SOURCE = '39c7c9c6b0107f12ac76cd2b6b54a0694c8540c1bb9588a380e05fcd3ee93e6b'
if hashlib.sha256(Path(upstream.__file__).read_bytes()).hexdigest() != EXPECTED_SOURCE:
    raise RuntimeError('Qwen3 source changed; adapter requires a new review')
ORIGINAL = upstream.Qwen3Attention.forward_prepare_native
COUNTERS = {'custom_dispatch': 0, 'original_fallback': 0}
logger = logging.getLogger(__name__)


def prepare(self, positions, hidden_states):
    rope = self.rotary_emb
    # Every test stays metadata-only: no host/GPU synchronization during capture.
    allowed = (os.environ.get('QK_ROPE_DISABLE', '0') == '0'
        and self.head_dim == 128 and self.num_heads == 32 and self.num_kv_heads == 8
        and self.q_norm.variance_epsilon == self.k_norm.variance_epsilon == 1e-6
        and type(rope).__name__ == 'RotaryEmbedding'
        and rope.rotary_dim == 128 and rope.is_neox_style and rope.base == 1000000
        and not getattr(self.q_norm, 'cast_x_before_out_mul', False)
        and hidden_states.is_cuda and hidden_states.dtype == torch.float16
        and torch.cuda.get_device_capability(hidden_states.device) == (8, 6))
    if not allowed:
        COUNTERS['original_fallback'] += 1
        return ORIGINAL(self, positions, hidden_states)
    # Raw-clock eager measurements favor the installed path below the measured
    # 1024-token prefill shape. Captured decode replays avoid Python launch cost.
    if hidden_states.shape[0]<1024 and not torch.cuda.is_current_stream_capturing():
        COUNTERS['original_fallback'] += 1
        return ORIGINAL(self, positions, hidden_states)
    qkv, _ = self.qkv_proj(hidden_states)
    if supported(qkv, self.q_norm.weight, self.k_norm.weight, positions, rope.cos_sin_cache):
        # Out-of-place Q/K keeps QKV intact if compilation fails before launch.
        # CUDA execution errors deliberately propagate: continuing a poisoned CUDA
        # context would not be a valid fallback or correctness result.
        try:
            result = fused_qk_rope(qkv, self.q_norm.weight, self.k_norm.weight,
                                   positions, rope.cos_sin_cache)
        except (ValueError, torch.OutOfMemoryError, CompilationError, OutOfResources) as error:
            logger.warning('QK/RoPE original path fallback: %s', error)
        else:
            if COUNTERS['custom_dispatch']==0:
                logger.warning('CUSTOM_QK_ROPE_ACTIVE: FP16 sm86, separate QK output, original V retained')
            COUNTERS['custom_dispatch'] += 1
            return result
    COUNTERS['original_fallback'] += 1
    q, k, v = qkv.split([self.q_size,self.kv_size,self.kv_size], dim=-1)
    q, k = upstream.apply_qk_norm(q, k, self.q_norm, self.k_norm,
                                 self.head_dim, self.alt_stream)
    q, k = rope(positions, q, k)
    return q, k, v


upstream.Qwen3Attention.forward_prepare_native = prepare
logger.warning('SM86 QK/RoPE adapter installed; original callable retained')
EntryClass = upstream.Qwen3ForCausalLM
