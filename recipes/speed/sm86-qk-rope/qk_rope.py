"""SM86 (RTX 30xx) Qwen3 FP16 QK RMSNorm + cached NeoX RoPE in one Triton kernel.

Locally authored Triton implementation; contracts inspected in SGLang 0.5.17
qknorm.cuh, impl/norm.cuh and rope.cuh (Apache-2.0).
No TMA, WGMMA, PDL or Hopper-only synchronization. Each program handles a head.
Q/K output is separate so dispatch/compile fallback retains the original QKV.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _qk_rope(QKV, OUT, QW, KW, POS, CACHE,
             INPUT_STRIDE: tl.constexpr, OUTPUT_STRIDE: tl.constexpr,
             POSITION_STRIDE: tl.constexpr, CACHE_STRIDE: tl.constexpr,
             CACHE_ROWS: tl.constexpr, EPS: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, 128)
    x = tl.load(QKV + token * INPUT_STRIDE + head * 128 + d).to(tl.float32)
    weight = tl.load(tl.where(head < 32, QW + d, KW + d)).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 128) + EPS)
    # The baseline writes norm output as FP16 before RoPE reloads it as FP32.
    norm = (x * inv_rms * weight).to(tl.float16).to(tl.float32)
    partner = tl.gather(norm, (d + 64) % 128, axis=0)
    pos = tl.load(POS + token * POSITION_STRIDE)
    valid = (pos >= 0) & (pos < CACHE_ROWS)
    cos = tl.load(CACHE + pos * CACHE_STRIDE + d % 64, mask=valid, other=0)
    sin = tl.load(CACHE + pos * CACHE_STRIDE + 64 + d % 64, mask=valid, other=0)
    # Explicit FMA ordering matches the CUDA expression contraction for RoPE.
    rotated = tl.where(d < 64,
                       tl.fma(norm, cos, -(partner * sin)),
                       tl.fma(partner, sin, norm * cos))
    # Invalid positions never access cache OOB; tests require NaN signalling.
    rotated = tl.where(valid, rotated, float('nan'))
    tl.store(OUT + token * OUTPUT_STRIDE + head * 128 + d, rotated)


def supported(qkv, qw, kw, positions, cache):
    return (qkv.is_cuda and qkv.dtype == torch.float16 and qkv.ndim == 2
            and qkv.shape[1] == 6144 and qkv.stride(1) == 1
            and qkv.stride(0) >= 6144
            and all(w.shape == (128,) and w.dtype == torch.float16
                    and w.is_contiguous() and w.device == qkv.device for w in (qw, kw))
            and positions.ndim == 1 and positions.shape[0] == qkv.shape[0]
            and positions.dtype in (torch.int32, torch.int64)
            and positions.device == qkv.device and positions.stride(0) > 0
            and cache.ndim == 2 and cache.shape[1] == 128
            and cache.dtype == torch.float32 and cache.device == qkv.device
            and cache.stride(1) == 1 and cache.stride(0) >= 128)


def fused_qk_rope(qkv, qw, kw, positions, cache, eps=1e-6):
    if not supported(qkv, qw, kw, positions, cache) or eps != 1e-6:
        raise ValueError('Unsupported Qwen3 FP16 QK/RoPE contract')
    output = torch.empty((qkv.shape[0], 5120), dtype=qkv.dtype, device=qkv.device)
    if qkv.shape[0]:
        _qk_rope[(qkv.shape[0], 40)](
            qkv, output, qw, kw, positions, cache, qkv.stride(0), output.stride(0),
            positions.stride(0), cache.stride(0), cache.shape[0], eps, num_warps=1,
            enable_fp_fusion=False)
    return output[:, :4096], output[:, 4096:], qkv[:, 5120:]
