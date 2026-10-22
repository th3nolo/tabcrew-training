# Speed recipes: Qwen3-4B-AWQ teacher on one RTX 3080 (10 GB)

The local teacher (`data-generation/local-sglang`) served `Qwen/Qwen3-4B-AWQ`
(revision `74d4bd2bd4bff9cafc9345221320bffb08b406a3`) with SGLang 0.5.17 in WSL2.
Four collector workers sent author and blind-review requests at the same time.
This page lists the serving settings, what changed throughput, and what did not.

## Serving settings

These are the flags that `data-generation/local-sglang/serve.py` passes to SGLang:

| Flag | Value | Note |
|---|---|---|
| `--dtype` | `float16` | |
| `--context-length` | `6144` | Maximum prompt plus output per request |
| `--max-running-requests` | `4` | One slot per collector worker |
| `--max-total-tokens` | `16384` | GPU KV pool. See the measurement below |
| `--mem-fraction-static` | `0.78` | |
| `--chunked-prefill-size` | `1024` | Long prompts are prefilled in 1,024-token pieces |
| `--cuda-graph-max-bs-decode` | `4` | CUDA graphs for decode batches 1 to 4 |
| `--cuda-graph-backend-prefill` | `disabled` | No CUDA graphs for prefill |
| `--attention-backend` | `triton` | |
| `--sampling-backend` | `pytorch` | |
| `--grammar-backend` | `xgrammar` | JSON schema constrained output for author and reviewer |
| `--default-chat-template-kwargs` | `{"enable_thinking":false}` | No thinking tokens |
| `--load-format` | `safetensors` | |
| `--model-loader-extra-config` | `{"enable_multithread_load":false}` | Single-threaded weight loading |
| `--weight-loader-drop-cache-after-load` | (flag) | Release file cache after the weights are on the GPU |
| `--enable-metrics`, `--disable-custom-all-reduce`, `--mm-feature-transport cpu` | | |

Plain SGLang equivalent (the collector starts it through `entrypoint.py`, which adds a streaming weight loader):

```bash
python -m sglang.launch_server --model-path <qwen3-4b-awq folder> --served-model-name qwen3-local-teacher \
  --host 127.0.0.1 --port 60647 --dtype float16 --load-format safetensors --context-length 6144 \
  --mem-fraction-static 0.78 --max-running-requests 4 --max-total-tokens 16384 \
  --chunked-prefill-size 1024 --cuda-graph-max-bs-decode 4 --cuda-graph-backend-prefill disabled \
  --attention-backend triton --sampling-backend pytorch --grammar-backend xgrammar \
  --enable-metrics --disable-custom-all-reduce --mm-feature-transport cpu \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

`sglang-runtime-pyproject.toml` is the project file of that runtime (SGLang 0.5.17, Python 3.12,
`exclude-newer = 2026-09-05`). Its lock file is `sglang-runtime-uv.lock` in this folder.

## Measurement: KV pool 8,192 vs 16,384 tokens

One collector revision ran with an 8,192-token KV pool. The next one went back to 16,384. Model, prompts,
decoding, loader, worker count and guards did not change.

| Measurement | 8,192 tokens | 16,384 tokens |
|---|---:|---:|
| Completed output tokens per generation second (whole pipeline) | 185.3 | 273.5 |
| Median decode-window tokens/s | 201.0 | 357.8 |
| Decode samples with queued requests | 92.6% | 1.1% |
| Generation time measured | 11.8 min | 4.3 min |

Throughput went from 185 to 274 tokens/s (+47.6%). The reason: before SGLang admits a request, it reserves
KV space for the prompt and for the future output. With 8,192 tokens, four workers did not mean four active
requests. In 1,003 of 1,442 decode samples only two requests were active; four were active in 80 samples.
The larger pool costs about 1.13 GiB more GPU memory.

Limits of this measurement: the two windows ran different sessions and retries, and the 16K window is only
4.3 minutes. It is an operational before/after observation, not a matched benchmark. An earlier, longer run
with the 16K pool and shorter prompts averaged 363.8 tokens/s over 23 minutes, with short decode windows up
to 623.5 tokens/s; that is not a sustained rate.

"Whole pipeline" divides completed author and reviewer output tokens by the time since the model was ready.
It includes prefill, queueing, CPU validation and rejected output, and excludes model loading.

## Host memory during startup

The machine has limited free RAM and Windows commit while the desktop runs. Startup was the risky phase.
These changes were tried; none of them changes model bytes or dependencies:

- One TorchInductor compile worker (`TORCHINDUCTOR_COMPILE_THREADS=1`, `MAX_JOBS=2`).
- `posix_fadvise(POSIX_FADV_DONTNEED)` on each model file after its checksum is verified and after the
  weights are loaded (`serve.py`), and on large installed `.so` files (`runtime_cache.py`). This asks Linux
  to drop clean file-cache pages. It is not a global cache flush.
- A streaming safetensors loader (`stream_weights.py`, `entrypoint.py`) that reads tensors into bounded
  buffers. It matched the reference reader byte for byte on FP16, BF16 and INT32 test tensors and validated
  all 902 tensor headers of the model. It was not compared tensor by tensor on the full model.
- A 12,288-token KV pool as a middle option.

Seven guarded startup trials stopped on the host guards (free commit under 6 GiB or free RAM under 1.5 GiB),
under changing desktop load. No complete benchmark passed in those trials, so they support no throughput claim.
The 24 GiB free-commit check in an earlier launcher was a conservative preflight value, not a measured need.

## AWQ re-shard script (untested)

`reshard-awq.py` rewrites an AWQ safetensors checkpoint into shards of about 256 MiB. Names, dtypes, shapes and
tensor bytes stay the same, and every tensor is hashed during the copy and again from the written file.
The idea is a lower host-memory peak during loading.

**Untested: never loaded in a serving run.** It was written for `Qwen/Qwen3-8B-AWQ` and is kept for reference.

## SM86 kernel experiments

Goal: raise whole-pipeline tokens/s of the 4B AWQ teacher with custom kernels for the RTX 3080 (SM86).
Result: **best observed +3.2%, inside run-to-run variation (inconclusive). The original SGLang kernels were kept.**

Where the time goes (one profiled window, decode batch 3 padded to 4, 29 decode steps):

| Operation | Share of decode GPU time |
|---|---:|
| AWQ Marlin matrix multiplications | 60.5% |
| FP16 dense GEMM (mostly the vocabulary projection) | 20.3% |
| Grouped attention (stage 1) | 10.4% |
| Fused residual RMSNorm | 3.6% |
| QK RMSNorm + RoPE | 1.2% |

Prefill steps were 85 to 86% Marlin.

What was tried:

| ID | Change | Outcome |
|---|---|---|
| QK-003 | Fused QK RMSNorm + cached RoPE Triton kernel (`sm86-qk-rope/`), used for CUDA-graph decode and 1,024-token prefill chunks; original path for smaller eager calls | Full pipeline: 337.8 to 348.5 tokens/s (+3.2%) over 2 runs per arm of 30 sessions with 4 concurrent pipelines. Runs spanned 326.8 to 352.2 (original) and 329.2 to 369.1 (candidate). Inconclusive. Not adopted. |
| QK-001, QK-002 | Earlier launch geometries of the same kernel | Slower for some shapes; replaced by QK-003 |
| MARLIN-001, -002, -008 | Existing smaller Marlin tile shapes for batches 1 to 4 | Some operator-level gains for some layers (about 1.3% on QKV for -008); no full-model gain shown |
| MARLIN-003 to -007, -009, -010 | Two blocks per SM and synchronization changes | Numerical mismatches at some batch sizes; not timed |
| LMHEAD-001 | Small-batch FP16 kernel for the vocabulary projection | Correct, but 2.4% to 4.7% slower than cuBLAS |
| SAMP-001 | Shorter seeded sampling path | Not reachable: the server samples without a seed |

The kernel passed 252 numerical cases against the original path (absolute tolerance 0.004, relative 0.002) on
all 36 layers, including CUDA-graph replay and the fallbacks. This is tolerance-level agreement, not bitwise.

`sm86-qk-rope/` holds the kernel (`qk_rope.py`) and an SGLang external model package
(`kernel_adapter/qwen3_sm86.py`). To try it, put the folder on `PYTHONPATH` of the SGLang process and set
`SGLANG_EXTERNAL_MODEL_PACKAGE=kernel_adapter`. The adapter refuses to load if SGLang's `qwen3.py` differs from
the reviewed 0.5.17 file. Set `QK_ROPE_DISABLE=1` to use the original path.
