#!/usr/bin/env bash
# Runs train.py in a bubblewrap sandbox: no network, no capabilities, read-only inputs.
# Linux (or WSL2 with /dev/dxg for the GPU). Set these variables before running:
#   CR_VENV            Linux virtual environment with torch and transformers (bound to /venv)
#   CR_PYTHON_BASE     base Python install that CR_VENV links to (bound at the same path)
#   CR_EXTRA_PACKAGES  site-packages folder that holds peft, accelerate and their dist-info folders
#   CR_BASE_MODEL      pinned Qwen/Qwen3.5-0.8B snapshot folder (bound to /qwen-base)
#   CR_ADAPTER         mixed-SFT adapter folder to continue (bound to /adapter)
#   CR_TRAIN_JSONL     training labels, one JSON row per session (bound to /data/train.jsonl)
#   CR_OUT             output root; checkpoints go to $CR_OUT/training
# Optional:
#   CR_PLAN            training plan (default: training-plan.json next to this script)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CR_VENV:?}" "${CR_PYTHON_BASE:?}" "${CR_EXTRA_PACKAGES:?}" "${CR_BASE_MODEL:?}" "${CR_ADAPTER:?}" "${CR_TRAIN_JSONL:?}" "${CR_OUT:?}"
plan="${CR_PLAN:-$here/training-plan.json}"
mkdir -p "$CR_OUT/training"
ulimit -f 1048576
ulimit -u 512
exec timeout --kill-after=10s 7200s bwrap --unshare-all --die-with-parent --new-session --cap-drop ALL \
 --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/sbin /sbin --symlink usr/lib /lib --symlink usr/lib64 /lib64 --ro-bind /etc/ld.so.cache /etc/ld.so.cache \
 --ro-bind "$CR_VENV" /venv --ro-bind "$CR_PYTHON_BASE" "$CR_PYTHON_BASE" \
 --ro-bind "$here" /input \
 --dir /data --ro-bind "$CR_TRAIN_JSONL" /data/train.jsonl --ro-bind "$plan" /data/training-plan.json \
 --ro-bind "$CR_BASE_MODEL" /qwen-base \
 --ro-bind "$CR_ADAPTER" /adapter \
 --ro-bind "$CR_EXTRA_PACKAGES/peft" /extra/peft \
 --ro-bind "$CR_EXTRA_PACKAGES/accelerate" /extra/accelerate \
 --ro-bind "$CR_EXTRA_PACKAGES/accelerate-1.12.0.dist-info" /extra/accelerate-1.12.0.dist-info \
 --ro-bind "$CR_EXTRA_PACKAGES/peft-0.18.1.dist-info" /extra/peft-0.18.1.dist-info \
 --bind "$CR_OUT/training" /output \
 --proc /proc --dev /dev --dev-bind /dev/dxg /dev/dxg --tmpfs /dev/shm --tmpfs /tmp \
 --clearenv --setenv PATH /venv/bin:/usr/bin --setenv PYTHONPATH /extra --setenv HOME /tmp --setenv TMPDIR /tmp \
 --setenv USER training --setenv LOGNAME training --setenv TORCHINDUCTOR_CACHE_DIR /tmp/inductor \
 --setenv LD_LIBRARY_PATH /usr/lib/wsl/lib --setenv TRITON_LIBCUDA_PATH /usr/lib/wsl/lib --setenv OMP_NUM_THREADS 4 --setenv MKL_NUM_THREADS 4 \
 --setenv HF_HUB_OFFLINE 1 --setenv TRANSFORMERS_OFFLINE 1 --setenv HF_HUB_DISABLE_TELEMETRY 1 \
 --setenv TOKENIZERS_PARALLELISM false --setenv PYTHONDONTWRITEBYTECODE 1 --setenv CUDA_CACHE_PATH /tmp/cuda-cache \
 --chdir /input -- /venv/bin/python -B /input/train.py
