#!/usr/bin/env bash
# Runs evaluate.py for one adapter in a bubblewrap sandbox: no network, no capabilities.
# Usage: launch-eval.sh original|repaired|expanded
#   original  = mixed-SFT adapter (CR_ADAPTER_ORIGINAL)
#   repaired  = first 128-label repair pilot, checkpoint 32 (CR_ADAPTER_REPAIRED)
#   expanded  = this run, checkpoint 744 (CR_ADAPTER_EXPANDED)
# Set CR_VENV, CR_PYTHON_BASE, CR_EXTRA_PACKAGES, CR_BASE_MODEL, CR_OUT as for launch-train.sh, and
#   CR_CASES           evaluation cases JSON (bound to /data/cases.json; see evaluate.py)
#   CR_ADAPTER_<MODE>  adapter folder for the selected mode
# Results go to $CR_OUT/evaluation.
set -euo pipefail
mode="${1:?original, repaired or expanded}"
case "$mode" in original|repaired|expanded) ;; *) exit 2;; esac
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CR_VENV:?}" "${CR_PYTHON_BASE:?}" "${CR_EXTRA_PACKAGES:?}" "${CR_BASE_MODEL:?}" "${CR_CASES:?}" "${CR_OUT:?}"
case "$mode" in
 original) adapter="${CR_ADAPTER_ORIGINAL:?}";;
 repaired) adapter="${CR_ADAPTER_REPAIRED:?}";;
 expanded) adapter="${CR_ADAPTER_EXPANDED:?}";;
esac
mkdir -p "$CR_OUT/evaluation"
ulimit -f 1048576
ulimit -u 512
exec timeout --kill-after=10s 2400s bwrap --unshare-all --die-with-parent --new-session --cap-drop ALL \
 --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/sbin /sbin --symlink usr/lib /lib --symlink usr/lib64 /lib64 --ro-bind /etc/ld.so.cache /etc/ld.so.cache \
 --ro-bind "$CR_VENV" /venv --ro-bind "$CR_PYTHON_BASE" "$CR_PYTHON_BASE" \
 --ro-bind "$here" /input \
 --dir /data --ro-bind "$CR_CASES" /data/cases.json \
 --ro-bind "$CR_BASE_MODEL" /qwen-base \
 --ro-bind "$adapter" /adapter \
 --ro-bind "$CR_EXTRA_PACKAGES/peft" /extra/peft \
 --ro-bind "$CR_EXTRA_PACKAGES/accelerate" /extra/accelerate \
 --ro-bind "$CR_EXTRA_PACKAGES/accelerate-1.12.0.dist-info" /extra/accelerate-1.12.0.dist-info \
 --ro-bind "$CR_EXTRA_PACKAGES/peft-0.18.1.dist-info" /extra/peft-0.18.1.dist-info \
 --bind "$CR_OUT/evaluation" /output \
 --proc /proc --dev /dev --dev-bind /dev/dxg /dev/dxg --tmpfs /dev/shm --tmpfs /tmp \
 --clearenv --setenv PATH /venv/bin:/usr/bin --setenv PYTHONPATH /extra --setenv HOME /tmp --setenv TMPDIR /tmp \
 --setenv USER training --setenv LOGNAME training --setenv TORCHINDUCTOR_CACHE_DIR /tmp/inductor \
 --setenv LD_LIBRARY_PATH /usr/lib/wsl/lib --setenv TRITON_LIBCUDA_PATH /usr/lib/wsl/lib --setenv OMP_NUM_THREADS 4 --setenv MKL_NUM_THREADS 4 \
 --setenv HF_HUB_OFFLINE 1 --setenv TRANSFORMERS_OFFLINE 1 --setenv HF_HUB_DISABLE_TELEMETRY 1 \
 --setenv TOKENIZERS_PARALLELISM false --setenv PYTHONDONTWRITEBYTECODE 1 --setenv CUDA_CACHE_PATH /tmp/cuda-cache \
 --chdir /input -- /venv/bin/python -B /input/evaluate.py "$mode"
