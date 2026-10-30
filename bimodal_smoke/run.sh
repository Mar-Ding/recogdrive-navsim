#!/usr/bin/env bash
set -euo pipefail
cd /data/ruiqiwang/recogdrive_baseline/bimodal_smoke
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
case "$CUDA_VISIBLE_DEVICES" in 0|2) ;; *) echo 'Only GPU0 or GPU2 allowed'; exit 2;; esac
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export TMPDIR="$PWD/tmp" MPLCONFIGDIR="$PWD/tmp/matplotlib" HF_HOME="$PWD/tmp/huggingface"
export PYTHONPATH="$PWD:/data/ruiqiwang/recogdrive_baseline/repo:/home/yiding/Multimodal-Flow2VLA/upstream/Multimodal-Flow-main/src:/home/yiding/Multimodal-Flow2VLA/upstream/TurboVLA-native"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" artifacts outputs
exec /home/yiding/TurboVLA-repro/env/bin/python -u "$@"
