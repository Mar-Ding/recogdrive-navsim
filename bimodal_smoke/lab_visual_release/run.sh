#!/usr/bin/env bash
set -euo pipefail
cd /data/ruiqiwang/recogdrive_baseline/visual_vla
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export TMPDIR="$PWD/tmp" HF_HOME="$PWD/tmp/huggingface" MPLCONFIGDIR="$PWD/tmp/matplotlib"
export PYTHONPATH="$PWD/multiview:/data/ruiqiwang/recogdrive_baseline/repo:$PWD/src/Multimodal-Flow/src"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" outputs/logs
exec /home/yiding/TurboVLA-repro/env/bin/python -u "$@"
