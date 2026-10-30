#!/usr/bin/env bash
set -euo pipefail
cd /data/ruiqiwang/recogdrive_baseline
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$PWD/hf_cache"
export MPLCONFIGDIR="$PWD/tmp/matplotlib"
export TMPDIR="$PWD/tmp"
export PYTHONPATH="$PWD/repo${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs "$TMPDIR" "$MPLCONFIGDIR"
env/bin/python -u run_demo.py 2>&1 | tee logs/demo.log
