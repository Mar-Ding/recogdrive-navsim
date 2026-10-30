#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/ruiqiwang/recogdrive_baseline/hf_cache
export PIP_CACHE_DIR=/data/ruiqiwang/recogdrive_baseline/pip_cache
export TMPDIR=/data/ruiqiwang/recogdrive_baseline/tmp
cd /data/ruiqiwang/recogdrive_baseline
mkdir -p tmp models data logs
/home/yiding/TurboVLA-repro/env/bin/python -m venv --system-site-packages env
env/bin/python -m pip install 'numpy==1.26.4' 'hydra-core==1.3.2' 'transformers==4.51.2' 'diffusers==0.32.2' 'timm==1.0.15' 'einops' 'accelerate' > logs/install.log 2>&1
env/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for name in ['ReCogDrive-VLM-2B','ReCogDrive-2B-IL']:
    snapshot_download('owl10/'+name,local_dir='models/'+name)
PY
