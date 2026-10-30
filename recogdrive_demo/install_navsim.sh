#!/bin/bash
set -euo pipefail
cd /data/ruiqiwang/recogdrive_baseline
export TMPDIR=$PWD/tmp PIP_CACHE_DIR=$PWD/pip_cache HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
# Install into the dedicated overlay only.
env/bin/python -m pip install 'pytorch-lightning==2.4.0' 'torchmetrics==1.4.3' 'numpy==1.26.4' 'geopandas==0.14.4' 'shapely==2.0.6' 'bokeh==2.4.3' 'control==0.9.1' 'guppy3==3.1.2' 'SQLAlchemy==1.4.27' 'pyarrow' 'pyquaternion' 'pyogrio' 'rasterio' 'rtree' 'retry' 'pyinstrument' 'ujson' 'ray' 'aioboto3' 'aiofiles' 'selenium' 'positional-encodings==6.0.1' 'tensorboard==2.16.2' 'protobuf==4.25.3'
env/bin/python -m pip install 'nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2' --no-deps
env/bin/python -m pip install -e repo --no-deps
