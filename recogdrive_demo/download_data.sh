#!/bin/bash
set -euo pipefail
cd /data/ruiqiwang/recogdrive_baseline/data
curl -fL --retry 4 'https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_test.tgz?download=true' -o metadata_test.tgz
tar -xzf metadata_test.tgz
curl -fL --retry 4 https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip -o maps.zip
unzip -q maps.zip
