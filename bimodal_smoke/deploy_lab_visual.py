"""Deploy adapted paths/protocol only; preserve AutoDL source snapshots."""
import json,hashlib,warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import paramiko
HERE=Path(__file__).resolve().parent;SOURCE=HERE/'evidence/visual_lab_migration/autodl';BASE='/data/ruiqiwang/recogdrive_baseline/visual_vla'
cfg=dict(l.split(':',1) for l in Path('login.md').read_text().splitlines() if ':' in l)
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(cfg['ip'],port=int(cfg['port']),username=cfg['user'],password=cfg['passwd'],timeout=25)
s=c.open_sftp();hashes={}
for name in ('model.py','visual_mf_vla.py','turbo_cross_vla.py','train_turbo_cross_vla.py','train_visual_mf_vla.py'):
 data=(SOURCE/name).read_text().replace('/root/autodl-tmp/mf-vla',BASE)
 if name=='visual_mf_vla.py':data=data.replace('from mf.modeling.layers import RMSNorm','from mf.modeling.layers import RMSNorm\nif not hasattr(nn,"RMSNorm"): nn.RMSNorm=RMSNorm')
 if name.startswith('train_'):
  data=data.replace("else:history=[dict(epoch=0,**evaluate())];save_json('history.json',history)","else:\n        torch.save(dict(model=net.state_dict(),epoch=0,config=config),OUT/'initial.pt')\n        history=[dict(epoch=0,**evaluate())];save_json('history.json',history)")
  data=data.replace("pdms='not run; test data/maps/cache not yet prepared'","pdms='automatic fixed-200 lab navtest evaluation queued by pipeline'")
  if name=='train_turbo_cross_vla.py':data=data.replace("fusion='existing TransformerEncoder chunk-causal VLA'","fusion='TurboVLA BiAttentionBlock VL + action cross-attention'")
  # Both methods must start from identical train-only kmeans anchors.
  data=data.replace("torch.manual_seed", "torch.manual_seed")
 with s.open(BASE+'/multiview/'+name,'w') as f:f.write(data)
 hashes[name]=hashlib.sha256(data.encode()).hexdigest()
for name in ('lab_visual_prepare.py','lab_visual_score.py','lab_visual_pipeline.py'):
 s.put(str(HERE/name),BASE+'/multiview/'+name)
shell='''#!/usr/bin/env bash
set -euo pipefail
cd /data/ruiqiwang/recogdrive_baseline/visual_vla
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export TMPDIR="$PWD/tmp" HF_HOME="$PWD/tmp/huggingface" MPLCONFIGDIR="$PWD/tmp/matplotlib"
export PYTHONPATH="$PWD/multiview:/data/ruiqiwang/recogdrive_baseline/repo:$PWD/src/Multimodal-Flow/src"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" outputs/logs
exec /home/yiding/TurboVLA-repro/env/bin/python -u "$@"
'''
with s.open(BASE+'/run.sh','w') as f:f.write(shell)
with s.open(BASE+'/multiview/sitecustomize.py','w') as f:f.write("import sys\nsys.path.append('/data/ruiqiwang/recogdrive_baseline/env/lib/python3.10/site-packages')\n")
with s.open(BASE+'/migration_sources.json','w') as f:f.write(json.dumps(hashes,indent=2))
report='''# 纯视觉三模态模型迁移训练日志

2026-11-02：从 AutoDL 迁移两版本源代码：TurboVLA 双向 VL cross-attention + action cross-attention；MF-VLA（视觉/语言/动作三路，轮换 flow MSE + 0.1 cosine，辅助权重前 60% 为 0.1 后线性归零）。冻结 DINOv3 / BERT；ego 为条件；DiffusionDrive 轨迹生成。

使用实验室原 demo10 数据清单，8688 训练 / 1930 验证；前视 288×160，左右各 144×80。保留 AutoDL 设计，重新初始化训练以复用本实验室历史协议；不接续 AutoDL 不同划分的权重。200 epoch，effective batch 128，微批量自动探测，seed42，AdamW / BF16 / clip1 / warmup3 + cosine。训练后自动以最佳验证 ADE 权重在原固定 200 navtest 场景上运行原生 PDM scorer，另评初始化权重。GPU0 cross，GPU2 MF；不操作 GPU1。

源代码哈希见 migration_sources.json；数据协议与哈希见 outputs/shared/protocol.json；进度 outputs/pipeline_status.json；各阶段日志 outputs/logs/；各版 history.json、config.json、evaluation_trained.json 保存结果。耗时预测待实验室 3090 前几轮实测速率更新。
'''
with s.open(BASE+'/RUN_LOG.md','w') as f:f.write(report)
s.close()
_,o,e=c.exec_command('cd '+BASE+'; bash run.sh -c "import torch,transformers; from model import Planner; from turbo_cross_vla import CrossPlanner; from visual_mf_vla import MFPlanner; print(\'IMPORT_OK\',torch.__version__,transformers.__version__)"',timeout=120)
print(o.read().decode(errors='replace'));print(e.read().decode(errors='replace'));code=o.channel.recv_exit_status();c.close();raise SystemExit(code)
