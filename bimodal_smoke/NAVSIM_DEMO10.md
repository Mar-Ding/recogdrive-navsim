# NAVSIM 约 10% 数据与单卡 IL

## 数据定义

取 OpenDriveLab/OpenScene 官方传感器分片 **0–19 / 200**，按固定 ReCogDrive 提交 `5831ee83` 的 `navtrain.yaml` 过滤，不再对每段日志截断到 64 个场景。

全量元数据实际计数为 103,288 个场景；选中 **10,618（10.28%）**，共 126 段日志。这是前 10% 分片对应的近似 10% 场景，并非按时间排序的前 10% token。

- 训练：8,688 个场景，107 段日志。
- 内部验证：1,930 个场景，19 段日志；按 SHA256 排序确定，并将旧验证日志保留在验证侧。
- 测试：保持原来的 200 个 navtest token、原生 ReCogDrive PDM scorer、仿真配置和地图。仅验证 ADE 选权重，测试成绩不参与选择。
- 下载选中场景的 F0/L0/R0 图像及 LiDAR；本轮两种模型只输入 F0 + LiDAR，并使用相同 ego status。
- 原始分片完整下载并校验文件大小和上游 SHA256，再只解压选中场景的传感器路径。下载支持断点续传，保留已有项目与权重。

40 个相机/LiDAR 压缩包合计 212,421,299,544 字节（约 197.8 GiB），第 0 分片已存在。部署时 `/data` 可用约 403 GiB；预算校验额外为传感器、特征和磁盘余量预留 70 GiB，无需清理。

## 官方训练预算核查

来源是服务器固定版本的实际文件，副本在 `evidence/demo10/official_budget`：

| 配置 | 卡数 | 每卡 batch | 累积 | 有效 batch | epochs |
|---|---:|---:|---:|---:|---:|
| 常规 IL：`scripts/training/run_recogdrive_train_multi_node_2b.sh` + default_training.yaml | 8 | 16 | 1 | 128 | 200 |
| Accelerate IL：`scripts/training_accelerate/run_recogdrive_train_multi_node_2b.sh` | 8 | 32 | 1 | 256 | 100 |

常规 IL 优化器：AdamW，lr=1e-4，weight decay=1e-4，betas=(0.9,0.95)，梯度裁剪 1；3 epoch warmup、200 epoch cosine、min lr=1e-6。此次选择常规 IL 预算作为迁移参考，未执行 VLM 预训练或 GRPO/RL，也不意味着复现原始模型的全量训练精度。

公开来源：[官方 IL 脚本](https://github.com/xiaomi-research/recogdrive/blob/main/scripts/training/run_recogdrive_train_multi_node_2b.sh)、[默认训练配置](https://github.com/xiaomi-research/recogdrive/blob/main/navsim/planning/script/config/training/default_training.yaml)、[优化器和调度器](https://github.com/xiaomi-research/recogdrive/blob/main/navsim/agents/recogdrive/recogdrive_agent.py)。这些链接可能更新，复现以保存的固定提交副本为准。

## 本轮配置与显存

| 模型 | GPU | micro-batch | 累积 | 有效 batch | batch128 allocated / reserved |
|---|---:|---:|---:|---:|---:|
| TurboAD | 0 | 128 | 1 | 128 | 6.69 / 7.32 GiB |
| Turbo-MFAD | 2 | 128 | 1 | 128 | 9.90 / 11.13 GiB |

显存是 RTX3090 上真实 BF16 前向、反向、裁剪、AdamW 更新的实测峰值；DINOv3 冻结并离线缓存，不含在线训练视觉编码器的开销。effective_batch = micro_batch × accumulation × world_size。batch128 已达到所选官方有效 batch，不再为填满显存而增加有效 batch；不足时可改 micro64 × accumulation2。

两模型均从 seed42 初始化重新训练，200 epochs，采用上述优化器和 warmup/cosine，模型、输入分辨率及 DiffusionDrive 解码器沿用既有基线。每 epoch 68 次更新，总 **13,600 次更新/模型**；末个 batch 为 112 个样本，梯度按实际样本数加权。验证噪声 seed123，测试推理噪声 seed999。完整相同的数据 manifest SHA256 写入配置与 checkpoint。

估计每模型训练及验证 1.5–2 小时，两张卡同时运行，合计约 3–4 GPU 小时。另有数据下载、解压、一次视觉编码和 4 次 200 场景评测；下载时间取决于镜像吞吐。该时间为实测吞吐推算，最终以 summary 为准。

## 部署与状态

服务器根目录：`/data/ruiqiwang/recogdrive_baseline/demo10`，日志 `demo10/logs`；主调度日志暂在 `turboad/logs/demo10_pipeline.log`。后台流程已启动：下载/校验 → 编码 → 两模型各单卡训练 → 各自初始与最佳验证权重的固定测试评测。

`pipeline_status.json` 表示当前阶段；训练过程中各模型的 `progress.json`、`history.json` 可查，完成后生成 `summary.json`、`comparison.json`。失败会记录 traceback；下载可断点续传，训练可恢复 optimizer 和 epoch，既有基线不覆盖。再次启动前须确认没有同一 pipeline 在运行。

```bash
cd /data/ruiqiwang/recogdrive_baseline/bimodal_smoke
GPU=0 bash run.sh demo10_pipeline.py
```

本地收集已存在的轻量结果：`python bimodal_smoke/collect_demo10.py`。

## 10% 数据基线已完成（2026-10-31）

实际划分为 8,688 train / 1,930 val；200 epoch，13,600 optimizer steps，有效 batch128。固定 navtest 200 场景全部有效。以下 PDMS 均为原始 0–1 数值。

| 模型 | 单卡训练时间 | 峰值 allocated 显存 | 最佳内部验证 ADE | 初始 → 训练后 PDMS |
|---|---:|---:|---:|---:|
| TurboAD | 95.5 分钟 | 6.69 GiB | 0.754 m | 0.07634 → 0.70645 |
| Turbo-MFAD | 109.2 分钟 | 9.89 GiB | 0.780 m | 0.11831 → 0.75241 |

该结果为单种子 demo 子集对比；证据见 `evidence/demo10`。训练后权重按内部验证 ADE 选择。

## 多视角与辅助视角缩小实验

两版沿用上面相同数据、训练预算和评测协议，均使用 MF 融合及 DiffusionDrive 解码器：

| 版本 | GPU | 前视 | 左右辅助视角（各） | 视觉 token |
|---|---:|---|---|---:|
| TurboAD-Plus | 0 | 288×160 | 288×160 | 540 |
| TurboAD-Plus-small | 2 | 288×160 | 144×80 | 270 |

前视特征直接复用；逐条验证前视、LiDAR、状态、目标及 token/split/log 与已有基线完全一致。辅助图像真实缩小后经冻结 DINO 编码；没有使用裁切旧特征代替缩小图像。两版均已通过真实编码数据前向/反向检查，所有视角和 LiDAR 输入梯度均有限且非零。位置参数形状不同，两版不是严格相同初始化的消融。

后台 `demo10_plus.py pipeline` 同时调度两卡，编码后以真实输入测试 micro batch64/128，在 reserved 显存低于21 GiB的候选中选择较大者；有效 batch 始终128。训练后自动评测初始与最佳验证权重，结果目录分别为 `demo10/turboad_plus`、`demo10/turboad_plus_small`。状态为 `plus_status.json` / `plus_small_status.json`，日志为 `demo10/logs/plus*`。本次提交时尚在编码，不能视为新两版训练已完成。

## 前一轮扩大训练结果

原有 129/338 划分、batch32、恒定 lr1e-4，从 epoch20 恢复到 epoch100（500 steps）：

| 模型 | 最佳验证 ADE：20→100 epoch | 固定 200 场景 PDMS：20→100 epoch |
|---|---:|---:|
| Turbo-MFAD | 6.11 → 3.60 m | 61.44 → 36.63 |
| TurboAD-Plus | 6.03 → 4.05 m | 43.88 → 23.82 |

两次评测均 200/200 有效。内部验证改善没有转化为固定测试收益，单纯增加小样本训练轮数不够；可能存在过拟合和验证分布代表性不足，不能只凭此区分原因。旧权重及 CSV 全部保留；本轮扩充数据另立实验，避免覆盖和混比。
