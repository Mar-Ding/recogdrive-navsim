# TurboAD 有界训练试验

TurboAD 指当前去掉 VLM/LLM 的前视 RGB + LiDAR cross-attention 版本：冻结 DINOv3、可训练 LiDAR CNN、TurboVLA BiAttentionBlock、DiffusionDrive 轨迹头。MF 变体本轮不训练。

## 数据和优化

- 数据索引严格读取固定 ReCogDrive 提交的 `navtrain.yaml` 中 log_names、tokens、4 帧历史、10 帧未来及 route 过滤；模型仅输入当前帧，监督使用官方坐标转换得到的未来 8 个 0.5 秒轨迹点。
- 每个日志最多均匀选择 64 个候选 token，只保留前视图和 LiDAR 均取得的样本。子集受第 0 个传感器分片可用范围限制，实际样本数以 manifest 和 summary 为准。
- 按日志 SHA256 排序，将约 15% 的完整日志留作内部验证，与训练日志无交集。此内部验证不是官方 navtest 分数。
- seed=42，batch=32，20 epoch，AdamW lr=1e-4、weight_decay=1e-4，BF16，梯度裁剪 1。随机初始化融合、LiDAR 编码和轨迹头；不加载 ReCogDrive 或 DiffusionDrive 规划权重。
- 每轮记录训练目标、固定噪声验证目标、推理 ADE/FDE、移动场景 ADE 和停车预测参照。保留 latest 和按内部验证 ADE 选择的 best checkpoint。navtest 不参与选 checkpoint。

## 官方评测协议

使用服务器固定 ReCogDrive 提交的原始 `run_metric_caching.py`、`run_pdm_score.py`、`pdm_score`、simulator/scorer 及其 YAML 参数，不重新定义 PDMS。地图采用 nuplan-maps-v1.0，模型输出 4 秒轨迹。scorer 使用官方 40×0.1 秒 proposal sampling。

第一轮最多 200 个具有配对传感器的官方 navtest token，只验证官方评测链路。**这仍是 navtest 子集，不是完整官方 benchmark，不能直接和论文总分比较。** 完整 navtest 需要另补其他传感器分片；不将未准备的场景默认为已评测。

服务器根目录 `/data/ruiqiwang/recogdrive_baseline`：

```bash
cd /data/ruiqiwang/recogdrive_baseline/bimodal_smoke
GPU=0 bash run.sh turboad_data.py index
GPU=0 bash run.sh turboad_data.py sensors
GPU=0 bash run.sh turboad_data.py encode
GPU=0 bash run.sh turboad_train.py --epochs 20 --batch 32
bash run.sh turboad_eval_data.py maps
bash run.sh turboad_eval_data.py prepare
bash run.sh turboad_eval.py cache --tokens ../turboad/navtest_subset_tokens.json
GPU=0 bash run.sh turboad_eval.py score --tokens ../turboad/navtest_subset_tokens.json
```

下载有超时，流水线总训练阶段限时一小时；仅 GPU0，GPU1 不使用。过程状态保存在 `turboad/pipeline_status.json`，日志在 `turboad/logs`，训练结果在 `turboad/pilot_seed42`。`turboad_pipeline.py` 等待本轮两个训练分片下载完成后，依次配对、编码、训练；不会自动启动完整训练、RL 或 MF。

地图原始 S3 连接重置后，改从 nuPlan CDN `https://d1qinkmu0ju04f.cloudfront.net/public/nuplan-v1.1/nuplan-maps-v1.1.zip` 下载，解压得到 nuplan-maps-v1.0。训练与测试传感器从 OpenDriveLab/OpenScene 镜像的各自第 0 分片取得。旧 smoke checkpoint 不覆盖。

## 2026-10-30 实测记录

已完成两次 seed=42、batch=32、20 epoch 的 TurboAD 训练。训练/验证按日志隔离（4/2 个日志），没有使用 navtest 选权重。

| 试验 | 训练/验证样本 | step | 初始→最终 ADE (m) | 初始→最终 FDE (m) | 最佳验证 ADE | 训练阶段耗时 |
|---|---:|---:|---:|---:|---:|---:|
| pilot | 116/128 | 80 | 9.208→5.658 | 16.508→10.058 | 5.590 | 50.9 秒 |
| expanded | 129/338 | 100 | 10.113→6.026 | 18.423→10.253 | 5.379 | 58.7 秒 |

expanded 使用 `dense_index` 补全同一 6 个日志的候选 token。验证集增加较多，训练集只增加 13 个，因此不能将两行当成受控的数据规模对比。相邻帧相关，实际日志数仍很少。两轮训练的 loss、梯度均有限，验证轨迹误差下降；后段有波动，不能认定已充分收敛。

峰值 PyTorch allocated 显存均 1.843 GiB。此处冻结 DINOv3 后缓存特征，训练耗时和显存不包含视觉编码、数据准备、CUDA 上下文或官方评测；端到端评测实际重新读取 RGB/LiDAR 并运行 DINOv3。

官方原始 scorer 的同一 200 个 navtest token：随机初始化 PDMS=0.0763393，pilot 内部验证最优权重 PDMS=0.1756586，均 200/200 成功。PDMS 使用原始 [0,1] 单位；乘 100 分别为 7.63 和 17.57。这个低绝对分数符合极小样本、随机初始化规划头的试跑定位，不代表 ReCogDrive 复现精度。证据在 `evidence/turboad`。

扩充版复现命令：`GPU=0 bash run.sh turboad_data.py dense_index`，随后 `GPU=0 bash run.sh turboad_expanded_pipeline.py`。这会刷新当前 manifest/features，但保留 pilot 权重及已归档的 pilot manifest。评测使用 GPU2，GPU1 未使用。

扩充版内部验证最优权重在相同 200 场景 PDMS=0.4644476（46.44/100），200/200 成功。逐 token 核验三次评分集合完全一致、均无失败；训练/验证 token 与本次 navtest 无交集，训练与验证日志无交集。核验脚本 `verify_turboad_evidence.py`，结果 `evidence/turboad/verification.json`。pilot 与 expanded 的训练采样、更新步数和最佳轮次均有变化，不能将两者的 PDMS 差异归因于单一因素，也不能据此推断 MF 的收益。本次只验证 TurboAD 训练能够有效优化。
