# Turbo-MFAD flow 辅助试验

## 原始实现核对

服务器 `/home/yiding/Multimodal-Flow2VLA`：

- `vla/extension_model.py` 的 `flow_loss`：目标模态 detach，采样标准高斯 noise 和均匀时间 t，构造 `x_t=(1-t)*noise+t*clean`。条件模态时钟为 1，目标时钟为 t。
- `vla/mf_action_extension.py` 的 `flow_objective`：速度目标为 `clean-noise`，按有效 token 和通道平均 MSE。
- `vla/dense_experiment.py` 的 E010：辅助权重前 60% 更新为 0.1，后 40% 线性衰减至 0；原任务在 image/text/action 间轮换。
- 已核对的路径没有余弦辅助损失；cosine 学习率与特征相似度诊断不能混称为训练目标。

原始源文件已归档到 `evidence/demo10/flow_audit/`。

## 本次适配

保留已完成的 Turbo-MFAD 无辅助结果，新增独立 `demo10/turbo_mfad_flow`。使用同一数据 manifest、train8688/val1930、seed42、有效 batch128、BF16、AdamW、200 epochs/13600 updates，仍按最低验证 ADE 选权重。只使用 GPU2。

训练目标 `trajectory_loss + weight(step) * flow_velocity_mse`。逐更新轮换 RGB、LiDAR 目标；余弦相似度只作诊断，不加入 loss。训练日志分别记录总损失、轨迹损失、两种 flow MSE、两种速度 cosine 与梯度范数。

RGB target 是冻结 DINOv3 的 768维特征；LiDAR target 是固定非重叠 16×16 BEV patch 的 256维向量，共256 tokens。二者均使用固定 per-token layer_norm，避免学习到的 CNN 自己移动重建目标。辅助分支有独立输入投影、速度输出头和 TimestepEmbedder；复用同一 MF fusion 参数与空间位置。目标 noisy，另一模态 clean。没有把未来轨迹作为条件输入。

这是双传感器辅助 flow 的迁移，不是完整 image/text/action 配方、独立预训练或完整 registry 生命周期的复现。DiffusionDrive 轨迹解码保持原任务，暂不另增加 action flow 模态。辅助分支不用于推理；评测导出时仅保留严格匹配原 BimodalPlanner 的参数。

## 检查与运行

真实缓存8样本、20步：RGB flow MSE 2.32124 → 1.90386；LiDAR 1.81308 → 1.60605。反向梯度有限、MF attention 收到梯度、t=0 目标信息泄漏检查通过。该结果仅是固定噪声小样本 smoke，不代表测试收益。

batch64 峰值 allocated 8.44 GiB；batch128 为16.44 GiB、reserved17.86 GiB，约0.467秒/更新（2 warmup + 3 measured）。选择 batch128，不累积。

`GPU=2 bash run.sh flow_pipeline.py` 依次执行200 epoch训练、按val ADE导出推理权重、调用原生NAVSIM scorer。评分使用原200 navtest tokens与metric cache；逐token核验200/200有效后保存 `demo10/score_mf_flow.csv`。失败记录在 `demo10/flow_pipeline_status.json`，不会自动换数据或协议。

初步估计训练加验证2–3小时，评测另计；需用首轮实际耗时校准。此前无flow MF的PDMS为0.75240693，本次完成后同口径对照。
