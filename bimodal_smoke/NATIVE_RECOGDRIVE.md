# 原生 ReCogDrive 同数据规模对比

目的：补跑原生 ReCogDrive 模型，不以协议对照表替代实测成绩。

- 固定上游提交 5831ee83；保留原生 ReCogDriveFeatureBuilder、InternVL 2B VLM、small DiT IL planner、图像预处理与提示词。
- 官方 VLM 预训练权重冻结，缓存完整 2800×1536 hidden states。原生 builder 将 BF16 输出转为 FP32；逐样本验证 BF16→FP32 可精确还原后用 BF16 存储，无 token 压缩、裁剪或改变图像尺寸。
- IL planner 随机初始化，不加载已在其他数据上训练的官方 IL 规划权重，避免将其作为“只用当前10%数据训练”的结果。
- 同一 demo10 manifest，train8688 / val1930，seed42，200 epochs，13600 updates；effective batch128，BF16，AdamW lr1e-4 / wd1e-4 / betas(0.9,0.95)，warmup3 + cosine最低1e-6，clip1。
- 保持原生前视、历史运动、导航与状态输入；没有改成我们的 RGB+LiDAR 架构。因此是同数据规模的模型对比，不是输入模态完全一致的消融。
- 按最低内部验证 ADE 选择原生规划权重，固定原200个navtest token，用原始pdm_score / simulator / scorer与既有metric cache评分，随机噪声seed999。评测直接循环调用同一原生评分函数，加载路径形式的图像以适配原生特征构建接口。

## 已完成实测

8个真实场景：每个缓存8,604,061字节，完整缓存约85GiB；单样本VLM约0.6秒（首次约1.4秒）。

规划头完整前向/反向/AdamW：batch64峰值8.50GiB、约0.244秒；batch128峰值16.53GiB、约0.422秒。另通过真实缓存batch128训练及推理，输出128×8×3且有限。

数据编码使用GPU0与GPU2分别处理manifest奇偶行；编码完成后GPU2单卡训练和评测。GPU1不使用。缓存空间检查要求预计缓存之外至少20GiB余量。

## 运行与证据

服务器：`/data/ruiqiwang/recogdrive_baseline/demo10/native_recogdrive/`。

流程：`env/bin/python -u bimodal_smoke/native_recogdrive_run.py pipeline`；须先检查同一流程未在运行。

状态：pipeline_status.json、cache_0/1_progress.json、progress.json；结果：summary.json、evaluation.json、scores.csv。日志位于demo10/logs/native_*.log。

最初全流程4–6小时的估计已被实际整轮测速否定。前三轮分别785.24、766.81、800.15秒，平均784.07秒/轮（13.07分钟），外推200轮约 **43.56小时**。这是当前服务器、缓存读取和逐轮验证配置下的训练预算外推，不是已完成耗时，也不含VLM特征编码与最终PDM评测。

用户要求停止后，已在第3轮之后终止训练及控制进程，保留完整特征缓存、历史、最近与最佳权重。记录见 `bimodal_smoke/evidence/recogdrive_stopped_timing.json`，服务器同名实验目录下为 `stopped_timing.json`。当前没有原生同规模训练的最终PDMS。

独立计算测速：原生规划头batch128前向/反向/更新约0.416秒，验证推理约1.742秒。整轮784秒不能直接归因于扩散头计算；缓存读取、数据拼接传输等主机侧开销需单独测量。后续混合版将保留原生VLM、替换为已有DiffusionDrive轨迹头与必要接口适配，独立记录，不标为原生ReCogDrive。
