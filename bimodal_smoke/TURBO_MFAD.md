# Turbo-MFAD 同配置试跑

2026-10-30，沿用 TurboAD expanded 的缓存特征和数据划分：129 train / 338 val，4/2 个日志；seed=42、batch=32、20 epoch / 100 step、AdamW lr=1e-4、weight_decay=1e-4、BF16、clip=1。输入、冻结 DINOv3、LiDAR CNN、DiffusionDrive 轨迹头、监督和内部验证选权重规则均相同，仅融合模块改为现有 MFFusion。公共模块在融合模块之前构建，保证相同随机种子的公共初始权重一致。

MF 使用已有 MFBlock、模态路由 FFN、SharedQKVOAttention、空间 mRoPE 与 SDPA；RGB 和 LiDAR 都是 clean conditioning stream，可双向互见。没有辅助 flow matching 预训练，不将此试跑描述为完整 Multimodal Flow 训练配方复现。TurboAD / MF 可训练参数约 16.52M / 21.64M，并非等参数量消融。

| 指标 | TurboAD | Turbo-MFAD |
|---|---:|---:|
| 初始验证 ADE (m) | 10.113 | 10.130 |
| 第20轮验证 ADE (m) | 6.026 | 6.274 |
| 最佳验证 ADE (m) | 5.379 | 6.114 |
| 初始→最终验证 FDE (m) | 18.423→10.253 | 18.266→10.984 |
| 最终验证 loss | 16.410 | 16.202 |
| 训练阶段秒数 | 58.7 | 69.8 |
| 峰值 allocated 显存 GiB | 1.843 | 2.714 |

训练阶段使用缓存的视觉特征，不含原图编码、数据下载、CUDA 上下文或 PDM 评测。MF loss、梯度和轨迹均有限，能够正常优化；当前单种子极小样本下，内部验证 ADE 尚未优于 cross-attention，不能据此判断总体优劣。

复现（服务器项目根目录的 bimodal_smoke 中）：

```bash
GPU=0 bash run.sh turboad_train.py --variant mf --run mf_expanded_seed42 --epochs 20 --batch 32
GPU=2 bash run.sh turbo_mfad_eval_pipeline.py
```

权重位于 `/data/ruiqiwang/recogdrive_baseline/turboad/mf_expanded_seed42`，保留 latest 和 best_val_ade。原有 TurboAD 权重不覆盖。评测调用与 TurboAD 相同的固定 ReCogDrive 原始 scorer，使用同一 200 navtest token 和相同缓存；这是子集评测。

## 同一 200 场景的官方 PDMS 结果

| 指标 | TurboAD | Turbo-MFAD |
|---|---:|---:|
| 初始 PDMS ×100 | 7.63 | 11.83 |
| 训练后 PDMS ×100 | 46.44 | **61.44** |
| 无责任碰撞指标均值 | .770 | .885 |
| 可行驶区域合规 | .835 | .915 |
| ego progress | .406 | .511 |
| TTC 合规 | .650 | .815 |
| 舒适度 | .815 | .740 |
| 成功/失败场景 | 200/0 | 200/0 |

两者均采用内部验证 ADE 最优 checkpoint；navtest 不选权重。逐 token 集合核验通过。MF 本轮 PDMS 高 14.99 分，但验证 ADE 较差，且舒适度更低；误差指标与闭环代理评分不完全一致。结果支持“MF 版本训练有效、本次子集评分更好”，不支持单种子极小样本下的稳健优越性结论。参数量和初始 PDMS 也不相等。

训练曲线见 `evidence/turboad/training_comparison.png`；逐场景 CSV、配置及核验结果均已归档。所有训练/评分已结束，GPU1 未使用。
