# TurboAD-Plus 三视角试验

本版以 Turbo-MFAD 为基础，新增当前帧 CAM_L0 / CAM_R0，与 CAM_F0、LiDAR 联合输入。其他视角属于 RGB 的多视角扩展，不宣称增加新的物理传感器模态。

三个视角共享冻结 DINOv3，各自 resize 为 288×160，产生 180 个 token，共 540 个视觉 token；LiDAR 仍为 256 个 BEV token。MF 的空间 mRoPE 使用不同 view index，结合可学习位置编码区分视角；不是基于标定的显式 3D 投影。各视角共享 RGB 模态路由，LiDAR 使用独立路由。DiffusionDrive 轨迹头保持一致。

沿用 expanded 的全部 467 个训练/验证样本（129/338，4/2 日志），官方元数据按 token 对齐三个视角；缺图直接报错，不静默删样本。测试保持同一 200 token 和原始 ReCogDrive scorer/cache。训练 seed42、batch32、20 epoch / 100 step、AdamW 1e-4、wd1e-4、BF16、clip1；只用内部验证 ADE 选权重。冻结编码器缓存特征，不加入额外 flow 预训练。

由于视觉 token 数和可学习位置参数增加，随机初始化时其他模块的 RNG 消耗也会变化；本版是扩展试跑，不能据单种子结果严格归因于视角数量。原有 cross/MF 单视角模型的形状和 mRoPE 不变，原 checkpoint 保留。

服务器 `bimodal_smoke` 中执行：

```bash
GPU=0 bash run.sh turboad_plus_data.py
GPU=0 bash run.sh turboad_plus_pipeline.py
```

数据特征单独写入 `turboad/plus_samples.pt`，对齐清单 `plus_manifest.json`；权重目录 `turboad/plus_seed42`；状态 `plus_status.json`；日志 `turboad/logs/plus_*` 和 `score_plus*`。使用 GPU0 训练、GPU2 评测，GPU1 不使用。

## 训练实测

20 epoch / 100 step 完成，无非有限 loss/梯度；验证 ADE 从 8.837 降至 6.369 米，最佳 6.032 米（epoch 2），FDE 从 16.539 降至 10.923 米。固定噪声验证 loss 从 18.169 降至 16.108。缓存特征后训练 83.34 秒，峰值 allocated 显存 4.030 GiB（不含 CUDA 上下文与图像编码）。可训练参数 21,736,706。

逐样本核验前后 token 顺序、划分、LiDAR、status、target 完全一致。单样本固定噪声下分别将左/右图像特征置零，输出均变化，确认两视角进入计算路径；这不是效果消融。最佳权重严格加载成功。

## 相同协议评测结果

同一批 200 个 navtest 场景，原始图像重新编码、原始 PDM scorer、内部验证 ADE 最优权重（epoch2），200/200 成功。PDMS ×100：初始化 11.05 → 训练后 **43.88**。

| 指标 | TurboAD | Turbo-MFAD | TurboAD-Plus |
|---|---:|---:|---:|
| PDMS ×100 | 46.44 | **61.44** | 43.88 |
| 最佳内部验证 ADE (m) | 5.379 | 6.114 | 6.032 |
| 无责任碰撞 | .770 | .885 | .880 |
| 可行驶区域合规 | .835 | .915 | .890 |
| ego progress | .406 | .511 | .392 |
| TTC 合规 | .650 | .815 | .695 |
| 舒适度 | .815 | .740 | **.165** |

Plus 能正常优化，但本次未出现多视角增益，尤其舒适度明显下降。此表不能说明多视角普遍无效：训练仅4个日志、100更新步，额外视角未进行显式几何对齐，初始化也变化。当前不替换 v0.5 MF 基线，不用 navtest 调参或重选权重；若后续诊断，需要独立设计验证集轨迹平滑性检查及受控初始化对比。

代码/结果版本 `v0.6-turboad-plus-navsim-pilot`。逐场景 CSV、对齐 manifest、输入一致性/模型依赖核验、训练曲线和 resolved 配置均在 `evidence/turboad`，可用 `verify_turboad_evidence.py` 重查 token 集合与全场景有效性。
