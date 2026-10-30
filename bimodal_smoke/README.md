# RGB + LiDAR 训练 smoke

2026-10-30 已完成 A/B 两组各 50 step 的真实 NAVSIM 数据训练、严格保存重载，以及独立进程从原始 RGB + PCD 开始的推理。

这是在已有 ReCogDrive 数据/环境旁新增的传感器融合实验，**不是完整 ReCogDrive 或 DiffusionDrive 的训练复现**。原有官方 ReCogDrive demo 和权重保持原样。所有本地文件位于 `F:\code\AD`；服务器实验目录为 `/data/ruiqiwang/recogdrive_baseline/bimodal_smoke`。

## 本轮结构

```text
前视 RGB 160×288 → 冻结 DINOv3 ViT-B → 180×768 → 投影 256
                                                           ↘
                                                A / B 传感器融合（6 层）
                                                           ↗
LiDAR PCD → [-32,32]m BEV 256×256 → 可训练 CNN → 16×16×256
                                      ↓ 融合后仍保留 BEV 网格
                               共用 BEV adapter / 场景 query decoder
                                  ↑ 自车状态 8 维
                                      ↓ 自车 + 30 个 agent queries
官方 20×8×2 轨迹锚点 → 截断噪声 → 官方 DiffusionDrive TrajectoryHead
                                      ↓
                                 选出一条 8×3 轨迹
```

- A (`cross`)：直接调用 TurboVLA 的 `BiAttentionBlock`，RGB↔LiDAR 双向 cross-attention，两个分支各有 FFN。沿用 `gamma=1e-4` 初始化；这会使最初的 RGB 梯度较小。
- B (`mf`)：直接调用上游 `MFBlock`，共享 QKVO、各模态 SwiGLU、RMSNorm、mRoPE；按 MFVLA E005 的方式使用 torch 2.3 SDPA。RGB 局部 ID 0、LiDAR 局部 ID 3，避免占用 text=1 或旧 action=2；两路均为干净条件，双向可见。mRoPE 使用各自网格的 `(sensor,row,column)`。
- 状态为 command(4)、velocity(2)、acceleration(2)，不作为第三个融合模态。
- 两组使用相同的编码器结构、投影、BEV adapter、场景 decoder、轨迹 decoder 和优化参数；审计确认 177 个共用初始 tensor 完全一致。但融合参数量不同，当前不是参数量匹配的效果对照。
- 轨迹 decoder 的类体从官方源码原样提取，AST 审计相等；官方锚点来自 release。保留官方训练 timestep `[0,50)`、`prediction_type=sample`、两层 refinement、两次推理扩散更新和分类+L1 损失。

## 训练范围与限制

- 使用现有 3 个已配对的 **test 调试样本**，每组 batch=1、50 step、AdamW lr=1e-4、weight_decay=1e-4、梯度裁剪 1、BF16 autocast。
- DINOv3 使用既有预训练权重且冻结；首次从真实 RGB 计算并缓存 patch features。训练更新投影、随机初始化的 LiDAR CNN、融合模块和解码端。轨迹头没有加载官方 DiffusionDrive 预训练 checkpoint。
- 独立 `demo.py` 会重新读取 RGB/PCD、重新运行 DINOv3、严格加载训练 checkpoint，再输出轨迹，不依赖缓存视觉特征进行预测。
- 现有三帧未来 GT 均接近静止；当前预测尚不平滑。这里只证明训练和推理链路通畅，不能证明驾驶质量、跨场景泛化或 MF 收益。
- 未运行 PDMS、闭环评测、完整训练、辅助 BEV 语义/检测监督。30 个 agent queries 只接受轨迹损失的间接梯度，不能解释为已校准的车辆检测结果。
- B 是复用真实 MF 核心的双传感器适配，尚未做辅助 flow 预训练，也尚未移植 E005 完整 codec/task 注册生命周期。以后增加模态时再补扩展契约；不把本轮称为完整 Multimodal Flow recipe。
- 相比官方 DiffusionDrive，本轮改用前视小图与 DINOv3、单尺度小型 LiDAR CNN 和自定义融合，保留解码机制；不声称官方性能。

## 验收结果

| 项目 | A：cross / GPU0 | B：MF / GPU2 |
|---|---:|---:|
| 可训练参数 | 16,524,034 | 21,644,546 |
| 固定噪声训练目标：前→后 | 25.6467 → 21.0702 | 25.6469 → 21.3659 |
| 50 step 训练循环耗时 | 5.09 秒 | 5.66 秒 |
| 缓存特征训练峰值 allocated | 416 MiB | 545 MiB |
| 完整编码+规划推理峰值 allocated | 441 MiB | 464 MiB |
| RGB / LiDAR 输入非零梯度 | 通过 / 通过 | 通过 / 通过 |
| 融合参数更新 | 通过 | 通过 |
| 固定噪声消融任一路会改变输出 | 通过 | 通过 |
| 严格重载最大输出差 | 0 | 0 |
| 原始 RGB + PCD 独立进程推理 | 3/3 通过 | 3/3 通过 |

耗时不含安装、加载、数据准备和审计；显存为 PyTorch allocated，非 nvidia-smi 进程总量。完整推理计时包含 GPU 编码和规划，排除磁盘/CPU 预处理，首帧冷启动。这里的耗时和 loss 均不能用来比较最终方法性能。

证据：`evidence/{cross,mf}/report.json`、`raw_demo.json`、`training.json`、`predictions.json`、`trajectories.png`。公共初始化及源码核对见 `evidence/audit.json`，实际依赖见 `evidence/environment.json`。

## 复现

本地从工作区运行（凭据仅从被 git 忽略的 `login.md` 读取，部署不会上传该文件）：

```powershell
python bimodal_smoke/extract_decoder.py
python bimodal_smoke/deploy.py
```

服务器入口复用现有 TurboVLA Python，追加既有 ReCogDrive 的缺失依赖目录，不安装或升级旧环境中的任何包。依赖包括 torch 2.3.1+cu121、transformers 4.56.2、diffusers 0.32.2。

```bash
cd /data/ruiqiwang/recogdrive_baseline/bimodal_smoke
GPU=0 bash run.sh prepare.py
GPU=0 bash run.sh train.py --variant cross --steps 50
GPU=2 bash run.sh train.py --variant mf --steps 50
GPU=0 bash run.sh demo.py --variant cross
GPU=2 bash run.sh demo.py --variant mf
bash run.sh audit.py
```

入口只接受 GPU=0 或 GPU=2，且所有临时文件/缓存落在本实验目录。再次训练会覆盖对应 `outputs/<variant>/latest.pt`。checkpoint 含模型、优化器、step、配置与源码哈希；本轮验证的是严格权重重载，不是完整随机状态的精确训练续跑。

本地收集轻量结果：`python bimodal_smoke/collect.py`。checkpoint 留在服务器 `outputs/cross/latest.pt`、`outputs/mf/latest.pt`，不写入 git。

## 源码出处

- [DiffusionDrive 轨迹 decoder](https://github.com/hustvl/DiffusionDrive/blob/main/navsim/agents/diffusiondrive/transfuser_model_v2.py)、[BEV attention](https://github.com/hustvl/DiffusionDrive/blob/main/navsim/agents/diffusiondrive/modules/blocks.py)、[轨迹损失](https://github.com/hustvl/DiffusionDrive/blob/main/navsim/agents/diffusiondrive/modules/multimodal_loss.py)。原文件和 MIT 许可保存在 `reference/DiffusionDrive`，SHA256 见 `vendor/source_hashes.json`。
- [官方轨迹锚点](https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/kmeans_navsim_traj_20.npy)。
- TurboVLA：服务器既有 `upstream/TurboVLA-native/turbovla/models/components/fusion.py`。
- MFVLA E005：既有 `vla/extension_model.py`、`vla/model.py` 和 `experiments/E005_EXTENSION.md`；核心为既有 `upstream/Multimodal-Flow-main/src/mf/modeling/block.py`。实际源文件哈希保存在 `evidence/audit.json`。
