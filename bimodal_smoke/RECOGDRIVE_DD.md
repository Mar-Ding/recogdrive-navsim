# ReCogDrive VLM + DiffusionDrive 混合版试验

按用户要求停止原生 ReCogDrive 长训练，尝试使用已有 DiffusionDrive action 生成部分。此版为独立混合架构，不作为原生 ReCogDrive 成绩。

## 结构与保留项

前视图、原生提示词/导航/历史 → 冻结官方 ReCogDrive InternVL 2B → 完整2800×1536特征 → 可训练1536→256投影 → 256个学习query以cross-attention读取全部token → 16×16潜在网格及31个场景query → 已有DiffusionDrive TrajectoryHead → 未来4秒8个SE(2)航点。

状态分支使用原生8维status与4×3历史轨迹，共20维。没有引入LiDAR；16×16网格是学习的潜在表征，不是观测得到的物理BEV。原生VLM和缓存内容不变；新增接口适配器随机初始化。复用 `vendor/diffusion_decoder.py` 的原始轨迹头、20个锚点、训练损失、两层decoder和两步推理，不修改该文件。

## 控制变量与试训

同一manifest：8688训练/1930验证，seed42，BF16，有效batch128。AdamW lr1e-4/wd1e-4/betas(0.9,0.95)，clip1。先3轮全规模pilot，使用200轮协议的前3轮warmup，不将学习率压缩到3轮结束。按验证ADE保存最佳权重。此pilot不代表200轮模型表现，不与已有200轮PDMS直接判断优劣。

原生缓存约85GiB，从磁盘反复读取明显影响整轮速度。本版先预载CPU内存，要求可用内存大于缓存大小另加32GiB，加载时保留至少24GiB；加载耗时单列。因加载方式和规划头同时改变，整轮加速不能单独归因为换头。纯GPU计算测速单独对照。

## 同机GPU2计算实测

| batch128 | 前向/反向/AdamW | 验证推理 | 峰值显存 |
|---|---:|---:|---:|
| 原生ReCogDrive头 | 0.412秒 | 1.733秒 | 16.52GiB |
| VLM适配器+DiffusionDrive头 | 0.160秒 | 0.072秒 | 4.40GiB |

真实完整特征形状；使用一个缓存样本重复成batch，前向/反向测速预热后取两次均值，推理单次。因此这是计算microbenchmark，不是整轮吞吐。混合版输出128×8×3且有限，训练loss/梯度均有限。读取128个真实缓存文件约1.10GB，实测4.11秒；不能以此单次读速精确解释原实验全部时间。

原生完整轮实测平均784.07秒，200轮外推 **43.56小时**，已按用户要求停止，权重与记录保留。该外推包含训练及逐轮验证，不含VLM编码与最终PDM评分，不是实际跑满时间。详见 `NATIVE_RECOGDRIVE.md` 和 `evidence/recogdrive_stopped_timing.json`。

## 运行与证据

### 已完成3轮试训

缓存预载697.43秒（11.62分钟）；3轮训练及验证共181.17秒。每轮60.61/57.34/59.66秒，平均59.20秒，外推200轮 **3.29小时**；加一次预载约3.48小时，不含最终PDM评测。这是外推预算，未跑满200轮。

训练loss：16.7322 → 16.3030 → 15.9233。验证ADE：初始化6.6929m，epoch1为7.1438m，epoch2为4.1834m，epoch3为2.6064m；验证FDE最终5.3564m。初期ADE并非单调下降，末轮已优于初始化，支持短程能够优化，不代表最终PDMS或优于其他模型。峰值4.403GiB，micro/effective batch均128。试训已正常结束，未自动启动200轮或最终PDM评测。

第3轮约37.76秒为内存组批及传输，前向/反向11.09秒，验证10.42秒。原生43.56小时与混合版3.29小时同时存在缓存加载方式变化，不能作为仅替换diffusion的因果加速比。

本地结果证据 `evidence/recogdrive_dd_pilot_summary.json`。

服务器目录 `/data/ruiqiwang/recogdrive_baseline/demo10/recogdrive_dd/`，benchmark.json/io_benchmark.json为测速；status.json为阶段；loading.json为加载耗时；history.json/summary.json为试训结果。日志 `demo10/logs/recogdrive_dd_pilot.log`。GPU2运行，GPU1不使用。

命令 `env/bin/python -u bimodal_smoke/recogdrive_dd.py benchmark` 与 `env/bin/python -u bimodal_smoke/recogdrive_dd.py pilot --epochs 3`，外部设置CUDA_VISIBLE_DEVICES=2及现有HF_HOME/TMPDIR/PYTHONPATH。

## 200轮训练与自动评测

用户已授权将混合版继续训练至200轮，并自动接续PDMS。运行 `env/bin/python -u bimodal_smoke/recogdrive_dd.py pipeline`，持有排他文件锁防止重复启动。训练子进程从latest.pt第3轮/step204恢复模型、AdamW状态和历史；除目标epochs从3扩至200外严格核对配置。第4轮开始按原200轮cosine计划降至最低1e-6，有效batch128不变。旧3轮summary/config/history/loading首次自动备份为pilot_*.json，已有本地pilot_summary证据不覆盖。

训练结束并成功退出后释放CPU特征缓存，自动启动 `recogdrive_dd_score.py`。仅选择内部验证ADE最低的best_val_ade.pt，严格验证模型参数、manifest和选权轮次，确认完成200轮后才允许正式评分。在线使用同一原生冻结VLM提取特征；轨迹头BF16推理，seed999；原固定200个navtest token、同一metric cache、原生pdm_score/simulator/scorer。输出scores.csv与evaluation.json（原始0–1 PDMS），保存checkpoint与token清单哈希。任何场景失败立即报告错误，不剔除失败场景后报告200场景均值。

正式启动前先以第3轮权重做单场景端到端评测检查，结果独立保存evaluation_smoke.json，不混入最终成绩。管线状态pipeline_status.json，训练状态status.json，评测进度evaluation_progress.json。日志为recogdrive_dd_train.log、recogdrive_dd_score.log。GPU2执行全链路，GPU1不使用。按已测3轮，剩余197轮约3.24小时，加重新预载约12分钟及最终评测；这是外推，实际以日志为准。
