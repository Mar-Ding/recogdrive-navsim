# 实验版本索引

最新部署：10% NAVSIM 的 TurboAD / Turbo-MFAD 已完成，原始 PDMS 分别为 0.70645 / 0.75241。TurboAD-Plus（GPU0）与辅助视角减半分辨率的 TurboAD-Plus-small（GPU2）同样已完成，200 场景有效，原始 PDMS 分别为 0.75558 / 0.75220，状态见各自 `plus_status.json` / `plus_small_status.json`。详细配置和状态见 [NAVSIM_DEMO10](bimodal_smoke/NAVSIM_DEMO10.md)。下方旧版记录保留原有显示尺度，请勿与原始 PDMS 混用。

各版本均为本地 Git 快照。checkpoint 保存在服务器，不纳入 Git；manifest、训练曲线数据、逐场景评测 CSV 和配置随对应代码提交保存。

| Tag | 提交 | 内容 |
|---|---|---|
| v0.1-front-camera-demo | b5ac91a | 官方 ReCogDrive 前视图三样本推理 |
| v0.2-aligned-sensors-demo | b9f5449 | 对齐的八摄像头与 LiDAR 数据子集 |
| v0.3-bimodal-fusion-smoke | c11d7c3 | RGB + LiDAR cross-attention / MF 训练 smoke |
| v0.4-turboad-navsim-pilot | ef61302 | TurboAD 训练及原生 NAVSIM 200 场景评测 |
| v0.5-turbo-mfad-navsim-pilot | 4108728 | Turbo-MFAD 同配置训练及与 TurboAD 对比 |

当前推荐实验基线为 `v0.5-turbo-mfad-navsim-pilot`。相同 129/338 样本划分、seed=42、batch=32、20 epoch：TurboAD / Turbo-MFAD 的 200 场景 PDMS（×100）分别为 46.44 / 61.44；最佳内部验证 ADE 分别为 5.38 / 6.11 米。两者评测均 200/200 成功。这是单种子、小样本子集结果，未进行完整 benchmark；MF 此版本没有额外 flow 预训练。

- [TurboAD 记录](bimodal_smoke/TURBOAD.md)
- [Turbo-MFAD 记录](bimodal_smoke/TURBO_MFAD.md)
- [逐场景和训练证据](bimodal_smoke/evidence/turboad)

服务器权重根目录：`/data/ruiqiwang/recogdrive_baseline/turboad`。

| 模型 | 目录 | 权重 |
|---|---|---|
| TurboAD 首轮 | pilot_seed42 | best_val_ade.pt / latest.pt |
| TurboAD 对比基线 | expanded_seed42 | best_val_ade.pt / latest.pt |
| Turbo-MFAD | mf_expanded_seed42 | best_val_ade.pt / latest.pt |

查看或建立独立复现分支：

```bash
git show v0.5-turbo-mfad-navsim-pilot
git switch -c experiment/my-run v0.5-turbo-mfad-navsim-pilot
```

后续实验应使用新的 run 目录与提交，保留这些基线权重和结果。远端仓库为 GitHub `Mar-Ding/recogdrive-navsim`（private），基线以 `main` 快照保存。

## v0.6-turboad-plus-navsim-pilot

三视角 RGB（F0/L0/R0）+ LiDAR、MF 融合；与 v0.5 相同训练划分与 200 场景原生评测。20 epoch、100 step、batch32。PDMS=43.88，200/200成功；本轮未优于单视角 MF，继续保留 v0.5 作为效果基线。权重 `turboad/plus_seed42/{best_val_ade,latest}.pt`，详情见 [Plus 记录](bimodal_smoke/TURBOAD_PLUS.md)。

## v0.7-navsim-demo10-deployment

归档 MF / Plus 从20扩展到100 epoch的已完成实验：200场景PDMS分别36.63 / 23.82，验证ADE改善但测试下降，保留旧权重。

新增约10%数据部署：官方前20/200传感器分片，共10,618场景（navtrain的10.28%），训练8,688 / 验证1,930，日志隔离。TurboAD与Turbo-MFAD分别单卡GPU0/2，参考官方常规IL的有效batch128、200epoch、AdamW和warmup/cosine。已通过显存实测与两轮训练smoke；此版本是下载、训练、评测自动流程的部署快照，尚不是新一轮完整训练结果。详情见 [数据与预算记录](bimodal_smoke/NAVSIM_DEMO10.md)。

## v0.9-navsim-visual-fivegroup

2026-11-02 实验室 RTX 3090 上完成的纯视觉五组对照。协议统一为 demo10 清单（8,688 训练 / 1,930 验证）、固定 200 场景 navtest、seed 42、200 epoch、有效 batch 128、AdamW lr1e-4；每组均按最低验证 ADE 选权重，评测 200/200 有效、0 失败。

| 方法 | PDMS（百分制） | 验证 ADE / FDE（m） | 训练+验证 |
|---|---:|---|---:|
| DINOv3-DD | 81.11 | 0.7543 / 1.8833 | 20.7 min |
| TurboAD Cross | 79.10 | 0.7561 / 1.8951 | 42.8 min |
| TurboMFAD + flow | 83.62 | 0.7136 / 1.7423 | 92.0 min |
| TurboMFAD no-flow | 85.80 | 0.7289 / 1.7863 | 58.6 min |
| ReCogDrive-TD front | 86.90 | 0.7107 / 1.7467 | 227.9 min |

指标原表 `bimodal_smoke/assets/navsim/metrics.csv`，逐场景分项 `output/charts/NAVSIM_detailed_metrics.csv`，图表在 `bimodal_smoke/assets/navsim/`（PNG + SVG + 曲线原始数据）。原生 ReCogDrive 按要求停止试训（外推 43.56 小时），不计入成绩；三视角 ReCogDrive-TD 只做过探针，已取消完整训练。ReCogDrive-TD 的视觉输入、场景查询层数与 anchors 与其他四组不同，差异不能全部归因于融合模块。部署与复现步骤见 [GITHUB_DEPLOYMENT](bimodal_smoke/GITHUB_DEPLOYMENT.md)。
