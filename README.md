# AD / ReCogDrive 基础实验

基础版本：`v0.1-front-camera-demo`。

已验证官方 ReCogDrive 2B IL 在真实 NAVSIM 前视图上的三样本推理。运行方式、结果及限制见 [demo 说明](recogdrive_demo/README.md)。

新增 RGB + LiDAR 双模态实验：TurboVLA cross-attention 与 MF 融合均完成 50 step 训练及原始传感器推理，结构、复现入口和结果见 [双模态 smoke 说明](bimodal_smoke/README.md)。当前仅验证三样本链路，不作为性能评测。

cross-attention 版本后续统一命名 **TurboAD**，正式训练子集和 ReCogDrive PDM 评测适配见 [TurboAD 说明](bimodal_smoke/TURBOAD.md)。

上游：https://github.com/xiaomi-research/recogdrive ，固定提交 `5831ee83c7a6a40bf6919715b96b0e9ba2b48335`。本仓库归档实验入口及结果，上游代码保留在服务器独立目录。

当前实验版本：**v0.5-turbo-mfad-navsim-pilot**。Turbo-MFAD 已完成相同训练配置及 200 场景评测，PDMS 为 61.44（TurboAD 为 46.44）。这仍是小样本单种子试跑，详见 [版本索引](VERSIONS.md) 和 [MF 实验记录](bimodal_smoke/TURBO_MFAD.md)。

新增 [TurboAD-Plus 三视角试验](bimodal_smoke/TURBOAD_PLUS.md)：已完成相同协议训练和 200 场景评测，PDMS=43.88；本轮没有多视角收益，保留单视角 MF 效果基线。
