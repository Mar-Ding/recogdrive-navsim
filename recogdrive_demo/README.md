# ReCogDrive 真实 NAVSIM 推理 demo

服务器目录：`/data/ruiqiwang/recogdrive_baseline`

复现命令：
```bash
bash /data/ruiqiwang/recogdrive_baseline/run_demo.sh
```

官方仓库提交：5831ee83c7a6a40bf6919715b96b0e9ba2b48335。
模型：ReCogDrive-VLM-2B + ReCogDrive 2B IL planner 官方权重。
使用 GPU0；没有使用 GPU1。基础环境通过独立 overlay venv 复用，未删除旧数据或任何 checkpoint。

从 NAVSIM/OpenScene test 元数据及真实 CAM_F0 图像选取 3 个相邻样本；仅加载当前前视图，不加载 LiDAR。模型仍包含官方 ego 状态输入。每例输出未来 4 秒、间隔 0.5 秒的 8 个 (x,y,heading) 点，已验证形状和有限值。

输出：服务器 demo/results.json 和 demo/demo.png；日志 logs/demo.log。本目录保存一份脚本及结果。

样本 ADE（m）：1.6143, 1.5639, 2.0399。
这些相邻样本的真实车辆基本静止，而预测有前进趋势。这里仅验证完整推理链路；样本规模和场景不支持模型效果结论，ADE 也不是 PDMS。

完整 NAVSIM PDMS 评测尚未完成：缺 nuPlan 地图，官方 S3 地图下载连接失败。下一步取得地图并构建 metric cache，再进行小规模 PDMS 评测。demo 推理本身不依赖地图。

## 基础版本 v0.1-front-camera-demo

本次归档范围：当前前视图 CAM_F0 + 官方自车状态输入，官方 2B IL 权重，3 个真实样本的推理及可视化。未新增图像缩小处理，未做多模态 Flow 改造。后续图像分辨率或模型修改从此版本继续。

工作区根目录为本地 Git 仓库；不包含 login.md、模型权重、数据集、虚拟环境或缓存。该版本尚未推送到远程托管平台。

脚本说明：run_demo.sh / run_demo.py 是已复跑验证的入口。bootstrap.sh、install_navsim.sh、download_data.sh、get_camera_subset.py 是部署过程脚本存档；它们包含本服务器绝对路径，不代表已验证的全新机器一键安装程序。environment-freeze.txt 记录最终环境（包含继承的基础环境依赖）。部署过程中另外补装了 decord==0.6.0，仓库源码固定在上述提交。

本地结果快照为首次成功推理的原始输出，seconds 含 GPU 异步执行等因素，不能作为严格性能基准。

## 对齐的全传感器子集

已补齐原有 16 张前视 RGB 对应的各相机和 LiDAR，并扩展原 3 个 demo 样本的 4 帧历史上下文，总计 25 个元数据帧。每帧包含 8 路相机和 1 个 MergedPointCloud；共 200 张图像、25 份点云，72,914,625 字节（约 69.54 MiB）。

服务器数据根目录：`/data/ruiqiwang/recogdrive_baseline/data/sensor_blobs/test`。
`multisensor_manifest.json` 保存逐帧 token、时间戳、文件路径、SHA256、相机内参及相机到 LiDAR 外参；原始完整标定及状态仍在对应 metadata pickle 中。关联依据是官方元数据；没有额外验证各相机实际曝光时间差。

下载：在服务器项目目录执行 `env/bin/python complete_sensors.py`，流式读取 camera_0 / lidar_0 官方分片并只保存目标文件，可跳过已有文件。该脚本会以现有本地前视图及 demo 历史上下文确定范围。
验证：`PYTHONPATH=$PWD/repo MPLCONFIGDIR=$PWD/tmp/matplotlib env/bin/python verify_sensors.py`。
所有相机已通过图片解码检查，所有点云通过 NAVSIM 使用的二进制读取接口与有限坐标检查；3 个 demo 样本均通过全传感器四帧历史加载检查。结果见 `multisensor_verification.json`，预览见 `multisensor_preview.png`。

本次只补齐输入数据；基础模型推理仍保持前视图配置，尚未增加多相机或 LiDAR 编码/融合网络。
