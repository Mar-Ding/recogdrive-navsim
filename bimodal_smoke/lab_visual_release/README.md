# 实验室纯视觉实验运行源码快照

2026-11-02 从已完成训练评测的实验室服务器拉取，Python 文件保持原样。

部署、数据、依赖、训练和评测命令见 [部署说明](../GITHUB_DEPLOYMENT.md)。

将 Python 文件部署到目标 `visual_vla/multiview/`，将 `run.sh` 放至 `visual_vla/`。源码保留实验室绝对路径；新环境需要统一调整路径。`sitecustomize.py` 引用实验室旧环境，只适合原环境，不应原样用于其他机器。

不包含上游依赖、数据、特征缓存或编码器。最佳权重另存本地 `output/lab_archive_20261102/outputs/`。不要在保留结果的目录直接重启训练覆盖记录。
