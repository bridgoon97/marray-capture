"""独立混响随机增强工具: 把一批裸多通道 IR wav 每条随机加 N 个混响尾。

跟主程序后处理页 (gui/post_page.py) 共用同一套增强核心 (augment_rir / 几何 / 相干性),
但入口是裸 wav 目录, 不依赖采集会话 / manifest / 质检门槛。用 ``uv run rir-augment-ui`` 启动。
"""
