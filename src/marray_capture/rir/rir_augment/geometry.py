"""阵列几何: 从 YAML 构建麦克风坐标, 计算两两距离。

坐标单位为米。可以直接给出 ``coords``, 或用 ``generator`` 生成。
默认: 3 麦, 边长 1cm 等边三角形, 位于 xy 平面, 质心在原点。
"""
from __future__ import annotations

from typing import Any

import numpy as np


def build_array(cfg: dict[str, Any]) -> np.ndarray:
    """返回麦克风坐标数组, 形状 (M, 3), 单位米。"""
    coords = cfg.get("coords")
    if coords:
        pts = np.asarray(coords, dtype=float)
    else:
        gen = cfg["generator"]
        gtype = gen["type"]
        if gtype == "equilateral_triangle":
            a = float(gen["edge"])
            # 外接圆半径 R = a/sqrt(3), 顶点距 = a
            r = a / np.sqrt(3.0)
            angs = np.deg2rad([90.0, 210.0, 330.0])
            pts = np.stack([r * np.cos(angs), r * np.sin(angs), np.zeros(3)], axis=1)
        elif gtype == "linear":
            n = int(gen["num"])
            d = float(gen["spacing"])
            xs = (np.arange(n) - (n - 1) / 2.0) * d
            pts = np.stack([xs, np.zeros(n), np.zeros(n)], axis=1)
        else:
            raise ValueError(f"未知的阵列生成器: {gtype}")
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(f"坐标形状应为 (M,2) 或 (M,3), 实际 {pts.shape}")
    if pts.shape[1] == 2:
        pts = np.concatenate([pts, np.zeros((len(pts), 1))], axis=1)
    # 平移到质心
    pts = pts - pts.mean(axis=0, keepdims=True)
    return pts


def pairwise_distances(pts: np.ndarray) -> np.ndarray:
    """两两欧氏距离矩阵, 形状 (M, M)。"""
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))
