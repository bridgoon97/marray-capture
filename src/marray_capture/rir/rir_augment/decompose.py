"""RIR 分解与 T60 估计。"""
from __future__ import annotations

import numpy as np


def find_direct(rir: np.ndarray) -> int:
    """跨通道能量峰值定位直达路径的样本索引。rir 形状 (M, T)。"""
    energy = (rir ** 2).sum(axis=0)
    return int(np.argmax(energy))


def split_early_late(rir: np.ndarray, fs: float, split_ms: float) -> tuple[int, int]:
    """返回 (direct_idx, split_idx)。early = [:split_idx], late = [split_idx:]。"""
    direct = find_direct(rir)
    split = direct + int(round(split_ms * 1e-3 * fs))
    split = min(split, rir.shape[1])
    return direct, split


def schroeder_t60(h: np.ndarray, fs: float, lo_db: float = -5.0, hi_db: float = -35.0) -> float:
    """单通道 Schroeder 反向积分 + 线性拟合估计 T60 (秒)。估计失败返回 nan。"""
    e = h.astype(float) ** 2
    edc = np.cumsum(e[::-1])[::-1]
    edc = edc / (edc[0] + 1e-20)
    db = 10.0 * np.log10(edc + 1e-12)
    below_lo = np.where(db <= lo_db)[0]
    below_hi = np.where(db <= hi_db)[0]
    if len(below_lo) == 0 or len(below_hi) == 0:
        return float("nan")
    i1, i2 = below_lo[0], below_hi[0]
    if i2 <= i1 + 1:
        return float("nan")
    t = np.arange(i1, i2) / fs
    slope = np.polyfit(t, db[i1:i2], 1)[0]  # dB/s
    if slope >= 0:
        return float("nan")
    return float(-60.0 / slope)
