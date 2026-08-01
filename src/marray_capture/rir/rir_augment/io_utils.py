"""多通道 RIR 读写。约定内存布局为 (M, T) = (通道, 样本)。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_rir(path: str | Path) -> tuple[np.ndarray, int]:
    """读取多通道 wav, 返回 ((M, T), fs)。"""
    data, fs = sf.read(str(path), always_2d=True)  # (T, M)
    return data.T.astype(float), int(fs)


def save_rir(path: str | Path, rir: np.ndarray, fs: int, peak_norm: bool = True) -> None:
    """保存 (M, T) RIR 为 wav。默认按峰值归一到 0.98 防止 float->int 削顶。"""
    x = np.atleast_2d(rir).astype(float)
    if peak_norm:
        peak = np.max(np.abs(x))
        if peak > 0:
            x = x / peak * 0.98
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x.T, fs, subtype="FLOAT")
