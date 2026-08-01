"""从一次录音里提取 IR。

流程:
    录音 (T, C) --与逆滤波器全长卷积--> 反卷积域 (T+N-1, C)
                 --按已知激励布局定位每次扫频的直达峰--> 逐次 IR
                 --一致性通过则对齐后平均--> 最终 IR

为什么整段一次性反卷积: 录音里两次扫频相隔 >6 秒, 反卷积输出里两条 IR 互不重叠,
一次 FFT 卷积就能同时拿到, 也避免了「先切段再反卷积」需要预先知道延迟的鸡生蛋问题。

蓝牙输出的延迟可达数百毫秒且每次不同, 因此第一次扫频的峰位在
[start0 + ir_offset, start0 + ir_offset + max_latency] 窗口内全局搜索;
后续扫频只在上次结果附近小窗搜索, 两者之差即时钟漂移。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy.signal import resample, resample_poly

from ..audio.sweep import Sweep, deconvolve, find_direct_index


@dataclass
class TakeIR:
    """一个位置的提取结果。"""

    irs: list[np.ndarray] = field(default_factory=list)   # 每次扫频的 IR, (L, C)
    ir_avg: np.ndarray | None = None                      # 对齐平均后的 IR, (L, C)
    direct_indices: list[int] = field(default_factory=list)  # 反卷积域中的绝对峰位
    latency_samples: int = 0
    drift_ppm: float = 0.0
    pre_samples: int = 0                                  # IR 里直达峰之前保留的样本数
    fs: int = 48000
    n_averaged: int = 1

    def use_single(self) -> None:
        """一致性不过时回退到第一次扫频的 IR, 放弃平均。"""
        if self.irs:
            self.ir_avg = self.irs[0]
            self.n_averaged = 1


def deconvolve_take(rec: np.ndarray, sweep: Sweep) -> np.ndarray:
    """整段录音反卷积。rec (T, C) → (T+N-1, C)。"""
    return deconvolve(np.asarray(rec, dtype=float), sweep)


def locate_peaks(
    deconv: np.ndarray,
    sweep: Sweep,
    starts: list[int],
    max_latency_s: float,
    energy_channels: list[int] | None = None,
    known_latency: int | None = None,
) -> tuple[list[int], int, float]:
    """定位每次扫频的直达峰。返回 (峰位列表, 延迟样本数, 漂移 ppm)。"""
    fs = sweep.fs
    sub = deconv if energy_channels is None else deconv[:, energy_channels]

    # 第一次: 全局窗搜索 (或用已标定的延迟收窄)
    base = starts[0] + sweep.ir_offset
    if known_latency is not None:
        first = find_direct_index(sub, expected=base + known_latency,
                                  search_radius=int(0.05 * fs))
    else:
        span = int(round(max_latency_s * fs))
        lo, hi = base, min(len(sub), base + span)
        seg = (sub ** 2).sum(axis=1)[lo:hi]
        first = int(lo + np.argmax(seg)) if len(seg) else base

    latency = first - base
    peaks = [first]
    radius = int(0.03 * fs)                    # 后续扫频只在 ±30ms 内找
    for k in range(1, len(starts)):
        exp = starts[k] + sweep.ir_offset + latency
        peaks.append(find_direct_index(sub, expected=exp, search_radius=radius))

    # 漂移: 峰位相对名义间隔的偏移 / 间隔时长
    drift_ppm = 0.0
    if len(starts) >= 2:
        nominal = starts[-1] - starts[0]
        actual = peaks[-1] - peaks[0]
        if nominal > 0:
            drift_ppm = (actual - nominal) / nominal * 1e6
    return peaks, latency, float(drift_ppm)


def _cut(deconv: np.ndarray, peak: int, pre: int, length: int) -> np.ndarray:
    """从反卷积输出里裁一条 IR, 越界补零。"""
    start = peak - pre
    out = np.zeros((length, deconv.shape[1]), dtype=float)
    src_lo = max(0, start)
    src_hi = min(len(deconv), start + length)
    if src_hi > src_lo:
        out[src_lo - start: src_hi - start] = deconv[src_lo:src_hi]
    return out


def extract_take(
    deconv: np.ndarray,
    sweep: Sweep,
    starts: list[int],
    ir_pre_ms: float,
    ir_len_ms: float,
    max_latency_s: float,
    energy_channels: list[int] | None = None,
    known_latency: int | None = None,
    average: bool = True,
) -> TakeIR:
    """定位 + 裁剪 + (可选) 平均。

    平均是否安全由上层根据 QC 的一致性指标决定: 两次扫频若存在亚样本错位
    (时钟漂移), 直接平均会梳状衰减高频。一致性不过时应回退到单次 IR,
    用 ``take.use_single()``。
    """
    fs = sweep.fs
    peaks, latency, drift = locate_peaks(
        deconv, sweep, starts, max_latency_s, energy_channels, known_latency
    )
    pre = int(round(ir_pre_ms * 1e-3 * fs))
    length = int(round(ir_len_ms * 1e-3 * fs))
    irs = [_cut(deconv, p, pre, length) for p in peaks]

    ir_avg = irs[0]
    n_avg = 1
    if average and len(irs) > 1:
        # 每条都已按自己的直达峰对齐到相同的 pre 偏移, 直接平均即可
        ir_avg = np.stack(irs, axis=0).mean(axis=0)
        n_avg = len(irs)

    return TakeIR(
        irs=irs, ir_avg=ir_avg, direct_indices=peaks, latency_samples=latency,
        drift_ppm=drift, pre_samples=pre, fs=fs, n_averaged=n_avg,
    )


# ---------------------------------------------------------------------- 重采样
def resample_ir(ir: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """(L, C) IR 重采样。整数比走 resample_poly (更干净), 否则退回 FFT 重采样。"""
    if fs_in == fs_out:
        return np.asarray(ir, dtype=float)
    frac = Fraction(int(fs_out), int(fs_in)).limit_denominator(1000)
    return resample_poly(np.asarray(ir, dtype=float), frac.numerator, frac.denominator, axis=0)


def correct_drift(rec: np.ndarray, ppm: float) -> np.ndarray:
    """按估计的 ppm 重采样录音, 补偿两台设备的时钟漂移。

    只在 |ppm| 明显但仍可接受时使用; 漂移太大说明蓝牙链路在重同步, 应该重录。
    """
    if abs(ppm) < 1e-9:
        return rec
    n_in = len(rec)
    n_out = int(round(n_in / (1.0 + ppm * 1e-6)))
    if n_out <= 1 or abs(n_out - n_in) > n_in * 0.01:
        return rec
    return resample(np.asarray(rec, dtype=float), n_out, axis=0)
