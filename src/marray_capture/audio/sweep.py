"""指数扫频 (ESS / Farina) 的生成、逆滤波器与反卷积。

用 ESS 而非 MLS/白噪的理由: 谐波失真在反卷积输出里落到线性 IR 的**负时间**上,
直接切掉即可。这一条在蓝牙音箱这种链路上尤其重要 —— 功放/单元的非线性不会污染 IR。

注意: 有损蓝牙编码 (SBC/AAC) 引入的量化噪声**不是**谐波, 不会被推到负时间,
它表现为 IR 上的宽带噪底。对策是多次扫频求平均 + 靠 QC 的一致性指标兜底。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


@dataclass
class Sweep:
    """一条 ESS 及其配套逆滤波器。"""

    signal: np.ndarray      # (N,) 扫频本体
    inverse: np.ndarray     # (N,) 逆滤波器, 已归一到 conv(signal, inverse) 峰值为 1
    fs: int
    f_start: float
    f_end: float
    duration_s: float

    @property
    def n(self) -> int:
        return len(self.signal)

    @property
    def ir_offset(self) -> int:
        """零延迟时线性 IR 在 fftconvolve(rec, inverse, 'full') 中的索引。"""
        return self.n - 1


def _raised_cosine_fade(x: np.ndarray, fs: int, fade_in_ms: float, fade_out_ms: float) -> np.ndarray:
    y = x.copy()
    ni = int(round(fade_in_ms * 1e-3 * fs))
    no = int(round(fade_out_ms * 1e-3 * fs))
    ni = min(ni, len(y) // 2)
    no = min(no, len(y) // 2)
    if ni > 0:
        y[:ni] *= 0.5 * (1.0 - np.cos(np.pi * np.arange(ni) / ni))
    if no > 0:
        y[-no:] *= 0.5 * (1.0 + np.cos(np.pi * np.arange(no) / no))
    return y


def generate_ess(
    fs: int,
    f_start: float = 40.0,
    f_end: float = 20000.0,
    duration_s: float = 5.0,
    fade_in_ms: float = 15.0,
    fade_out_ms: float = 40.0,
) -> Sweep:
    """生成指数扫频与逆滤波器。

    x(t) = sin(K * (exp(t/L) - 1)),  L = T / ln(w2/w1),  K = w1 * L
    逆滤波器 = 时间反转的 x 再乘 exp(-t/L) 补偿 -3dB/oct 的粉噪谱。
    """
    f_start = float(max(f_start, 1.0))
    f_end = float(min(f_end, fs / 2.0 * 0.98))
    if f_end <= f_start:
        raise ValueError(f"扫频上限 {f_end} 必须大于下限 {f_start}")

    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    w1, w2 = 2 * np.pi * f_start, 2 * np.pi * f_end
    ratio = np.log(w2 / w1)
    L = duration_s / ratio
    K = w1 * L

    x = np.sin(K * (np.exp(t / L) - 1.0))
    x = _raised_cosine_fade(x, fs, fade_in_ms, fade_out_ms)

    # 逆滤波器 = 时间反转 + 随时间衰减的包络。
    # 反转后 t 处的瞬时频率是 f2·exp(-t/L); ESS 的谱是 -3dB/oct (|X| ∝ 1/√f),
    # 而 1/X 要求 |Inv| ∝ √f, 即在反转信号上乘一个正比于瞬时频率的包络 exp(-t/L)。
    # 注意包络必须**在反转之后**施加 —— 反转前乘会把频率补偿方向做反。
    inv = x[::-1] * np.exp(-t / L)

    # 数值归一: 让 conv(x, inv) 的峰值恰为 1
    peak = float(np.max(np.abs(fftconvolve(x, inv, mode="full"))))
    if peak > 0:
        inv /= peak

    return Sweep(signal=x, inverse=inv, fs=fs, f_start=f_start, f_end=f_end, duration_s=duration_s)


def deconvolve(rec: np.ndarray, sweep: Sweep) -> np.ndarray:
    """把录音与逆滤波器卷积得到 IR。

    rec: (T,) 或 (T, C)。返回 (T+N-1,) 或 (T+N-1, C)。
    线性 IR 的直达峰在 sweep.ir_offset + 系统延迟 处; 谐波失真在其**之前**。
    """
    rec = np.asarray(rec, dtype=float)
    if rec.ndim == 1:
        return fftconvolve(rec, sweep.inverse, mode="full")
    out = [fftconvolve(rec[:, c], sweep.inverse, mode="full") for c in range(rec.shape[1])]
    return np.stack(out, axis=1)


def find_direct_index(
    ir: np.ndarray,
    expected: int | None = None,
    search_radius: int | None = None,
) -> int:
    """在 (T,) 或 (T, C) 的反卷积输出里定位直达峰 (跨通道能量最大点)。

    给了 expected/search_radius 就只在窗口内搜, 避免被谐波残留或前置噪声骗到。
    """
    energy = ir ** 2 if ir.ndim == 1 else (ir ** 2).sum(axis=1)
    if expected is not None and search_radius:
        lo = max(0, expected - search_radius)
        hi = min(len(energy), expected + search_radius)
        if hi > lo:
            return int(lo + np.argmax(energy[lo:hi]))
    return int(np.argmax(energy))


def build_excitation(
    sweep: Sweep,
    repeats: int,
    preroll_s: float,
    gap_s: float,
    tail_s: float,
    amplitude: float = 0.5,
) -> tuple[np.ndarray, list[int]]:
    """拼出一个位置的完整播放波形。

    返回 (波形, 每次扫频起点的样本索引)。索引用于在录音里定位搜索窗。
    布局: [preroll 静音][sweep][gap][sweep]...[tail 静音]
    """
    fs = sweep.fs
    pre = int(round(preroll_s * fs))
    gap = int(round(gap_s * fs))
    tail = int(round(tail_s * fs))
    n = sweep.n

    total = pre + repeats * n + max(0, repeats - 1) * gap + tail
    out = np.zeros(total, dtype=float)
    starts = []
    pos = pre
    for i in range(repeats):
        out[pos: pos + n] = sweep.signal * amplitude
        starts.append(pos)
        pos += n + (gap if i < repeats - 1 else 0)
    return out, starts
