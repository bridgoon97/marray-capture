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


class PeakNotFoundError(RuntimeError):
    """搜索窗内没有明显高于噪底的直达峰。

    通常意味着输出设备的启动延迟超过了 max_latency_s (搜索窗), 或根本没录到信号。
    这时若静默抓窗内 argmax 当峰, 下游会算出看似合理实则垃圾的 drift/NCC/DDR
    (实测出现过 -4400 ppm、NCC 0.02 这种)。改成显式失败, 让用户去查
    output_latency / 声卡驱动, 而不是被垃圾指标误导。
    """


# 真实直达峰在反卷积域里远高于噪底 (匹配滤波相干积分, 即使 SNR≈0 的弱扫频也有
# ~42 dB prominence)。峰落在搜索窗外时, 窗内只剩噪底 + 启动脉冲的反卷积残渣,
# prominence ~18-24 dB; 纯噪底 ~10 dB。30 dB 卡在中间, 两侧都留 6+ dB 余量。
_PROMINENCE_DB = 30.0


def _prominence_db(seg_e: np.ndarray, peak_idx: int) -> float:
    """窗内峰能 vs 中位能 (dB)。真实直达峰 42+ dB; 峰落窗外/纯噪底时 10~24 dB。"""
    if len(seg_e) < 8 or not (0 <= peak_idx < len(seg_e)):
        return 0.0
    peak = float(seg_e[peak_idx])
    med = float(np.median(seg_e))
    if peak <= 0.0 or med <= 0.0:
        return 0.0
    return 10.0 * np.log10(peak / med)


def _locate_subsequent(
    sub: np.ndarray, ref_peak: int, expected: int, radius: int, fs: int,
) -> int:
    """后续扫频的峰位: 用第一次扫频直达峰段做模板互相关, 不用窄窗 argmax。

    直达声被头遮挡时, ±radius 窗内的 argmax 会锁到稍晚/稍早的反射 (遮挡下
    反射幅度可能比直达峰还高), 两次扫频的峰位差 1~3ms, 被算成 400~500 ppm 的
    假漂移并伴生电平差/NCC 掉 —— 看着像"佩戴者动了 / 蓝牙重同步", 实际是
    onset 没对齐。实测: 同一位置两次扫频的真实直达在名义位置 ±0, 而 argmax
    锁到 ±90~105 样本处的反射, 报 521/-446 ppm, 但按名义位置切 IR 算 NCC=0.99。

    改成拿第一次扫频直达峰前后 ±1ms 当模板, 在期望位置 ±radius 内做互相关:
    直达对齐时两次扫频**共享的全部早反射**同时对上, 总相关最高, 单个强反射
    (哪怕幅度更大) 拼不过; 反射的邻域与直达邻域不同, 区别于幅度。互相关是
    相位感知 + 跨通道求和, 比能量 argmax / 单样本 onset 都稳。±30ms 窗远小于
    二次谐波失真产物 (-L·ln(k), 几百 ms), 不会误锁到失真产物。

    只要第一次峰是真的 (进这里之前已过 prominence 守卫), 模板就是有效的。
    """
    n = sub.shape[0]
    half = max(2, int(0.001 * fs))                 # 模板 ±1ms: 直达峰 + 紧邻
    ref_lo, ref_hi = max(0, ref_peak - half), min(n, ref_peak + half)
    ref = sub[ref_lo:ref_hi]
    peak_in_ref = ref_peak - ref_lo                 # 直达峰在模板里的下标
    if len(ref) < 4 or np.linalg.norm(ref) <= 0.0:
        return find_direct_index(sub, expected=expected, search_radius=radius)

    seg_lo = max(0, expected - radius)
    seg_hi = min(n, expected + radius + len(ref))
    seg = sub[seg_lo:seg_hi]
    if len(seg) < len(ref):
        return find_direct_index(sub, expected=expected, search_radius=radius)

    # 跨通道求和的 valid 互相关。多通道相位一致, 比单通道或能量 argmax 更难被骗。
    cc = np.zeros(len(seg) - len(ref) + 1, dtype=float)
    for c in range(sub.shape[1]):
        cc += np.correlate(seg[:, c], ref[:, c], mode="valid")
    lag = int(np.argmax(cc))
    return seg_lo + lag + peak_in_ref


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
    """定位每次扫频的直达峰。返回 (峰位列表, 延迟样本数, 漂移 ppm)。

    抛 ``PeakNotFoundError`` 当搜索窗内没有明显高于噪底的峰 —— 多半是输出设备
    启动延迟超过 max_latency_s, 这时抓噪声当峰会污染下游全部指标。
    """
    fs = sweep.fs
    sub = deconv if energy_channels is None else deconv[:, energy_channels]
    sub_e = (sub ** 2).sum(axis=1)        # 跨通道能量, 算一次复用
    base = starts[0] + sweep.ir_offset
    span = int(round(max_latency_s * fs))

    # 第一次: 用标定值窄窗 (若有) 先试, 命中就用; 延迟漂移出 ±50ms 没命中就
    # 回退全局窗再搜 —— 标定值是上次跑的, 这次实际延迟可能差几十 ms (WASAPI
    # 共享模式启动抖动), 死守窄窗会把好峰判成"找不到"。
    if known_latency is not None:
        # ±100ms: WASAPI 共享模式启动延迟每次抖几十 ms (实测过 104/193/186ms),
        # 标定值是上次的, 窗太窄 (原 ±50ms) 会把延迟漂移的那条误杀走回退。
        r = int(0.10 * fs)
        first = find_direct_index(sub, expected=base + known_latency, search_radius=r)
        lo, hi = max(0, base + known_latency - r), min(len(sub), base + known_latency + r)
        if _prominence_db(sub_e[lo:hi], first - lo) < _PROMINENCE_DB:
            lo, hi = base, min(len(sub), base + span)
            first = int(lo + np.argmax(sub_e[lo:hi])) if hi > lo else base
    else:
        lo, hi = base, min(len(sub), base + span)
        first = int(lo + np.argmax(sub_e[lo:hi])) if hi > lo else base

    # 护栏: 全局窗内峰仍不明显高于噪底, 才算真没峰 (设备延迟超 max_latency_s
    # 或根本没录到信号), 别静默抓噪声当峰。
    if _prominence_db(sub_e[lo:hi], first - lo) < _PROMINENCE_DB:
        raise PeakNotFoundError(
            f"在 {max_latency_s:.2f}s 搜索窗内未找到明显高于噪底的直达峰"
            f" (峰仅比噪底高 {_prominence_db(sub_e[lo:hi], first - lo):.1f} dB)。"
            f" 输出设备启动延迟可能远超搜索窗 —— 检查 output_latency 设置"
            f" (实测过 8~19s 的) 与声卡驱动, 或调大 max_latency_s。"
        )

    latency = first - base
    peaks = [first]
    radius = int(0.03 * fs)                    # 后续扫频只在 ±30ms 内找
    for k in range(1, len(starts)):
        exp = starts[k] + sweep.ir_offset + latency
        peaks.append(_locate_subsequent(sub, first, exp, radius, fs))

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

    # 多次扫频各自按整样本直达峰对齐, 残留的亚样本错位直接平均会梳状衰减高频,
    # 静止位置两次扫频的 NCC 也会被压到 0.8x 看着像动了。这里把第 2 条起按互相关
    # 峰做亚样本平移对到第 1 条上 —— 用 FFT 相位旋转 (全通, 不动频谱幅度,
    # 绝对电平关系是后面 ILD/DDR 的依据, 不能被动)。对齐后再平均 / 再算一致性,
    # NCC 能到 0.99+。对齐跟 average 开不开解耦: 一致性指标无论是否平均都该公平。
    if len(irs) > 1:
        chs = energy_channels if energy_channels is not None else list(range(irs[0].shape[1]))
        for k in range(1, len(irs)):
            lag = _estimate_lag(irs[0], irs[k], chs)
            if abs(lag) > 1e-6:
                irs[k] = _fractional_shift(irs[k], lag)

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


def _estimate_lag(a: np.ndarray, b: np.ndarray, chs: list[int]) -> float:
    """返回把 b 对齐到 a 所需的样本平移 (可亚样本)。用指定通道算归一化互相关。

    直达峰已经在 ``_cut`` 里按整样本对齐过, 残差 <1 样本, ±3 样本窗足够;
    再用抛物线插值得亚样本峰。窗开大了会把无关反射也对上, 反而虚高。
    """
    aa = a[:, chs] if chs else a
    bb = b[:, chs] if chs else b
    n = min(len(aa), len(bb))
    if n < 8:
        return 0.0
    aa = aa[:n].astype(float)
    bb = bb[:n].astype(float)
    na, nb = float(np.linalg.norm(aa)), float(np.linalg.norm(bb))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    W = 3
    cors: dict[int, float] = {}
    for lag in range(-W, W + 1):
        if lag >= 0:
            x, y = aa[lag:], bb[:n - lag]
        else:
            x, y = aa[:n + lag], bb[-lag:]
        if len(x) > 1:
            nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
            if nx > 0.0 and ny > 0.0:
                cors[lag] = float((x * y).sum() / (nx * ny))
    if not cors:
        return 0.0
    best = max(cors, key=cors.get)
    if -W < best < W:
        ym1, y0, yp1 = cors[best - 1], cors[best], cors[best + 1]
        denom = ym1 - 2.0 * y0 + yp1
        if denom != 0.0:
            delta = 0.5 * (ym1 - yp1) / denom
            if abs(delta) <= 1.0:
                return float(best) + float(delta)
    return float(best)


def _fractional_shift(x: np.ndarray, shift: float) -> np.ndarray:
    """按 shift 个样本平移 (可亚样本), 沿 axis=0。FFT 相位旋转, 全通 ——
    不改频谱幅度, 所以绝对电平关系 (ILD/DRR 的依据) 不会被动, 这点比样条插值好。"""
    if abs(shift) < 1e-9:
        return np.asarray(x, dtype=float)
    n = x.shape[0]
    X = np.fft.rfft(np.asarray(x, dtype=float), axis=0)
    f = np.fft.rfftfreq(n, 1.0)[:, None]
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * shift), n=n, axis=0)


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
