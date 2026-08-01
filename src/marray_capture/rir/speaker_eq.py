"""去音箱响应。

测出来的 IR = 房间/头部/阵列 ⊗ **音箱自身的频响**。做 RTF (麦间相对量) 时音箱响应
是公共项会约掉, 但把 IR 直接和干净语音卷积生成训练音频时, 它会给干扰人染色 ——
网络可能学到"带某种音色的就是干扰人"。所以正式数据建议做一次去音箱响应。

用法: 用测量麦在音箱正轴 1 米处录一条参考 IR (软件里当成一个普通位置录即可,
把该通道角色设为 ref), 然后在后处理页选这条参考 IR。

实现要点:
- 只取参考 IR 的**前若干毫秒**估幅度, 避开房间反射, 否则会把房间也一起反掉。
- 幅度做 1/3 倍频程平滑, 只反掉整体趋势, 不去追细结构。
- 反滤波器做成**最小相位**并限幅提升量, 避免在音箱滚降的频段疯狂放大噪声。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def _to_mono(ir: np.ndarray, channel: int | None = None) -> np.ndarray:
    x = np.atleast_2d(np.asarray(ir, dtype=float))
    if x.shape[0] < x.shape[1]:      # (C, L) → (L, C)
        x = x.T
    if x.shape[1] == 1:
        return x[:, 0]
    if channel is not None:
        return x[:, channel]
    return x[:, int(np.argmax((x ** 2).sum(axis=0)))]


def _fractional_octave_smooth(mag: np.ndarray, freqs: np.ndarray, frac: float = 3.0) -> np.ndarray:
    """1/frac 倍频程平滑 (对数频率上的滑动平均)。"""
    out = mag.copy()
    ratio = 2.0 ** (1.0 / (2.0 * frac))
    for i, f in enumerate(freqs):
        if f <= 0:
            continue
        lo, hi = f / ratio, f * ratio
        m = (freqs >= lo) & (freqs <= hi)
        if m.any():
            out[i] = mag[m].mean()
    return out


def estimate_response(
    ref_ir: np.ndarray, fs: int, window_ms: float = 20.0,
    n_fft: int = 8192, channel: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """从参考 IR 估音箱幅度响应。返回 (freqs, mag)。"""
    h = _to_mono(ref_ir, channel)
    peak = int(np.argmax(np.abs(h)))
    n = int(round(window_ms * 1e-3 * fs))
    pre = int(0.001 * fs)
    start = max(0, peak - pre)
    seg = h[start: start + n]
    if len(seg) < n:
        seg = np.pad(seg, (0, n - len(seg)))

    # 必须用「起始平坦、尾部淡出」的窗 (Tukey 式)。对称 Hann 会把直达峰削掉,
    # 而直达正是我们要估的那部分响应。
    w = np.ones(n)
    fi = min(peak - start, n // 8)
    fo = max(1, n // 4)
    if fi > 0:
        w[:fi] = 0.5 * (1.0 - np.cos(np.pi * np.arange(fi) / fi))
    w[-fo:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(fo) / fo))
    spec = np.fft.rfft(seg * w, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    mag = _fractional_octave_smooth(np.abs(spec), freqs, frac=3.0)
    return freqs, mag


def design_inverse(
    freqs: np.ndarray, mag: np.ndarray, fs: int,
    f_lo: float = 80.0, f_hi: float | None = None,
    max_boost_db: float = 12.0, fir_len: int = 2048,
) -> np.ndarray:
    """由幅度响应设计最小相位反滤波器 FIR。"""
    f_hi = f_hi if f_hi is not None else fs / 2.0 * 0.9
    n_fft = 2 * (len(freqs) - 1)

    ref = float(np.median(mag[(freqs >= max(f_lo, 200.0)) & (freqs <= min(f_hi, 4000.0))]) + 1e-20)
    inv = ref / np.maximum(mag, 1e-20)
    inv = np.clip(inv, 10.0 ** (-max_boost_db / 20.0), 10.0 ** (max_boost_db / 20.0))

    # 带外不反 (平滑过渡), 避免在音箱没有输出的频段放大噪声
    band = np.ones_like(inv)
    lo_m = freqs < f_lo
    hi_m = freqs > f_hi
    if lo_m.any():
        band[lo_m] = np.linspace(0.0, 1.0, lo_m.sum())
    if hi_m.any():
        band[hi_m] = np.linspace(1.0, 0.0, hi_m.sum())
    inv = 1.0 + (inv - 1.0) * band

    # 最小相位: 由 log|G| 经倒谱折叠得到
    log_mag = np.log(np.maximum(inv, 1e-12))
    ceps = np.fft.irfft(log_mag, n=n_fft)
    w = np.zeros(n_fft)
    w[0] = 1.0
    w[1: n_fft // 2] = 2.0
    w[n_fft // 2] = 1.0
    spec = np.exp(np.fft.rfft(ceps * w, n=n_fft))
    fir = np.fft.irfft(spec, n=n_fft)[:fir_len]
    fade = min(64, fir_len // 8)
    fir[-fade:] *= np.linspace(1.0, 0.0, fade)
    return fir


def apply_inverse(ir: np.ndarray, fir: np.ndarray, keep_length: bool = True) -> np.ndarray:
    """对 (L, C) 的 IR 施加反滤波器。最小相位 FIR 附加群延迟很小, 直接从 0 截取。"""
    x = np.atleast_2d(np.asarray(ir, dtype=float))
    if x.shape[0] < x.shape[1]:
        x = x.T
    out = np.stack([fftconvolve(x[:, c], fir, mode="full") for c in range(x.shape[1])], axis=1)
    return out[: x.shape[0]] if keep_length else out


def build_from_reference(
    ref_ir: np.ndarray, fs: int, channel: int | None = None,
    f_lo: float = 80.0, f_hi: float | None = None,
    max_boost_db: float = 12.0, fir_len: int = 2048, window_ms: float = 20.0,
) -> np.ndarray:
    """一步到位: 参考 IR → 反滤波器 FIR。"""
    freqs, mag = estimate_response(ref_ir, fs, window_ms=window_ms, channel=channel)
    return design_inverse(freqs, mag, fs, f_lo=f_lo, f_hi=f_hi,
                          max_boost_db=max_boost_db, fir_len=fir_len)
