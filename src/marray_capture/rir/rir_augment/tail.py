"""晚期混响尾合成: 频带相关的衰减 + 通道间扩散相干性。

对每个倍频带独立施加指数衰减包络 (高频带 T60 更短, 更真实), 再求和。
带通滤波与实值包络对所有通道一致, 因此不破坏 generate_diffuse_noise
建立的通道间相干性。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .coherence import generate_diffuse_noise

# 60dB 幅度衰减常数: A(t)=exp(-K t/T60), A(T60)=10^-3
_DECAY_K = 3.0 * np.log(10.0)  # ≈ 6.9078


def _octave_sos(fc: float, fs: float, order: int = 4):
    lo = fc / np.sqrt(2.0)
    hi = min(fc * np.sqrt(2.0), fs / 2.0 * 0.999)
    lo = max(lo, 1e-3)
    return butter(order, [lo, hi], btype="band", fs=fs, output="sos")


def band_t60(t60_ref: float, fc: float, ref_freq: float, hf_damping: float, fs: float) -> float:
    """由宽带 T60 与高频阻尼比推出某倍频带的 T60。

    hf_damping = T60(nyquist)/T60(ref) ∈ (0, 1]; =1 表示频率无关。
    幂律模型 T60(f) = t60_ref * (f/ref_freq)^p。
    """
    damp = float(np.clip(hf_damping, 1e-3, 1.0))
    nyq = fs / 2.0
    p = np.log(damp) / np.log(nyq / ref_freq)
    return float(t60_ref * (fc / ref_freq) ** p)


def synth_tail(
    dist: np.ndarray,
    length: int,
    fs: float,
    c: float,
    t60_ref: float,
    hf_damping: float,
    centers: list[float],
    ref_freq: float,
    rng: np.random.Generator,
    tilt_db_per_oct: float = 0.0,
) -> np.ndarray:
    """合成多通道晚期混响尾, 形状 (M, length)。能量未归一 (由 DRR 步骤缩放)。"""
    noise = generate_diffuse_noise(dist, length, fs, c, rng)  # (M, length)
    t = np.arange(length) / fs
    out = np.zeros_like(noise)
    for fc in centers:
        if fc >= fs / 2.0:
            continue
        sos = _octave_sos(fc, fs)
        band = sosfiltfilt(sos, noise, axis=1)
        t60_b = max(band_t60(t60_ref, fc, ref_freq, hf_damping, fs), 1e-3)
        env = np.exp(-_DECAY_K * t / t60_b)                    # (length,)
        tilt_gain = 10.0 ** (tilt_db_per_oct * np.log2(fc / ref_freq) / 20.0)
        out += band * env[None, :] * tilt_gain
    return out
