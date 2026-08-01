"""在实测多通道 RIR 上做混响随机增强。

保留实测早期段 (直达+早期反射, 承载阵列空间/近场线索),
按采样得到的 T60/DRR/着色/尾长, 重新合成扩散晚期尾并拼接。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import sample_value
from .decompose import split_early_late
from .tail import synth_tail


def _crossfade_splice(early: np.ndarray, tail: np.ndarray, split: int, cf: int) -> np.ndarray:
    """早期段与合成尾在边界处做 cf 样本的交叉淡化拼接。"""
    m = early.shape[0]
    cf = max(0, min(cf, split, tail.shape[1]))
    start = split - cf
    out_len = start + tail.shape[1]
    out = np.zeros((m, out_len), dtype=float)
    out[:, :split] = early
    if cf > 0:
        fade_out = np.linspace(1.0, 0.0, cf)
        fade_in = np.linspace(0.0, 1.0, cf)
        out[:, split - cf: split] *= fade_out[None, :]
        tail = tail.copy()
        tail[:, :cf] *= fade_in[None, :]
    out[:, start: start + tail.shape[1]] += tail
    return out


def augment_rir(
    rir: np.ndarray,
    fs: float,
    c: float,
    dist: np.ndarray,
    acfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """对一条 (M, T) 实测 RIR 做一次增强, 返回 (增强后 RIR, 采样到的参数)。"""
    rir = np.atleast_2d(rir).astype(float)
    m = rir.shape[0]

    # 采样本次增强参数
    t60 = sample_value(acfg["t60"], rng)
    damp = sample_value(acfg["hf_damping"], rng)
    drr = sample_value(acfg["drr"], rng)
    tilt = sample_value(acfg["spectral_tilt_db_per_oct"], rng)
    tail_ms = sample_value(acfg["tail_len_ms"], rng)
    tail_len = int(round(tail_ms * 1e-3 * fs))

    _, split = split_early_late(rir, fs, acfg["early_late_split_ms"])
    early = rir[:, :split].copy()

    sub = acfg["subbands"]
    tail = synth_tail(
        dist, tail_len, fs, c, t60, damp,
        centers=[float(x) for x in sub["octave_centers"]],
        ref_freq=float(sub["ref_freq"]),
        rng=rng, tilt_db_per_oct=tilt,
    )

    # 按目标 DRR 缩放尾能量: E_early / E_tail = 10^(DRR/10)
    e_early = float((early ** 2).sum())
    e_tail = float((tail ** 2).sum()) + 1e-20
    target_e_tail = e_early / (10.0 ** (drr / 10.0))
    tail *= np.sqrt(target_e_tail / e_tail)

    cf = int(round(acfg.get("crossfade_ms", 5) * 1e-3 * fs))
    out = _crossfade_splice(early, tail, split, cf)

    # 可选传感器噪声 (通道间不相干)
    noise_cfg = acfg.get("noise", {})
    snr = float("nan")
    if noise_cfg.get("enable", False):
        snr = sample_value(noise_cfg["snr_db"], rng)
        sig_p = float((out ** 2).mean())
        noise_p = sig_p / (10.0 ** (snr / 10.0))
        out = out + rng.standard_normal(out.shape) * np.sqrt(noise_p)

    params = {
        "t60": round(t60, 4), "hf_damping": round(damp, 4), "drr_db": round(drr, 3),
        "spectral_tilt": round(tilt, 3), "tail_len_ms": round(tail_ms, 1),
        "noise_snr_db": round(snr, 2) if snr == snr else None,
        "n_ch": m,
    }
    return out, params
