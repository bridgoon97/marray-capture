"""从零合成多通道 RIR (bootstrap 模式)。

无实测数据时的引导生成: 直达路径 (按随机声源位置的近场 1/r 幅度与到达时延) +
若干随机早期反射 + 扩散晚期尾。

注意: 合成的直达/早期段是几何点声源近似, 不含真实人嘴指向性,
无法完全复现近场指向性效应 —— 正式数据应以扫频实测的早期段替代 (见 augment 模式)。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import sample_value
from .geometry import pairwise_distances
from .tail import synth_tail

_SINC_HALF = 16  # 分数延迟窗长半径


def _add_fractional_impulse(buf: np.ndarray, ch: int, delay_samp: float, gain: float) -> None:
    """在 buf[ch] 的 delay_samp (可为小数) 处叠加一个幅度 gain 的冲激 (加窗 sinc 插值)。"""
    n0 = int(np.floor(delay_samp))
    frac = delay_samp - n0
    idx = np.arange(-_SINC_HALF, _SINC_HALF + 1)
    kernel = np.sinc(idx - frac) * np.hamming(2 * _SINC_HALF + 1)
    lo = n0 - _SINC_HALF
    for k, off in enumerate(idx):
        pos = n0 + off
        if 0 <= pos < buf.shape[1]:
            buf[ch, pos] += gain * kernel[k]
    _ = lo  # 保留可读性


def _direction_to_point(distance: float, az_deg: float, el_deg: float) -> np.ndarray:
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return distance * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def synth_rir(
    pts: np.ndarray,
    fs: float,
    c: float,
    acfg: dict[str, Any],
    scfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """合成一条 (M, T) RIR, 返回 (RIR, 采样参数)。"""
    m = len(pts)
    dist = pairwise_distances(pts)

    # 直达声源位置
    r = sample_value(scfg["source"]["distance_m"], rng)
    az = sample_value(scfg["source"]["azimuth_deg"], rng)
    el = sample_value(scfg["source"]["elevation_deg"], rng)
    src = _direction_to_point(r, az, el)
    dmic = np.linalg.norm(pts - src[None, :], axis=1)  # (M,)

    # 尾长决定总长度
    tail_ms = sample_value(acfg["tail_len_ms"], rng)
    tail_len = int(round(tail_ms * 1e-3 * fs))
    head = int(round(0.06 * fs))  # 60ms 早期窗
    total = head + tail_len
    rir = np.zeros((m, total), dtype=float)

    # 直达: 近场 1/r 幅度, 时延对齐到最近麦
    toa = dmic / c
    t0 = toa.min()
    gain = 1.0 / np.maximum(dmic, 1e-3)
    gain = gain / gain.max()
    for ch in range(m):
        _add_fractional_impulse(rir, ch, (toa[ch] - t0) * fs + 2, gain[ch])

    # 早期反射: 各自随机方向 + 时延 + 增益
    er = scfg["early_reflections"]
    n_er = int(round(sample_value(er["num"], rng)))
    for _ in range(n_er):
        d_ms = sample_value(er["delay_ms"], rng)
        g_db = sample_value(er["gain_db"], rng)
        r_az = rng.uniform(-180, 180)
        r_el = rng.uniform(-30, 30)
        ref_src = _direction_to_point(r + c * d_ms * 1e-3, r_az, r_el)
        r_dmic = np.linalg.norm(pts - ref_src[None, :], axis=1)
        r_toa = r_dmic / c
        base = (r_toa.min() - t0) * fs + 2
        g = 10.0 ** (g_db / 20.0)
        rg = (1.0 / np.maximum(r_dmic, 1e-3))
        rg = rg / rg.max() * g
        for ch in range(m):
            _add_fractional_impulse(rir, ch, base + (r_toa[ch] - r_toa.min()) * fs, rg[ch])

    # 晚期扩散尾, 拼接在 head 之后
    t60 = sample_value(acfg["t60"], rng)
    damp = sample_value(acfg["hf_damping"], rng)
    tilt = sample_value(acfg["spectral_tilt_db_per_oct"], rng)
    drr = sample_value(acfg["drr"], rng)
    sub = acfg["subbands"]
    tail = synth_tail(
        dist, tail_len, fs, c, t60, damp,
        centers=[float(x) for x in sub["octave_centers"]],
        ref_freq=float(sub["ref_freq"]), rng=rng, tilt_db_per_oct=tilt,
    )
    e_early = float((rir[:, :head] ** 2).sum())
    e_tail = float((tail ** 2).sum()) + 1e-20
    tail *= np.sqrt((e_early / (10.0 ** (drr / 10.0))) / e_tail)
    rir[:, head:head + tail_len] += tail

    params = {
        "mode": "synth", "src_dist_m": round(r, 3), "az_deg": round(az, 1),
        "el_deg": round(el, 1), "t60": round(t60, 4), "hf_damping": round(damp, 4),
        "drr_db": round(drr, 3), "spectral_tilt": round(tilt, 3),
        "tail_len_ms": round(tail_ms, 1), "n_early_refl": n_er, "n_ch": m,
    }
    return rir, params
