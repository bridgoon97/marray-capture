"""各向同性 (球面) 扩散场的通道间相干性, 及满足该相干性的多通道噪声生成。

球面各向同性扩散场两麦相干性:
    Gamma_ij(f) = sinc(2*pi*f*d_ij / c) = sin(x)/x,  x = 2*pi*f*d_ij/c
numpy.sinc(t) = sin(pi t)/(pi t), 故 Gamma = np.sinc(2*f*d/c)。

多通道扩散噪声用频域混合法 (Habets 等, "generating sensor signals in
isotropic noise fields") 生成: 每个频点求满足 C C^H = Gamma 的混合矩阵,
作用到独立白噪声谱上。用特征分解并把负特征值截零, 比 Cholesky 更鲁棒
(sinc 相干矩阵在部分频点可能非正定)。
"""
from __future__ import annotations

import numpy as np


def diffuse_coherence(dist: np.ndarray, freqs: np.ndarray, c: float) -> np.ndarray:
    """返回 (F, M, M) 的球面各向同性相干矩阵。"""
    d = dist[None, :, :]          # (1, M, M)
    f = freqs[:, None, None]      # (F, 1, 1)
    return np.sinc(2.0 * f * d / c)


def generate_diffuse_noise(
    dist: np.ndarray, n: int, fs: float, c: float, rng: np.random.Generator
) -> np.ndarray:
    """生成长度 n 的多通道扩散噪声, 通道间相干性匹配球面各向同性场。

    返回形状 (M, n)。
    """
    m = dist.shape[0]
    freqs = np.fft.rfftfreq(n, 1.0 / fs)          # (F,)
    w = rng.standard_normal((m, n))
    W = np.fft.rfft(w, axis=1)                     # (M, F)

    gamma = diffuse_coherence(dist, freqs, c)      # (F, M, M)
    vals, vecs = np.linalg.eigh(gamma)             # 批量特征分解
    vals = np.clip(vals, 0.0, None)
    # 对称平方根 C = V diag(sqrt(vals)) V^H, 满足 C C^H = Gamma
    c_mat = vecs @ (np.sqrt(vals)[..., None] * np.swapaxes(vecs.conj(), -1, -2))  # (F,M,M)

    X = np.einsum("kij,jk->ik", c_mat, W)          # (M, F)
    x = np.fft.irfft(X, n=n, axis=1)               # (M, n)
    return x


def measured_coherence(x: np.ndarray, nfft: int = 1024) -> np.ndarray:
    """估计多通道信号的幅度平方相干 |gamma|^2, 返回 (F, M, M)。用于自检。"""
    from scipy.signal import csd

    m = x.shape[0]
    f = None
    pxx = []
    for i in range(m):
        f, p = csd(x[i], x[i], nperseg=nfft)
        pxx.append(p)
    coh = np.zeros((len(f), m, m))
    for i in range(m):
        for j in range(m):
            _, pij = csd(x[i], x[j], nperseg=nfft)
            coh[:, i, j] = np.abs(pij) ** 2 / (np.abs(pxx[i]) * np.abs(pxx[j]) + 1e-20)
    return coh
