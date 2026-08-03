"""逐位置的 IR 可靠性验证。

指标分三组:

**录音层** —— 削顶、录音段 SNR。采不好的第一现场, 当场就该发现。

**IR 层** —— 直达/噪底比 (DDR)、可靠带宽、时间弥散。
噪声估计取**直达峰之前**的一段: ESS 的谐波失真产物落在 -L·ln(k) 处
(L = T/ln(f2/f1)), 二次谐波之后到直达峰之间那段既没有失真产物也没有混响,
是干净的反卷积域噪底。用峰后的尾巴估噪底会把真实混响算成噪声, 偏悲观。

**重复一致性** —— 同一位置两次扫频的归一化互相关、电平差、时钟漂移。
这是最有价值的一个指标: 佩戴者动了、蓝牙重同步、编码噪声过大, 全都会在这里暴露。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from ..settings import QCThresholds

OCTAVE_CENTERS = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_RANK = {PASS: 0, WARN: 1, FAIL: 2}


@dataclass
class ChannelQC:
    label: str
    peak_dbfs: float = float("nan")
    rec_snr_db: float = float("nan")
    ir_ddr_db: float = float("nan")
    ir_peak_db: float = float("nan")     # IR 直达峰电平 (绝对 dB), DDR 的分子
    ir_noise_db: float = float("nan")    # 反卷积域噪底电平 (绝对 dB), DDR 的分母
    reliable_bw_hz: float = float("nan")
    rel_level_db: float = 0.0          # 相对参考麦的电平 (查死麦/增益不一致)


@dataclass
class TakeQC:
    verdict: str = PASS
    reasons: list[str] = field(default_factory=list)
    channels: list[ChannelQC] = field(default_factory=list)
    repeat_ncc: float = float("nan")
    repeat_level_diff_db: float = float("nan")
    drift_ppm: float = 0.0
    latency_ms: float = 0.0
    smear_ms: float = float("nan")
    stream_warnings: str = ""

    def worsen(self, verdict: str, reason: str) -> None:
        self.reasons.append(reason)
        if _RANK[verdict] > _RANK[self.verdict]:
            self.verdict = verdict

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channels"] = [asdict(c) for c in self.channels]
        return d

    def summary(self) -> str:
        if not self.reasons:
            return "全部指标通过"
        return " / ".join(self.reasons)


# ---------------------------------------------------------------------- 工具
def _db(x: float) -> float:
    return float(10.0 * np.log10(max(float(x), 1e-20)))


def _band_energies(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 (倍频带中心, 各带能量)。x 为 (L,) 单通道。"""
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    out = []
    centers = []
    for fc in OCTAVE_CENTERS:
        lo, hi = fc / np.sqrt(2.0), fc * np.sqrt(2.0)
        if lo >= fs / 2.0:
            continue
        m = (freqs >= lo) & (freqs < min(hi, fs / 2.0))
        if m.sum() == 0:
            continue
        centers.append(fc)
        out.append(float(spec[m].mean()))
    return np.array(centers), np.array(out)


def _reliable_bandwidth(sig: np.ndarray, noise: np.ndarray, fs: int, min_snr_db: float = 6.0) -> float:
    """从低频起连续满足带内 SNR ≥ min_snr_db 的最高倍频带上边界 (Hz)。"""
    n = min(len(sig), len(noise))
    if n < 64:
        return float("nan")
    c_s, e_s = _band_energies(sig[:n], fs)
    _, e_n = _band_energies(noise[:n], fs)
    ok_upto = 0.0
    started = False
    for fc, es, en in zip(c_s, e_s, e_n):
        snr = _db(es) - _db(en)
        if snr >= min_snr_db:
            started = True
            ok_upto = fc * np.sqrt(2.0)
        elif started:
            break
    return float(min(ok_upto, fs / 2.0))


def _smear_ms(ir_mono: np.ndarray, peak_idx: int, fs: int, drop_db: float = 10.0) -> float:
    """直达峰在 -10dB 处的时间宽度 (ms)。时钟漂移/编码劣化会让它变宽。"""
    env = np.abs(ir_mono)
    if peak_idx >= len(env) or env[peak_idx] <= 0:
        return float("nan")
    thr = env[peak_idx] * 10.0 ** (-drop_db / 20.0)
    i = peak_idx
    while i > 0 and env[i] > thr:
        i -= 1
    j = peak_idx
    while j < len(env) - 1 and env[j] > thr:
        j += 1
    return float((j - i) / fs * 1000.0)


# ---------------------------------------------------------------------- 主入口
def evaluate_take(
    rec: np.ndarray,
    deconv: np.ndarray,
    ir_list: list[np.ndarray],
    ir_avg: np.ndarray,
    peaks: list[int],
    latency: int,
    drift_ppm: float,
    starts: list[int],
    sweep_n: int,
    pre_samples: int,
    fs: int,
    labels: list[str],
    mic_cols: list[int],
    thr: QCThresholds,
    stream_warnings: str = "",
    early_ms: float = 50.0,
    vpu_cols: list[int] | None = None,
) -> TakeQC:
    """对一个位置的采集做完整质检。

    rec:      (T, C) 原始录音 (已按通道映射切好)
    deconv:   (T+N-1, C) 整段反卷积输出
    ir_list:  每次扫频裁出的 IR, 每条 (L, C)
    peaks:    每次扫频在 deconv 里的直达峰绝对索引
    mic_cols: 参与阵列判定的列号 (VPU/loopback 不计入可靠带宽的硬门限)
    """
    qc = TakeQC(drift_ppm=drift_ppm, latency_ms=latency / fs * 1000.0,
                stream_warnings=stream_warnings)
    n_ch = rec.shape[1]
    d0 = starts[0] + latency                       # 录音域中第一次扫频的到达位置

    # ---- 噪声窗 (直达峰之前, 避开谐波失真产物) 与信号窗
    noise_lo = max(0, peaks[0] - int(0.40 * fs))
    noise_hi = max(noise_lo + 1, peaks[0] - int(0.01 * fs))
    dec_noise = deconv[noise_lo:noise_hi]

    # 录音最前面的一段是声卡开流瞬态 (输入缓冲区初始化/ADC 启动), 每通道都会
    # 有一个接近 0 dBFS 的窄脉冲, 落在 guard 静音区里。它不是信号, 却会同时
    # 污染两个指标: ① peak_dbfs 对整段取 max → 误判削顶; ② rec_noise 从 t=0
    # 起算时把它当噪底 → SNR 偏低。guard 本就是静音, 把这前若干 ms 从录音层
    # 指标里整体剔掉。脉冲实测约 5 ms, 取 10 ms 留余量。
    startup = int(0.01 * fs)
    rec_noise = rec[max(startup, d0 - int(1.0 * fs)): max(startup + 1, d0 - int(0.05 * fs))]
    rec_sig = rec[max(startup, d0): min(len(rec), d0 + sweep_n)]

    early_n = int(round(early_ms * 1e-3 * fs))
    ref_rms = None

    for c in range(n_ch):
        label = labels[c] if c < len(labels) else f"ch{c + 1}"
        ch = ChannelQC(label=label)

        # 峰值取在扫频信号窗, 不取整段 rec —— 开流瞬态在 guard 区, 信号窗在
        # d0 之后, 天然避开它; 也避开尾部的静音/漂移段。这样接近 0 dBFS 的
        # 启动脉冲不再误触发削顶判据。
        sig = rec_sig if len(rec_sig) else rec
        peak = float(np.max(np.abs(sig[:, c]))) if len(sig) else 0.0
        ch.peak_dbfs = 20.0 * np.log10(max(peak, 1e-12))

        if len(rec_noise) > 16 and len(rec_sig) > 16:
            p_n = float((rec_noise[:, c] ** 2).mean())
            p_s = float((rec_sig[:, c] ** 2).mean())
            ch.rec_snr_db = _db(max(p_s - p_n, 1e-20)) - _db(p_n)

        ir_c = ir_avg[:, c]
        pk = float(np.max(np.abs(ir_c))) if len(ir_c) else 0.0
        # 同时存 DDR 的两个分量: 分子 ir_peak_db (直达峰电平)、分母 ir_noise_db
        # (噪底电平)。比值异常时一眼看出是峰太小 (定位错/扫频没录好) 还是噪底太高
        # (漂移失配/失真/爆音铺进噪底窗)。光看比值分不清哪头坏。
        ch.ir_peak_db = _db(pk ** 2)
        if len(dec_noise) > 16:
            n_pow = float((dec_noise[:, c] ** 2).mean())
            ch.ir_noise_db = _db(n_pow)
            ch.ir_ddr_db = ch.ir_peak_db - ch.ir_noise_db
            sig_win = ir_c[pre_samples: pre_samples + early_n]
            ch.reliable_bw_hz = _reliable_bandwidth(sig_win, dec_noise[:, c], fs)

        rms = float(np.sqrt((ir_c[pre_samples: pre_samples + early_n] ** 2).mean() + 1e-30))
        if ref_rms is None and c in mic_cols:
            ref_rms = rms
        ch.rel_level_db = 20.0 * np.log10(rms / ref_rms) if ref_rms else 0.0

        qc.channels.append(ch)

    # ---- 时间弥散 (取麦克风通道的能量和)
    if mic_cols:
        mono = np.sqrt((ir_avg[:, mic_cols] ** 2).sum(axis=1))
        qc.smear_ms = _smear_ms(mono, int(np.argmax(mono)), fs)

    # ---- 重复一致性
    if len(ir_list) >= 2:
        cols = mic_cols or list(range(n_ch))
        a = ir_list[0][pre_samples: pre_samples + early_n, cols].ravel()
        b = ir_list[1][pre_samples: pre_samples + early_n, cols].ravel()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        qc.repeat_ncc = float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")
        qc.repeat_level_diff_db = 20.0 * np.log10((na + 1e-20) / (nb + 1e-20))

    _apply_thresholds(qc, thr, mic_cols, vpu_cols)
    return qc


def _apply_thresholds(qc: TakeQC, thr: QCThresholds, mic_cols: list[int],
                      vpu_cols: list[int] | None = None) -> None:
    """把指标翻译成 PASS / WARN / FAIL。差一点点的判 WARN, 差得多的判 FAIL。"""

    # VPU 是非声学通道 (骨导/接触), 中低频本来就没信号, SNR/DDR 偏低是物理必然,
    # 不该当失败判据。削顶和"几乎没信号"对所有通道照判 —— 那是录音层故障, 与用途无关。
    vpu = set(vpu_cols or [])

    def check(value: float, limit: float, higher_is_better: bool, name: str, unit: str) -> None:
        if value != value:                       # nan
            qc.worsen(WARN, f"{name} 无法计算")
            return
        margin = abs(limit) * thr.warn_margin if limit else 1.0
        if higher_is_better:
            if value < limit - margin:
                qc.worsen(FAIL, f"{name} {value:.1f}{unit} < {limit:.1f}{unit}")
            elif value < limit:
                qc.worsen(WARN, f"{name} {value:.1f}{unit} 偏低")
        else:
            if value > limit + margin:
                qc.worsen(FAIL, f"{name} {value:.1f}{unit} > {limit:.1f}{unit}")
            elif value > limit:
                qc.worsen(WARN, f"{name} {value:.1f}{unit} 偏高")

    for i, ch in enumerate(qc.channels):
        if ch.peak_dbfs > thr.clip_dbfs:
            qc.worsen(FAIL, f"{ch.label} 削顶 ({ch.peak_dbfs:.1f} dBFS) — 降声卡硬件输入增益")
        elif ch.peak_dbfs < -50.0:
            qc.worsen(FAIL, f"{ch.label} 几乎没有信号 ({ch.peak_dbfs:.1f} dBFS), 疑似死麦或接线问题")
        if i not in vpu:
            check(ch.rec_snr_db, thr.min_rec_snr_db, True, f"{ch.label} 录音 SNR", " dB")
            check(ch.ir_ddr_db, thr.min_ir_ddr_db, True, f"{ch.label} IR 直达噪底比", " dB")
        if i in mic_cols:
            check(ch.reliable_bw_hz, thr.min_reliable_bw_hz, True, f"{ch.label} 可靠带宽", " Hz")

    # 死通道检测: 同一次采集里所有麦录的是同一个源, 直达峰应在彼此 ~十几 dB 内。
    # 某条麦比最响的那条低 30 dB 以上, 几乎肯定是没接/接错/增益没开 —— 不是"SNR 偏低"
    # 能解释的。单独标出来, 让用户去通道表里取消该通道的麦角色, 而不是整场当废片。
    mic_peaks = [qc.channels[i].ir_peak_db for i in mic_cols
                 if qc.channels[i].ir_peak_db == qc.channels[i].ir_peak_db]
    if mic_peaks:
        loudest = max(mic_peaks)
        for i in mic_cols:
            ch = qc.channels[i]
            if ch.ir_peak_db == ch.ir_peak_db and loudest - ch.ir_peak_db > 30.0:
                qc.worsen(FAIL, f"{ch.label} 疑似死通道/未接线 (直达峰 {ch.ir_peak_db:.0f} dB, "
                                f"比最响麦低 {loudest - ch.ir_peak_db:.0f} dB) — "
                                f"在通道表里取消该通道的麦角色")

    check(qc.repeat_ncc, thr.min_repeat_ncc, True, "两次扫频一致性", "")
    check(abs(qc.repeat_level_diff_db), thr.max_repeat_level_diff_db, False, "两次扫频电平差", " dB")
    check(abs(qc.drift_ppm), thr.max_drift_ppm, False, "时钟漂移", " ppm")

    if qc.stream_warnings:
        qc.worsen(WARN, f"音频流告警: {qc.stream_warnings}")
