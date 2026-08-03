"""端到端自检: 用仿真的已知 IR 走一遍 扫频 → 录音 → 反卷积 → 提取 → 质检 → 增强。"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve

from marray_capture.audio.sweep import build_excitation, deconvolve, generate_ess
from marray_capture.protocol.plan import build_plan, coarse_label
from marray_capture.qc.metrics import evaluate_take
from marray_capture.rir.extract import deconvolve_take, extract_take, resample_ir
from marray_capture.rir import speaker_eq
from marray_capture.settings import ProtocolConfig, QCThresholds

FS = 48000


def make_ir(n_ch: int = 3, length: int = 4800, seed: int = 0,
            bandlimit: bool = False) -> np.ndarray:
    """一条带直达 + 几个早反射 + 指数尾的多通道 IR, (L, C)。

    bandlimit=True 时把 IR 限制到扫频覆盖的频带内 —— 反卷积只能还原激励覆盖到的
    频段, 拿全频带的仿真 IR 直接比对是不公平的。
    """
    rng = np.random.default_rng(seed)
    ir = np.zeros((length, n_ch))
    for c in range(n_ch):
        ir[40 + c, c] = 1.0 - 0.1 * c                     # 直达, 通道间小延迟
        for _ in range(6):
            k = int(rng.integers(60, 900))
            ir[k, c] += rng.normal(0, 0.15)
        t = np.arange(length) / FS
        ir[:, c] += rng.normal(0, 0.02, length) * np.exp(-t / 0.12)
    if bandlimit:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(4, [80.0, 18000.0], btype="band", fs=FS, output="sos")
        ir = sosfiltfilt(sos, ir, axis=0)
    return ir


def simulate(play: np.ndarray, ir: np.ndarray, latency: int, snr_db: float = 45.0,
             seed: int = 1, peak_dbfs: float = -12.0) -> np.ndarray:
    """播放波形经过 IR 后被录到的信号, 含固定延迟与底噪。

    整体缩放到 peak_dbfs, 模拟真实录音的电平余量 (不缩放的话 QC 会判削顶)。
    """
    rng = np.random.default_rng(seed)
    n_ch = ir.shape[1]
    out = np.stack([fftconvolve(play, ir[:, c])[: len(play) + 2000] for c in range(n_ch)], axis=1)
    rec = np.zeros((latency + len(out), n_ch))
    rec[latency:] = out
    p = float((out ** 2).mean())
    rec += rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10.0)), rec.shape)
    peak = float(np.max(np.abs(rec))) or 1.0
    return rec / peak * 10.0 ** (peak_dbfs / 20.0)


# ---------------------------------------------------------------------- 扫频
def test_ess_deconvolution_recovers_delta():
    sw = generate_ess(FS, 40, 20000, 2.0)
    y = deconvolve(sw.signal, sw)
    peak = int(np.argmax(np.abs(y)))
    assert peak == sw.ir_offset
    assert np.isclose(y[peak], 1.0, atol=0.05)
    # 峰旁瓣要低: 除去峰附近 1ms, 其余不应超过峰的 -30dB
    mask = np.ones(len(y), bool)
    mask[max(0, peak - 48): peak + 48] = False
    assert np.max(np.abs(y[mask])) < 0.05


def test_ess_recovers_known_ir():
    sw = generate_ess(FS, 40, 20000, 2.0)
    ir = make_ir(bandlimit=True)
    rec = simulate(sw.signal, ir, latency=0, snr_db=60.0, peak_dbfs=0.0)
    est = deconvolve(rec, sw)
    got = est[sw.ir_offset: sw.ir_offset + 2000]
    ref = ir[:2000]
    for c in range(ir.shape[1]):
        num = float(got[:, c] @ ref[:, c])
        den = float(np.linalg.norm(got[:, c]) * np.linalg.norm(ref[:, c]))
        assert num / den > 0.98, f"通道 {c} 相关性只有 {num / den:.3f}"


# ---------------------------------------------------------------------- 提取
@pytest.mark.parametrize("latency_ms", [0.0, 120.0, 480.0])
def test_extract_handles_unknown_latency(latency_ms):
    sw = generate_ess(FS, 40, 20000, 2.0)
    exc, starts = build_excitation(sw, repeats=2, preroll_s=1.0, gap_s=0.8,
                                   tail_s=1.0, amplitude=0.5)
    lat = int(latency_ms * 1e-3 * FS)
    ir = make_ir()
    rec = simulate(exc, ir, latency=lat, snr_db=45.0)

    deconv = deconvolve_take(rec, sw)
    take = extract_take(deconv, sw, starts, ir_pre_ms=5.0, ir_len_ms=200.0,
                        max_latency_s=1.0, energy_channels=[0, 1, 2])
    # 直达在 IR 里第 40 个样本, 所以估到的延迟 = 真实延迟 + 40
    assert abs(take.latency_samples - (lat + 40)) <= 2
    assert abs(take.drift_ppm) < 200.0
    assert take.n_averaged == 2
    assert take.ir_avg.shape == (int(0.2 * FS), 3)
    # 平均后的 IR 直达峰应落在 pre_samples 处
    mono = np.abs(take.ir_avg).sum(axis=1)
    assert abs(int(np.argmax(mono)) - take.pre_samples) <= 1


def test_qc_passes_on_clean_take_and_fails_on_noisy():
    sw = generate_ess(FS, 40, 20000, 2.0)
    exc, starts = build_excitation(sw, 2, 1.0, 0.8, 1.0, 0.5)
    ir = make_ir()
    thr = QCThresholds()

    for snr_db, expect_pass in [(50.0, True), (3.0, False)]:
        rec = simulate(exc, ir, latency=int(0.1 * FS), snr_db=snr_db, seed=7)
        deconv = deconvolve_take(rec, sw)
        take = extract_take(deconv, sw, starts, 5.0, 200.0, 1.0, energy_channels=[0, 1, 2])
        qc = evaluate_take(
            rec=rec, deconv=deconv, ir_list=take.irs, ir_avg=take.ir_avg,
            peaks=take.direct_indices, latency=take.latency_samples,
            drift_ppm=take.drift_ppm, starts=starts, sweep_n=sw.n,
            pre_samples=take.pre_samples, fs=FS,
            labels=["mic1", "mic2", "mic3"], mic_cols=[0, 1, 2], thr=thr,
        )
        if expect_pass:
            assert qc.verdict == "PASS", qc.summary()
            assert qc.repeat_ncc > 0.99
        else:
            assert qc.verdict == "FAIL", qc.summary()


def test_qc_catches_movement_between_sweeps():
    """两次扫频之间佩戴者动了 → 一致性指标必须掉下来。"""
    sw = generate_ess(FS, 40, 20000, 2.0)
    exc, starts = build_excitation(sw, 2, 1.0, 0.8, 1.0, 0.5)
    ir_a, ir_b = make_ir(seed=1), make_ir(seed=99)
    lat = int(0.1 * FS)

    # 第一次扫频用 ir_a, 第二次用 ir_b: 分别仿真再拼起来
    sweep_len = sw.n
    rec = np.zeros((lat + len(exc) + 2000, 3))
    for k, ir in enumerate((ir_a, ir_b)):
        seg = np.zeros(len(exc))
        seg[starts[k]: starts[k] + sweep_len] = exc[starts[k]: starts[k] + sweep_len]
        part = simulate(seg, ir, latency=lat, snr_db=55.0, seed=3 + k)
        rec[: len(part)] += part

    deconv = deconvolve_take(rec, sw)
    take = extract_take(deconv, sw, starts, 5.0, 200.0, 1.0, energy_channels=[0, 1, 2])
    qc = evaluate_take(
        rec=rec, deconv=deconv, ir_list=take.irs, ir_avg=take.ir_avg,
        peaks=take.direct_indices, latency=take.latency_samples,
        drift_ppm=take.drift_ppm, starts=starts, sweep_n=sw.n,
        pre_samples=take.pre_samples, fs=FS,
        labels=["mic1", "mic2", "mic3"], mic_cols=[0, 1, 2], thr=QCThresholds(),
    )
    assert qc.repeat_ncc < 0.95
    assert qc.verdict == "FAIL"


def test_subsample_alignment_recovers_ncc():
    """两次扫频只差亚样本 (真实蓝牙漂移的典型量级) → 对齐后 NCC 该到 0.99+。

    不做亚样本对齐时, 整样本对齐残留的 0.5 样本错位会在宽带早期段梳状衰减,
    把 NCC 压到 0.8x, 看着像佩戴者动过 —— 这是这组改动要治的根。
    """
    sw = generate_ess(FS, 40, 20000, 2.0)
    exc, starts = build_excitation(sw, 2, 1.0, 0.8, 1.0, 0.5)
    ir = make_ir(seed=1)

    def _shift(x, s):
        n = x.shape[0]
        X = np.fft.rfft(x, axis=0)
        f = np.fft.rfftfreq(n, 1.0)[:, None]
        return np.fft.irfft(X * np.exp(-2j * np.pi * f * s), n=n, axis=0)

    ir_b = _shift(ir, 0.5)                       # 第二条用的 IR 晚 0.5 样本
    lat = int(0.1 * FS)
    rec = np.zeros((lat + len(exc) + 2000, 3))
    for k, irk in enumerate((ir, ir_b)):
        seg = np.zeros(len(exc))
        seg[starts[k]: starts[k] + sw.n] = exc[starts[k]: starts[k] + sw.n]
        part = simulate(seg, irk, latency=lat, snr_db=55.0, seed=3 + k)
        rec[: len(part)] += part

    deconv = deconvolve_take(rec, sw)
    take = extract_take(deconv, sw, starts, 5.0, 200.0, 1.0, energy_channels=[0, 1, 2])
    qc = evaluate_take(
        rec=rec, deconv=deconv, ir_list=take.irs, ir_avg=take.ir_avg,
        peaks=take.direct_indices, latency=take.latency_samples,
        drift_ppm=take.drift_ppm, starts=starts, sweep_n=sw.n,
        pre_samples=take.pre_samples, fs=FS,
        labels=["mic1", "mic2", "mic3"], mic_cols=[0, 1, 2], thr=QCThresholds(),
    )
    assert qc.repeat_ncc > 0.99, f"亚样本对齐失效, NCC={qc.repeat_ncc:.3f}"


def test_resample_to_16k_preserves_direct_peak():
    ir = make_ir(length=9600)
    ir16 = resample_ir(ir, FS, 16000)
    assert ir16.shape[0] == 3200
    assert abs(int(np.argmax(np.abs(ir16[:, 0]))) - 40 // 3) <= 2


# ---------------------------------------------------------------------- 方案
def test_plan_counts_and_labels():
    cfg = ProtocolConfig(
        distances_cm=[50, 100], heights_cm=[120, 160],
        dense_distance_cm=100, dense_height_cm=160, dense_steps=12, sparse_steps=8,
        speaker_orientations=[0, 180], orientation_subset_steps=4,
        random_positions=5, rewearing_rings=1, rewearing_steps=8,
    )
    plan = build_plan(cfg)
    # 主网格 2×2 圈: 一圈 dense(12) + 三圈 sparse(8) = 36; 朝向 4; 重戴 8; 随机 5
    assert len(plan.measures) == 12 + 8 * 3 + 4 + 8 + 5
    assert all(s.take_id for s in plan.measures)
    assert len({s.take_id for s in plan.measures}) == len(plan.measures)
    assert plan.estimated_seconds(15.0) > 0
    # 转四分之一圈后音箱应该在左侧
    assert coarse_label(90) == "正左"
    assert coarse_label(270) == "正右"


def test_plan_roundtrip(tmp_path):
    plan = build_plan(ProtocolConfig(random_positions=2, rewearing_rings=0))
    p = plan.save(tmp_path / "plan.json")
    back = type(plan).load(p)
    assert len(back.steps) == len(plan.steps)
    assert back.steps[3].instruction == plan.steps[3].instruction


# ---------------------------------------------------------------------- 后处理
def test_speaker_eq_flattens_response():
    """造一个带限 + 有明显斜率的"音箱", 反滤波后通带内应显著更平。"""
    from scipy.signal import butter, sosfilt

    fs = 48000
    imp = np.zeros(2048)
    imp[0] = 1.0
    sos = butter(2, [150.0, 7000.0], btype="band", fs=fs, output="sos")
    b = sosfilt(sos, imp)
    b = sosfilt(butter(1, 2500.0, btype="low", fs=fs, output="sos"), b)   # 再加一段高频滚降
    ref = b[:, None]

    fir = speaker_eq.build_from_reference(ref, fs, channel=0, f_lo=200, f_hi=6000,
                                          max_boost_db=20, fir_len=2048)
    corrected = speaker_eq.apply_inverse(ref, fir, keep_length=False)[:, 0]

    def spread(x):
        f = np.fft.rfftfreq(8192, 1 / fs)
        m = 20 * np.log10(np.abs(np.fft.rfft(x, 8192)) + 1e-12)
        band = (f > 300) & (f < 5000)
        return float(m[band].std())

    assert spread(corrected) < spread(b) * 0.4


def test_augment_end_to_end(tmp_path):
    import soundfile as sf
    from marray_capture.rir.augment_runner import default_config, run_augment

    fs = 16000
    ir = np.zeros((1600, 4))          # 3 麦 + 1 路 VPU
    ir[20, 0] = 1.0
    ir[21, 1] = 0.9
    ir[22, 2] = 0.8
    ir[20, 3] = 0.3
    ir[100:400] += np.random.default_rng(0).normal(0, 0.02, (300, 4))
    src = tmp_path / "take.wav"
    sf.write(str(src), ir.astype(np.float32), fs, subtype="FLOAT")

    cfg = default_config()
    cfg["output"]["num_per_rir"] = 2
    out = tmp_path / "aug"
    manifest = run_augment([(src, [0, 1, 2])], out, cfg)

    files = sorted(out.glob("*.wav"))
    assert len(files) == 2
    data, got_fs = sf.read(str(files[0]), always_2d=True)
    assert got_fs == fs
    assert data.shape[1] == 4
    # 实测早期段必须被原样保留
    assert abs(data[20, 0] - 1.0) < 1e-3
    # VPU 通道走直通, 不合成扩散尾
    assert abs(data[20, 3] - 0.3) < 1e-3
    assert manifest.exists()
    assert len(manifest.read_text(encoding="utf-8").strip().splitlines()) == 2
