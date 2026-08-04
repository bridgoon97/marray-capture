"""对已落盘的录音用最新 QC 逻辑重跑, 改判旧 manifest。

录制时 QC 是内联的 (run.py 里 extract_take + evaluate_take 一把跑完), 逻辑修过
之后旧 manifest 上的判据就是过时的 —— 典型如 onset 误锁修复前, 遮挡下的两次
扫频被错判 FAIL (假漂移 500ppm + NCC 崩)。这里从 raw 录音重跑 deconv→定位→切
IR→evaluate, 用更新后的代码重算 qc, 写回 manifest 与 IR。

不依赖录制时的 starts/leadin: 两次扫频的直达峰是反卷积里最显著的一对 (间隔
≈ 名义间隔 ± 真实漂移), 直接按间隔配对找到 peak0, 再用 ``_locate_subsequent``
(互相关, 防遮挡下 argmax 锁反射) 定 peak1。这样不必重建 leadin (TTS 指令 +
提示音 + 倒计时, 长度随步变化且依赖 TTS 后端, 重建脆弱), 也不需要录制时落盘
starts。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

from ..audio.sweep import build_excitation, generate_ess
from ..rir.extract import (
    PeakNotFoundError, _cut, _estimate_lag, _fractional_shift,
    _locate_subsequent, deconvolve_take,
)
from ..settings import QCThresholds
from .metrics import evaluate_take


def _role_map(session_settings: dict) -> dict[str, str]:
    """label → role, 从 session.json 的通道表建。manifest 的 channels 只存了 label。"""
    out: dict[str, str] = {}
    for c in session_settings["audio"]["channels"]:
        if c.get("enabled"):
            out[c.get("label", "")] = c.get("role", "")
    return out


def _find_first_peak(
    sub_e: np.ndarray, nominal: int, fs: int, prominence_db: float = 30.0,
) -> int:
    """配对找第一次扫频的直达峰: 反卷积里其后约 nominal 处还有一个显著峰的
    那些峰里, 取**最高**的一个 = 直达 (匹配滤波相干增益最高, 比反射/谐波失真
    产物都高)。

    直接重建 leadin 太脆 (TTS 后端/提示音/倒计时), 配对绕开它。容差 1% 名义
    间隔 ≈ 2000 ppm, 远超 200 ppm 的漂移失败线, 真实漂移再大也配得上; 同时窄到
    不会被无关的提示音/启动瞬态误配。

    不按"间隔最接近名义"挑 —— 同一相对偏移的反射对 (反射1↔反射2) 间隔也
    = 名义+真实漂移, 误差和直达对一样, 靠误差分辨不开; 而且遮挡下反射可能
    反而更高, 不能取"最高"以外的东西。但同一对里峰0必是较早的那个 (Q>P),
    所以选的是 sweep1 的直达不是 sweep2 的。
    """
    noise = float(np.median(sub_e))
    pk, _ = find_peaks(
        sub_e, prominence=noise * 10.0 ** (prominence_db / 10.0),
        distance=int(0.005 * fs),
    )
    if len(pk) < 2:
        raise PeakNotFoundError(
            f"反卷积里找不到两个显著峰 (只找到 {len(pk)} 个), 无法配对两次扫频。"
        )
    tol = int(nominal * 0.01)
    pkset = set(int(p) for p in pk)
    candidates: list[tuple[int, float]] = []
    for p in pk:
        p = int(p)
        for q in pkset:
            if q > p and abs((q - p) - nominal) <= tol:
                candidates.append((p, float(sub_e[p])))
                break
    if not candidates:
        raise PeakNotFoundError(
            "找不到间隔 ≈ 名义间隔的一对显著峰 —— 录音可能不完整或扫频未触发。")
    candidates.sort(key=lambda t: -t[1])
    return candidates[0][0]


def requantify_take(
    session_dir: str | Path, row: dict, session_settings: dict,
    progress=None,
) -> dict:
    """对一个 take 从 raw 重跑 QC, 返回要写回 manifest 的字段。

    返回 dict 含: qc, ir_avg, ir16k (若需要), n_averaged, latency_samples,
    drift_ppm, direct_indices。不在这里写盘, 由调用者落盘 + 改 manifest。
    """
    sdir = Path(session_dir)
    au = session_settings["audio"]
    sw = session_settings["sweep"]
    ex = session_settings["export"]
    fs = int(au["samplerate"])

    sweep = generate_ess(fs, sw["f_start"], sw["f_end"], sw["duration_s"])
    _exc, starts = build_excitation(
        sweep, sw["repeats"], sw["preroll_s"], sw["gap_s"], sw["tail_s"],
        sw["amplitude"],
    )
    nominal = starts[-1] - starts[0]
    radius = int(0.03 * fs)

    labels = list(row.get("channels") or [])
    mic_cols = list(row.get("mic_cols") or list(range(len(labels))))
    roles = _role_map(session_settings)
    vpu_cols = [i for i, lb in enumerate(labels) if roles.get(lb, "") == "vpu"]
    n_ch = len(labels)

    raw_rel = row.get("raw_file")
    if not raw_rel:
        raise FileNotFoundError(f"{row.get('take_id')} 没有存 raw 录音, 无法重跑。")
    # manifest 在 Windows 上落盘用反斜杠, 跨平台读回来要归一化
    raw_path = sdir / Path(str(raw_rel).replace("\\", "/"))
    rec, fs2 = sf.read(str(raw_path), always_2d=True)
    if fs2 != fs:
        raise ValueError(f"{row['take_id']} raw 采样率 {fs2} != 配置 {fs}")

    deconv = deconvolve_take(rec, sweep)
    sub = deconv[:, :n_ch] if n_ch else deconv
    sub_mic = sub[:, mic_cols] if mic_cols else sub
    sub_e = (sub_mic ** 2).sum(axis=1)

    peak0 = _find_first_peak(sub_e, nominal, fs)
    # peak1 走互相关 (修复后的路径), 防遮挡下 argmax 锁反射
    peak1 = _locate_subsequent(sub_mic, peak0, peak0 + nominal, radius, fs)
    peaks = [peak0, peak1]
    drift = (peak1 - peak0 - nominal) / nominal * 1e6 if nominal else 0.0

    pre = int(round(ex["ir_pre_ms"] * 1e-3 * fs))
    length = int(round(ex["ir_len_ms"] * 1e-3 * fs))
    irs = [_cut(deconv, p, pre, length) for p in peaks]
    for k in range(1, len(irs)):
        lag = _estimate_lag(irs[0], irs[k], mic_cols)
        if abs(lag) > 1e-6:
            irs[k] = _fractional_shift(irs[k], lag)

    average = bool(ex.get("average_repeats", True)) and len(irs) > 1
    ir_avg = np.stack(irs, axis=0).mean(axis=0) if average else irs[0]
    n_avg = len(irs) if average else 1

    # d0 (录音域第一次扫频到达) = peak0 - ir_offset; 用原 latency 保持 latency_ms
    # 信息性一致 (设备延迟没变, 没必要也不易重算 —— 没存 starts)。
    latency = int(row.get("latency_samples") or 0)
    starts_for_eval = [peak0 - sweep.ir_offset - latency]

    thr = QCThresholds(**(session_settings.get("qc") or {}))
    stream_warnings = (row.get("qc") or {}).get("stream_warnings", "")
    qc = evaluate_take(
        rec=rec, deconv=deconv, ir_list=irs, ir_avg=ir_avg,
        peaks=peaks, latency=latency, drift_ppm=drift,
        starts=starts_for_eval, sweep_n=sweep.n, pre_samples=pre, fs=fs,
        labels=labels, mic_cols=mic_cols, thr=thr,
        stream_warnings=stream_warnings, vpu_cols=vpu_cols,
    )
    # 一致性不过不平均 (亚样本错位会梳状衰减高频), 与 runner 一致
    if qc.repeat_ncc == qc.repeat_ncc and qc.repeat_ncc < thr.min_repeat_ncc:
        ir_avg = irs[0]
        n_avg = 1

    out = {
        "qc": qc.to_dict(),
        "ir_avg": ir_avg,
        "n_averaged": n_avg,
        "latency_samples": latency,
        "drift_ppm": float(drift),
        "direct_indices": peaks,
        "fs": fs,
    }
    if ex.get("export_16k") and fs != 16000:
        from ..rir.extract import resample_ir
        out["ir16k"] = resample_ir(ir_avg, fs, 16000)
    return out


def requantify_session(
    session_dir: str | Path, take_ids: set[str] | None, settings: dict,
    log=None, should_stop=None,
) -> list[tuple[str, str, str]]:
    """对整个会话重跑 QC, 改判 manifest + 重存 IR。

    take_ids 为 None 重跑全部, 否则只跑指定的。返回 [(take_id, 旧判, 新判)]。
    重存的 IR 落到原 ir_file 路径 (save_ir 按 take_id 定位, 覆盖同路径文件),
    所以 manifest 里的 ir_file/ir16k_file 字符串不动。
    """
    from ..store import Session

    log = log or (lambda *a: None)
    sdir = Path(session_dir)
    session = Session(sdir.parent, sdir.name, create=False)
    rows = session.load_manifest()
    changed: list[tuple[str, str, str]] = []
    for r in rows:
        if should_stop and should_stop():
            break
        tid = r.get("take_id", "")
        if take_ids is not None and tid not in take_ids:
            continue
        if not r.get("raw_file"):
            log(f"跳过 {tid}: 未存 raw 录音, 无法重跑")
            continue
        old_v = (r.get("qc") or {}).get("verdict", "")
        try:
            out = requantify_take(sdir, r, settings)
        except Exception as e:
            log(f"✗ {tid}: 重算失败 — {e}")
            continue
        r["qc"] = out["qc"]
        r["n_averaged"] = out["n_averaged"]
        r["latency_samples"] = out["latency_samples"]
        # IR 重存到原路径 (文件名 = take_id, save_ir 覆盖)。manifest 里的路径串不动。
        for sub in ("ir48k", "ir16k"):
            (session.dir / sub).mkdir(exist_ok=True)
        session.save_ir(tid, out["ir_avg"], out["fs"])
        if out.get("ir16k") is not None:
            session.save_ir(tid, out["ir16k"], 16000)
        new_v = out["qc"]["verdict"]
        changed.append((tid, old_v, new_v))
        log(f"✓ {tid}: {old_v} → {new_v}  (NCC {out['qc']['repeat_ncc']:.2f}, "
            f"漂移 {out['qc']['drift_ppm']:+.0f} ppm)")
    session.rewrite_manifest(rows)
    session.write_qc_csv()
    return changed
