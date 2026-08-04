"""分析一个 session: 定位低 DDR 的真正原因。

从 raw wav 重跑 deconv→extract→evaluate, 拿到 DDR 的两个分量 (IR 峰电平 /
噪底电平), 对比 OK take 和失败 take, 看是峰太小 (定位错/扫频没录好) 还是
噪底太高 (漂移失配/失真/爆音铺进噪底窗)。
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from marray_capture.audio.sweep import generate_ess, build_excitation
from marray_capture.qc.metrics import evaluate_take
from marray_capture.rir.extract import deconvolve_take, extract_take
from marray_capture.settings import QCThresholds


def db(x):
    return 10.0 * np.log10(max(float(x), 1e-20))


def load_session(sdir):
    sdir = Path(sdir)
    sess = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
    manifest = [json.loads(l) for l in (sdir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return sdir, sess, manifest


def rebuild_sweep(sess):
    sw = sess["sweep"]; au = sess["audio"]
    sweep = generate_ess(au["samplerate"], sw["f_start"], sw["f_end"], sw["duration_s"])
    exc, starts = build_excitation(
        sweep, sw["repeats"], sw["preroll_s"], sw["gap_s"], sw["tail_s"], sw["amplitude"])
    return sweep, exc, starts, au


def recompute(sdir, sess, row):
    """从 raw wav 重跑全链路, 返回 (qc, take, dec, peaks, sweep_n, fs)."""
    sweep, exc, starts, au = rebuild_sweep(sess)
    fs = au["samplerate"]
    sw = sess["sweep"]
    raw = sdir / "raw" / row["raw_file"]
    rec, fs2 = sf.read(str(raw), always_2d=True)
    if fs2 != fs:
        print(f"  !! fs 不匹配: raw={fs2} config={fs}")
    dec = deconvolve_take(rec, sweep)
    take = extract_take(
        dec, sweep, starts, sess["export"]["ir_pre_ms"], sess["export"]["ir_len_ms"],
        sess["sweep"]["guard_s"] + sess["sweep"]["max_latency_s"],
        energy_channels=None,  # 用全部
    )
    labels = [c.get("label", f"ch{i}") for i, c in enumerate(row.get("channels", []))]
    mic_cols = [i for i, c in enumerate(row.get("channels", [])) if c.get("role") == "mic"]
    if not mic_cols:
        mic_cols = list(range(rec.shape[1]))
    qc = evaluate_take(
        rec=rec, deconv=dec, ir_list=take.irs, ir_avg=take.ir_avg,
        peaks=take.direct_indices, latency=take.latency_samples,
        drift_ppm=take.drift_ppm, starts=starts, sweep_n=sweep.n,
        pre_samples=take.pre_samples, fs=fs, labels=labels, mic_cols=mic_cols,
        thr=QCThresholds(), stream_warnings=(row.get("qc") or {}).get("stream_warnings", ""),
    )
    return qc, take, dec, rec, starts, sweep


def main(sdir):
    sdir, sess, manifest = load_session(sdir)
    au = sess["audio"]; sw = sess["sweep"]
    print("=" * 70)
    print(f"session: {sdir.name}")
    print(f"  duplex_mode={au.get('duplex_mode')}  drift_correction={au.get('drift_correction')}  drift_ppm(cfg)={au.get('drift_ppm')}")
    print(f"  input={au.get('input_device')}  output={au.get('output_device')}")
    print(f"  sweep: f=[{sw['f_start']},{sw['f_end']}] dur={sw['duration_s']}s repeats={sw['repeats']} amp={sw['amplitude']} guard={sw['guard_s']}s gap={sw['gap_s']}s max_lat={sw['max_latency_s']}s")
    print(f"  export: ir_pre={sess['export']['ir_pre_ms']}ms ir_len={sess['export']['ir_len_ms']}ms")
    print(f"  takes: {len(manifest)}")

    # 先用 manifest 里现成的 qc 排个序, 找出 DDR 最低和最高的几个
    def min_ddr(r):
        chans = (r.get("qc") or {}).get("channels") or []
        ds = [c.get("ir_ddr_db") for c in chans if isinstance(c.get("ir_ddr_db"), (int, float)) and c["ir_ddr_db"] == c["ir_ddr_db"]]
        return min(ds) if ds else 999
    rows_sorted = sorted(manifest, key=min_ddr)
    print("\n--- manifest 里的 DDR (录制时算的) ---")
    for r in rows_sorted[:6] + (rows_sorted[-3:] if len(rows_sorted) > 9 else []):
        qc = r.get("qc") or {}
        print(f"  {r.get('take_id',''):20s} v={qc.get('verdict','')[:4]:4s} DDR_min={min_ddr(r):6.1f} "
              f"drift={qc.get('drift_ppm'):.0f}ppm smear={qc.get('smear_ms')}ms ncc={qc.get('repeat_ncc')} "
              f"warn={qc.get('stream_warnings','')[:30]}")

    # 挑 1 个最低 + 1 个最高 (OK), 重跑全链路对比
    targets = []
    if rows_sorted:
        targets.append(("LOW", rows_sorted[0]))
        if len(rows_sorted) > 1 and min_ddr(rows_sorted[-1]) > 20:
            targets.append(("OK ", rows_sorted[-1]))

    for tag, r in targets:
        print("\n" + "=" * 70)
        print(f"[{tag}] {r.get('take_id')}  verdict={(r.get('qc') or {}).get('verdict')}")
        try:
            qc, take, dec, rec, starts, sweep = recompute(sdir, sess, r)
        except Exception as e:
            print(f"  重算失败: {e}")
            continue
        fs = sess["audio"]["samplerate"]
        print(f"  drift(实测)={take.drift_ppm:.1f}ppm  latency={take.latency_samples/fs*1000:.1f}ms  "
              f"peaks={[p for p in take.direct_indices]}  nominal1st={starts[0]+sweep.ir_offset}")
        print(f"  rec shape={rec.shape}  rec peak={20*np.log10(max(np.abs(rec).max(),1e-12)):.1f}dBFS")
        # 检查扫频段是否削波
        d0 = starts[0] + take.latency_samples
        sweep_n = sweep.n
        sig = rec[max(0,d0):min(len(rec),d0+sweep_n)]
        print(f"  扫频段: peak={20*np.log10(max(np.abs(sig).max(),1e-12)):.1f}dBFS "
              f"rms={20*np.log10(max(np.sqrt((sig**2).mean()),1e-12)):.1f}dBFS "
              f"crest={20*np.log10(max(np.abs(sig).max()/max(np.sqrt((sig**2).mean()),1e-12))):.1f}dB")
        # 分量
        print(f"  通道: {'label':6s} {'peak_dbfs':>9s} {'rec_snr':>7s} {'ir_peak':>8s} {'ir_noise':>9s} {'DDR':>7s}")
        for c in qc.channels:
            print(f"        {c.label:6s} {c.peak_dbfs:9.1f} {c.rec_snr_db:7.1f} "
                  f"{c.ir_peak_db:8.1f} {c.ir_noise_db:9.1f} {c.ir_ddr_db:7.1f}")
        # 噪底窗位置 & 内容 (第 0 通道)
        ch0 = 0
        pk = take.direct_indices[0]
        nlo = max(0, pk - int(0.40 * fs)); nhi = max(nlo + 1, pk - int(0.01 * fs))
        dec_noise = dec[nlo:nhi, ch0]
        print(f"  噪底窗 [{nlo}:{nhi}] (peak={pk}, 0.4~0.01s 前)  rms={20*np.log10(max(np.sqrt((dec_noise**2).mean()),1e-12)):.1f}dB")
        # 对比: 直达峰前 0.4s 但避开谐波区 (0.05~0.01s 前, 更靠峰)
        nlo2 = max(0, pk - int(0.05*fs)); nhi2 = max(nlo2+1, pk - int(0.01*fs))
        near = dec[nlo2:nhi2, ch0]
        print(f"  峰前近窗 [{nlo2}:{nhi2}] (0.05~0.01s)    rms={20*np.log10(max(np.sqrt((near**2).mean()),1e-12)):.1f}dB")
        # 看直达峰本身多宽 (smear)
        print(f"  smear={qc.smear_ms:.2f}ms (直达峰 -10dB 宽度)")
        # 谐波区: 2nd 谐波在 -L ln2, L=T/ln(f2/f1)
        L = sw["duration_s"] / np.log(sw["f_end"]/sw["f_start"])
        h2 = pk - int(L*np.log(2)*fs)
        h3 = pk - int(L*np.log(3)*fs)
        print(f"  谐波位置: 2nd@{h2} ({L*np.log(2)*1000:.0f}ms前) 3rd@{h3} ({L*np.log(3)*1000:.0f}ms前)  在噪底窗内? {h2>nlo and h2<nhi}")
        if 0 <= h2 < len(dec):
            print(f"    2nd谐波区 deconv rms={20*np.log10(max(np.sqrt((dec[max(0,h2-200):h2+200,ch0]**2).mean()),1e-12)):.1f}dB")

if __name__ == "__main__":
    main(sys.argv[1])
