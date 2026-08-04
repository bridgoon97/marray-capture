"""混响随机增强的批处理入口 (包着 vendored rir-augment)。

与 rir-augment 原版的两点差异:

1. **只对麦克风通道合成扩散尾。** IR 文件里可能还有 VPU 这类非声学通道 ——
   VPU 拾到的是骨导/接触振动, 它对远场干扰人的响应主要是泄漏, 套球面各向同性
   相干模型是错的。这些通道保留实测 IR 原样 (截断/补零到相同长度)。
2. **按质检结论筛输入。** 只用 PASS (可选含 WARN) 的位置, FAIL 的不进增强。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import soundfile as sf

from .rir_augment.augment import augment_rir
from .rir_augment.geometry import build_array, pairwise_distances
from .speaker_eq import apply_inverse

ProgressCb = Callable[[int, int, str], None]


def default_config() -> dict[str, Any]:
    """与 rir-augment configs/default.yaml 一致的默认值 (16k 训练管线)。"""
    return {
        "seed": 2026,
        "sample_rate": 16000,
        "speed_of_sound": 343.0,
        "array": {
            "coords": None,
            "generator": {"type": "equilateral_triangle", "edge": 0.01},
        },
        "augment": {
            "early_late_split_ms": 50,
            "crossfade_ms": 5,
            "tail_len_ms": [300, 1200],
            "t60": {"range": [0.2, 0.8], "sampling": "log"},
            "hf_damping": {"range": [0.4, 0.9], "sampling": "linear"},
            "drr": {"range": [0.0, 15.0], "sampling": "linear"},
            "spectral_tilt_db_per_oct": {"range": [-3.0, 1.0], "sampling": "linear"},
            "subbands": {"ref_freq": 1000.0, "octave_centers": [125, 250, 500, 1000, 2000, 4000]},
            "noise": {"enable": True, "snr_db": [40, 60]},
        },
        "output": {"num_per_rir": 4, "manifest": "manifest.jsonl"},
    }


def parse_coords(text: str) -> list[list[float]]:
    """把「每行一个 x, y, z」的文本解析成坐标表 (单位米)。缺 z 补 0。

    非 3 麦的阵列 (比如 4 麦 + VPU) 只能靠显式坐标 —— 扩散尾的通道间相干性
    sinc(2πf·d/c) 完全由麦间距决定, 填错会让下游波束形成器得到过于乐观的结果。
    """
    pts: list[list[float]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.replace("，", ",").strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in line.replace("\t", " ").replace(",", " ").split() if p]
        try:
            vals = [float(p) for p in parts]
        except ValueError as e:
            raise ValueError(f"第 {lineno} 行解析失败: {raw.strip()!r}") from e
        if len(vals) == 2:
            vals.append(0.0)
        if len(vals) != 3:
            raise ValueError(f"第 {lineno} 行应有 2 或 3 个数, 实际 {len(vals)} 个")
        pts.append(vals)
    if len(pts) < 2:
        raise ValueError("至少要两个麦克风坐标")
    return pts


def select_inputs(
    session_dir: str | Path,
    fs: int = 16000,
    accept: Iterable[str] = ("PASS",),
    tags: Iterable[str] | None = None,
) -> list[tuple[Path, list[int]]]:
    """从会话 manifest 里挑出可用的 IR。返回 [(路径, 麦克风列号)]。"""
    root = Path(session_dir)
    mf = root / "manifest.jsonl"
    if not mf.exists():
        raise FileNotFoundError(f"{mf} 不存在")
    accept = set(accept)
    tags = set(tags) if tags else None
    key = "ir16k_file" if fs <= 16000 else "ir_file"
    out: list[tuple[Path, list[int]]] = []
    seen: set[str] = set()
    for line in mf.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("qc") or {}).get("verdict") not in accept:
            continue
        if tags and r.get("tag") not in tags:
            continue
        rel = r.get(key) or r.get("ir_file")
        if not rel or rel in seen:
            continue
        p = root / rel
        if p.exists():
            seen.add(rel)
            out.append((p, list(r.get("mic_cols") or [])))
    return out


def run_augment(
    inputs: list[tuple[Path, list[int]]],
    output_dir: str | Path,
    cfg: dict[str, Any],
    eq_fir: np.ndarray | None = None,
    progress: ProgressCb | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """对每条实测 IR 生成 num_per_rir 条增强版本。返回 manifest 路径。"""
    fs = int(cfg["sample_rate"])
    c = float(cfg["speed_of_sound"])
    acfg = cfg["augment"]
    n_per = int(cfg["output"]["num_per_rir"])
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / cfg["output"].get("manifest", "manifest.jsonl")

    pts = build_array(cfg["array"])
    n_geom = len(pts)
    total = len(inputs) * n_per
    done = 0

    # 全批单一标量归一化: 先攒着所有输出 IR, 算出整批的全局峰值, 再用同一个
    # 0.98/peak 标量写出。通道间/位置间的绝对电平关系是 ILD/DRR 的依据, 单一
    # 公共标量只做整体缩放、不破坏相对关系 —— 与 AGENTS.md「IR 一律不做峰值归一」
    # 的本意(反对按单条 IR 各自归一)不冲突。
    global_norm = bool(cfg.get("output", {}).get("global_norm"))
    pending: list[tuple[str, np.ndarray, dict]] = [] if global_norm else []

    with open(manifest, "w", encoding="utf-8") as mf:
        for path, mic_cols in inputs:
            if should_stop and should_stop():
                break
            data, src_fs = sf.read(str(path), always_2d=True)      # (L, C)
            if src_fs != fs:
                raise ValueError(f"{path.name} 采样率 {src_fs} != 配置 {fs}")
            if eq_fir is not None:
                data = apply_inverse(data, eq_fir)

            n_ch = data.shape[1]
            cols = mic_cols or list(range(min(n_geom, n_ch)))
            if len(cols) != n_geom:
                raise ValueError(
                    f"{path.name} 有 {len(cols)} 个麦克风通道, 但阵列几何只配了 {n_geom} 个。\n"
                    "在「阵列几何」里改成同样的麦克风数 —— 3 麦以外的布局选「自定义坐标」, "
                    "按实际间距逐行填 x, y, z (单位米)。\n"
                    "扩散尾的通道间相干性 sinc(2πf·d/c) 完全由这个几何决定, 填错会让"
                    "下游波束形成器得到过于乐观的结果。"
                )
            others = [i for i in range(n_ch) if i not in cols]

            mic_ir = data[:, cols].T                                # (M, L)
            for k in range(n_per):
                if should_stop and should_stop():
                    break
                aug, params = augment_rir(mic_ir, fs, c, pairwise_distances(pts), acfg, rng)
                out_len = aug.shape[1]
                full = np.zeros((out_len, n_ch), dtype=float)
                for m, col in enumerate(cols):
                    full[:, col] = aug[m]
                for col in others:                                  # 非声学通道保留实测
                    n = min(out_len, data.shape[0])
                    full[:n, col] = data[:n, col]

                name = f"{path.stem}_aug{k:03d}.wav"
                rec = {
                    "file": name, "source": path.name,
                    "mic_cols": cols, "passthrough_cols": others,
                    **params,
                }
                if global_norm:
                    pending.append((name, full, rec))
                else:
                    sf.write(str(out_root / name), full.astype(np.float32), fs, subtype="FLOAT")
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done += 1
                if progress:
                    progress(done, total, name)

        if global_norm and pending:
            peak = 0.0
            for _, full, _ in pending:
                p = float(np.max(np.abs(full)))
                if p > peak:
                    peak = p
            scale = 0.98 / peak if peak > 0 else 1.0
            for name, full, rec in pending:
                rec["global_norm_scale"] = scale
                sf.write(str(out_root / name), (full * scale).astype(np.float32),
                         fs, subtype="FLOAT")
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return manifest
