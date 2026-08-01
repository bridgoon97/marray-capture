"""批处理编排: augment (增强实测 RIR) 与 synth (从零合成)。

每条输出写一个 wav, 并把采样到的参数追加进 manifest (jsonl), 便于复现与筛选。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .augment import augment_rir
from .config import load_config
from .geometry import build_array, pairwise_distances
from .io_utils import load_rir, save_rir
from .synth import synth_rir


def _rng(cfg: dict[str, Any], salt: int = 0) -> np.random.Generator:
    return np.random.default_rng(int(cfg.get("seed", 0)) + salt)


def run_augment(cfg: dict[str, Any], input_dir: str, output_dir: str) -> Path:
    fs = int(cfg["sample_rate"])
    c = float(cfg["speed_of_sound"])
    pts = build_array(cfg["array"])
    dist = pairwise_distances(pts)
    acfg = cfg["augment"]
    n_per = int(cfg["output"]["num_per_rir"])
    rng = _rng(cfg)

    in_paths = sorted(Path(input_dir).glob("*.wav"))
    if not in_paths:
        raise FileNotFoundError(f"{input_dir} 下没有 .wav")
    out_root = Path(output_dir)
    manifest = out_root / cfg["output"]["manifest"]
    out_root.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(manifest, "w", encoding="utf-8") as mf:
        for src in in_paths:
            rir, src_fs = load_rir(src)
            if src_fs != fs:
                raise ValueError(f"{src.name} 采样率 {src_fs} != 配置 {fs}")
            for k in range(n_per):
                aug, params = augment_rir(rir, fs, c, dist, acfg, rng)
                name = f"{src.stem}_aug{k:03d}.wav"
                save_rir(out_root / name, aug, fs)
                rec = {"file": name, "source": src.name, **params}
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
    print(f"[augment] 由 {len(in_paths)} 条实测 RIR 生成 {count} 条 -> {out_root}")
    return manifest


def run_synth(cfg: dict[str, Any], output_dir: str) -> Path:
    fs = int(cfg["sample_rate"])
    c = float(cfg["speed_of_sound"])
    pts = build_array(cfg["array"])
    acfg = cfg["augment"]
    scfg = cfg["synth"]
    n = int(cfg["output"]["num"])
    rng = _rng(cfg, salt=101)

    out_root = Path(output_dir)
    manifest = out_root / cfg["output"]["manifest"]
    out_root.mkdir(parents=True, exist_ok=True)

    with open(manifest, "w", encoding="utf-8") as mf:
        for i in range(n):
            rir, params = synth_rir(pts, fs, c, acfg, scfg, rng)
            name = f"synth_{i:04d}.wav"
            save_rir(out_root / name, rir, fs)
            mf.write(json.dumps({"file": name, **params}, ensure_ascii=False) + "\n")
    print(f"[synth] 合成 {n} 条 RIR -> {out_root}")
    return manifest


def run_from_config(config_path: str, mode: str, input_dir: str | None, output_dir: str) -> Path:
    cfg = load_config(config_path)
    if mode == "augment":
        if not input_dir:
            raise ValueError("augment 模式需要 --input 指向实测 RIR 目录")
        return run_augment(cfg, input_dir, output_dir)
    if mode == "synth":
        return run_synth(cfg, output_dir)
    raise ValueError(f"未知模式: {mode}")
