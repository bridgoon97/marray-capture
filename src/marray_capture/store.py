"""会话目录与清单。

目录布局::

    <session_root>/<session_name>/
        session.json      配置快照 + 设备信息
        plan.json         采集方案
        manifest.jsonl    每个位置一行: 元数据 + QC 指标 + 文件名
        qc.csv            质检汇总 (给现场看的扁平表)
        raw/              原始多通道录音 (可关)
        ir48k/            提取出的 IR, 采集采样率
        ir16k/            重采样到训练管线采样率的 IR
        noise/            环境底噪 / 电平检查录音
        prompts/          TTS 缓存
        aug/              增强产物 (早期段实测 + 随机晚期尾)
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

SUBDIRS = ["raw", "ir48k", "ir16k", "noise", "prompts", "aug"]


class Session:
    def __init__(self, root: str | Path, name: str | None = None, create: bool = True):
        self.root = Path(root)
        self.name = name or datetime.now().strftime("session_%Y%m%d_%H%M")
        self.dir = self.root / self.name
        if create:
            for sub in SUBDIRS:
                (self.dir / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 路径
    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.jsonl"

    @property
    def plan_path(self) -> Path:
        return self.dir / "plan.json"

    @property
    def prompts_dir(self) -> Path:
        return self.dir / "prompts"

    def raw_path(self, take_id: str, fmt: str = "FLAC") -> Path:
        ext = "flac" if fmt.upper() == "FLAC" else "wav"
        return self.dir / "raw" / f"{take_id}.{ext}"

    def ir_path(self, take_id: str, fs: int) -> Path:
        sub = "ir16k" if fs <= 16000 else "ir48k"
        return self.dir / sub / f"{take_id}.wav"

    # ------------------------------------------------------------------ 写入
    def save_json(self, name: str, payload: dict[str, Any]) -> Path:
        p = self.dir / name
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return p

    def append_manifest(self, record: dict[str, Any]) -> None:
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    def load_manifest(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            return []
        rows = []
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def rewrite_manifest(self, rows: Iterable[dict[str, Any]]) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")

    # ------------------------------------------------------------------ 音频
    def save_raw(self, take_id: str, rec: np.ndarray, fs: int, fmt: str = "FLAC") -> Path:
        p = self.raw_path(take_id, fmt)
        if fmt.upper() == "FLAC":
            data = np.clip(rec, -1.0, 1.0)
            sf.write(str(p), data, fs, subtype="PCM_24", format="FLAC")
        else:
            sf.write(str(p), rec, fs, subtype="FLOAT")
        return p

    def save_ir(self, take_id: str, ir: np.ndarray, fs: int) -> Path:
        """IR 存 32-bit float 且**不做峰值归一** —— 通道间与位置间的绝对电平关系
        是后面算 ILD / DRR 的依据, 归一化会毁掉它。"""
        p = self.ir_path(take_id, fs)
        sf.write(str(p), np.asarray(ir, dtype=np.float32), fs, subtype="FLOAT")
        return p

    # ------------------------------------------------------------------ 汇总
    def write_qc_csv(self) -> Path:
        rows = self.load_manifest()
        p = self.dir / "qc.csv"
        if not rows:
            p.write_text("", encoding="utf-8")
            return p
        fields = [
            "take_id", "tag", "subject_id", "wearing_id", "side",
            "distance_cm", "height_cm", "speaker_deg", "az_index", "az_nominal_deg", "az_label",
            "stance", "verdict", "reasons", "repeat_ncc", "repeat_level_diff_db",
            "drift_ppm", "latency_ms", "smear_ms", "n_averaged",
            "min_rec_snr_db", "min_ir_ddr_db", "min_ir_peak_db", "max_ir_noise_db",
            "min_reliable_bw_hz", "max_peak_dbfs",
            "attempt", "timestamp",
        ]
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                flat = dict(r)
                qc = r.get("qc") or {}
                chans = qc.get("channels") or []
                flat.update({
                    "verdict": qc.get("verdict", ""),
                    "reasons": " / ".join(qc.get("reasons", [])),
                    "repeat_ncc": _r(qc.get("repeat_ncc")),
                    "repeat_level_diff_db": _r(qc.get("repeat_level_diff_db")),
                    "drift_ppm": _r(qc.get("drift_ppm")),
                    "latency_ms": _r(qc.get("latency_ms")),
                    "smear_ms": _r(qc.get("smear_ms")),
                    "min_rec_snr_db": _r(_agg(chans, "rec_snr_db", min)),
                    "min_ir_ddr_db": _r(_agg(chans, "ir_ddr_db", min)),
                    "min_ir_peak_db": _r(_agg(chans, "ir_peak_db", min)),
                    "max_ir_noise_db": _r(_agg(chans, "ir_noise_db", max)),
                    "min_reliable_bw_hz": _r(_agg(chans, "reliable_bw_hz", min)),
                    "max_peak_dbfs": _r(_agg(chans, "peak_dbfs", max)),
                })
                w.writerow(flat)
        return p

    def stats(self) -> dict[str, int]:
        rows = self.load_manifest()
        out = {"total": len(rows), "PASS": 0, "WARN": 0, "FAIL": 0}
        for r in rows:
            v = (r.get("qc") or {}).get("verdict", "")
            if v in out:
                out[v] += 1
        return out


def list_sessions(root: str | Path) -> list[str]:
    p = Path(root)
    if not p.exists():
        return []
    return sorted(d.name for d in p.iterdir() if d.is_dir() and (d / "manifest.jsonl").exists())


def _agg(channels: list[dict], key: str, fn):
    vals = [c.get(key) for c in channels if isinstance(c.get(key), (int, float))]
    vals = [v for v in vals if v == v]
    return fn(vals) if vals else None


def _r(v, nd: int = 2):
    return round(v, nd) if isinstance(v, (int, float)) and v == v else ""


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
