"""独立增强批驱动的端到端自检。

重点抓「短 IR」这个静默坑: 这批自说话 IR 只有几百样本 (~10-30 ms), 默认早/晚分界
50 ms 比整条还长。要确认 split_early_late 截到 IR 长度内、尾真的拼上去了, 而不是
静默吞掉短输入。
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

from marray_capture.rir.augment_runner import default_config
from marray_capture.rir_aug_ui.batch import (
    build_inputs, channel_counts, list_inputs, run_augment_dir,
)


def _write_short_ir(path, ir: np.ndarray, fs: int = 16000) -> None:
    """ir: (L, C)。"""
    sf.write(str(path), ir.astype(np.float32), fs, subtype="FLOAT")


def _make_short_ir(n_ch: int = 3, length: int = 300, seed: int = 0) -> np.ndarray:
    """一条带直达 + 几个早反射 + 指数尾的短多通道 IR, (L, C)。"""
    rng = np.random.default_rng(seed)
    ir = np.zeros((length, n_ch))
    for c in range(n_ch):
        ir[20 + c, c] = 1.0 - 0.1 * c
        for _ in range(4):
            k = int(rng.integers(30, 250))
            ir[k, c] += rng.normal(0, 0.15)
        t = np.arange(length) / 16000
        ir[:, c] += rng.normal(0, 0.02, length) * np.exp(-t / 0.05)
    return ir


def test_list_inputs_and_channel_counts(tmp_path):
    _write_short_ir(tmp_path / "a.wav", _make_short_ir(3))
    _write_short_ir(tmp_path / "b.wav", _make_short_ir(3))
    _write_short_ir(tmp_path / "c.wav", _make_short_ir(3))
    # 非 wav 不该进列表
    (tmp_path / "notes.txt").write_text("hi")

    paths = list_inputs(tmp_path)
    assert [p.name for p in paths] == ["a.wav", "b.wav", "c.wav"]
    assert channel_counts(paths) == {3: 3}

    inputs = build_inputs(paths)
    assert all(cols == [0, 1, 2] for _, cols in inputs)


def test_list_inputs_empty(tmp_path):
    try:
        list_inputs(tmp_path)
    except FileNotFoundError:
        return
    raise AssertionError("空目录应该报 FileNotFoundError")


def test_run_augment_dir_end_to_end(tmp_path):
    """2 条 3 通道 300 样本 @16k 的短 IR → 每条 3 条增强。"""
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    a, b = src_dir / "a.wav", src_dir / "b.wav"
    ir_a = _make_short_ir(3, 300, seed=1)
    ir_b = _make_short_ir(3, 300, seed=2)
    _write_short_ir(a, ir_a)
    _write_short_ir(b, ir_b)
    in_len = ir_a.shape[0]

    out_dir = tmp_path / "out"
    cfg = default_config()
    cfg["sample_rate"] = 16000
    cfg["array"] = {"coords": None,
                    "generator": {"type": "equilateral_triangle", "edge": 0.01}}
    # 早/晚分界压到 20 ms, 匹配短 IR
    cfg["augment"]["early_late_split_ms"] = 20
    cfg["output"]["num_per_rir"] = 3

    manifest = run_augment_dir(src_dir, out_dir, cfg)

    outs = sorted(out_dir.glob("*.wav"))
    assert len(outs) == 2 * 3, f"应有 6 条输出, 实际 {len(outs)}"
    # 每条输出长度 > 输入: 证明合成尾拼上去了 (短 IR 的早期段 + 尾)
    for o in outs:
        data, fs = sf.read(str(o), always_2d=True)
        assert fs == 16000
        assert data.shape[0] > in_len, f"{o.name} 长度 {data.shape[0]} 未超过输入 {in_len}"
        assert data.shape[1] == 3, f"{o.name} 应保持 3 通道"

    lines = manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6
    # 每条记录都带了采样到的参数 (t60/drr/...), 说明走的是真实 augment_rir
    import json
    for ln in lines:
        rec = json.loads(ln)
        assert "t60" in rec and "drr_db" in rec and rec.get("source") in ("a.wav", "b.wav")
