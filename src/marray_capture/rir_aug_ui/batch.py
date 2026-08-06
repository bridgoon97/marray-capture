"""裸 wav 目录版的批驱动 (包着 augment_runner.run_augment)。

与 augment_runner.run_augment 的唯一区别: 输入不是会话 manifest 里筛出来的 (路径, mic_cols),
而是一个目录下的裸 wav。自说话阵列 IR 没有 VPU 这类非声学通道, **所有通道都当麦克风**,
几何校验/manifest/global_norm 全部复用 run_augment —— 不在这里重写一遍, 免得两份逻辑漂移。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import soundfile as sf

from ..rir.augment_runner import ProgressCb, run_augment


def list_inputs(input_dir: str | Path) -> list[Path]:
    """目录下所有 .wav, 按文件名排序。空目录直接报错。"""
    paths = sorted(Path(input_dir).glob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"{input_dir} 下没有 .wav 文件")
    return paths


def channel_counts(paths: list[Path]) -> dict[int, int]:
    """{通道数: 该通道数的文件数}。UI 预检用 —— 一批文件通道数不一致或与几何对不上时,
    在跑之前就提示, 而不是跑到一半被 run_augment 逐个报错。"""
    counts: dict[int, int] = {}
    for p in paths:
        info = sf.info(str(p))      # 只读头, 不读数据
        counts[info.channels] = counts.get(info.channels, 0) + 1
    return counts


def build_inputs(paths: list[Path]) -> list[tuple[Path, list[int]]]:
    """每个 wav 映射成 (路径, [0..n_ch-1])。所有通道都当麦克风通道。

    mic_cols 在主工具里来自 manifest, 这里没有 manifest, 就取全部通道。下游 run_augment
    会拿这个去和阵列几何的 n_geom 比对, 不一致会报它那条 helpful 错误。
    """
    out: list[tuple[Path, list[int]]] = []
    for p in paths:
        info = sf.info(str(p))
        out.append((p, list(range(info.channels))))
    return out


def run_augment_dir(
    input_dir: str | Path,
    output_dir: str | Path,
    cfg: dict,
    progress: ProgressCb | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """对 input_dir 下每条 wav 生成 num_per_rir 条增强版本。返回 manifest 路径。"""
    inputs = build_inputs(list_inputs(input_dir))
    # eq_fir=None: 自说话 IR 不做去音箱响应这一步。
    return run_augment(inputs, output_dir, cfg, None, progress, should_stop)
