"""配置加载与参数采样。

参数可写成三种形式, 统一由 ``sample_value`` 解析:
- 标量:            ``0.5``            -> 固定值
- 两元列表:        ``[0.2, 0.8]``     -> 线性区间, 均匀采样
- 字典:            ``{range: [0.2, 0.8], sampling: log}``  -> 指定 log/linear
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_value(spec: Any, rng: np.random.Generator) -> float:
    """把参数规格解析成一个采样值 (每次调用重新采样区间)。"""
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, (list, tuple)):
        lo, hi = float(spec[0]), float(spec[1])
        return float(rng.uniform(lo, hi))
    if isinstance(spec, dict):
        lo, hi = spec["range"]
        sampling = spec.get("sampling", "linear")
        lo, hi = float(lo), float(hi)
        if sampling == "log":
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        return float(rng.uniform(lo, hi))
    raise ValueError(f"无法解析的参数规格: {spec!r}")


def range_of(spec: Any) -> tuple[float, float]:
    """返回参数规格的 (min, max), 用于日志/校验。"""
    if isinstance(spec, (int, float)):
        return float(spec), float(spec)
    if isinstance(spec, (list, tuple)):
        return float(spec[0]), float(spec[1])
    if isinstance(spec, dict):
        lo, hi = spec["range"]
        return float(lo), float(hi)
    raise ValueError(f"无法解析的参数规格: {spec!r}")
