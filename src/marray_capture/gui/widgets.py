"""通用控件与线程工具。"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget,
)

pg.setConfigOptions(antialias=True, background="w", foreground="k")

VERDICT_COLOR = {"PASS": "#1a7f37", "WARN": "#bf8700", "FAIL": "#cf222e"}


class Worker(QThread):
    """在后台跑一个函数, 结果/异常通过信号回主线程。"""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()
        self._fn, self._args, self._kwargs = fn, args, kwargs

    def run(self) -> None:
        try:
            self.done.emit(self._fn(*self._args, **self._kwargs))
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class RunnerBridge(QObject):
    """把 SessionRunner 的回调转成 Qt 信号 (跨线程安全)。"""

    step = Signal(object, int, int)
    state = Signal(str)
    level = Signal(object)
    take = Signal(object, object, object)
    log = Signal(str)
    finished = Signal(str)
    progress = Signal(int, int)


class LevelMeter(QWidget):
    """多通道电平表, 单位 dBFS。红色表示逼近削顶。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._bars: list[QProgressBar] = []
        self._labels: list[QLabel] = []
        self._peak_hold: list[float] = []

    def set_channels(self, labels: list[str]) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._bars, self._labels, self._peak_hold = [], [], []
        for lb in labels:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            name = QLabel(lb)
            name.setMinimumWidth(70)
            bar = QProgressBar()
            bar.setRange(-72, 0)
            bar.setValue(-72)
            bar.setTextVisible(False)
            bar.setFixedHeight(14)
            val = QLabel("--")
            val.setMinimumWidth(88)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            h.addWidget(name)
            h.addWidget(bar, 1)
            h.addWidget(val)
            self._layout.addWidget(row)
            self._bars.append(bar)
            self._labels.append(val)
            self._peak_hold.append(-99.0)

    def update_levels(self, rms: np.ndarray) -> None:
        rms = np.atleast_1d(np.asarray(rms, dtype=float))
        for i, bar in enumerate(self._bars):
            if i >= len(rms):
                break
            db = 20.0 * np.log10(max(float(rms[i]), 1e-12))
            self._peak_hold[i] = max(self._peak_hold[i] * 0.995 - 0.02, db)
            bar.setValue(int(np.clip(db, -72, 0)))
            color = "#cf222e" if self._peak_hold[i] > -3 else (
                "#bf8700" if self._peak_hold[i] > -12 else "#1a7f37")
            bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
            self._labels[i].setText(f"{db:6.1f} / 峰 {self._peak_hold[i]:5.1f}")

    def reset_peaks(self) -> None:
        self._peak_hold = [-99.0] * len(self._peak_hold)


class IRView(QWidget):
    """IR 时域包络 + 频响。用来肉眼复核某个位置录得对不对。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.time_plot = pg.PlotWidget(title="IR 包络 (dB)")
        self.time_plot.setLabel("bottom", "时间", units="ms")
        self.time_plot.setLabel("left", "幅度", units="dB")
        self.time_plot.showGrid(x=True, y=True, alpha=0.3)
        self.freq_plot = pg.PlotWidget(title="幅频响应 (早期 50ms)")
        self.freq_plot.setLabel("bottom", "频率", units="Hz")
        self.freq_plot.setLabel("left", "幅度", units="dB")
        self.freq_plot.setLogMode(x=True, y=False)
        self.freq_plot.showGrid(x=True, y=True, alpha=0.3)
        v.addWidget(self.time_plot, 1)
        v.addWidget(self.freq_plot, 1)

    def show_ir(self, ir: np.ndarray, fs: int, labels: list[str] | None = None,
                early_ms: float = 50.0) -> None:
        self.time_plot.clear()
        self.freq_plot.clear()
        if ir is None or not len(ir):
            return
        x = np.atleast_2d(np.asarray(ir, dtype=float))
        if x.shape[0] < x.shape[1]:
            x = x.T
        n, ch = x.shape
        t = np.arange(n) / fs * 1000.0
        peak = float(np.max(np.abs(x))) or 1.0
        self.time_plot.addLegend(offset=(-10, 10))
        for c in range(ch):
            name = labels[c] if labels and c < len(labels) else f"ch{c + 1}"
            pen = pg.mkPen(pg.intColor(c, hues=max(3, ch)), width=1)
            env = 20.0 * np.log10(np.abs(x[:, c]) / peak + 1e-6)
            self.time_plot.plot(t, env, pen=pen, name=name)

            m = int(early_ms * 1e-3 * fs)
            seg = x[:m, c] * np.hanning(min(m, n))[:min(m, n)]
            spec = 20.0 * np.log10(np.abs(np.fft.rfft(seg, n=8192)) + 1e-12)
            freqs = np.fft.rfftfreq(8192, 1.0 / fs)
            ok = freqs > 20
            self.freq_plot.plot(freqs[ok], spec[ok], pen=pen)
        self.time_plot.setYRange(-100, 5)


def group(title: str, inner: QWidget) -> QGroupBox:
    box = QGroupBox(title)
    lay = QVBoxLayout(box)
    lay.setContentsMargins(8, 8, 8, 8)
    lay.addWidget(inner)
    return box


def big_label(text: str = "") -> QLabel:
    lb = QLabel(text)
    f = QFont()
    f.setPointSize(20)
    f.setBold(True)
    lb.setFont(f)
    lb.setWordWrap(True)
    lb.setAlignment(Qt.AlignCenter)
    lb.setMinimumHeight(90)
    return lb
