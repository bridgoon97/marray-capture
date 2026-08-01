"""通用控件与线程工具。"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame, QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QVBoxLayout, QWidget,
)

from . import theme

VERDICT_COLOR = {k: v[0] for k, v in theme.VERDICT.items()}
VERDICT_GLYPH = {k: v[2] for k, v in theme.VERDICT.items()}


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
        self._layout.setSpacing(3)
        for lb in labels:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)
            name = QLabel(lb)
            name.setMinimumWidth(64)
            name.setObjectName("readout")
            bar = QProgressBar()
            bar.setRange(-72, 0)
            bar.setValue(-72)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            val = QLabel("  --.-  /  --.-")
            val.setObjectName("readout")
            val.setMinimumWidth(112)
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
            # 峰值保持决定颜色: 逼近削顶报警, 太小也要看得出来
            hold = self._peak_hold[i]
            color = theme.BAD if hold > -3 else (theme.WARN if hold > -12 else theme.OK)
            bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
            self._labels[i].setText(f"{db:6.1f} / {hold:6.1f}")

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
            pen = pg.mkPen(theme.PLOT_SERIES[c % len(theme.PLOT_SERIES)], width=1.2)
            env = 20.0 * np.log10(np.abs(x[:, c]) / peak + 1e-6)
            self.time_plot.plot(t, env, pen=pen, name=name)

            m = int(early_ms * 1e-3 * fs)
            seg = x[:m, c] * np.hanning(min(m, n))[:min(m, n)]
            spec = 20.0 * np.log10(np.abs(np.fft.rfft(seg, n=8192)) + 1e-12)
            freqs = np.fft.rfftfreq(8192, 1.0 / fs)
            ok = freqs > 20
            self.freq_plot.plot(freqs[ok], spec[ok], pen=pen)
        self.time_plot.setYRange(-100, 5)


class RoleToggle(QPushButton):
    """通道用途的 toggle：点一下换一个（麦克风 → VPU → 参考麦）。

    比下拉框快 —— 8 通道的卡逐路点下拉太慢，而实际上只有这三种用途。
    """

    changed = Signal(str)

    def __init__(self, role: str = "mic", parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("点击切换：麦克风 / VPU / 参考麦")
        self._role = role if role in theme.ROLE_ORDER else "mic"
        self._render()
        self.clicked.connect(self._cycle)

    def role(self) -> str:
        return self._role

    def set_role(self, role: str) -> None:
        if role in theme.ROLE_ORDER and role != self._role:
            self._role = role
            self._render()

    def _cycle(self) -> None:
        i = theme.ROLE_ORDER.index(self._role)
        self._role = theme.ROLE_ORDER[(i + 1) % len(theme.ROLE_ORDER)]
        self._render()
        self.changed.emit(self._role)

    def _render(self) -> None:
        label, color, tint = theme.ROLE_STYLE[self._role]
        self.setText(label)
        self.setMinimumWidth(76)
        self.setStyleSheet(
            f"QPushButton {{ background: {tint}; color: {color};"
            f" border: 1px solid {color}; border-radius: 4px;"
            f" padding: 3px 10px; font-weight: 600; }}"
            f"QPushButton:disabled {{ background: {theme.PAPER}; color: {theme.INK_FAINT};"
            f" border-color: {theme.LINE}; }}")


def group(title: str, inner: QWidget) -> QGroupBox:
    box = QGroupBox(title)
    lay = QVBoxLayout(box)
    lay.setContentsMargins(8, 8, 8, 8)
    lay.addWidget(inner)
    return box


class InstructionCard(QWidget):
    """采集页的主角: 一张大字指令卡, 左侧一道判定色条。

    这是整套界面唯一用饱和色的地方。色条永远配一个字符 (✓ ! ✕), 不靠颜色单打独斗,
    所以隔着三米、或者色觉障碍下都读得出来。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        self.rail = QFrame()
        self.rail.setFixedWidth(5)
        self.rail.setStyleSheet(f"background: {theme.LINE_STRONG}; border: none;")
        row.addWidget(self.rail)

        body = QWidget()
        body.setObjectName("card")
        v = QVBoxLayout(body)
        v.setContentsMargins(22, 16, 22, 18)
        v.setSpacing(8)

        self.eyebrow = QLabel("待开始")
        self.eyebrow.setObjectName("eyebrow")

        # 字号由 QSS 的 QLabel#display 给 —— QSS 优先级高于 setFont()
        self.text = QLabel("按「开始采集」")
        self.text.setObjectName("display")
        self.text.setWordWrap(True)
        self.text.setMinimumHeight(150)
        self.text.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.status = QLabel("就绪")
        self.status.setObjectName("hint")

        v.addWidget(self.eyebrow)
        v.addWidget(self.text, 1)
        v.addWidget(self.status)
        row.addWidget(body, 1)

        self._paint(theme.LINE_STRONG, theme.SURFACE)

    def _paint(self, rail_color: str, bg: str) -> None:
        self.rail.setStyleSheet(f"background: {rail_color}; border: none;")
        self.setStyleSheet(
            f"QWidget#card {{ background: {bg}; border: 1px solid {theme.LINE};"
            f" border-left: none; border-radius: 0 6px 6px 0; }}")

    def show_step(self, eyebrow: str, text: str) -> None:
        self.eyebrow.setText(eyebrow)
        self.text.setText(text)

    def show_status(self, text: str) -> None:
        self.status.setText(text)

    def show_verdict(self, verdict: str) -> None:
        color, tint, glyph = theme.VERDICT.get(
            verdict, (theme.LINE_STRONG, theme.SURFACE, ""))
        self._paint(color, tint)
        if glyph:
            self.status.setText(f"{glyph}  上一条 {verdict}")

    def reset_verdict(self) -> None:
        self._paint(theme.LINE_STRONG, theme.SURFACE)


def verdict_cell(verdict: str):
    """质检表里的判定单元格: 字符 + 判定色的浅底。"""
    from PySide6.QtWidgets import QTableWidgetItem

    color, tint, glyph = theme.VERDICT.get(verdict, ("", "", ""))
    item = QTableWidgetItem(f"{glyph} {verdict}" if glyph else verdict)
    if color:
        item.setForeground(QColor(color))
        item.setBackground(QColor(tint))
    return item
