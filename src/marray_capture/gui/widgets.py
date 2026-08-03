"""通用控件与线程工具。"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QFrame, QGraphicsItem, QGroupBox, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
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
        # 每个 bar 当前 chunk 颜色。setStyleSheet 会触发整条 QSS 重新解析+重绘,
        # 电平表在音频回调下高频更新, 每帧都设会把 GUI 线程的 GIL 占死,
        # 卡住 PortAudio 回调 → 输出 underrun (扫频卡顿)。只在跨档时才设一次。
        self._bar_color: list[str] = []
        # 分离双流模式下电平回调每个块都 emit (~上百 Hz), 不节流会撑死 GUI 线程
        # (setText+setValue × 每通道, 拖不动窗口)。这里把刷新限到 ~25 Hz,
        # 对电平表足够顺滑, 全双工模式本来就已经在回调里节流过, 不受影响。
        self._last_t: float = 0.0

    def set_channels(self, labels: list[str]) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._bars, self._labels, self._peak_hold, self._bar_color = [], [], [], []
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
            self._bar_color.append("")

    def update_levels(self, rms: np.ndarray) -> None:
        now = time.monotonic()
        dt = now - self._last_t
        if dt < 0.04:                    # 节流: 丢弃 40 ms 内的帧
            return
        self._last_t = now
        # 峰值保持按固定 dB/s 衰减回当前电平。注意必须用「减」而不是 ×0.995:
        # hold 是负 dB, 乘 0.995 会让它更接近 0 (更大), 渐近爬到 -4 dB ——
        # 表现为右数随时间不断变大, 同样音量第二次扫频时颜色从绿变黄。
        decay = 12.0 * min(dt, 0.25)     # 12 dB/s, 长间隔封顶防一次掉太多
        rms = np.atleast_1d(np.asarray(rms, dtype=float))
        for i, bar in enumerate(self._bars):
            if i >= len(rms):
                break
            db = 20.0 * np.log10(max(float(rms[i]), 1e-12))
            self._peak_hold[i] = max(self._peak_hold[i] - decay, db)
            bar.setValue(int(np.clip(db, -72, 0)))
            # 峰值保持决定颜色: 逼近削顶报警, 太小也要看得出来
            hold = self._peak_hold[i]
            color = theme.BAD if hold > -3 else (theme.WARN if hold > -12 else theme.OK)
            # 同档之下只改 value; 只有跨档才 setStyleSheet, 避免高频重绘卡住音频回调。
            if self._bar_color[i] != color:
                self._bar_color[i] = color
                bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
            self._labels[i].setText(f"{db:6.1f} / {hold:6.1f}")

    def reset_peaks(self) -> None:
        self._peak_hold = [-99.0] * len(self._peak_hold)
        self._bar_color = [""] * len(self._bar_color)


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
        # 鼠标平移/缩放/右键菜单全关 —— 这套界面是三米外看的采集仪器,
        # 不需要在图上拖拽探查; 关掉既杜绝交互卡顿, 也免得误操作把视图拖飞。
        # 需要细看时去会话目录拿原始 IR 文件。
        # (降采样/clipToView 不在这里设: PlotItem 级的 setDownsampling 只作用于
        # 已存在的曲线, 而 __init__ 时还没曲线; 必须在 show_ir 里逐条曲线设。)
        for p in (self.time_plot, self.freq_plot):
            p.plotItem.setMouseEnabled(x=False, y=False)
            p.plotItem.setMenuEnabled(False)
            # 坐标轴 (含 grid) 缓存成设备坐标 pixmap: 视图范围固定 (鼠标已关、
            # Y 轴固定、X 轴 show_ir 时一次性 fit), 滚轮滚页面时 QGraphicsView
            # 每帧重绘, 不缓存的话 grid 每帧重画 ~30 ms → 卡; 缓存后只 blit。
            for ax in p.plotItem.axes.values():
                ax["item"].setCacheMode(QGraphicsItem.DeviceCoordinateCache)
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
        self.time_plot.addLegend(offset=(-10, 10))
        for c in range(ch):
            name = labels[c] if labels and c < len(labels) else f"ch{c + 1}"
            pen = pg.mkPen(theme.PLOT_SERIES[c % len(theme.PLOT_SERIES)], width=1.2)
            # 绝对 dB, 不做峰值归一: 逆滤波器已把自卷积峰归一到 1, 所以 IR 的
            # 0 dB = 全刻度直通, 真实录制远低于此 (麦+音箱+距离都是衰减)。
            # 之前除以 peak 会让任何 IR 的峰都画在 0 dB, 看起来像削顶,
            # 与频响图 (本就是绝对值) 的 -50 dB 自相矛盾。
            env = 20.0 * np.log10(np.abs(x[:, c]) + 1e-12)
            tc = self.time_plot.plot(t, env, pen=pen, name=name)

            m = int(early_ms * 1e-3 * fs)
            seg = x[:m, c] * np.hanning(min(m, n))[:min(m, n)]
            spec = 20.0 * np.log10(np.abs(np.fft.rfft(seg, n=8192)) + 1e-12)
            freqs = np.fft.rfftfreq(8192, 1.0 / fs)
            ok = freqs > 20
            # 逐条曲线开降采样 + clipToView + 关抗锯齿: IR 1 s = 4.8 万点/通道,
            # 滚轮滚页面时 QScrollArea 重绘可见的图, 不降采样 + 抗锯齿每帧画
            # 几万个反锯齿线段 → 卡死。peak 法保住窄直达峰不被均值抹掉;
            # 关抗锯齿: 点这么密的包络看不出锯齿差别, 换来几十倍提速。useCache
            # 已开, 首帧渲染成 pixmap 后滚轮只是位图拷贝。PlotItem 级的
            # setDownsampling 不透传给之后才加的曲线, 必须拿到 PlotDataItem 逐条设。
            for curve in (tc, self.freq_plot.plot(freqs[ok], spec[ok], pen=pen)):
                curve.opts["antialias"] = False      # 见上方注释: 关抗锯齿换速度
                curve.setDownsampling(auto=True, method="peak")
                curve.setClipToView(True)
                # 曲线也缓存成 pixmap: 数据/视图范围不变时滚轮重绘只 blit,
                # 不再每帧光栅化几千条线段。换 IR 时 updateItems 会让缓存失效重画。
                curve.curve.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        # 固定 0 dB 为顶 (全刻度), 峰就不会贴着顶被误读成削顶; -120 盖住噪底。
        self.time_plot.setYRange(-120, 0)


def notes_html(items: list, max_width: int = 0) -> str:
    """把注意事项渲染成富文本。tooltip 和小窗共用同一段渲染。"""
    if not items:
        return "<i>这一步没有特别要注意的。</i>"
    rows = []
    for n in items:
        color, tint = theme.NOTE_LEVEL.get(n.level, (theme.INK_MUTED, theme.PAPER))
        chip = (f"<span style='background:{tint}; color:{color}; "
                f"padding:1px 5px; border-radius:3px; font-weight:600;'>"
                f"{theme.NOTE_LABEL.get(n.level, '')}</span>")
        rows.append(
            f"<tr><td valign='top' style='padding:0 8px 8px 0; white-space:nowrap;'>{chip}</td>"
            f"<td valign='top' style='padding:0 0 8px 0; line-height:155%;'>{n.text}</td></tr>")
    width = f" width='{max_width}'" if max_width else ""
    return f"<table cellspacing='0' cellpadding='0'{width}>{''.join(rows)}</table>"


class InfoDialog(QDialog):
    """一个通用的只读小窗: 标题 + 可滚动富文本。注意事项和参数说明都用它。"""

    def __init__(self, title: str, html: str, parent=None, size=(660, 560)):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)                     # 非模态: 可以边看边改
        self.resize(*size)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        self.area.setFrameShape(QFrame.NoFrame)
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.RichText)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.label.setAlignment(Qt.AlignTop)
        self.label.setContentsMargins(16, 14, 16, 16)
        self.area.setWidget(self.label)
        v.addWidget(self.area)
        self.set_html(html)

    def set_html(self, html: str) -> None:
        self.label.setText(html)


class NotesButton(QPushButton):
    """页面角上的「注意事项」按钮: 悬停看摘要, 点开看小窗。

    之前是常驻的卡片, 但在 1440p 笔记本上它会把下面的表单挤出屏幕 ——
    注意事项是**偶尔查**的东西, 不该长期占版面。
    """

    def __init__(self, page: str, settings, parent=None):
        super().__init__(parent)
        self.page = page
        self.settings = settings
        self._notes: list = []
        self._dialog: InfoDialog | None = None
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(self.open_dialog)
        self.refresh()

    def refresh(self) -> None:
        from . import notes as notes_mod

        self._notes = notes_mod.notes_for(self.page, self.settings)
        n_crit = sum(1 for n in self._notes if n.level == theme.NOTE_CRITICAL)
        mark = "⚠ " if n_crit else ""
        extra = f" · {n_crit} 必看" if n_crit else ""
        self.setText(f"{mark}注意事项 {len(self._notes)}{extra}")
        color = theme.BAD if n_crit else theme.INK_MUTED
        tint = theme.BAD_TINT if n_crit else theme.NOTE_TINT
        self.setStyleSheet(
            f"QPushButton {{ background: {tint}; color: {color};"
            f" border: 1px solid {theme.NOTE_LINE}; border-radius: 5px;"
            f" padding: 4px 12px; font-size: {theme.FS_SMALL}pt; font-weight: 600; }}"
            f"QPushButton:hover {{ border-color: {color}; }}")
        self.setToolTip(notes_html(self._notes, max_width=560))
        if self._dialog is not None:
            self._dialog.set_html(notes_html(self._notes))

    def open_dialog(self) -> None:
        if self._dialog is None:
            self._dialog = InfoDialog(f"注意事项 · {self.page}", notes_html(self._notes), self)
        self._dialog.set_html(notes_html(self._notes))
        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()


def scrollable(widget: QWidget) -> QScrollArea:
    """给页面套一层滚动区 —— 1440p 笔记本上窗口放不下整页内容。"""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setWidget(widget)
    return area


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
