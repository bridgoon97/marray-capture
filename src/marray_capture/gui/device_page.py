"""设备页: 选声卡、分配通道用途、做开录前的三项检查。

开录前必做的三件事都在这一页:
  1. **电平检查** —— 看有没有死麦、增益是否离谱、会不会削顶。
  2. **环境底噪** —— 记录本场房间的噪声水平, 后面质检的 SNR 才有参照。
  3. **延迟标定** —— 蓝牙音箱每次连接的延迟都不同; 标定一次可以收窄峰值搜索窗,
     顺便暴露"根本没出声"这类接线问题, 也给出时钟漂移的初值。
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from ..audio import devices as dev
from ..audio.prompts import tone
from ..audio.engine import AudioEngine
from ..protocol.runner import RunnerHooks, calibrate_latency, channel_layout, measure_ambient
from ..settings import CHANNEL_ROLES, AppSettings, ChannelMap
from .widgets import IRView, LevelMeter, RunnerBridge, Worker


class DevicePage(QWidget):
    def __init__(self, settings: AppSettings, get_session, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.get_session = get_session
        self._worker: Worker | None = None
        self._loading = False
        # 电平回调来自音频线程, 必须经信号跨回 GUI 线程
        self.bridge = RunnerBridge()
        self._build()
        self.bridge.level.connect(self.meter.update_levels)
        # 必须走 pull(): 否则控件停在 Qt 的默认值上, 随后 _push() 会把这些默认值
        # 写回 settings —— 输出通道数会变成 1、输出增益会变成 0 dB。
        self.pull()

    @property
    def _hooks(self) -> RunnerHooks:
        return RunnerHooks(on_level=self.bridge.level.emit)

    # ------------------------------------------------------------------ 布局
    def _build(self) -> None:
        root = QHBoxLayout(self)

        # ---- 左: 设备与参数
        left = QVBoxLayout()
        form = QFormLayout()
        self.cb_in = QComboBox()
        self.cb_out = QComboBox()
        self.cb_rate = QComboBox()
        self.sp_block = QSpinBox()
        self.sp_block.setRange(0, 8192)
        self.sp_block.setSingleStep(256)
        self.sp_out_ch = QSpinBox()
        self.sp_out_ch.setRange(1, 8)
        self.sp_gain = QDoubleSpinBox()
        self.sp_gain.setRange(-40.0, 6.0)
        self.sp_gain.setSuffix(" dB")
        self.cb_in_lat = QComboBox()
        self.cb_in_lat.addItems(["low", "high"])
        self.cb_out_lat = QComboBox()
        self.cb_out_lat.addItems(["high", "low"])
        self.cb_duplex = QComboBox()
        for key, label in [
            ("auto", "自动 (同设备走全双工)"),
            ("duplex", "强制全双工单流 (ASIO 必选)"),
            ("split", "强制分离双流 (跨设备)"),
        ]:
            self.cb_duplex.addItem(label, key)

        btn_refresh = QPushButton("刷新设备列表")
        btn_refresh.clicked.connect(self.rescan_devices)
        self.btn_probe = QPushButton("检测该设备真正支持的采样率 (会占用设备几秒)")
        self.btn_probe.clicked.connect(self.probe_rates)

        form.addRow("输入设备 (声卡)", self.cb_in)
        form.addRow("输出设备 (音箱)", self.cb_out)
        form.addRow("采样率", self.cb_rate)
        form.addRow("块大小 (0=自动)", self.sp_block)
        form.addRow("输出通道数", self.sp_out_ch)
        form.addRow("输出增益", self.sp_gain)
        form.addRow("收发模式", self.cb_duplex)
        form.addRow("输入延迟模式", self.cb_in_lat)
        form.addRow("输出延迟模式", self.cb_out_lat)
        form.addRow("", btn_refresh)
        form.addRow("", self.btn_probe)

        self.lb_mode = QLabel()
        self.lb_mode.setWordWrap(True)
        self.lb_mode.setObjectName("hint")
        form.addRow("", self.lb_mode)

        box_dev = QGroupBox("设备")
        box_dev.setLayout(form)
        left.addWidget(box_dev)

        # ---- 通道表
        self.tbl = QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(["物理通道", "用途", "标签"])
        self.tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        box_ch = QGroupBox("输入通道映射")
        lay_ch = QVBoxLayout(box_ch)

        # 8 通道的卡逐行点太慢, 给一个批量分配
        quick = QHBoxLayout()
        self.sp_nmic = QSpinBox()
        self.sp_nmic.setRange(1, 32)
        self.sp_nmic.setValue(3)
        self.ck_vpu = QCheckBox("其后一路作 VPU")
        self.ck_vpu.setChecked(True)
        btn_apply = QPushButton("套用")
        btn_clear = QPushButton("全部不用")
        btn_apply.clicked.connect(self.quick_assign)
        btn_clear.clicked.connect(self.clear_assign)
        quick.addWidget(QLabel("前"))
        quick.addWidget(self.sp_nmic)
        quick.addWidget(QLabel("路作麦克风"))
        quick.addWidget(self.ck_vpu)
        quick.addWidget(btn_apply)
        quick.addWidget(btn_clear)
        quick.addStretch(1)
        lay_ch.addLayout(quick)

        lay_ch.addWidget(self.tbl)
        self.lb_chsum = QLabel()
        self.lb_chsum.setWordWrap(True)
        self.lb_chsum.setObjectName("hint")
        lay_ch.addWidget(self.lb_chsum)
        left.addWidget(box_ch, 1)

        root.addLayout(left, 3)

        # ---- 右: 检查动作
        right = QVBoxLayout()
        self.meter = LevelMeter()
        right.addWidget(QLabel("<b>输入电平</b>"))
        right.addWidget(self.meter)

        grid = QGridLayout()
        self.btn_tone = QPushButton("① 播放测试音 (查音箱)")
        self.btn_level = QPushButton("② 电平检查 (录 10 秒)")
        self.btn_noise = QPushButton("③ 环境底噪 (录 30 秒)")
        self.btn_cal = QPushButton("④ 延迟标定 (播一次扫频)")
        for i, b in enumerate([self.btn_tone, self.btn_level, self.btn_noise, self.btn_cal]):
            grid.addWidget(b, i // 2, i % 2)
        self.btn_tone.clicked.connect(self.play_tone)
        self.btn_level.clicked.connect(lambda: self.record_check(10.0, "电平检查"))
        self.btn_noise.clicked.connect(lambda: self.record_check(30.0, "环境底噪"))
        self.btn_cal.clicked.connect(self.calibrate)
        right.addLayout(grid)

        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMinimumHeight(140)
        right.addWidget(self.txt, 1)

        self.ir_view = IRView()
        right.addWidget(self.ir_view, 2)
        root.addLayout(right, 4)

        # ---- 信号
        self.cb_in.currentIndexChanged.connect(self._on_input_changed)
        self.cb_out.currentIndexChanged.connect(self._push)
        self.cb_rate.currentIndexChanged.connect(self._push)
        self.sp_block.valueChanged.connect(self._push)
        self.sp_out_ch.valueChanged.connect(self._push)
        self.sp_gain.valueChanged.connect(self._push)
        self.cb_in_lat.currentIndexChanged.connect(self._push)
        self.cb_out_lat.currentIndexChanged.connect(self._push)
        self.cb_duplex.currentIndexChanged.connect(self._push)

    # ------------------------------------------------------------------ 设备
    def rescan_devices(self) -> None:
        """重新初始化 PortAudio 后再列一次 (热插拔的设备才会出现)。"""
        dev.rescan()
        self.refresh_devices()

    def probe_rates(self) -> None:
        """真正打开设备验证采样率。放工作线程 —— 虚拟声卡/蓝牙设备可能卡住几十秒。"""
        idx = self.cb_in.currentData()
        if idx is None:
            return
        self.btn_probe.setEnabled(False)
        self._log("检测输入设备支持的采样率…")
        self._worker = Worker(dev.probe_rates, idx, "input")

        def ok(rates: list[int]) -> None:
            self.btn_probe.setEnabled(True)
            if not rates:
                self._log("✗ 该设备没有响应任何常用采样率, 可能被别的程序占用。")
                return
            self._log("  支持: " + ", ".join(f"{r} Hz" for r in rates))
            cur = self.settings.audio.samplerate
            self.cb_rate.blockSignals(True)
            self.cb_rate.clear()
            for r in rates:
                self.cb_rate.addItem(f"{r} Hz", r)
            self._select(self.cb_rate, cur if cur in rates else rates[-1])
            self.cb_rate.blockSignals(False)
            self._push()

        self._worker.done.connect(ok)
        self._worker.failed.connect(lambda e: (self.btn_probe.setEnabled(True),
                                               self._log("✗ " + e.splitlines()[0])))
        self._worker.start()

    def refresh_devices(self) -> None:
        cur_in = self.settings.audio.input_device
        cur_out = self.settings.audio.output_device
        self.cb_in.blockSignals(True)
        self.cb_out.blockSignals(True)
        self.cb_in.clear()
        self.cb_out.clear()
        try:
            ins, outs = dev.input_devices(), dev.output_devices()
        except Exception as e:
            QMessageBox.critical(self, "枚举设备失败", str(e))
            ins, outs = [], []
        for d in ins:
            self.cb_in.addItem(d.label(), d.index)
        for d in outs:
            self.cb_out.addItem(d.label(), d.index)
        self._select(self.cb_in, cur_in)
        self._select(self.cb_out, cur_out)
        self.cb_in.blockSignals(False)
        self.cb_out.blockSignals(False)
        self._on_input_changed()

    @staticmethod
    def _select(cb: QComboBox, value) -> None:
        if value is None:
            return
        for i in range(cb.count()):
            if cb.itemData(i) == value:
                cb.setCurrentIndex(i)
                return

    def _on_input_changed(self) -> None:
        idx = self.cb_in.currentData()
        info = dev.describe(idx)
        rates = dev.supported_rates(idx)          # 只读描述, 不打开设备
        cur = self.settings.audio.samplerate
        self.cb_rate.blockSignals(True)
        self.cb_rate.clear()
        for r in rates:
            self.cb_rate.addItem(f"{r} Hz", r)
        self._select(self.cb_rate, cur if cur in rates else (48000 if 48000 in rates else rates[-1]))
        self.cb_rate.blockSignals(False)
        self._rebuild_channel_table(info.max_input if info else 0)
        self._push()

    def _rebuild_channel_table(self, n_ch: int) -> None:
        existing = {c.index: c for c in self.settings.audio.channels}
        self.tbl.blockSignals(True)
        self.tbl.setRowCount(n_ch)
        for i in range(n_ch):
            item = QTableWidgetItem(f"ch{i + 1}")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.tbl.setItem(i, 0, item)

            cb = QComboBox()
            cb.addItems(CHANNEL_ROLES)
            prev = existing.get(i)
            default_role = prev.role if prev else ("mic" if i < 3 else "ignore")
            cb.setCurrentText(default_role if default_role in CHANNEL_ROLES else "ignore")
            cb.currentIndexChanged.connect(self._push_channels)
            self.tbl.setCellWidget(i, 1, cb)

            le = QLineEdit(prev.label if prev else ("" if i >= 3 else f"mic{i + 1}"))
            le.setPlaceholderText(f"ch{i + 1}")
            le.editingFinished.connect(self._push_channels)
            self.tbl.setCellWidget(i, 2, le)
        self.tbl.blockSignals(False)
        self._push_channels()

    # ------------------------------------------------------------------ 同步
    def quick_assign(self) -> None:
        """前 N 路设为麦克风, 可选再把下一路设为 VPU, 其余不用。"""
        n = self.sp_nmic.value()
        for i in range(self.tbl.rowCount()):
            cb = self.tbl.cellWidget(i, 1)
            le = self.tbl.cellWidget(i, 2)
            if cb is None:
                continue
            if i < n:
                role, label = "mic", f"mic{i + 1}"
            elif self.ck_vpu.isChecked() and i == n:
                role, label = "vpu", "vpu"
            else:
                role, label = "ignore", ""
            cb.blockSignals(True)
            cb.setCurrentText(role)
            cb.blockSignals(False)
            if le is not None:
                le.setText(label)
        self._push_channels()

    def clear_assign(self) -> None:
        for i in range(self.tbl.rowCount()):
            cb = self.tbl.cellWidget(i, 1)
            if cb is not None:
                cb.blockSignals(True)
                cb.setCurrentText("ignore")
                cb.blockSignals(False)
        self._push_channels()

    def _push_channels(self) -> None:
        if getattr(self, "_loading", False):
            return
        chans = []
        for i in range(self.tbl.rowCount()):
            cb = self.tbl.cellWidget(i, 1)
            le = self.tbl.cellWidget(i, 2)
            if cb is None:
                continue
            chans.append(ChannelMap(index=i, role=cb.currentText(), label=le.text() if le else ""))
        self.settings.audio.channels = chans
        _, labels, mic_cols = channel_layout(self.settings)
        self.meter.set_channels(labels or ["(未选通道)"])

        a = self.settings.audio
        n_rec = a.n_record_channels()
        roles = {r: len(a.role_indices(r)) for r in ("mic", "vpu", "ref", "loopback")}
        bits = [f"{n} 路 {r}" for r, n in roles.items() if n]
        self.lb_chsum.setText(
            f"当前：{'、'.join(bits) or '未分配'}，实际录制前 {n_rec} 个物理通道后按此表切片。"
            f"通道顺序即 IR 文件里的顺序。"
            + (f"　⚠ 增强时阵列几何要配成 {len(mic_cols)} 个麦克风（后处理页）。"
               if len(mic_cols) not in (0, 3) else "")
        )

    def _push(self) -> None:
        if getattr(self, "_loading", False):
            return
        a = self.settings.audio
        a.input_device = self.cb_in.currentData()
        a.output_device = self.cb_out.currentData()
        if self.cb_rate.currentData():
            a.samplerate = int(self.cb_rate.currentData())
        a.blocksize = self.sp_block.value()
        a.output_channels = self.sp_out_ch.value()
        a.output_gain_db = self.sp_gain.value()
        a.input_latency = self.cb_in_lat.currentText()
        a.output_latency = self.cb_out_lat.currentText()
        a.duplex_mode = self.cb_duplex.currentData() or "auto"

        duplex = AudioEngine(a).use_duplex()
        asio = dev.is_asio(a.input_device) or dev.is_asio(a.output_device)
        mode = "全双工单流（收发采样锁定，无时钟漂移）" if duplex else "分离双流（跨设备，会估计时钟漂移）"
        extra = "　检测到 ASIO —— 输入输出必须是同一个 ASIO 条目。" if asio and not duplex else ""
        self.lb_mode.setText(f"当前：{mode}。{extra}")

    def pull(self) -> None:
        """从 settings 回填控件 (构造时和加载配置后调用)。

        回填期间必须挡住 _push —— 每个 setValue 都会触发 valueChanged, 而那时
        其余控件还停在 Qt 默认值上, _push 会把这些默认值写回 settings, 把还没
        回填的字段冲掉 (输出增益 -6 dB 会变成 0 dB)。
        """
        a = self.settings.audio
        self._loading = True
        try:
            self.sp_block.setValue(a.blocksize)
            self.sp_out_ch.setValue(a.output_channels)
            self.sp_gain.setValue(a.output_gain_db)
            self.cb_in_lat.setCurrentText(a.input_latency)
            self.cb_out_lat.setCurrentText(a.output_latency)
            i = self.cb_duplex.findData(a.duplex_mode)
            if i >= 0:
                self.cb_duplex.setCurrentIndex(i)
            self.refresh_devices()
        finally:
            self._loading = False
        self._push()

    # ------------------------------------------------------------------ 动作
    def _busy(self, on: bool) -> None:
        for b in (self.btn_tone, self.btn_level, self.btn_noise, self.btn_cal):
            b.setEnabled(not on)

    def _log(self, text: str) -> None:
        self.txt.append(text)

    def _check_ready(self) -> bool:
        a = self.settings.audio
        problems = dev.validate(
            a.input_device, a.output_device, a.samplerate,
            max(1, a.n_record_channels()), a.output_channels,
            duplex=AudioEngine(a).use_duplex(),
        )
        hard = [p for p in problems if not p.startswith("提示")]
        for p in problems:
            self._log(("⚠ " if p.startswith("提示") else "✗ ") + p)
        if hard:
            QMessageBox.warning(self, "设备检查", "\n".join(hard))
            return False
        return True

    def play_tone(self) -> None:
        if not self._check_ready():
            return
        fs = self.settings.audio.samplerate
        sig = np.concatenate([tone(fs, f, 0.35, 0.3) for f in (250, 500, 1000, 2000, 4000)])
        self._busy(True)
        self._worker = Worker(AudioEngine(self.settings.audio).play, sig)
        self._worker.done.connect(lambda _: (self._busy(False), self._log("测试音播放完毕。")))
        self._worker.failed.connect(lambda e: (self._busy(False), self._log("✗ " + e.splitlines()[0])))
        self._worker.start()

    def record_check(self, seconds: float, title: str) -> None:
        if not self._check_ready():
            return
        self.meter.reset_peaks()
        self._busy(True)
        self._log(f"—— {title}: 录 {seconds:.0f} 秒 ——")
        session = self.get_session() if title == "环境底噪" else None
        self._worker = Worker(measure_ambient, self.settings, seconds, session, self._hooks)

        def ok(res: dict) -> None:
            self._busy(False)
            for lb, db in zip(res["labels"], res["rms_dbfs"]):
                self._log(f"  {lb:<10} RMS {db:6.1f} dBFS")
            spread = max(res["rms_dbfs"]) - min(res["rms_dbfs"])
            if spread > 12:
                self._log(f"⚠ 通道间电平差 {spread:.1f} dB, 检查是否有死麦或增益不一致。")
            if res["file"]:
                self._log(f"  已保存: {res['file']}")

        self._worker.done.connect(ok)
        self._worker.failed.connect(lambda e: (self._busy(False), self._log("✗ " + e.splitlines()[0])))
        self._worker.start()

    def calibrate(self) -> None:
        if not self._check_ready():
            return
        self.meter.reset_peaks()
        self._busy(True)
        self._log("—— 延迟标定: 播一次扫频, 请保持安静 ——")
        self._worker = Worker(calibrate_latency, self.settings, self._hooks)

        def ok(res: dict) -> None:
            self._busy(False)
            a = self.settings.audio
            a.measured_latency_samples = int(res["latency_samples"])
            a.drift_ppm = float(res["drift_ppm"])
            self._log(f"  往返延迟 {res['latency_ms']:.1f} ms ({res['latency_samples']} 样本)")
            self._log(f"  时钟漂移 {res['drift_ppm']:+.1f} ppm")
            if abs(res["drift_ppm"]) > 200:
                self._log("⚠ 漂移偏大。蓝牙链路在重同步的话 IR 会被时间弥散, "
                          "建议改用有线音箱接到同一块声卡的输出。")
            if res["warnings"]:
                self._log(f"⚠ {res['warnings']}")
            if res["ir"] is not None:
                self.ir_view.show_ir(res["ir"], self.settings.audio.samplerate, res["labels"])

        self._worker.done.connect(ok)
        self._worker.failed.connect(lambda e: (self._busy(False), self._log("✗ " + e.splitlines()[0])))
        self._worker.start()
