"""方案页: 配置采集协议、扫频参数与导出参数, 生成并预览步骤表。

方案里的方位全部用「向右转一格」表达, 不给绝对角度 —— 角度标签只是名义值,
用于分组统计, 不参与任何计算。想改密度就改「基准圈格数 / 其余圈格数」。
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressBar,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..audio import tts
from ..audio.engine import AudioEngine
from ..audio.prompts import PromptRenderer
from ..protocol.plan import Plan, build_plan
from ..settings import AppSettings
from . import notes
from .widgets import RunnerBridge, Worker


def _parse_ints(text: str, fallback: list[int]) -> list[int]:
    try:
        vals = [int(x.strip()) for x in text.replace("，", ",").split(",") if x.strip()]
        return vals or fallback
    except ValueError:
        return fallback


class PlanPage(QWidget):
    def __init__(self, settings: AppSettings, on_plan_ready, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.on_plan_ready = on_plan_ready
        self.plan: Plan | None = None
        self._worker: Worker | None = None
        self._param_dialog = None
        self._build()
        self._apply_param_tooltips()
        self.pull()

    # ------------------------------------------------------------------ 布局
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        self.notes_btn = notes.build_notes_button("plan", self.settings)
        self.params_btn = QPushButton("参数说明")
        self.params_btn.clicked.connect(self.show_param_help)
        bar = QHBoxLayout()
        bar.addStretch(1)
        bar.addWidget(self.params_btn)
        bar.addWidget(self.notes_btn)
        outer.addLayout(bar)
        root = QHBoxLayout()
        outer.addLayout(root, 1)
        left = QVBoxLayout()

        # ---- 协议
        f1 = QFormLayout()
        self.le_subject = QLineEdit()
        self.le_wearing = QLineEdit()
        self.cb_side = QComboBox()
        self.cb_side.addItems(["R", "L"])
        self.le_dist = QLineEdit()
        self.le_height = QLineEdit()
        self.sp_dense_d = QSpinBox(); self.sp_dense_d.setRange(10, 500); self.sp_dense_d.setSuffix(" cm")
        self.sp_dense_h = QSpinBox(); self.sp_dense_h.setRange(10, 250); self.sp_dense_h.setSuffix(" cm")
        self.sp_dense_n = QSpinBox(); self.sp_dense_n.setRange(3, 36)
        self.sp_sparse_n = QSpinBox(); self.sp_sparse_n.setRange(3, 36)
        self.le_orient = QLineEdit()
        self.sp_orient_n = QSpinBox(); self.sp_orient_n.setRange(1, 24)
        self.sp_random = QSpinBox(); self.sp_random.setRange(0, 200)
        self.sp_rewear = QSpinBox(); self.sp_rewear.setRange(0, 10)
        self.sp_rewear_n = QSpinBox(); self.sp_rewear_n.setRange(1, 36)
        self.sp_settle = QDoubleSpinBox(); self.sp_settle.setRange(1.0, 30.0); self.sp_settle.setSuffix(" s")
        self.sp_setup_settle = QDoubleSpinBox(); self.sp_setup_settle.setRange(3.0, 120.0); self.sp_setup_settle.setSuffix(" s")

        f1.addRow("被试编号", self.le_subject)
        f1.addRow("佩戴编号", self.le_wearing)
        f1.addRow("佩戴侧", self.cb_side)
        f1.addRow("距离 (cm, 逗号分隔)", self.le_dist)
        f1.addRow("音箱高度 (cm, 逗号分隔)", self.le_height)
        f1.addRow("基准圈距离", self.sp_dense_d)
        f1.addRow("基准圈高度", self.sp_dense_h)
        f1.addRow("基准圈格数", self.sp_dense_n)
        f1.addRow("其余圈格数", self.sp_sparse_n)
        f1.addRow("音箱朝向 (度)", self.le_orient)
        f1.addRow("朝向变体圈格数", self.sp_orient_n)
        f1.addRow("随机位数量", self.sp_random)
        f1.addRow("重戴次数", self.sp_rewear)
        f1.addRow("每次重戴格数", self.sp_rewear_n)
        f1.addRow("测量位稳定时间", self.sp_settle)
        f1.addRow("挪位/调高稳定时间", self.sp_setup_settle)
        box1 = QGroupBox("采集协议")
        box1.setLayout(f1)
        left.addWidget(box1)
        left.addStretch(1)
        root.addLayout(left, 3)
        mid = QVBoxLayout()

        # ---- 扫频
        f2 = QFormLayout()
        self.sp_f1 = QDoubleSpinBox(); self.sp_f1.setRange(10, 500); self.sp_f1.setSuffix(" Hz")
        self.sp_f2 = QDoubleSpinBox(); self.sp_f2.setRange(1000, 24000); self.sp_f2.setSuffix(" Hz")
        self.sp_dur = QDoubleSpinBox(); self.sp_dur.setRange(1.0, 30.0); self.sp_dur.setSuffix(" s")
        self.sp_rep = QSpinBox(); self.sp_rep.setRange(1, 6)
        self.sp_gap = QDoubleSpinBox(); self.sp_gap.setRange(0.2, 10.0); self.sp_gap.setSuffix(" s")
        self.sp_pre = QDoubleSpinBox(); self.sp_pre.setRange(0.5, 10.0); self.sp_pre.setSuffix(" s")
        self.sp_tail = QDoubleSpinBox(); self.sp_tail.setRange(0.5, 10.0); self.sp_tail.setSuffix(" s")
        self.sp_maxlat = QDoubleSpinBox(); self.sp_maxlat.setRange(0.1, 3.0); self.sp_maxlat.setSuffix(" s")
        self.sp_amp = QDoubleSpinBox(); self.sp_amp.setRange(0.05, 1.0); self.sp_amp.setSingleStep(0.05)
        f2.addRow("起始频率", self.sp_f1)
        f2.addRow("终止频率", self.sp_f2)
        f2.addRow("扫频时长", self.sp_dur)
        f2.addRow("每位置扫几次", self.sp_rep)
        f2.addRow("两次之间静音", self.sp_gap)
        f2.addRow("扫频前静音 (噪声窗)", self.sp_pre)
        f2.addRow("扫频后静音", self.sp_tail)
        f2.addRow("最大延迟余量", self.sp_maxlat)
        f2.addRow("播放幅度", self.sp_amp)
        box2 = QGroupBox("扫频信号")
        box2.setLayout(f2)
        mid.addWidget(box2)

        # ---- 导出
        f3 = QFormLayout()
        self.sp_pre_ms = QDoubleSpinBox(); self.sp_pre_ms.setRange(0.0, 50.0); self.sp_pre_ms.setSuffix(" ms")
        self.sp_len_ms = QDoubleSpinBox(); self.sp_len_ms.setRange(100.0, 4000.0); self.sp_len_ms.setSuffix(" ms")
        self.ck_16k = QCheckBox("同时导出 16 kHz")
        self.ck_raw = QCheckBox("保存原始录音")
        self.cb_rawfmt = QComboBox(); self.cb_rawfmt.addItems(["FLAC", "WAV"])
        self.ck_avg = QCheckBox("多次扫频对齐后求平均")
        self.ck_retry = QCheckBox("质检 FAIL 时自动重录一次")
        f3.addRow("直达峰前保留", self.sp_pre_ms)
        f3.addRow("IR 长度", self.sp_len_ms)
        f3.addRow("", self.ck_16k)
        f3.addRow("", self.ck_raw)
        f3.addRow("原始录音格式", self.cb_rawfmt)
        f3.addRow("", self.ck_avg)
        f3.addRow("", self.ck_retry)
        box3 = QGroupBox("提取与导出")
        box3.setLayout(f3)
        mid.addWidget(box3)

        # ---- 语音导播
        f4 = QFormLayout()
        self.ck_tts = QCheckBox("启用语音导播 (关掉就只用提示音 + 屏幕大字)")
        self.cb_backend = QComboBox()
        for key in ("auto", "piper", "edge", "sapi", "say", "none"):
            self.cb_backend.addItem(tts.BACKEND_LABELS[key], key)
        self.cb_voice = QComboBox()
        self.cb_voice.setEditable(True)
        self.btn_dl = QPushButton("下载中文语音模型 (piper)")
        self.btn_try = QPushButton("试听一句")
        self.lb_tts = QLabel("-")
        self.lb_tts.setWordWrap(True)
        self.lb_tts.setStyleSheet("color:#555;")
        f4.addRow("", self.ck_tts)
        f4.addRow("后端", self.cb_backend)
        f4.addRow("语音", self.cb_voice)
        f4.addRow("", self.btn_dl)
        f4.addRow("", self.btn_try)
        f4.addRow("状态", self.lb_tts)
        box4 = QGroupBox("语音导播")
        box4.setLayout(f4)
        mid.addWidget(box4)
        mid.addStretch(1)
        root.addLayout(mid, 3)

        self.cb_backend.currentIndexChanged.connect(self.refresh_tts)
        self.ck_tts.toggled.connect(self.refresh_tts)
        self.cb_voice.currentIndexChanged.connect(self._voice_chosen)
        self.btn_dl.clicked.connect(self.download_voice)
        self.btn_try.clicked.connect(self.preview_voice)

        # ---- 右侧: 预览
        right = QVBoxLayout()
        bar = QHBoxLayout()
        self.btn_gen = QPushButton("生成方案")
        self.btn_tts = QPushButton("预渲染语音")
        self.btn_save = QPushButton("保存方案")
        self.btn_load = QPushButton("载入方案")
        for b in (self.btn_gen, self.btn_tts, self.btn_save, self.btn_load):
            bar.addWidget(b)
        bar.addStretch(1)
        right.addLayout(bar)

        self.lb_summary = QLabel("尚未生成方案")
        self.lb_summary.setWordWrap(True)
        right.addWidget(self.lb_summary)
        self.pb = QProgressBar()
        self.pb.setVisible(False)
        right.addWidget(self.pb)

        self.tbl = QTableWidget(0, 8)
        self.tbl.setHorizontalHeaderLabels(
            ["#", "类型", "take_id", "标签", "距离", "高度", "音箱朝向", "指令"])
        self.tbl.horizontalHeader().setSectionResizeMode(7, QHeaderView.Stretch)
        right.addWidget(self.tbl, 1)
        root.addLayout(right, 4)

        self.btn_gen.clicked.connect(self.generate)
        self.btn_tts.clicked.connect(self.prewarm_tts)
        self.btn_save.clicked.connect(self.save_plan)
        self.btn_load.clicked.connect(self.load_plan)

    # ------------------------------------------------------------------ 同步
    def pull(self) -> None:
        p, s, e = self.settings.protocol, self.settings.sweep, self.settings.export
        self.le_subject.setText(p.subject_id)
        self.le_wearing.setText(p.wearing_id)
        self.cb_side.setCurrentText(p.side)
        self.le_dist.setText(", ".join(map(str, p.distances_cm)))
        self.le_height.setText(", ".join(map(str, p.heights_cm)))
        self.sp_dense_d.setValue(p.dense_distance_cm)
        self.sp_dense_h.setValue(p.dense_height_cm)
        self.sp_dense_n.setValue(p.dense_steps)
        self.sp_sparse_n.setValue(p.sparse_steps)
        self.le_orient.setText(", ".join(map(str, p.speaker_orientations)))
        self.sp_orient_n.setValue(p.orientation_subset_steps)
        self.sp_random.setValue(p.random_positions)
        self.sp_rewear.setValue(p.rewearing_rings)
        self.sp_rewear_n.setValue(p.rewearing_steps)
        self.sp_settle.setValue(p.settle_s)
        self.sp_setup_settle.setValue(p.setup_settle_s)

        self.sp_f1.setValue(s.f_start); self.sp_f2.setValue(s.f_end)
        self.sp_dur.setValue(s.duration_s); self.sp_rep.setValue(s.repeats)
        self.sp_gap.setValue(s.gap_s); self.sp_pre.setValue(s.preroll_s)
        self.sp_tail.setValue(s.tail_s); self.sp_maxlat.setValue(s.max_latency_s)
        self.sp_amp.setValue(s.amplitude)

        self.sp_pre_ms.setValue(e.ir_pre_ms); self.sp_len_ms.setValue(e.ir_len_ms)
        self.ck_16k.setChecked(e.export_16k); self.ck_raw.setChecked(e.save_raw)
        self.cb_rawfmt.setCurrentText(e.raw_format); self.ck_avg.setChecked(e.average_repeats)
        self.ck_retry.setChecked(self.settings.auto_retry_on_fail)
        self.ck_tts.setChecked(self.settings.tts_enabled)
        for i in range(self.cb_backend.count()):
            if self.cb_backend.itemData(i) == self.settings.tts_backend:
                self.cb_backend.setCurrentIndex(i)
                break
        self.refresh_tts()

    def push(self) -> None:
        p, s, e = self.settings.protocol, self.settings.sweep, self.settings.export
        p.subject_id = self.le_subject.text().strip() or "S01"
        p.wearing_id = self.le_wearing.text().strip() or "W1"
        p.side = self.cb_side.currentText()
        p.distances_cm = _parse_ints(self.le_dist.text(), p.distances_cm)
        p.heights_cm = _parse_ints(self.le_height.text(), p.heights_cm)
        p.dense_distance_cm = self.sp_dense_d.value()
        p.dense_height_cm = self.sp_dense_h.value()
        p.dense_steps = self.sp_dense_n.value()
        p.sparse_steps = self.sp_sparse_n.value()
        p.speaker_orientations = _parse_ints(self.le_orient.text(), p.speaker_orientations)
        p.orientation_subset_steps = self.sp_orient_n.value()
        p.random_positions = self.sp_random.value()
        p.rewearing_rings = self.sp_rewear.value()
        p.rewearing_steps = self.sp_rewear_n.value()
        p.settle_s = self.sp_settle.value()
        p.setup_settle_s = self.sp_setup_settle.value()

        s.f_start = self.sp_f1.value(); s.f_end = self.sp_f2.value()
        s.duration_s = self.sp_dur.value(); s.repeats = self.sp_rep.value()
        s.gap_s = self.sp_gap.value(); s.preroll_s = self.sp_pre.value()
        s.tail_s = self.sp_tail.value(); s.max_latency_s = self.sp_maxlat.value()
        s.amplitude = self.sp_amp.value()

        e.ir_pre_ms = self.sp_pre_ms.value(); e.ir_len_ms = self.sp_len_ms.value()
        e.export_16k = self.ck_16k.isChecked(); e.save_raw = self.ck_raw.isChecked()
        e.raw_format = self.cb_rawfmt.currentText(); e.average_repeats = self.ck_avg.isChecked()
        self.settings.auto_retry_on_fail = self.ck_retry.isChecked()
        self.settings.tts_enabled = self.ck_tts.isChecked()
        self.settings.tts_backend = self.cb_backend.currentData() or "auto"
        self.settings.tts_voice = (self.cb_voice.currentData()
                                   or self.cb_voice.currentText().strip())

    # ------------------------------------------------------------------ 说明
    def _apply_param_tooltips(self) -> None:
        """把参数说明挂到各个控件上, 鼠标悬停即可看。"""
        for attr, grp, name, desc in notes.PARAMS:
            w = getattr(self, attr, None)
            if w is None:
                continue
            w.setToolTip(f"<b>{name}</b>　<span style='color:#777'>{grp}</span>"
                         f"<hr style='margin:4px 0'>{desc}")

    def show_param_help(self) -> None:
        from .widgets import InfoDialog

        if self._param_dialog is None:
            self._param_dialog = InfoDialog(
                "采集参数说明", notes.params_html(), self, size=(700, 640))
        self._param_dialog.show()
        self._param_dialog.raise_()
        self._param_dialog.activateWindow()

    # ------------------------------------------------------------------ 语音
    def _make_renderer(self) -> PromptRenderer:
        """用当前界面设置造一个渲染器 (共享缓存, 不绑定会话)。"""
        return PromptRenderer(
            self.settings.audio.samplerate,
            voice=self.settings.tts_voice,
            enabled=self.settings.tts_enabled,
            backend=self.settings.tts_backend,
        )

    def refresh_tts(self) -> None:
        """切后端/开关时刷新语音列表与状态说明。不做任何耗时操作。"""
        self.settings.tts_enabled = self.ck_tts.isChecked()
        self.settings.tts_backend = self.cb_backend.currentData() or "auto"
        r = self._make_renderer()
        self._renderer = r

        cur = self.settings.tts_voice
        self.cb_voice.blockSignals(True)
        self.cb_voice.clear()
        for v in r.available_voices():
            self.cb_voice.addItem(v.label, v.id)
        idx = self.cb_voice.findData(cur or r.voice)
        if idx >= 0:
            self.cb_voice.setCurrentIndex(idx)
        elif cur:
            self.cb_voice.setEditText(cur)
        self.cb_voice.blockSignals(False)

        self.lb_tts.setText(r.describe())
        is_piper = r.backend == "piper" or self.settings.tts_backend == "piper"
        self.btn_dl.setEnabled(self.settings.tts_enabled and is_piper)
        self.btn_try.setEnabled(self.settings.tts_enabled)

    def _voice_chosen(self) -> None:
        self.settings.tts_voice = (self.cb_voice.currentData()
                                   or self.cb_voice.currentText().strip())
        self.refresh_tts()

    def download_voice(self) -> None:
        """下载 piper 中文模型。放工作线程 —— 有几十 MB。"""
        name = (self.cb_voice.currentData() or self.cb_voice.currentText().strip()
                or tts.PIPER_ZH_VOICES[0][0])
        self.pb.setVisible(True)
        self.pb.setRange(0, 100)
        self.btn_dl.setEnabled(False)
        self.lb_tts.setText(f"正在下载 {name} …")

        bridge = RunnerBridge()
        bridge.progress.connect(lambda a, b: (
            self.pb.setRange(0, max(1, b)), self.pb.setValue(a)))
        self._bridge = bridge

        self._worker = Worker(tts.download_piper_voice, name, None, bridge.progress.emit)

        def ok(path) -> None:
            self.pb.setVisible(False)
            self.btn_dl.setEnabled(True)
            self.settings.tts_voice = name
            self.refresh_tts()
            QMessageBox.information(self, "下载完成", f"模型已保存到\n{path}")

        self._worker.done.connect(ok)
        self._worker.failed.connect(lambda e: (
            self.pb.setVisible(False), self.btn_dl.setEnabled(True),
            self.lb_tts.setText("下载失败"),
            QMessageBox.critical(self, "下载失败", e.splitlines()[0])))
        self._worker.start()

    def preview_voice(self) -> None:
        """合成并从音箱播一句, 确认音量与吐字都合适。"""
        self.push()
        r = self._make_renderer()
        text = "面向音箱，保持不动。向右转一格，大约三十度。"
        self.btn_try.setEnabled(False)

        def job():
            wave = r.render(text)
            if not len(wave):
                raise RuntimeError(r.last_error or "没有可用的 TTS 后端")
            AudioEngine(self.settings.audio).play(wave)
            return r.describe()

        self._worker = Worker(job)
        self._worker.done.connect(lambda d: (self.btn_try.setEnabled(True),
                                             self.lb_tts.setText(d)))
        self._worker.failed.connect(lambda e: (
            self.btn_try.setEnabled(True),
            QMessageBox.warning(self, "试听失败", e.splitlines()[0])))
        self._worker.start()

    # ------------------------------------------------------------------ 动作
    def generate(self) -> None:
        self.push()
        self.plan = build_plan(self.settings.protocol)
        self._fill_table()
        self.on_plan_ready(self.plan)

    def _take_seconds(self) -> float:
        s = self.settings.sweep
        return s.preroll_s + s.repeats * s.duration_s + max(0, s.repeats - 1) * s.gap_s + s.tail_s

    def _fill_table(self) -> None:
        plan = self.plan
        if plan is None:
            return
        steps = plan.steps
        self.tbl.setRowCount(len(steps))
        for r, st in enumerate(steps):
            vals = [
                str(st.idx), st.kind, st.take_id, st.tag,
                "" if st.distance_cm is None else f"{st.distance_cm} cm",
                "" if st.height_cm is None else f"{st.height_cm} cm",
                f"{st.speaker_deg}°", st.instruction,
            ]
            for c, v in enumerate(vals):
                self.tbl.setItem(r, c, QTableWidgetItem(v))
        self.tbl.resizeColumnsToContents()
        self.tbl.horizontalHeader().setSectionResizeMode(7, QHeaderView.Stretch)

        n_meas = len(plan.measures)
        secs = plan.estimated_seconds(self._take_seconds())
        self.lb_summary.setText(
            f"共 <b>{len(steps)}</b> 步, 其中测量位 <b>{n_meas}</b> 个; "
            f"每个位置激励时长 {self._take_seconds():.1f} s; "
            f"预计耗时 <b>{secs / 60:.0f} 分钟</b> (不含中途休息与重录)。"
        )

    def prewarm_tts(self) -> None:
        """把方案里所有指令合成好并落盘, 采集时零延迟、现场断网也不影响。"""
        if self.plan is None:
            QMessageBox.information(self, "预渲染语音", "先生成方案。")
            return
        self.push()
        session = self.on_plan_ready(self.plan, want_session=True)
        renderer = self._make_renderer()
        if not renderer.ok:
            QMessageBox.warning(
                self, "TTS 不可用",
                "没有可用的语音合成后端, 采集时会自动降级为提示音 + 屏幕大字。\n\n"
                f"诊断: {renderer.last_error}\n\n"
                "装开源离线后端: uv sync --extra tts, 然后点「下载中文语音模型」。",
            )
            return

        texts = self.plan.all_instructions()
        self.pb.setVisible(True)
        self.pb.setRange(0, len(texts))
        self.btn_tts.setEnabled(False)

        bridge = RunnerBridge()
        bridge.progress.connect(lambda a, b: self.pb.setValue(a))
        self._bridge = bridge

        def job():
            n = renderer.prewarm(texts, progress=bridge.progress.emit)
            # 另存一份到会话目录, 开录前可以逐条试听核对
            index = renderer.export(texts, session.prompts_dir)
            return n, index

        self._worker = Worker(job)

        def ok(res) -> None:
            n, index = res
            self.pb.setVisible(False)
            self.btn_tts.setEnabled(True)
            msg = (f"{n}/{len(texts)} 条指令已合成 ({renderer.describe()})。\n"
                   f"音频与文本索引: {index.parent}")
            if n < len(texts):
                msg += f"\n\n失败原因: {renderer.last_error}"
            QMessageBox.information(self, "预渲染语音", msg)
            self.lb_tts.setText(renderer.describe())

        self._worker.done.connect(ok)
        self._worker.failed.connect(lambda e: (
            self.pb.setVisible(False), self.btn_tts.setEnabled(True),
            QMessageBox.critical(self, "预渲染失败", e.splitlines()[0])))
        self._worker.start()

    def save_plan(self) -> None:
        if self.plan is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存方案", "plan.json", "JSON (*.json)")
        if path:
            self.plan.save(path)

    def load_plan(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "载入方案", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.plan = Plan.load(Path(path))
        except Exception as e:
            QMessageBox.critical(self, "载入失败", str(e))
            return
        self._fill_table()
        self.on_plan_ready(self.plan)
