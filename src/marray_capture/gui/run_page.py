"""采集页: 跑方案。

采集时佩戴者在房间里、手够不到电脑, 所以流程是**全自动语音导播**的:
每一步的指令由音箱播出, 倒计时最后一声升调表示「别动了」, 采完给上行/下行两声反馈。
这一页主要给操作者盯状态用 —— 大字指令是给隔着几米也能看见的场合准备的。
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar, QPushButton,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from ..audio.prompts import PromptRenderer
from ..protocol.plan import Plan
from ..protocol.runner import RunnerHooks, SessionRunner
from ..settings import AppSettings
from ..store import Session
from .widgets import (
    InstructionCard, IRView, LevelMeter, RunnerBridge, Worker, verdict_cell,
)


class RunPage(QWidget):
    def __init__(self, settings: AppSettings, get_session, get_plan, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.get_session = get_session
        self.get_plan = get_plan
        self.runner: SessionRunner | None = None
        self._worker: Worker | None = None
        self.bridge = RunnerBridge()
        self._build()
        self._wire()

    # ------------------------------------------------------------------ 布局
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)

        # ---- 顶栏: 会话 + 进度。进度条只表示"走到第几步", 不抢注意力
        top = QHBoxLayout()
        self.lb_session = QLabel("会话 —")
        self.lb_session.setObjectName("hint")
        self.lb_progress = QLabel("0 / 0")
        self.lb_progress.setObjectName("readout")
        self.pb = QProgressBar()
        self.pb.setTextVisible(False)
        self.pb.setFixedWidth(220)
        top.addWidget(self.lb_session, 1)
        top.addWidget(self.lb_progress)
        top.addWidget(self.pb)
        root.addLayout(top)

        # ---- 主角: 指令卡。占满上部, 三米开外可读
        self.card = InstructionCard()
        root.addWidget(self.card)

        # ---- 操作条: 开始是主按钮, 停止是危险色, 其余安静
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.btn_start = QPushButton("开始采集")
        self.btn_start.setObjectName("primary")
        self.btn_pause = QPushButton("暂停")
        self.btn_redo = QPushButton("重录这个位置")
        self.btn_skip = QPushButton("跳过这个位置")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("danger")
        for b in (self.btn_start, self.btn_pause, self.btn_redo, self.btn_skip, self.btn_stop):
            b.setMinimumHeight(34)
            bar.addWidget(b)
        bar.addStretch(1)
        self.lb_counts = QLabel("✓ 0    ! 0    ✕ 0")
        self.lb_counts.setObjectName("readout")
        bar.addWidget(self.lb_counts)
        root.addLayout(bar)

        # ---- 下部: 坐下来复核时才看的东西
        mid = QHBoxLayout()
        mid.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(6)
        eyebrow = QLabel("输入电平  dBFS / 峰值")
        eyebrow.setObjectName("eyebrow")
        left.addWidget(eyebrow)
        self.meter = LevelMeter()
        left.addWidget(self.meter)
        self.log = QTextEdit()
        self.log.setObjectName("log")
        self.log.setReadOnly(True)
        left.addWidget(self.log, 1)
        mid.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(6)
        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["位置", "结论", "一致性", "说明"])
        self.tbl.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setMaximumHeight(210)
        right.addWidget(self.tbl)
        self.ir_view = IRView()
        right.addWidget(self.ir_view, 1)
        mid.addLayout(right, 5)
        root.addLayout(mid, 1)

        self.btn_start.clicked.connect(self.start)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_redo.clicked.connect(lambda: self.runner and self.runner.redo())
        self.btn_skip.clicked.connect(lambda: self.runner and self.runner.skip())
        self.btn_stop.clicked.connect(self.stop)
        self._set_running(False)

    def _wire(self) -> None:
        self.bridge.level.connect(self.meter.update_levels)
        self.bridge.state.connect(self.card.show_status)
        self.bridge.log.connect(self.log.append)
        self.bridge.step.connect(self._on_step)
        self.bridge.take.connect(self._on_take)
        self.bridge.progress.connect(self._on_progress)
        self.bridge.finished.connect(self._on_finished)

    def _set_running(self, on: bool) -> None:
        self.btn_start.setEnabled(not on)
        for b in (self.btn_pause, self.btn_redo, self.btn_skip, self.btn_stop):
            b.setEnabled(on)

    # ------------------------------------------------------------------ 控制
    def start(self, only_take_ids: set[str] | None = None) -> None:
        plan: Plan | None = self.get_plan()
        if plan is None or not plan.steps:
            QMessageBox.information(self, "开始采集", "先在「方案」页生成方案。")
            return
        if not self.settings.audio.mic_indices():
            QMessageBox.warning(self, "开始采集", "还没有把任何输入通道标为 mic, 去「设备」页配一下。")
            return
        session: Session = self.get_session()
        self.lb_session.setText(f"会话 {session.name}　·　{session.dir}")
        session.save_json("session.json", {
            "settings": self.settings.to_dict(),
            "plan_steps": len(plan.steps),
        })
        plan.save(session.plan_path)

        # 用共享缓存: 在「方案」页预渲染过就直接命中, 不会现场跑 TTS
        renderer = PromptRenderer(
            self.settings.audio.samplerate,
            voice=self.settings.tts_voice,
            enabled=self.settings.tts_enabled,
            backend=self.settings.tts_backend,
        )
        self.log.append(f"语音导播: {renderer.describe()}")
        if self.settings.tts_enabled and not renderer.ok:
            self.log.append("⚠ 已降级为提示音 + 屏幕大字。")

        hooks = RunnerHooks(
            on_step=lambda s, i, n: self.bridge.step.emit(s, i, n),
            on_state=self.bridge.state.emit,
            on_level=self.bridge.level.emit,
            on_take=lambda s, q, r: self.bridge.take.emit(s, q, r),
            on_log=self.bridge.log.emit,
            on_finished=self.bridge.finished.emit,
            on_progress=lambda a, b: self.bridge.progress.emit(a, b),
        )
        self.runner = SessionRunner(
            self.settings, plan, session, renderer, hooks, only_take_ids=only_take_ids,
        )
        self.meter.set_channels([c.display() for c in self.settings.audio.ordered_channels()])
        self.meter.reset_peaks()
        self.tbl.setRowCount(0)
        self._set_running(True)
        self.btn_pause.setText("暂停")
        self._worker = Worker(self.runner.run)
        self._worker.failed.connect(lambda e: self.log.append("✗ " + e))
        self._worker.start()

    def rerun(self, take_ids: set[str]) -> None:
        """质检页发起的补录。"""
        self.start(only_take_ids=take_ids)

    def toggle_pause(self) -> None:
        if not self.runner:
            return
        if self.btn_pause.text() == "暂停":
            self.runner.pause()
            self.btn_pause.setText("继续")
        else:
            self.runner.resume()
            self.btn_pause.setText("暂停")

    def stop(self) -> None:
        if self.runner:
            self.runner.stop()

    # ------------------------------------------------------------------ 回调
    def _on_step(self, step, i: int, n: int) -> None:
        if step.kind == "setup":
            self.card.reset_verdict()
            self.card.show_step("调整位置", step.instruction)
        else:
            self.card.show_step(step.take_id, step.instruction)

    def _on_progress(self, i: int, n: int) -> None:
        self.pb.setMaximum(n)
        self.pb.setValue(i)
        self.lb_progress.setText(f"{i} / {n}")

    def _on_take(self, step, qc, record) -> None:
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        ncc = qc.repeat_ncc
        self.tbl.setItem(r, 0, QTableWidgetItem(record.get("take_id", "")))
        self.tbl.setItem(r, 1, verdict_cell(qc.verdict))
        self.tbl.setItem(r, 2, QTableWidgetItem("  —  " if ncc != ncc else f"{ncc:.3f}"))
        self.tbl.setItem(r, 3, QTableWidgetItem(qc.summary()))
        self.tbl.scrollToBottom()
        self.card.show_verdict(qc.verdict)

        stats = self.get_session().stats()
        self.lb_counts.setText(
            f"✓ {stats['PASS']}    ! {stats['WARN']}    ✕ {stats['FAIL']}")
        session = self.get_session()
        ir_rel = record.get("ir_file")
        if ir_rel:
            try:
                import soundfile as sf
                data, fs = sf.read(str(session.dir / ir_rel), always_2d=True)
                self.ir_view.show_ir(data, fs, record.get("channels"))
            except Exception:
                pass

    def _on_finished(self, reason: str) -> None:
        self._set_running(False)
        self.card.show_step("本轮结束", reason)
        self.card.show_status("去「质检」页看分组统计，整组失败的当场补录")
        self.log.append(f"—— {reason} ——")
