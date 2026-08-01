"""采集页: 跑方案。

采集时佩戴者在房间里、手够不到电脑, 所以流程是**全自动语音导播**的:
每一步的指令由音箱播出, 倒计时最后一声升调表示「别动了」, 采完给上行/下行两声反馈。
这一页主要给操作者盯状态用 —— 大字指令是给隔着几米也能看见的场合准备的。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar, QPushButton,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from ..audio.prompts import PromptRenderer
from ..protocol.plan import Plan
from ..protocol.runner import RunnerHooks, SessionRunner
from ..settings import AppSettings
from ..store import Session
from .widgets import VERDICT_COLOR, IRView, LevelMeter, RunnerBridge, Worker, big_label


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

        self.lb_session = QLabel("会话: -")
        self.lb_session.setStyleSheet("color:#555;")
        root.addWidget(self.lb_session)

        self.lb_instruction = big_label("按「开始采集」")
        self.lb_instruction.setStyleSheet(
            "background:#f6f8fa; border:1px solid #d0d7de; border-radius:8px; padding:12px;")
        root.addWidget(self.lb_instruction)

        self.lb_state = QLabel("就绪")
        self.lb_state.setAlignment(Qt.AlignCenter)
        root.addWidget(self.lb_state)

        self.pb = QProgressBar()
        self.pb.setFormat("%v / %m")
        root.addWidget(self.pb)

        bar = QHBoxLayout()
        self.btn_start = QPushButton("开始采集")
        self.btn_pause = QPushButton("暂停")
        self.btn_redo = QPushButton("重录当前位置")
        self.btn_skip = QPushButton("跳过当前位置")
        self.btn_stop = QPushButton("停止")
        for b in (self.btn_start, self.btn_pause, self.btn_redo, self.btn_skip, self.btn_stop):
            b.setMinimumHeight(34)
            bar.addWidget(b)
        root.addLayout(bar)

        mid = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("<b>输入电平</b>"))
        self.meter = LevelMeter()
        left.addWidget(self.meter)
        self.lb_counts = QLabel("PASS 0 / WARN 0 / FAIL 0")
        left.addWidget(self.lb_counts)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        left.addWidget(self.log, 1)
        mid.addLayout(left, 3)

        right = QVBoxLayout()
        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["take_id", "结论", "一致性", "说明"])
        self.tbl.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.tbl.setMaximumHeight(220)
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
        self.bridge.state.connect(self.lb_state.setText)
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
        self.lb_session.setText(f"会话: {session.dir}")
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
        self.meter.set_channels([c.display() for c in
                                 sorted((c for c in self.settings.audio.channels if c.role != "ignore"),
                                        key=lambda c: c.index)])
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
        prefix = "【调整】" if step.kind == "setup" else f"【{step.take_id}】"
        self.lb_instruction.setText(f"{prefix}\n{step.instruction}")

    def _on_progress(self, i: int, n: int) -> None:
        self.pb.setMaximum(n)
        self.pb.setValue(i)

    def _on_take(self, step, qc, record) -> None:
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        ncc = qc.repeat_ncc
        vals = [record.get("take_id", ""), qc.verdict,
                "-" if ncc != ncc else f"{ncc:.3f}", qc.summary()]
        for c, v in enumerate(vals):
            item = QTableWidgetItem(v)
            if c == 1:
                item.setForeground(Qt.GlobalColor.black)
                item.setBackground(_qcolor(VERDICT_COLOR.get(qc.verdict, "#ffffff")))
            self.tbl.setItem(r, c, item)
        self.tbl.scrollToBottom()

        session = self.get_session()
        stats = session.stats()
        self.lb_counts.setText(
            f"PASS {stats['PASS']} / WARN {stats['WARN']} / FAIL {stats['FAIL']}  "
            f"(共 {stats['total']})")
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
        self.lb_state.setText(f"结束: {reason}")
        self.lb_instruction.setText("本轮结束")
        self.log.append(f"—— {reason} ——")


def _qcolor(hex_str: str):
    from PySide6.QtGui import QColor
    c = QColor(hex_str)
    c.setAlpha(60)
    return c
