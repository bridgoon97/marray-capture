"""主窗口: 五个页签串起「设备 → 方案 → 采集 → 质检 → 后处理」这条完整链路。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QTabWidget, QVBoxLayout, QWidget,
)

from ..protocol.plan import Plan
from ..settings import SETTINGS_PATH, AppSettings
from ..store import Session
from .device_page import DevicePage
from .plan_page import PlanPage
from .post_page import PostPage
from .qc_page import QCPage
from .run_page import RunPage
from .widgets import scrollable


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings):
        super().__init__()
        self.settings = settings
        self.plan: Plan | None = None
        self._session: Session | None = None
        self.setWindowTitle("marray-capture —— 干扰人扫频实录采集")
        # 1440p 笔记本 (常见 150% 缩放 → 有效 1707x960) 也要放得下; 内容超了走滚动
        self.resize(1240, 800)
        self.setMinimumSize(900, 600)
        self._build()

    # ------------------------------------------------------------------ 布局
    def _build(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        bar = QHBoxLayout()
        self.le_root = QLineEdit(self.settings.session_root)
        btn_root = QPushButton("…")
        btn_root.setFixedWidth(32)
        btn_root.clicked.connect(self.pick_root)
        self.le_name = QLineEdit(datetime.now().strftime("session_%Y%m%d_%H%M"))
        btn_new = QPushButton("新建会话")
        btn_new.clicked.connect(self.new_session)
        btn_save_cfg = QPushButton("保存配置")
        btn_save_cfg.clicked.connect(self.save_settings)
        btn_load_cfg = QPushButton("载入配置")
        btn_load_cfg.clicked.connect(self.load_settings)
        bar.addWidget(QLabel("会话根目录"))
        bar.addWidget(self.le_root, 2)
        bar.addWidget(btn_root)
        bar.addWidget(QLabel("会话名"))
        bar.addWidget(self.le_name, 1)
        bar.addWidget(btn_new)
        bar.addStretch(1)
        bar.addWidget(btn_save_cfg)
        bar.addWidget(btn_load_cfg)
        root.addLayout(bar)

        self.tabs = QTabWidget()
        self.page_device = DevicePage(self.settings, self.get_session)
        self.page_plan = PlanPage(self.settings, self.on_plan_ready)
        self.page_run = RunPage(self.settings, self.get_session, lambda: self.plan)
        self.page_qc = QCPage(self.settings, self.get_session_optional, self.rerun)
        self.page_post = PostPage(self.settings, self.get_session_optional)
        # 每页套一层滚动区 —— 表单很长, 小屏上必须能滚
        self.tabs.addTab(scrollable(self.page_device), "1 设备")
        self.tabs.addTab(scrollable(self.page_plan), "2 方案")
        self.tabs.addTab(scrollable(self.page_run), "3 采集")
        self.tabs.addTab(scrollable(self.page_qc), "4 质检")
        self.tabs.addTab(scrollable(self.page_post), "5 后处理")
        self.tabs.currentChanged.connect(self._on_tab)
        root.addWidget(self.tabs, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("先在「设备」页选好声卡与通道, 再去「方案」页生成采集方案。")

    def _on_tab(self, idx: int) -> None:
        # 页签里装的是滚动区, 真正的页面在它的 widget() 里
        area = self.tabs.widget(idx)
        page = area.widget() if hasattr(area, "widget") else area
        if page is self.page_qc:
            self.page_qc.refresh()
        elif page is self.page_post:
            self.page_post.refresh()

    # ------------------------------------------------------------------ 会话
    def pick_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择会话根目录", self.le_root.text())
        if path:
            self.le_root.setText(path)
            self.settings.session_root = path

    def new_session(self) -> None:
        self._session = None
        self.le_name.setText(datetime.now().strftime("session_%Y%m%d_%H%M"))
        s = self.get_session()
        self.statusBar().showMessage(f"新会话: {s.dir}")

    def get_session(self) -> Session:
        self.settings.session_root = self.le_root.text().strip() or self.settings.session_root
        name = self.le_name.text().strip() or datetime.now().strftime("session_%Y%m%d_%H%M")
        if self._session is None or self._session.name != name or \
                str(self._session.root) != self.settings.session_root:
            self._session = Session(self.settings.session_root, name, create=True)
        return self._session

    def get_session_optional(self) -> Session | None:
        return self._session

    # ------------------------------------------------------------------ 联动
    def on_plan_ready(self, plan: Plan, want_session: bool = False):
        self.plan = plan
        self.statusBar().showMessage(
            f"方案已就绪: {len(plan.measures)} 个测量位 / {len(plan.steps)} 步")
        if want_session:
            return self.get_session()
        return None

    def rerun(self, take_ids: set[str]) -> None:
        for i in range(self.tabs.count()):
            area = self.tabs.widget(i)
            if getattr(area, "widget", lambda: None)() is self.page_run:
                self.tabs.setCurrentIndex(i)
                break
        self.page_run.rerun(take_ids)

    # ------------------------------------------------------------------ 配置
    def save_settings(self) -> None:
        self.page_plan.push()
        self.settings.session_root = self.le_root.text().strip()
        path, _ = QFileDialog.getSaveFileName(
            self, "保存配置", str(SETTINGS_PATH), "JSON (*.json)")
        if path:
            self.settings.save(Path(path))
            self.statusBar().showMessage(f"配置已保存到 {path}")

    def load_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "载入配置", str(SETTINGS_PATH), "JSON (*.json)")
        if not path:
            return
        try:
            loaded = AppSettings.load(Path(path))
        except Exception as e:
            QMessageBox.critical(self, "载入失败", str(e))
            return
        for field in ("audio", "sweep", "protocol", "export", "qc"):
            setattr(self.settings, field, getattr(loaded, field))
        self.settings.session_root = loaded.session_root
        self.settings.tts_enabled = loaded.tts_enabled
        self.settings.tts_voice = loaded.tts_voice
        self.settings.auto_retry_on_fail = loaded.auto_retry_on_fail
        self.le_root.setText(self.settings.session_root)
        self.page_device.pull()
        self.page_plan.pull()
        self.statusBar().showMessage(f"已载入 {path}")

    # ------------------------------------------------------------------ 关闭
    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self.page_plan.push()
            self.settings.session_root = self.le_root.text().strip()
            self.settings.save()
        except Exception:
            pass
        if self.page_run.runner is not None:
            self.page_run.stop()
        event.accept()
