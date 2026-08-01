"""后处理页: 去音箱响应 + 早期段保留/晚期尾随机合成。

两步是独立的, 可以只做其中一步:

**去音箱响应** —— 用测量麦在音箱正轴录的参考 IR 设计最小相位反滤波器。做 RTF
(麦间相对量) 时音箱响应会自己约掉, 但把 IR 直接卷积干净语音生成训练音频时不会,
所以正式数据建议做。

**混响随机增强** —— 保留实测早期段 (直达+早期反射, 承载阵列/近场/头部线索),
按随机 T60/DRR 重新合成带正确通道间相干性的晚期尾。非声学通道 (VPU) 保留实测原样。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

from ..rir import speaker_eq
from ..rir.augment_runner import default_config, parse_coords, run_augment, select_inputs
from ..settings import AppSettings
from ..store import Session, list_sessions
from .widgets import IRView, RunnerBridge, Worker


def _range_row(lo: float, hi: float, step: float, suffix: str = "") -> tuple[QWidget, QDoubleSpinBox, QDoubleSpinBox]:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    a, b = QDoubleSpinBox(), QDoubleSpinBox()
    for sp in (a, b):
        sp.setRange(-1e6, 1e6)
        sp.setSingleStep(step)
        sp.setSuffix(suffix)
    a.setValue(lo)
    b.setValue(hi)
    h.addWidget(a)
    h.addWidget(QLabel("~"))
    h.addWidget(b)
    return w, a, b


class PostPage(QWidget):
    def __init__(self, settings: AppSettings, get_session, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.get_session = get_session
        self.eq_fir: np.ndarray | None = None
        self._worker: Worker | None = None
        self.bridge = RunnerBridge()
        self._build()
        self.bridge.log.connect(self.log.append)
        self.bridge.progress.connect(self._on_progress)
        self.refresh()

    # ------------------------------------------------------------------ 布局
    def _build(self) -> None:
        root = QHBoxLayout(self)
        left = QVBoxLayout()

        # ---- 会话与输入筛选
        f0 = QFormLayout()
        self.cb_session = QComboBox()
        self.cb_accept = QComboBox()
        self.cb_accept.addItems(["只用 PASS", "PASS + WARN"])
        self.le_tags = QLineEdit()
        self.le_tags.setPlaceholderText("留空=全部; 例: grid,rewear")
        self.cb_fs = QComboBox()
        self.cb_fs.addItems(["16000", "48000"])
        btn_refresh = QPushButton("刷新会话列表")
        btn_refresh.clicked.connect(self.refresh)
        f0.addRow("会话", self.cb_session)
        f0.addRow("采样率", self.cb_fs)
        f0.addRow("质检门槛", self.cb_accept)
        f0.addRow("只用这些标签", self.le_tags)
        f0.addRow("", btn_refresh)
        box0 = QGroupBox("输入")
        box0.setLayout(f0)
        left.addWidget(box0)

        # ---- 去音箱响应
        f1 = QFormLayout()
        self.le_ref = QLineEdit()
        self.le_ref.setPlaceholderText("测量麦在音箱正轴 1 米录的参考 IR (wav)")
        btn_pick = QPushButton("选择…")
        btn_pick.clicked.connect(self.pick_ref)
        ref_row = QWidget()
        rh = QHBoxLayout(ref_row)
        rh.setContentsMargins(0, 0, 0, 0)
        rh.addWidget(self.le_ref, 1)
        rh.addWidget(btn_pick)
        self.sp_ref_ch = QSpinBox(); self.sp_ref_ch.setRange(-1, 15); self.sp_ref_ch.setValue(-1)
        self.sp_eq_lo = QDoubleSpinBox(); self.sp_eq_lo.setRange(20, 2000); self.sp_eq_lo.setValue(80); self.sp_eq_lo.setSuffix(" Hz")
        self.sp_eq_hi = QDoubleSpinBox(); self.sp_eq_hi.setRange(1000, 24000); self.sp_eq_hi.setValue(7500); self.sp_eq_hi.setSuffix(" Hz")
        self.sp_boost = QDoubleSpinBox(); self.sp_boost.setRange(0, 30); self.sp_boost.setValue(12); self.sp_boost.setSuffix(" dB")
        self.sp_win = QDoubleSpinBox(); self.sp_win.setRange(2, 100); self.sp_win.setValue(20); self.sp_win.setSuffix(" ms")
        self.ck_eq = QCheckBox("增强前先去音箱响应")
        btn_design = QPushButton("设计反滤波器并预览")
        btn_design.clicked.connect(self.design_eq)
        f1.addRow("参考 IR", ref_row)
        f1.addRow("参考通道 (-1=自动选最强)", self.sp_ref_ch)
        f1.addRow("反演下限", self.sp_eq_lo)
        f1.addRow("反演上限", self.sp_eq_hi)
        f1.addRow("最大提升量", self.sp_boost)
        f1.addRow("估响应用的窗长", self.sp_win)
        f1.addRow("", btn_design)
        f1.addRow("", self.ck_eq)
        box1 = QGroupBox("① 去音箱响应 (可选)")
        box1.setLayout(f1)
        left.addWidget(box1)

        # ---- 增强参数
        f2 = QFormLayout()
        self.sp_edge = QDoubleSpinBox(); self.sp_edge.setRange(0.001, 1.0); self.sp_edge.setDecimals(4)
        self.sp_edge.setValue(0.01); self.sp_edge.setSuffix(" m")
        self.cb_geom = QComboBox()
        for key, label in [
            ("equilateral_triangle", "等边三角形 (3 麦)"),
            ("linear", "线阵 (等间距)"),
            ("coords", "自定义坐标 (任意麦克风数)"),
        ]:
            self.cb_geom.addItem(label, key)
        self.sp_nmic = QSpinBox(); self.sp_nmic.setRange(2, 32); self.sp_nmic.setValue(3)
        self.te_coords = QPlainTextEdit()
        self.te_coords.setPlaceholderText(
            "每行一个麦克风的 x, y, z，单位米。例如 4 麦 1cm 正方形：\n"
            "0.005,  0.005, 0\n0.005, -0.005, 0\n-0.005, -0.005, 0\n-0.005,  0.005, 0")
        self.te_coords.setMaximumHeight(110)
        self.lb_geom = QLabel("-")
        self.lb_geom.setWordWrap(True)
        self.lb_geom.setObjectName("hint")
        self.sp_split = QDoubleSpinBox(); self.sp_split.setRange(5, 200); self.sp_split.setValue(50); self.sp_split.setSuffix(" ms")
        self.sp_cf = QDoubleSpinBox(); self.sp_cf.setRange(0, 50); self.sp_cf.setValue(5); self.sp_cf.setSuffix(" ms")
        w_t60, self.sp_t60a, self.sp_t60b = _range_row(0.2, 0.8, 0.05, " s")
        w_damp, self.sp_dampa, self.sp_dampb = _range_row(0.4, 0.9, 0.05)
        w_drr, self.sp_drra, self.sp_drrb = _range_row(0.0, 15.0, 1.0, " dB")
        w_tilt, self.sp_tilta, self.sp_tiltb = _range_row(-3.0, 1.0, 0.5, " dB/oct")
        w_tail, self.sp_taila, self.sp_tailb = _range_row(300, 1200, 50, " ms")
        w_snr, self.sp_snra, self.sp_snrb = _range_row(40, 60, 1, " dB")
        self.ck_noise = QCheckBox("叠加通道间不相干传感器噪声")
        self.ck_noise.setChecked(True)
        self.sp_num = QSpinBox(); self.sp_num.setRange(1, 64); self.sp_num.setValue(4)
        self.sp_seed = QSpinBox(); self.sp_seed.setRange(0, 10 ** 6); self.sp_seed.setValue(2026)
        f2.addRow("阵列几何", self.cb_geom)
        f2.addRow("麦克风数", self.sp_nmic)
        f2.addRow("间距 / 边长", self.sp_edge)
        f2.addRow("坐标", self.te_coords)
        f2.addRow("", self.lb_geom)
        f2.addRow("早/晚分界", self.sp_split)
        f2.addRow("交叉淡化", self.sp_cf)
        f2.addRow("T60 区间 (log 采样)", w_t60)
        f2.addRow("高频阻尼比", w_damp)
        f2.addRow("DRR 区间", w_drr)
        f2.addRow("尾部谱斜率", w_tilt)
        f2.addRow("尾长区间", w_tail)
        f2.addRow("传感器噪声 SNR", w_snr)
        f2.addRow("", self.ck_noise)
        f2.addRow("每条 IR 生成几条", self.sp_num)
        f2.addRow("随机种子", self.sp_seed)
        box2 = QGroupBox("② 混响随机增强")
        box2.setLayout(f2)
        left.addWidget(box2)
        left.addStretch(1)
        root.addLayout(left, 4)

        # ---- 右侧
        right = QVBoxLayout()
        bar = QHBoxLayout()
        self.le_out = QLineEdit()
        self.le_out.setPlaceholderText("输出目录 (默认 <会话>/aug)")
        btn_out = QPushButton("选择…")
        btn_out.clicked.connect(self.pick_out)
        self.btn_run = QPushButton("开始增强")
        self.btn_run.clicked.connect(self.run)
        bar.addWidget(QLabel("输出"))
        bar.addWidget(self.le_out, 1)
        bar.addWidget(btn_out)
        bar.addWidget(self.btn_run)
        right.addLayout(bar)

        self.pb = QProgressBar()
        right.addWidget(self.pb)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(180)
        right.addWidget(self.log)
        self.view = IRView()
        right.addWidget(self.view, 1)
        root.addLayout(right, 5)

        self.cb_geom.currentIndexChanged.connect(self.refresh_geometry)
        self.sp_nmic.valueChanged.connect(self.refresh_geometry)
        self.sp_edge.valueChanged.connect(self.refresh_geometry)
        self.te_coords.textChanged.connect(self.refresh_geometry)
        self.cb_session.currentIndexChanged.connect(self.check_channels)
        self.refresh_geometry()

    # ------------------------------------------------------------------ 数据
    def refresh(self) -> None:
        cur = self.cb_session.currentText()
        self.cb_session.clear()
        names = list_sessions(self.settings.session_root)
        active = self.get_session()
        if active and active.name not in names:
            names.append(active.name)
        self.cb_session.addItems(names)
        if cur in names:
            self.cb_session.setCurrentText(cur)
        elif active:
            self.cb_session.setCurrentText(active.name)

    def check_channels(self) -> None:
        """拿会话里实际的麦克风通道数比对几何, 不一致就当场提示怎么改。"""
        session = self._session()
        if session is None:
            return
        rows = session.load_manifest()
        counts = {len(r.get("mic_cols") or []) for r in rows if r.get("mic_cols")}
        if not counts:
            return
        try:
            from ..rir.rir_augment.geometry import build_array
            n_geom = len(build_array(self._array_cfg()))
        except Exception:
            return
        if counts != {n_geom}:
            got = "/".join(str(c) for c in sorted(counts))
            msg = (f"⚠ 这个会话的 IR 有 {got} 个麦克风通道，当前阵列几何是 {n_geom} 个。"
                   f"把「阵列几何」改成「自定义坐标」并按实物填 {got} 行坐标。")
            if msg != getattr(self, "_last_channel_warning", None):
                self._last_channel_warning = msg
                self.log.append(msg)

    def _session(self) -> Session | None:
        name = self.cb_session.currentText()
        if not name:
            return None
        return Session(self.settings.session_root, name, create=False)

    def _array_cfg(self) -> dict:
        kind = self.cb_geom.currentData()
        if kind == "coords":
            return {"coords": parse_coords(self.te_coords.toPlainText()), "generator": None}
        if kind == "equilateral_triangle":
            return {"coords": None,
                    "generator": {"type": "equilateral_triangle", "edge": self.sp_edge.value()}}
        return {"coords": None,
                "generator": {"type": "linear", "num": self.sp_nmic.value(),
                              "spacing": self.sp_edge.value()}}

    def refresh_geometry(self) -> None:
        """切换几何方式时更新可用控件与摘要。"""
        kind = self.cb_geom.currentData()
        self.te_coords.setEnabled(kind == "coords")
        self.sp_nmic.setEnabled(kind == "linear")
        self.sp_edge.setEnabled(kind in ("equilateral_triangle", "linear"))
        try:
            from ..rir.rir_augment.geometry import build_array, pairwise_distances
            pts = build_array(self._array_cfg())
            d = pairwise_distances(pts)
            self.lb_geom.setText(
                f"当前几何：{len(pts)} 个麦克风，最大间距 {d.max() * 100:.2f} cm，"
                f"最小 {d[d > 0].min() * 100:.2f} cm。"
                "扩散尾的通道间相干性完全由它决定，务必与实物一致。")
        except Exception as e:
            self.lb_geom.setText(f"⚠ 几何无效：{e}")

    def _cfg(self) -> dict:
        cfg = default_config()
        cfg["sample_rate"] = int(self.cb_fs.currentText())
        cfg["seed"] = self.sp_seed.value()
        cfg["array"] = self._array_cfg()
        a = cfg["augment"]
        a["early_late_split_ms"] = self.sp_split.value()
        a["crossfade_ms"] = self.sp_cf.value()
        a["t60"] = {"range": [self.sp_t60a.value(), self.sp_t60b.value()], "sampling": "log"}
        a["hf_damping"] = {"range": [self.sp_dampa.value(), self.sp_dampb.value()], "sampling": "linear"}
        a["drr"] = {"range": [self.sp_drra.value(), self.sp_drrb.value()], "sampling": "linear"}
        a["spectral_tilt_db_per_oct"] = {
            "range": [self.sp_tilta.value(), self.sp_tiltb.value()], "sampling": "linear"}
        a["tail_len_ms"] = [self.sp_taila.value(), self.sp_tailb.value()]
        a["noise"] = {"enable": self.ck_noise.isChecked(),
                      "snr_db": [self.sp_snra.value(), self.sp_snrb.value()]}
        cfg["output"]["num_per_rir"] = self.sp_num.value()
        return cfg

    # ------------------------------------------------------------------ 动作
    def pick_ref(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择参考 IR", "", "WAV (*.wav)")
        if path:
            self.le_ref.setText(path)

    def pick_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.le_out.setText(path)

    def design_eq(self) -> None:
        path = self.le_ref.text().strip()
        if not path or not Path(path).exists():
            QMessageBox.information(self, "去音箱响应", "先选一条参考 IR。")
            return
        try:
            data, fs = sf.read(path, always_2d=True)
            ch = None if self.sp_ref_ch.value() < 0 else self.sp_ref_ch.value()
            freqs, mag = speaker_eq.estimate_response(
                data, fs, window_ms=self.sp_win.value(), channel=ch)
            self.eq_fir = speaker_eq.design_inverse(
                freqs, mag, fs, f_lo=self.sp_eq_lo.value(), f_hi=self.sp_eq_hi.value(),
                max_boost_db=self.sp_boost.value())
            corrected = speaker_eq.apply_inverse(data, self.eq_fir)
            self.view.show_ir(
                np.stack([data[:, ch or 0], corrected[:, ch or 0]], axis=1), fs,
                ["参考 IR", "去响应后"])
            self.log.append(f"反滤波器已生成: {len(self.eq_fir)} 抽头 @ {fs} Hz。"
                            f"频响预览里第二条应比第一条平坦。")
            self.ck_eq.setChecked(True)
        except Exception as e:
            QMessageBox.critical(self, "设计失败", str(e))

    def run(self) -> None:
        session = self._session()
        if session is None:
            QMessageBox.information(self, "增强", "先选一个会话。")
            return
        fs = int(self.cb_fs.currentText())
        accept = ("PASS",) if self.cb_accept.currentIndex() == 0 else ("PASS", "WARN")
        tags = [t.strip() for t in self.le_tags.text().replace("，", ",").split(",") if t.strip()]
        try:
            inputs = select_inputs(session.dir, fs=fs, accept=accept, tags=tags or None)
        except Exception as e:
            QMessageBox.critical(self, "读取输入失败", str(e))
            return
        if not inputs:
            QMessageBox.information(self, "增强", "按当前筛选没有可用的 IR。")
            return

        out_dir = self.le_out.text().strip() or str(session.dir / "aug")
        cfg = self._cfg()
        self.log.append(f"输入 {len(inputs)} 条 IR → 每条生成 {cfg['output']['num_per_rir']} 条, "
                        f"输出 {out_dir}")
        self.pb.setRange(0, len(inputs) * cfg["output"]["num_per_rir"])
        self.btn_run.setEnabled(False)
        eq = self.eq_fir if self.ck_eq.isChecked() else None

        def progress(done: int, total: int, name: str) -> None:
            self.bridge.progress.emit(done, total)

        self._worker = Worker(run_augment, inputs, out_dir, cfg, eq, progress)
        self._worker.done.connect(lambda p: (
            self.btn_run.setEnabled(True),
            self.log.append(f"完成, manifest: {p}")))
        self._worker.failed.connect(lambda e: (
            self.btn_run.setEnabled(True),
            QMessageBox.critical(self, "增强失败", e.splitlines()[0]),
            self.log.append("✗ " + e)))
        self._worker.start()

    def _on_progress(self, done: int, total: int) -> None:
        self.pb.setMaximum(total)
        self.pb.setValue(done)
