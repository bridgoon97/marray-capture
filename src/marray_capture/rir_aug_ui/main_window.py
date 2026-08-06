"""独立增强工具的界面。

布局与 gui/post_page.py 的增强部分同源 (同一个 theme + 同一组控件), 差异:
- 去掉会话 / 质检门槛 / 去音箱响应 —— 入口是一个裸 wav 目录。
- 早/晚分界默认 20 ms (这批自说话 IR 只有几百样本, 50 ms 默认比 IR 还长,
  split_early_late 本就会截到 IR 长度内, 但默认值压低更顺)。
- 每条 IR 默认生成 10 条。
- 左侧加一个文件列表, 点其中一条即时预览输入 IR。
"""
from __future__ import annotations

from pathlib import Path

import soundfile as sf
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

from ..gui.widgets import IRView, Worker
from ..rir.augment_runner import default_config, parse_coords
from ..rir.rir_augment.geometry import build_array, pairwise_distances
from .batch import channel_counts, list_inputs, run_augment_dir


def _range_row(lo: float, hi: float, step: float, suffix: str = "") -> tuple[QWidget, QDoubleSpinBox, QDoubleSpinBox]:
    """[lo ~ hi] 两个 spinbox 一行。从 post_page 拷过来, 不去动主包 widgets。"""
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


class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("混响随机增强 · 独立版")
        self.resize(1180, 760)
        self._worker: Worker | None = None
        self._paths: list[Path] = []
        self._build()
        self.refresh_geometry()

    # ------------------------------------------------------------------ 布局
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        # ---- 输入目录
        bar = QHBoxLayout()
        self.le_in = QLineEdit()
        self.le_in.setPlaceholderText("放自说话 IR 的目录 (里头一堆 .wav)")
        btn_in = QPushButton("选择…")
        btn_in.clicked.connect(lambda: self.pick_dir(self.le_in, "选择 IR 目录"))
        self.btn_scan = QPushButton("扫描")
        self.btn_scan.clicked.connect(self.scan)
        bar.addWidget(QLabel("输入目录"))
        bar.addWidget(self.le_in, 1)
        bar.addWidget(btn_in)
        bar.addWidget(self.btn_scan)
        outer.addLayout(bar)

        body = QHBoxLayout()
        outer.addLayout(body, 1)
        left = QVBoxLayout()

        # ---- 文件列表 + 摘要
        self.lb_summary = QLabel("尚未扫描")
        self.lb_summary.setObjectName("hint")
        self.lw_files = QListWidget()
        self.lw_files.setMaximumWidth(300)
        self.lw_files.setAlternatingRowColors(True)
        self.lw_files.currentRowChanged.connect(self.preview_input)
        box_f = QGroupBox("输入文件")
        bf = QVBoxLayout(box_f)
        bf.addWidget(self.lb_summary)
        bf.addWidget(self.lw_files, 1)
        left.addWidget(box_f)

        # ---- 阵列几何
        fg = QFormLayout()
        self.cb_geom = QComboBox()
        for key, label in [
            ("equilateral_triangle", "等边三角形 (3 麦)"),
            ("linear", "线阵 (等间距)"),
            ("coords", "自定义坐标 (任意麦克风数)"),
        ]:
            self.cb_geom.addItem(label, key)
        self.sp_nmic = QSpinBox(); self.sp_nmic.setRange(2, 32); self.sp_nmic.setValue(3)
        self.sp_edge = QDoubleSpinBox(); self.sp_edge.setRange(0.001, 1.0)
        self.sp_edge.setDecimals(4); self.sp_edge.setValue(0.01); self.sp_edge.setSuffix(" m")
        self.te_coords = QPlainTextEdit()
        self.te_coords.setPlaceholderText(
            "每行一个麦克风的 x, y, z，单位米。例如 4 麦 1cm 正方形：\n"
            "0.005,  0.005, 0\n0.005, -0.005, 0\n-0.005, -0.005, 0\n-0.005,  0.005, 0")
        self.te_coords.setMaximumHeight(110)
        self.lb_geom = QLabel("-")
        self.lb_geom.setWordWrap(True)
        self.lb_geom.setObjectName("hint")
        fg.addRow("阵列几何", self.cb_geom)
        fg.addRow("麦克风数", self.sp_nmic)
        fg.addRow("间距 / 边长", self.sp_edge)
        fg.addRow("坐标", self.te_coords)
        fg.addRow("", self.lb_geom)
        box_g = QGroupBox("阵列几何")
        box_g.setLayout(fg)
        left.addWidget(box_g)

        # ---- 增强参数
        fa = QFormLayout()
        # 这批 IR 很短 (几百样本 ≈ 10~30 ms), 50 ms 默认比整条还长, 压到 20 ms。
        self.sp_split = QDoubleSpinBox(); self.sp_split.setRange(1, 200)
        self.sp_split.setValue(20); self.sp_split.setSuffix(" ms")
        self.sp_cf = QDoubleSpinBox(); self.sp_cf.setRange(0, 50)
        self.sp_cf.setValue(5); self.sp_cf.setSuffix(" ms")
        w_t60, self.sp_t60a, self.sp_t60b = _range_row(0.2, 0.8, 0.05, " s")
        w_damp, self.sp_dampa, self.sp_dampb = _range_row(0.4, 0.9, 0.05)
        w_drr, self.sp_drra, self.sp_drrb = _range_row(0.0, 15.0, 1.0, " dB")
        w_tilt, self.sp_tilta, self.sp_tiltb = _range_row(-3.0, 1.0, 0.5, " dB/oct")
        w_tail, self.sp_taila, self.sp_tailb = _range_row(300, 1200, 50, " ms")
        w_snr, self.sp_snra, self.sp_snrb = _range_row(40, 60, 1, " dB")
        self.ck_noise = QCheckBox("叠加通道间不相干传感器噪声")
        self.ck_noise.setChecked(True)
        # 用户要每条 IR 加 10 个混响尾。
        self.sp_num = QSpinBox(); self.sp_num.setRange(1, 64); self.sp_num.setValue(10)
        self.sp_seed = QSpinBox(); self.sp_seed.setRange(0, 10 ** 6); self.sp_seed.setValue(2026)
        fa.addRow("早/晚分界", self.sp_split)
        fa.addRow("交叉淡化", self.sp_cf)
        fa.addRow("T60 区间 (log 采样)", w_t60)
        fa.addRow("高频阻尼比", w_damp)
        fa.addRow("DRR 区间", w_drr)
        fa.addRow("尾部谱斜率", w_tilt)
        fa.addRow("尾长区间", w_tail)
        fa.addRow("传感器噪声 SNR", w_snr)
        fa.addRow("", self.ck_noise)
        fa.addRow("每条 IR 生成几条", self.sp_num)
        fa.addRow("随机种子", self.sp_seed)
        box_a = QGroupBox("增强参数")
        box_a.setLayout(fa)
        left.addWidget(box_a)
        left.addStretch(1)

        body.addLayout(left, 4)

        # ---- 右侧: 运行 + 预览
        right = QVBoxLayout()
        rbar = QHBoxLayout()
        self.le_out = QLineEdit()
        self.le_out.setPlaceholderText("输出目录")
        btn_out = QPushButton("选择…")
        btn_out.clicked.connect(lambda: self.pick_dir(self.le_out, "选择输出目录"))
        self.ck_norm = QCheckBox("全局归一化")
        self.ck_norm.setToolTip(
            "取整批所有增强 IR 的全局峰值, 用同一个 0.98/peak 标量缩放全部输出。\n"
            "通道间/位置间的相对电平完全保留 (单一公共标量), 不影响下游 ILD/DRR。")
        self.btn_run = QPushButton("开始增强")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self.run)
        rbar.addWidget(QLabel("输出"))
        rbar.addWidget(self.le_out, 1)
        rbar.addWidget(btn_out)
        rbar.addWidget(self.ck_norm)
        rbar.addWidget(self.btn_run)
        right.addLayout(rbar)

        self.pb = QProgressBar()
        right.addWidget(self.pb)
        self.log = QTextEdit()
        self.log.setObjectName("log")
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        right.addWidget(self.log)
        self.view = IRView()
        right.addWidget(self.view, 1)
        body.addLayout(right, 5)

        # 几何联动
        self.cb_geom.currentIndexChanged.connect(self.refresh_geometry)
        self.sp_nmic.valueChanged.connect(self.refresh_geometry)
        self.sp_edge.valueChanged.connect(self.refresh_geometry)
        self.te_coords.textChanged.connect(self.refresh_geometry)

    # ------------------------------------------------------------------ 数据
    def pick_dir(self, line: QLineEdit, title: str) -> None:
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            line.setText(path)

    def scan(self) -> None:
        d = self.le_in.text().strip()
        if not d:
            QMessageBox.information(self, "扫描", "先选一个输入目录。")
            return
        try:
            self._paths = list_inputs(d)
        except Exception as e:
            QMessageBox.critical(self, "扫描失败", str(e))
            return
        self.lw_files.clear()
        for p in self._paths:
            self.lw_files.addItem(p.name)
        cch = channel_counts(self._paths)
        n = len(self._paths)
        chs = " / ".join(f"{k}通道×{v}" for k, v in sorted(cch.items()))
        self.lb_summary.setText(f"{n} 条 wav · {chs}")
        self.log.append(f"扫描到 {n} 条: {chs}")
        self._check_channels(cch)

    def _check_channels(self, cch: dict[int, int]) -> None:
        """扫描完先比对通道数和几何, 不一致在跑之前提示。"""
        if not cch:
            return
        try:
            n_geom = len(build_array(self._array_cfg()))
        except Exception:
            return
        if set(cch) != {n_geom}:
            got = " / ".join(str(k) for k in sorted(cch))
            self.log.append(
                f"⚠ 文件通道数 {got}, 当前几何是 {n_geom} 个麦克风。"
                f"把「阵列几何」改成「自定义坐标」并按实物填 {got} 行坐标。")

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
        kind = self.cb_geom.currentData()
        self.te_coords.setEnabled(kind == "coords")
        self.sp_nmic.setEnabled(kind == "linear")
        self.sp_edge.setEnabled(kind in ("equilateral_triangle", "linear"))
        try:
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
        # 自说话 IR 固定 16k; 留这个键, batch 不校验来源 fs 之外的事。
        cfg["sample_rate"] = 16000
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
        cfg["output"]["global_norm"] = self.ck_norm.isChecked()
        return cfg

    # ------------------------------------------------------------------ 动作
    def preview_input(self, row: int) -> None:
        if row < 0 or row >= len(self._paths):
            return
        try:
            data, fs = sf.read(str(self._paths[row]), always_2d=True)
        except Exception as e:
            self.log.append(f"✗ 读 {self._paths[row].name} 失败: {e}")
            return
        self.view.show_ir(data, fs, [f"ch{c + 1}" for c in range(data.shape[1])])

    def run(self) -> None:
        d = self.le_in.text().strip()
        if not d or not self._paths:
            QMessageBox.information(self, "增强", "先扫描输入目录。")
            return
        out_dir = self.le_out.text().strip()
        if not out_dir:
            QMessageBox.information(self, "增强", "先选输出目录。")
            return
        cfg = self._cfg()
        n_per = cfg["output"]["num_per_rir"]
        self.pb.setRange(0, len(self._paths) * n_per)
        self.pb.setValue(0)
        self.btn_run.setEnabled(False)
        self.log.append(
            f"输入 {len(self._paths)} 条 → 每条 {n_per} 条, 输出 {out_dir}"
            + ("，全局归一化(全批单一标量)" if cfg["output"]["global_norm"] else ""))

        def progress(done: int, total: int, name: str) -> None:
            self.pb.setMaximum(total)
            self.pb.setValue(done)

        self._worker = Worker(run_augment_dir, d, out_dir, cfg, progress)

        def done(mf: str) -> None:
            self.btn_run.setEnabled(True)
            self.log.append(f"完成, manifest: {mf}")
            # 跑完预览第一条输出, 直观看到尾拼上去了。
            out_path = Path(out_dir)
            outs = sorted(out_path.glob("*.wav"))
            if outs:
                data, fs = sf.read(str(outs[0]), always_2d=True)
                self.view.show_ir(data, fs, [f"ch{c + 1}" for c in range(data.shape[1])])
                self.log.append(f"已预览输出: {outs[0].name}")

        def failed(e: str) -> None:
            self.btn_run.setEnabled(True)
            QMessageBox.critical(self, "增强失败", e.splitlines()[0])
            self.log.append("✗ " + e)

        self._worker.done.connect(done)
        self._worker.failed.connect(failed)
        self._worker.start()
