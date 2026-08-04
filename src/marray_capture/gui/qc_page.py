"""质检页: 逐位置复核 IR 是否可信, 并生成补录清单。

分组统计那一栏是刻意做的 —— 总体均值会掩盖问题, 真正有用的是「哪一档距离/
哪个标签/哪次佩戴的位置在批量失败」。现场发现一整圈都 FAIL, 才有机会当场补录;
回去再看就来不及了。
"""
from __future__ import annotations

from collections import defaultdict

import soundfile as sf
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from ..settings import AppSettings
from ..store import Session, list_sessions
from ..qc.requantify import requantify_session
from . import notes
from .widgets import IRView, RunnerBridge, Worker, verdict_cell

COLUMNS = ["take_id", "结论", "标签", "距离", "高度", "朝向", "方位",
           "一致性", "电平差 dB", "漂移 ppm", "弥散 ms", "最低 SNR", "最低 DDR", "说明"]


class QCPage(QWidget):
    def __init__(self, settings: AppSettings, get_session, on_rerun, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.get_session = get_session
        self.on_rerun = on_rerun
        self.rows: list[dict] = []
        self._session: Session | None = None
        self._worker: Worker | None = None
        self.bridge = RunnerBridge()
        self._build()
        self.bridge.log.connect(self._on_requantify_log)

    def _build(self) -> None:
        root = QVBoxLayout(self)
        self.notes_btn = notes.build_notes_button("qc", self.settings)

        bar = QHBoxLayout()
        self.cb_session = QComboBox()
        self.cb_filter = QComboBox()
        self.cb_filter.addItems(["全部", "仅 WARN 和 FAIL", "仅 FAIL"])
        btn_refresh = QPushButton("刷新")
        btn_csv = QPushButton("导出 qc.csv")
        btn_rerun = QPushButton("把选中的位置加入补录")
        btn_requse = QPushButton("重新质检")
        btn_open = QPushButton("打开会话目录")
        bar.addWidget(QLabel("会话"))
        bar.addWidget(self.cb_session, 2)
        bar.addWidget(QLabel("筛选"))
        bar.addWidget(self.cb_filter)
        bar.addWidget(btn_refresh)
        bar.addWidget(btn_csv)
        bar.addWidget(btn_rerun)
        bar.addWidget(btn_requse)
        bar.addWidget(btn_open)
        bar.addStretch(1)
        bar.addWidget(self.notes_btn)
        root.addLayout(bar)

        self.lb_stats = QLabel("-")
        self.lb_stats.setWordWrap(True)
        root.addWidget(self.lb_stats)

        mid = QHBoxLayout()
        self.tbl = QTableWidget(0, len(COLUMNS))
        self.tbl.setHorizontalHeaderLabels(COLUMNS)
        self.tbl.horizontalHeader().setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.Stretch)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.itemSelectionChanged.connect(self._on_select)
        mid.addWidget(self.tbl, 5)

        right = QVBoxLayout()
        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setMaximumHeight(190)
        right.addWidget(self.detail)
        self.ir_view = IRView()
        right.addWidget(self.ir_view, 1)
        mid.addLayout(right, 4)
        root.addLayout(mid, 1)

        btn_refresh.clicked.connect(self.refresh)
        btn_csv.clicked.connect(self.export_csv)
        btn_rerun.clicked.connect(self.queue_rerun)
        btn_requse.clicked.connect(self.requantify)
        btn_open.clicked.connect(self.open_dir)
        self.cb_session.currentIndexChanged.connect(self.reload)
        self.cb_filter.currentIndexChanged.connect(self._fill)

    # ------------------------------------------------------------------ 数据
    def refresh(self) -> None:
        cur = self.cb_session.currentText()
        self.cb_session.blockSignals(True)
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
        self.cb_session.blockSignals(False)
        self.reload()

    def reload(self) -> None:
        name = self.cb_session.currentText()
        if not name:
            self.rows = []
            self._fill()
            return
        self._session = Session(self.settings.session_root, name, create=False)
        self.rows = self._session.load_manifest()
        self._fill()
        self._update_stats()

    def _passes_filter(self, verdict: str) -> bool:
        mode = self.cb_filter.currentIndex()
        if mode == 1:
            return verdict in ("WARN", "FAIL")
        if mode == 2:
            return verdict == "FAIL"
        return True

    def _fill(self) -> None:
        shown = [r for r in self.rows
                 if self._passes_filter((r.get("qc") or {}).get("verdict", ""))]
        self.tbl.setRowCount(len(shown))
        self._shown = shown
        for i, r in enumerate(shown):
            qc = r.get("qc") or {}
            chans = qc.get("channels") or []
            snrs = [c.get("rec_snr_db") for c in chans if isinstance(c.get("rec_snr_db"), (int, float))]
            ddrs = [c.get("ir_ddr_db") for c in chans if isinstance(c.get("ir_ddr_db"), (int, float))]
            vals = [
                r.get("take_id", ""), qc.get("verdict", ""), r.get("tag", ""),
                _s(r.get("distance_cm")), _s(r.get("height_cm")), _s(r.get("speaker_deg")),
                r.get("az_label", "") or _s(r.get("az_index")),
                _f(qc.get("repeat_ncc"), 3), _f(qc.get("repeat_level_diff_db")),
                _f(qc.get("drift_ppm"), 0), _f(qc.get("smear_ms"), 2),
                _f(min(snrs) if snrs else None), _f(min(ddrs) if ddrs else None),
                " / ".join(qc.get("reasons") or []),
            ]
            for c, v in enumerate(vals):
                item = verdict_cell(str(v)) if c == 1 else QTableWidgetItem(str(v))
                self.tbl.setItem(i, c, item)
        self.tbl.resizeColumnsToContents()
        self.tbl.horizontalHeader().setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.Stretch)

    def _update_stats(self) -> None:
        if not self.rows:
            self.lb_stats.setText("这个会话还没有数据。")
            return
        counts = defaultdict(int)
        by_group: dict[tuple, list[str]] = defaultdict(list)
        for r in self.rows:
            v = (r.get("qc") or {}).get("verdict", "")
            counts[v] += 1
            key = (r.get("tag", ""), r.get("wearing_id", ""), r.get("distance_cm"))
            by_group[key].append(v)
        bad = []
        for key, vs in sorted(by_group.items(), key=lambda kv: str(kv[0])):
            n_bad = sum(1 for v in vs if v == "FAIL")
            if n_bad and n_bad / len(vs) >= 0.3:
                bad.append(f"{key[0]}/{key[1]}/{_s(key[2])}cm: {n_bad}/{len(vs)} FAIL")
        from . import theme
        text = (f"共 {len(self.rows)} 个位置 —— "
                f"<b style='color:{theme.OK}'>✓ {counts['PASS']}</b>　"
                f"<b style='color:{theme.WARN}'>! {counts['WARN']}</b>　"
                f"<b style='color:{theme.BAD}'>✕ {counts['FAIL']}</b>")
        if bad:
            text += (f"<br><b style='color:{theme.BAD}'>整组失败，优先当场补录：</b> "
                     + "；".join(bad))
        self.lb_stats.setText(text)

    # ------------------------------------------------------------------ 交互
    def _on_select(self) -> None:
        rows = {i.row() for i in self.tbl.selectedIndexes()}
        if not rows or self._session is None:
            return
        r = self._shown[min(rows)]
        qc = r.get("qc") or {}
        lines = [f"<b>{r.get('take_id','')}</b> — {qc.get('verdict','')} "
                 f"(第 {r.get('attempt', 1)} 次, 平均 {r.get('n_averaged', 1)} 条)",
                 f"延迟 {_f(qc.get('latency_ms'))} ms, 漂移 {_f(qc.get('drift_ppm'), 0)} ppm, "
                 f"弥散 {_f(qc.get('smear_ms'), 2)} ms",
                 "<table cellpadding=3><tr><th>通道</th><th>峰值 dBFS</th><th>录音 SNR</th>"
                 "<th>IR 峰</th><th>噪底</th><th>DDR</th><th>可靠带宽</th><th>相对电平</th></tr>"]
        for c in qc.get("channels") or []:
            lines.append(
                f"<tr><td>{c.get('label','')}</td><td>{_f(c.get('peak_dbfs'))}</td>"
                f"<td>{_f(c.get('rec_snr_db'))}</td><td>{_f(c.get('ir_peak_db'))}</td>"
                f"<td>{_f(c.get('ir_noise_db'))}</td><td>{_f(c.get('ir_ddr_db'))}</td>"
                f"<td>{_f(c.get('reliable_bw_hz'), 0)}</td><td>{_f(c.get('rel_level_db'))}</td></tr>")
        lines.append("</table>")
        if qc.get("reasons"):
            lines.append("<b>问题:</b> " + " / ".join(qc["reasons"]))
        self.detail.setHtml("<br>".join(lines))

        rel = r.get("ir_file")
        if rel:
            p = self._session.dir / rel
            if p.exists():
                try:
                    data, fs = sf.read(str(p), always_2d=True)
                    self.ir_view.show_ir(data, fs, r.get("channels"))
                except Exception as e:
                    self.detail.append(f"<br>读取 IR 失败: {e}")

    def selected_take_ids(self) -> set[str]:
        rows = {i.row() for i in self.tbl.selectedIndexes()}
        out = set()
        for i in rows:
            r = self._shown[i]
            # 补录要用方案里的原始 take_id (去掉重试后缀)
            tid = r.get("take_id", "")
            out.add(tid.split("_try")[0])
        return out

    def queue_rerun(self) -> None:
        ids = self.selected_take_ids()
        if not ids:
            QMessageBox.information(self, "补录", "先在表里选中要补录的行。")
            return
        if QMessageBox.question(
            self, "补录",
            f"将只跑这 {len(ids)} 个位置 (方案里的调整步仍会播报)。现在切到采集页开始吗?",
        ) == QMessageBox.Yes:
            self.on_rerun(ids)

    def requantify(self) -> None:
        """用最新 QC 逻辑对已落盘的录音重跑一遍, 改判旧 manifest + 重存 IR。

        修过 QC 逻辑 (如 onset 误锁、VPU 弱信号误判) 之后, 旧 manifest 上的
        判据是过时的。这里从 raw 重跑, 把误判的位置救回来。选中行只重跑选中
        的, 没选则重跑全部 (PASS 的重跑不变, 幂等)。
        """
        if self._session is None:
            return
        import json
        sess_path = self._session.dir / "session.json"
        if not sess_path.exists():
            QMessageBox.warning(self, "重新质检", "session.json 不在, 拿不到配置。")
            return
        selected = self.selected_take_ids()
        if selected:
            scope = f"选中的 {len(selected)} 个位置"
            take_ids = selected
        else:
            n_fail = sum(1 for r in self.rows if (r.get("qc") or {}).get("verdict") == "FAIL")
            n_warn = sum(1 for r in self.rows if (r.get("qc") or {}).get("verdict") == "WARN")
            scope = f"全部 {len(self.rows)} 个位置 (含 {n_fail} FAIL / {n_warn} WARN)"
            take_ids = None
        if QMessageBox.question(
            self, "重新质检",
            f"对 {scope} 用最新质检逻辑重跑 (从 raw 录音重算, 会覆盖 IR 与 manifest)。\n"
            "重跑期间别动会话目录。现在开始吗?",
        ) != QMessageBox.Yes:
            return

        settings = json.loads(sess_path.read_text(encoding="utf-8")).get("settings", {})
        self.detail.setPlainText(f"重新质检中… ({scope})")
        for w in self.findChildren(QPushButton):
            w.setEnabled(False)

        def log_cb(msg: str) -> None:
            self.bridge.log.emit(msg)

        self._worker = Worker(requantify_session, str(self._session.dir),
                              take_ids, settings, log=log_cb)
        self._worker.done.connect(self._on_requantify_done)
        self._worker.failed.connect(self._on_requantify_failed)
        self._worker.start()

    def _on_requantify_log(self, msg: str) -> None:
        self.detail.append(msg)

    def _on_requantify_done(self, changed: list) -> None:
        for w in self.findChildren(QPushButton):
            w.setEnabled(True)
        n_flip = sum(1 for _o, n in changed if n in ("PASS", "WARN") and _o == "FAIL")
        self.detail.append(f"完成: 重跑 {len(changed)} 个, 其中 {n_flip} 个 FAIL 改判通过。")
        self.reload()

    def _on_requantify_failed(self, err: str) -> None:
        for w in self.findChildren(QPushButton):
            w.setEnabled(True)
        QMessageBox.critical(self, "重新质检", err.splitlines()[0])
        self.detail.append("✗ " + err)

    def export_csv(self) -> None:
        if self._session is None:
            return
        p = self._session.write_qc_csv()
        QMessageBox.information(self, "导出", f"已写出 {p}")

    def open_dir(self) -> None:
        if self._session is None:
            return
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._session.dir)))


def _s(v) -> str:
    return "" if v is None else str(v)


def _f(v, nd: int = 1) -> str:
    if not isinstance(v, (int, float)) or v != v:
        return "-"
    return f"{v:.{nd}f}"
