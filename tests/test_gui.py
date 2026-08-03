"""界面层的回归测试。离屏运行, 不需要显示器也不碰声卡。"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication          # noqa: E402

from marray_capture.gui import theme                # noqa: E402
from marray_capture.settings import AppSettings     # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    theme.apply(app)
    return app


@pytest.fixture
def settings(tmp_path):
    s = AppSettings()
    s.session_root = str(tmp_path)
    return s


def test_device_page_does_not_clobber_settings(qapp, settings):
    """构造设备页不能改动配置。

    回填控件时每个 setValue 都会触发 valueChanged, 那时其余控件还停在 Qt 默认值上;
    没有防护的话 _push 会把默认值写回去 —— 输出增益 -6 dB 变 0 dB (扫频响 6 dB),
    输出通道数 2 变 1 (只出一个声道)。
    """
    from marray_capture.gui.device_page import DevicePage

    before = settings.to_dict()["audio"]
    DevicePage(settings, lambda: None)
    qapp.processEvents()
    after = settings.to_dict()["audio"]

    for key in ("output_channels", "output_gain_db", "duplex_mode",
                "input_latency", "output_latency", "blocksize"):
        assert after[key] == before[key], f"{key} 被构造过程改掉了"


def test_channel_summary_and_meter_populated_on_startup(qapp, settings, monkeypatch):
    """加载守卫不能把通道摘要和电平表也一起挡掉。"""
    from marray_capture.audio import devices as dev
    from marray_capture.gui.device_page import DevicePage

    fake = dev.DeviceInfo(index=0, name="4ch", hostapi="ASIO",
                          max_input=4, max_output=4, default_samplerate=48000.0)
    for name in ("list_devices", "input_devices", "output_devices"):
        monkeypatch.setattr(dev, name, lambda f=fake: [f])
    monkeypatch.setattr(dev, "describe", lambda i: fake if i == 0 else None)

    page = DevicePage(settings, lambda: None)
    qapp.processEvents()
    assert settings.audio.channels, "启动后通道表应已写回配置"
    assert "IR 通道顺序" in page.lb_chsum.text()
    assert page.meter._bars, "电平表应已按通道建好"


def test_quick_assign_builds_channel_map(qapp, settings, monkeypatch):
    """8 通道卡上一键分配 4 麦 + VPU。"""
    from marray_capture.audio import devices as dev
    from marray_capture.gui.device_page import DevicePage

    fake = dev.DeviceInfo(index=0, name="8ch card", hostapi="ASIO",
                          max_input=8, max_output=8, default_samplerate=48000.0)
    monkeypatch.setattr(dev, "list_devices", lambda: [fake])
    monkeypatch.setattr(dev, "input_devices", lambda: [fake])
    monkeypatch.setattr(dev, "output_devices", lambda: [fake])
    monkeypatch.setattr(dev, "describe", lambda i: fake if i == 0 else None)

    page = DevicePage(settings, lambda: None)
    qapp.processEvents()
    assert page.tbl.rowCount() == 8

    page.sp_nmic.setValue(4)
    page.ck_vpu.setChecked(True)
    page.quick_assign()

    a = settings.audio
    assert a.mic_indices() == [0, 1, 2, 3]
    assert a.role_indices("vpu") == [4]
    assert a.n_record_channels() == 5
    assert "IR 通道顺序" in page.lb_chsum.text()
    # 输入输出同为 ASIO 设备 → 必须走全双工
    assert "全双工" in page.lb_mode.text()


def test_channel_table_supports_scrambled_wiring(qapp, settings, monkeypatch):
    """麦克风挂在任意物理通道上, 顺序由「麦序号」决定。"""
    from marray_capture.audio import devices as dev
    from marray_capture.gui.device_page import DevicePage

    fake = dev.DeviceInfo(index=0, name="8ch", hostapi="ASIO",
                          max_input=8, max_output=8, default_samplerate=48000.0)
    for name in ("list_devices", "input_devices", "output_devices"):
        monkeypatch.setattr(dev, name, lambda f=fake: [f])
    monkeypatch.setattr(dev, "describe", lambda i: fake if i == 0 else None)

    page = DevicePage(settings, lambda: None)
    page.clear_assign()

    # mic1→ch6, mic2→ch3, mic3→ch2, mic4→ch5, vpu→ch7 (界面上是 0-based 行号)
    wiring = [(5, "mic", 1, "mic1"), (2, "mic", 2, "mic2"), (1, "mic", 3, "mic3"),
              (4, "mic", 4, "mic4"), (6, "vpu", 1, "vpu")]
    for row, role, order, label in wiring:
        ck, btn, sp, le = page._row_widgets(row)
        ck.setChecked(True)
        btn.set_role(role)
        sp.setValue(order)
        le.setText(label)
    page._push_channels()
    qapp.processEvents()

    a = settings.audio
    assert a.mic_indices() == [5, 2, 1, 4]
    assert a.active_indices() == [5, 2, 1, 4, 6]
    assert a.n_record_channels() == 7
    assert "mic1" in page.lb_chsum.text() and "ch6" in page.lb_chsum.text()


def test_role_toggle_cycles(qapp):
    from marray_capture.gui.widgets import RoleToggle

    t = RoleToggle("mic")
    seen = [t.role()]
    for _ in range(3):
        t.click()
        seen.append(t.role())
    assert seen == ["mic", "vpu", "ref", "mic"]
    t.set_role("vpu")
    assert t.role() == "vpu" and "VPU" in t.text()


def test_post_page_custom_geometry(qapp, settings):
    from marray_capture.gui.post_page import PostPage

    page = PostPage(settings, lambda: None)
    idx = page.cb_geom.findData("coords")
    page.cb_geom.setCurrentIndex(idx)
    page.te_coords.setPlainText(
        "0.005,0.005,0\n0.005,-0.005,0\n-0.005,-0.005,0\n-0.005,0.005,0")
    qapp.processEvents()

    cfg = page._array_cfg()
    assert len(cfg["coords"]) == 4
    assert "4 个麦克风" in page.lb_geom.text()
    assert page.te_coords.isEnabled()

    page.cb_geom.setCurrentIndex(page.cb_geom.findData("equilateral_triangle"))
    qapp.processEvents()
    assert "3 个麦克风" in page.lb_geom.text()
    assert not page.te_coords.isEnabled()


def test_instruction_card_verdict_rail(qapp):
    """判定色条要真的换色, 且永远带一个字符 —— 不能只靠颜色。"""
    from marray_capture.gui.widgets import InstructionCard

    card = InstructionCard()
    card.show_step("S01_A03", "向右转一格，大约 30 度。")
    assert card.text.text().startswith("向右转")
    assert card.text.objectName() == "display"      # 字号靠 QSS, 不是 setFont

    for verdict, glyph in [("PASS", "✓"), ("WARN", "!"), ("FAIL", "✕")]:
        card.show_verdict(verdict)
        assert glyph in card.status.text()
        assert theme.VERDICT[verdict][0].lower() in card.rail.styleSheet().lower()


def test_display_font_size_survives_global_stylesheet(qapp):
    """全局 QSS 的通配字号不能把大字指令压回正文大小。"""
    from marray_capture.gui.widgets import InstructionCard

    card = InstructionCard()
    card.setStyleSheet(theme.stylesheet())
    qapp.processEvents()
    assert card.text.font().pointSize() >= theme.FS_LARGE


def test_every_page_has_a_notes_button(qapp, settings):
    """五页都要有注意事项按钮, 悬停能看到内容。"""
    import tempfile

    from marray_capture.gui.main_window import MainWindow

    settings.session_root = tempfile.mkdtemp()
    w = MainWindow(settings)
    pages = {
        "device": w.page_device, "plan": w.page_plan, "run": w.page_run,
        "qc": w.page_qc, "post": w.page_post,
    }
    for name, page in pages.items():
        btn = getattr(page, "notes_btn", None)
        assert btn is not None, f"{name} 页没有注意事项按钮"
        assert btn._notes, f"{name} 页的注意事项是空的"
        assert "注意事项" in btn.text()
        assert "必看" in btn.toolTip()          # 悬停就能读到


def test_notes_dialog_opens_and_updates(qapp, settings, monkeypatch):
    from marray_capture.audio import devices as dev
    from marray_capture.gui import notes

    btn = notes.build_notes_button("device", settings)
    btn.open_dialog()
    assert btn._dialog is not None and btn._dialog.isVisible()
    before = btn._dialog.label.text()

    asio = dev.DeviceInfo(0, "ASIO4ALL", "ASIO", 8, 8, 48000.0)
    monkeypatch.setattr(dev, "describe", lambda i: asio if i == 0 else None)
    settings.audio.input_device = settings.audio.output_device = 0
    btn.refresh()
    assert "单实例独占" in btn._dialog.label.text()
    assert btn._dialog.label.text() != before
    btn._dialog.close()


def test_pages_are_scrollable(qapp, settings):
    """小屏上内容超出必须能滚, 不能被截断。"""
    import tempfile

    from PySide6.QtWidgets import QScrollArea

    from marray_capture.gui.main_window import MainWindow

    settings.session_root = tempfile.mkdtemp()
    w = MainWindow(settings)
    assert w.width() <= 1280 and w.height() <= 820, "默认窗口对 1440p 笔记本来说太大"
    for i in range(w.tabs.count()):
        area = w.tabs.widget(i)
        assert isinstance(area, QScrollArea), f"第 {i} 页没有套滚动区"
        assert area.widgetResizable()


def test_param_help_covers_plan_widgets(qapp, settings):
    """参数说明要覆盖到方案页的控件, 并真的挂成 tooltip。"""
    from marray_capture.gui import notes
    from marray_capture.gui.plan_page import PlanPage

    page = PlanPage(settings, lambda plan, want_session=False: None)
    missing = [attr for attr, *_ in notes.PARAMS if getattr(page, attr, None) is None]
    assert not missing, f"参数说明指向了不存在的控件: {missing}"
    for attr, _, name, _ in notes.PARAMS:
        assert name in getattr(page, attr).toolTip()

    # 关键参数不能漏
    covered = {name for _, _, name, _ in notes.PARAMS}
    for must in ("每位置扫几次", "扫频前静音（噪声窗）", "重戴次数", "基准圈格数", "播放幅度"):
        assert must in covered

    page.show_param_help()
    assert page._param_dialog.isVisible()
    assert "每位置扫几次" in page._param_dialog.label.text()
    page._param_dialog.close()


def test_device_notes_follow_config(qapp, settings, monkeypatch):
    """ASIO 的坑只在选了 ASIO 时提示, 蓝牙的坑只在跨设备时提示。"""
    from marray_capture.audio import devices as dev
    from marray_capture.gui import notes

    asio = dev.DeviceInfo(0, "ASIO4ALL", "ASIO", 8, 8, 48000.0)
    bt = dev.DeviceInfo(1, "BT speaker", "Core Audio", 0, 2, 48000.0)
    monkeypatch.setattr(dev, "describe", lambda i: {0: asio, 1: bt}.get(i))

    def texts(**kw):
        settings.audio.input_device = kw.get("i")
        settings.audio.output_device = kw.get("o")
        return " ".join(n.text for n in notes.notes_for("device", settings))

    same_asio = texts(i=0, o=0)
    assert "单实例独占" in same_asio
    assert "立体声" not in same_asio          # 同设备不该提蓝牙

    cross = texts(i=0, o=1)
    assert "立体声" in cross and "HFP" in cross

    none_selected = texts()
    assert "单实例独占" not in none_selected


def test_notes_sorted_critical_first(settings):
    from marray_capture.gui import notes

    for page in notes.PAGES:
        items = notes.notes_for(page, settings)
        assert items, f"{page} 没有条目"
        levels = [n.level for n in items]
        rank = {"critical": 0, "warn": 1, "info": 2}
        assert levels == sorted(levels, key=lambda x: rank[x]), f"{page} 没按轻重排序"
        assert any(n.level == "critical" for n in items), f"{page} 应至少有一条必看"


def test_main_window_builds_all_tabs(qapp, settings):
    from marray_capture.gui.main_window import MainWindow

    settings.session_root = tempfile.mkdtemp()
    w = MainWindow(settings)
    for i in range(w.tabs.count()):
        w.tabs.setCurrentIndex(i)
        qapp.processEvents()
    assert w.tabs.count() == 5

    w.page_plan.generate()
    qapp.processEvents()
    assert len(w.plan.measures) > 0
    assert w.page_plan.tbl.rowCount() == len(w.plan.steps)


def test_generated_param_doc_is_in_sync():
    """docs/protocol-params.md 必须与 notes.PARAMS 同步。

    它由 `marray-capture --dump-params` 生成 —— 手改那个 md 会被下次生成覆盖，
    所以这里卡一道, 提醒改的是 PARAMS 而不是文档。
    """
    from pathlib import Path

    from marray_capture.gui import notes

    doc = Path(__file__).resolve().parents[1] / "docs" / "protocol-params.md"
    assert doc.exists(), "docs/protocol-params.md 不存在"
    expected = notes.params_markdown().rstrip() + "\n"
    actual = doc.read_text(encoding="utf-8").rstrip() + "\n"
    assert actual == expected, (
        "docs/protocol-params.md 与 notes.PARAMS 不同步。"
        "运行: uv run marray-capture --dump-params > docs/protocol-params.md")


def test_agent_docs_exist():
    """AGENTS.md 是给后续 agent 的说明, CLAUDE.md 指向它。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    claude = (root / "CLAUDE.md").read_text(encoding="utf-8")
    assert "AGENTS.md" in claude, "CLAUDE.md 应指向 AGENTS.md"
    # 那些会静默出错的地方必须在里面写清楚
    for topic in ("麦序号", "map_levels", "guard", "clicked", "QSS", "WASAPI"):
        assert topic in agents, f"AGENTS.md 少了 {topic} 这一条"
