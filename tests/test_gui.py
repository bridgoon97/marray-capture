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
