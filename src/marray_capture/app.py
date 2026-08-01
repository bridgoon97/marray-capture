"""入口。

    uv run marray-capture          启动 GUI
    uv run marray-capture --check  只做环境自检 (不开窗口), 用来在新机器上排查依赖
"""
from __future__ import annotations

import sys


def _self_check() -> int:
    ok = True
    print("=== marray-capture 环境自检 ===")
    try:
        import numpy, scipy, soundfile        # noqa: F401
        print("✓ numpy / scipy / soundfile")
    except Exception as e:
        ok = False
        print(f"✗ 科学计算依赖: {e}")
    try:
        import sounddevice as sd
        print(f"✓ sounddevice {sd.__version__}")
        ins = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
        outs = [d for d in sd.query_devices() if d["max_output_channels"] > 0]
        print(f"  输入设备 {len(ins)} 个, 输出设备 {len(outs)} 个")
        for d in ins:
            print(f"   [in ] {d['name']}  ({d['max_input_channels']} ch)")
        for d in outs:
            print(f"   [out] {d['name']}  ({d['max_output_channels']} ch)")
    except Exception as e:
        ok = False
        print(f"✗ sounddevice: {e}")
    try:
        import PySide6, pyqtgraph             # noqa: F401
        print("✓ PySide6 / pyqtgraph")
    except Exception as e:
        ok = False
        print(f"✗ 界面依赖: {e}")
    print("--- 语音导播后端 ---")
    try:
        from .audio import tts
        for name in tts.BACKEND_ORDER:
            b = tts.make_backend(name)
            if b.available():
                vs = b.voices()
                print(f"✓ {name}: 可用, 默认语音 {b.default_voice() or '(系统默认)'}")
                for v in vs[:6]:
                    print(f"   {v.label}  ({v.id})")
            else:
                print(f"· {name}: {getattr(b, 'last_error', '不可用')}")
        chosen, notes = tts.pick_backend("auto")
        if chosen is None:
            print("⚠ 没有可用 TTS —— 采集会降级为提示音 + 屏幕大字, 流程不受影响。")
            print("  想要中文语音: uv sync --extra tts, 再在界面上点「下载中文语音模型」。")
        else:
            print(f"→ 自动选择: {chosen.name}")
    except Exception as e:
        print(f"⚠ TTS 检查失败: {e}")
    try:
        from .rir.rir_augment.augment import augment_rir   # noqa: F401
        print("✓ vendored rir-augment")
    except Exception as e:
        ok = False
        print(f"✗ rir-augment: {e}")
    print("=== 自检" + ("通过" if ok else "有问题") + " ===")
    return 0 if ok else 1


def main() -> int:
    if "--check" in sys.argv:
        return _self_check()

    from PySide6.QtWidgets import QApplication

    from .gui.main_window import MainWindow
    from .settings import AppSettings

    app = QApplication(sys.argv)
    app.setApplicationName("marray-capture")
    win = MainWindow(AppSettings.load())
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
