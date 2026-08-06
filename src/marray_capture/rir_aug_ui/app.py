"""入口。

    uv run rir-augment-ui        启动独立增强 GUI
"""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ..gui import theme
from .main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    theme.apply(app)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
