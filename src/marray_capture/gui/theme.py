"""视觉规范。

**这是什么东西的界面**: 一台声学测量仪器。操作者大部分时间不在电脑前 —— 他戴着
待测耳机、坐在房间另一头的转椅上。所以采集页的唯一任务是: **三米开外能读懂
"现在该做什么" 和 "上一条过没过"**。其余页面是坐下来配置和复核用的, 可以密。

**取向**: 参照测量报告 / 校准图表, 不是控制台也不是网页后台。
- 界面底色是**纸**(暖白), 数据表面是**纯白** —— 数据像印在纸上, 层级自然分开。
- 字色是**墨**(近黑非纯黑), 主色是**仪表蓝**, 判定用**三态量规色**。
- **所有测得的数字一律等宽字体**。等宽的表格数字在实时刷新时不会左右跳动,
  列也能对齐 —— 这是仪器读数该有的样子, 也是这套界面唯一的字体"个性"。
  正文用系统人文无衬线, 与读数形成对比。

**招牌元素**: **判定色条** —— 采集页指令卡左侧和质检表每行左侧的一道 4px 竖条,
颜色随 PASS/WARN/FAIL 变化。它不靠颜色单打独斗, 永远配一个字符 (✓ ! ✕),
所以色觉障碍和隔着三米看都成立。整套界面只在这一处用饱和色, 其余保持安静。
"""
from __future__ import annotations

import sys

# ---------------------------------------------------------------- 颜色 token
PAPER = "#FAF9F7"        # 界面底色: 暖白纸
SURFACE = "#FFFFFF"      # 数据表面: 纯白 (表格 / 图 / 输入框)
LINE = "#E3E0DA"         # 分隔线
LINE_STRONG = "#CBC7BF"
INK = "#16181D"          # 正文
INK_MUTED = "#63636B"    # 次要说明
INK_FAINT = "#8E8D95"

BLUE = "#1F4E9C"         # 仪表蓝: 主操作 / 选中
BLUE_DARK = "#173C79"
BLUE_TINT = "#E8EEF8"

OK = "#2C6E49"           # 氧化绿
WARN = "#B07503"         # 琥珀
BAD = "#B3261E"          # 朱红
OK_TINT = "#EAF2ED"
WARN_TINT = "#FBF2E0"
BAD_TINT = "#FAEBEA"

VERDICT = {
    "PASS": (OK, OK_TINT, "✓"),
    "WARN": (WARN, WARN_TINT, "!"),
    "FAIL": (BAD, BAD_TINT, "✕"),
}

# 绘图用的分类色: 与判定色拉开距离, 免得图上的曲线看着像在报警
PLOT_SERIES = ["#1F4E9C", "#8A5FBF", "#0E7C86", "#C2683A", "#5A6570", "#A03E7A"]

# 通道用途的 toggle: 循环顺序与配色。用途是分类不是状态, 所以不用判定色。
ROLE_ORDER = ["mic", "vpu", "ref"]
ROLE_STYLE = {
    "mic": ("麦克风", BLUE, BLUE_TINT),
    "vpu": ("VPU", "#0E7C86", "#E4F1F2"),
    "ref": ("参考麦", "#7A5AA8", "#EFEAF7"),
}


# ---------------------------------------------------------------- 字体 token
def ui_family() -> str:
    if sys.platform == "darwin":
        return "PingFang SC"
    if sys.platform.startswith("win"):
        return "Microsoft YaHei UI"
    return "Noto Sans CJK SC"


def mono_family() -> str:
    """读数字体。要求: 等宽 + 数字等宽 (tabular figures)。"""
    if sys.platform == "darwin":
        return "SF Mono"
    if sys.platform.startswith("win"):
        return "Cascadia Mono"
    return "DejaVu Sans Mono"


FONT_STACK = f'"{ui_family()}", "Segoe UI", "Helvetica Neue", sans-serif'
MONO_STACK = f'"{mono_family()}", "Consolas", "Menlo", monospace'

# 字号阶梯 (pt)
FS_MICRO = 10
FS_SMALL = 11
FS_BASE = 12
FS_MED = 13
FS_LARGE = 17
FS_DISPLAY = 30      # 采集页指令 —— 三米可读


def stylesheet() -> str:
    return f"""
* {{ font-family: {FONT_STACK}; font-size: {FS_BASE}pt; }}

QWidget {{ background: {PAPER}; color: {INK}; }}
/* 标签不画自己的底 —— 否则套在卡片/高亮底色上会露出一块方块 */
QLabel {{ background: transparent; }}
QMainWindow::separator {{ background: {LINE}; width: 1px; height: 1px; }}

/* ---- 采集页指令卡: 三米开外可读。字号必须走 QSS ——
   QSS 的优先级高于 setFont(), 用 QFont 设的字号会被上面的通配规则盖掉 ---- */
QLabel#display {{
    font-size: {FS_DISPLAY}pt;
    font-weight: 600;
    line-height: 140%;
    color: {INK};
}}
QLabel#title {{ font-size: {FS_LARGE}pt; font-weight: 600; }}

/* ---- 分区 ---- */
QGroupBox {{
    background: {SURFACE};
    border: 1px solid {LINE};
    border-radius: 6px;
    margin-top: 20px;
    padding: 10px 12px 12px 12px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 2px 6px;
    color: {INK_MUTED};
    font-size: {FS_SMALL}pt;
    font-weight: 600;
    letter-spacing: 0.4px;
}}

/* ---- 页签 ---- */
QTabWidget::pane {{ border: 1px solid {LINE}; border-radius: 6px; background: {PAPER}; top: -1px; }}
QTabBar::tab {{
    background: transparent;
    color: {INK_MUTED};
    padding: 8px 18px;
    margin-right: 2px;
    border: 1px solid transparent;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {BLUE}; border-bottom: 2px solid {BLUE}; font-weight: 600; }}
QTabBar::tab:hover:!selected {{ color: {INK}; }}

/* ---- 按钮 ---- */
QPushButton {{
    background: {SURFACE};
    border: 1px solid {LINE_STRONG};
    border-radius: 5px;
    padding: 6px 14px;
    color: {INK};
}}
QPushButton:hover {{ border-color: {BLUE}; color: {BLUE}; }}
QPushButton:pressed {{ background: {BLUE_TINT}; }}
QPushButton:disabled {{ color: {INK_FAINT}; border-color: {LINE}; background: {PAPER}; }}
QPushButton#primary {{
    background: {BLUE}; color: #FFFFFF; border-color: {BLUE}; font-weight: 600;
}}
QPushButton#primary:hover {{ background: {BLUE_DARK}; border-color: {BLUE_DARK}; color: #FFFFFF; }}
QPushButton#primary:disabled {{ background: {LINE}; border-color: {LINE}; color: {INK_FAINT}; }}
QPushButton#danger:hover {{ border-color: {BAD}; color: {BAD}; }}

/* ---- 输入 ---- */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background: {SURFACE};
    border: 1px solid {LINE_STRONG};
    border-radius: 5px;
    padding: 4px 8px;
    selection-background-color: {BLUE_TINT};
    selection-color: {INK};
}}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{ border: 1px solid {BLUE}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE}; border: 1px solid {LINE_STRONG};
    selection-background-color: {BLUE_TINT}; selection-color: {INK};
}}
QCheckBox {{ spacing: 7px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border: 1px solid {LINE_STRONG};
    border-radius: 3px; background: {SURFACE};
}}
QCheckBox::indicator:checked {{ background: {BLUE}; border-color: {BLUE}; }}

/* ---- 表格: 数据表面, 读数等宽 ---- */
QTableWidget {{
    background: {SURFACE};
    alternate-background-color: {PAPER};
    border: 1px solid {LINE};
    border-radius: 6px;
    gridline-color: {LINE};
    font-family: {MONO_STACK};
    font-size: {FS_SMALL}pt;
}}
QTableWidget::item {{ padding: 4px 6px; }}
QTableWidget::item:selected {{ background: {BLUE_TINT}; color: {INK}; }}
QHeaderView::section {{
    background: {PAPER};
    color: {INK_MUTED};
    border: none;
    border-bottom: 1px solid {LINE_STRONG};
    padding: 6px;
    font-family: {FONT_STACK};
    font-size: {FS_MICRO}pt;
    font-weight: 600;
    letter-spacing: 0.3px;
}}

/* ---- 日志: 也是读数 ---- */
QTextEdit#log {{ font-family: {MONO_STACK}; font-size: {FS_SMALL}pt; }}

/* ---- 进度 / 电平 ---- */
QProgressBar {{
    background: {PAPER}; border: 1px solid {LINE}; border-radius: 4px;
    text-align: center; color: {INK_MUTED}; font-size: {FS_MICRO}pt; height: 16px;
}}
QProgressBar::chunk {{ background: {BLUE}; border-radius: 3px; }}

/* ---- 说明文字 ---- */
QLabel#hint {{ color: {INK_MUTED}; font-size: {FS_SMALL}pt; }}
QLabel#eyebrow {{
    color: {INK_FAINT}; font-size: {FS_MICRO}pt;
    font-weight: 600; letter-spacing: 1.2px;
}}
QLabel#readout {{ font-family: {MONO_STACK}; font-size: {FS_SMALL}pt; color: {INK}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {LINE_STRONG}; border-radius: 5px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QStatusBar {{ background: {PAPER}; color: {INK_MUTED}; border-top: 1px solid {LINE}; }}
QStatusBar::item {{ border: none; }}
"""


def apply(app) -> None:
    """给 QApplication 套上主题, 并同步 pyqtgraph 的配色。"""
    import pyqtgraph as pg
    from PySide6.QtGui import QFont

    pg.setConfigOptions(antialias=True, background=SURFACE, foreground=INK_MUTED)
    f = QFont(ui_family())
    f.setPointSize(FS_BASE)
    app.setFont(f)
    app.setStyleSheet(stylesheet())
