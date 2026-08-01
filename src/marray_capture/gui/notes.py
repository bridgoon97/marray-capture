"""每个环节的注意事项。

这些条目原本散在 README 和部署指南里 —— 但采集现场没人会去翻文档, 所以搬进界面。

两条编写原则:

1. **只写会付学费的。** 通用建议("注意音量")不写; 只写那些做错了会**静默出错**、
   或者事后无法补救的。分三档:
   - 必看: 做错了不报错, 但数据是废的 (麦序号错位、阵列几何不符)
   - 注意: 做错了会明显影响质量, 但看得出来 (蓝牙掉 HFP、电平差太大)
   - 提示: 省时间或帮助理解的
2. **跟当前配置联动。** 选了 ASIO 才提示 ASIO 的坑, 跨设备才提示蓝牙的坑 ——
   常驻的通用清单会被当成墙纸, 只在相关时出现的才有人读。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

from ..settings import AppSettings

CRITICAL, WARN, INFO = "critical", "warn", "info"
LEVEL_LABEL = {CRITICAL: "必看", WARN: "注意", INFO: "提示"}


@dataclass
class Note:
    level: str
    text: str
    when: Callable[[AppSettings], bool] | None = None

    def applies(self, settings: AppSettings) -> bool:
        return self.when is None or bool(self.when(settings))


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _asio(s: AppSettings) -> bool:
    from ..audio import devices as dev
    return dev.is_asio(s.audio.input_device) or dev.is_asio(s.audio.output_device)


def _cross_device(s: AppSettings) -> bool:
    a = s.audio
    return (a.input_device is not None and a.output_device is not None
            and a.input_device != a.output_device)


def _many_mics(s: AppSettings) -> bool:
    return len(s.audio.mic_indices()) not in (0, 3)


# ---------------------------------------------------------------- 1 设备
DEVICE = [
    Note(CRITICAL,
         "麦序号决定 IR 里的麦克风顺序，<b>物理通道号不决定</b>。填错不会报错，"
         "只会让后处理的阵列几何和实际麦克风错位。开录前核对下方摘要的"
         "「IR 通道顺序 ↔ 对应声卡通道」两行。"),
    Note(CRITICAL,
         "麦克风权限没开时，录到的是一整段数字零，质检会报「几乎没有信号，疑似死麦」。"
         + ("Windows：设置 → 隐私和安全性 → 麦克风 → 打开「让桌面应用访问」。"
            if _is_windows() else
            "macOS：系统设置 → 隐私与安全性 → 麦克风，授权对象是启动它的终端/IDE；"
            "改完要 <b>Cmd+Q 完全退出</b>再开，关窗口不算。")),
    Note(CRITICAL,
         "检测到 ASIO。ASIO 驱动<b>单实例独占</b> —— 输入和输出必须选<b>同一个</b> "
         "ASIO 条目，收发模式用「自动」或「强制全双工」，否则开流会直接失败。"
         "用 ASIO4ALL 的话，声卡和音箱在 ASIO4ALL 面板里选，这里只选那一个设备；"
         "面板里没启用的输入端口，这里根本看不见。",
         when=_asio),
    Note(WARN,
         "输入通道数只有 2？多半是选到了 <b>MME</b> 条目。认准括号里写 "
         "<b>Windows WASAPI</b> 或 <b>ASIO</b> 的那一条。",
         when=lambda s: _is_windows() and not _asio(s)),
    Note(WARN,
         "跨设备播放（蓝牙音箱）：必须选<b>「立体声 / Stereo」</b>端点，"
         "不要选「免提电话 / Hands-Free」。而且<b>绝不能把蓝牙设备选成输入</b> —— "
         "一旦系统认为你在用它的麦克风，整条链路会掉到 HFP，8/16 kHz 单声道，高频全没。"
         "判断方法：点「① 播放测试音」，听起来像打电话就是掉了。",
         when=_cross_device),
    Note(WARN,
         "「② 电平检查」时逐只麦克风说话：看哪一路电平在动，就能确定物理通道对应关系。"
         "通道间电平差 &gt; 12 dB 要查接线或增益；峰值留 −6 dBFS 余量。"),
    Note(INFO,
         "四项检查按 ①②③④ 顺序做完再开录。「④ 延迟标定」除了收窄搜索窗，"
         "还能提前暴露「根本没出声」这类接线问题，并给出时钟漂移初值。"),
    Note(INFO,
         "有条件就把音箱用线接到声卡输出：软件会自动切成全双工单流，"
         "收发采样锁定、没有时钟漂移，也不受蓝牙有损编码影响。",
         when=_cross_device),
]

# ---------------------------------------------------------------- 2 方案
PLAN = [
    Note(CRITICAL,
         "循环顺序是按<b>操作成本</b>排的：高度（要走到支架旁）&gt; 距离（挪椅子）"
         "&gt; 方位（转椅子）。别自己改成别的顺序，否则同样的位置数会多花一倍时间。"),
    Note(WARN,
         "音箱朝向是<b>圈级</b>参数，不是位置级的。音箱固定指向椅子时，佩戴者原地旋转"
         "并不改变两者的相对朝向 —— 想要「干扰人没正对着你」，只能整圈重来一次并"
         "把音箱转个方向。所以每种朝向只需要走过去调一次。"),
    Note(WARN,
         "「预渲染语音」必须在开录前做完。采集过程中不跑 TTS，所以现场断网也照跑；"
         "但没预渲染就会降级成提示音。用 edge-tts 的话，只有预渲染这一步需要联网。"),
    Note(INFO,
         "方位全靠「向右转一格」，<b>不需要任何角度标定</b>。角度标签只是名义值，"
         "只用于分组统计，不参与任何计算。每转四分之一圈会播一句粗方向校验，防止累计误差。"),
    Note(INFO,
         "距离维度真正要覆盖的是<b>近场头部效应</b>（&lt;1 m 时低频 ILD 会显著抬升，"
         "远场 HRTF 覆盖不了），混响量交给后处理的 DRR 旋钮随机化。所以距离点可以很少。"),
    Note(INFO,
         "点「试听一句」确认佩戴者在两三米外能听清、音量合适，再去预渲染整套。"),
]

# ---------------------------------------------------------------- 3 采集
RUN = [
    Note(CRITICAL,
         "倒计时<b>最后一声是升调</b>，听到就别动了。两次扫频之间只要动一下，"
         "一致性指标就会掉下来，这一条会被判 FAIL 并自动重录。"),
    Note(WARN,
         "扫频期间保持安静：别说话、别咳嗽、别调整姿势。呼吸声也会进近嘴的麦克风。"),
    Note(WARN,
         "采集期间不要开会议软件（微信/腾讯会议/Teams）。它们会抢占音频设备，"
         "蓝牙会被切到 HFP，正在跑的采集会直接废掉。"),
    Note(WARN,
         "出现<b>连续 FAIL</b> 就按「暂停」查原因，别硬跑完。多半是佩戴松了、"
         "音箱被挪动了，或者蓝牙重连了。"),
    Note(INFO,
         "反馈音：<b>上行两声 = 过</b>，<b>下行两声 = 没过</b>（会自动重录一次）。"
         "指令卡左侧的色条同步显示上一条的结论，隔着几米也看得见。"),
    Note(INFO,
         "每个位置的录音里都嵌了位置编号的提示音串。万一 manifest 和音频对不上，"
         "可以直接从音频里读回来。"),
]

# ---------------------------------------------------------------- 4 质检
QC = [
    Note(CRITICAL,
         "<b>采完立刻看分组统计，不要留到回去。</b> 整组失败说明是系统性问题"
         "（某个距离电平不够、某次佩戴松了、蓝牙重连过），当场补录成本极低；"
         "协助者走了就补不回来了。"),
    Note(WARN,
         "补录：在表里选中要重来的行 → 「把选中的位置加入补录」，会自动切到采集页"
         "并只跑这些位置（挪位置的调整步仍会播报）。"),
    Note(INFO,
         "<b>两次扫频一致性（NCC）是最有用的一个指标。</b> 佩戴者动了、蓝牙重同步、"
         "有损编码劣化，全都会在这里暴露。它不通过时软件会自动放弃多次平均，"
         "回退到单次 IR —— 亚样本错位下直接平均会梳状衰减高频。"),
    Note(INFO,
         "「可靠带宽」告诉你这一条实际能用到多少 Hz。蓝牙链路下这个数会明显低于 8 kHz，"
         "那是编码噪底造成的，不是模型问题。"),
    Note(INFO,
         "噪声估计取的是<b>直达峰之前</b>那一段：ESS 的谐波失真落在 −L·ln(k) 处，"
         "二次谐波之后到直达峰之间既没有失真产物也没有混响，是干净的反卷积域噪底。"),
]

# ---------------------------------------------------------------- 5 后处理
POST = [
    Note(CRITICAL,
         "<b>阵列几何必须与实物一致，坐标第 k 行要对应第 k 个麦克风。</b> "
         "晚期尾的通道间相干性 sinc(2πf·d/c) 完全由它决定 —— 填错不会报错，"
         "只会让下游波束形成器得到<b>过于乐观</b>的结果。这是整条链路里唯一"
         "会静默出错的地方。"),
    Note(CRITICAL,
         "麦克风数不是 3 时，「阵列几何」要选<b>「自定义坐标」</b>，逐行填 x, y, z"
         "（单位米）。3 麦等边三角形的预设只对默认阵型成立。",
         when=_many_mics),
    Note(WARN,
         "「早/晚分界」默认 50 ms：分界之前是<b>保留实测</b>的直达 + 早期反射"
         "（承载阵列/近场/头部的空间线索），之后才是随机合成的扩散尾。"
         "调得太小会把真实空间线索一起丢掉。"),
    Note(WARN,
         "质检门槛先用「只用 PASS」。数据量不够时再放宽到 PASS + WARN，"
         "但不要把 FAIL 放进训练集。"),
    Note(INFO,
         "去音箱响应是可选的：做 RTF（麦间相对量）时音箱响应是公共项会自己约掉，"
         "但把 IR 直接卷积干净语音生成训练音频时不会 —— 它会给干扰人染色。"
         "正式数据建议做。"),
    Note(INFO,
         "VPU 等非声学通道<b>保留实测原样</b>，不合成扩散尾。骨导/接触振动"
         "套球面各向同性相干模型是错的。"),
]

PAGES = {"device": DEVICE, "plan": PLAN, "run": RUN, "qc": QC, "post": POST}


def build_note_card(page: str, settings: AppSettings):
    """造好一张卡并接上折叠状态的持久化。采集页默认折叠 —— 采集时不该有阅读材料。"""
    from .widgets import NoteCard

    default_collapsed = page == "run"
    card = NoteCard(page, collapsed=bool(settings.notes_collapsed.get(page, default_collapsed)))
    card.set_notes(notes_for(page, settings))
    card.toggled.connect(lambda c, p=page: settings.notes_collapsed.__setitem__(p, c))
    return card


def notes_for(page: str, settings: AppSettings | None = None) -> list[Note]:
    """取某一页当前适用的注意事项。必看的排在前面。"""
    items = PAGES.get(page, [])
    if settings is not None:
        items = [n for n in items if n.applies(settings)]
    rank = {CRITICAL: 0, WARN: 1, INFO: 2}
    return sorted(items, key=lambda n: rank.get(n.level, 9))
