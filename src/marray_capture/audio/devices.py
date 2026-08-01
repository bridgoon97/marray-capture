"""声卡枚举与能力查询。输入与输出允许是**不同设备** (USB 声卡录音 + 蓝牙音箱播放)。

重要: PortAudio 的 ``check_input_settings`` / ``check_output_settings`` 会真的去打开
设备。碰上虚拟声卡 (会议软件、串流工具) 或正在重连的蓝牙设备时**可能长时间阻塞**。
所以这里的默认路径全部不做探测:

- ``list_devices`` / ``supported_rates`` / ``validate`` 只读设备描述, 不打开设备。
- 真正的探测放在 ``probe_rates`` / ``probe_settings``, 调用方必须放到工作线程里。
- 参数不合适时, 让真正开流的那一刻报错 —— 错误信息更准确, 也不会冻界面。
"""
from __future__ import annotations

from dataclasses import dataclass

import sounddevice as sd

COMMON_RATES = [16000, 32000, 44100, 48000, 96000]


@dataclass
class DeviceInfo:
    index: int
    name: str
    hostapi: str
    max_input: int
    max_output: int
    default_samplerate: float

    def label(self) -> str:
        io = f"in {self.max_input} / out {self.max_output}"
        return f"[{self.index}] {self.name}  ({self.hostapi}, {io})"


def rescan() -> None:
    """重新初始化 PortAudio, 让热插拔的设备出现在列表里。只在用户点「刷新」时调用。"""
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        pass


def list_devices() -> list[DeviceInfo]:
    apis = sd.query_hostapis()
    out: list[DeviceInfo] = []
    for i, d in enumerate(sd.query_devices()):
        out.append(
            DeviceInfo(
                index=i,
                name=str(d["name"]),
                hostapi=str(apis[d["hostapi"]]["name"]) if d["hostapi"] < len(apis) else "?",
                max_input=int(d["max_input_channels"]),
                max_output=int(d["max_output_channels"]),
                default_samplerate=float(d["default_samplerate"]),
            )
        )
    return out


def input_devices() -> list[DeviceInfo]:
    return [d for d in list_devices() if d.max_input > 0]


def output_devices() -> list[DeviceInfo]:
    return [d for d in list_devices() if d.max_output > 0]


def describe(device: int | None) -> DeviceInfo | None:
    if device is None:
        return None
    for d in list_devices():
        if d.index == device:
            return d
    return None


def supported_rates(device: int | None) -> list[int]:
    """候选采样率 —— 常用档 + 该设备的默认采样率。不打开设备, 不阻塞。"""
    rates = set(COMMON_RATES)
    info = describe(device)
    if info and info.default_samplerate > 0:
        rates.add(int(round(info.default_samplerate)))
    return sorted(rates)


def probe_rates(device: int | None, kind: str) -> list[int]:
    """**会阻塞**: 逐个真开设备验证采样率。必须在工作线程里调用。"""
    if device is None:
        return []
    ok = []
    for r in supported_rates(device):
        try:
            if kind == "input":
                sd.check_input_settings(device=device, samplerate=r)
            else:
                sd.check_output_settings(device=device, samplerate=r)
            ok.append(r)
        except Exception:
            continue
    return ok


def probe_settings(input_device: int | None, output_device: int | None,
                   samplerate: int, n_in: int, n_out: int) -> list[str]:
    """**会阻塞**: 真开一次设备验证完整参数组合。必须在工作线程里调用。"""
    problems: list[str] = []
    if input_device is not None:
        try:
            sd.check_input_settings(device=input_device, channels=n_in, samplerate=samplerate)
        except Exception as e:
            problems.append(f"输入设备不支持 {n_in} 通道 @ {samplerate} Hz: {e}")
    if output_device is not None:
        try:
            sd.check_output_settings(device=output_device, channels=n_out, samplerate=samplerate)
        except Exception as e:
            problems.append(f"输出设备不支持 {n_out} 通道 @ {samplerate} Hz: {e}")
    return problems


def is_asio(device: int | None) -> bool:
    info = describe(device)
    return bool(info and "asio" in info.hostapi.lower())


def validate(input_device: int | None, output_device: int | None, samplerate: int,
             n_in: int, n_out: int, duplex: bool = False) -> list[str]:
    """只读描述信息的快速检查。以 "提示:" 开头的是提醒, 不阻塞流程。"""
    problems: list[str] = []
    if input_device is None:
        problems.append("未选择输入设备")
    if output_device is None:
        problems.append("未选择输出设备")

    din, dout = describe(input_device), describe(output_device)
    if din is not None and n_in > din.max_input:
        problems.append(
            f"输入设备「{din.name}」只有 {din.max_input} 个输入通道, 但通道映射用到了第 {n_in} 个。"
            + (" ASIO4ALL 请在驱动面板里把需要的输入端口全部启用。" if is_asio(input_device)
               else " Windows 上如果这里只有 2 个通道, 多半是选到了 MME 条目, 换成 WASAPI/ASIO 那条。")
        )
    if dout is not None and n_out > dout.max_output:
        problems.append(f"输出设备只有 {dout.max_output} 个通道, 但配置要求 {n_out} 个")
    if samplerate <= 0:
        problems.append("采样率无效")

    asio = is_asio(input_device) or is_asio(output_device)
    same = input_device is not None and input_device == output_device

    if asio and not same and input_device is not None and output_device is not None:
        problems.append(
            "ASIO 驱动通常独占设备, 输入和输出必须是同一个 ASIO 条目。"
            "用 ASIO4ALL 的话, 声卡和音箱在 ASIO4ALL 面板里选, 这里只选那一个 ASIO 设备。"
        )
    if duplex or same:
        problems.append("提示: 走全双工单流, 收发采样锁定 —— 没有时钟漂移, 延迟只有设备往返延迟。")
    elif input_device is not None and output_device is not None:
        problems.append(
            "提示: 输入/输出是不同设备, 时钟不同源。软件会按扫频反卷积峰值对齐, "
            "并在每个位置估计漂移 (ppm); 漂移超阈值会在质检里报警。"
        )
    return problems
