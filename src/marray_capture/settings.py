"""全局配置对象与持久化。

配置分四块:
  AudioConfig    —— 声卡 / 通道 / 采样率 / 延迟
  SweepConfig    —— ESS 扫频参数与每个位置的时间安排
  ProtocolConfig —— 采集方案 (距离 / 高度 / 转向格数 / 重戴 / 随机抖动位)
  ExportConfig   —— IR 裁剪、重采样与导出

增强 (rir-augment) 那部分参数仍然走 YAML, 见 configs/augment.yaml。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

APP_DIR = Path.home() / ".marray-capture"
SETTINGS_PATH = APP_DIR / "settings.json"

# 通道用途。mic 进阵列 IR (有序); vpu 单独存一路; ref 是测量麦 (用于去音箱响应)。
CHANNEL_ROLES = ["mic", "vpu", "ref"]
ROLE_LABELS = {"mic": "麦克风", "vpu": "VPU", "ref": "参考麦"}


@dataclass
class ChannelMap:
    """一路输入通道。index 是声卡上的 0-based 物理通道号。

    麦克风在声卡上的通道号是任意的, 顺序也不一定和阵列几何一致, 所以:
    - ``enabled`` 决定这一路录不录 (界面上是勾选框)
    - ``role`` 决定它是什么 (界面上是一个 toggle)
    - ``order`` **只对麦克风有意义**: 它是第几个麦克风。IR 文件里的麦克风顺序按
      order 排, 必须与后处理页阵列几何的坐标行一一对应 —— 错位不会报错,
      只会让扩散尾的通道间相干性算错。
    """

    index: int
    role: str = "mic"
    label: str = ""
    enabled: bool = False
    order: int = 0

    def display(self) -> str:
        return self.label or f"ch{self.index + 1}"


def _channel_from_dict(d: dict[str, Any]) -> ChannelMap:
    """读旧配置。旧版用 role="ignore" 表示不录, 也没有 enabled / order 字段。"""
    role = d.get("role", "mic")
    enabled = d.get("enabled")
    if enabled is None:
        enabled = role in CHANNEL_ROLES
    if role not in CHANNEL_ROLES:
        role = "mic"
    return ChannelMap(
        index=int(d.get("index", 0)), role=role, label=d.get("label", ""),
        enabled=bool(enabled), order=int(d.get("order", 0)),
    )


@dataclass
class AudioConfig:
    input_device: int | None = None
    output_device: int | None = None
    samplerate: int = 48000
    input_latency: str = "low"      # sounddevice latency 提示
    output_latency: str = "high"    # 蓝牙音箱必须给 high, 否则 underrun
    blocksize: int = 0              # 0 = 由驱动决定
    output_channels: int = 2
    output_gain_db: float = -6.0
    # auto = 同设备走全双工单流, 异设备走分离双流。ASIO 必须全双工 (驱动单实例独占)
    duplex_mode: str = "auto"       # auto | duplex | split
    # 实际向声卡请求的输入通道数。0 = 自动 (先按用到的最大通道号试, 不行退到全开)。
    # WASAPI 共享模式只接受端点的完整通道数, 这里会被自动填成声卡的输入通道数。
    input_channels_override: int = 0
    # 输入通道映射; 录制时会录到 max(index)+1 路再按顺序切片
    channels: list[ChannelMap] = field(default_factory=list)
    # 输出→输入的往返延迟 (样本数)。由「延迟标定」测得; None 表示每次全局搜索
    measured_latency_samples: int | None = None
    # 两台设备不同源时的时钟漂移 (ppm), 由标定或每个 take 估计
    drift_ppm: float = 0.0
    drift_correction: bool = False

    def ordered_channels(self) -> list[ChannelMap]:
        """落盘顺序: 先按 order 排好的麦克风, 然后 VPU, 最后参考麦。

        这样 IR 文件的第 k 列 (k < 麦克风数) 恒等于第 k 个麦克风, 与阵列几何的
        第 k 行坐标对应 —— 不依赖麦克风挂在哪几个物理通道上。
        """
        on = [c for c in self.channels if c.enabled and c.role in CHANNEL_ROLES]
        mics = sorted((c for c in on if c.role == "mic"), key=lambda c: (c.order, c.index))
        rest = sorted((c for c in on if c.role != "mic"),
                      key=lambda c: (CHANNEL_ROLES.index(c.role), c.index))
        return mics + rest

    def mic_indices(self) -> list[int]:
        """麦克风的物理通道号, 按麦克风顺序 (不是物理顺序)。"""
        return [c.index for c in self.ordered_channels() if c.role == "mic"]

    def role_indices(self, role: str) -> list[int]:
        return [c.index for c in self.ordered_channels() if c.role == role]

    def active_indices(self) -> list[int]:
        """所有要落盘的物理通道号, **按落盘顺序**排列 (不是物理顺序)。"""
        return [c.index for c in self.ordered_channels()]

    def n_record_channels(self) -> int:
        """要向声卡请求几个通道 —— 必须录到用到的最大物理通道号为止。"""
        act = self.active_indices()
        return (max(act) + 1) if act else 0

    def duplicate_orders(self) -> list[int]:
        """返回被重复使用的麦克风序号, 用于界面提示。"""
        seen: dict[int, int] = {}
        for c in self.channels:
            if c.enabled and c.role == "mic":
                seen[c.order] = seen.get(c.order, 0) + 1
        return sorted(k for k, v in seen.items() if v > 1)


@dataclass
class SweepConfig:
    f_start: float = 40.0
    f_end: float = 20000.0          # 会自动夹到 0.98*Nyquist
    duration_s: float = 5.0
    repeats: int = 2                # 每个位置扫几次 (>=2 才有一致性质检)
    gap_s: float = 1.2              # 两次扫频之间的静音
    preroll_s: float = 1.5          # 扫频前静音, 同时用于底噪测量
    tail_s: float = 1.5             # 最后一次扫频后的静音 (混响尾 + 蓝牙延迟余量)
    # 输入流先于播放启动的保护间隔。它会计入测得的"延迟"里, 所以标定与正式采集
    # 必须用同一个值, 否则 measured_latency_samples 的搜索窗会偏掉。
    guard_s: float = 0.4
    fade_in_ms: float = 15.0
    fade_out_ms: float = 40.0
    max_latency_s: float = 1.0      # 搜索直达峰的时间窗半径, 蓝牙需要给大
    amplitude: float = 0.5          # 播放幅度 (线性), 与 output_gain_db 叠加


@dataclass
class ProtocolConfig:
    subject_id: str = "S01"
    wearing_id: str = "W1"
    side: str = "R"                                 # 耳机戴在哪一侧
    distances_cm: list[int] = field(default_factory=lambda: [50, 100, 200])
    heights_cm: list[int] = field(default_factory=lambda: [120, 160, 90])
    # 基准环 (距离, 高度) 用密集角度, 其余用稀疏角度
    dense_distance_cm: int = 100
    dense_height_cm: int = 160
    dense_steps: int = 12                           # 12 格 ≈ 每格 30°
    sparse_steps: int = 8                           # 8 格 ≈ 每格 45°
    # 音箱朝向子集: 只在基准环上做 (0=正对佩戴者, 90=侧向, 180=背对)
    speaker_orientations: list[int] = field(default_factory=lambda: [0, 90, 180])
    orientation_subset_steps: int = 4               # 朝向变体只在这么多个方位上做
    random_positions: int = 20                      # 随机抖动位数量
    rewearing_rings: int = 2                        # 摘下重戴几次
    rewearing_steps: int = 8                        # 每次重戴录几个方位
    rewearing_distance_cm: int = 100
    rewearing_height_cm: int = 160
    settle_s: float = 4.0                           # 指令播完到开扫的稳定时间
    setup_settle_s: float = 20.0                    # 需要挪椅子/调支架的步骤给更长时间


@dataclass
class ExportConfig:
    ir_pre_ms: float = 5.0          # 直达峰之前保留多少
    ir_len_ms: float = 1000.0       # IR 总长
    export_16k: bool = True
    save_raw: bool = True
    raw_format: str = "FLAC"        # FLAC | WAV
    average_repeats: bool = True    # 一致性通过时对多次扫频求平均 (涨 SNR)
    speaker_ref_ir: str = ""        # 音箱参考 IR 路径, 空则不做去音箱响应


@dataclass
class QCThresholds:
    clip_dbfs: float = -0.5         # 峰值超过判削顶
    min_rec_snr_db: float = 25.0
    min_ir_ddr_db: float = 30.0     # 直达峰 / IR 远端噪声
    min_repeat_ncc: float = 0.95    # 两次扫频早期段的归一化互相关
    max_repeat_level_diff_db: float = 1.0
    max_drift_ppm: float = 200.0
    min_reliable_bw_hz: float = 6000.0
    warn_margin: float = 0.5        # 未过阈值但差距在此比例内 → WARN 而非 FAIL


@dataclass
class AppSettings:
    audio: AudioConfig = field(default_factory=AudioConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    qc: QCThresholds = field(default_factory=QCThresholds)
    session_root: str = str(Path.home() / "marray-sessions")
    # 各页注意事项卡片的折叠状态 (页名 -> 是否折叠)
    notes_collapsed: dict[str, bool] = field(default_factory=dict)
    tts_enabled: bool = True
    tts_backend: str = "auto"       # auto | piper | edge | sapi | say | none
    tts_voice: str = ""
    auto_retry_on_fail: bool = True

    # ---- 持久化 ----
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AppSettings":
        s = cls()
        for key, klass in [
            ("audio", AudioConfig), ("sweep", SweepConfig),
            ("protocol", ProtocolConfig), ("export", ExportConfig),
            ("qc", QCThresholds),
        ]:
            sub = d.get(key) or {}
            obj = klass()
            for f_name, val in sub.items():
                if hasattr(obj, f_name):
                    setattr(obj, f_name, val)
            if key == "audio":
                obj.channels = [_channel_from_dict(c) for c in sub.get("channels", [])]
            setattr(s, key, obj)
        for f_name in ("session_root", "notes_collapsed", "tts_enabled", "tts_backend",
                       "tts_voice", "auto_retry_on_fail"):
            if f_name in d:
                setattr(s, f_name, d[f_name])
        return s

    def save(self, path: Path | None = None) -> Path:
        p = Path(path or SETTINGS_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> "AppSettings":
        p = Path(path or SETTINGS_PATH)
        if not p.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            return cls()
