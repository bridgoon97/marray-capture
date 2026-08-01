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

# 通道用途。mic 会进入阵列 IR; vpu 单独存一路; loopback 用于延迟标定; ref 是测量麦。
CHANNEL_ROLES = ["ignore", "mic", "vpu", "loopback", "ref"]


@dataclass
class ChannelMap:
    """一路输入通道的角色定义。index 是声卡上的 0-based 物理通道号。"""

    index: int
    role: str = "ignore"
    label: str = ""

    def display(self) -> str:
        return self.label or f"ch{self.index + 1}"


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
    # 输入通道映射; 录制时会录到 max(index)+1 路再按顺序切片
    channels: list[ChannelMap] = field(default_factory=list)
    # 输出→输入的往返延迟 (样本数)。由「延迟标定」测得; None 表示每次全局搜索
    measured_latency_samples: int | None = None
    # 两台设备不同源时的时钟漂移 (ppm), 由标定或每个 take 估计
    drift_ppm: float = 0.0
    drift_correction: bool = False

    def mic_indices(self) -> list[int]:
        return [c.index for c in self.channels if c.role == "mic"]

    def role_indices(self, role: str) -> list[int]:
        return [c.index for c in self.channels if c.role == role]

    def active_indices(self) -> list[int]:
        """所有需要落盘的通道 (按声卡物理顺序)。"""
        return sorted(c.index for c in self.channels if c.role != "ignore")

    def n_record_channels(self) -> int:
        act = self.active_indices()
        return (max(act) + 1) if act else 0


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
                obj.channels = [ChannelMap(**c) for c in sub.get("channels", [])]
            setattr(s, key, obj)
        for f_name in ("session_root", "tts_enabled", "tts_backend", "tts_voice",
                       "auto_retry_on_fail"):
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
