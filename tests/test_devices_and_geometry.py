"""收发模式选择、全双工流、以及任意麦克风数的阵列几何。

全双工是 ASIO 唯一能用的路子 (驱动单实例独占), 所以这里用一个替身 sd.Stream
把回调驱动起来, 验证「录到的东西和播出去的东西逐样本对齐」。
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from marray_capture.audio import engine as engine_mod
from marray_capture.audio.engine import AudioEngine
from marray_capture.rir.augment_runner import default_config, parse_coords, run_augment
from marray_capture.rir.rir_augment.geometry import build_array, pairwise_distances
from marray_capture.settings import AudioConfig, ChannelMap

FS = 16000


# ---------------------------------------------------------------- 模式选择
@pytest.mark.parametrize("mode, in_dev, out_dev, expect", [
    ("auto", 3, 3, True),        # 同设备 (ASIO4ALL 就是这种) → 全双工
    ("auto", 3, 7, False),       # 声卡 + 蓝牙音箱 → 分离双流
    ("auto", None, None, False),
    ("duplex", 3, 7, True),      # 强制
    ("split", 3, 3, False),
])
def test_duplex_mode_selection(mode, in_dev, out_dev, expect):
    cfg = AudioConfig(input_device=in_dev, output_device=out_dev, duplex_mode=mode)
    assert AudioEngine(cfg).use_duplex() is expect


# ---------------------------------------------------------------- 全双工流
class FakeStream:
    """驱动 duplex 回调的替身: 把播出去的东西延迟一个块之后回灌成输入。

    延迟取整块是为了让回灌只依赖**之前**的块 —— 否则本块的输入会依赖本块尚未
    填好的输出, 在替身里无法自洽。
    """

    BLOCK = 512
    LATENCY_BLOCKS = 1
    latency = BLOCK * LATENCY_BLOCKS
    captured: list[np.ndarray] = []

    def __init__(self, device, channels, samplerate, dtype, blocksize,
                 latency, callback, finished_callback):
        self.channels = channels
        self.blocksize = blocksize or self.BLOCK
        self.callback = callback
        self.finished_callback = finished_callback
        FakeStream.captured = []

    def __enter__(self):
        import sounddevice as sd

        n_in, n_out = self.channels
        history = [np.zeros((self.blocksize, n_out), np.float32)
                   for _ in range(self.LATENCY_BLOCKS)]
        while True:
            out = np.zeros((self.blocksize, n_out), dtype=np.float32)
            src = history[0]
            indata = np.zeros((self.blocksize, n_in), dtype=np.float32)
            for c in range(min(n_in, n_out)):
                indata[:, c] = src[:, c] * (1.0 - 0.1 * c)   # 各通道给点差异
            stop = False
            try:
                self.callback(indata, out, self.blocksize, None, None)
            except (sd.CallbackStop, sd.CallbackAbort):
                stop = True
            FakeStream.captured.append(out.copy())
            history = history[1:] + [out.copy()]
            if stop:
                break
        self.finished_callback()
        return self

    def __exit__(self, *a):
        return False


def test_duplex_records_and_aligns(monkeypatch):
    monkeypatch.setattr(engine_mod.sd, "Stream", FakeStream)
    cfg = AudioConfig(input_device=3, output_device=3, samplerate=FS,
                      duplex_mode="duplex", output_channels=2, blocksize=512)
    eng = AudioEngine(cfg)

    play = np.zeros(FS)                    # 1 秒, 中间放一个脉冲
    play[FS // 2] = 0.8
    levels: list[np.ndarray] = []
    rec = eng.play_record(play, n_in_channels=4, extra_tail_s=0.2, guard_s=0.4,
                          level_cb=levels.append)

    guard = int(0.4 * FS)
    assert rec.shape[1] == 4
    assert rec.shape[0] >= guard + len(play)
    # 脉冲应出现在 guard + 播放位置 + 设备延迟 处
    peak = int(np.argmax(np.abs(rec[:, 0])))
    assert abs(peak - (guard + FS // 2 + FakeStream.latency)) <= 1
    # 播放缓冲区里 guard 段必须是静音 (质检的噪声窗依赖这一点)
    played = np.concatenate(FakeStream.captured, axis=0)
    assert np.max(np.abs(played[:guard])) < 1e-9
    assert levels, "应该有电平回调"


def test_open_failure_message_mentions_wasapi(monkeypatch):
    import sounddevice as sd

    def boom(**kwargs):
        raise sd.PortAudioError("Invalid number of channels")

    monkeypatch.setattr(engine_mod.sd, "Stream", boom)
    eng = AudioEngine(AudioConfig(input_device=3, output_device=3, duplex_mode="duplex"))
    with pytest.raises(engine_mod.EngineError, match="WASAPI"):
        eng.play_record(np.zeros(1000), n_in_channels=4)


def test_open_failure_message_mentions_asio_when_asio(monkeypatch):
    import sounddevice as sd

    from marray_capture.audio import devices as dev

    asio = dev.DeviceInfo(3, "ASIO4ALL", "ASIO", 8, 8, 48000.0)
    monkeypatch.setattr(dev, "describe", lambda i: asio if i == 3 else None)
    monkeypatch.setattr(engine_mod.sd, "Stream",
                        lambda **kw: (_ for _ in ()).throw(sd.PortAudioError("busy")))
    eng = AudioEngine(AudioConfig(input_device=3, output_device=3, duplex_mode="duplex"))
    with pytest.raises(engine_mod.EngineError, match="单实例独占"):
        eng.play_record(np.zeros(1000), n_in_channels=4)


def test_input_channel_fallback_to_full_count(monkeypatch):
    """WASAPI 上请求 5/8 通道会被拒, 应自动退到 8 通道再切片。"""
    import sounddevice as sd

    from marray_capture.audio import devices as dev

    card = dev.DeviceInfo(0, "8ch", "Windows WASAPI", 8, 2, 48000.0)
    monkeypatch.setattr(dev, "describe", lambda i: card if i == 0 else None)

    tried: list[int] = []

    class Stream(FakeStream):
        def __init__(self, device, channels, samplerate, dtype, blocksize,
                     latency, callback, finished_callback):
            tried.append(channels[0])
            if channels[0] != 8:                     # 只接受完整通道数
                raise sd.PortAudioError("Invalid number of channels")
            super().__init__(device, channels, samplerate, dtype, blocksize,
                             latency, callback, finished_callback)

    monkeypatch.setattr(engine_mod.sd, "Stream", Stream)
    cfg = AudioConfig(input_device=0, output_device=0, samplerate=FS,
                      duplex_mode="duplex", output_channels=2, blocksize=512)
    eng = AudioEngine(cfg)
    rec = eng.play_record(np.zeros(FS), n_in_channels=5, extra_tail_s=0.1, guard_s=0.2)

    assert tried == [5, 8]
    assert rec.shape[1] == 8                          # 全开, 由上层切片
    assert cfg.input_channels_override == 8           # 记住了, 下次直接用
    assert "WASAPI" in eng.last_warnings

    tried.clear()
    eng.play_record(np.zeros(FS), n_in_channels=5, extra_tail_s=0.1, guard_s=0.2)
    assert tried == [8], "记住之后不该再试错"


# ---------------------------------------------------------------- 通道映射
def test_channel_map_handles_scrambled_eight_channel_card():
    """4 麦 + VPU 挂在 8 通道卡上, 物理通道号打乱, 麦克风顺序由 order 决定。

    这是真实接线的样子: mic1 在 ch6、mic2 在 ch3 …… 麦克风顺序必须跟着 order 走,
    否则 IR 的通道顺序和阵列几何的坐标行对不上, 而且不会报错。
    """
    cfg = AudioConfig(channels=[
        ChannelMap(0, "mic", "", enabled=False, order=9),   # 没勾, 不该出现
        ChannelMap(1, "mic", "mic3", enabled=True, order=3),
        ChannelMap(2, "mic", "mic2", enabled=True, order=2),
        ChannelMap(3, "ref", "refmic", enabled=True),
        ChannelMap(4, "mic", "mic4", enabled=True, order=4),
        ChannelMap(5, "mic", "mic1", enabled=True, order=1),
        ChannelMap(6, "vpu", "vpu", enabled=True),
        ChannelMap(7, "mic", "", enabled=False, order=8),
    ])
    # 麦克风按 order 排, 不是按物理通道号
    assert cfg.mic_indices() == [5, 2, 1, 4]
    # 落盘顺序: 麦克风 → VPU → 参考麦
    assert cfg.active_indices() == [5, 2, 1, 4, 6, 3]
    assert cfg.n_record_channels() == 7        # 要录到 ch7 才能拿到 vpu
    assert cfg.role_indices("vpu") == [6]
    assert cfg.duplicate_orders() == []

    from marray_capture.protocol.runner import channel_layout
    from marray_capture.settings import AppSettings
    s = AppSettings()
    s.audio = cfg
    indices, labels, mic_cols = channel_layout(s)
    assert indices == [5, 2, 1, 4, 6, 3]
    assert labels == ["mic1", "mic2", "mic3", "mic4", "vpu", "refmic"]
    assert mic_cols == [0, 1, 2, 3]            # 麦克风恒在最前, 与几何坐标行对齐


def test_duplicate_mic_order_is_reported():
    cfg = AudioConfig(channels=[
        ChannelMap(0, "mic", "a", enabled=True, order=1),
        ChannelMap(1, "mic", "b", enabled=True, order=1),
        ChannelMap(2, "mic", "c", enabled=True, order=2),
    ])
    assert cfg.duplicate_orders() == [1]
    # 同号时按物理通道号先后排, 不会丢通道
    assert cfg.mic_indices() == [0, 1, 2]


def test_legacy_channel_config_migrates():
    """旧配置用 role="ignore" 表示不录, 没有 enabled / order。"""
    from marray_capture.settings import AppSettings

    s = AppSettings.from_dict({"audio": {"channels": [
        {"index": 0, "role": "mic", "label": "mic1"},
        {"index": 1, "role": "ignore", "label": ""},
        {"index": 2, "role": "vpu", "label": "vpu"},
    ]}})
    assert [c.enabled for c in s.audio.channels] == [True, False, True]
    assert s.audio.mic_indices() == [0]
    assert s.audio.role_indices("vpu") == [2]


# ---------------------------------------------------------------- 阵列几何
def test_parse_coords():
    pts = parse_coords("""
        # 4 麦 1cm 正方形
        0.005,  0.005, 0
        0.005, -0.005, 0
        -0.005, -0.005
        -0.005  0.005  0
    """)
    assert len(pts) == 4
    arr = build_array({"coords": pts, "generator": None})
    d = pairwise_distances(arr)
    assert arr.shape == (4, 3)
    assert abs(d.max() - 0.01 * np.sqrt(2)) < 1e-9      # 对角线
    assert abs(d[d > 0].min() - 0.01) < 1e-9            # 边长

    with pytest.raises(ValueError, match="第 1 行"):
        parse_coords("a, b, c")
    with pytest.raises(ValueError, match="至少要两个"):
        parse_coords("0,0,0")


def test_augment_with_four_mics_plus_vpu(tmp_path):
    """4 麦 + VPU：自定义坐标必须能跑通，VPU 通道保留实测。"""
    fs = 16000
    ir = np.zeros((1600, 5), np.float32)
    for c in range(4):
        ir[20 + c, c] = 1.0 - 0.1 * c
    ir[20, 4] = 0.3                                     # VPU
    ir[100:400] += np.random.default_rng(0).normal(0, 0.02, (300, 5)).astype(np.float32)
    src = tmp_path / "take.wav"
    sf.write(str(src), ir, fs, subtype="FLOAT")

    cfg = default_config()
    cfg["array"] = {"coords": parse_coords(
        "0.005,0.005,0\n0.005,-0.005,0\n-0.005,-0.005,0\n-0.005,0.005,0"), "generator": None}
    cfg["output"]["num_per_rir"] = 2

    out = tmp_path / "aug"
    run_augment([(src, [0, 1, 2, 3])], out, cfg)
    files = sorted(out.glob("*.wav"))
    assert len(files) == 2
    data, _ = sf.read(str(files[0]), always_2d=True)
    assert data.shape[1] == 5
    for c in range(4):
        assert abs(data[20 + c, c] - (1.0 - 0.1 * c)) < 1e-3
    assert abs(data[20, 4] - 0.3) < 1e-3                # VPU 直通


def test_augment_rejects_geometry_mismatch(tmp_path):
    fs = 16000
    ir = np.zeros((800, 5), np.float32)
    ir[20, 0] = 1.0
    src = tmp_path / "t.wav"
    sf.write(str(src), ir, fs, subtype="FLOAT")
    cfg = default_config()                              # 默认是 3 麦等边三角形
    with pytest.raises(ValueError, match="自定义坐标"):
        run_augment([(src, [0, 1, 2, 3])], tmp_path / "o", cfg)
