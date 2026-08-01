"""SessionRunner 集成测试: 用假声卡把一整个会话跑完, 校验落盘与 manifest。

假引擎复现真实链路里最容易出错的三件事: 输入输出不同源带来的未知延迟、
录音里除了扫频还有语音/提示音、以及多通道 (麦 + VPU) 的切片顺序。
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from marray_capture.audio import prompts as P
from marray_capture.protocol import runner as runner_mod
from marray_capture.protocol.plan import build_plan
from marray_capture.protocol.runner import RunnerHooks, SessionRunner
from marray_capture.settings import AppSettings, ChannelMap
from marray_capture.store import Session

FS = 16000
LATENCY = int(0.08 * FS)


def _ir(n_ch: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ir = np.zeros((1600, n_ch))
    for c in range(n_ch):
        ir[30 + c, c] = 1.0 - 0.15 * c
        ir[np.array([90, 140, 260]), c] += rng.normal(0, 0.12, 3)
        t = np.arange(1600) / FS
        ir[:, c] += rng.normal(0, 0.01, 1600) * np.exp(-t / 0.08)
    return ir


class FakeEngine:
    """替身声卡: 播放缓冲区经过固定 IR + 延迟 + 底噪后被"录"回来。"""

    IR = _ir()
    played: list[np.ndarray] = []          # 每次 play_record 的播放缓冲区, 供断言

    def __init__(self, cfg):
        self.cfg = cfg
        self.last_warnings = ""
        self.calls = 0

    def abort(self) -> None:
        pass

    def play(self, mono) -> None:
        pass

    def play_record(self, play_mono, n_in_channels, extra_tail_s=0.0,
                    guard_s=0.4, level_cb=None):
        self.calls += 1
        FakeEngine.played.append(np.asarray(play_mono, dtype=float).copy())
        rng = np.random.default_rng(1234 + self.calls)
        guard = int(guard_s * FS)
        tail = int(extra_tail_s * FS)
        total = guard + len(play_mono) + tail
        rec = np.zeros((total, n_in_channels))
        for c in range(min(n_in_channels, self.IR.shape[1])):
            y = fftconvolve(np.asarray(play_mono, dtype=float), self.IR[:, c])
            start = guard + LATENCY
            n = min(len(y), total - start)
            rec[start: start + n, c] = y[:n]
        rec += rng.normal(0, 3e-4, rec.shape)
        peak = float(np.max(np.abs(rec))) or 1.0
        rec = rec / peak * 0.25
        if level_cb is not None:
            level_cb(np.sqrt((rec[:512] ** 2).mean(axis=0)))
        return rec


def _settings(tmp_path) -> AppSettings:
    s = AppSettings()
    s.session_root = str(tmp_path)
    s.tts_enabled = False
    s.auto_retry_on_fail = False
    s.audio.samplerate = FS
    s.audio.input_device = 0
    s.audio.output_device = 1
    s.audio.channels = [
        ChannelMap(0, "mic", "mic1"), ChannelMap(1, "mic", "mic2"),
        ChannelMap(2, "mic", "mic3"), ChannelMap(3, "vpu", "vpu"),
    ]
    s.sweep.f_start = 60.0
    s.sweep.f_end = 7800.0
    s.sweep.duration_s = 0.8
    s.sweep.repeats = 2
    s.sweep.gap_s = 0.4
    s.sweep.preroll_s = 0.6
    s.sweep.tail_s = 0.4
    s.sweep.max_latency_s = 0.5
    s.export.ir_len_ms = 300.0
    s.export.export_16k = False          # 采集就是 16k, 不再重复导出
    s.export.raw_format = "WAV"
    s.protocol.distances_cm = [100]
    s.protocol.heights_cm = [160]
    s.protocol.dense_distance_cm = 100
    s.protocol.dense_height_cm = 160
    s.protocol.dense_steps = 3
    s.protocol.sparse_steps = 2
    s.protocol.speaker_orientations = [0]
    s.protocol.random_positions = 1
    s.protocol.rewearing_rings = 1
    s.protocol.rewearing_steps = 2
    s.protocol.settle_s = 3.0
    s.protocol.setup_settle_s = 3.0
    return s


def test_full_session_run(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "AudioEngine", FakeEngine)
    settings = _settings(tmp_path)
    plan = build_plan(settings.protocol)
    session = Session(settings.session_root, "t1")
    renderer = P.PromptRenderer(FS, session.prompts_dir, enabled=False)

    logs: list[str] = []
    runner = SessionRunner(settings, plan, session, renderer,
                           RunnerHooks(on_log=logs.append))
    runner.run()

    rows = session.load_manifest()
    # 距离/高度各只有一档且正是基准圈 → 没有稀疏圈: 基准圈 3 + 重戴 2 + 随机 1
    assert len(rows) == len(plan.measures) == 3 + 2 + 1
    assert {r["take_id"] for r in rows} == {s.take_id for s in plan.measures}

    for r in rows:
        qc = r["qc"]
        assert qc["verdict"] == "PASS", f"{r['take_id']}: {qc['reasons']}"
        assert qc["repeat_ncc"] > 0.99
        # 未知延迟必须被正确估回。测得值 = 保护间隔 + 设备延迟 + IR 直达偏移(30)
        guard = int(settings.sweep.guard_s * FS)
        assert abs(r["latency_samples"] - (guard + LATENCY + 30)) <= 2
        assert r["n_averaged"] == 2
        assert r["channels"] == ["mic1", "mic2", "mic3", "vpu"]
        assert r["mic_cols"] == [0, 1, 2]

        ir_path = session.dir / r["ir_file"]
        assert ir_path.exists()
        data, fs = sf.read(str(ir_path), always_2d=True)
        assert fs == FS
        assert data.shape == (int(0.3 * FS), 4)
        # 直达峰应落在 pre_samples 处 (默认 5ms)
        pre = int(settings.export.ir_pre_ms * 1e-3 * FS)
        assert abs(int(np.argmax(np.abs(data).sum(axis=1))) - pre) <= 1
        assert (session.dir / r["raw_file"]).exists()

    assert (session.dir / "qc.csv").exists()
    stats = session.stats()
    assert stats["PASS"] == len(rows) and stats["FAIL"] == 0
    assert any("PASS" in line for line in logs)


def test_rerun_only_selected(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "AudioEngine", FakeEngine)
    settings = _settings(tmp_path)
    plan = build_plan(settings.protocol)
    session = Session(settings.session_root, "t2")
    renderer = P.PromptRenderer(FS, session.prompts_dir, enabled=False)

    target = plan.measures[1].take_id
    runner = SessionRunner(settings, plan, session, renderer,
                           only_take_ids={target})
    runner.run()

    rows = session.load_manifest()
    assert [r["take_id"] for r in rows] == [target]


def test_voice_prompt_is_played_before_the_sweep(tmp_path, monkeypatch):
    """语音指导词必须真的进到播放缓冲区里, 且落在扫频窗之前的静音期。"""
    from marray_capture.audio import tts

    voice_s = 1.2
    rate = 22050

    class VoiceBackend:
        name = "fake"
        last_error = ""

        def available(self): return True
        def voices(self): return [tts.Voice("v1", "假语音")]
        def default_voice(self): return "v1"

        def synth(self, text, out_wav, voice_id):
            n = int(voice_s * rate)
            t = np.arange(n) / rate
            sf.write(str(out_wav), (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), rate)

    monkeypatch.setattr(tts, "pick_backend",
                        lambda preferred="auto", models_dir=None: (VoiceBackend(), ""))
    monkeypatch.setattr(runner_mod, "AudioEngine", FakeEngine)

    settings = _settings(tmp_path)
    settings.tts_enabled = True
    plan = build_plan(settings.protocol)
    session = Session(settings.session_root, "voice")
    renderer = P.PromptRenderer(FS, session.prompts_dir, enabled=True)
    assert renderer.ok

    FakeEngine.played.clear()
    SessionRunner(settings, plan, session, renderer).run()
    assert FakeEngine.played

    # 关掉语音再跑一遍, 两者播放缓冲区的长度差应约等于语音时长
    FakeEngine.played.clear()
    quiet = P.PromptRenderer(FS, tmp_path / "empty_cache", enabled=False)
    settings.tts_enabled = False
    SessionRunner(settings, plan, Session(settings.session_root, "quiet"), quiet).run()
    silent_len = len(FakeEngine.played[0])

    FakeEngine.played.clear()
    settings.tts_enabled = True
    SessionRunner(settings, plan, Session(settings.session_root, "voice2"), renderer).run()
    voiced = FakeEngine.played[0]

    # 有语音时, 播放缓冲区应该比降级版长「语音 + 0.4s 停顿 - 降级用的提示音」
    fallback_s = (len(P.cue_ready(FS)) + len(P.silence(FS, 0.3))) / FS
    delta_s = (len(voiced) - silent_len) / FS
    assert abs(delta_s - (voice_s + 0.4 - fallback_s)) < 0.05

    # 语音落在开头, 而扫频前的 preroll 必须是静音 (否则会污染质检的噪声窗)
    assert np.max(np.abs(voiced[:int(voice_s * FS)])) > 0.1
    sw = settings.sweep
    exc_s = (sw.preroll_s + sw.repeats * sw.duration_s
             + (sw.repeats - 1) * sw.gap_s + sw.tail_s)
    lead = len(voiced) - int(round(exc_s * FS))
    preroll = voiced[lead: lead + int(sw.preroll_s * FS)]
    assert np.max(np.abs(preroll)) < 1e-9


def test_stop_halts_run(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "AudioEngine", FakeEngine)
    settings = _settings(tmp_path)
    plan = build_plan(settings.protocol)
    session = Session(settings.session_root, "t3")
    renderer = P.PromptRenderer(FS, session.prompts_dir, enabled=False)

    runner = SessionRunner(settings, plan, session, renderer)
    finished: list[str] = []
    runner.hooks = RunnerHooks(
        on_take=lambda *_: runner.stop(),
        on_finished=finished.append,
    )
    runner.run()

    assert finished and finished[0] == "已停止"
    assert len(session.load_manifest()) == 1
