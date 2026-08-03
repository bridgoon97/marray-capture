"""会话执行器: 把方案跑成一次真实采集。

每个测量位置只发起**一次** play_record, 播放缓冲区的布局是::

    [语音指令][停顿][位置编号提示音][倒计时][preroll 静音][扫频][gap][扫频][tail 静音]
                                                  ^ 噪声窗        ^ 信号窗

好处:
- 指令、编号音、扫频在同一个录音文件里, 事后可追溯是哪个位置 (蓝牙/操作失误时救命)。
- 指令与扫频之间没有流的启停间隙, 佩戴者不会被"什么时候开始"搞糊涂。
- 编号提示音编码了位置序号, 万一 manifest 和音频对不上可以从音频里读回来。

执行器本身不依赖 Qt, GUI 用回调接。
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np

from ..audio import prompts as P
from ..audio.engine import AudioEngine, EngineError, map_levels, slice_channels
from ..audio.sweep import Sweep, build_excitation, generate_ess
from ..qc.metrics import FAIL, TakeQC, evaluate_take
from ..rir.extract import (
    PeakNotFoundError, TakeIR, correct_drift, deconvolve_take, extract_take,
    resample_ir,
)
from ..settings import AppSettings
from ..store import Session
from .plan import Plan, Step


@dataclass
class RunnerHooks:
    on_step: Callable[[Step, int, int], None] | None = None
    on_state: Callable[[str], None] | None = None
    on_level: Callable[[np.ndarray], None] | None = None
    on_take: Callable[[Step, TakeQC, dict], None] | None = None
    on_log: Callable[[str], None] | None = None
    on_finished: Callable[[str], None] | None = None
    on_progress: Callable[[int, int], None] | None = None

    def call(self, name: str, *args) -> None:
        fn = getattr(self, name, None)
        if fn is not None:
            try:
                fn(*args)
            except Exception:
                pass


def channel_layout(settings: AppSettings) -> tuple[list[int], list[str], list[int]]:
    """返回 (物理通道索引, 标签, 麦克风在切片后的列号)。

    索引是**落盘顺序**而非物理顺序 —— 麦克风按用户指定的序号排在最前面, 所以
    mic_cols 恒为 [0, 1, ... n-1], 与阵列几何的坐标行天然对齐。
    """
    active = settings.audio.ordered_channels()
    indices = [c.index for c in active]
    labels = [c.display() for c in active]
    mic_cols = [i for i, c in enumerate(active) if c.role == "mic"]
    return indices, labels, mic_cols


def make_sweep(settings: AppSettings) -> Sweep:
    s = settings.sweep
    return generate_ess(
        fs=settings.audio.samplerate, f_start=s.f_start, f_end=s.f_end,
        duration_s=s.duration_s, fade_in_ms=s.fade_in_ms, fade_out_ms=s.fade_out_ms,
    )


class SessionRunner:
    """跑一遍方案。run() 是阻塞的, 由 GUI 放到工作线程里。"""

    def __init__(
        self,
        settings: AppSettings,
        plan: Plan,
        session: Session,
        renderer: P.PromptRenderer,
        hooks: RunnerHooks | None = None,
        start_index: int = 0,
        only_take_ids: set[str] | None = None,
    ):
        self.settings = settings
        self.plan = plan
        self.session = session
        self.renderer = renderer
        self.hooks = hooks or RunnerHooks()
        self.start_index = start_index
        # 兜底: 上游要是把非集合的东西传进来 (Qt 的 clicked 会塞一个 bool), 当成"全跑"
        self.only_take_ids = set(only_take_ids) if isinstance(
            only_take_ids, (set, frozenset, list, tuple)) else None

        self.engine = AudioEngine(settings.audio)
        self.sweep = make_sweep(settings)
        self.ch_indices, self.ch_labels, self.mic_cols = channel_layout(settings)
        # VPU 是非声学通道, SNR/DDR 偏低是物理必然, 质检不当判据 (见 _apply_thresholds)
        self.vpu_cols = [i for i, c in enumerate(settings.audio.ordered_channels())
                         if c.role == "vpu"]

        self._pause = threading.Event()
        self._stop = threading.Event()
        self._skip = threading.Event()
        self._redo = threading.Event()
        self.current_index = start_index

    # ------------------------------------------------------------------ 控制
    def pause(self) -> None:
        self._pause.set()
        self.hooks.call("on_state", "已暂停")

    def resume(self) -> None:
        self._pause.clear()

    def stop(self) -> None:
        self._stop.set()
        self._pause.clear()
        self.engine.abort()

    def skip(self) -> None:
        self._skip.set()

    def redo(self) -> None:
        self._redo.set()

    def _wait_if_paused(self) -> None:
        while self._pause.is_set() and not self._stop.is_set():
            time.sleep(0.1)

    # ------------------------------------------------------------------ 主循环
    def run(self) -> None:
        reason = "完成"
        try:
            steps = self.plan.steps
            todo = list(range(self.start_index, len(steps)))
            if self.only_take_ids is not None:
                todo = [i for i in todo
                        if steps[i].kind == "setup" or steps[i].take_id in self.only_take_ids]
            total = len(todo)
            for n, i in enumerate(todo):
                if self._stop.is_set():
                    reason = "已停止"
                    break
                self._wait_if_paused()
                if self._stop.is_set():
                    reason = "已停止"
                    break
                step = steps[i]
                self.current_index = i
                self.hooks.call("on_step", step, n + 1, total)
                self.hooks.call("on_progress", n + 1, total)
                if step.kind == "setup":
                    self._run_setup(step)
                else:
                    self._run_measure(step)
        except EngineError as e:
            # 用户在录音中途按停止时, 引擎也会抛 EngineError, 别报成音频故障
            reason = "已停止" if self._stop.is_set() else f"音频错误: {e}"
            self.hooks.call("on_log", reason)
        except Exception as e:  # 采集中途崩了要把栈打出来, 别静默
            reason = f"异常: {e}"
            self.hooks.call("on_log", reason + "\n" + traceback.format_exc())
        finally:
            try:
                self.session.write_qc_csv()
            except Exception:
                pass
            self.hooks.call("on_finished", reason)

    # ------------------------------------------------------------------ 单步
    def _leadin(self, step: Step, countdown_s: float) -> np.ndarray:
        fs = self.settings.audio.samplerate
        parts: list[np.ndarray] = []
        voice = self.renderer.render(step.instruction) if self.settings.tts_enabled else np.zeros(0)
        if len(voice):
            parts += [voice, P.silence(fs, 0.4)]
        else:
            parts += [P.cue_ready(fs), P.silence(fs, 0.3)]
        if step.kind == "measure":
            parts += [P.cue_index(fs, step.idx), P.silence(fs, 0.3)]
        cd = max(2, int(round(countdown_s)))
        parts.append(P.cue_countdown(fs, cd))
        return np.concatenate(parts) if parts else np.zeros(0)

    def _run_setup(self, step: Step) -> None:
        self.hooks.call("on_state", step.instruction)
        lead = self._leadin(step, step.settle_s)
        # 引导语走可中止/可挂起的 play: 按停止立刻断, 按暂停原地停住
        self.engine.play(lead, pause_event=self._pause)

    def _run_measure(self, step: Step, attempt: int = 1) -> None:
        st = self.settings
        fs = st.audio.samplerate
        self.hooks.call("on_state", step.instruction)

        lead = self._leadin(step, step.settle_s)
        exc, starts = build_excitation(
            self.sweep, st.sweep.repeats, st.sweep.preroll_s, st.sweep.gap_s,
            st.sweep.tail_s, st.sweep.amplitude,
        )
        play = np.concatenate([lead, exc])
        starts = [s + len(lead) for s in starts]

        n_rec_ch = st.audio.n_record_channels()
        rec_all = self.engine.play_record(
            play, n_rec_ch,
            extra_tail_s=st.sweep.max_latency_s + 0.3,
            guard_s=st.sweep.guard_s,
            level_cb=lambda r: self.hooks.call("on_level", map_levels(r, self.ch_indices)),
        )
        warnings = self.engine.last_warnings
        rec = slice_channels(rec_all, self.ch_indices)

        if st.audio.drift_correction and abs(st.audio.drift_ppm) > 1.0:
            rec = correct_drift(rec, st.audio.drift_ppm)

        self.hooks.call("on_state", "反卷积与质检…")
        deconv = deconvolve_take(rec, self.sweep)
        # 搜索窗要同时覆盖保护间隔和设备延迟 —— 录音是在播放之前 guard_s 就开始的
        try:
            take = extract_take(
                deconv, self.sweep, starts,
                ir_pre_ms=st.export.ir_pre_ms, ir_len_ms=st.export.ir_len_ms,
                max_latency_s=st.sweep.guard_s + st.sweep.max_latency_s,
                energy_channels=self.mic_cols or None,
                known_latency=st.audio.measured_latency_samples,
                average=st.export.average_repeats,
            )
            qc = evaluate_take(
                rec=rec, deconv=deconv, ir_list=take.irs, ir_avg=take.ir_avg,
                peaks=take.direct_indices, latency=take.latency_samples,
                drift_ppm=take.drift_ppm, starts=starts, sweep_n=self.sweep.n,
                pre_samples=take.pre_samples, fs=fs, labels=self.ch_labels,
                mic_cols=self.mic_cols, thr=st.qc, stream_warnings=warnings,
                vpu_cols=self.vpu_cols,
            )
            # 一致性不过就别平均, 亚样本错位会梳状衰减高频
            if qc.repeat_ncc == qc.repeat_ncc and qc.repeat_ncc < st.qc.min_repeat_ncc:
                take.use_single()
        except PeakNotFoundError as e:
            # 直达峰定位失败 (设备延迟超搜索窗/没录到信号): 不反卷积下游,
            # 直接记 FAIL, 让用户去查 output_latency/声卡, 而不是被垃圾指标误导。
            pre = int(round(st.export.ir_pre_ms * 1e-3 * fs))
            length = int(round(st.export.ir_len_ms * 1e-3 * fs))
            take = TakeIR(
                ir_avg=np.zeros((length, rec.shape[1]), dtype=float),
                direct_indices=[0], latency_samples=0, pre_samples=pre,
                fs=fs, n_averaged=1)
            qc = TakeQC(drift_ppm=0.0, latency_ms=0.0, stream_warnings=warnings)
            qc.worsen(FAIL, str(e))

        files = self._persist(step, rec, take, attempt)
        record = {
            **step.to_dict(),
            "attempt": attempt,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "fs": fs,
            "channels": self.ch_labels,
            "mic_cols": self.mic_cols,
            "n_averaged": take.n_averaged,
            "latency_samples": take.latency_samples,
            **files,
            "qc": qc.to_dict(),
        }
        self.session.append_manifest(record)
        self.hooks.call("on_take", step, qc, record)
        self.hooks.call("on_log", f"[{qc.verdict}] {step.take_id} — {qc.summary()}")

        # 一条采完了: 暂停在这里挂住, 先别播反馈音也别进下一步
        self._wait_if_paused()
        if self._stop.is_set():
            return
        # 反馈音: 过了上行两声, 没过下行两声
        self.engine.play(P.cue_fail(fs) if qc.verdict == FAIL else P.cue_done(fs),
                         pause_event=self._pause)
        if self._stop.is_set():
            return

        if self._redo.is_set():
            self._redo.clear()
            self._run_measure(step, attempt + 1)
            return
        if self._skip.is_set():
            self._skip.clear()
            return
        if qc.verdict == FAIL and self.settings.auto_retry_on_fail and attempt == 1:
            self.hooks.call("on_log", f"{step.take_id} 质检不过, 自动重录一次")
            retry = Step(**{**step.to_dict(), "instruction": "刚才那次没通过，保持原位不要动，马上重录一次。"})
            self._run_measure(retry, attempt + 1)

    # ------------------------------------------------------------------ 落盘
    def _persist(self, step: Step, rec: np.ndarray, take, attempt: int) -> dict:
        st = self.settings
        tid = step.take_id if attempt == 1 else f"{step.take_id}_try{attempt}"
        out: dict[str, str] = {"take_id": tid}
        if st.export.save_raw:
            p = self.session.save_raw(tid, rec, st.audio.samplerate, st.export.raw_format)
            out["raw_file"] = str(p.relative_to(self.session.dir))
        p48 = self.session.save_ir(tid, take.ir_avg, st.audio.samplerate)
        out["ir_file"] = str(p48.relative_to(self.session.dir))
        if st.export.export_16k and st.audio.samplerate != 16000:
            ir16 = resample_ir(take.ir_avg, st.audio.samplerate, 16000)
            p16 = self.session.save_ir(tid, ir16, 16000)
            out["ir16k_file"] = str(p16.relative_to(self.session.dir))
        return out


# ---------------------------------------------------------------------- 单次动作
def calibrate_latency(settings: AppSettings, hooks: RunnerHooks | None = None) -> dict:
    """播一次扫频测输出→输入的往返延迟与时钟漂移。

    蓝牙音箱每次连接的延迟都不一样, 每场开录前跑一次可以把后续的峰值搜索窗收窄,
    也能提前发现"根本没声音"这类接线问题。
    """
    hooks = hooks or RunnerHooks()
    engine = AudioEngine(settings.audio)
    sweep = make_sweep(settings)
    indices, labels, mic_cols = channel_layout(settings)
    exc, starts = build_excitation(
        sweep, max(2, settings.sweep.repeats), settings.sweep.preroll_s,
        settings.sweep.gap_s, settings.sweep.tail_s, settings.sweep.amplitude,
    )
    rec_all = engine.play_record(
        exc, settings.audio.n_record_channels(),
        extra_tail_s=settings.sweep.max_latency_s + 0.3,
        guard_s=settings.sweep.guard_s,
        level_cb=lambda r: hooks.call("on_level", map_levels(r, indices)),
    )
    rec = slice_channels(rec_all, indices)
    deconv = deconvolve_take(rec, sweep)
    take = extract_take(
        deconv, sweep, starts, settings.export.ir_pre_ms, settings.export.ir_len_ms,
        settings.sweep.guard_s + settings.sweep.max_latency_s,
        energy_channels=mic_cols or None, average=False,
    )
    fs = settings.audio.samplerate
    # 报给用户的是扣掉保护间隔后的真实设备延迟; 存进 settings 的仍是含保护间隔的原值,
    # 因为后续搜索窗是按同一个原点算的。
    device_latency = take.latency_samples - int(round(settings.sweep.guard_s * fs))
    return {
        "latency_samples": take.latency_samples,
        "latency_ms": device_latency / fs * 1000.0,
        "drift_ppm": take.drift_ppm,
        "peaks": take.direct_indices,
        "ir": take.irs[0] if take.irs else None,
        "labels": labels,
        "warnings": engine.last_warnings,
    }


def record_reference_ir(settings: AppSettings, session: Session,
                        hooks: RunnerHooks | None = None) -> dict:
    """录一条音箱参考 IR（用于去音箱响应）。**整场只需要录一次。**

    它和采集网格无关: 音箱的频响不随位置变化, 所以不用每个角度都录。
    做法是把测量麦放在音箱正轴约 1 米处, 播一次扫频, 提取 IR 存下来。

    也不需要长期固定某个通道作参考麦 —— 只要录这一条时测量麦插在某个被勾选的通道上,
    后处理页用「参考通道」挑出那一列即可。若通道表里已经有 `ref` 角色, 会自动选它。
    """
    import soundfile as sf

    hooks = hooks or RunnerHooks()
    engine = AudioEngine(settings.audio)
    sweep = make_sweep(settings)
    indices, labels, mic_cols = channel_layout(settings)
    if not indices:
        raise EngineError("还没勾选任何输入通道")

    exc, starts = build_excitation(
        sweep, max(2, settings.sweep.repeats), settings.sweep.preroll_s,
        settings.sweep.gap_s, settings.sweep.tail_s, settings.sweep.amplitude,
    )
    rec_all = engine.play_record(
        exc, settings.audio.n_record_channels(),
        extra_tail_s=settings.sweep.max_latency_s + 0.3,
        guard_s=settings.sweep.guard_s,
        level_cb=lambda r: hooks.call("on_level", map_levels(r, indices)),
    )
    rec = slice_channels(rec_all, indices)
    deconv = deconvolve_take(rec, sweep)
    take = extract_take(
        deconv, sweep, starts, settings.export.ir_pre_ms, settings.export.ir_len_ms,
        settings.sweep.guard_s + settings.sweep.max_latency_s,
        energy_channels=None, average=settings.export.average_repeats,
    )

    ref_cols = [i for i, c in enumerate(settings.audio.ordered_channels()) if c.role == "ref"]
    fs = settings.audio.samplerate
    out_dir = session.dir / "ref"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ref_{datetime.now():%Y%m%d_%H%M%S}.wav"
    sf.write(str(path), np.asarray(take.ir_avg, dtype=np.float32), fs, subtype="FLOAT")

    return {
        "path": str(path),
        "labels": labels,
        "ref_col": ref_cols[0] if ref_cols else -1,
        "mic_cols": mic_cols,
        "fs": fs,
        "ir": take.ir_avg,
        "drift_ppm": take.drift_ppm,
        "warnings": engine.last_warnings,
    }


def measure_ambient(settings: AppSettings, seconds: float = 30.0,
                    session: Session | None = None,
                    hooks: RunnerHooks | None = None) -> dict:
    """录一段环境底噪, 报每通道噪声电平与倍频带谱。"""
    hooks = hooks or RunnerHooks()
    engine = AudioEngine(settings.audio)
    indices, labels, _ = channel_layout(settings)
    rec_all = engine.record(seconds, settings.audio.n_record_channels(),
                            level_cb=lambda r: hooks.call("on_level", map_levels(r, indices)))
    rec = slice_channels(rec_all, indices)
    fs = settings.audio.samplerate
    levels = 20.0 * np.log10(np.sqrt((rec ** 2).mean(axis=0)) + 1e-12)
    path = ""
    if session is not None:
        import soundfile as sf
        p = session.dir / "noise" / f"ambient_{datetime.now():%Y%m%d_%H%M%S}.wav"
        sf.write(str(p), rec.astype(np.float32), fs, subtype="FLOAT")
        path = str(p.relative_to(session.dir))
    return {"labels": labels, "rms_dbfs": levels.tolist(), "seconds": seconds, "file": path,
            "data": rec, "fs": fs}
