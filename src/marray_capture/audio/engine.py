"""播放/录音引擎。两种模式:

**全双工 (duplex)** —— 输入输出是同一台设备时用。一条 ``sd.Stream`` 同时收发,
两端采样锁定: 没有时钟漂移, 延迟只有设备本身的往返延迟。**ASIO 必须走这条路** ——
绝大多数 ASIO 驱动 (含 ASIO4ALL) 是单实例独占的, 同一块 ASIO 设备开两条流会失败。

**分离双流 (split)** —— 输入输出是两台设备时用 (USB 声卡录音 + 蓝牙音箱播放),
此时 sd.playrec / duplex 都用不了:

    输入流先启动 → 等一个保护间隔 → 输出流播放 → 播完再多录 tail → 停输入

两条流时钟不同源, 录音里扫频的绝对位置未知, 且会随时间漂移。

两种模式下, 播放缓冲区的布局都是 ``[guard 静音][play][tail 静音]``, 录音从 0 开始,
所以「录音域到达位置 = guard + starts[k] + 设备延迟」这条关系对两者一致 ——
上层的峰值定位逻辑不用区分模式。对齐一律在反卷积域按峰值做
(见 sweep.find_direct_index), 不依赖流之间的时序。
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import numpy as np
import sounddevice as sd

from ..settings import AudioConfig

LevelCallback = Callable[[np.ndarray], None]   # 每块的 per-channel RMS (线性)


class EngineError(RuntimeError):
    pass


class AudioEngine:
    """一次只跑一个采集动作; 上层负责串行化。"""

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self.last_warnings: str = ""
        self._abort = threading.Event()

    # ------------------------------------------------------------------ 控制
    def abort(self) -> None:
        self._abort.set()

    def _reset(self) -> None:
        self._abort.clear()
        self.last_warnings = ""

    @property
    def aborted(self) -> bool:
        return self._abort.is_set()

    # ------------------------------------------------------------------ 工具
    def _to_output(self, mono: np.ndarray) -> np.ndarray:
        gain = 10.0 ** (self.cfg.output_gain_db / 20.0)
        x = np.clip(np.asarray(mono, dtype=float) * gain, -1.0, 1.0).astype(np.float32)
        return np.repeat(x[:, None], max(1, int(self.cfg.output_channels)), axis=1)

    @staticmethod
    def _pump(
        q: "queue.Queue[np.ndarray]",
        sink: list[np.ndarray],
        level_cb: LevelCallback | None,
        timeout: float,
    ) -> int:
        """取一块录音数据。返回帧数, 队列空返回 0。"""
        try:
            blk = q.get(timeout=timeout) if timeout > 0 else q.get_nowait()
        except queue.Empty:
            return 0
        sink.append(blk)
        if level_cb is not None and len(blk):
            level_cb(np.sqrt((blk.astype(float) ** 2).mean(axis=0)))
        return len(blk)

    def _drain_rest(self, q, sink, level_cb) -> None:
        while self._pump(q, sink, level_cb, 0.0):
            pass

    # ------------------------------------------------------------------ 模式
    def use_duplex(self) -> bool:
        """是否走全双工单流。"""
        mode = (self.cfg.duplex_mode or "auto").lower()
        if mode == "duplex":
            return True
        if mode == "split":
            return False
        # auto: 同一台设备就全双工 (ASIO 只能这样)
        return (self.cfg.input_device is not None
                and self.cfg.input_device == self.cfg.output_device)

    # -------------------------------------------------------------- 播放+录音
    def play_record(
        self,
        play_mono: np.ndarray,
        n_in_channels: int,
        extra_tail_s: float = 0.0,
        guard_s: float = 0.4,
        level_cb: LevelCallback | None = None,
    ) -> np.ndarray:
        """播放 play_mono 的同时录音, 返回 (T, n_in_channels)。

        录音窗 = guard + 播放时长 + extra_tail, 保证蓝牙的大延迟也落在窗内。
        """
        if self.use_duplex():
            return self._play_record_duplex(
                play_mono, n_in_channels, extra_tail_s, guard_s, level_cb)
        return self._play_record_split(
            play_mono, n_in_channels, extra_tail_s, guard_s, level_cb)

    # ---------------------------------------------------------------- 全双工
    def _play_record_duplex(
        self,
        play_mono: np.ndarray,
        n_in_channels: int,
        extra_tail_s: float,
        guard_s: float,
        level_cb: LevelCallback | None,
    ) -> np.ndarray:
        """一条流同时收发。录音与播放逐样本对齐, 无时钟漂移。"""
        self._reset()
        fs = self.cfg.samplerate
        if n_in_channels <= 0:
            raise EngineError("没有选择任何输入通道")

        play_buf = self._to_output(play_mono)
        n_out = play_buf.shape[1]
        guard = int(round(guard_s * fs))
        total = guard + len(play_buf) + int(round(extra_tail_s * fs))

        out_buf = np.zeros((total, n_out), dtype=np.float32)
        out_buf[guard: guard + len(play_buf)] = play_buf
        rec = np.zeros((total, n_in_channels), dtype=np.float32)

        pos = {"i": 0}
        warnings: set[str] = set()
        level_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
        finished = threading.Event()
        blk = {"n": 0}

        def cb(indata, outdata, frames, time_info, status):  # noqa: ANN001
            if status:
                warnings.add(f"全双工流 xrun: {status}")
            i = pos["i"]
            n = min(frames, total - i)
            outdata[:n] = out_buf[i: i + n]
            if n < frames:
                outdata[n:] = 0.0
            if n > 0:
                rec[i: i + n] = indata[:n]
            pos["i"] = i + n
            # 电平不在音频线程里算完就发, 只丢进队列, 由等待循环取
            blk["n"] += 1
            if level_cb is not None and blk["n"] % 4 == 0:
                try:
                    level_q.put_nowait(
                        np.sqrt((indata[:n].astype(np.float64) ** 2).mean(axis=0)))
                except queue.Full:
                    pass
            if self._abort.is_set():
                raise sd.CallbackAbort
            if pos["i"] >= total:
                raise sd.CallbackStop

        try:
            stream = sd.Stream(
                device=(self.cfg.input_device, self.cfg.output_device),
                channels=(n_in_channels, n_out),
                samplerate=fs, dtype="float32",
                blocksize=int(self.cfg.blocksize or 0),
                latency=(self.cfg.input_latency, self.cfg.output_latency),
                callback=cb, finished_callback=finished.set,
            )
            with stream:
                # 给足余量: 缓冲区时长 + 5 秒, 防止驱动异常时死等
                deadline = time.monotonic() + total / fs + 5.0
                while not finished.wait(0.1):
                    self._drain_levels(level_q, level_cb)
                    if self._abort.is_set() or time.monotonic() > deadline:
                        break
        except sd.PortAudioError as e:
            raise EngineError(
                f"打开全双工流失败: {e}\n"
                "ASIO 设备请确认输入输出在驱动面板里选的是同一块设备, "
                "且采样率/缓冲区与软件一致。"
            ) from e
        finally:
            self._drain_levels(level_q, level_cb)

        self.last_warnings = "; ".join(sorted(warnings))
        if self._abort.is_set():
            raise EngineError("已中止")
        got = pos["i"]
        if got <= 0:
            raise EngineError("没有录到任何数据, 检查输入设备")
        return rec[:got]

    @staticmethod
    def _drain_levels(q: "queue.Queue[np.ndarray]", level_cb: LevelCallback | None) -> None:
        while True:
            try:
                rms = q.get_nowait()
            except queue.Empty:
                return
            if level_cb is not None:
                level_cb(rms)

    # ------------------------------------------------------------ 分离双流
    def _play_record_split(
        self,
        play_mono: np.ndarray,
        n_in_channels: int,
        extra_tail_s: float,
        guard_s: float,
        level_cb: LevelCallback | None,
    ) -> np.ndarray:
        self._reset()
        fs = self.cfg.samplerate
        if n_in_channels <= 0:
            raise EngineError("没有选择任何输入通道")

        play_buf = self._to_output(play_mono)
        rec_q: "queue.Queue[np.ndarray]" = queue.Queue()
        chunks: list[np.ndarray] = []
        play_done = threading.Event()
        warnings: set[str] = set()

        def in_cb(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                warnings.add(f"输入流 xrun: {status}")
            rec_q.put(indata.copy())
            if self._abort.is_set():
                raise sd.CallbackAbort

        pos = {"i": 0}

        def out_cb(outdata, frames, time_info, status):  # noqa: ANN001
            if status:
                warnings.add(f"输出流 xrun: {status}")
            i = pos["i"]
            end = min(i + frames, len(play_buf))
            n = end - i
            outdata[:n] = play_buf[i:end]
            if n < frames:
                outdata[n:] = 0.0
            pos["i"] = end
            if self._abort.is_set():
                play_done.set()
                raise sd.CallbackAbort
            if end >= len(play_buf):
                play_done.set()
                raise sd.CallbackStop

        blocksize = int(self.cfg.blocksize or 0)
        in_stream = sd.InputStream(
            device=self.cfg.input_device, channels=n_in_channels, samplerate=fs,
            dtype="float32", blocksize=blocksize,
            latency=self.cfg.input_latency, callback=in_cb,
        )
        out_stream = sd.OutputStream(
            device=self.cfg.output_device, channels=max(1, int(self.cfg.output_channels)),
            samplerate=fs, dtype="float32", blocksize=blocksize,
            latency=self.cfg.output_latency, callback=out_cb,
        )

        guard_frames = int(round(guard_s * fs))
        total_frames = guard_frames + len(play_buf) + int(round(extra_tail_s * fs))
        collected = 0
        idle_rounds = 0
        try:
            in_stream.start()
            while collected < guard_frames and not self._abort.is_set():
                n = self._pump(rec_q, chunks, level_cb, 0.5)
                if n == 0:
                    idle_rounds += 1
                    if idle_rounds > 8:
                        raise EngineError("输入流启动后 4 秒没有数据, 检查声卡")
                collected += n
            idle_rounds = 0
            out_stream.start()
            while collected < total_frames and not self._abort.is_set():
                n = self._pump(rec_q, chunks, level_cb, 0.5)
                collected += n
                if n == 0:
                    idle_rounds += 1
                    if play_done.is_set() or idle_rounds > 8:
                        break
                else:
                    idle_rounds = 0
        finally:
            for st in (out_stream, in_stream):
                try:
                    st.stop()
                    st.close()
                except Exception:
                    pass
            self._drain_rest(rec_q, chunks, level_cb)

        self.last_warnings = "; ".join(sorted(warnings))
        if self._abort.is_set():
            raise EngineError("已中止")
        if not chunks:
            raise EngineError("没有录到任何数据, 检查输入设备")
        return np.concatenate(chunks, axis=0)

    # ------------------------------------------------------------------ 只录音
    def record(
        self,
        seconds: float,
        n_in_channels: int,
        level_cb: LevelCallback | None = None,
    ) -> np.ndarray:
        """纯录音 (底噪测量 / 电平检查)。"""
        self._reset()
        fs = self.cfg.samplerate
        rec_q: "queue.Queue[np.ndarray]" = queue.Queue()
        chunks: list[np.ndarray] = []

        def in_cb(indata, frames, time_info, status):  # noqa: ANN001
            rec_q.put(indata.copy())
            if self._abort.is_set():
                raise sd.CallbackAbort

        stream = sd.InputStream(
            device=self.cfg.input_device, channels=n_in_channels, samplerate=fs,
            dtype="float32", blocksize=int(self.cfg.blocksize or 0),
            latency=self.cfg.input_latency, callback=in_cb,
        )
        need = int(round(seconds * fs))
        got, idle = 0, 0
        try:
            stream.start()
            while got < need and not self._abort.is_set():
                n = self._pump(rec_q, chunks, level_cb, 0.5)
                got += n
                if n == 0:
                    idle += 1
                    if idle > 8:
                        break
                else:
                    idle = 0
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            self._drain_rest(rec_q, chunks, level_cb)
        if not chunks:
            raise EngineError("没有录到任何数据, 检查输入设备")
        return np.concatenate(chunks, axis=0)[:need]

    # ------------------------------------------------------------------ 只播放
    def play(self, mono: np.ndarray) -> None:
        """播放提示音/语音, 阻塞直到播完。"""
        buf = self._to_output(mono)
        sd.play(buf, samplerate=self.cfg.samplerate, device=self.cfg.output_device, blocking=False)
        try:
            sd.wait()
        except Exception:
            pass


def slice_channels(rec: np.ndarray, indices: list[int]) -> np.ndarray:
    """从 (T, Nmax) 里按物理通道号取通道, 顺序与 indices 一致。"""
    if not indices:
        return rec[:, :0]
    if rec.shape[1] <= max(indices):
        raise EngineError(f"录到 {rec.shape[1]} 通道, 但需要通道 {max(indices) + 1}")
    return rec[:, indices]
