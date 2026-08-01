"""语音导播: 把指令文本渲染成音频, 供采集时通过音箱播给佩戴者听。

设计取舍:
- 提示语走**和扫频同一个输出设备** (音箱)。佩戴者戴着待测耳机, 不能用耳机放提示音,
  否则会被自己的麦克风录进去。
- 提示语只在「稳定期」之前播, 扫频窗内是纯静音, 所以提示音不会污染 IR。
- 所有提示在方案生成时**预渲染并缓存**, 采集过程中零 TTS 延迟, 现场断网也照跑。
- 缓存是**跨会话共享**的 (``~/.marray-capture/prompt_cache``), 键含后端与语音,
  换人换场次不必重新合成。预渲染时会另外把音频导出一份到会话目录留档。
- TTS 全不可用时自动降级为提示音编码 + GUI 大字显示, 流程不中断。

具体后端 (piper / edge-tts / SAPI) 见 :mod:`marray_capture.audio.tts`。
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from ..settings import APP_DIR
from . import tts

SHARED_CACHE = APP_DIR / "prompt_cache"


class PromptRenderer:
    """文本 → 单声道 float 波形。带磁盘缓存与后端自动选择。"""

    def __init__(
        self,
        fs: int,
        cache_dir: Path | None = None,
        voice: str = "",
        enabled: bool = True,
        backend: str = "auto",
        models_dir: Path | None = None,
    ):
        self.fs = int(fs)
        self.cache_dir = Path(cache_dir or SHARED_CACHE)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.voice = voice
        self.enabled = enabled
        self.models_dir = models_dir
        self.requested = backend
        self.last_error = ""
        self._mem: dict[str, np.ndarray] = {}

        self._backend = None
        self.backend = "none"
        if enabled and backend != "none":
            self._backend, notes = tts.pick_backend(backend, models_dir)
            if self._backend is not None:
                self.backend = self._backend.name
                if not self.voice:
                    self.voice = self._backend.default_voice()
            self.last_error = notes

    # ------------------------------------------------------------------ 后端
    @property
    def ok(self) -> bool:
        return self._backend is not None

    def available_voices(self) -> list[tts.Voice]:
        return self._backend.voices() if self._backend else []

    def describe(self) -> str:
        if not self.enabled or self.requested == "none":
            return "语音导播已关闭, 采集时用提示音 + 屏幕大字"
        if self._backend is None:
            return f"没有可用的 TTS 后端, 将降级为提示音。{self.last_error}"
        label = tts.BACKEND_LABELS.get(self.backend, self.backend)
        return f"{label}, 语音: {self.voice or '默认'}"

    # ------------------------------------------------------------------ 渲染
    def _cache_path(self, text: str) -> Path:
        key = hashlib.sha1(
            f"{self.backend}|{self.voice}|{self.fs}|{text}".encode()
        ).hexdigest()[:16]
        return self.cache_dir / f"p_{key}.wav"

    def render(self, text: str) -> np.ndarray:
        """返回单声道波形。失败时返回空数组 (调用方用提示音兜底)。"""
        if not text or not self.enabled:
            return np.zeros(0)
        if text in self._mem:
            return self._mem[text]

        path = self._cache_path(text)
        if not path.exists():
            if self._backend is None:
                self._mem[text] = np.zeros(0)
                return self._mem[text]
            try:
                self._backend.synth(text, path, self.voice)
            except Exception as e:
                self.last_error = f"TTS 渲染失败: {e}"
                path.unlink(missing_ok=True)
                self._mem[text] = np.zeros(0)
                return self._mem[text]

        try:
            data, fs = sf.read(str(path), always_2d=True)
            wave = resample_to(data.mean(axis=1), fs, self.fs)
            peak = float(np.max(np.abs(wave))) if len(wave) else 0.0
            if peak > 0:
                wave = wave / peak * 0.7          # 与提示音大致等响
        except Exception as e:
            self.last_error = f"读取 TTS 音频失败: {e}"
            wave = np.zeros(0)
        self._mem[text] = wave
        return wave

    def prewarm(self, texts: list[str], progress=None) -> int:
        """预渲染一批文本。返回成功条数。"""
        ok = 0
        for i, t in enumerate(texts):
            if len(self.render(t)):
                ok += 1
            if progress is not None:
                progress(i + 1, len(texts))
        return ok

    def export(self, texts: list[str], out_dir: Path) -> Path:
        """把已渲染的提示音导出一份到会话目录, 附带文本索引, 方便开录前试听核对。"""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        index = []
        for i, t in enumerate(texts):
            wave = self.render(t)
            name = f"{i:03d}.wav"
            if len(wave):
                sf.write(str(out / name), wave.astype(np.float32), self.fs, subtype="FLOAT")
            else:
                name = ""
            index.append({"idx": i, "text": t, "file": name})
        meta = {"backend": self.backend, "voice": self.voice, "fs": self.fs, "items": index}
        p = out / "index.json"
        p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return p


# ---------------------------------------------------------------------- 提示音
def resample_to(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out or len(x) == 0:
        return np.asarray(x, dtype=float)
    frac = Fraction(int(fs_out), int(fs_in)).limit_denominator(1000)
    return resample_poly(np.asarray(x, dtype=float), frac.numerator, frac.denominator)


def tone(fs: int, freq: float, dur_s: float, amp: float = 0.3) -> np.ndarray:
    n = int(round(dur_s * fs))
    t = np.arange(n) / fs
    x = amp * np.sin(2 * np.pi * freq * t)
    fade = max(1, int(0.005 * fs))
    x[:fade] *= np.linspace(0, 1, fade)
    x[-fade:] *= np.linspace(1, 0, fade)
    return x


def silence(fs: int, dur_s: float) -> np.ndarray:
    return np.zeros(int(round(dur_s * fs)))


def cue_ready(fs: int) -> np.ndarray:
    """「准备」提示: 两声低音。"""
    return np.concatenate([tone(fs, 660, 0.12), silence(fs, 0.08), tone(fs, 660, 0.12)])


def cue_countdown(fs: int, seconds: int) -> np.ndarray:
    """倒计时: 每秒一声短音, 最后一声升高 —— 听到高音就别动了。"""
    parts = []
    for i in range(seconds):
        last = i == seconds - 1
        parts.append(tone(fs, 1320 if last else 880, 0.10, 0.35))
        parts.append(silence(fs, 0.90))
    return np.concatenate(parts) if parts else np.zeros(0)


def cue_done(fs: int) -> np.ndarray:
    """本位置采完: 上行两声。"""
    return np.concatenate([tone(fs, 880, 0.09), tone(fs, 1320, 0.14)])


def cue_fail(fs: int) -> np.ndarray:
    """质检不过要重录: 下行两声。"""
    return np.concatenate([tone(fs, 700, 0.15), silence(fs, 0.05), tone(fs, 440, 0.25)])


def cue_index(fs: int, idx: int) -> np.ndarray:
    """把位置序号编成短音串, 录进音频里做防错标记 (每 5 个一组高音)。"""
    parts = [tone(fs, 2000, 0.05, 0.25)]
    n = max(0, idx) % 100
    for _ in range(n // 5):
        parts += [silence(fs, 0.04), tone(fs, 2500, 0.04, 0.22)]
    for _ in range(n % 5):
        parts += [silence(fs, 0.04), tone(fs, 1500, 0.04, 0.22)]
    return np.concatenate(parts)
