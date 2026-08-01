"""语音合成后端。

四个后端, 按优先级自动挑选:

| 后端 | 开源 | 联网 | 中文音质 | 备注 |
|---|---|---|---|---|
| ``piper`` | 是 (MIT) | 只在下载模型时 | 好 | ONNX 纯离线推理, 模型 ~60 MB, **默认首选** |
| ``edge`` | 客户端开源, 服务是微软的 | 预渲染时要联网 | 最好 | 装了 edge-tts 才可用 |
| ``sapi`` | 否 | 否 | 取决于系统语音包 | Windows 自带, 缺中文包就不可用 |
| ``say`` | 否 | 否 | — | macOS 开发调试用 |

全部不可用时 PromptRenderer 会降级为提示音 + 屏幕大字, 采集流程不中断。

提示语是**预渲染并缓存**的, 采集过程中不跑 TTS —— 所以 edge 的联网需求只影响
准备阶段, 现场断网照跑。缓存按 (后端, 语音, 采样率, 文本) 做键, 跨会话复用。
"""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from ..settings import APP_DIR

PIPER_DIR = APP_DIR / "piper"
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# 中文可选模型。medium 音质明显好于 x_low, 体积也就 60 MB 左右, 默认用 medium。
PIPER_ZH_VOICES = [
    ("zh_CN-huayan-medium", "华言 (女声, medium, ~63 MB)"),
    ("zh_CN-huayan-x_low", "华言 (女声, x_low, ~20 MB)"),
]

EDGE_ZH_VOICES = [
    ("zh-CN-XiaoxiaoNeural", "晓晓 (女声)"),
    ("zh-CN-YunxiNeural", "云希 (男声)"),
    ("zh-CN-YunyangNeural", "云扬 (男声, 播报腔)"),
    ("zh-CN-XiaoyiNeural", "晓伊 (女声)"),
]

ProgressCb = Callable[[int, int], None]


@dataclass
class Voice:
    id: str
    label: str


class BackendError(RuntimeError):
    pass


# ====================================================================== Piper
def piper_model_path(name: str, models_dir: Path | None = None) -> Path:
    return Path(models_dir or PIPER_DIR) / f"{name}.onnx"


def piper_installed_voices(models_dir: Path | None = None) -> list[Voice]:
    d = Path(models_dir or PIPER_DIR)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.onnx")):
        label = dict(PIPER_ZH_VOICES).get(p.stem, p.stem)
        out.append(Voice(p.stem, f"{label}  [已下载]"))
    return out


def download_piper_voice(name: str, models_dir: Path | None = None,
                         progress: ProgressCb | None = None) -> Path:
    """从 HuggingFace 下载 piper 模型 (.onnx + .onnx.json)。返回 .onnx 路径。

    命名规则 zh_CN-huayan-medium → zh/zh_CN/huayan/medium/ 。
    """
    d = Path(models_dir or PIPER_DIR)
    d.mkdir(parents=True, exist_ok=True)
    try:
        lang, speaker, quality = name.split("-", 2)
    except ValueError as e:
        raise BackendError(f"无法解析模型名 {name!r}, 期望形如 zh_CN-huayan-medium") from e
    family = lang.split("_")[0]
    base = f"{PIPER_BASE}/{family}/{lang}/{speaker}/{quality}/{name}"

    onnx = d / f"{name}.onnx"
    for url, dest in ((f"{base}.onnx", onnx), (f"{base}.onnx.json", d / f"{name}.onnx.json")):
        if dest.exists() and dest.stat().st_size > 0:
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                while chunk := r.read(1 << 16):
                    f.write(chunk)
                    got += len(chunk)
                    if progress:
                        progress(got, total)
            tmp.replace(dest)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise BackendError(f"下载 {url} 失败: {e}") from e
    return onnx


class PiperBackend:
    """纯离线 ONNX 推理。piper 的 Python API 跨版本变过, 这里逐个形状试。"""

    name = "piper"

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = Path(models_dir or PIPER_DIR)
        self._cache: dict[str, object] = {}
        self.last_error = ""

    def available(self) -> bool:
        try:
            import piper  # noqa: F401
        except Exception as e:
            self.last_error = f"未安装 piper: {e} (uv sync --extra tts)"
            return False
        if not piper_installed_voices(self.models_dir):
            self.last_error = "piper 已安装但还没有下载中文模型"
            return False
        return True

    def voices(self) -> list[Voice]:
        installed = piper_installed_voices(self.models_dir)
        have = {v.id for v in installed}
        return installed + [Voice(n, f"{lb}  [未下载]")
                            for n, lb in PIPER_ZH_VOICES if n not in have]

    def default_voice(self) -> str:
        v = piper_installed_voices(self.models_dir)
        return v[0].id if v else PIPER_ZH_VOICES[0][0]

    def _load(self, voice_id: str):
        if voice_id in self._cache:
            return self._cache[voice_id]
        from piper import PiperVoice

        model = piper_model_path(voice_id, self.models_dir)
        if not model.exists():
            raise BackendError(f"模型不存在: {model}。先在界面上点「下载中文语音模型」。")
        obj = PiperVoice.load(str(model))
        self._cache[voice_id] = obj
        return obj

    def synth(self, text: str, out_wav: Path, voice_id: str) -> None:
        v = self._load(voice_id)
        errors: list[str] = []

        # 形状 1/2: 直接写进 wave 对象 (新版的 synthesize_wav / 老版的 synthesize)。
        # 新版 synthesize 的第二个参数是 syn_config 而不是 wav_file, 传错不会报错只会出怪东西,
        # 所以先按签名确认。
        for meth in ("synthesize_wav", "synthesize"):
            fn = getattr(v, meth, None)
            if fn is None or not _takes_wav_file(fn):
                continue
            try:
                with wave.open(str(out_wav), "wb") as wf:
                    fn(text, wf)
                if out_wav.exists() and out_wav.stat().st_size > 44:
                    return
            except Exception as e:
                errors.append(f"{meth}: {e}")
                out_wav.unlink(missing_ok=True)

        # 形状 3: 返回音频块的迭代器
        for meth in ("synthesize", "synthesize_stream_raw"):
            fn = getattr(v, meth, None)
            if fn is None:
                continue
            try:
                pcm, rate = self._collect_chunks(fn(text), v)
                if pcm:
                    _write_wav_bytes(out_wav, pcm, rate)
                    return
            except Exception as e:
                errors.append(f"{meth}(iter): {e}")

        # 形状 4: 退回命令行
        try:
            self._synth_cli(text, out_wav, voice_id)
            return
        except Exception as e:
            errors.append(f"cli: {e}")

        raise BackendError("piper 合成失败 —— " + "; ".join(errors))

    @staticmethod
    def _collect_chunks(chunks, voice_obj) -> tuple[bytes, int]:
        rate = 22050
        cfg = getattr(voice_obj, "config", None)
        if cfg is not None:
            rate = int(getattr(cfg, "sample_rate", rate))
        buf = io.BytesIO()
        for ch in chunks:
            if isinstance(ch, (bytes, bytearray)):
                buf.write(ch)
                continue
            data = getattr(ch, "audio_int16_bytes", None)
            if data is None:
                arr = getattr(ch, "audio_float_array", None)
                if arr is not None:
                    data = (np.clip(arr, -1, 1) * 32767).astype("<i2").tobytes()
            if data:
                buf.write(data)
            rate = int(getattr(ch, "sample_rate", rate))
        return buf.getvalue(), rate

    def _synth_cli(self, text: str, out_wav: Path, voice_id: str) -> None:
        model = piper_model_path(voice_id, self.models_dir)
        exe = shutil.which("piper")
        cmd = ([exe] if exe else [sys.executable, "-m", "piper"]) + \
              ["-m", str(model), "-f", str(out_wav)]
        r = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True)
        if r.returncode != 0 or not out_wav.exists():
            raise BackendError(r.stderr.decode("utf-8", "ignore")[:200] or "piper CLI 失败")


# ==================================================================== edge-tts
class EdgeBackend:
    name = "edge"

    def __init__(self):
        self.last_error = ""

    def available(self) -> bool:
        try:
            import edge_tts  # noqa: F401
            return True
        except Exception as e:
            self.last_error = f"未安装 edge-tts: {e} (uv sync --extra tts)"
            return False

    def voices(self) -> list[Voice]:
        return [Voice(i, lb) for i, lb in EDGE_ZH_VOICES]

    def default_voice(self) -> str:
        return EDGE_ZH_VOICES[0][0]

    def synth(self, text: str, out_wav: Path, voice_id: str) -> None:
        import asyncio

        import edge_tts

        async def _go(dst: Path) -> None:
            await edge_tts.Communicate(text, voice_id or self.default_voice()).save(str(dst))

        with tempfile.TemporaryDirectory() as td:
            mp3 = Path(td) / "t.mp3"
            try:
                asyncio.run(_go(mp3))
            except Exception as e:
                raise BackendError(f"edge-tts 请求失败 (需要联网): {e}") from e
            try:
                data, fs = sf.read(str(mp3), always_2d=True)
            except Exception as e:
                raise BackendError(
                    f"无法解码 edge-tts 的 mp3 ({e})。libsndfile 需要 1.1+ 才支持 mp3, "
                    "升级 soundfile 或改用 piper 后端。") from e
            sf.write(str(out_wav), data, fs)


# ======================================================================= SAPI
class Pyttsx3Backend:
    name = "sapi"

    def __init__(self, rate_scale: float = 0.95):
        self.rate_scale = rate_scale
        self.last_error = ""

    def available(self) -> bool:
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.stop()
            return True
        except Exception as e:
            self.last_error = f"pyttsx3 不可用: {e}"
            return False

    def voices(self) -> list[Voice]:
        try:
            import pyttsx3
            eng = pyttsx3.init()
            out = [Voice(v.id, getattr(v, "name", v.id)) for v in eng.getProperty("voices")]
            eng.stop()
            return out
        except Exception as e:
            self.last_error = str(e)
            return []

    def default_voice(self) -> str:
        for v in self.voices():
            low = (v.id + v.label).lower()
            if "zh" in low or "chinese" in low or "中文" in v.label:
                return v.id
        return ""

    def synth(self, text: str, out_wav: Path, voice_id: str) -> None:
        import pyttsx3

        eng = pyttsx3.init()
        try:
            if voice_id:
                eng.setProperty("voice", voice_id)
            eng.setProperty("rate", int(eng.getProperty("rate") * self.rate_scale))
            eng.save_to_file(text, str(out_wav))
            eng.runAndWait()
        finally:
            try:
                eng.stop()
            except Exception:
                pass
        if not out_wav.exists() or out_wav.stat().st_size == 0:
            raise BackendError("SAPI 没有产出音频, 系统可能没装中文语音包")


class SayBackend:
    """macOS 的 say, 只用于开发调试。"""

    name = "say"

    def __init__(self):
        self.last_error = ""

    def available(self) -> bool:
        return shutil.which("say") is not None

    def voices(self) -> list[Voice]:
        return [Voice("Tingting", "Tingting (zh-CN)"), Voice("", "系统默认")]

    def default_voice(self) -> str:
        return "Tingting"

    def synth(self, text: str, out_wav: Path, voice_id: str) -> None:
        with tempfile.TemporaryDirectory() as td:
            aiff = Path(td) / "t.aiff"
            cmd = ["say", "-o", str(aiff)]
            if voice_id:
                cmd += ["-v", voice_id]
            cmd.append(text)
            subprocess.run(cmd, check=True, capture_output=True)
            data, fs = sf.read(str(aiff), always_2d=True)
            sf.write(str(out_wav), data, fs)


# ==================================================================== 选择器
BACKEND_ORDER = ["piper", "edge", "sapi", "say"]
BACKEND_LABELS = {
    "auto": "自动 (piper → edge-tts → 系统 SAPI)",
    "piper": "piper (开源, 纯离线)",
    "edge": "edge-tts (音质最好, 预渲染需联网)",
    "sapi": "系统 SAPI (Windows 自带)",
    "say": "macOS say (仅调试)",
    "none": "关闭 (只用提示音)",
}


def make_backend(name: str, models_dir: Path | None = None):
    if name == "piper":
        return PiperBackend(models_dir)
    if name == "edge":
        return EdgeBackend()
    if name == "sapi":
        return Pyttsx3Backend()
    if name == "say":
        return SayBackend()
    raise BackendError(f"未知的 TTS 后端: {name}")


def pick_backend(preferred: str = "auto", models_dir: Path | None = None):
    """返回 (后端对象或 None, 诊断信息)。"""
    notes: list[str] = []
    names = BACKEND_ORDER if preferred in ("auto", "", None) else [preferred]
    for n in names:
        try:
            b = make_backend(n, models_dir)
        except BackendError as e:
            notes.append(str(e))
            continue
        if b.available():
            return b, "; ".join(notes)
        notes.append(f"{n}: {getattr(b, 'last_error', '不可用')}")
    return None, "; ".join(notes)


def _takes_wav_file(fn) -> bool:
    """判断该方法的第二个位置参数是不是 wave 文件对象。签名读不到就当是。"""
    import inspect

    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return True
    return bool(params) and params[0] in ("wav_file", "wav", "wave_file")


def _write_wav_bytes(path: Path, pcm: bytes, rate: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(rate))
        wf.writeframes(pcm)
