"""语音导播的后端选择、缓存与降级。

用假后端跑, 不联网、不需要模型 —— 真实后端 (piper / edge / SAPI) 的可用性
由 ``marray-capture --check`` 在目标机器上现场确认。
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from marray_capture.audio import prompts as P
from marray_capture.audio import tts


class FakeBackend:
    """合成一段固定时长的正弦, 记录被调用了哪些文本。"""

    name = "fake"

    def __init__(self, dur_s: float = 0.5, rate: int = 22050):
        self.dur_s, self.rate = dur_s, rate
        self.calls: list[str] = []
        self.last_error = ""

    def available(self) -> bool:
        return True

    def voices(self):
        return [tts.Voice("v1", "假语音 1"), tts.Voice("v2", "假语音 2")]

    def default_voice(self) -> str:
        return "v1"

    def synth(self, text: str, out_wav, voice_id: str) -> None:
        self.calls.append(text)
        n = int(self.dur_s * self.rate)
        t = np.arange(n) / self.rate
        sf.write(str(out_wav), (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), self.rate)


@pytest.fixture
def fake(monkeypatch):
    b = FakeBackend()
    monkeypatch.setattr(tts, "pick_backend", lambda preferred="auto", models_dir=None: (b, ""))
    return b


def test_renders_and_resamples(tmp_path, fake):
    r = P.PromptRenderer(48000, tmp_path)
    w = r.render("面向音箱，保持不动。")
    assert r.ok and r.backend == "fake" and r.voice == "v1"
    assert abs(len(w) / 48000 - 0.5) < 0.01          # 22.05k → 48k 重采样
    assert abs(float(np.max(np.abs(w))) - 0.7) < 0.02  # 归一到与提示音等响


def test_disk_cache_avoids_resynthesis(tmp_path, fake):
    text = "向右转一格，大约三十度。"
    P.PromptRenderer(48000, tmp_path).render(text)
    assert fake.calls == [text]
    # 新的渲染器实例 (内存缓存是空的) 应该命中磁盘缓存
    w = P.PromptRenderer(48000, tmp_path).render(text)
    assert fake.calls == [text]
    assert len(w) > 0


def test_cache_key_separates_voices(tmp_path, fake):
    text = "面向音箱。"
    P.PromptRenderer(48000, tmp_path, voice="v1").render(text)
    P.PromptRenderer(48000, tmp_path, voice="v2").render(text)
    assert len(fake.calls) == 2
    assert len(list(tmp_path.glob("p_*.wav"))) == 2


def test_export_writes_audio_and_index(tmp_path, fake):
    texts = ["第一句。", "第二句。"]
    r = P.PromptRenderer(48000, tmp_path / "cache")
    r.prewarm(texts)
    index = r.export(texts, tmp_path / "out")

    import json
    meta = json.loads(index.read_text(encoding="utf-8"))
    assert meta["backend"] == "fake" and meta["fs"] == 48000
    assert [i["text"] for i in meta["items"]] == texts
    for item in meta["items"]:
        assert (tmp_path / "out" / item["file"]).exists()


def test_synthesis_failure_degrades_quietly(tmp_path, monkeypatch):
    class Broken(FakeBackend):
        def synth(self, text, out_wav, voice_id):
            raise RuntimeError("模型炸了")

    monkeypatch.setattr(tts, "pick_backend",
                        lambda preferred="auto", models_dir=None: (Broken(), ""))
    r = P.PromptRenderer(48000, tmp_path)
    assert len(r.render("随便什么")) == 0        # 返回空数组, 由提示音兜底
    assert "模型炸了" in r.last_error
    assert not list(tmp_path.glob("p_*.wav"))    # 半截文件必须被清掉


def test_no_backend_and_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "pick_backend",
                        lambda preferred="auto", models_dir=None: (None, "都没装"))
    r = P.PromptRenderer(48000, tmp_path)
    assert not r.ok and r.backend == "none"
    assert len(r.render("你好")) == 0
    assert "都没装" in r.describe()

    off = P.PromptRenderer(48000, tmp_path, enabled=False)
    assert len(off.render("你好")) == 0
    assert "关闭" in off.describe()


def test_piper_voice_name_parsing():
    with pytest.raises(tts.BackendError):
        tts.download_piper_voice("不是合法名字")
    assert tts.piper_model_path("zh_CN-huayan-medium").name == "zh_CN-huayan-medium.onnx"


def test_backend_labels_cover_all_choices():
    for key in ["auto", "none", *tts.BACKEND_ORDER]:
        assert key in tts.BACKEND_LABELS
