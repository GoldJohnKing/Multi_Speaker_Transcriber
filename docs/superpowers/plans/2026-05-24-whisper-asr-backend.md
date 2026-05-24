# Whisper ASR 后端集成实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 ASR 后端框架中集成 faster-whisper 作为第四种后端，提供多语言识别能力。

**Architecture:** 遵循现有自注册模式：新建 `whisper.py` 实现 `ASRBase` 子类，通过 `register_backend("Whisper", ...)` 注册，在 `__init__.py` 中 import 触发注册。使用 faster-whisper 的原生 `hotwords` 参数做热词偏置，`vad_filter=True` 做 VAD 分段，段级时间戳直接映射为 `TranscriptSegment`。

**Tech Stack:** faster-whisper (CTranslate2)，numpy，pytest (mock-based unit tests)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `transcribe/models/asr/whisper.py` | Create | WhisperTranscriber 实现 |
| `tests/test_whisper.py` | Create | Whisper 后端单元测试 |
| `transcribe/models/asr/__init__.py` | Modify | import whisper 模块触发注册 |
| `transcribe/cli.py` | Modify | --backend choices 加 "Whisper" |
| `pyproject.toml` | Modify | 新增 whisper 依赖组 |
| `AGENTS.md` | Modify | ASR 后端差异表新增 Whisper 行 |

---

### Task 1: Add whisper dependency group to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add whisper optional dependency group**

In `pyproject.toml`, after the `qwen-asr` group, add a new `whisper` group and update the `all` group:

```toml
[project.optional-dependencies]
# ─── ASR backends ───
funasr = [
    "funasr @ git+https://github.com/modelscope/FunASR.git@2ca745e5d11ad9650b94691d0d346d1435dc9b63",
    "modelscope",
    "tiktoken",
    "torch>=2.8",
    "torchaudio>=2.8",
    "transformers",
]
qwen-asr = [
    "qwen-asr",
    "torch>=2.8",
    "torchaudio>=2.8",
    "transformers>=4.45",
    "accelerate",
    "librosa",
]
whisper = [
    "faster-whisper>=1.0",
]

# ─── Diarization ───
diarize = [
    "addict",
    "modelscope",
    "pyannote.audio>=4.0.4",
    "scipy",
    "simplejson",
]

# ─── Convenience ───
all = [
    "multi-speaker-transcribe[funasr,diarize,qwen-asr,whisper]",
]
dev = [
    "pytest>=8.0",
]
```

- [ ] **Step 2: Verify toml is valid**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"`
Expected: no output (success, no parse error)

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add whisper optional dependency group"
```

---

### Task 2: Create WhisperTranscriber with tests

**Files:**
- Create: `transcribe/models/asr/whisper.py`
- Create: `tests/test_whisper.py`

- [ ] **Step 1: Write test file with all unit tests**

Create `tests/test_whisper.py`:

```python
"""Tests for the Whisper ASR backend."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transcribe.data.types import AudioSegment, TranscriptSegment
from transcribe.models.asr import list_backends
from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.factory import _BACKENDS


# --- Registration tests ---


def test_whisper_registered() -> None:
    """"Whisper" should appear in the backend registry."""
    assert "Whisper" in list_backends()


def test_whisper_is_asr_base_subclass() -> None:
    """The registered class should be an ASRBase subclass."""
    from transcribe.models.asr.whisper import WhisperTranscriber

    assert issubclass(WhisperTranscriber, ASRBase)


def test_supports_hotwords_true() -> None:
    """Whisper should declare hotword support."""
    cls = _BACKENDS["Whisper"]
    instance = cls.__new__(cls)
    assert instance.supports_hotwords is True


# --- Import error test ---


def test_import_error_without_faster_whisper() -> None:
    """Instantiation without faster-whisper installed should give a helpful error."""
    from transcribe.models.asr.whisper import WhisperTranscriber

    with patch.dict(sys.modules, {"faster_whisper": None}):
        with pytest.raises(ImportError, match="uv sync --extra whisper"):
            WhisperTranscriber(device="cpu")


# --- Hotword loading tests ---


def test_load_hotwords_reads_file(tmp_path) -> None:
    """_load_hotwords should read hotword file and comma-join."""
    cls = _BACKENDS["Whisper"]
    instance = cls.__new__(cls)

    hw_file = tmp_path / "hotwords.txt"
    hw_file.write_text("张三\n李四\n王五\n", encoding="utf-8")

    result = instance._load_hotwords(str(hw_file))
    assert result == "张三,李四,王五"


def test_load_hotwords_none_returns_none() -> None:
    """_load_hotwords with None path returns None."""
    cls = _BACKENDS["Whisper"]
    instance = cls.__new__(cls)
    assert instance._load_hotwords(None) is None


def test_load_hotwords_empty_file_returns_none(tmp_path) -> None:
    """_load_hotwords with empty file returns None."""
    cls = _BACKENDS["Whisper"]
    instance = cls.__new__(cls)

    hw_file = tmp_path / "empty.txt"
    hw_file.write_text("", encoding="utf-8")

    result = instance._load_hotwords(str(hw_file))
    assert result is None


def test_load_hotwords_missing_file_returns_none() -> None:
    """_load_hotwords with missing file returns None (no crash)."""
    cls = _BACKENDS["Whisper"]
    instance = cls.__new__(cls)
    result = instance._load_hotwords("/nonexistent/path/hotwords.txt")
    assert result is None


# --- Transcribe tests with mocked model ---


def _make_instance() -> types.SimpleNamespace:
    """Create a WhisperTranscriber instance via __new__ with mocked internals."""
    cls = _BACKENDS["Whisper"]
    instance = cls.__new__(cls)
    instance._device = "cpu"
    instance._language = None
    instance._hotwords = None
    instance._model = MagicMock()
    return instance


def _make_segment(
    start: float, end: float, text: str
) -> types.SimpleNamespace:
    """Create a mock faster-whisper Segment."""
    return types.SimpleNamespace(
        id=0,
        seek=0,
        start=start,
        end=end,
        text=text,
        tokens=[],
        avg_logprob=0.0,
        compression_ratio=1.0,
        no_speech_prob=0.0,
        words=None,
        temperature=0.0,
    )


def _make_info(language: str = "zh") -> types.SimpleNamespace:
    """Create a mock faster-whisper TranscriptionInfo."""
    return types.SimpleNamespace(
        language=language,
        language_probability=0.99,
        duration=2.0,
        duration_after_vad=2.0,
        all_language_probs=None,
        transcription_options=None,
        vad_options=None,
    )


def test_transcribe_single_segment() -> None:
    """transcribe() should map a single segment to TranscriptSegment."""
    instance = _make_instance()
    segment = _make_segment(0.5, 1.5, "你好世界")
    info = _make_info()

    instance._model.transcribe.return_value = (iter([segment]), info)

    audio = AudioSegment(
        waveform=np.zeros(24000, dtype=np.float32),
        sample_rate=16000,
        start_time=10.0,
        end_time=12.0,
    )
    result = instance.transcribe(audio)

    assert len(result) == 1
    assert result[0].text == "你好世界"
    assert result[0].start_time == pytest.approx(10.5)  # 0.5 + 10.0 offset
    assert result[0].end_time == pytest.approx(11.5)  # 1.5 + 10.0 offset
    assert result[0].speaker_id == "SPEAKER_00"


def test_transcribe_multiple_segments() -> None:
    """transcribe() should map multiple segments preserving order."""
    instance = _make_instance()
    seg1 = _make_segment(0.0, 1.0, "第一段")
    seg2 = _make_segment(1.5, 2.5, "第二段")
    info = _make_info()

    instance._model.transcribe.return_value = (iter([seg1, seg2]), info)

    audio = AudioSegment(
        waveform=np.zeros(40000, dtype=np.float32),
        sample_rate=16000,
        start_time=5.0,
        end_time=7.5,
    )
    result = instance.transcribe(audio)

    assert len(result) == 2
    assert result[0].text == "第一段"
    assert result[0].start_time == pytest.approx(5.0)
    assert result[1].text == "第二段"
    assert result[1].start_time == pytest.approx(6.5)


def test_transcribe_empty_segments() -> None:
    """transcribe() returns empty list when model produces no segments."""
    instance = _make_instance()
    info = _make_info()

    instance._model.transcribe.return_value = (iter([]), info)

    audio = AudioSegment(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
    )
    result = instance.transcribe(audio)
    assert result == []


def test_transcribe_skips_empty_text_segments() -> None:
    """transcribe() should skip segments with empty text."""
    instance = _make_instance()
    seg1 = _make_segment(0.0, 0.5, "")
    seg2 = _make_segment(0.5, 1.5, "有效内容")
    info = _make_info()

    instance._model.transcribe.return_value = (iter([seg1, seg2]), info)

    audio = AudioSegment(
        waveform=np.zeros(24000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.5,
    )
    result = instance.transcribe(audio)

    assert len(result) == 1
    assert result[0].text == "有效内容"


def test_transcribe_strips_whitespace() -> None:
    """transcribe() should strip leading/trailing whitespace from text."""
    instance = _make_instance()
    segment = _make_segment(0.0, 1.0, "  你好  ")
    info = _make_info()

    instance._model.transcribe.return_value = (iter([segment]), info)

    audio = AudioSegment(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
    )
    result = instance.transcribe(audio)

    assert result[0].text == "你好"


def test_transcribe_passes_hotwords() -> None:
    """transcribe() should pass hotwords to model.transcribe()."""
    instance = _make_instance()
    instance._hotwords = "张三,李四"
    segment = _make_segment(0.0, 1.0, "你好")
    info = _make_info()

    instance._model.transcribe.return_value = (iter([segment]), info)

    audio = AudioSegment(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
    )
    instance.transcribe(audio)

    call_kwargs = instance._model.transcribe.call_args
    assert call_kwargs[1]["hotwords"] == "张三,李四"


def test_transcribe_zero_offset() -> None:
    """transcribe() with start_time=0 should use segment timestamps directly."""
    instance = _make_instance()
    segment = _make_segment(0.36, 0.89, "测试")
    info = _make_info()

    instance._model.transcribe.return_value = (iter([segment]), info)

    audio = AudioSegment(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
    )
    result = instance.transcribe(audio)

    assert result[0].start_time == pytest.approx(0.36)
    assert result[0].end_time == pytest.approx(0.89)


def test_transcribe_passes_language() -> None:
    """transcribe() should pass language to model.transcribe()."""
    instance = _make_instance()
    instance._language = "en"
    segment = _make_segment(0.0, 1.0, "hello")
    info = _make_info(language="en")

    instance._model.transcribe.return_value = (iter([segment]), info)

    audio = AudioSegment(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
    )
    instance.transcribe(audio)

    call_kwargs = instance._model.transcribe.call_args
    assert call_kwargs[1]["language"] == "en"


# --- Cleanup tests ---


def test_cleanup_deletes_model() -> None:
    """cleanup() should delete _model attribute."""
    instance = _make_instance()
    assert hasattr(instance, "_model")
    instance.cleanup()
    assert not hasattr(instance, "_model")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_whisper.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'transcribe.models.asr.whisper'`

- [ ] **Step 3: Write WhisperTranscriber implementation**

Create `transcribe/models/asr/whisper.py`:

```python
"""Whisper ASR backend using faster-whisper (CTranslate2)."""

from __future__ import annotations

from pathlib import Path

from transcribe.data.types import AudioSegment, TranscriptSegment
from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.factory import register_backend


class WhisperTranscriber(ASRBase):
    """Multi-language speech recognition using faster-whisper.

    Uses CTranslate2 for efficient inference with native hotword support
    and built-in Silero VAD for automatic segmentation. Supports 99 languages
    with automatic language detection.

    Requires the ``whisper`` optional dependency group.
    """

    def __init__(
        self,
        device: str = "cpu",
        hotword_path: str | None = None,
        *,
        model_size: str = "large-v3",
        language: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "Whisper 后端需要 faster-whisper 包。"
                "请运行: uv sync --extra whisper"
            )

        self._device = device
        self._language = language
        self._hotwords = self._load_hotwords(hotword_path)

        if compute_type is None:
            compute_type = "int8" if device == "cpu" else "float16"

        self._model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type,
        )

    @property
    def supports_hotwords(self) -> bool:
        return True

    def _load_hotwords(self, path: str | None) -> str | None:
        """Read hotword file and comma-join for faster-whisper ``hotwords`` param.

        Args:
            path: Path to hotword file (one word per line), or None.

        Returns:
            Comma-joined hotword string, or None if no hotwords.
        """
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            return None
        words = [
            line.strip()
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return ",".join(words) if words else None

    def transcribe(self, audio: AudioSegment) -> list[TranscriptSegment]:
        """Transcribe audio to text segments with timestamps."""
        segments_iter, info = self._model.transcribe(
            audio.waveform,
            language=self._language,
            hotwords=self._hotwords,
            vad_filter=True,
        )

        segments: list[TranscriptSegment] = []
        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue

            segments.append(
                TranscriptSegment(
                    speaker_id="SPEAKER_00",
                    start_time=seg.start + audio.start_time,
                    end_time=seg.end + audio.start_time,
                    text=text,
                )
            )

        return segments

    def cleanup(self) -> None:
        """Release model from memory."""
        del self._model
        if self._device != "cpu":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


register_backend("Whisper", WhisperTranscriber)
```

- [ ] **Step 4: Register backend in __init__.py**

In `transcribe/models/asr/__init__.py`, add `whisper` to the import line:

```python
# Import backend modules to trigger register_backend() calls
from transcribe.models.asr import funasr_nano, funasr_paraformer, qwen3_asr, whisper  # noqa: F401
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_whisper.py -v`
Expected: All tests PASS

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `uv run pytest -v --timeout=60`
Expected: All existing tests still pass, new whisper tests pass

- [ ] **Step 7: Commit**

```bash
git add transcribe/models/asr/whisper.py transcribe/models/asr/__init__.py tests/test_whisper.py
git commit -m "feat: add Whisper ASR backend using faster-whisper"
```

---

### Task 3: Update CLI choices

**Files:**
- Modify: `transcribe/cli.py`

- [ ] **Step 1: Add "Whisper" to --backend choices**

In `transcribe/cli.py`, update the `--backend` argument:

```python
parser.add_argument(
    "--backend",
    choices=["Fun-ASR-Paraformer", "Fun-ASR-Nano", "Qwen3-ASR", "Whisper"],
    default="Fun-ASR-Nano",
    help="ASR backend (default: Fun-ASR-Nano)",
)
```

- [ ] **Step 2: Verify CLI help text**

Run: `uv run python -m transcribe --help`
Expected: Output includes `--backend` with choices showing `Whisper`

- [ ] **Step 3: Commit**

```bash
git add transcribe/cli.py
git commit -m "feat: add Whisper to --backend CLI choices"
```

---

### Task 4: Update AGENTS.md backend documentation

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Add Whisper row to ASR backend difference table**

In the ASR 后端差异 table in `AGENTS.md`, add a Whisper column:

```markdown
| 后端 | 模型 | 热词机制 | 时间戳来源 | 字幕分段 |
|------|------|---------|-----------|---------|
| Fun-ASR-Nano | FunAudioLLM/Fun-ASR-Nano-2512 | `hotwords=list[str]` 解码器偏置 | FSMN-VAD 分段 | VAD 分段 |
| Fun-ASR-Paraformer | speech_seaco_paraformer_large | `hotword=str` 解码器偏置 | FSMN-VAD 分段 | VAD 分段 |
| Qwen3-ASR | Qwen3-ASR-1.7B + ForcedAligner-0.6B | `context=str` LLM prompt 偏置 | ForcedAligner 字符级对齐 | 标点 + 时长混合分段 |
| Whisper | large-v3 (faster-whisper/CTranslate2) | `hotwords=str` 原生热词偏置 | Silero VAD 分段 | VAD 分段 |
```

- [ ] **Step 2: Update 项目结构约定 section**

In the 项目结构约定 code block listing `models/asr/` files, add `whisper.py`:

```markdown
    └── asr/           # ASR 后端包
        ├── __init__.py         # 注册 + 重导出
        ├── base.py             # ASRBase 抽象基类
        ├── factory.py          # create_asr 工厂函数
        ├── utils.py            # 共享工具（热词修复、时间戳解析、字幕分段）
        ├── funasr_nano.py      # Fun-ASR-Nano 后端
        ├── funasr_paraformer.py # Fun-ASR-Paraformer 后端
        ├── qwen3_asr.py       # Qwen3-ASR 后端
        └── whisper.py          # Whisper 后端 (faster-whisper)
```

- [ ] **Step 3: Update 依赖结构 table**

Add whisper row to the 依赖结构 table:

```markdown
| `whisper` | `uv sync --extra whisper` | faster-whisper (CTranslate2, 无 PyTorch) |
```

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md
git commit -m "docs: update AGENTS.md with Whisper backend info"
```

---

### Task 5: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass

- [ ] **Step 2: Verify backend registration**

Run: `uv run python -c "from transcribe.models.asr import list_backends; print(list_backends())"`
Expected: `['Fun-ASR-Nano', 'Fun-ASR-Paraformer', 'Qwen3-ASR', 'Whisper']`

- [ ] **Step 3: Verify CLI help**

Run: `uv run python -m transcribe --help`
Expected: `--backend` shows all 4 choices including `Whisper`
