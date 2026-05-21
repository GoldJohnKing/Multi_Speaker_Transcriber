# Qwen3-ASR Backend Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Qwen3-ASR-1.7B (with ForcedAligner-0.6B) as a third ASR backend, following the existing abstract base class + factory + self-registration pattern.

**Architecture:** New `Qwen3ASRTranscriber` class in `transcribe/models/asr/qwen3_asr.py` implements `ASRBase`. Audio passed as `(np.ndarray, sr)` tuple. Hotwords mapped to `context` string. Character-level timestamps from ForcedAligner segmented into subtitle-grade chunks via hybrid punctuation/duration algorithm in `utils.py`.

**Tech Stack:** Python 3.12, `qwen-asr` PyPI package, PyTorch, existing `ASRBase`/`register_backend` pattern.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `transcribe/models/asr/utils.py` | Modify | Add `segment_by_timestamps()` function |
| `transcribe/models/asr/qwen3_asr.py` | Create | `Qwen3ASRTranscriber(ASRBase)` class |
| `transcribe/models/asr/__init__.py` | Modify | +1 import line for registration trigger |
| `transcribe/cli.py` | Modify | Add `"Qwen3-ASR"` to `--backend` choices |
| `pyproject.toml` | Modify | Add `qwen-asr` dependency group, update `all` |
| `tests/test_qwen3_asr.py` | Create | Unit + integration tests |

---

### Task 1: Add `segment_by_timestamps()` to `utils.py`

**Files:**
- Modify: `transcribe/models/asr/utils.py` (append after line 97)
- Test: `tests/test_asr.py` (append new test class)

This is a pure function with no external dependencies, so it can be fully tested without models.

- [ ] **Step 1: Write failing tests for `segment_by_timestamps`**

Append to `tests/test_asr.py`:

```python
from transcribe.models.asr.utils import segment_by_timestamps


# --- segment_by_timestamps tests (pure function, no model needed) ---


def test_segment_empty_input() -> None:
    """Empty input returns empty list."""
    assert segment_by_timestamps([]) == []


def test_segment_single_char() -> None:
    """Single character produces one segment."""
    result = segment_by_timestamps([("你", 0.0, 0.2)])
    assert len(result) == 1
    assert result[0].text == "你"
    assert result[0].start_time == pytest.approx(0.0)
    assert result[0].end_time == pytest.approx(0.2)


def test_segment_splits_at_sentence_end() -> None:
    """Segments split at sentence-ending punctuation (。！？)."""
    char_ts = [
        ("你", 0.0, 0.2), ("好", 0.2, 0.4), ("。", 0.4, 0.5),
        ("我", 1.0, 1.2), ("是", 1.2, 1.4), ("。", 1.4, 1.5),
    ]
    result = segment_by_timestamps(char_ts)
    assert len(result) == 2
    assert result[0].text == "你好。"
    assert result[0].start_time == pytest.approx(0.0)
    assert result[0].end_time == pytest.approx(0.5)
    assert result[1].text == "我是。"
    assert result[1].start_time == pytest.approx(1.0)
    assert result[1].end_time == pytest.approx(1.5)


def test_segment_max_duration_splits_at_clause() -> None:
    """When duration exceeds max_duration, split at nearest clause punctuation."""
    # 8 seconds with no sentence-end, only commas
    char_ts = [
        ("今", 0.0, 0.5), ("天", 0.5, 1.0),
        ("，", 1.0, 1.1),
        ("天", 1.1, 1.6), ("气", 1.6, 2.1),
        ("，", 2.1, 2.2),
        ("很", 2.2, 2.7), ("好", 2.7, 3.2),
        ("，", 3.2, 3.3),
        ("我", 3.3, 3.8), ("们", 3.8, 4.3),
        ("，", 4.3, 4.4),
        ("出", 4.4, 4.9), ("去", 4.9, 5.4),
        ("，", 5.4, 5.5),
        ("玩", 5.5, 6.0), ("吧", 6.0, 6.5),
        ("。", 6.5, 6.6),
    ]
    # With max_duration=4.0, should split at first comma after 4s
    result = segment_by_timestamps(char_ts, max_duration=4.0)
    assert len(result) >= 2
    # First segment should end at or before 4s
    assert result[0].end_time <= 4.5  # allows for clause punctuation inclusion
    # Each segment should not exceed max_duration by much
    for seg in result:
        duration = seg.end_time - seg.start_time
        assert duration <= 5.0  # some tolerance for punctuation


def test_segment_max_chars_hard_cut() -> None:
    """When no punctuation and max_chars reached, hard-cut."""
    # 30 characters with no punctuation
    char_ts = [(c, float(i) * 0.1, float(i) * 0.1 + 0.1) for i, c in enumerate("一" * 30)]
    result = segment_by_timestamps(char_ts, max_duration=100.0, max_chars=10)
    assert len(result) >= 3
    for seg in result:
        assert len(seg.text) <= 10


def test_segment_no_punctuation_long_duration() -> None:
    """Long unpunctuated speech triggers hard duration cut."""
    # 10 seconds of speech with no punctuation, max_duration=5
    char_ts = [(f"w{i}", float(i), float(i) + 1.0) for i in range(10)]
    result = segment_by_timestamps(char_ts, max_duration=5.0, max_chars=100)
    assert len(result) >= 2
    # No segment should wildly exceed max_duration
    for seg in result:
        duration = seg.end_time - seg.start_time
        assert duration <= 7.0  # tolerance: cut at next available char


def test_segment_speaker_id_always_speaker_00() -> None:
    """All returned segments have speaker_id SPEAKER_00."""
    char_ts = [
        ("你", 0.0, 0.2), ("好", 0.2, 0.4), ("。", 0.4, 0.5),
        ("我", 1.0, 1.2), ("是", 1.2, 1.4), ("。", 1.4, 1.5),
    ]
    result = segment_by_timestamps(char_ts)
    for seg in result:
        assert seg.speaker_id == "SPEAKER_00"


def test_segment_preserves_time_offset() -> None:
    """Timestamps should preserve any offset in input (e.g. audio.start_time)."""
    offset = 10.5
    char_ts = [
        ("你", offset + 0.0, offset + 0.2),
        ("好", offset + 0.2, offset + 0.4),
        ("。", offset + 0.4, offset + 0.5),
    ]
    result = segment_by_timestamps(char_ts)
    assert result[0].start_time == pytest.approx(offset + 0.0)
    assert result[0].end_time == pytest.approx(offset + 0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_asr.py -v -k "segment_" --no-header -q 2>&1 | head -20
```

Expected: `ImportError` or `FAILED` — `segment_by_timestamps` not yet defined.

- [ ] **Step 3: Implement `segment_by_timestamps()`**

Append to `transcribe/models/asr/utils.py` (after line 97):

```python
# --- Subtitle segmentation from character-level timestamps ---


_SENTENCE_END = frozenset("。！？!?")
_CLAUSE_END = frozenset("，；：,;:")


def segment_by_timestamps(
    char_ts: list[tuple[str, float, float]],
    max_duration: float = 7.0,
    max_chars: int = 25,
) -> list[TranscriptSegment]:
    """Split character-level timestamps into subtitle-grade segments.

    Hybrid strategy:
    1. Split at sentence-ending punctuation (。！？).
    2. When accumulated duration exceeds *max_duration*, split at the
       nearest preceding clause punctuation (，；：).
    3. When no clause punctuation is found, hard-cut at *max_duration*.
    4. When accumulated characters exceed *max_chars*, hard-cut (fallback).

    Args:
        char_ts: ``[(text, start_sec, end_sec), ...]`` character-level timestamps.
        max_duration: Maximum duration per subtitle segment (seconds).
        max_chars: Maximum characters per subtitle segment.

    Returns:
        List of :class:`TranscriptSegment`.
    """
    if not char_ts:
        return []

    from transcribe.data.types import TranscriptSegment

    segments: list[TranscriptSegment] = []
    buf_text: list[str] = []
    buf_start: float = char_ts[0][1]
    last_clause_idx: int | None = None  # index into buf_text

    for i, (text, start, end) in enumerate(char_ts):
        buf_text.append(text)

        # Flush at sentence-ending punctuation
        if text in _SENTENCE_END:
            segments.append(TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=buf_start,
                end_time=end,
                text="".join(buf_text),
            ))
            buf_text = []
            buf_start = char_ts[i + 1][1] if i + 1 < len(char_ts) else end
            last_clause_idx = None
            continue

        # Track clause punctuation positions for fallback splitting
        if text in _CLAUSE_END:
            last_clause_idx = len(buf_text) - 1

        duration = end - buf_start
        char_count = len(buf_text)

        # Max duration exceeded — try splitting at clause punctuation
        if duration > max_duration and last_clause_idx is not None and last_clause_idx > 0:
            before = buf_text[: last_clause_idx + 1]
            after = buf_text[last_clause_idx + 1 :]
            before_end = char_ts[i - len(after)][2]  # end time of last char in 'before'

            segments.append(TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=buf_start,
                end_time=before_end,
                text="".join(before),
            ))
            buf_text = after
            buf_start = start
            last_clause_idx = None
            continue

        # Max duration exceeded with no clause punctuation — hard cut
        if duration > max_duration and len(buf_text) > 1:
            segments.append(TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=buf_start,
                end_time=end,
                text="".join(buf_text),
            ))
            buf_text = []
            buf_start = char_ts[i + 1][1] if i + 1 < len(char_ts) else end
            last_clause_idx = None
            continue

        # Max chars exceeded — hard cut
        if char_count >= max_chars:
            segments.append(TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=buf_start,
                end_time=end,
                text="".join(buf_text),
            ))
            buf_text = []
            buf_start = char_ts[i + 1][1] if i + 1 < len(char_ts) else end
            last_clause_idx = None
            continue

    # Flush remaining buffer
    if buf_text:
        segments.append(TranscriptSegment(
            speaker_id="SPEAKER_00",
            start_time=buf_start,
            end_time=char_ts[-1][2],
            text="".join(buf_text),
        ))

    return segments
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_asr.py -v -k "segment_" --no-header -q
```

Expected: All `test_segment_*` tests PASS.

- [ ] **Step 5: Run full test suite to verify no regressions**

```bash
uv run pytest tests/test_asr.py -v --no-header -q -m "not slow"
```

Expected: All non-slow tests PASS (including existing hotword/timestamp tests).

- [ ] **Step 6: Commit**

```bash
git add transcribe/models/asr/utils.py tests/test_asr.py
git commit -m "feat: add segment_by_timestamps() for character-level subtitle splitting"
```

---

### Task 2: Create `Qwen3ASRTranscriber` backend

**Files:**
- Create: `transcribe/models/asr/qwen3_asr.py`
- Test: `tests/test_qwen3_asr.py`

- [ ] **Step 1: Write failing unit tests (no model required)**

Create `tests/test_qwen3_asr.py`:

```python
"""Tests for the Qwen3-ASR backend."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from transcribe.models.asr import list_backends
from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.factory import _BACKENDS


def test_qwen3_asr_registered() -> None:
    """"Qwen3-ASR" should appear in the backend registry."""
    assert "Qwen3-ASR" in list_backends()


def test_qwen3_asr_is_asr_base_subclass() -> None:
    """The registered class should be an ASRBase subclass."""
    from transcribe.models.asr.qwen3_asr import Qwen3ASRTranscriber
    assert issubclass(Qwen3ASRTranscriber, ASRBase)


def test_supports_hotwords_true() -> None:
    """Qwen3-ASR should declare hotword support (via context)."""
    from transcribe.models.asr.qwen3_asr import Qwen3ASRTranscriber
    # We can't instantiate without qwen_asr installed, but we can check
    # the property exists on the class level via __init__ signature
    cls = _BACKENDS["Qwen3-ASR"]
    # Create a mock instance to check the property
    instance = cls.__new__(cls)
    assert instance.supports_hotwords is True


def test_import_error_without_qwen_asr(tmp_path) -> None:
    """Instantiation without qwen-asr installed should give a helpful error."""
    from transcribe.models.asr.qwen3_asr import Qwen3ASRTranscriber

    # Ensure qwen_asr is not importable
    with patch.dict(sys.modules, {"qwen_asr": None}):
        with pytest.raises(ImportError, match="uv sync --extra qwen-asr"):
            Qwen3ASRTranscriber(device="cpu")


def test_load_context_reads_hotword_file(tmp_path) -> None:
    """_load_context should read hotword file and join with spaces."""
    from transcribe.models.asr.qwen3_asr import Qwen3ASRTranscriber

    hw_file = tmp_path / "hotwords.txt"
    hw_file.write_text("张三\n李四\n王五\n", encoding="utf-8")

    cls = _BACKENDS["Qwen3-ASR"]
    instance = cls.__new__(cls)
    result = instance._load_context(str(hw_file))
    assert result == "张三 李四 王五"


def test_load_context_none_returns_empty() -> None:
    """_load_context with None path returns empty string."""
    from transcribe.models.asr.qwen3_asr import Qwen3ASRTranscriber

    cls = _BACKENDS["Qwen3-ASR"]
    instance = cls.__new__(cls)
    assert instance._load_context(None) == ""


def test_load_context_empty_file_returns_empty() -> None:
    """_load_context with empty file returns empty string."""
    from transcribe.models.asr.qwen3_asr import Qwen3ASRTranscriber

    hw_file = types.SimpleNamespace()
    cls = _BACKENDS["Qwen3-ASR"]
    instance = cls.__new__(cls)
    # Use a non-existent path — should be handled gracefully
    # Actually _load_context just reads lines, empty file = empty string
    import tempfile, pathlib
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("")
        path = f.name
    result = instance._load_context(path)
    assert result == ""
    pathlib.Path(path).unlink()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_qwen3_asr.py -v --no-header -q 2>&1 | head -20
```

Expected: `ModuleNotFoundError` — `transcribe.models.asr.qwen3_asr` not yet created.

- [ ] **Step 3: Create `transcribe/models/asr/qwen3_asr.py`**

```python
"""Qwen3-ASR backend with ForcedAligner for character-level timestamps."""

from __future__ import annotations

import torch

from transcribe.data.types import AudioSegment, TranscriptSegment
from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.factory import register_backend
from transcribe.models.asr.utils import segment_by_timestamps


class Qwen3ASRTranscriber(ASRBase):
    """Speech recognition using Qwen3-ASR-1.7B with Qwen3-ForcedAligner-0.6B.

    Encapsulates both the ASR model and the forced aligner in a single
    ``ASRBase`` subclass.  Audio is passed as ``(np.ndarray, sample_rate)``
    tuples.  Hotwords are mapped to the ``context`` parameter (LLM system
    prompt biasing) rather than traditional weighted decoder biasing.

    Requires the ``qwen-asr`` optional dependency group.
    """

    def __init__(
        self,
        device: str = "cpu",
        hotword_path: str | None = None,
        *,
        asr_model: str = "Qwen/Qwen3-ASR-1.7B",
        aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        language: str | None = None,
    ) -> None:
        # Deferred import — module must be importable without qwen-asr
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            raise ImportError(
                "Qwen3-ASR 后端需要 qwen-asr 包。"
                "请运行: uv sync --extra qwen-asr"
            )

        self._device = device
        self._language = language
        self._context = self._load_context(hotword_path)

        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        device_map = "cuda:0" if device == "cuda" else "cpu"

        self._model = Qwen3ASRModel.from_pretrained(
            asr_model,
            dtype=dtype,
            device_map=device_map,
            forced_aligner=aligner_model,
            forced_aligner_kwargs=dict(dtype=dtype, device_map=device_map),
            max_inference_batch_size=32,
            max_new_tokens=256,
        )

    @property
    def supports_hotwords(self) -> bool:
        return True

    def transcribe(self, audio: AudioSegment) -> list[TranscriptSegment]:
        """Transcribe audio to text segments with character-level timestamps."""
        results = self._model.transcribe(
            audio=(audio.waveform, audio.sample_rate),
            context=self._context,
            language=self._language,
            return_time_stamps=True,
        )

        if not results or not results[0].text:
            return []

        r = results[0]

        # Fallback: no timestamps → single segment covering entire audio
        if not r.time_stamps:
            return [TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=audio.start_time,
                end_time=audio.end_time,
                text=r.text,
            )]

        # Build character-level timestamp list with offset
        char_ts = [
            (ts.text, audio.start_time + ts.start_time, audio.start_time + ts.end_time)
            for ts in r.time_stamps
        ]

        return segment_by_timestamps(char_ts)

    def cleanup(self) -> None:
        """Release ASR + ForcedAligner from GPU/CPU memory."""
        del self._model
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_context(self, path: str | None) -> str:
        """Read hotword file, join with spaces for Qwen3-ASR ``context`` param."""
        if not path:
            return ""
        with open(path, encoding="utf-8") as f:
            terms = [line.strip() for line in f if line.strip()]
        return " ".join(terms)


register_backend("Qwen3-ASR", Qwen3ASRTranscriber)
```

- [ ] **Step 4: Register backend in `__init__.py`**

In `transcribe/models/asr/__init__.py`, change line 12 from:

```python
from transcribe.models.asr import funasr_nano, funasr_paraformer  # noqa: F401
```

to:

```python
from transcribe.models.asr import funasr_nano, funasr_paraformer, qwen3_asr  # noqa: F401
```

- [ ] **Step 5: Run unit tests**

```bash
uv run pytest tests/test_qwen3_asr.py -v --no-header -q
```

Expected: All unit tests PASS (registration, subclass check, hotword loading, import error).

- [ ] **Step 6: Run full non-slow test suite**

```bash
uv run pytest -m "not slow" --no-header -q
```

Expected: All non-slow tests PASS across all test files.

- [ ] **Step 7: Commit**

```bash
git add transcribe/models/asr/qwen3_asr.py transcribe/models/asr/__init__.py tests/test_qwen3_asr.py
git commit -m "feat: add Qwen3-ASR backend with ForcedAligner support"
```

---

### Task 3: Update CLI and dependencies

**Files:**
- Modify: `transcribe/cli.py` (line 43)
- Modify: `pyproject.toml` (lines 13–31)

- [ ] **Step 1: Update CLI `--backend` choices**

In `transcribe/cli.py`, change line 43 from:

```python
choices=["Fun-ASR-Paraformer", "Fun-ASR-Nano"],
```

to:

```python
choices=["Fun-ASR-Paraformer", "Fun-ASR-Nano", "Qwen3-ASR"],
```

- [ ] **Step 2: Add `qwen-asr` dependency group in `pyproject.toml`**

After the existing `diarize` group (line 28), add a new group and update `all`:

```toml
qwen-asr = [
    "qwen-asr",
    "torch>=2.1",
    "torchaudio>=2.1",
    "transformers>=4.45",
    "accelerate",
    "librosa",
    "soundfile",
]
all = [
    "multi-speaker-transcribe[funasr,diarize,qwen-asr]",
]
```

The full `[project.optional-dependencies]` section should look like:

```toml
[project.optional-dependencies]
funasr = [
    "funasr @ git+https://github.com/modelscope/FunASR.git@2ca745e5d11ad9650b94691d0d346d1435dc9b63",
    "modelscope",
    "tiktoken",
    "torch>=2.1",
    "torchaudio>=2.1",
    "transformers",
]
diarize = [
    "addict",
    "modelscope",
    "pyannote.audio>=4.0",
    "scipy",
    "simplejson",
]
qwen-asr = [
    "qwen-asr",
    "torch>=2.1",
    "torchaudio>=2.1",
    "transformers>=4.45",
    "accelerate",
    "librosa",
    "soundfile",
]
all = [
    "multi-speaker-transcribe[funasr,diarize,qwen-asr]",
]
dev = [
    "pytest>=8.0",
]
```

- [ ] **Step 3: Verify CLI accepts the new backend**

```bash
uv run python -c "from transcribe.cli import build_parser; p = build_parser(); args = p.parse_args(['test.wav', '--backend', 'Qwen3-ASR']); print(args.backend)"
```

Expected: `Qwen3-ASR`

- [ ] **Step 4: Verify dependency groups are valid**

```bash
uv sync --extra dev 2>&1 | tail -5
```

Expected: Success (no errors about invalid dependency groups).

- [ ] **Step 5: Commit**

```bash
git add transcribe/cli.py pyproject.toml
git commit -m "feat: add Qwen3-ASR to CLI choices and pyproject dependencies"
```

---

### Task 4: Final verification

- [ ] **Step 1: Run full non-slow test suite**

```bash
uv run pytest -m "not slow" -v --no-header -q
```

Expected: All tests PASS.

- [ ] **Step 2: Verify all three backends are registered**

```bash
uv run python -c "from transcribe.models.asr import list_backends; print(list_backends())"
```

Expected: `['Fun-ASR-Paraformer', 'Fun-ASR-Nano', 'Qwen3-ASR']`

- [ ] **Step 3: Verify Qwen3-ASR graceful import error**

```bash
uv run python -c "
from transcribe.models.asr import create_asr
try:
    create_asr('Qwen3-ASR', device='cpu')
except ImportError as e:
    print(f'OK: {e}')
"
```

Expected: `OK: Qwen3-ASR 后端需要 qwen-asr 包。请运行: uv sync --extra qwen-asr`

- [ ] **Step 4: Verify `--help` output**

```bash
uv run python -m transcribe --help 2>&1 | grep -A2 backend
```

Expected: Shows `--backend` with three choices including `Qwen3-ASR`.
