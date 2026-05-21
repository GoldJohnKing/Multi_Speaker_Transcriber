# Dual ASR Backend Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support both Fun-ASR-Paraformer and Fun-ASR-Nano as selectable ASR backends via `--backend` CLI argument and `config.yaml`.

**Architecture:** Abstract base class `ASRBase` defines the uniform interface. Each backend (Paraformer, Nano) implements it in its own file and self-registers via `register_backend()`. A factory function `create_asr()` instantiates the chosen backend. The pipeline uses the factory instead of importing a concrete class.

**Tech Stack:** Python 3.12, FunASR (git version), PyTorch, argparse, PyYAML, pytest

**Design Spec:** `docs/superpowers/specs/2026-05-21-dual-asr-backend-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| **Delete** | `transcribe/models/asr.py` | Single-file ASR (replaced by package) |
| **Create** | `transcribe/models/asr/__init__.py` | Package init, triggers registration, re-exports |
| **Create** | `transcribe/models/asr/base.py` | `ASRBase` abstract base class |
| **Create** | `transcribe/models/asr/utils.py` | `restore_hotwords`, `parse_timestamps` |
| **Create** | `transcribe/models/asr/factory.py` | `register_backend`, `create_asr`, `list_backends` |
| **Create** | `transcribe/models/asr/paraformer.py` | `FunASRParaformerTranscriber` |
| **Create** | `transcribe/models/asr/nano.py` | `FunASRNanoTranscriber` |
| **Modify** | `transcribe/data/types.py:58-67` | Add `backend` field to `PipelineConfig` |
| **Modify** | `transcribe/cli.py` | Add `--backend` argument |
| **Modify** | `transcribe/config.py:13` | Add `"backend"` to `_PIPELINE_CONFIG_FIELDS` |
| **Modify** | `transcribe/pipeline.py:18,135` | Use `create_asr` factory |
| **Modify** | `config.yaml` | Add `backend: Fun-ASR-Nano`, update ASR comment |
| **Modify** | `tests/test_asr.py` | Restructure for dual-backend, update imports |

---

### Task 1: Remove old asr.py and create package scaffolding

**Files:**
- Delete: `transcribe/models/asr.py`
- Create: `transcribe/models/asr/__init__.py` (minimal, will be expanded in Task 6)

**IMPORTANT:** This must be done first because Python cannot have both `asr.py` (file) and `asr/` (package directory) at the same path. All subsequent tasks create files inside the new `asr/` package.

- [ ] **Step 1: Delete the old single-file module**

```bash
git rm transcribe/models/asr.py
```

- [ ] **Step 2: Create a minimal `transcribe/models/asr/__init__.py` (placeholder)**

A minimal init is needed so that submodules can be imported. It will be expanded in Task 6.

```python
"""ASR backend package — placeholder, expanded after all submodules are created."""
```

- [ ] **Step 3: Commit**

```bash
git add transcribe/models/asr/__init__.py
git commit -m "feat(asr): finalize __init__.py with registration and re-exports"
```

---

### Task 8: Update PipelineConfig to add backend field

**Files:**
- Modify: `transcribe/data/types.py:58-67`

- [ ] **Step 1: Add `backend` field to `PipelineConfig`**

In `transcribe/data/types.py`, add the `backend` field to the `PipelineConfig` dataclass:

```python
@dataclass
class PipelineConfig:
    """Pipeline configuration."""

    device: str = "auto"  # "cpu" | "cuda" | "auto"
    diarize: bool = True  # enable speaker diarization (Pyannote)
    backend: str = "Fun-ASR-Nano"  # ASR backend name
    hotwords: str | None = None  # hotword file path
    language: str = "zh"
    cache_dir: str = ".cache"
    num_speakers: int | None = None  # known speaker count, auto-detect if None
    speaker_references: str | None = None  # directory of speaker reference audio samples
```

- [ ] **Step 2: Commit**

```bash
git add transcribe/data/types.py
git commit -m "feat(config): add backend field to PipelineConfig"
```

---

### Task 9: Update CLI to add --backend argument

**Files:**
- Modify: `transcribe/cli.py`

- [ ] **Step 1: Add `--backend` argument to `build_parser()`**

In `transcribe/cli.py`, add the following argument after the `--device` argument (around line 41):

```python
    parser.add_argument(
        "--backend",
        choices=["Fun-ASR-Paraformer", "Fun-ASR-Nano"],
        default="Fun-ASR-Nano",
        help="ASR backend (default: Fun-ASR-Nano)",
    )
```

- [ ] **Step 2: Commit**

```bash
git add transcribe/cli.py
git commit -m "feat(cli): add --backend argument for ASR backend selection"
```

---

### Task 10: Update config loader to handle backend field

**Files:**
- Modify: `transcribe/config.py:13`

- [ ] **Step 1: Add `"backend"` to `_PIPELINE_CONFIG_FIELDS`**

In `transcribe/config.py`, update line 14 to include `"backend"`:

```python
_PIPELINE_CONFIG_FIELDS = frozenset(
    {"device", "diarize", "backend", "hotwords", "language", "cache_dir", "num_speakers", "speaker_references"}
)
```

- [ ] **Step 2: Commit**

```bash
git add transcribe/config.py
git commit -m "feat(config): add backend to pipeline config fields"
```

---

### Task 11: Update pipeline.py to use factory

**Files:**
- Modify: `transcribe/pipeline.py:18,135`

- [ ] **Step 1: Update import on line 18**

Change:
```python
from transcribe.models.asr import ASRTranscriber
```
To:
```python
from transcribe.models.asr import create_asr
```

- [ ] **Step 2: Update transcriber instantiation on line 135**

Change:
```python
    transcriber = ASRTranscriber(device=device, hotword_path=config.hotwords)
```
To:
```python
    transcriber = create_asr(config.backend, device=device, hotword_path=config.hotwords)
```

- [ ] **Step 3: Commit**

```bash
git add transcribe/pipeline.py
git commit -m "feat(pipeline): use create_asr factory for backend selection"
```

---

### Task 12: Update config.yaml

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Add backend field and update comments**

Update `config.yaml` to:

```yaml
device: auto
diarize: true
backend: Fun-ASR-Nano  # 可选 Fun-ASR-Paraformer / Fun-ASR-Nano
language: zh
speaker_references: null

diarizer:
  model: pyannote/speaker-diarization-community-1
  hf_token: null
  clustering: hidden_markov

matcher:
  embedding_model: iic/speech_eres2netv2_sv_zh-cn_16k-common
  match_threshold: 0.5
  min_segment_seconds: 0.5

asr:
  model: FunAudioLLM/Fun-ASR-Nano-2512
  vad_model: fsmn-vad
  vad_max_single_segment_time: 30000
  # No punc_model — Fun-ASR-Nano generates punctuation via LLM
  # bf16 is auto-enabled for GPU in FunASRNanoTranscriber

srt:
  max_chars_per_line: 20
  min_duration: 1.0
  merge_gap: 0.5
  speaker_label: true
```

- [ ] **Step 2: Commit**

```bash
git add config.yaml
git commit -m "feat(config): add backend field to config.yaml"
```

---

### Task 13: Update __main__.py to pass backend

**Files:**
- Modify: `transcribe/__main__.py`

Need to verify whether `__main__.py` maps CLI args to config and whether `backend` needs explicit wiring.

- [ ] **Step 1: Read `transcribe/__main__.py` and determine if backend is auto-mapped**

Read the file. If it uses a mapping from `args` to `cli_overrides`, add `"backend": args.backend` to the mapping. If it relies on `parse_args()` + `load_config()` auto-merge, verify that the field name `"backend"` in the argparse namespace matches `_PIPELINE_CONFIG_FIELDS` (it does after Task 9).

- [ ] **Step 2: Make any necessary changes and commit**

```bash
git add transcribe/__main__.py
git commit -m "feat(cli): wire --backend argument through to pipeline config"
```

---

### Task 14: Update tests

**Files:**
- Modify: `tests/test_asr.py`

The test file must be restructured to:
1. Update imports from the new package structure
2. Test `create_asr` factory
3. Keep existing `restore_hotwords` and `parse_timestamps` tests (update imports)
4. Test `FunASRNanoTranscriber` (existing slow tests)
5. Test `FunASRParaformerTranscriber` (mirrored slow tests)
6. Update `parse_timestamps` imports to use the new `utils` path

- [ ] **Step 1: Update test imports and add factory tests**

Replace the entire `tests/test_asr.py` with:

```python
"""Tests for the ASR backend modules."""

from __future__ import annotations

import numpy as np
import pytest

from transcribe.data.types import AudioSegment, TranscriptSegment
from transcribe.models.asr import create_asr, list_backends, restore_hotwords, parse_timestamps
from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.nano import FunASRNanoTranscriber
from transcribe.models.asr.paraformer import FunASRParaformerTranscriber


@pytest.fixture
def silence_audio() -> AudioSegment:
    """1 second of silence at 16 kHz."""
    return AudioSegment(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
    )


# --- Factory tests ---


def test_list_backends_contains_both() -> None:
    """Both backends should be registered."""
    backends = list_backends()
    assert "Fun-ASR-Paraformer" in backends
    assert "Fun-ASR-Nano" in backends


def test_create_asr_unknown_backend_raises() -> None:
    """Unknown backend name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown ASR backend"):
        create_asr("nonexistent")


def test_create_asr_returns_correct_type() -> None:
    """create_asr should return the correct backend class."""
    nano = create_asr("Fun-ASR-Nano", device="cpu")
    assert isinstance(nano, FunASRNanoTranscriber)
    assert isinstance(nano, ASRBase)

    paraformer = create_asr("Fun-ASR-Paraformer", device="cpu")
    assert isinstance(paraformer, FunASRParaformerTranscriber)
    assert isinstance(paraformer, ASRBase)


def test_backends_support_hotwords() -> None:
    """Both backends should declare hotword support."""
    nano = create_asr("Fun-ASR-Nano", device="cpu")
    paraformer = create_asr("Fun-ASR-Paraformer", device="cpu")
    assert nano.supports_hotwords is True
    assert paraformer.supports_hotwords is True


# --- FunASRNanoTranscriber slow tests ---


@pytest.mark.slow
def test_nano_init() -> None:
    """FunASRNanoTranscriber should initialise without error."""
    transcriber = FunASRNanoTranscriber(device="cpu")
    assert transcriber is not None
    assert transcriber._hotword_list == []


@pytest.mark.slow
def test_nano_transcribe_silence(silence_audio: AudioSegment) -> None:
    """Silence should produce empty or minimal output, not crash."""
    transcriber = FunASRNanoTranscriber(device="cpu")
    results = transcriber.transcribe(silence_audio)
    assert isinstance(results, list)
    for seg in results:
        assert isinstance(seg, TranscriptSegment)


@pytest.mark.slow
def test_nano_transcribe_with_hotwords(silence_audio: AudioSegment, tmp_path) -> None:
    """FunASRNanoTranscriber should accept a hotword file without error."""
    hw_file = tmp_path / "hotwords.txt"
    hw_file.write_text("张三\n李四\n", encoding="utf-8")
    transcriber = FunASRNanoTranscriber(device="cpu", hotword_path=str(hw_file))
    assert transcriber._hotword_list == ["张三", "李四"]
    results = transcriber.transcribe(silence_audio)
    assert isinstance(results, list)


@pytest.mark.slow
def test_nano_load_hotwords_missing_file() -> None:
    """Missing hotword file should be treated as no hotwords."""
    transcriber = FunASRNanoTranscriber(device="cpu", hotword_path="/nonexistent/path.txt")
    assert transcriber._hotword_list == []


# --- FunASRParaformerTranscriber slow tests ---


@pytest.mark.slow
def test_paraformer_init() -> None:
    """FunASRParaformerTranscriber should initialise without error."""
    transcriber = FunASRParaformerTranscriber(device="cpu")
    assert transcriber is not None
    assert transcriber._hotwords is None
    assert transcriber._hotword_list == []


@pytest.mark.slow
def test_paraformer_transcribe_silence(silence_audio: AudioSegment) -> None:
    """Silence should produce empty or minimal output, not crash."""
    transcriber = FunASRParaformerTranscriber(device="cpu")
    results = transcriber.transcribe(silence_audio)
    assert isinstance(results, list)
    for seg in results:
        assert isinstance(seg, TranscriptSegment)


@pytest.mark.slow
def test_paraformer_transcribe_with_hotwords(silence_audio: AudioSegment, tmp_path) -> None:
    """FunASRParaformerTranscriber should accept a hotword file without error."""
    hw_file = tmp_path / "hotwords.txt"
    hw_file.write_text("张三\n李四\n", encoding="utf-8")
    transcriber = FunASRParaformerTranscriber(device="cpu", hotword_path=str(hw_file))
    assert transcriber._hotwords == "张三 李四"
    assert transcriber._hotword_list == ["张三", "李四"]
    results = transcriber.transcribe(silence_audio)
    assert isinstance(results, list)


@pytest.mark.slow
def test_paraformer_load_hotwords_missing_file() -> None:
    """Missing hotword file should be treated as no hotwords."""
    transcriber = FunASRParaformerTranscriber(device="cpu", hotword_path="/nonexistent/path.txt")
    assert transcriber._hotwords is None
    assert transcriber._hotword_list == []


# --- Hotword punctuation restoration tests (pure function, no model needed) ---


def test_restore_removes_comma_inside_term() -> None:
    """Comma inserted inside a hotword term should be removed."""
    assert restore_hotwords("大家看朽，叶飞过来了", ["朽叶"]) == "大家看朽叶飞过来了"


def test_restore_preserves_external_punctuation() -> None:
    """Punctuation before/after a hotword should be preserved."""
    assert restore_hotwords("朽，叶，来了", ["朽叶"]) == "朽叶，来了"


def test_restore_multiple_hotwords() -> None:
    """Multiple hotwords with internal punctuation should all be restored."""
    result = restore_hotwords("硅，基流动的朽。叶团队", ["硅基流动", "朽叶"])
    assert result == "硅基流动的朽叶团队"


def test_restore_no_hotwords() -> None:
    """With no hotwords, text should pass through unchanged."""
    assert restore_hotwords("朽，叶来了", []) == "朽，叶来了"


def test_restore_no_hotwords_none_list() -> None:
    """With empty hotword list, text passes through."""
    assert restore_hotwords("朽，叶来了", []) == "朽，叶来了"


def test_restore_single_char_hotword() -> None:
    """Single-character hotwords can't have internal punctuation, pass through."""
    assert restore_hotwords("好，的", ["好"]) == "好，的"


def test_restore_preserves_unrelated_text() -> None:
    """Text that doesn't match any hotword should be untouched."""
    assert restore_hotwords("今天天气很好，我们出去玩", ["硅基流动"]) == "今天天气很好，我们出去玩"


def test_restore_empty_string() -> None:
    """Empty string should return empty string."""
    assert restore_hotwords("", ["朽叶"]) == ""


def test_restore_various_punctuation() -> None:
    """Various Chinese punctuation types inside hotword should all be removed."""
    assert restore_hotwords("硅基？流动", ["硅基流动"]) == "硅基流动"


def test_restore_period_inside_term() -> None:
    """Period inserted inside a hotword should be removed."""
    assert restore_hotwords("来到了硅。基流动公司", ["硅基流动"]) == "来到了硅基流动公司"


def test_restore_two_char_hotword_with_space() -> None:
    """Space inside a hotword should be removed."""
    assert restore_hotwords("朽 叶", ["朽叶"]) == "朽叶"


def test_restore_hotword_at_string_boundaries() -> None:
    """Hotword at the start/end of text should be restored."""
    assert restore_hotwords("朽，叶来了", ["朽叶"]) == "朽叶来了"
    assert restore_hotwords("去找朽，叶", ["朽叶"]) == "去找朽叶"


def test_restore_repeated_hotword() -> None:
    """Multiple occurrences of the same hotword should all be restored."""
    assert restore_hotwords("朽，叶和朽。叶", ["朽叶"]) == "朽叶和朽叶"


def test_restore_overlapping_text_not_hotword() -> None:
    """Text containing hotword chars in different order should not be touched."""
    assert restore_hotwords("叶朽来了", ["朽叶"]) == "叶朽来了"


# --- Timestamp parsing tests (pure function, no model needed) ---


def test_parse_timestamps_dict_format() -> None:
    """Fun-ASR-Nano dict-format timestamps: seconds, first start / last end."""
    timestamps = [
        {"token": "你", "start_time": 0.10, "end_time": 0.20, "score": 0.99},
        {"token": "好", "start_time": 0.20, "end_time": 0.30, "score": 0.98},
    ]
    start, end = parse_timestamps(timestamps)
    assert start == pytest.approx(0.10)
    assert end == pytest.approx(0.30)


def test_parse_timestamps_single_entry() -> None:
    """Single-entry timestamp list."""
    timestamps = [
        {"token": "嗨", "start_time": 0.50, "end_time": 0.60, "score": 0.95},
    ]
    start, end = parse_timestamps(timestamps)
    assert start == pytest.approx(0.50)
    assert end == pytest.approx(0.60)


def test_parse_timestamps_empty_returns_none() -> None:
    """Empty timestamp list returns (None, None)."""
    start, end = parse_timestamps([])
    assert start is None
    assert end is None


def test_parse_timestamps_nested_list_format() -> None:
    """Legacy Paraformer format [[start_ms, end_ms], ...] should still be handled."""
    timestamps = [[100, 200], [200, 350]]
    start, end = parse_timestamps(timestamps)
    assert start == pytest.approx(0.100)
    assert end == pytest.approx(0.350)


def test_parse_timestamps_flat_list_format() -> None:
    """Legacy flat format [start, end, start, end, ...] should still be handled."""
    timestamps = [100, 200, 200, 350]
    start, end = parse_timestamps(timestamps)
    assert start == pytest.approx(0.100)
    assert end == pytest.approx(0.350)


def test_parse_timestamps_dict_missing_keys_returns_none() -> None:
    """Dict entries without start_time/end_time keys should not crash."""
    timestamps = [{"token": "<sil>"}]
    start, end = parse_timestamps(timestamps)
    assert start is None
    assert end is None
```

- [ ] **Step 2: Run non-slow tests to verify imports and factory**

Run: `uv run pytest tests/test_asr.py -m "not slow" -v`
Expected: All pure-function tests and factory tests PASS. Slow tests are deselected.

- [ ] **Step 3: Commit**

```bash
git add tests/test_asr.py
git commit -m "test(asr): restructure tests for dual-backend support"
```

---

### Task 15: Verify end-to-end integration

**Files:** None (verification only)

- [ ] **Step 1: Run all non-slow tests**

Run: `uv run pytest -m "not slow" -v`
Expected: ALL tests PASS (including non-ASR tests).

- [ ] **Step 2: Verify CLI help shows --backend**

Run: `uv run python -m transcribe --help`
Expected: Output includes `--backend {Fun-ASR-Paraformer,Fun-ASR-Nano}` with default `Fun-ASR-Nano`.

- [ ] **Step 3: Verify import path works**

Run: `uv run python -c "from transcribe.models.asr import create_asr, list_backends; print(list_backends())"`
Expected: `['Fun-ASR-Nano', 'Fun-ASR-Paraformer']`

---

### Task 16: Final commit — clean up any remaining references

**Files:** Check for any stale references

- [ ] **Step 1: Search for any remaining references to old `ASRTranscriber`**

Run: `grep -rn "ASRTranscriber" transcribe/ tests/`
Expected: No matches (all replaced with factory or backend-specific class names).

- [ ] **Step 2: Search for any remaining direct imports of old module**

Run: `grep -rn "from transcribe.models.asr import ASRTranscriber" transcribe/ tests/`
Expected: No matches.

- [ ] **Step 3: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore: clean up stale ASRTranscriber references"
```
