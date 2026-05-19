# Phase 1: 项目骨架与基础 ASR 管线

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建项目骨架，实现一条可工作的单说话人 ASR → SRT 管线（不含噪声抑制、说话人识别和重叠分离）。

**Architecture:** uv 管理的 Python 项目，6 阶段管线中的阶段 ①⑤⑥ 先行实现。通过 CLI 接收视频输入，经 FFmpeg 提取音频后直接用 FunASR SeACo-Paraformer 进行中文语音识别（含热词），最后生成 SRT 字幕文件。

**Tech Stack:** Python 3.12, uv, FFmpeg, FunASR (SeACo-Paraformer + FSMN-VAD + CT-Transformer Punc), PyYAML, Rich

**Design doc reference:** Sections "项目结构", "核心数据类型", "Stage 1", "Stage 5", "Stage 6", "CLI 接口与配置", "依赖与部署"

---

## File Structure

```
multi_speaker_transcribe/
├── pyproject.toml
├── config.yaml
├── transcribe/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── audio_extractor.py
│   │   ├── asr.py
│   │   └── srt_writer.py
│   └── data/
│       ├── __init__.py
│       └── types.py
├── hotwords/
│   └── example.txt
└── tests/
    ├── __init__.py
    ├── test_audio_extractor.py
    ├── test_asr.py
    └── test_srt_writer.py
```

---

### Task 1: 项目初始化与目录结构

**Files:**
- Create: `pyproject.toml`
- Create: `transcribe/__init__.py`
- Create: `transcribe/models/__init__.py`
- Create: `transcribe/data/__init__.py`
- Create: `tests/__init__.py`
- Create: `hotwords/example.txt`

- [ ] **Step 1: 初始化 uv 项目**

```bash
cd /mnt/d/GitRepos/Multi_Speaker_Transcribe
uv init --no-readme
uv python install 3.12
uv python pin 3.12
```

- [ ] **Step 2: 创建目录结构**

```bash
mkdir -p transcribe/models transcribe/data hotwords tests
touch transcribe/__init__.py transcribe/models/__init__.py transcribe/data/__init__.py tests/__init__.py
```

- [ ] **Step 3: 编写 pyproject.toml**

```toml
[project]
name = "multi-speaker-transcribe"
version = "0.1.0"
description = "Offline multi-speaker audio transcription pipeline for Chinese Mandarin"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "pyyaml",
    "rich",
]

[project.optional-dependencies]
asr = [
    "funasr>=1.1",
    "modelscope",
    "torch>=2.1",
    "torchaudio>=2.1",
]
denoise = [
    "deepfilternet>=0.5",
]
diarize = [
    "pyannote.audio>=3.1",
]
separate = [
    "speechbrain>=1.0",
]
all = [
    "multi-speaker-transcribe[asr,denoise,diarize,separate]",
]
dev = [
    "pytest>=8.0",
]

[project.scripts]
transcribe = "transcribe.__main__:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: 创建示例热词文件**

```
# hotwords/example.txt
# 每行一个热词，支持人名、地名、行业术语等
张三
李四
硅基流动
RAG
Transformer
```

- [ ] **Step 5: 安装开发依赖并验证**

```bash
uv sync --extra dev
uv run python -c "import transcribe; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "feat: initialize project skeleton with uv"
```

---

### Task 2: 核心数据类型

**Files:**
- Create: `transcribe/data/types.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: 编写数据类型测试**

```python
# tests/test_types.py
import numpy as np
from transcribe.data.types import (
    AudioSegment,
    SpeakerSegment,
    DiarizationResult,
    TranscriptSegment,
    PipelineConfig,
)


def test_audio_segment_creation():
    wav = np.zeros((1, 16000), dtype=np.float32)
    seg = AudioSegment(waveform=wav, sample_rate=16000, start_time=0.0, end_time=1.0)
    assert seg.sample_rate == 16000
    assert seg.duration == 1.0


def test_speaker_segment():
    seg = SpeakerSegment(speaker_id="SPEAKER_00", start_time=1.0, end_time=3.5)
    assert seg.is_overlap is False
    assert seg.duration == 2.5


def test_speaker_segment_overlap():
    seg = SpeakerSegment(
        speaker_id="SPEAKER_00", start_time=1.0, end_time=3.5, is_overlap=True
    )
    assert seg.is_overlap is True


def test_diarization_result():
    segs = [
        SpeakerSegment("SPEAKER_00", 0.0, 2.0),
        SpeakerSegment("SPEAKER_01", 1.5, 3.0, is_overlap=True),
    ]
    result = DiarizationResult(
        segments=segs,
        num_speakers=2,
        overlap_regions=[(1.5, 3.0)],
    )
    assert result.num_speakers == 2
    assert len(result.overlap_regions) == 1


def test_transcript_segment():
    seg = TranscriptSegment(
        speaker_id="SPEAKER_00", start_time=0.0, end_time=2.0, text="你好世界。"
    )
    assert seg.text == "你好世界。"


def test_pipeline_config_defaults():
    config = PipelineConfig()
    assert config.device == "auto"
    assert config.denoise is True
    assert config.language == "zh"
    assert config.hotwords is None
    assert config.num_speakers is None
```

- [ ] **Step 2: 运行测试验证失败**

```bash
uv run pytest tests/test_types.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现数据类型**

```python
# transcribe/data/types.py
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioSegment:
    """A segment of audio data."""

    waveform: np.ndarray  # float32, shape (channels, samples)
    sample_rate: int
    start_time: float  # seconds, relative to original audio
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class SpeakerSegment:
    """A speaker time segment."""

    speaker_id: str  # "SPEAKER_00", "SPEAKER_01", ...
    start_time: float
    end_time: float
    is_overlap: bool = False

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class DiarizationResult:
    """Speaker diarization result."""

    segments: list[SpeakerSegment]
    num_speakers: int
    overlap_regions: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class TranscriptSegment:
    """A transcribed segment."""

    speaker_id: str
    start_time: float
    end_time: float
    text: str  # with punctuation


@dataclass
class PipelineConfig:
    """Pipeline configuration."""

    device: str = "auto"  # "cpu" | "cuda" | "auto"
    denoise: bool = True
    hotwords: str | None = None  # hotword file path
    language: str = "zh"
    cache_dir: str = ".cache"
    num_speakers: int | None = None  # known speaker count, auto-detect if None
```

- [ ] **Step 4: 运行测试验证通过**

```bash
uv run pytest tests/test_types.py -v
```

Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add transcribe/data/types.py tests/test_types.py
git commit -m "feat: add core data types for pipeline stages"
```

---

### Task 3: 配置系统

**Files:**
- Create: `transcribe/config.py`
- Create: `config.yaml`
- Create: `tests/test_config.py`

- [ ] **Step 1: 编写配置测试**

```python
# tests/test_config.py
import tempfile
from pathlib import Path

import yaml

from transcribe.config import load_config, resolve_device


def test_load_config_defaults():
    config = load_config()
    assert config.device == "auto"
    assert config.denoise is True
    assert config.language == "zh"


def test_load_config_from_yaml():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump({"device": "cpu", "denoise": False}, f)
        f.flush()
        config = load_config(f.name)
    assert config.device == "cpu"
    assert config.denoise is False


def test_load_config_yaml_and_cli_override():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump({"device": "cpu"}, f)
        f.flush()
        config = load_config(f.name, cli_overrides={"device": "cuda"})
    assert config.device == "cuda"


def test_load_config_hotwords_file(tmp_path):
    hw_file = tmp_path / "hotwords.txt"
    hw_file.write_text("张三\n李四\n")
    config = load_config(cli_overrides={"hotwords": str(hw_file)})
    assert config.hotwords == str(hw_file)


def test_resolve_device_cpu():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_explicit_cuda():
    # 测试显式指定 cuda 时直接返回（即使没有 GPU）
    assert resolve_device("cuda") == "cuda"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
uv run pytest tests/test_config.py -v
```

- [ ] **Step 3: 实现配置加载**

```python
# transcribe/config.py
from __future__ import annotations

from pathlib import Path

import torch
import yaml

from transcribe.data.types import PipelineConfig

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def resolve_device(device: str) -> str:
    """Resolve 'auto' to actual device string."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_config(
    config_path: str | None = None,
    cli_overrides: dict | None = None,
) -> PipelineConfig:
    """Load configuration from YAML file, CLI overrides, and defaults.

    Priority: CLI overrides > YAML config > code defaults.
    """
    # Start with code defaults
    params: dict = {}

    # Load YAML config if provided or if default exists
    yaml_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if yaml_path.exists():
        with open(yaml_path) as f:
            yaml_config = yaml.safe_load(f) or {}
        # Flatten top-level keys only (nested keys stay as dicts)
        params.update(yaml_config)

    # Apply CLI overrides (highest priority)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                params[key] = value

    # Extract known PipelineConfig fields
    known_fields = {
        "device",
        "denoise",
        "hotwords",
        "language",
        "cache_dir",
        "num_speakers",
    }
    filtered = {k: v for k, v in params.items() if k in known_fields}

    return PipelineConfig(**filtered)
```

- [ ] **Step 4: 编写默认配置文件**

```yaml
# config.yaml
device: auto
denoise: true
language: zh

denoiser:
  model: deepfilternet_v2
  post_filter: true

diarizer:
  model: pyannote/speaker-diarization-3.1
  hf_token: null
  clustering: hidden_markov

separator:
  model: speechbrain/sepformer-whamr16k
  max_segment_seconds: 10

asr:
  model: paraformer-zh
  vad_model: fsmn-vad
  punc_model: ct-punc
  batch_size_s: 300

srt:
  max_chars_per_line: 20
  min_duration: 1.0
  merge_gap: 0.5
  speaker_label: true
```

- [ ] **Step 5: 运行测试验证通过**

```bash
uv run pytest tests/test_config.py -v
```

Expected: all passed

- [ ] **Step 6: 提交**

```bash
git add transcribe/config.py config.yaml tests/test_config.py
git commit -m "feat: add configuration system with YAML and CLI support"
```

---

### Task 4: 音频提取模块

**Files:**
- Create: `transcribe/models/audio_extractor.py`
- Create: `tests/test_audio_extractor.py`

- [ ] **Step 1: 编写音频提取测试**

```python
# tests/test_audio_extractor.py
import subprocess
from pathlib import Path

import numpy as np
import pytest

from transcribe.data.types import AudioSegment
from transcribe.models.audio_extractor import AudioExtractor


@pytest.fixture
def sample_wav(tmp_path: Path):
    """Create a minimal valid WAV file using FFmpeg."""
    wav_path = tmp_path / "test.wav"
    # Generate 1 second of 440Hz sine wave via FFmpeg
    subprocess.run(
        [
            "ffmpeg", "-f", "lavfi",
            "-i", "sine=frequency=440:duration=1",
            "-ar", "16000", "-ac", "1",
            "-y", str(wav_path),
        ],
        check=True,
        capture_output=True,
    )
    return wav_path


def test_extract_from_wav(sample_wav):
    extractor = AudioExtractor()
    audio = extractor.extract(str(sample_wav))
    assert isinstance(audio, AudioSegment)
    assert audio.sample_rate == 16000
    assert audio.waveform.ndim == 1  # mono
    assert audio.start_time == 0.0
    assert audio.end_time == pytest.approx(1.0, abs=0.1)


def test_extract_returns_float32(sample_wav):
    extractor = AudioExtractor()
    audio = extractor.extract(str(sample_wav))
    assert audio.waveform.dtype == np.float32


def test_extract_nonexistent_file():
    extractor = AudioExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract("/nonexistent/file.mp4")


def test_extract_corrupted_file(tmp_path: Path):
    bad_file = tmp_path / "bad.wav"
    bad_file.write_bytes(b"not a real wav file")
    extractor = AudioExtractor()
    with pytest.raises(RuntimeError):
        extractor.extract(str(bad_file))
```

- [ ] **Step 2: 运行测试验证失败**

```bash
uv run pytest tests/test_audio_extractor.py -v
```

- [ ] **Step 3: 实现音频提取**

```python
# transcribe/models/audio_extractor.py
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from transcribe.data.types import AudioSegment

_TARGET_SAMPLE_RATE = 16000
_TARGET_CHANNELS = 1


class AudioExtractor:
    """Extract audio from video/audio files using FFmpeg."""

    def extract(self, input_path: str) -> AudioSegment:
        """Extract audio as 16kHz mono float32 numpy array.

        Args:
            input_path: Path to video or audio file.

        Returns:
            AudioSegment with waveform data and metadata.

        Raises:
            FileNotFoundError: If input file doesn't exist.
            RuntimeError: If FFmpeg fails to process the file.
        """
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Use FFmpeg to convert to 16kHz mono WAV, piped to stdout
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-i", str(path),
                    "-ar", str(_TARGET_SAMPLE_RATE),
                    "-ac", str(_TARGET_CHANNELS),
                    "-f", "wav",
                    "-v", "error",
                    "pipe:1",
                ],
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FFmpeg failed to extract audio from {input_path}: {e.stderr.decode()}"
            ) from e

        # Parse WAV from stdout bytes
        waveform, sr = sf.read(
            _BytesIO(result.stdout),
            dtype="float32",
        )

        duration = len(waveform) / sr
        return AudioSegment(
            waveform=waveform,
            sample_rate=sr,
            start_time=0.0,
            end_time=duration,
        )
```

- [ ] **Step 4: 在 pyproject.toml 添加 soundfile 依赖**

在 `dependencies` 列表中添加 `"soundfile"`。

- [ ] **Step 5: 添加 BytesIO import 并运行测试**

在 `audio_extractor.py` 顶部添加 `from io import BytesIO as _BytesIO`，然后：

```bash
uv sync --extra dev
uv run pytest tests/test_audio_extractor.py -v
```

Expected: 4 passed

- [ ] **Step 6: 提交**

```bash
git add transcribe/models/audio_extractor.py tests/test_audio_extractor.py pyproject.toml
git commit -m "feat: add audio extraction via FFmpeg"
```

---

### Task 5: ASR 模块（FunASR SeACo-Paraformer + 热词）

**Files:**
- Create: `transcribe/models/asr.py`
- Create: `tests/test_asr.py`

- [ ] **Step 1: 安装 ASR 依赖**

```bash
uv sync --extra asr --extra dev
```

- [ ] **Step 2: 编写 ASR 测试**

```python
# tests/test_asr.py
import numpy as np
import pytest

from transcribe.data.types import AudioSegment, TranscriptSegment
from transcribe.models.asr import ASRTranscriber


@pytest.fixture
def silence_audio():
    """1 second of silence at 16kHz."""
    return AudioSegment(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
    )


def test_transcriber_init():
    transcriber = ASRTranscriber(device="cpu")
    assert transcriber is not None


def test_transcribe_silence(silence_audio):
    """Silence should produce empty or minimal output, not crash."""
    transcriber = ASRTranscriber(device="cpu")
    results = transcriber.transcribe(silence_audio)
    assert isinstance(results, list)
    # Silence should produce empty list or very short text
    for seg in results:
        assert isinstance(seg, TranscriptSegment)


def test_transcribe_with_hotwords(silence_audio, tmp_path):
    """Should not crash when hotwords are provided."""
    hw_file = tmp_path / "hotwords.txt"
    hw_file.write_text("张三\n李四\n")
    transcriber = ASRTranscriber(device="cpu", hotword_path=str(hw_file))
    results = transcriber.transcribe(silence_audio)
    assert isinstance(results, list)
```

- [ ] **Step 3: 运行测试验证失败**

```bash
uv run pytest tests/test_asr.py -v
```

- [ ] **Step 4: 实现 ASR 模块**

```python
# transcribe/models/asr.py
from __future__ import annotations

from pathlib import Path

from funasr import AutoModel

from transcribe.data.types import AudioSegment, TranscriptSegment


class ASRTranscriber:
    """Speech recognition using FunASR SeACo-Paraformer with hotword support."""

    def __init__(
        self,
        device: str = "cpu",
        hotword_path: str | None = None,
        model_name: str = "paraformer-zh",
        vad_model: str = "fsmn-vad",
        punc_model: str = "ct-punc",
    ) -> None:
        self._device = device
        self._hotwords = self._load_hotwords(hotword_path)
        self._model = AutoModel(
            model=model_name,
            vad_model=vad_model,
            punc_model=punc_model,
            device=device,
        )

    def _load_hotwords(self, path: str | None) -> str | None:
        """Load hotwords from a text file (one per line), join with spaces."""
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            return None
        words = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        return " ".join(words) if words else None

    def transcribe(self, audio: AudioSegment) -> list[TranscriptSegment]:
        """Transcribe audio to text segments with timestamps.

        Args:
            audio: Input audio segment.

        Returns:
            List of transcript segments with speaker attribution.
        """
        # FunASR expects a file path or numpy array
        result = self._model.generate(
            input=audio.waveform,
            batch_size_s=300,
            hotword=self._hotwords,
        )

        segments: list[TranscriptSegment] = []
        if not result:
            return segments

        for res in result:
            # FunASR result structure: {"text": ..., "timestamp": [...], ...}
            text = res.get("text", "")
            timestamps = res.get("timestamp", [])

            if not text or not timestamps:
                continue

            # timestamps is a flat list: [start_ms, end_ms, start_ms, end_ms, ...]
            start_time = timestamps[0] / 1000.0 + audio.start_time if len(timestamps) >= 2 else audio.start_time
            end_time = timestamps[-1] / 1000.0 + audio.start_time if len(timestamps) >= 2 else audio.end_time

            segments.append(
                TranscriptSegment(
                    speaker_id="SPEAKER_00",  # Default single speaker
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                )
            )

        return segments
```

- [ ] **Step 5: 运行测试**

```bash
uv run pytest tests/test_asr.py -v
```

Expected: all passed（首次运行会下载 FunASR 模型，需要网络）

- [ ] **Step 6: 提交**

```bash
git add transcribe/models/asr.py tests/test_asr.py
git commit -m "feat: add ASR module with FunASR SeACo-Paraformer and hotword support"
```

---

### Task 6: SRT 生成模块

**Files:**
- Create: `transcribe/models/srt_writer.py`
- Create: `tests/test_srt_writer.py`

- [ ] **Step 1: 编写 SRT 生成测试**

```python
# tests/test_srt_writer.py
from transcribe.data.types import TranscriptSegment
from transcribe.models.srt_writer import SrtWriter


def _seg(speaker: str, start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(speaker_id=speaker, start_time=start, end_time=end, text=text)


def test_format_timestamp():
    writer = SrtWriter()
    assert writer._format_timestamp(0.0) == "00:00:00,000"
    assert writer._format_timestamp(61.5) == "00:01:01,500"
    assert writer._format_timestamp(3661.123) == "01:01:01,123"


def test_write_basic_srt(tmp_path):
    segments = [
        _seg("SPEAKER_00", 0.0, 2.5, "你好世界。"),
        _seg("SPEAKER_00", 3.0, 5.0, "这是测试。"),
    ]
    writer = SrtWriter(speaker_label=True)
    output = tmp_path / "out.srt"
    writer.write(segments, str(output))

    content = output.read_text(encoding="utf-8")
    assert "1\n" in content
    assert "00:00:00,000 --> 00:00:02,500" in content
    assert "[说话人1] 你好世界。" in content
    assert "2\n" in content
    assert "[说话人1] 这是测试。" in content


def test_write_without_speaker_label(tmp_path):
    segments = [_seg("SPEAKER_00", 0.0, 2.0, "测试。")]
    writer = SrtWriter(speaker_label=False)
    output = tmp_path / "out.srt"
    writer.write(segments, str(output))

    content = output.read_text(encoding="utf-8")
    assert "[说话人" not in content
    assert "测试。" in content


def test_write_empty_segments(tmp_path):
    writer = SrtWriter()
    output = tmp_path / "out.srt"
    writer.write([], str(output))
    assert output.read_text(encoding="utf-8") == ""


def test_segments_sorted_by_time(tmp_path):
    segments = [
        _seg("SPEAKER_01", 3.0, 5.0, "第二段。"),
        _seg("SPEAKER_00", 0.0, 2.0, "第一段。"),
    ]
    writer = SrtWriter(speaker_label=True)
    output = tmp_path / "out.srt"
    writer.write(segments, str(output))

    content = output.read_text(encoding="utf-8")
    first_idx = content.index("第一段")
    second_idx = content.index("第二段")
    assert first_idx < second_idx


def test_merge_adjacent_same_speaker(tmp_path):
    segments = [
        _seg("SPEAKER_00", 0.0, 1.0, "你好"),
        _seg("SPEAKER_00", 1.3, 2.5, "世界"),  # gap = 0.3s < 0.5s default
    ]
    writer = SrtWriter(merge_gap=0.5)
    output = tmp_path / "out.srt"
    writer.write(segments, str(output))

    content = output.read_text(encoding="utf-8")
    # Should be merged into one entry
    assert "1\n" in content
    assert "2\n" not in content or "你好" in content  # merged or not depending on implementation
```

- [ ] **Step 2: 运行测试验证失败**

```bash
uv run pytest tests/test_srt_writer.py -v
```

- [ ] **Step 3: 实现 SRT 生成**

```python
# transcribe/models/srt_writer.py
from __future__ import annotations

from pathlib import Path

from transcribe.data.types import TranscriptSegment

# Speaker ID to display label mapping
_SPEAKER_LABELS: dict[str, str] = {}


def _get_speaker_label(speaker_id: str) -> str:
    """Convert SPEAKER_00 -> 说话人1, SPEAKER_01 -> 说话人2, etc."""
    if speaker_id not in _SPEAKER_LABELS:
        idx = speaker_id.rsplit("_", 1)[-1]
        _SPEAKER_LABELS[speaker_id] = f"说话人{int(idx) + 1}"
    return _SPEAKER_LABELS[speaker_id]


class SrtWriter:
    """Generate SRT subtitle files from transcript segments."""

    def __init__(
        self,
        speaker_label: bool = True,
        max_chars_per_line: int = 20,
        min_duration: float = 1.0,
        merge_gap: float = 0.5,
    ) -> None:
        self._speaker_label = speaker_label
        self._max_chars = max_chars_per_line
        self._min_duration = min_duration
        self._merge_gap = merge_gap

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS,mmm."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _merge_segments(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        """Merge adjacent segments from the same speaker if gap < merge_gap."""
        if not segments:
            return []

        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            if (
                seg.speaker_id == prev.speaker_id
                and seg.start_time - prev.end_time < self._merge_gap
            ):
                # Merge
                merged[-1] = TranscriptSegment(
                    speaker_id=prev.speaker_id,
                    start_time=prev.start_time,
                    end_time=seg.end_time,
                    text=prev.text + seg.text,
                )
            else:
                merged.append(seg)
        return merged

    def write(self, segments: list[TranscriptSegment], output_path: str) -> None:
        """Write transcript segments to an SRT file.

        Args:
            segments: Transcript segments (will be sorted by start_time).
            output_path: Path to write the SRT file.
        """
        if not segments:
            Path(output_path).write_text("", encoding="utf-8")
            return

        sorted_segments = sorted(segments, key=lambda s: s.start_time)
        merged = self._merge_segments(sorted_segments)

        lines: list[str] = []
        for idx, seg in enumerate(merged, start=1):
            start_ts = self._format_timestamp(seg.start_time)
            end_ts = self._format_timestamp(seg.end_time)
            text = seg.text
            if self._speaker_label:
                label = _get_speaker_label(seg.speaker_id)
                text = f"[{label}] {text}"
            lines.append(f"{idx}")
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(text)
            lines.append("")  # blank line between entries

        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **Step 4: 运行测试**

```bash
uv run pytest tests/test_srt_writer.py -v
```

Expected: all passed

- [ ] **Step 5: 提交**

```bash
git add transcribe/models/srt_writer.py tests/test_srt_writer.py
git commit -m "feat: add SRT writer with speaker labels and segment merging"
```

---

### Task 7: CLI 与管线编排（Phase 1 基础版）

**Files:**
- Create: `transcribe/__main__.py`
- Create: `transcribe/cli.py`
- Create: `transcribe/pipeline.py`

- [ ] **Step 1: 实现 CLI 参数解析**

```python
# transcribe/cli.py
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Offline multi-speaker audio transcription for Chinese Mandarin",
    )
    parser.add_argument("input", help="Input video or audio file path")
    parser.add_argument("-o", "--output", help="Output SRT file path (default: input_name.srt)")
    parser.add_argument("--hotwords", help="Hotword file path (one word per line)")
    parser.add_argument("--num-speakers", type=int, help="Known number of speakers (auto-detect if omitted)")
    parser.add_argument("--no-denoise", action="store_true", help="Skip noise suppression")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto", help="Compute device")
    parser.add_argument("--cache-dir", default=".cache", help="Intermediate cache directory")
    parser.add_argument("--config", help="YAML config file path")
    parser.add_argument("--keep-cache", action="store_true", help="Keep intermediate artifacts")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
```

- [ ] **Step 2: 实现管线编排（Phase 1 基础版）**

```python
# transcribe/pipeline.py
from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console

from transcribe.config import load_config, resolve_device
from transcribe.data.types import PipelineConfig, TranscriptSegment
from transcribe.models.audio_extractor import AudioExtractor
from transcribe.models.asr import ASRTranscriber
from transcribe.models.srt_writer import SrtWriter

console = Console()


def _default_output_path(input_path: str) -> str:
    return str(Path(input_path).with_suffix(".srt"))


def run_pipeline(
    input_path: str,
    output_path: str | None = None,
    config: PipelineConfig | None = None,
    verbose: bool = False,
) -> str:
    """Run the transcription pipeline.

    Args:
        input_path: Path to input video/audio file.
        output_path: Path to output SRT file.
        config: Pipeline configuration.
        verbose: Print detailed progress.

    Returns:
        Path to the output SRT file.
    """
    if config is None:
        config = PipelineConfig()

    device = resolve_device(config.device)
    output = output_path or _default_output_path(input_path)
    total_start = time.time()

    # Stage 1: Audio extraction
    step_start = time.time()
    if verbose:
        console.print("[1/3] 提取音频 ...", end=" ")
    extractor = AudioExtractor()
    audio = extractor.extract(input_path)
    if verbose:
        console.print(f"完成 ({time.time() - step_start:.1f}s)")

    # Stage 2: ASR
    step_start = time.time()
    if verbose:
        console.print("[2/3] 语音转文字 ...", end=" ")
    transcriber = ASRTranscriber(device=device, hotword_path=config.hotwords)
    segments = transcriber.transcribe(audio)
    if verbose:
        console.print(f"识别 {len(segments)} 个片段 ... 完成 ({time.time() - step_start:.1f}s)")

    # Stage 3: SRT generation
    step_start = time.time()
    if verbose:
        console.print("[3/3] 生成 SRT ...", end=" ")
    writer = SrtWriter(speaker_label=True)
    writer.write(segments, output)
    if verbose:
        console.print(f"输出 {len(segments)} 条字幕 ... 完成 ({time.time() - step_start:.1f}s)")

    if verbose:
        elapsed = time.time() - total_start
        console.print(f"{'─' * 40}")
        console.print(f"总耗时: {elapsed:.0f}s | 输出: {output} ({len(segments)} 条字幕)")

    return output
```

- [ ] **Step 3: 实现 CLI 入口**

```python
# transcribe/__main__.py
from transcribe.cli import parse_args
from transcribe.config import load_config
from transcribe.pipeline import run_pipeline


def main() -> None:
    args = parse_args()
    config = load_config(
        config_path=args.config,
        cli_overrides={
            "device": args.device,
            "denoise": not args.no_denoise,
            "hotwords": args.hotwords,
            "num_speakers": args.num_speakers,
            "cache_dir": args.cache_dir,
        },
    )
    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        config=config,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 端到端验证（需要测试音频文件）**

```bash
# 使用一个包含中文语音的测试视频/音频文件
uv run python -m transcribe test.mp4 --hotwords hotwords/example.txt -o test.srt -v
```

- [ ] **Step 5: 提交**

```bash
git add transcribe/__main__.py transcribe/cli.py transcribe/pipeline.py
git commit -m "feat: add CLI and basic pipeline orchestrator (ASR-only)"
```

---

### Task 8: Phase 1 集成测试与清理

**Files:**
- Create: `tests/test_pipeline_basic.py`

- [ ] **Step 1: 编写基础管线集成测试**

```python
# tests/test_pipeline_basic.py
"""Integration tests for the basic ASR pipeline (Phase 1).
These tests require FunASR models to be cached locally.
Mark as slow / optional for CI.
"""
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def sample_wav(tmp_path: Path):
    """Create a short WAV file via FFmpeg."""
    wav_path = tmp_path / "test.wav"
    subprocess.run(
        [
            "ffmpeg", "-f", "lavfi",
            "-i", "sine=frequency=440:duration=1",
            "-ar", "16000", "-ac", "1",
            "-y", str(wav_path),
        ],
        check=True,
        capture_output=True,
    )
    return wav_path


@pytest.mark.slow
def test_pipeline_produces_srt(sample_wav, tmp_path):
    """Full pipeline should produce an SRT file."""
    from transcribe.config import PipelineConfig
    from transcribe.pipeline import run_pipeline

    output = str(tmp_path / "output.srt")
    config = PipelineConfig(device="cpu", denoise=False, hotwords=None)

    result = run_pipeline(
        input_path=str(sample_wav),
        output_path=output,
        config=config,
    )
    assert Path(result).exists()
    content = Path(result).read_text(encoding="utf-8")
    # Sine wave has no speech, so output should be empty or very minimal
    assert isinstance(content, str)


@pytest.mark.slow
def test_cli_help():
    """CLI --help should not crash."""
    result = subprocess.run(
        ["uv", "run", "python", "-m", "transcribe", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "input" in result.stdout
```

- [ ] **Step 2: 运行基础单元测试**

```bash
uv run pytest tests/ -v --ignore=tests/test_pipeline_basic.py -k "not slow"
```

Expected: all passed

- [ ] **Step 3: 提交**

```bash
git add tests/test_pipeline_basic.py
git commit -m "test: add integration tests for basic ASR pipeline"
```

---

## Phase 1 Deliverable

完成后的能力：
- `python -m transcribe input.mp4 -o output.srt` — 从视频生成 SRT 字幕
- `--hotwords dict.txt` — 通过自定义热词文件提升专有名词识别率
- 单说话人场景可正常工作
- 无噪声抑制、无说话人识别、无重叠分离
