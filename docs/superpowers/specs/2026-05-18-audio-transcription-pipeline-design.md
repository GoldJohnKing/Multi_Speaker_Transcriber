# Audio Transcription Pipeline Design

## Overview

A fully offline, local audio transcription pipeline that takes a video file as input and produces an SRT subtitle file with speaker attribution. Designed for Chinese Mandarin outdoor variety show audio with frequent multi-speaker overlap and background noise.

**Key constraints:**
- Language: Chinese Mandarin only
- Deployment: Fully offline/local, no cloud APIs
- Hardware: AMD 7900XTX (20GB VRAM), ROCm
- Python environment: managed by `uv`
- Processing mode: Offline batch processing

## Architecture

```
Video → ① Audio Extract → ② Denoise → ③ Diarize → ④ Separate Overlaps → ⑤ ASR (hotwords) → ⑥ SRT Writer
```

Each stage is an independent Python class with a unified `process(input) -> output` interface. Models are loaded on-demand and released after each stage to manage VRAM (peak ~10GB).

### Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                        CLI Entry (main.py)                       │
│  python -m transcribe input.mp4 --hotwords dict.txt -o out.srt  │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Pipeline Orchestrator                         │
│               (pipeline.py — serial stage dispatch)              │
└──┬────────┬────────┬────────┬────────┬────────┬─────────────────┘
   ▼        ▼        ▼        ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│Extract│ │Denoi-│ │Diari-│ │Sepa- │ │ ASR  │ │ SRT  │
│Audio  │ │se    │ │zation│ │ration│ │      │ │Writer│
│FFmpeg │ │Deep  │ │Pyan- │ │Sep-  │ │SeACo │ │      │
│       │ │Filter│ │note  │ │Former│ │Para- │ │Merge │
│       │ │Net v2│ │3.1   │ │      │ │former│ │spkr+ │
│       │ │      │ │      │ │      │ │      │ │ts→srt│
└──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘
   ①        ②        ③        ④       ⑤        ⑥
```

## Project Structure

```
multi_speaker_transcribe/
├── pyproject.toml              # uv project config (Python 3.10+)
├── config.yaml                 # Default configuration
├── transcribe/
│   ├── __init__.py
│   ├── __main__.py             # CLI entry point
│   ├── cli.py                  # Argument parsing
│   ├── pipeline.py             # Pipeline orchestrator
│   ├── config.py               # Configuration dataclass
│   ├── models/
│   │   ├── __init__.py
│   │   ├── audio_extractor.py  # ① FFmpeg audio extraction
│   │   ├── denoiser.py         # ② DeepFilterNet noise suppression
│   │   ├── diarizer.py         # ③ Pyannote speaker diarization
│   │   ├── separator.py        # ④ SepFormer speech separation
│   │   ├── asr.py              # ⑤ FunASR SeACo-Paraformer
│   │   └── srt_writer.py       # ⑥ SRT generation
│   └── data/
│       ├── __init__.py
│       └── types.py            # Dataclasses (AudioSegment, SpeakerSegment, etc.)
├── hotwords/
│   └── example.txt             # Example hotword file
└── tests/
    └── ...
```

## Core Data Types

```python
# data/types.py

@dataclass
class AudioSegment:
    """A segment of audio data."""
    waveform: np.ndarray          # float32, shape (channels, samples)
    sample_rate: int
    start_time: float             # seconds, relative to original audio
    end_time: float

@dataclass
class SpeakerSegment:
    """A speaker time segment."""
    speaker_id: str               # "SPEAKER_00", "SPEAKER_01", ...
    start_time: float
    end_time: float
    is_overlap: bool = False      # whether this is an overlap region

@dataclass
class DiarizationResult:
    """Speaker diarization result."""
    segments: list[SpeakerSegment]
    num_speakers: int
    overlap_regions: list[tuple[float, float]]  # list of overlap time ranges

@dataclass
class TranscriptSegment:
    """A transcribed segment."""
    speaker_id: str
    start_time: float
    end_time: float
    text: str                     # text with punctuation

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    device: str = "auto"          # "cpu" | "cuda" | "auto" (auto-detect ROCm)
    denoise: bool = True          # enable noise suppression
    hotwords: str | None = None   # hotword file path
    language: str = "zh"
    cache_dir: str = ".cache"     # intermediate artifacts cache directory
    num_speakers: int | None = None  # known speaker count (optional)
```

## Stage Details

### Stage 1: Audio Extraction (`audio_extractor.py`)

- **Input**: Video file path (mp4/mkv/avi/mov etc.)
- **Output**: `AudioSegment` (16kHz mono float32 WAV)
- **Implementation**: FFmpeg subprocess, convert to 16kHz mono WAV, load into memory
- **Key params**: 16kHz sample rate (standard ASR input), mono channel

### Stage 2: Noise Suppression (`denoiser.py`) — Optional

- **Input**: `AudioSegment`
- **Output**: `AudioSegment` (denoised)
- **Model**: DeepFilterNet v2 (~4MB, MIT license)
- **Behavior**:
  - Load DeepFilterNet model -> enhance speech -> return clean audio
  - Skipped when `denoise: false` in config or `--no-denoise` CLI flag
- **Device**: GPU preferred (ROCm via PyTorch), falls back to CPU (RTF ~0.19)
- **Rationale**: DeepFilterNet is trained on DNS4 outdoor noise dataset (wind, traffic, crowd), achieving SOTA DNSMOS scores while preserving speech quality

### Stage 3: Speaker Diarization (`diarizer.py`)

- **Input**: `AudioSegment` (denoised)
- **Output**: `DiarizationResult`
- **Model**: Pyannote `pyannote/speaker-diarization-3.1`
- **Behavior**:
  - Detect all speaker segments with `speaker_id` + time range
  - **Explicitly mark overlap regions** (Pyannote's core capability via powerset multi-class cross-entropy loss)
  - Output `overlap_regions` list for the next stage
  - If `--num-speakers` is provided, pass to Pyannote for higher accuracy
- **Note**: Pyannote models require HuggingFace download on first run (gated access, free token required)

### Stage 4: Speech Separation (`separator.py`) — Overlap regions only

- **Input**: `AudioSegment` + `DiarizationResult`
- **Output**: `list[AudioSegment]` (separated per-speaker audio for overlap regions, with original time offsets)
- **Model**: SpeechBrain SepFormer (`speechbrain/sepformer-whamr`)
- **Behavior**:
  - Crop only the audio corresponding to `overlap_regions` (non-overlap regions pass through unchanged)
  - Run SepFormer on each overlap fragment to separate into 2 independent speaker tracks
  - Associate separated tracks with Pyannote speaker labels (via similarity matching)
  - Non-overlap segments pass through as-is
- **Optimization**: Process in chunks of max 10 seconds each to avoid VRAM overflow

### Stage 5: Speech Recognition (`asr.py`)

- **Input**: All speaker-segmented audio (overlaps separated into individual tracks, non-overlap used directly)
- **Output**: `list[TranscriptSegment]`
- **Model**: FunASR SeACo-Paraformer + FSMN-VAD + CT-Transformer punctuation
- **Hotword mechanism**:
  ```
  # Hotword file format (hotwords/example.txt)
  张三
  李四
  硅基流动
  RAG
  Transformer
  ```
  - Read user hotword file -> space-join -> pass as `hotword` parameter
  - SeACo-Paraformer's neural hotword architecture (3rd generation, bias decoder + ASF) achieves 87% recall on low-frequency words
- **Behavior**:
  - Run ASR on each speaker segment, obtain timestamped text
  - Punctuation restoration via CT-Transformer model
  - Output each segment's `speaker_id` + `start/end_time` + `text`
- **Chinese quality**: Trained on 60,000+ hours Mandarin, SOTA on AISHELL-1/2, WenetSpeech

### Stage 6: SRT Generation (`srt_writer.py`)

- **Input**: `list[TranscriptSegment]`
- **Output**: SRT file
- **Behavior**:
  - Sort all segments by `start_time`
  - Merge adjacent short sentences from the same speaker (gap < 0.5s)
  - Limit max chars per line (default 20) and min display duration (default 1.0s)
  - Speaker label in SRT output:
  ```
  1
  00:00:01,500 --> 00:00:04,200
  [说话人1] 大家好，欢迎来到今天的节目。

  2
  00:00:03,800 --> 00:00:06,100
  [说话人2] 你好你好！
  ```

## VRAM Management

Models are loaded on-demand and released after each stage:

| Stage | Model | VRAM |
|-------|-------|------|
| 2 - Denoise | DeepFilterNet v2 | ~2 GB |
| 3 - Diarize | Pyannote 3.1 | ~4-8 GB |
| 4 - Separate | SepFormer | ~2-4 GB |
| 5 - ASR | Paraformer + VAD + Punc | ~3-4 GB |

Peak VRAM: ~10GB (one stage at a time). Each stage calls `torch.cuda.empty_cache()` after completion. All stages fall back to CPU if GPU is unavailable.

## CLI Interface

```bash
# Basic usage
python -m transcribe input.mp4 -o output.srt

# Full parameters
python -m transcribe input.mp4 \
  --output output.srt \
  --hotwords hotwords/my_dict.txt \
  --num-speakers 4 \
  --no-denoise \
  --device cuda \
  --cache-dir .pipeline_cache \
  --config config.yaml \
  --keep-cache \
  --verbose
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | positional | required | Input video/audio file path |
| `-o, --output` | str | same name `.srt` | Output SRT file path |
| `--hotwords` | str | None | Hotword file path (one word per line) |
| `--num-speakers` | int | None | Known speaker count (auto-detect if omitted) |
| `--no-denoise` | flag | False | Skip noise suppression stage |
| `--device` | str | auto | `cpu` / `cuda` / `auto` (auto-detect ROCm) |
| `--cache-dir` | str | .cache | Intermediate artifacts cache directory |
| `--config` | str | None | YAML config file path |
| `--keep-cache` | flag | False | Keep intermediate artifacts (deleted by default) |
| `-v, --verbose` | flag | False | Print detailed per-stage logs and timing |

## Configuration

```yaml
# config.yaml — default configuration
device: auto
denoise: true
language: zh

# Noise suppression
denoiser:
  model: deepfilternet_v2
  post_filter: true          # Enable post-filter (stronger denoising)

# Speaker diarization
diarizer:
  model: pyannote/speaker-diarization-3.1
  hf_token: null             # HuggingFace token (needed for first model download)
  clustering: hidden_markov  # Clustering algorithm

# Speech separation (overlap regions only)
separator:
  model: speechbrain/sepformer-whamr
  max_segment_seconds: 10    # Max fragment length per separation pass

# Speech recognition
asr:
  model: paraformer-zh       # SeACo-Paraformer
  vad_model: fsmn-vad
  punc_model: ct-punc
  batch_size_s: 300          # Audio seconds per batch

# SRT generation
srt:
  max_chars_per_line: 20     # Max chars per subtitle entry
  min_duration: 1.0          # Min display duration (seconds)
  merge_gap: 0.5             # Same-speaker sentence merge threshold (seconds)
  speaker_label: true        # Include speaker labels in subtitles
```

Priority: CLI args > YAML config > code defaults

## Error Handling

### Device Detection and Fallback

- Auto-detect: `torch.cuda.is_available()` (ROCm exposes as CUDA on Linux)
- Print device info on first run: `使用设备: AMD 7900 XTX (ROCm)`
- Any stage GPU failure -> automatic CPU fallback with warning

### Model Download and Caching

- Pyannote: Requires HuggingFace token (gated access, free to apply)
- FunASR / DeepFilterNet / SpeechBrain: Auto-download from HuggingFace to `~/.cache/huggingface/` on first run
- `--offline` mode: Require all models cached, error instead of attempting download

### Edge Cases

| Scenario | Strategy |
|----------|----------|
| No speech in audio (noise/music only) | VAD detects silence, return empty SRT with warning |
| Single speaker, no overlap | Pyannote returns 1 speaker, skip separation stage entirely |
| Overlap region exceeds 30 seconds | Split into 10-second chunks for processing |
| Empty or missing hotword file | Print warning, continue without hotwords |
| Corrupted video / no audio track | FFmpeg returns non-zero exit code, print clear error |
| VRAM OOM | Auto-retry current stage on CPU |
| Output file exists | Overwrite by default, `--no-clobber` to skip |
| Very long video (> 3 hours) | Process in 30-minute segments, merge at end |

### Progress Display

```
[1/6] 提取音频 ... 完成 (2.1s)
[2/6] 噪声抑制 ... 完成 (1m 23s)
[3/6] 说话人识别 ... 检测到 4 位说话人, 12 个重叠区域 ... 完成 (3m 05s)
[4/6] 重叠语音分离 ... 处理 12 个重叠片段 ... 完成 (2m 18s)
[5/6] 语音转文字 ... 识别 347 个片段 ... 完成 (4m 42s)
[6/6] 生成 SRT ... 输出 312 条字幕 ... 完成 (0.3s)
──────────────────────────────────
总耗时: 11m 30s | 输出: output.srt (312 条字幕)
```

## Dependencies and Deployment

### Python Dependencies (pyproject.toml)

```toml
[project]
name = "multi-speaker-transcribe"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "torchaudio>=2.1",
    "funasr>=1.1",
    "modelscope",
    "pyannote.audio>=3.1",
    "speechbrain>=1.0",
    "deepfilternet>=0.5",
    "numpy",
    "pyyaml",
    "rich",
]
```

### Installation (ROCm, using uv)

```bash
# 1. Initialize project with uv
uv init --no-readme
uv python install 3.12
uv python pin 3.12

# 2. Install PyTorch ROCm version
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# 3. Install project dependencies
uv sync

# 4. First run (downloads models, requires internet)
uv run python -m transcribe test.mp4 --hotwords hotwords/example.txt -o test.srt

# 5. Subsequent runs can be offline
uv run python -m transcribe test.mp4 --hotwords hotwords/example.txt -o test.srt --offline
```

## Component Selection Rationale

| Component | Tool | Why |
|-----------|------|-----|
| ASR | FunASR SeACo-Paraformer | Best Chinese Mandarin quality (60k hrs training). SOTA hotword support (87% recall via neural bias decoder). Built-in timestamps + punctuation. |
| Diarization | Pyannote 3.1 | Best overlap detection in open-source (powerset loss). MIT license. Proven on Chinese datasets (DER 11.7% on AISHELL-4). |
| Separation | SepFormer (SpeechBrain) | Pure PyTorch (ROCm compatible). Apache 2.0. SI-SNRi 22.3 dB on WSJ0-2mix. |
| Denoise | DeepFilterNet v2 | Best outdoor noise suppression (trained on DNS4). SOTA speech quality preservation. MIT license. PyTorch/ROCm. |
| Packaging | uv | Fast Python package manager. Reliable dependency resolution. |
