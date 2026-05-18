# AGENT.md — Multi-Speaker Transcribe Project Guide

## Project Overview

Offline audio transcription pipeline for Chinese Mandarin. Takes video input, produces SRT subtitles with speaker attribution. Designed for outdoor variety show audio with frequent multi-speaker overlap and background noise.

**Key constraints:** Fully offline/local, AMD 7900XTX 20GB VRAM (ROCm), Python managed by `uv`, Chinese Mandarin only.

---

## Design Document

| Document | Description |
|----------|-------------|
| [Design Spec](docs/superpowers/specs/2026-05-18-audio-transcription-pipeline-design.md) | Complete system design — architecture, data types, CLI, configuration, error handling, deployment |

### Design Sections → Component Mapping

| Design Section | Component | Open-Source Tool | Purpose |
|---------------|-----------|-----------------|---------|
| Stage 1 | `transcribe/models/audio_extractor.py` | FFmpeg | Video → 16kHz mono WAV |
| Stage 2 | `transcribe/models/denoiser.py` | DeepFilterNet v2 | Outdoor noise suppression |
| Stage 3 | `transcribe/models/diarizer.py` | Pyannote 3.1 | Speaker diarization + overlap detection |
| Stage 4 | `transcribe/models/separator.py` | SpeechBrain SepFormer | Overlap speech separation |
| Stage 5 | `transcribe/models/asr.py` | FunASR SeACo-Paraformer | Chinese ASR + hotwords |
| Stage 6 | `transcribe/models/srt_writer.py` | — | Transcript → SRT file |
| Infrastructure | `transcribe/config.py`, `transcribe/cli.py`, `transcribe/pipeline.py` | PyYAML, Rich | Config, CLI, pipeline orchestration |
| Data Types | `transcribe/data/types.py` | — | AudioSegment, SpeakerSegment, DiarizationResult, TranscriptSegment, PipelineConfig |

---

## Implementation Phases

### Phase 1: 项目骨架与基础 ASR 管线

| Item | Details |
|------|---------|
| **Plan** | [phase1-foundation.md](docs/superpowers/plans/2026-05-18-phase1-foundation.md) |
| **Tasks** | 8 tasks: project init, data types, config, audio extractor, ASR, SRT writer, CLI/pipeline, integration tests |
| **Design sections** | "项目结构", "核心数据类型", "Stage 1", "Stage 5", "Stage 6", "CLI 接口与配置", "依赖与部署" |
| **Delivers** | Working single-speaker ASR → SRT pipeline: `python -m transcribe input.mp4 -o output.srt --hotwords dict.txt` |
| **Dependencies** | torch, torchaudio, funasr, modelscope, numpy, pyyaml, rich, soundfile |

### Phase 2: 噪声抑制

| Item | Details |
|------|---------|
| **Plan** | [phase2-denoise.md](docs/superpowers/plans/2026-05-18-phase2-denoise.md) |
| **Tasks** | 2 tasks: Denoiser module, pipeline integration |
| **Design sections** | "Stage 2: Noise Suppression", "VRAM Management" |
| **Delivers** | DeepFilterNet v2 noise suppression (off by default), enabled via `--denoise`, SNR-gated adaptive denoising |
| **Dependencies** | deepfilternet (new) |
| **Prerequisite** | Phase 1 |

### Phase 3: 说话人识别与标注

| Item | Details |
|------|---------|
| **Plan** | [phase3-diarization.md](docs/superpowers/plans/2026-05-18-phase3-diarization.md) |
| **Tasks** | 2 tasks: Diarizer module, pipeline integration |
| **Design sections** | "Stage 3: Speaker Diarization", "VRAM Management" |
| **Delivers** | Multi-speaker detection and labeling with `[说话人1]` / `[说话人2]` in SRT; `--num-speakers N` support |
| **Dependencies** | pyannote.audio (new, requires HF token) |
| **Prerequisite** | Phase 1 + Phase 2 |

### Phase 4: 重叠语音分离

| Item | Details |
|------|---------|
| **Plan** | [phase4-separation.md](docs/superpowers/plans/2026-05-18-phase4-separation.md) |
| **Tasks** | 3 tasks: Separator module, pipeline integration, full pipeline tests |
| **Design sections** | "Stage 4: Speech Separation", "VRAM Management" |
| **Delivers** | Full 6-stage pipeline with overlap handling: extract → denoise → diarize → separate → ASR → SRT. Separation is optional (off by default), enabled via `--separate`. |
| **Dependencies** | speechbrain (new) |
| **Prerequisite** | Phase 1 + Phase 2 + Phase 3 |

---

## Execution Order

```
Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4
(foundation)  (denoise)   (diarize)  (separation)
```

Each phase must be fully complete (tests passing, committed) before starting the next.

## Quick Start (after all phases)

```bash
# Setup
uv sync --extra all
# PyTorch ROCm
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Run — default: ASR + speaker diarization
uv run python -m transcribe input.mp4 \
  --hotwords hotwords/my_dict.txt \
  -o output.srt \
  -v

# Run with noise suppression
uv run python -m transcribe input.mp4 \
  --denoise \
  --hotwords hotwords/my_dict.txt \
  -o output.srt \
  -v

# Run — pure ASR, no speaker diarization
uv run python -m transcribe input.mp4 \
  --no-diarize \
  --hotwords hotwords/my_dict.txt \
  -o output.srt \
  -v

# Run — full pipeline with overlap separation
uv run python -m transcribe input.mp4 \
  --denoise --separate \
  --hotwords hotwords/my_dict.txt \
  -o output.srt \
  -v
```

## CLI Flags Summary

| Flag | Default | Description |
|------|---------|-------------|
| *(none)* | ASR + diarization | Best recognition quality, speaker labels in SRT |
| `--no-diarize` | — | Pure ASR, no speaker labels, faster |
| `--denoise` | off | Enable DeepFilterNet noise suppression |
| `--separate` | off | Enable SepFormer overlap speech separation |
| `--num-speakers N` | auto | Hint known speaker count to diarizer |
| `--hotwords FILE` | none | Hotword file for ASR boosting |
| `-v, --verbose` | off | Print per-stage progress and timing |
