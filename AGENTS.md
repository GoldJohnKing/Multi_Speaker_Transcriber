# AGENT.md — Multi-Speaker Transcribe Project Guide

## Project Overview

Offline audio transcription pipeline for Chinese Mandarin. Takes video input, produces SRT subtitles with speaker attribution. Designed for outdoor variety show audio with frequent multi-speaker overlap and background noise.

**Key constraints:** Fully offline/local, AMD 7900XTX 20GB VRAM (ROCm), Python managed by `uv`, Chinese Mandarin only.

---

## Design Documents

| Document | Description |
|----------|-------------|
| [Pipeline Design](docs/superpowers/specs/2026-05-18-audio-transcription-pipeline-design.md) | Original system design — architecture, data types, CLI, configuration, error handling, deployment |
| [ClearVoice Integration](docs/superpowers/specs/2026-05-19-clearvoice-integration-design.md) | ClearVoice integration design — replaces denoiser and separator, adds speaker matching and TSE |

### Design Sections → Component Mapping

| Design Section | Component | Tool | Purpose |
|---------------|-----------|------|---------|
| Stage 1 | `transcribe/models/audio_extractor.py` | FFmpeg | Video → 16kHz mono WAV |
| Stage 2 | `transcribe/models/denoiser.py` | ClearVoice `MossFormerGAN_SE_16K` | 16kHz 原生噪声抑制 |
| Stage 3 | `transcribe/models/diarizer.py` | Pyannote 3.1 | 说话人识别 + 重叠检测 |
| Stage 4 | `transcribe/models/separator.py` | ClearVoice `MossFormer2_SS_16K` | 16kHz 原生重叠语音分离 |
| Stage 4A | `transcribe/models/extractor.py` | ClearVoice `AV_MossFormer2_TSE_16K` | 目标说话人提取（视频人脸） |
| Stage 4.5 | `transcribe/models/matcher.py` | Pyannote PretrainedSpeakerEmbedding | 声纹嵌入余弦相似度匹配 |
| Stage 5 | `transcribe/models/asr.py` | FunASR SeACo-Paraformer | 中文 ASR + 热词 |
| Stage 6 | `transcribe/models/srt_writer.py` | — | Transcript → SRT |
| Infrastructure | `transcribe/config.py`, `transcribe/cli.py`, `transcribe/pipeline.py` | PyYAML, Rich | 配置、CLI、管线编排 |
| Data Types | `transcribe/data/types.py` | — | AudioSegment, SpeakerSegment, DiarizationResult, TranscriptSegment, PipelineConfig |
| ROCm Patch | `transcribe/models/rocm_compat.py` | — | ClearVoice ROCm GPU 检测兼容 |

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

### Phase 2: ClearVoice 噪声抑制

| Item | Details |
|------|---------|
| **Plan** | [phase2-clearvoice-denoise.md](docs/superpowers/plans/2026-05-19-phase2-clearvoice-denoise.md) |
| **Tasks** | 3 tasks: ROCm 兼容模块, Denoiser 重写（ClearVoice SE）, 管线集成 |
| **Design sections** | ClearVoice Integration: "Stage 2 噪声抑制", "ROCm 兼容性", "采样率简化" |
| **Delivers** | ClearVoice `MossFormerGAN_SE_16K` 噪声抑制（默认关闭），通过 `--denoise` 启用，SNR 门控自适应降噪，全程 16kHz 无重采样 |
| **Dependencies** | clearvoice（新增，替代 deepfilternet） |
| **Prerequisite** | Phase 1 |
| **Key changes** | 移除 48kHz 提取和重采样步骤；denoiser.py 内部实现替换为 ClearVoice SE；新增 rocm_compat.py |

### Phase 3: 说话人识别与标注

| Item | Details |
|------|---------|
| **Plan** | [phase3-diarization.md](docs/superpowers/plans/2026-05-18-phase3-diarization.md) |
| **Tasks** | 2 tasks: Diarizer module, pipeline integration |
| **Design sections** | "Stage 3: Speaker Diarization", "VRAM Management" |
| **Delivers** | Multi-speaker detection and labeling with `[说话人1]` / `[说话人2]` in SRT; `--num-speakers N` support |
| **Dependencies** | pyannote.audio (new, requires HF token) |
| **Prerequisite** | Phase 1 + Phase 2 |

### Phase 4: ClearVoice 语音分离 + 声纹匹配

| Item | Details |
|------|---------|
| **Plan** | [phase4-clearvoice-separation.md](docs/superpowers/plans/2026-05-19-phase4-clearvoice-separation.md) |
| **Tasks** | 3 tasks: Separator 重写（ClearVoice SS）, SpeakerMatcher 新增（声纹嵌入匹配）, 管线集成 |
| **Design sections** | ClearVoice Integration: "Stage 4 语音分离", "Stage 4.5 声纹嵌入匹配" |
| **Delivers** | ClearVoice `MossFormer2_SS_16K` 重叠分离 + Pyannote 声纹嵌入余弦匹配。修复采样率 bug，分离质量 SI-SDRi +24%，说话人标签准确匹配 |
| **Dependencies** | 无新依赖（复用 clearvoice + pyannote.audio） |
| **Prerequisite** | Phase 1 + Phase 2 + Phase 3 |
| **Key changes** | separator.py 内部实现替换为 ClearVoice SS；新增 matcher.py；pipeline.py 中分离→匹配→ASR 流程重构 |

### Phase 5: 目标说话人提取（TSE）

| Item | Details |
|------|---------|
| **Plan** | [phase5-tse.md](docs/superpowers/plans/2026-05-19-phase5-tse.md) |
| **Tasks** | 3 tasks: TargetSpeakerExtractor 新增（ClearVoice TSE）, CLI 互斥校验, 管线集成 + 端到端测试 |
| **Design sections** | ClearVoice Integration: "Stage 4A 目标说话人提取", "CLI 变更" |
| **Delivers** | ClearVoice `AV_MossFormer2_TSE_16K` 目标说话人提取，通过 `--tse` 启用。利用视频人脸信息从重叠语音中提取特定说话人，配合声纹匹配关联说话人标签 |
| **Dependencies** | 无新依赖（复用 clearvoice，TSE 模型已包含在内） |
| **Prerequisite** | Phase 1 + Phase 2 + Phase 3 + Phase 4 |
| **Key changes** | 新增 extractor.py；cli.py 新增 `--tse` 标志（与 `--separate` 互斥）；pipeline.py 新增 TSE 分支 |

---

## Execution Order

```
Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4 ──→ Phase 5
(基础 ASR)   (降噪)      (识别)     (分离+匹配)   (TSE)
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

# Run — full pipeline with overlap separation (ClearVoice SS + speaker matching)
uv run python -m transcribe input.mp4 \
  --denoise --separate \
  --hotwords hotwords/my_dict.txt \
  -o output.srt \
  -v

# Run — with target speaker extraction (ClearVoice TSE, requires video)
uv run python -m transcribe input.mp4 \
  --tse \
  --hotwords hotwords/my_dict.txt \
  -o output.srt \
  -v
```

## CLI Flags Summary

| Flag | Default | Description |
|------|---------|-------------|
| *(none)* | ASR + diarization | Best recognition quality, speaker labels in SRT |
| `--no-diarize` | — | Pure ASR, no speaker labels, faster |
| `--denoise` | off | Enable ClearVoice noise suppression (SNR-gated) |
| `--separate` | off | Enable ClearVoice overlap speech separation + speaker matching |
| `--tse` | off | Enable ClearVoice target speaker extraction (requires video, mutually exclusive with `--separate`) |
| `--num-speakers N` | auto | Hint known speaker count to diarizer |
| `--hotwords FILE` | none | Hotword file for ASR boosting |
| `-v, --verbose` | off | Print per-stage progress and timing |
