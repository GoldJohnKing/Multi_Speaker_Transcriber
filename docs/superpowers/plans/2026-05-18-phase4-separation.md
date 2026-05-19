# Phase 4: 重叠语音分离

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对 Phase 3 检测到的重叠语音区域使用 SepFormer 进行说话人分离，将混合语音拆分为独立说话人音轨后再进行 ASR，提升多人同时说话场景下的识别准确率。

**Architecture:** 新增 `Separator` 类，仅在重叠区域上运行。管线流程变为：降噪 → 说话人识别 → 重叠区域分离 → 按说话人分段 ASR → SRT 生成。

**Tech Stack:** SpeechBrain SepFormer (Apache 2.0), PyTorch

**Design doc reference:** Sections "Stage 4: Speech Separation", "VRAM Management"

**Prerequisite:** Phase 1 + Phase 2 + Phase 3 complete

---

## File Structure

```
# 新增/修改的文件
transcribe/models/separator.py   # 新增
tests/test_separator.py          # 新增
transcribe/pipeline.py           # 修改 — 插入分离阶段
```

---

### Task 1: Separator 模块

**Files:**
- Create: `transcribe/models/separator.py`
- Create: `tests/test_separator.py`

- [ ] **Step 1: 安装语音分离依赖**

```bash
uv sync --extra separate --extra dev
```

- [ ] **Step 2: 编写分离测试**

```python
# tests/test_separator.py
import numpy as np
import pytest

from transcribe.data.types import AudioSegment, DiarizationResult, SpeakerSegment
from transcribe.models.separator import Separator


@pytest.fixture
def overlap_audio():
    """2 seconds of synthetic mixed audio at 16kHz."""
    sr = 16000
    duration = 2.0
    waveform = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1
    return AudioSegment(
        waveform=waveform,
        sample_rate=sr,
        start_time=0.0,
        end_time=duration,
    )


@pytest.fixture
def diarization_with_overlap():
    """Diarization result with overlap regions."""
    return DiarizationResult(
        segments=[
            SpeakerSegment("SPEAKER_00", 0.0, 1.0),
            SpeakerSegment("SPEAKER_01", 0.5, 1.5, is_overlap=True),
            SpeakerSegment("SPEAKER_00", 1.5, 2.0),
        ],
        num_speakers=2,
        overlap_regions=[(0.5, 1.5)],
    )


def test_separator_init():
    sep = Separator(device="cpu")
    assert sep is not None


def test_separator_returns_list(overlap_audio, diarization_with_overlap):
    sep = Separator(device="cpu")
    separated = sep.separate_overlaps(overlap_audio, diarization_with_overlap)
    assert isinstance(separated, list)


def test_separator_output_are_audio_segments(overlap_audio, diarization_with_overlap):
    sep = Separator(device="cpu")
    separated = sep.separate_overlaps(overlap_audio, diarization_with_overlap)
    for seg in separated:
        assert isinstance(seg, AudioSegment)


def test_separator_no_overlap_returns_empty(overlap_audio):
    diarization = DiarizationResult(
        segments=[SpeakerSegment("SPEAKER_00", 0.0, 2.0)],
        num_speakers=1,
        overlap_regions=[],
    )
    sep = Separator(device="cpu")
    separated = sep.separate_overlaps(overlap_audio, diarization)
    assert separated == []


def test_separator_cleanup():
    sep = Separator(device="cpu")
    sep.cleanup()  # should not crash
```

- [ ] **Step 3: 运行测试验证失败**

```bash
uv run pytest tests/test_separator.py -v
```

- [ ] **Step 4: 实现 Separator**

```python
# transcribe/models/separator.py
from __future__ import annotations

import torch
import torchaudio

from transcribe.data.types import AudioSegment, DiarizationResult


class Separator:
    """Speech separation for overlapping regions using SepFormer."""

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "speechbrain/sepformer-whamr16k",
        max_segment_seconds: float = 10.0,
    ) -> None:
        self._device = device
        self._max_segment_seconds = max_segment_seconds
        self._model = self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load SepFormer model from SpeechBrain."""
        from speechbrain.inference.separation import SepformerSeparation

        model = SepformerSeparation.from_hparams(
            source=model_name,
            savedir=f"pretrained_models/{model_name.split('/')[-1]}",
            run_opts={"device": self._device},
        )
        return model

    def separate_overlaps(
        self, audio: AudioSegment, diarization: DiarizationResult
    ) -> list[AudioSegment]:
        """Separate overlapping speech regions into individual speaker tracks.

        Args:
            audio: Full audio segment.
            diarization: Diarization result with overlap regions.

        Returns:
            List of separated AudioSegments, each from an overlap region.
            Non-overlap regions are NOT included (handled directly by ASR).
        """
        if not diarization.overlap_regions:
            return []

        separated_segments: list[AudioSegment] = []

        for overlap_start, overlap_end in diarization.overlap_regions:
            # Clip duration limit
            clip_duration = min(
                overlap_end - overlap_start, self._max_segment_seconds
            )
            actual_end = overlap_start + clip_duration

            # Extract overlap audio chunk
            start_sample = int(
                (overlap_start - audio.start_time) * audio.sample_rate
            )
            end_sample = int(
                (actual_end - audio.start_time) * audio.sample_rate
            )
            chunk = audio.waveform[start_sample:end_sample]

            # Convert to tensor
            wav_tensor = torch.tensor(chunk, dtype=torch.float32)
            if wav_tensor.ndim == 1:
                wav_tensor = wav_tensor.unsqueeze(0)  # add batch dim

            # Run separation
            est_sources = self._model.separate_batch(wav_tensor)

            # est_sources shape: (batch, time, num_speakers)
            separated = est_sources[0]  # (time, num_speakers)

            for spk_idx in range(separated.shape[-1]):
                spk_waveform = separated[:, spk_idx].cpu().numpy()
                spk_waveform = spk_waveform.astype(np.float32)

                separated_segments.append(
                    AudioSegment(
                        waveform=spk_waveform,
                        sample_rate=audio.sample_rate,
                        start_time=overlap_start,
                        end_time=actual_end,
                    )
                )

        return separated_segments

    def cleanup(self) -> None:
        """Release model from memory."""
        del self._model
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
```

- [ ] **Step 5: 运行测试**

```bash
uv run pytest tests/test_separator.py -v
```

Expected: all passed

- [ ] **Step 6: 提交**

```bash
git add transcribe/models/separator.py tests/test_separator.py
git commit -m "feat: add SepFormer speech separation for overlap regions"
```

---

### Task 2: 集成分离到管线

**Files:**
- Modify: `transcribe/pipeline.py`

- [ ] **Step 1: 更新 pipeline.py，在说话人识别后插入分离阶段**

在 Phase 3 的 diarization 之后、ASR 之前，插入 Stage 4 (Speech Separation)：

```python
# Stage 4: Speech separation (overlap regions only)
overlap_separated: dict[tuple[float, float], list[AudioSegment]] = {}
if diarization.overlap_regions:
    step_start = time.time()
    if verbose:
        console.print(
            f"[4/6] 重叠语音分离 ... "
            f"处理 {len(diarization.overlap_regions)} 个重叠片段 ...",
            end=" ",
        )
    separator = Separator(device=device)
    separated_audios = separator.separate_overlaps(audio, diarization)

    # Group separated audio by overlap region
    sep_idx = 0
    for overlap_start, overlap_end in diarization.overlap_regions:
        region_segments = []
        # 2 speakers per overlap region (SepFormer outputs 2 sources)
        for _ in range(2):
            if sep_idx < len(separated_audios):
                region_segments.append(separated_audios[sep_idx])
                sep_idx += 1
        overlap_separated[(overlap_start, overlap_end)] = region_segments

    separator.cleanup()
    if verbose:
        console.print(f"完成 ({time.time() - step_start:.1f}s)")

# Stage 5: ASR per speaker segment
# ... (modify existing ASR loop)
# For overlap regions: use separated audio instead of mixed audio
# For non-overlap regions: use cropped original audio as before
```

修改 ASR 循环逻辑：
- **非重叠区域**：直接裁剪原始音频送入 ASR（同 Phase 3）
- **重叠区域**：使用 `overlap_separated` 中分离后的独立说话人音轨分别送入 ASR
- 重叠区域的 `speaker_id` 通过音轨索引与 Pyannote 标签关联

- [ ] **Step 2: 端到端验证**

```bash
uv run python -m transcribe outdoor_show.mp4 --hotwords hotwords/example.txt -o output.srt -v
```

Expected:
```
[1/6] 提取音频 ... 完成
[2/6] 噪声抑制 ... 完成
[3/6] 说话人识别 ... 检测到 4 位说话人, 12 个重叠区域 ... 完成
[4/6] 重叠语音分离 ... 处理 12 个重叠片段 ... 完成
[5/6] 语音转文字 ... 识别 347 个片段 ... 完成
[6/6] 生成 SRT ... 输出 312 条字幕 ... 完成
```

- [ ] **Step 3: 提交**

```bash
git add transcribe/pipeline.py
git commit -m "feat: integrate speech separation into full pipeline"
```

---

### Task 3: 全管线集成测试

**Files:**
- Create: `tests/test_pipeline_full.py`

- [ ] **Step 1: 编写全管线测试**

```python
# tests/test_pipeline_full.py
"""Full pipeline integration tests (all 6 stages).
Requires all models to be cached locally.
Mark as slow / optional for CI.
"""
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def sample_wav(tmp_path: Path):
    wav_path = tmp_path / "test.wav"
    subprocess.run(
        [
            "ffmpeg", "-f", "lavfi",
            "-i", "sine=frequency=440:duration=2",
            "-ar", "16000", "-ac", "1",
            "-y", str(wav_path),
        ],
        check=True,
        capture_output=True,
    )
    return wav_path


@pytest.mark.slow
def test_full_pipeline_with_all_stages(sample_wav, tmp_path):
    from transcribe.config import PipelineConfig
    from transcribe.pipeline import run_pipeline

    output = str(tmp_path / "output.srt")
    config = PipelineConfig(device="cpu", denoise=False)

    result = run_pipeline(
        input_path=str(sample_wav),
        output_path=output,
        config=config,
    )
    assert Path(result).exists()


@pytest.mark.slow
def test_full_pipeline_with_denoise(sample_wav, tmp_path):
    from transcribe.config import PipelineConfig
    from transcribe.pipeline import run_pipeline

    output = str(tmp_path / "output.srt")
    config = PipelineConfig(device="cpu", denoise=True)

    result = run_pipeline(
        input_path=str(sample_wav),
        output_path=output,
        config=config,
    )
    assert Path(result).exists()
```

- [ ] **Step 2: 运行所有测试**

```bash
uv run pytest tests/ -v -k "not slow"
```

Expected: all unit tests passed

- [ ] **Step 3: 提交**

```bash
git add tests/test_pipeline_full.py
git commit -m "test: add full pipeline integration tests"
```

---

## Phase 4 Deliverable

完成后的新增能力：
- 重叠语音区域自动分离为独立说话人音轨
- 分离后的音轨独立进行 ASR，提升重叠场景识别准确率
- 完整的 6 阶段管线全部就位：音频提取 → 噪声抑制 → 说话人识别 → 重叠分离 → ASR → SRT
- 总显存峰值 ~10GB（模型分时加载），适配 7900XTX 20GB 显存
