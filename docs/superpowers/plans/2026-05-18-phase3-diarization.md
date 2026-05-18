# Phase 3: 说话人识别与标注

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在管线中集成 Pyannote 说话人日记系统，为每段语音标注说话人身份。此阶段不处理重叠语音分离，重叠区域将在 Phase 4 处理。

**Architecture:** 新增 `Diarizer` 类作为管线第 3 阶段（降噪之后、ASR 之前）。Pyannote 输出 `DiarizationResult`，包含说话人分段和重叠区域标记。管线根据说话人分段将音频裁剪后分别送入 ASR，实现说话人级别的转录。

**Tech Stack:** Pyannote Audio 3.1 (MIT), PyTorch, HuggingFace Hub

**Design doc reference:** Sections "Stage 3: Speaker Diarization", "VRAM Management"

**Prerequisite:** Phase 1 + Phase 2 complete

---

## File Structure

```
# 新增/修改的文件
transcribe/models/diarizer.py     # 新增
tests/test_diarizer.py            # 新增
transcribe/pipeline.py            # 修改 — 插入说话人识别阶段，按说话人分段送入 ASR
```

---

### Task 1: Diarizer 模块

**Files:**
- Create: `transcribe/models/diarizer.py`
- Create: `tests/test_diarizer.py`

- [ ] **Step 1: 安装说话人识别依赖**

```bash
uv sync --extra diarize --extra dev
```

- [ ] **Step 2: 编写说话人识别测试**

```python
# tests/test_diarizer.py
import numpy as np
import pytest

from transcribe.data.types import AudioSegment, DiarizationResult, SpeakerSegment
from transcribe.models.diarizer import Diarizer


@pytest.fixture
def multi_speaker_audio():
    """3 seconds of synthetic audio at 16kHz."""
    sr = 16000
    duration = 3.0
    waveform = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1
    return AudioSegment(
        waveform=waveform,
        sample_rate=sr,
        start_time=0.0,
        end_time=duration,
    )


def test_diarizer_init():
    diarizer = Diarizer(device="cpu")
    assert diarizer is not None


def test_diarizer_returns_diarization_result(multi_speaker_audio):
    """Diarizer should return a DiarizationResult."""
    diarizer = Diarizer(device="cpu")
    result = diarizer.process(multi_speaker_audio)
    assert isinstance(result, DiarizationResult)


def test_diarizer_segments_have_speaker_ids(multi_speaker_audio):
    """Each segment should have a valid speaker_id."""
    diarizer = Diarizer(device="cpu")
    result = diarizer.process(multi_speaker_audio)
    for seg in result.segments:
        assert seg.speaker_id.startswith("SPEAKER_")
        assert seg.start_time >= 0
        assert seg.end_time > seg.start_time


def test_diarizer_with_known_num_speakers(multi_speaker_audio):
    """Should accept num_speakers parameter."""
    diarizer = Diarizer(device="cpu", num_speakers=2)
    result = diarizer.process(multi_speaker_audio)
    assert isinstance(result, DiarizationResult)
    assert result.num_speakers <= 2


def test_diarizer_cleanup():
    """Cleanup should not crash."""
    diarizer = Diarizer(device="cpu")
    diarizer.cleanup()
```

- [ ] **Step 3: 运行测试验证失败**

```bash
uv run pytest tests/test_diarizer.py -v
```

- [ ] **Step 4: 实现 Diarizer**

```python
# transcribe/models/diarizer.py
from __future__ import annotations

import torch

from transcribe.data.types import AudioSegment, DiarizationResult, SpeakerSegment


class Diarizer:
    """Speaker diarization using Pyannote Audio 3.1."""

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "pyannote/speaker-diarization-3.1",
        hf_token: str | None = None,
        num_speakers: int | None = None,
    ) -> None:
        self._device = device
        self._num_speakers = num_speakers
        self._pipeline = self._load_pipeline(model_name, hf_token)

    def _load_pipeline(self, model_name: str, hf_token: str | None):
        """Load Pyannote diarization pipeline."""
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            model_name,
            use_auth_token=hf_token,
        )
        if self._device != "cpu" and torch.cuda.is_available():
            pipeline = pipeline.to(torch.device(self._device))
        return pipeline

    def process(self, audio: AudioSegment) -> DiarizationResult:
        """Run speaker diarization on audio.

        Args:
            audio: Input audio segment.

        Returns:
            DiarizationResult with speaker segments and overlap regions.
        """
        # Pyannote expects a dict with "waveform" (tensor) and "sample_rate"
        waveform_tensor = torch.tensor(audio.waveform, dtype=torch.float32)
        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)  # add channel dim

        input_dict = {
            "waveform": waveform_tensor,
            "sample_rate": audio.sample_rate,
        }

        kwargs = {}
        if self._num_speakers is not None:
            kwargs["num_speakers"] = self._num_speakers

        diarization = self._pipeline(input_dict, **kwargs)

        segments: list[SpeakerSegment] = []
        overlap_regions: list[tuple[float, float]] = []
        speaker_set: set[str] = set()

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_set.add(speaker)
            is_overlap = False

            # Check if this turn overlaps with any existing segment
            for existing in segments:
                if (
                    existing.speaker_id != speaker
                    and existing.start_time < turn.end
                    and existing.end_time > turn.start
                ):
                    is_overlap = True
                    overlap_start = max(existing.start_time, turn.start)
                    overlap_end = min(existing.end_time, turn.end)
                    overlap_regions.append((overlap_start, overlap_end))

            segments.append(
                SpeakerSegment(
                    speaker_id=speaker,
                    start_time=turn.start + audio.start_time,
                    end_time=turn.end + audio.start_time,
                    is_overlap=is_overlap,
                )
            )

        return DiarizationResult(
            segments=segments,
            num_speakers=len(speaker_set),
            overlap_regions=overlap_regions,
        )

    def cleanup(self) -> None:
        """Release model from memory."""
        del self._pipeline
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
```

- [ ] **Step 5: 运行测试**

```bash
uv run pytest tests/test_denoiser.py -v
```

注意：首次运行需要从 HuggingFace 下载 Pyannote 模型（需 token）。设置环境变量 `HF_TOKEN` 或在 config.yaml 中配置。

Expected: all passed

- [ ] **Step 6: 提交**

```bash
git add transcribe/models/diarizer.py tests/test_diarizer.py
git commit -m "feat: add Pyannote speaker diarization module"
```

---

### Task 2: 集成说话人识别到管线

**Files:**
- Modify: `transcribe/pipeline.py`

- [ ] **Step 1: 更新 pipeline.py**

核心变化：ASR 阶段不再对整段音频做一次转录，而是按说话人分段逐段转录：

```python
# Stage 3: Speaker diarization
step_start = time.time()
if verbose:
    console.print("[3/5] 说话人识别 ...", end=" ")
diarizer = Diarizer(device=device, num_speakers=config.num_speakers)
diarization = diarizer.process(audio)
diarizer.cleanup()
if verbose:
    console.print(
        f"检测到 {diarization.num_speakers} 位说话人, "
        f"{len(diarization.overlap_regions)} 个重叠区域 ... "
        f"完成 ({time.time() - step_start:.1f}s)"
    )

# Stage 4: ASR per speaker segment
step_start = time.time()
if verbose:
    console.print("[4/5] 语音转文字 ...", end=" ")
transcriber = ASRTranscriber(device=device, hotword_path=config.hotwords)
all_segments: list[TranscriptSegment] = []

for spk_seg in diarization.segments:
    if spk_seg.is_overlap:
        # Overlap regions handled in Phase 4; for now, transcribe as-is
        start_sample = int(spk_seg.start_time * audio.sample_rate)
        end_sample = int(spk_seg.end_time * audio.sample_rate)
        segment_audio = AudioSegment(
            waveform=audio.waveform[start_sample:end_sample],
            sample_rate=audio.sample_rate,
            start_time=spk_seg.start_time,
            end_time=spk_seg.end_time,
        )
    else:
        start_sample = int((spk_seg.start_time - audio.start_time) * audio.sample_rate)
        end_sample = int((spk_seg.end_time - audio.start_time) * audio.sample_rate)
        segment_audio = AudioSegment(
            waveform=audio.waveform[start_sample:end_sample],
            sample_rate=audio.sample_rate,
            start_time=spk_seg.start_time,
            end_time=spk_seg.end_time,
        )

    transcripts = transcriber.transcribe(segment_audio)
    for t in transcripts:
        all_segments.append(
            TranscriptSegment(
                speaker_id=spk_seg.speaker_id,
                start_time=t.start_time,
                end_time=t.end_time,
                text=t.text,
            )
        )

transcriber = None  # release
if verbose:
    console.print(f"识别 {len(all_segments)} 个片段 ... 完成 ({time.time() - step_start:.1f}s)")

# Stage 5: SRT generation (unchanged)
```

- [ ] **Step 2: 端到端验证**

```bash
uv run python -m transcribe multi_speaker_video.mp4 --hotwords hotwords/example.txt -o output.srt -v
```

Expected: SRT 中每个字幕条目带有 `[说话人1]` / `[说话人2]` 标签

- [ ] **Step 3: 提交**

```bash
git add transcribe/pipeline.py
git commit -m "feat: integrate speaker diarization into pipeline"
```

---

## Phase 3 Deliverable

完成后的新增能力：
- 自动检测说话人数量并标注（`[说话人1]`、`[说话人2]`...）
- `--num-speakers N` 可指定已知说话人数量提升精度
- 重叠区域被标记但尚未分离处理（转录质量可能较低）
- 需要 HuggingFace token 下载 Pyannote 模型
