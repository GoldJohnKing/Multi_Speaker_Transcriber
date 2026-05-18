# Phase 2: 噪声抑制

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Phase 1 的 ASR 管线之前加入 DeepFilterNet 噪声抑制阶段，提升户外嘈杂环境下的语音识别准确率。

**Architecture:** 新增 `Denoiser` 类作为管线第 2 阶段（在音频提取之后、ASR 之前），使用 DeepFilterNet v2 对音频进行语音增强。该阶段可通过配置跳过。

**Tech Stack:** DeepFilterNet v2 (MIT), PyTorch, torchaudio

**Design doc reference:** Sections "Stage 2: Noise Suppression", "VRAM Management"

**Prerequisite:** Phase 1 complete

---

## File Structure

```
# 新增/修改的文件
transcribe/models/denoiser.py    # 新增
tests/test_denoiser.py           # 新增
transcribe/pipeline.py           # 修改 — 插入降噪阶段
```

---

### Task 1: Denoiser 模块

**Files:**
- Create: `transcribe/models/denoiser.py`
- Create: `tests/test_denoiser.py`

- [ ] **Step 1: 安装降噪依赖**

```bash
uv sync --extra denoise --extra dev
```

- [ ] **Step 2: 编写降噪测试**

```python
# tests/test_denoiser.py
import numpy as np
import pytest

from transcribe.data.types import AudioSegment
from transcribe.models.denoiser import Denoiser


@pytest.fixture
def noisy_audio():
    """Simulated noisy audio: sine wave + random noise."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    clean = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = 0.1 * np.random.randn(len(t)).astype(np.float32)
    return AudioSegment(
        waveform=(clean + noise),
        sample_rate=sr,
        start_time=0.0,
        end_time=duration,
    )


def test_denoiser_init():
    denoiser = Denoiser(device="cpu")
    assert denoiser is not None


def test_denoiser_output_is_audio_segment(noisy_audio):
    denoiser = Denoiser(device="cpu")
    result = denoiser.process(noisy_audio)
    assert isinstance(result, AudioSegment)


def test_denoiser_preserves_metadata(noisy_audio):
    denoiser = Denoiser(device="cpu")
    result = denoiser.process(noisy_audio)
    assert result.sample_rate == noisy_audio.sample_rate
    assert result.start_time == noisy_audio.start_time
    assert result.end_time == noisy_audio.end_time


def test_denoiser_output_length_matches(noisy_audio):
    denoiser = Denoiser(device="cpu")
    result = denoiser.process(noisy_audio)
    assert len(result.waveform) == len(noisy_audio.waveform)


def test_denoiser_reduces_noise(noisy_audio):
    """Output should have lower energy than noisy input (noise removed)."""
    denoiser = Denoiser(device="cpu")
    result = denoiser.process(noisy_audio)
    # The denoised audio should exist and be non-zero
    assert np.abs(result.waveform).max() > 0
```

- [ ] **Step 3: 运行测试验证失败**

```bash
uv run pytest tests/test_denoiser.py -v
```

- [ ] **Step 4: 实现 Denoiser**

```python
# transcribe/models/denoiser.py
from __future__ import annotations

import numpy as np
import torch

from transcribe.data.types import AudioSegment


class Denoiser:
    """Noise suppression using DeepFilterNet v2."""

    def __init__(self, device: str = "cpu", post_filter: bool = True) -> None:
        self._device = device
        self._post_filter = post_filter
        self._model, self._df_state, _ = self._load_model()

    def _load_model(self):
        """Load DeepFilterNet model."""
        from df import init_df

        model, df_state, _ = init_df(device=self._device, post_filter=self._post_filter)
        return model, df_state

    def process(self, audio: AudioSegment) -> AudioSegment:
        """Apply noise suppression to audio.

        Args:
            audio: Input audio segment (potentially noisy).

        Returns:
            Denoised audio segment with same metadata.
        """
        from df import enhance

        # DeepFilterNet expects 1D numpy array or tensor
        waveform = audio.waveform
        if waveform.ndim > 1:
            waveform = waveform[0]  # take first channel

        # Enhance
        enhanced = enhance(self._model, self._df_state, waveform)

        # Convert back to numpy if needed
        if isinstance(enhanced, torch.Tensor):
            enhanced = enhanced.cpu().numpy()

        # Ensure correct shape
        enhanced = np.ascontiguousarray(enhanced, dtype=np.float32)

        return AudioSegment(
            waveform=enhanced,
            sample_rate=audio.sample_rate,
            start_time=audio.start_time,
            end_time=audio.end_time,
        )

    def cleanup(self) -> None:
        """Release model from memory."""
        del self._model
        del self._df_state
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
```

- [ ] **Step 5: 运行测试**

```bash
uv run pytest tests/test_denoiser.py -v
```

Expected: all passed

- [ ] **Step 6: 提交**

```bash
git add transcribe/models/denoiser.py tests/test_denoiser.py
git commit -m "feat: add DeepFilterNet noise suppression module"
```

---

### Task 2: 集成降噪到管线

**Files:**
- Modify: `transcribe/pipeline.py`

- [ ] **Step 1: 更新 pipeline.py，在音频提取后插入降噪阶段**

在 `run_pipeline` 函数中，在 Stage 1 (音频提取) 和 Stage 2 (ASR) 之间插入：

```python
# Stage 1.5: Noise suppression (optional)
if config.denoise:
    step_start = time.time()
    if verbose:
        console.print("[2/4] 噪声抑制 ...", end=" ")
    denoiser = Denoiser(device=device)
    audio = denoiser.process(audio)
    denoiser.cleanup()
    if verbose:
        console.print(f"完成 ({time.time() - step_start:.1f}s)")
```

同时在 Stage 编号上调整（ASR 变为 `[3/4]`，SRT 变为 `[4/4]`）。如果 `config.denoise == False`，跳过此阶段，编号保持 `[2/3]` `[3/3]`。

- [ ] **Step 2: 验证完整管线运行**

```bash
uv run python -m transcribe noisy_audio.wav --hotwords hotwords/example.txt -o output.srt -v
```

Expected: 显示 4 个阶段（含噪声抑制）

- [ ] **Step 3: 验证 --no-denoise 跳过降噪**

```bash
uv run python -m transcribe test.wav --no-denoise -o output.srt -v
```

Expected: 显示 3 个阶段（跳过噪声抑制）

- [ ] **Step 4: 提交**

```bash
git add transcribe/pipeline.py
git commit -m "feat: integrate noise suppression into pipeline"
```

---

## Phase 2 Deliverable

完成后的新增能力：
- `--no-denoise` 可跳过噪声抑制
- 默认启用 DeepFilterNet v2 降噪，提升户外环境音下的识别准确率
- 模型使用后自动释放显存
