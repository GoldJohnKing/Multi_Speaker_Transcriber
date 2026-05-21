# Qwen3-ASR Backend Integration — Design Spec

**Date**: 2026-05-21
**Scope**: Integrate Qwen3-ASR-1.7B (with ForcedAligner-0.6B) as a third ASR backend

## Background

The project currently supports two FunASR-based ASR backends (Fun-ASR-Paraformer and Fun-ASR-Nano). Qwen3-ASR is a newer LLM-based ASR model from Alibaba (released Jan 2026, Apache 2.0) that achieves SOTA accuracy on Chinese benchmarks (AISHELL-2: 2.71% CER vs Paraformer's 2.85%) and supports 30+ languages plus 22 Chinese dialects.

This spec designs its integration following the existing abstract base class + factory + self-registration pattern.

## Approach

**Single unified backend class** encapsulating both Qwen3-ASR-1.7B and Qwen3-ForcedAligner-0.6B. The pipeline sees a standard `ASRBase` subclass — no pipeline modifications needed.

## Model Architecture

Qwen3-ASR uses a two-stage approach:

1. **Qwen3-ASR-1.7B** — Audio encoder + LLM that produces transcript text + detected language
2. **Qwen3-ForcedAligner-0.6B** — Takes ASR output + original audio, produces character-level timestamps (42.9ms avg accuracy)

Both models are loaded via `Qwen3ASRModel.from_pretrained(forced_aligner=...)` in a single call.

## Directory Structure

```
transcribe/models/asr/
├── __init__.py           # +1 行 import qwen3_asr
├── base.py               # 不变
├── factory.py            # 不变
├── utils.py              # 不变
├── paraformer.py         # 不变
├── nano.py               # 不变
└── qwen3_asr.py          # 🆕 新增
```

## Qwen3ASRTranscriber Class

```python
# qwen3_asr.py
import numpy as np
import torch
from transcribe.data.types import AudioSegment, TranscriptSegment
from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.factory import register_backend


class Qwen3ASRTranscriber(ASRBase):
    def __init__(self, device="cpu", hotword_path=None, *,
                 asr_model="Qwen/Qwen3-ASR-1.7B",
                 aligner_model="Qwen/Qwen3-ForcedAligner-0.6B",
                 language=None):
        # 延迟导入，避免未安装 qwen-asr 时 import 整个 asr 包报错
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            raise ImportError(
                "Qwen3-ASR 后端需要 qwen-asr 包。"
                "请运行: uv sync --extra qwen-asr"
            )

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
        results = self._model.transcribe(
            audio=(audio.waveform, audio.sample_rate),
            context=self._context,
            language=self._language,
            return_time_stamps=True,
        )

        if not results or not results[0].text:
            return []

        r = results[0]
        text = r.text

        # 从字符级时间戳中提取段边界
        start_time = audio.start_time
        end_time = audio.end_time
        if r.time_stamps:
            ts_first = r.time_stamps[0]
            ts_last = r.time_stamps[-1]
            start_time = audio.start_time + ts_first.start_time
            end_time = audio.start_time + ts_last.end_time

        return [TranscriptSegment(
            speaker_id="SPEAKER_00",
            start_time=start_time,
            end_time=end_time,
            text=text,
        )]

    def cleanup(self) -> None:
        del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_context(self, path: str | None) -> str:
        """读取热词文件，空格分隔后作为 Qwen3-ASR context 字符串。"""
        if not path:
            return ""
        with open(path) as f:
            terms = [line.strip() for line in f if line.strip()]
        return " ".join(terms)


register_backend("Qwen3-ASR", Qwen3ASRTranscriber)
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 音频直接传 `(np.ndarray, sr)` 元组 | 与现有后端一致无文件 I/O；`qwen-asr` 原生支持 |
| `supports_hotwords = True` | 通过 `context` 参数实现，对管线透明 |
| 热词 → context 直接空格 join | 官方示例用法；模型训练为将 system prompt token 作为背景知识，无需额外提示语 |
| 延迟导入 `qwen_asr` | 未安装 `qwen-asr` 时 `import transcribe.models.asr` 不会报错 |
| 时间戳：首/尾字符级 → 段边界 | 管线按 diarization 段切分传入，每个片段较短，首尾映射足够 |

## Hotwords / Context Biasing

Qwen3-ASR 不使用 FunASR 风格的加权热词解码，而是通过 LLM system prompt 注入背景知识：

| 对比 | FunASR Nano | FunASR Paraformer | Qwen3-ASR |
|------|-------------|-------------------|-----------|
| 参数名 | `hotwords` | `hotword` | `context` |
| 格式 | `list[str]` | 空格 join 的 `str` | 空格 join 的 `str` |
| 机制 | 加权解码器偏置 | 加权解码器偏置 | LLM prompt 背景知识 |
| 最大容量 | 数百词 | 数百词 | ~10,000 tokens |

用户侧完全透明：同一个热词文件，三个后端各用各自的方式消费。

## Files to Modify

| File | Change |
|------|--------|
| `transcribe/models/asr/qwen3_asr.py` | 🆕 新增 — Qwen3ASRTranscriber 类 |
| `transcribe/models/asr/__init__.py` | +1 行 `from transcribe.models.asr import qwen3_asr` |
| `transcribe/cli.py` | `--backend` choices 添加 `"Qwen3-ASR"` |
| `pyproject.toml` | 新增 `qwen-asr` 依赖组 |
| `tests/test_qwen3_asr.py` | 🆕 新增 — 单元 + 集成测试 |

`pipeline.py`、`config.py`、`data/types.py` 无需修改。

## Dependency Management

```toml
[project.optional-dependencies]
# 现有 asr 组不变（funasr）
asr = [
    "funasr @ git+https://github.com/modelscope/FunASR.git@2ca745e5d11ad9650b94691d0d346d1435dc9b63",
    "modelscope",
    "tiktoken",
    "torch>=2.1",
    "torchaudio>=2.1",
    "transformers",
]

# 🆕 独立依赖组
qwen-asr = [
    "qwen-asr",
    "torch>=2.1",
    "torchaudio>=2.1",
    "transformers>=4.45",
    "accelerate",
    "librosa",
    "soundfile",
]

# 更新 all 组
all = [
    "Multi_Speaker_Transcribe[asr,diarize,qwen-asr]",
]
```

`qwen-asr` 作为独立依赖组，不放入现有 `asr` 组，避免与 `funasr` 的 `transformers` 版本冲突。

安装方式：
- `uv sync --extra asr` — FunASR 后端
- `uv sync --extra qwen-asr` — Qwen3-ASR 后端
- `uv sync --extra all` — 全部后端

## Testing

新增 `tests/test_qwen3_asr.py`：

| Test | Marker | Description |
|------|--------|-------------|
| `test_register_backend` | 无 | 验证 `"Qwen3-ASR"` 已注册到 factory |
| `test_supports_hotwords_true` | 无 | 验证 `supports_hotwords` 返回 `True` |
| `test_import_error_message` | 无 | mock `qwen_asr` 不存在时，实例化报错含安装提示 |
| `test_load_context` | 无 | 验证热词文件读取 + 空格 join |
| `test_transcribe` | `@pytest.mark.slow` | 集成测试：加载模型 + 转写短音频 |
| `test_cleanup` | `@pytest.mark.slow` | 验证 cleanup 后显存释放 |

## VRAM Considerations

| Component | VRAM |
|-----------|------|
| Qwen3-ASR-1.7B (BF16) | ~5 GB |
| Qwen3-ForcedAligner-0.6B (BF16) | ~2 GB |
| **Total (同时加载)** | **~7 GB** |

与现有后端对比：
- Fun-ASR-Paraformer: ~1.5 GB
- Fun-ASR-Nano: ~2 GB

`cleanup()` 在管线中按阶段释放显存（ASR → diarization → matcher），与其他后端行为一致。Qwen3-ASR 的 `cleanup()` 一次性释放 ASR + ForcedAligner 两个模型。

## CLI Usage

```bash
# 使用 Qwen3-ASR 后端
uv run python -m transcribe audio.wav --backend Qwen3-ASR

# 使用 Qwen3-ASR + 热词
uv run python -m transcribe audio.wav --backend Qwen3-ASR --hotwords hotwords.txt

# 无 diarization（仅转写）
uv run python -m transcribe audio.wav --backend Qwen3-ASR --no-diarize
```
