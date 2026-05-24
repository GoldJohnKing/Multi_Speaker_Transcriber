# Whisper ASR 后端集成设计

**日期**: 2026-05-24
**状态**: 已批准

## 概述

在现有 3 种 ASR 后端（Fun-ASR-Nano、Fun-ASR-Paraformer、Qwen3-ASR）基础上，集成第四种基于 faster-whisper 的 Whisper 后端，提供多语言识别能力。

## 决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Whisper 实现 | faster-whisper (CTranslate2) | 高效推理、原生 hotwords 参数、无 PyTorch 依赖、社区活跃 |
| 推理模式 | 标准 WhisperModel（非 Batched） | 与现有后端模式一致；项目管线已做分段，无需额外吞吐 |
| 语言支持 | 多语言自动检测 | Whisper 的核心优势在于多语言 |
| 热词机制 | 原生 `hotwords` 参数 | faster-whisper 1.x 原生支持逗号分隔热词，比 `initial_prompt` 效果好 |
| 时间戳粒度 | 段级（segment.start/end） | 用户选择；直接映射到 TranscriptSegment，无需额外分段逻辑 |
| VAD | `vad_filter=True`（Silero VAD） | 内置 VAD 自动分段，与 FunASR 后端的 VAD 模式对齐 |
| 默认模型 | large-v3 | 中文质量最好 |
| compute_type | CPU→int8，GPU→float16 | 自动选择，兼顾速度和精度 |
| 注册名称 | `"Whisper"` | 简洁，与现有后端命名风格一致 |

## 架构

遵循现有 ASR 后端的自注册模式：

```
transcribe/models/asr/
├── whisper.py          ← 新增：WhisperTranscriber
├── __init__.py         ← 改动：import whisper 触发注册
├── factory.py          ← 不变
├── base.py             ← 不变
└── utils.py            ← 不变

pyproject.toml          ← 改动：新增 whisper 依赖组
transcribe/cli.py       ← 改动：--backend choices 加 "Whisper"
```

## 核心类设计

### WhisperTranscriber

```python
class WhisperTranscriber(ASRBase):
    """基于 faster-whisper 的多语言 ASR 后端。"""

    def __init__(
        self,
        device: str = "cpu",
        hotword_path: str | None = None,
        *,
        model_size: str = "large-v3",
        language: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        # 延迟导入 faster-whisper，缺少时给出友好提示
        # 根据 device 自动选择 compute_type
        # 加载热词文件并逗号拼接
        ...

    def transcribe(self, audio: AudioSegment) -> list[TranscriptSegment]:
        # model.transcribe(waveform, hotwords=..., vad_filter=True, language=...)
        # 遍历 segments，映射为 TranscriptSegment
        # 时间戳偏移：+ audio.start_time
        ...

    def cleanup(self) -> None:
        # del self._model; torch.cuda.empty_cache() if GPU
        ...

    @property
    def supports_hotwords(self) -> bool:
        return True
```

### 热词处理

- 读取热词文件（每行一个词，与现有后端格式一致）
- 逗号拼接为 faster-whisper 的 `hotwords` 参数格式：`"词1,词2,词3"`
- `supports_hotwords` 返回 `True`

### 数据流

```
AudioSegment(waveform: np.ndarray, sample_rate, start_time, end_time)
    │
    ▼  model.transcribe(waveform, hotwords=..., vad_filter=True, language=...)
    │
    ▼  Iterable[Segment]  (start, end, text — 秒级时间戳)
    │
    ▼  + audio.start_time 偏移  →  list[TranscriptSegment]
```

faster-whisper 接受 `np.ndarray`（float32, 16kHz mono），与 `AudioSegment.waveform` 格式完全匹配，无需转换。

## 依赖管理

### pyproject.toml 新增

```toml
[project.optional-dependencies]
whisper = [
    "faster-whisper>=1.0",
]
all = [
    "multi-speaker-transcribe[funasr,diarize,qwen-asr,whisper]",
]
```

`faster-whisper` 依赖 CTranslate2 而非 PyTorch，与其他后端的 torch 依赖不冲突。用户可单独安装 `uv sync --extra whisper`，无需 torch。

### 安装命令

```bash
uv sync --extra whisper            # 仅 Whisper 后端
uv sync --extra all                # 全部后端（含 Whisper）
```

## CLI 集成

```python
# cli.py — --backend choices 新增 "Whisper"
parser.add_argument(
    "--backend",
    choices=["Fun-ASR-Paraformer", "Fun-ASR-Nano", "Qwen3-ASR", "Whisper"],
    default="Fun-ASR-Nano",
    help="ASR backend (default: Fun-ASR-Nano)",
)
```

## 错误处理

延迟导入 + 友好提示，与 Qwen3-ASR 后端一致：

```python
try:
    from faster_whisper import WhisperModel
except ImportError:
    raise ImportError(
        "Whisper 后端需要 faster-whisper 包。"
        "请运行: uv sync --extra whisper"
    )
```

## 与现有后端的差异对比

| 维度 | Fun-ASR-Nano | Fun-ASR-Paraformer | Qwen3-ASR | **Whisper (新)** |
|------|-------------|-------------------|-----------|----------------|
| 推理引擎 | FunASR (PyTorch) | FunASR (PyTorch) | qwen-asr (PyTorch) | CTranslate2 |
| 语言 | 中文 | 中文 | 中文/多语言 | **多语言（99种）** |
| 热词机制 | list[str] 解码器偏置 | str 解码器偏置 | str LLM prompt | **str 原生 hotwords** |
| 时间戳 | VAD 分段 | VAD 分段 | ForcedAligner 字符级 | **Silero VAD 段级** |
| 字幕分段 | VAD 分段 | VAD 分段 | segment_by_timestamps | **VAD 分段** |
| 推理后端依赖 | torch | torch | torch + transformers | **CTranslate2（无 torch）** |

## 涉及文件清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `transcribe/models/asr/whisper.py` | 新增 | WhisperTranscriber 实现 |
| `transcribe/models/asr/__init__.py` | 改动 | import whisper 模块触发注册 |
| `transcribe/cli.py` | 改动 | --backend choices 加 "Whisper" |
| `pyproject.toml` | 改动 | 新增 whisper 依赖组，更新 all |
| `AGENTS.md` | 改动 | ASR 后端差异表新增 Whisper 行 |

## 不在范围内

- BatchedInferencePipeline 批处理模式（未来可扩展）
- word-level timestamps（未来可扩展）
- Whisper 模型微调 / LoRA
- 流式推理
