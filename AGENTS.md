# AGENTS.md — 开发注意事项

## 环境与依赖管理

### Python 版本

- 项目要求 Python >= 3.10，开发使用 3.12（见 `.python-version`）
- 版本文件由 [uv](https://docs.astral.sh/uv/) 读取，不要手动删除或修改

### 包管理器：uv

本项目使用 **uv** 而非 pip/poetry 管理依赖。所有安装、运行、测试命令均通过 uv 执行：

```bash
uv sync --extra all          # 安装全部可选依赖
uv run python -m transcribe  # 运行管线
uv run pytest                # 运行测试
```

**不要**使用 `pip install` 直接安装项目依赖，应使用 `uv sync` 或 `uv pip install`。

### 依赖结构

依赖分核心依赖和可选依赖组，定义在 `pyproject.toml`：

| 类型 | 安装命令 | 说明 |
|------|----------|------|
| 核心 | `uv sync` | numpy, pyyaml, rich, soundfile（始终安装） |
| `funasr` | `uv sync --extra funasr` | FunASR, torch, torchaudio, modelscope |
| `qwen-asr` | `uv sync --extra qwen-asr` | qwen-asr, torch, torchaudio, transformers, accelerate |
| `diarize` | `uv sync --extra diarize` | pyannote.audio, scipy, modelscope |
| `all` | `uv sync --extra all` | 以上全部 |
| `dev` | `uv sync --extra dev` | pytest（通常与 `--extra all` 组合） |

### PyTorch 安装

PyTorch（torch/torchaudio）通过 `pyproject.toml` 中的 `[[tool.uv.index]]` 和 `[tool.uv.sources]` 配置自动从 PyTorch 官方 wheel 索引安装，**无需手动 `pip install`**。

当前默认使用 **CPU 版本**（`https://download.pytorch.org/whl/cpu`）。切换到 CUDA：

```toml
# pyproject.toml 中修改索引 URL：
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cu121"   # ← 改为 CUDA
```

修改后执行 `uv lock && uv sync` 即可。

## 测试

```bash
uv run pytest                      # 全部测试
uv run pytest -m "not slow"        # 排除需下载模型的慢测试
uv run pytest tests/test_xxx.py    # 单个文件
```

- 测试标记：`@pytest.mark.slow` — 需要下载模型的集成测试
- 测试目录：`tests/`，与源码目录 `transcribe/` 平级

## 项目结构约定

```
transcribe/
├── __main__.py        # CLI 入口
├── cli.py             # 参数解析（argparse）
├── config.py          # 配置加载（YAML + CLI 合并）
├── pipeline.py        # 管线编排
├── data/types.py      # 核心数据类型（dataclass）
└── models/            # 各阶段模型实现
    └── asr/           # ASR 后端包
        ├── __init__.py         # 注册 + 重导出
        ├── base.py             # ASRBase 抽象基类
        ├── factory.py          # create_asr 工厂函数
        ├── utils.py            # 共享工具（热词修复、时间戳解析、字幕分段）
        ├── funasr_nano.py      # Fun-ASR-Nano 后端
        ├── funasr_paraformer.py # Fun-ASR-Paraformer 后端
        └── qwen3_asr.py       # Qwen3-ASR 后端
```

- 每个模型类实现 `process()` / `extract()` / `transcribe()` 等方法，以及 `cleanup()` 用于释放 GPU 显存
- ASR 后端通过 `create_asr(backend_name, ...)` 工厂函数创建，后端在各自模块中通过 `register_backend()` 自注册
- 新增管线阶段时，同步更新 `pipeline.py`、`cli.py`（如有新参数）、`data/types.py`（如有新数据类型）、`config.yaml`（如有新配置项）
- 新增 ASR 后端时，在 `transcribe/models/asr/` 下新建文件，实现 `ASRBase` 子类并调用 `register_backend()`，然后在 `__init__.py` 中 import 触发注册，最后更新 `cli.py` 的 `--backend` choices
- `config.yaml` 中的顶层字段由 `config.py` 加载，子配置段（`diarizer:`、`matcher:` 等）目前仅为参考，实际参数硬编码在模型类构造函数中

### ASR 后端差异

| 后端 | 模型 | 热词机制 | 时间戳来源 | 字幕分段 |
|------|------|---------|-----------|---------|
| Fun-ASR-Nano | FunAudioLLM/Fun-ASR-Nano-2512 | `hotwords=list[str]` 解码器偏置 | FSMN-VAD 分段 | VAD 分段 |
| Fun-ASR-Paraformer | speech_seaco_paraformer_large | `hotword=str` 解码器偏置 | FSMN-VAD 分段 | VAD 分段 |
| Qwen3-ASR | Qwen3-ASR-1.7B + ForcedAligner-0.6B | `context=str` LLM prompt 偏置 | ForcedAligner 字符级对齐 | 标点 + 时长混合分段 |

- FunASR 后端内置 VAD 自动分段；Qwen3-ASR 无 VAD，使用 `segment_by_timestamps()` 混合分段策略
- Qwen3-ASR 的 `text` 含标点但 `time_stamps` 不含标点，需通过 `_align_text_to_timestamps()` 对齐
- `segment_by_timestamps()` 在句末标点（。！？）处分段并移除标点；逗号级标点保留在字幕行内

## 运行时配置

- 配置优先级：**CLI 参数 > YAML 配置文件 > 代码默认值**
- 默认配置文件：项目根目录 `config.yaml`
- 说话人识别（Pyannote 4.0 Community-1）需要 HuggingFace Token，通过 `HF_TOKEN` 环境变量或 `config.yaml` 的 `diarizer.hf_token` 配置

## 注意事项

- 所有音频处理统一使用 **16kHz 单声道**，不要在模型中添加重采样逻辑
- 各阶段模型通过 `cleanup()` 释放显存，避免同时加载所有模型导致 OOM
- `samples/`、`pretrained_models/`、`checkpoints/` 目录已在 `.gitignore` 中，不要提交大文件
