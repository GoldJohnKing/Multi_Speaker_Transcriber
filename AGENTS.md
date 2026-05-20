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
| `asr` | `uv sync --extra asr` | FunASR, torch, torchaudio, modelscope |
| `diarize` | `uv sync --extra diarize` | pyannote.audio, speechbrain |
| `all` | `uv sync --extra all` | 以上全部 |
| `dev` | `uv sync --extra dev` | pytest（通常与 `--extra all` 组合） |

### PyTorch 安装

PyTorch（torch/torchaudio）需要根据硬件平台单独安装，不在 `uv sync` 中自动处理：

```bash
# NVIDIA CUDA
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# AMD ROCm
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# CPU only
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

修改 pyproject.toml 中 `asr` 组的 torch 版本约束时，注意与上述安装方式保持兼容。

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
```

- 每个模型类实现 `process()` / `extract()` / `transcribe()` 等方法，以及 `cleanup()` 用于释放 GPU 显存
- 新增管线阶段时，同步更新 `pipeline.py`、`cli.py`（如有新参数）、`data/types.py`（如有新数据类型）、`config.yaml`（如有新配置项）
- `config.yaml` 中的顶层字段由 `config.py` 加载，子配置段（`diarizer:`、`matcher:` 等）目前仅为参考，实际参数硬编码在模型类构造函数中

## 运行时配置

- 配置优先级：**CLI 参数 > YAML 配置文件 > 代码默认值**
- 默认配置文件：项目根目录 `config.yaml`
- 说话人识别（Pyannote）需要 HuggingFace Token，通过 `HF_TOKEN` 环境变量或 `config.yaml` 的 `diarizer.hf_token` 配置

## 注意事项

- 所有音频处理统一使用 **16kHz 单声道**，不要在模型中添加重采样逻辑
- 各阶段模型通过 `cleanup()` 释放显存，避免同时加载所有模型导致 OOM
- `samples/`、`pretrained_models/`、`checkpoints/` 目录已在 `.gitignore` 中，不要提交大文件
