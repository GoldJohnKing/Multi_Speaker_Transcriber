# Multi-Speaker Transcribe

离线中文多人语音转录管线。输入视频/音频文件，输出带说话人标注的 SRT 字幕。

专为户外综艺等复杂音频场景设计——多说话人重叠、背景噪声大、说话人数量不固定。

**特点：**

- 完全离线运行，无需联网
- 全程 16kHz 统一采样率，无重采样损耗
- 说话人声纹参考匹配，通过参考音频自动标注说话人姓名
- 热词增强 + 标点修复，提高领域专有词汇识别率

## 目录

- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [CLI 参数说明](#cli-参数说明)
- [Pipeline 架构](#pipeline-架构)
- [配置文件](#配置文件)
- [项目结构](#项目结构)
- [开源依赖](#开源依赖)
- [说话人参考音频](#说话人参考音频)
- [开发与测试](#开发与测试)

---

## 环境配置

### 前置条件

- Python >= 3.10（推荐 3.12）
- [uv](https://docs.astral.sh/uv/)（Python 包管理器）
- FFmpeg（系统安装，用于音视频提取）
- **GPU 用户：** NVIDIA CUDA 或 AMD ROCm 驱动

### 安装步骤

```bash
# 1. 克隆仓库
git clone <repo-url>
cd Multi_Speaker_Transcribe

# 2. 安装全部依赖（含 ASR、识别、声纹匹配）
uv sync --extra all

# 3a. NVIDIA GPU 用户 — 安装 CUDA 版 PyTorch
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3b. AMD ROCm 用户 — 安装 ROCm 版 PyTorch
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# 3c. 仅 CPU — 安装 CPU 版 PyTorch
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 按需安装

如果只需部分功能，可以按需安装可选依赖组：

| 依赖组 | 命令 | 包含功能 |
|--------|------|----------|
| `asr` | `uv sync --extra asr` | 语音识别（FunASR + PyTorch） |
| `diarize` | `uv sync --extra diarize` | 说话人识别 + 声纹匹配（Pyannote + ModelScope + scipy） |
| `all` | `uv sync --extra all` | 全部功能 |

### Pyannote 说话人识别（HF Token）

说话人识别使用 Pyannote Audio 4.0 Community-1 模型，需要 HuggingFace 访问令牌：

1. 在 [huggingface.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) 接受用户协议
2. 生成 HF Access Token：Settings → Access Tokens → New token
3. 配置方式（二选一）：

```bash
# 方式一：环境变量
export HF_TOKEN="hf_xxxxxxxx"

# 方式二：config.yaml
# diarizer:
#   hf_token: "hf_xxxxxxxx"
```

---

## 快速开始

```bash
# 基础用法 — ASR + 说话人识别（默认模式）
uv run python -m transcribe input.mp4 -o output.srt -v

# 纯 ASR，不识别说话人（更快）
uv run python -m transcribe input.mp4 --no-diarize -o output.srt -v

# 指定说话人数量（提高识别准确性）
uv run python -m transcribe input.mp4 --num-speakers 3 -o output.srt -v

# 使用热词文件
uv run python -m transcribe input.mp4 --hotwords hotwords/example.txt -o output.srt -v

# 使用参考音频自动标注说话人姓名
uv run python -m transcribe input.mp4 --speaker-ref speakers/ -o output.srt -v

# 选择 ASR 后端（默认 Fun-ASR-Nano）
uv run python -m transcribe input.mp4 --backend Fun-ASR-Paraformer -o output.srt -v
```

### 输出示例

```
1
00:00:01,200 --> 00:00:04,500
[张三] 大家好欢迎来到这一期的节目

2
00:00:04,800 --> 00:00:07,200
[李四] 今天我们来聊一个很有意思的话题

3
00:00:07,500 --> 00:00:10,100
[张三] 对，这个话题我之前就一直想讨论
```

---

## CLI 参数说明

```
usage: transcribe [-h] [-o OUTPUT] [--hotwords FILE] [--num-speakers N]
                  [--no-diarize] [--speaker-ref DIR]
                  [--backend {Fun-ASR-Paraformer,Fun-ASR-Nano}]
                  [--device {cpu,cuda,auto}] [--cache-dir DIR]
                  [--config FILE] [--keep-cache] [-v]
                  input

位置参数:
  input                  输入视频或音频文件路径

可选参数:
  -o, --output           输出 SRT 文件路径（默认: 输入文件名.srt）
  --hotwords FILE        热词文件路径（每行一个词）
  --num-speakers N       已知说话人数量（默认自动检测）
  --no-diarize           禁用说话人识别（纯 ASR 模式）
  --speaker-ref DIR      说话人参考音频目录（文件名即说话人名）
  --backend BACKEND      ASR 后端：Fun-ASR-Paraformer / Fun-ASR-Nano（默认 Fun-ASR-Nano）
  --device DEVICE        计算设备：cpu / cuda / auto（默认 auto）
  --cache-dir DIR        中间缓存目录（默认 .cache）
  --config FILE          YAML 配置文件路径
  --keep-cache           保留中间产物
  -v, --verbose          详细输出（显示各阶段进度和耗时）
```

### 标志组合说明

| 标志组合 | 行为 |
|----------|------|
| （无标志） | ASR + 说话人识别，输出 `[说话人N]` 标签 |
| `--no-diarize` | 纯 ASR，无说话人标签，速度最快 |
| `--speaker-ref DIR` | 说话人识别后，通过参考音频将 `[说话人N]` 替换为实际姓名 |
| `--speaker-ref` + `--no-diarize` | 参考匹配跳过（需说话人识别） |

---

## Pipeline 架构

```
输入文件 (视频/音频)
     │
     ▼
┌─────────────────────────────────┐
│  Stage 1: 音频提取 (FFmpeg)      │  视频/音频 → 16kHz 单声道 WAV
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 2: 说话人识别            │  Pyannote 4.0 Community-1
│  (默认启用, --no-diarize 关闭)  │  输出: 说话人片段 + 重叠区域
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 2.5: 说话人参考匹配     │  ERes2NetV2 (3D-Speaker)
│  (--speaker-ref DIR)            │  多段加权嵌入 + 匈牙利算法匹配
│  将 SPEAKER_XX 映射为实际姓名   │  需要 Stage 2 已启用
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 3: 语音识别 (ASR)       │  Fun-ASR-Nano (默认)
│  热词增强 + 标点修复           │  或 Fun-ASR-Paraformer
│  --backend 选择后端           │  + FSMN-VAD
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 4: SRT 生成             │  合并相邻片段 → 输出 SRT
│  说话人标签 + 时间戳对齐       │
└─────────────────────────────────┘
```

### 数据流

```
AudioSegment ──→ DiarizationResult ──→ (--speaker-ref)
                 (SpeakerSegment[])     SPEAKER_XX → 实际姓名
                        │
                        ▼
               TranscriptSegment[]
                        │
                        ▼
                     SRT 文件
```

### 各阶段详解

#### Stage 1 — 音频提取
使用 FFmpeg 将任意格式的视频/音频转为 16kHz 单声道 float32 WAV。支持 MP4、MKV、AVI、MP3、WAV 等常见格式。

#### Stage 2 — 说话人识别
使用 Pyannote Audio 4.0 Community-1 模型进行说话人分割（CC-BY-4.0 许可）。输出每个说话人的时间片段（`SpeakerSegment`）及重叠区域列表（`overlap_regions`）。支持 `--num-speakers` 提示已知说话人数量以提高准确性。

#### Stage 2.5 — 说话人参考匹配（可选，`--speaker-ref DIR`）
当提供说话人参考音频目录时，使用 ERes2NetV2（3D-Speaker）提取 192 维声纹嵌入。对每个说话人，从最长 3 个非重叠片段中提取嵌入并计算时长加权平均值，然后通过匈牙利算法（`scipy.optimize.linear_sum_assignment`）进行全局最优 1:1 匹配，将匿名说话人标签（`SPEAKER_00` 等）关联到参考音频文件名（即说话人姓名）。例如目录下放置 `张三.wav`、`李四.wav`，输出中将显示 `[张三]`、`[李四]` 而非 `[说话人1]`、`[说话人2]`。此阶段需要说话人识别（Stage 2）已启用。

#### Stage 3 — 语音识别

支持两种 ASR 后端，通过 `--backend` 参数选择：

**Fun-ASR-Nano（默认）** — 基于 LLM 的语音识别模型，标点由 LLM 原生生成，无需单独标点模型。支持热词增强，并内置热词标点修复逻辑。GPU 自动启用 BF16（Ampere+）。

**Fun-ASR-Paraformer** — 经典 SeACo-Paraformer 非自回归模型，配合 ct-punc 标点恢复和热词增强。识别速度更快，但标点质量略低于 LLM 方案。

两种后端均内置 FSMN-VAD 语音活动检测。在说话人识别模式下，每个说话人片段独立送入 ASR；在 `--no-diarize` 模式下，整段音频作为单一说话人转录。

#### Stage 4 — SRT 生成
将转录片段排序、合并相邻同说话人片段，生成标准 SRT 字幕文件。说话人标签格式默认为 `[说话人1]`、`[说话人2]` 等；若提供了 `--speaker-ref` 参考音频，则替换为实际姓名（如 `[张三]`、`[李四]`）。

---

## 配置文件

项目根目录的 `config.yaml` 为默认配置文件：

```yaml
device: auto              # 计算设备: auto / cpu / cuda
backend: Fun-ASR-Nano     # ASR 后端: Fun-ASR-Paraformer / Fun-ASR-Nano
diarize: true             # 说话人识别
language: zh              # 语言（当前仅支持中文）
speaker_references: null  # 说话人参考音频目录路径

diarizer:
  model: pyannote/speaker-diarization-community-1
  hf_token: null          # HuggingFace token（或使用 HF_TOKEN 环境变量）
  clustering: hidden_markov

matcher:
  embedding_model: iic/speech_eres2netv2_sv_zh-cn_16k-common
  match_threshold: 0.5    # 余弦相似度匹配阈值
  min_segment_seconds: 0.5

asr:
  model: FunAudioLLM/Fun-ASR-Nano-2512
  vad_model: fsmn-vad
  vad_max_single_segment_time: 30000

srt:
  max_chars_per_line: 20
  min_duration: 1.0       # 最短字幕时长 (秒)
  merge_gap: 0.5          # 相邻片段合并间隔 (秒)
  speaker_label: true     # 说话人标签前缀
```

配置优先级：**CLI 参数 > YAML 配置文件 > 代码默认值**。

> **注意：** `config.yaml` 中的顶层字段（`device`、`backend`、`diarize` 等）会在运行时加载并生效；下方的子配置段（`diarizer:`、`matcher:` 等）目前仅作参数参考，实际模型参数硬编码在各模型类的构造函数中。

---

## 项目结构

```
Multi_Speaker_Transcribe/
├── config.yaml                        # 默认配置文件
├── pyproject.toml                     # 项目元数据和依赖
├── .python-version                    # Python 版本 (3.12)
├── transcribe/
│   ├── __init__.py
│   ├── __main__.py                    # CLI 入口点
│   ├── cli.py                         # 参数解析
│   ├── config.py                      # 配置加载（YAML + CLI 合并）
│   ├── pipeline.py                    # 管线编排器
│   ├── data/
│   │   ├── types.py                   # 核心数据类型定义
│   │   └── __init__.py
│   └── models/
│       ├── audio_extractor.py         # Stage 1: FFmpeg 音频提取
│       ├── diarizer.py                # Stage 2: Pyannote 说话人识别
│       ├── matcher.py                 # Stage 2.5: 声纹嵌入匹配
│       ├── asr/                       # Stage 3: ASR 后端包
│       │   ├── __init__.py            #   注册 + 重导出
│       │   ├── base.py                #   ASRBase 抽象基类
│       │   ├── factory.py             #   create_asr 工厂函数
│       │   ├── utils.py               #   共享工具（热词修复、时间戳解析）
│       │   ├── nano.py                #   Fun-ASR-Nano 后端
│       │   └── paraformer.py          #   Fun-ASR-Paraformer 后端
│       ├── srt_writer.py             # Stage 4: SRT 生成
│       └── __init__.py
├── tests/                             # 测试目录
├── hotwords/                          # 热词文件目录
│   └── example.txt                    # 示例热词文件
├── samples/                           # 测试样本（gitignore）
└── checkpoints/                       # 预训练模型权重（gitignore）
```

### 核心数据类型

| 类型 | 用途 |
|------|------|
| `AudioSegment` | 音频片段（waveform + 采样率 + 时间范围） |
| `SpeakerSegment` | 说话人时间片段（说话人 ID + 起止时间 + 是否重叠） |
| `DiarizationResult` | 说话人识别结果（片段列表 + 说话人数量 + 重叠区域） |
| `TranscriptSegment` | 转录片段（说话人 ID + 起止时间 + 文本） |
| `PipelineConfig` | 管线配置（含 `backend` 后端选择、`speaker_references` 参考音频目录） |

---

## 说话人参考音频

`--speaker-ref` 参数允许你提供一组已知说话人的参考音频，自动将匿名标签（`说话人1`、`说话人2`）替换为实际姓名。

### 使用方法

1. 创建目录，放入参考音频文件（WAV 格式），**文件名（不含扩展名）即为说话人姓名**：

```
speakers/
├── 张三.wav
├── 李四.wav
└── 王五.wav
```

2. 运行管线时指定参考音频目录：

```bash
uv run python -m transcribe input.mp4 --speaker-ref speakers/ -o output.srt -v
```

### 工作原理

- 使用 ERes2NetV2（3D-Speaker / ModelScope）为每段参考音频提取 192 维声纹嵌入。该模型在 20 万中文说话人数据上训练，CN-Celeb EER 达 6.14%
- 对说话人识别产生的每个说话人，从最长 3 个非重叠片段中提取嵌入并计算时长加权平均
- 通过匈牙利算法进行全局最优 1:1 匹配，余弦相似度低于阈值（默认 0.5）的配对将被拒绝
- 需要说话人识别（Stage 2）已启用；若使用 `--no-diarize`，参考匹配将跳过并发出警告

---

## 开源依赖

本项目使用以下开源项目：

| 项目 | 用途 | 许可证 |
|------|------|--------|
| [PyTorch](https://pytorch.org/) | 深度学习框架 | BSD-3-Clause |
| [FunASR](https://github.com/modelscope/FunASR) | 中文语音识别（Fun-ASR-Nano / SeACo-Paraformer + VAD） | MIT |
| [Pyannote Audio](https://github.com/pyannote/pyannote-audio) | 说话人识别（Speaker Diarization 4.0 Community-1） | MIT |
| [3D-Speaker](https://github.com/modelscope/3D-Speaker) | 声纹嵌入提取（ERes2NetV2 中文模型） | Apache-2.0 |
| [FFmpeg](https://ffmpeg.org/) | 音视频格式转换 | LGPL / GPL |
| [NumPy](https://numpy.org/) | 数值计算 | BSD-3-Clause |
| [Rich](https://github.com/Textualize/rich) | 终端彩色输出 | MIT |
| [PyYAML](https://github.com/yaml/pyyaml) | YAML 配置文件解析 | MIT |
| [SoundFile](https://github.com/bastibe/python-soundfile) | WAV 音频读写 | BSD-3-Clause |
| [SciPy](https://scipy.org/) | 匈牙利算法最优匹配 | BSD-3-Clause |
| [ModelScope](https://github.com/modelscope/modelscope) | 模型下载与管理 | Apache-2.0 |

---

## 开发与测试

```bash
# 安装开发依赖
uv sync --extra all --extra dev

# 运行全部测试
uv run pytest

# 运行快速测试（排除需要下载模型的慢测试）
uv run pytest -m "not slow"

# 运行特定模块测试
uv run pytest tests/test_srt_writer.py -v
```

### 测试文件

| 测试文件 | 覆盖范围 |
|----------|----------|
| `test_types.py` | 核心数据类型（5 个 dataclass） |
| `test_config.py` | 配置加载（默认值、YAML、CLI 覆盖） |
| `test_audio_extractor.py` | FFmpeg 音频提取 |
| `test_srt_writer.py` | SRT 生成（时间戳、标签、合并、排序） |
| `test_asr.py` | 双后端 ASR + 工厂 + 热词修复 + 时间戳解析 |
| `test_diarizer.py` | 说话人识别（mock） |
| `test_matcher.py` | 声纹匹配（余弦相似度、参考匹配） |
| `test_pipeline_basic.py` | CLI 解析 + 基础管线 |
| `test_pipeline_full.py` | 完整管线集成测试 |

### 测试标记

- `@pytest.mark.slow` — 需要下载模型的慢速集成测试

---

## 硬件要求

| 配置 | VRAM | 说明 |
|------|------|------|
| 最低 | 无（CPU） | 可运行，速度较慢 |
| 推荐 | 8GB+ GPU | 流畅运行全部功能 |

### VRAM 占用估算

| 阶段 | 模型 | 预估显存 |
|------|------|----------|
| 说话人识别 | Pyannote 4.0 | ~2-3 GB |
| 声纹匹配 | ERes2NetV2 | ~0.5 GB |
| ASR | Fun-ASR-Nano (默认) | ~1-2 GB |
| ASR | Fun-ASR-Paraformer | ~1-2 GB |

管线在各阶段之间释放模型显存（`cleanup()`），避免同时加载所有模型。

---

## License

待定。
