# Multi-Speaker Transcribe

离线中文多人语音转录管线。输入视频/音频文件，输出带说话人标注的 SRT 字幕。

专为户外综艺等复杂音频场景设计——多说话人重叠、背景噪声大、说话人数量不固定。

**特点：**

- 完全离线运行，无需联网
- 全程 16kHz 统一采样率，无重采样损耗
- SNR 自适应降噪门控（音频干净时自动跳过）
- 支持盲源分离（ClearVoice SS）和目标说话人提取（ClearVoice TSE）
- 声纹嵌入余弦相似度匹配，分离音轨自动关联说话人标签
- 热词增强 + 标点修复，提高领域专有词汇识别率
- AMD ROCm 兼容（自动修补 ClearVoice GPU 检测）

## 目录

- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [CLI 参数说明](#cli-参数说明)
- [Pipeline 架构](#pipeline-架构)
- [配置文件](#配置文件)
- [项目结构](#项目结构)
- [开源依赖](#开源依赖)
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

# 2. 安装全部依赖（含 ASR、降噪、识别、分离、TSE）
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
| `denoise` | `uv sync --extra denoise` | 噪声抑制（ClearVoice） |
| `diarize` | `uv sync --extra diarize` | 说话人识别（Pyannote） |
| `separate` | `uv sync --extra separate` | 重叠语音分离（ClearVoice SS） |
| `tse` | `uv sync --extra tse` | 目标说话人提取（ClearVoice TSE + 视觉） |
| `all` | `uv sync --extra all` | 全部功能 |

### Pyannote 说话人识别（HF Token）

说话人识别使用 Pyannote Audio 3.1，需要 HuggingFace 访问令牌：

1. 在 [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) 接受用户协议
2. 在 [huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) 接受用户协议
3. 生成 HF Access Token：Settings → Access Tokens → New token
4. 配置方式（二选一）：

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

# 开启噪声抑制
uv run python -m transcribe input.mp4 --denoise -o output.srt -v

# 完整管线 — 降噪 + 重叠分离 + 声纹匹配
uv run python -m transcribe input.mp4 --denoise --separate -o output.srt -v

# 目标说话人提取（需要视频文件，利用人脸信息）
uv run python -m transcribe input.mp4 --tse -o output.srt -v

# 指定说话人数量（提高识别准确性）
uv run python -m transcribe input.mp4 --num-speakers 3 -o output.srt -v

# 使用热词文件
uv run python -m transcribe input.mp4 --hotwords hotwords/my_dict.txt -o output.srt -v
```

### 输出示例

```
1
00:00:01,200 --> 00:00:04,500
[说话人1] 大家好欢迎来到这一期的节目

2
00:00:04,800 --> 00:00:07,200
[说话人2] 今天我们来聊一个很有意思的话题

3
00:00:07,500 --> 00:00:10,100
[说话人1] 对，这个话题我之前就一直想讨论
```

---

## CLI 参数说明

```
usage: transcribe [-h] [-o OUTPUT] [--hotwords FILE] [--num-speakers N]
                  [--denoise] [--no-diarize] [--separate] [--tse]
                  [--device {cpu,cuda,auto}] [--cache-dir DIR]
                  [--config FILE] [--keep-cache] [-v]
                  input

位置参数:
  input                  输入视频或音频文件路径

可选参数:
  -o, --output           输出 SRT 文件路径（默认: 输入文件名.srt）
  --hotwords FILE        热词文件路径（每行一个词）
  --num-speakers N       已知说话人数量（默认自动检测）
  --denoise              启用噪声抑制（SNR 门控自适应）
  --no-diarize           禁用说话人识别（纯 ASR 模式）
  --separate             启用重叠语音分离 + 声纹匹配
  --tse                  启用目标说话人提取（需视频输入，与 --separate 互斥）
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
| `--denoise` | 噪声抑制 → ASR + 说话人识别 |
| `--denoise --separate` | 降噪 → 识别 → 重叠分离 → 声纹匹配 → ASR |
| `--tse` | 降噪 → 识别 → 人脸追踪 TSE → 声纹匹配 → ASR |
| `--tse` + 音频文件 | 报错（TSE 需要视频文件） |
| `--tse --separate` | 报错（两者互斥） |

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
│  Stage 2: 噪声抑制 (--denoise)  │  ClearVoice MossFormerGAN_SE_16K
│  SNR < 25dB 时启用              │  SNR 门控：音频干净则自动跳过
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 3: 说话人识别            │  Pyannote 3.1
│  (默认启用, --no-diarize 关闭)  │  输出: 说话人片段 + 重叠区域
└───────────────┬─────────────────┘
                │
          ┌─────┴─────────────┐
          │                   │
          ▼                   ▼
┌──────────────────┐  ┌──────────────────┐
│  Stage 4:        │  │  Stage 4A:       │
│  重叠语音分离    │  │  目标说话人提取  │
│  (--separate)    │  │  (--tse)         │
│  ClearVoice SS   │  │  ClearVoice TSE  │
│  MossFormer2     │  │  AV-MossFormer2  │
│  _SS_16K         │  │  _TSE_16K        │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └──────┬──────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 4.5: 声纹匹配           │  SpeechBrain ECAPA-TDNN
│  (分离/TSE 模式自动启用)       │  192维嵌入 → 余弦相似度
│                                 │  分离音轨 → 说话人标签
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 5: 语音识别 (ASR)       │  FunASR SeACo-Paraformer
│  热词增强 + 标点修复           │  + FSMN-VAD + ct-punc
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 6: SRT 生成             │  合并相邻片段 → 输出 SRT
│  说话人标签 + 时间戳对齐       │
└─────────────────────────────────┘
```

### 数据流

```
AudioSegment ──→ (Denoiser) ──→ AudioSegment
                                       │
                               ┌───────┴───────┐
                               ▼               ▼
                    DiarizationResult    (Separator/TSE)
                    (SpeakerSegment[])   AudioSegment[]
                               │               │
                               └───────┬───────┘
                                       ▼
                              (SpeakerMatcher)
                              track_idx → speaker_id
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

#### Stage 2 — 噪声抑制（可选）
使用 ClearVoice 的 `MossFormerGAN_SE_16K` 语音增强模型。内置 SNR（信噪比）估计器，仅在 SNR < 25dB 时启动降噪，避免对干净音频的过度处理。16kHz 原生处理，无需重采样。

#### Stage 3 — 说话人识别
使用 Pyannote Audio 3.1 进行说话人分割。输出每个说话人的时间片段（`SpeakerSegment`）及重叠区域列表（`overlap_regions`）。支持 `--num-speakers` 提示已知说话人数量以提高准确性。

#### Stage 4 — 重叠语音分离（可选，`--separate`）
对说话人识别检测到的重叠区域，使用 ClearVoice `MossFormer2_SS_16K` 盲源分离模型将混合语音拆分为独立说话人音轨。

#### Stage 4A — 目标说话人提取（可选，`--tse`）
利用视频的人脸追踪信息，使用 ClearVoice `AV_MossFormer2_TSE_16K` 音视频联合模型，从混合语音中提取特定说话人的音频。需要视频文件输入（非纯音频）。

#### Stage 4.5 — 声纹匹配
分离/TSE 模式自动启用。使用 SpeechBrain ECAPA-TDNN 提取 192 维声纹嵌入，通过余弦相似度将匿名分离音轨关联到说话人识别产生的说话人标签。

#### Stage 5 — 语音识别
使用 FunASR SeACo-Paraformer 中文语音识别模型，配合 FSMN-VAD 语音活动检测和 ct-punc 标点恢复。支持热词文件增强领域词汇识别率，并内置热词标点修复逻辑——ct-punc 可能会在热词内部插入标点（如"朽，叶"→"朽叶"），此模块自动修复。

#### Stage 6 — SRT 生成
将转录片段排序、合并相邻同说话人片段，生成标准 SRT 字幕文件。说话人标签格式为 `[说话人1]`、`[说话人2]` 等。

---

## 配置文件

项目根目录的 `config.yaml` 为默认配置文件：

```yaml
device: auto              # 计算设备: auto / cpu / cuda
denoise: false            # 噪声抑制
diarize: true             # 说话人识别
separate: false           # 重叠语音分离
tse: false                # 目标说话人提取
language: zh              # 语言（当前仅支持中文）

denoiser:
  model: MossFormerGAN_SE_16K
  snr_threshold: 25       # SNR 门限 (dB)

diarizer:
  model: pyannote/speaker-diarization-3.1
  hf_token: null          # HuggingFace token（或使用 HF_TOKEN 环境变量）
  clustering: hidden_markov

separator:
  model: MossFormer2_SS_16K
  max_segment_seconds: 10 # 单次分离最大时长

extractor:
  model: AV_MossFormer2_TSE_16K
  min_track_frames: 50    # 最小人脸轨迹帧数

matcher:
  embedding_model: speechbrain/spkrec-ecapa-voxceleb
  match_threshold: 0.5    # 余弦相似度匹配阈值
  min_segment_seconds: 0.5

asr:
  model: paraformer-zh
  vad_model: fsmn-vad
  punc_model: ct-punc
  batch_size_s: 300

srt:
  max_chars_per_line: 20
  min_duration: 1.0       # 最短字幕时长 (秒)
  merge_gap: 0.5          # 相邻片段合并间隔 (秒)
  speaker_label: true     # 说话人标签前缀
```

配置优先级：**CLI 参数 > YAML 配置文件 > 代码默认值**。

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
│       ├── denoiser.py                # Stage 2: ClearVoice 噪声抑制
│       ├── diarizer.py                # Stage 3: Pyannote 说话人识别
│       ├── separator.py              # Stage 4: ClearVoice 重叠分离
│       ├── extractor.py              # Stage 4A: ClearVoice TSE 提取
│       ├── matcher.py                # Stage 4.5: 声纹嵌入匹配
│       ├── asr.py                    # Stage 5: FunASR 语音识别
│       ├── srt_writer.py             # Stage 6: SRT 生成
│       ├── rocm_compat.py            # AMD ROCm 兼容补丁
│       └── __init__.py
├── tests/                             # 测试目录
├── hotwords/                          # 热词文件目录
├── docs/                              # 设计文档
└── samples/                           # 测试样本（gitignore）
```

### 核心数据类型

| 类型 | 用途 |
|------|------|
| `AudioSegment` | 音频片段（waveform + 采样率 + 时间范围） |
| `SpeakerSegment` | 说话人时间片段（说话人 ID + 起止时间 + 是否重叠） |
| `DiarizationResult` | 说话人识别结果（片段列表 + 说话人数量 + 重叠区域） |
| `TranscriptSegment` | 转录片段（说话人 ID + 起止时间 + 文本） |
| `PipelineConfig` | 管线配置 |

---

## 开源依赖

本项目使用以下开源项目：

| 项目 | 用途 | 许可证 |
|------|------|--------|
| [PyTorch](https://pytorch.org/) | 深度学习框架 | BSD-3-Clause |
| [FunASR](https://github.com/modelscope/FunASR) | 中文语音识别（SeACo-Paraformer + VAD + 标点） | MIT |
| [ClearVoice](https://github.com/modelscope/ClearerVoice-Studio) | 噪声抑制、语音分离、目标说话人提取 | Apache-2.0 |
| [Pyannote Audio](https://github.com/pyannote/pyannote-audio) | 说话人识别（Speaker Diarization 3.1） | MIT |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) | 声纹嵌入提取（ECAPA-TDNN） | Apache-2.0 |
| [FFmpeg](https://ffmpeg.org/) | 音视频格式转换 | LGPL / GPL |
| [NumPy](https://numpy.org/) | 数值计算 | BSD-3-Clause |
| [Rich](https://github.com/Textualize/rich) | 终端彩色输出 | MIT |
| [PyYAML](https://github.com/yaml/pyyaml) | YAML 配置文件解析 | MIT |
| [soundfile](https://github.com/bastibe/python-soundfile) | WAV 音频读写 | BSD-3-Clause |
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

### 测试标记

- `@pytest.mark.slow` — 需要下载模型的慢速集成测试

---

## 硬件要求

| 配置 | VRAM | 说明 |
|------|------|------|
| 最低 | 无（CPU） | 可运行，速度较慢 |
| 推荐 | 8GB+ GPU | 流畅运行全部功能 |
| 开发环境 | AMD 7900XTX 20GB | 全部管线 + 模型常驻 |

### VRAM 占用估算

| 阶段 | 模型 | 预估显存 |
|------|------|----------|
| 噪声抑制 | MossFormerGAN_SE_16K | ~1-2 GB |
| 说话人识别 | Pyannote 3.1 | ~2-3 GB |
| 语音分离 | MossFormer2_SS_16K | ~1-2 GB |
| TSE | AV_MossFormer2_TSE_16K | ~2-3 GB |
| 声纹匹配 | ECAPA-TDNN | ~0.5 GB |
| ASR | SeACo-Paraformer | ~1-2 GB |

管线在各阶段之间释放模型显存（`cleanup()`），避免同时加载所有模型。

---

## License

待定。
