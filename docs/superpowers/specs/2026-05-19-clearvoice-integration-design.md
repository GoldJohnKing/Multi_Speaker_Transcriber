# ClearVoice 集成设计方案

## 概述

将 ClearVoice（阿里巴巴通义实验室 ClearerVoice-Studio）集成到转录管线中，替换现有的降噪和语音分离模块，并新增声纹匹配和目标说话人提取功能。

### 动机

| 问题 | 现有方案 | ClearVoice 方案 |
|------|---------|----------------|
| 降噪采样率不匹配 | DeepFilterNet 需要 48kHz，管线需 16→48→16kHz 重采样 | MossFormerGAN_SE_16K 原生 16kHz，无重采样 |
| 语音分离采样率 bug | SepFormer-whamr 训练于 8kHz，管线喂入 16kHz | MossFormer2_SS_16K 原生 16kHz |
| 分离质量 | SepFormer SI-SNRi 14.0dB (WHAMR!) | MossFormer2 SS SI-SNRi 17.4dB (WHAMR!)，+24% |
| 说话人-音轨映射 | 按索引硬匹配，标签随机互换 | 声纹嵌入余弦相似度匹配 |
| 中文支持 | 降噪/分离模型均未含中文训练数据 | 阿里巴巴训练，含 MISP 中文数据集 |
| 依赖管理 | deepfilternet + speechbrain 两个独立依赖 | clearvoice 一个统一依赖 |

### 关键约束

- 语言：中文普通话
- 硬件：AMD 7900XTX 20GB VRAM (ROCm)
- ClearVoice 的 GPU 检测依赖 `nvidia-smi`，ROCm 环境需 monkey-patch
- Python 包管理：`uv`

---

## 架构变更

### 变更前

```
Video → ① Audio Extract → ② Denoise (DeepFilterNet, opt) → ③ Diarize (Pyannote) → ④ Separate (SepFormer, opt) → ⑤ ASR (FunASR) → ⑥ SRT
```

### 变更后

```
Video → ① Audio Extract → ② Denoise (ClearVoice SE, opt) → ③ Diarize (Pyannote) → ④ Separate (ClearVoice SS, opt)
                                                                                        → ④.5 Speaker Match (声纹嵌入, new)
                                                                                        → ④A TSE (ClearVoice TSE, opt)
                                                                    → ⑤ ASR (FunASR) → ⑥ SRT
```

### 管线流程图

```
┌──────────────────────────────────────────────────────────────────┐
│                     Pipeline Orchestrator                         │
└──┬────────┬────────┬────────┬────────┬────────┬──────────────────┘
   ▼        ▼        ▼        ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│Extract│ │Denoi-│ │Diari-│ │Sepa- │ │ ASR  │ │ SRT  │
│Audio  │ │se    │ │zation│ │ration│ │      │ │Writer│
│FFmpeg │ │Clear │ │Pyan- │ │Clear │ │SeACo │ │      │
│       │ │Voice │ │note  │ │Voice │ │Para- │ │Merge │
│       │ │SE    │ │3.1   │ │SS    │ │former│ │spkr+ │
│       │ │ opt  │ │default│ │opt   │ │      │ │ts→srt│
└──────┘ └──────┘ └──────┘ └──┬───┘ └──────┘ └──────┘
    ①        ②        ③      │        ⑤        ⑥
                              ▼
                        ┌─────────────┐
                        │ Speaker     │ ← 声纹嵌入余弦匹配
                        │ Matcher     │
                        └─────────────┘
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
              ┌──────────┐        ┌──────────┐
              │ClearVoice│        │ClearVoice│
              │ SS 盲分离 │        │ TSE 目标 │
              │ (默认)    │        │ 说话人   │
              └──────────┘        │ (--tse)  │
                                  └──────────┘
```

---

## 项目结构变更

```
multi_speaker_transcribe/
├── pyproject.toml              # 移除 deepfilternet/speechbrain，新增 clearvoice
├── config.yaml                 # 更新配置项
├── transcribe/
│   ├── __init__.py
│   ├── __main__.py             # CLI entry point
│   ├── cli.py                  # 新增 --tse 标志
│   ├── pipeline.py             # 重写降噪/分离/匹配集成逻辑
│   ├── config.py               # 新增 tse 配置字段
│   ├── models/
│   │   ├── __init__.py
│   │   ├── audio_extractor.py  # ① 不变
│   │   ├── denoiser.py         # ② 重写：ClearVoice SE
│   │   ├── diarizer.py         # ③ 不变（仍用 Pyannote）
│   │   ├── separator.py        # ④ 重写：ClearVoice SS
│   │   ├── matcher.py          # ④.5 新增：声纹嵌入匹配
│   │   ├── extractor.py        # ④A 新增：ClearVoice TSE
│   │   ├── rocm_compat.py      # 新增：ROCm monkey-patch
│   │   ├── asr.py              # ⑤ 不变
│   │   └── srt_writer.py       # ⑥ 不变
│   └── data/
│       ├── __init__.py
│       └── types.py            # PipelineConfig 新增 tse 字段
└── tests/
    ├── test_denoiser.py        # 重写
    ├── test_separator.py       # 重写
    ├── test_matcher.py         # 新增
    └── test_extractor.py       # 新增
```

---

## 数据类型变更

`transcribe/data/types.py` 中 `PipelineConfig` 新增字段：

```python
@dataclass
class PipelineConfig:
    device: str = "auto"          # "cpu" | "cuda" | "auto"
    denoise: bool = False         # 启用 ClearVoice 噪声抑制
    diarize: bool = True          # 启用 Pyannote 说话人识别
    separate: bool = False        # 启用 ClearVoice 盲分离
    tse: bool = False             # 启用 ClearVoice 目标说话人提取
    hotwords: str | None = None   # 热词文件路径
    language: str = "zh"
    cache_dir: str = ".cache"
    num_speakers: int | None = None
```

其余数据类型（`AudioSegment`、`SpeakerSegment`、`DiarizationResult`、`TranscriptSegment`）不变。

---

## Stage 2：噪声抑制（重写）

- **输入**：`AudioSegment`（16kHz）
- **输出**：`AudioSegment`（16kHz，降噪后）
- **模型**：ClearVoice `MossFormerGAN_SE_16K`
- **默认**：关闭，通过 `--denoise` 启用

### 行为

1. SNR 估算（沿用现有 `estimate_snr()` 逻辑）
2. 若 SNR >= 25dB：跳过降噪
3. 若 SNR < 25dB：加载 ClearVoice SE 模型 → 增强 → 返回

### 关键改进

| 维度 | 变更前（DeepFilterNet） | 变更后（ClearVoice SE） |
|------|----------------------|----------------------|
| 采样率 | 48kHz（需 16→48→16 重采样） | **16kHz（直接匹配 ASR）** |
| PESQ | 3.03 | **3.57** |
| SI-SDR | 15.71 dB | **20.60 dB** |
| STOI | 0.94 | **0.98** |
| 重采样 | 2 次（有损） | **0 次** |
| `end_time` 偏移 bug | 有 | **无** |

### 接口

```python
class Denoiser:
    def __init__(self, device: str = "cpu") -> None: ...
    def process(self, audio: AudioSegment) -> AudioSegment: ...
    def cleanup(self) -> None: ...
```

接口签名不变，内部实现替换为 ClearVoice。

### ClearVoice 调用方式

```python
from clearvoice import ClearVoice

cv = ClearVoice(task='speech_enhancement', model_names=['MossFormerGAN_SE_16K'])
# numpy 数组输入输出，无需临时文件
input_array = audio.waveform.reshape(1, -1).astype(np.float32)  # [1, T]
output_array = cv(input_array, False)  # [1, T]
result = output_array.squeeze(0).astype(np.float32)
```

---

## Stage 4：语音分离（重写）

- **输入**：`AudioSegment` + `DiarizationResult`
- **输出**：`list[AudioSegment]`（每个重叠区域的分离音轨）
- **模型**：ClearVoice `MossFormer2_SS_16K`
- **默认**：关闭，通过 `--separate` 启用

### 行为

1. 遍历 `diarization.overlap_regions`
2. 对每个重叠区域裁剪音频
3. 调用 ClearVoice SS 分离为 2 条音轨
4. 返回所有分离音轨（保留原始时间偏移）

### 接口

```python
class Separator:
    def __init__(self, device: str = "cpu") -> None: ...
    def separate_overlaps(self, audio: AudioSegment, diarization: DiarizationResult) -> list[AudioSegment]: ...
    def cleanup(self) -> None: ...
```

### ClearVoice 调用方式

```python
cv = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])
input_array = audio.waveform.reshape(1, -1).astype(np.float32)  # [1, T]
output_array = cv(input_array, False)  # [num_spks, 1, T]
# output_array[0] → speaker 1, output_array[1] → speaker 2
```

### 改进

| 维度 | 变更前（SepFormer-whamr） | 变更后（ClearVoice SS） |
|------|------------------------|----------------------|
| 采样率 | 8kHz 模型接收 16kHz（**致命 bug**） | **16kHz 原生** |
| SI-SDRi (WHAMR!) | 14.0 dB | **17.4 dB** |
| 训练数据 | 英语 WSJ0 | **11 个数据集含中文 MISP** |
| 长音频 | 需手动分块，OOM 风险 | **内置分块（30s/10s 窗口）** |

---

## Stage 4.5：声纹嵌入匹配（新增）

解决核心问题：**分离出的音轨是匿名的，不知道哪条对应哪个说话人**。

### 原理

```
1. 从 Pyannote 识别结果中，为每个说话人找一段干净的非重叠语音（≥2秒）
2. 用 Pyannote 的 PretrainedSpeakerEmbedding 提取每个说话人的声纹嵌入（192维）
3. 对每条分离音轨，同样提取声纹嵌入
4. 用余弦相似度匹配：音轨嵌入 vs 说话人嵌入 → 最佳匹配
5. 阈值门控：相似度 < 0.5 时标记为"未匹配"
```

### 接口

```python
class SpeakerMatcher:
    def __init__(self, device: str = "cpu") -> None: ...
    def match_tracks_to_speakers(
        self,
        separated_tracks: list[AudioSegment],
        audio: AudioSegment,
        diarization: DiarizationResult,
    ) -> dict[int, str]:
        """将分离音轨索引映射到说话人 ID。

        Returns:
            dict[int, str]: {track_index: speaker_id}
        """
        ...
    def cleanup(self) -> None: ...
```

### 嵌入提取

使用 Pyannote 内置的 `PretrainedSpeakerEmbedding`：

```python
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb"
)
# 输入: torch.Tensor [batch, 1, samples] (16kHz)
# 输出: numpy [batch, 192]
```

### 匹配算法

```python
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# 对每条音轨，找最佳匹配说话人
for track_idx, track_embedding in enumerate(track_embeddings):
    best_speaker = None
    best_sim = -1.0
    for speaker_id, ref_embedding in reference_embeddings.items():
        sim = _cosine_similarity(track_embedding, ref_embedding)
        if sim > best_sim:
            best_sim = sim
            best_speaker = speaker_id
    if best_sim >= MATCH_THRESHOLD:
        mapping[track_idx] = best_speaker
```

### 退化处理

| 情况 | 处理 |
|------|------|
| 说话人无非重叠段落（全部重叠） | 使用最长重叠前/后的短暂独白段落 |
| 音轨 < 0.5s（嵌入提取最少需要 ~0.5s） | 跳过匹配，标记为 `UNKNOWN` |
| 两条音轨匹配到同一说话人 | 保留相似度更高的匹配，另一条标记为 `UNKNOWN` |
| 所有匹配相似度 < 0.5 | 回退到按索引匹配（兼容旧行为） |

---

## Stage 4A：目标说话人提取（新增，可选）

- **输入**：视频文件路径 + `AudioSegment`（降噪后）+ `DiarizationResult`
- **输出**：`dict[str, AudioSegment]`（{人脸轨迹ID: 说话人音频}）
- **模型**：ClearVoice `AV_MossFormer2_TSE_16K`
- **默认**：关闭，通过 `--tse` 启用

### 行为

1. 从视频中检测人脸轨迹（S3FD 人脸检测器）
2. 对每条人脸轨迹，提取唇部视觉特征
3. 用视觉特征引导 TSE 模型从混合音频中提取该说话人的语音
4. 每条人脸轨迹产生一个独立的说话人音频输出

### 与管线的关系

TSE 与 `--separate` 互斥：

| 组合 | 行为 |
|------|------|
| `--tse`（单独） | TSE 替代盲分离，用视觉身份直接关联说话人 |
| `--separate`（单独） | ClearVoice SS 盲分离 + 声纹匹配 |
| `--tse --separate` | **错误**，互斥标志 |
| 无标志 | 不处理重叠，ASR 直接转录混合音频 |

### TSE 的声纹匹配

TSE 输出的每条音轨带有人脸轨迹 ID（非说话人 ID）。仍需声纹匹配将人脸轨迹映射到 Pyannote 说话人标签：

```
TSE 输出: {face_0: audio_A, face_1: audio_B}
声纹匹配: face_0 → SPEAKER_01, face_1 → SPEAKER_00
```

### 接口

```python
class TargetSpeakerExtractor:
    def __init__(self, device: str = "cpu") -> None: ...
    def extract(
        self,
        video_path: str,
        audio: AudioSegment,
        diarization: DiarizationResult,
    ) -> dict[str, AudioSegment]:
        """从视频中提取每个说话人的独立音频。

        Args:
            video_path: 输入视频文件路径
            audio: 降噪后的音频
            diarization: 说话人识别结果

        Returns:
            {face_track_id: AudioSegment} 每条人脸轨迹的独立说话人音频
        """
        ...
    def cleanup(self) -> None: ...
```

### TSE 管线集成逻辑

```
if config.tse:
    extractor = TargetSpeakerExtractor(device=device)
    face_tracks = extractor.extract(input_path, audio, diarization)
    extractor.cleanup()

    matcher = SpeakerMatcher(device=device)
    mapping = matcher.match_tracks_to_speakers(
        list(face_tracks.values()), audio, diarization
    )
    matcher.cleanup()

    # 用 mapping 将 face_track_id → speaker_id
    # 单说话人区域：用降噪后原始音频
    # 重叠区域：用 TSE 提取的对应说话人音频
```

### 限制

- **需要视频输入**：纯音频文件无法使用 TSE
- **需要可见人脸**：远景、背面、遮挡会导致提取失败
- **人脸检测耗时**：约 7-15 分钟/小时视频（GPU）
- **人脸轨迹与说话人不一定一一对应**：同一说话人可能产生多条轨迹（多次入画），需要声纹匹配进行合并
- **盲分离模式（`--separate`）当前仅支持 2 说话人**：ClearVoice `MossFormer2_SS_16K` 的 `num_spks` 默认为 2

---

## ROCm 兼容性

ClearVoice 的 GPU 检测使用 `nvidia-smi`，在 AMD ROCm 环境下会静默回退到 CPU。

### 修复方案：`transcribe/models/rocm_compat.py`

在导入 ClearVoice 之前执行 monkey-patch，强制使用 CUDA 设备：

```python
"""ClearVoice ROCm compatibility patch.

ClearVoice's SpeechModel.__init__ uses nvidia-smi to detect GPU availability.
On AMD ROCm, nvidia-smi doesn't exist, causing models to silently fall back to CPU.

This patch overrides the device detection to use torch.cuda.is_available() instead.
"""
import torch


def patch_clearvoice_for_rocm() -> None:
    """Apply ROCm compatibility patch for ClearVoice.

    Safe to call multiple times (idempotent). No-op if ClearVoice is not installed.
    """
    try:
        from clearvoice.clearvoice.networks import SpeechModel
    except ImportError:
        return

    if getattr(SpeechModel, "_rocm_patched", False):
        return

    _original_init = SpeechModel.__init__

    def _patched_init(self, args) -> None:
        _original_init(self, args)
        if torch.cuda.is_available() and self.device.type == "cpu":
            import torch as _torch
            args.use_cuda = 1
            self.device = _torch.device("cuda")

    SpeechModel.__init__ = _patched_init
    SpeechModel._rocm_patched = True
```

### 使用方式

在 `denoiser.py`、`separator.py`、`extractor.py` 的模型加载前调用：

```python
from transcribe.models.rocm_compat import patch_clearvoice_for_rocm
patch_clearvoice_for_rocm()
```

---

## CLI 变更

### 新增标志

| 标志 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--tse` | flag | False | 启用 ClearVoice 目标说话人提取（需视频输入） |

### 互斥规则

- `--tse` 与 `--separate` 互斥
- `--tse` 隐含 `--denoise`（TSE 应在降噪后音频上工作）
- `--tse` 隐含 `--diarize`（需要识别结果作为说话人标签来源）
- `--separate` 隐含 `--diarize`

### 用法示例

```bash
# 默认：ASR + 说话人识别
uv run python -m transcribe input.mp4 -o output.srt -v

# 启用降噪
uv run python -m transcribe input.mp4 --denoise -o output.srt -v

# 启用盲分离（ClearVoice SS + 声纹匹配）
uv run python -m transcribe input.mp4 --denoise --separate -o output.srt -v

# 启用 TSE（ClearVoice TSE + 声纹匹配）
uv run python -m transcribe input.mp4 --tse -o output.srt -v

# 纯 ASR
uv run python -m transcribe input.mp4 --no-diarize -o output.srt -v
```

---

## 配置变更

```yaml
# config.yaml — 更新后的默认配置
device: auto
denoise: false
diarize: true
separate: false
tse: false              # 新增：目标说话人提取
language: zh

# 噪声抑制
denoiser:
  model: MossFormerGAN_SE_16K   # ClearVoice 模型
  snr_threshold: 25             # SNR >= 25dB 时跳过降噪

# 语音分离
separator:
  model: MossFormer2_SS_16K     # ClearVoice 模型
  max_segment_seconds: 10       # 最大分段长度

# 目标说话人提取
extractor:
  model: AV_MossFormer2_TSE_16K # ClearVoice TSE 模型
  min_track_frames: 50          # 最少人脸轨迹帧数（25fps 下 = 2 秒）

# 声纹匹配
matcher:
  embedding_model: speechbrain/spkrec-ecapa-voxceleb
  match_threshold: 0.5          # 余弦相似度匹配阈值
  min_segment_seconds: 0.5      # 最短参考音频时长
```

---

## VRAM 管理

所有模型仍然顺序加载、使用后立即释放：

| 阶段 | 模型 | VRAM |
|------|------|------|
| 2 - 降噪 | ClearVoice MossFormerGAN_SE_16K | ~0.5 GB |
| 3 - 识别 | Pyannote 3.1 | ~4-8 GB |
| 4 - 分离 | ClearVoice MossFormer2_SS_16K | ~2-4 GB |
| 4 - TSE | ClearVoice AV_MossFormer2_TSE_16K | ~1-2 GB |
| 4.5 - 匹配 | Pyannote PretrainedSpeakerEmbedding | ~0.5 GB |
| 5 - ASR | Paraformer + VAD + Punc | ~3-4 GB |

峰值 VRAM：~10GB（每次仅加载一个模型）。每个阶段完成后调用 `torch.cuda.empty_cache()`。

### 采样率简化

| 情况 | 提取采样率 | 说明 |
|------|----------|------|
| `--denoise` | 16kHz | ClearVoice SE 直接处理 16kHz，无需 48kHz |
| 无 `--denoise` | 16kHz | 直接用于 ASR |

**消除了 48kHz 提取和重采样步骤。** 管线全程 16kHz。

---

## 依赖变更

```toml
[project]
name = "multi-speaker-transcribe"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "torchaudio>=2.1",
    "funasr>=1.1",
    "modelscope",
    "pyannote.audio>=3.1",
    "clearvoice",           # 新增：替代 deepfilternet + speechbrain
    "numpy<2.0",            # clearvoice 限制
    "pyyaml",
    "rich",
    "soundfile==0.12.1",    # clearvoice 固定版本
]
```

移除的依赖：
- `deepfilternet` — 被 ClearVoice SE 替代
- `speechbrain` — 被 ClearVoice SS 替代

新增的依赖（通过 clearvoice 间接引入）：
- `librosa==0.10.2.post1`
- `opencv-python==4.10.0.84`（TSE 人脸检测需要）
- `torchvision`（TSE 需要）
- `scenedetect==0.6.6`（TSE 场景检测需要）

**注意**：`clearvoice` 固定了 `numpy<2.0` 和 `librosa==0.10.2.post1`，需验证与 FunASR/Pyannote 的兼容性。

---

## 进度显示

```
# --denoise --separate（完整管线）
[1/6] 提取音频 (16000Hz) ... 完成 (2.1s)
[2/6] 噪声抑制 ... SNR=12.3dB < 25dB，需要降噪 ... 完成 (1m 23s)
[3/6] 说话人识别 ... 检测到 4 位说话人, 12 个重叠区域 ... 完成 (3m 05s)
[4/6] 重叠语音分离 ... 处理 12 个重叠片段 ... 完成 (2m 18s)
[5/6] 语音转文字 ... 识别 347 个片段 ... 完成 (4m 42s)
[6/6] 生成 SRT ... 输出 312 条字幕 ... 完成 (0.3s)

# --tse（TSE 管线）
[1/6] 提取音频 (16000Hz) ... 完成 (2.1s)
[2/6] 噪声抑制 ... SNR=12.3dB < 25dB，需要降噪 ... 完成 (1m 23s)
[3/6] 说话人识别 ... 检测到 4 位说话人, 12 个重叠区域 ... 完成 (3m 05s)
[4/6] 目标说话人提取 ... 检测到 4 个人脸轨迹，提取 4 条说话人音频 ... 完成 (8m 12s)
[5/6] 语音转文字 ... 识别 347 个片段 ... 完成 (4m 42s)
[6/6] 生成 SRT ... 输出 312 条字幕 ... 完成 (0.3s)
```

---

## 错误处理

### 新增边界情况

| 场景 | 策略 |
|------|------|
| ClearVoice 导入失败（未安装） | 打印错误：`请安装 clearvoice: uv pip install clearvoice` |
| `--tse` 与音频文件（非视频） | 错误：`--tse 需要视频文件输入（检测到音频文件）` |
| `--tse` 与 `--separate` 同时使用 | 错误：`--tse 和 --separate 互斥，请选择其一` |
| TSE 人脸检测未发现人脸 | 警告：`未检测到人脸，回退到盲分离模式 (--separate)` |
| 声纹匹配全部失败 | 警告：`声纹匹配失败，回退到索引匹配` |
| ClearVoice 依赖版本冲突 | 安装时提示 `numpy<2.0` 和 `librosa==0.10.2.post1` 约束 |
