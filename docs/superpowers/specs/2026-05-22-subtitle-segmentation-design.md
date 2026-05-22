# 字幕分段算法改进设计

> 日期: 2026-05-22
> 状态: 已审批
> 范围: `SubtitleSegmenter` 多趟管线重构

## 背景

### 当前问题

项目中存在两套独立的字幕分段实现：

1. `segment_by_timestamps()`（`transcribe/models/asr/utils.py:109-202`）— 字符级，仅 Qwen3-ASR 使用
2. `SubtitleSegmenter`（`transcribe/models/segmentation.py`）— 词级，较新

两者共享相同的**单趟贪心 + 4 级优先切分**算法，导致以下问题：

- **切分位置不自然**：硬切策略粗糙（`len//2` 或 flush 全部），不考虑语音停顿
- **无间隙感知**：两个词之间 0.5s+ 的自然停顿是最好的切分信号，但完全被忽略
- **无 CPS 约束**：允许 25 字/0.5s（50 CPS = 不可读），不符合 Netflix 中文标准（9 CPS）
- **代码重复**：两套实现逻辑高度重叠，需同步维护

### 业界标准

| 参数 | Netflix 中文标准 | 本项目当前值 |
|------|-----------------|-------------|
| CPS 上限 | 9（成人），11（SDH） | 无 |
| 每行最大字符 | 16（Originals），23（非 Originals） | 25 |
| 最短时长 | 0.833s（5/6 秒） | 0.833s（仅 SubtitleSegmenter） |
| 最长时长 | 7s | 7s |

### 业界方案调研结论

对 stable-ts、WhisperJAV、CapsWriter、VideoCaptioner 等工具的调研表明：

- **所有成熟工具都使用多趟规则管线**（非 DP 优化）
- 核心模式：句末标点切 → 间隙切 → 短片段合并 → 超限修正 → CPS 校验
- **无可用现成库**：stable-ts-whisperless 最接近但需大量格式适配，引入 3000 行依赖换 70 行功能不值得
- 最实际的做法是自实现多趟管线

## 设计

### 算法架构：5 趟管线

```
Input: list[WordTimestamp]
  │
  ├─ Pass 1: 句末标点硬切（。！？!?…—）
  │   → 标点丢弃，产出 segments
  │
  ├─ Pass 2: 间隙感知切分
  │   → 对 Pass 1 每个 segment，在词间间隙处切分
  │   → 双阈值: gap > 0.5s 且 ≥5字符 / gap > 1.0s 无条件切
  │
  ├─ Pass 3: 短片段合并（< min_duration）
  │   → 合并 < 0.833s 的片段到邻居
  │
  ├─ Pass 4: 超限修正
  │   → 对仍 > max_duration 或 > max_chars 的 segment
  │   → 在最佳候选点切分: 间隙质量(0-8) + 高斯中点偏好(0-2)
  │
  ├─ Pass 5: CPS 校验
  │   → 对 CPS > max_cps 的 segment 尝试用 Pass 4 逻辑再切分
  │
  └─ Output: list[TranscriptSegment]
```

### 参数

```python
max_duration: float = 7.0              # Netflix 标准
max_chars: int = 25                    # 略高于 Netflix 23（合理余量）
min_duration: float = 0.833            # 5/6 秒，Netflix 标准
max_cps: float = 12.0                  # Netflix 9 + 余量
gap_soft: float = 0.5                  # stable-ts 默认值
gap_hard: float = 1.0                  # CapsWriter 硬阈值
min_chars_for_gap_split: int = 5       # CapsWriter 守卫，防单字碎片
```

### 各 Pass 实现

#### Pass 1: 句末标点硬切

复用现有 `_split_by_punctuation()` 中对 `_SENTENCE_END` 的处理。遍历 words，遇到句末标点时 flush buffer 并丢弃标点。无变化。

#### Pass 2: 间隙感知切分（核心新增）

对 Pass 1 产出的每个 word group，扫描词间间隙：

- `gap > gap_hard`（1.0s）：无条件切分
- `gap > gap_soft`（0.5s）且已积累 ≥ `min_chars_for_gap_split`（5）字符：切分

双阈值设计参考 CapsWriter 的 soft/hard 模式，防止自然语流中的微小停顿产生碎片。

#### Pass 3: 短片段合并

直接复用现有 `_merge_short()`。单趟左到右扫描，将 < `min_duration` 的片段与前一片段合并（需满足合并后不超限）。

#### Pass 4: 超限修正

替换当前的硬切逻辑（`len//2`）。在完整 buffer 中寻找最佳切分点：

评分函数：
```
score(i) = gap_score(i) + midpoint_score(i)

gap_score = min(8.0, gap_seconds * 8.0)     # 间隙越大越好
midpoint_score = 2.0 * exp(-((ratio - 0.5)² / (2 * 0.09)))  # 靠近中点加分
```

约束：前后至少各 2 字符。递归处理右侧（可能仍超限）。

#### Pass 5: CPS 校验

计算每段 CPS = `len(text) / duration`。对超过 `max_cps` 的段，尝试用 Pass 4 逻辑再切分。无法再切（单字/单词）时保留原样。

### 主入口

```python
def segment(self, words: list[WordTimestamp]) -> list[TranscriptSegment]:
    if not words:
        return []
    pass1 = self._split_sentence_end(words)           # Pass 1
    pass2 = [g for grp in pass1 for g in self._split_by_gap(grp)]  # Pass 2
    raw = [self._build_segment(g) for g in pass2]
    merged = self._merge_short(raw)                    # Pass 3
    fixed = self._fix_oversized(merged)                # Pass 4
    return self._validate_cps(fixed)                   # Pass 5
```

## 兼容性

### 对 Qwen3-ASR 的影响

`segment_by_timestamps()` 改为薄适配器：将 `char_ts` 转为 `WordTimestamp`，委托给 `SubtitleSegmenter`。

```python
def segment_by_timestamps(char_ts, max_duration=7.0, max_chars=25):
    words = [WordTimestamp(word=t, start_time=s, end_time=e) for t, s, e in char_ts]
    return SubtitleSegmenter(max_duration=max_duration, max_chars=max_chars).segment(words)
```

- `qwen3_asr.py` 无需修改
- `tests/test_asr.py` 中 7 个现有测试继续通过（签名不变）

### 对 FunASR 的影响

**无影响**。FunASR 的分段逻辑完全由内置 VAD 决定，代码路径中不引用 `segment_by_timestamps` 或 `SubtitleSegmenter`。

两套后端的分段差异在改动前已存在（VAD chunk vs 精确时间戳分段），本轮不扩大差异。

### 修改文件清单

| 文件 | 改动 |
|------|------|
| `transcribe/models/segmentation.py` | 重写 `SubtitleSegmenter`，实现 5 趟管线 |
| `transcribe/models/asr/utils.py` | `segment_by_timestamps()` 改为委托适配器 |
| `tests/test_asr.py` | 更新现有测试 + 新增 Pass 2/4/5 测试用例 |

## 测试计划

| 测试 | 验证点 |
|------|--------|
| 现有 7 个测试全部通过 | 向后兼容 |
| 间隙切分：gap > 0.5s 且 ≥5 字符触发 | Pass 2 soft 阈值 |
| 间隙切分：gap > 1.0s 无条件触发 | Pass 2 hard 阈值 |
| 间隙守卫：< 5 字符时不切 | Pass 2 防碎片 |
| 短片段合并 | Pass 3 |
| 超限修正：间隙/中点评分选择 | Pass 4 |
| CPS 超标触发再切分 | Pass 5 |
| 端到端：Qwen3 字符级时间戳 | 集成验证 |
