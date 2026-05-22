# 重叠语音词级说话人归属设计

## 背景

当前管线采用「ASR 全量转录 → 说话人分离 → 段级归属」的级联架构。重叠语音区域仅被标记 `is_overlap=True`，但实际上混合在一起的语音只被转录为一条文本，归属给占时更长的说话人，另一说话人的内容丢失。

### 使用场景

- 多人会议中存在短暂插话（"嗯"、"对"）和持续数秒的交叉讨论
- 需要分别转录同时说话的各个说话人的字幕
- 需要正确标注说话人
- 输出为同一时间段的多条独立 SRT 条目
- 质量优先

### 方案选择

经对比三种方案（级联优化 / 音频分离 / 多说话人 ASR），选择**级联优化方案**：

- 不引入新模型，改动最小
- 非重叠区域质量不受影响
- 利用现有 ASR 的词级时间戳 + diarization 的说话人活跃时段做词级归属
- 未来可扩展（音频分离、端到端多说话人 ASR）时按需重构

## 设计

### 核心思路

在重叠区域内，对 ASR 输出的每个词（word）计算其与 diarization 各说话人活跃时段的时间交叉，将词归属给交叉时间最长的说话人，然后按连续说话人分组形成多条独立子段。

### 管线变更

```
当前流程：
  音频 → ASR(全文) → 分段 → 说话人分离 → 段级归属(一人一段) → 输出

新流程：
  音频 → ASR(全文) → 分段 → 说话人分离 → 归属
                                              ├─ 非重叠段：段级归属（不变）
                                              └─ 重叠段：词级归属 + 按说话人分组
                                                        → 合并 → 匹配 → 输出
```

### 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `models/attribution/overlap.py` | 重写 | 删除 `MarkOverlapHandler`，实现 `OverlapHandler` 词级归属 |
| `models/attribution/engine.py` | 修改 | 调用新的 `OverlapHandler` |
| `pipeline.py` | 修改 | 适配新的 `OverlapHandler` |
| `models/srt_writer.py` | 可能微调 | 确认重叠多条目输出正确（预计无需改动） |
| `data/types.py` | 不变 | `TranscriptSegment.is_overlap` 字段已存在 |
| `config.py` | 不变 | 不新增配置项 |
| `cli.py` | 不变 | 不新增 CLI 参数 |

### OverlapHandler 实现

替换现有的 `MarkOverlapHandler`，实现词级归属逻辑：

```python
class OverlapHandler:
    """对重叠区域做词级说话人归属，按连续说话人分组为多条子段。"""

    def handle(
        self,
        segments: list[TranscriptSegment],
        diarization: DiarizationResult,
    ) -> list[TranscriptSegment]:
        if not diarization.overlap_regions:
            return segments

        result = []
        for seg in segments:
            overlap = self._find_overlap(seg, diarization.overlap_regions)
            if overlap is None:
                result.append(seg)
                continue

            result.extend(self._split_by_speaker(seg, diarization.segments))

        return result

    def _find_overlap(self, seg, overlap_regions):
        """判断段是否落在重叠区域内。"""
        ...

    def _split_by_speaker(self, seg, speaker_segments):
        """对重叠段做词级归属，按连续说话人分组。"""
        if not seg.words:
            # 无词级时间戳，保持原段不变，标记 is_overlap
            seg.is_overlap = True
            return [seg]

        # 1. 对每个词，计算与各说话人活跃段的时间交叉
        word_speakers = []
        for word in seg.words:
            speaker_id = self._attribute_word(word, speaker_segments)
            word_speakers.append((word, speaker_id))

        # 2. 按连续说话人分组
        groups = self._group_consecutive(word_speakers)

        # 3. 每组生成一个 TranscriptSegment
        sub_segments = []
        for group in groups:
            words = [w for w, _ in group]
            speaker_id = group[0][1]
            text = "".join(w.word for w in words)
            sub_segments.append(TranscriptSegment(
                speaker_id=speaker_id,
                start_time=words[0].start_time,
                end_time=words[-1].end_time,
                text=text,
                is_overlap=True,
                words=words,
            ))

        return sub_segments

    def _attribute_word(self, word, speaker_segments):
        """将单个词归属给时间交叉最长的说话人。"""
        best_speaker = None
        best_overlap = 0.0

        for spk_seg in speaker_segments:
            overlap = self._compute_overlap_duration(
                word.start_time, word.end_time,
                spk_seg.start_time, spk_seg.end_time,
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = spk_seg.speaker_id

        return best_speaker

    @staticmethod
    def _compute_overlap_duration(s1, e1, s2, e2):
        """计算两个时间段的重叠时长。"""
        overlap_start = max(s1, s2)
        overlap_end = min(e1, e2)
        return max(0.0, overlap_end - overlap_start)

    @staticmethod
    def _group_consecutive(word_speakers):
        """按连续说话人分组。"""
        if not word_speakers:
            return []
        groups = []
        current = [word_speakers[0]]
        for ws in word_speakers[1:]:
            if ws[1] == current[-1][1]:
                current.append(ws)
            else:
                groups.append(current)
                current = [ws]
        groups.append(current)
        return groups
```

### AttributionEngine 集成

```python
class AttributionEngine:
    def __init__(self):
        self.timestamp_strategy = TimestampStrategy()
        self.overlap_handler = OverlapHandler()

    def attribute(self, segments, diarization):
        # Step 1: 段级归属
        attributed = self.timestamp_strategy.attribute(segments, diarization)

        # Step 2: 重叠区域词级归属
        attributed = self.overlap_handler.handle(attributed, diarization)

        return attributed
```

### 重叠段判断逻辑

沿用当前 `MarkOverlapHandler` 的中心点判断逻辑：计算 `TranscriptSegment` 的中心时刻 `(start_time + end_time) / 2`，若落在 `diarization.overlap_regions` 的某个区间内，则视为重叠段。

### SRT 输出

重叠区域的多条子段作为独立的 `TranscriptSegment`，具有各自的起止时间。SrtWriter 按时间排序输出，自然产生同一时间段的多条条目。无需修改 SrtWriter。

### 端到端数据流示例

**输入**：张三（0:00-0:15）和李四（0:08-0:20），重叠区 0:08-0:15。

**ASR 输出词级时间戳**（重叠区部分）：
```
"这个"  0:10.0-0:10.3 → 交叉 SPEAKER_00 更长 → 归 SPEAKER_00
"方案"  0:10.3-0:10.6 → 归 SPEAKER_00
"嗯"    0:10.5-0:10.7 → 归 SPEAKER_00
"可以"  0:10.8-0:11.1 → 归 SPEAKER_00
"对"    0:10.9-0:11.2 → 归 SPEAKER_01
"同意"  0:11.3-0:11.6 → 归 SPEAKER_01
```

**按连续说话人分组后**：
```
TranscriptSegment(SPEAKER_00, 0:10.0, 0:11.1, "这个方案嗯可以", is_overlap=True)
TranscriptSegment(SPEAKER_01, 0:10.9, 0:11.6, "对同意", is_overlap=True)
```

**SRT 输出**：
```
5
00:00:10,000 --> 00:00:11,100
[张三] 这个方案嗯可以

6
00:00:10,900 --> 00:00:11,600
[李四] 对同意
```

### 局限性

1. **ASR 对混合音频的转录质量有限**：两人同时说话时 ASR 可能遗漏或混淆部分内容，词级归属无法弥补 ASR 本身的限制
2. **完全同步说话时无法区分**：两人完全同步说出相同音节时，词级时间戳重叠，归属可能不准确
3. **无词级时间戳时的退化处理**：某些 ASR 后端可能不提供 `words` 字段，此时保持原段不变仅标记 `is_overlap=True`

### 测试策略

- 单元测试：`OverlapHandler` 的词级归属、分组逻辑
- 集成测试：使用合成重叠音频验证端到端输出
- 回归测试：确保非重叠区域输出不变
