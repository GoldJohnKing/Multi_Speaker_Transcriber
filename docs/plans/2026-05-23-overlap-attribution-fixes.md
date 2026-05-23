# 重叠归因与字幕拆行修复方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复说话人归因和重叠处理中的正确性缺陷，提升户外综艺多人同时说话场景下的可靠性。

**Architecture:** 在现有 ASR-first 管线不变的前提下，修复 diarization 输出信息的完整传递（non-exclusive segments、is_overlap 标记），改进重叠检测和合并算法，增加说话人边界感知分段和置信度评分。

**Tech Stack:** Python 3.12, dataclass, numpy, pyannote.audio, pytest

---

## 依赖关系与分组

```
Group A: types.py + diarizer.py (P1#2, P1#3, P2#8, P3#10)
  ↓
Group B: attribution/ (P1#4, P3#9, P3#11) — 依赖 Group A 的数据结构变更
  ↓
Group C: pipeline.py (P2#6) — 依赖 Group A+B 的逻辑变更
```

Group A 内的 4 个修复改动紧密耦合（都涉及 `DiarizationResult` 和 `diarizer.py`），应作为一个原子任务实施。Group B 的 3 个修复相对独立，可并行或顺序实施。

---

## Group A: Diarization 输出完整性修复

### Task 1: 扩展 `DiarizationResult` 数据结构

**Files:**
- Modify: `transcribe/data/types.py:39-44`

**背景:**
当前 `DiarizationResult` 只存储 exclusive diarization 的 segments（每个时间点只有 1 个说话人）。OverlapHandler 在重叠区域做词级归因时，需要知道**所有活跃说话人**的 segments，但 exclusive 输出在重叠区域只保留了 dominant speaker，丢失了其他说话人信息。

**修改内容:**

在 `DiarizationResult` 中新增 `non_exclusive_segments` 字段：

```python
@dataclass
class DiarizationResult:
    """Speaker diarization result."""

    segments: list[SpeakerSegment]  # exclusive diarization (1 speaker per time)
    num_speakers: int
    overlap_regions: list[tuple[float, float]] = field(default_factory=list)
    # NEW: non-exclusive segments showing all active speakers (including overlaps)
    # Used by OverlapHandler for word-level attribution in overlap regions.
    non_exclusive_segments: list[SpeakerSegment] = field(default_factory=list)
```

**设计决策:**
- 使用独立字段而非替换 `segments`，避免破坏 `TimestampStrategy` 对 exclusive segments 的依赖
- 默认空列表确保向后兼容

- [ ] **Step 1: 修改 `types.py` 添加 `non_exclusive_segments` 字段**

```python
# transcribe/data/types.py:39-46
@dataclass
class DiarizationResult:
    """Speaker diarization result."""

    segments: list[SpeakerSegment]
    num_speakers: int
    overlap_regions: list[tuple[float, float]] = field(default_factory=list)
    non_exclusive_segments: list[SpeakerSegment] = field(default_factory=list)
```

- [ ] **Step 2: 验证既有测试不被破坏**

Run: `uv run pytest tests/test_diarizer.py tests/test_attribution.py -v`
Expected: 全部 PASS

---

### Task 2: 重写 `Diarizer.process()` — 完整输出 + is_overlap 标记 + 合并修复 + 扫描线

**Files:**
- Modify: `transcribe/models/diarizer.py:178-263`
- Test: `tests/test_diarizer.py`

**背景:**
4 个问题需要在 `diarizer.py` 中一起修复：
1. **P1 #2**: 缺少 non-exclusive segments 存储
2. **P1 #3**: `SpeakerSegment.is_overlap` 从未被设为 True
3. **P2 #8**: `_merge_intervals` 合并了仅首尾相接的不同说话人对的重叠区域
4. **P3 #10**: O(n²) pairwise 重叠检测

**修改 1: 用扫描线算法替换 O(n²) pairwise 检测 (P3 #10)**

```python
def _extract_overlap_regions(
    self, diarization_output, audio_start_time: float = 0.0
) -> list[tuple[float, float]]:
    """Detect time regions where multiple speakers overlap.

    Uses a sweep-line algorithm (O(n log n)) instead of pairwise
    intersection (O(n²)). Collects (time, delta, speaker) events,
    sorts them, and tracks active speaker count to find overlap spans.
    """
    if hasattr(diarization_output, "speaker_diarization"):
        annotation = diarization_output.speaker_diarization
    else:
        annotation = diarization_output

    # Build sweep-line events
    events: list[tuple[float, int, str]] = []  # (time, +1/-1, speaker)
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        events.append((turn.start, 1, speaker))
        events.append((turn.end, -1, speaker))
    events.sort()

    # Sweep: overlap when >1 active speaker
    active_speakers: dict[str, int] = {}  # speaker -> ref count
    raw_overlaps: list[tuple[float, float]] = []
    ov_start: float | None = None

    for time, delta, speaker in events:
        prev_count = len(active_speakers)
        if delta > 0:
            active_speakers[speaker] = active_speakers.get(speaker, 0) + 1
        else:
            # Guard: ignore end-events for speakers not yet tracked.
            # This handles zero-duration turns and edge cases where
            # end events sort before start events at the same timestamp.
            if speaker not in active_speakers:
                continue
            active_speakers[speaker] -= 1
            if active_speakers[speaker] <= 0:
                active_speakers.pop(speaker, None)
        curr_count = len(active_speakers)

        if prev_count <= 1 and curr_count > 1:
            ov_start = time
        elif prev_count > 1 and curr_count <= 1 and ov_start is not None:
            raw_overlaps.append(
                (ov_start + audio_start_time, time + audio_start_time)
            )
            ov_start = None

    return self._merge_intervals(raw_overlaps)
```

**设计决策:**
- 使用 `dict[str, int]` 而非 `set` 处理同一说话人多个重叠 turn 的引用计数
- `prev_count <= 1` 而非 `== 0` 确保 1→2 的转换被捕获
- 排序时同一时间戳的 end(-1) 排在 start(+1) 前面（`(2.0, -1, A)` < `(2.0, 1, A)`）。guard `if speaker not in active_speakers: continue` 防止零时长 turn 的 end 事件污染计数

**修改 2: 修复合并逻辑 — 只合并真正重叠的区间 (P2 #8)**

```python
@staticmethod
def _merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge truly overlapping time intervals.

    Unlike the previous version which merged adjacent intervals (start <= prev_end),
    this only merges intervals with genuine temporal overlap (start < prev_end).
    This prevents merging overlap regions from unrelated speaker pairs that
    merely touch at a boundary.
    """
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[float, float]] = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start < prev_end:  # only merge genuine overlap, not adjacency
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged
```

**修改 3: 重写 `process()` — 提取 non-exclusive segments + 标记 is_overlap (P1 #2, P1 #3)**

```python
def process(self, audio: AudioSegment) -> DiarizationResult:
    """Run speaker diarization on audio."""
    waveform_tensor = torch.tensor(audio.waveform, dtype=torch.float32)
    if waveform_tensor.ndim == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)

    input_dict = {
        "waveform": waveform_tensor,
        "sample_rate": audio.sample_rate,
    }

    kwargs = {}
    if self._num_speakers is not None:
        kwargs["num_speakers"] = self._num_speakers

    diarization_output = self._pipeline(input_dict, **kwargs)

    # ── Exclusive annotation (1 speaker per time) ──────────────
    if hasattr(diarization_output, "exclusive_speaker_diarization"):
        exclusive = diarization_output.exclusive_speaker_diarization
    elif hasattr(diarization_output, "speaker_diarization"):
        exclusive = diarization_output.speaker_diarization
    else:
        exclusive = diarization_output  # pyannote 3.x

    # ── Non-exclusive annotation (all active speakers) ─────────
    if hasattr(diarization_output, "speaker_diarization"):
        non_exclusive = diarization_output.speaker_diarization
    else:
        non_exclusive = diarization_output  # pyannote 3.x fallback

    # ── Extract overlap regions (from non-exclusive) ───────────
    overlap_regions = self._extract_overlap_regions(
        diarization_output, audio.start_time
    )

    # ── Build exclusive segments with is_overlap flag ──────────
    segments: list[SpeakerSegment] = []
    speaker_set: set[str] = set()
    for turn, _, speaker in exclusive.itertracks(yield_label=True):
        seg_start = turn.start + audio.start_time
        seg_end = turn.end + audio.start_time
        seg_dur = seg_end - seg_start
        speaker_set.add(speaker)
        # Mark as overlap if >= 50% of segment duration falls in an overlap region.
        # Using the same ratio threshold as OverlapHandler._in_overlap() (Task 4)
        # to avoid aggressively excluding matcher reference segments.
        is_ov = False
        if seg_dur > 0:
            for ov_start, ov_end in overlap_regions:
                intersection = max(0.0, min(seg_end, ov_end) - max(seg_start, ov_start))
                if intersection >= seg_dur * 0.5:
                    is_ov = True
                    break
        segments.append(SpeakerSegment(
            speaker_id=speaker,
            start_time=seg_start,
            end_time=seg_end,
            is_overlap=is_ov,
        ))

    # ── Build non-exclusive segments for overlap attribution ───
    non_exclusive_segs: list[SpeakerSegment] = []
    for turn, _, speaker in non_exclusive.itertracks(yield_label=True):
        non_exclusive_segs.append(SpeakerSegment(
            speaker_id=speaker,
            start_time=turn.start + audio.start_time,
            end_time=turn.end + audio.start_time,
        ))

    return DiarizationResult(
        segments=segments,
        num_speakers=len(speaker_set),
        overlap_regions=overlap_regions,
        non_exclusive_segments=non_exclusive_segs,
    )
```

**设计决策:**
- `is_overlap` 判定条件：`seg_start < ov_end and seg_end > ov_start`（存在交集即标记）
- 使用较宽松的交集检测（any overlap）而非 majority，宁可多标也不漏标——因为漏标的后果（错误归因）比多标（多触发一次词级归因）更严重
- Non-exclusive segments 不设 `is_overlap`：它们本身包含重叠信息，该字段无意义

- [ ] **Step 1: 编写 diarizer 新行为测试**

在 `tests/test_diarizer.py` 中添加：

```python
def test_diarizer_marks_is_overlap():
    """Exclusive segments in overlap regions should have is_overlap=True."""
    mock = MagicMock()
    # Exclusive: SPEAKER_00 gets t=0-2 (dominant in overlap), SPEAKER_01 gets t=2-3
    exclusive_tracks = [
        (MockTurn(0.0, 2.0), None, "SPEAKER_00"),
        (MockTurn(2.0, 3.0), None, "SPEAKER_01"),
    ]
    # Non-exclusive: both speakers active in t=0-1.5 overlap
    non_exclusive_tracks = [
        (MockTurn(0.0, 2.0), None, "SPEAKER_00"),
        (MockTurn(0.0, 1.5), None, "SPEAKER_01"),
        (MockTurn(2.0, 3.0), None, "SPEAKER_01"),
    ]
    mock.return_value.exclusive_speaker_diarization.itertracks.return_value = exclusive_tracks
    mock.return_value.speaker_diarization.itertracks.return_value = non_exclusive_tracks

    with patch("transcribe.models.diarizer.Diarizer._load_pipeline", return_value=mock):
        diarizer = Diarizer(device="cpu")
        result = diarizer.process(_make_audio())

    # SPEAKER_00 segment t=0-2 overlaps with SPEAKER_01 t=0-1.5
    # overlap region should be t=0-1.5
    assert len(result.overlap_regions) > 0
    # SPEAKER_00's segment (t=0-2) intersects overlap → is_overlap=True
    s00 = [s for s in result.segments if s.speaker_id == "SPEAKER_00"]
    assert any(s.is_overlap for s in s00)
    # SPEAKER_01's segment (t=2-3) does not intersect overlap → is_overlap=False
    s01 = [s for s in result.segments if s.speaker_id == "SPEAKER_01"]
    assert all(not s.is_overlap for s in s01)


def test_diarizer_stores_non_exclusive_segments():
    """DiarizationResult should contain non-exclusive segments."""
    mock = MagicMock()
    exclusive_tracks = [
        (MockTurn(0.0, 1.5), None, "SPEAKER_00"),
        (MockTurn(1.0, 2.5), None, "SPEAKER_01"),
    ]
    non_exclusive_tracks = [
        (MockTurn(0.0, 1.5), None, "SPEAKER_00"),
        (MockTurn(1.0, 2.5), None, "SPEAKER_01"),
        (MockTurn(2.5, 3.0), None, "SPEAKER_00"),
    ]
    mock.return_value.exclusive_speaker_diarization.itertracks.return_value = exclusive_tracks
    mock.return_value.speaker_diarization.itertracks.return_value = non_exclusive_tracks

    with patch("transcribe.models.diarizer.Diarizer._load_pipeline", return_value=mock):
        diarizer = Diarizer(device="cpu")
        result = diarizer.process(_make_audio())

    assert len(result.non_exclusive_segments) == 3


def test_merge_intervals_does_not_merge_adjacent():
    """Adjacent intervals from different speaker pairs should NOT be merged."""
    intervals = [(1.0, 2.0), (2.0, 3.0)]  # touch at t=2, but not overlap
    result = Diarizer._merge_intervals(intervals)
    assert len(result) == 2
    assert result[0] == (1.0, 2.0)
    assert result[1] == (2.0, 3.0)


def test_merge_intervals_merges_genuine_overlap():
    """Genuinely overlapping intervals should still be merged."""
    intervals = [(1.0, 2.5), (2.0, 3.0)]  # overlap from 2.0-2.5
    result = Diarizer._merge_intervals(intervals)
    assert len(result) == 1
    assert result[0] == (1.0, 3.0)


def test_sweep_line_three_speakers():
    """Three-way overlap should produce a single merged region."""
    mock = MagicMock()
    exclusive_tracks = [
        (MockTurn(0.0, 3.0), None, "SPEAKER_00"),
    ]
    non_exclusive_tracks = [
        (MockTurn(0.0, 3.0), None, "SPEAKER_00"),
        (MockTurn(1.0, 2.0), None, "SPEAKER_01"),
        (MockTurn(1.5, 2.5), None, "SPEAKER_02"),
    ]
    mock.return_value.exclusive_speaker_diarization.itertracks.return_value = exclusive_tracks
    mock.return_value.speaker_diarization.itertracks.return_value = non_exclusive_tracks

    with patch("transcribe.models.diarizer.Diarizer._load_pipeline", return_value=mock):
        diarizer = Diarizer(device="cpu")
        result = diarizer.process(_make_audio())

    # Overlap from 1.0 (2nd speaker enters) to 2.5 (3rd speaker leaves)
    assert len(result.overlap_regions) == 1
    ov_start, ov_end = result.overlap_regions[0]
    assert ov_start == pytest.approx(1.0, abs=0.01)
    assert ov_end == pytest.approx(2.5, abs=0.01)
```

- [ ] **Step 2: 运行新测试确认失败**

Run: `uv run pytest tests/test_diarizer.py::test_diarizer_marks_is_overlap tests/test_diarizer.py::test_diarizer_stores_non_exclusive_segments tests/test_diarizer.py::test_merge_intervals_does_not_merge_adjacent -v`
Expected: FAIL (AttributeError 或 assert False)

- [ ] **Step 3: 实现 diarizer.py 的全部修改**

替换 `process()`, `_extract_overlap_regions()`, `_merge_intervals()` 为上述代码。

- [ ] **Step 4: 运行全部 diarizer 测试**

Run: `uv run pytest tests/test_diarizer.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add transcribe/data/types.py transcribe/models/diarizer.py tests/test_diarizer.py
git commit -m "fix: complete diarization output — non-exclusive segments, is_overlap flag, sweep-line overlap detection, fix interval merging"
```

---

## Group B: Attribution 归因修复

### Task 3: OverlapHandler 使用 non-exclusive segments (P1 #2)

**Files:**
- Modify: `transcribe/models/attribution/overlap.py:57-60`
- Test: `tests/test_attribution.py`

**背景:**
当前 `OverlapHandler._split_by_speaker()` 接收 `diarization.segments`（exclusive），在重叠区域只有一个说话人。改为使用 `diarization.non_exclusive_segments`，使重叠区域内多个说话人的 segments 可见，词级归因才能正确地将词分配给各自说话人。

**修改内容:**

**修改 1: `handle()` — 使用 non-exclusive segments**

```python
# overlap.py: handle() 方法
def handle(
    self,
    segments: list[TranscriptSegment],
    diarization: DiarizationResult,
) -> list[TranscriptSegment]:
    """Process overlap regions and return updated segment list."""
    overlap_regions = diarization.overlap_regions
    if not overlap_regions:
        return segments

    # Use non-exclusive segments for overlap attribution so all speakers
    # in overlap regions are visible for word-level attribution.
    dia_segs = (
        diarization.non_exclusive_segments
        if diarization.non_exclusive_segments
        else diarization.segments
    )

    result: list[TranscriptSegment] = []
    for seg in segments:
        if not self._in_overlap(seg, overlap_regions):
            result.append(seg)
            continue
        result.extend(self._split_by_speaker(seg, dia_segs))

    return result
```

**修改 2: `_split_by_speaker()` — 重叠子段设置低置信度**

在 `_split_by_speaker()` 中创建 `TranscriptSegment` 时，设置 `attribution_confidence=0.0`。
重叠区域的词级归因本身不确定性高（基于交叉时长的投票），不应报告 1.0 的默认置信度。

```python
# 在 _split_by_speaker() 的 sub_segs.append(...) 中添加：
sub_segs.append(
    TranscriptSegment(
        speaker_id=speaker_id,
        start_time=group[0].start_time,
        end_time=group[-1].end_time,
        text=text,
        is_overlap=True,
        words=list(group),
        attribution_confidence=0.0,  # overlap attribution is inherently uncertain
    )
)
```

**修改 3: 更新 OverlapHandler 类 docstring**

```python
class OverlapHandler:
    """Per-speaker subtitle lines for overlap regions.

    For each TranscriptSegment whose temporal overlap with a detected overlap
    region exceeds 50% of its duration, attribute every word to the diarization
    speaker with the most temporal intersection.  Then collect words by speaker
    independently (not linearly), group each speaker's words into temporally
    continuous spans, and build one TranscriptSegment per span.  Segments from
    different speakers may overlap in time, enabling simultaneous display.
    """
```

**设计决策:**
- Fallback 到 `diarization.segments` 当 `non_exclusive_segments` 为空（向后兼容 pyannote 3.x 或旧数据）
- 将 `dia_segs` 提取到 `handle()` 级别，避免每次 `_split_by_speaker()` 调用时重复判断

- [ ] **Step 1: 编写测试 — non-exclusive segments 让重叠区域词级归因正确**

```python
# tests/test_attribution.py
def test_overlap_uses_non_exclusive_segments():
    """OverlapHandler should use non-exclusive segments for word attribution.

    In this scenario, exclusive diarization assigns the entire overlap region
    to SPEAKER_00, but non-exclusive shows both speakers are active.
    Without non-exclusive segments, all words would be attributed to SPEAKER_00.
    """
    words = [
        WordTimestamp("我", 5.0, 5.2),
        WordTimestamp("去", 5.2, 5.4),
        WordTimestamp("不", 5.4, 5.6),
        WordTimestamp("行", 5.6, 5.8),
    ]
    segments = [
        TranscriptSegment(
            speaker_id="SPEAKER_00",
            start_time=5.0,
            end_time=5.8,
            text="我去不行",
            is_overlap=False,
            words=words,
        )
    ]
    diarization = DiarizationResult(
        # Exclusive: entire overlap goes to SPEAKER_00
        segments=[
            SpeakerSegment("SPEAKER_00", 5.0, 5.8),
        ],
        num_speakers=2,
        overlap_regions=[(5.0, 5.8)],
        # Non-exclusive: both speakers visible
        non_exclusive_segments=[
            SpeakerSegment("SPEAKER_00", 5.0, 5.4),
            SpeakerSegment("SPEAKER_01", 5.4, 5.8),
        ],
    )
    result = OverlapHandler().handle(segments, diarization)

    # With non-exclusive segments, words should be split between speakers
    assert len(result) == 2
    s00 = [s for s in result if s.speaker_id == "SPEAKER_00"]
    s01 = [s for s in result if s.speaker_id == "SPEAKER_01"]
    assert len(s00) == 1
    assert s00[0].text == "我去"
    assert len(s01) == 1
    assert s01[0].text == "不行"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_attribution.py::test_overlap_uses_non_exclusive_segments -v`
Expected: FAIL (所有词归给 SPEAKER_00)

- [ ] **Step 3: 修改 `overlap.py` 的 `handle()` 方法**

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add transcribe/models/attribution/overlap.py tests/test_attribution.py
git commit -m "fix: OverlapHandler uses non-exclusive diarization for word-level attribution"
```

---

### Task 4: 改进重叠检测启发式 — 交叉占比替代中心点 (P1 #4)

**Files:**
- Modify: `transcribe/models/attribution/overlap.py:48-55`
- Test: `tests/test_attribution.py`

**背景:**
中心点检测对短片段（<0.5s 反应词）不可靠——片段起止恰好在重叠区域边缘时会漏检。改用交叉时长占片段时长比例（>=50%）作为判据。

**修改内容:**

```python
@staticmethod
def _in_overlap(
    seg: TranscriptSegment,
    overlap_regions: list[tuple[float, float]],
    min_ratio: float = 0.5,
) -> bool:
    """Check if a segment substantially falls within any overlap region.

    Uses intersection-duration ratio instead of center-point: if >= 50%
    of the segment's duration overlaps with an overlap region, the segment
    is treated as overlapping. This is more robust for short interjections
    and segments at overlap region boundaries.
    """
    seg_dur = seg.end_time - seg.start_time
    if seg_dur <= 0:
        return False

    for ov_start, ov_end in overlap_regions:
        intersection = max(
            0.0, min(seg.end_time, ov_end) - max(seg.start_time, ov_start)
        )
        if intersection >= seg_dur * min_ratio:
            return True
    return False
```

**设计决策:**
- 阈值 50%：平衡漏检和误检。>50% 意味着片段大部分在重叠区，应当被处理
- 使用 `>=` 而非 `>` 处理精确 50% 的边界情况
- 短片段优势：一个 0.2s 反应词如果有 0.1s 在重叠区（50%），会被正确捕获。旧的中心点检测对于这类短片段容易漏检

**边缘情况分析:**

| 场景 | 片段 | 重叠区 | 中心点结果 | 交叉占比结果 | 修正？ |
|------|------|--------|-----------|-------------|--------|
| 短片段大部分在重叠区 | 4.9-5.1 (0.2s) | 4.5-5.0 | ❌ center=5.0 不在 [4.5,5.0) | ✅ 0.1/0.2=50% | 是 |
| 长片段小部分在边缘 | 0.0-5.0 (5s) | 4.5-5.0 | ❌ center=2.5 不在 [4.5,5.0) | ❌ 0.5/5=10% | 否（正确） |
| 片段完全在重叠区 | 3.0-4.0 (1s) | 2.0-5.0 | ✅ center=3.5 在 [2.0,5.0) | ✅ 1.0/1.0=100% | 等价 |
| 片段大部分在重叠区 | 1.0-3.0 (2s) | 2.0-4.0 | ✅ center=2.0 在 [2.0,4.0) | ✅ 1.0/2.0=50% | 等价 |

- [ ] **Step 1: 编写测试**

```python
# tests/test_attribution.py
def test_short_segment_at_overlap_boundary():
    """Short segment mostly inside overlap should be detected.

    Old center-point check: center=5.0 NOT in [4.5, 5.0) → missed.
    New intersection-ratio: 0.1/0.2 = 50% >= 50% → detected.
    """
    words = [WordTimestamp("哇", 4.9, 5.1)]
    segments = [
        TranscriptSegment(
            speaker_id="SPEAKER_00",
            start_time=4.9,
            end_time=5.1,
            text="哇",
            words=words,
        )
    ]
    diarization = DiarizationResult(
        segments=[SpeakerSegment("SPEAKER_00", 4.0, 5.5)],
        num_speakers=1,
        overlap_regions=[(4.5, 5.0)],
    )
    result = OverlapHandler().handle(segments, diarization)
    assert result[0].is_overlap is True


def test_long_segment_barely_touching_overlap():
    """Segment with <50% overlap should NOT be treated as overlap."""
    segments = [
        TranscriptSegment(
            speaker_id="SPEAKER_00",
            start_time=0.0,
            end_time=5.0,
            text="很长的一段话",
            words=[WordTimestamp("很", 0.0, 5.0)],
        )
    ]
    diarization = DiarizationResult(
        segments=[SpeakerSegment("SPEAKER_00", 0.0, 5.0)],
        num_speakers=1,
        overlap_regions=[(4.5, 5.0)],  # only 10% of segment
    )
    result = OverlapHandler().handle(segments, diarization)
    assert result[0].is_overlap is False


def test_segment_fully_inside_overlap():
    """Segment fully inside overlap should always be detected."""
    segments = [
        TranscriptSegment(
            speaker_id="SPEAKER_00",
            start_time=3.0,
            end_time=4.0,
            text="测试",
            words=[WordTimestamp("测", 3.0, 3.5), WordTimestamp("试", 3.5, 4.0)],
        )
    ]
    diarization = DiarizationResult(
        segments=[SpeakerSegment("SPEAKER_00", 0.0, 5.0)],
        num_speakers=1,
        overlap_regions=[(2.0, 5.0)],
    )
    result = OverlapHandler().handle(segments, diarization)
    assert result[0].is_overlap is True
```

- [ ] **Step 2: 运行测试确认失败/通过**

Run: `uv run pytest tests/test_attribution.py::test_short_segment_at_overlap_boundary tests/test_attribution.py::test_long_segment_barely_touching_overlap -v`

注意：`test_long_segment_barely_touching_overlap` 在旧实现下会 PASS（center=2.5 不在 [4.5,5.0) 内，返回 False，测试也期望 False——结果一致）。**真正在旧实现下会 FAIL 的是 `test_short_segment_at_overlap_boundary`**（center=5.0 不在 [4.5,5.0) 内但 50% 的片段在重叠区内）。如果两个测试都 PASS，说明旧代码对这两个场景恰好都正确，但仍应替换为交叉占比实现以覆盖更多边界场景。

- [ ] **Step 3: 修改 `_in_overlap()`**

- [ ] **Step 4: 运行全部 attribution 测试**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add transcribe/models/attribution/overlap.py tests/test_attribution.py
git commit -m "fix: replace center-point overlap check with intersection-ratio for short segment reliability"
```

---

### Task 5: 说话人轮次边界分段 (P3 #9)

**Files:**
- Modify: `transcribe/models/attribution/engine.py`
- Test: `tests/test_attribution.py`

**背景:**
`SubtitleSegmenter` 不感知说话人边界。一条字幕可能横跨 A→B 的轮流发言。`TimestampStrategy` 把整条归给 dominant speaker，另一方的贡献被吞没。

**方案:**
在 `AttributionEngine` 中，`TimestampStrategy` 之后、`OverlapHandler` 之前，增加一个 `_split_at_turn_boundaries()` 步骤。该步骤检测每个字幕段内是否包含 diarization 说话人切换点，如有则沿切换点拆分，并**对每个子段重新归因**。

```python
# engine.py — 完整替换
"""Attribution engine — orchestrates strategy + turn splitting + overlap handling."""
from __future__ import annotations

from transcribe.data.types import (
    DiarizationResult,
    SpeakerSegment,
    TranscriptSegment,
    WordTimestamp,
)
from transcribe.models.attribution.overlap import OverlapHandler
from transcribe.models.attribution.strategy import TimestampStrategy


class AttributionEngine:
    """Run speaker attribution on pre-segmented subtitle lines.

    Pipeline:
    1. TimestampStrategy: assign dominant speaker per segment
    2. Turn splitting: split segments that span speaker turn boundaries
    3. OverlapHandler: word-level splitting for overlap regions
    """

    def __init__(self) -> None:
        self._strategy = TimestampStrategy()
        self._overlap_handler = OverlapHandler()

    def run(
        self,
        segments: list[TranscriptSegment],
        diarization: DiarizationResult,
    ) -> list[TranscriptSegment]:
        """Assign speakers to segments, split at turns and overlaps."""
        segments = self._strategy.attribute(segments, diarization)
        segments = self._split_at_turn_boundaries(segments, diarization.segments)
        segments = self._overlap_handler.handle(segments, diarization)
        return segments

    def _split_at_turn_boundaries(
        self,
        segments: list[TranscriptSegment],
        dia_segs: list[SpeakerSegment],
    ) -> list[TranscriptSegment]:
        """Split subtitle segments at speaker turn boundaries.

        When a subtitle segment spans a point where the diarized speaker
        changes, split it so each sub-segment is attributed to its correct
        speaker. This prevents a single subtitle line from containing
        two different speakers' words.
        """
        if not dia_segs or len(dia_segs) <= 1:
            return segments

        # Collect all turn boundaries (where speaker changes)
        boundaries: list[float] = []
        for i in range(1, len(dia_segs)):
            if dia_segs[i].speaker_id != dia_segs[i - 1].speaker_id:
                boundaries.append(dia_segs[i].start_time)

        if not boundaries:
            return segments

        result: list[TranscriptSegment] = []
        for seg in segments:
            # Find boundaries that fall within this segment
            inner_bounds = [
                b for b in boundaries
                if seg.start_time < b < seg.end_time
            ]

            if not inner_bounds:
                result.append(seg)
                continue

            # Split segment at each boundary, re-attributing each sub-segment
            if not seg.words:
                result.extend(
                    self._split_segment_without_words(seg, inner_bounds, dia_segs)
                )
                continue

            result.extend(self._split_segment_with_words(seg, inner_bounds, dia_segs))

        return result

    @staticmethod
    def _split_segment_with_words(
        seg: TranscriptSegment,
        boundaries: list[float],
        dia_segs: list[SpeakerSegment],
    ) -> list[TranscriptSegment]:
        """Split a segment with word timestamps at turn boundaries.

        Words are assigned to sub-segments by midpoint. Each sub-segment
        is re-attributed to the dominant diarization speaker in its time range.
        """
        splits = [seg.start_time] + boundaries + [seg.end_time]
        sub_segments: list[TranscriptSegment] = []

        for i in range(len(splits) - 1):
            sub_start, sub_end = splits[i], splits[i + 1]

            # Partition words by midpoint — handles straddling words correctly.
            # A word [1.7, 1.9] at boundary 1.8 has midpoint 1.8 → falls in
            # sub [1.8, ...] (the `<=` side of the boundary).
            sub_words = [
                w for w in seg.words
                if sub_start <= (w.start_time + w.end_time) / 2 < sub_end
            ]

            # Handle words exactly at the last split point's end
            if i == len(splits) - 2:
                sub_words = [
                    w for w in seg.words
                    if sub_start <= (w.start_time + w.end_time) / 2 <= sub_end
                ]

            if not sub_words:
                continue

            # Re-attribute: find dominant speaker in [sub_start, sub_end]
            best_speaker = seg.speaker_id
            best_overlap = 0.0
            for dia in dia_segs:
                ov = max(0.0, min(sub_end, dia.end_time) - max(sub_start, dia.start_time))
                if ov > best_overlap:
                    best_overlap = ov
                    best_speaker = dia.speaker_id

            text = "".join(w.word for w in sub_words)
            sub_segments.append(TranscriptSegment(
                speaker_id=best_speaker,
                start_time=sub_words[0].start_time,
                end_time=sub_words[-1].end_time,
                text=text,
                is_overlap=seg.is_overlap,
                words=sub_words,
                attribution_confidence=seg.attribution_confidence,
            ))

        return sub_segments

    @staticmethod
    def _split_segment_without_words(
        seg: TranscriptSegment,
        boundaries: list[float],
        dia_segs: list[SpeakerSegment],
    ) -> list[TranscriptSegment]:
        """Split a segment without word timestamps at turn boundaries."""
        splits = [seg.start_time] + boundaries + [seg.end_time]
        sub_segments: list[TranscriptSegment] = []

        for i in range(len(splits) - 1):
            sub_start, sub_end = splits[i], splits[i + 1]
            # Find dominant speaker for this sub-segment
            best_speaker = seg.speaker_id
            best_overlap = 0.0
            for dia in dia_segs:
                ov = max(0.0, min(sub_end, dia.end_time) - max(sub_start, dia.start_time))
                if ov > best_overlap:
                    best_overlap = ov
                    best_speaker = dia.speaker_id

            sub_segments.append(TranscriptSegment(
                speaker_id=best_speaker,
                start_time=sub_start,
                end_time=sub_end,
                text=seg.text,  # can't split text without words
                is_overlap=seg.is_overlap,
                attribution_confidence=seg.attribution_confidence,
            ))

        return sub_segments
```

**设计决策:**
- `_split_segment_with_words` **必须接收 `dia_segs` 并对每个子段做交叉投票重新归因**。不能继承父段的 `speaker_id`——OverlapHandler 只处理重叠区域，非重叠区域的子段不会被修正
- 词分配使用**中点归属法**（`midpoint >= sub_start and midpoint < sub_end`）：跨边界的词 [1.7, 1.9] 在边界 1.8 处的中点为 1.8，归入右侧子段。这避免了"两个子段都不包含该词"的静默丢失
- 最后一个子段使用 `<=` 闭合上限，确保尾部词不被丢弃
- 两个拆分方法都继承父段的 `attribution_confidence`

**局限性:**
- 无词级时间戳时，`text` 无法被拆分，所有子段共享原文。这是一个已知降级——需要词级时间戳才能正确拆分文本

- [ ] **Step 1: 编写测试**

```python
# tests/test_attribution.py
class TestTurnBoundarySplitting:
    def test_segment_spanning_two_speakers_is_split(self):
        """A segment spanning A→B turn boundary should be split."""
        words = [
            WordTimestamp("大", 0.0, 0.5),
            WordTimestamp("家", 0.5, 1.0),
            WordTimestamp("好", 1.0, 1.5),
            WordTimestamp("谢", 2.0, 2.5),
            WordTimestamp("谢", 2.5, 3.0),
        ]
        segments = [
            TranscriptSegment(
                speaker_id="SPEAKER_00",  # will be overwritten
                start_time=0.0,
                end_time=3.0,
                text="大家好谢谢",
                words=words,
            )
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 1.8),
                SpeakerSegment("SPEAKER_01", 1.8, 4.0),
            ],
            num_speakers=2,
        )
        engine = AttributionEngine()
        result = engine.run(segments, diarization)

        # Should be split into two segments at the turn boundary (t=1.8)
        assert len(result) == 2
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].text == "大家好"
        assert result[1].speaker_id == "SPEAKER_01"
        assert result[1].text == "谢谢"

    def test_no_turn_boundary_no_split(self):
        """Segment entirely within one speaker's turn should not be split."""
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 1.0, "你好"),
        ]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 2.0)],
            num_speakers=1,
        )
        engine = AttributionEngine()
        result = engine.run(segments, diarization)
        assert len(result) == 1
```

- [ ] **Step 2: 运行测试确认失败**

- [ ] **Step 3: 修改 `engine.py`**

- [ ] **Step 4: 运行全部 attribution 测试**

Run: `uv run pytest tests/test_attribution.py -v`

- [ ] **Step 5: Commit**

```bash
git add transcribe/models/attribution/engine.py tests/test_attribution.py
git commit -m "feat: split subtitle segments at speaker turn boundaries"
```

---

### Task 6: 归因置信度评分 (P3 #11)

**Files:**
- Modify: `transcribe/data/types.py:48-56` (TranscriptSegment)
- Modify: `transcribe/models/attribution/strategy.py:48-70`
- Modify: `transcribe/models/srt_writer.py` (可选：输出置信度)
- Test: `tests/test_attribution.py`

**背景:**
当交叉投票的胜出优势很小时（如 SPEAKER_00 0.51s vs SPEAKER_01 0.49s），归因结果不可靠。添加置信度分数可帮助后期人工审校定位需要检查的字幕行。

**修改 1: TranscriptSegment 添加置信度字段**

```python
# types.py
@dataclass
class TranscriptSegment:
    """A transcribed segment."""

    speaker_id: str
    start_time: float
    end_time: float
    text: str
    is_overlap: bool = False
    words: list[WordTimestamp] | None = None
    attribution_confidence: float = 1.0  # NEW: 0.0-1.0, speaker attribution confidence
```

**修改 2: TimestampStrategy 计算置信度**

```python
# strategy.py — 修改 _assign_segment 返回 (speaker, confidence)
def attribute(
    self,
    segments: list[TranscriptSegment],
    diarization: DiarizationResult,
) -> list[TranscriptSegment]:
    """Assign a speaker to each segment based on temporal overlap."""
    if not segments:
        return []

    dia_segs = diarization.segments
    result: list[TranscriptSegment] = []

    for seg in segments:
        speaker, confidence = self._assign_segment(seg, dia_segs)
        result.append(replace(seg, speaker_id=speaker, attribution_confidence=confidence))

    return result

def _assign_segment(
    self, segment: TranscriptSegment, dia_segs: list[SpeakerSegment]
) -> tuple[str, float]:
    """Find the dominant speaker and confidence score.

    Returns:
        (speaker_id, confidence) where confidence is the winner's overlap
        divided by (winner + runner-up) overlap. 1.0 when unambiguous.
    """
    if not dia_segs:
        return "SPEAKER_00", 0.0

    speaker_overlap: dict[str, float] = {}

    for dia in dia_segs:
        intersection = (
            min(segment.end_time, dia.end_time)
            - max(segment.start_time, dia.start_time)
        )
        if intersection > 0:
            speaker_overlap[dia.speaker_id] = (
                speaker_overlap.get(dia.speaker_id, 0.0) + intersection
            )

    if not speaker_overlap:
        speaker = self._nearest_speaker(segment, dia_segs)
        return speaker, 0.0  # no overlap evidence → low confidence

    sorted_speakers = sorted(speaker_overlap.items(), key=lambda x: x[1], reverse=True)
    winner_id, winner_overlap = sorted_speakers[0]

    if len(sorted_speakers) == 1:
        confidence = 1.0
    else:
        runner_up_overlap = sorted_speakers[1][1]
        total = winner_overlap + runner_up_overlap
        confidence = winner_overlap / total if total > 0 else 1.0

    return winner_id, confidence
```

**置信度语义:**
- `1.0`：只有一个说话人在该时间范围内活动，归因明确
- `0.5`：两个说话人交叉时长完全相等，完全不确定
- `0.0`：没有任何交叉（使用了最近说话人回退），最不可靠

**设计决策:**
- 置信度 = winner_overlap / (winner + runner_up)。这是最直接的"胜出优势"度量
- 不使用 softmax 或归一化到所有说话人——只比较前两名，其余说话人已经输了
- 默认值 `1.0` 确保向后兼容

- [ ] **Step 1: 编写测试**

```python
# tests/test_attribution.py
class TestAttributionConfidence:
    def test_single_speaker_high_confidence(self):
        """Segment fully within one speaker → confidence 1.0."""
        segments = [TranscriptSegment("SPEAKER_00", 0.5, 2.0, "你好")]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 3.0)],
            num_speakers=1,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert result[0].attribution_confidence == 1.0

    def test_equal_overlap_low_confidence(self):
        """Equal overlap with two speakers → confidence ~0.5."""
        segments = [TranscriptSegment("SPEAKER_00", 0.0, 2.0, "你好")]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 1.0),
                SpeakerSegment("SPEAKER_01", 1.0, 2.0),
            ],
            num_speakers=2,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert 0.45 <= result[0].attribution_confidence <= 0.55

    def test_fallback_zero_confidence(self):
        """No overlap, nearest-speaker fallback → confidence 0.0."""
        segments = [TranscriptSegment("SPEAKER_00", 1.2, 1.3, "嗯")]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 1.0),
                SpeakerSegment("SPEAKER_01", 2.0, 3.0),
            ],
            num_speakers=2,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert result[0].attribution_confidence == 0.0

    def test_dominant_speaker_high_confidence(self):
        """75/25 split → confidence ~0.75."""
        segments = [TranscriptSegment("SPEAKER_00", 0.0, 4.0, "你好世界")]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 3.0),
                SpeakerSegment("SPEAKER_01", 3.0, 4.0),
            ],
            num_speakers=2,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert result[0].attribution_confidence == pytest.approx(0.75, abs=0.01)
```

- [ ] **Step 2: 运行测试确认失败**

- [ ] **Step 3: 修改 types.py 和 strategy.py**

- [ ] **Step 4: 运行全部测试**

Run: `uv run pytest tests/ -v`

- [ ] **Step 5: Commit**

```bash
git add transcribe/data/types.py transcribe/models/attribution/strategy.py tests/test_attribution.py
git commit -m "feat: add attribution confidence score to TranscriptSegment"
```

---

## Group C: Pipeline 级改进

### Task 7: 词级时间戳可用性验证与降级警告 (P2 #6)

**Files:**
- Modify: `transcribe/pipeline.py:75-84`
- Modify: `transcribe/models/attribution/overlap.py:69-71`

**背景:**
如果 ASR 后端未产出有效的词级时间戳，`OverlapHandler` 的降级路径几乎不做事。应在管线层面提前检测并发出警告。

**修改 1: pipeline.py — ASR 后检测词级时间戳质量**

```python
# pipeline.py — 在 Stage 2 之后添加验证
# ── Stage 2: ASR on full audio ──────────────────────────────────
step += 1
step_start = time.time()
if verbose:
    console.print(f"[{step}/{total_stages}] 语音转文字（全音频）...", end=" ")
transcriber = create_asr(config.backend, device=device, hotword_path=config.hotwords)
words = transcriber.transcribe_words(audio)
transcriber.cleanup()

# Validate word-level timestamp quality for diarization pipeline
if config.diarize and words:
    zero_dur_count = sum(1 for w in words if abs(w.end_time - w.start_time) < 1e-6)
    zero_ratio = zero_dur_count / len(words) if words else 0
    if zero_ratio > 0.8:
        if verbose:
            console.print(
                f"\n[bold yellow]警告: {zero_dur_count}/{len(words)} 个词的时长为零，"
                "词级时间戳可能无效。重叠区域的说话人归因将严重受限。[/bold yellow]"
            )
    elif verbose:
        console.print(f"识别 {len(words)} 个词 ... 完成 ({time.time() - step_start:.1f}s)")
elif config.diarize and not words:
    if verbose:
        console.print(
            "\n[bold yellow]警告: ASR 未产出任何词级时间戳，"
            "重叠处理和说话人归因将不可用。[/bold yellow]"
        )
elif verbose:
    console.print(f"识别 {len(words)} 个词 ... 完成 ({time.time() - step_start:.1f}s)")
```

**修改 2: overlap.py — 降级路径添加 logging**

```python
# overlap.py — 在文件顶部添加
import logging

_logger = logging.getLogger(__name__)

# 在 _split_by_speaker 的 degraded path:
if not seg.words:
    _logger.warning(
        "Overlap segment at %.2f-%.2f has no word timestamps, "
        "cannot split by speaker. Text: '%s'",
        seg.start_time, seg.end_time, seg.text[:50],
    )
    return [replace(seg, is_overlap=True)]
```

**设计决策:**
- 使用 `logging` 而非 `print` — 不污染 CLI 输出，但可被日志系统捕获
- 阈值 80% 零时长词：给一些容错（少数零时长词是正常的，如标点）
- Pipeline 级的检测在 verbose 模式下向用户展示，模块级的 logging 面向开发者

- [ ] **Step 1: 修改 pipeline.py 添加验证逻辑**

- [ ] **Step 2: 修改 overlap.py 添加 logging**

- [ ] **Step 3: 运行全部测试确保无回归**

Run: `uv run pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 4: Commit**

```bash
git add transcribe/pipeline.py transcribe/models/attribution/overlap.py
git commit -m "feat: add word-level timestamp validation and overlap degradation warnings"
```

---

## 自检清单

### 1. 规格覆盖
- [x] P1 #2: Non-exclusive diarization → Task 2 (diarizer) + Task 3 (overlap handler)
- [x] P1 #3: is_overlap 死字段 → Task 2 (diarizer) 中标记 is_overlap（50% 交叉占比阈值）
- [x] P1 #4: 重叠检测启发式 → Task 4 (intersection-ratio)
- [x] P2 #8: 激进合并 → Task 2 (merge_intervals 修复)
- [x] P2 #6: 词级时间戳警告 → Task 7
- [x] P3 #10: O(n²) → Task 2 (sweep-line)
- [x] P3 #9: 说话人边界分段 → Task 5
- [x] P3 #11: 置信度评分 → Task 6

### 2. 占位符扫描
无 TBD、TODO、"implement later"、"add validation" 等占位符。所有步骤包含完整代码。

### 3. 类型一致性
- `DiarizationResult.non_exclusive_segments: list[SpeakerSegment]` — Task 1 定义，Task 2 填充，Task 3 消费 ✓
- `TranscriptSegment.attribution_confidence: float = 1.0` — Task 6 定义，Task 5/6/7 消费 ✓
- `OverlapHandler._in_overlap(min_ratio=0.5)` — Task 4 定义，Task 4 使用 ✓
- `AttributionEngine._split_at_turn_boundaries()` — Task 5 定义，`run()` 调用 ✓
- `AttributionEngine._split_segment_with_words(seg, boundaries, dia_segs)` — Task 5 定义，调用时传入 `dia_segs` ✓

### 4. 审查修正记录（Oracle Review Fixes）

**Critical Issues (已修复):**
- **C1 修复**: `_split_segment_with_words` 现在接收 `dia_segs` 参数，对每个子段做交叉投票重新归因，不再盲目继承父段 `speaker_id`
- **C2 修复**: 词分配改为中点归属法 `sub_start <= midpoint < sub_end`，跨边界的词 [1.7,1.9] 在边界 1.8 处中点=1.8 归入右侧子段，不再被静默丢弃

**Warnings (已处理):**
- **W1**: Sweep-line 增加 `if speaker not in active_speakers: continue` guard，防止零时长 turn 的 end 事件污染计数
- **W2**: `is_overlap` 标记改用 50% 交叉占比阈值（而非"any intersection"），避免过长地排除 matcher 参考段
- **W3**: Task 4 TDD 步骤描述已修正，正确指出 `test_short_segment_at_overlap_boundary` 是旧实现下会 FAIL 的测试
- **W4**: 已知限制，不影响功能（下游 `_in_overlap` 逐个检查），可接受
- **W5**: 已知限制，warning 场景下不显示计时信息是合理的

**Suggestions (已采纳):**
- **S2**: 添加 3-speaker sweep-line 测试 (`test_sweep_line_three_speakers`)
- **S3**: OverlapHandler `_split_by_speaker()` 创建的重叠子段设置 `attribution_confidence=0.0`
- **S5**: OverlapHandler 类 docstring 已更新为反映 intersection-ratio 逻辑
