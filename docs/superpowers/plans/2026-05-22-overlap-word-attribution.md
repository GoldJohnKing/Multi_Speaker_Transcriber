# Overlap Word-Level Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the passive overlap marking with word-level speaker attribution that splits overlap regions into per-speaker subtitle entries.

**Architecture:** The `OverlapHandler` in `overlap.py` is rewritten to, for each overlap-region segment, attribute each word to the diarization speaker with the most temporal intersection, then group consecutive same-speaker words into separate `TranscriptSegment` entries. The `AttributionEngine.run()` API is simplified to accept `DiarizationResult` directly (overlap_regions are already inside it).

**Tech Stack:** Python 3.12, dataclasses, pytest

---

### Task 1: Rewrite `OverlapHandler` in `overlap.py`

**Files:**
- Rewrite: `transcribe/models/attribution/overlap.py`

- [ ] **Step 1: Write the new `OverlapHandler` implementation**

Replace the entire file content. The old `OverlapHandler` ABC and `MarkOverlapHandler` are removed. The new class does word-level speaker attribution for overlap segments.

```python
"""Overlap handling — word-level speaker attribution for overlap regions."""
from __future__ import annotations

from dataclasses import replace

from transcribe.data.types import (
    DiarizationResult,
    SpeakerSegment,
    TranscriptSegment,
    WordTimestamp,
)


class OverlapHandler:
    """Split overlap-region segments into per-speaker sub-segments.

    For each TranscriptSegment whose center falls in a detected overlap
    region, attribute every word to the diarization speaker with the most
    temporal intersection, then group consecutive same-speaker words into
    independent TranscriptSegments.
    """

    def handle(
        self,
        segments: list[TranscriptSegment],
        diarization: DiarizationResult,
    ) -> list[TranscriptSegment]:
        """Process overlap regions and return updated segment list."""
        overlap_regions = diarization.overlap_regions
        if not overlap_regions:
            return segments

        result: list[TranscriptSegment] = []
        for seg in segments:
            if not self._in_overlap(seg, overlap_regions):
                result.append(seg)
                continue
            result.extend(self._split_by_speaker(seg, diarization.segments))

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _in_overlap(
        seg: TranscriptSegment,
        overlap_regions: list[tuple[float, float]],
    ) -> bool:
        """Center-point check (same logic as old MarkOverlapHandler)."""
        center = (seg.start_time + seg.end_time) / 2.0
        return any(s <= center < e for s, e in overlap_regions)

    def _split_by_speaker(
        self,
        seg: TranscriptSegment,
        dia_segs: list[SpeakerSegment],
    ) -> list[TranscriptSegment]:
        """Word-level attribution → group by speaker → build sub-segments."""
        # Degraded path: no word-level timestamps
        if not seg.words:
            return [replace(seg, is_overlap=True)]

        # 1. Attribute each word to a speaker
        attributed: list[tuple[WordTimestamp, str]] = []
        for w in seg.words:
            speaker = self._attribute_word(w, dia_segs)
            attributed.append((w, speaker))

        # 2. Group consecutive same-speaker words
        groups = self._group_consecutive(attributed)

        # 3. Build one TranscriptSegment per group
        sub_segs: list[TranscriptSegment] = []
        for group in groups:
            words = [w for w, _ in group]
            speaker_id = group[0][1]
            text = "".join(w.word for w in words)
            sub_segs.append(
                TranscriptSegment(
                    speaker_id=speaker_id,
                    start_time=words[0].start_time,
                    end_time=words[-1].end_time,
                    text=text,
                    is_overlap=True,
                    words=words,
                )
            )
        return sub_segs

    @staticmethod
    def _attribute_word(
        word: WordTimestamp,
        dia_segs: list[SpeakerSegment],
    ) -> str:
        """Assign word to the speaker with the most temporal intersection."""
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0
        for dia in dia_segs:
            overlap_start = max(word.start_time, dia.start_time)
            overlap_end = min(word.end_time, dia.end_time)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dia.speaker_id
        return best_speaker

    @staticmethod
    def _group_consecutive(
        attributed: list[tuple[WordTimestamp, str]],
    ) -> list[list[tuple[WordTimestamp, str]]]:
        """Group consecutive (word, speaker) pairs by speaker identity."""
        if not attributed:
            return []
        groups: list[list[tuple[WordTimestamp, str]]] = []
        current = [attributed[0]]
        for item in attributed[1:]:
            if item[1] == current[-1][1]:
                current.append(item)
            else:
                groups.append(current)
                current = [item]
        groups.append(current)
        return groups
```

- [ ] **Step 2: Verify the file is syntactically correct**

Run: `python -c "from transcribe.models.attribution.overlap import OverlapHandler; print('OK')"`
Expected: `OK`

---

### Task 2: Update `AttributionEngine` in `engine.py`

**Files:**
- Modify: `transcribe/models/attribution/engine.py`

- [ ] **Step 1: Rewrite `engine.py`**

Simplify the API: `run()` now takes `diarization` only (overlap_regions are inside it). The new `OverlapHandler` replaces `MarkOverlapHandler`.

```python
"""Attribution engine — orchestrates strategy + overlap handling."""
from __future__ import annotations

from transcribe.data.types import (
    DiarizationResult,
    TranscriptSegment,
)
from transcribe.models.attribution.overlap import OverlapHandler
from transcribe.models.attribution.strategy import TimestampStrategy


class AttributionEngine:
    """Run speaker attribution on pre-segmented subtitle lines.

    Takes subtitle segments (from SubtitleSegmenter) and diarization
    output, assigns a dominant speaker to each segment via interval
    intersection voting, then applies word-level overlap splitting.
    """

    def __init__(self) -> None:
        self._strategy = TimestampStrategy()
        self._overlap_handler = OverlapHandler()

    def run(
        self,
        segments: list[TranscriptSegment],
        diarization: DiarizationResult,
    ) -> list[TranscriptSegment]:
        """Assign speakers to segments and handle overlaps."""
        segments = self._strategy.attribute(segments, diarization)
        segments = self._overlap_handler.handle(segments, diarization)
        return segments
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from transcribe.models.attribution.engine import AttributionEngine; print('OK')"`
Expected: `OK`

---

### Task 3: Update `pipeline.py` call site

**Files:**
- Modify: `transcribe/pipeline.py` (lines 97–118)

- [ ] **Step 1: Update the diarization+attribution block**

The `engine.run()` call changes from `engine.run(all_segments, diarization, overlap_regions)` to `engine.run(all_segments, diarization)`. The separate `overlap_regions` variable is no longer needed for the engine call.

In `transcribe/pipeline.py`, replace lines 96–118 (the Stage 4 block) with:

```python
    # ── Stage 4: Speaker diarization + attribution ──────────────────
    diarization: DiarizationResult | None = None
    if config.diarize:
        step += 1
        step_start = time.time()
        if verbose:
            console.print(f"[{step}/{total_stages}] 说话人识别 + 归因 ...", end=" ")

        diarizer = Diarizer(device=device, num_speakers=config.num_speakers)
        diarization = diarizer.process(audio)
        diarizer.cleanup()

        engine = AttributionEngine()
        all_segments = engine.run(all_segments, diarization)

        if verbose:
            console.print(
                f"检测到 {diarization.num_speakers} 位说话人, "
                f"{len(diarization.overlap_regions)} 个重叠区域 ... "
                f"完成 ({time.time() - step_start:.1f}s)"
            )
```

Key changes:
- Remove `overlap_regions: list[tuple[float, float]] = []` (line 98)
- Remove `overlap_regions = diarization.overlap_regions` (line 107)
- Change `engine.run(all_segments, diarization, overlap_regions)` → `engine.run(all_segments, diarization)` (line 111)
- Change `len(overlap_regions)` → `len(diarization.overlap_regions)` in the verbose print (line 116)

- [ ] **Step 2: Verify pipeline imports work**

Run: `python -c "from transcribe.pipeline import run_pipeline; print('OK')"`
Expected: `OK`

---

### Task 4: Update tests

**Files:**
- Modify: `tests/test_attribution.py`

- [ ] **Step 1: Rewrite tests to match new API**

The test file needs these changes:
1. Remove `MarkOverlapHandler` import, add `OverlapHandler` import
2. Rename `TestMarkOverlapHandler` to `TestOverlapHandler` and update tests
3. Update `TestAttributionEngine` tests (simpler `engine.run()` API)
4. Add new tests for word-level attribution splitting

Replace the entire `tests/test_attribution.py`:

```python
"""Tests for the speaker attribution module."""
from __future__ import annotations

import pytest

from transcribe.data.types import (
    DiarizationResult,
    SpeakerSegment,
    TranscriptSegment,
    WordTimestamp,
)
from transcribe.models.attribution.engine import AttributionEngine
from transcribe.models.attribution.overlap import OverlapHandler
from transcribe.models.attribution.strategy import TimestampStrategy


class TestTimestampStrategyBasic:
    def test_single_segment_single_speaker(self) -> None:
        segments = [
            TranscriptSegment("SPEAKER_00", 0.5, 2.0, "你好世界"),
        ]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 3.0)],
            num_speakers=1,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert len(result) == 1
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].text == "你好世界"

    def test_two_segments_two_speakers(self) -> None:
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 1.0, "你好"),
            TranscriptSegment("SPEAKER_00", 2.0, 3.0, "世界"),
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 1.5),
                SpeakerSegment("SPEAKER_01", 1.5, 3.0),
            ],
            num_speakers=2,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert len(result) == 2
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[1].speaker_id == "SPEAKER_01"

    def test_dominant_speaker_by_intersection(self) -> None:
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 2.0, "你好世界"),
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 1.5),
                SpeakerSegment("SPEAKER_01", 1.5, 3.0),
            ],
            num_speakers=2,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert result[0].speaker_id == "SPEAKER_00"

    def test_fallback_to_nearest_when_no_overlap(self) -> None:
        segments = [
            TranscriptSegment("SPEAKER_00", 1.2, 1.3, "嗯"),
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 1.0),
                SpeakerSegment("SPEAKER_01", 2.0, 3.0),
            ],
            num_speakers=2,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert len(result) == 1
        assert result[0].speaker_id == "SPEAKER_00"

    def test_empty_segments(self) -> None:
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 1.0)],
            num_speakers=1,
        )
        result = TimestampStrategy().attribute([], diarization)
        assert result == []

    def test_empty_diarization(self) -> None:
        segments = [TranscriptSegment("SPEAKER_00", 0.0, 1.0, "你好")]
        diarization = DiarizationResult(segments=[], num_speakers=0)
        result = TimestampStrategy().attribute(segments, diarization)
        assert result[0].speaker_id == "SPEAKER_00"

    def test_multiple_segments_preserved(self) -> None:
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 1.0, "你好"),
            TranscriptSegment("SPEAKER_00", 1.0, 2.0, "世界"),
            TranscriptSegment("SPEAKER_00", 2.0, 3.0, "再见"),
        ]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 3.0)],
            num_speakers=1,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert len(result) == 3
        assert [s.text for s in result] == ["你好", "世界", "再见"]

    def test_timing_preserved(self) -> None:
        segments = [TranscriptSegment("SPEAKER_00", 0.5, 1.5, "你好")]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 2.0)],
            num_speakers=1,
        )
        result = TimestampStrategy().attribute(segments, diarization)
        assert result[0].start_time == pytest.approx(0.5)
        assert result[0].end_time == pytest.approx(1.5)


class TestOverlapHandler:
    def test_no_overlap_regions_returns_unchanged(self) -> None:
        segments = [TranscriptSegment("SPEAKER_00", 0.0, 1.0, "你好")]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 1.0)],
            num_speakers=1,
        )
        result = OverlapHandler().handle(segments, diarization)
        assert len(result) == 1
        assert result[0].is_overlap is False

    def test_overlap_no_words_marks_flag(self) -> None:
        """Degraded path: segment in overlap but no word timestamps."""
        segments = [TranscriptSegment("SPEAKER_00", 1.0, 2.0, "你好")]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 3.0),
                SpeakerSegment("SPEAKER_01", 0.0, 3.0),
            ],
            num_speakers=2,
            overlap_regions=[(0.5, 2.5)],
        )
        result = OverlapHandler().handle(segments, diarization)
        assert len(result) == 1
        assert result[0].is_overlap is True

    def test_overlap_splits_by_speaker(self) -> None:
        """Two speakers overlap; words attributed and split into sub-segments."""
        words = [
            WordTimestamp("这", 10.0, 10.1),
            WordTimestamp("个", 10.1, 10.2),
            WordTimestamp("对", 10.3, 10.4),
            WordTimestamp("好", 10.5, 10.6),
        ]
        segments = [
            TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=10.0,
                end_time=10.6,
                text="这个对好",
                is_overlap=False,
                words=words,
            )
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 10.0, 10.3),
                SpeakerSegment("SPEAKER_01", 10.3, 10.7),
            ],
            num_speakers=2,
            overlap_regions=[(10.0, 10.6)],
        )
        result = OverlapHandler().handle(segments, diarization)

        # Should produce 2 sub-segments (one per speaker)
        assert len(result) == 2
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].text == "这个"
        assert result[0].start_time == pytest.approx(10.0)
        assert result[0].end_time == pytest.approx(10.2)
        assert result[0].is_overlap is True

        assert result[1].speaker_id == "SPEAKER_01"
        assert result[1].text == "对好"
        assert result[1].start_time == pytest.approx(10.3)
        assert result[1].end_time == pytest.approx(10.6)
        assert result[1].is_overlap is True

    def test_non_overlap_segment_passes_through(self) -> None:
        """Segment outside overlap region is unchanged."""
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 1.0, "你好"),
            TranscriptSegment("SPEAKER_00", 3.0, 4.0, "世界"),
        ]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 5.0)],
            num_speakers=1,
            overlap_regions=[(1.5, 2.5)],
        )
        result = OverlapHandler().handle(segments, diarization)
        assert len(result) == 2
        assert all(s.is_overlap is False for s in result)

    def test_all_words_same_speaker_produces_one_segment(self) -> None:
        """All words in overlap attributed to same speaker → one sub-segment."""
        words = [
            WordTimestamp("你", 1.0, 1.1),
            WordTimestamp("好", 1.1, 1.2),
        ]
        segments = [
            TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=1.0,
                end_time=1.2,
                text="你好",
                words=words,
            )
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 3.0),
                SpeakerSegment("SPEAKER_01", 0.0, 0.5),  # barely touches
            ],
            num_speakers=2,
            overlap_regions=[(0.5, 2.5)],
        )
        result = OverlapHandler().handle(segments, diarization)
        assert len(result) == 1
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].text == "你好"
        assert result[0].is_overlap is True


class TestAttributionEngine:
    def test_full_attribution_flow(self) -> None:
        segments = [
            TranscriptSegment("SPEAKER_00", 0.0, 0.4, "你好"),
            TranscriptSegment("SPEAKER_00", 1.0, 1.4, "世界"),
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0, 0.8),
                SpeakerSegment("SPEAKER_01", 0.8, 2.0),
            ],
            num_speakers=2,
            overlap_regions=[(0.1, 0.3)],
        )
        engine = AttributionEngine()
        result = engine.run(segments, diarization)

        assert len(result) == 2
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].text == "你好"
        assert result[0].is_overlap is True  # center=0.2 in [0.1, 0.3)
        assert result[1].speaker_id == "SPEAKER_01"
        assert result[1].text == "世界"
        assert result[1].is_overlap is False

    def test_no_overlap_regions(self) -> None:
        segments = [TranscriptSegment("SPEAKER_00", 0.0, 0.5, "你好")]
        diarization = DiarizationResult(
            segments=[SpeakerSegment("SPEAKER_00", 0.0, 1.0)],
            num_speakers=1,
        )
        engine = AttributionEngine()
        result = engine.run(segments, diarization)
        assert result[0].is_overlap is False

    def test_overlap_with_word_splitting(self) -> None:
        """Integration: engine splits overlap by word-level attribution."""
        words = [
            WordTimestamp("这", 10.0, 10.1),
            WordTimestamp("个", 10.1, 10.2),
            WordTimestamp("好", 10.3, 10.4),
        ]
        segments = [
            TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=10.0,
                end_time=10.4,
                text="这个好",
                words=words,
            )
        ]
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 10.0, 10.25),
                SpeakerSegment("SPEAKER_01", 10.25, 10.5),
            ],
            num_speakers=2,
            overlap_regions=[(10.0, 10.4)],
        )
        engine = AttributionEngine()
        result = engine.run(segments, diarization)

        # Should split into 2 sub-segments
        assert len(result) == 2
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].text == "这个"
        assert result[1].speaker_id == "SPEAKER_01"
        assert result[1].text == "好"
```

- [ ] **Step 2: Run tests and verify all pass**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: All tests PASS

---

### Task 5: Run full test suite and commit

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 2: Commit all changes**

```bash
git add transcribe/models/attribution/overlap.py transcribe/models/attribution/engine.py transcribe/pipeline.py tests/test_attribution.py
git commit -m "feat: word-level speaker attribution for overlap regions

Replace passive overlap marking with word-level attribution that splits
overlap-region segments into per-speaker subtitle entries. Each word is
attributed to the diarization speaker with the most temporal intersection,
then consecutive same-speaker words are grouped into independent
TranscriptSegments."
```
