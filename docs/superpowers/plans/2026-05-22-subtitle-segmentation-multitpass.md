# 字幕分段多趟管线实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `SubtitleSegmenter` 从单趟贪心算法重构为 5 趟管线（句末标点切 → 间隙切 → 短合并 → 超限修正 → CPS 校验），并将 `segment_by_timestamps()` 改为委托适配器。

**Architecture:** 所有中间 Pass 操作统一的 `list[list[WordTimestamp]]` 表示（含逗号词），标点在最终构建 `TranscriptSegment` 时一次性剥离。逗号保留在词流中用于 Pass 4 的评分定位。`segment_by_timestamps()` 转为轻量适配器：`char_ts → WordTimestamp → SubtitleSegmenter.segment()`。

**Tech Stack:** Python 3.12, pytest, 现有 dataclass 类型（`WordTimestamp`, `TranscriptSegment`），`math` 标准库。

**Spec:** `docs/superpowers/specs/2026-05-22-subtitle-segmentation-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `transcribe/models/segmentation.py` | **Rewrite** | 5 趟管线 `SubtitleSegmenter` |
| `transcribe/models/asr/utils.py` | **Modify** | `segment_by_timestamps()` → 适配器 |
| `tests/test_segmentation.py` | **Update + extend** | 更新断言 + 新增 Pass 2/4/5 测试 |
| `tests/test_asr.py` | **Update** | 修正因算法改进变化的断言 |

**不修改的文件：** `qwen3_asr.py`, `funasr_nano.py`, `funasr_paraformer.py`, `base.py`, `pipeline.py`, `constants.py`, `data/types.py`

---

## Key Design Decisions

1. **内部表示**：所有 Pass 操作 `list[list[WordTimestamp]]`，逗号词保留在流中直至最终构建。
2. **标点剥离**：仅在新 `_to_segments()` 最终步骤中统一剥离句末 + 逗号标点。
3. **Pass 4 评分**：逗号邻近加分(+6) + 间隙质量(0-8) + 高斯中点偏好(0-2)。
4. **计时语义**：segment 的 `start_time`/`end_time` 取自内容词（不含标点），即字幕覆盖语音区间而非静默区间。
5. **新增参数**都有默认值，现有调用无需修改。

---

### Task 1: Add new constructor parameters and content-word helpers

**Files:**
- Modify: `transcribe/models/segmentation.py`

- [ ] **Step 1: Update constructor with new parameters**

**Additive changes only** — do NOT remove any existing methods yet. The old `_split_by_punctuation`, `_check_limits`, `_merge_short`, and `_build_segment` must remain functional until Task 7.

Changes to make in `transcribe/models/segmentation.py`:

1. Add `import math` after `from __future__ import annotations`
2. Add `_ALL_PUNCT = _SENTENCE_END | _CLAUSE_END` after the existing imports
3. Update the class docstring to describe the multi-pass pipeline
4. Add 4 new parameters to `__init__`: `max_cps=12.0`, `gap_soft=0.5`, `gap_hard=1.0`, `min_chars_for_gap_split=5`
5. Add `self.max_cps = max_cps` etc. assignments in `__init__` body

The file header should look like:

```python
"""Pure text/timing subtitle segmentation — no speaker awareness."""

from __future__ import annotations

import math

from transcribe.constants import CLAUSE_END as _CLAUSE_END, SENTENCE_END as _SENTENCE_END
from transcribe.data.types import TranscriptSegment, WordTimestamp

# Union of all punctuation for content-word filtering.
_ALL_PUNCT = _SENTENCE_END | _CLAUSE_END
```

The `__init__` should be:

```python
    def __init__(
        self,
        max_duration: float = 7.0,
        max_chars: int = 25,
        min_duration: float = 0.833,
        max_cps: float = 12.0,
        gap_soft: float = 0.5,
        gap_hard: float = 1.0,
        min_chars_for_gap_split: int = 5,
    ) -> None:
        self.max_duration = max_duration
        self.max_chars = max_chars
        self.min_duration = min_duration
        self.max_cps = max_cps
        self.gap_soft = gap_soft
        self.gap_hard = gap_hard
        self.min_chars_for_gap_split = min_chars_for_gap_split
```

**Do NOT remove** the existing `_split_by_punctuation`, `_check_limits`, `_merge_short`, `_build_segment`, or `segment` methods — they remain active until Task 7.

- [ ] **Step 2: Add static helper methods for content-word operations**

Add these helpers right after `__init__`:

```python
    # ------------------------------------------------------------------
    # Content-word helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _content_words(words: list[WordTimestamp]) -> list[WordTimestamp]:
        """Return only non-punctuation words."""
        return [w for w in words if w.word not in _ALL_PUNCT]

    @staticmethod
    def _content_duration(words: list[WordTimestamp]) -> float:
        """Duration spanned by content words (ignoring punctuation)."""
        cw = [w for w in words if w.word not in _ALL_PUNCT]
        if not cw:
            return 0.0
        return cw[-1].end_time - cw[0].start_time

    @staticmethod
    def _content_chars(words: list[WordTimestamp]) -> int:
        """Total character count of content words."""
        return sum(len(w.word) for w in words if w.word not in _ALL_PUNCT)
```

- [ ] **Step 3: Run existing tests to verify no breakage yet**

Run: `uv run pytest tests/test_segmentation.py tests/test_asr.py -x -v`
Expected: All pass (we only added helpers, didn't change behavior).

- [ ] **Step 4: Commit**

```bash
git add transcribe/models/segmentation.py
git commit -m "feat(segmentation): add new constructor params and content-word helpers"
```

---

### Task 2: Implement Pass 1 — sentence-end split

**Files:**
- Modify: `transcribe/models/segmentation.py`

- [ ] **Step 1: Write test for `_split_sentence_end`**

Add to `tests/test_segmentation.py`:

```python
class TestSplitSentenceEnd:
    """Pass 1: split at sentence-end punctuation, discard it, keep commas."""

    def test_no_sentence_end_returns_single_group(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0)])
        seg = SubtitleSegmenter()
        groups = seg._split_sentence_end(words)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_sentence_end_creates_split(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0), ("。", 1.0, 1.0), ("世", 1.0, 1.5)])
        seg = SubtitleSegmenter()
        groups = seg._split_sentence_end(words)
        assert len(groups) == 2
        # Comma should remain, sentence-end discarded
        assert [w.word for w in groups[0]] == ["你", "好"]
        assert [w.word for w in groups[1]] == ["世"]

    def test_comma_preserved_in_groups(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("，", 0.5, 0.5), ("好", 0.5, 1.0)])
        seg = SubtitleSegmenter()
        groups = seg._split_sentence_end(words)
        assert len(groups) == 1
        assert [w.word for w in groups[0]] == ["你", "，", "好"]

    def test_empty_input(self) -> None:
        seg = SubtitleSegmenter()
        assert seg._split_sentence_end([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_segmentation.py::TestSplitSentenceEnd -v`
Expected: FAIL — `_split_sentence_end` method doesn't exist yet.

- [ ] **Step 3: Implement `_split_sentence_end`**

Add to `SubtitleSegmenter` class:

```python
    # ------------------------------------------------------------------
    # Pass 1: Sentence-end split
    # ------------------------------------------------------------------

    def _split_sentence_end(
        self, words: list[WordTimestamp]
    ) -> list[list[WordTimestamp]]:
        """Split at sentence-ending punctuation; discard it. Keep clause punct."""
        groups: list[list[WordTimestamp]] = []
        buf: list[WordTimestamp] = []
        for w in words:
            if w.word in _SENTENCE_END:
                if buf:
                    groups.append(buf)
                    buf = []
                continue  # discard sentence-end punct
            buf.append(w)  # keep clause-end and content words
        if buf:
            groups.append(buf)
        return groups
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_segmentation.py::TestSplitSentenceEnd -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add transcribe/models/segmentation.py tests/test_segmentation.py
git commit -m "feat(segmentation): implement Pass 1 — sentence-end split with tests"
```

---

### Task 3: Implement Pass 2 — gap-aware splitting

**Files:**
- Modify: `transcribe/models/segmentation.py`
- Modify: `tests/test_segmentation.py`

- [ ] **Step 1: Write tests for `_split_by_gap`**

Add to `tests/test_segmentation.py`:

```python
class TestSplitByGap:
    """Pass 2: split at large inter-word gaps."""

    def test_no_gap_returns_single_group(self) -> None:
        words = _ws([("你", 0.0, 0.5), ("好", 0.5, 1.0), ("世", 1.0, 1.5)])
        seg = SubtitleSegmenter()
        groups = seg._split_by_gap(words)
        assert len(groups) == 1

    def test_hard_gap_always_splits(self) -> None:
        """Gap > 1.0s splits unconditionally."""
        words = _ws([
            ("你", 0.0, 0.5), ("好", 0.5, 1.0),
            ("世", 2.5, 3.0), ("界", 3.0, 3.5),
        ])
        seg = SubtitleSegmenter()
        groups = seg._split_by_gap(words)
        assert len(groups) == 2
        assert [w.word for w in groups[0]] == ["你", "好"]
        assert [w.word for w in groups[1]] == ["世", "界"]

    def test_soft_gap_splits_with_min_chars(self) -> None:
        """Gap > 0.5s splits if ≥5 content chars accumulated."""
        words = _ws([
            ("今", 0.0, 0.1), ("天", 0.1, 0.2), ("天", 0.2, 0.3),
            ("气", 0.3, 0.4), ("很", 0.4, 0.5),
            # 5 chars accumulated, then 0.6s gap
            ("好", 1.1, 1.2),
        ])
        seg = SubtitleSegmenter(gap_soft=0.5)
        groups = seg._split_by_gap(words)
        assert len(groups) == 2

    def test_soft_gap_no_split_below_min_chars(self) -> None:
        """Gap > 0.5s does NOT split if <5 content chars accumulated."""
        words = _ws([
            ("你", 0.0, 0.1), ("好", 0.1, 0.2),
            # only 2 chars, then 0.6s gap
            ("世", 0.8, 0.9), ("界", 0.9, 1.0),
        ])
        seg = SubtitleSegmenter(gap_soft=0.5)
        groups = seg._split_by_gap(words)
        assert len(groups) == 1

    def test_single_word_returns_single_group(self) -> None:
        words = _ws([("好", 0.0, 0.5)])
        seg = SubtitleSegmenter()
        groups = seg._split_by_gap(words)
        assert len(groups) == 1

    def test_comma_between_words_no_extra_split(self) -> None:
        """Comma (zero-width) between words should not create a false gap."""
        words = _ws([("你", 0.0, 0.5), ("，", 0.5, 0.5), ("好", 0.5, 1.0)])
        seg = SubtitleSegmenter()
        groups = seg._split_by_gap(words)
        assert len(groups) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_segmentation.py::TestSplitByGap -v`
Expected: FAIL — `_split_by_gap` doesn't exist.

- [ ] **Step 3: Implement `_split_by_gap`**

Add to `SubtitleSegmenter`:

```python
    # ------------------------------------------------------------------
    # Pass 2: Gap-aware splitting
    # ------------------------------------------------------------------

    def _split_by_gap(
        self, words: list[WordTimestamp]
    ) -> list[list[WordTimestamp]]:
        """Split at speech pauses (inter-content-word gaps)."""
        if len(words) <= 1:
            return [words]

        groups: list[list[WordTimestamp]] = []
        buf: list[WordTimestamp] = [words[0]]
        accumulated = self._content_chars(buf)

        for i in range(1, len(words)):
            w = words[i]
            is_content = w.word not in _ALL_PUNCT

            if is_content:
                # Find previous content word for gap calculation
                prev_content: WordTimestamp | None = None
                for j in range(i - 1, -1, -1):
                    if words[j].word not in _ALL_PUNCT:
                        prev_content = words[j]
                        break

                if prev_content is not None:
                    gap = w.start_time - prev_content.end_time
                    should_split = False
                    if gap > self.gap_hard:
                        should_split = True
                    elif gap > self.gap_soft and accumulated >= self.min_chars_for_gap_split:
                        should_split = True

                    if should_split:
                        groups.append(buf)
                        buf = []
                        accumulated = 0

            buf.append(w)
            if is_content:
                accumulated += len(w.word)

        if buf:
            groups.append(buf)
        return groups if groups else [words]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmentation.py::TestSplitByGap -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add transcribe/models/segmentation.py tests/test_segmentation.py
git commit -m "feat(segmentation): implement Pass 2 — gap-aware splitting with tests"
```

---

### Task 4: Implement Pass 3 — refactor `_merge_short` for word groups

**Files:**
- Modify: `transcribe/models/segmentation.py`

- [ ] **Step 1: Write test for `_merge_short_groups`**

Add to `tests/test_segmentation.py`:

```python
class TestMergeShortGroups:
    """Pass 3: merge word groups whose content duration < min_duration."""

    def test_short_group_merged_with_next(self) -> None:
        groups = [
            _ws([("你", 0.0, 0.3)]),                # 0.3s < 0.833
            _ws([("好", 0.3, 2.0)]),                  # 1.7s
        ]
        seg = SubtitleSegmenter(min_duration=0.833)
        merged = seg._merge_short_groups(groups)
        assert len(merged) == 1
        assert merged[0] == groups[0] + groups[1]

    def test_no_merge_when_over_max_chars(self) -> None:
        groups = [
            _ws([("A" * 15, 0.0, 0.3)]),             # short, 15 chars
            _ws([("B" * 15, 0.3, 2.0)]),              # 15 chars
        ]
        seg = SubtitleSegmenter(max_chars=20, min_duration=0.833)
        merged = seg._merge_short_groups(groups)
        assert len(merged) == 2  # merged would be 30 chars > 20

    def test_no_merge_when_over_max_duration(self) -> None:
        groups = [
            _ws([("A", 0.0, 0.3)]),                   # short
            _ws([("B", 0.3, 8.0)]),                    # long
        ]
        seg = SubtitleSegmenter(max_duration=7.0, min_duration=0.833)
        merged = seg._merge_short_groups(groups)
        assert len(merged) == 2  # merged would be 8.0s > 7.0

    def test_chain_of_short_groups_merged(self) -> None:
        groups = [
            _ws([("A", 0.0, 0.3)]),                   # 0.3s
            _ws([("B", 0.3, 0.6)]),                   # 0.3s
            _ws([("C", 0.6, 2.0)]),                    # 1.4s
        ]
        seg = SubtitleSegmenter(min_duration=0.833)
        merged = seg._merge_short_groups(groups)
        assert len(merged) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_segmentation.py::TestMergeShortGroups -v`
Expected: FAIL — `_merge_short_groups` doesn't exist.

- [ ] **Step 3: Implement `_merge_short_groups`**

Add to `SubtitleSegmenter`. This replaces the old `_merge_short` which operated on `TranscriptSegment`. The new version operates on `list[list[WordTimestamp]]`:

```python
    # ------------------------------------------------------------------
    # Pass 3: Short group merging
    # ------------------------------------------------------------------

    def _merge_short_groups(
        self, groups: list[list[WordTimestamp]]
    ) -> list[list[WordTimestamp]]:
        """Merge groups whose content duration < min_duration."""
        if not groups:
            return []

        result: list[list[WordTimestamp]] = [groups[0]]
        for i in range(1, len(groups)):
            prev = result[-1]
            prev_dur = self._content_duration(prev)
            if prev_dur < self.min_duration:
                candidate = prev + groups[i]
                merged_dur = self._content_duration(candidate)
                merged_chars = self._content_chars(candidate)
                if merged_dur <= self.max_duration and merged_chars <= self.max_chars:
                    result[-1] = candidate
                    continue
            result.append(groups[i])
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmentation.py::TestMergeShortGroups -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add transcribe/models/segmentation.py tests/test_segmentation.py
git commit -m "feat(segmentation): implement Pass 3 — merge short groups with tests"
```

---

### Task 5: Implement Pass 4 — over-limit correction with scoring

**Files:**
- Modify: `transcribe/models/segmentation.py`
- Modify: `tests/test_segmentation.py`

- [ ] **Step 1: Write tests for `_split_oversized`**

Add to `tests/test_segmentation.py`:

```python
class TestSplitOversized:
    """Pass 4: split groups exceeding max_duration/max_chars at best point."""

    def test_within_limits_not_split(self) -> None:
        group = _ws([("你", 0.0, 1.0), ("好", 1.0, 2.0)])
        seg = SubtitleSegmenter(max_chars=10, max_duration=7.0)
        result = seg._split_oversized(group)
        assert len(result) == 1

    def test_over_duration_splits_at_comma(self) -> None:
        """Comma position gets scoring bonus → preferred split point."""
        group = _ws([
            ("这", 0.0, 0.5), ("是", 0.5, 1.0),
            ("，", 1.0, 1.0),
            ("一", 1.0, 3.0), ("段", 3.0, 5.0), ("很", 5.0, 7.0), ("长", 7.0, 8.5),
        ])
        seg = SubtitleSegmenter(max_duration=7.0)
        result = seg._split_oversized(group)
        assert len(result) == 2
        # Left group content: [这, 是]
        assert SubtitleSegmenter._content_chars(result[0]) == 2
        # Right group content: [一, 段, 很, 长]
        assert SubtitleSegmenter._content_chars(result[1]) == 4

    def test_over_chars_splits_at_midpoint_when_no_punct(self) -> None:
        """No punctuation → split near midpoint."""
        group = _ws([(f"w{i}", float(i) * 0.2, float(i) * 0.2 + 0.1) for i in range(30)])
        seg = SubtitleSegmenter(max_chars=15, max_duration=999.0)
        result = seg._split_oversized(group)
        assert len(result) >= 2
        for g in result:
            assert SubtitleSegmenter._content_chars(g) <= 15

    def test_recursive_split_for_very_long(self) -> None:
        """Very long input gets split into multiple compliant groups."""
        group = _ws([(f"c{i}", float(i) * 0.1, float(i) * 0.1 + 0.05) for i in range(60)])
        seg = SubtitleSegmenter(max_chars=10, max_duration=999.0)
        result = seg._split_oversized(group)
        assert len(result) >= 5
        for g in result:
            assert SubtitleSegmenter._content_chars(g) <= 10

    def test_single_word_not_split(self) -> None:
        group = _ws([("好", 0.0, 8.0)])
        seg = SubtitleSegmenter(max_duration=7.0)
        result = seg._split_oversized(group)
        assert len(result) == 1  # can't split a single word
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_segmentation.py::TestSplitOversized -v`
Expected: FAIL — `_split_oversized` doesn't exist.

- [ ] **Step 3: Implement `_split_oversized` and `_score_split_point`**

Add to `SubtitleSegmenter`:

```python
    # ------------------------------------------------------------------
    # Pass 4: Over-limit correction
    # ------------------------------------------------------------------

    def _split_oversized(
        self, words: list[WordTimestamp]
    ) -> list[list[WordTimestamp]]:
        """Split a group that exceeds max_duration or max_chars."""
        content_chars = self._content_chars(words)
        content_dur = self._content_duration(words)

        if content_chars <= self.max_chars and content_dur <= self.max_duration:
            return [words]

        content = self._content_words(words)
        if len(content) <= 1:
            return [words]  # can't split further

        # Find best split point
        best_idx = self._find_best_split(words)
        left = words[:best_idx]
        right = words[best_idx:]

        if not left or not right:
            # Safety: at least 1 word on each side
            mid = max(1, len(words) // 2)
            left = words[:mid]
            right = words[mid:]

        result = [left]
        result.extend(self._split_oversized(right))  # recurse for right side
        return result

    def _find_best_split(self, words: list[WordTimestamp]) -> int:
        """Find the index at which to split *words* (into words[:i] | words[i:])."""
        content = self._content_words(words)
        total_chars = sum(len(w.word) for w in content)
        best_idx = len(words) // 2  # fallback midpoint
        best_score = -1.0

        for i in range(1, len(words)):
            left_content = [w for w in words[:i] if w.word not in _ALL_PUNCT]
            right_content = [w for w in words[i:] if w.word not in _ALL_PUNCT]
            if not left_content or not right_content:
                continue

            score = self._score_split_point(
                words, i, left_content, right_content, total_chars
            )
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    @staticmethod
    def _score_split_point(
        words: list[WordTimestamp],
        i: int,
        left_content: list[WordTimestamp],
        right_content: list[WordTimestamp],
        total_content_chars: int,
    ) -> float:
        """Score a candidate split at index *i*."""
        score = 0.0

        # Factor 1: Comma proximity (+6 if words[i-1] is clause punctuation)
        if words[i - 1].word in _CLAUSE_END:
            score += 6.0

        # Factor 2: Gap quality between adjacent content words (0–8)
        gap = right_content[0].start_time - left_content[-1].end_time
        score += min(8.0, max(0.0, gap) * 8.0)

        # Factor 3: Gaussian midpoint preference (0–2)
        left_chars = sum(len(w.word) for w in left_content)
        ratio = left_chars / total_content_chars if total_content_chars > 0 else 0.5
        score += 2.0 * math.exp(-((ratio - 0.5) ** 2) / 0.18)

        return score
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmentation.py::TestSplitOversized -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add transcribe/models/segmentation.py tests/test_segmentation.py
git commit -m "feat(segmentation): implement Pass 4 — over-limit correction with scoring"
```

---

### Task 6: Implement Pass 5 — CPS validation

**Files:**
- Modify: `transcribe/models/segmentation.py`
- Modify: `tests/test_segmentation.py`

- [ ] **Step 1: Write tests for `_validate_cps`**

Add to `tests/test_segmentation.py`:

```python
class TestValidateCps:
    """Pass 5: re-split groups where CPS exceeds limit."""

    def test_normal_cps_not_split(self) -> None:
        # 6 chars in 3.0s = 2 CPS
        groups = [_ws([("你", 0.0, 1.0), ("好", 1.0, 2.0), ("世", 2.0, 3.0)])]
        seg = SubtitleSegmenter(max_cps=12.0)
        result = seg._validate_cps(groups)
        assert len(result) == 1

    def test_high_cps_split(self) -> None:
        # 12 chars in 0.5s = 24 CPS > max_cps=12
        chars = "ABCDEFGHIJKL"
        group = _ws([(c, i * 0.04, (i + 1) * 0.04) for i, c in enumerate(chars)])
        seg = SubtitleSegmenter(max_cps=12.0, max_chars=20, max_duration=999.0)
        result = seg._validate_cps([group])
        assert len(result) >= 2
        for g in result:
            dur = SubtitleSegmenter._content_duration(g)
            chars_count = SubtitleSegmenter._content_chars(g)
            if dur > 0:
                cps = chars_count / dur
                # Each result should have better CPS than original
                assert cps <= 15.0  # allow some tolerance for edge cases

    def test_single_word_high_cps_not_split(self) -> None:
        # 1 char in 0.01s → very high CPS but can't split further
        group = _ws([("好", 0.0, 0.01)])
        seg = SubtitleSegmenter(max_cps=12.0)
        result = seg._validate_cps([group])
        assert len(result) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_segmentation.py::TestValidateCps -v`
Expected: FAIL — `_validate_cps` doesn't exist.

- [ ] **Step 3: Implement `_validate_cps`**

Add to `SubtitleSegmenter`:

```python
    # ------------------------------------------------------------------
    # Pass 5: CPS validation
    # ------------------------------------------------------------------

    def _validate_cps(
        self, groups: list[list[WordTimestamp]]
    ) -> list[list[WordTimestamp]]:
        """Re-split groups where content CPS exceeds *max_cps*."""
        result: list[list[WordTimestamp]] = []
        for group in groups:
            dur = self._content_duration(group)
            chars = self._content_chars(group)
            if dur <= 0 or chars / dur <= self.max_cps:
                result.append(group)
                continue
            # CPS exceeded → try splitting
            content = self._content_words(group)
            if len(content) <= 1:
                result.append(group)  # can't split further
                continue
            sub = self._split_oversized(group)
            result.extend(sub)
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmentation.py::TestValidateCps -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add transcribe/models/segmentation.py tests/test_segmentation.py
git commit -m "feat(segmentation): implement Pass 5 — CPS validation with tests"
```

---

### Task 7: Wire up 5-pass pipeline and final segment builder

**Files:**
- Modify: `transcribe/models/segmentation.py`

This task replaces the old `segment()`, `_split_by_punctuation()`, `_check_limits()`, `_merge_short()`, and `_build_segment()` with the new pipeline.

- [ ] **Step 1: Replace `segment()` with 5-pass pipeline**

Replace the existing `segment()`, `_split_by_punctuation()`, `_check_limits()`, `_merge_short()`, and `_build_segment()` methods with:

```python
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, words: list[WordTimestamp]) -> list[TranscriptSegment]:
        """Segment *words* into subtitle-ready ``TranscriptSegment``s.

        5-pass pipeline:
        1. Sentence-end punctuation hard split
        2. Gap-aware splitting (speech pauses)
        3. Short segment merging
        4. Over-limit correction (comma/gap/midpoint scoring)
        5. CPS (characters-per-second) validation
        """
        if not words:
            return []

        # Pass 1: Split at sentence-end punctuation
        groups = self._split_sentence_end(words)

        # Pass 2: Split at large gaps
        gap_groups: list[list[WordTimestamp]] = []
        for g in groups:
            gap_groups.extend(self._split_by_gap(g))

        # Pass 3: Merge short groups
        merged = self._merge_short_groups(gap_groups)

        # Pass 4: Split oversized groups
        fixed: list[list[WordTimestamp]] = []
        for g in merged:
            if self._content_chars(g) > self.max_chars or self._content_duration(g) > self.max_duration:
                fixed.extend(self._split_oversized(g))
            else:
                fixed.append(g)

        # Pass 5: CPS validation
        validated = self._validate_cps(fixed)

        # Final: build TranscriptSegments (strip all punctuation)
        return self._to_segments(validated)

    # ------------------------------------------------------------------
    # Final segment builder
    # ------------------------------------------------------------------

    @staticmethod
    def _to_segments(groups: list[list[WordTimestamp]]) -> list[TranscriptSegment]:
        """Build TranscriptSegments from word groups, stripping all punctuation."""
        segments: list[TranscriptSegment] = []
        for group in groups:
            content = [w for w in group if w.word not in _ALL_PUNCT]
            if not content:
                continue
            text = "".join(w.word for w in content)
            segments.append(TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=content[0].start_time,
                end_time=content[-1].end_time,
                text=text,
                words=list(content),
            ))
        return segments
```

- [ ] **Step 2: Run existing SubtitleSegmenter tests**

Run: `uv run pytest tests/test_segmentation.py -v`

Expected: Some tests may fail due to behavior changes. **Do not fix yet** — that's Task 8.

Key tests that may need attention:
- `test_clause_split_when_over_duration` — timing now uses content words, not comma timestamps
- `test_no_split_within_limits` — should still pass (comma stripped, no split)
- `test_hard_cut_no_clause` — should still pass (punctuation-free text unchanged)

- [ ] **Step 3: Commit**

```bash
git add transcribe/models/segmentation.py
git commit -m "feat(segmentation): wire up 5-pass pipeline, replace old single-pass algorithm"
```

---

### Task 8: Update `segment_by_timestamps()` adapter

**Files:**
- Modify: `transcribe/models/asr/utils.py`

- [ ] **Step 1: Replace `segment_by_timestamps` body with adapter**

Replace the entire `segment_by_timestamps` function (lines 107–202 in the current file) with:

```python
# --- Subtitle segmentation from character-level timestamps ---


def segment_by_timestamps(
    char_ts: list[tuple[str, float, float]],
    max_duration: float = 7.0,
    max_chars: int = 25,
) -> list[TranscriptSegment]:
    """Split character-level timestamps into subtitle-grade segments.

    Delegates to :class:`SubtitleSegmenter` for the actual segmentation.

    Args:
        char_ts: ``[(text, start_sec, end_sec), ...]`` character-level timestamps.
        max_duration: Maximum duration per subtitle segment (seconds).
        max_chars: Maximum characters per subtitle segment.

    Returns:
        List of :class:`TranscriptSegment`.
    """
    if not char_ts:
        return []

    from transcribe.data.types import WordTimestamp
    from transcribe.models.segmentation import SubtitleSegmenter

    words = [
        WordTimestamp(word=text, start_time=start, end_time=end)
        for text, start, end in char_ts
    ]
    segmenter = SubtitleSegmenter(max_duration=max_duration, max_chars=max_chars)
    return segmenter.segment(words)
```

Also clean up: remove the now-unused `_SENTENCE_END` and `_CLAUSE_END` imports at the top of `utils.py` if they are no longer referenced by other code in the file. Check with:

```
grep -n '_SENTENCE_END\|_CLAUSE_END' transcribe/models/asr/utils.py
```

If only the removed function used them, remove the import line:
```python
from transcribe.constants import CLAUSE_END as _CLAUSE_END, SENTENCE_END as _SENTENCE_END
```

Note: `_SENTENCE_END` and `_CLAUSE_END` might still be used in other functions in this file — check carefully before removing.

- [ ] **Step 2: Run segment_by_timestamps tests**

Run: `uv run pytest tests/test_asr.py -k "segment_" -v`

Expected: Some tests may fail due to timing changes (segment end times now use content word timestamps instead of punctuation timestamps). **Do not fix yet** — that's Task 9.

- [ ] **Step 3: Commit**

```bash
git add transcribe/models/asr/utils.py
git commit -m "refactor: convert segment_by_timestamps() to SubtitleSegmenter adapter"
```

---

### Task 9: Fix failing tests in `tests/test_segmentation.py`

**Files:**
- Modify: `tests/test_segmentation.py`

- [ ] **Step 1: Run all segmentation tests and capture failures**

Run: `uv run pytest tests/test_segmentation.py -v 2>&1 | head -100`

Analyze each failure. The most likely changes:

**A. Timing assertions** — `start_time`/`end_time` now use content words, not punctuation. Where a test asserts end_time equals a comma's timestamp, update to use the last content word's end_time.

**B. `test_clause_split_when_over_duration`** — currently asserts `result[0].end_time == pytest.approx(3.3)` (comma time). With new algorithm, content words determine timing:
- Left content: [这(0.0-0.5), 是(0.5-1.0)] → `end_time = 1.0`
- Right content: [一(1.0-3.0), 段(3.0-5.0), 很(5.0-7.0), 长(7.0-8.5)] → `start_time = 1.0`
- Update assertion to: `result[0].end_time == pytest.approx(1.0)` and `result[1].start_time == pytest.approx(1.0)`

**C. `test_multiple_commas_splits_when_over_duration`** — update duration/char count expectations based on content words.

**D. `test_stale_clause_idx_after_split`** — this tests the old `_check_limits` stale index bug. The new algorithm doesn't have `_check_limits`, so this test may need to be rewritten to test a different aspect of the new pipeline, or removed if the scenario is now covered by the new pass tests.

For each failing test, update the assertions to match the new algorithm's behavior. **Do not weaken test assertions** — ensure the test still validates meaningful behavior.

- [ ] **Step 2: Re-run tests until all pass**

Run: `uv run pytest tests/test_segmentation.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_segmentation.py
git commit -m "test: update segmentation tests for multi-pass pipeline behavior"
```

---

### Task 10: Fix failing tests in `tests/test_asr.py`

**Files:**
- Modify: `tests/test_asr.py`

- [ ] **Step 1: Run all segment_by_timestamps tests and capture failures**

Run: `uv run pytest tests/test_asr.py -k "segment_" -v 2>&1 | head -100`

Most likely failure:

**`test_segment_max_duration_splits_at_clause`** — currently asserts:
```python
assert result[0].end_time == pytest.approx(3.3)  # comma time
assert result[1].start_time == pytest.approx(3.3)
```

With new algorithm, the split happens at the comma position (Pass 4 scoring gives comma bonus), but timing uses content words. The content words around the split:
- Left content ends at 好(2.7-3.2), `end_time = 3.2`
- Right content starts at 我(3.3-3.8), `start_time = 3.3`

Update to:
```python
assert result[0].end_time == pytest.approx(3.2)
assert result[1].start_time == pytest.approx(3.3)
```

Also check:
- Duration constraints still hold (`duration <= 5.0`)
- No gaps or overlaps still hold

- [ ] **Step 2: Re-run tests until all pass**

Run: `uv run pytest tests/test_asr.py -k "segment_" -v`
Expected: All 7 segment tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_asr.py
git commit -m "test: update segment_by_timestamps tests for new pipeline timing"
```

---

### Task 11: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL tests pass.

- [ ] **Step 2: Run tests excluding slow (model-download) tests**

Run: `uv run pytest -m "not slow" -v`
Expected: All non-slow tests pass.

- [ ] **Step 3: Verify public API contract**

Quick sanity check that the import paths still work:

```bash
uv run python -c "
from transcribe.models.asr.utils import segment_by_timestamps
from transcribe.models.segmentation import SubtitleSegmenter
from transcribe.data.types import WordTimestamp, TranscriptSegment

# Test SubtitleSegmenter
words = [WordTimestamp('你', 0.0, 0.5), WordTimestamp('好', 0.5, 1.0)]
segs = SubtitleSegmenter().segment(words)
assert len(segs) == 1
assert segs[0].text == '你好'

# Test segment_by_timestamps adapter
result = segment_by_timestamps([('你', 0.0, 0.5), ('好', 0.5, 1.0), ('。', 1.0, 1.0)])
assert len(result) == 1
assert result[0].text == '你好'

print('All sanity checks passed')
"
```

Expected: `All sanity checks passed`

- [ ] **Step 4: Final commit (if any remaining changes)**

```bash
git add -A
git commit -m "chore: final cleanup for multi-pass subtitle segmentation"
```
