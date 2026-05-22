"""Pure text/timing subtitle segmentation — no speaker awareness."""

from __future__ import annotations

import math

from transcribe.constants import CLAUSE_END as _CLAUSE_END, SENTENCE_END as _SENTENCE_END
from transcribe.data.types import TranscriptSegment, WordTimestamp

_ALL_PUNCT = _SENTENCE_END | _CLAUSE_END


class SubtitleSegmenter:
    """Split ``list[WordTimestamp]`` into ``list[TranscriptSegment]``.

    Uses Netflix Chinese subtitle standards by default:
    - max_duration: 7.0 s
    - max_chars: 25
    - min_duration: 0.833 s

    Multi-pass pipeline (new methods, not yet wired into ``segment()``):
    - Pass 1: Split at sentence-ending punctuation (``_split_sentence_end``).
    - Pass 2: Gap-aware splitting at speech pauses (``_split_by_gap``).
    - Pass 3: Merge short groups below *min_duration* (``_merge_short_groups``).
    - Pass 4: Split oversized groups exceeding limits (``_split_oversized``).
    - Pass 5: CPS validation — re-split if characters-per-second is too high
      (``_validate_cps``).
    """

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, words: list[WordTimestamp]) -> list[TranscriptSegment]:
        """Segment *words* into subtitle-ready ``TranscriptSegment``s."""
        if not words:
            return []

        raw = self._split_by_punctuation(words)
        merged = self._merge_short(raw)
        return merged

    # ------------------------------------------------------------------
    # Core splitting
    # ------------------------------------------------------------------

    def _split_by_punctuation(
        self, words: list[WordTimestamp]
    ) -> list[TranscriptSegment]:
        buf: list[WordTimestamp] = []
        last_clause_idx: int | None = None  # index into *buf*
        segments: list[TranscriptSegment] = []

        for _word in words:
            text = _word.word

            # --- sentence-end: hard flush, discard punctuation ---
            if text in _SENTENCE_END:
                if buf:
                    segments.append(self._build_segment(buf))
                    buf = []
                    last_clause_idx = None
                continue

            # --- clause-end: record soft split position, discard punctuation ---
            if text in _CLAUSE_END:
                if buf:
                    # Record position before adding anything; the comma is
                    # discarded so we track the end of the content words.
                    last_clause_idx = len(buf)
                continue

            # --- normal word ---
            buf.append(_word)

            # Check limits after every add
            prev_len = len(buf)
            self._check_limits(buf, last_clause_idx, segments)
            # If buf was trimmed (clause split or hard cut), the index is stale
            if len(buf) < prev_len:
                last_clause_idx = None

        # Flush remaining buffer
        if buf:
            segments.append(self._build_segment(buf))

        return segments

    def _check_limits(
        self,
        buf: list[WordTimestamp],
        last_clause_idx: int | None,
        segments: list[TranscriptSegment],
    ) -> None:
        """Flush or trim *buf* if duration/char limits are exceeded."""
        duration = buf[-1].end_time - buf[0].start_time
        char_count = sum(len(w.word) for w in buf)

        over_duration = duration > self.max_duration
        over_chars = char_count >= self.max_chars

        if not over_duration and not over_chars:
            return

        # Try clause split
        if last_clause_idx is not None and last_clause_idx > 0:
            segments.append(self._build_segment(buf[:last_clause_idx]))
            replaced = buf[last_clause_idx:]
            buf[:] = replaced
            return

        # Hard cut: split in half (or single element flush)
        mid = max(1, len(buf) // 2)
        segments.append(self._build_segment(buf[:mid]))
        replaced = buf[mid:]
        buf[:] = replaced

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def _merge_short(
        self, segments: list[TranscriptSegment]
    ) -> list[TranscriptSegment]:
        """Merge segments shorter than *min_duration* with neighbours.

        Single left-to-right pass: each segment is merged into the previous
        one when the previous segment's duration is below *min_duration*.
        This naturally handles chains of short segments.
        """
        if not segments:
            return []

        result: list[TranscriptSegment] = [segments[0]]
        for i in range(1, len(segments)):
            prev = result[-1]
            prev_dur = prev.end_time - prev.start_time
            if prev_dur < self.min_duration:
                merged_text = prev.text + segments[i].text
                merged_dur = segments[i].end_time - prev.start_time
                merged_chars = len(merged_text)
                if merged_dur <= self.max_duration and merged_chars <= self.max_chars:
                    merged_words = (prev.words or []) + (segments[i].words or [])
                    result[-1] = TranscriptSegment(
                        speaker_id=prev.speaker_id,
                        start_time=prev.start_time,
                        end_time=segments[i].end_time,
                        text=merged_text,
                        words=merged_words or None,
                    )
                    continue
            result.append(segments[i])
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_segment(words: list[WordTimestamp]) -> TranscriptSegment:
        text = "".join(w.word for w in words)
        return TranscriptSegment(
            speaker_id="SPEAKER_00",
            start_time=words[0].start_time,
            end_time=words[-1].end_time,
            text=text,
            words=list(words),
        )

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

        result = list(self._split_oversized(left))  # recurse on left side
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
            # Force split by constraining max_chars to what max_cps allows
            target_chars = max(1, int(dur * self.max_cps))
            saved = self.max_chars
            self.max_chars = min(self.max_chars, target_chars)
            sub = self._split_oversized(group)
            self.max_chars = saved
            result.extend(sub)
        return result

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
