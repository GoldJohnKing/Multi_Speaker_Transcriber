"""Pure text/timing subtitle segmentation — no speaker awareness."""

from __future__ import annotations

from transcribe.constants import CLAUSE_END as _CLAUSE_END, SENTENCE_END as _SENTENCE_END
from transcribe.data.types import TranscriptSegment, WordTimestamp


class SubtitleSegmenter:
    """Split ``list[WordTimestamp]`` into ``list[TranscriptSegment]``.

    Uses Netflix Chinese subtitle standards by default:
    - max_duration: 7.0 s
    - max_chars: 25
    - min_duration: 0.833 s
    """

    def __init__(
        self,
        max_duration: float = 7.0,
        max_chars: int = 25,
        min_duration: float = 0.833,
    ) -> None:
        self.max_duration = max_duration
        self.max_chars = max_chars
        self.min_duration = min_duration

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
