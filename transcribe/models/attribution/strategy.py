"""Timestamp-based speaker attribution strategy."""
from __future__ import annotations

from transcribe.constants import ALIGNMENT_PUNCT as _PUNCT_CHARS
from transcribe.data.types import (
    DiarizationResult,
    SpeakerSegment,
    TranscriptSegment,
    WordTimestamp,
)


class TimestampStrategy:
    """Attribute words to speakers via temporal overlap with diarization segments.

    For each word, finds which diarization segment contains the word's center
    time point. Uses exclusive diarization output (each time point belongs to
    exactly one speaker).
    """

    def __init__(self, *, min_segment_duration: float = 0.2) -> None:
        self._min_segment_duration = min_segment_duration

    def attribute(
        self,
        words: list[WordTimestamp],
        diarization: DiarizationResult,
    ) -> list[TranscriptSegment]:
        if not words:
            return []

        # Step 1: Assign speaker to each word
        assigned: list[tuple[str, WordTimestamp]] = []
        for w in words:
            spk = self._assign_word(w, diarization.segments)
            assigned.append((spk, w))

        # Step 1.5: Fix punctuation attribution at speaker boundaries
        assigned = self._fix_punctuation_attribution(assigned)

        # Step 2: Merge consecutive same-speaker words
        merged = self._merge_consecutive(assigned)

        # Step 3: Smooth short interruptions
        smoothed = self._smooth_short_segments(merged)

        # Step 4: Re-merge consecutive same-speaker segments that smoothing
        # may have produced (e.g. [A+B, A] → [A+B+A])
        final = self._final_merge(smoothed)

        return final

    @staticmethod
    def _fix_punctuation_attribution(
        assigned: list[tuple[str, WordTimestamp]],
    ) -> list[tuple[str, WordTimestamp]]:
        """Re-assign punctuation to follow its neighboring non-punct word's speaker.

        Punctuation characters have tiny interpolated timestamps (e.g. 20 ms)
        that can place them in the wrong speaker's time range at boundaries.
        """
        if len(assigned) <= 1:
            return assigned

        # Find the speaker for the first non-punct word (for leading punct)
        first_non_punct_spk = assigned[0][0]
        for spk, w in assigned:
            if len(w.word) != 1 or w.word not in _PUNCT_CHARS:
                first_non_punct_spk = spk
                break

        result: list[tuple[str, WordTimestamp]] = []
        for i, (spk, w) in enumerate(assigned):
            if len(w.word) == 1 and w.word in _PUNCT_CHARS:
                # Use previous non-punct word's speaker, or next if at start
                if result:
                    fixed_spk = result[-1][0]
                else:
                    # Leading punctuation — use first non-punct speaker
                    fixed_spk = first_non_punct_spk
                result.append((fixed_spk, w))
            else:
                result.append((spk, w))
        return result

    def _assign_word(
        self, word: WordTimestamp, segments: list[SpeakerSegment]
    ) -> str:
        """Find which diarization segment contains the word's center.

        Uses two-pointer scan since both words and diarization segments
        are time-ordered. O(n + m) instead of O(n · m).
        """
        if not segments:
            return "SPEAKER_00"

        center = (word.start_time + word.end_time) / 2.0

        # Two-pointer: advance through segments to find the one containing center
        for seg in segments:
            if seg.start_time <= center < seg.end_time:
                return seg.speaker_id
            if seg.start_time > center:
                break

        return self._nearest_speaker(center, segments)

    def _nearest_speaker(
        self, time: float, segments: list[SpeakerSegment]
    ) -> str:
        """Fallback: find speaker of nearest segment by midpoint distance."""
        if not segments:
            return "SPEAKER_00"
        best_seg = min(
            segments,
            key=lambda s: abs((s.start_time + s.end_time) / 2.0 - time),
        )
        return best_seg.speaker_id

    def _merge_consecutive(
        self, assigned: list[tuple[str, WordTimestamp]]
    ) -> list[TranscriptSegment]:
        """Merge consecutive words with the same speaker into segments."""
        if not assigned:
            return []

        segments: list[TranscriptSegment] = []
        cur_spk = assigned[0][0]
        cur_words: list[WordTimestamp] = [assigned[0][1]]

        for spk, w in assigned[1:]:
            if spk == cur_spk:
                cur_words.append(w)
            else:
                segments.append(self._build_segment(cur_spk, cur_words))
                cur_spk = spk
                cur_words = [w]

        if cur_words:
            segments.append(self._build_segment(cur_spk, cur_words))

        return segments

    def _smooth_short_segments(
        self, segments: list[TranscriptSegment]
    ) -> list[TranscriptSegment]:
        """Merge short interruptions between same-speaker segments.

        Looks ahead through consecutive short segments from different speakers.
        If the same speaker resumes after the run of short segments, the entire
        run is absorbed into the preceding segment.
        """
        if len(segments) < 3:
            return segments

        def _is_short(s: TranscriptSegment) -> bool:
            return (s.end_time - s.start_time) < self._min_segment_duration

        def _merge_word_lists(
            a: list[WordTimestamp] | None, b: list[WordTimestamp] | None,
        ) -> list[WordTimestamp] | None:
            if a is not None and b is not None:
                return a + b
            return None

        result = [segments[0]]
        i = 1
        while i < len(segments):
            seg = segments[i]
            prev = result[-1]

            if _is_short(seg) and prev.speaker_id != seg.speaker_id:
                # Look ahead through consecutive short segments
                j = i + 1
                while j < len(segments):
                    ahead = segments[j]
                    if ahead.speaker_id == prev.speaker_id:
                        break  # Found same speaker
                    if not _is_short(ahead):
                        break  # Long segment — stop
                    j += 1

                if j < len(segments) and segments[j].speaker_id == prev.speaker_id:
                    # Merge segments[i:j+1] into prev
                    merged_words = prev.words
                    merged_text = prev.text
                    merge_end = prev.end_time
                    for k in range(i, j + 1):
                        merged_words = _merge_word_lists(merged_words, segments[k].words)
                        merged_text += segments[k].text
                        merge_end = segments[k].end_time

                    result[-1] = TranscriptSegment(
                        speaker_id=prev.speaker_id,
                        start_time=prev.start_time,
                        end_time=merge_end,
                        text=merged_text,
                        words=merged_words,
                    )
                    i = j + 1
                    continue

            result.append(seg)
            i += 1

        return result

    def _final_merge(
        self, segments: list[TranscriptSegment]
    ) -> list[TranscriptSegment]:
        """Re-merge consecutive same-speaker segments after smoothing.

        Smoothing can leave adjacent same-speaker segments (e.g. when a short
        segment B is absorbed into preceding A, but the following A segment
        remains separate). This pass collapses such runs.
        """
        if len(segments) < 2:
            return segments

        result = [segments[0]]
        for seg in segments[1:]:
            prev = result[-1]
            if seg.speaker_id == prev.speaker_id:
                merged_words = (
                    (prev.words or []) + (seg.words or [])
                    if (prev.words is not None and seg.words is not None)
                    else None
                )
                result[-1] = TranscriptSegment(
                    speaker_id=prev.speaker_id,
                    start_time=prev.start_time,
                    end_time=seg.end_time,
                    text=prev.text + seg.text,
                    words=merged_words,
                )
            else:
                result.append(seg)
        return result

    @staticmethod
    def _build_segment(
        speaker_id: str, words: list[WordTimestamp]
    ) -> TranscriptSegment:
        return TranscriptSegment(
            speaker_id=speaker_id,
            start_time=words[0].start_time,
            end_time=words[-1].end_time,
            text="".join(w.word for w in words),
            words=list(words),
        )
