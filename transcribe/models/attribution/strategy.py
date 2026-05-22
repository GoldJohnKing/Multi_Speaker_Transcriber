"""Timestamp-based speaker attribution strategy."""
from __future__ import annotations

from transcribe.data.types import (
    DiarizationResult,
    SpeakerSegment,
    TranscriptSegment,
    WordTimestamp,
)


class TimestampStrategy:
    """Attribute words to speakers via temporal overlap with diarization segments.

    Uses exclusive_speaker_diarization (each time point belongs to exactly
    one speaker). For each word, finds which diarization segment contains
    the word's center time point.
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

        # Step 2: Merge consecutive same-speaker words
        merged = self._merge_consecutive(assigned)

        # Step 3: Smooth short interruptions
        smoothed = self._smooth_short_segments(merged)

        return smoothed

    def _assign_word(
        self, word: WordTimestamp, segments: list[SpeakerSegment]
    ) -> str:
        """Find which diarization segment contains the word's center."""
        center = (word.start_time + word.end_time) / 2.0
        for seg in segments:
            if seg.start_time <= center < seg.end_time:
                return seg.speaker_id
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
        """Merge short interruptions between same-speaker segments."""
        if len(segments) < 3:
            return segments

        result = [segments[0]]
        for i in range(1, len(segments)):
            seg = segments[i]
            prev = result[-1]
            dur = seg.end_time - seg.start_time

            # Check if this short segment is flanked by same speaker
            if (
                dur < self._min_segment_duration
                and prev.speaker_id != seg.speaker_id
            ):
                # Look at the segment after this one in original list
                if i + 1 < len(segments) and segments[i + 1].speaker_id == prev.speaker_id:
                    # Merge this short segment into previous
                    result[-1] = TranscriptSegment(
                        speaker_id=prev.speaker_id,
                        start_time=prev.start_time,
                        end_time=seg.end_time,
                        text=prev.text + seg.text,
                    )
                    continue

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
        )
