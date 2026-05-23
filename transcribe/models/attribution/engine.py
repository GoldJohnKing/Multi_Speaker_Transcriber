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

            # Partition words by midpoint.
            # Last sub-segment uses <= to include words exactly at end boundary.
            if i == len(splits) - 2:
                sub_words = [
                    w for w in seg.words
                    if sub_start <= (w.start_time + w.end_time) / 2 <= sub_end
                ]
            else:
                sub_words = [
                    w for w in seg.words
                    if sub_start <= (w.start_time + w.end_time) / 2 < sub_end
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
