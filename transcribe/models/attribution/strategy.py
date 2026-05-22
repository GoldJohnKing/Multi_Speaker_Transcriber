"""Segment-level speaker attribution via interval intersection voting.

For each subtitle segment, computes the temporal intersection with each
diarization segment and assigns the speaker with the most overlap.
Falls back to nearest speaker by midpoint distance when there is no overlap.

This is the standard approach used by WhisperX and recommended by the
pyannote.audio official documentation.
"""
from __future__ import annotations

from dataclasses import replace

from transcribe.data.types import (
    DiarizationResult,
    SpeakerSegment,
    TranscriptSegment,
)


class TimestampStrategy:
    """Attribute subtitle segments to speakers via temporal overlap.

    Uses the "dominant speaker by intersection duration" algorithm:
    for each segment, find the diarization speaker with the most temporal
    overlap. When a segment falls in a gap between diarization segments,
    assign the nearest speaker by midpoint distance.
    """

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
            speaker = self._assign_segment(seg, dia_segs)
            result.append(replace(seg, speaker_id=speaker))

        return result

    def _assign_segment(
        self, segment: TranscriptSegment, dia_segs: list[SpeakerSegment]
    ) -> str:
        """Find the dominant speaker for a segment by intersection duration."""
        if not dia_segs:
            return "SPEAKER_00"

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

        if speaker_overlap:
            return max(speaker_overlap.items(), key=lambda x: x[1])[0]

        return self._nearest_speaker(segment, dia_segs)

    @staticmethod
    def _nearest_speaker(
        segment: TranscriptSegment, dia_segs: list[SpeakerSegment]
    ) -> str:
        """Fallback: find speaker of nearest segment by midpoint distance."""
        if not dia_segs:
            return "SPEAKER_00"

        seg_mid = (segment.start_time + segment.end_time) / 2.0
        best_seg = min(
            dia_segs,
            key=lambda s: abs((s.start_time + s.end_time) / 2.0 - seg_mid),
        )
        return best_seg.speaker_id
