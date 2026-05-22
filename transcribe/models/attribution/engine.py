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
