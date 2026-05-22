"""Attribution engine — orchestrates strategy + overlap handling."""
from __future__ import annotations

from transcribe.data.types import (
    DiarizationResult,
    TranscriptSegment,
    WordTimestamp,
)
from transcribe.models.attribution.overlap import (
    MarkOverlapHandler,
    OverlapHandler,
)
from transcribe.models.attribution.strategy import TimestampStrategy


class AttributionEngine:
    """Run speaker attribution on ASR word-level output.

    Config: TimestampStrategy + MarkOverlapHandler.
    """

    def __init__(
        self,
        *,
        min_segment_duration: float = 0.2,
        overlap_handler: OverlapHandler | None = None,
    ) -> None:
        self._strategy = TimestampStrategy(
            min_segment_duration=min_segment_duration,
        )
        self._overlap_handler = overlap_handler or MarkOverlapHandler()

    def run(
        self,
        words: list[WordTimestamp],
        diarization: DiarizationResult,
        overlap_regions: list[tuple[float, float]],
    ) -> list[TranscriptSegment]:
        segments = self._strategy.attribute(words, diarization)
        segments = self._overlap_handler.process(segments, overlap_regions)
        return segments
