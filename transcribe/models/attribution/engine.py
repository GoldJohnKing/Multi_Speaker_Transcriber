"""Attribution engine — segment-level speaker attribution via temporal overlap."""
from __future__ import annotations

from transcribe.data.types import DiarizationResult, TranscriptSegment
from transcribe.models.attribution.strategy import TimestampStrategy


class AttributionEngine:
    """Assign speakers to subtitle segments based on temporal overlap.

    Each segment is attributed to the diarization speaker with the most
    temporal intersection.  No turn-boundary splitting or overlap handling
    is performed — a single dominant speaker is assigned per segment.
    """

    def __init__(self) -> None:
        self._strategy = TimestampStrategy()

    def run(
        self,
        segments: list[TranscriptSegment],
        diarization: DiarizationResult,
    ) -> list[TranscriptSegment]:
        """Assign speakers to segments based on temporal overlap."""
        return self._strategy.attribute(segments, diarization)
