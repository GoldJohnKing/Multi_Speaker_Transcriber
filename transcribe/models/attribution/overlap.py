"""Overlap handling for speaker attribution."""
from __future__ import annotations

from abc import ABC, abstractmethod

from transcribe.data.types import TranscriptSegment


class OverlapHandler(ABC):
    """Base class for overlap region handling."""

    @abstractmethod
    def process(
        self,
        segments: list[TranscriptSegment],
        overlap_regions: list[tuple[float, float]],
    ) -> list[TranscriptSegment]:
        ...


class MarkOverlapHandler(OverlapHandler):
    """Phase 1: Mark segments that overlap with detected overlap regions."""

    def process(
        self,
        segments: list[TranscriptSegment],
        overlap_regions: list[tuple[float, float]],
    ) -> list[TranscriptSegment]:
        if not overlap_regions:
            return segments

        for seg in segments:
            for ov_start, ov_end in overlap_regions:
                if seg.start_time < ov_end and seg.end_time > ov_start:
                    seg.is_overlap = True
                    break
        return segments


class SeparateOverlapHandler(OverlapHandler):
    """Phase 2 placeholder: speech separation for overlap regions.

    Will use pyannote SpeechSeparation or SepFormer to separate
    overlapping speakers, then ASR each stream independently.
    """

    def __init__(self, separator, asr_backend) -> None:
        self._separator = separator
        self._asr_backend = asr_backend

    def process(
        self,
        segments: list[TranscriptSegment],
        overlap_regions: list[tuple[float, float]],
    ) -> list[TranscriptSegment]:
        raise NotImplementedError("Phase 2: speech separation not yet implemented")
