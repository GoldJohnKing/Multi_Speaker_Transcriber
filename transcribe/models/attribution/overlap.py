"""Overlap handling for speaker attribution."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

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

        result: list[TranscriptSegment] = []
        for seg in segments:
            # Use center-point check instead of "any intersection"
            # to avoid marking entire long segments for brief touches
            center = (seg.start_time + seg.end_time) / 2.0
            is_overlap = any(
                ov_start <= center < ov_end
                for ov_start, ov_end in overlap_regions
            )
            if is_overlap:
                result.append(replace(seg, is_overlap=True))
            else:
                result.append(seg)
        return result

