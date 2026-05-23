"""Overlap handling — word-level speaker attribution for overlap regions."""
from __future__ import annotations

from dataclasses import replace

from transcribe.data.types import (
    DiarizationResult,
    SpeakerSegment,
    TranscriptSegment,
    WordTimestamp,
)


class OverlapHandler:
    """Split overlap-region segments into per-speaker sub-segments.

    For each TranscriptSegment whose center falls in a detected overlap
    region, attribute every word to the diarization speaker with the most
    temporal intersection, then group consecutive same-speaker words into
    independent TranscriptSegments.
    """

    def handle(
        self,
        segments: list[TranscriptSegment],
        diarization: DiarizationResult,
    ) -> list[TranscriptSegment]:
        """Process overlap regions and return updated segment list."""
        overlap_regions = diarization.overlap_regions
        if not overlap_regions:
            return segments

        result: list[TranscriptSegment] = []
        for seg in segments:
            if not self._in_overlap(seg, overlap_regions):
                result.append(seg)
                continue
            result.extend(self._split_by_speaker(seg, diarization.segments))

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _in_overlap(
        seg: TranscriptSegment,
        overlap_regions: list[tuple[float, float]],
    ) -> bool:
        """Center-point check (same logic as old MarkOverlapHandler)."""
        center = (seg.start_time + seg.end_time) / 2.0
        return any(s <= center < e for s, e in overlap_regions)

    def _split_by_speaker(
        self,
        seg: TranscriptSegment,
        dia_segs: list[SpeakerSegment],
    ) -> list[TranscriptSegment]:
        """Word-level attribution → group by speaker → build sub-segments."""
        # Degraded path: no word-level timestamps
        if not seg.words:
            return [replace(seg, is_overlap=True)]

        # 1. Attribute each word to a speaker
        attributed: list[tuple[WordTimestamp, str]] = []
        for w in seg.words:
            speaker = self._attribute_word(w, dia_segs)
            attributed.append((w, speaker))

        # 2. Group consecutive same-speaker words
        groups = self._group_consecutive(attributed)

        # 3. Build one TranscriptSegment per group
        sub_segs: list[TranscriptSegment] = []
        for group in groups:
            words = [w for w, _ in group]
            speaker_id = group[0][1]
            text = "".join(w.word for w in words)
            sub_segs.append(
                TranscriptSegment(
                    speaker_id=speaker_id,
                    start_time=words[0].start_time,
                    end_time=words[-1].end_time,
                    text=text,
                    is_overlap=True,
                    words=words,
                )
            )
        return sub_segs

    @staticmethod
    def _attribute_word(
        word: WordTimestamp,
        dia_segs: list[SpeakerSegment],
    ) -> str:
        """Assign word to the speaker with the most temporal intersection.

        Falls back to nearest speaker by midpoint when intersection is zero
        for all speakers (e.g. zero-duration words).
        """
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0
        for dia in dia_segs:
            overlap_start = max(word.start_time, dia.start_time)
            overlap_end = min(word.end_time, dia.end_time)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dia.speaker_id

        if best_overlap > 0.0:
            return best_speaker

        # Fallback: nearest speaker by midpoint distance
        word_mid = (word.start_time + word.end_time) / 2.0
        nearest = min(
            dia_segs,
            key=lambda d: abs((d.start_time + d.end_time) / 2.0 - word_mid),
            default=None,
        )
        return nearest.speaker_id if nearest else best_speaker

    @staticmethod
    def _group_consecutive(
        attributed: list[tuple[WordTimestamp, str]],
    ) -> list[list[tuple[WordTimestamp, str]]]:
        """Group consecutive (word, speaker) pairs by speaker identity."""
        if not attributed:
            return []
        groups: list[list[tuple[WordTimestamp, str]]] = []
        current = [attributed[0]]
        for item in attributed[1:]:
            if item[1] == current[-1][1]:
                current.append(item)
            else:
                groups.append(current)
                current = [item]
        groups.append(current)
        return groups
