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
    """Per-speaker subtitle lines for overlap regions.

    For each TranscriptSegment whose center falls in a detected overlap
    region, attribute every word to the diarization speaker with the most
    temporal intersection.  Then collect words by speaker independently
    (not linearly), group each speaker's words into temporally continuous
    spans, and build one TranscriptSegment per span.  Segments from
    different speakers may overlap in time, enabling simultaneous display.
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
        """Per-speaker word collection → temporal span grouping → sub-segments.

        Unlike linear consecutive grouping, this collects all words attributed
        to the same speaker and groups them into temporally continuous spans.
        This prevents a single interjection from another speaker from
        fragmenting the primary speaker's subtitle line.
        """
        # Degraded path: no word-level timestamps
        if not seg.words:
            return [replace(seg, is_overlap=True)]

        # 1. Attribute each word to a speaker
        attributed: list[tuple[WordTimestamp, str]] = []
        for w in seg.words:
            speaker = self._attribute_word(w, dia_segs)
            attributed.append((w, speaker))

        # 2. Collect words per speaker (preserving original order)
        from collections import OrderedDict
        speaker_words: OrderedDict[str, list[WordTimestamp]] = OrderedDict()
        for w, speaker in attributed:
            speaker_words.setdefault(speaker, []).append(w)

        # 3. Within each speaker, group into temporally continuous spans
        sub_segs: list[TranscriptSegment] = []
        for speaker_id, words in speaker_words.items():
            for group in self._group_temporal_spans(words):
                text = "".join(w.word for w in group)
                sub_segs.append(
                    TranscriptSegment(
                        speaker_id=speaker_id,
                        start_time=group[0].start_time,
                        end_time=group[-1].end_time,
                        text=text,
                        is_overlap=True,
                        words=list(group),
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
    def _group_temporal_spans(
        words: list[WordTimestamp],
        max_gap: float = 0.5,
    ) -> list[list[WordTimestamp]]:
        """Group words into temporally continuous spans.

        A gap between consecutive words exceeding *max_gap* starts a new span.
        This ensures that if a speaker has non-contiguous utterances within an
        overlap region (e.g. speaks, pauses, speaks again), they produce
        separate subtitle lines.
        """
        if not words:
            return []
        groups: list[list[WordTimestamp]] = [[words[0]]]
        for w in words[1:]:
            prev = groups[-1][-1]
            gap = w.start_time - prev.end_time
            if gap > max_gap:
                groups.append([w])
            else:
                groups[-1].append(w)
        return groups
