"""Overlap speech separation — PixIT-based separation for overlap regions."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np

from transcribe.data.types import AudioSegment, TranscriptSegment, WordTimestamp

_logger = logging.getLogger(__name__)


@dataclass
class OverlapClip:
    """A padded, merged audio clip containing one or more overlap regions."""

    waveform: np.ndarray
    sample_rate: int
    start_time: float  # absolute time in original audio
    end_time: float
    source_overlaps: list[tuple[float, float]]  # original (unpadded) overlap regions


def extract_overlap_clips(
    audio: AudioSegment,
    overlap_regions: list[tuple[float, float]],
    padding: float = 3.0,
) -> list[OverlapClip]:
    """Extract padded, merged audio clips for overlap regions.

    For each overlap region, creates a clip extended by *padding* seconds on
    each side. Adjacent or overlapping padded clips are merged into a single
    clip. Each clip tracks which original overlap regions it covers.

    Padding is clamped to audio boundary. Returns empty list if no overlaps.
    """
    if not overlap_regions:
        return []

    audio_start = audio.start_time
    audio_end = audio.end_time

    # Step 1: Pad each overlap region
    padded: list[tuple[float, float, tuple[float, float]]] = []
    for ov_start, ov_end in overlap_regions:
        padded_start = max(ov_start - padding, audio_start)
        padded_end = min(ov_end + padding, audio_end)
        padded.append((padded_start, padded_end, (ov_start, ov_end)))

    # Sort by start time
    padded.sort(key=lambda x: x[0])

    # Step 2: Merge overlapping/adjacent padded regions
    merged: list[tuple[float, float, list[tuple[float, float]]]] = []
    cur_start, cur_end, cur_sources = padded[0][0], padded[0][1], [padded[0][2]]

    for p_start, p_end, source in padded[1:]:
        if p_start <= cur_end:  # overlapping or adjacent → merge
            cur_end = max(cur_end, p_end)
            cur_sources.append(source)
        else:
            merged.append((cur_start, cur_end, cur_sources))
            cur_start, cur_end, cur_sources = p_start, p_end, [source]
    merged.append((cur_start, cur_end, cur_sources))

    # Step 3: Extract waveform for each merged clip
    clips: list[OverlapClip] = []
    for clip_start, clip_end, sources in merged:
        start_sample = int((clip_start - audio_start) * audio.sample_rate)
        end_sample = int((clip_end - audio_start) * audio.sample_rate)
        start_sample = max(0, start_sample)
        end_sample = min(len(audio.waveform), end_sample)

        clips.append(OverlapClip(
            waveform=audio.waveform[start_sample:end_sample],
            sample_rate=audio.sample_rate,
            start_time=clip_start,
            end_time=clip_end,
            source_overlaps=sources,
        ))

    return clips


def _trim_words_to_overlaps(
    words: list[WordTimestamp],
    overlap_regions: list[tuple[float, float]],
) -> list[WordTimestamp]:
    """Keep only words that intersect with any overlap region.

    A word is kept if any part of its time range falls within an overlap
    region. This is intentionally lenient — it's better to keep a boundary
    word than to lose it.
    """
    if not words or not overlap_regions:
        return []

    result: list[WordTimestamp] = []
    for w in words:
        for ov_start, ov_end in overlap_regions:
            if w.start_time < ov_end and w.end_time > ov_start:
                result.append(w)
                break
    return result


def _map_local_to_global_speakers(
    local_labels: list[str],
    local_times: list[tuple[float, float]],
    global_speakers: list[str],
    global_segs: list,
) -> dict[str, str]:
    """Map PixIT's local speaker labels to global diarization speaker IDs.

    Uses temporal intersection voting: for each local speaker, find the global
    speaker with the most temporal overlap.

    Args:
        local_labels: Speaker labels from PixIT output (e.g., ["A", "B"]).
        local_times: (start, end) for each local speaker's active time.
        global_speakers: Global speaker IDs from diarization.
        global_segs: Global SpeakerSegment-like objects with speaker_id, start_time, end_time.

    Returns:
        Mapping from local label → global speaker ID.
    """
    mapping: dict[str, str] = {}

    for label, (local_start, local_end) in zip(local_labels, local_times):
        best_speaker = global_speakers[0] if global_speakers else label
        best_overlap = 0.0
        for seg in global_segs:
            ov = max(0.0, min(local_end, seg.end_time) - max(local_start, seg.start_time))
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = seg.speaker_id
        mapping[label] = best_speaker

    return mapping
