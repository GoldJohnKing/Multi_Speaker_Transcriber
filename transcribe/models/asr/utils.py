"""Shared utilities for ASR backends."""

from __future__ import annotations

import re

# Chinese punctuation characters that may be inserted between hotword chars
_PUNC_PATTERN = r"[，。？！、；：""''（）【】《》…—· ]*"


def restore_hotwords(text: str, hotword_list: list[str]) -> str:
    """Remove punctuation inserted inside hotword terms.

    The ASR model's punctuation mechanism may insert commas, periods, etc. in
    the middle of a correctly-recognised hotword (e.g. ``朽，叶`` for the
    hotword ``朽叶``).  This function detects such breakages and restores the
    original hotword form.

    Args:
        text: ASR output text (already punctuated).
        hotword_list: List of hotword terms to protect.

    Returns:
        Text with hotword-internal punctuation removed.
    """
    if not text or not hotword_list:
        return text

    # Only multi-char hotwords can have internal punctuation
    multi_char = [w for w in hotword_list if len(w) >= 2]
    if not multi_char:
        return text

    # Build combined alternation pattern
    parts: list[str] = []
    for hw in multi_char:
        chars = list(hw)
        pattern = "".join(
            re.escape(c) + _PUNC_PATTERN for c in chars[:-1]
        ) + re.escape(chars[-1])
        parts.append(pattern)
    combined = re.compile("|".join(parts))

    hotword_set = set(hotword_list)

    def _replace(match: re.Match[str]) -> str:
        matched = match.group(0)
        # Strip all punctuation/spaces to get bare characters
        bare = re.sub(_PUNC_PATTERN, "", matched)
        if bare in hotword_set:
            return bare
        return matched

    return combined.sub(_replace, text)


def parse_timestamps(
    timestamps: list,
) -> tuple[float | None, float | None]:
    """Extract segment start and end times from ASR timestamp output.

    Handles two timestamp formats:

    1. Fun-ASR-Nano dict format:
       [{"start_time": 0.36, "end_time": 0.42, ...}, ...] — times in **seconds**

    2. Paraformer list format:
       [[start_ms, end_ms], ...] — times in **milliseconds**
       [start_ms, end_ms, start_ms, end_ms, ...] — flat, in **milliseconds**

    Args:
        timestamps: Raw timestamp data from ASR model output.

    Returns:
        Tuple of (start_time_seconds, end_time_seconds).
        (None, None) if timestamps is empty.
    """
    if not timestamps:
        return None, None

    first = timestamps[0]

    if isinstance(first, dict):
        # Fun-ASR-Nano format: already in seconds
        start = first.get("start_time")
        end = timestamps[-1].get("end_time")
        return start, end

    if isinstance(first, (list, tuple)):
        # Nested Paraformer format: [[start_ms, end_ms], ...]
        return first[0] / 1000.0, timestamps[-1][1] / 1000.0

    if isinstance(first, (int, float)):
        # Flat Paraformer format: [start_ms, end_ms, start_ms, end_ms, ...]
        return timestamps[0] / 1000.0, timestamps[-1] / 1000.0

    return None, None


# --- Subtitle segmentation from character-level timestamps ---

_SENTENCE_END = frozenset("。！？!?")
_CLAUSE_END = frozenset("，；：,;:")


def segment_by_timestamps(
    char_ts: list[tuple[str, float, float]],
    max_duration: float = 7.0,
    max_chars: int = 25,
) -> list[TranscriptSegment]:
    """Split character-level timestamps into subtitle-grade segments.

    Hybrid strategy:
    1. Split at sentence-ending punctuation (。！？).
    2. When accumulated duration exceeds *max_duration*, split at the
       nearest preceding clause punctuation (，；：).
    3. When no clause punctuation is found, hard-cut at *max_duration*.
    4. When accumulated characters exceed *max_chars*, hard-cut (fallback).

    Args:
        char_ts: ``[(text, start_sec, end_sec), ...]`` character-level timestamps.
        max_duration: Maximum duration per subtitle segment (seconds).
        max_chars: Maximum characters per subtitle segment.

    Returns:
        List of :class:`TranscriptSegment`.
    """
    if not char_ts:
        return []

    from transcribe.data.types import TranscriptSegment

    segments: list[TranscriptSegment] = []
    buf_text: list[str] = []
    buf_start: float = char_ts[0][1]
    last_clause_idx: int | None = None  # index into buf_text

    for i, (text, start, end) in enumerate(char_ts):
        buf_text.append(text)

        # Flush at sentence-ending punctuation
        if text in _SENTENCE_END:
            segments.append(TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=buf_start,
                end_time=end,
                text="".join(buf_text),
            ))
            buf_text = []
            buf_start = char_ts[i + 1][1] if i + 1 < len(char_ts) else end
            last_clause_idx = None
            continue

        # Track clause punctuation positions for fallback splitting
        if text in _CLAUSE_END:
            last_clause_idx = len(buf_text) - 1

        duration = end - buf_start
        char_count = len(buf_text)

        # Max duration exceeded — try splitting at clause punctuation
        if duration > max_duration and last_clause_idx is not None and last_clause_idx > 0:
            before = buf_text[: last_clause_idx + 1]
            after = buf_text[last_clause_idx + 1 :]
            before_end = char_ts[i - len(after)][2]

            segments.append(TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=buf_start,
                end_time=before_end,
                text="".join(before),
            ))
            buf_text = after
            buf_start = start
            last_clause_idx = None
            continue

        # Max duration exceeded with no clause punctuation — hard cut
        if duration > max_duration and len(buf_text) > 1:
            segments.append(TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=buf_start,
                end_time=end,
                text="".join(buf_text),
            ))
            buf_text = []
            buf_start = char_ts[i + 1][1] if i + 1 < len(char_ts) else end
            last_clause_idx = None
            continue

        # Max chars exceeded — hard cut
        if char_count >= max_chars:
            segments.append(TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=buf_start,
                end_time=end,
                text="".join(buf_text),
            ))
            buf_text = []
            buf_start = char_ts[i + 1][1] if i + 1 < len(char_ts) else end
            last_clause_idx = None
            continue

    # Flush remaining buffer
    if buf_text:
        segments.append(TranscriptSegment(
            speaker_id="SPEAKER_00",
            start_time=buf_start,
            end_time=char_ts[-1][2],
            text="".join(buf_text),
        ))

    return segments
