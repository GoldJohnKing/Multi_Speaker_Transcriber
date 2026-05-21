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
