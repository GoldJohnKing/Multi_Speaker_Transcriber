"""Pure text/timing subtitle segmentation — no speaker awareness.

Adapted from Si-ris-B/Qwen3-ASR-FastAPI-Docker
(``src/stt_service/services/subtitle_utils.py``), using the
``punctuation_or_pause`` split mode.
"""

from __future__ import annotations

from transcribe.data.types import TranscriptSegment, WordTimestamp

# Punctuation that triggers a split (sentence-ending marks).
_SPLIT_PUNCT = frozenset("。！？!?.")

# All known punctuation characters.
_ALL_PUNCT = frozenset("。！？!?.…—，；：,;:")
# Punctuation retained in output text (attached to preceding segment).
_RETAIN_PUNCT = frozenset("！？！!")
# Punctuation stripped from output text.
_STRIP_PUNCT = _ALL_PUNCT - _RETAIN_PUNCT


def _is_cjk(ch: str) -> bool:
    """Return ``True`` if *ch* is a CJK Unified Ideograph."""
    return "\u4e00" <= ch <= "\u9fff"


def _join_tokens(a: str, b: str) -> str:
    """CJK-aware token joining — no space between CJK characters."""
    if not a:
        return b
    if not b:
        return a
    # If either boundary character is CJK, join without space.
    for ch in (a[-1], b[0]):
        if _is_cjk(ch):
            return f"{a}{b}"
    return f"{a} {b}"


class SubtitleSegmenter:
    """Split ``list[WordTimestamp]`` into ``list[TranscriptSegment]``.

    Single-pass greedy algorithm (``punctuation_or_pause`` mode):

    - **Punctuation**: when the last word in the buffer is a
      sentence-ending punctuation mark (``。！？!?.``), the next
      word starts a new group.
    - **Pause**: when the gap between the last buffered word and the
      next word exceeds *max_gap_sec*, a new group starts.

    No merging, length-based splitting, or CPS validation is performed.
    """

    def __init__(self, max_gap_sec: float = 0.6) -> None:
        self.max_gap_sec = max_gap_sec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, words: list[WordTimestamp]) -> list[TranscriptSegment]:
        """Segment *words* into subtitle-ready ``TranscriptSegment`` s."""
        if not words:
            return []
        groups = self._group_words(words)
        return self._to_segments(groups)

    # ------------------------------------------------------------------
    # Core grouping (punctuation_or_pause)
    # ------------------------------------------------------------------

    def _group_words(
        self, words: list[WordTimestamp]
    ) -> list[list[WordTimestamp]]:
        """Single-pass grouping by punctuation or pause boundaries."""
        groups: list[list[WordTimestamp]] = []
        buf: list[WordTimestamp] = []

        for w in words:
            if not buf:
                buf.append(w)
                continue

            # Gap between this word and the last word in buffer.
            gap = w.start_time - buf[-1].end_time

            # Punctuation trigger: last buffered word is sentence-ending punct.
            end_sentence = buf[-1].word in _SPLIT_PUNCT

            should_split = end_sentence or gap > self.max_gap_sec

            if should_split:
                groups.append(buf)
                buf = [w]
            else:
                buf.append(w)

        if buf:
            groups.append(buf)

        return groups

    # ------------------------------------------------------------------
    # Output builder
    # ------------------------------------------------------------------

    @staticmethod
    def _to_segments(groups: list[list[WordTimestamp]]) -> list[TranscriptSegment]:
        """Build ``TranscriptSegment`` s from word groups."""
        segments: list[TranscriptSegment] = []
        for group in groups:
            # Skip groups that contain only punctuation.
            if all(w.word in _ALL_PUNCT for w in group):
                continue
            # Keep content words and retained punctuation; strip the rest.
            content = [w for w in group if w.word not in _STRIP_PUNCT]
            if not content:
                continue
            # CJK-aware text joining.
            text = content[0].word
            for w in content[1:]:
                text = _join_tokens(text, w.word)
            segments.append(TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=content[0].start_time,
                end_time=content[-1].end_time,
                text=text.strip(),
                words=list(content),
            ))
        return segments
