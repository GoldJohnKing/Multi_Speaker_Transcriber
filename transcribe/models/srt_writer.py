"""SRT subtitle writer with punctuation-based splitting."""
from __future__ import annotations

import re

from transcribe.data.types import TranscriptSegment

# Sentence-ending punctuation — hard split, discard
_SENTENCE_END = frozenset("。！？!?……—")

# Clause-internal punctuation — soft split (duration-gated), discard
_CLAUSE_END = frozenset("，；：,;:")

# Non-split punctuation — replace with space
_REPLACE_SPACE = frozenset("\u2018\u2019'\u00b7\u2013-")


def _collapse_spaces(text: str) -> str:
    """Collapse consecutive spaces, strip edges."""
    return re.sub(r" +", " ", text).strip()


class SrtWriter:
    def __init__(
        self,
        speaker_label: bool = True,
        min_split_duration: float = 1.0,
        merge_gap: float = 0.5,
    ) -> None:
        self.speaker_label = speaker_label
        self.min_split_duration = min_split_duration
        self.merge_gap = merge_gap

    def write(
        self,
        segments: list[TranscriptSegment],
        output_path: str,
        speaker_name_map: dict[str, str] | None = None,
    ) -> None:
        if not segments:
            with open(output_path, "w", encoding="utf-8") as f:
                pass
            return

        sorted_segments = sorted(segments, key=lambda s: s.start_time)
        merged = self._merge_adjacent(sorted_segments)
        processed = self._process_all(merged)
        self._write_file(processed, output_path, speaker_name_map)

    # ── Merge adjacent same-speaker ────────────────────────

    def _merge_adjacent(
        self, segments: list[TranscriptSegment]
    ) -> list[TranscriptSegment]:
        if not segments:
            return []
        merged: list[TranscriptSegment] = [segments[0]]
        for seg in segments[1:]:
            last = merged[-1]
            if (
                seg.speaker_id == last.speaker_id
                and seg.is_overlap == last.is_overlap
                and (seg.start_time - last.end_time) < self.merge_gap
            ):
                merged[-1] = TranscriptSegment(
                    speaker_id=last.speaker_id,
                    start_time=last.start_time,
                    end_time=max(last.end_time, seg.end_time),
                    text=last.text + seg.text,
                    is_overlap=last.is_overlap,
                )
            else:
                merged.append(seg)
        return merged

    # ── Pass 1: split + clean ──────────────────────────────

    def _split_and_clean(
        self, seg: TranscriptSegment
    ) -> list[tuple[str, float, float, bool]]:
        """Single-pass: split at punctuation, clean other punctuation → spaces.

        Uses uniform time-per-char interpolation for subtitle timing.
        This is accurate enough for subtitle display since the actual
        word-level timestamps are already used for speaker attribution
        upstream (TimestampStrategy).
        """
        text = seg.text
        start = seg.start_time
        end = seg.end_time
        n = len(text)
        if n == 0:
            return []

        time_per_char = (end - start) / n

        chunks: list[tuple[str, float, float, bool]] = []
        buf: list[str] = []
        buf_start_idx = 0

        for i, ch in enumerate(text):
            if ch in _SENTENCE_END:
                if buf:
                    cleaned = _collapse_spaces("".join(buf))
                    if cleaned:
                        chunks.append((
                            cleaned,
                            start + buf_start_idx * time_per_char,
                            start + i * time_per_char,
                            True,
                        ))
                    buf = []
                buf_start_idx = i + 1
            elif ch in _CLAUSE_END:
                if buf:
                    cleaned = _collapse_spaces("".join(buf))
                    if cleaned:
                        chunks.append((
                            cleaned,
                            start + buf_start_idx * time_per_char,
                            start + i * time_per_char,
                            False,
                        ))
                    buf = []
                buf_start_idx = i + 1
            elif ch in _REPLACE_SPACE:
                buf.append(" ")
            else:
                buf.append(ch)

        if buf:
            cleaned = _collapse_spaces("".join(buf))
            if cleaned:
                chunks.append((cleaned, start + buf_start_idx * time_per_char, end, True))

        return chunks

    # ── Pass 2: duration enforcement ───────────────────────

    def _enforce_duration(
        self,
        chunks: list[tuple[str, float, float, bool]],
        speaker_id: str,
        is_overlap: bool,
    ) -> list[TranscriptSegment]:
        if not chunks:
            return []

        work: list[list] = [[t, s, e, h] for t, s, e, h in chunks]

        i = 1
        while i < len(work):
            text, cs, ce, is_hard = work[i]
            prev_text, prev_start, prev_end, _ = work[i - 1]

            chunk_dur = ce - prev_end
            prev_dur = prev_end - prev_start

            if not is_hard and (
                chunk_dur < self.min_split_duration
                or prev_dur < self.min_split_duration
            ):
                work[i - 1][0] = _collapse_spaces(prev_text + " " + text)
                work[i - 1][2] = ce
                work.pop(i)
            else:
                i += 1

        return [
            TranscriptSegment(
                speaker_id=speaker_id,
                start_time=s,
                end_time=e,
                text=t,
                is_overlap=is_overlap,
            )
            for t, s, e, _ in work
        ]

    # ── Process all ────────────────────────────────────────

    def _process_all(
        self, segments: list[TranscriptSegment]
    ) -> list[TranscriptSegment]:
        result: list[TranscriptSegment] = []
        for seg in segments:
            chunks = self._split_and_clean(seg)
            result.extend(self._enforce_duration(
                chunks, seg.speaker_id, seg.is_overlap,
            ))
        return result

    # ── File output ────────────────────────────────────────

    def _write_file(
        self,
        segments: list[TranscriptSegment],
        output_path: str,
        speaker_name_map: dict[str, str] | None,
    ) -> None:
        lines: list[str] = []
        for idx, seg in enumerate(segments, start=1):
            text = seg.text
            if seg.is_overlap:
                text = f"[重叠] {text}"
            if self.speaker_label:
                display_id = (
                    speaker_name_map.get(seg.speaker_id, seg.speaker_id)
                    if speaker_name_map
                    else seg.speaker_id
                )
                text = f"[{self._speaker_label(display_id)}] {text}"
            lines.append(str(idx))
            lines.append(
                f"{self._format_timestamp(seg.start_time)} --> "
                f"{self._format_timestamp(seg.end_time)}"
            )
            lines.append(text)
            lines.append("")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total_ms = int(round(seconds * 1000))
        ms = total_ms % 1000
        total_s = total_ms // 1000
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _speaker_label(speaker_id: str) -> str:
        if speaker_id.startswith("SPEAKER_"):
            try:
                num = int(speaker_id.split("_", 1)[1]) + 1
                return f"说话人{num}"
            except (ValueError, IndexError):
                return speaker_id
        return speaker_id
