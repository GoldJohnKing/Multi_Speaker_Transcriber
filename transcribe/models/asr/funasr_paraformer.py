"""Fun-ASR (Paraformer) ASR backend with hotword support."""

from __future__ import annotations

from pathlib import Path

import torch
from funasr import AutoModel

from transcribe.data.types import AudioSegment, TranscriptSegment, WordTimestamp
from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.factory import register_backend
from transcribe.models.asr.utils import CHINESE_PUNCTUATION, parse_timestamps, restore_hotwords


class FunASRParaformerTranscriber(ASRBase):
    """Speech recognition using FunASR SeACo-Paraformer with hotword support.

    Uses the paraformer-zh model with a separate ct-punc punctuation model.
    Timestamps are in Paraformer format (milliseconds, nested or flat lists).
    """

    def __init__(
        self,
        device: str = "cpu",
        hotword_path: str | None = None,
        model_name: str = "paraformer-zh",
        vad_model: str = "fsmn-vad",
        punc_model: str = "ct-punc",
    ) -> None:
        self._device = device
        self._hotwords, self._hotword_list = self._load_hotwords(hotword_path)
        self._model = AutoModel(
            model=model_name,
            vad_model=vad_model,
            punc_model=punc_model,
            device=device,
        )

    def _load_hotwords(
        self, path: str | None
    ) -> tuple[str | None, list[str]]:
        """Load hotwords from text file (one per line), space-joined.

        Paraformer accepts hotwords as a single space-joined string via the
        ``hotword`` parameter.

        Args:
            path: Path to hotword file, or None.

        Returns:
            Tuple of (space-joined hotword string or None, list of individual words).
        """
        if path is None:
            return None, []
        p = Path(path)
        if not p.exists():
            return None, []
        words = [
            line.strip()
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return (" ".join(words) if words else None), words

    def transcribe(self, audio: AudioSegment) -> list[TranscriptSegment]:
        """Transcribe audio to text segments with timestamps."""
        result = self._model.generate(
            input=audio.waveform,
            batch_size_s=300,
            hotword=self._hotwords,
        )

        segments: list[TranscriptSegment] = []
        if not result:
            return segments

        for res in result:
            text = res.get("text", "")
            if not text:
                continue

            timestamps = res.get("timestamp", [])

            # Restore hotword terms broken by ct-punc punctuation insertion
            text = restore_hotwords(text, self._hotword_list)

            # Parse timestamps — handles nested/flat ms formats
            parsed_start, parsed_end = parse_timestamps(timestamps)

            if parsed_start is not None and parsed_end is not None:
                start_time = parsed_start + audio.start_time
                end_time = parsed_end + audio.start_time
            else:
                start_time = audio.start_time
                end_time = audio.end_time

            segments.append(
                TranscriptSegment(
                    speaker_id="SPEAKER_00",
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                )
            )

        return segments

    @staticmethod
    def _build_token_groups(text: str) -> list[tuple[str, bool]]:
        """Group text characters into alignment tokens.

        Returns list of (token_text, is_punctuation) tuples where:
        - CJK characters → individual tokens (1 char each)
        - ASCII runs (English words/numbers) → single consolidated token
        - Punctuation → individual tokens

        This matches Paraformer's timestamp granularity: CJK chars get one
        timestamp each, but English words like "npc" share a single timestamp.
        """
        _PUNC = CHINESE_PUNCTUATION
        groups: list[tuple[str, bool]] = []
        ascii_buf: list[str] = []

        def _flush_ascii() -> None:
            nonlocal ascii_buf
            if ascii_buf:
                groups.append(("".join(ascii_buf), False))
                ascii_buf = []

        for ch in text:
            if ch in _PUNC:
                _flush_ascii()
                groups.append((ch, True))
            elif ord(ch) < 128:  # ASCII (English, digits, etc.)
                ascii_buf.append(ch)
            else:  # CJK or other non-ASCII, non-punctuation
                _flush_ascii()
                groups.append((ch, False))

        _flush_ascii()
        return groups

    def _align_char_timestamps(
        self,
        text: str,
        timestamps: list[list[int]],
        audio_start: float,
    ) -> list[WordTimestamp]:
        """Align text (with punctuation) to per-char timestamps (without punctuation).

        Paraformer's ``text`` contains punctuation but ``timestamp`` only covers
        non-punctuation characters. Additionally, consecutive ASCII characters
        (e.g. "npc") share a single timestamp entry.

        This method:
        1. Groups text into tokens (CJK=1 char, ASCII runs=1 token, punctuation=1)
        2. Verifies non-punctuation token count matches timestamp count
        3. Walks groups and timestamps in parallel
        """
        groups = self._build_token_groups(text)

        # Count non-punctuation tokens and verify against timestamps
        non_punc_tokens = [(t, p) for t, p in groups if not p]
        if len(non_punc_tokens) != len(timestamps):
            return []  # Cannot align — still mismatched after consolidation

        words: list[WordTimestamp] = []
        ts_idx = 0

        for token_text, is_punc in groups:
            if is_punc:
                # Punctuation — inherit timestamp from last non-punct word
                if words:
                    last = words[-1]
                    words.append(WordTimestamp(
                        word=token_text,
                        start_time=last.start_time,
                        end_time=last.end_time,
                    ))
                elif ts_idx < len(timestamps):
                    ts = timestamps[ts_idx]
                    words.append(WordTimestamp(
                        word=token_text,
                        start_time=ts[0] / 1000.0 + audio_start,
                        end_time=ts[1] / 1000.0 + audio_start,
                    ))
            else:
                if ts_idx < len(timestamps):
                    ts = timestamps[ts_idx]
                    # Apply hotword restoration to individual tokens
                    restored_token = restore_hotwords(token_text, self._hotword_list)
                    words.append(WordTimestamp(
                        word=restored_token,
                        start_time=ts[0] / 1000.0 + audio_start,
                        end_time=ts[1] / 1000.0 + audio_start,
                    ))
                    ts_idx += 1

        return words

    @staticmethod
    def _fallback_char_timestamps(
        text: str,
        timestamps: list[list[int]],
        audio_start: float,
    ) -> list[WordTimestamp]:
        """Per-char interpolation fallback when exact alignment fails.

        Distributes the overall time range uniformly across all characters,
        preserving character-level granularity for downstream speaker attribution.
        """
        if not timestamps or not text:
            return []

        start_ms = timestamps[0][0]
        end_ms = timestamps[-1][1]
        total_dur = end_ms - start_ms
        n = len(text)
        ms_per_char = total_dur / n if n > 0 else 0.0

        words: list[WordTimestamp] = []
        for i, ch in enumerate(text):
            s = (start_ms + i * ms_per_char) / 1000.0 + audio_start
            e = (start_ms + (i + 1) * ms_per_char) / 1000.0 + audio_start
            words.append(WordTimestamp(word=ch, start_time=s, end_time=e))
        return words

    def transcribe_words(self, audio: AudioSegment) -> list[WordTimestamp]:
        """Override: return per-word timestamps from Paraformer ms-format output."""
        result = self._model.generate(
            input=audio.waveform,
            batch_size_s=300,
            hotword=self._hotwords,
        )

        words: list[WordTimestamp] = []
        if not result:
            return words

        for res in result:
            text = res.get("text", "")
            timestamps = res.get("timestamp", [])

            if (
                timestamps
                and isinstance(timestamps, list)
                and len(timestamps) > 0
                and isinstance(timestamps[0], (list, tuple))
                and len(timestamps) == len(text)
            ):
                # Exact 1:1 match — text has no punctuation, use directly
                restored = restore_hotwords(text, self._hotword_list)
                if len(restored) == len(timestamps):
                    text = restored
                for i, ts in enumerate(timestamps):
                    ch = text[i]
                    words.append(WordTimestamp(
                        word=ch,
                        start_time=ts[0] / 1000.0 + audio.start_time,
                        end_time=ts[1] / 1000.0 + audio.start_time,
                    ))
            elif (
                timestamps
                and isinstance(timestamps, list)
                and len(timestamps) > 0
                and isinstance(timestamps[0], (list, tuple))
                and len(timestamps) < len(text)
            ):
                # text has punctuation that timestamps don't cover — align them.
                # Assumption: len(timestamps) < len(text) is caused by ct-punc
                # adding punctuation to text that is absent from the timestamp
                # array. If timestamps are corrupted or a different format,
                # alignment will fail and fall through to single-word fallback.
                aligned = self._align_char_timestamps(text, timestamps, audio.start_time)
                if aligned:
                    words.extend(aligned)
                else:
                    # Exact alignment failed — per-char interpolation fallback
                    fallback = self._fallback_char_timestamps(
                        text, timestamps, audio.start_time,
                    )
                    if fallback:
                        words.extend(fallback)
                    else:
                        # Ultimate fallback: single word covering entire segment
                        text = restore_hotwords(text, self._hotword_list)
                        parsed_start, parsed_end = parse_timestamps(timestamps)
                        if parsed_start is not None:
                            words.append(WordTimestamp(
                                word=text,
                                start_time=parsed_start + audio.start_time,
                                end_time=parsed_end + audio.start_time,
                            ))
                        else:
                            words.append(WordTimestamp(
                                word=text,
                                start_time=audio.start_time,
                                end_time=audio.end_time,
                            ))
            elif timestamps:
                # Non-char-level timestamps — fallback
                text = restore_hotwords(text, self._hotword_list)
                parsed_start, parsed_end = parse_timestamps(timestamps)
                if parsed_start is not None:
                    words.append(WordTimestamp(
                        word=text,
                        start_time=parsed_start + audio.start_time,
                        end_time=parsed_end + audio.start_time,
                    ))
                else:
                    words.append(WordTimestamp(
                        word=text,
                        start_time=audio.start_time,
                        end_time=audio.end_time,
                    ))
            elif text:
                text = restore_hotwords(text, self._hotword_list)
                words.append(WordTimestamp(
                    word=text,
                    start_time=audio.start_time,
                    end_time=audio.end_time,
                ))

        return words

    def cleanup(self) -> None:
        """Release model from memory."""
        del self._model
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()


register_backend("Fun-ASR-Paraformer", FunASRParaformerTranscriber)
