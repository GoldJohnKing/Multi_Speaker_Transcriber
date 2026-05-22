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

    def _align_char_timestamps(
        self,
        text: str,
        timestamps: list[list[int]],
        audio_start: float,
    ) -> list[WordTimestamp]:
        """Align text (with punctuation) to per-char timestamps (without punctuation).

        Paraformer's ``text`` contains punctuation but ``timestamp`` only covers
        non-punctuation characters.  This method strips punctuation from text to
        verify length match, then walks both in parallel — punctuation characters
        are emitted without timestamps (inheriting the nearest timestamp), while
        non-punctuation characters consume the next timestamp entry.
        """
        # Punctuation that Paraformer adds to text but excludes from timestamps
        # Shared with utils.restore_hotwords — kept in sync via CHINESE_PUNCTUATION
        _PUNC = CHINESE_PUNCTUATION

        # Count non-punctuation chars
        non_punc_chars = [ch for ch in text if ch not in _PUNC]
        if len(non_punc_chars) != len(timestamps):
            return []  # Cannot align — lengths don't match even after stripping

        words: list[WordTimestamp] = []
        ts_idx = 0

        # Apply hotword restoration to the punctuation-stripped text first
        stripped = "".join(non_punc_chars)
        restored = restore_hotwords(stripped, self._hotword_list)

        # If restoration changed length, fall back to non-restored
        if len(restored) != len(timestamps):
            restored = stripped

        # Build a mapping: position in stripped text → position in restored text
        # They're the same length (both stripped), but content may differ
        restored_chars = list(restored)

        for ch in text:
            if ch in _PUNC:
                # Punctuation — inherit timestamp from last non-punct char
                # or use the next one if this is the first char
                if words:
                    last = words[-1]
                    words.append(WordTimestamp(
                        word=ch,
                        start_time=last.start_time,
                        end_time=last.end_time,
                    ))
                elif ts_idx < len(timestamps):
                    ts = timestamps[ts_idx]
                    words.append(WordTimestamp(
                        word=ch,
                        start_time=ts[0] / 1000.0 + audio_start,
                        end_time=ts[1] / 1000.0 + audio_start,
                    ))
            else:
                if ts_idx < len(timestamps) and ts_idx < len(restored_chars):
                    ts = timestamps[ts_idx]
                    words.append(WordTimestamp(
                        word=restored_chars[ts_idx],
                        start_time=ts[0] / 1000.0 + audio_start,
                        end_time=ts[1] / 1000.0 + audio_start,
                    ))
                    ts_idx += 1

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
                    # Alignment failed — fallback to single word
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
